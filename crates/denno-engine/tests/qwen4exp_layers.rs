//! P-MOE-F3: ssm 層を [`SsmAttn`] で層ループに接続し、合成 GGUF で
//! `block 4 / ssm 3 層 + full attention 1 層` を通すテスト。
//!
//! 実 GGUF (111 GB) はこのコンテナに無いので開かない。同じ構造で小さい次元の
//! 合成 GGUF を一時ディレクトリに書き、`Qwen4Exp::with_attn` に
//! `SsmAttn::from_gguf` を渡して 1 トークンの forward を回す。
//!
//! # 次元を目録 (実モデル) から縮める規則
//!
//! 目録の比を保ったまま縮めるが、**量子型のブロック長が下限を決める**。
//! `validate` は目録と同じ量子型を要求するので、各テンソルの `ne0` は
//! その型のブロック長の倍数でなければならない (`GgufLinear` が拒否する)。
//!
//! | 制約 | 出どころ | この合成モデル |
//! |---|---|---|
//! | `n_embd % 256 == 0` | `ffn_gate_exps` が Q4_K (ブロック 256) | 256 |
//! | `n_ff_exp % 32 == 0` | `ffn_down_exps` が Q5_1 (ブロック 32) | 32 |
//! | `hc_low_rank % 32 == 0` | `hc_attn_up` / `output_hc_up` が Q8_0 | 32 |
//! | `n_embd_per_layer % 32 == 0` | `per_layer_token_embd` が IQ4_NL | 64 |
//!
//! ssm の諸元は目録の比のまま縮める (`state 8 / group 2 / dt_rank 4 /
//! inner 32 / conv 4`)。`inner == state * dt_rank` と
//! `n_embd_per_layer * (ngram-1) * heads_per_ngram == n_embd` は
//! メタデータ側で検査されるので、その 2 つは満たすように選ぶ。
//!
//! 数値の正しさ (実モデルとの一致) は F4 で判定する。ここで見るのは
//! 「層の結線が通る・有限・状態が層ごとに独立・履歴が効く」ことだけ。

use std::path::{Path, PathBuf};

use denno_engine::gdn::GdnParams;
use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::{
    validate, IdentityAttn, LayerAttn, LayerKind, Qwen4Exp, Qwen4ExpError, Qwen4ExpParams,
};
use denno_engine::model::qwen4exp_ssm::{gdn_params, SsmAttn};
use denno_engine::ple::{PleNgram, PleTable, PLE_TABLE_NAME};

const ALIGN: usize = 32;

// GGUF メタデータ型 id (gguf.h enum gguf_type)
const GT_U32: u32 = 4;
const GT_F32: u32 = 6;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;
const GT_U64: u32 = 10;

// ggml_type id
const TY_F32: u32 = 0;
const TY_Q5_1: u32 = 7;
const TY_Q8_0: u32 = 8;
const TY_Q4_K: u32 = 12;
const TY_IQ4_NL: u32 = 20;
const TY_BF16: u32 = 30;

// ---------------------------------------------------------------------------
// 合成 GGUF の書き出し
// ---------------------------------------------------------------------------

enum Kv {
    U32(u32),
    F32(f32),
    Str(&'static str),
    ArrU32(Vec<u32>),
    ArrU64(Vec<u64>),
    ArrStr(Vec<String>),
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::F32(v) => {
                b.extend_from_slice(&GT_F32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::Str(s) => {
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                push_str(b, s);
            }
            Kv::ArrU32(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Kv::ArrU64(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U64.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Kv::ArrStr(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for s in v {
                    push_str(b, s);
                }
            }
        }
    }
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    ty: u32,
}

fn ts(name: impl Into<String>, dims: Vec<u64>, ty: u32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dims,
        ty,
    }
}

fn blck_size(ty: u32) -> u64 {
    match ty {
        TY_F32 | TY_BF16 => 1,
        TY_Q5_1 | TY_Q8_0 | TY_IQ4_NL => 32,
        TY_Q4_K => 256,
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
}

fn type_size(ty: u32) -> u64 {
    match ty {
        TY_F32 => 4,
        TY_BF16 => 2,
        TY_Q5_1 => 24,
        TY_Q8_0 => 34,
        TY_IQ4_NL => 18,
        TY_Q4_K => 144,
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
}

/// gguf.rs のパーサと同じ式で nbytes を求める。
fn nbytes(dims: &[u64], ty: u32) -> u64 {
    let ne: u64 = dims.iter().product();
    let bs = blck_size(ty);
    ne / bs * type_size(ty) + if ne.is_multiple_of(bs) { 0 } else { type_size(ty) }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// 決定的な擬似乱数バイト (0..=255)。
fn pat(i: usize, seed: u32) -> u8 {
    let mut x = (i as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40_503));
    x ^= x >> 13;
    x = x.wrapping_mul(0x9E37_79B1);
    x ^= x >> 16;
    x as u8
}

fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

/// テンソルの中身。**有限値だけ**を作る (スケール欄を乱数にすると NaN になる)。
///
/// - 名前に `norm` を含む gamma は 1.0 (変換器が `1 + w` に畳んだ前提の値)
/// - `ssm_a` は負値 (-0.5)。`ssm_dt.bias` は 0
/// - `ssm_conv1d` は tap ごとに違う係数 (`k` が古いほど小さい) —
///   履歴が効くことの検査 (検査 4) がこの係数に乗っている
/// - 量子型はスケール欄だけ固定し、量子値はブロックごとに変える
///   (全行同一にすると出力が全要素同値になり、経路の生死が分からない)
/// - `zero_ple` のときは PLE 表 (`per_layer_token_embd`) を厳密な 0 で書く
///   (検査 6 の「PLE が効いている」比較用)
fn tensor_bytes(name: &str, dims: &[u64], ty: u32, seed: u32, zero_ple: bool) -> Vec<u8> {
    let nb = nbytes(dims, ty) as usize;
    let mut b: Vec<u8> = (0..nb).map(|i| pat(i, 7 + seed)).collect();
    match ty {
        TY_F32 => {
            for i in 0..nb / 4 {
                let v = if name.contains("norm") {
                    1.0f32
                } else if name.ends_with(".ssm_a") {
                    -0.5f32
                } else if name.ends_with("ssm_dt.bias") {
                    0.0f32
                } else if name.ends_with("ssm_conv1d.weight") {
                    // [conv_dim][d_conv] — 古い tap ほど小さい係数
                    0.05 * (((i % SSM_CONV as usize) + 1) as f32)
                } else if name.ends_with("ple_conv1d.weight") {
                    // [hc_dim][kern] — 遅延 0 (最後の tap) を主にする
                    0.1 * (((i % PLE_CONV as usize) + 1) as f32)
                } else {
                    (((i as u32).wrapping_add(seed * 5) % 17) as f32 - 8.0) * 0.01
                };
                b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        TY_BF16 => {
            for i in 0..nb / 2 {
                let v = (((i as u32).wrapping_add(seed * 3) % 13) as f32 - 6.0) * 0.01;
                let bits = (v.to_bits() >> 16) as u16;
                b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        TY_Q8_0 => {
            for blk in 0..nb / 34 {
                let o = blk * 34;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                for j in 0..32u32 {
                    let q = ((j as i32 + blk as i32 * 3 + seed as i32 * 7) % 11) - 5;
                    b[o + 2 + j as usize] = q as i8 as u8;
                }
            }
        }
        TY_Q5_1 => {
            for blk in 0..nb / 24 {
                let o = blk * 24;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
            }
        }
        TY_Q4_K => {
            for blk in 0..nb / 144 {
                let o = blk * 144;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                // dmin = 0 にして、scales[12] の 6 bit をすべて 1 にする
                // (乱数のままだと sc が 63 まで出て expert の出力が 1e4 級になり、
                //  層の差分が相対 1e-4 になって trace が読みにくい)
                // ggml の get_scale_min_k4: j<4 は q[j] / q[j+4] の下位 6 bit、
                // j>=4 は q[j+4] の nibble + q[j-4]/q[j] の上位 2 bit。
                b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.0));
                for j in 0..8 {
                    b[o + 4 + j] = 0x01;
                }
                for j in 8..12 {
                    b[o + 4 + j] = 0x11;
                }
            }
        }
        TY_IQ4_NL => {
            let zero = zero_ple && name == PLE_TABLE_NAME;
            for blk in 0..nb / 18 {
                let o = blk * 18;
                b[o..o + 2].copy_from_slice(&f16_bytes(if zero { 0.0 } else { 0.02 }));
                if zero {
                    for j in 0..16 {
                        b[o + 2 + j] = 0;
                    }
                }
            }
        }
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
    b
}

/// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列、テンソルは与えた順に並ぶ。
fn write_gguf(path: &Path, kvs: &[(String, Kv)], tensors: &[TensorSpec], zero_ple: bool) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        push_str(&mut head, k);
        v.write(&mut head);
    }

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for t in tensors {
        offsets.push(cur);
        cur = align_up((cur + nbytes(&t.dims, t.ty)) as usize, ALIGN) as u64;
    }

    for (t, off) in tensors.iter().zip(&offsets) {
        push_str(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_le_bytes());
        head.extend_from_slice(&off.to_le_bytes());
    }

    let mut out = head;
    out.resize(align_up(out.len(), ALIGN), 0);
    for (i, t) in tensors.iter().enumerate() {
        out.extend_from_slice(&tensor_bytes(&t.name, &t.dims, t.ty, i as u32, zero_ple));
        out.resize(align_up(out.len(), ALIGN), 0);
    }
    std::fs::write(path, out).expect("テスト GGUF の書き出し");
}

/// テスト用の一時ディレクトリ (終了時に消す)。
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_qwen4exp_layers_{tag}_{}_{n}",
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

// ---------------------------------------------------------------------------
// 小さな合成モデル (構造は目録どおり・次元だけ小さい)
// ---------------------------------------------------------------------------

const N_EMBD: u64 = 256;
const N_VOCAB: u64 = 8;
const N_HEAD: u64 = 2;
const HEAD: u64 = 16;
const N_HEAD_KV: u64 = 1;
const N_EXPERT: u64 = 4;
const N_EXPERT_USED: u32 = 2;
const N_FF: u64 = 32;
const HC: u64 = 2;
const HC_LR: u64 = 32;
const SSM_CONV: u64 = 4;
const SSM_STATE: u64 = 8;
const SSM_GROUP: u64 = 2;
const SSM_DT: u64 = 4;
const SSM_INNER: u64 = 32;
const INTERVAL: u64 = 4;
const IDX_HEAD: u64 = 2;
const IDX_DIM: u64 = 8;
const IDX_TOP_K: u64 = 8;
const RATIO: u32 = 4;
const PLE_NGRAM: u64 = 3;
const PLE_HEADS_PER_NGRAM: u64 = 2;
/// `n_ple_heads * n_embd_per_layer == n_embd` かつ IQ4_NL のブロック長 32 の倍数。
const N_EMBD_PER_LAYER: u64 = N_EMBD / ((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
const PLE_LAYER: usize = 1;
const PLE_CONV: u64 = 4;
/// head_offsets [0,10,20,30] + head_vocab_sizes [10,10,10,10] の最大値
const PLE_MIN_ROWS: u64 = 40;
const DEFAULT_LAYERS: usize = 4;

const HC_DIM: u64 = N_EMBD * HC;
const KEY_DIM: u64 = SSM_STATE * SSM_GROUP;
const VALUE_DIM: u64 = SSM_INNER;
const CONV_DIM: u64 = 2 * KEY_DIM + VALUE_DIM;

fn synth_kvs(layers: usize) -> Vec<(String, Kv)> {
    let mut ratios = vec![0u32; layers];
    for (il, r) in ratios.iter_mut().enumerate() {
        if (il + 1) % INTERVAL as usize == 0 {
            *r = RATIO;
        }
    }
    vec![
        ("general.architecture".to_string(), Kv::Str("qwen4exp")),
        ("general.alignment".to_string(), Kv::U32(ALIGN as u32)),
        (
            "qwen4exp.block_count".to_string(),
            Kv::U32(layers as u32),
        ),
        ("qwen4exp.context_length".to_string(), Kv::U32(1024)),
        (
            "qwen4exp.embedding_length".to_string(),
            Kv::U32(N_EMBD as u32),
        ),
        (
            "qwen4exp.attention.head_count".to_string(),
            Kv::U32(N_HEAD as u32),
        ),
        (
            "qwen4exp.attention.head_count_kv".to_string(),
            Kv::U32(N_HEAD_KV as u32),
        ),
        (
            "qwen4exp.attention.key_length".to_string(),
            Kv::U32(HEAD as u32),
        ),
        (
            "qwen4exp.attention.value_length".to_string(),
            Kv::U32(HEAD as u32),
        ),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon".to_string(),
            Kv::F32(1e-6),
        ),
        ("qwen4exp.rope.dimension_count".to_string(), Kv::U32(4)),
        (
            "qwen4exp.rope.dimension_sections".to_string(),
            Kv::ArrU32(vec![2, 2, 2, 0]),
        ),
        ("qwen4exp.rope.freq_base".to_string(), Kv::F32(1e7)),
        (
            "qwen4exp.expert_count".to_string(),
            Kv::U32(N_EXPERT as u32),
        ),
        (
            "qwen4exp.expert_used_count".to_string(),
            Kv::U32(N_EXPERT_USED),
        ),
        (
            "qwen4exp.expert_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        (
            "qwen4exp.ssm.conv_kernel".to_string(),
            Kv::U32(SSM_CONV as u32),
        ),
        (
            "qwen4exp.ssm.state_size".to_string(),
            Kv::U32(SSM_STATE as u32),
        ),
        (
            "qwen4exp.ssm.group_count".to_string(),
            Kv::U32(SSM_GROUP as u32),
        ),
        (
            "qwen4exp.ssm.time_step_rank".to_string(),
            Kv::U32(SSM_DT as u32),
        ),
        (
            "qwen4exp.ssm.inner_size".to_string(),
            Kv::U32(SSM_INNER as u32),
        ),
        (
            "qwen4exp.full_attention_interval".to_string(),
            Kv::U32(INTERVAL as u32),
        ),
        (
            "qwen4exp.hyper_connection.count".to_string(),
            Kv::U32(HC as u32),
        ),
        (
            "qwen4exp.hyper_connection.low_rank".to_string(),
            Kv::U32(HC_LR as u32),
        ),
        (
            "qwen4exp.attention.indexer.head_count".to_string(),
            Kv::U32(IDX_HEAD as u32),
        ),
        (
            "qwen4exp.attention.indexer.key_length".to_string(),
            Kv::U32(IDX_DIM as u32),
        ),
        (
            "qwen4exp.attention.indexer.top_k".to_string(),
            Kv::U32(IDX_TOP_K as u32),
        ),
        (
            "qwen4exp.attention.compress_ratios".to_string(),
            Kv::ArrU32(ratios),
        ),
        (
            "qwen4exp.embedding_length_per_layer_input".to_string(),
            Kv::U32(N_EMBD_PER_LAYER as u32),
        ),
        (
            "qwen4exp.ple.layers".to_string(),
            Kv::ArrU32(vec![PLE_LAYER as u32]),
        ),
        (
            "qwen4exp.ple.ngram_size".to_string(),
            Kv::U32(PLE_NGRAM as u32),
        ),
        (
            "qwen4exp.ple.heads_per_ngram".to_string(),
            Kv::U32(PLE_HEADS_PER_NGRAM as u32),
        ),
        (
            "qwen4exp.ple.conv_kernel".to_string(),
            Kv::U32(PLE_CONV as u32),
        ),
        ("qwen4exp.ple.eos_token_id".to_string(), Kv::U32(7)),
        ("qwen4exp.ple.image_token_id".to_string(), Kv::U32(9)),
        (
            "qwen4exp.ple.layer_multipliers".to_string(),
            Kv::ArrU64(vec![1, 2, 3]),
        ),
        (
            "qwen4exp.ple.head_offsets".to_string(),
            Kv::ArrU64(vec![0, 10, 20, 30]),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes".to_string(),
            Kv::ArrU64(vec![10, 10, 10, 10]),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Kv::ArrStr(vec![String::new(); N_VOCAB as usize]),
        ),
    ]
}

/// 全層共通の 16 本 (`blk.{il}.` の後ろ)。
fn common_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], TY_F32),
        ts(p("ffn_gate_inp_shexp.weight"), vec![N_EMBD], TY_F32),
        ts(
            p("ffn_gate_exps.weight"),
            vec![N_EMBD, N_FF, N_EXPERT],
            TY_Q4_K,
        ),
        ts(
            p("ffn_up_exps.weight"),
            vec![N_EMBD, N_FF, N_EXPERT],
            TY_Q4_K,
        ),
        ts(
            p("ffn_down_exps.weight"),
            vec![N_FF, N_EMBD, N_EXPERT],
            TY_Q5_1,
        ),
        ts(p("ffn_gate_shexp.weight"), vec![N_EMBD, N_FF], TY_Q8_0),
        ts(p("ffn_up_shexp.weight"), vec![N_EMBD, N_FF], TY_Q8_0),
        ts(p("ffn_down_shexp.weight"), vec![N_FF, N_EMBD], TY_Q8_0),
        ts(p("hc_attn_norm.weight"), vec![HC_DIM], TY_F32),
        ts(p("hc_attn_up.weight"), vec![HC_LR, HC_DIM], TY_Q8_0),
        ts(p("hc_attn_down.weight"), vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(p("hc_attn_inject.weight"), vec![HC_DIM, HC], TY_F32),
        ts(p("hc_ffn_norm.weight"), vec![HC_DIM], TY_F32),
        ts(p("hc_ffn_up.weight"), vec![HC_LR, HC_DIM], TY_Q8_0),
        ts(p("hc_ffn_down.weight"), vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(p("hc_ffn_inject.weight"), vec![HC_DIM, HC], TY_F32),
    ]
}

/// ssm 層の 9 本 (`ssm_a` と `ssm_dt.bias` はサフィックス無し)。
fn ssm_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ssm_a"), vec![SSM_DT], TY_F32),
        ts(p("ssm_dt.bias"), vec![SSM_DT], TY_F32),
        ts(p("ssm_alpha.weight"), vec![N_EMBD, SSM_DT], TY_F32),
        ts(p("ssm_beta.weight"), vec![N_EMBD, SSM_DT], TY_F32),
        ts(p("ssm_conv1d.weight"), vec![SSM_CONV, CONV_DIM], TY_F32),
        ts(p("ssm_norm.weight"), vec![SSM_STATE], TY_F32),
        ts(p("ssm_out.weight"), vec![VALUE_DIM, N_EMBD], TY_Q8_0),
        ts(p("attn_qkv.weight"), vec![N_EMBD, CONV_DIM], TY_Q8_0),
        ts(p("attn_gate.weight"), vec![N_EMBD, VALUE_DIM], TY_Q8_0),
    ]
}

/// full attention 層の 10 本。
fn full_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(
            p("attn_q.weight"),
            vec![N_EMBD, 2 * N_HEAD * HEAD],
            TY_Q8_0,
        ),
        ts(p("attn_k.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_v.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_q_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_k_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_output.weight"), vec![N_HEAD * HEAD, N_EMBD], TY_Q8_0),
        ts(
            p("indexer.q_proj.weight"),
            vec![N_EMBD, IDX_HEAD * IDX_DIM],
            TY_BF16,
        ),
        ts(p("indexer.k_proj.weight"), vec![N_EMBD, IDX_DIM], TY_BF16),
        ts(p("indexer.q_norm.weight"), vec![IDX_DIM], TY_F32),
        ts(p("indexer.k_norm.weight"), vec![IDX_DIM], TY_F32),
    ]
}

/// PLE 層の 6 本。
fn ple_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ple_key.weight"), vec![N_EMBD, HC_DIM], TY_Q8_0),
        ts(p("ple_value.weight"), vec![N_EMBD, N_EMBD], TY_Q8_0),
        ts(p("ple_norm_key.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_norm_query.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_norm_conv.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_conv1d.weight"), vec![PLE_CONV, HC_DIM], TY_F32),
    ]
}

/// 目録のとおりのテンソル束 (小さい次元で同じ構造)。
fn synth_tensors(layers: usize) -> Vec<TensorSpec> {
    let mut out = vec![
        ts("token_embd.weight", vec![N_EMBD, N_VOCAB], TY_Q8_0),
        ts("output.weight", vec![N_EMBD, N_VOCAB], TY_Q8_0),
        ts("output_hc_norm.weight", vec![HC_DIM], TY_F32),
        ts("output_hc_up.weight", vec![HC_LR, HC_DIM], TY_Q8_0),
        ts("output_hc_down.weight", vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(
            PLE_TABLE_NAME,
            vec![N_EMBD_PER_LAYER, PLE_MIN_ROWS],
            TY_IQ4_NL,
        ),
    ];
    for il in 0..layers {
        out.extend(common_tensors(il));
        if (il + 1) % INTERVAL as usize == 0 {
            out.extend(full_tensors(il));
        } else {
            out.extend(ssm_tensors(il));
        }
        if il == PLE_LAYER {
            out.extend(ple_tensors(il));
        }
    }
    out
}

/// 合成 GGUF を書いて開く。`zero_ple` は PLE 表を 0 で埋める。
fn open_synth(dir: &TempDir, name: &str, zero_ple: bool) -> GgufFile {
    let path = dir.path(name);
    write_gguf(
        &path,
        &synth_kvs(DEFAULT_LAYERS),
        &synth_tensors(DEFAULT_LAYERS),
        zero_ple,
    );
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// 0. 実モデルのメタデータでは導出値が `GdnParams::qwen4exp` と一致する
// ---------------------------------------------------------------------------

/// 目録 (`flashnext_meta.txt`) の実値。`gdn_params` が読む 7 フィールドの
/// 突き合わせにだけ使う (他のフィールドも目録の値で埋める)。
fn real_params() -> Qwen4ExpParams {
    let mut compress = vec![0u32; 48];
    for (il, r) in compress.iter_mut().enumerate() {
        if (il + 1) % 4 == 0 {
            *r = 4;
        }
    }
    Qwen4ExpParams {
        n_layer: 48,
        n_ctx: 262144,
        n_embd: 2560,
        n_head: 24,
        n_head_kv: 2,
        n_embd_head_k: 256,
        n_embd_head_v: 256,
        rms_eps: 1e-6,
        n_rot: 64,
        rope_sections: [11, 11, 10, 0],
        rope_freq_base: 1e7,
        n_expert: 512,
        n_expert_used: 10,
        n_ff_exp: 640,
        n_ff_shexp: 640,
        ssm_d_conv: 4,
        ssm_d_state: 128,
        ssm_n_group: 16,
        ssm_dt_rank: 48,
        ssm_d_inner: 6144,
        full_attn_interval: 4,
        n_hc: 4,
        hc_low_rank: 320,
        indexer_n_head: 4,
        indexer_head_size: 128,
        indexer_top_k: 2048,
        compress_ratios: compress,
        n_embd_per_layer: 160,
        ple_layers: vec![1],
        ple_ngram_size: 3,
        ple_heads_per_ngram: 8,
        ple_conv_kernel: 4,
        ple_eos_token_id: 248044,
        ple_image_token_id: 248056,
        ple_layer_multipliers: vec![23703573157769, 20109073645365, 8052911324071],
        ple_head_offsets: vec![
            0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
            180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
        ],
        ple_head_vocab_sizes: vec![
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        ],
        n_vocab: 248320,
    }
}

/// F2 まで `SsmLayer::from_gguf` が使っていた固定値 (`GdnParams::qwen4exp`) と、
/// メタデータからの導出値が**同じ**であること。ここがずれると実モデルの
/// ssm 層が静かに別の次元で回る。
#[test]
fn gdn_params_of_real_metadata_matches_qwen4exp_preset() {
    let p = real_params();
    let g = gdn_params(&p).expect("gdn_params");
    let want = GdnParams::qwen4exp();

    assert_eq!(g.n_embd, want.n_embd, "n_embd");
    assert_eq!(g.n_k_heads, want.n_k_heads, "n_k_heads");
    assert_eq!(g.n_v_heads, want.n_v_heads, "n_v_heads");
    assert_eq!(g.head_k_dim, want.head_k_dim, "head_k_dim");
    assert_eq!(g.head_v_dim, want.head_v_dim, "head_v_dim");
    assert_eq!(g.d_conv, want.d_conv, "d_conv");
    assert_eq!(g.eps.to_bits(), want.eps.to_bits(), "eps");

    assert_eq!(g.key_dim(), 2048);
    assert_eq!(g.value_dim(), 6144);
    assert_eq!(g.conv_dim(), 10240);
    assert_eq!(g.ssm_state_len(), 48 * 128 * 128);
    assert_eq!(g.conv_state_len(), 10240 * 3);
}

// ---------------------------------------------------------------------------
// 1. 層の振り分け
// ---------------------------------------------------------------------------

#[test]
fn ssm_attn_assigns_ssm_and_full_layers() {
    let dir = TempDir::new("kinds");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    assert_eq!(p.n_layer, DEFAULT_LAYERS);
    assert_eq!(p.full_attn_interval, INTERVAL as usize);

    let attn = SsmAttn::from_gguf(&f, &p).expect("SsmAttn::from_gguf");
    assert_eq!(
        attn.kinds(),
        vec![
            LayerKind::Ssm,
            LayerKind::Ssm,
            LayerKind::Ssm,
            LayerKind::FullAttn
        ],
        "interval 4 の 4 層は 0/1/2 が ssm、3 が full attention"
    );
    assert_eq!(attn.n_ssm(), 3);
    assert_eq!(attn.n_full(), 1);
    assert_eq!(attn.n_ssm(), p.n_ssm_layers());
    assert_eq!(attn.n_full(), p.n_full_attn_layers());

    // ssm 層はメタデータから導出した次元で組まれている
    let l0 = attn.ssm_layer(0).expect("層 0 は ssm");
    assert_eq!(l0.p.n_embd, N_EMBD as usize);
    assert_eq!(l0.p.n_k_heads, SSM_GROUP as usize);
    assert_eq!(l0.p.n_v_heads, SSM_DT as usize);
    assert_eq!(l0.p.head_k_dim, SSM_STATE as usize);
    assert_eq!(l0.p.head_v_dim, (SSM_INNER / SSM_DT) as usize);
    assert_eq!(l0.p.d_conv, SSM_CONV as usize);
    assert_eq!(l0.qkv_split(), (KEY_DIM as usize, 2 * KEY_DIM as usize, CONV_DIM as usize));
    assert!(attn.ssm_layer(3).is_none(), "層 3 は full attention");
    assert!(attn.ssm_layer(1).is_some() && attn.ssm_layer(2).is_some());
}

// ---------------------------------------------------------------------------
// 2. 4 層の層別 trace (有限・隣接層で変化)
// ---------------------------------------------------------------------------

#[test]
fn forward_token_traces_four_layers_and_all_finite() {
    let dir = TempDir::new("trace");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    let attn = SsmAttn::from_gguf(&f, &p).expect("SsmAttn");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("with_attn");

    let mut norms: Vec<f32> = Vec::new();
    let mut heads: Vec<Vec<f32>> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    let logits = m
        .forward_token_with(&[1, 2, 3], |il, res| {
            let mut ss = 0.0f64;
            for &v in res {
                if !v.is_finite() {
                    return Err(Qwen4ExpError(vec![format!(
                        "層 {il}: res_hc に NaN/inf が出た ({v})"
                    )]));
                }
                ss += (v as f64) * (v as f64);
            }
            let norm = ss.sqrt() as f32;
            let head = res[..4].to_vec();
            eprintln!("LAYER {il:2} norm={norm:.9e} head={head:?}");
            seen.push(il);
            norms.push(norm);
            heads.push(head);
            Ok(())
        })
        .expect("forward_token_with");

    assert_eq!(seen, (0..DEFAULT_LAYERS).collect::<Vec<_>>());
    assert!(
        norms.iter().all(|v| v.is_finite() && *v > 0.0),
        "res_hc のノルムが有限でない: {norms:?}"
    );
    // 隣接層で値が変わっている (恒等でない)
    for il in 1..DEFAULT_LAYERS {
        assert_ne!(
            bits(&heads[il]),
            bits(&heads[il - 1]),
            "層 {il} の res_hc が層 {} と同値 (層が効いていない)",
            il - 1
        );
    }
    assert_eq!(logits.len(), N_VOCAB as usize);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logits[{i}] が有限でない: {v}");
    }
    let first = logits[0].to_bits();
    assert!(
        logits.iter().any(|v| v.to_bits() != first),
        "logits が全要素同値 (計算が効いていない)"
    );
}

// ---------------------------------------------------------------------------
// 3. 履歴を進めない設定 (新しい SsmAttn) ではビット一致
// ---------------------------------------------------------------------------

#[test]
fn fresh_ssm_state_is_bit_identical() {
    let dir = TempDir::new("fresh");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    let ctx = [1u32, 2, 3];

    let mut a = Qwen4Exp::with_attn(&f, Box::new(SsmAttn::from_gguf(&f, &p).expect("attn a")))
        .expect("model a");
    let mut b = Qwen4Exp::with_attn(&f, Box::new(SsmAttn::from_gguf(&f, &p).expect("attn b")))
        .expect("model b");

    let la = a.forward_token(&ctx).expect("a");
    let lb = b.forward_token(&ctx).expect("b");
    assert_eq!(la.len(), lb.len());
    assert_eq!(
        bits(&la),
        bits(&lb),
        "状態を進めていない 2 つの SsmAttn で logits が違う"
    );
}

// ---------------------------------------------------------------------------
// 4. 同じ SsmAttn で 2 回呼ぶと 2 回目が変わる (履歴が効いている)
// ---------------------------------------------------------------------------

#[test]
fn second_call_with_same_input_differs() {
    let dir = TempDir::new("hist");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(SsmAttn::from_gguf(&f, &p).expect("attn")))
        .expect("model");
    let ctx = [1u32, 2, 3];

    let first = m.forward_token(&ctx).expect("1 回目");
    let second = m.forward_token(&ctx).expect("2 回目");
    assert_eq!(first.len(), second.len());
    assert!(
        bits(&first) != bits(&second),
        "同じ入力で 2 回目がビット一致 (conv / ssm の履歴が効いていない)"
    );
    assert!(second.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// 5. IdentityAttn と SsmAttn で出力が違う
// ---------------------------------------------------------------------------

#[test]
fn ssm_attn_differs_from_identity() {
    let dir = TempDir::new("ident");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    let ctx = [1u32, 2, 3];

    let mut ident = Qwen4Exp::with_attn(&f, Box::new(IdentityAttn)).expect("identity");
    let mut ssm = Qwen4Exp::with_attn(&f, Box::new(SsmAttn::from_gguf(&f, &p).expect("ssm")))
        .expect("ssm model");

    let li = ident.forward_token(&ctx).expect("identity");
    let ls = ssm.forward_token(&ctx).expect("ssm");
    assert_eq!(li.len(), ls.len());
    assert!(
        bits(&li) != bits(&ls),
        "SsmAttn と IdentityAttn の logits が同じ (ssm が効いていない)"
    );
}

// ---------------------------------------------------------------------------
// 6. PLE 層 (il=1) の ple_emb が非ゼロで、層 1 の出口に効いている
// ---------------------------------------------------------------------------

#[test]
fn ple_embedding_is_nonzero_and_changes_layer_one() {
    let dir = TempDir::new("ple");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    assert!(p.is_ple(PLE_LAYER), "層 {PLE_LAYER} が PLE 層でない");

    // 6a. 表の gather が非ゼロを返す (合成表に 0 以外が入っている)
    let tr = f.tensor_ref(PLE_TABLE_NAME).expect("PLE 表が無い");
    let table = PleTable::open(&tr, &f).expect("PleTable::open");
    let ngram = PleNgram::from_meta(&f).expect("PleNgram::from_meta");
    assert_eq!(ngram.n_heads(), ((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM) as usize);
    let emb = ngram.gather(&table, &[1, 2, 3]).expect("gather");
    assert_eq!(emb.len(), N_EMBD as usize, "ple_emb の長さが n_embd と違う");
    assert!(
        emb.iter().any(|&v| v != 0.0),
        "ple_emb が全要素 0 (合成表が 0 埋め)"
    );
    assert!(emb.iter().all(|v| v.is_finite()));

    // 6b. 同じ構造で PLE 表だけ 0 にしたモデルと比べ、層 1 の出口が変わる
    let fz = open_synth(&dir, "model_zero_ple.gguf", true);
    let pz = validate(&fz).expect("validate (zero ple)");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(SsmAttn::from_gguf(&f, &p).expect("attn")))
        .expect("model");
    let mut mz = Qwen4Exp::with_attn(
        &fz,
        Box::new(SsmAttn::from_gguf(&fz, &pz).expect("attn zero")),
    )
    .expect("model zero ple");

    let mut res = Vec::new();
    m.forward_token_with(&[1, 2, 3], |il, r| {
        res.push((il, r.to_vec()));
        Ok(())
    })
    .expect("forward");
    let mut res_z = Vec::new();
    mz.forward_token_with(&[1, 2, 3], |il, r| {
        res_z.push((il, r.to_vec()));
        Ok(())
    })
    .expect("forward zero ple");

    assert_eq!(res.len(), DEFAULT_LAYERS);
    assert_eq!(res_z.len(), DEFAULT_LAYERS);
    // 層 0 は PLE を持たないので一致する (差は PLE の層だけ)
    assert_eq!(
        bits(&res[0].1),
        bits(&res_z[0].1),
        "層 0 は PLE を持たないのに差が出た"
    );
    assert!(
        bits(&res[PLE_LAYER].1) != bits(&res_z[PLE_LAYER].1),
        "層 {PLE_LAYER} の res_hc が PLE 表の有無で変わらない (PLE が効いていない)"
    );
}

// ---------------------------------------------------------------------------
// 7. 範囲外の層はエラー
// ---------------------------------------------------------------------------

#[test]
fn forward_rejects_out_of_range_layer() {
    let dir = TempDir::new("range");
    let f = open_synth(&dir, "model.gguf", false);
    let p = validate(&f).expect("validate");
    let mut attn = SsmAttn::from_gguf(&f, &p).expect("SsmAttn");
    let cur = vec![0.0f32; N_EMBD as usize];
    let errs = match attn.forward(DEFAULT_LAYERS, &cur) {
        Ok(_) => panic!("範囲外の層で Ok が返った"),
        Err(Qwen4ExpError(errs)) => errs,
    };
    assert_eq!(errs.len(), 1);
    assert!(errs[0].contains("範囲外"), "{}", errs[0]);
    assert!(errs[0].contains(&DEFAULT_LAYERS.to_string()), "{}", errs[0]);
}
