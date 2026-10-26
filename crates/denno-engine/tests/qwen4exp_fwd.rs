//! P-MOE-F1: `Qwen4Exp` (1 トークン decode の骨格) のテスト。
//!
//! 実 GGUF (111 GB) はこのコンテナに無いので開かない。同じ構造で小さい次元の
//! 合成 GGUF を一時ディレクトリに書き、`Qwen4Exp::new` の検査と 1 トークンの
//! forward を通す。
//!
//! `DENNO_Q4_MODEL=<分割 GGUF の先頭>` を設定したときだけ `--ignored` で
//! 実モデルを 1 トークン走らせる (`real_model_smoke_one_token`)。実行は
//! メインセッションが行う。
//!
//! 数値の正しさは F1 では判定しない。ここで見るのは
//! 「形が通る・NaN が出ない・決定性がある」の 3 点だけ。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::hyper::{self, HyperParams, HyperStreams};
use denno_engine::model::qwen4exp::{ple_window, LayerAttn, Qwen4Exp, Qwen4ExpError};

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
/// - norm の gamma (`*norm.weight`) は 1.0 (変換器が `1 + w` に畳んだ前提の値)
/// - F32 のその他は ±0.08 程度
/// - 量子型はスケール欄だけ固定し、量子値はブロックごとに変える
///   (全行同一にすると `output` の logits が全部同値になり、経路の生死が分からない)
fn tensor_bytes(name: &str, dims: &[u64], ty: u32, seed: u32) -> Vec<u8> {
    let nb = nbytes(dims, ty) as usize;
    let mut b: Vec<u8> = (0..nb).map(|i| pat(i, 7 + seed)).collect();
    match ty {
        TY_F32 => {
            let is_gamma = name.ends_with("norm.weight");
            for i in 0..nb / 4 {
                let v = if is_gamma {
                    1.0f32
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
                    // ブロックごとに量子値をずらす (行ごとに違う値になる)
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
                b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
            }
        }
        TY_IQ4_NL => {
            for blk in 0..nb / 18 {
                let o = blk * 18;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
            }
        }
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
    b
}

/// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列、テンソルは与えた順に並ぶ。
fn write_gguf(path: &Path, kvs: &[(String, Kv)], tensors: &[TensorSpec]) {
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
        out.extend_from_slice(&tensor_bytes(&t.name, &t.dims, t.ty, i as u32));
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
            "denno_qwen4exp_fwd_{tag}_{}_{n}",
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
const HEAD: u64 = 256;
const N_HEAD_KV: u64 = 1;
const N_EXPERT: u64 = 2;
const N_FF: u64 = 256;
const HC: u64 = 2;
const HC_LR: u64 = 32;
const SSM_CONV: u64 = 4;
const SSM_STATE: u64 = 32;
const SSM_GROUP: u64 = 2;
const SSM_DT: u64 = 2;
const SSM_INNER: u64 = 64;
const INTERVAL: u64 = 4;
const IDX_HEAD: u64 = 2;
const IDX_DIM: u64 = 32;
const IDX_TOP_K: u64 = 8;
const RATIO: u32 = 2;
/// `n_ple_heads * n_embd_per_layer == n_embd` でなければ `ple_key` の入力長が
/// 合わない (実ファイルは 16 * 160 = 2560)。ここでは 4 * 64 = 256。
const PLE_HEADS_PER_NGRAM: u64 = 2;
const N_EMBD_PER_LAYER: u64 = 64;
const PLE_LAYER: usize = 1;
const PLE_NGRAM: u64 = 3;
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
        ("general.alignment".to_string(), Kv::U32(32)),
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
        ("qwen4exp.expert_used_count".to_string(), Kv::U32(1)),
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

/// 共通 16 本 (`blk.{il}.` の後ろ)。
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
            "per_layer_token_embd.weight",
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

fn write_synth(
    dir: &TempDir,
    name: &str,
    layers: usize,
    tensors: &[TensorSpec],
) -> PathBuf {
    let path = dir.path(name);
    write_gguf(&path, &synth_kvs(layers), tensors);
    path
}

fn open_model(dir: &TempDir) -> (GgufFile, Qwen4Exp) {
    let path = write_synth(dir, "model.gguf", DEFAULT_LAYERS, &synth_tensors(DEFAULT_LAYERS));
    let f = GgufFile::open(&path).expect("open synth");
    let m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
    (f, m)
}

// ---------------------------------------------------------------------------
// 1. hc_mix の既知値
// ---------------------------------------------------------------------------

fn hp() -> HyperParams {
    HyperParams {
        hc: 4,
        n_embd: 3,
        low_rank: 2,
        eps: 1e-6,
    }
}

/// 全ストリーム同一なら、`mixed = sigmoid(gate) * mean(xn)` になる。
///
/// `w_down = 0` なら `lo = silu(0) = 0`、`gate = 0`、`sigmoid(0) = 0.5` なので
/// **mixed は平均の 0.5 倍**。手計算の既知値として `0.5 * xn[d]` と
/// `to_bits` で比べる (`hc_mixed` は `s` の昇順に足してから `1/hc` を掛ける)。
#[test]
fn hc_mix_identical_streams_is_half_mean() {
    let p = hp();
    let v = [0.5f32, -1.5, 2.0];
    let x = HyperStreams::from_embed(&v, &p);

    // 全ストリーム同一なら mean は元のベクトルそのもの (s の昇順加算 → 1/hc)
    assert_eq!(x.mean(), v, "mean が元のベクトルと一致しない");

    let w_norm = vec![1.0f32; p.hc * p.n_embd];
    let w_down = vec![0.0f32; p.low_rank * p.hc * p.n_embd];
    let w_up = vec![0.0f32; p.hc * p.n_embd * p.low_rank];
    let (mixed, inject) = hyper::hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);

    // inject は down_inject が無いので 0
    assert_eq!(inject, vec![0.0f32; p.hc]);

    // 手計算: xn = rmsnorm(v, 1.0) (二乗和 / n_embd → +eps → rsqrt → *v * 1)
    let ss: f32 = v.iter().map(|a| a * a).sum::<f32>() / p.n_embd as f32;
    let inv = 1.0f32 / (ss + p.eps).sqrt();
    for d in 0..p.n_embd {
        let xn = v[d] * inv * 1.0;
        // hc 本ぶん xn * sigmoid(0) を s の昇順に足して 1/hc
        let mut acc = xn * 0.5f32;
        for _ in 1..p.hc {
            acc += xn * 0.5f32;
        }
        let want = acc * (1.0f32 / p.hc as f32);
        assert_eq!(
            mixed[d].to_bits(),
            want.to_bits(),
            "mixed[{d}]: {} != {want}",
            mixed[d]
        );
    }
}

/// 全ストリーム同一 + `w_up` の行がストリーム間で同一なら、gate はストリームに
/// 依らず `mixed[d] = xn[d] * sigmoid(gate[d])` になる (構造の確認)。
#[test]
fn hc_mix_identical_streams_uses_one_gate() {
    let p = hp();
    let v = [0.25f32, -0.75, 1.0];
    let x = HyperStreams::from_embed(&v, &p);
    let w_norm = vec![1.0f32; p.hc * p.n_embd];

    // down / up に非ゼロを入れる (行優先: [out][in])
    let hc_dim = p.hc * p.n_embd;
    let mut w_down = vec![0.0f32; p.low_rank * hc_dim];
    for r in 0..p.low_rank {
        for i in 0..hc_dim {
            w_down[r * hc_dim + i] = 0.05 * ((r + i) as f32) - 0.1;
        }
    }
    // up は **ストリームごとに同じ行** にする (row s*n_embd+d は d だけで決まる)
    let mut w_up = vec![0.0f32; hc_dim * p.low_rank];
    for s in 0..p.hc {
        for d in 0..p.n_embd {
            let r = s * p.n_embd + d;
            for i in 0..p.low_rank {
                w_up[r * p.low_rank + i] = 0.1 * ((d + 2 * i) as f32) - 0.2;
            }
        }
    }
    let (mixed, _) = hyper::hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);

    // 同じ計算を独立に書き下す (xn → lo → gate → mixed)
    let ss: f32 = v.iter().map(|a| a * a).sum::<f32>() / p.n_embd as f32;
    let inv = 1.0f32 / (ss + p.eps).sqrt();
    let xn: Vec<f32> = (0..hc_dim).map(|i| v[i % p.n_embd] * inv).collect();
    let mut lo = vec![0.0f32; p.low_rank];
    for r in 0..p.low_rank {
        let mut acc = 0.0f32;
        for i in 0..hc_dim {
            acc += w_down[r * hc_dim + i] * xn[i];
        }
        let z = acc * (1.0f32 / p.hc as f32);
        lo[r] = z * (1.0 / (1.0 + (-z).exp()));
    }
    for d in 0..p.n_embd {
        let mut g = 0.0f32;
        for i in 0..p.low_rank {
            g += w_up[d * p.low_rank + i] * lo[i];
        }
        let sig = 1.0 / (1.0 + (-g).exp());
        let mut acc = xn[d] * sig;
        for s in 1..p.hc {
            acc += xn[s * p.n_embd + d] * sig;
        }
        let want = acc * (1.0f32 / p.hc as f32);
        assert_eq!(mixed[d].to_bits(), want.to_bits(), "mixed[{d}]");
    }
}

// ---------------------------------------------------------------------------
// 2. hc_combine
// ---------------------------------------------------------------------------

/// `inject = 0` なら `w = 2 * sigmoid(0) = 1` なので素の加算になる。
#[test]
fn hc_combine_zero_inject_is_plain_add() {
    let p = hp();
    let res = HyperStreams {
        data: (0..p.hc * p.n_embd)
            .map(|i| (i as f32) * 0.25 - 1.0)
            .collect(),
        hc: p.hc,
        n_embd: p.n_embd,
    };
    let block: Vec<f32> = (0..p.n_embd).map(|i| (i as f32) * 0.5 + 0.125).collect();
    let inject = vec![0.0f32; p.hc];

    let out = hyper::hc_combine(&res, &block, &inject, &p);
    assert_eq!(out.data.len(), res.data.len());
    for s in 0..p.hc {
        for (d, &b) in block.iter().enumerate() {
            let i = s * p.n_embd + d;
            let want = res.data[i] + b;
            assert_eq!(
                out.data[i].to_bits(),
                want.to_bits(),
                "stream {s} dim {d}"
            );
        }
    }
}

/// inject が非ゼロなら `2 * sigmoid(inject[s])` が掛かる (係数が 1 でないこと)。
#[test]
fn hc_combine_scales_each_stream() {
    let p = hp();
    let res = HyperStreams {
        data: vec![0.0f32; p.hc * p.n_embd],
        hc: p.hc,
        n_embd: p.n_embd,
    };
    let block = vec![1.0f32; p.n_embd];
    let inject = vec![0.5f32, 1.0, -1.0, 2.0];
    let out = hyper::hc_combine(&res, &block, &inject, &p);
    for (s, &inj) in inject.iter().enumerate() {
        let w = 2.0f32 * (1.0 / (1.0 + (-inj).exp()));
        assert!((w - 1.0).abs() > 1e-6, "stream {s} の係数が 1 のまま");
        for d in 0..p.n_embd {
            let want = 0.0f32 + 1.0f32 * w;
            assert_eq!(out.data[s * p.n_embd + d].to_bits(), want.to_bits());
        }
    }
}

// ---------------------------------------------------------------------------
// 3. grouped_rms_norm
// ---------------------------------------------------------------------------

/// gamma を掛けたあと、各ストリームの RMS が 1 になること。
#[test]
fn grouped_rms_norm_makes_stream_rms_one() {
    let p = HyperParams {
        hc: 4,
        n_embd: 8,
        low_rank: 2,
        eps: 1e-6,
    };
    let data: Vec<f32> = (0..p.hc * p.n_embd)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0 + 0.1 * (i as f32))
        .collect();
    let x = HyperStreams {
        data,
        hc: p.hc,
        n_embd: p.n_embd,
    };
    let gamma = vec![1.0f32; p.hc * p.n_embd];
    let xn = hyper::grouped_rms_norm(&x, &gamma, &p);
    assert_eq!(xn.len(), p.hc * p.n_embd);
    for s in 0..p.hc {
        let row = &xn[s * p.n_embd..(s + 1) * p.n_embd];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / p.n_embd as f32;
        let rms = ss.sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-5,
            "stream {s}: rms {rms} が 1 でない"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. 形状検査 (欠けたら Err)
// ---------------------------------------------------------------------------

#[test]
fn new_rejects_missing_tensor() {
    let dir = TempDir::new("missing");
    let all = synth_tensors(DEFAULT_LAYERS);
    let target = "blk.2.hc_ffn_up.weight";
    let tensors: Vec<TensorSpec> = all.into_iter().filter(|t| t.name != target).collect();
    let path = write_synth(&dir, "missing.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");
    match Qwen4Exp::new(&f) {
        Ok(_) => panic!("1 本欠けたのに Ok が返った"),
        Err(Qwen4ExpError(errs)) => {
            assert_eq!(errs.len(), 1, "エラーは 1 件のはず: {errs:?}");
            assert!(
                errs[0].contains(target),
                "エラーにテンソル名が無い: {}",
                errs[0]
            );
            assert!(
                errs[0].contains("見つからない"),
                "エラーが欠落を指していない: {}",
                errs[0]
            );
        }
    }
}

#[test]
fn new_rejects_wrong_shape() {
    let dir = TempDir::new("shape");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    // hc_attn_down の hc_lr を 1 つ減らす (形状だけ壊す)
    for t in tensors.iter_mut() {
        if t.name == "blk.0.hc_attn_down.weight" {
            t.dims = vec![HC_DIM, HC_LR - 1];
        }
    }
    let path = write_synth(&dir, "shape.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");
    let errs = match Qwen4Exp::new(&f) {
        Ok(_) => panic!("形状違いなのに Ok が返った"),
        Err(Qwen4ExpError(errs)) => errs,
    };
    assert_eq!(errs.len(), 1, "エラーは 1 件のはず: {errs:?}");
    assert!(errs[0].contains("blk.0.hc_attn_down.weight"), "{}", errs[0]);
    assert!(errs[0].contains("形状"), "{}", errs[0]);
}

// ---------------------------------------------------------------------------
// 5. forward_token (決定性・形・有限性)
// ---------------------------------------------------------------------------

#[test]
fn forward_token_is_deterministic() {
    let dir = TempDir::new("det");
    let (_f, mut m) = open_model(&dir);
    let ctx = [1u32, 2, 3];

    let a = m.forward_token(&ctx).expect("1 回目");
    let b = m.forward_token(&ctx).expect("2 回目");
    assert_eq!(a.len(), N_VOCAB as usize, "logits の長さが n_vocab と違う");
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logits[{i}] が 2 回で違う");
        assert!(x.is_finite(), "logits[{i}] が有限でない: {x}");
    }
    // 全部同じ値ではないこと (経路が死んでいない確認)
    let first = a[0].to_bits();
    assert!(
        a.iter().any(|v| v.to_bits() != first),
        "logits が全要素同値 (計算が効いていない)"
    );
}

#[test]
fn forward_token_rejects_empty_ctx() {
    let dir = TempDir::new("empty");
    let (_f, mut m) = open_model(&dir);
    let errs = match m.forward_token(&[]) {
        Ok(_) => panic!("空の ctx で Ok が返った"),
        Err(Qwen4ExpError(errs)) => errs,
    };
    assert_eq!(errs.len(), 1);
    assert!(errs[0].contains("ctx"), "{}", errs[0]);
}

/// attention の差し替え口が**全層で呼ばれる**こと (恒等でも経路が通る証明)。
#[test]
fn layer_attn_hook_is_called_for_every_layer() {
    use std::sync::{Arc, Mutex};

    struct Recorder {
        seen: Arc<Mutex<Vec<usize>>>,
    }
    impl LayerAttn for Recorder {
        fn forward(&mut self, layer: usize, cur: &[f32]) -> Result<Vec<f32>, Qwen4ExpError> {
            self.seen.lock().expect("lock").push(layer);
            Ok(cur.to_vec())
        }
    }

    let dir = TempDir::new("attn");
    let path = write_synth(
        &dir,
        "model.gguf",
        DEFAULT_LAYERS,
        &synth_tensors(DEFAULT_LAYERS),
    );
    let f = GgufFile::open(&path).expect("open");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut m = Qwen4Exp::with_attn(
        &f,
        Box::new(Recorder {
            seen: Arc::clone(&seen),
        }),
    )
    .expect("with_attn");

    let logits = m.forward_token(&[3]).expect("forward");
    assert_eq!(logits.len(), N_VOCAB as usize);
    let got = seen.lock().expect("lock").clone();
    assert_eq!(
        got,
        (0..DEFAULT_LAYERS).collect::<Vec<_>>(),
        "attention の差し替え口が全層で呼ばれていない"
    );
}

/// 各層の出口フックが全層ぶん呼ばれ、`res_hc` の長さが `hc_dim` であること。
#[test]
fn layer_hook_sees_every_layer_and_full_width() {
    let dir = TempDir::new("hook");
    let (_f, mut m) = open_model(&dir);
    let mut seen: Vec<usize> = Vec::new();
    let mut widths: Vec<usize> = Vec::new();
    m.forward_token_with(&[3], |il, res| {
        seen.push(il);
        widths.push(res.len());
        Ok(())
    })
    .expect("forward");
    assert_eq!(seen, (0..DEFAULT_LAYERS).collect::<Vec<_>>());
    assert!(
        widths.iter().all(|&w| w == HC_DIM as usize),
        "res_hc の長さが hc_dim でない: {widths:?}"
    );
}

/// PLE の層 (1) でだけ res_hc が変わること (PLE 注入が発火している証明)。
#[test]
fn ple_layer_changes_residual() {
    let dir = TempDir::new("ple");
    let (_f, mut m) = open_model(&dir);
    let mut norms: Vec<f32> = Vec::new();
    m.forward_token_with(&[1, 2, 3], |_il, res| {
        norms.push(res.iter().map(|v| v * v).sum::<f32>().sqrt());
        Ok(())
    })
    .expect("forward");
    assert_eq!(norms.len(), DEFAULT_LAYERS);
    // 層 1 の出口で値が動いている (層 0 と同値なら PLE も FFN も効いていない)
    assert!(norms[1] != norms[0], "層 1 で res_hc が変わっていない");
    assert!(norms.iter().all(|v| v.is_finite()), "res_hc が有限でない");
}

/// 合成重みでの `forward_token` 1 回の所要時間 (**参考値**。合否判定はしない)。
///
/// `cargo test --release --test qwen4exp_fwd -- --nocapture forward_token_timing`
/// で数値が出る。実モデルの所要時間とは比較できない (次元が 1/10・層 4 本)。
#[test]
fn forward_token_timing_reference() {
    let dir = TempDir::new("time");
    let (_f, mut m) = open_model(&dir);
    let ctx = [1u32, 2, 3];
    // 1 回目は PLE 表のページフォルトを含むので捨てる
    m.forward_token(&ctx).expect("warmup");
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        m.forward_token(&ctx).expect("forward");
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    eprintln!(
        "合成重み (n_embd={} hc={} 層 {} 本 expert {} top-1) で {:.3} us/token",
        m.p.n_embd, m.p.n_hc, m.p.n_layer, m.p.n_expert, per * 1e6
    );
    assert!(per > 0.0);
}

// ---------------------------------------------------------------------------
// 実モデルのスモーク (--ignored。実行はメインセッション)
// ---------------------------------------------------------------------------
/// `DENNO_Q4_MODEL` が指す分割 GGUF から 1 トークン走らせ、層ごとの
/// `||res_hc||` を stderr に出す。NaN / inf が出たらその層番号で落ちる。
#[test]
#[ignore = "実モデル (111 GB) を使う。DENNO_Q4_MODEL を設定して --ignored で実行する"]
fn real_model_smoke_one_token() {
    let Some(first) = std::env::var_os("DENNO_Q4_MODEL").map(PathBuf::from) else {
        eprintln!("DENNO_Q4_MODEL が未設定なのでスキップ");
        return;
    };
    let parts = split_parts(&first);
    eprintln!("GGUF パート {} 個: {:?}", parts.len(), parts);
    let f = if parts.len() > 1 {
        GgufFile::open_split(&parts)
    } else {
        GgufFile::open(&first)
    }
    .expect("GGUF を開けない");

    let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new (形状検査)");
    eprintln!(
        "params: n_layer={} n_embd={} hc={} hc_dim={} n_expert={} top_k={} n_vocab={}",
        m.p.n_layer,
        m.p.n_embd,
        m.p.n_hc,
        m.p.hc_dim(),
        m.p.n_expert,
        m.p.n_expert_used,
        m.p.n_vocab
    );

    // 現在トークンは語彙内なら何でもよい (数値は F4 で判定する)
    let tok = 1u32.min((m.p.n_vocab as u32).saturating_sub(1));
    let ctx = [tok];

    let t0 = std::time::Instant::now();
    let mut n_norm = 0usize;
    let logits = m
        .forward_token_with(&ctx, |il, res| {
            let mut ss = 0.0f64;
            for &v in res {
                if !v.is_finite() {
                    return Err(Qwen4ExpError(vec![format!(
                        "層 {il}: res_hc に NaN/inf が出た ({v})"
                    )]));
                }
                ss += (v as f64) * (v as f64);
            }
            n_norm += 1;
            eprintln!("層 {il:2}: ||res_hc|| = {:.6e}", ss.sqrt());
            Ok(())
        })
        .expect("forward_token");
    let dt = t0.elapsed();

    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logits[{i}] が有限でない: {v}");
        if logits[i] > logits[best] {
            best = i;
        }
    }
    eprintln!(
        "logits: n={} argmax={} max={:.6e}",
        logits.len(),
        best,
        logits[best]
    );
    eprintln!("所要: {:.3} s/token (層 {n_norm} 本ぶんのノルム出力込み)", dt.as_secs_f64());
}

/// `<prefix>-00001-of-00006.gguf` から全パートのパスを組み立てる。
fn split_parts(first: &Path) -> Vec<PathBuf> {
    let Some(name) = first.file_name().and_then(|n| n.to_str()) else {
        return vec![first.to_path_buf()];
    };
    let Some(stem) = name.strip_suffix(".gguf") else {
        return vec![first.to_path_buf()];
    };
    let Some(dash) = stem.rfind('-') else {
        return vec![first.to_path_buf()];
    };
    let tail = &stem[dash + 1..];
    let Some((num, total)) = tail.split_once("-of-") else {
        return vec![first.to_path_buf()];
    };
    if num.len() != 5 || total.len() != 5 {
        return vec![first.to_path_buf()];
    }
    let Ok(n) = total.parse::<u32>() else {
        return vec![first.to_path_buf()];
    };
    let prefix = &stem[..dash];
    let dir = first.parent().unwrap_or_else(|| Path::new("."));
    (1..=n)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{n:05}.gguf")))
        .collect()
}


// ---------------------------------------------------------------------------
// PLE の n-gram 窓
// ---------------------------------------------------------------------------

/// `ple_window` は末尾から `n_gram` 個を **新しい順** で返す。
/// `PleNgram::indices` が `ctx[0]` を現在のトークンとして扱うため、ここを
/// 古い順のまま渡すと n-gram が 2 つずれた窓で引かれる (2026-10-26 に踏んだ)。
#[test]
fn ple_window_takes_the_tail_newest_first() {
    let ctx = [10u32, 11, 12, 13];
    assert_eq!(ple_window(&ctx, 3), vec![13, 12, 11]);
    assert_eq!(ple_window(&ctx, 1), vec![13]);
    assert_eq!(ple_window(&ctx, 8), vec![13, 12, 11, 10]);
    assert_eq!(ple_window(&[7u32], 3), vec![7]);
    assert_eq!(ple_window(&[], 3), Vec::<u32>::new());
}

/// 4 トークンの ctx でも、PLE が見るのは末尾 `n_gram` 個だけ。
/// 窓より前のトークンを変えても forward の結果は変わらない (窓を丸ごと渡していない証拠)。
#[test]
fn ple_only_reads_the_tail_window() {
    let dir = TempDir::new("ple_window");
    // 先頭だけ違う 4 トークン。末尾 3 個は同じ。
    let mut a = open_model(&dir).1;
    let mut b = open_model(&dir).1;
    let la = a.forward_token(&[1, 5, 6, 7]).expect("forward a");
    let lb = b.forward_token(&[3, 5, 6, 7]).expect("forward b");
    assert_eq!(la.len(), lb.len());
    let diff = la
        .iter()
        .zip(lb.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(diff, 0, "窓の外のトークンで {diff} 個の logit が変わった");

    // 窓の中を変えれば変わる (比較が効いていることの確認)
    let mut c = open_model(&dir).1;
    let lc = c.forward_token(&[3, 5, 6, 2]).expect("forward c");
    let same = la
        .iter()
        .zip(lc.iter())
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    assert!(same < la.len(), "窓の中を変えても logit が同じ");
}
