//! P-MOE-F5: `qwen4exp` の貪欲 decode 駆動部
//! ([`Qwen4ExpDecoder`]) の検査。
//!
//! 実 GGUF (111 GB) はこのコンテナに無いので開かない。F3 の
//! `tests/qwen4exp_layers.rs` と同じ組み方で `block 4 / ssm 3 層 + full 1 層` の
//! 合成 GGUF を一時ディレクトリに書き、それを `Qwen4Exp::with_attn` +
//! `SsmAttn::from_gguf` で回す。次元の縮め方は F3 の表のとおり
//! (`validate` が目録と同じ量子型を要求するので、`ne0` は型のブロック長の倍数)。
//!
//! ここで見るのは「駆動部の規則」だけ:
//! 本数・決定性・同値の解決・`max_ctx`・trace の回数と有限性・`tokens()` の順。
//! 数値の正しさ (実モデルとの一致) は F6 で判定する。
//!
//! 検査 4 (argmax の同値規則) の作り方: logits は外から触れないので、
//! 素の重みで 1 回 decode して**勝った行**を求め、その行を `output.weight` の
//! 0 番へバイト単位で複写した重みを作る。0 番と勝った行がビット同値の最大値に
//! なるので、添字の小さい方 (0 番) が選ばれなければならない。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::{validate, Qwen4Exp, Qwen4ExpParams};
use denno_engine::model::qwen4exp_decode::{
    argmax, DecoderConfig, Qwen4ExpDecoder, DEFAULT_PLE_WINDOW,
};
use denno_engine::model::qwen4exp_ssm::SsmAttn;
use denno_engine::ple::PLE_TABLE_NAME;

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
// 合成 GGUF の書き出し (F3 の `tests/qwen4exp_layers.rs` と同じ組み方)
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

/// `output.weight` の 1 行を別の行へバイト単位で複写する指定
/// (`(dst_row, src_row)`)。検査 4 (同値の解決) だけが使う。
type CopyRow = Option<(usize, usize)>;

/// テンソルの中身。**有限値だけ**を作る (スケール欄を乱数にすると NaN になる)。
fn tensor_bytes(
    name: &str,
    dims: &[u64],
    ty: u32,
    seed: u32,
    zero_ple: bool,
    copy_row: CopyRow,
) -> Vec<u8> {
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
                    0.05 * (((i % SSM_CONV as usize) + 1) as f32)
                } else if name.ends_with("ple_conv1d.weight") {
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
            // 行あたりのブロック数 (ne0 / 32)。`copy_row` はこの単位で行を写す。
            let bpr = (dims[0] / 32).max(1) as usize;
            for blk in 0..nb / 34 {
                let o = blk * 34;
                let src = match copy_row {
                    Some((dst, s)) if name == "output.weight" && blk / bpr == dst => {
                        s * bpr + blk % bpr
                    }
                    _ => blk,
                };
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                for j in 0..32u32 {
                    let q = ((j as i32 + src as i32 * 3 + seed as i32 * 7) % 11) - 5;
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

fn write_gguf(
    path: &Path,
    kvs: &[(String, Kv)],
    tensors: &[TensorSpec],
    zero_ple: bool,
    copy_row: CopyRow,
) {
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
        out.extend_from_slice(&tensor_bytes(&t.name, &t.dims, t.ty, i as u32, zero_ple, copy_row));
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
            "denno_qwen4exp_decode_{tag}_{}_{n}",
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
const N_EMBD_PER_LAYER: u64 = N_EMBD / ((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
const PLE_LAYER: usize = 1;
const PLE_CONV: u64 = 4;
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

/// 合成 GGUF を書いて開く。
fn open_synth(dir: &TempDir, name: &str, zero_ple: bool, copy_row: CopyRow) -> GgufFile {
    let path = dir.path(name);
    write_gguf(
        &path,
        &synth_kvs(DEFAULT_LAYERS),
        &synth_tensors(DEFAULT_LAYERS),
        zero_ple,
        copy_row,
    );
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

fn params(f: &GgufFile) -> Qwen4ExpParams {
    validate(f).expect("validate")
}

/// 合成 GGUF から decoder を組む (`SsmAttn` を attention として渡す)。
fn make_decoder(f: &GgufFile, max_ctx: usize) -> Qwen4ExpDecoder {
    let p = params(f);
    let attn = SsmAttn::from_gguf(f, &p).expect("SsmAttn::from_gguf");
    let m = Qwen4Exp::with_attn(f, Box::new(attn)).expect("Qwen4Exp::with_attn");
    Qwen4ExpDecoder::new(m, DecoderConfig::new(max_ctx))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 検査で使うプロンプト (3 トークン = PLE の n-gram 長)。
const PROMPT: [u32; 3] = [1, 2, 3];
/// 列が変わることを見るための別プロンプト (`PROMPT` とは別の固定点に落ちる)。
const PROMPT_B: [u32; 3] = [0, 0, 0];

// ---------------------------------------------------------------------------
// 0. 既定値
// ---------------------------------------------------------------------------

#[test]
fn decoder_config_defaults_to_ngram_window() {
    let cfg = DecoderConfig::new(64);
    assert_eq!(cfg.max_ctx, 64);
    assert_eq!(cfg.ple_window, DEFAULT_PLE_WINDOW);
    assert_eq!(cfg.ple_window, PLE_NGRAM as usize, "既定はメタデータの ngram_size");
}

// ---------------------------------------------------------------------------
// 1. decode_greedy が n_new 個ちょうどの id を返す
// ---------------------------------------------------------------------------

#[test]
fn decode_greedy_returns_exactly_n_new_ids() {
    let dir = TempDir::new("count");
    let f = open_synth(&dir, "model.gguf", false, None);
    let p = params(&f);
    let mut d = make_decoder(&f, 64);

    let ids = d.decode_greedy(&PROMPT, 5).expect("decode_greedy");
    assert_eq!(ids.len(), 5, "生成 id の本数が n_new と違う");
    assert!(
        ids.iter().all(|&t| (t as u64) < N_VOCAB),
        "語彙 ({N_VOCAB}) の外の id が出た: {ids:?}"
    );
    assert_eq!(d.tokens().len(), PROMPT.len() + 5);
    assert_eq!(p.n_layer, DEFAULT_LAYERS);
}

// ---------------------------------------------------------------------------
// 2. 同じプロンプトを 2 回で同じ列 (決定性)
// ---------------------------------------------------------------------------

#[test]
fn decode_greedy_is_deterministic() {
    let dir = TempDir::new("det");
    let f = open_synth(&dir, "model.gguf", false, None);

    // 別々に組んだ 2 つの decoder (状態を共有しない)
    let mut a = make_decoder(&f, 64);
    let mut b = make_decoder(&f, 64);
    let ia = a.decode_greedy(&PROMPT, 6).expect("a");
    let ib = b.decode_greedy(&PROMPT, 6).expect("b");
    assert_eq!(ia, ib, "同じプロンプトで列が違う (決定性が無い)");

    // 同じ decoder を組み直しても同じ (順序に依存した状態が残っていない)
    let mut c = make_decoder(&f, 64);
    let ic = c.decode_greedy(&PROMPT, 6).expect("c");
    assert_eq!(ia, ic, "decoder を組み直すと列が変わる");

    // 生成列だけでなく logits もビット一致する (丸め順序が入力に依らない)
    let mut l1 = make_decoder(&f, 64);
    let mut l2 = make_decoder(&f, 64);
    let a1 = l1.prefill(&PROMPT).expect("prefill a");
    let a2 = l2.prefill(&PROMPT).expect("prefill b");
    assert_eq!(bits(&a1), bits(&a2), "同じ入力で logits がビット一致しない");

    // 途中まで伸ばした列は、長く伸ばした列の接頭辞になっている
    let mut e = make_decoder(&f, 64);
    let short = e.decode_greedy(&PROMPT, 3).expect("short");
    assert_eq!(short[..], ia[..3], "n_new を変えると列の先頭が変わる");
}

// ---------------------------------------------------------------------------
// 3. 別のプロンプトでは別の列になる (定数に落ちていない)
// ---------------------------------------------------------------------------

#[test]
fn different_prompts_give_different_sequences() {
    let dir = TempDir::new("diff");
    let f = open_synth(&dir, "model.gguf", false, None);

    let mut a = make_decoder(&f, 64);
    let mut b = make_decoder(&f, 64);
    let ia = a.decode_greedy(&PROMPT, 6).expect("a");
    let ib = b.decode_greedy(&PROMPT_B, 6).expect("b");
    assert_ne!(ia, ib, "プロンプトを変えても同じ列 (入力が効いていない)");

    // 末尾 1 トークンだけ違うプロンプトでも logits はビット一致しない
    // (合成重みでは貪欲の列が同じ固定点に落ちることがあるので、ここは
    //  列ではなく logits で「入力が効いている」ことを見る)
    let mut c = make_decoder(&f, 64);
    let mut e = make_decoder(&f, 64);
    let la = c.prefill(&PROMPT).expect("prefill a");
    let lb = e.prefill(&[1, 2, 4]).expect("prefill b");
    assert_ne!(
        bits(&la),
        bits(&lb),
        "末尾 1 トークンの違いが logits に出ない"
    );
}

// ---------------------------------------------------------------------------
// 4. argmax の同値規則: 同点なら添字が小さい方
// ---------------------------------------------------------------------------

#[test]
fn argmax_tie_takes_lower_index() {
    let dir = TempDir::new("tie");

    // 4a. 素の重みで「勝つ行」を測る
    let plain = open_synth(&dir, "plain.gguf", false, None);
    let mut probe = make_decoder(&plain, 64);
    let logits = probe.prefill(&PROMPT).expect("prefill");
    assert_eq!(logits.len(), N_VOCAB as usize);
    let win = argmax(&logits).expect("argmax");
    assert_eq!(win, argmax(&logits).expect("argmax"), "argmax が決定的でない");

    // 4b. 勝った行を 0 番へバイト単位で複写する。
    //     0 番が勝ってしまうと「添字の小さい方が選ばれる」検査が空振りする
    //     (0 番が答えになるので規則が効いたか分からない) ので、明示的に落とす。
    assert_ne!(
        win, 0,
        "素の重みで 0 番が勝った。この重みでは同点規則の検査が空振りする"
    );
    let (dst, src) = (0usize, win);
    let tied = open_synth(&dir, "tied.gguf", false, Some((dst, src)));

    // 複写した 2 行は同じ重みなので、logits はビット同値になる
    let mut d = make_decoder(&tied, 64);
    let l2 = d.prefill(&PROMPT).expect("prefill tied");
    assert_eq!(
        l2[dst].to_bits(),
        l2[src].to_bits(),
        "複写した 2 行の logits がビット同値でない (dst={dst} src={src})"
    );
    assert!(
        l2[dst].is_finite(),
        "同点の logits が有限でない: {}",
        l2[dst]
    );
    assert!(
        logits[win] > logits[0],
        "素の重みで 0 番が勝っていない (win={win} の検査が空振り)"
    );

    // 同点の最大値が 2 つあるので、添字の小さい方 (0 番) が選ばれる
    assert_eq!(
        argmax(&l2).expect("argmax"),
        0,
        "同点のとき添字の大きい方が選ばれた (logits={l2:?})"
    );

    // 生成側でも 0 番が選ばれる
    let mut g = make_decoder(&tied, 64);
    let ids = g.decode_greedy(&PROMPT, 1).expect("decode_greedy");
    assert_eq!(ids[0], 0, "decode_greedy が同点で 0 番を選ばない: {ids:?}");
}

// ---------------------------------------------------------------------------
// 5. n_new == 0 で空
// ---------------------------------------------------------------------------

#[test]
fn decode_greedy_with_zero_new_is_empty() {
    let dir = TempDir::new("zero");
    let f = open_synth(&dir, "model.gguf", false, None);
    let mut d = make_decoder(&f, 64);

    let ids = d.decode_greedy(&PROMPT, 0).expect("n_new=0 でエラーになった");
    assert!(ids.is_empty(), "n_new=0 で {ids:?} が返った");
    assert!(
        d.tokens().is_empty(),
        "n_new=0 なのにトークンを食わせた: {:?}",
        d.tokens()
    );

    // traced 側も同じ
    let mut n = 0usize;
    let ids = d
        .decode_greedy_traced(&PROMPT, 0, |_, _, _, _| {
            n += 1;
            Ok(())
        })
        .expect("traced n_new=0");
    assert!(ids.is_empty());
    assert_eq!(n, 0, "n_new=0 なのにフックが {n} 回呼ばれた");
}

// ---------------------------------------------------------------------------
// 6. max_ctx 超過でエラー
// ---------------------------------------------------------------------------

#[test]
fn max_ctx_overflow_is_an_error() {
    let dir = TempDir::new("ctx");
    let f = open_synth(&dir, "model.gguf", false, None);

    // ちょうど max_ctx 本は通る
    let mut ok = make_decoder(&f, PROMPT.len());
    let ids = ok.decode_greedy(&PROMPT, 0).expect("max_ctx ちょうどは通るはず");
    assert!(ids.is_empty());
    let logits = ok.prefill(&PROMPT).expect("max_ctx ちょうど");
    assert_eq!(logits.len(), N_VOCAB as usize);

    // 1 本でも超えたらエラー (黙って古いトークンを捨てない)
    let mut over = make_decoder(&f, PROMPT.len() - 1);
    let errs = over
        .prefill(&PROMPT)
        .expect_err("max_ctx を超えたのに Ok が返った");
    let msg = errs.to_string();
    assert!(msg.contains("max_ctx"), "{msg}");
    assert!(msg.contains(&(PROMPT.len() - 1).to_string()), "{msg}");

    // 生成で超える場合もエラー
    let mut grow = make_decoder(&f, PROMPT.len() + 1);
    let errs = grow
        .decode_greedy(&PROMPT, 5)
        .expect_err("生成で max_ctx を超えたのに Ok が返った");
    assert!(errs.to_string().contains("max_ctx"), "{errs}");
}

// ---------------------------------------------------------------------------
// 7. decode_greedy_traced が n_new * n_layer 回フックを呼び、norm が有限
// ---------------------------------------------------------------------------

#[test]
fn traced_decode_calls_hooks_once_per_layer_per_step() {
    let dir = TempDir::new("trace");
    let f = open_synth(&dir, "model.gguf", false, None);
    let p = params(&f);
    let n_layer = p.n_layer;
    let n_new = 3usize;

    let mut d = make_decoder(&f, 64);
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let mut norms: Vec<f32> = Vec::new();
    let ids = d
        .decode_greedy_traced(&PROMPT, n_new, |step, il, norm, head| {
            assert!(
                norm.is_finite() && norm > 0.0,
                "step {step} 層 {il}: ||res_hc|| が有限でない ({norm})"
            );
            assert!(
                head.iter().all(|v| v.is_finite()),
                "step {step} 層 {il}: head に NaN/inf ({head:?})"
            );
            calls.push((step, il));
            norms.push(norm);
            Ok(())
        })
        .expect("decode_greedy_traced");

    assert_eq!(ids.len(), n_new);
    assert_eq!(
        calls.len(),
        n_new * n_layer,
        "フックの回数が n_new({n_new}) * n_layer({n_layer}) と違う: {}",
        calls.len()
    );
    // 生成 step の順に、各 step で層 0..n_layer が 1 回ずつ
    let want: Vec<(usize, usize)> = (0..n_new)
        .flat_map(|s| (0..n_layer).map(move |il| (s, il)))
        .collect();
    assert_eq!(calls, want, "フックの (step, 層) の順序が違う");
    assert_eq!(norms.len(), n_new * n_layer);
    // 隣接層でノルムが変わっている (同じ値を流していない)
    for s in 0..n_new {
        let row = &norms[s * n_layer..(s + 1) * n_layer];
        assert!(
            row.windows(2).all(|w| w[0].to_bits() != w[1].to_bits()),
            "step {s}: 隣接層で ||res_hc|| が同値 (層が効いていない): {row:?}"
        );
    }

    // フックを付けても生成列は decode_greedy と同じ (trace が数値を変えない)
    let mut plain = make_decoder(&f, 64);
    let plain_ids = plain.decode_greedy(&PROMPT, n_new).expect("plain");
    assert_eq!(ids, plain_ids, "trace の有無で生成列が変わる");
}

// ---------------------------------------------------------------------------
// 8. tokens() が「プロンプト + 生成」の順になっている
// ---------------------------------------------------------------------------

#[test]
fn tokens_are_prompt_then_generated() {
    let dir = TempDir::new("toks");
    let f = open_synth(&dir, "model.gguf", false, None);
    let mut d = make_decoder(&f, 64);
    assert!(d.tokens().is_empty(), "組み立て直後から tokens が空でない");

    // prefill だけ: プロンプトそのもの
    let mut pf = make_decoder(&f, 64);
    pf.prefill(&PROMPT).expect("prefill");
    assert_eq!(pf.tokens(), &PROMPT[..], "prefill 後の tokens がプロンプトと違う");

    let ids = d.decode_greedy(&PROMPT, 4).expect("decode_greedy");
    let mut want: Vec<u32> = PROMPT.to_vec();
    want.extend_from_slice(&ids);
    assert_eq!(d.tokens(), &want[..], "tokens() が「プロンプト + 生成」になっていない");

    // step を直接呼んでも末尾に積まれる
    let mut s = make_decoder(&f, 64);
    s.step(5).expect("step");
    s.step(6).expect("step");
    assert_eq!(s.tokens(), &[5u32, 6][..]);
}

// ---------------------------------------------------------------------------
// 9. 空のプロンプト・窓 0 はエラー (黙って別の意味にしない)
// ---------------------------------------------------------------------------

#[test]
fn empty_prompt_and_zero_window_are_errors() {
    let dir = TempDir::new("edge");
    let f = open_synth(&dir, "model.gguf", false, None);

    let mut d = make_decoder(&f, 64);
    let errs = d.prefill(&[]).expect_err("空のプロンプトで Ok が返った");
    assert!(errs.to_string().contains("空"), "{errs}");
    assert!(d.tokens().is_empty(), "失敗した prefill が tokens を残した");

    let p = params(&f);
    let attn = SsmAttn::from_gguf(&f, &p).expect("SsmAttn");
    let m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("with_attn");
    let mut z = Qwen4ExpDecoder::new(
        m,
        DecoderConfig {
            max_ctx: 64,
            ple_window: 0,
        },
    );
    let errs = z.step(1).expect_err("ple_window=0 で Ok が返った");
    assert!(errs.to_string().contains("ple_window"), "{errs}");
    assert!(z.tokens().is_empty(), "失敗した step が tokens を残した");
}

