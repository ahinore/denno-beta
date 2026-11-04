//! P118: serve_flash のモデル再利用 (no-reuse 経路で reset する) の発火テスト。
//!
//! engine 側 `tests/qwen4exp_decode.rs` の合成 GGUF 書き出しと同じ組み方で、
//! トークナイザ KV (byte-level 256 語 + `<|im_start|>` / `<|im_end|>` +
//! chat_template) を足した 1 本の GGUF を書き、`Flash` を直接組んで
//! `generate()` を回す。`build_model` の呼び出し数は serve_flash.rs の
//! `TEST_BUILD_COUNT` (`cfg(test)` 専用カウンタ) で数える。
//!
//! このファイルは `#[cfg(test)]` のときだけコンパイルされる
//! (serve_flash.rs から `#[cfg(test)] #[path] mod flash_synth;` で include)。

use super::{
    model_reuse_enabled, prefix_miss_line, prefix_splice_enabled, Flash, GenReq, TEST_BUILD_COUNT,
    TEST_FAIL_STEP, TEST_LAST_REUSE,
};
use crate::flash_telemetry::Telemetry;
use denno_engine::gguf::GgufFile;
use denno_engine::tok_gguf::GgufTokenizer;
use denno_telemetry::Frame;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// 合成 GGUF の書き出し (tests/qwen4exp_decode.rs と同じ組み方)
// ---------------------------------------------------------------------------

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

// llama_token_type (tokenizer.ggml.token_type)
const TT_NORMAL: u32 = 1;
const TT_CONTROL: u32 = 3;

enum Kv {
    U32(u32),
    F32(f32),
    Str(String),
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
    ne / bs * type_size(ty) + if ne % bs == 0 { 0 } else { type_size(ty) }
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

/// テストで使う f16 定数 (0.0 / 0.001 / 0.02) のビット。
/// `half` クレートに依存しないための直書き (この 3 値しか使わない)。
fn f16_bytes(v: f32) -> [u8; 2] {
    let bits: u16 = if v == 0.0 {
        0x0000
    } else if v == 0.02 {
        0x251F // 2^-6 * (1 + 287/1024) ~= 0.020004
    } else if v == 0.001 {
        0x1419 // 2^-10 * (1 + 25/1024) ~= 0.001022
    } else {
        panic!("テストヘルパ未対応の f16 定数 {v}")
    };
    bits.to_le_bytes()
}

/// テンソルの中身。有限値だけを作る (スケール欄を乱数にすると NaN になる)。
fn tensor_bytes(name: &str, dims: &[u64], ty: u32, seed: u32) -> Vec<u8> {
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
            for blk in 0..nb / 18 {
                let o = blk * 18;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
            }
        }
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
    b
}

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

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_serve_flash_{tag}_{}_{n}",
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
// 小さな合成モデル (構造は目録どおり・次元だけ小さい) + トークナイザ
// ---------------------------------------------------------------------------

const N_EMBD: u64 = 256;
/// byte 256 語 + `<|im_start|>` / `<|im_end|>`。n_vocab は
/// `tokenizer.ggml.tokens` の要素数から決まる (validate) ので、
/// トークナイザの語彙数と必ず一致させる。
const N_VOCAB: u64 = 258;
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

/// Qwen 系 chat_template の縮小版 (minijinja で描画できる)。
///
/// task 131: 本番テンプレートと同じく content を `|trim` する。生成テキストの
/// 末尾の空白がここで落ちるのが prefix 不一致の原因なので、縮小版でも同じ
/// 挙動にしてテストで再現する。
const CHAT_TPL: &str = concat!(
    "{%- for m in messages %}",
    "{{- '<|im_start|>' + m.role + '\\n' + m.content|trim + '<|im_end|>\\n' }}",
    "{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
);

/// GPT-2 / Qwen の byte-level 写像 (byte → 表示文字)。
fn byte_level_char(b: u8) -> char {
    fn printable(x: u8) -> bool {
        (33..=126).contains(&x) || (161..=172).contains(&x) || (174..=255).contains(&x)
    }
    if printable(b) {
        return b as char;
    }
    let mut n = 0u32;
    for x in 0u8..=255 {
        if printable(x) {
            continue;
        }
        if x == b {
            return char::from_u32(256 + n).unwrap();
        }
        n += 1;
    }
    unreachable!()
}

fn synth_kvs(layers: usize) -> Vec<(String, Kv)> {
    let mut ratios = vec![0u32; layers];
    for (il, r) in ratios.iter_mut().enumerate() {
        if (il + 1) % INTERVAL as usize == 0 {
            *r = RATIO;
        }
    }
    vec![
        ("general.architecture".to_string(), Kv::Str("qwen4exp".into())),
        ("general.alignment".to_string(), Kv::U32(ALIGN as u32)),
        ("qwen4exp.block_count".to_string(), Kv::U32(layers as u32)),
        ("qwen4exp.context_length".to_string(), Kv::U32(1024)),
        ("qwen4exp.embedding_length".to_string(), Kv::U32(N_EMBD as u32)),
        ("qwen4exp.attention.head_count".to_string(), Kv::U32(N_HEAD as u32)),
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
        ("qwen4exp.expert_count".to_string(), Kv::U32(N_EXPERT as u32)),
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
        ("qwen4exp.ssm.state_size".to_string(), Kv::U32(SSM_STATE as u32)),
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
        ("qwen4exp.ple.ngram_size".to_string(), Kv::U32(PLE_NGRAM as u32)),
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
    ]
}

fn tok_kvs() -> Vec<(String, Kv)> {
    let mut tokens: Vec<String> = (0u8..=255).map(|b| byte_level_char(b).to_string()).collect();
    tokens.push("<|im_start|>".into()); // id 256
    tokens.push("<|im_end|>".into()); // id 257
    let mut types: Vec<u32> = vec![TT_NORMAL; 256];
    types.push(TT_CONTROL);
    types.push(TT_CONTROL);
    vec![
        ("tokenizer.ggml.model".into(), Kv::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Kv::Str("qwen35".into())),
        ("tokenizer.ggml.tokens".into(), Kv::ArrStr(tokens)),
        ("tokenizer.ggml.merges".into(), Kv::ArrStr(vec![])),
        ("tokenizer.ggml.token_type".into(), Kv::ArrU32(types)),
        ("tokenizer.ggml.bos_token_id".into(), Kv::U32(256)),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(257)),
        ("tokenizer.chat_template".into(), Kv::Str(CHAT_TPL.into())),
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
            denno_engine::ple::PLE_TABLE_NAME,
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

// ---------------------------------------------------------------------------
// Flash を組んで generate() を回す
// ---------------------------------------------------------------------------

/// 合成 GGUF (モデル + トークナイザ) を書いて `Flash` を組む。
fn build_flash(dir: &TempDir) -> Flash {
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    kvs.extend(tok_kvs());
    let path = dir.path("flash.gguf");
    write_gguf(&path, &kvs, &synth_tensors(DEFAULT_LAYERS));
    let f = GgufFile::open(&path).expect("合成 GGUF を開けない");
    let tok = GgufTokenizer::from_gguf(&f).expect("トークナイザを組めない");
    let (telem, _) = Telemetry::new(
        false,
        Frame::ServerInfo {
            model: "synth".into(),
            model_id: "synth".into(),
            quant: "gguf".into(),
            slots: 1,
            context: 64,
            layers: DEFAULT_LAYERS as u32,
        },
    );
    Flash {
        parts: vec![path],
        max_ctx: 128,
        name: "synth".into(),
        default_max: 6,
        tok,
        batch: true,
        cache: None,
        telem,
    }
}

/// user 1 ターンの要求 (temperature 0 = 貪欲、max_tokens 6)。
fn gen_req(content: &str) -> GenReq {
    GenReq::from_json(
        &serde_json::json!({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 6,
            "temperature": 0.0,
        }),
        6,
    )
    .expect("GenReq")
}

fn no_emit(_: &str) -> anyhow::Result<()> {
    Ok(())
}

fn builds() -> usize {
    TEST_BUILD_COUNT.load(Ordering::Relaxed)
}

/// task 123: env (`DENNO_FLASH_MODEL_REUSE`) と `TEST_FAIL_STEP` を書き替える
/// テストの直列化。`cargo test -p denno-cli` は既定で並列に走るので、env や
/// 注入スイッチを触るテスト同士が重なると build 数の assert が割れる。
static REUSE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// task 123: K 番目の差分で失敗する emit (それまでは成功)。クライアント切断
/// (broken pipe) の代わり。失敗のエラーはそのまま呼び出し元へ返る。
fn fail_at(k: usize) -> impl FnMut(&str) -> anyhow::Result<()> {
    let mut n = 0usize;
    move |_: &str| {
        n += 1;
        if n == k {
            Err(anyhow::anyhow!("emit failed (test injection {k})"))
        } else {
            Ok(())
        }
    }
}

/// user 1 ターンのプロンプトトークン (テンプレート済み)。
fn prompt_toks(flash: &Flash, content: &str) -> Vec<u32> {
    flash
        .tok
        .encode_messages(None, &[("user".into(), content.into())], None, false)
        .expect("encode_messages")
}

/// task 123 (a): emit 失敗 (クライアント切断) で decoder を cache に残す。
///
/// 合成モデルは "hi" を 6 トークンすべて 0x09 (tab) にし、各トークンが差分
/// 1 個 ("\t") を出す (プローブで確認した決定列。eos には当たらない)。
/// 通常経路は emit → step の順なので、K 番目の差分で失敗すると step は
/// K-1 回しか走らない (失敗した差分のトークンは step されず、その後も
/// 何も step しない = 余分な step は無い)。
#[test]
fn disconnect_caches_consistent_decoder() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");

    // fail_at(1): 最初の差分で失敗 → step は 0 回。toks はプロンプトのまま。
    let dir = TempDir::new("flash-disc-a1");
    let mut flash = build_flash(&dir);
    let plen = prompt_toks(&flash, "hi").len();
    let b0 = builds();
    let err = flash.generate(&gen_req("hi"), fail_at(1)).err().expect("generate が失敗するはず");
    assert!(
        err.to_string().contains("emit failed (test injection"),
        "emit のエラーがそのまま返っていない: {err}"
    );
    assert_eq!(builds() - b0, 1, "最初のリクエストで build_model が 1 回");
    let c = flash.cache.as_ref().expect("emit 失敗なのに cache が空");
    assert_eq!(c.toks, c.dec.tokens(), "toks が decoder 状態と不一致");
    assert_eq!(
        c.toks.len(),
        plen,
        "失敗した差分のトークン以降が step されていないはず"
    );
    assert_eq!(&c.toks[..plen], &prompt_toks(&flash, "hi")[..], "プロンプト部が変わった");

    // fail_at(3): 2 差分成功 → step 2 回で止まる (3 個目は step しない)。
    let dir2 = TempDir::new("flash-disc-a2");
    let mut flash2 = build_flash(&dir2);
    let err2 = flash2.generate(&gen_req("hi"), fail_at(3)).err().expect("generate が失敗するはず");
    assert!(err2.to_string().contains("emit failed (test injection"), "err={err2}");
    let c2 = flash2.cache.as_ref().expect("emit 失敗なのに cache が空");
    assert_eq!(c2.toks, c2.dec.tokens());
    assert_eq!(
        c2.toks.len(),
        plen + 2,
        "成功した 2 差分ぶんだけ step しているはず"
    );
    assert_eq!(&c2.toks[plen..], &[9u32, 9], "合成モデルの決定列 (tab 2 個)");
}

/// task 123 (b): 切断の次に前置きが違うリクエスト → reset 経路で
/// `build_model` なし。出力は新規構築とバイト一致。
#[test]
fn disconnect_then_other_prompt_reuses_model_without_rebuild() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");

    let dir = TempDir::new("flash-disc-b");
    let mut flash = build_flash(&dir);
    flash.generate(&gen_req("hi"), fail_at(2)).err().expect("generate が失敗するはず");
    assert!(flash.cache.is_some(), "切断後に cache が空");

    let b1 = builds();
    let g2 = flash.generate(&gen_req("yo"), no_emit).expect("generate 2");
    assert_eq!(
        builds() - b1,
        0,
        "別プロンプトで build_model が走った (reset 再利用になっていない)"
    );
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);
    assert_eq!(g2.text, "\t\t\t\t\t\t", "合成モデルの決定列 (tab 6 個)");

    // 新規構築 (cache 無し) との比較。出力はバイト一致のはず。
    let dir2 = TempDir::new("flash-disc-b2");
    let mut fresh = build_flash(&dir2);
    let b2 = builds();
    let f2 = fresh.generate(&gen_req("yo"), no_emit).expect("fresh generate");
    assert_eq!(builds() - b2, 1, "新規構築で build_model が 1 回走るはず");
    assert_eq!(g2.text, f2.text, "reset 再利用の出力が新規構築と一致しない");
}

/// task 123 (c): 切断の次に保存した toks を延長するプロンプト → 前置き一致
/// 経路で差分 prefill (reuse > 0)。出力は新規構築の full run と一致。
///
/// 継続リクエストは `continue: true` (最後の assistant メッセージを閉じずに
/// プロンプト末尾へ生テキストで継ぐ)。byte-level 語彙 (merge 無し) なので
/// decode → encode がトークン列で戻る (プローブで全 prefix を確認済み)。
#[test]
fn disconnect_then_extended_prompt_reuses_prefix() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");

    let dir = TempDir::new("flash-disc-c");
    let mut flash = build_flash(&dir);
    flash.generate(&gen_req("hi"), fail_at(2)).err().expect("generate が失敗するはず");
    let plen = prompt_toks(&flash, "hi").len();
    let stored = flash.cache.as_ref().expect("cache").toks.clone();
    assert_eq!(stored.len(), plen + 1, "プロンプト + 成功差分 1 (tab)");

    // 保存した生成部を decode し、2 倍にして延長 (保存列 + 1 トークぶん長い)。
    let part = flash.tok.decode(&stored[plen..]).expect("decode");
    let req2 = GenReq::from_json(
        &serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": format!("{part}{part}")}
            ],
            "continue": true,
            "max_tokens": 6,
            "temperature": 0.0,
        }),
        6,
    )
    .expect("GenReq");

    let b1 = builds();
    let g2 = flash.generate(&req2, no_emit).expect("generate 2");
    assert_eq!(builds() - b1, 0, "延長リクエストで build_model が走った");
    let reuse = TEST_LAST_REUSE.load(Ordering::Relaxed);
    assert_eq!(
        reuse,
        stored.len(),
        "保存した前置き全体が再利用されていない (reuse={reuse})"
    );
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);

    // 新規構築 + full prefill との比較。
    let dir2 = TempDir::new("flash-disc-c2");
    let mut fresh = build_flash(&dir2);
    let f2 = fresh.generate(&req2, no_emit).expect("fresh generate");
    assert_eq!(g2.text, f2.text, "前置き再利用の出力が新規構築と一致しない");
    assert!(f2.n_prompt > stored.len(), "延長リクエストのプロンプトが短い");
}

/// task 123 (d): engine 側の失敗 (step) は従来どおり cache を捨てて、
/// 次のリクエストで組み直す。失敗の注入は cfg(test) の `TEST_FAIL_STEP`
/// (本番コード経路には存在しない) で行う。
#[test]
fn engine_failure_drops_decoder_and_rebuilds() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");

    let dir = TempDir::new("flash-disc-d");
    let mut flash = build_flash(&dir);
    TEST_FAIL_STEP.store(true, Ordering::Relaxed);
    let err = flash.generate(&gen_req("hi"), no_emit).err().expect("generate が失敗するはず");
    TEST_FAIL_STEP.store(false, Ordering::Relaxed);
    assert!(err.to_string().contains("test: step 失敗を注入"), "err={err}");
    assert!(flash.cache.is_none(), "engine 失敗なのに cache が残った");

    // 次のリクエストは組み直す (cache が無いので build_model が走る)。
    let b1 = builds();
    let g2 = flash.generate(&gen_req("hi"), no_emit).expect("generate 2");
    assert_eq!(
        builds() - b1,
        1,
        "engine 失敗の次で build_model が走っていない"
    );
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);
}

/// task 123 (e): gate off (`DENNO_FLASH_MODEL_REUSE=0`) では切断しても
/// cache を残さない (従来どおり次のリクエストで組み直す)。
#[test]
fn gate_off_disconnect_keeps_old_behaviour() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::set_var("DENNO_FLASH_MODEL_REUSE", "0");

    let dir = TempDir::new("flash-disc-e");
    let mut flash = build_flash(&dir);
    let err = flash.generate(&gen_req("hi"), fail_at(2)).err().expect("generate が失敗するはず");
    assert!(err.to_string().contains("emit failed (test injection"), "err={err}");
    assert!(flash.cache.is_none(), "gate off で切断 cache が残った");

    // 従来どおり次のリクエストで組み直す。
    let b1 = builds();
    flash.generate(&gen_req("yo"), no_emit).expect("generate 2");
    assert_eq!(builds() - b1, 1, "gate off で build_model が走っていない");
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");
}

/// (e) `model_reuse_enabled` の真偽表。
///
/// 既定は on (`true`)。「0」/「false」/「no」だけが off。未設定・空・その他の
/// 値は既定に従う (typo で毎回組み直しになるのを避ける)。
#[test]
fn model_reuse_truth_table() {
    let cases: &[(&str, Option<&str>, bool)] = &[
        ("unset", None, true),
        ("empty", Some(""), true),
        ("0", Some("0"), false),
        ("false", Some("false"), false),
        ("no", Some("no"), false),
        ("1", Some("1"), true),
        ("true", Some("true"), true),
        ("yes", Some("yes"), true),
        ("other", Some("ON"), true),
        ("00", Some("00"), true),
    ];
    for (tag, v, want) in cases {
        assert_eq!(
            model_reuse_enabled(*v),
            *want,
            "DENNO_FLASH_MODEL_REUSE={tag}"
        );
    }
}

/// (f) no-reuse リクエストが reset 経路を通り、`build_model` を呼ばないこと。
///
/// 手順 (build_model の呼び出し数を `TEST_BUILD_COUNT` で数える):
/// 1. "hi" (cache 無し) → 1 回 build。
/// 2. "yo" (前置きが一致しない) → **reset 経路。build は増えない**。
/// 3. "yo" を `DENNO_FLASH_MODEL_REUSE=0` で → 従来経路 (build が増える)。
///    出力は手順 2 の reset 経路とビットでなくても (テキストとして) 同じ。
/// 4. gate を戻して "hi" → reset 経路 (build 増えない)。出力は手順 1 と同じ。
/// 5. "yo" の繰り返し → reset 経路 (build 増えない)。出力は手順 2 と同じ。
///
/// 手順 2 で build が増えていたら (= branch が発火していなかったら) この
/// テストは落ちる。出力の一致は「reset + prefill が新規構築 + prefill と
/// 同じ」ことの serve レベルの確認 (エンジン側の to_bits 比較は
/// tests/qwen4exp_decode.rs の reset_* テストが持つ)。
#[test]
fn flash_no_reuse_resets_decoder_without_rebuilding() {
    // task 123: env を書き替えるので env を触る他のテストと直列化する。
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    let dir = TempDir::new("flash-reuse");
    let mut flash = build_flash(&dir);
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");

    let b0 = builds();
    let g1 = flash.generate(&gen_req("hi"), no_emit).expect("generate 1");
    assert_eq!(
        builds() - b0,
        1,
        "cache 無しの最初のリクエストで build_model が 1 回走るはず"
    );
    assert!(g1.n_out > 0, "生成が空 (stop={})", g1.stop_reason);

    // 前置きが一致しない別プロンプト → reset 経路。build は増えない。
    let g2 = flash.generate(&gen_req("yo"), no_emit).expect("generate 2");
    assert_eq!(
        builds() - b0,
        1,
        "no-reuse 経路で build_model が増えた (reset 再利用になっていない)"
    );
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);

    // gate off なら従来どおり組み直す。出力は reset 経路と同じ。
    std::env::set_var("DENNO_FLASH_MODEL_REUSE", "0");
    let g3 = flash.generate(&gen_req("yo"), no_emit).expect("generate 3");
    assert_eq!(builds() - b0, 2, "gate off で build_model が走っていない");
    assert_eq!(g3.text, g2.text, "reset 経路の出力が組み直しと一致しない");

    // gate を戻して "hi" → reset 経路。出力は新規構築 (手順 1) と同じ。
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");
    let g4 = flash.generate(&gen_req("hi"), no_emit).expect("generate 4");
    assert_eq!(
        builds() - b0,
        2,
        "gate on に戻したあとの no-reuse で build_model が増えた"
    );
    assert_eq!(g4.text, g1.text, "reset 経路の \"hi\" 出力が新規構築と一致しない");

    // 同じプロンプト ("yo") の繰り返し → reset 経路で出力同じ。
    let g5 = flash.generate(&gen_req("yo"), no_emit).expect("generate 5");
    assert_eq!(builds() - b0, 2, "同一プロンプトで build_model が増えた");
    assert_eq!(g5.text, g2.text, "同一プロンプトの繰り返しで出力が変わった");
}

// ---------------------------------------------------------------------------
// task 131: 生成テキスト末尾の空白と prefix cache
// ---------------------------------------------------------------------------

/// N メッセージの要求 (temperature 0 = 貪欲、max_tokens 6)。
fn gen_req_msgs(msgs: &[(&str, &str)]) -> GenReq {
    let v: Vec<serde_json::Value> = msgs
        .iter()
        .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
        .collect();
    GenReq::from_json(
        &serde_json::json!({"messages": v, "max_tokens": 6, "temperature": 0.0}),
        6,
    )
    .expect("GenReq")
}

/// `DENNO_FLASH_PREFIX_SPLICE` の真偽表 (model_reuse と同じ規約)。
#[test]
fn prefix_splice_truth_table() {
    let cases: &[(&str, Option<&str>, bool)] = &[
        ("unset", None, true),
        ("empty", Some(""), true),
        ("0", Some("0"), false),
        ("false", Some("false"), false),
        ("no", Some("no"), false),
        ("1", Some("1"), true),
        ("true", Some("true"), true),
        ("yes", Some("yes"), true),
        ("other", Some("ON"), true),
        ("00", Some("00"), true),
    ];
    for (tag, v, want) in cases {
        assert_eq!(
            prefix_splice_enabled(*v),
            *want,
            "DENNO_FLASH_PREFIX_SPLICE={tag}"
        );
    }
}

/// 不一致の 1 行ログ: 最初の不一致位置と両側のトークンを出し、1 行を保つ。
#[test]
fn prefix_miss_line_reports_first_mismatch() {
    let dir = TempDir::new("flash-missline");
    let flash = build_flash(&dir);
    let line = prefix_miss_line(&flash.tok, &[1, 2, 3, 10], &[1, 2, 9]);
    assert!(
        line.starts_with("[flash] prefix miss: cache=4 prompt=3 lcp=2 "),
        "{line}"
    );
    assert!(line.contains("cached_tok=3("), "不一致トークンが無い: {line}");
    assert!(line.contains("prompt_tok=9("), "不一致トークンが無い: {line}");
    assert!(!line.contains('\n'), "1 行でない: {line:?}");

    // 短い方で頭打ち (prompt が cache の接頭辞) でも落ちない。
    let line2 = prefix_miss_line(&flash.tok, &[1, 2, 3], &[1, 2, 3]);
    assert!(line2.contains("lcp=3"), "{line2}");
    assert!(line2.contains("cached_tok=-"), "{line2}");
}

/// task 131: 生成テキストの末尾が空白で、テンプレートの `|trim` に落とされても
/// 前置きを捨てない (splice)。
///
/// 合成モデルは "hi" に tab 6 個を返す。クライアントはその本文をそのまま
/// 返送するので、テンプレートが trim して 0 文字になったぶんだけ前回の
/// プロンプトとずれる。ここで reuse=0 になっていたのが本番の 18.19 秒 miss。
#[test]
fn trimmed_assistant_tail_keeps_reuse_by_splice() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");
    std::env::remove_var("DENNO_FLASH_PREFIX_SPLICE");

    let dir = TempDir::new("flash-splice");
    let mut flash = build_flash(&dir);
    let b0 = builds();
    let g1 = flash.generate(&gen_req("hi"), no_emit).expect("generate 1");
    assert_eq!(g1.text, "\t\t\t\t\t\t", "合成モデルの決定列 (tab 6 個)");

    let c1 = flash.cache.as_ref().expect("cache").toks.clone();
    assert_eq!(c1.len(), g1.n_prompt + 6, "プロンプト + 生成 6");
    let echo = g1.text.clone();
    assert_eq!(echo.trim(), "", "この再現には末尾が空白の生成が要る");

    // 2 ターン目: 生の本文 (tab 6 個) を返送。
    let req2 = gen_req_msgs(&[("user", "hi"), ("assistant", &echo), ("user", "yo")]);
    let b1 = builds();
    let g2 = flash.generate(&req2, no_emit).expect("generate 2");
    assert_eq!(builds() - b1, 0, "splice 経路で build_model が走った");
    let reuse = TEST_LAST_REUSE.load(Ordering::Relaxed);
    assert_eq!(
        reuse,
        c1.len(),
        "生成テキストを前置きに残せていない (reuse={reuse}, cache={})",
        c1.len()
    );
    // trim して組んだプロンプトより、食った生成トークン 6 個ぶん長い。
    let trimmed = flash
        .tok
        .encode_messages(
            None,
            &[
                ("user".into(), "hi".into()),
                ("assistant".into(), "".into()),
                ("user".into(), "yo".into()),
            ],
            None,
            false,
        )
        .expect("encode trimmed");
    assert_eq!(
        g2.n_prompt,
        trimmed.len() + 6,
        "生の生成テキストがプロンプトに残っていない"
    );
    assert_eq!(g2.text, "\t\t\t\t\t\t", "2 ターン目の出力");

    // 3 ターン目: splice の続きでもう一度 splice できること。
    let c2len = flash.cache.as_ref().expect("cache").toks.len();
    let req3 = gen_req_msgs(&[
        ("user", "hi"),
        ("assistant", &echo),
        ("user", "yo"),
        ("assistant", &g2.text),
        ("user", "zz"),
    ]);
    let g3 = flash.generate(&req3, no_emit).expect("generate 3");
    assert_eq!(
        TEST_LAST_REUSE.load(Ordering::Relaxed),
        c2len,
        "splice の次のターンで前置きが再利用されていない"
    );
    assert_eq!(builds() - b0, 1, "build_model は最初の 1 回だけのはず");
    assert!(g3.n_out > 0, "生成が空 (stop={})", g3.stop_reason);
}

/// task 131: 返送された本文が食った生成トークンと違うときは splice せず、
/// 従来どおり full prefill に落ちる (安全側)。出力は新規構築と一致。
#[test]
fn changed_assistant_echo_falls_back_to_full_prefill() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");
    std::env::remove_var("DENNO_FLASH_PREFIX_SPLICE");

    let dir = TempDir::new("flash-splice-fb");
    let mut flash = build_flash(&dir);
    flash.generate(&gen_req("hi"), no_emit).expect("generate 1");

    // 本文を trim した (= クライアントが空白を落とした) 返送。
    let req2 = gen_req_msgs(&[("user", "hi"), ("assistant", ""), ("user", "yo")]);
    let b1 = builds();
    let g2 = flash.generate(&req2, no_emit).expect("generate 2");
    assert_eq!(
        TEST_LAST_REUSE.load(Ordering::Relaxed),
        0,
        "本文が違うのに reuse した"
    );
    assert_eq!(builds() - b1, 0, "full prefill で build_model が走った (reset 経路のはず)");
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);

    // 新規構築 + full prefill と同じ出力。
    let dir2 = TempDir::new("flash-splice-fb2");
    let mut fresh = build_flash(&dir2);
    let f2 = fresh.generate(&req2, no_emit).expect("fresh generate");
    assert_eq!(g2.text, f2.text, "fallback の出力が新規構築と一致しない");
}

/// task 131: gate off (`DENNO_FLASH_PREFIX_SPLICE=0`) では splice しない
/// (従来どおり reuse=0 の full prefill。モデルの組み直しはしない)。
#[test]
fn prefix_splice_gate_off_keeps_full_prefill() {
    let _g = REUSE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("DENNO_FLASH_MODEL_REUSE");
    std::env::set_var("DENNO_FLASH_PREFIX_SPLICE", "0");

    let dir = TempDir::new("flash-splice-off");
    let mut flash = build_flash(&dir);
    let g1 = flash.generate(&gen_req("hi"), no_emit).expect("generate 1");
    let b1 = builds();
    let req2 = gen_req_msgs(&[("user", "hi"), ("assistant", &g1.text), ("user", "yo")]);
    let g2 = flash.generate(&req2, no_emit).expect("generate 2");
    std::env::remove_var("DENNO_FLASH_PREFIX_SPLICE");
    assert_eq!(TEST_LAST_REUSE.load(Ordering::Relaxed), 0, "gate off で reuse した");
    assert_eq!(builds() - b1, 0, "gate off で build_model が走った");
    assert!(g2.n_out > 0, "生成が空 (stop={})", g2.stop_reason);
}
