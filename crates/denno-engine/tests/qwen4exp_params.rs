//! P-MOE-E3: qwen4exp のパラメータ構造とテンソル対応表のテスト。
//!
//! 実 GGUF (111 GB) はこのコンテナに無いので開かない。目録
//! (`/tmp/ref/flashnext_meta.txt`) の実値をこのファイルに固定し、
//! メタデータだけの合成 GGUF から params を組み立てて期待値を照合する。
//! テンソル検証は小さな合成 GGUF (同じ構造・小さい次元) で全経路を通す。
//!
//! GPU / CUDA / 推論経路は一切通さない。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::kquant::GgmlType;
use denno_engine::model::qwen4exp::{
    expected_layer_tensors, expected_tensors, validate, LayerKind, LayerTensors, MixerTensors,
    Qwen4ExpError, Qwen4ExpParams,
};

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
    fill: u8,
}

fn ts(name: impl Into<String>, dims: Vec<u64>, ty: u32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dims,
        ty,
        fill: 0x5a,
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
    ne / bs * type_size(ty) + if ne % bs != 0 { type_size(ty) } else { 0 }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
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
    for t in tensors {
        out.extend(std::iter::repeat(t.fill).take(nbytes(&t.dims, t.ty) as usize));
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
            "denno_qwen4exp_{tag}_{}_{n}",
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
// 目録の実値 (flashnext_meta.txt から転記。ここが唯一の出所)
// ---------------------------------------------------------------------------

const REAL_BLOCK_COUNT: u32 = 48;
const REAL_CTX: u32 = 262144;
const REAL_EMBD: u32 = 2560;
const REAL_HEAD: u32 = 24;
const REAL_HEAD_KV: u32 = 2;
const REAL_KEY_LEN: u32 = 256;
const REAL_VALUE_LEN: u32 = 256;
const REAL_RMS_EPS: f32 = 9.999999974752427e-07;
const REAL_ROPE_DIM: u32 = 64;
const REAL_ROPE_FREQ: f32 = 10000000.0;
const REAL_EXPERT: u32 = 512;
const REAL_EXPERT_USED: u32 = 10;
const REAL_FF_EXP: u32 = 640;
const REAL_FF_SHEXP: u32 = 640;
const REAL_SSM_CONV: u32 = 4;
const REAL_SSM_STATE: u32 = 128;
const REAL_SSM_GROUP: u32 = 16;
const REAL_SSM_DT: u32 = 48;
const REAL_SSM_INNER: u32 = 6144;
const REAL_INTERVAL: u32 = 4;
const REAL_HC: u32 = 4;
const REAL_HC_LR: u32 = 320;
const REAL_IDX_HEAD: u32 = 4;
const REAL_IDX_DIM: u32 = 128;
const REAL_IDX_TOP_K: u32 = 2048;
const REAL_EMBD_PER_LAYER: u32 = 160;
const REAL_PLE_NGRAM: u32 = 3;
const REAL_PLE_HEADS_PER_NGRAM: u32 = 8;
const REAL_PLE_CONV: u32 = 4;
const REAL_PLE_EOS: u32 = 248044;
const REAL_PLE_IMAGE: u32 = 248056;
const REAL_VOCAB: usize = 248320;
/// 目録の `per_layer_token_embd.weight` の行数 (頭の値)。
const REAL_PLE_TABLE_ROWS: u64 = 320001536;

/// 目録の `rope.dimension_sections` = [11, 11, 10, 0]
const REAL_ROPE_SECTIONS: [u32; 4] = [11, 11, 10, 0];
/// 目録の `attention.compress_ratios` = 0,0,0,4 の 48 要素
const REAL_COMPRESS_RATIOS: [u32; 48] = [
    0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4,
    0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4,
];
/// 目録の `ple.layer_multipliers`
const REAL_PLE_MULTIPLIERS: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];
/// 目録の `ple.head_offsets` (16 本)
const REAL_PLE_OFFSETS: [u64; 16] = [
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
    180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
];
/// 目録の `ple.head_vocab_sizes` (16 本)
const REAL_PLE_VOCAB_SIZES: [u64; 16] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];

/// 実モデルのメタデータだけを持つ GGUF (テンソル 0 本)。
fn real_meta_kvs() -> Vec<(String, Kv)> {
    vec![
        ("general.architecture".to_string(), Kv::Str("qwen4exp")),
        ("general.alignment".to_string(), Kv::U32(32)),
        ("qwen4exp.block_count".to_string(), Kv::U32(REAL_BLOCK_COUNT)),
        ("qwen4exp.context_length".to_string(), Kv::U32(REAL_CTX)),
        ("qwen4exp.embedding_length".to_string(), Kv::U32(REAL_EMBD)),
        ("qwen4exp.attention.head_count".to_string(), Kv::U32(REAL_HEAD)),
        (
            "qwen4exp.attention.head_count_kv".to_string(),
            Kv::U32(REAL_HEAD_KV),
        ),
        (
            "qwen4exp.attention.key_length".to_string(),
            Kv::U32(REAL_KEY_LEN),
        ),
        (
            "qwen4exp.attention.value_length".to_string(),
            Kv::U32(REAL_VALUE_LEN),
        ),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon".to_string(),
            Kv::F32(REAL_RMS_EPS),
        ),
        (
            "qwen4exp.rope.dimension_count".to_string(),
            Kv::U32(REAL_ROPE_DIM),
        ),
        (
            "qwen4exp.rope.dimension_sections".to_string(),
            Kv::ArrU32(REAL_ROPE_SECTIONS.to_vec()),
        ),
        (
            "qwen4exp.rope.freq_base".to_string(),
            Kv::F32(REAL_ROPE_FREQ),
        ),
        ("qwen4exp.expert_count".to_string(), Kv::U32(REAL_EXPERT)),
        (
            "qwen4exp.expert_used_count".to_string(),
            Kv::U32(REAL_EXPERT_USED),
        ),
        (
            "qwen4exp.expert_feed_forward_length".to_string(),
            Kv::U32(REAL_FF_EXP),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length".to_string(),
            Kv::U32(REAL_FF_SHEXP),
        ),
        ("qwen4exp.ssm.conv_kernel".to_string(), Kv::U32(REAL_SSM_CONV)),
        ("qwen4exp.ssm.state_size".to_string(), Kv::U32(REAL_SSM_STATE)),
        ("qwen4exp.ssm.group_count".to_string(), Kv::U32(REAL_SSM_GROUP)),
        (
            "qwen4exp.ssm.time_step_rank".to_string(),
            Kv::U32(REAL_SSM_DT),
        ),
        ("qwen4exp.ssm.inner_size".to_string(), Kv::U32(REAL_SSM_INNER)),
        (
            "qwen4exp.full_attention_interval".to_string(),
            Kv::U32(REAL_INTERVAL),
        ),
        (
            "qwen4exp.hyper_connection.count".to_string(),
            Kv::U32(REAL_HC),
        ),
        (
            "qwen4exp.hyper_connection.low_rank".to_string(),
            Kv::U32(REAL_HC_LR),
        ),
        (
            "qwen4exp.attention.indexer.head_count".to_string(),
            Kv::U32(REAL_IDX_HEAD),
        ),
        (
            "qwen4exp.attention.indexer.key_length".to_string(),
            Kv::U32(REAL_IDX_DIM),
        ),
        (
            "qwen4exp.attention.indexer.top_k".to_string(),
            Kv::U32(REAL_IDX_TOP_K),
        ),
        (
            "qwen4exp.attention.compress_ratios".to_string(),
            Kv::ArrU32(REAL_COMPRESS_RATIOS.to_vec()),
        ),
        (
            "qwen4exp.embedding_length_per_layer_input".to_string(),
            Kv::U32(REAL_EMBD_PER_LAYER),
        ),
        ("qwen4exp.ple.layers".to_string(), Kv::ArrU32(vec![1])),
        (
            "qwen4exp.ple.ngram_size".to_string(),
            Kv::U32(REAL_PLE_NGRAM),
        ),
        (
            "qwen4exp.ple.heads_per_ngram".to_string(),
            Kv::U32(REAL_PLE_HEADS_PER_NGRAM),
        ),
        (
            "qwen4exp.ple.conv_kernel".to_string(),
            Kv::U32(REAL_PLE_CONV),
        ),
        (
            "qwen4exp.ple.eos_token_id".to_string(),
            Kv::U32(REAL_PLE_EOS),
        ),
        (
            "qwen4exp.ple.image_token_id".to_string(),
            Kv::U32(REAL_PLE_IMAGE),
        ),
        (
            "qwen4exp.ple.layer_multipliers".to_string(),
            Kv::ArrU64(REAL_PLE_MULTIPLIERS.to_vec()),
        ),
        (
            "qwen4exp.ple.head_offsets".to_string(),
            Kv::ArrU64(REAL_PLE_OFFSETS.to_vec()),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes".to_string(),
            Kv::ArrU64(REAL_PLE_VOCAB_SIZES.to_vec()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Kv::ArrStr(vec![String::new(); REAL_VOCAB]),
        ),
    ]
}

/// 実モデルのメタデータから params を作る。
fn real_params(dir: &TempDir) -> Qwen4ExpParams {
    let path = dir.path("real_meta.gguf");
    write_gguf(&path, &real_meta_kvs(), &[]);
    let f = GgufFile::open(&path).expect("open real meta");
    Qwen4ExpParams::from_meta(&f).expect("実メタデータは通るはず")
}

// ---------------------------------------------------------------------------
// 小さな合成モデル (同じ構造・小さい次元)
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
const N_EMBD_PER_LAYER: u64 = 160;
const PLE_LAYER: usize = 1;
const PLE_NGRAM: u64 = 3;
const PLE_HEADS_PER_NGRAM: u64 = 2;
const PLE_CONV: u64 = 4;
/// head_offsets [0,10,20,30] + head_vocab_sizes [10,10,10,10] の最大値
const PLE_MIN_ROWS: u64 = 40;

const HC_DIM: u64 = N_EMBD * HC;
const KEY_DIM: u64 = SSM_STATE * SSM_GROUP;
const VALUE_DIM: u64 = SSM_INNER;
const CONV_DIM: u64 = 2 * KEY_DIM + VALUE_DIM;
const DEFAULT_LAYERS: usize = 4;

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
        ("qwen4exp.block_count".to_string(), Kv::U32(layers as u32)),
        ("qwen4exp.context_length".to_string(), Kv::U32(1024)),
        ("qwen4exp.embedding_length".to_string(), Kv::U32(N_EMBD as u32)),
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
        ("qwen4exp.expert_count".to_string(), Kv::U32(N_EXPERT as u32)),
        (
            "qwen4exp.expert_used_count".to_string(),
            Kv::U32(1),
        ),
        (
            "qwen4exp.expert_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        ("qwen4exp.ssm.conv_kernel".to_string(), Kv::U32(SSM_CONV as u32)),
        ("qwen4exp.ssm.state_size".to_string(), Kv::U32(SSM_STATE as u32)),
        ("qwen4exp.ssm.group_count".to_string(), Kv::U32(SSM_GROUP as u32)),
        ("qwen4exp.ssm.time_step_rank".to_string(), Kv::U32(SSM_DT as u32)),
        ("qwen4exp.ssm.inner_size".to_string(), Kv::U32(SSM_INNER as u32)),
        (
            "qwen4exp.full_attention_interval".to_string(),
            Kv::U32(INTERVAL as u32),
        ),
        ("qwen4exp.hyper_connection.count".to_string(), Kv::U32(HC as u32)),
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
        ("qwen4exp.ple.conv_kernel".to_string(), Kv::U32(PLE_CONV as u32)),
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
        ts(p("ffn_gate_exps.weight"), vec![N_EMBD, N_FF, N_EXPERT], TY_Q4_K),
        ts(p("ffn_up_exps.weight"), vec![N_EMBD, N_FF, N_EXPERT], TY_Q4_K),
        ts(p("ffn_down_exps.weight"), vec![N_FF, N_EMBD, N_EXPERT], TY_Q5_1),
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
        ts(p("attn_q.weight"), vec![N_EMBD, 2 * N_HEAD * HEAD], TY_Q8_0),
        ts(p("attn_k.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_v.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_q_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_k_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_output.weight"), vec![N_HEAD * HEAD, N_EMBD], TY_Q8_0),
        ts(p("indexer.q_proj.weight"), vec![N_EMBD, IDX_HEAD * IDX_DIM], TY_BF16),
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

fn write_synth(dir: &TempDir, name: &str, layers: usize, tensors: &[TensorSpec]) -> PathBuf {
    let path = dir.path(name);
    write_gguf(&path, &synth_kvs(layers), tensors);
    path
}

fn expect_err<T>(r: Result<T, Qwen4ExpError>) -> Vec<String> {
    match r {
        Ok(_) => panic!("エラーになるはずが通った"),
        Err(e) => e.0,
    }
}

fn joined(errs: &[String]) -> String {
    errs.join("\n")
}

// ---------------------------------------------------------------------------
// 1. 目録の実値が params に入る
// ---------------------------------------------------------------------------

#[test]
fn real_metadata_matches_manifest() {
    let dir = TempDir::new("real");
    let p = real_params(&dir);

    assert_eq!(p.n_layer, 48);
    assert_eq!(p.n_ctx, 262144);
    assert_eq!(p.n_embd, 2560);
    assert_eq!(p.n_head, 24);
    assert_eq!(p.n_head_kv, 2);
    assert_eq!(p.n_embd_head_k, 256);
    assert_eq!(p.n_embd_head_v, 256);
    assert_eq!(p.rms_eps, 1e-6);
    assert_eq!(p.n_rot, 64);
    assert_eq!(p.rope_sections, [11, 11, 10, 0]);
    assert_eq!(p.rope_freq_base, 1e7);
    assert_eq!(p.n_expert, 512);
    assert_eq!(p.n_expert_used, 10);
    assert_eq!(p.n_ff_exp, 640);
    assert_eq!(p.n_ff_shexp, 640);
    assert_eq!(p.ssm_d_conv, 4);
    assert_eq!(p.ssm_d_state, 128);
    assert_eq!(p.ssm_n_group, 16);
    assert_eq!(p.ssm_dt_rank, 48);
    assert_eq!(p.ssm_d_inner, 6144);
    assert_eq!(p.full_attn_interval, 4);
    assert_eq!(p.n_hc, 4);
    assert_eq!(p.hc_low_rank, 320);
    assert_eq!(p.indexer_n_head, 4);
    assert_eq!(p.indexer_head_size, 128);
    assert_eq!(p.indexer_top_k, 2048);
    assert_eq!(p.compress_ratios.len(), 48);
    assert_eq!(p.n_embd_per_layer, 160);
    assert_eq!(p.ple_layers, vec![1]);
    assert_eq!(p.ple_ngram_size, 3);
    assert_eq!(p.ple_heads_per_ngram, 8);
    assert_eq!(p.ple_conv_kernel, 4);
    assert_eq!(p.ple_eos_token_id, 248044);
    assert_eq!(p.ple_image_token_id, 248056);
    assert_eq!(p.ple_layer_multipliers, REAL_PLE_MULTIPLIERS.to_vec());
    assert_eq!(p.n_vocab, REAL_VOCAB);

    // 派生値
    assert_eq!(p.hc_dim(), 10240, "embedding_length * hyper_connection.count");
    assert_eq!(p.n_ple_heads(), 16, "(ngram_size - 1) * heads_per_ngram");
    assert_eq!(p.ssm_key_dim(), 2048, "state_size * group_count");
    assert_eq!(p.ssm_value_dim(), 6144, "inner_size");
    assert_eq!(p.ssm_conv_dim(), 10240, "2*key + value");
    assert_eq!(p.indexer_kpool(), 4);
    // PLE 表の必要行数は 300001275 + 20000171 = 320001446。実テンソルは 90 行多い。
    assert_eq!(p.ple_min_rows(), 320001446);
    assert!(
        REAL_PLE_TABLE_ROWS >= p.ple_min_rows(),
        "目録の {} 行は PLE head 範囲 {} 行を満たす",
        REAL_PLE_TABLE_ROWS,
        p.ple_min_rows()
    );
}

// ---------------------------------------------------------------------------
// 2. 層の種別
// ---------------------------------------------------------------------------

#[test]
fn layer_kind_follows_full_attention_interval() {
    let dir = TempDir::new("kind");
    let p = real_params(&dir);

    for il in [0usize, 1, 2] {
        assert_eq!(p.layer_kind(il).unwrap(), LayerKind::Ssm, "層 {il} は ssm");
    }
    for il in [3usize, 7, 11, 47] {
        assert_eq!(
            p.layer_kind(il).unwrap(),
            LayerKind::FullAttn,
            "層 {il} は full attention"
        );
    }
    assert_eq!(p.n_full_attn_layers(), 12);
    assert_eq!(p.n_ssm_layers(), 36);
    assert!(p.is_ple(1) && !p.is_ple(0) && !p.is_ple(3));

    let e = p.layer_kind(48).unwrap_err();
    assert!(e.0.len() == 1);
    assert!(e.0[0].contains("48"), "エラー: {e}");
    assert!(e.0[0].contains("block_count"), "エラー: {e}");
}

// ---------------------------------------------------------------------------
// 3. 期待テンソル表 (目録の 1224 本)
// ---------------------------------------------------------------------------

#[test]
fn expected_table_matches_manifest() {
    let dir = TempDir::new("table");
    let p = real_params(&dir);
    let all = expected_tensors(&p).expect("期待表");

    // 目録の split.tensors.count = 1224
    assert_eq!(all.len(), 1224, "6 + 48*16 + 36*9 + 12*10 + 6");
    // 名前の重複が無い
    let mut names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "期待表に名前の重複がある");

    let find = |n: &str| -> &denno_engine::model::qwen4exp::ExpectedTensor {
        all.iter()
            .find(|e| e.name == n)
            .unwrap_or_else(|| panic!("{n} が期待表に無い"))
    };

    // 全体系
    let t = find("token_embd.weight");
    assert_eq!(t.dims, vec![2560, 248320]);
    assert_eq!(t.ty, GgmlType::Q8_0);
    let t = find("per_layer_token_embd.weight");
    assert_eq!(t.dims, vec![160, 320001446], "必要最小行数");
    assert!(t.min_last_dim, "行数は「以上」(実テンソルは 320001536)");
    assert_eq!(t.ty, GgmlType::IQ4Nl);
    assert_eq!(find("output_hc_norm.weight").dims, vec![10240]);
    assert_eq!(find("output_hc_up.weight").dims, vec![320, 10240]);
    assert_eq!(find("output_hc_down.weight").dims, vec![10240, 320]);

    // 共通 (ssm 層と full 層の両方にある)
    for il in [0usize, 3] {
        let t = find(&format!("blk.{il}.ffn_gate_inp.weight"));
        assert_eq!(t.dims, vec![2560, 512]);
        assert_eq!(t.ty, GgmlType::F32);
        let t = find(&format!("blk.{il}.ffn_down_exps.weight"));
        assert_eq!(t.dims, vec![640, 2560, 512]);
        assert_eq!(t.ty, GgmlType::Q5_1, "ffn_down_exps は Q5_1");
        let t = find(&format!("blk.{il}.ffn_gate_exps.weight"));
        assert_eq!(t.dims, vec![2560, 640, 512]);
        assert_eq!(t.ty, GgmlType::Q4K);
        assert_eq!(
            find(&format!("blk.{il}.hc_attn_norm.weight")).dims,
            vec![10240]
        );
        assert_eq!(
            find(&format!("blk.{il}.hc_attn_inject.weight")).dims,
            vec![10240, 4]
        );
    }

    // ssm 層
    let t = find("blk.0.ssm_out.weight");
    assert_eq!(t.dims, vec![6144, 2560]);
    assert_eq!(t.ty, GgmlType::Q8_0);
    assert_eq!(find("blk.0.ssm_a").dims, vec![48], "サフィックス無し");
    assert_eq!(find("blk.0.ssm_dt.bias").dims, vec![48]);
    assert_eq!(find("blk.0.ssm_conv1d.weight").dims, vec![4, 10240]);
    assert_eq!(find("blk.0.attn_qkv.weight").dims, vec![2560, 10240]);
    assert_eq!(find("blk.0.attn_gate.weight").dims, vec![2560, 6144]);
    assert_eq!(find("blk.0.ssm_norm.weight").dims, vec![128]);

    // full attention 層
    let t = find("blk.3.attn_q.weight");
    assert_eq!(t.dims, vec![2560, 12288], "q 6144 + gate 6144");
    assert_eq!(t.ty, GgmlType::Q8_0);
    assert_eq!(find("blk.3.attn_k.weight").dims, vec![2560, 512]);
    assert_eq!(find("blk.3.attn_v.weight").dims, vec![2560, 512]);
    assert_eq!(find("blk.3.attn_q_norm.weight").dims, vec![256]);
    assert_eq!(find("blk.3.attn_output.weight").dims, vec![6144, 2560]);
    let t = find("blk.3.indexer.q_proj.weight");
    assert_eq!(t.dims, vec![2560, 512], "4 heads * 128");
    assert_eq!(t.ty, GgmlType::Bf16, "indexer は BF16");
    assert_eq!(find("blk.3.indexer.k_proj.weight").dims, vec![2560, 128]);
    assert_eq!(find("blk.3.indexer.q_norm.weight").dims, vec![128]);

    // PLE は blk.1 にだけ 6 本
    let ple: Vec<&str> = all
        .iter()
        .filter(|e| e.name.contains(".ple_"))
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(ple.len(), 6);
    assert!(ple.iter().all(|n| n.starts_with("blk.1.")), "{ple:?}");
    assert_eq!(find("blk.1.ple_key.weight").dims, vec![2560, 10240]);
    assert_eq!(find("blk.1.ple_value.weight").dims, vec![2560, 2560]);
    assert_eq!(find("blk.1.ple_conv1d.weight").dims, vec![4, 10240]);

    // 層種別の固有分の本数 (層 0 は ssm・PLE 無し、層 1 は ssm + PLE、層 3 は full)
    assert_eq!(expected_layer_tensors(&p, 0).unwrap().len(), 16 + 9);
    assert_eq!(expected_layer_tensors(&p, 1).unwrap().len(), 16 + 9 + 6);
    assert_eq!(expected_layer_tensors(&p, 3).unwrap().len(), 16 + 10);
}

// ---------------------------------------------------------------------------
// 4. validate: 通る / 欠落 / 形状 / 型 / 種別の取り違え / 全件収集
// ---------------------------------------------------------------------------

#[test]
fn validate_accepts_complete_synthetic_model() {
    let dir = TempDir::new("ok");
    let tensors = synth_tensors(DEFAULT_LAYERS);
    assert_eq!(tensors.len(), 6 + 4 * 16 + 3 * 9 + 1 * 10 + 6);
    let path = write_synth(&dir, "ok.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let p = validate(&f).expect("完全な合成モデルは通るはず");
    assert_eq!(p.n_layer, DEFAULT_LAYERS);
    assert_eq!(p.hc_dim(), 512);
    assert_eq!(p.n_ple_heads(), 4);
    assert_eq!(p.ple_min_rows(), PLE_MIN_ROWS);

    // 期待表の本数も一致する
    assert_eq!(expected_tensors(&p).unwrap().len(), tensors.len());
}

#[test]
fn validate_reports_missing_tensor_by_name() {
    let dir = TempDir::new("missing");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    tensors.retain(|t| t.name != "blk.0.ssm_a");
    let path = write_synth(&dir, "missing.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    assert_eq!(errs.len(), 1, "1 本欠け = 1 件: {errs:?}");
    assert!(errs[0].contains("blk.0.ssm_a"), "名前が入る: {}", errs[0]);
    assert!(errs[0].contains("見つからない"), "{}", errs[0]);
    assert!(errs[0].contains("[2]"), "期待形状が入る: {}", errs[0]);
    assert!(errs[0].contains("F32"), "期待型が入る: {}", errs[0]);
}

#[test]
fn validate_reports_wrong_shape_with_expected_and_actual() {
    let dir = TempDir::new("shape");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    for t in tensors.iter_mut() {
        if t.name == "blk.0.ssm_out.weight" {
            t.dims = vec![VALUE_DIM, 128]; // 期待は [64, 256]
        }
    }
    let path = write_synth(&dir, "shape.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("blk.0.ssm_out.weight"), "{}", errs[0]);
    assert!(errs[0].contains("[64, 256]"), "期待が入る: {}", errs[0]);
    assert!(errs[0].contains("[64, 128]"), "実値が入る: {}", errs[0]);
}

#[test]
fn validate_reports_wrong_quant_type() {
    let dir = TempDir::new("type");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    for t in tensors.iter_mut() {
        if t.name == "blk.2.ffn_down_exps.weight" {
            t.ty = TY_Q8_0; // 期待は Q5_1
        }
        if t.name == "blk.3.indexer.q_proj.weight" {
            t.ty = TY_Q8_0; // 期待は BF16
        }
    }
    let path = write_synth(&dir, "type.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    let all = joined(&errs);
    assert_eq!(errs.len(), 2, "{all}");
    assert!(
        all.contains("blk.2.ffn_down_exps.weight: 型 Q8_0 が期待 Q5_1 と違う"),
        "{all}"
    );
    assert!(
        all.contains("blk.3.indexer.q_proj.weight: 型 Q8_0 が期待 BF16 と違う"),
        "{all}"
    );
}

#[test]
fn validate_detects_layer_kind_mixup() {
    let dir = TempDir::new("mixup");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    // ssm 層 (層 0) に full attention の attn_q を置く
    tensors.push(ts(
        "blk.0.attn_q.weight",
        vec![N_EMBD, 2 * N_HEAD * HEAD],
        TY_Q8_0,
    ));
    let path = write_synth(&dir, "mixup.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("blk.0.attn_q.weight"), "{}", errs[0]);
    assert!(errs[0].contains("余分なテンソル"), "{}", errs[0]);
    assert!(errs[0].contains("ssm"), "層種別が入る: {}", errs[0]);
}

#[test]
fn validate_collects_every_error() {
    let dir = TempDir::new("collect");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    tensors.retain(|t| t.name != "blk.1.ssm_beta.weight");
    for t in tensors.iter_mut() {
        if t.name == "blk.3.attn_k.weight" {
            t.dims = vec![N_EMBD, 999];
        }
        if t.name == "blk.2.ffn_up_shexp.weight" {
            t.ty = TY_Q4_K;
        }
    }
    let path = write_synth(&dir, "collect.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    assert_eq!(errs.len(), 3, "3 件すべて返る: {errs:?}");
    let all = joined(&errs);
    assert!(all.contains("blk.1.ssm_beta.weight"), "{all}");
    assert!(all.contains("blk.3.attn_k.weight"), "{all}");
    assert!(all.contains("blk.2.ffn_up_shexp.weight"), "{all}");
    // 1 行 1 件 (行数 = 件数)
    assert_eq!(Qwen4ExpError(errs.clone()).to_string().lines().count(), 3);
}

#[test]
fn validate_reports_every_missing_tensor_on_metadata_only_file() {
    // 実メタデータだけの GGUF = 1224 本すべて欠落。1 件で止まらず全件返る。
    let dir = TempDir::new("allmiss");
    let path = dir.path("meta_only.gguf");
    write_gguf(&path, &real_meta_kvs(), &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(validate(&f));
    assert_eq!(errs.len(), 1224, "目録のテンソル数ぶん返る");
    assert!(errs.iter().all(|e| e.contains("見つからない")), "全件が欠落");
}

// ---------------------------------------------------------------------------
// 5. LayerTensors::load
// ---------------------------------------------------------------------------

#[test]
fn layer_tensors_load_covers_all_48_layers() {
    let dir = TempDir::new("load48");
    let layers = 48;
    let tensors = synth_tensors(layers);
    let path = write_synth(&dir, "load48.gguf", layers, &tensors);
    let f = GgufFile::open(&path).expect("open");
    let p = validate(&f).expect("48 層の合成モデルは通る");

    let mut n_ssm = 0;
    let mut n_full = 0;
    for il in 0..layers {
        let l = LayerTensors::load(&f, &p, il).unwrap_or_else(|e| panic!("層 {il}: {e}"));
        assert_eq!(l.il, il);
        let want = if (il + 1) % INTERVAL as usize == 0 {
            LayerKind::FullAttn
        } else {
            LayerKind::Ssm
        };
        assert_eq!(l.kind, want, "層 {il}");
        match (&l.mixer, want) {
            (MixerTensors::Ssm(s), LayerKind::Ssm) => {
                n_ssm += 1;
                assert_eq!(s.ssm_out.ne(), vec![VALUE_DIM, N_EMBD]);
                assert_eq!(s.attn_qkv.ne(), vec![N_EMBD, CONV_DIM]);
                assert_eq!(s.ssm_a.ne(), vec![SSM_DT], "サフィックス無しで引けている");
                assert_eq!(s.ssm_dt.ne(), vec![SSM_DT], ".bias で引けている");
                assert_eq!(s.ssm_conv1d.ne(), vec![SSM_CONV, CONV_DIM]);
            }
            (MixerTensors::FullAttn(a), LayerKind::FullAttn) => {
                n_full += 1;
                assert_eq!(a.attn_q.ne(), vec![N_EMBD, 2 * N_HEAD * HEAD]);
                assert_eq!(a.attn_output.ne(), vec![N_HEAD * HEAD, N_EMBD]);
                assert_eq!(a.indexer_q_proj.ne(), vec![N_EMBD, IDX_HEAD * IDX_DIM]);
                assert_eq!(a.indexer_q_proj.ty(), GgmlType::Bf16);
            }
            _ => panic!("層 {il} の mixer と kind が食い違う"),
        }
        // 共通 16 本が層の名前を指している
        assert_eq!(
            l.ffn_gate_inp.name(),
            format!("blk.{il}.ffn_gate_inp.weight")
        );
        assert_eq!(
            l.ffn_down_exps.name(),
            format!("blk.{il}.ffn_down_exps.weight")
        );
        assert_eq!(l.ffn_down_exps.ty(), GgmlType::Q5_1);
        assert_eq!(l.hc_attn_norm.ne(), vec![HC_DIM]);
        // PLE は層 1 だけ
        if il == PLE_LAYER {
            let ple = l.ple.as_ref().expect("層 1 は PLE を持つ");
            assert_eq!(ple.ple_key.ne(), vec![N_EMBD, HC_DIM]);
            assert_eq!(ple.ple_conv1d.ne(), vec![PLE_CONV, HC_DIM]);
        } else {
            assert!(l.ple.is_none(), "層 {il} は PLE を持たない");
        }
    }
    assert_eq!(n_full, 12);
    assert_eq!(n_ssm, 36);
    // 全 48 層ぶんの参照が別のテンソルを指している (取り違えていない)
    let a = LayerTensors::load(&f, &p, 0).unwrap();
    let b = LayerTensors::load(&f, &p, 1).unwrap();
    assert_ne!(a.ffn_gate_inp.offset(), b.ffn_gate_inp.offset());
}

#[test]
fn layer_tensors_load_rejects_broken_layer() {
    let dir = TempDir::new("loadbad");
    let mut tensors = synth_tensors(DEFAULT_LAYERS);
    for t in tensors.iter_mut() {
        if t.name == "blk.2.hc_ffn_inject.weight" {
            t.dims = vec![HC_DIM, HC + 1];
        }
    }
    let path = write_synth(&dir, "loadbad.gguf", DEFAULT_LAYERS, &tensors);
    let f = GgufFile::open(&path).expect("open");
    // params はメタデータから作る (validate は通らないので使わない)
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータは正常");

    let errs = expect_err(LayerTensors::load(&f, &p, 2));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("blk.2.hc_ffn_inject.weight"), "{}", errs[0]);
    assert!(errs[0].contains("[512, 2]"), "期待: {}", errs[0]);
    assert!(errs[0].contains("[512, 3]"), "実値: {}", errs[0]);

    // 範囲外の層
    let e = LayerTensors::load(&f, &p, DEFAULT_LAYERS).unwrap_err();
    assert!(e.0[0].contains("範囲外"), "{e}");
}

#[test]
fn ple_table_accepts_padding_and_rejects_short_table() {
    // 行数は「以上」: 40 (必要) / 48 (パディング) は通り、32 は落ちる
    for (rows, want_ok) in [(PLE_MIN_ROWS, true), (48, true), (32, false)] {
        let dir = TempDir::new("pletable");
        let mut tensors = synth_tensors(DEFAULT_LAYERS);
        for t in tensors.iter_mut() {
            if t.name == "per_layer_token_embd.weight" {
                t.dims = vec![N_EMBD_PER_LAYER, rows];
            }
        }
        let path = write_synth(&dir, "ple.gguf", DEFAULT_LAYERS, &tensors);
        let f = GgufFile::open(&path).expect("open");
        match validate(&f) {
            Ok(_) => assert!(want_ok, "{rows} 行は落ちるはず"),
            Err(e) => {
                assert!(!want_ok, "{rows} 行は通るはず: {e}");
                assert!(
                    e.0.iter().any(|l| l.contains("per_layer_token_embd.weight")
                        && l.contains("以上")),
                    "エラー: {e}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. メタデータのエラー収集
// ---------------------------------------------------------------------------

#[test]
fn metadata_read_errors_are_collected_not_fatal() {
    let dir = TempDir::new("meta_err");
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    // 1 本消す + 1 本を文字列にする (読み取り段階のエラーは 2 件そろって返る)
    kvs.retain(|(k, _)| k != "qwen4exp.ssm.inner_size");
    for (k, v) in kvs.iter_mut() {
        if k == "qwen4exp.block_count" {
            *v = Kv::Str("48");
        }
    }
    let path = dir.path("meta_err.gguf");
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(Qwen4ExpParams::from_meta(&f));
    let all = joined(&errs);
    assert_eq!(errs.len(), 2, "読み取りエラーは全件返る: {all}");
    assert!(all.contains("qwen4exp.ssm.inner_size: キーが無い"), "{all}");
    assert!(all.contains("qwen4exp.block_count: 整数でない"), "{all}");
}

#[test]
fn metadata_value_errors_are_collected_not_fatal() {
    let dir = TempDir::new("meta_val");
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    for (k, v) in kvs.iter_mut() {
        if k == "qwen4exp.hyper_connection.low_rank" {
            *v = Kv::U32(0);
        }
        if k == "qwen4exp.ssm.inner_size" {
            *v = Kv::U32(SSM_INNER as u32 + 1); // state_size * time_step_rank と違う
        }
        if k == "qwen4exp.hyper_connection.count" {
            *v = Kv::U32(1); // HC は 2 以上
        }
        if k == "qwen4exp.attention.compress_ratios" {
            *v = Kv::ArrU32(vec![0, 0, 0, 3]); // top_k=8 を割り切らない
        }
    }
    let path = dir.path("meta_val.gguf");
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(Qwen4ExpParams::from_meta(&f));
    let all = joined(&errs);
    assert_eq!(errs.len(), 4, "値の不整合も全件返る: {all}");
    assert!(all.contains("qwen4exp.hyper_connection.low_rank: 0 は不正"), "{all}");
    assert!(all.contains("qwen4exp.ssm.inner_size:"), "{all}");
    assert!(all.contains("1 以下は不正"), "{all}");
    assert!(all.contains("割り切れない"), "{all}");
}

#[test]
fn metadata_rejects_unknown_architecture() {
    let dir = TempDir::new("meta_arch");
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    for (k, v) in kvs.iter_mut() {
        if k == "general.architecture" {
            *v = Kv::Str("qwen35");
        }
    }
    let path = dir.path("meta_arch.gguf");
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(Qwen4ExpParams::from_meta(&f));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("general.architecture"), "{}", errs[0]);
    assert!(errs[0].contains("qwen4exp"), "{}", errs[0]);
}

#[test]
fn metadata_rejects_bad_rope_sections() {
    // 4 要素でない rope sections は読み取り段階のエラー。
    let dir = TempDir::new("meta_arr");
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    for (k, v) in kvs.iter_mut() {
        if k == "qwen4exp.rope.dimension_sections" {
            *v = Kv::ArrU32(vec![11, 11, 10]);
        }
    }
    let path = dir.path("meta_arr.gguf");
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(Qwen4ExpParams::from_meta(&f));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].contains("rope.dimension_sections: 4 要素でない"),
        "{}",
        errs[0]
    );
}

#[test]
fn ple_layer_must_be_a_ssm_layer() {
    let dir = TempDir::new("ple_full");
    let mut kvs = synth_kvs(DEFAULT_LAYERS);
    for (k, v) in kvs.iter_mut() {
        if k == "qwen4exp.ple.layers" {
            *v = Kv::ArrU32(vec![3]); // 層 3 は full attention
        }
    }
    let path = dir.path("ple_full.gguf");
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");

    let errs = expect_err(Qwen4ExpParams::from_meta(&f));
    let all = joined(&errs);
    assert!(all.contains("full attention 層"), "{all}");
}
