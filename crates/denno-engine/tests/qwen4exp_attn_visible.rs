//! G2: QSA 選択を full attention 経路に配線したことのテスト。
//!
//! 実モデル (111 GB) は使わない。このファイル内で **合成 GGUF** を書き出し、
//! 本物と同じ構造 (attn は Q8_0、indexer は BF16) の小モデルで検査する。
//!
//! 検査項目 (タスク G2 の A〜D + 配線):
//! A. `attend_visible(0..len)` が `attend` とビット一致 (len = 1, 5, 16)
//! B. cell を飛ばした visible 列は結果を変え、手計算の softmax と一致する
//! C. 不正な cell 列 (非昇順・重複・範囲外・空) はエラー
//! D. 短い文脈 (`n_pool <= top_k / kpool`) では選択が全 cell を返すので
//!    `DENNO_QSA=1` の出力が dense とビット一致する
//! E. 配線: 長い文脈では `FullAttn` の出力が
//!    「QsaState が返した cell 列での attend_visible」の手回しとビット一致し、
//!    dense とは異なる (= 選択結果が実際に使われている)
//!
//! 数値はすべて `to_bits()` で比べる。期待値は「同じ関数を呼んで作る」か
//! 「f32 の同じ演算列を手で書く」に限る。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::Qwen4ExpParams;
use denno_engine::model::qwen4exp_attn::{AttnCache, AttnLayer, FullAttn};
use denno_engine::model::qwen4exp_qsa::QsaState;

// ---- 小モデルの諸元 (構造は本物と同じ、寸法だけ小さい) --------------------
const N_LAYER: usize = 4;
/// full attention は層 3 だけ ((3 + 1) % 4 == 0)。
const ATTN_LAYER: usize = 3;
const N_EMBD: usize = 32;
const N_HEAD: usize = 2;
const N_HEAD_KV: usize = 1;
const HEAD_DIM: usize = 16;
const N_ROT: usize = 8;
const ROPE_THETA: f32 = 1e4;
/// `attention.indexer.key_length` (n_rot 以上・偶数)
const IDX_DIM: usize = 8;
/// `attention.indexer.head_count`
const N_IDX_H: usize = 2;
/// `attention.indexer.top_k`。kpool = 4 なので top_k / kpool = 1 pool。
const TOP_K: usize = 4;
/// `compress_ratios[3]` (QSA 層の pool サイズ)
const KPOOL: usize = 4;
const MAX_CTX: usize = 16;

// ---- GGUF 書き出し -------------------------------------------------------
const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const TY_F32: u32 = 0;
const TY_Q8_0: u32 = 8;
const TY_BF16: u32 = 30;

/// f16 の 1.0 (Q8_0 のスケール。1.0 は f16 で厳密)。
const F16_ONE: [u8; 2] = [0x00, 0x3c];
/// BF16 の 1.0 (上位 16 bit が 0x3F80)。
const BF16_ONE: [u8; 2] = [0x80, 0x3f];

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
            "denno_qwen4exp_visible_{tag}_{}_{n}",
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

/// BF16 行列 (k 列 n 行)。`ones` の (行, 列) だけ 1.0、残りは 0。
fn bf16_matrix(k: usize, n: usize, ones: &[(usize, usize)]) -> Vec<u8> {
    let mut out = vec![0u8; n * k * 2];
    for &(r, c) in ones {
        assert!(r < n && c < k, "ones の ({r}, {c}) が範囲外");
        let o = (r * k + c) * 2;
        out[o..o + 2].copy_from_slice(&BF16_ONE);
    }
    out
}

fn f32_data(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `blk.{ATTN_LAYER}.attn_*` (dense attention の重み)。値そのものは検査しないので
/// 行 r が入力列 `r % N_EMBD` を読む疎行列でよい。
fn attn_tensors() -> Vec<TensorSpec> {
    let n = |s: &str| format!("blk.{ATTN_LAYER}.{s}");
    let q_rows = 2 * N_HEAD * HEAD_DIM;
    let kv_rows = N_HEAD_KV * HEAD_DIM;
    let q_ones: Vec<(usize, usize)> = (0..q_rows).map(|r| (r, r % N_EMBD)).collect();
    let kv_ones: Vec<(usize, usize)> = (0..kv_rows).map(|r| (r, r % N_EMBD)).collect();
    let out_ones: Vec<(usize, usize)> = (0..N_EMBD).map(|r| (r, r)).collect();
    vec![
        TensorSpec {
            name: n("attn_q.weight"),
            dims: vec![N_EMBD as u64, q_rows as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_EMBD, q_rows, &q_ones),
        },
        TensorSpec {
            name: n("attn_k.weight"),
            dims: vec![N_EMBD as u64, kv_rows as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_EMBD, kv_rows, &kv_ones),
        },
        TensorSpec {
            name: n("attn_v.weight"),
            dims: vec![N_EMBD as u64, kv_rows as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_EMBD, kv_rows, &kv_ones),
        },
        TensorSpec {
            name: n("attn_q_norm.weight"),
            dims: vec![HEAD_DIM as u64],
            ty: TY_F32,
            data: f32_data(&vec![1.0f32; HEAD_DIM]),
        },
        TensorSpec {
            name: n("attn_k_norm.weight"),
            dims: vec![HEAD_DIM as u64],
            ty: TY_F32,
            data: f32_data(&vec![1.0f32; HEAD_DIM]),
        },
        TensorSpec {
            name: n("attn_output.weight"),
            dims: vec![(N_HEAD * HEAD_DIM) as u64, N_EMBD as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_HEAD * HEAD_DIM, N_EMBD, &out_ones),
        },
    ]
}

/// `blk.{ATTN_LAYER}.indexer.*`。`k_proj` / `q_proj` は恒等なので
/// `k_raw = cur[0..IDX_DIM]`、indexer の q は `cur` を 2 head に割ったもの。
fn indexer_tensors() -> Vec<TensorSpec> {
    let n = |s: &str| format!("blk.{ATTN_LAYER}.indexer.{s}");
    let q_ones: Vec<(usize, usize)> = (0..N_IDX_H * IDX_DIM).map(|r| (r, r)).collect();
    let k_ones: Vec<(usize, usize)> = (0..IDX_DIM).map(|r| (r, r)).collect();
    vec![
        TensorSpec {
            name: n("q_proj.weight"),
            dims: vec![N_EMBD as u64, (N_IDX_H * IDX_DIM) as u64],
            ty: TY_BF16,
            data: bf16_matrix(N_EMBD, N_IDX_H * IDX_DIM, &q_ones),
        },
        TensorSpec {
            name: n("k_proj.weight"),
            dims: vec![N_EMBD as u64, IDX_DIM as u64],
            ty: TY_BF16,
            data: bf16_matrix(N_EMBD, IDX_DIM, &k_ones),
        },
        TensorSpec {
            name: n("q_norm.weight"),
            dims: vec![IDX_DIM as u64],
            ty: TY_F32,
            data: f32_data(&vec![1.0f32; IDX_DIM]),
        },
        TensorSpec {
            name: n("k_norm.weight"),
            dims: vec![IDX_DIM as u64],
            ty: TY_F32,
            data: f32_data(&vec![1.0f32; IDX_DIM]),
        },
    ]
}

fn kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.block_count", Kv::U32(N_LAYER as u32)),
        ("qwen4exp.embedding_length", Kv::U32(N_EMBD as u32)),
        ("qwen4exp.attention.head_count", Kv::U32(N_HEAD as u32)),
        (
            "qwen4exp.attention.head_count_kv",
            Kv::U32(N_HEAD_KV as u32),
        ),
    ]
}

fn params() -> Qwen4ExpParams {
    let mut ratios = vec![0u32; N_LAYER];
    ratios[ATTN_LAYER] = KPOOL as u32;
    Qwen4ExpParams {
        n_layer: N_LAYER,
        n_ctx: 128,
        n_embd: N_EMBD,
        n_head: N_HEAD,
        n_head_kv: N_HEAD_KV,
        n_embd_head_k: HEAD_DIM,
        n_embd_head_v: HEAD_DIM,
        rms_eps: 1e-6,
        n_rot: N_ROT,
        rope_sections: [2, 2, 2, 0],
        rope_freq_base: ROPE_THETA,
        n_expert: 2,
        n_expert_used: 1,
        n_ff_exp: 8,
        n_ff_shexp: 8,
        ssm_d_conv: 4,
        ssm_d_state: 4,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 4,
        full_attn_interval: 4,
        n_hc: 2,
        hc_low_rank: 4,
        indexer_n_head: N_IDX_H,
        indexer_head_size: IDX_DIM,
        indexer_top_k: TOP_K,
        compress_ratios: ratios,
        n_embd_per_layer: 0,
        ple_layers: Vec::new(),
        ple_ngram_size: 0,
        ple_heads_per_ngram: 0,
        ple_conv_kernel: 0,
        ple_eos_token_id: 0,
        ple_image_token_id: 0,
        ple_layer_multipliers: Vec::new(),
        ple_head_offsets: Vec::new(),
        ple_head_vocab_sizes: Vec::new(),
        n_vocab: 8,
    }
}

/// 合成 GGUF (attn + indexer) を書いて開く。
fn open_fixture(dir: &TempDir, tag: &str) -> GgufFile {
    let path = dir.path(&format!("{tag}.gguf"));
    let mut tensors = attn_tensors();
    tensors.extend(indexer_tensors());
    write_gguf(&path, &kvs(), &tensors);
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

// ---- 決定的な入力 --------------------------------------------------------

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: 長さが違う");
    for i in 0..got.len() {
        assert_eq!(
            got[i].to_bits(),
            want[i].to_bits(),
            "{what}: 要素 {i} が {} と {} で違う",
            got[i],
            want[i]
        );
    }
}

/// キャッシュに積む k / v の 1 行 (`seed` = cell 番号)。cell ごとに違う値。
fn k_row(seed: usize) -> Vec<f32> {
    (0..HEAD_DIM)
        .map(|d| ((seed * 37 + d * 11) % 17) as f32 * 0.125 - 0.5)
        .collect()
}

fn v_row(seed: usize) -> Vec<f32> {
    (0..HEAD_DIM)
        .map(|d| ((seed * 29 + d * 7) % 13) as f32 * 0.25 - 1.0)
        .collect()
}

/// `len` 位置ぶん積んだキャッシュ。
fn cache_with(len: usize) -> AttnCache {
    let mut cache = AttnCache::new(N_LAYER, N_HEAD_KV, HEAD_DIM, MAX_CTX).expect("cache");
    for j in 0..len {
        cache
            .push(ATTN_LAYER, &k_row(j), &v_row(j))
            .expect("cache push");
    }
    cache
}

/// 正規化も rope も受けない生の q (値が cell ごとに違うのでスコアが一様でない)。
fn raw_q() -> Vec<f32> {
    (0..N_HEAD * HEAD_DIM)
        .map(|d| ((d * 7) % 5) as f32 * 0.25 + 0.125)
        .collect()
}

/// 1 トークンぶんの層入力。
fn token(t: usize) -> Vec<f32> {
    (0..N_EMBD)
        .map(|d| (((t * 13 + d * 5) % 19) as f32 - 9.0) * 0.0625)
        .collect()
}

fn layer_of(dir: &TempDir, tag: &str) -> (GgufFile, Qwen4ExpParams, FullAttn) {
    let f = open_fixture(dir, tag);
    let p = params();
    let fa = FullAttn::from_gguf_with(&f, &p, MAX_CTX, false).expect("from_gguf_with(false)");
    (f, p, fa)
}

// ---- A. attend_visible(0..len) == attend --------------------------------

/// `cells = 0..len` は dense と同じ演算順序なのでビット一致する。
/// len = 1, 5, 16 (キャッシュ満杯) で見る。
#[test]
fn visible_all_cells_is_bit_identical_to_dense() {
    let dir = TempDir::new("all");
    let (_f, _p, fa) = layer_of(&dir, "all");
    let l = fa.layer(ATTN_LAYER).expect("層 3 が full attention でない");
    let q = raw_q();
    for len in [1usize, 5, 16] {
        let cache = cache_with(len);
        let dense = l.attend(&q, &cache, ATTN_LAYER).expect("attend");
        let all: Vec<usize> = (0..len).collect();
        let vis = l
            .attend_visible(&q, &cache, ATTN_LAYER, &all)
            .expect("attend_visible");
        assert_bits_eq(&vis, &dense, &format!("len={len} の全 cell"));
    }
}

// ---- B. cell を飛ばすと結果が変わる --------------------------------------

/// 手計算: `cells` の昇順に dot → `* kq_scale` → max 引き softmax → v の加重和。
fn hand_attend(l: &AttnLayer, cache: &AttnCache, q: &[f32], cells: &[usize]) -> Vec<f32> {
    let dh = l.head_dim;
    let row = cache.row();
    let kb = cache.k(ATTN_LAYER).expect("k");
    let vb = cache.v(ATTN_LAYER).expect("v");
    let mut out = vec![0f32; l.q_dim()];
    for h in 0..l.n_head {
        let hk = l.kv_head(h);
        let mut s: Vec<f32> = Vec::with_capacity(cells.len());
        for &j in cells {
            let krow = &kb[j * row + hk * dh..][..dh];
            let mut acc = 0f32;
            for d in 0..dh {
                acc += q[h * dh + d] * krow[d];
            }
            s.push(acc * l.kq_scale);
        }
        let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in s.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in s.iter_mut() {
                *v /= sum;
            }
        }
        for (i, &j) in cells.iter().enumerate() {
            let vrow = &vb[j * row + hk * dh..][..dh];
            for d in 0..dh {
                out[h * dh + d] += s[i] * vrow[d];
            }
        }
    }
    out
}

#[test]
fn skipping_cells_changes_the_result_and_matches_hand_math() {
    let dir = TempDir::new("skip");
    let (_f, _p, fa) = layer_of(&dir, "skip");
    let l = fa.layer(ATTN_LAYER).expect("層 3");
    let q = raw_q();
    let len = 5usize;
    let cache = cache_with(len);

    let dense = l.attend(&q, &cache, ATTN_LAYER).expect("attend");
    let cells = vec![0usize, 2, 4];
    let vis = l
        .attend_visible(&q, &cache, ATTN_LAYER, &cells)
        .expect("attend_visible");
    let hand = hand_attend(l, &cache, &q, &cells);
    assert_bits_eq(&vis, &hand, "cells = [0, 2, 4] の手計算");
    assert_ne!(bits(&vis), bits(&dense), "cell を飛ばしたのに dense と同じ");

    // 飛ばす cell が違えば結果も違う (引数が効いている)。
    let cells2 = vec![1usize, 2, 3];
    let vis2 = l
        .attend_visible(&q, &cache, ATTN_LAYER, &cells2)
        .expect("attend_visible 2");
    assert_bits_eq(&vis2, &hand_attend(l, &cache, &q, &cells2), "cells = [1, 2, 3] の手計算");
    assert_ne!(bits(&vis), bits(&vis2), "別の cell 列なのに同じ");
}

// ---- C. 不正な cell 列はエラー -------------------------------------------

#[test]
fn invalid_cell_lists_are_errors() {
    let dir = TempDir::new("bad");
    let (_f, _p, fa) = layer_of(&dir, "bad");
    let l = fa.layer(ATTN_LAYER).expect("層 3");
    let q = raw_q();
    let len = 4usize;
    let cache = cache_with(len);

    // 非昇順
    let e = l
        .attend_visible(&q, &cache, ATTN_LAYER, &[2, 1])
        .expect_err("非昇順が通った");
    assert!(format!("{e}").contains("昇順"), "想定外のエラー: {e}");

    // 重複 (昇順の判定で落ちる)
    let e = l
        .attend_visible(&q, &cache, ATTN_LAYER, &[0, 1, 1])
        .expect_err("重複が通った");
    assert!(format!("{e}").contains("昇順"), "想定外のエラー: {e}");

    // 範囲外
    let e = l
        .attend_visible(&q, &cache, ATTN_LAYER, &[0, len])
        .expect_err("範囲外が通った");
    assert!(format!("{e}").contains("範囲外"), "想定外のエラー: {e}");

    // 空
    let e = l
        .attend_visible(&q, &cache, ATTN_LAYER, &[])
        .expect_err("空が通った");
    assert!(format!("{e}").contains("空"), "想定外のエラー: {e}");

    // dense 経路は従来どおり (len = 0 でもエラーにしない)。
    let empty = AttnCache::new(N_LAYER, N_HEAD_KV, HEAD_DIM, MAX_CTX).expect("cache");
    let out = l.attend(&q, &empty, ATTN_LAYER).expect("空キャッシュの attend");
    assert!(out.iter().all(|v| *v == 0.0), "空キャッシュの出力が 0 でない");
}

// ---- D. 短い文脈では dense と一致 ---------------------------------------

/// `n_pool = n_cells / kpool` が `top_k / kpool` (= 1) 以下なら全 cell が残る。
/// ここでは 7 cell (= 1 pool + tail) まで。8 cell で初めて pool が落ちる。
#[test]
fn short_context_selects_everything_and_matches_dense() {
    let dir = TempDir::new("short");
    let f = open_fixture(&dir, "short");
    let p = params();
    let mut qsa_on = FullAttn::from_gguf_with(&f, &p, MAX_CTX, true).expect("QSA 有効");
    let mut dense = FullAttn::from_gguf_with(&f, &p, MAX_CTX, false).expect("QSA 無効");

    assert!(qsa_on.qsa().is_some(), "QSA 有効なのに状態が無い");
    assert!(qsa_on.qsa_weight_bytes() > 0, "QSA の重みが 0 バイト");
    assert!(dense.qsa().is_none(), "QSA 無効なのに状態がある");
    assert_eq!(dense.qsa_weight_bytes(), 0, "QSA 無効なのに重みを数えている");

    for t in 0..7 {
        let cur = token(t);
        let a = qsa_on.forward_token(ATTN_LAYER, &cur).expect("qsa forward");
        let b = dense.forward_token(ATTN_LAYER, &cur).expect("dense forward");
        assert_bits_eq(&a, &b, &format!("短い文脈 (pos {t}) の QSA と dense"));
    }
    assert_eq!(qsa_on.pos(), 7);
}

// ---- F. 環境変数のゲート -------------------------------------------------

/// `DENNO_QSA` は `"1"` のときだけ有効。未設定 / それ以外は dense 経路。
/// (このテストだけが `from_gguf` を呼ぶので、環境変数の書き換えが他へ漏れない)
#[test]
fn denno_qsa_env_switches_the_path() {
    let dir = TempDir::new("env");
    let f = open_fixture(&dir, "env");
    let p = params();

    std::env::remove_var("DENNO_QSA");
    let off = FullAttn::from_gguf(&f, &p, MAX_CTX).expect("from_gguf (未設定)");
    assert!(off.qsa().is_none(), "未設定で QSA が組まれている");
    assert_eq!(off.qsa_weight_bytes(), 0, "未設定で重みを数えている");

    std::env::set_var("DENNO_QSA", "1");
    let on = FullAttn::from_gguf(&f, &p, MAX_CTX).expect("from_gguf (1)");
    assert!(on.qsa().is_some(), "DENNO_QSA=1 で QSA が組まれていない");
    assert!(on.qsa_weight_bytes() > 0, "QSA の重みが 0 バイト");

    std::env::set_var("DENNO_QSA", "true");
    let odd = FullAttn::from_gguf(&f, &p, MAX_CTX).expect("from_gguf (true)");
    assert!(odd.qsa().is_none(), "DENNO_QSA=true を有効として扱っている");

    std::env::remove_var("DENNO_QSA");
}

// ---- E. 配線 (選択結果が実際に attention に届く) -------------------------

/// QsaState が返した cell 列で手回しした 1 トークンぶんの出力。
fn manual_visible_forward(
    l: &AttnLayer,
    cache: &mut AttnCache,
    cur: &[f32],
    pos: usize,
    cells: &[usize],
) -> Vec<f32> {
    let qg = l.project_q_gate(cur).expect("project_q_gate");
    let (mut q, gate) = l.split_q_gate(&qg).expect("split_q_gate");
    l.norm_rope_q(&mut q, pos).expect("norm_rope_q");
    let mut k = l.project_k(cur).expect("project_k");
    l.norm_rope_k(&mut k, pos).expect("norm_rope_k");
    let v = l.project_v(cur).expect("project_v");
    cache.push(ATTN_LAYER, &k, &v).expect("cache push");
    let attn = l
        .attend_visible(&q, cache, ATTN_LAYER, cells)
        .expect("attend_visible");
    let gated = l.gate_mul(&attn, &gate).expect("gate_mul");
    l.project_out(&gated).expect("project_out")
}

#[test]
fn qsa_selection_reaches_attention() {
    let dir = TempDir::new("wire");
    let f = open_fixture(&dir, "wire");
    let p = params();
    let mut qsa_on = FullAttn::from_gguf_with(&f, &p, MAX_CTX, true).expect("QSA 有効");
    let mut dense = FullAttn::from_gguf_with(&f, &p, MAX_CTX, false).expect("QSA 無効");
    let manual = FullAttn::from_gguf_with(&f, &p, MAX_CTX, false).expect("手回し用");
    let l = manual.layer(ATTN_LAYER).expect("層 3");
    let mut mcache = AttnCache::new(N_LAYER, N_HEAD_KV, HEAD_DIM, MAX_CTX).expect("cache");
    let mut qsa = QsaState::from_gguf(&f, &p, MAX_CTX).expect("QsaState");

    let n = 12usize;
    for t in 0..n {
        let cur = token(t);
        let cells = qsa.push_token(ATTN_LAYER, &cur, t).expect("push_token");
        // 短い文脈では全 cell、8 cell 目からは選択が cell を落とす。
        if t <= 6 {
            assert_eq!(cells, (0..=t).collect::<Vec<_>>(), "pos {t} で全 cell でない");
        } else {
            assert!(
                cells.len() < t + 1,
                "pos {t} で 1 つも落ちていない (cells = {cells:?})"
            );
        }
        assert_eq!(cells.last().copied(), Some(t), "現在のトークンが入っていない");

        let got = qsa_on.forward_token(ATTN_LAYER, &cur).expect("qsa forward");
        let want = manual_visible_forward(l, &mut mcache, &cur, t, &cells);
        assert_bits_eq(&got, &want, &format!("pos {t} の QSA 出力と手回し"));

        let d = dense.forward_token(ATTN_LAYER, &cur).expect("dense forward");
        if t <= 6 {
            assert_bits_eq(&got, &d, &format!("pos {t} の QSA と dense (短い文脈)"));
        } else {
            assert_ne!(bits(&got), bits(&d), "pos {t} で選択が効いていない");
        }
    }
    assert_eq!(qsa_on.pos(), n);

    // reset で QSA の状態も空になり、同じ列をもう一度回すと同じ値になる。
    qsa_on.reset();
    assert_eq!(qsa_on.pos(), 0);
    let first: Vec<Vec<f32>> = (0..3)
        .map(|t| qsa_on.forward_token(ATTN_LAYER, &token(t)).expect("forward"))
        .collect();
    qsa_on.reset();
    assert_eq!(qsa_on.pos(), 0);
    for (t, want) in first.iter().enumerate() {
        let got = qsa_on.forward_token(ATTN_LAYER, &token(t)).expect("forward 2");
        assert_bits_eq(&got, want, &format!("reset 後の pos {t}"));
    }
}
