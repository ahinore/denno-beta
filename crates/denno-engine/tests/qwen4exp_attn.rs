//! P-MOE-F4: `model::qwen4exp_attn` (full attention 12 層の dense 経路) のテスト。
//!
//! 実モデル (111 GB) は使わない。このファイル内で **合成 GGUF** を書き出し、
//! 本物と同じ構造・同じ量子型 (Q8_0) の小モデルで検査する。
//!
//! 検査項目 (タスク F4 の 1〜8):
//! 1. `attn_q` の切り分け (head ごとに先頭 head_dim = q、後半 = gate)
//! 2. M-RoPE の 3 軸が同じ位置なら通常 rope とビット一致 (sections の効果が消える)
//! 3. rope の手計算 (対 `(0, n_rot/2)`、角度は有理数)
//! 4. GQA の対応 (q head が同じ kv head を読む)
//! 5. softmax (一様 / 一点集中)
//! 6. 因果性 (キャッシュに積んだ本数だけを見る)
//! 7. キャッシュ (push / clear / 容量超過)
//! 8. 出力ゲート (単調・大きな正で attention そのもの・大きな負でほぼ 0)
//!
//! 数値の一致はすべて `to_bits()` で見る。softmax / sigmoid / cos / sin は
//! 実装と同じ f32 の関数を呼んで期待値を作る (実装差を持ち込まない)。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::Qwen4ExpParams;
use denno_engine::model::qwen4exp_attn::{
    rope_mrope_partial, rope_neox_partial, sigmoid, AttnCache, FullAttn,
};

// ---- 小モデルの諸元 (構造は本物と同じ、寸法だけ小さい) --------------------
const N_LAYER: usize = 4;
const N_EMBD: usize = 32;
const HEAD_DIM: usize = 16;
const N_ROT: usize = 8;
const ROPE_THETA: f32 = 1e4;
/// full attention は層 3 だけ ((3 + 1) % 4 == 0)。
const ATTN_LAYER: usize = 3;

// ---- GGUF 書き出し -------------------------------------------------------
const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const TY_F32: u32 = 0;
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
            "denno_qwen4exp_attn_{tag}_{}_{n}",
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

fn params(n_head: usize, n_head_kv: usize) -> Qwen4ExpParams {
    Qwen4ExpParams {
        n_layer: N_LAYER,
        n_ctx: 128,
        n_embd: N_EMBD,
        n_head,
        n_head_kv,
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
        indexer_n_head: 1,
        indexer_head_size: 4,
        indexer_top_k: 8,
        compress_ratios: vec![0, 0, 0, 4],
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

/// full attention 層 1 本ぶんのテンソル。
///
/// - `attn_q`: 行 r は入力列 `r % N_EMBD` を読む。head h の q 行
///   (`h*2*dh + d`) は列 d、gate 行 (`h*2*dh + dh + d`) は列 `dh + d` を読むので、
///   q と gate が**別の入力**から立ち、取り違えれば値が入れ替わる。
/// - `attn_k` / `attn_v`: 行 r は入力列 r を読む。
/// - `attn_output`: 恒等 (行 r が列 r) — ゲートの検査で `attn` と直接比べられる。
fn layer_tensors(n_head: usize, n_head_kv: usize, break_k: bool) -> Vec<TensorSpec> {
    let n = |s: &str| format!("blk.{ATTN_LAYER}.{s}");
    let q_rows = 2 * n_head * HEAD_DIM;
    let kv_rows = n_head_kv * HEAD_DIM;
    let q_ones: Vec<(usize, usize)> = (0..q_rows).map(|r| (r, r % N_EMBD)).collect();
    let kv_ones: Vec<(usize, usize)> = (0..kv_rows).map(|r| (r, r % N_EMBD)).collect();
    let out_ones: Vec<(usize, usize)> = (0..N_EMBD).map(|r| (r, r)).collect();
    // 壊したテンソル (形状検査の確認用): attn_k の行数を 2 倍にする。
    let k_rows = if break_k { 2 * kv_rows } else { kv_rows };
    let k_ones: Vec<(usize, usize)> = (0..k_rows).map(|r| (r, r % N_EMBD)).collect();
    vec![
        TensorSpec {
            name: n("attn_q.weight"),
            dims: vec![N_EMBD as u64, q_rows as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_EMBD, q_rows, &q_ones),
        },
        TensorSpec {
            name: n("attn_k.weight"),
            dims: vec![N_EMBD as u64, k_rows as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(N_EMBD, k_rows, &k_ones),
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
            dims: vec![(n_head * HEAD_DIM) as u64, N_EMBD as u64],
            ty: TY_Q8_0,
            data: q8_0_matrix(n_head * HEAD_DIM, N_EMBD, &out_ones),
        },
    ]
}

fn kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.block_count", Kv::U32(N_LAYER as u32)),
        ("qwen4exp.embedding_length", Kv::U32(N_EMBD as u32)),
        ("qwen4exp.attention.head_count", Kv::U32(2)),
        ("qwen4exp.attention.head_count_kv", Kv::U32(1)),
    ]
}

/// 合成 GGUF を書いて開く。
fn open_fixture(
    dir: &TempDir,
    tag: &str,
    n_head: usize,
    n_head_kv: usize,
    break_k: bool,
) -> GgufFile {
    let path = dir.path(&format!("{tag}.gguf"));
    write_gguf(&path, &kvs(), &layer_tensors(n_head, n_head_kv, break_k));
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

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

/// 小モデル (n_head=2 / n_head_kv=1) の full attention 層を組む。
fn full_attn(dir: &TempDir, tag: &str, max_ctx: usize) -> (GgufFile, Qwen4ExpParams, FullAttn) {
    let f = open_fixture(dir, tag, 2, 1, false);
    let p = params(2, 1);
    let fa = FullAttn::from_gguf(&f, &p, max_ctx).expect("from_gguf");
    (f, p, fa)
}

// ---- 1. attn_q の切り分け ------------------------------------------------

/// head h の 2*head_dim 要素のうち先頭 head_dim が q、後半が gate。
/// 入力の別々の列から立ててあるので、オフセットを取り違えると値が入れ替わる。
#[test]
fn q_gate_split_uses_head_local_offsets() {
    let dir = TempDir::new("split");
    let (_f, _p, fa) = full_attn(&dir, "split", 4);
    let l = fa.layer(ATTN_LAYER).expect("層 3 が full attention でない");

    assert_eq!(l.qg_dim(), 2 * l.q_dim());
    assert_eq!(l.q_dim(), 2 * HEAD_DIM);
    assert_eq!(l.kv_dim(), HEAD_DIM);

    // 入力 i (0..32) は値 i+1。q は列 0..15、gate は列 16..31 を読む。
    let cur: Vec<f32> = (0..N_EMBD).map(|i| (i as f32) + 1.0).collect();
    let qg = l.project_q_gate(&cur).expect("project_q_gate");
    assert_eq!(qg.len(), 2 * 2 * HEAD_DIM);
    for (r, &v) in qg.iter().enumerate() {
        assert_eq!(v, cur[r % N_EMBD], "attn_q の行 {r} が列 {} を読んでいない", r % N_EMBD);
    }

    let (q, gate) = l.split_q_gate(&qg).expect("split_q_gate");
    assert_eq!(q.len(), 2 * HEAD_DIM);
    assert_eq!(gate.len(), 2 * HEAD_DIM);
    for h in 0..2 {
        for d in 0..HEAD_DIM {
            assert_eq!(q[h * HEAD_DIM + d], cur[d], "head {h} dim {d} の q");
            assert_eq!(
                gate[h * HEAD_DIM + d],
                cur[HEAD_DIM + d],
                "head {h} dim {d} の gate"
            );
        }
    }
    // 取り違え検出: q と gate が同じ並びなら、この検査は何も見ていない。
    assert_ne!(q, gate);
    assert_eq!(q[0], 1.0);
    assert_eq!(gate[0], 17.0);

    // 長さの取り違えはエラー
    assert!(l.split_q_gate(&qg[..qg.len() - 1]).is_err());
}

// ---- 2. M-RoPE の等価性 --------------------------------------------------

fn rope_input() -> Vec<f32> {
    (0..2 * HEAD_DIM)
        .map(|i| ((i as f32) * 0.375) - 2.0)
        .collect()
}

/// 3 軸が同じ位置なら、sections の分割は結果に影響しない (通常 rope とビット一致)。
#[test]
fn mrope_with_equal_axes_is_bit_exact_plain_rope() {
    for &pos in &[0usize, 1, 5, 13, 4096] {
        let mut plain = rope_input();
        rope_neox_partial(&mut plain, 2, HEAD_DIM, N_ROT, pos, ROPE_THETA);

        // 実モデルの sections と、小モデル用の小さい sections の両方で同じ値になる。
        for sections in [[2u32, 2, 2, 0], [11, 11, 10, 0]] {
            let mut mrope = rope_input();
            rope_mrope_partial(
                &mut mrope,
                2,
                HEAD_DIM,
                N_ROT,
                sections,
                pos,
                pos,
                pos,
                pos,
                ROPE_THETA,
            );
            assert_bits_eq(&mrope, &plain, &format!("pos {pos} sections {sections:?}"));
        }
    }

    // 逆の確認: 3 軸が違えば sections は効く (等価性テストが空振りでないこと)。
    let mut plain = rope_input();
    rope_neox_partial(&mut plain, 2, HEAD_DIM, N_ROT, 5, ROPE_THETA);
    let mut mrope = rope_input();
    rope_mrope_partial(
        &mut mrope,
        2,
        HEAD_DIM,
        N_ROT,
        [2, 2, 2, 0],
        5,
        6,
        7,
        8,
        ROPE_THETA,
    );
    assert_ne!(bits(&mrope), bits(&plain));
}

// ---- 3. rope の手計算 ----------------------------------------------------

/// pos=1 の対 (0, n_rot/2) は角度 1.0 (有理数) なので手で書ける。
#[test]
fn rope_matches_hand_computed_pair() {
    let mut x = vec![0f32; HEAD_DIM];
    x[0] = 1.5;
    x[N_ROT / 2] = -0.25;
    let before = x.clone();

    rope_neox_partial(&mut x, 1, HEAD_DIM, N_ROT, 1, ROPE_THETA);

    // 角度 = 1 * theta^(0) = 1.0 (厳密)
    let ang = 1.0f32 * ROPE_THETA.powf(0.0);
    assert_eq!(ang, 1.0);
    let (c, s) = (ang.cos(), ang.sin());
    let (x0, x1) = (before[0], before[N_ROT / 2]);
    assert_eq!(x[0].to_bits(), (x0 * c - x1 * s).to_bits(), "x cos - y sin");
    assert_eq!(
        x[N_ROT / 2].to_bits(),
        (x0 * s + x1 * c).to_bits(),
        "x sin + y cos"
    );

    // 回転しない範囲: n_rot..head_dim は不変、対でない次元 (1..n_rot/2) も 0 のまま。
    for d in N_ROT..HEAD_DIM {
        assert_eq!(x[d], before[d], "次元 {d} は回転しないはず");
    }
    for d in 1..N_ROT / 2 {
        assert_eq!(x[d], 0.0, "次元 {d} は 0 のまま (対は {})", d + N_ROT / 2);
    }

    // pos=0 は恒等 (cos=1, sin=0 で厳密)
    let mut y = rope_input();
    let y0 = y.clone();
    rope_neox_partial(&mut y, 2, HEAD_DIM, N_ROT, 0, ROPE_THETA);
    assert_bits_eq(&y, &y0, "pos 0");
}

// ---- 4. GQA の対応 -------------------------------------------------------

/// q head h は kv head `h * n_head_kv / n_head` を読む。
/// 2 kv head の小モデルで、head 0/1 が同じ出力・head 2/3 が別の出力になることを見る。
#[test]
fn gqa_heads_share_the_mapped_kv_head() {
    let dir = TempDir::new("gqa");
    let f = open_fixture(&dir, "gqa", 4, 2, false);
    let p = params(4, 2);
    let fa = FullAttn::from_gguf(&f, &p, 4).expect("from_gguf");
    let l = fa.layer(ATTN_LAYER).expect("層 3 が full attention でない");

    assert_eq!((0..4).map(|h| l.kv_head(h)).collect::<Vec<_>>(), vec![0, 0, 1, 1]);

    // 1 位置だけ積んだキャッシュ。kv head 0 と 1 で k/v を変える。
    let row = 2 * HEAD_DIM;
    let mut k = vec![0f32; row];
    let mut v = vec![0f32; row];
    k[0] = 1.0; // kv head 0 の k = e_0
    k[HEAD_DIM + 1] = 1.0; // kv head 1 の k = e_1
    v[0] = 10.0; // kv head 0 の v
    v[HEAD_DIM + 1] = 20.0; // kv head 1 の v
    let mut cache = AttnCache::new(N_LAYER, 2, HEAD_DIM, 4).expect("cache");
    cache.push(ATTN_LAYER, &k, &v).expect("push");

    // head 0/1 は e_0 を、head 2/3 は e_1 を向く。
    let mut q = vec![0f32; 4 * HEAD_DIM];
    q[0] = 1.0;
    q[HEAD_DIM] = 1.0;
    q[2 * HEAD_DIM + 1] = 1.0;
    q[3 * HEAD_DIM + 1] = 1.0;
    let out = l.attend(&q, &cache, ATTN_LAYER).expect("attend");

    let h0 = &out[0..HEAD_DIM];
    let h1 = &out[HEAD_DIM..2 * HEAD_DIM];
    let h2 = &out[2 * HEAD_DIM..3 * HEAD_DIM];
    let h3 = &out[3 * HEAD_DIM..4 * HEAD_DIM];
    assert_bits_eq(h0, h1, "head 0 と head 1 (同じ kv head 0)");
    assert_bits_eq(h2, h3, "head 2 と head 3 (同じ kv head 1)");
    assert_ne!(bits(h0), bits(h2), "kv head が違うのに出力が同じ");
    assert_eq!(h0[0], 10.0, "head 0 が kv head 0 の v を読んでいない");
    assert_eq!(h2[1], 20.0, "head 2 が kv head 1 の v を読んでいない");

    // 1 kv head の小モデルでも全 head が同じ kv head を読む。
    let dir2 = TempDir::new("gqa1");
    let (_f2, _p2, fa2) = full_attn(&dir2, "gqa1", 4);
    let l2 = fa2.layer(ATTN_LAYER).expect("層 3");
    assert_eq!((0..2).map(|h| l2.kv_head(h)).collect::<Vec<_>>(), vec![0, 0]);
}

// ---- 5. softmax ----------------------------------------------------------

/// スコアが全部同じなら一様、1 つだけ大きければその位置に集中する。
#[test]
fn softmax_is_uniform_for_equal_scores_and_peaked_for_one() {
    let dir = TempDir::new("softmax");
    let (_f, _p, fa) = full_attn(&dir, "softmax", 8);
    let l = fa.layer(ATTN_LAYER).expect("層 3");
    let dh = HEAD_DIM;
    let len = 4usize;

    // k は全位置同じ、v_j = e_j。q = e_0 → スコアは全部同じ。
    let mut k = vec![0f32; len * dh];
    let mut v = vec![0f32; len * dh];
    for j in 0..len {
        k[j * dh] = 1.0;
        v[j * dh + j] = 1.0;
    }
    let mut cache = AttnCache::new(N_LAYER, 1, dh, 8).expect("cache");
    for j in 0..len {
        cache
            .push(ATTN_LAYER, &k[j * dh..(j + 1) * dh], &v[j * dh..(j + 1) * dh])
            .expect("push");
    }
    let mut q = vec![0f32; 2 * dh];
    q[0] = 1.0;
    let out = l.attend(&q, &cache, ATTN_LAYER).expect("attend");
    // 一様分布 = 1/4 (f32 で厳密) が各位置に乗る。
    for j in 0..len {
        assert_eq!(out[j], 0.25, "位置 {j} の重みが 1/4 でない");
    }
    for d in len..dh {
        assert_eq!(out[d], 0.0);
    }

    // 位置 2 の k だけ 800 倍 → スコア差 199.75 で exp が 0 に落ちる。
    let mut k2 = k.clone();
    k2[2 * dh] = 800.0;
    let mut cache2 = AttnCache::new(N_LAYER, 1, dh, 8).expect("cache");
    for j in 0..len {
        cache2
            .push(ATTN_LAYER, &k2[j * dh..(j + 1) * dh], &v[j * dh..(j + 1) * dh])
            .expect("push");
    }
    let out2 = l.attend(&q, &cache2, ATTN_LAYER).expect("attend");
    assert_eq!(out2[2], 1.0, "位置 2 に集中していない");
    for j in [0usize, 1, 3] {
        assert_eq!(out2[j], 0.0, "位置 {j} に重みが残っている");
    }
}

// ---- 6. 因果性 -----------------------------------------------------------

/// attention が見るのは**積んだ本数 (`len`)** だけで、容量 (`max_ctx`) ではない。
#[test]
fn causal_attention_sees_only_pushed_keys() {
    let dir = TempDir::new("causal");
    let (_f, _p, fa) = full_attn(&dir, "causal", 16);
    let l = fa.layer(ATTN_LAYER).expect("層 3");
    let dh = HEAD_DIM;
    let mut q = vec![0f32; 2 * dh];
    q[0] = 1.0;
    q[1] = 1.0;
    q[2] = 1.0;

    let mut k = vec![0f32; 3 * dh];
    let mut v = vec![0f32; 3 * dh];
    for j in 0..3 {
        k[j * dh + j] = 1.0;
        v[j * dh + j] = 10.0 * (j as f32 + 1.0);
    }

    // 容量だけ違う 2 つのキャッシュ (4 と 9)。同じ 3 本を積んで比べる。
    let mut small = AttnCache::new(N_LAYER, 1, dh, 4).expect("cache");
    let mut large = AttnCache::new(N_LAYER, 1, dh, 9).expect("cache");
    for j in 0..3 {
        small
            .push(ATTN_LAYER, &k[j * dh..(j + 1) * dh], &v[j * dh..(j + 1) * dh])
            .expect("push");
        large
            .push(ATTN_LAYER, &k[j * dh..(j + 1) * dh], &v[j * dh..(j + 1) * dh])
            .expect("push");
    }
    assert_eq!(small.len(ATTN_LAYER).unwrap(), 3);
    let out3 = l.attend(&q, &small, ATTN_LAYER).expect("attend");
    let out3b = l.attend(&q, &large, ATTN_LAYER).expect("attend");
    assert_bits_eq(&out3, &out3b, "容量だけ違うキャッシュ");
    // 3 本・スコアが同じなので一様 1/3。1/3 は f32 で厳密でないため、
    // 期待値は softmax と同じ割り算 (1.0 / 3.0) と積で作る。
    let p = 1.0f32 / 3.0;
    assert_eq!(out3[0], p * 10.0);
    assert_eq!(out3[1], p * 20.0);
    assert_eq!(out3[2], p * 30.0);

    // 4 本目 (未来の位置) を積むと結果が変わる = 積んだ分はちゃんと見ている。
    let mut k4 = vec![0f32; dh];
    let mut v4 = vec![0f32; dh];
    k4[0] = 1000.0;
    v4[3] = 7.0;
    small.push(ATTN_LAYER, &k4, &v4).expect("push");
    let out4 = l.attend(&q, &small, ATTN_LAYER).expect("attend");
    assert_ne!(bits(&out3), bits(&out4), "積んだ 4 本目が見えていない");
    assert_eq!(out4[3], 7.0, "4 本目に集中していない");
    assert_eq!(out4[0], 0.0);

    // pos 2 (= 3 本積んだ状態) の結果は、4 本目を積む前とビット一致のまま。
    // 未来のキーはキャッシュに無いのでスコアに入らない。
    let mut again = AttnCache::new(N_LAYER, 1, dh, 3).expect("cache");
    for j in 0..3 {
        again
            .push(ATTN_LAYER, &k[j * dh..(j + 1) * dh], &v[j * dh..(j + 1) * dh])
            .expect("push");
    }
    let out3c = l.attend(&q, &again, ATTN_LAYER).expect("attend");
    assert_bits_eq(&out3c, &out3, "3 本だけのキャッシュ (再現性)");
}

// ---- 7. キャッシュ -------------------------------------------------------

#[test]
fn cache_push_clear_and_capacity() {
    let dh = HEAD_DIM;
    let mut c = AttnCache::new(2, 1, dh, 2).expect("cache");
    assert_eq!(c.max_ctx(), 2);
    assert_eq!(c.row(), dh);
    assert_eq!(c.n_layer(), 2);
    assert!(c.is_empty(0).unwrap());
    assert_eq!(c.k(0).unwrap().len(), 0);
    assert_eq!(c.v(0).unwrap().len(), 0);

    let k = vec![1.0f32; dh];
    let v = vec![2.0f32; dh];
    c.push(0, &k, &v).expect("push 1");
    assert_eq!(c.len(0).unwrap(), 1);
    assert_eq!(c.k(0).unwrap().len(), dh);
    assert_eq!(c.k(0).unwrap()[0], 1.0);
    assert_eq!(c.v(0).unwrap()[0], 2.0);
    // 層 1 は影響を受けない
    assert_eq!(c.len(1).unwrap(), 0);

    c.push(0, &k, &v).expect("push 2");
    assert_eq!(c.len(0).unwrap(), 2);
    assert_eq!(c.k(0).unwrap().len(), 2 * dh);

    // 容量超過はエラーで、長さは変わらない (黙って捨てない)
    let err = format!("{}", c.push(0, &k, &v).expect_err("容量超過が通った"));
    assert!(err.contains("max_ctx"), "{err}");
    assert_eq!(c.len(0).unwrap(), 2);

    // clear でその層だけ 0 に戻る
    c.push(1, &k, &v).expect("層 1 push");
    c.clear(0).expect("clear");
    assert_eq!(c.len(0).unwrap(), 0);
    assert_eq!(c.len(1).unwrap(), 1);
    c.push(0, &k, &v).expect("clear 後の push");
    assert_eq!(c.len(0).unwrap(), 1);

    c.clear_all();
    assert_eq!(c.len(0).unwrap(), 0);
    assert_eq!(c.len(1).unwrap(), 0);

    // 長さ・層番号の取り違え
    assert!(c.push(0, &k[..dh - 1], &v).is_err());
    assert!(c.push(0, &k, &v[..dh - 1]).is_err());
    assert!(c.len(2).is_err());
    assert!(c.push(2, &k, &v).is_err());
    assert!(c.clear(2).is_err());
    assert!(AttnCache::new(2, 1, dh, 0).is_err());
}

// ---- 8. 出力ゲート -------------------------------------------------------

/// sigmoid の単調性と、大きな正で 1.0 (厳密) になること。
#[test]
fn output_gate_is_monotone() {
    let dir = TempDir::new("gate");
    let (_f, _p, fa) = full_attn(&dir, "gate", 4);
    let l = fa.layer(ATTN_LAYER).expect("層 3");
    let n = l.q_dim();

    assert_eq!(sigmoid(0.0), 0.5);
    assert_eq!(sigmoid(60.0), 1.0, "sigmoid(60) が厳密に 1.0 でない");

    let attn: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -2.0 }).collect();
    let gates = [-60.0f32, -10.0, 0.0, 10.0, 60.0];
    let mut prev = f32::NEG_INFINITY;
    for g in gates {
        let gate = vec![g; n];
        let out = l.gate_mul(&attn, &gate).expect("gate_mul");
        // 正の要素は単調増加
        assert!(out[0] > prev, "gate {g} で単調でない ({} <= {prev})", out[0]);
        prev = out[0];
        // 負の要素は単調減少 (絶対値が増える)
        assert!(out[1] < 0.0);
    }
    // gate=+60 → attention そのもの (ビット一致)
    let hi = l.gate_mul(&attn, &vec![60.0; n]).expect("gate_mul");
    assert_bits_eq(&hi, &attn, "gate +60");
    // gate=-60 → ほぼ 0 (sigmoid(-60) = 1/(1+exp(60)) = 8.8e-27)
    let lo = l.gate_mul(&attn, &vec![-60.0; n]).expect("gate_mul");
    for (i, v) in lo.iter().enumerate() {
        assert!(v.abs() < 1e-25, "gate -60 で要素 {i} が {} のまま", v);
    }
    // 長さの取り違え
    assert!(l.gate_mul(&attn[..n - 1], &vec![0.0; n]).is_err());
    assert!(l.gate_mul(&attn, &vec![0.0; n - 1]).is_err());
}

/// 1 トークンを最後まで通し、ゲートが `attn_output` の手前で効いていることを見る
/// (大きな正のゲートなら出力は `attn_output(attn)` にビット一致)。
#[test]
fn forward_token_applies_gate_before_output_projection() {
    let dir = TempDir::new("fwd");
    let (_f, _p, mut fa) = full_attn(&dir, "fwd", 4);

    // q は列 0..15、gate は列 16..31 を読む。gate を +100 にすると sigmoid = 1.0。
    let mut cur: Vec<f32> = (0..N_EMBD).map(|i| (i as f32) * 0.125).collect();
    for x in cur[HEAD_DIM..].iter_mut() {
        *x = 100.0;
    }

    let (want_out, gate, attn) = {
        let l = fa.layer(ATTN_LAYER).expect("層 3");
        let qg = l.project_q_gate(&cur).expect("project_q_gate");
        let (mut q, gate) = l.split_q_gate(&qg).expect("split_q_gate");
        l.norm_rope_q(&mut q, 0).expect("norm_rope_q");
        let mut k = l.project_k(&cur).expect("project_k");
        l.norm_rope_k(&mut k, 0).expect("norm_rope_k");
        let v = l.project_v(&cur).expect("project_v");
        let mut cache = AttnCache::new(N_LAYER, 1, HEAD_DIM, 4).expect("cache");
        cache.push(ATTN_LAYER, &k, &v).expect("push");
        let attn = l.attend(&q, &cache, ATTN_LAYER).expect("attend");
        let out = l.project_out(&attn).expect("project_out");
        (out, gate, attn)
    };
    for (i, g) in gate.iter().enumerate() {
        assert_eq!(*g, 100.0, "gate 要素 {i} が 100.0 でない");
    }
    assert_eq!(sigmoid(gate[0]), 1.0);

    let got = fa.forward_token(ATTN_LAYER, &cur).expect("forward_token");
    assert_eq!(fa.pos(), 1);
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 1);
    // attn_output は恒等なので、ゲートが 1.0 なら出力 = attn
    assert_bits_eq(&got, &attn, "ゲート +100 の出力 (attn_output は恒等)");
    assert_bits_eq(&got, &want_out, "公開部品を組み直した期待値");

    // ゲートを -100 にすると出力はほぼ 0
    fa.reset();
    assert_eq!(fa.pos(), 0);
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 0);
    let mut cur2 = cur.clone();
    for x in cur2[HEAD_DIM..].iter_mut() {
        *x = -100.0;
    }
    let got2 = fa.forward_token(ATTN_LAYER, &cur2).expect("forward_token");
    for (i, v) in got2.iter().enumerate() {
        assert!(v.abs() < 1e-40, "ゲート -100 で要素 {i} が {} のまま", v);
    }
}

// ---- 担当範囲と形状検査 --------------------------------------------------

/// ssm 層は「担当外」を返す。full attention の層だけが `Some` になる。
#[test]
fn ssm_layers_are_out_of_scope() {
    let dir = TempDir::new("scope");
    let (_f, _p, mut fa) = full_attn(&dir, "scope", 4);
    let full: Vec<usize> = (0..N_LAYER).filter(|&il| fa.is_full(il)).collect();
    assert_eq!(full, vec![ATTN_LAYER]);
    for il in 0..N_LAYER {
        assert_eq!(fa.layer(il).is_some(), il == ATTN_LAYER, "層 {il}");
    }
    assert!(fa.layer(N_LAYER).is_none());

    let cur = vec![0.1f32; N_EMBD];
    for il in [0usize, 1, 2] {
        let err = format!("{}", fa.forward_token(il, &cur).expect_err("ssm 層が通った"));
        assert!(err.contains("担当外"), "層 {il}: {err}");
    }
    let err = format!("{}", fa.forward_token(N_LAYER, &cur).expect_err("範囲外が通った"));
    assert!(err.contains("範囲外"), "{err}");
    // 入力長の取り違え
    assert!(fa.forward_token(ATTN_LAYER, &cur[..N_EMBD - 1]).is_err());
    // 位置は進んでいない (エラーで状態を壊さない)
    assert_eq!(fa.pos(), 0);
}

/// テンソルの形状が違えばロードで落ちる (黙って誤読しない)。
#[test]
fn wrong_tensor_shape_is_rejected_at_load() {
    let dir = TempDir::new("shape");
    let f = open_fixture(&dir, "shape", 2, 1, true);
    let p = params(2, 1);
    let err = match FullAttn::from_gguf(&f, &p, 4) {
        Ok(_) => panic!("形状違いが通った"),
        Err(e) => format!("{e:#}"),
    };
    assert!(err.contains("attn_k.weight"), "{err}");
}


// ---- 位置の進み方 (full attention 層が複数ある場合) ----------------------

/// 位置は**層ごとのキャッシュ長の最大**。full attention 層が 2 本あるモデルで、
/// 層をまたいで順序が入れ替わっても (層優先の prefill)、各層の位置が
/// その層のキャッシュ長から決まることを見る。
#[test]
fn pos_follows_per_layer_cache_len() {
    let dir = TempDir::new("pos_multi");
    let path = dir.path("pos_multi.gguf");
    // 層 1 と層 3 を full attention にする (interval 2)。
    let mut tensors = layer_tensors(2, 1, false);
    let mut l1: Vec<TensorSpec> = layer_tensors(2, 1, false)
        .into_iter()
        .map(|t| TensorSpec {
            name: t.name.replace(&format!("blk.{ATTN_LAYER}."), "blk.1."),
            ..t
        })
        .collect();
    tensors.append(&mut l1);
    write_gguf(&path, &kvs(), &tensors);
    let f = GgufFile::open(&path).expect("合成 GGUF を開けない");
    let mut p = params(2, 1);
    p.full_attn_interval = 2;
    let mut fa = FullAttn::from_gguf(&f, &p, 4).expect("from_gguf");
    let full: Vec<usize> = (0..N_LAYER).filter(|&il| fa.is_full(il)).collect();
    assert_eq!(full, vec![1, ATTN_LAYER]);

    let cur = vec![0.25f32; N_EMBD];
    fa.forward_token(1, &cur).expect("層 1");
    assert_eq!(fa.cache().len(1).unwrap(), 1);
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 0);
    assert_eq!(fa.pos(), 1, "位置は層ごとのキャッシュ長の最大");
    fa.forward_token(ATTN_LAYER, &cur).expect("層 3");
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 1);
    assert_eq!(fa.pos(), 1, "2 本とも 1 トークン目なので 1 のまま");

    // 層 3 だけ 2 トークン目 (層優先の順序で起きる並び)
    fa.forward_token(ATTN_LAYER, &cur).expect("層 3 (2 トークン目)");
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 2);
    assert_eq!(fa.pos(), 2);
    fa.forward_token(1, &cur).expect("層 1 (2 トークン目)");
    assert_eq!(fa.pos(), 2);

    // 層 3 を 3 トークン目まで進めても、層 1 は 2 のまま (層ごとに独立)
    fa.forward_token(ATTN_LAYER, &cur).expect("層 3 (3 トークン目)");
    assert_eq!(fa.cache().len(1).unwrap(), 2);
    assert_eq!(fa.cache().len(ATTN_LAYER).unwrap(), 3);
    assert_eq!(fa.pos(), 3, "位置は最大のキャッシュ長");
}
