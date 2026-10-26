//! P-MOE-F5b: `model::qwen4exp_qsa` (indexer による key 選択) のテスト。
//!
//! 実モデル (111 GB) は使わない。このファイル内で合成 GGUF を書き出し、
//! 本物と同じ構造 (indexer の投影は BF16) の小モデルで検査する。
//!
//! 検査項目 (タスク F5b の 1〜8):
//! 1. pool の平均 (kpool=4、手計算とビット一致)
//! 2. 平均 → norm → rope の順序 (rope を先にやった値とは違う)
//! 3. rope の位置が pool の先頭メンバー (末尾ではない)
//! 4. スコア式 (ReLU・head の和・`1/sqrt(idx_dim)` の倍率)
//! 5. top-k (上位が選ばれる・同点は添字の小さい方)
//! 6. 短い文脈では dense と一致 (全 cell が返る)
//! 7. tail と現在のトークンが常に含まれる
//! 8. 容量超過はエラー
//!
//! 数値はすべて `to_bits()` で比べる。期待値は「同じ関数を呼んで作る」か
//! 「2 進で厳密に表せる有理数の手計算」に限る。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::indexer::{pool_keys, score_pools, select_cells, IndexerParams};
use denno_engine::model::qwen4exp::Qwen4ExpParams;
use denno_engine::model::qwen4exp_attn::rope_neox_partial;
use denno_engine::model::qwen4exp_qsa::{select_visible, QsaState};
use denno_engine::tensor::qknorm_rows_inplace;

// ---- 小モデルの諸元 (構造は本物と同じ、寸法だけ小さい) --------------------
const N_LAYER: usize = 4;
/// full attention は層 3 だけ ((3 + 1) % 4 == 0)。
const ATTN_LAYER: usize = 3;
const N_EMBD: usize = 16;
/// `attention.indexer.key_length`
const IDX_DIM: usize = 8;
/// `attention.indexer.head_count`
const N_IDX_H: usize = 2;
/// `rope.dimension_count` (idx_dim 以下・偶数)
const N_ROT: usize = 4;
const ROPE_THETA: f32 = 1e4;
const RMS_EPS: f32 = 1e-6;
/// `attention.indexer.top_k`。kpool=2 なら 4 pool、kpool=4 なら 2 pool。
const TOP_K: usize = 8;
const MAX_CTX: usize = 16;

/// `k_norm` は 1 以外を混ぜる。norm と rope の順序を入れ替えると値が変わるので、
/// 順序の検査 (2) が「たまたま一致した」で通らない。
const K_NORM: [f32; IDX_DIM] = [1.0, 1.5, 0.5, 2.0, 1.0, 1.5, 0.5, 2.0];

// ---- GGUF 書き出し -------------------------------------------------------
const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const TY_F32: u32 = 0;
const TY_BF16: u32 = 30;

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
            "denno_qwen4exp_qsa_{tag}_{}_{n}",
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

/// BF16 の 1.0 (上位 16 bit が 0x3F80)。
const BF16_ONE: [u8; 2] = [0x80, 0x3f];

/// BF16 行列 (k 列 n 行)。`ones` の (行, 列) だけ 1.0、残りは 0。
/// 1 本の行に 1 つだけ 1.0 を置くと `gemv` は入力の該当要素を厳密に返す。
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

/// `blk.{ATTN_LAYER}.indexer.*`。
///
/// - `k_proj`: 行 d が入力列 d を読む → `k_raw = cur[0..IDX_DIM]`
/// - `q_proj`: 恒等 (行 r が列 r) → head 0 は `cur[0..8]`、head 1 は `cur[8..16]`
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
            data: f32_data(&K_NORM),
        },
    ]
}

fn kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.block_count", Kv::U32(N_LAYER as u32)),
        ("qwen4exp.embedding_length", Kv::U32(N_EMBD as u32)),
        (
            "qwen4exp.attention.indexer.head_count",
            Kv::U32(N_IDX_H as u32),
        ),
        (
            "qwen4exp.attention.indexer.key_length",
            Kv::U32(IDX_DIM as u32),
        ),
        (
            "qwen4exp.attention.indexer.top_k",
            Kv::U32(TOP_K as u32),
        ),
    ]
}

fn params(kpool: u32) -> Qwen4ExpParams {
    let mut ratios = vec![0u32; N_LAYER];
    ratios[ATTN_LAYER] = kpool;
    Qwen4ExpParams {
        n_layer: N_LAYER,
        n_ctx: 64,
        n_embd: N_EMBD,
        n_head: 2,
        n_head_kv: 1,
        n_embd_head_k: 8,
        n_embd_head_v: 8,
        rms_eps: RMS_EPS,
        n_rot: N_ROT,
        rope_sections: [1, 1, 1, 0],
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

/// 合成 GGUF を書いて開く。
fn open_fixture(dir: &TempDir, tag: &str, tensors: &[TensorSpec]) -> GgufFile {
    let path = dir.path(&format!("{tag}.gguf"));
    write_gguf(&path, &kvs(), tensors);
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

/// 小モデルの `QsaState` を組む (max_ctx は呼び出し側が決める)。
fn state(dir: &TempDir, tag: &str, kpool: u32, max_ctx: usize) -> (GgufFile, Qwen4ExpParams, QsaState) {
    let f = open_fixture(dir, tag, &indexer_tensors());
    let p = params(kpool);
    let s = QsaState::from_gguf(&f, &p, max_ctx).expect("from_gguf");
    (f, p, s)
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

/// 入力ベクトル (`n_embd = 16`)。前半 8 次元が `k_proj` と q head 0、
/// 後半 8 次元が q head 1 を駆動する。
fn cur(head0: [f32; IDX_DIM], head1: [f32; IDX_DIM]) -> Vec<f32> {
    let mut v = head0.to_vec();
    v.extend_from_slice(&head1);
    v
}

fn zeros() -> [f32; IDX_DIM] {
    [0.0; IDX_DIM]
}

/// 平均 (手計算用)。`pool_keys` と同じ順序 (先頭をコピー → 昇順に加算 → 1/n 乗算)。
fn hand_mean(rows: &[[f32; IDX_DIM]]) -> [f32; IDX_DIM] {
    let mut acc = rows[0];
    for r in &rows[1..] {
        for i in 0..IDX_DIM {
            acc[i] += r[i];
        }
    }
    let inv = 1.0f32 / rows.len() as f32;
    for v in acc.iter_mut() {
        *v *= inv;
    }
    acc
}

/// 参照実装の「平均 → norm → rope」をテスト側で組み立てる。
fn expect_pool(mean: &[f32; IDX_DIM], k_norm: &[f32], start: usize) -> Vec<f32> {
    let mut row = mean.to_vec();
    qknorm_rows_inplace(&mut row, 1, 1, IDX_DIM, k_norm, RMS_EPS);
    rope_neox_partial(&mut row, 1, IDX_DIM, N_ROT, start, ROPE_THETA);
    row
}

/// 独立実装の「残す cell」。スコア降順・同点は添字昇順に並べて上位を採る。
fn expected_visible(
    scores: &[f32],
    n_cells: usize,
    kpool: usize,
    top_k: usize,
    cur_cell: usize,
) -> Vec<usize> {
    let n_complete = n_cells / kpool;
    let n_top = (top_k / kpool).min(n_complete);
    let mut order: Vec<usize> = (0..n_complete).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("スコアに NaN")
            .then(a.cmp(&b))
    });
    let mut keep = vec![false; n_cells];
    for &p in &order[..n_top] {
        for c in p * kpool..(p + 1) * kpool {
            keep[c] = true;
        }
    }
    for c in n_complete * kpool..n_cells {
        keep[c] = true;
    }
    keep[cur_cell] = true;
    (0..n_cells).filter(|&c| keep[c]).collect()
}

// ---- 1. pool の平均 -------------------------------------------------------

/// kpool=4。生キーを既知の値にして、平均が手計算 (2 進で厳密) とビット一致する。
/// pool 0 は cell 0 から始まるので rope の位置は 0 (= 恒等) になり、平均と norm だけが残る。
#[test]
fn pool_mean_matches_hand_calc() {
    let dir = TempDir::new("mean");
    let (_f, _p, mut s) = state(&dir, "mean", 4, MAX_CTX);
    let ip = s.layer(ATTN_LAYER).expect("QSA 層が無い").indexer_params();
    assert_eq!(ip.kpool, 4);
    assert_eq!(ip.idx_dim, IDX_DIM);

    let rows: [[f32; IDX_DIM]; 4] = [
        [1.0, 3.0, 5.0, 7.0, -1.0, -3.0, -5.0, -7.0],
        [3.0, 1.0, 7.0, 5.0, -3.0, -1.0, -7.0, -5.0],
        [0.0, 2.0, 4.0, 6.0, 0.0, -2.0, -4.0, -6.0],
        [4.0, 2.0, 6.0, 4.0, -4.0, -2.0, -6.0, -4.0],
    ];
    // 手計算: 各次元の和は [8, 8, 22, 22, -8, -8, -22, -22]、/4 で [2, 2, 5.5, 5.5, -2, -2, -5.5, -5.5]。
    let hand = [2.0f32, 2.0, 5.5, 5.5, -2.0, -2.0, -5.5, -5.5];
    assert_eq!(bits(&hand_mean(&rows)), bits(&hand), "テスト側の手計算");

    // (a) 共有実装の平均そのもの。
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let pooled = pool_keys(&flat, 4, &ip);
    assert_bits_eq(&pooled, &hand, "pool_keys の平均");

    // (b) push_token が同じ平均を作っているか (norm と rope は同じ関数で作る)。
    for (pos, r) in rows.iter().enumerate() {
        s.push_token(ATTN_LAYER, &cur(*r, zeros()), pos)
            .expect("push_token");
        assert_eq!(s.cells(ATTN_LAYER).unwrap(), pos + 1);
    }
    assert_eq!(s.n_pools(ATTN_LAYER).unwrap(), 1, "4 cell で pool は 1 本");
    assert_eq!(s.tail_start(ATTN_LAYER).unwrap(), 4, "tail は cell 4 から");
    assert_bits_eq(s.raw_key(ATTN_LAYER, 2).unwrap(), &rows[2], "生キーは入力をそのまま");
    let want = expect_pool(&hand, &K_NORM, 0);
    assert_bits_eq(s.pooled_keys(ATTN_LAYER).unwrap(), &want, "pool キー (平均→norm→rope)");
}

// ---- 2. norm と rope の順序 ----------------------------------------------

/// kpool=2 で 4 トークン積むと pool 1 は cell 2 から始まる (rope の位置が 0 でない)。
/// 「平均 → norm → rope」の値と一致し、「norm を最後」「平均より先に rope」とは違う。
#[test]
fn pool_uses_mean_then_norm_then_rope() {
    let dir = TempDir::new("order");
    let (_f, _p, mut s) = state(&dir, "order", 2, MAX_CTX);
    let rows: [[f32; IDX_DIM]; 4] = [
        [1.0, 2.0, -1.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.5, -2.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.5, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];
    for (pos, r) in rows.iter().enumerate() {
        s.push_token(ATTN_LAYER, &cur(*r, zeros()), pos)
            .expect("push_token");
    }
    assert_eq!(s.n_pools(ATTN_LAYER).unwrap(), 2);
    let got = s.pooled_keys(ATTN_LAYER).unwrap();
    let got1 = &got[IDX_DIM..2 * IDX_DIM];

    // 手計算の平均 (すべて 2 進で厳密): pool 1 = ((-1.5+0.5)/2, (-1+3)/2, (2+1)/2, (1+1)/2)
    let mean1: [f32; IDX_DIM] = [-0.5, 1.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        bits(&hand_mean(&rows[2..4])),
        bits(&mean1),
        "テスト側の手計算 (pool 1)"
    );

    let want = expect_pool(&mean1, &K_NORM, 2);
    assert_bits_eq(got1, &want, "pool 1 = 平均 → norm → rope(2)");

    // (a) norm を最後に回した値 (rope → norm) とは違う。
    let mut norm_last = mean1.to_vec();
    rope_neox_partial(&mut norm_last, 1, IDX_DIM, N_ROT, 2, ROPE_THETA);
    qknorm_rows_inplace(&mut norm_last, 1, 1, IDX_DIM, &K_NORM, RMS_EPS);
    assert_ne!(bits(got1), bits(&norm_last), "norm を最後に回すと値が変わる");

    // (b) 平均より先に rope を当てた値 (rope → 平均) とも違う。
    let mut r2 = rows[2].to_vec();
    qknorm_rows_inplace(&mut r2, 1, 1, IDX_DIM, &K_NORM, RMS_EPS);
    rope_neox_partial(&mut r2, 1, IDX_DIM, N_ROT, 2, ROPE_THETA);
    let mut r3 = rows[3].to_vec();
    qknorm_rows_inplace(&mut r3, 1, 1, IDX_DIM, &K_NORM, RMS_EPS);
    rope_neox_partial(&mut r3, 1, IDX_DIM, N_ROT, 3, ROPE_THETA);
    let rope_first: Vec<f32> = (0..IDX_DIM).map(|i| (r2[i] + r3[i]) * 0.5).collect();
    assert_ne!(bits(got1), bits(&rope_first), "平均より先に rope を当てると値が変わる");
}

// ---- 3. rope の位置 -------------------------------------------------------

/// pool のキーは**先頭メンバー**の位置で回る (末尾の位置ではない)。
#[test]
fn pool_rope_uses_first_member_position() {
    let dir = TempDir::new("ropepos");
    let (_f, _p, mut s) = state(&dir, "ropepos", 2, MAX_CTX);
    let rows: [[f32; IDX_DIM]; 4] = [
        [1.0, 2.0, -1.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.5, -2.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.5, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];
    for (pos, r) in rows.iter().enumerate() {
        s.push_token(ATTN_LAYER, &cur(*r, zeros()), pos)
            .expect("push_token");
    }
    let got = s.pooled_keys(ATTN_LAYER).unwrap();
    let mean0 = hand_mean(&rows[0..2]);
    let mean1 = hand_mean(&rows[2..4]);

    // pool 0 は cell 0 から: rope(0) は恒等。
    let want0 = expect_pool(&mean0, &K_NORM, 0);
    assert_bits_eq(&got[..IDX_DIM], &want0, "pool 0 は位置 0 で回る");
    let at1 = expect_pool(&mean0, &K_NORM, 1);
    assert_ne!(bits(&got[..IDX_DIM]), bits(&at1), "pool 0 を末尾の位置 1 で回すと違う");

    // pool 1 は cell 2 から: 先頭の位置 2 で回る (末尾の 3 ではない)。
    let want1 = expect_pool(&mean1, &K_NORM, 2);
    assert_bits_eq(&got[IDX_DIM..], &want1, "pool 1 は位置 2 で回る");
    let at3 = expect_pool(&mean1, &K_NORM, 3);
    assert_ne!(bits(&got[IDX_DIM..]), bits(&at3), "pool 1 を末尾の位置 3 で回すと違う");
}

// ---- 4. スコア式 ----------------------------------------------------------

/// `score[p] = Σ_h max(0, dot(q_h, pooled_p)) * (1/sqrt(idx_dim))`。
/// idx_dim=4 なら倍率は厳密に 0.5。負の内積は head ごとに 0 へ落ちる。
#[test]
fn score_is_relu_per_head_then_scaled_sum() {
    let ip = IndexerParams {
        n_head: 2,
        idx_dim: 4,
        top_k: 8,
        kpool: 2,
        eps: RMS_EPS,
    };
    // head 0 は e0、head 1 は e1 を見る。
    let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pooled = [
        2.0f32, 3.0, 0.0, 0.0, // head 0 の dot = 2、head 1 = 3 → (2+3)*0.5
        -2.0, 3.0, 0.0, 0.0, // head 0 は負 → 0、head 1 = 3 → 3*0.5
        -1.0, -1.0, 0.0, 0.0, // 両方負 → 0
    ];
    let scores = score_pools(&q, &pooled, 3, &ip);
    assert_bits_eq(&scores, &[2.5, 1.5, 0.0], "スコア式");
    // 倍率が 1/sqrt(idx_dim) であること (0.5 倍されていないなら 5.0 になる)。
    assert_eq!(scores[0].to_bits(), 2.5f32.to_bits());
    assert_ne!(scores[0].to_bits(), 5.0f32.to_bits());

    // 状態のスコアが「自分の q と pooled から同じ式で出した値」と一致すること。
    let dir = TempDir::new("score");
    let (_f, _p, mut s) = state(&dir, "score", 2, MAX_CTX);
    for pos in 0..4 {
        let h0 = [0.5 * pos as f32 + 1.0, 2.0, -1.0, 0.25, 0.0, 0.0, 0.0, 0.0];
        let h1 = [1.0, -0.5, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0];
        s.push_token(ATTN_LAYER, &cur(h0, h1), pos).expect("push_token");
    }
    let l = s.layer(ATTN_LAYER).unwrap();
    let ip2 = l.indexer_params();
    let np = s.n_pools(ATTN_LAYER).unwrap();
    assert_eq!(np, 2);
    let want = score_pools(
        s.query(ATTN_LAYER).unwrap(),
        s.pooled_keys(ATTN_LAYER).unwrap(),
        np,
        &ip2,
    );
    assert_bits_eq(s.scores(ATTN_LAYER).unwrap(), &want, "状態のスコア");
}

// ---- 5. top-k -------------------------------------------------------------

/// 上位 `min(n_pool, top_k/kpool)` 個が選ばれる。同点は添字の小さい方が勝つ。
#[test]
fn top_k_picks_upper_pools_and_breaks_ties_by_index() {
    let ip = IndexerParams {
        n_head: 1,
        idx_dim: 1,
        top_k: 4,
        kpool: 2,
        eps: RMS_EPS,
    };
    // n_cells=12 (6 pool)、n_top = 4/2 = 2。
    let scores = [1.0f32, 3.0, 3.0, 2.0, 0.0, 3.0];
    // 同点 3.0 が 1, 2, 5 にある。2 本選ぶので 1 と 2 (添字の小さい方) が勝つ。
    assert_eq!(select_cells(&scores, 12, &ip), vec![2, 3, 4, 5]);

    // 残す cell は「選ばれた pool の全 cell + tail + 現在のトークン」。
    // 12 cell はちょうど 6 pool なので tail は無い。cur=11 は pool 5 (未選択) の末尾。
    let vis = select_visible(&scores, 12, &ip, 11).expect("select_visible");
    assert_eq!(vis, vec![2, 3, 4, 5, 11]);

    // 状態でも同じ選択になること (12 token で 6 pool > 4 pool の枠)。
    let dir = TempDir::new("topk");
    let (_f, _p, mut s) = state(&dir, "topk", 2, MAX_CTX);
    for pos in 0..12 {
        let a = 0.25 * ((pos % 5) as f32) + 1.0;
        let b = 0.5 * ((pos % 3) as f32) - 0.5;
        let h0 = [a, b, 1.0, -1.0, 0.5, 0.0, 0.0, 0.0];
        let h1 = [b, a, -0.5, 0.75, 0.0, 0.0, 0.0, 0.0];
        let vis = s
            .push_token(ATTN_LAYER, &cur(h0, h1), pos)
            .expect("push_token");
        let np = s.n_pools(ATTN_LAYER).unwrap();
        let want = expected_visible(
            s.scores(ATTN_LAYER).unwrap(),
            pos + 1,
            2,
            TOP_K,
            pos,
        );
        assert_eq!(vis, want, "pos {pos}: 選ばれた cell が独立実装と違う (n_pool={np})");
        assert!(vis.windows(2).all(|w| w[0] < w[1]), "pos {pos}: 昇順でない");
        assert!(vis.contains(&pos), "pos {pos}: 現在のトークンが入っていない");
        assert!(vis.len() <= TOP_K + 2, "pos {pos}: 残す cell が多すぎる");
    }
    // 6 pool のうち 4 pool しか選ばれないので、全部は残らない。
    let np = s.n_pools(ATTN_LAYER).unwrap();
    assert_eq!(np, 6);
    let all: Vec<usize> = (0..12).collect();
    let last = expected_visible(s.scores(ATTN_LAYER).unwrap(), 12, 2, TOP_K, 11);
    assert_ne!(last, all, "6 pool > 4 pool の枠なので全部は残らない");
}

// ---- 6. dense 一致 --------------------------------------------------------

/// `n_pool <= top_k/kpool` の範囲では全 pool が選ばれるので、返る添字は `0..=pos` と一致する。
#[test]
fn short_context_matches_dense() {
    for &kpool in &[2u32, 4u32] {
        let dir = TempDir::new(&format!("dense{kpool}"));
        let (_f, _p, mut s) = state(&dir, &format!("dense{kpool}"), kpool, MAX_CTX);
        // kpool=2 なら 8 cell (4 pool)、kpool=4 なら 8 cell (2 pool) まで dense と一致。
        let n_top_cells = (TOP_K / kpool as usize) * kpool as usize;
        for pos in 0..=n_top_cells {
            let a = 0.5 * ((pos % 4) as f32) - 0.75;
            let h0 = [a, 1.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0];
            let h1 = [1.0, a, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0];
            let vis = s
                .push_token(ATTN_LAYER, &cur(h0, h1), pos)
                .expect("push_token");
            let want: Vec<usize> = (0..=pos).collect();
            assert_eq!(
                vis, want,
                "kpool={kpool} pos={pos}: 短い文脈で dense と一致しない"
            );
        }
    }
}

// ---- 7. tail と現在のトークン --------------------------------------------

/// tail (未完成 pool の生 cell) は選択に関わらず常に残る。
/// 現在のトークンが未選択の完成 pool の末尾でも、その cell は残る。
#[test]
fn tail_and_current_token_are_always_kept() {
    let ip = IndexerParams {
        n_head: 1,
        idx_dim: 1,
        top_k: 8,
        kpool: 2,
        eps: RMS_EPS,
    };
    // 6 pool・n_top=4。pool 0..3 だけ選ばれる。13 cell なので tail は cell 12。
    let scores = [9.0f32, 9.0, 9.0, 9.0, 0.0, 0.0];
    let vis = select_visible(&scores, 13, &ip, 12).expect("select_visible");
    assert_eq!(vis, vec![0, 1, 2, 3, 4, 5, 6, 7, 12], "tail の cell 12 が残る");

    // cur=11 は pool 5 (未選択) の末尾。それでも残る。
    let vis = select_visible(&scores, 12, &ip, 11).expect("select_visible");
    assert_eq!(vis, vec![0, 1, 2, 3, 4, 5, 6, 7, 11], "現在のトークンが残る");

    // 端数が 1 のとき (n_cells=13) は 6 pool のうち 4 pool + tail。
    //
    // 入力は +1 と -1 の交互。各 pool の平均は厳密に 0 になるので全 pool のスコアが
    // 0 で並ぶ (同点は添字の小さい方が勝つ)。そのため pool 4 以降は選ばれず、
    // 「末尾の完成 pool が選ばれず現在のトークンだけが残る」分岐も実際に通る。
    let dir = TempDir::new("tail");
    let (_f, _p, mut s) = state(&dir, "tail", 2, MAX_CTX);
    let mut saw_cur_alone = false;
    for pos in 0..MAX_CTX {
        let sign = if pos % 2 == 0 { 1.0f32 } else { -1.0f32 };
        let h0 = [sign, sign, sign, sign, 0.0, 0.0, 0.0, 0.0];
        let vis = s
            .push_token(ATTN_LAYER, &cur(h0, zeros()), pos)
            .expect("push_token");
        let n_cells = pos + 1;
        let tail_start = s.tail_start(ATTN_LAYER).unwrap();
        assert_eq!(tail_start, (n_cells / 2) * 2, "tail_start が完成 pool の末尾でない");
        for c in tail_start..n_cells {
            assert!(vis.contains(&c), "pos {pos}: tail の cell {c} が残っていない");
        }
        assert!(vis.contains(&pos), "pos {pos}: 現在のトークンが残っていない");
        if (pos + 1) % 2 == 0 && !vis.contains(&(pos - 1)) {
            // tail は空なので、pos は「選択されなかった完成 pool の末尾」として
            // 明示的に足されている (現在のトークンの分岐)。
            saw_cur_alone = true;
        }
    }
    assert!(
        saw_cur_alone,
        "現在のトークンだけを残す分岐が一度も通っていない (テストの入力を見直す)"
    );
    assert_eq!(s.n_pools(ATTN_LAYER).unwrap(), 8);
}

// ---- 8. 容量と入力の検査 --------------------------------------------------

#[test]
fn capacity_overflow_is_an_error() {
    let dir = TempDir::new("cap");
    let (f, p, mut s) = state(&dir, "cap", 2, 3);
    assert_eq!(s.max_ctx(), 3);
    for pos in 0..3 {
        s.push_token(ATTN_LAYER, &cur(zeros(), zeros()), pos)
            .expect("容量内は通る");
    }
    let e = s
        .push_token(ATTN_LAYER, &cur(zeros(), zeros()), 3)
        .expect_err("容量超過はエラーのはず");
    assert!(
        format!("{e:#}").contains("max_ctx"),
        "エラーに max_ctx が出ていない: {e:#}"
    );

    // max_ctx=0 は組めない。
    let e = match QsaState::from_gguf(&f, &p, 0) {
        Ok(_) => panic!("max_ctx=0 はエラーのはず"),
        Err(e) => e,
    };
    assert!(format!("{e:#}").contains("max_ctx"), "{e:#}");
}

#[test]
fn input_checks_reject_wrong_layer_and_position() {
    let dir = TempDir::new("input");
    let (_f, _p, mut s) = state(&dir, "input", 2, MAX_CTX);

    // ssm 層 (層 0) は QSA を持たない。
    assert!(!s.is_qsa(0));
    assert!(s.is_qsa(ATTN_LAYER));
    assert!(s.push_token(0, &cur(zeros(), zeros()), 0).is_err());

    // 位置は積んだ順でなければならない。
    s.push_token(ATTN_LAYER, &cur(zeros(), zeros()), 0)
        .expect("pos 0");
    assert!(s.push_token(ATTN_LAYER, &cur(zeros(), zeros()), 2).is_err());

    // 入力長の取り違え。
    assert!(s.push_token(ATTN_LAYER, &[0.0; 3], 1).is_err());

    // QSA 層が 1 本も無い params は組めない。
    let mut p = params(2);
    p.compress_ratios = vec![0; N_LAYER];
    let f = open_fixture(&dir, "input2", &indexer_tensors());
    assert!(QsaState::from_gguf(&f, &p, MAX_CTX).is_err());

    // QSA が ssm 層に付いている params は組めない。
    let mut p = params(2);
    p.compress_ratios[0] = 2;
    assert!(QsaState::from_gguf(&f, &p, MAX_CTX).is_err());

    // テンソルが欠けていれば落ちる。
    let mut tensors = indexer_tensors();
    tensors.retain(|t| !t.name.ends_with("k_proj.weight"));
    let f = open_fixture(&dir, "input3", &tensors);
    assert!(QsaState::from_gguf(&f, &params(2), MAX_CTX).is_err());
}
