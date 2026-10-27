//! full attention 層のバッチ版 (`FullAttn::forward_batch` / `AttnLayer::attend_batch`)
//! が 1 トークンずつの `forward_token` と**ビット一致**することを固定するテスト。
//!
//! 実モデル (111 GB) は使わない。ヘッダを書いただけの合成 GGUF に、head ごとに
//! 値の違う F32 の重みを置いて小さな full attention 層を組み、同じ入力列を
//! 2 つの `FullAttn` に「逐次」と「チャンク」で食わせて比べる。
//!
//! 検査項目 (タスク (a)(b)(c)):
//! - (a) 空のキャッシュから 5 トークン: 逐次 5 回と `forward_batch`(5) が一致
//! - (b) キャッシュに 3 トークン積んだ状態から 5 トークン (位置が進んだ状態)
//! - (c) head 数 1 / kv head 数 1 と、head 数 4 / kv head 数 2 (GQA) の両方
//!
//! 比較は `to_bits()`。出力だけでなく、KV キャッシュの中身と `pos` も比べる
//! (attention の値が合っていても積み方が違えば次のチャンクで壊れる)。
//!
//! 重みは head・行ごとに違う値にしてあるので、q/k/v の head の取り違えや
//! GQA の対応 (`kv_head`) の誤りは値の差として出る。

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::Qwen4ExpParams;
use denno_engine::model::qwen4exp_attn::{attend_batch_calls, FullAttn};

/// プロセス通算のカウンタ (`attend_batch_calls`) を読むテストは直列に走らせる。
/// 並列だと他のテストの呼び出しが混ざり、「+1 回」を検査できない。
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---- 小モデルの諸元 (構造は本物と同じ、寸法だけ小さい) --------------------
const N_LAYER: usize = 4;
const N_EMBD: usize = 16;
const HEAD_DIM: usize = 8;
const N_ROT: usize = 4;
const ROPE_THETA: f32 = 1e4;
const RMS_EPS: f32 = 1e-6;
/// full attention は層 3 だけ ((3 + 1) % 4 == 0)。
const ATTN_LAYER: usize = 3;

/// (head 数, kv head 数) の組み合わせ。(c) の「1 と複数の両方」を満たす。
const CASES: [(usize, usize); 3] = [(1, 1), (2, 1), (4, 2)];

// ---- GGUF 書き出し -------------------------------------------------------
const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const TY_F32: u32 = 0;

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
            "denno_qwen4exp_attn_batch_{tag}_{}_{n}",
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

// ---- 決定的な擬似乱数 (外部 crate を使わない) ----------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(2) | 1)
    }
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as u32;
        u as f32 / (1u32 << 24) as f32
    }
    /// [-0.5, 0.5)
    fn next_sym(&mut self) -> f32 {
        self.next_unit() - 0.5
    }
}

fn f32_data(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `k` 列 `n` 行 (GGUF は ne0 = 列が連続) の F32 行列を擬似乱数で埋める。
fn f32_matrix(k: usize, n: usize, rng: &mut Rng) -> Vec<u8> {
    let mut v = Vec::with_capacity(k * n);
    for _ in 0..k * n {
        v.push(rng.next_sym());
    }
    f32_data(&v)
}

fn params(n_head: usize, n_head_kv: usize, dh: usize) -> Qwen4ExpParams {
    Qwen4ExpParams {
        n_layer: N_LAYER,
        n_ctx: 64,
        n_embd: N_EMBD,
        n_head,
        n_head_kv,
        n_embd_head_k: dh,
        n_embd_head_v: dh,
        rms_eps: RMS_EPS,
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

/// full attention 層 1 本ぶんのテンソル (全部 F32、値は擬似乱数)。
///
/// head ごとに別の行・別の列を読むので、head の取り違えは値の差になる。
fn layer_tensors(n_head: usize, n_head_kv: usize, dh: usize, seed: u64) -> Vec<TensorSpec> {
    let n = |s: &str| format!("blk.{ATTN_LAYER}.{s}");
    let q_rows = 2 * n_head * dh;
    let kv_rows = n_head_kv * dh;
    let mut rng = Rng::new(seed);
    let spec = |name: String, dims: Vec<u64>, k: usize, rows: usize, rng: &mut Rng| TensorSpec {
        name,
        dims,
        ty: TY_F32,
        data: f32_matrix(k, rows, rng),
    };
    let q = spec(
        n("attn_q.weight"),
        vec![N_EMBD as u64, q_rows as u64],
        N_EMBD,
        q_rows,
        &mut rng,
    );
    let k = spec(
        n("attn_k.weight"),
        vec![N_EMBD as u64, kv_rows as u64],
        N_EMBD,
        kv_rows,
        &mut rng,
    );
    let v = spec(
        n("attn_v.weight"),
        vec![N_EMBD as u64, kv_rows as u64],
        N_EMBD,
        kv_rows,
        &mut rng,
    );
    let o = spec(
        n("attn_output.weight"),
        vec![(n_head * dh) as u64, N_EMBD as u64],
        n_head * dh,
        N_EMBD,
        &mut rng,
    );
    // norm の重みは 0.5..1.5 (0 だと正規化が潰れる)。
    let qn: Vec<f32> = (0..dh).map(|_| 0.5 + rng.next_unit()).collect();
    let kn: Vec<f32> = (0..dh).map(|_| 0.5 + rng.next_unit()).collect();
    vec![
        q,
        k,
        v,
        o,
        TensorSpec {
            name: n("attn_q_norm.weight"),
            dims: vec![dh as u64],
            ty: TY_F32,
            data: f32_data(&qn),
        },
        TensorSpec {
            name: n("attn_k_norm.weight"),
            dims: vec![dh as u64],
            ty: TY_F32,
            data: f32_data(&kn),
        },
    ]
}

fn kvs() -> Vec<(&'static str, Kv)> {
    vec![("general.architecture", Kv::Str("qwen4exp"))]
}

/// 合成 GGUF を書いて full attention 層を組む (QSA は使わない)。
fn build(dir: &TempDir, n_head: usize, n_head_kv: usize, dh: usize, max_ctx: usize) -> FullAttn {
    let path = dir.path(&format!("nh{n_head}_kv{n_head_kv}_dh{dh}.gguf"));
    let seed = 1000 + (n_head * 31 + n_head_kv + dh) as u64;
    write_gguf(&path, &kvs(), &layer_tensors(n_head, n_head_kv, dh, seed));
    let f = GgufFile::open(&path).expect("合成 GGUF を開けない");
    let p = params(n_head, n_head_kv, dh);
    FullAttn::from_gguf_with(&f, &p, max_ctx, false).expect("FullAttn::from_gguf")
}

/// 決定的な入力列 (トークンごとに違う値)。
fn tokens(from: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let t = from + i;
            (0..N_EMBD)
                .map(|j| {
                    let x = ((t * 17 + j * 5) % 23) as f32 - 11.0;
                    x * 0.05 + (t as f32) * 0.01
                })
                .collect()
        })
        .collect()
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
            "{what}: 要素 {i} が {} ({:#010x}) と {} ({:#010x}) で違う",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

/// 逐次 5 回の出力と `forward_batch` の出力を比べる共通部分。
fn compare_paths(fa_seq: &mut FullAttn, fa_bat: &mut FullAttn, curs: &[Vec<f32>], tag: &str) {
    let before = attend_batch_calls();
    let bat = fa_bat
        .forward_batch(ATTN_LAYER, curs)
        .expect("forward_batch");
    let after = attend_batch_calls();
    assert_eq!(
        after - before,
        1,
        "{tag}: attend_batch が 1 回呼ばれていない (バッチ経路が発動していない)"
    );
    assert_eq!(bat.len(), curs.len(), "{tag}: 出力トークン数が違う");

    let mut seq = Vec::with_capacity(curs.len());
    for cur in curs {
        seq.push(fa_seq.forward_token(ATTN_LAYER, cur).expect("forward_token"));
    }

    for t in 0..curs.len() {
        assert_bits_eq(&bat[t], &seq[t], &format!("{tag}: トークン {t} の出力"));
        assert!(
            bat[t].iter().all(|v| v.is_finite()),
            "{tag}: トークン {t} に NaN/inf"
        );
    }
    // 出力が全トークン同値だと「計算が効いていない」ので、差があることも見る。
    assert!(
        (1..curs.len()).any(|t| bits(&bat[t]) != bits(&bat[0])),
        "{tag}: 全トークンの出力が同一 (head/位置が効いていない)"
    );

    // KV キャッシュの中身と位置も一致していること (積み方が違えば次で壊れる)。
    assert_eq!(
        fa_bat.cache().len(ATTN_LAYER).expect("len"),
        fa_seq.cache().len(ATTN_LAYER).expect("len"),
        "{tag}: キャッシュ長が違う"
    );
    assert_bits_eq(
        fa_bat.cache().k(ATTN_LAYER).expect("k"),
        fa_seq.cache().k(ATTN_LAYER).expect("k"),
        &format!("{tag}: キャッシュ k"),
    );
    assert_bits_eq(
        fa_bat.cache().v(ATTN_LAYER).expect("v"),
        fa_seq.cache().v(ATTN_LAYER).expect("v"),
        &format!("{tag}: キャッシュ v"),
    );
    assert_eq!(fa_bat.pos(), fa_seq.pos(), "{tag}: pos が違う");
}

// ---------------------------------------------------------------------------
// (a) 空のキャッシュから 5 トークン
// ---------------------------------------------------------------------------

#[test]
fn batch_matches_sequential_from_empty_cache() {
    let _g = lock();
    let dir = TempDir::new("empty");
    for (nh, nkv) in CASES {
        let tag = format!("heads {nh} / kv {nkv}");
        let mut seq = build(&dir, nh, nkv, HEAD_DIM, 32);
        let mut bat = build(&dir, nh, nkv, HEAD_DIM, 32);
        let curs = tokens(0, 5);
        compare_paths(&mut seq, &mut bat, &curs, &tag);
        assert_eq!(seq.cache().len(ATTN_LAYER).expect("len"), 5);
    }
}

// ---------------------------------------------------------------------------
// (b) キャッシュに 3 トークン積んだ状態から 5 トークン (位置が進んだ状態)
// ---------------------------------------------------------------------------

#[test]
fn batch_matches_sequential_with_prefilled_cache() {
    let _g = lock();
    let dir = TempDir::new("prefilled");
    for (nh, nkv) in CASES {
        let tag = format!("heads {nh} / kv {nkv} (継続)");
        let mut seq = build(&dir, nh, nkv, HEAD_DIM, 32);
        let mut bat = build(&dir, nh, nkv, HEAD_DIM, 32);
        // 先頭 3 トークンは両方とも逐次で積む (= 同じキャッシュを作る)。
        let head = tokens(0, 3);
        for cur in &head {
            let a = seq.forward_token(ATTN_LAYER, cur).expect("seq 3");
            let b = bat.forward_token(ATTN_LAYER, cur).expect("bat 3");
            assert_bits_eq(&b, &a, &format!("{tag}: 前置 3 トークン"));
        }
        let curs = tokens(3, 5);
        compare_paths(&mut seq, &mut bat, &curs, &tag);
        assert_eq!(seq.cache().len(ATTN_LAYER).expect("len"), 8);
    }
}

// ---------------------------------------------------------------------------
// (c) head 数・kv head 数ごとに、複数チャンクを跨いでも一致する
// ---------------------------------------------------------------------------

/// 2 チャンク (3 + 4) を続けて流す。2 チャンク目はキャッシュに 3 行積まれた状態で
/// 始まるので、(b) の「位置が進んだ状態」を head 構成ごとに 1 本で確かめることになる。
#[test]
fn batch_matches_sequential_across_two_chunks() {
    let _g = lock();
    let dir = TempDir::new("chunks");
    for (nh, nkv) in CASES {
        let tag = format!("heads {nh} / kv {nkv} (2 チャンク)");
        let mut seq = build(&dir, nh, nkv, HEAD_DIM, 32);
        let mut bat = build(&dir, nh, nkv, HEAD_DIM, 32);
        compare_paths(&mut seq, &mut bat, &tokens(0, 3), &format!("{tag} 1"));
        compare_paths(&mut seq, &mut bat, &tokens(3, 4), &format!("{tag} 2"));
        assert_eq!(seq.cache().len(ATTN_LAYER).expect("len"), 7);
    }
}

// ---------------------------------------------------------------------------
// 1 トークンのチャンクと範囲外の層は逐次経路 / エラー
// ---------------------------------------------------------------------------

#[test]
fn single_token_batch_falls_back_and_bad_inputs_are_rejected() {
    let _g = lock();
    let dir = TempDir::new("edges");
    let mut fa = build(&dir, 2, 1, HEAD_DIM, 4);
    let before = attend_batch_calls();
    let curs = tokens(0, 1);
    let one = fa.forward_batch(ATTN_LAYER, &curs).expect("1 トークン");
    assert_eq!(one.len(), 1);
    assert_eq!(
        attend_batch_calls(),
        before,
        "1 トークンでバッチ版が呼ばれた (逐次に落とす約束)"
    );

    // ssm 層 (この合成モデルでは層 0)
    let err = fa.forward_batch(0, &tokens(0, 2)).expect_err("ssm 層");
    assert!(format!("{err:#}").contains("担当外"), "{err:#}");

    // 範囲外
    let err = fa.forward_batch(N_LAYER, &tokens(0, 2)).expect_err("範囲外");
    assert!(format!("{err:#}").contains("範囲外"), "{err:#}");

    // 入力長の取り違え
    let bad = vec![vec![0.0f32; N_EMBD - 1]; 2];
    let err = fa.forward_batch(ATTN_LAYER, &bad).expect_err("入力長");
    assert!(format!("{err:#}").contains("入力長"), "{err:#}");

    // 容量超過 (max_ctx = 4 に 5 トークン)
    let mut small = build(&dir, 2, 1, HEAD_DIM, 4);
    let err = small
        .forward_batch(ATTN_LAYER, &tokens(0, 5))
        .expect_err("容量超過");
    assert!(format!("{err:#}").contains("max_ctx"), "{err:#}");
}

// ---------------------------------------------------------------------------
// CPU マイクロベンチ (実機の prefill 計測ではない。main が実機で測る前の目安)
// ---------------------------------------------------------------------------

/// 実モデルと同じ形の attend を CPU で A/B する。
///
/// 24 head / 2 kv head / head_dim 256 / チャンク 64 / 位置 738 から 802。
/// 測るのは **attention 本体だけ** (投影は含めない)。逐次 = 1 トークンずつ
/// push して `attend`、バッチ = 全部 push して `attend_batch`。
///
/// `--ignored` で手動実行する。数字は CPU・単一スレッド・合成 KV のものなので、
/// 実機 (802 トークン / CUDA) の ms/tok と直接は比べられない。
#[test]
#[ignore = "CPU マイクロベンチ (実機の prefill 計測ではない)"]
fn bench_attend_sequential_vs_batch() {
    let _g = lock();
    const NH: usize = 24;
    const NKV: usize = 2;
    const DH: usize = 256;
    const NT: usize = 64;
    const BASE: usize = 738;

    let dir = TempDir::new("bench");
    let mut fa = build(&dir, NH, NKV, DH, 1024);
    let row = NKV * DH;
    let qd = NH * DH;

    // 位置ごとに違う KV を作る (全部同じ値だと内積が同じになる)。
    let mut rng = Rng::new(7);
    let kbuf: Vec<f32> = (0..(BASE + NT) * row).map(|_| rng.next_sym()).collect();
    let vbuf: Vec<f32> = (0..(BASE + NT) * row).map(|_| rng.next_sym()).collect();
    let qs: Vec<f32> = (0..NT * qd).map(|_| rng.next_sym()).collect();

    let mut seq_ms = f64::INFINITY;
    let mut bat_ms = f64::INFINITY;
    let reps = 3;
    for _ in 0..reps {
        // --- 逐次: t ごとに push → attend(0..base+t+1) ---
        fa.reset();
        for j in 0..BASE {
            fa.cache_mut()
                .push(ATTN_LAYER, &kbuf[j * row..(j + 1) * row], &vbuf[j * row..(j + 1) * row])
                .expect("push");
        }
        let t0 = std::time::Instant::now();
        for t in 0..NT {
            let j = BASE + t;
            fa.cache_mut()
                .push(ATTN_LAYER, &kbuf[j * row..(j + 1) * row], &vbuf[j * row..(j + 1) * row])
                .expect("push");
            let out = {
                let l = fa.layer(ATTN_LAYER).expect("layer");
                l.attend(&qs[t * qd..(t + 1) * qd], fa.cache(), ATTN_LAYER)
                    .expect("attend")
            };
            assert_eq!(out.len(), qd);
        }
        seq_ms = seq_ms.min(t0.elapsed().as_secs_f64() * 1e3);

        // --- バッチ: 全部 push → attend_batch ---
        fa.reset();
        for j in 0..BASE {
            fa.cache_mut()
                .push(ATTN_LAYER, &kbuf[j * row..(j + 1) * row], &vbuf[j * row..(j + 1) * row])
                .expect("push");
        }
        let t0 = std::time::Instant::now();
        for t in 0..NT {
            let j = BASE + t;
            fa.cache_mut()
                .push(ATTN_LAYER, &kbuf[j * row..(j + 1) * row], &vbuf[j * row..(j + 1) * row])
                .expect("push");
        }
        let out = {
            let l = fa.layer(ATTN_LAYER).expect("layer");
            l.attend_batch(&qs, NT, fa.cache(), ATTN_LAYER)
                .expect("attend_batch")
        };
        assert_eq!(out.len(), NT * qd);
        bat_ms = bat_ms.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "CPU attend 1 層 1 チャンク (heads {NH}/kv {NKV}/dh {DH}/nt {NT}/pos {BASE}..{}): \
         逐次 {seq_ms:.2} ms / バッチ {bat_ms:.2} ms / 比 {:.2}x",
        BASE + NT,
        seq_ms / bat_ms
    );
    assert!(bat_ms.is_finite() && seq_ms.is_finite());
}
