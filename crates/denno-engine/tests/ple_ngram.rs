//! P-MOE-E2: PLE の n-gram 索引計算のテスト。
//!
//! 実モデルに依存しないよう、このファイル内で最小 GGUF を書いて検証する。
//! GPU / CUDA / 推論経路は一切通さない。
//!
//! 索引の期待値は **手計算** (数式に実データを当てて求めた値) を直書きしている。
//! 実装の出力をコピーしないこと。
//!
//! 実データ (実 GGUF のメタデータから取得した値):
//! ```text
//! qwen4exp.ple.layers            = [1]
//! qwen4exp.ple.ngram_size        = 3
//! qwen4exp.ple.heads_per_ngram   = 8
//! qwen4exp.ple.eos_token_id      = 248044
//! qwen4exp.embedding_length_per_layer_input = 160
//! qwen4exp.ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]
//! per_layer_token_embd.weight    : dims=[160, 320001536] ty=IQ4_NL
//! ```

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::kquant::{dequant_row, GgmlType};
use denno_engine::ple::{PleNgram, PleTable, PLE_TABLE_NAME};

const ALIGN: usize = 32;

// GGUF メタデータ型 id
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;
const GT_U64: u32 = 10;

// ggml_type id
const TY_IQ4_NL: u32 = 20;

// ---- 実データ ----------------------------------------------------------------

const EOS: u32 = 248044;
const N_DIM: usize = 160;
const TABLE_ROWS: u64 = 320_001_536;

const REAL_MULT: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];

const REAL_OFF: [u64; 16] = [
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
    180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
];

const REAL_VOC: [u64; 16] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];

// ---- 最小 GGUF の書き出し ----------------------------------------------------

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// メタデータ 1 個分。
#[derive(Clone)]
enum Kv {
    Str(&'static str),
    U32(u32),
    U64(u64),
    /// u64 の配列 (実ファイルの ple.* はこの型)
    ArrU64(Vec<u64>),
    /// u32 の配列 (layers など)
    ArrU32(Vec<u32>),
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::Str(s) => {
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                push_str(b, s);
            }
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::U64(v) => {
                b.extend_from_slice(&GT_U64.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::ArrU64(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U64.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Kv::ArrU32(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
}

struct TensorSpec {
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
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
        push_str(&mut head, t.name);
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
        let p = std::env::temp_dir().join(format!("denno_ple_ngram_{tag}_{}_{n}", std::process::id()));
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

/// 決定的な擬似乱数 (xorshift32)。外部クレートに依存しない。
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

const ROW_BYTES: usize = (N_DIM / 32) * 18; // 5 ブロック × 18 B = 90

/// 合成 IQ4_NL 表の生バイト列 (rows × 90 B)。
fn synth_rows(rows: usize, seed: u32) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let scales: [u16; 4] = [0x3C00, 0x3800, 0x3400, 0x4000]; // 1.0, 0.5, 0.25, 2.0
    let mut out = Vec::with_capacity(rows * ROW_BYTES);
    for _ in 0..rows {
        for _ in 0..(N_DIM / 32) {
            let d = scales[(rng.next_u32() & 3) as usize];
            out.extend_from_slice(&d.to_le_bytes());
            for _ in 0..16 {
                out.push((rng.next_u32() & 0xff) as u8);
            }
        }
    }
    assert_eq!(out.len(), rows * ROW_BYTES);
    out
}

// ---- メタデータの組み立て ----------------------------------------------------

/// 実ファイルのメタデータ (表テンソルは含めない = 合成 GGUF には無い)。
fn real_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.ple.layers", Kv::ArrU64(vec![1])),
        ("qwen4exp.ple.ngram_size", Kv::U32(3)),
        ("qwen4exp.ple.heads_per_ngram", Kv::U32(8)),
        ("qwen4exp.ple.conv_kernel", Kv::U32(4)),
        ("qwen4exp.ple.eos_token_id", Kv::U32(EOS)),
        (
            "qwen4exp.embedding_length_per_layer_input",
            Kv::U32(N_DIM as u32),
        ),
        (
            "qwen4exp.ple.layer_multipliers",
            Kv::ArrU64(REAL_MULT.to_vec()),
        ),
        ("qwen4exp.ple.head_offsets", Kv::ArrU64(REAL_OFF.to_vec())),
        (
            "qwen4exp.ple.head_vocab_sizes",
            Kv::ArrU64(REAL_VOC.to_vec()),
        ),
    ]
}

fn set_kv(kvs: &mut [(&'static str, Kv)], key: &str, v: Kv) {
    for (k, slot) in kvs.iter_mut() {
        if *k == key {
            *slot = v;
            return;
        }
    }
    panic!("{key} が無い");
}

fn drop_kv(kvs: &mut Vec<(&'static str, Kv)>, key: &str) {
    let before = kvs.len();
    kvs.retain(|(k, _)| *k != key);
    assert_eq!(kvs.len(), before - 1, "{key} を消せていない");
}

/// 実メタデータだけの GGUF を書いて開く (表テンソルなし)。
fn open_real(tag: &str) -> (TempDir, GgufFile) {
    let dir = TempDir::new(tag);
    let path = dir.path("ple_meta.gguf");
    write_gguf(&path, &real_kvs(), &[]);
    let f = GgufFile::open(&path).expect("open");
    (dir, f)
}

/// 小さい合成メタデータ + 64 行の表を持つ GGUF (gather の配線テスト用)。
///
/// 乗数を [1, 2, 3] にして手計算できるようにしてある。
const SMALL_ROWS: usize = 64;
const SMALL_EOS: u32 = 5;

fn small_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.ple.layers", Kv::ArrU64(vec![1])),
        ("qwen4exp.ple.ngram_size", Kv::U32(3)),
        ("qwen4exp.ple.heads_per_ngram", Kv::U32(8)),
        ("qwen4exp.ple.eos_token_id", Kv::U32(SMALL_EOS)),
        (
            "qwen4exp.embedding_length_per_layer_input",
            Kv::U32(N_DIM as u32),
        ),
        ("qwen4exp.ple.layer_multipliers", Kv::ArrU64(vec![1, 2, 3])),
        (
            "qwen4exp.ple.head_offsets",
            Kv::ArrU64((0..16).map(|h| (h * 4) as u64).collect()),
        ),
        ("qwen4exp.ple.head_vocab_sizes", Kv::ArrU64(vec![4; 16])),
    ]
}

fn open_small(tag: &str, seed: u32) -> (TempDir, GgufFile, Vec<u8>) {
    let dir = TempDir::new(tag);
    let path = dir.path("ple_small.gguf");
    let data = synth_rows(SMALL_ROWS, seed);
    let tensors = [TensorSpec {
        name: PLE_TABLE_NAME,
        dims: vec![N_DIM as u64, SMALL_ROWS as u64],
        ty: TY_IQ4_NL,
        data: data.clone(),
    }];
    write_gguf(&path, &small_kvs(), &tensors);
    let f = GgufFile::open(&path).expect("open");
    (dir, f, data)
}

// ---- 1. ヘッド数 -------------------------------------------------------------

#[test]
fn real_metadata_gives_sixteen_heads() {
    let (_d, f) = open_real("heads");
    let ng = PleNgram::from_meta(&f).expect("from_meta");
    assert_eq!(ng.n_heads(), 16);
    assert_eq!(ng.n_gram(), 3);
    assert_eq!(ng.eos(), EOS);
}

/// 実データのヘッド範囲が表の行数に収まること (タスク指定の検算)。
#[test]
fn real_head_ranges_fit_table_rows() {
    let mut max_end = 0u64;
    for h in 0..16 {
        let end = REAL_OFF[h] + REAL_VOC[h];
        assert!(end <= TABLE_ROWS, "ヘッド {h} の終端 {end} が行数 {TABLE_ROWS} を超える");
        max_end = max_end.max(end);
    }
    assert_eq!(max_end, 320_001_446, "最大終端 (offset[15] + vocab[15])");
    assert!(max_end <= TABLE_ROWS);
}

// ---- 2. 既知ベクトル (手計算の期待値) ----------------------------------------

/// 実データの乗数・オフセット・語彙サイズで 4 本の ctx を引く。
///
/// 期待値の求め方 (手計算):
/// ```text
/// mult = [23703573157769, 20109073645365, 8052911324071], eos = 248044
/// mixed(2) = ctx[0]*mult[0] ^ ctx[1]*mult[1]
/// mixed(3) = ctx[0]*mult[0] ^ ctx[1]*mult[1] ^ ctx[2]*mult[2]
/// row(h)   = mixed(n) % vocab[h] + offset[h]   (n = h/8 + 2)
/// ```
#[test]
fn known_vectors_match_hand_computed_indices() {
    let (_d, f) = open_real("known");
    let ng = PleNgram::from_meta(&f).expect("from_meta");

    // ctx = [100, 200, eos] → n=3 は eos を含む組
    let a = ng.indices(&[100, 200, EOS]).expect("indices");
    assert_eq!(
        &a[..8],
        &[3789882, 37548566, 54429218, 66063567, 90322970, 105076404, 127206819, 156714535],
        "ctx=[100,200,eos] の n=2"
    );
    assert_eq!(
        &a[8..],
        &[176591313, 190338839, 214863519, 230014596, 240666691, 271678783, 282096113, 314782342],
        "ctx=[100,200,eos] の n=3"
    );

    // ctx = [100, 200, 300]
    let b = ng.indices(&[100, 200, 300]).expect("indices");
    assert_eq!(
        &b[..8],
        &[3789882, 37548566, 54429218, 66063567, 90322970, 105076404, 127206819, 156714535],
        "ctx=[100,200,300] の n=2 (n=2 は ctx[0..2] だけなので a と同じ)"
    );
    assert_eq!(
        &b[8..],
        &[178314574, 193067435, 213616235, 229492120, 246876475, 264261580, 290056942, 319034207],
        "ctx=[100,200,300] の n=3"
    );

    // ctx = [100, 200] (履歴不足 → ctx[2] = eos) は a と同一
    let c = ng.indices(&[100, 200]).expect("indices");
    assert_eq!(c, a, "履歴不足の古い側は eos として扱う");

    // ctx = [eos, 200, 300] → ctx[0] はカットしないので n=2 は eos を含む組
    let d = ng.indices(&[EOS, 200, 300]).expect("indices");
    assert_eq!(
        &d[..8],
        &[16967003, 36865578, 41224014, 76264126, 82312536, 111936038, 137253066, 159321812],
        "ctx=[eos,200,300] の n=2 (ctx[0]=eos はカットしない)"
    );
    assert_eq!(
        &d[8..],
        &[162951404, 192807203, 202990777, 229559485, 245602902, 262704941, 281974310, 300084881],
        "ctx=[eos,200,300] の n=3"
    );

    // 4 本とも 16 個
    for (name, v) in [("a", &a), ("b", &b), ("c", &c), ("d", &d)] {
        assert_eq!(v.len(), 16, "{name}");
    }
    // ctx[0] が違えば結果も違う (取り違えていない)
    assert_ne!(a, d);
}

// ---- 3. 全添字がヘッド範囲に入る ---------------------------------------------

/// 1000 通りの擬似乱数 ctx で、各 index が [offset[h], offset[h] + vocab[h]) に入ること。
#[test]
fn random_contexts_stay_in_head_ranges() {
    let (_d, f) = open_real("ranges");
    let ng = PleNgram::from_meta(&f).expect("from_meta");
    let mut rng = Rng::new(0xC0FF_EE01);

    for t in 0..1000u32 {
        // 長さ 1..=6 (n_gram = 3 を超える長さも渡す: 先頭 3 個だけ使う)
        let len = 1 + (rng.next_u32() % 6) as usize;
        let mut ctx = Vec::with_capacity(len);
        for _ in 0..len {
            let r = rng.next_u32();
            // 1/8 の確率で eos を混ぜる (カット分岐を通す)
            let tok = if r % 8 == 0 { EOS } else { r % 300_000 };
            ctx.push(tok);
        }
        let idx = ng.indices(&ctx).expect("indices");
        assert_eq!(idx.len(), 16);
        for h in 0..16 {
            let lo = REAL_OFF[h] as i64;
            let hi = lo + REAL_VOC[h] as i64;
            let v = idx[h] as i64;
            assert!(
                v >= lo && v < hi,
                "t={t} ctx={ctx:?} ヘッド {h}: index {v} が [{lo}, {hi}) の外"
            );
        }
    }
}

// ---- 4. from_meta の検証 -----------------------------------------------------

#[test]
fn from_meta_rejects_missing_keys() {
    for key in [
        "qwen4exp.ple.layers",
        "qwen4exp.ple.ngram_size",
        "qwen4exp.ple.heads_per_ngram",
        "qwen4exp.ple.eos_token_id",
        "qwen4exp.embedding_length_per_layer_input",
        "qwen4exp.ple.layer_multipliers",
        "qwen4exp.ple.head_offsets",
        "qwen4exp.ple.head_vocab_sizes",
    ] {
        let dir = TempDir::new("missing");
        let path = dir.path("m.gguf");
        let mut kvs = real_kvs();
        drop_kv(&mut kvs, key);
        write_gguf(&path, &kvs, &[]);
        let f = GgufFile::open(&path).expect("open");
        let e = PleNgram::from_meta(&f).expect_err("キー欠落はエラー");
        let msg = e.to_string();
        assert!(msg.contains(key), "エラーにキー名が無い: {msg}");
    }
}

#[test]
fn from_meta_rejects_wrong_array_lengths() {
    // layer_multipliers が n_gram 個でない
    let dir = TempDir::new("len_mult");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    set_kv(&mut kvs, "qwen4exp.ple.layer_multipliers", Kv::ArrU64(vec![1, 2]));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("要素数不一致はエラー");
    assert!(e.to_string().contains("layer_multipliers"), "{e}");

    // head_offsets が 15 個
    let dir = TempDir::new("len_off");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    set_kv(
        &mut kvs,
        "qwen4exp.ple.head_offsets",
        Kv::ArrU64(REAL_OFF[..15].to_vec()),
    );
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("要素数不一致はエラー");
    assert!(e.to_string().contains("head_offsets"), "{e}");

    // head_vocab_sizes が 17 個
    let dir = TempDir::new("len_voc");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    let mut voc = REAL_VOC.to_vec();
    voc.push(20000181);
    set_kv(&mut kvs, "qwen4exp.ple.head_vocab_sizes", Kv::ArrU64(voc));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("要素数不一致はエラー");
    assert!(e.to_string().contains("head_vocab_sizes"), "{e}");
}

#[test]
fn from_meta_rejects_bad_scalars() {
    // ngram_size = 1
    let dir = TempDir::new("ng1");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    set_kv(&mut kvs, "qwen4exp.ple.ngram_size", Kv::U32(1));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("n_gram < 2 はエラー");
    assert!(e.to_string().contains("ngram_size"), "{e}");

    // heads_per_ngram = 0
    let dir = TempDir::new("pg0");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    set_kv(&mut kvs, "qwen4exp.ple.heads_per_ngram", Kv::U32(0));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("per_gram = 0 はエラー");
    assert!(e.to_string().contains("heads_per_ngram"), "{e}");

    // layers が 2 個
    let dir = TempDir::new("layers2");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    set_kv(&mut kvs, "qwen4exp.ple.layers", Kv::ArrU64(vec![1, 2]));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("layers が 2 個はエラー");
    assert!(e.to_string().contains("ple.layers"), "{e}");

    // head_vocab_sizes に 0
    let dir = TempDir::new("voc0");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    let mut voc = REAL_VOC.to_vec();
    voc[3] = 0;
    set_kv(&mut kvs, "qwen4exp.ple.head_vocab_sizes", Kv::ArrU64(voc));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("vocab = 0 はエラー");
    assert!(e.to_string().contains("head_vocab_sizes[3]"), "{e}");
}

/// `offset + vocab` が表の行数を超える場合は弾く (表テンソルがある場合)。
#[test]
fn from_meta_rejects_head_range_over_table_rows() {
    // 行数 64・ヘッド範囲の最大 64 → 通る
    let (_d, f, _data) = open_small("rows_ok", 0x1234);
    assert!(PleNgram::from_meta(&f).is_ok(), "64 行ちょうどは通る");

    // vocab を 5 にすると最大 65 > 64 で弾く
    let dir = TempDir::new("rows_ng");
    let path = dir.path("m.gguf");
    let mut kvs = small_kvs();
    set_kv(&mut kvs, "qwen4exp.ple.head_vocab_sizes", Kv::ArrU64(vec![5; 16]));
    let data = synth_rows(SMALL_ROWS, 0x1234);
    write_gguf(
        &path,
        &kvs,
        &[TensorSpec {
            name: PLE_TABLE_NAME,
            dims: vec![N_DIM as u64, SMALL_ROWS as u64],
            ty: TY_IQ4_NL,
            data,
        }],
    );
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("行数超過はエラー");
    let msg = e.to_string();
    assert!(msg.contains("行数"), "{msg}");
    assert!(msg.contains("65"), "{msg}");

    // 次元が embedding_length_per_layer_input と食い違う
    let dir = TempDir::new("dim_ng");
    let path = dir.path("m.gguf");
    let mut kvs = small_kvs();
    set_kv(
        &mut kvs,
        "qwen4exp.embedding_length_per_layer_input",
        Kv::U32(128),
    );
    let data = synth_rows(SMALL_ROWS, 0x1234);
    write_gguf(
        &path,
        &kvs,
        &[TensorSpec {
            name: PLE_TABLE_NAME,
            dims: vec![N_DIM as u64, SMALL_ROWS as u64],
            ty: TY_IQ4_NL,
            data,
        }],
    );
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("次元不一致はエラー");
    assert!(e.to_string().contains("次元"), "{e}");
}

/// `offset + vocab` が i32 に収まらない場合は弾く (表テンソルが無くても弾く)。
#[test]
fn from_meta_rejects_head_range_over_i32() {
    let dir = TempDir::new("i32");
    let path = dir.path("m.gguf");
    let mut kvs = real_kvs();
    let mut off = REAL_OFF.to_vec();
    off[0] = i32::MAX as u64; // + vocab で i32 を超える
    set_kv(&mut kvs, "qwen4exp.ple.head_offsets", Kv::ArrU64(off));
    write_gguf(&path, &kvs, &[]);
    let f = GgufFile::open(&path).expect("open");
    let e = PleNgram::from_meta(&f).expect_err("i32 超過はエラー");
    assert!(e.to_string().contains("i32"), "{e}");
}

// ---- 5. gather ---------------------------------------------------------------

/// gather が `n_heads * n_dim` を返し、各ヘッドが手計算の行を読んでいること。
///
/// 小さいメタデータ (mult = [1,2,3], offset = 4h, vocab = 4) と ctx = [7,8,9]:
/// ```text
/// mixed(2) = 7*1 ^ 8*2   = 7 ^ 16 = 23 → 23 % 4 = 3 → 行 3, 7, 11, ..., 31
/// mixed(3) = 23 ^ 9*3    = 23 ^ 27 = 12 → 12 % 4 = 0 → 行 32, 36, ..., 60
/// ```
#[test]
fn gather_reads_hand_computed_rows() {
    let (_d, f, data) = open_small("gather", 0x0BAD_F00D);
    let ng = PleNgram::from_meta(&f).expect("from_meta");
    let tref = f.tensor_ref(PLE_TABLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &f).expect("PleTable::open");

    let ctx = [7u32, 8, 9];
    let idx = ng.indices(&ctx).expect("indices");
    assert_eq!(
        idx,
        vec![3, 7, 11, 15, 19, 23, 27, 31, 32, 36, 40, 44, 48, 52, 56, 60],
        "手計算の行番号"
    );

    let out = ng.gather(&ple, &ctx).expect("gather");
    assert_eq!(out.len(), 16 * N_DIM);

    for (h, &r) in idx.iter().enumerate() {
        let want = dequant_row(
            GgmlType::IQ4Nl,
            &data[r as usize * ROW_BYTES..(r as usize + 1) * ROW_BYTES],
            N_DIM,
        );
        let got = &out[h * N_DIM..(h + 1) * N_DIM];
        for d in 0..N_DIM {
            assert_eq!(
                got[d].to_bits(),
                want[d].to_bits(),
                "ヘッド {h} (行 {r}) 次元 {d}"
            );
        }
    }
}

/// 実メタデータ (表 3.2 億行) の索引は 64 行の表の範囲外 → gather はエラー。
#[test]
fn gather_rejects_out_of_range_rows() {
    let (_d1, f_meta) = open_real("gather_oor_meta");
    let ng = PleNgram::from_meta(&f_meta).expect("from_meta");

    let (_d2, f_tab, _data) = open_small("gather_oor_tab", 0x5555);
    let tref = f_tab.tensor_ref(PLE_TABLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &f_tab).expect("PleTable::open");

    let e = ng.gather(&ple, &[100, 200, 300]).expect_err("範囲外はエラー");
    let msg = e.to_string();
    assert!(msg.contains("範囲外"), "{msg}");
}

#[test]
fn indices_reject_empty_ctx() {
    let (_d, f) = open_real("empty");
    let ng = PleNgram::from_meta(&f).expect("from_meta");
    let e = ng.indices(&[]).expect_err("空の ctx はエラー");
    assert!(e.to_string().contains("ctx"), "{e}");
}
