//! P-MOE-A1: GGUF の MoE レイアウト読解と split GGUF 結合のテスト。
//!
//! 実モデルに依存しないよう、このファイル内で **最小 GGUF** (metadata +
//! ダミーテンソル、量子型は Q4_0 か F32) を書き出して検証する。
//! GPU / CUDA / エンジン推論経路は一切通さない。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;

const ALIGN: usize = 32;

// GGUF メタデータ型 id (gguf.h enum gguf_type)
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_U64: u32 = 10;

// ggml_type id
const TY_F32: u32 = 0;
const TY_Q4_0: u32 = 2;

enum Kv {
    U32(u32),
    U64(u64),
    Str(&'static str),
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::U64(v) => {
                b.extend_from_slice(&GT_U64.to_le_bytes());
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
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    fill: u8,
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn blck_size(ty: u32) -> u64 {
    match ty {
        TY_F32 => 1,
        TY_Q4_0 => 32,
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
}

fn type_size(ty: u32) -> u64 {
    match ty {
        TY_F32 => 4,
        TY_Q4_0 => 18,
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
fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[TensorSpec]) {
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
        push_str(&mut head, t.name);
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
            "denno_moe_{tag}_{}_{n}",
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

/// MoE 1 モデル分のメタデータ (層 2・エキスパート 4・shared 1)。
fn moe_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("general.alignment", Kv::U32(32)),
        ("qwen35.block_count", Kv::U32(2)),
        ("qwen35.embedding_length", Kv::U32(64)),
        ("qwen35.expert_count", Kv::U32(4)),
        ("qwen35.expert_used_count", Kv::U32(2)),
        ("qwen35.expert_feed_forward_length", Kv::U32(32)),
        ("qwen35.expert_shared_count", Kv::U32(1)),
        ("qwen35.expert_shared_feed_forward_length", Kv::U32(32)),
    ]
}

const N_EMBD: u64 = 64;
const N_FFN: u64 = 32;
const N_EXPERT: u64 = 4;
const N_SHARED_FFN: u64 = 32;

/// 1 層分の MoE テンソル (llama.cpp の正規名・ne 順)。
fn moe_layer_tensors(layer: usize, fill: u8) -> Vec<TensorSpec> {
    let p = |s: &'static str| -> &'static str { Box::leak(format!("blk.{layer}.{s}").into_boxed_str()) };
    vec![
        TensorSpec { name: p("ffn_gate_exps.weight"), dims: vec![N_EMBD, N_FFN, N_EXPERT], ty: TY_Q4_0, fill },
        TensorSpec { name: p("ffn_up_exps.weight"), dims: vec![N_EMBD, N_FFN, N_EXPERT], ty: TY_Q4_0, fill },
        TensorSpec { name: p("ffn_down_exps.weight"), dims: vec![N_FFN, N_EMBD, N_EXPERT], ty: TY_Q4_0, fill },
        TensorSpec { name: p("ffn_gate_inp.weight"), dims: vec![N_EMBD, N_EXPERT], ty: TY_F32, fill },
        TensorSpec { name: p("ffn_gate_inp.bias"), dims: vec![N_EXPERT], ty: TY_F32, fill },
        TensorSpec { name: p("ffn_gate_shexp.weight"), dims: vec![N_EMBD, N_SHARED_FFN], ty: TY_Q4_0, fill },
        TensorSpec { name: p("ffn_up_shexp.weight"), dims: vec![N_EMBD, N_SHARED_FFN], ty: TY_Q4_0, fill },
        TensorSpec { name: p("ffn_down_shexp.weight"), dims: vec![N_SHARED_FFN, N_EMBD], ty: TY_Q4_0, fill },
    ]
}

/// 1. MoE メタデータを読める + 2. `ffn_*_exps` の形状が表どおり。
#[test]
fn moe_metadata_and_expert_shapes() {
    let dir = TempDir::new("meta");
    let path = dir.path("moe.gguf");
    let mut tensors = moe_layer_tensors(0, 0x11);
    tensors.extend(moe_layer_tensors(1, 0x22));
    write_gguf(&path, &moe_kvs(), &tensors);

    let f = GgufFile::open(&path).expect("open");
    let l = f.moe_layout().expect("MoE として読めるはず");

    assert_eq!(l.n_expert, 4);
    assert_eq!(l.n_expert_used, 2);
    assert_eq!(l.n_ffn, 32);
    assert_eq!(l.n_shared, 1);
    assert_eq!(l.n_shared_ffn, 32);
    assert_eq!(l.per_layer.len(), 2);

    let t0 = l.per_layer[0].as_ref().expect("層 0 はエキスパートを持つ");
    assert_eq!(t0.gate_exps.ne(), vec![64, 32, 4], "ffn_gate_exps = [n_embd, n_ffn, n_expert]");
    assert_eq!(t0.up_exps.ne(), vec![64, 32, 4], "ffn_up_exps = [n_embd, n_ffn, n_expert]");
    assert_eq!(t0.down_exps.ne(), vec![32, 64, 4], "ffn_down_exps = [n_ffn, n_embd, n_expert]");
    assert_eq!(t0.gate_inp.ne(), vec![64, 4], "ffn_gate_inp = [n_embd, n_expert]");
    assert_eq!(t0.gate_inp_bias.as_ref().map(|t| t.ne()), Some(vec![4]));
    assert_eq!(t0.shexp_gate.as_ref().map(|t| t.ne()), Some(vec![64, 32]));
    assert_eq!(t0.shexp_up.as_ref().map(|t| t.ne()), Some(vec![64, 32]));
    assert_eq!(t0.shexp_down.as_ref().map(|t| t.ne()), Some(vec![32, 64]));

    // 名前・型・要素数・バイト数も引けている
    assert_eq!(t0.gate_exps.name(), "blk.0.ffn_gate_exps.weight");
    assert_eq!(t0.gate_exps.ty(), denno_engine::kquant::GgmlType::Q4_0);
    assert_eq!(t0.gate_exps.n_elements(), 64 * 32 * 4);
    assert_eq!(t0.gate_exps.nbytes(), (64 * 32 * 4) / 32 * 18);
    assert_eq!(t0.gate_inp.ty(), denno_engine::kquant::GgmlType::F32);
    assert_eq!(t0.gate_inp.nbytes(), 64 * 4 * 4);

    // 層 1 も別テンソルを指している (取り違えていない)
    let t1 = l.per_layer[1].as_ref().expect("層 1");
    assert_eq!(t1.gate_exps.name(), "blk.1.ffn_gate_exps.weight");
    assert_ne!(t0.gate_exps.offset(), t1.gate_exps.offset());
}

/// 3. dense のメタデータでは `moe_layout() == None`。
#[test]
fn dense_metadata_has_no_moe_layout() {
    let dir = TempDir::new("dense");
    let path = dir.path("dense.gguf");
    let kvs = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("general.alignment", Kv::U32(32)),
        ("qwen35.block_count", Kv::U32(1)),
        ("qwen35.embedding_length", Kv::U32(64)),
        ("qwen35.feed_forward_length", Kv::U32(32)),
    ];
    let tensors = vec![
        TensorSpec { name: "blk.0.ffn_gate.weight", dims: vec![64, 32], ty: TY_Q4_0, fill: 0x33 },
        TensorSpec { name: "blk.0.ffn_up.weight", dims: vec![64, 32], ty: TY_Q4_0, fill: 0x33 },
        TensorSpec { name: "blk.0.ffn_down.weight", dims: vec![32, 64], ty: TY_Q4_0, fill: 0x33 },
    ];
    write_gguf(&path, &kvs, &tensors);

    let f = GgufFile::open(&path).expect("open");
    assert!(f.moe_layout().is_none(), "dense は None");
    // dense のテンソル参照は従来どおり
    assert!(f.tensor("blk.0.ffn_gate.weight").is_some());
}

/// 4. split 2 分割を結合でき、テンソル数・パート番号・オフセット・バイト列が正しい。
#[test]
fn split_two_shards_merge_into_one_table() {
    let dir = TempDir::new("split");
    let shard0 = dir.path("model-00001-of-00002.gguf");
    let shard1 = dir.path("model-00002-of-00002.gguf");

    let mut kv0 = moe_kvs();
    kv0.push(("split.no", Kv::U32(0)));
    kv0.push(("split.count", Kv::U32(2)));
    kv0.push(("split.tensors.count", Kv::U64(4)));
    let t0 = vec![
        TensorSpec { name: "blk.0.ffn_gate_exps.weight", dims: vec![N_EMBD, N_FFN, N_EXPERT], ty: TY_Q4_0, fill: 0xAA },
        TensorSpec { name: "blk.0.ffn_up_exps.weight", dims: vec![N_EMBD, N_FFN, N_EXPERT], ty: TY_Q4_0, fill: 0xAA },
    ];
    write_gguf(&shard0, &kv0, &t0);

    let kv1 = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("split.no", Kv::U32(1)),
        ("split.count", Kv::U32(2)),
        ("split.tensors.count", Kv::U64(4)),
    ];
    let t1 = vec![
        TensorSpec { name: "blk.0.ffn_down_exps.weight", dims: vec![N_FFN, N_EMBD, N_EXPERT], ty: TY_Q4_0, fill: 0xBB },
        TensorSpec { name: "blk.0.ffn_gate_inp.weight", dims: vec![N_EMBD, N_EXPERT], ty: TY_F32, fill: 0xBB },
    ];
    write_gguf(&shard1, &kv1, &t1);

    // わざと逆順で渡す (split.no で並べ直すことの確認)
    let f = GgufFile::open_split(&[shard1.clone(), shard0.clone()]).expect("split open");
    assert_eq!(f.part_count(), 2);
    assert_eq!(f.tensors.len(), 4);
    assert_eq!(
        f.tensor_order,
        vec![
            "blk.0.ffn_gate_exps.weight".to_string(),
            "blk.0.ffn_up_exps.weight".to_string(),
            "blk.0.ffn_down_exps.weight".to_string(),
            "blk.0.ffn_gate_inp.weight".to_string(),
        ],
        "split.no 昇順の連結"
    );

    // 各テンソルが正しいパートを指し、オフセットが書き出し時の値と一致する
    for (i, spec) in t0.iter().enumerate() {
        let info = f.tensor(spec.name).expect("tensor");
        assert_eq!(info.file, 0, "{} は shard 0", spec.name);
        let want_off: u64 = t0[..i]
            .iter()
            .map(|s| align_up(nbytes(&s.dims, s.ty) as usize, ALIGN) as u64)
            .sum();
        assert_eq!(info.offset, want_off, "{} のオフセット", spec.name);
        assert_eq!(info.nbytes, nbytes(&spec.dims, spec.ty));
    }
    for (i, spec) in t1.iter().enumerate() {
        let info = f.tensor(spec.name).expect("tensor");
        assert_eq!(info.file, 1, "{} は shard 1", spec.name);
        let want_off: u64 = t1[..i]
            .iter()
            .map(|s| align_up(nbytes(&s.dims, s.ty) as usize, ALIGN) as u64)
            .sum();
        assert_eq!(info.offset, want_off, "{} のオフセット", spec.name);
        assert_eq!(info.nbytes, nbytes(&spec.dims, spec.ty));
    }

    // データ部がパートを跨いで正しく読める (shard 1 のテンソルは 0xBB)
    let b = f.tensor_bytes("blk.0.ffn_down_exps.weight").expect("bytes");
    assert_eq!(b.len() as u64, nbytes(&t1[0].dims, t1[0].ty));
    assert!(b.iter().all(|&x| x == 0xBB), "shard 1 のテンソルは 0xBB");
    let b0 = f.tensor_bytes("blk.0.ffn_gate_exps.weight").expect("bytes");
    assert!(b0.iter().all(|&x| x == 0xAA), "shard 0 のテンソルは 0xAA");

    // パート 0 以外は tensor_mapping (単一ファイル用) では None、parts 版で引ける
    assert!(f.tensor_mapping("blk.0.ffn_down_exps.weight").is_none());
    let (part, off, len) = f.tensor_mapping_parts("blk.0.ffn_down_exps.weight").expect("parts");
    assert_eq!(part, 1);
    // 返ってきた絶対範囲がそのパートの mmap 上で同じバイト列を指す
    let h = f.mmap_handle_at(part).expect("part mmap");
    assert_eq!(
        &h[off..off + len],
        f.tensor_bytes("blk.0.ffn_down_exps.weight").unwrap()
    );
    assert_eq!(len as u64, nbytes(&t1[0].dims, t1[0].ty));
    assert!(f.mmap_handle_at(1).is_some());
    assert!(f.mmap_handle_at(2).is_none());

    // split でも moe_layout が 1 つの表として読める
    let l = f.moe_layout().expect("MoE");
    let t = l.per_layer[0].as_ref().expect("層 0");
    assert_eq!(t.gate_exps.file, 0);
    assert_eq!(t.down_exps.file, 1);
    assert_eq!(t.gate_inp.file, 1);
}

/// 5. `split.no` 欠番でエラー。
#[test]
fn split_missing_index_is_error() {
    let dir = TempDir::new("gap");
    let a = dir.path("a.gguf");
    let b = dir.path("b.gguf");

    let kvs = |no: u32| {
        vec![
            ("general.architecture", Kv::Str("qwen35")),
            ("split.no", Kv::U32(no)),
            ("split.count", Kv::U32(2)),
        ]
    };
    write_gguf(
        &a,
        &kvs(0),
        &[TensorSpec { name: "blk.0.ffn_gate_inp.weight", dims: vec![8, 2], ty: TY_F32, fill: 0x01 }],
    );
    // split.no = 2 が欠番を作る (0,1 の 2 分割なのに 2 が来る)
    write_gguf(
        &b,
        &kvs(2),
        &[TensorSpec { name: "blk.0.ffn_down_exps.weight", dims: vec![8, 8, 2], ty: TY_F32, fill: 0x02 }],
    );

    let e = match GgufFile::open_split(&[a, b]) {
        Ok(_) => panic!("欠番はエラーになるはず"),
        Err(e) => e,
    };
    assert!(e.0.contains("split.no"), "エラー内容: {e}");
}

/// おまけ: 実 dense 27B (Qwen3.8-27B Q4_0) で `moe_layout()` が None を返す。
/// `QWEN35_27B_DIR` 未設定なら skip。ヘッダだけ読む (dequant しない)。
#[test]
#[ignore]
fn real_dense_27b_has_no_moe_layout() {
    let Ok(dir) = std::env::var("QWEN35_27B_DIR") else {
        eprintln!("skip: QWEN35_27B_DIR 未設定");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut cands: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("dir")
        .filter_map(|en| {
            let p = en.ok()?.path();
            (p.extension().and_then(|s| s.to_str()) == Some("gguf")).then_some(p)
        })
        .collect();
    assert!(!cands.is_empty(), "GGUF が見つからない: {dir:?}");
    cands.sort();
    let gguf = cands
        .iter()
        .find(|p| p.to_string_lossy().contains("Q4_0"))
        .unwrap_or(&cands[0])
        .clone();

    let f = GgufFile::open(&gguf).expect("open");
    eprintln!("dense GGUF: {gguf:?} ({} tensors)", f.tensors.len());
    assert!(
        f.moe_layout().is_none(),
        "dense 27B は moe_layout() == None でなければならない"
    );
    // expert 系メタデータが本当に無いこと (None の理由を明示)
    let arch = f.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("?");
    assert!(
        f.get(&format!("{arch}.expert_count")).is_none(),
        "{arch}.expert_count が存在する"
    );
}

#[test]
fn open_split_single_path_is_single_file() {
    let dir = TempDir::new("single");
    let path = dir.path("moe.gguf");
    write_gguf(&path, &moe_kvs(), &moe_layer_tensors(0, 0x44));
    let f = GgufFile::open_split(&[path.clone()]).expect("open_split");
    assert_eq!(f.part_count(), 1);
    assert!(f.moe_layout().is_some());
    let g = GgufFile::open(&path).expect("open");
    assert_eq!(g.tensors.len(), f.tensors.len());
}

/// 5. `split.tensors.count` は **モデル全体** のテンソル数で、全パートに同じ値が入る。
///    パート間で食い違ったら開けない (実モデル 1,224 本 / 4 分割で踏んだ穴の再発防止)。
#[test]
fn split_tensors_count_must_agree_across_parts() {
    let dir = TempDir::new("splitcnt");
    let shard0 = dir.path("model-00001-of-00002.gguf");
    let shard1 = dir.path("model-00002-of-00002.gguf");

    let mut kv0 = moe_kvs();
    kv0.push(("split.no", Kv::U32(0)));
    kv0.push(("split.count", Kv::U32(2)));
    kv0.push(("split.tensors.count", Kv::U64(2)));
    let t0 = vec![TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![N_EMBD, N_FFN, N_EXPERT],
        ty: TY_Q4_0,
        fill: 0xAA,
    }];
    write_gguf(&shard0, &kv0, &t0);

    let kv1 = vec![
        ("general.architecture", Kv::Str("qwen35")),
        ("split.no", Kv::U32(1)),
        ("split.count", Kv::U32(2)),
        // ここだけ違う値 (総和 2+1=3 でも実際 2 本なので、旧実装は別の理由で落ちていた)
        ("split.tensors.count", Kv::U64(1)),
    ];
    let t1 = vec![TensorSpec {
        name: "blk.0.ffn_up_exps.weight",
        dims: vec![N_EMBD, N_FFN, N_EXPERT],
        ty: TY_Q4_0,
        fill: 0xBB,
    }];
    write_gguf(&shard1, &kv1, &t1);

    let e = match GgufFile::open_split(&[shard0, shard1]) {
        Ok(_) => panic!("食い違いはエラーになるはず"),
        Err(e) => e,
    };
    let msg = format!("{e:?}");
    assert!(msg.contains("食い違う"), "予期したエラーでない: {msg}");
}
