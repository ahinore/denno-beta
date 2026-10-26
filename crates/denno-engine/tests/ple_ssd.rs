//! P-MOE-C1: PLE 表 (`per_layer_token_embd.weight`, IQ4_NL) の SSD 常駐行 gather のテスト。
//!
//! 実モデルに依存しないよう、このファイル内で最小 GGUF を書いて検証する。
//! GPU / CUDA / 推論経路は一切通さない。
//!
//! 実ファイルに対する計測は `#[ignore]` の `ple_gather_timing_real_gguf`。
//! 実行方法はそのテストの doc コメントを参照。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::kquant::{dequant_row, GgmlType};
use denno_engine::ple::{PleTable, DEFAULT_MAX_READ};

const ALIGN: usize = 32;

// GGUF メタデータ型 id
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_U64: u32 = 10;

// ggml_type id
const TY_F32: u32 = 0;
const TY_IQ4_NL: u32 = 20;

const ROWS: usize = 64;
const DIM: usize = 160;
const ROW_BYTES: usize = (DIM / 32) * 18; // 5 ブロック × 18 B = 90

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// メタデータ 1 個分。
enum Kv {
    Str(&'static str),
    U32(u32),
    U64(u64),
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
        }
    }
}

/// 1 テンソル分の仕様 (データは生バイト列をそのまま書く)。
struct TensorSpec {
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
}

/// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列、テンソルは与えた順に並ぶ。
fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[TensorSpec]) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&((kvs.len() + 1) as u64).to_le_bytes());
    for (k, v) in kvs {
        push_str(&mut head, k);
        v.write(&mut head);
    }
    // general.alignment
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

/// 合成 IQ4_NL 表の生バイト列 (rows × 90 B)。スケールは妥当な f16 値にしておく。
fn synth_rows(rows: usize, seed: u32) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let scales: [u16; 4] = [0x3C00, 0x3800, 0x3400, 0x4000]; // 1.0, 0.5, 0.25, 2.0
    let mut out = Vec::with_capacity(rows * ROW_BYTES);
    for _ in 0..rows {
        for _ in 0..(DIM / 32) {
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

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("denno_ple_{tag}_{}_{n}", std::process::id()));
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

const PLE_NAME: &str = "per_layer_token_embd.weight";

/// 合成表を書いた GGUF と、そのテンソル参照を返す。
fn setup(tag: &str, rows: usize, seed: u32) -> (TempDir, PathBuf, Vec<u8>) {
    let dir = TempDir::new(tag);
    let path = dir.path("ple.gguf");
    let data = synth_rows(rows, seed);
    let kvs = [("general.architecture", Kv::Str("qwen35next"))];
    let tensors = [TensorSpec {
        name: PLE_NAME,
        dims: vec![DIM as u64, rows as u64],
        ty: TY_IQ4_NL,
        data: data.clone(),
    }];
    write_gguf(&path, &kvs, &tensors);
    (dir, path, data)
}

/// 1. 64 行 × 160 次元の IQ4_NL 表を一時ファイルに書き、`gather_rows` の結果が
///    `kquant::dequant_row` とビット一致すること。
#[test]
fn gather_matches_dequant_row_bitwise() {
    let (_dir, path, data) = setup("bitwise", ROWS, 0x1234_5678);
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");

    assert_eq!(ple.n_rows(), ROWS);
    assert_eq!(ple.n_dim(), DIM);
    assert_eq!(ple.row_bytes(), ROW_BYTES);
    assert_eq!(ple.ty(), GgmlType::IQ4Nl);
    assert_eq!(ple.nbytes(), (ROWS * ROW_BYTES) as u64);

    let rows: Vec<u32> = (0..ROWS as u32).collect();
    let mut out = vec![0f32; ROWS * DIM];
    ple.gather_rows(&rows, &mut out).expect("gather");

    for r in 0..ROWS {
        let want = dequant_row(
            GgmlType::IQ4Nl,
            &data[r * ROW_BYTES..(r + 1) * ROW_BYTES],
            DIM,
        );
        for d in 0..DIM {
            let got = out[r * DIM + d];
            assert_eq!(
                got.to_bits(),
                want[d].to_bits(),
                "行 {r} 次元 {d}: gather={got} dequant_row={}",
                want[d]
            );
        }
    }
}

/// 2. 行 0 / 最終行 / 重複 index を含む gather。出力順は入力順。
#[test]
fn gather_first_last_and_duplicate_rows() {
    let (_dir, path, data) = setup("edges", ROWS, 0x0bad_f00d);
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");

    let rows: Vec<u32> = vec![0, (ROWS - 1) as u32, 7, 7, 0, (ROWS - 1) as u32];
    let mut out = vec![0f32; rows.len() * DIM];
    ple.gather_rows(&rows, &mut out).expect("gather");

    for (i, &r) in rows.iter().enumerate() {
        let want = dequant_row(
            GgmlType::IQ4Nl,
            &data[r as usize * ROW_BYTES..(r as usize + 1) * ROW_BYTES],
            DIM,
        );
        let got = &out[i * DIM..(i + 1) * DIM];
        for d in 0..DIM {
            assert_eq!(got[d].to_bits(), want[d].to_bits(), "位置 {i} (行 {r}) 次元 {d}");
        }
    }
    // 重複 index は同じ値になる
    assert_eq!(&out[0..DIM], &out[4 * DIM..5 * DIM]);
    assert_eq!(&out[DIM..2 * DIM], &out[5 * DIM..6 * DIM]);
}

/// 2b. 連続 8 行のまとめ読みと、上限を縮めた分割読みが同じ結果になること
///     (まとめ読みの分割分岐が実際に通ることも同時に確かめる)。
#[test]
fn contiguous_gather_equals_split_gather() {
    let (_dir, path, _data) = setup("contig", ROWS, 0x5eed_1234);
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");

    let rows: Vec<u32> = (10..18).collect();
    let mut merged = vec![0f32; rows.len() * DIM];
    ple.gather_rows(&rows, &mut merged).expect("gather (merged)");

    // 2 行ずつしかまとめない上限 (= 4 回に分割される)
    let mut split = PleTable::open(&tref, &gguf).expect("open 2");
    split.set_max_read(ROW_BYTES * 2).expect("set_max_read");
    assert_eq!(split.max_read(), ROW_BYTES * 2);
    let mut split_out = vec![0f32; rows.len() * DIM];
    split
        .gather_rows(&rows, &mut split_out)
        .expect("gather (split)");

    for i in 0..rows.len() * DIM {
        assert_eq!(merged[i].to_bits(), split_out[i].to_bits(), "要素 {i}");

    }
    // 上限は 64 KiB を超えられない
    let mut p = PleTable::open(&tref, &gguf).expect("open 3");
    assert!(p.set_max_read(DEFAULT_MAX_READ + 1).is_err());
    assert!(p.set_max_read(0).is_err());
}

/// 3. 範囲外 index はエラー (黙って 0 にしない)。
#[test]
fn out_of_range_index_is_error() {
    let (_dir, path, _data) = setup("oor", ROWS, 0x7777_1111);
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");

    for bad in [ROWS as u32, ROWS as u32 + 1, u32::MAX] {
        let mut out = vec![0f32; DIM];
        let e = ple.gather_rows(&[bad], &mut out).expect_err("範囲外はエラー");
        let msg = e.to_string();
        assert!(msg.contains("範囲外"), "{msg}");
        assert!(msg.contains(&bad.to_string()), "{msg}");
        // 書き込みも起きていない (先頭は 0 のまま)
        assert_eq!(out[0].to_bits(), 0f32.to_bits());
    }

    // out の長さ不一致もエラー
    let mut short = vec![0f32; DIM - 1];
    assert!(ple.gather_rows(&[0], &mut short).is_err());
    let mut long = vec![0f32; 2 * DIM];
    assert!(ple.gather_rows(&[0], &mut long).is_err());
}

/// 4. `n_rows` / `n_dim` がメタデータと一致しないテンソル参照でエラー。
#[test]
fn inconsistent_tensor_ref_is_error() {
    let (_dir, path, _data) = setup("meta", ROWS, 0x2468_ace0);
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");

    // 行数だけ 1 少ない参照
    let mut bad_rows = tref.clone();
    bad_rows.info.dims = [DIM as u64, (ROWS - 1) as u64, 1, 1];
    let e = PleTable::open(&bad_rows, &gguf).expect_err("行数不一致はエラー");
    assert!(e.to_string().contains("一致しない"), "{e}");

    // 行長だけ違う参照
    let mut bad_dim = tref.clone();
    bad_dim.info.dims = [128, ROWS as u64, 1, 1];
    assert!(PleTable::open(&bad_dim, &gguf).is_err());

    // バイト数だけ違う参照
    let mut bad_bytes = tref.clone();
    bad_bytes.info.nbytes += 18;
    assert!(PleTable::open(&bad_bytes, &gguf).is_err());

    // 型だけ違う参照 (IQ4_NL 以外は未対応)
    let mut bad_ty = tref.clone();
    bad_ty.info.ty = GgmlType::Q4_0;
    let e = PleTable::open(&bad_ty, &gguf).expect_err("型不一致はエラー");
    assert!(e.to_string().contains("未対応の量子型"), "{e}");

    // オフセットだけ違う参照 (別テンソルを指している)
    let mut bad_off = tref.clone();
    bad_off.info.offset += 32;
    assert!(PleTable::open(&bad_off, &gguf).is_err());

    // パート番号だけ違う参照
    let mut bad_part = tref.clone();
    bad_part.file = 1;
    bad_part.info.file = 1;
    assert!(PleTable::open(&bad_part, &gguf).is_err());

    // 存在しない名前
    let mut bad_name = tref.clone();
    bad_name.info.name = "nope.weight".to_string();
    assert!(PleTable::open(&bad_name, &gguf).is_err());

    // 正しい参照は通る
    assert!(PleTable::open(&tref, &gguf).is_ok());
}

/// 5. split GGUF のパート 1 に置かれた表も、パート番号から fd を開いて読めること
///    (`GgufFile::path_at(file)` の `file != 0` 分岐を通す)。
#[test]
fn gather_from_nonzero_split_part() {
    let dir = TempDir::new("split");
    let shard0 = dir.path("model-00001-of-00002.gguf");
    let shard1 = dir.path("model-00002-of-00002.gguf");
    let data = synth_rows(ROWS, 0xabcd_1234);

    // パート 0 にはダミーの F32 テンソルだけを置く。
    write_gguf(
        &shard0,
        &[
            ("general.architecture", Kv::Str("qwen35next")),
            ("split.no", Kv::U32(0)),
            ("split.count", Kv::U32(2)),
            ("split.tensors.count", Kv::U64(1)),
        ],
        &[TensorSpec {
            name: "blk.0.attn_norm.weight",
            dims: vec![4],
            ty: TY_F32,
            data: vec![0u8; 16],
        }],
    );
    // パート 1 に PLE 表。
    write_gguf(
        &shard1,
        &[
            ("split.no", Kv::U32(1)),
            ("split.count", Kv::U32(2)),
            ("split.tensors.count", Kv::U64(1)),
        ],
        &[TensorSpec {
            name: PLE_NAME,
            dims: vec![DIM as u64, ROWS as u64],
            ty: TY_IQ4_NL,
            data: data.clone(),
        }],
    );

    // わざと逆順で渡す (split.no で並べ直される)
    let gguf = GgufFile::open_split(&[shard1.clone(), shard0.clone()]).expect("open_split");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    assert_eq!(tref.file, 1, "PLE 表はパート 1");
    assert_eq!(gguf.path_at(1).expect("path"), shard1.as_path());

    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");
    assert_eq!(ple.path(), shard1.as_path());
    let rows: Vec<u32> = vec![3, 4, 5, (ROWS - 1) as u32];
    let mut out = vec![0f32; rows.len() * DIM];
    ple.gather_rows(&rows, &mut out).expect("gather");
    for (i, &r) in rows.iter().enumerate() {
        let want = dequant_row(
            GgmlType::IQ4Nl,
            &data[r as usize * ROW_BYTES..(r as usize + 1) * ROW_BYTES],
            DIM,
        );
        for d in 0..DIM {
            assert_eq!(
                out[i * DIM + d].to_bits(),
                want[d].to_bits(),
                "位置 {i} (行 {r}) 次元 {d}"
            );
        }
    }
}

/// 実ファイルに対する計測 (未実施の環境では何もせず return)。
///
/// 実行:
///   PLE_GGUF=/home/iegamo/models/qwen38-flash-next/<表を含むパート>.gguf \
///   cargo test --release -p denno-engine --test ple_ssd -- --ignored --nocapture \
///     ple_gather_timing_real_gguf
///
/// ランダム 8 行 × 1,000 回と連続 8 行 × 1,000 回の平均 µs を出す。
/// 1 回目はページキャッシュが冷えているので、warm 5 回の後 1,000 回を測る。
///
/// 実ファイルが無い環境では `PLE_SYNTH_ROWS=<行数>` を渡すと合成 IQ4_NL 表で
/// 同じ計測を行う。ただし合成表は全行がページキャッシュに載るため、その数字は
/// **SSD 読みの実測ではなく mmap + デコードの下限**として扱うこと。
#[test]
#[ignore]
fn ple_gather_timing_real_gguf() {
    let mut _keep: Option<TempDir> = None;
    let path = match std::env::var("PLE_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => match std::env::var("PLE_SYNTH_ROWS") {
            Ok(n) => {
                let rows: usize = n.parse().expect("PLE_SYNTH_ROWS は数値");
                let dir = TempDir::new("bench");
                let path = dir.path("ple_synth.gguf");
                let data = synth_rows(rows, 0x1234_abcd);
                write_gguf(
                    &path,
                    &[("general.architecture", Kv::Str("qwen35next"))],
                    &[TensorSpec {
                        name: PLE_NAME,
                        dims: vec![DIM as u64, rows as u64],
                        ty: TY_IQ4_NL,
                        data,
                    }],
                );
                eprintln!(
                    "注意: 合成表 ({rows} 行, {} MB) を使用。全行がページキャッシュに載るため \
                     この数字は SSD 読みの実測ではない",
                    rows * ROW_BYTES / (1024 * 1024)
                );
                _keep = Some(dir);
                path
            }
            Err(_) => {
                eprintln!(
                    "skip: PLE_GGUF も PLE_SYNTH_ROWS も未設定。/home/iegamo/models/\
                     qwen38-flash-next/ はこのコンテナから見えないため実測は未実施 \
                     (REPORT_MOE_C1.md 参照)"
                );
                return;
            }
        },
    };
    if !path.exists() {
        eprintln!("skip: {} が無い", path.display());
        return;
    }
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tref = gguf.tensor_ref(PLE_NAME).expect("tensor_ref");
    let ple = PleTable::open(&tref, &gguf).expect("PleTable::open");
    let n_rows = ple.n_rows();
    let n_dim = ple.n_dim();
    eprintln!("表: {n_rows} 行 × {n_dim} 次元 ({} B/行)", ple.row_bytes());

    let mut rng = Rng::new(0xdead_beef);
    let mut out = vec![0f32; 8 * n_dim];
    let run = |rows: &[u32], out: &mut Vec<f32>, iters: usize| -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            ple.gather_rows(rows, out).expect("gather");
        }
        let ns = t.elapsed().as_nanos() as f64;
        std::hint::black_box(&out[0]);
        ns / iters as f64 / 1000.0
    };

    // warm-up
    let rand_rows: Vec<u32> = (0..8).map(|_| rng.next_u32() % n_rows as u32).collect();
    let _ = run(&rand_rows, &mut out, 5);

    // RSS (VmRSS) の増分: 8 行 gather が本当にページ単位だけしか載せないことの確認。
    let rss_kb = || -> u64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        s.lines()
            .find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let rss_before = rss_kb();

    let t = std::time::Instant::now();
    for _ in 0..1000 {
        let rows: Vec<u32> = (0..8).map(|_| rng.next_u32() % n_rows as u32).collect();
        ple.gather_rows(&rows, &mut out).expect("gather");
    }
    let random_us = t.elapsed().as_nanos() as f64 / 1000.0 / 1000.0;
    let rss_after = rss_kb();

    let start = (rng.next_u32() as usize) % (n_rows - 8);
    let contig: Vec<u32> = (0..8).map(|i| (start + i) as u32).collect();
    let _ = run(&contig, &mut out, 5);
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        ple.gather_rows(&contig, &mut out).expect("gather");
    }
    let contig_us = t.elapsed().as_nanos() as f64 / 1000.0 / 1000.0;

    eprintln!("ランダム 8 行 × 1000 回: 平均 {random_us:.3} µs/gather");
    eprintln!("連続   8 行 × 1000 回: 平均 {contig_us:.3} µs/gather");
    eprintln!(
        "RSS (ランダム 8000 行の前後): {rss_before} kB -> {rss_after} kB (+{} kB)",
        rss_after as i64 - rss_before as i64
    );
}
