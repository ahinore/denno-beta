//! Task 1404: GGUF v2/v3 の壊れたヘッダ・桁溢れ・split シャードの回帰テスト。
//!
//! `GgufFile::open` / `GgufFile::open_split` を **実ファイル** に対して呼び、
//! 以下の入力がパニックせず `Err` になることを確かめる (P912 の単体テストを
//! 公開 API 経由で補完する):
//!
//! - magic / version 不正
//! - 途中で切れたメタデータ文字列・配列 (宣言長がバッファを超える)
//! - `kv_count` / `tensor_count` / 配列要素数の巨大宣言 (過剰な確保をしない)
//! - 次元の積・`nbytes`・テンソル offset の桁溢れ
//! - 未知の GGML 型タグ・未知のメタデータ型・`n_dims > 4`
//! - split シャードの欠番 / 重複 / 件数不一致 / パート跨ぎのテンソル名重複
//! - 壊れたシャードを含む split、offset 桁溢れを含む split
//!
//! すべて CPU のみ・1 秒未満で終わる。デバッグ出力は出さない。

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use denno_engine::gguf::GgufFile;

const GT_U8: u32 = 0;
const GT_STRING: u32 = 8;
const GT_U32: u32 = 4;
const GT_U64: u32 = 10;
const GT_ARRAY: u32 = 9;

// ===== 割り当て量の監視 =====
//
// 「壊れた宣言数で巨大な確保をしない」ことを実際のバイト数で確かめるため、
// このテストバイナリ内だけの集計アロケータを入れる。全テストは `serial()` で
// 直列化するので、他スレッドの割り当てが計測に混ざらない。

struct CountingAlloc;

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn note_alloc(bytes: usize) {
    let now = CUR.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = std::alloc::System.alloc(layout);
        if !p.is_null() {
            note_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        CUR.fetch_sub(layout.size(), Ordering::SeqCst);
        std::alloc::System.dealloc(ptr, layout)
    }

    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        old: std::alloc::Layout,
        new_size: usize,
    ) -> *mut u8 {
        let p = std::alloc::System.realloc(ptr, old, new_size);
        if !p.is_null() {
            CUR.fetch_sub(old.size(), Ordering::SeqCst);
            note_alloc(new_size);
        }
        p
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// 計測区間の最大確保バイト数 (テスト開始時点からの増分) を返す。
fn measure_peak_growth(f: impl FnOnce()) -> usize {
    let base = CUR.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    f();
    PEAK.swap(base, Ordering::SeqCst).saturating_sub(base)
}

/// テスト同士を直列化する (env と確保量の計測がプロセス全体で共有のため)。
fn serial() -> MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

/// テスト中だけ env を立てるガード。drop で元に戻す。
struct EnvGuard {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var_os(key);
        std::env::set_var(key, val);
        EnvGuard { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

// ===== 一時ディレクトリ =====

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::AtomicU32;
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_1404_{tag}_{}_{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp dir");
        TempDir(p)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).expect("write gguf");
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ===== 合成 GGUF =====

fn u32b(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn u64b(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn strb(v: &mut Vec<u8>, s: &str) {
    u64b(v, s.len() as u64);
    v.extend_from_slice(s.as_bytes());
}

#[derive(Clone)]
enum Val<'a> {
    Str(&'a str),
    /// GT_STRING を宣言するが中身を書かない (宣言長だけで切る)。
    StrLen(u64),
    U32(u32),
    U64(u64),
    /// GT_ARRAY (要素型 u32)。`count` は `data` の実長と一致しなくてよい。
    ArrU8 { count: u64, data: &'a [u8] },
    /// GT_ARRAY (要素型は任意 id)。
    ArrRawTy { elem_ty: u32, count: u64 },
}

impl Val<'_> {
    fn write(&self, v: &mut Vec<u8>) {
        match *self {
            Val::Str(s) => {
                u32b(v, GT_STRING);
                strb(v, s);
            }
            Val::StrLen(n) => {
                u32b(v, GT_STRING);
                u64b(v, n);
            }
            Val::U32(x) => {
                u32b(v, GT_U32);
                u32b(v, x);
            }
            Val::U64(x) => {
                u32b(v, GT_U64);
                u64b(v, x);
            }
            Val::ArrU8 { count, data } => {
                u32b(v, GT_ARRAY);
                u32b(v, GT_U8);
                u64b(v, count);
                v.extend_from_slice(data);
            }
            Val::ArrRawTy { elem_ty, count } => {
                u32b(v, GT_ARRAY);
                u32b(v, elem_ty);
                u64b(v, count);
            }
        }
    }
}

struct TensorSpec<'a> {
    name: &'a str,
    dims: &'a [u64],
    ty: u32,
    offset: u64,
}

const DEFAULT_ALIGN: usize = 32;

/// 合成 GGUF をバイト列で返す。ヘッダの後に `data_len` バイトのデータ部を付ける。
fn synth(
    version: u32,
    alignment: Option<u64>,
    kvs: &[(&str, Val)],
    tensors: &[TensorSpec],
    data_len: usize,
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    u32b(&mut b, version);
    u64b(&mut b, tensors.len() as u64);
    u64b(&mut b, (kvs.len() + usize::from(alignment.is_some())) as u64);
    for (k, val) in kvs {
        strb(&mut b, k);
        val.write(&mut b);
    }
    if let Some(a) = alignment {
        strb(&mut b, "general.alignment");
        Val::U64(a).write(&mut b);
    }
    for t in tensors {
        strb(&mut b, t.name);
        u32b(&mut b, t.dims.len() as u32);
        for d in t.dims {
            u64b(&mut b, *d);
        }
        u32b(&mut b, t.ty);
        u64b(&mut b, t.offset);
    }
    let align = match alignment {
        Some(a) if a.is_power_of_two() && a <= 65536 => a as usize,
        _ => DEFAULT_ALIGN,
    };
    while b.len() % align != 0 {
        b.push(0);
    }
    let n = b.len();
    b.resize(n + data_len, 0);
    b
}

fn q4_bytes(k: u64, n: u64) -> u64 {
    (k / 32) * 18 * n
}

// ===== 単一ファイル: ヘッダ異常 =====

#[test]
fn bad_magic_rejected() {
    let _g = serial();
    let dir = TempDir::new("magic");
    let mut b = synth(3, None, &[], &[], 0);
    b[0] = b'X';
    let p = dir.write("bad_magic.gguf", &b);
    let e = GgufFile::open(&p).err().expect("bad magic が通った");
    assert!(e.0.contains("bad magic"), "予期しないエラー: {}", e.0);

    // 全 4 バイトが別物でも同じく Err
    let mut b2 = synth(3, None, &[], &[], 0);
    b2[..4].copy_from_slice(b"GGML");
    let p2 = dir.write("bad_magic2.gguf", &b2);
    assert!(GgufFile::open(&p2).is_err());
}

#[test]
fn version_gate_accepts_v2_v3_and_rejects_others() {
    let _g = serial();
    let dir = TempDir::new("version");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    for bad in [0u32, 1, 4, 5, u32::MAX] {
        let b = synth(
            bad,
            None,
            &[],
            &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
            nb,
        );
        let p = dir.write(&format!("v{bad}.gguf"), &b);
        let e = GgufFile::open(&p)
            .err()
            .unwrap_or_else(|| panic!("v{bad} が通った"));
        assert!(e.0.contains("未対応"), "v{bad} のエラー: {}", e.0);
    }
    for (i, v) in [2u32, 3].into_iter().enumerate() {
        let b = synth(
            v,
            None,
            &[],
            &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
            nb,
        );
        let p = dir.write(&format!("ok_v{v}.gguf"), &b);
        let f = GgufFile::open(&p).unwrap_or_else(|e| panic!("v{v}: {}", e.0));
        assert_eq!(f.version, v);
        assert_eq!(f.part_count(), 1);
        assert_eq!(f.tensor_bytes("t.weight").expect("bytes").len(), nb, "v{v} #{}", i);
        f.validate_layout().expect("validate_layout");
    }
}

#[test]
fn truncated_metadata_strings_are_eof_errors() {
    let _g = serial();
    let dir = TempDir::new("str");
    for (tag, len) in [("max", u64::MAX), ("huge", 1u64 << 40)] {
        let b = synth(
            3,
            None,
            &[("k", Val::StrLen(len))],
            &[],
            0,
        );
        let p = dir.write(&format!("str_{tag}.gguf"), &b);
        let e = GgufFile::open(&p).err().expect("巨大な文字列長が通った");
        assert!(e.0.starts_with("EOF"), "len={len} のエラー: {}", e.0);
    }

    // キー自体の宣言長が桁溢れ
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    u32b(&mut b, 3);
    u64b(&mut b, 0);
    u64b(&mut b, 1);
    u64b(&mut b, u64::MAX);
    let p = dir.write("key_len.gguf", &b);
    let e = GgufFile::open(&p).err().expect("キー長 u64::MAX が通った");
    assert!(e.0.starts_with("EOF"), "予期しないエラー: {}", e.0);

    // テンソル名の宣言長が桁溢れ
    let mut b2 = Vec::new();
    b2.extend_from_slice(b"GGUF");
    u32b(&mut b2, 3);
    u64b(&mut b2, 1); // tensor_count
    u64b(&mut b2, 0); // kv_count
    u64b(&mut b2, u64::MAX); // name 長
    let p2 = dir.write("name_len.gguf", &b2);
    let e2 = GgufFile::open(&p2).err().expect("テンソル名長 u64::MAX が通った");
    assert!(e2.0.starts_with("EOF"), "予期しないエラー: {}", e2.0);
}

#[test]
fn truncated_metadata_arrays_are_eof_errors() {
    let _g = serial();
    let dir = TempDir::new("arr");
    // alignment 詰め (最大 31 B) を足しても宣言数に届かない値を選ぶ。
    let cases: [(&str, u64, &[u8]); 3] = [
        ("max", u64::MAX, &[]),
        ("partial", 100, &[1, 2, 3]),
        ("one_short", 64, &[1, 2, 3]),
    ];
    for (tag, count, data) in cases {
        let b = synth(
            3,
            None,
            &[("a", Val::ArrU8 { count, data })],
            &[],
            0,
        );
        let p = dir.write(&format!("arr_{tag}.gguf"), &b);
        let e = GgufFile::open(&p).err().expect("切れた配列が通った");
        assert!(e.0.starts_with("EOF"), "{tag} のエラー: {}", e.0);
    }

    // 未知の配列要素型 (1 要素以上あるとき要素読み取りで検出される)
    let b = synth(
        3,
        None,
        &[("a", Val::ArrRawTy { elem_ty: 99, count: 1 })],
        &[],
        0,
    );
    let p = dir.write("arr_ty.gguf", &b);
    let e = GgufFile::open(&p).err().expect("未知の配列要素型が通った");
    assert!(e.0.contains("配列要素型"), "予期しないエラー: {}", e.0);

    // 未知のメタデータ型
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    u32b(&mut raw, 3);
    u64b(&mut raw, 0);
    u64b(&mut raw, 1);
    strb(&mut raw, "k");
    u32b(&mut raw, 99);
    let p2 = dir.write("meta_ty.gguf", &raw);
    let e2 = GgufFile::open(&p2).err().expect("未知のメタデータ型が通った");
    assert!(e2.0.contains("メタデータ型"), "予期しないエラー: {}", e2.0);
}

#[test]
fn huge_declared_counts_error_without_excessive_allocation() {
    let _g = serial();
    let dir = TempDir::new("counts");

    // tensor_count / kv_count = u64::MAX。バッファは 24 B しかない。
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    u32b(&mut b, 3);
    u64b(&mut b, u64::MAX);
    u64b(&mut b, u64::MAX);
    let p = dir.write("counts.gguf", &b);
    let e = GgufFile::open(&p).err().expect("u64::MAX 件数が通った");
    assert!(e.0.starts_with("EOF"), "予期しないエラー: {}", e.0);

    // 宣言数だけ巨大な配列。確保はバッファ長で上限を切られているはず。
    let arr = synth(
        3,
        None,
        &[("a", Val::ArrU8 { count: u64::MAX, data: &[] })],
        &[],
        0,
    );
    assert!(arr.len() < 1024, "前置きが大きすぎる: {} B", arr.len());
    let p2 = dir.write("huge_arr.gguf", &arr);
    let mut err_seen = false;
    let growth = measure_peak_growth(|| {
        err_seen = GgufFile::open(&p2).is_err();
    });
    assert!(err_seen, "巨大な配列宣言が通った");
    assert!(
        growth < 1024 * 1024,
        "壊れた宣言で {growth} B 確保した (上限 1 MiB)"
    );
}

#[test]
fn n_dims_above_four_rejected() {
    let _g = serial();
    let dir = TempDir::new("ndims");
    let dims = [2u64; 5];
    let b = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
        0,
    );
    let p = dir.write("ndims5.gguf", &b);
    let e = GgufFile::open(&p).err().expect("n_dims=5 が通った");
    assert!(e.0.contains("n_dims"), "予期しないエラー: {}", e.0);
}

#[test]
fn unknown_ggml_type_tags_rejected() {
    let _g = serial();
    let dir = TempDir::new("tyid");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    for ty in [4u32, 5, 15, 24, 31, 999, u32::MAX] {
        let b = synth(
            3,
            None,
            &[],
            &[TensorSpec { name: "t.weight", dims: &dims, ty, offset: 0 }],
            nb,
        );
        let p = dir.write(&format!("ty{ty}.gguf"), &b);
        let e = GgufFile::open(&p)
            .err()
            .unwrap_or_else(|| panic!("型 id {ty} が通った"));
        assert!(e.0.contains("型 id"), "型 id {ty} のエラー: {}", e.0);
    }
}

#[test]
fn dimension_and_nbytes_overflow_rejected() {
    let _g = serial();
    let dir = TempDir::new("dimovf");

    // 次元の積が u64 を超える
    let dims = [u64::MAX, 4, 2, 2];
    let b = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "huge.weight", dims: &dims, ty: 2, offset: 0 }],
        0,
    );
    let p = dir.write("dim_overflow.gguf", &b);
    let e = GgufFile::open(&p).err().expect("次元の積の桁溢れが通った");
    assert!(e.0.contains("次元の積"), "予期しないエラー: {}", e.0);

    // 要素数は収まるが nbytes が溢れる (F32: 1 要素 4 B、2^63 要素)
    let dims2 = [1u64 << 63];
    let b2 = synth(
        2,
        None,
        &[],
        &[TensorSpec { name: "huge2.weight", dims: &dims2, ty: 0, offset: 0 }],
        0,
    );
    let p2 = dir.write("nbytes_overflow.gguf", &b2);
    let e2 = GgufFile::open(&p2).err().expect("nbytes の桁溢れが通った");
    assert!(e2.0.contains("バイト数"), "予期しないエラー: {}", e2.0);
}

#[test]
fn duplicate_tensor_names_in_one_file_rejected() {
    let _g = serial();
    let dir = TempDir::new("dupname");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let b = synth(
        3,
        None,
        &[],
        &[
            TensorSpec { name: "dup.weight", dims: &dims, ty: 2, offset: 0 },
            TensorSpec { name: "dup.weight", dims: &dims, ty: 2, offset: 32 },
        ],
        2 * nb,
    );
    let p = dir.write("dup.gguf", &b);
    let e = GgufFile::open(&p).err().expect("テンソル名重複が通った");
    assert!(e.0.contains("重複"), "予期しないエラー: {}", e.0);
}

#[test]
fn invalid_alignment_rejected_and_valid_alignment_accepted() {
    let _g = serial();
    let dir = TempDir::new("align");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    for bad in [0u64, 3, 24, 6, (1 << 25) + 1] {
        let b = synth(
            3,
            Some(bad),
            &[],
            &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
            nb,
        );
        let p = dir.write(&format!("align{bad}.gguf"), &b);
        let e = GgufFile::open(&p)
            .err()
            .unwrap_or_else(|| panic!("alignment={bad} が通った"));
        assert!(e.0.contains("alignment"), "alignment={bad} のエラー: {}", e.0);
    }
    for ok in [1u64, 2, 32, 64, 256] {
        let b = synth(
            3,
            Some(ok),
            &[],
            &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
            nb,
        );
        let p = dir.write(&format!("ok_align{ok}.gguf"), &b);
        let f = GgufFile::open(&p).unwrap_or_else(|e| panic!("alignment={ok}: {}", e.0));
        assert_eq!(f.alignment, ok);
        assert_eq!(f.data_start % ok, 0, "data_start が整列していない");
    }
}

// ===== 単一ファイル: テンソル範囲・offset の桁溢れ =====

#[test]
fn tensor_offset_overflow_surfaces_as_clean_error() {
    let _g = serial();
    let dir = TempDir::new("offovf");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let b = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: u64::MAX }],
        nb,
    );
    let p = dir.write("off_overflow.gguf", &b);

    // 既定 (環境ゲート off) では open は通り、アクセス時に Err になる。
    let f = GgufFile::open(&p).expect("open 自体はメタデータ検査のみ");
    let e = f.tensor_bytes("t.weight").err().expect("offset 桁溢れが読めた");
    assert!(e.0.contains("桁溢れ"), "予期しないエラー: {}", e.0);
    assert!(f.tensor_mapping_parts("t.weight").is_none());
    let e2 = f.validate_layout().err().expect("validate_layout が通った");
    assert!(e2.0.contains("桁溢れ"), "予期しないエラー: {}", e2.0);

    // 環境ゲート on なら open 時点で Err。
    {
        let _env = EnvGuard::set("DENNO_GGUF_VALIDATE_LAYOUT", "1");
        let e3 = GgufFile::open(&p)
            .err()
            .expect("ゲート on でも open が通った");
        assert!(e3.0.contains("桁溢れ"), "予期しないエラー: {}", e3.0);
    }
}

#[test]
fn tensor_offset_past_eof_is_clean_error() {
    let _g = serial();
    let dir = TempDir::new("offeof");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    // データ部 18 B しかないのに offset 4096
    let b = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 4096 }],
        nb,
    );
    let p = dir.write("off_eof.gguf", &b);
    let f = GgufFile::open(&p).expect("open");
    let e = f.tensor_bytes("t.weight").err().expect("終端超過が読めた");
    assert!(e.0.contains("ファイル終端"), "予期しないエラー: {}", e.0);
    assert!(f.validate_layout().is_err());
}

#[test]
fn truncated_file_prefixes_are_clean_errors() {
    let _g = serial();
    let dir = TempDir::new("prefix");
    let dims = [64u64, 2];
    let nb = q4_bytes(64, 2) as usize;
    let b = synth(
        3,
        None,
        &[("general.architecture", Val::Str("qwen4exp"))],
        &[
            TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 },
            TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: 72 },
        ],
        2 * nb,
    );
    let full = GgufFile::open(dir.write("full.gguf", &b)).expect("完全なファイルは開ける");
    let header_len = full.data_start as usize;
    drop(full);

    let p = dir.write("prefix.gguf", &[]);
    let mut ok = 0usize;
    for len in 0..header_len {
        std::fs::write(&p, &b[..len]).expect("write prefix");
        match GgufFile::open(&p) {
            Ok(_) => ok += 1,
            Err(e) => assert!(!e.0.is_empty(), "len={len}: 空のエラー"),
        }
    }
    assert_eq!(ok, 0, "ヘッダ手前までの切り詰めが {} 件通った", ok);
}

// ===== split シャード =====

fn split_shard(
    no: u64,
    count: u64,
    tensors_count: Option<u64>,
    kvs: &[(&str, Val)],
    tensors: &[TensorSpec],
    data: &[u8],
) -> Vec<u8> {
    let mut kv: Vec<(&str, Val)> = vec![
        ("split.no", Val::U32(no as u32)),
        ("split.count", Val::U32(count as u32)),
    ];
    if let Some(n) = tensors_count {
        kv.push(("split.tensors.count", Val::U64(n)));
    }
    kv.extend_from_slice(kvs);
    // data は任意バイト列にしたいので、synth のゼロ埋めの後に上書きする。
    let mut b = synth(3, None, &kv, tensors, data.len());
    let n = b.len();
    b[n - data.len()..].copy_from_slice(data);
    b
}

#[test]
fn split_two_shards_load_and_route_tensor_bytes() {
    let _g = serial();
    let dir = TempDir::new("split_ok");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;

    let shard0 = split_shard(
        0,
        2,
        Some(2),
        &[("general.architecture", Val::Str("qwen3_5"))],
        &[TensorSpec { name: "blk.0.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0x11u8; nb],
    );
    let shard1 = split_shard(
        1,
        2,
        Some(2),
        &[("qwen3_5.block_count", Val::U32(2))],
        &[TensorSpec { name: "blk.1.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0x22u8; nb],
    );
    let p0 = dir.write("p0.gguf", &shard0);
    let p1 = dir.write("p1.gguf", &shard1);

    // 渡す順は逆でも split.no で並べ直される。
    let f = GgufFile::open_split(&[p1.clone(), p0.clone()]).expect("split が開けない");
    assert_eq!(f.part_count(), 2);
    assert_eq!(f.version, 3);
    assert_eq!(f.path_at(0), Some(p0.as_path()));
    assert_eq!(f.path_at(1), Some(p1.as_path()));
    // メタデータ: パート 0 が正・他パートの固有キーは補われる。
    assert_eq!(f.get("general.architecture").and_then(|v| v.as_str()), Some("qwen3_5"));
    assert_eq!(f.get("qwen3_5.block_count").and_then(|v| v.as_u64()), Some(2));
    assert_eq!(f.tensor_order.len(), 2);

    let b0 = f.tensor_bytes("blk.0.weight").expect("part0 bytes");
    assert_eq!(b0.len(), nb);
    assert!(b0.iter().all(|&x| x == 0x11), "part0 のデータが別パートから読まれた");
    let b1 = f.tensor_bytes("blk.1.weight").expect("part1 bytes");
    assert_eq!(b1.len(), nb);
    assert!(b1.iter().all(|&x| x == 0x22), "part1 のデータが別パートから読まれた");

    let (file, _off, len) = f.tensor_mapping_parts("blk.1.weight").expect("mapping");
    assert_eq!(file, 1);
    assert_eq!(len, nb);
    assert!(f.tensor_mapping("blk.1.weight").is_none(), "パート 1 が単一ファイル経路に出た");
    f.validate_layout().expect("validate_layout");
}

#[test]
fn split_requires_split_no_and_count() {
    let _g = serial();
    let dir = TempDir::new("split_nokv");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let with = split_shard(
        0,
        2,
        Some(1),
        &[],
        &[TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    // split.no / split.count を書かないパート
    let without = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: 0 }],
        nb,
    );
    let p0 = dir.write("p0.gguf", &with);
    let p1 = dir.write("p1.gguf", &without);
    let e = GgufFile::open_split(&[p0, p1]).err().expect("split.no 無しが通った");
    assert!(e.0.contains("split.no"), "予期しないエラー: {}", e.0);
}

#[test]
fn split_rejects_non_sequential_and_duplicate_numbers() {
    let _g = serial();
    let dir = TempDir::new("split_no");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let mk = |name: &str, no: u64, count: u64| {
        synth(
            3,
            None,
            &[
                ("split.no", Val::U32(no as u32)),
                ("split.count", Val::U32(count as u32)),
                ("split.tensors.count", Val::U64(2)),
            ],
            &[TensorSpec { name, dims: &dims, ty: 2, offset: 0 }],
            nb,
        )
    };

    // 欠番 (0 と 2)
    let a = dir.write("gap0.gguf", &mk("a.weight", 0, 2));
    let b = dir.write("gap2.gguf", &mk("b.weight", 2, 2));
    let e = GgufFile::open_split(&[a, b]).err().expect("欠番が通った");
    assert!(e.0.contains("連番でない"), "予期しないエラー: {}", e.0);

    // 重複 (0 と 0)
    let c = dir.write("dup0.gguf", &mk("c.weight", 0, 2));
    let d = dir.write("dup0b.gguf", &mk("d.weight", 0, 2));
    let e2 = GgufFile::open_split(&[c, d]).err().expect("重複 split.no が通った");
    assert!(e2.0.contains("連番でない"), "予期しないエラー: {}", e2.0);
}

#[test]
fn split_count_mismatch_rejected() {
    let _g = serial();
    let dir = TempDir::new("split_count");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let mk = |name: &str, no: u64, count: u64| {
        synth(
            3,
            None,
            &[
                ("split.no", Val::U32(no as u32)),
                ("split.count", Val::U32(count as u32)),
            ],
            &[TensorSpec { name, dims: &dims, ty: 2, offset: 0 }],
            nb,
        )
    };
    let a = dir.write("a.gguf", &mk("a.weight", 0, 3));
    let b = dir.write("b.gguf", &mk("b.weight", 1, 3));
    let e = GgufFile::open_split(&[a, b]).err().expect("count 不一致が通った");
    assert!(e.0.contains("一致しない"), "予期しないエラー: {}", e.0);
}

#[test]
fn split_tensors_count_mismatch_rejected() {
    let _g = serial();
    let dir = TempDir::new("split_tc");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;

    // パート間で宣言値が食い違う
    let a = split_shard(
        0,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let b = split_shard(
        1,
        2,
        Some(3),
        &[],
        &[TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let p0 = dir.write("tc0.gguf", &a);
    let p1 = dir.write("tc1.gguf", &b);
    let e = GgufFile::open_split(&[p0, p1]).err().expect("宣言値の食い違いが通った");
    assert!(e.0.contains("食い違う"), "予期しないエラー: {}", e.0);

    // 実テンソル数と宣言値が食い違う (2 本しかないのに 5 を宣言)
    let a2 = split_shard(
        0,
        2,
        Some(5),
        &[],
        &[TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let b2 = split_shard(
        1,
        2,
        Some(5),
        &[],
        &[TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let q0 = dir.write("tc0b.gguf", &a2);
    let q1 = dir.write("tc1b.gguf", &b2);
    let e2 = GgufFile::open_split(&[q0, q1]).err().expect("宣言値 5 が通った");
    assert!(e2.0.contains("宣言値"), "予期しないエラー: {}", e2.0);
}

#[test]
fn split_duplicate_tensor_name_across_parts_rejected() {
    let _g = serial();
    let dir = TempDir::new("split_dupt");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let a = split_shard(
        0,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "same.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let b = split_shard(
        1,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "same.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let p0 = dir.write("dt0.gguf", &a);
    let p1 = dir.write("dt1.gguf", &b);
    let e = GgufFile::open_split(&[p0, p1]).err().expect("パート跨ぎ重複が通った");
    assert!(e.0.contains("重複"), "予期しないエラー: {}", e.0);
}

#[test]
fn split_with_corrupted_shard_header_rejected() {
    let _g = serial();
    let dir = TempDir::new("split_corrupt");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let good = split_shard(
        0,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let mut bad = split_shard(
        1,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    bad.truncate(bad.len() / 2);
    let p0 = dir.write("c0.gguf", &good);
    let p1 = dir.write("c1.gguf", &bad);
    let e = GgufFile::open_split(&[p0, p1]).err().expect("切れたシャードが通った");
    assert!(!e.0.is_empty(), "空のエラー文字列");

    // split.count = u64::MAX 級の宣言でも過剰確保せず Err
    let huge = synth(
        3,
        None,
        &[("split.no", Val::U32(1)), ("split.count", Val::U64(u64::MAX))],
        &[],
        0,
    );
    let q0 = dir.write("h0.gguf", &good);
    let q1 = dir.write("h1.gguf", &huge);
    let e2 = GgufFile::open_split(&[q0, q1]).err().expect("巨大 split.count が通った");
    assert!(e2.0.contains("一致しない"), "予期しないエラー: {}", e2.0);
}

#[test]
fn split_tensor_offset_overflow_surfaces_as_clean_error() {
    let _g = serial();
    let dir = TempDir::new("split_offovf");
    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let a = split_shard(
        0,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "a.weight", dims: &dims, ty: 2, offset: 0 }],
        &vec![0u8; nb],
    );
    let b = split_shard(
        1,
        2,
        Some(2),
        &[],
        &[TensorSpec { name: "b.weight", dims: &dims, ty: 2, offset: u64::MAX }],
        &vec![0u8; nb],
    );
    let p0 = dir.write("o0.gguf", &a);
    let p1 = dir.write("o1.gguf", &b);
    let f = GgufFile::open_split(&[p0.clone(), p1.clone()]).expect("open_split");
    let e = f.tensor_bytes("b.weight").err().expect("offset 桁溢れが読めた");
    assert!(e.0.contains("桁溢れ"), "予期しないエラー: {}", e.0);
    let e2 = f.validate_layout().err().expect("validate_layout が通った");
    assert!(e2.0.contains("桁溢れ"), "予期しないエラー: {}", e2.0);
    drop(f);

    {
        let _env = EnvGuard::set("DENNO_GGUF_VALIDATE_LAYOUT", "1");
        let e3 = GgufFile::open_split(&[p0, p1])
            .err()
            .expect("ゲート on でも open_split が通った");
        assert!(e3.0.contains("桁溢れ"), "予期しないエラー: {}", e3.0);
    }
}

#[test]
fn split_empty_and_single_path_handling() {
    let _g = serial();
    let dir = TempDir::new("split_edge");
    let e = GgufFile::open_split(&[]).err().expect("空のパス列が通った");
    assert!(e.0.contains("空"), "予期しないエラー: {}", e.0);

    let dims = [32u64, 1];
    let nb = q4_bytes(32, 1) as usize;
    let b = synth(
        3,
        None,
        &[],
        &[TensorSpec { name: "t.weight", dims: &dims, ty: 2, offset: 0 }],
        nb,
    );
    let p = dir.write("single.gguf", &b);
    let f = GgufFile::open_split(&[p]).expect("1 パスは open に委譲される");
    assert_eq!(f.part_count(), 1);
    assert_eq!(f.tensor_bytes("t.weight").expect("bytes").len(), nb);
}

// ===== ファイル全体のバイト改変 (パニックしないこと) =====

#[test]
fn byte_mutation_fuzz_never_panics() {
    let _g = serial();
    let dir = TempDir::new("fuzz");
    let dims = [64u64, 2];
    let nb = q4_bytes(64, 2) as usize;
    let base = synth(
        3,
        Some(32),
        &[
            ("general.architecture", Val::Str("qwen4exp")),
            ("qwen4exp.block_count", Val::U32(2)),
            ("qwen4exp.x", Val::U64(u64::MAX)),
        ],
        &[
            TensorSpec { name: "blk.0.a.weight", dims: &dims, ty: 2, offset: 0 },
            TensorSpec { name: "blk.1.b.weight", dims: &dims, ty: 8, offset: 144 },
        ],
        2 * nb,
    );
    GgufFile::open(dir.write("base.gguf", &base)).expect("基準ファイルが開けない");

    let path = dir.write("mut.gguf", &base);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ok = 0usize;
    let mut err = 0usize;
    const ITERS: usize = 400;
    for _ in 0..ITERS {
        let mut b = base.clone();
        let mutations = 1 + (next() % 4) as usize;
        for _ in 0..mutations {
            let i = (next() as usize) % b.len();
            if next() % 10 < 8 {
                b[i] = (next() & 0xff) as u8;
            } else {
                let i = i.min(b.len().saturating_sub(8));
                let v = if next() % 2 == 0 { u64::MAX } else { 1u64 << 62 };
                let n = 8.min(b.len() - i);
                b[i..i + n].copy_from_slice(&v.to_le_bytes()[..n]);
            }
        }
        if next() % 10 < 3 {
            let n = (next() as usize) % b.len();
            b.truncate(n);
        }
        std::fs::write(&path, &b).expect("write mut");
        match GgufFile::open(&path) {
            Ok(f) => {
                ok += 1;
                // 壊れた offset / 範囲でもアクセスは Err で返る (パニックしない)。
                for name in &f.tensor_order {
                    let _ = f.tensor_bytes(name);
                }
                let _ = f.validate_layout();
            }
            Err(e) => {
                assert!(!e.0.is_empty(), "空のエラー文字列");
                err += 1;
            }
        }
    }
    assert!(err > 0, "壊した入力が 1 件もエラーにならなかった");
    assert!(ok > 0, "健全な入力が 1 件も通らなかった (fuzz が効いていない)");
}
