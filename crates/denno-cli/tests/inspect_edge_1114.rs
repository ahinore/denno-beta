//! task 1114: `denno inspect` の**異常系**と `--expert-cache-mib` の境界の試験。
//!
//! task 921 の正常系 (`inspect_921.rs`) を補い、実バイナリ
//! (`CARGO_BIN_EXE_denno`) を起動して次を確かめる:
//!
//! 1. 存在しないパス / ディレクトリ / 0 バイトの空ファイル / magic 不正 /
//!    途中で切れたヘッダは、いずれも panic せず非 0 で終わる
//!    (panic の 101 ではない = 通常のエラー経路)。
//! 2. `--expert-cache-mib 0 / 2048 / 16384` の値がテキスト出力と `--json` の
//!    両方の VRAM 見積りに正しく反映される (expert 全量より小さい予算では
//!    予算で頭打ち、大きい予算では expert 全量で頭打ち)。
//! 3. `--json` の stdout は単体で解析できる 1 個の JSON オブジェクトだけで、
//!    テキストやログが混ざらない。失敗時は stdout が空のまま。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const MIB: u64 = 1 << 20;

/// `inspect::mib` と同じ丸め (最近傍 MiB)。
fn mib(bytes: u64) -> u64 {
    bytes.saturating_add(MIB / 2) / MIB
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ (Drop で再帰削除)。
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "denno1114_{}_{}_{}_{}",
            tag,
            std::process::id(),
            n,
            nanos
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.path.join(name);
        std::fs::write(&p, bytes).expect("write file");
        p
    }

    fn as_path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ===== 合成 GGUF の書き出し =====

enum Kv {
    Str(String),
    U32(u32),
    StrArr(Vec<String>),
}

fn w_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

/// gguf メタデータ型 id (gguf.h の enum gguf_type)。
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;

/// GGML 型 id (ggml.h の enum ggml_type)。
const T_F32: u32 = 0;
const T_Q8_0: u32 = 8;

/// `(名前, 次元, GGML 型 id, データ部オフセット)`。
type TensorSpec = (&'static str, Vec<u64>, u32, u64);

/// ヘッダ + ゼロ埋めしたデータ部からなる GGUF を組み立てる。
fn gguf_bytes(kvs: &[(&str, Kv)], tensors: &[TensorSpec], data_len: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kvs.len() as u64).to_le_bytes());

    for (k, v) in kvs {
        w_str(&mut b, k);
        match v {
            Kv::Str(s) => {
                b.extend(GT_STRING.to_le_bytes());
                w_str(&mut b, s);
            }
            Kv::U32(x) => {
                b.extend(GT_U32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            Kv::StrArr(items) => {
                b.extend(GT_ARRAY.to_le_bytes());
                b.extend(GT_STRING.to_le_bytes());
                b.extend((items.len() as u64).to_le_bytes());
                for it in items {
                    w_str(&mut b, it);
                }
            }
        }
    }

    for (name, dims, ty, off) in tensors {
        w_str(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
    }

    // データ部は既定 alignment 32 に整列する。
    let data_start = b.len().div_ceil(32) * 32;
    b.resize(data_start + data_len as usize, 0);
    b
}

/// メタデータだけを持つアーキ情報 (qwen4exp / 4 層 / ctx 1024 / kv 2x32 / vocab 5)。
fn arch_kvs() -> Vec<(&'static str, Kv)> {
    let tokens: Vec<String> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    vec![
        ("general.architecture", Kv::Str("qwen4exp".to_string())),
        ("qwen4exp.block_count", Kv::U32(4)),
        ("qwen4exp.embedding_length", Kv::U32(64)),
        ("qwen4exp.expert_count", Kv::U32(8)),
        ("qwen4exp.context_length", Kv::U32(1024)),
        ("qwen4exp.full_attention_interval", Kv::U32(2)),
        ("qwen4exp.attention.head_count_kv", Kv::U32(2)),
        ("qwen4exp.attention.key_length", Kv::U32(32)),
        ("tokenizer.ggml.tokens", Kv::StrArr(tokens)),
    ]
}

/// 異常系用の小さな合成 GGUF (データ部 0 バイト)。
///
/// `token_embd.weight` F32 [64, 5] = 1,280 B (dense/host)、
/// `blk.0.attn_q.weight` Q8_0 [64, 2] = 136 B (dense/dev)、
/// `blk.{0,1}.ffn_gate_exps.weight` Q8_0 [64, 2, 8] = 1,088 B ずつ (expert)。
fn small_fixture() -> Vec<u8> {
    let tensors: Vec<TensorSpec> = vec![
        ("token_embd.weight", vec![64, 5], T_F32, 0),
        ("blk.0.attn_q.weight", vec![64, 2], T_Q8_0, 1280),
        ("blk.0.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 1440),
        ("blk.1.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 2528),
    ];
    gguf_bytes(&arch_kvs(), &tensors, 0)
}

/// `--expert-cache-mib` の頭打ちを区別できる大きさの合成 GGUF。
///
/// - dense: `token_embd.weight` F32 [65536, 32768] = 2^31 elem * 4 B
///   = 8,589,934,592 B (= 8,192 MiB、host 側)
/// - expert: `blk.0.ffn_gate_exps.weight` Q8_0 [65536, 32768]
///   = 2^31 / 32 * 34 = 2,281,701,376 B (= 2,176 MiB)
///
/// expert 全量 2,176 MiB なので:
/// - `--expert-cache-mib 0`     → 0 B (予算 0)
/// - `--expert-cache-mib 2048`  → 2,147,483,648 B (予算で頭打ち)
/// - `--expert-cache-mib 16384` → 2,281,701,376 B (expert 全量で頭打ち)
///
/// データ部は 0 バイト (重みは読まれない)。
fn budget_fixture() -> Vec<u8> {
    let tensors: Vec<TensorSpec> = vec![
        ("token_embd.weight", vec![65536, 32768], T_F32, 0),
        (
            "blk.0.ffn_gate_exps.weight",
            vec![65536, 32768],
            T_Q8_0,
            8_589_934_592,
        ),
    ];
    gguf_bytes(&arch_kvs(), &tensors, 0)
}

// 見積りの期待値 (ヘッダの次元と型から独立に計算した値)。
const DENSE_BYTES: u64 = 8_589_934_592;
const EXPERT_BYTES: u64 = 2_281_701_376;
/// 2 full 層 * ctx 1024 * kv_dim 64 * 2 * 4 B
const KV_BYTES: u64 = 1_048_576;
const FULL_GPU_BYTES: u64 = DENSE_BYTES + EXPERT_BYTES + KV_BYTES;

// ===== プロセス起動 =====

fn run_inspect(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denno"))
        .arg("inspect")
        .args(args)
        .output()
        .expect("run denno inspect")
}

fn run_inspect_path(path: &Path, extra: &[&str]) -> Output {
    let mut args: Vec<std::ffi::OsString> = vec![path.as_os_str().to_os_string()];
    for e in extra {
        args.push(std::ffi::OsString::from(*e));
    }
    run_inspect(&args.iter().map(|s| s.as_os_str()).collect::<Vec<_>>())
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn json_of(out: &Output) -> serde_json::Value {
    serde_json::from_str(&stdout_of(out)).expect("stdout is a single JSON object")
}

/// 失敗の共通契約: 非 0 / panic (101) ではない / stdout 空 / 理由が stderr。
fn assert_clean_failure(out: &Output, what: &str, needle: &str) {
    let code = out
        .status
        .code()
        .unwrap_or_else(|| panic!("{what}: no exit code (killed by signal?)"));
    assert_ne!(code, 0, "{what}: must exit non-zero");
    assert_ne!(code, 101, "{what}: exit 101 means a panic: {}", stderr_of(out));
    assert!(
        out.stdout.is_empty(),
        "{what}: stdout must stay empty, got {:?}",
        stdout_of(out)
    );
    let e = stderr_of(out);
    assert!(
        e.contains(needle),
        "{what}: stderr should mention {needle:?}, got:\n{e}"
    );
    assert!(
        !e.contains("panicked"),
        "{what}: must not panic, stderr:\n{e}"
    );
    assert!(
        !e.contains("RUST_BACKTRACE"),
        "{what}: must not print a backtrace, stderr:\n{e}"
    );
}

// ===== 1. 異常系 (パス / ディレクトリ / 空 / magic / 切れ) =====

#[test]
fn missing_path_fails_cleanly() {
    let dir = TempDir::new("missing");
    let p = dir.as_path().join("nope.gguf");
    let out = run_inspect_path(&p, &[]);
    assert_clean_failure(&out, "missing path", "nope.gguf");
}

#[test]
fn directory_path_fails_cleanly() {
    let dir = TempDir::new("isdir");
    let out = run_inspect(&[dir.as_path().as_os_str()]);
    assert_clean_failure(&out, "directory path", "ディレクトリ");
}

#[test]
fn empty_file_fails_cleanly() {
    let dir = TempDir::new("empty");
    let p = dir.write("empty.gguf", &[]);
    assert_eq!(std::fs::metadata(&p).expect("stat").len(), 0);

    let out = run_inspect_path(&p, &[]);
    assert_clean_failure(&out, "0-byte file", "EOF");

    // `--json` でも stdout に中途半端な JSON を出さない。
    let json_out = run_inspect_path(&p, &["--json"]);
    assert_clean_failure(&json_out, "0-byte file (--json)", "EOF");
}

#[test]
fn bad_magic_fails_cleanly() {
    let dir = TempDir::new("badmagic");

    // magic だけ差し替えた完全なヘッダ。
    let mut b = small_fixture();
    b[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
    let p = dir.write("bad.gguf", &b);
    let out = run_inspect_path(&p, &[]);
    assert_clean_failure(&out, "bad magic", "not GGUF");

    // 4 バイトだけの極小ファイル (magic 不正 + 以後も無い)。
    let p2 = dir.write("four.gguf", &0xdead_beefu32.to_le_bytes());
    let out2 = run_inspect_path(&p2, &[]);
    assert_clean_failure(&out2, "bad magic (4 bytes)", "not GGUF");
}

#[test]
fn truncated_header_fails_cleanly() {
    let dir = TempDir::new("truncated");
    let b = small_fixture();

    // 先頭 10 バイト (version の途中まで) で切る。
    let p1 = dir.write("cut10.gguf", &b[..10]);
    assert_clean_failure(&run_inspect_path(&p1, &[]), "truncated @10", "EOF");

    // magic + version + 件数だけの 24 バイトで切る (メタデータ先頭で途切れる)。
    let p2 = dir.write("cut24.gguf", &b[..24]);
    assert_clean_failure(&run_inspect_path(&p2, &[]), "truncated @24", "EOF");

    // メタデータの後半で切る (KV 数は揃っているが中身が足りない)。
    let cut = b.len() / 3;
    let p3 = dir.write("cut_mid.gguf", &b[..cut]);
    assert_clean_failure(&run_inspect_path(&p3, &[]), "truncated mid", "EOF");
}

// ===== 2. --expert-cache-mib の反映 (テキスト + JSON) =====

#[test]
fn expert_cache_mib_is_reflected_in_text_and_json() {
    let dir = TempDir::new("cache");
    let p = dir.write("budget.gguf", &budget_fixture());

    // (指定 MiB, 期待するキャッシュ実バイト数)
    let cases: [(u64, u64); 3] = [
        (0, 0),
        (2048, 2048 * MIB),          // expert 全量 (2,176 MiB) より小さい → 予算で頭打ち
        (16384, EXPERT_BYTES),       // expert 全量より大きい → expert 全量で頭打ち
    ];

    for (requested, want_cache_bytes) in cases {
        let budget = requested.to_string();

        // --- JSON ---
        let out = run_inspect_path(&p, &["--json", "--expert-cache-mib", &budget]);
        assert!(
            out.status.success(),
            "--expert-cache-mib {requested}: {}",
            stderr_of(&out)
        );
        let v = json_of(&out);
        assert_eq!(v["vram"]["expert_cache_mib"].as_u64(), Some(requested));
        assert_eq!(
            v["vram"]["expert_cache_bytes"].as_u64(),
            Some(want_cache_bytes),
            "--expert-cache-mib {requested}: cache bytes"
        );
        assert_eq!(v["vram"]["dense_bytes"].as_u64(), Some(DENSE_BYTES));
        assert_eq!(v["vram"]["expert_bytes"].as_u64(), Some(EXPERT_BYTES));
        assert_eq!(v["vram"]["kv_cache_bytes"].as_u64(), Some(KV_BYTES));
        // 全段 GPU は予算に依存しない。
        assert_eq!(
            v["vram"]["full_gpu_bytes"].as_u64(),
            Some(FULL_GPU_BYTES)
        );
        assert_eq!(
            v["vram"]["dense_plus_expert_cache_bytes"].as_u64(),
            Some(DENSE_BYTES + KV_BYTES + want_cache_bytes)
        );

        // --- テキスト ---
        let text = run_inspect_path(&p, &["--expert-cache-mib", &budget]);
        assert!(
            text.status.success(),
            "--expert-cache-mib {requested} (text): {}",
            stderr_of(&text)
        );
        let s = stdout_of(&text);
        // 行の書式は inspect::render_text と同じ (`  {label:<26}{mib:>10} MiB`)。
        let row = |label: String, num: u64| format!("  {label:<26}{num:>10} MiB");
        for expect in [
            row(format!("expert cache ({requested} MiB)"), mib(want_cache_bytes)),
            row(
                "dense + expert cache".to_string(),
                mib(DENSE_BYTES + KV_BYTES + want_cache_bytes),
            ),
            row("kv cache".to_string(), mib(KV_BYTES)),
            row("dense".to_string(), mib(DENSE_BYTES)),
            row("experts (all)".to_string(), mib(EXPERT_BYTES)),
            row("full gpu".to_string(), mib(FULL_GPU_BYTES)),
        ] {
            assert!(
                s.contains(&expect),
                "--expert-cache-mib {requested}: missing row {expect:?} in:\n{s}"
            );
        }
    }
}

// ===== 3. --json の stdout 純度 =====

#[test]
fn json_stdout_is_a_single_pure_object() {
    let dir = TempDir::new("purity");
    let p = dir.write("pure.gguf", &budget_fixture());
    let out = run_inspect_path(&p, &["--json", "--expert-cache-mib", "2048"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let s = stdout_of(&out);
    // 前後の空白以外に JSON 以外の文字が無いこと (from_str は余剰テキストを拒否する)。
    let trimmed = s.trim();
    assert!(trimmed.starts_with('{'), "stdout must start with '{{': {s:?}");
    assert!(trimmed.ends_with('}'), "stdout must end with '}}': {s:?}");
    let v: serde_json::Value = serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("stdout is not one JSON document: {e}\n{s}"));
    assert!(v.is_object(), "top level must be a JSON object: {s}");

    // テキスト出力の見出しが混ざっていない。
    for needle in ["vram estimate:", "quantization:", "architecture:", "path:"] {
        assert!(!s.contains(needle), "text leaked into --json stdout: {needle:?}");
    }
    // ログ・panic も stdout に混ざらない。
    assert!(!s.contains("panicked"), "panic text in stdout");
    assert!(!s.contains("\u{1b}["), "ANSI escapes in stdout");

    // 期待するキーがすべて 1 個のオブジェクト直下にある。
    for key in ["path", "gguf_version", "quant", "vram", "tensor_count"] {
        assert!(v.get(key).is_some(), "missing key {key}: {s}");
    }

    // stderr 側に JSON の中身が漏れていない (出力先が分かれている)。
    let e = stderr_of(&out);
    assert!(!e.contains("\"vram\""), "JSON leaked into stderr:\n{e}");
    assert!(!e.contains("panicked"), "stderr:\n{e}");
}

#[test]
fn json_mode_keeps_stdout_empty_on_error() {
    let dir = TempDir::new("purityerr");
    let mut b = small_fixture();
    b[0..4].copy_from_slice(&0u32.to_le_bytes());
    let p = dir.write("broken.gguf", &b);

    let out = run_inspect_path(&p, &["--json"]);
    assert_clean_failure(&out, "--json on bad magic", "not GGUF");
    // 構造化エラー封筒も出さない (stdout は空のまま = 契約)。
    assert!(out.stdout.is_empty(), "stdout must stay empty in --json errors");
}
