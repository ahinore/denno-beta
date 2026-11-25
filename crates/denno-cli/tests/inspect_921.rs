//! task 921: `denno inspect <gguf> [--json]` の CLI 契約の試験。
//!
//! 合成の最小 GGUF を一時ディレクトリに書き、実際のバイナリ
//! (`CARGO_BIN_EXE_denno`) を起動して確かめる:
//!
//! 1. テキスト出力にアーキ情報 / 量子型ヒストグラム / VRAM 見積りが出る。
//! 2. `--json` は stdout が単体で解析できる JSON で、数値が期待どおり。
//! 3. 重み (テンソルのデータ部) を読まない — データ部が 0 バイトでも成功し、
//!    テンソルのバイト数はヘッダの次元・型から出る。
//! 4. 存在しないファイル / magic 不正 / 途中で切れたファイルは非 0 で終わり、
//!    panic せずに理由を stderr に出す。
//! 5. `--expert-cache-mib` が見積りに効く。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

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
            "denno921_{}_{}_{}_{}",
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
///
/// `data_len` はデータ部に置くバイト数 (0 ならヘッダだけで終わる =
/// 「重みは 1 バイトも無い」ファイル)。
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

/// 試験用の既定ヘッダ。e=64 / 層 4 / expert 8 / ctx 1024 / vocab 5。
///
/// テンソル:
/// - `token_embd.weight` F32 [64, 5] = 1,280 B (dense / host 側)
/// - `blk.0.attn_q.weight` Q8_0 [64, 2] = 128/32*34 = 136 B (dense / dev 側)
/// - `blk.{0,1}.ffn_gate_exps.weight` Q8_0 [64, 2, 8] = 1,088 B ずつ (expert)
fn default_fixture(data_len: u64) -> Vec<u8> {
    let tokens = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ];
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen4exp".to_string())),
        ("qwen4exp.block_count", Kv::U32(4)),
        ("qwen4exp.embedding_length", Kv::U32(64)),
        ("qwen4exp.expert_count", Kv::U32(8)),
        ("qwen4exp.context_length", Kv::U32(1024)),
        ("qwen4exp.full_attention_interval", Kv::U32(2)),
        ("qwen4exp.attention.head_count_kv", Kv::U32(2)),
        ("qwen4exp.attention.key_length", Kv::U32(32)),
        ("tokenizer.ggml.tokens", Kv::StrArr(tokens)),
    ];
    let tensors: Vec<TensorSpec> = vec![
        ("token_embd.weight", vec![64, 5], T_F32, 0),
        ("blk.0.attn_q.weight", vec![64, 2], T_Q8_0, 1280),
        ("blk.0.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 1440),
        ("blk.1.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 2528),
    ];
    gguf_bytes(&kvs, &tensors, data_len)
}

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
    Command::new(env!("CARGO_BIN_EXE_denno"))
        .arg("inspect")
        .args(&args)
        .output()
        .expect("run denno inspect")
}

fn json_of(out: &Output) -> serde_json::Value {
    let stdout = String::from_utf8(out.stdout.clone()).expect("stdout utf8");
    serde_json::from_str(&stdout).expect("stdout is a single JSON object")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

// ===== 1. テキスト出力 =====

#[test]
fn text_reports_arch_quant_and_vram() {
    let dir = TempDir::new("text");
    let p = dir.write("m.gguf", &default_fixture(4096));
    let out = run_inspect_path(&p, &[]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let s = String::from_utf8(out.stdout).expect("utf8");
    for needle in [
        "architecture: qwen4exp",
        "layers: 4",
        "hidden size: 64",
        "experts: 8",
        "context length: 1024",
        "vocab size: 5",
        "tensors: 4",
        "quantization:",
        "Q8_0",
        "F32",
        "vram estimate:",
        "full gpu",
        "dense + expert cache",
        "expert cache (4096 MiB)",
    ] {
        assert!(s.contains(needle), "missing {needle:?} in:\n{s}");
    }
}

// ===== 2. JSON の数値 =====

#[test]
fn json_reports_exact_sizes() {
    let dir = TempDir::new("json");
    let p = dir.write("m.gguf", &default_fixture(4096));
    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);

    assert_eq!(v["gguf_version"], 3);
    assert_eq!(v["architecture"], "qwen4exp");
    assert_eq!(v["layer_count"], 4);
    assert_eq!(v["hidden_size"], 64);
    assert_eq!(v["expert_count"], 8);
    assert_eq!(v["context_length"], 1024);
    assert_eq!(v["vocab_size"], 5);
    assert_eq!(v["tensor_count"], 4);
    // 1280 + 136 + 1088 + 1088
    assert_eq!(v["total_tensor_bytes"], 3592);

    // ヒストグラムは型名で引く。
    let quant = v["quant"].as_array().expect("quant array");
    let f32e = quant.iter().find(|q| q["ty"] == "F32").expect("F32 entry");
    assert_eq!(f32e["tensors"], 1);
    assert_eq!(f32e["bytes"], 1280);
    let q8 = quant.iter().find(|q| q["ty"] == "Q8_0").expect("Q8_0 entry");
    assert_eq!(q8["tensors"], 3);
    assert_eq!(q8["bytes"], 136 + 1088 + 1088);

    // dense = F32 1280 (host) + Q8_0 136 (dev) / expert = 2 * 1088
    assert_eq!(v["vram"]["dense_bytes"], 1416);
    assert_eq!(v["vram"]["expert_bytes"], 2176);
    // full 層 = ceil(4/2) = 2, kv_dim = 2 * 32 = 64,
    // kv = 2 * 1024 * 64 * 2 * 4 = 1,048,576
    assert_eq!(v["vram"]["kv_cache_bytes"], 1_048_576u64);
    assert_eq!(
        v["vram"]["full_gpu_bytes"],
        1416u64 + 2176 + 1_048_576
    );
    // 既定 4096 MiB の予算は expert 全量 (2176 B) より大きいので頭打ち。
    assert_eq!(v["vram"]["expert_cache_mib"], 4096);
    assert_eq!(v["vram"]["expert_cache_bytes"], 2176);
    assert_eq!(
        v["vram"]["dense_plus_expert_cache_bytes"],
        1416u64 + 2176 + 1_048_576
    );
}

#[test]
fn expert_cache_mib_changes_the_budget() {
    let dir = TempDir::new("cache");
    let p = dir.write("m.gguf", &default_fixture(0));
    let out = run_inspect_path(&p, &["--json", "--expert-cache-mib", "0"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);
    assert_eq!(v["vram"]["expert_cache_mib"], 0);
    assert_eq!(v["vram"]["expert_cache_bytes"], 0);
    // dense + kv だけになる。
    assert_eq!(
        v["vram"]["dense_plus_expert_cache_bytes"],
        1416u64 + 1_048_576
    );
    // 全段 GPU は予算に依存しない。
    assert_eq!(v["vram"]["full_gpu_bytes"], 1416u64 + 2176 + 1_048_576);
}

// ===== 3. 重みを読まない =====

#[test]
fn header_only_file_reports_tensor_bytes() {
    // データ部 0 バイト = 重みが 1 バイトも無いファイルでも成功し、
    // テンソルのバイト数は次元と型から出る (データ部を読んでいない証拠)。
    let dir = TempDir::new("headeronly");
    let bytes = default_fixture(0);
    let p = dir.write("head_only.gguf", &bytes);
    assert_eq!(std::fs::metadata(&p).expect("stat").len() as usize, bytes.len());

    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);
    assert_eq!(v["tensor_count"], 4);
    assert_eq!(v["total_tensor_bytes"], 3592);
    let q8 = v["quant"]
        .as_array()
        .expect("quant")
        .iter()
        .find(|q| q["ty"] == "Q8_0")
        .expect("Q8_0")
        .clone();
    assert_eq!(q8["bytes"], 2312);
}

// ===== 4. 壊れた入力 =====

#[test]
fn missing_file_fails_cleanly() {
    let dir = TempDir::new("missing");
    let p = dir.as_path().join("nope.gguf");
    let out = run_inspect_path(&p, &[]);
    assert!(!out.status.success(), "should fail");
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(&out);
    assert!(e.contains("nope.gguf"), "stderr should name the file: {e}");
    assert!(!e.contains("panicked"), "must not panic: {e}");
}

#[test]
fn bad_magic_fails_cleanly() {
    let dir = TempDir::new("badmagic");
    let mut b = default_fixture(0);
    b[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
    let p = dir.write("bad.gguf", &b);
    let out = run_inspect_path(&p, &[]);
    assert!(!out.status.success(), "should fail");
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(&out);
    assert!(e.contains("not GGUF"), "stderr should say not GGUF: {e}");
    assert!(!e.contains("panicked"), "must not panic: {e}");
}

#[test]
fn truncated_header_fails_cleanly() {
    let dir = TempDir::new("truncated");
    let b = default_fixture(0);
    // メタデータの途中で切る (magic + version + counts すら途中)。
    let p = dir.write("cut.gguf", &b[..10]);
    let out = run_inspect_path(&p, &[]);
    assert!(!out.status.success(), "should fail");
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(&out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    assert!(
        e.contains("gguf") || e.contains("EOF"),
        "stderr should explain: {e}"
    );
}

#[test]
fn directory_is_rejected() {
    let dir = TempDir::new("dir");
    let out = run_inspect(&[dir.as_path().as_os_str()]);
    assert!(!out.status.success(), "should fail for a directory");
    let e = stderr_of(&out);
    assert!(e.contains("ディレクトリ"), "stderr: {e}");
}

// ===== 5. メタデータ欠けでも落ちない =====

#[test]
fn missing_metadata_prints_question_marks() {
    let dir = TempDir::new("nometa");
    let kvs: Vec<(&str, Kv)> = vec![("general.architecture", Kv::Str("llama".to_string()))];
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![32, 4], T_F32, 0)];
    let p = dir.write("bare.gguf", &gguf_bytes(&kvs, &tensors, 512));

    let text = run_inspect_path(&p, &[]);
    assert!(text.status.success(), "stderr: {}", stderr_of(&text));
    let s = String::from_utf8(text.stdout).expect("utf8");
    assert!(s.contains("architecture: llama"), "{s}");
    assert!(s.contains("layers: ?"), "{s}");
    assert!(s.contains("experts: ?"), "{s}");

    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);
    assert_eq!(v["architecture"], "llama");
    assert!(v["layer_count"].is_null(), "unknown stays null");
    assert_eq!(v["vram"]["kv_cache_bytes"], 0);
    assert_eq!(v["total_tensor_bytes"], 512);
}
