//! task 1096: `denno inspect` の型別内訳 (`quant_breakdown`) の試験。
//!
//! task 921 の `quant` (型ごとの本数とバイト数) はそのまま残し、次の契約を
//! 足す:
//!
//! 1. `--json` の `quant_breakdown` は型名の昇順で、各要素が
//!    `ty` / `tensors` / `bytes` / `mib` / `pct_bp` を持つ。
//! 2. 本数の合計は `tensor_count`、バイト数の合計は `total_tensor_bytes`、
//!    割合の合計は 10000 bp (100%) に (丸め誤差の範囲で) 一致する。
//! 3. 同じファイルを 2 回 inspect すると stdout が 1 バイトも変わらない
//!    (決定的)。
//! 4. 既存のキー (`quant` / `vram` / `tensor_count` など) は値も並びも不変。
//! 5. テキスト出力に `quant breakdown:` の節があり、割合が `12.34%` 形式で
//!    出る。テンソル 0 個のヘッダでも割合は 0 で、合計行は 0.00%。
//!
//! 合成の GGUF を一時ディレクトリに書き、実バイナリ (`CARGO_BIN_EXE_denno`)
//! を起動して確かめる (エンジンのホットパスには触らない)。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const MIB: u64 = 1 << 20;

/// `inspect::mib` と同じ丸め (最近傍 MiB)。
fn mib(bytes: u64) -> u64 {
    bytes.saturating_add(MIB / 2) / MIB
}

/// `inspect::pct_bp` と同じ式 (整数の四捨五入)。
fn pct_bp(bytes: u64, total: u64) -> u32 {
    if total == 0 {
        return 0;
    }
    (bytes.saturating_mul(10_000).saturating_add(total / 2) / total).min(10_000) as u32
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

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
            "denno1096_{}_{}_{}_{}",
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
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ===== 合成 GGUF の書き出し (inspect_921 / inspect_edge_1114 と同じ形) =====

enum Kv {
    Str(String),
    U32(u32),
    StrArr(Vec<String>),
}

fn w_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;

/// GGML 型 id (ggml.h の enum ggml_type)。
const T_F32: u32 = 0;
const T_F16: u32 = 1;
const T_Q8_0: u32 = 8;
const T_Q4_K: u32 = 12;

type TensorSpec = (&'static str, Vec<u64>, u32, u64);

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

    let data_start = b.len().div_ceil(32) * 32;
    b.resize(data_start + data_len as usize, 0);
    b
}

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

/// 4 種類の型が混在する合成 GGUF (データ部 0 バイト = 重みは読まれない)。
///
/// | テンソル | 型 | 次元 | バイト数 |
/// |---|---|---|---|
/// | token_embd.weight | F32 [64,5] | 1,280 |
/// | blk.0.attn_k.weight | F16 [64,2] | 256 |
/// | blk.0.attn_q.weight | Q8_0 [64,2] | 136 |
/// | blk.0.ffn_gate_exps.weight | Q8_0 [64,2,8] | 1,088 |
/// | blk.0.ffn_up.weight | Q4_K [256,4] | 576 |
fn mixed_fixture() -> Vec<u8> {
    let tensors: Vec<TensorSpec> = vec![
        ("token_embd.weight", vec![64, 5], T_F32, 0),
        ("blk.0.attn_k.weight", vec![64, 2], T_F16, 1280),
        ("blk.0.attn_q.weight", vec![64, 2], T_Q8_0, 1536),
        ("blk.0.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 1672),
        ("blk.0.ffn_up.weight", vec![256, 4], T_Q4_K, 2760),
    ];
    gguf_bytes(&arch_kvs(), &tensors, 0)
}

/// 期待する (型名, 本数, バイト数)。型名の昇順。
const MIXED: [(&str, u64, u64); 4] = [
    ("F16", 1, 256),
    ("F32", 1, 1280),
    ("Q4_K", 1, 576),
    ("Q8_0", 2, 1224),
];

fn mixed_total_bytes() -> u64 {
    MIXED.iter().map(|(_, _, b)| *b).sum()
}

// ===== プロセス起動 =====

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

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn json_of(out: &Output) -> serde_json::Value {
    serde_json::from_str(&stdout_of(out)).expect("stdout is a single JSON object")
}

// ===== 1. JSON: 合計と要素 =====

#[test]
fn json_breakdown_sums_match_totals() {
    let dir = TempDir::new("sums");
    let p = dir.write("mixed.gguf", &mixed_fixture());
    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);

    let arr = v["quant_breakdown"]
        .as_array()
        .expect("quant_breakdown must be an array");
    assert_eq!(arr.len(), MIXED.len(), "one entry per ggml type");

    // (ty, tensors, bytes) が期待どおりで、型名の昇順。
    for (i, (ty, tensors, bytes)) in MIXED.iter().enumerate() {
        let e = &arr[i];
        assert_eq!(e["ty"], *ty, "entry {i} type name");
        assert_eq!(e["tensors"].as_u64(), Some(*tensors), "{ty}: tensors");
        assert_eq!(e["bytes"].as_u64(), Some(*bytes), "{ty}: bytes");
        assert_eq!(e["mib"].as_u64(), Some(mib(*bytes)), "{ty}: mib");
        assert!(e["pct_bp"].is_u64(), "{ty}: pct_bp must be an integer");
    }

    // 本数・バイト数の合計がトップレベルの合計と一致する。
    let sum_tensors: u64 = arr.iter().map(|e| e["tensors"].as_u64().unwrap()).sum();
    let sum_bytes: u64 = arr.iter().map(|e| e["bytes"].as_u64().unwrap()).sum();
    assert_eq!(sum_tensors, v["tensor_count"].as_u64().unwrap());
    assert_eq!(sum_bytes, v["total_tensor_bytes"].as_u64().unwrap());
    assert_eq!(sum_bytes, mixed_total_bytes());

    // 割合はそれぞれ bytes / total の四捨五入。合計は丸め誤差の範囲で 100%。
    let sum_bp: u64 = arr.iter().map(|e| e["pct_bp"].as_u64().unwrap()).sum();
    assert!(
        sum_bp.abs_diff(10_000) <= arr.len() as u64,
        "pct_bp sum {sum_bp} should be ~10000"
    );
    for e in arr {
        let ty = e["ty"].as_str().unwrap();
        let bytes = e["bytes"].as_u64().unwrap();
        assert_eq!(
            e["pct_bp"].as_u64().unwrap() as u32,
            pct_bp(bytes, sum_bytes),
            "{ty}: pct_bp"
        );
    }
}

// ===== 2. 既存キーの後方互換 =====

#[test]
fn json_keeps_all_previous_keys_and_quant_values() {
    let dir = TempDir::new("compat");
    let p = dir.write("mixed.gguf", &mixed_fixture());
    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);

    for key in [
        "path",
        "gguf_version",
        "architecture",
        "layer_count",
        "hidden_size",
        "expert_count",
        "context_length",
        "vocab_size",
        "tensor_count",
        "total_tensor_bytes",
        "quant",
        "vram",
    ] {
        assert!(v.get(key).is_some(), "missing pre-1096 key {key}");
    }
    assert!(v.get("quant_breakdown").is_some(), "missing quant_breakdown");

    // `quant` は不変: (ty, tensors, bytes, mib) が `quant_breakdown` と同値。
    let quant = v["quant"].as_array().expect("quant array");
    let breakdown = v["quant_breakdown"].as_array().expect("breakdown array");
    assert_eq!(quant.len(), breakdown.len());
    for (q, b) in quant.iter().zip(breakdown.iter()) {
        assert_eq!(q["ty"], b["ty"]);
        assert_eq!(q["tensors"], b["tensors"]);
        assert_eq!(q["bytes"], b["bytes"]);
        assert_eq!(q["mib"], b["mib"]);
    }
}

// ===== 3. 決定性 =====

#[test]
fn json_output_is_deterministic_across_runs() {
    let dir = TempDir::new("determinism");
    let p = dir.write("mixed.gguf", &mixed_fixture());
    let a = run_inspect_path(&p, &["--json"]);
    let b = run_inspect_path(&p, &["--json"]);
    assert!(a.status.success(), "stderr: {}", stderr_of(&a));
    assert!(b.status.success(), "stderr: {}", stderr_of(&b));
    assert_eq!(
        stdout_of(&a),
        stdout_of(&b),
        "two runs on the same file must produce identical stdout"
    );
}

// ===== 4. テキスト出力 =====

#[test]
fn text_shows_breakdown_section_with_percentages() {
    let dir = TempDir::new("text");
    let p = dir.write("mixed.gguf", &mixed_fixture());
    let out = run_inspect_path(&p, &[]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let s = stdout_of(&out);

    assert!(s.contains("quant breakdown:"), "missing section in:\n{s}");
    // 節は `quantization:` の後、`vram estimate:` の前。
    let q = s.find("quantization:").expect("quantization section");
    let b = s.find("quant breakdown:").expect("breakdown section");
    let v = s.find("vram estimate:").expect("vram section");
    assert!(q < b && b < v, "section order:\n{s}");

    let total = mixed_total_bytes();
    for (ty, tensors, bytes) in MIXED {
        let want = format!(
            "  {:<10} tensors={:<8} {:>12} MiB {:>7}%",
            ty,
            tensors,
            mib(bytes),
            format!("{}.{:02}", pct_bp(bytes, total) / 100, pct_bp(bytes, total) % 100)
        );
        assert!(s.contains(&want), "missing row {want:?} in:\n{s}");
    }
    // 合計行。
    let want_total = format!(
        "  {:<10} tensors={:<8} {:>12} MiB {:>7}%",
        "total",
        MIXED.iter().map(|(_, t, _)| *t).sum::<u64>(),
        mib(total),
        "100.00"
    );
    assert!(s.contains(&want_total), "missing total row {want_total:?} in:\n{s}");
}

// ===== 5. 端: テンソル 0 個 / 単一型 =====

#[test]
fn no_tensors_yields_empty_breakdown() {
    let dir = TempDir::new("zero");
    let p = dir.write("empty.gguf", &gguf_bytes(&arch_kvs(), &[], 0));
    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);
    assert_eq!(v["tensor_count"], 0);
    assert_eq!(v["total_tensor_bytes"], 0);
    assert_eq!(
        v["quant_breakdown"].as_array().map(|a| a.len()),
        Some(0),
        "no tensors -> empty breakdown"
    );

    let text = run_inspect_path(&p, &[]);
    assert!(text.status.success(), "stderr: {}", stderr_of(&text));
    let s = stdout_of(&text);
    assert!(s.contains("quant breakdown:"), "{s}");
    // 合計行は 0 個 / 0 MiB / 0.00%。
    let want_total = format!(
        "  {:<10} tensors={:<8} {:>12} MiB {:>7}%",
        "total", 0, 0, "0.00"
    );
    assert!(s.contains(&want_total), "missing {want_total:?} in:\n{s}");
}

#[test]
fn single_type_is_a_hundred_percent() {
    let dir = TempDir::new("single");
    let tensors: Vec<TensorSpec> = vec![
        ("blk.0.attn_q.weight", vec![64, 2], T_Q8_0, 0),
        ("blk.1.attn_q.weight", vec![64, 2], T_Q8_0, 136),
    ];
    let p = dir.write("q8.gguf", &gguf_bytes(&arch_kvs(), &tensors, 0));
    let out = run_inspect_path(&p, &["--json"]);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let v = json_of(&out);
    let arr = v["quant_breakdown"].as_array().expect("breakdown");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["ty"].as_str(), Some("Q8_0"));
    assert_eq!(arr[0]["tensors"].as_u64(), Some(2));
    assert_eq!(arr[0]["pct_bp"].as_u64(), Some(10_000), "single type is 100%");
    assert_eq!(arr[0]["bytes"].as_u64(), Some(272));
}
