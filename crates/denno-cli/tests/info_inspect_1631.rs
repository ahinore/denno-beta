//! task 1631: `denno info` / `denno inspect` の GGUF メタデータ出力の監査。
//!
//! 合成の GGUF (v2 / v3、dense / MoE、分割 shard) を一時ディレクトリに書き、
//! 実バイナリ (`CARGO_BIN_EXE_denno`) を起動して次を固定する:
//!
//! 1. `inspect` は v2 と v3 で同じ数値を出し、`gguf_version` だけが違う。
//! 2. MoE のメタデータ (`<arch>.expert_count`) は `experts:` 行と
//!    `vram.expert_bytes` にだけ現れ、dense の分と混ざらない。
//! 3. `<arch>.` 接頭辞は `general.architecture` の値から組み立てる
//!    (arch 名の決め打ちが無い)。vocab はトークン表 → `<arch>.vocab_size` の順。
//! 4. 分割 GGUF は part ごとに独立して読める (メタデータだけの part は
//!    tensor 0 本で成功)。`info` は名前順の先頭 part を見る。
//! 5. 欠けたメタデータは `?` / JSON null。無いファイル / 壊れた入力 /
//!    config.json も GGUF も無いディレクトリは exit 1 (panic の 101 ではない)
//!    で stdout を空に保つ。
//! 6. テキスト出力の行ラベルは固定順で、JSON は既知キーだけを持つ
//!    (同じ値が両方に出る)。
//!
//! CPU だけで完結する (GPU / モデルの重みは触らない)。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const MIB: u64 = 1 << 20;

/// `inspect::mib` と同じ丸め (最近傍 MiB)。
fn mib(bytes: u64) -> u64 {
    bytes.saturating_add(MIB / 2) / MIB
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
            "denno1631_{}_{}_{}_{}",
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
/// `version` は 2 でも 3 でも同じレイアウト (v2 以降は件数が u64)。
fn gguf_bytes(
    version: u32,
    kvs: &[(&str, Kv)],
    tensors: &[TensorSpec],
    data_len: u64,
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
    b.extend(version.to_le_bytes());
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

fn tokens(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("t{i}")).collect()
}

/// MoE (qwen4exp) のアーキ情報: 4 層 / e=64 / expert 8 / ctx 1024 / vocab 5。
fn moe_kvs(arch: &str) -> Vec<(&'static str, Kv)> {
    // キーは `&'static str` なので、arch は静的な 2 種類に限る。
    assert!(arch == "qwen4exp" || arch == "llama");
    let arch: &'static str = if arch == "qwen4exp" { "qwen4exp" } else { "llama" };
    vec![
        ("general.architecture", Kv::Str(arch.to_string())),
        (key(arch, "block_count"), Kv::U32(4)),
        (key(arch, "embedding_length"), Kv::U32(64)),
        (key(arch, "expert_count"), Kv::U32(8)),
        (key(arch, "expert_used_count"), Kv::U32(2)),
        (key(arch, "context_length"), Kv::U32(1024)),
        (key(arch, "full_attention_interval"), Kv::U32(2)),
        (key(arch, "attention.head_count_kv"), Kv::U32(2)),
        (key(arch, "attention.key_length"), Kv::U32(32)),
        ("tokenizer.ggml.tokens", Kv::StrArr(tokens(5))),
    ]
}

fn key(arch: &'static str, suffix: &'static str) -> &'static str {
    match (arch, suffix) {
        ("qwen4exp", "block_count") => "qwen4exp.block_count",
        ("qwen4exp", "embedding_length") => "qwen4exp.embedding_length",
        ("qwen4exp", "expert_count") => "qwen4exp.expert_count",
        ("qwen4exp", "expert_used_count") => "qwen4exp.expert_used_count",
        ("qwen4exp", "context_length") => "qwen4exp.context_length",
        ("qwen4exp", "full_attention_interval") => "qwen4exp.full_attention_interval",
        ("qwen4exp", "attention.head_count_kv") => "qwen4exp.attention.head_count_kv",
        ("qwen4exp", "attention.key_length") => "qwen4exp.attention.key_length",
        ("qwen4exp", "vocab_size") => "qwen4exp.vocab_size",
        ("llama", "block_count") => "llama.block_count",
        ("llama", "embedding_length") => "llama.embedding_length",
        ("llama", "expert_count") => "llama.expert_count",
        ("llama", "expert_used_count") => "llama.expert_used_count",
        ("llama", "context_length") => "llama.context_length",
        ("llama", "full_attention_interval") => "llama.full_attention_interval",
        ("llama", "attention.head_count_kv") => "llama.attention.head_count_kv",
        ("llama", "attention.key_length") => "llama.attention.key_length",
        ("llama", "vocab_size") => "llama.vocab_size",
        _ => panic!("unhandled key {arch}.{suffix}"),
    }
}

/// dense 2 本 + expert 2 本。データ部 0 バイト (重みは読まれない)。
/// - `token_embd.weight` F32 [64,5] = 1,280 B (dense/host)
/// - `blk.0.attn_q.weight` Q8_0 [64,2] = 136 B (dense/dev)
/// - `blk.{0,1}.ffn_gate_exps.weight` Q8_0 [64,2,8] = 1,088 B ずつ (expert)
const TOTAL_TENSOR_BYTES: u64 = 1280 + 136 + 1088 + 1088;
const DENSE_BYTES: u64 = 1280 + 136;
const EXPERT_BYTES: u64 = 1088 * 2;
/// 2 full 層 * ctx 1024 * kv_dim 64 * 2 * 4 B
const KV_BYTES: u64 = 1_048_576;

fn moe_tensors() -> Vec<TensorSpec> {
    vec![
        ("token_embd.weight", vec![64, 5], T_F32, 0),
        ("blk.0.attn_q.weight", vec![64, 2], T_Q8_0, 1280),
        ("blk.0.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 1440),
        ("blk.1.ffn_gate_exps.weight", vec![64, 2, 8], T_Q8_0, 2528),
    ]
}

fn moe_fixture(version: u32, arch: &str) -> Vec<u8> {
    gguf_bytes(version, &moe_kvs(arch), &moe_tensors(), 0)
}

/// メタデータだけ (tensor 0 本、データ部 0 バイト) の分割 part。
fn shard_meta_only(version: u32, block_count: u32, vocab: usize) -> Vec<u8> {
    let kvs = vec![
        ("general.architecture", Kv::Str("qwen4exp".to_string())),
        ("qwen4exp.block_count", Kv::U32(block_count)),
        ("qwen4exp.embedding_length", Kv::U32(64)),
        ("qwen4exp.context_length", Kv::U32(1024)),
        ("tokenizer.ggml.tokens", Kv::StrArr(tokens(vocab))),
    ];
    gguf_bytes(version, &kvs, &[], 0)
}

// ===== プロセス起動 =====

fn denno(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_denno"))
        .args(args)
        .output()
        .expect("run denno")
}

fn run_inspect(path: &Path, extra: &[&str]) -> Output {
    let mut args: Vec<std::ffi::OsString> = vec![
        std::ffi::OsString::from("inspect"),
        path.as_os_str().to_os_string(),
    ];
    for e in extra {
        args.push(std::ffi::OsString::from(*e));
    }
    denno(&args.iter().map(|s| s.as_os_str()).collect::<Vec<_>>())
}

fn run_info(dir: &Path) -> Output {
    let mut args: Vec<std::ffi::OsString> = vec![std::ffi::OsString::from("info")];
    args.push(dir.as_os_str().to_os_string());
    denno(&args.iter().map(|s| s.as_os_str()).collect::<Vec<_>>())
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn json_of(out: &Output) -> serde_json::Value {
    serde_json::from_str(&stdout_of(out)).expect("stdout is a single JSON object")
}

fn exit_code(out: &Output) -> i32 {
    out.status
        .code()
        .unwrap_or_else(|| panic!("no exit code (killed by signal?)"))
}

/// 成功の共通契約: exit 0 / stderr に panic なし。
fn assert_ok(out: &Output, what: &str) {
    assert_eq!(exit_code(out), 0, "{what}: stderr:\n{}", stderr_of(out));
    assert!(
        !stderr_of(out).contains("panicked"),
        "{what}: stderr:\n{}",
        stderr_of(out)
    );
}

/// 失敗の共通契約: exit 1 / stdout 空 / stderr に理由 / panic なし。
fn assert_clean_failure(out: &Output, what: &str, needle: &str) {
    assert_eq!(
        exit_code(out),
        1,
        "{what}: CLI errors exit 1, stderr:\n{}",
        stderr_of(out)
    );
    assert!(
        out.stdout.is_empty(),
        "{what}: stdout must stay empty, got {:?}",
        stdout_of(out)
    );
    let e = stderr_of(out);
    assert!(e.contains(needle), "{what}: stderr should mention {needle:?}:\n{e}");
    assert!(!e.contains("panicked"), "{what}: must not panic:\n{e}");
    assert!(!e.contains("RUST_BACKTRACE"), "{what}: no backtrace:\n{e}");
}

/// 行のラベル (インデントされていない行だけ)。`?` 行はそのまま。
fn labels(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| !l.starts_with(' '))
        .map(|l| l.split(':').next().unwrap_or(l))
        .collect()
}

/// `label` で始まる行の値 (コロン以降を trim)。
fn value_of<'a>(text: &'a str, label: &str) -> Option<&'a str> {
    text.lines()
        .find(|l| l.starts_with(label))
        .map(|l| l[label.len()..].trim_start_matches(':').trim())
}

/// inspect の JSON 直下のキー (既知キーだけを持つことの監査)。
/// `serde_json::Value` のオブジェクトはキー順を保たないので集合として比べる
/// (テキスト側のキー順は Serialize の宣言順)。
const INSPECT_KEYS: [&str; 13] = [
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
    "quant_breakdown",
    "vram",
];

const VRAM_KEYS: [&str; 7] = [
    "dense_bytes",
    "expert_bytes",
    "kv_cache_bytes",
    "expert_cache_mib",
    "expert_cache_bytes",
    "full_gpu_bytes",
    "dense_plus_expert_cache_bytes",
];

fn assert_inspect_schema(v: &serde_json::Value) {
    let obj = v.as_object().expect("inspect JSON is an object");
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut want = INSPECT_KEYS;
    want.sort_unstable();
    assert_eq!(keys, want, "inspect top-level key set changed");
    let vram = v["vram"].as_object().expect("vram is an object");
    let mut vkeys: Vec<&str> = vram.keys().map(String::as_str).collect();
    vkeys.sort_unstable();
    let mut vwant = VRAM_KEYS;
    vwant.sort_unstable();
    assert_eq!(vkeys, vwant, "vram key set changed");
    for q in v["quant"].as_array().expect("quant array") {
        let qkeys: Vec<&str> = q.as_object().expect("quant entry").keys().map(String::as_str).collect();
        assert_eq!(qkeys, ["bytes", "mib", "tensors", "ty"], "quant entry keys");
    }
    for b in v["quant_breakdown"].as_array().expect("quant_breakdown array") {
        let bkeys: Vec<&str> = b
            .as_object()
            .expect("breakdown entry")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            bkeys,
            ["bytes", "mib", "pct_bp", "tensors", "ty"],
            "quant_breakdown entry keys"
        );
    }
}

// ===== 1. GGUF v2 と v3 は同じ数値 =====

#[test]
fn inspect_v2_matches_v3_numbers() {
    let dir = TempDir::new("v23");
    let p2 = dir.write("v2.gguf", &moe_fixture(2, "qwen4exp"));
    let p3 = dir.write("v3.gguf", &moe_fixture(3, "qwen4exp"));

    let out2 = run_inspect(&p2, &["--json"]);
    let out3 = run_inspect(&p3, &["--json"]);
    assert_ok(&out2, "inspect v2");
    assert_ok(&out3, "inspect v3");

    let mut v2 = json_of(&out2);
    let v3 = json_of(&out3);
    assert_eq!(v2["gguf_version"], 2);
    assert_eq!(v3["gguf_version"], 3);
    assert_inspect_schema(&v2);
    assert_inspect_schema(&v3);

    // path と version 以外は 1 バイトも違わない。
    v2["path"] = v3["path"].clone();
    v2["gguf_version"] = v3["gguf_version"].clone();
    assert_eq!(v2, v3, "v2 and v3 headers must analyze identically");

    // テキストも version 行だけが違う。
    let t2 = run_inspect(&p2, &[]);
    let t3 = run_inspect(&p3, &[]);
    assert_ok(&t2, "inspect v2 text");
    assert_ok(&t3, "inspect v3 text");
    assert_eq!(value_of(&stdout_of(&t2), "gguf version"), Some("2"));
    assert_eq!(value_of(&stdout_of(&t3), "gguf version"), Some("3"));
    // 先頭の path 行だけはファイル名で違うので落として比べる。
    let body = |s: &str| -> String {
        let mut lines = s.lines();
        lines.next();
        lines.collect::<Vec<_>>().join("\n")
    };
    let s2 = stdout_of(&t2).replace("gguf version: 2", "gguf version: X");
    let s3 = stdout_of(&t3).replace("gguf version: 3", "gguf version: X");
    assert_eq!(
        body(&s2),
        body(&s3),
        "v2 and v3 text output must differ only in the version line"
    );
}

// ===== 2. MoE メタデータと dense / expert の分離 =====

#[test]
fn inspect_moe_expert_metadata_is_reported_separately() {
    let dir = TempDir::new("moe");
    let p = dir.write("moe.gguf", &moe_fixture(3, "qwen4exp"));

    let out = run_inspect(&p, &["--json"]);
    assert_ok(&out, "inspect moe");
    let v = json_of(&out);
    assert_inspect_schema(&v);

    assert_eq!(v["architecture"], "qwen4exp");
    assert_eq!(v["layer_count"], 4);
    assert_eq!(v["hidden_size"], 64);
    assert_eq!(v["expert_count"], 8);
    assert_eq!(v["context_length"], 1024);
    assert_eq!(v["vocab_size"], 5);
    assert_eq!(v["tensor_count"], 4);
    assert_eq!(v["total_tensor_bytes"], TOTAL_TENSOR_BYTES);

    // dense と expert が混ざらない (expert は _exps の 2 本だけ)。
    assert_eq!(v["vram"]["dense_bytes"], DENSE_BYTES);
    assert_eq!(v["vram"]["expert_bytes"], EXPERT_BYTES);
    assert_eq!(v["vram"]["kv_cache_bytes"], KV_BYTES);
    assert_eq!(
        v["vram"]["full_gpu_bytes"],
        DENSE_BYTES + EXPERT_BYTES + KV_BYTES
    );
    // 既定予算 4096 MiB は expert 全量より大きいので頭打ち。
    assert_eq!(v["vram"]["expert_cache_mib"], 4096);
    assert_eq!(v["vram"]["expert_cache_bytes"], EXPERT_BYTES);

    // MoE のキーが dense の数値に漏れていない。
    let text = stdout_of(&run_inspect(&p, &[]));
    assert_eq!(value_of(&text, "experts"), Some("8"));
    assert_eq!(value_of(&text, "layers"), Some("4"));
    assert!(
        text.contains(&format!("  {:<26}{:>10} MiB", "dense", mib(DENSE_BYTES))),
        "dense row:\n{text}"
    );
    assert!(
        text.contains(&format!("  {:<26}{:>10} MiB", "experts (all)", mib(EXPERT_BYTES))),
        "experts row:\n{text}"
    );
}

// ===== 3. arch 接頭辞と vocab のフォールバック =====

#[test]
fn inspect_arch_prefix_follows_general_architecture() {
    let dir = TempDir::new("arch");
    // llama という別名でも同じ形のキーが読める (arch 名の決め打ちが無い)。
    let p = dir.write("llama.gguf", &moe_fixture(3, "llama"));
    let out = run_inspect(&p, &["--json"]);
    assert_ok(&out, "inspect llama-prefixed");
    let v = json_of(&out);
    assert_eq!(v["architecture"], "llama");
    assert_eq!(v["layer_count"], 4);
    assert_eq!(v["hidden_size"], 64);
    assert_eq!(v["expert_count"], 8);
    assert_eq!(v["context_length"], 1024);
    assert_eq!(v["vram"]["kv_cache_bytes"], KV_BYTES);

    // qwen4exp.* のキーは llama.architecture では読まれない (取り違えない)。
    let kvs = vec![
        ("general.architecture", Kv::Str("llama".to_string())),
        ("qwen4exp.block_count", Kv::U32(9)),
        ("llama.block_count", Kv::U32(3)),
        ("llama.vocab_size", Kv::U32(7)),
    ];
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![32, 4], T_F32, 0)];
    let p2 = dir.write("pref.gguf", &gguf_bytes(3, &kvs, &tensors, 0));
    let out2 = run_inspect(&p2, &["--json"]);
    assert_ok(&out2, "inspect prefix isolation");
    let v2 = json_of(&out2);
    assert_eq!(v2["layer_count"], 3, "llama.* wins, qwen4exp.* must not leak");
    assert_eq!(v2["hidden_size"], serde_json::Value::Null);
    // トークン表が無いので `<arch>.vocab_size` に落ちる。
    assert_eq!(v2["vocab_size"], 7);
    // kv の材料が無いので 0 (見積りを捏造しない)。
    assert_eq!(v2["vram"]["kv_cache_bytes"], 0);
}

// ===== 4. 分割 shard =====

#[test]
fn inspect_reads_each_split_part_independently() {
    let dir = TempDir::new("splitinspect");
    // part 1 はメタデータだけ (tensor 0 本)、part 2 はテンソルだけ。
    let p1 = dir.write(
        "model-00001-of-00002.gguf",
        &shard_meta_only(3, 4, 5),
    );
    let p2 = dir.write(
        "model-00002-of-00002.gguf",
        &gguf_bytes(3, &[], &moe_tensors(), 0),
    );

    let o1 = run_inspect(&p1, &["--json"]);
    assert_ok(&o1, "inspect split part 1");
    let v1 = json_of(&o1);
    assert_inspect_schema(&v1);
    assert_eq!(v1["layer_count"], 4);
    assert_eq!(v1["vocab_size"], 5);
    assert_eq!(v1["tensor_count"], 0);
    assert_eq!(v1["total_tensor_bytes"], 0);
    assert_eq!(v1["quant"].as_array().map(Vec::len), Some(0));
    assert_eq!(v1["quant_breakdown"].as_array().map(Vec::len), Some(0));

    // 空でもテキストの節は出る (合計は 0.00%)。
    let t1 = run_inspect(&p1, &[]);
    assert_ok(&t1, "inspect split part 1 text");
    let s1 = stdout_of(&t1);
    assert!(s1.contains("quantization:"), "{s1}");
    assert!(s1.contains("quant breakdown:"), "{s1}");
    assert!(s1.contains("0.00%"), "{s1}");

    let o2 = run_inspect(&p2, &["--json"]);
    assert_ok(&o2, "inspect split part 2");
    let v2 = json_of(&o2);
    assert_inspect_schema(&v2);
    // メタデータが無い part は null で、テンソルだけを数える。
    assert_eq!(v2["architecture"], serde_json::Value::Null);
    assert_eq!(v2["layer_count"], serde_json::Value::Null);
    assert_eq!(v2["expert_count"], serde_json::Value::Null);
    assert_eq!(v2["tensor_count"], 4);
    assert_eq!(v2["total_tensor_bytes"], TOTAL_TENSOR_BYTES);
}

#[test]
fn info_uses_name_sorted_first_shard() {
    let dir = TempDir::new("splitinfo");
    // 00002 を先に書き、名前順で 00001 が選ばれることを確かめる。
    dir.write(
        "model-00002-of-00002.gguf",
        &moe_fixture(3, "qwen4exp"),
    );
    let first = gguf_bytes(
        2,
        &[
            ("general.architecture", Kv::Str("qwen4exp".to_string())),
            ("qwen4exp.block_count", Kv::U32(6)),
            ("qwen4exp.embedding_length", Kv::U32(48)),
            ("qwen4exp.context_length", Kv::U32(2048)),
            ("tokenizer.ggml.tokens", Kv::StrArr(tokens(9))),
        ],
        &[
            ("blk.0.attn_q.weight", vec![48, 2], T_Q8_0, 0),
            ("blk.0.attn_norm.weight", vec![48], T_F32, 96),
        ],
        0,
    );
    dir.write("model-00001-of-00002.gguf", &first);

    let out = run_info(dir.as_path());
    assert_ok(&out, "info split dir");
    let text = stdout_of(&out);
    assert_eq!(value_of(&text, "layers:"), Some("6"));
    assert_eq!(value_of(&text, "hidden:"), Some("48"));
    assert_eq!(value_of(&text, "vocab:"), Some("9"));
    assert_eq!(value_of(&text, "context:"), Some("2048"));
    // quant は先頭 part のテンソルから (F32 は投票から外れる)。
    assert_eq!(value_of(&text, "quant:"), Some("Q8_0 (8)"));
    // config.json も safetensors も無いので weights 行は出ない。
    assert!(value_of(&text, "weights:").is_none(), "{text}");
}

// ===== 5. 欠けたメタデータ / 失敗経路 =====

#[test]
fn inspect_missing_metadata_is_question_mark_and_null() {
    let dir = TempDir::new("nometa");
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![32, 4], T_F32, 0)];
    let p = dir.write("bare.gguf", &gguf_bytes(2, &[], &tensors, 0));
    let out = run_inspect(&p, &["--json"]);
    assert_ok(&out, "inspect bare");
    let v = json_of(&out);
    assert_inspect_schema(&v);
    for k in [
        "architecture",
        "layer_count",
        "hidden_size",
        "expert_count",
        "context_length",
        "vocab_size",
    ] {
        assert_eq!(v[k], serde_json::Value::Null, "{k} must be null");
    }
    assert_eq!(v["gguf_version"], 2);
    assert_eq!(v["tensor_count"], 1);
    assert_eq!(v["vram"]["kv_cache_bytes"], 0);

    let text = stdout_of(&run_inspect(&p, &[]));
    for label in [
        "architecture: ?",
        "layers: ?",
        "hidden size: ?",
        "experts: ?",
        "context length: ?",
        "vocab size: ?",
    ] {
        assert!(text.contains(label), "missing {label:?}:\n{text}");
    }
}

#[test]
fn inspect_text_section_order_is_stable() {
    let dir = TempDir::new("order");
    let p = dir.write("m.gguf", &moe_fixture(3, "qwen4exp"));
    let text = stdout_of(&run_inspect(&p, &[]));
    assert_eq!(
        labels(&text),
        [
            "path",
            "gguf version",
            "architecture",
            "layers",
            "hidden size",
            "experts",
            "context length",
            "vocab size",
            "tensors",
            "quantization",
            "quant breakdown",
            "vram estimate",
        ],
        "text section order changed:\n{text}"
    );
    // 内訳の行 (割合つき) は型名の昇順、割合は 2 桁固定。
    let pct_rows: Vec<&str> = text.lines().filter(|l| l.contains('%')).collect();
    let f32_row = pct_rows
        .iter()
        .find(|l| l.trim_start().starts_with("F32"))
        .expect("F32 breakdown row");
    let q8_row = pct_rows
        .iter()
        .find(|l| l.trim_start().starts_with("Q8_0"))
        .expect("Q8_0 breakdown row");
    assert!(f32_row.contains("35.63%"), "{f32_row}");
    assert!(q8_row.contains("64.37%"), "{q8_row}");
    assert!(pct_rows[0].contains("F32"), "types must be sorted: {pct_rows:?}");
    assert!(text.contains("total"), "{text}");
}

#[test]
fn missing_and_broken_inputs_exit_one_with_empty_stdout() {
    let dir = TempDir::new("errors");
    let missing = dir.as_path().join("nope.gguf");
    assert_clean_failure(&run_inspect(&missing, &[]), "missing file", "nope.gguf");
    assert_clean_failure(
        &run_inspect(&missing, &["--json"]),
        "missing file (--json)",
        "nope.gguf",
    );

    let empty = dir.write("empty.gguf", &[]);
    assert_clean_failure(&run_inspect(&empty, &[]), "empty file", "EOF");

    let mut bad = moe_fixture(3, "qwen4exp");
    bad[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
    let pbad = dir.write("bad.gguf", &bad);
    assert_clean_failure(&run_inspect(&pbad, &[]), "bad magic", "not GGUF");

    // 未対応の version (v1 / v4) は理由つきで失敗する。
    for v in [1u32, 4u32] {
        let b = moe_fixture(v, "qwen4exp");
        let p = dir.write(&format!("v{v}.gguf"), &b);
        assert_clean_failure(
            &run_inspect(&p, &[]),
            &format!("gguf v{v}"),
            "未対応",
        );
    }

    // truncated: 24 バイトで切る (メタデータ先頭で途切れる)。
    let full = moe_fixture(3, "qwen4exp");
    let cut = dir.write("cut.gguf", &full[..24]);
    assert_clean_failure(&run_inspect(&cut, &[]), "truncated", "EOF");

    // ディレクトリは inspect の対象外。
    assert_clean_failure(
        &run_inspect(dir.as_path(), &[]),
        "directory",
        "ディレクトリ",
    );
}

// ===== 6. denno info =====

#[test]
fn info_reads_dims_from_v2_and_v3_gguf() {
    for version in [2u32, 3u32] {
        let dir = TempDir::new(&format!("infov{version}"));
        dir.write("model.gguf", &moe_fixture(version, "qwen4exp"));
        let out = run_info(dir.as_path());
        assert_ok(&out, &format!("info v{version}"));
        let text = stdout_of(&out);
        assert_eq!(value_of(&text, "family:"), Some("?"), "{text}");
        assert_eq!(value_of(&text, "layers:"), Some("4"), "{text}");
        assert_eq!(value_of(&text, "hidden:"), Some("64"), "{text}");
        assert_eq!(value_of(&text, "vocab:"), Some("5"), "{text}");
        assert_eq!(value_of(&text, "context:"), Some("1024"), "{text}");
        assert_eq!(value_of(&text, "quant:"), Some("Q8_0 (8)"), "{text}");
        // 14 行の設定表示 + quant 行。weights 行は無い。
        assert!(value_of(&text, "weights:").is_none(), "{text}");
        assert!(value_of(&text, "repo:").is_some(), "{text}");
        assert_eq!(labels(&text).len(), 16, "{text}");
    }
}

#[test]
fn info_config_json_wins_and_gguf_fills_the_gaps() {
    let dir = TempDir::new("merge");
    // config.json は層数と family だけ。hidden / vocab / context は GGUF から。
    dir.write(
        "config.json",
        br#"{"model_type":"qwen3","num_hidden_layers":12,
             "rope_scaling":{"rope_type":"yarn","factor":4.0},
             "quantization_config":{"quant_method":"gptq","bits":4}}"#,
    );
    dir.write("model.gguf", &moe_fixture(3, "qwen4exp"));

    let out = run_info(dir.as_path());
    assert_ok(&out, "info merge");
    let text = stdout_of(&out);
    assert_eq!(value_of(&text, "family:"), Some("qwen3"), "{text}");
    // config.json の値が勝つ (GGUF の block_count=4 で上書きしない)。
    assert_eq!(value_of(&text, "layers:"), Some("12"), "{text}");
    assert_eq!(value_of(&text, "hidden:"), Some("64"), "{text}");
    assert_eq!(value_of(&text, "vocab:"), Some("5"), "{text}");
    assert_eq!(value_of(&text, "context:"), Some("1024"), "{text}");
    assert_eq!(value_of(&text, "rope_scaling:"), Some("yarn factor=4"), "{text}");
    assert_eq!(value_of(&text, "quant_method:"), Some("gptq"), "{text}");
    assert_eq!(value_of(&text, "quant_bits:"), Some("4"), "{text}");
    // rope_theta は config に無いので `?` (既定値を捏造しない)。
    assert_eq!(value_of(&text, "rope_theta:"), Some("?"), "{text}");
}

#[test]
fn info_without_config_or_gguf_exits_one() {
    // 1. 存在しないリポジトリ。
    let dir = TempDir::new("infomissing");
    let missing = dir.as_path().join("not-a-model");
    assert_clean_failure(&run_info(&missing), "info missing repo", "not cached");

    // 2. 空のディレクトリ (config.json も GGUF も無い)。
    assert_clean_failure(&run_info(dir.as_path()), "info empty dir", "config.json");

    // 3. GGUF はあるが次元が 1 つも読めない (general.architecture が無い)。
    let dir2 = TempDir::new("infonodims");
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![8, 2], T_F32, 0)];
    dir2.write("model.gguf", &gguf_bytes(3, &[], &tensors, 0));
    assert_clean_failure(&run_info(dir2.as_path()), "info gguf without dims", "config.json");
}

#[test]
fn info_broken_config_falls_back_to_gguf_metadata() {
    // config.json が壊れていても GGUF から次元が取れれば info は成功し、
    // config 由来の行は `?` / `none` になる (既定値を捏造しない)。
    let dir = TempDir::new("badcfg");
    dir.write("config.json", b"{not json");
    dir.write("model.gguf", &moe_fixture(3, "qwen4exp"));
    let out = run_info(dir.as_path());
    assert_ok(&out, "info broken config with readable gguf");
    let text = stdout_of(&out);
    assert_eq!(value_of(&text, "family:"), Some("?"), "{text}");
    assert_eq!(value_of(&text, "layers:"), Some("4"), "{text}");
    assert_eq!(value_of(&text, "hidden:"), Some("64"), "{text}");
    assert_eq!(value_of(&text, "quant_method:"), Some("none"), "{text}");
    assert_eq!(value_of(&text, "quant_bits:"), Some("none"), "{text}");
    assert_eq!(value_of(&text, "rope_theta:"), Some("?"), "{text}");

    // GGUF からも次元が取れないときは config.json のエラーを返す。
    let dir2 = TempDir::new("badcfgempty");
    dir2.write("config.json", b"{not json");
    assert_clean_failure(
        &run_info(dir2.as_path()),
        "info broken config without gguf",
        "config.json",
    );
}

// ===== 7. JSON と text が同じ数値を出す =====

#[test]
fn inspect_json_and_text_agree_on_mib_rows() {
    let dir = TempDir::new("agree");
    let p = dir.write("m.gguf", &moe_fixture(3, "qwen4exp"));
    let out = run_inspect(&p, &["--json"]);
    assert_ok(&out, "inspect json");
    let v = json_of(&out);
    let text = stdout_of(&run_inspect(&p, &[]));
    let row = |label: &str, bytes: u64| format!("  {label:<26}{:>10} MiB", mib(bytes));
    for (label, key) in [
        ("kv cache", "kv_cache_bytes"),
        ("dense", "dense_bytes"),
        ("experts (all)", "expert_bytes"),
        ("full gpu", "full_gpu_bytes"),
        ("expert cache (4096 MiB)", "expert_cache_bytes"),
        ("dense + expert cache", "dense_plus_expert_cache_bytes"),
    ] {
        let bytes = v["vram"][key]
            .as_u64()
            .unwrap_or_else(|| panic!("vram.{key} is not a u64"));
        assert!(text.contains(&row(label, bytes)), "missing {label} row:\n{text}");
    }
    // tensors 行の MiB は total_tensor_bytes の丸めと一致する。
    assert_eq!(
        value_of(&text, "tensors"),
        Some(format!("4 ({} MiB)", mib(TOTAL_TENSOR_BYTES)).as_str())
    );
}
