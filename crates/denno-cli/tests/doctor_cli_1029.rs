//! task 1029: `denno doctor` の CLI 契約 (テキスト / `--json` / 終了コード) の試験。
//!
//! `src/doctor.rs` の単体試験は `check_shards` と整形関数を直接叩くが、ここでは
//! 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動して次を確かめる:
//!
//! 1. テキスト出力に [CPU]/[メモリ]/[GPU]/[モデル]/[推奨] が出る。
//! 2. `--json` は stdout が単体で解析できる JSON オブジェクトで、主要キーが揃う。
//! 3. 欠けの無い分割ディレクトリは終了コード 0。
//! 4. シャード欠け / 0 バイトは「レポートを先に出してから」非 0 で終わる。
//! 5. 分割命名でない単一 GGUF は偽の欠け警告を出さない。
//!
//! バイナリはハードウェアを読むだけで GPU を初期化しないので、この環境で安全。
//! 一時ディレクトリはテストごとに一意な名前で作り、終了時に消す。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ。Drop で再帰削除する。
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
            "denno1029_{}_{}_{}_{}",
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

/// 環境変数を掃除して `denno` を起動する (ホストの DENNO_*/LD_LIBRARY_PATH に
/// 左右されないようにする)。
fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    for k in ["DENNO_FLASH_DIR", "DENNO_EXPERT_CACHE_MB", "DENNO_EXPERT_IQ_GPU"] {
        cmd.env_remove(k);
    }
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn parse_json(o: &Output) -> serde_json::Value {
    let s = stdout(o);
    serde_json::from_str(&s).unwrap_or_else(|e| {
        panic!(
            "stdout is not a single JSON document: {e}\n--- stdout ---\n{s}\n--- stderr ---\n{}",
            stderr(o)
        )
    })
}

// ---------------------------------------------------------------------------
// 1. テキスト出力の骨格 (dir 無し)
// ---------------------------------------------------------------------------

#[test]
fn text_mode_has_all_sections_without_dir() {
    let o = denno(&["doctor"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    for sec in ["[CPU]", "[メモリ]", "[GPU]", "[モデル]", "[推奨]"] {
        assert!(t.contains(sec), "missing {sec} in:\n{t}");
    }
    // dir 未指定ならモデル行は「未指定」で、NG にも失敗にもならない
    assert!(
        t.contains("(未指定: --dir で検査する)"),
        "unexpected model line in:\n{t}"
    );
    assert!(!t.contains("状態            NG"), "unexpected NG in:\n{t}");
    // 推奨行は実在するサブコマンドだけを案内する
    assert!(t.contains("denno serve-flash --dir"), "no recommendation:\n{t}");
}

// ---------------------------------------------------------------------------
// 2. --json の契約 (dir 無しでも JSON オブジェクト)
// ---------------------------------------------------------------------------

#[test]
fn json_mode_without_dir_is_structured_object() {
    let o = denno(&["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);

    assert!(v.is_object(), "top level is not an object: {v}");
    for key in ["cpu", "memory", "gpu", "model", "recommendations"] {
        assert!(v.get(key).is_some(), "missing section `{key}` in {v}");
    }

    // cpu
    assert!(v["cpu"]["arch"].is_string());
    assert!(!v["cpu"]["arch"].as_str().unwrap().is_empty());
    assert!(v["cpu"]["logical_cores"].as_u64().unwrap() >= 1);
    assert!(v["cpu"]["simd"].is_array());

    // memory: null か数値のどちらか (取れないものを捏造しない)
    for key in ["total_bytes", "available_bytes"] {
        let m = &v["memory"][key];
        assert!(m.is_null() || m.is_u64(), "memory.{key} = {m}");
    }

    // gpu: 検出値は null か文字列、gpu_devices は配列
    for key in [
        "cuda_driver",
        "nvidia_kernel_driver",
        "level_zero_loader",
        "level_zero_intel_driver",
    ] {
        let g = &v["gpu"][key];
        assert!(g.is_null() || g.is_string(), "gpu.{key} = {g}");
    }
    assert!(v["gpu"]["gpu_devices"].is_array());

    // model: dir 未指定なので null、shards は空配列、期待数も null
    assert!(v["model"]["dir"].is_null(), "model.dir = {}", v["model"]["dir"]);
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(0));
    assert!(v["model"]["expected_shards"].is_null());
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(
        v["model"]["zero_byte_shards"].as_array().map(Vec::len),
        Some(0)
    );

    assert!(
        v["recommendations"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "recommendations empty: {v}"
    );
}

// ---------------------------------------------------------------------------
// 3. 完全な分割 GGUF は終了コード 0
// ---------------------------------------------------------------------------

#[test]
fn complete_split_dir_exits_zero() {
    let td = TempDir::new("complete");
    td.write("model-00001-of-00002.gguf", b"GGUF-shard-1");
    td.write("model-00002-of-00002.gguf", b"GGUF-shard-2");
    let dir = td.as_path().to_str().unwrap();

    // テキスト
    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("2/2 本"), "unexpected:\n{t}");
    assert!(t.contains("OK"), "unexpected:\n{t}");
    assert!(!t.contains("欠け"), "false missing-shard warning:\n{t}");
    assert!(!t.contains("0 バイト"), "false zero-byte warning:\n{t}");

    // JSON
    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(0), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(2));
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(
        v["model"]["zero_byte_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
    let total: u64 = v["model"]["shards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["bytes"].as_u64().unwrap())
        .sum();
    assert_eq!(v["model"]["total_bytes"].as_u64(), Some(total));
    assert_eq!(total, (b"GGUF-shard-1".len() + b"GGUF-shard-2".len()) as u64);
}

// ---------------------------------------------------------------------------
// 4a. シャード欠け: レポートを先に出してから非 0
// ---------------------------------------------------------------------------

#[test]
fn missing_shard_reports_then_exits_nonzero() {
    let td = TempDir::new("missing");
    td.write("model-00001-of-00002.gguf", b"only-one");
    let dir = td.as_path().to_str().unwrap();

    // テキスト: 本文に欠けが出てから終了コード 1
    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("[モデル]"), "no model section:\n{t}");
    assert!(t.contains("00002"), "no missing list:\n{t}");
    assert!(t.contains("NG"), "wrong status:\n{t}");
    // 失敗理由は stderr に 1 行のエラーとして残る
    assert!(
        stderr(&o).contains("Error:"),
        "no error line on stderr:\n{}",
        stderr(&o)
    );

    // JSON: 失敗しても stdout は解析可能な JSON のまま
    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(1), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(2));
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["model"]["missing_shards"][0], "00002");
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(false));
    assert!(
        v["recommendations"]
            .as_array()
            .map(|a| a.iter().any(|s| s.as_str().unwrap_or("").contains("シャード")))
            .unwrap_or(false),
        "no shard recommendation: {v}"
    );
}

// ---------------------------------------------------------------------------
// 4b. 0 バイトのシャード: 同じくレポートを先に出してから非 0
// ---------------------------------------------------------------------------

#[test]
fn zero_byte_shard_reports_then_exits_nonzero() {
    let td = TempDir::new("zerobyte");
    td.write("model-00001-of-00002.gguf", b"");
    td.write("model-00002-of-00002.gguf", b"tail");
    let dir = td.as_path().to_str().unwrap();

    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("[モデル]"), "no model section:\n{t}");
    assert!(
        t.contains("0 バイト") && t.contains("model-00001-of-00002.gguf"),
        "no zero-byte list:\n{t}"
    );
    assert!(t.contains("NG"), "wrong status:\n{t}");
    // 0 バイトのみなので欠け行は出ない (両者を取り違えない)
    assert!(!t.lines().any(|l| l.trim_start().starts_with("欠け")), "false missing line:\n{t}");

    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(1), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["zero_byte_shards"][0], "model-00001-of-00002.gguf");
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(false));
    assert_eq!(v["model"]["total_bytes"].as_u64(), Some(4));
}

// ---------------------------------------------------------------------------
// 5. 分割命名でない単一 GGUF は偽の欠け警告を出さず終了 0
// ---------------------------------------------------------------------------

#[test]
fn single_unsplit_gguf_is_accepted_cleanly() {
    let td = TempDir::new("unsplit");
    td.write("solo.gguf", b"GGUF-single-model-bytes");
    // .gguf 以外は無視される
    td.write("README.txt", b"not a model");
    let dir = td.as_path().to_str().unwrap();

    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("1/? 本"), "unexpected:\n{t}");
    assert!(t.contains("状態            OK"), "unexpected:\n{t}");
    assert!(!t.contains("欠け"), "false missing-shard warning:\n{t}");

    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(0), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    // 期待数は unknown (null) で、欠け判定をしない
    assert!(v["model"]["expected_shards"].is_null());
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["model"]["shards"][0]["name"], "solo.gguf");
    assert_eq!(
        v["model"]["total_bytes"].as_u64(),
        Some(b"GGUF-single-model-bytes".len() as u64)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
}

// ---------------------------------------------------------------------------
// 6. 読めない dir はレポートを出さずに非 0 (別種の失敗として区別する)
// ---------------------------------------------------------------------------

#[test]
fn unreadable_dir_fails_without_report() {
    let td = TempDir::new("nodir");
    let missing = td.as_path().join("does-not-exist");
    let dir = missing.to_str().unwrap();

    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    assert!(
        stdout(&o).is_empty(),
        "report should not be printed when dir is unreadable:\n{}",
        stdout(&o)
    );
    assert!(
        stderr(&o).contains("Error:"),
        "no error line on stderr:\n{}",
        stderr(&o)
    );

    // --json でも同様 (JSON を騙って部分的に出すことはしない)
    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(1), "stderr:\n{}", stderr(&j));
    assert!(
        stdout(&j).trim().is_empty(),
        "unexpected stdout:\n{}",
        stdout(&j)
    );
}
