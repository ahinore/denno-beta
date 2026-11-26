//! task 1344: `denno doctor` の JSON Schema 契約と複数シャード診断の境界試験。
//!
//! task 1029 の CLI 契約試験を補い、次を実バイナリで確かめる:
//!
//! 1. `--json` の主要セクション (cpu/memory/gpu/model/recommendations) の型契約。
//! 2. 3 分割が揃った合成ディレクトリは ok=true / 終了コード 0。
//! 3. 中間シャード欠け (00002-of-00003) は missing_shards に 00002 が出て非 0。
//! 4. 0 バイトのシャードは zero_byte_shards に出て非 0 (欠けとは取り違えない)。
//! 5. 範囲外の余分なシャード (00004-of-00003) は期待数を増やさず偽の欠けを出さない。
//! 6. `--dir` が通常ファイル / 存在しないパスのときは、レポートを出さずに
//!    非 0 + stderr の 1 行エラーで終わる (JSON モードでも同じ)。
//!
//! GPU を初期化しない読み取りだけの診断なので CPU-only ホストでも安全。
//! GPU 検出値はホスト依存なので、値ではなく「null か文字列」という型だけを見る。
//! 一時ディレクトリはテストごとに一意な名前で作り、Drop で消す。

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
            "denno1344_{}_{}_{}_{}",
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

    fn as_str(&self) -> &str {
        self.path.to_str().expect("temp dir path is UTF-8")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// ホストの DENNO_* に左右されないよう掃除して `denno` を起動する。
fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    for k in [
        "DENNO_FLASH_DIR",
        "DENNO_EXPERT_CACHE_MB",
        "DENNO_EXPERT_IQ_GPU",
        "DENNO_EXPERT_VRAM_MB",
        "DENNO_API_KEY",
    ] {
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

/// `null` か `u64` のどちらか (取れない値を 0 として捏造しない契約)。
fn assert_opt_u64(v: &serde_json::Value, path: &str) {
    assert!(
        v.is_null() || v.is_u64(),
        "{path} must be null or an integer, got {v}"
    );
}

/// `null` か文字列のどちらか。
fn assert_opt_string(v: &serde_json::Value, path: &str) {
    assert!(
        v.is_null() || v.is_string(),
        "{path} must be null or a string, got {v}"
    );
}

fn assert_string_array(v: &serde_json::Value, path: &str) {
    let a = v.as_array().unwrap_or_else(|| panic!("{path} must be an array, got {v}"));
    for (i, item) in a.iter().enumerate() {
        assert!(
            item.is_string(),
            "{path}[{i}] must be a string, got {item}"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. --json の Schema 契約
// ---------------------------------------------------------------------------

#[test]
fn json_schema_contract_for_core_sections() {
    // dir 無しでも全セクションが既定値で存在する (キーの欠落を許さない)。
    let o = denno(&["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    assert!(v.is_object(), "top level is not an object: {v}");

    for key in ["cpu", "memory", "gpu", "model", "recommendations"] {
        assert!(v.get(key).is_some(), "missing section `{key}` in {v}");
    }

    // cpu: 実在する値。arch は非空、論理コアは 1 以上、SIMD は文字列配列。
    let cpu = &v["cpu"];
    assert!(cpu.is_object(), "cpu is not an object: {cpu}");
    assert!(cpu["arch"].is_string(), "cpu.arch = {}", cpu["arch"]);
    assert!(
        !cpu["arch"].as_str().unwrap_or("").is_empty(),
        "cpu.arch is empty"
    );
    assert!(
        cpu["logical_cores"].as_u64().unwrap_or(0) >= 1,
        "cpu.logical_cores = {}",
        cpu["logical_cores"]
    );
    assert_opt_u64(&cpu["physical_cores"], "cpu.physical_cores");
    assert_string_array(&cpu["simd"], "cpu.simd");

    // memory: 取れなければ null。負値や文字列は契約違反。
    assert_opt_u64(&v["memory"]["total_bytes"], "memory.total_bytes");
    assert_opt_u64(&v["memory"]["available_bytes"], "memory.available_bytes");

    // gpu: ホスト依存の検出値は null か文字列だけを要求する
    // (CPU-only / NVIDIA のどちらでも通る)。
    let gpu = &v["gpu"];
    assert!(gpu.is_object(), "gpu is not an object: {gpu}");
    for key in [
        "cuda_driver",
        "nvidia_kernel_driver",
        "level_zero_loader",
        "level_zero_intel_driver",
    ] {
        assert_opt_string(&gpu[key], &format!("gpu.{key}"));
    }
    assert_string_array(&gpu["gpu_devices"], "gpu.gpu_devices");

    // model: dir 未指定なので null、配列は空、期待数も null。
    let model = &v["model"];
    assert!(model.is_object(), "model is not an object: {model}");
    assert!(model["dir"].is_null(), "model.dir = {}", model["dir"]);
    assert_string_array(&model["shards"], "model.shards");
    assert_eq!(model["shards"].as_array().map(Vec::len), Some(0));
    assert!(model["expected_shards"].is_null());
    assert_string_array(&model["missing_shards"], "model.missing_shards");
    assert_string_array(
        &model["zero_byte_shards"],
        "model.zero_byte_shards",
    );
    assert_eq!(model["total_bytes"].as_u64(), Some(0));
    assert_eq!(model["ok"], serde_json::Value::Bool(false));

    // recommendations: 非空の文字列配列。
    let recs = v["recommendations"]
        .as_array()
        .unwrap_or_else(|| panic!("recommendations must be an array: {v}"));
    assert!(!recs.is_empty(), "recommendations empty: {v}");
    assert!(
        recs.iter().all(|s| s.is_string()),
        "recommendations contains a non-string: {recs:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. 3 分割が揃ったディレクトリは ok=true / 終了 0
// ---------------------------------------------------------------------------

#[test]
fn complete_three_shard_set_is_ok() {
    let td = TempDir::new("complete3");
    td.write("model-00001-of-00003.gguf", b"first");
    td.write("model-00002-of-00003.gguf", b"second");
    td.write("model-00003-of-00003.gguf", b"third");
    // .gguf 以外は数えない
    td.write("notes.txt", b"ignored");

    let o = denno(&["doctor", "--dir", td.as_str()]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("[モデル]"), "no model section:\n{t}");
    assert!(t.contains("3/3 本"), "unexpected shard count:\n{t}");
    assert!(t.contains("状態            OK"), "status is not OK:\n{t}");
    assert!(!t.contains("欠け"), "false missing-shard warning:\n{t}");
    assert!(!t.contains("0 バイト"), "false zero-byte warning:\n{t}");

    let j = denno(&["doctor", "--dir", td.as_str(), "--json"]);
    assert_eq!(j.status.code(), Some(0), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(3));
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(3));
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(
        v["model"]["zero_byte_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
    assert_eq!(
        v["model"]["total_bytes"].as_u64(),
        Some((b"first".len() + b"second".len() + b"third".len()) as u64)
    );
    // 各シャードは name/bytes を持つ
    for s in v["model"]["shards"].as_array().unwrap() {
        assert!(s["name"].is_string(), "shard.name = {}", s["name"]);
        assert!(s["bytes"].is_u64(), "shard.bytes = {}", s["bytes"]);
    }
}

// ---------------------------------------------------------------------------
// 3. 中間シャード欠け (00002-of-00003) は missing_shards=[00002] / 非 0
// ---------------------------------------------------------------------------

#[test]
fn missing_middle_shard_reports_and_exits_nonzero() {
    let td = TempDir::new("missingmid");
    td.write("model-00001-of-00003.gguf", b"aaa");
    td.write("model-00003-of-00003.gguf", b"ccc");
    let dir = td.as_str();

    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("[モデル]"), "no model section:\n{t}");
    assert!(t.contains("2/3 本"), "unexpected shard count:\n{t}");
    assert!(t.contains("00002"), "no missing shard in text:\n{t}");
    assert!(t.contains("状態            NG"), "status is not NG:\n{t}");
    // 中間の 1 本だけが欠け。00001/00003 を欠け扱いしない。
    assert!(
        !t.contains("00001,") && !t.contains("00003,"),
        "wrong missing list:\n{t}"
    );
    assert!(
        stderr(&o).contains("Error:"),
        "no error line on stderr:\n{}",
        stderr(&o)
    );

    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(1), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(3));
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(2));
    assert_eq!(v["model"]["missing_shards"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["model"]["missing_shards"][0], "00002");
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(false));
    assert_eq!(
        v["model"]["zero_byte_shards"].as_array().map(Vec::len),
        Some(0)
    );
    // NG の理由が recommendations に載る
    assert!(
        v["recommendations"].as_array().map_or(false, |a| a
            .iter()
            .any(|s| s.as_str().unwrap_or("").contains("00002"))),
        "no recommendation mentioning the missing shard: {v}"
    );
}

// ---------------------------------------------------------------------------
// 4. 0 バイトのシャードは zero_byte_shards に出て非 0 (欠けとは別)
// ---------------------------------------------------------------------------

#[test]
fn zero_byte_middle_shard_reports_and_exits_nonzero() {
    let td = TempDir::new("zero3");
    td.write("model-00001-of-00003.gguf", b"head");
    td.write("model-00002-of-00003.gguf", b"");
    td.write("model-00003-of-00003.gguf", b"tail");
    let dir = td.as_str();

    let o = denno(&["doctor", "--dir", dir]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("[モデル]"), "no model section:\n{t}");
    assert!(t.contains("3/3 本"), "unexpected shard count:\n{t}");
    assert!(
        t.contains("0 バイト") && t.contains("model-00002-of-00003.gguf"),
        "no zero-byte entry in text:\n{t}"
    );
    assert!(t.contains("状態            NG"), "status is not NG:\n{t}");
    // 0 バイトと欠けを取り違えない
    assert!(
        !t.lines().any(|l| l.trim_start().starts_with("欠け")),
        "false missing-shard line:\n{t}"
    );

    let j = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(j.status.code(), Some(1), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(3));
    assert_eq!(
        v["model"]["zero_byte_shards"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(v["model"]["zero_byte_shards"][0], "model-00002-of-00003.gguf");
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(false));
    assert_eq!(
        v["model"]["total_bytes"].as_u64(),
        Some((b"head".len() + b"tail".len()) as u64)
    );
}

// ---------------------------------------------------------------------------
// 5. 範囲外の余分なシャードは期待数を増やさず、偽の欠けを出さない
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_extra_shard_is_ignored() {
    let td = TempDir::new("extra");
    td.write("model-00001-of-00003.gguf", b"a");
    td.write("model-00002-of-00003.gguf", b"b");
    td.write("model-00003-of-00003.gguf", b"c");
    // idx > total は分割命名として不正 = 期待数に数えない
    td.write("model-00004-of-00003.gguf", b"stray");

    let o = denno(&["doctor", "--dir", td.as_str()]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("状態            OK"), "status is not OK:\n{t}");
    assert!(!t.contains("欠け"), "false missing-shard warning:\n{t}");

    let j = denno(&["doctor", "--dir", td.as_str(), "--json"]);
    assert_eq!(j.status.code(), Some(0), "stderr:\n{}", stderr(&j));
    let v = parse_json(&j);
    assert_eq!(v["model"]["expected_shards"].as_u64(), Some(3));
    // .gguf は 4 本すべて列挙されるが、期待数は 3 のまま。
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(4));
    assert_eq!(
        v["model"]["missing_shards"].as_array().map(Vec::len),
        Some(0)
    );
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
}

// ---------------------------------------------------------------------------
// 6. --dir が通常ファイル / 存在しないパス -> レポート無しで非 0
// ---------------------------------------------------------------------------

#[test]
fn dir_pointing_to_regular_file_fails_cleanly() {
    let td = TempDir::new("filepath");
    let file = td.write("model-00001-of-00001.gguf", b"not-a-dir");
    let dir = file.to_str().expect("file path is UTF-8");

    for extra in [&[][..], &["--json"][..]] {
        let mut args = vec!["doctor", "--dir", dir];
        args.extend_from_slice(extra);
        let o = denno(&args);
        assert_eq!(
            o.status.code(),
            Some(1),
            "args {args:?} stderr:\n{}",
            stderr(&o)
        );
        assert!(
            stdout(&o).trim().is_empty(),
            "report must not be printed for a file path (args {args:?}):\n{}",
            stdout(&o)
        );
        assert!(
            stderr(&o).contains("Error:"),
            "no error line on stderr (args {args:?}):\n{}",
            stderr(&o)
        );
    }
}

#[test]
fn nonexistent_dir_fails_cleanly() {
    let td = TempDir::new("nodir");
    let missing = td.as_path().join("does-not-exist");
    let dir = missing.to_str().expect("path is UTF-8");

    for extra in [&[][..], &["--json"][..]] {
        let mut args = vec!["doctor", "--dir", dir];
        args.extend_from_slice(extra);
        let o = denno(&args);
        assert_eq!(
            o.status.code(),
            Some(1),
            "args {args:?} stderr:\n{}",
            stderr(&o)
        );
        assert!(
            stdout(&o).trim().is_empty(),
            "report must not be printed for a missing dir (args {args:?}):\n{}",
            stdout(&o)
        );
        assert!(
            stderr(&o).contains("Error:"),
            "no error line on stderr (args {args:?}):\n{}",
            stderr(&o)
        );
    }
}
