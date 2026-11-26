//! task 1046: `denno doctor` の拡張検査 (ディスク空き / ファイル記述子の上限 /
//! `DENNO_*` env 監査) の CLI 契約。
//!
//! `src/doctor.rs` の単体試験は純関数と検出関数を直接叩く。ここでは実際の
//! バイナリ (`CARGO_BIN_EXE_denno`) を起動して:
//!
//! 1. `--json` に `disk` / `fd_limits` / `env` セクションが出て型が正しい。
//! 2. ディスクの検査対象は `--dir`、未指定ならカレント。
//! 3. `DENNO_API_KEY` / `DENNO_API_KEYS` の値が stdout のどこにも出ない
//!    (JSON でもテキストでも)。伏せたことは `redacted` で分かる。
//! 4. `ulimit -n` を下げた子プロセスでは soft 上限がその値になり、
//!    推奨行に「ファイル記述子」が出る (= 値が実測である証拠)。
//!
//! バイナリは `statvfs` / `getrlimit` / env を読むだけで、GPU も
//! 子プロセスも起動しない。一時ディレクトリはテストごとに一意で消す。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

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
            "denno1046_{}_{}_{}_{}",
            tag,
            std::process::id(),
            n,
            nanos
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
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

/// ホストの `DENNO_*` を落として `denno` を起動する (継承 env に左右されない)。
fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    for k in [
        "DENNO_FLASH_DIR",
        "DENNO_EXPERT_CACHE_MB",
        "DENNO_EXPERT_IQ_GPU",
        "DENNO_API_KEY",
        "DENNO_API_KEYS",
        "DENNO_MAX_CTX",
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

// ---------------------------------------------------------------------------
// 1. JSON: disk / fd_limits / env の形
// ---------------------------------------------------------------------------

#[test]
fn json_reports_disk_fd_limits_and_env() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(["doctor", "--json"]);
    cmd.env("DENNO_MAX_CTX", "1234");
    cmd.env("DENNO_API_KEY", "sk-SECRET-1046-a");
    cmd.env("DENNO_API_KEYS", "sk-SECRET-1046-b,sk-SECRET-1046-c");
    cmd.env("PATH", std::env::var("PATH").unwrap_or_default());
    let o = cmd.output().expect("spawn denno");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));

    let raw = stdout(&o);
    // 秘密の値が生で出ていない
    for leaked in ["sk-SECRET-1046-a", "sk-SECRET-1046-b", "sk-SECRET-1046-c"] {
        assert!(!raw.contains(leaked), "secret leaked into stdout: {leaked}\n{raw}");
        assert!(!stderr(&o).contains(leaked), "secret leaked into stderr");
    }

    let v = parse_json(&o);

    // disk: パスは文字列、容量は数値 (このホストでは statvfs が通る)
    assert!(v["disk"]["path"].is_string(), "{}", v["disk"]);
    let total = v["disk"]["total_bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("disk.total_bytes not u64: {}", v["disk"]));
    let avail = v["disk"]["available_bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("disk.available_bytes not u64: {}", v["disk"]));
    assert!(total > 0);
    assert!(avail <= total, "avail {avail} > total {total}");

    // fd_limits: soft/hard は u64 か null、無制限フラグは bool
    for key in ["soft", "hard"] {
        let x = &v["fd_limits"][key];
        assert!(x.is_u64() || x.is_null(), "fd_limits.{key} = {x}");
    }
    for key in ["soft_unlimited", "hard_unlimited"] {
        assert!(
            v["fd_limits"][key].is_boolean(),
            "fd_limits.{key} = {}",
            v["fd_limits"][key]
        );
    }
    if let (Some(s), Some(h)) = (
        v["fd_limits"]["soft"].as_u64(),
        v["fd_limits"]["hard"].as_u64(),
    ) {
        assert!(s <= h, "soft {s} > hard {h}");
    }

    // env: DENNO_* だけ、key 昇順、秘密は伏せる
    let entries = v["env"]["entries"].as_array().expect("env.entries array");
    let keys: Vec<&str> = entries
        .iter()
        .map(|e| e["key"].as_str().expect("key is string"))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(keys, sorted, "env entries not sorted: {keys:?}");
    for k in &keys {
        assert!(k.starts_with("DENNO_"), "non-DENNO key leaked: {k}");
    }

    let find = |name: &str| {
        entries
            .iter()
            .find(|e| e["key"] == name)
            .unwrap_or_else(|| panic!("{name} missing from env entries: {v}"))
    };
    let max_ctx = find("DENNO_MAX_CTX");
    assert_eq!(max_ctx["value"], "1234");
    assert_eq!(max_ctx["redacted"], false);
    for name in ["DENNO_API_KEY", "DENNO_API_KEYS"] {
        let e = find(name);
        assert_eq!(e["value"], "<redacted>", "{name} value: {}", e["value"]);
        assert_eq!(e["redacted"], true, "{name} not marked redacted");
    }
    let red: Vec<&str> = v["env"]["redacted_keys"]
        .as_array()
        .expect("redacted_keys array")
        .iter()
        .map(|x| x.as_str().expect("string"))
        .collect();
    assert!(red.contains(&"DENNO_API_KEY"), "{red:?}");
    assert!(red.contains(&"DENNO_API_KEYS"), "{red:?}");
    assert!(!red.contains(&"DENNO_MAX_CTX"), "{red:?}");
}

// ---------------------------------------------------------------------------
// 2. ディスクの検査対象: --dir があればそのパス、無ければカレント
// ---------------------------------------------------------------------------

#[test]
fn disk_check_targets_dir_flag_or_cwd() {
    let td = TempDir::new("disk");
    // 空ディレクトリはモデル検査が NG になるので、単一 GGUF を置く。
    std::fs::write(td.as_path().join("solo.gguf"), b"gguf").expect("write gguf");
    let dir = td.as_path().to_str().unwrap();

    let o = denno(&["doctor", "--dir", dir, "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    // tmp と作業ディレクトリは同じ FS とは限らないので、パス一致だけを見る
    assert_eq!(v["disk"]["path"], dir);
    assert!(v["disk"]["total_bytes"].as_u64().unwrap_or(0) > 0, "{v}");

    let o2 = denno(&["doctor", "--json"]);
    assert_eq!(o2.status.code(), Some(0), "stderr:\n{}", stderr(&o2));
    let v2 = parse_json(&o2);
    let cwd = std::env::current_dir().expect("cwd");
    assert_eq!(
        v2["disk"]["path"], cwd.display().to_string(),
        "--dir 省略時はカレントを見る"
    );
}

// ---------------------------------------------------------------------------
// 3. テキスト出力: 新セクションと伏せ字
// ---------------------------------------------------------------------------

#[test]
fn text_mode_shows_new_sections_and_redacts() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(["doctor"]);
    cmd.env("DENNO_API_KEY", "sk-TEXT-SECRET-1046");
    cmd.env("DENNO_MAX_CTX", "777");
    cmd.env_remove("DENNO_API_KEYS");
    let o = cmd.output().expect("spawn denno");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);

    for sec in [
        "[CPU]",
        "[メモリ]",
        "[ディスク]",
        "[ファイル記述子]",
        "[GPU]",
        "[モデル]",
        "[環境変数 (DENNO_*)]",
        "[推奨]",
    ] {
        assert!(t.contains(sec), "missing {sec} in:\n{t}");
    }
    assert!(t.contains("DENNO_MAX_CTX=777"), "env not listed:\n{t}");
    assert!(t.contains("DENNO_API_KEY=<redacted>"), "not redacted:\n{t}");
    assert!(!t.contains("sk-TEXT-SECRET-1046"), "secret leaked:\n{t}");
}

// ---------------------------------------------------------------------------
// 4. `ulimit -n` を下げると soft 上限に反映される (実測であることの証明)
// ---------------------------------------------------------------------------

#[test]
fn lowered_ulimit_is_reflected_in_report() {
    let bin = env!("CARGO_BIN_EXE_denno");
    // shell の中で soft 上限を 512 にしてから exec する (子プロセスは 1 本)。
    let script = format!("ulimit -n 512; exec '{}' doctor --json", bin);
    let o = Command::new("sh")
        .arg("-c")
        .arg(script)
        .env_remove("DENNO_API_KEY")
        .env_remove("DENNO_API_KEYS")
        .output()
        .expect("spawn sh");
    if o.status.code() != Some(0) {
        // 環境が 512 まで下げられない (hard が極端に小さい等) なら判定不能。
        // 偽の合格は出さず、理由付きで落とす。
        panic!(
            "cannot lower ulimit -n in this environment: status {:?}\nstderr:\n{}",
            o.status.code(),
            stderr(&o)
        );
    }
    let v = parse_json(&o);
    assert_eq!(
        v["fd_limits"]["soft"].as_u64(),
        Some(512),
        "soft limit not reflected: {}",
        v["fd_limits"]
    );
    assert_eq!(v["fd_limits"]["soft_unlimited"], false);
    let recs = v["recommendations"].as_array().expect("recommendations");
    assert!(
        recs.iter()
            .any(|s| s.as_str().unwrap_or("").contains("ファイル記述子")),
        "no fd recommendation: {recs:?}"
    );
}
