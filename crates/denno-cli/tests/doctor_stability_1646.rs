//! task 1646: `denno doctor` のクロスプラットフォーム契約監査。
//!
//! task 1029 / 1046 / 1344 / 1394 が個別の機能を押さえているのに対し、ここは
//! 「自動化が依存してよい不変条件」を 1 か所に固定する:
//!
//! 1. JSON Schema の全セクション (cpu / memory / gpu / model / disk /
//!    fd_limits / env) のキー集合とキー順。キーの追加・改名は破壊的変更なので、
//!    集合一致で落とす (意図した変更ならこのテストを同じ commit で更新する)。
//! 2. テキスト出力のセクション集合と順序。`{:?}` の残骸 (Some(...) / Ok(...))
//!    のようなデバッグ出力が混ざっていないこと。
//! 3. 終了コードの契約: 0 = 正常, 1 = モデル検査 NG または `--dir` 不正,
//!    2 = clap の使い方エラー。text / --json で同じ値になること。
//! 4. 推奨 (recommendations) の決定性と GPU 分岐の排他性。GPU 検出の有無で
//!    出るのは 1 系統だけ (GPU (CUDA) / NVIDIA カーネルのみ / CPU のみ)。
//! 5. 権限の制限 (読めないディレクトリ) と、ツールが無い環境 (PATH 無し /
//!    空 env) で panic しないこと。stderr に "panicked" を出さないこと。
//! 6. 変な `DENNO_*` (改行・非 UTF-8・巨大値) と非 UTF-8 のシャード名で
//!    panic せず、JSON が単一文書のままであること。
//! 7. 出力にホスト識別子 (ホスト名・`/home/`・メール風の `@`・私設 IP) が
//!    混ざらないこと。cwd を一時ディレクトリにして、`--dir` 無しで検査する。
//!
//! GPU は初期化しない (doctor はライブラリのファイル存在しか見ない) ので、
//! CPU-only のランナーでも安全。検出値そのものはホスト依存なので、値ではなく
//! 「null かどうか」と分岐の整合だけを見る。

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

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
            "denno1646_{}_{}_{}_{}",
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
        // 権限を落としたディレクトリでも消せるように戻してから消す。
        let _ = std::fs::set_permissions(
            &self.path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        );
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// 親の `DENNO_*` / `PATH` に左右されないよう env を捨てて起動する。
fn denno_in(dir: Option<&Path>, envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin");
    if let Some(d) = dir {
        cmd.current_dir(d);
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args);
    cmd.output().expect("spawn denno")
}

fn denno(envs: &[(&str, &str)], args: &[&str]) -> Output {
    denno_in(None, envs, args)
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

/// panic の痕跡を残さないこと (doctor は失敗を Err で返す契約)。
fn assert_no_panic(o: &Output, what: &str) {
    let e = stderr(o);
    assert!(
        !e.contains("panicked"),
        "{what} panicked:\n{e}\n--- stdout ---\n{}",
        stdout(o)
    );
    assert!(
        !e.contains("RUST_BACKTRACE"),
        "{what} looks like a crash:\n{e}"
    );
}

fn sorted_keys(v: &serde_json::Value) -> Vec<String> {
    let mut k: Vec<String> = v
        .as_object()
        .unwrap_or_else(|| panic!("not an object: {v}"))
        .keys()
        .cloned()
        .collect();
    k.sort_unstable();
    k
}

/// 現行 euid (root かどうかで権限テストの期待値が変わる)。
fn euid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    // /proc/<pid> の所有者は実効 uid。
    std::fs::metadata("/proc/self")
        .map(|m| m.uid())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// 1. JSON Schema: 全セクションのキー集合とキー順
// ---------------------------------------------------------------------------

#[test]
fn json_schema_key_sets_and_order_are_pinned() {
    let td = TempDir::new("schema");
    td.write("m-00001-of-00001.gguf", b"abc");

    let o = denno(
        &[("DENNO_API_KEY", "sk-should-not-leak"), ("DENNO_MAX_CTX", "4096")],
        &["doctor", "--dir", td.as_str(), "--json"],
    );
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    assert_no_panic(&o, "schema run");
    let v = parse_json(&o);

    let expect = |path: &str, want: &[&str]| {
        let node = if path.is_empty() {
            &v
        } else {
            path.split('.').fold(&v, |acc, seg| &acc[seg])
        };
        let got: Vec<String> = sorted_keys(node);
        let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
        assert_eq!(got, want, "schema of `{path}` changed: {node}");
    };

    expect(
        "",
        &[
            "cpu",
            "disk",
            "env",
            "fd_limits",
            "gpu",
            "memory",
            "model",
            "recommendations",
        ],
    );
    expect(
        "cpu",
        &["arch", "logical_cores", "physical_cores", "simd"],
    );
    expect("memory", &["available_bytes", "total_bytes"]);
    expect(
        "gpu",
        &[
            "cuda_driver",
            "gpu_devices",
            "level_zero_intel_driver",
            "level_zero_loader",
            "nvidia_kernel_driver",
        ],
    );
    expect(
        "model",
        &[
            "dir",
            "expected_shards",
            "missing_shards",
            "ok",
            "shards",
            "total_bytes",
            "zero_byte_shards",
        ],
    );
    expect("disk", &["available_bytes", "path", "total_bytes"]);
    expect(
        "fd_limits",
        &["hard", "hard_unlimited", "soft", "soft_unlimited"],
    );
    expect("env", &["entries", "redacted_keys"]);

    // 入れ子の要素のキーも固定する (自動化が読む構造)。
    let shard = &v["model"]["shards"][0];
    assert_eq!(sorted_keys(shard), vec!["bytes", "name"], "shard: {shard}");
    let entry = v["env"]["entries"]
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or_else(|| panic!("env.entries is empty: {v}"));
    assert_eq!(
        sorted_keys(entry),
        vec!["key", "redacted", "value"],
        "env entry: {entry}"
    );

    // キー順も契約 (差分ベースの自動化)。JSON 本文中の初出位置で見る。
    let s = stdout(&o);
    let order = [
        "\"cpu\"",
        "\"memory\"",
        "\"gpu\"",
        "\"model\"",
        "\"disk\"",
        "\"fd_limits\"",
        "\"env\"",
        "\"recommendations\"",
    ];
    let mut last = 0usize;
    for key in order {
        let at = s
            .find(key)
            .unwrap_or_else(|| panic!("key {key} missing from stdout: {s}"));
        assert!(
            at >= last,
            "top-level key order changed at {key} (offset {at} < {last}):\n{s}"
        );
        last = at;
    }

    // 型契約: 取れない値は null で、0 に潰していない。
    for (node, key) in [
        (&v["cpu"], "physical_cores"),
        (&v["memory"], "total_bytes"),
        (&v["memory"], "available_bytes"),
        (&v["gpu"], "cuda_driver"),
        (&v["gpu"], "nvidia_kernel_driver"),
        (&v["gpu"], "level_zero_loader"),
        (&v["gpu"], "level_zero_intel_driver"),
        (&v["model"], "expected_shards"),
    ] {
        let x = &node[key];
        assert!(
            x.is_null() || x.is_u64() || x.is_string(),
            "optional field type drifted: {x}"
        );
    }
    assert!(v["model"]["ok"].is_boolean(), "model.ok = {}", v["model"]["ok"]);
    assert!(v["cpu"]["logical_cores"].as_u64().unwrap_or(0) >= 1);
}

// ---------------------------------------------------------------------------
// 2. テキスト出力: セクション集合・順序・デバッグ出力の不在
// ---------------------------------------------------------------------------

#[test]
fn text_sections_are_pinned_and_free_of_debug_print_residue() {
    let td = TempDir::new("text");
    td.write("m-00001-of-00001.gguf", b"abc");

    let o = denno(
        &[("DENNO_MAX_CTX", "4096")],
        &["doctor", "--dir", td.as_str()],
    );
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);

    let sections = [
        "[CPU]",
        "[メモリ]",
        "[ディスク]",
        "[ファイル記述子]",
        "[GPU]",
        "[モデル]",
        "[環境変数 (DENNO_*)]",
        "[推奨]",
    ];
    let mut last = 0usize;
    for sec in sections {
        let at = t
            .find(sec)
            .unwrap_or_else(|| panic!("missing section {sec} in:\n{t}"));
        assert!(at >= last, "section order changed at {sec}:\n{t}");
        last = at;
    }

    // `{:?}` の残骸を出さない (人間向け 1 行 1 項目の契約)。
    for bad in ["Some(", "None)", "Ok(", "Err(", "{\"", "dbg!", "eprintln!"] {
        assert!(!t.contains(bad), "debug residue {bad:?} in text:\n{t}");
    }
}

// ---------------------------------------------------------------------------
// 3. 終了コードの契約 (text / json で一致)
// ---------------------------------------------------------------------------

#[test]
fn exit_codes_are_consistent_across_modes() {
    let ok = TempDir::new("exit_ok");
    ok.write("m-00001-of-00001.gguf", b"a");

    let missing = TempDir::new("exit_missing");
    missing.write("m-00001-of-00002.gguf", b"a");

    let zero = TempDir::new("exit_zero");
    zero.write("m-00001-of-00001.gguf", b"");

    let nodir = TempDir::new("exit_nodir");
    let absent = nodir.as_path().join("does-not-exist");
    let absent_s = absent.to_str().expect("UTF-8");

    // (args, 期待する終了コード)
    let cases: Vec<(Vec<&str>, i32)> = vec![
        (vec!["doctor"], 0),
        (vec!["doctor", "--json"], 0),
        (vec!["doctor", "--dir", ok.as_str()], 0),
        (vec!["doctor", "--dir", ok.as_str(), "--json"], 0),
        (vec!["doctor", "--dir", missing.as_str()], 1),
        (vec!["doctor", "--dir", missing.as_str(), "--json"], 1),
        (vec!["doctor", "--dir", zero.as_str()], 1),
        (vec!["doctor", "--dir", zero.as_str(), "--json"], 1),
        (vec!["doctor", "--dir", absent_s], 1),
        (vec!["doctor", "--dir", absent_s, "--json"], 1),
        // clap の使い方エラーは 2 (1 = 診断 NG と取り違えない)。
        (vec!["doctor", "--no-such-flag"], 2),
        (vec!["doctor", "--dir"], 2),
    ];

    for (args, want) in cases {
        let o = denno(&[], &args);
        assert_eq!(
            o.status.code(),
            Some(want),
            "args {args:?}: expected exit {want}, got {:?}\n--- stderr ---\n{}",
            o.status.code(),
            stderr(&o)
        );
        assert_no_panic(&o, &format!("args {args:?}"));

        // 成功時は stderr が空、失敗時は stderr に 1 行以上の理由がある。
        if want == 0 {
            assert!(
                stderr(&o).trim().is_empty(),
                "success run wrote to stderr (args {args:?}):\n{}",
                stderr(&o)
            );
        } else {
            assert!(
                !stderr(&o).trim().is_empty(),
                "failure run wrote nothing to stderr (args {args:?})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. 推奨: 決定性と GPU 分岐の排他性
// ---------------------------------------------------------------------------

/// 2 回実行して recommendations が一致する (環境依存の閾値行は除く)。
#[test]
fn recommendations_are_deterministic_across_runs() {
    let td = TempDir::new("det");
    td.write("m-00001-of-00002.gguf", b"a");
    td.write("m-00002-of-00002.gguf", b"b");

    let envs = [
        ("DENNO_MAX_CTX", "4096"),
        ("DENNO_EXPERT_CACHE_MB", "2048"),
        ("DENNO_API_KEY", "sk-det"),
    ];
    let run = || {
        let o = denno(&envs, &["doctor", "--dir", td.as_str(), "--json"]);
        assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
        parse_json(&o)["recommendations"]
            .as_array()
            .unwrap_or_else(|| panic!("recommendations is not an array"))
            .iter()
            .map(|s| s.as_str().expect("recommendation string").to_string())
            // 空き RAM / 空きディスク / fd 上限はホストの現在値で決まるので、
            // 決定性の比較からだけ外す (出現は下で別途確認する)。
            .filter(|s| {
                !(s.contains("空き RAM")
                    || s.contains("空きディスク")
                    || s.contains("ファイル記述子の soft 上限"))
            })
            .collect::<Vec<String>>()
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "recommendations are not deterministic");
    assert!(!a.is_empty(), "no recommendations at all");
}

/// GPU の分岐は排他: 出るのは 1 系統だけ。
#[test]
fn gpu_branch_is_exclusive_and_matches_detection() {
    let o = denno(&[], &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    let recs: Vec<String> = v["recommendations"]
        .as_array()
        .expect("recommendations array")
        .iter()
        .map(|s| s.as_str().expect("string").to_string())
        .collect();
    let joined = recs.join("\n");

    let cuda = recs.iter().any(|s| s.contains("GPU (CUDA)"));
    let kernel_only = recs
        .iter()
        .any(|s| s.contains("NVIDIA カーネルドライバは居る"));
    let cpu_only = recs.iter().any(|s| s.contains("CPU のみ"));

    let hits = [cuda, kernel_only, cpu_only].iter().filter(|b| **b).count();
    assert_eq!(
        hits, 1,
        "GPU 分岐が排他でない (cuda={cuda} kernel_only={kernel_only} cpu_only={cpu_only}):\n{joined}"
    );

    // 検出値と分岐が一致する。
    let has_cuda = v["gpu"]["cuda_driver"].is_string();
    let has_kernel = v["gpu"]["nvidia_kernel_driver"].is_string();
    if has_cuda {
        assert!(cuda, "cuda_driver があるのに GPU 行が無い:\n{joined}");
    } else if has_kernel {
        assert!(
            kernel_only,
            "kernel driver だけあるのに案内が違う:\n{joined}"
        );
    } else {
        assert!(
            cpu_only,
            "GPU が無いのに CPU 行が無い:\n{joined}"
        );
    }

    // 実在しないフラグを勧めない (Level Zero は検出しても情報行だけ)。
    // `--gpus` は NVIDIA の container runtime 引数で実在するので、前方一致する
    // `--gpu` 単体では判定しない。
    for s in &recs {
        for invented in [
            "--backend",
            "--level-zero",
            "--nvidia-smi",
            "--tensorrt",
            "--rocm",
            "--device",
        ] {
            assert!(!s.contains(invented), "invented flag {invented} in: {s}");
        }
    }

    // CPU-only 環境向けの数値は GPU 向けより小さい (取り違えの検出)。
    if cpu_only {
        assert!(joined.contains("--max-ctx 4096"), "{joined}");
        assert!(joined.contains("--max-tokens 256"), "{joined}");
    } else if cuda {
        assert!(joined.contains("--max-ctx 8192"), "{joined}");
        assert!(joined.contains("--max-tokens 512"), "{joined}");
    }
}

// ---------------------------------------------------------------------------
// 5. GPU 検出は子プロセスを起動しない (CPU-only ランナーで安全)
// ---------------------------------------------------------------------------

#[test]
fn gpu_detection_spawns_no_tools() {
    let td = TempDir::new("notools");
    let fake_bin = td.as_path().join("bin");
    std::fs::create_dir_all(&fake_bin).expect("create fake bin");

    // nvidia-smi / lspci / rocm-smi が呼ばれたら印を残す。
    let mut markers = Vec::new();
    for tool in ["nvidia-smi", "lspci", "rocm-smi", "nvidia-debugdump"] {
        let marker = td.as_path().join(format!("called-{tool}"));
        let script = fake_bin.join(tool);
        let mut f = std::fs::File::create(&script).expect("create fake tool");
        writeln!(f, "#!/bin/sh").unwrap();
        writeln!(f, "echo called > {}", marker.display()).unwrap();
        writeln!(f, "echo fake").unwrap();
        drop(f);
        let mut perm = std::fs::metadata(&script).expect("stat").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&script, perm).expect("chmod");
        markers.push(marker);
    }

    let path = format!("{}:/usr/bin:/bin", fake_bin.display());
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.env_clear();
    cmd.env("PATH", &path);
    cmd.args(["doctor", "--json"]);
    let o = cmd.output().expect("spawn denno");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let _ = parse_json(&o);

    for m in &markers {
        assert!(
            !m.exists(),
            "doctor が外部ツールを起動した: {}",
            m.display()
        );
    }
}

// ---------------------------------------------------------------------------
// 6. 権限の制限で panic しない
// ---------------------------------------------------------------------------

#[test]
fn restricted_permissions_do_not_panic() {
    use std::os::unix::fs::PermissionsExt;

    let td = TempDir::new("perm");
    let locked = td.as_path().join("locked");
    std::fs::create_dir_all(&locked).expect("create dir");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let locked_s = locked.to_str().expect("UTF-8");

    let root = euid() == 0;
    for extra in [&[][..], &["--json"][..]] {
        let mut args = vec!["doctor", "--dir", locked_s];
        args.extend_from_slice(extra);
        let o = denno(&[], &args);
        assert_no_panic(&o, &format!("locked dir {args:?}"));
        if root {
            // root は読めてしまうので終了コードは問わない (crash しないことだけ)。
            assert!(
                matches!(o.status.code(), Some(0) | Some(1)),
                "unexpected exit {:?}",
                o.status.code()
            );
        } else {
            assert_eq!(
                o.status.code(),
                Some(1),
                "unreadable dir should exit 1 (args {args:?}):\n{}",
                stderr(&o)
            );
            assert!(
                stdout(&o).trim().is_empty(),
                "report printed for an unreadable dir (args {args:?}):\n{}",
                stdout(&o)
            );
        }
    }

    // 読めないファイルはメタデータだけなので列挙できる (0 バイト扱いにしない)。
    let f = td.write("m-00001-of-00001.gguf", b"abc");
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let o = denno(&[], &["doctor", "--dir", td.as_str(), "--json"]);
    assert_no_panic(&o, "unreadable shard");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["model"]["total_bytes"].as_u64(), Some(3));
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
}

// ---------------------------------------------------------------------------
// 7. ツールが無い環境 (PATH 無し / 空 env) で panic しない
// ---------------------------------------------------------------------------

#[test]
fn empty_environment_without_tools_does_not_panic() {
    for (label, keep_path) in [("no PATH", false), ("minimal PATH", true)] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
        cmd.env_clear();
        if keep_path {
            cmd.env("PATH", "/usr/bin:/bin");
        }
        cmd.args(["doctor", "--json"]);
        let o = cmd.output().expect("spawn denno");
        assert_eq!(
            o.status.code(),
            Some(0),
            "{label}: stderr:\n{}",
            stderr(&o)
        );
        assert_no_panic(&o, label);
        let v = parse_json(&o);
        // PATH が無くても CPU / メモリは取れる (procfs 由来)。取れない値は null。
        assert!(v["cpu"]["arch"].is_string(), "{label}: {}", v["cpu"]);
        assert!(
            v["cpu"]["logical_cores"].as_u64().unwrap_or(0) >= 1,
            "{label}: {}",
            v["cpu"]
        );
        assert!(v["cpu"]["simd"].is_array(), "{label}: {}", v["cpu"]);
        // DENNO_* が無いので env 一覧は空。
        assert_eq!(
            v["env"]["entries"].as_array().map(Vec::len),
            Some(0),
            "{label}: {}",
            v["env"]
        );
        assert_eq!(
            v["env"]["redacted_keys"].as_array().map(Vec::len),
            Some(0),
            "{label}: {}",
            v["env"]
        );
    }
}

// ---------------------------------------------------------------------------
// 8. 変な値 (改行 / 非 UTF-8 / 巨大値) と非 UTF-8 名のシャードで panic しない
// ---------------------------------------------------------------------------

#[test]
fn hostile_values_and_non_utf8_names_do_not_panic() {
    use std::os::unix::ffi::OsStrExt;

    let big = "x".repeat(8192);

    // 非 UTF-8 の env 値を 1 本と、改行入りの秘密を 1 本。
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin");
    cmd.env(OsStr::from_bytes(b"DENNO_WEIRD_BYTES"), OsStr::from_bytes(b"\xff\xfe\x80"));
    cmd.env("DENNO_MULTILINE_SECRET_TOKEN", "line1\nsk-leak\nline3");
    cmd.env("DENNO_BIG_VALUE", &big);
    cmd.args(["doctor", "--json"]);
    let o = cmd.output().expect("spawn denno");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    assert_no_panic(&o, "hostile env");
    let v = parse_json(&o);

    assert_eq!(
        v["env"]["entries"].as_array().map(Vec::len),
        Some(3),
        "env entries: {}",
        v["env"]
    );
    // 秘密名は伏せ、生値は JSON のどこにも出さない。
    let s = stdout(&o);
    assert!(!s.contains("sk-leak"), "secret leaked into json");
    let secret = v["env"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["key"] == "DENNO_MULTILINE_SECRET_TOKEN")
        .expect("secret entry present");
    assert_eq!(secret["value"], "<redacted>");
    assert_eq!(secret["redacted"], serde_json::Value::Bool(true));

    // テキストでは改行を潰して 1 行 1 変数を保つ。
    let t = {
        let mut c = Command::new(env!("CARGO_BIN_EXE_denno"));
        c.env_clear();
        c.env("PATH", "/usr/bin:/bin");
        c.env("DENNO_MULTILINE_SECRET_TOKEN", "line1\nsk-leak\nline3");
        c.args(["doctor"]);
        let o2 = c.output().expect("spawn denno");
        assert_eq!(o2.status.code(), Some(0), "stderr:\n{}", stderr(&o2));
        stdout(&o2)
    };
    assert!(!t.contains("sk-leak"), "secret leaked into text:\n{t}");
    let lines_with = t
        .lines()
        .filter(|l| l.contains("DENNO_MULTILINE_SECRET_TOKEN"))
        .count();
    assert_eq!(lines_with, 1, "env 行が複数行に割れた:\n{t}");

    // 非 UTF-8 のシャード名でも列挙でき、JSON が壊れない。
    let td = TempDir::new("nonutf8");
    let name = OsStr::from_bytes(b"bad-\xff-model-00001-of-00001.gguf");
    std::fs::write(td.as_path().join(name), b"abc").expect("write shard");
    let o = denno(&[], &["doctor", "--dir", td.as_str(), "--json"]);
    assert_no_panic(&o, "non-utf8 shard name");
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    assert_eq!(v["model"]["shards"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["model"]["ok"], serde_json::Value::Bool(true));
    assert_eq!(v["model"]["total_bytes"].as_u64(), Some(3));
}

// ---------------------------------------------------------------------------
// 9. 出力にホスト識別子を混ぜない
// ---------------------------------------------------------------------------

#[test]
fn output_carries_no_host_identifiers() {
    let td = TempDir::new("ident");

    // cwd を一時ディレクトリにして、disk.path を /tmp 配下に閉じ込める。
    let o = denno_in(Some(td.as_path()), &[], &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    let s = stdout(&o);

    assert!(
        !s.contains("/home/"),
        "home directory leaked into output:\n{s}"
    );
    // `@` (email / URL userinfo) を出さない。ただし `gpu.nvidia_kernel_driver` は
    // カーネルドライバが自分で名乗る NVRM 行の丸ごと写しで、`(user@build-host)`
    // を含むことがある (task 1646 の監査で確認した既存の通過点)。ここだけは
    // 例外として、その値を取り除いた残りに対して検査する。
    let mut rest = s.clone();
    if let Some(nvrm) = v["gpu"]["nvidia_kernel_driver"].as_str() {
        rest = rest.replace(nvrm, "");
    }
    assert!(
        !rest.contains('@'),
        "`@` (email/userinfo) leaked into output outside the NVRM line:\n{rest}"
    );

    // 私設 IPv4 (10/8, 172.16/12, 192.168/16, 127/8) を出さない。
    for bad in ["10.", "172.16.", "172.17.", "192.168.", "127.0.0.1"] {
        assert!(
            !s.contains(bad),
            "private/LAN address fragment {bad:?} leaked into output:\n{s}"
        );
    }

    // ホスト名 (取れる場合) を出さない。
    if let Ok(host) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
        let host = host.trim();
        if host.len() >= 6 {
            assert!(
                !s.contains(host),
                "hostname {host:?} leaked into output:\n{s}"
            );
        }
    }

    // 診断対象として渡したパス自体は出てよい (利用者が指定した情報)。
    assert_eq!(
        v["disk"]["path"].as_str(),
        Some(td.as_str()),
        "disk.path should mirror cwd: {}",
        v["disk"]
    );

    // デバッグ出力の痕跡 (stderr に何も出さない = ログ洪水なし)。
    assert!(
        stderr(&o).trim().is_empty(),
        "doctor wrote diagnostics to stderr:\n{}",
        stderr(&o)
    );
}
