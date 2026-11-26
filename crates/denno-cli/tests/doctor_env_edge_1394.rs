//! task 1394: `denno doctor` の拡張検査 — ディスク / fd 上限 / `DENNO_*` env の
//! 境界試験 (task 1046 の CLI 契約を補う)。
//!
//! 実バイナリ (`CARGO_BIN_EXE_denno`) を起動して確かめること:
//!
//! 1. `--json` のスキーマ安定性 — `disk` / `fd_limits` / `env` のキー集合と型が
//!    変わっていない (キーの追加・改名は自動化側の破壊なので、集合一致で見る)。
//! 2. 既知の `DENNO_*` だけを設定したときは静かに通る (doctor の env 一覧と
//!    `config::diag` の診断が同じ集合を「問題なし」と言う)。
//! 3. 未知 / 綴り違いの `DENNO_*` は doctor の env 一覧に**値ごと**出る
//!    (doctor は絞り込まない) 一方、`config::diag` は `env_unknown` として
//!    警告し、近い名前を提案する。診断メッセージに値は載せない。
//!    `DENNO_` 接頭辞の欠けは doctor の一覧には出ないが diag は拾う。
//! 4. `DENNO_API_KEY` / `DENNO_API_KEYS` などの秘密名は、JSON / テキスト /
//!    診断のどこにも生値を出さない (`<redacted>` に置き換える)。
//! 5. ディスクの検査対象は `--dir` をそのまま反映する (相対 `.` もそのまま)。
//! 6. 存在しない `--dir` はレポートを出さず非 0 + stderr のエラーで終わる。
//! 7. fd 上限の推奨は境界 (4096 未満で出る / 4096 以上で出ない) で切り替わる。
//!
//! GPU を初期化しない読み取りだけの診断なので CPU-only でも安全。GPU 検出値は
//! ホスト依存なので値ではなく型だけを見る。一時ディレクトリはテストごとに一意で
//! 作り、Drop で消す。
//!
//! `config::diag` は bin クレートの内部モジュールなので、既存の
//! `config_diag_1066.rs` と同じく `#[path]` で `src/config.rs` を読み込む。
//! 診断関数は env を引数で受ける純関数で、プロセスの env は触らない。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use config::diag::{diagnose_env, known_env_vars, Level};

static SEQ: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

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
            "denno1394_{}_{}_{}_{}",
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

/// `denno` を最小 env で起動する。`env_clear` でホストの `DENNO_*` を必ず落とす
/// (一覧の集合一致を見るテストが親プロセスの env に左右されない)。
fn denno(envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args);
    cmd.output().expect("spawn denno")
}

fn denno_in(dir: &Path, envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.current_dir(dir);
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args);
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

/// オブジェクトのキーを昇順で返す (`serde_json` の内部順に依存しない)。
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

/// env の (key, value) を BTreeMap にする (`diagnose_env` の入力形)。
fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn codes(diags: &[config::diag::Diag]) -> Vec<&'static str> {
    diags.iter().map(|d| d.code).collect()
}

fn diag_text(diags: &[config::diag::Diag]) -> String {
    diags.iter().map(|d| format!("{}\n", d.line())).collect()
}

/// `env` 一覧 (doctor の JSON) から key → value を取り出す。
fn env_entries(v: &serde_json::Value) -> BTreeMap<String, (String, bool)> {
    let mut out = BTreeMap::new();
    for e in v["env"]["entries"]
        .as_array()
        .unwrap_or_else(|| panic!("env.entries not array: {}", v["env"]))
    {
        let key = e["key"].as_str().expect("key string").to_string();
        let value = e["value"].as_str().expect("value string").to_string();
        let redacted = e["redacted"].as_bool().expect("redacted bool");
        out.insert(key, (value, redacted));
    }
    out
}

/// 既知の `DENNO_*` 10 個 (config の SPEC 由来) を、範囲内の値で埋める。
fn all_known_envs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("DENNO_AUTO_CONFIG", "1"),
        ("DENNO_FLASH_DIR", "/tmp/models"),
        ("DENNO_FLASH_HOST", "127.0.0.1"),
        ("DENNO_FLASH_MAX_CTX", "4096"),
        ("DENNO_FLASH_MAX_TOKENS", "256"),
        ("DENNO_FLASH_NAME", "qwen3.8-flash-next"),
        ("DENNO_FLASH_PORT", "18098"),
        ("DENNO_KV_SLOTS", "3"),
        ("DENNO_MTP_PATH", "/tmp/mtp.safetensors"),
        ("DENNO_STEER_RECIPE", "/tmp/recipe.json"),
    ]
}

// ---------------------------------------------------------------------------
// 1. JSON スキーマの安定性
// ---------------------------------------------------------------------------

#[test]
fn json_schema_of_disk_fd_and_env_is_stable() {
    let o = denno(&[("DENNO_FLASH_PORT", "18098")], &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);

    assert_eq!(
        sorted_keys(&v),
        vec![
            "cpu",
            "disk",
            "env",
            "fd_limits",
            "gpu",
            "memory",
            "model",
            "recommendations"
        ],
        "top-level sections changed: {v}"
    );

    assert_eq!(
        sorted_keys(&v["disk"]),
        vec!["available_bytes", "path", "total_bytes"],
        "disk schema: {}",
        v["disk"]
    );
    assert_eq!(
        sorted_keys(&v["fd_limits"]),
        vec!["hard", "hard_unlimited", "soft", "soft_unlimited"],
        "fd_limits schema: {}",
        v["fd_limits"]
    );
    assert_eq!(
        sorted_keys(&v["env"]),
        vec!["entries", "redacted_keys"],
        "env schema: {}",
        v["env"]
    );

    // 型契約: disk は文字列 + u64/null、fd は u64/null + bool
    assert!(v["disk"]["path"].is_string(), "{}", v["disk"]);
    for key in ["total_bytes", "available_bytes"] {
        let x = &v["disk"][key];
        assert!(x.is_u64() || x.is_null(), "disk.{key} = {x}");
    }
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
    // 無制限フラグと null は一致している (片方だけ立たない)
    for (num, unl) in [("soft", "soft_unlimited"), ("hard", "hard_unlimited")] {
        if v["fd_limits"][unl].as_bool() == Some(true) {
            assert!(
                v["fd_limits"][num].is_null(),
                "fd_limits.{num} = {} but {unl}=true",
                v["fd_limits"][num]
            );
        }
    }
    if let (Some(s), Some(h)) = (
        v["fd_limits"]["soft"].as_u64(),
        v["fd_limits"]["hard"].as_u64(),
    ) {
        assert!(s <= h, "soft {s} > hard {h}");
    }

    // env の entry もキー集合を固定する
    let entries = v["env"]["entries"].as_array().expect("entries array");
    for e in entries {
        assert_eq!(
            sorted_keys(e),
            vec!["key", "redacted", "value"],
            "env entry schema: {e}"
        );
        assert!(e["key"].as_str().unwrap_or("").starts_with("DENNO_"), "{e}");
    }
    assert!(v["env"]["redacted_keys"].is_array(), "{}", v["env"]);
    assert!(v["recommendations"].is_array(), "{}", v["recommendations"]);
}

// ---------------------------------------------------------------------------
// 2. 既知の DENNO_* だけなら静かに通る
// ---------------------------------------------------------------------------

#[test]
fn known_vars_only_pass_cleanly() {
    let envs = all_known_envs();
    let o = denno(&envs, &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    let entries = env_entries(&v);

    let mut want: Vec<String> = known_env_vars().iter().map(|s| s.to_string()).collect();
    want.sort_unstable();
    let got: Vec<String> = entries.keys().cloned().collect();
    assert_eq!(got, want, "doctor の env 一覧が既知名の集合と一致しない");

    // 既知名は値を隠さない。ただし `DENNO_FLASH_MAX_TOKENS` は名前が
    // `*TOKEN*` を含むため、doctor の「伏せすぎる側に倒す」規約で伏せられる
    // (値は秘密ではないが、名前だけでは区別できない)。この過剰伏せ字も契約として固定する。
    let over_redacted = ["DENNO_FLASH_MAX_TOKENS"];
    for (key, (value, redacted)) in &entries {
        if over_redacted.contains(&key.as_str()) {
            assert!(*redacted, "{key} が伏せられていない");
            assert_eq!(value, "<redacted>", "{key} の値: {value}");
            continue;
        }
        assert!(!redacted, "{key} が不要に伏せられている");
        let expected = envs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("unexpected key: {key}"));
        assert_eq!(value, expected, "{key} の値が変わっている");
    }
    let red: Vec<&str> = v["env"]["redacted_keys"]
        .as_array()
        .expect("redacted_keys array")
        .iter()
        .map(|x| x.as_str().expect("string"))
        .collect();
    assert_eq!(red, over_redacted, "{red:?}");

    // 同じ env を診断に掛けても何も出ない
    let diags = diagnose_env(&env_map(&envs));
    assert_eq!(diags, Vec::new(), "{diags:?}");
}

// ---------------------------------------------------------------------------
// 3. 未知 / 綴り違い / 接頭辞欠け
// ---------------------------------------------------------------------------

#[test]
fn unknown_and_typo_vars_are_listed_then_warned() {
    let envs = [
        ("DENNO_FLASH_PROT", "18098"),                  // typo (PORT)
        ("DENNO_TOTALLY_UNRELATED_THING", "1"),         // 近い名前なし
        ("DENNO_FLASH_PORT", "18098"),                  // 正当
        ("ENNO_FLASH_PORT", "18098"),                   // 接頭辞欠け
    ];
    let o = denno(&envs, &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    let entries = env_entries(&v);

    // doctor は絞り込まず、値ごと出す (監査する人が生の一覧を見られる)
    for (key, value) in [("DENNO_FLASH_PROT", "18098"), ("DENNO_TOTALLY_UNRELATED_THING", "1")] {
        let (got, redacted) = entries
            .get(key)
            .unwrap_or_else(|| panic!("{key} が env 一覧に無い: {entries:?}"));
        assert_eq!(got, value, "{key}");
        assert!(!redacted, "{key} は秘密名ではない");
    }
    // 接頭辞の無い名前は `DENNO_*` の一覧には入らない
    assert!(!entries.contains_key("ENNO_FLASH_PORT"), "{entries:?}");

    // diag 側は「読まれない名前」として警告する
    let diags = diagnose_env(&env_map(&envs));
    let unknown: Vec<&str> = diags
        .iter()
        .filter(|d| d.code == "env_unknown")
        .flat_map(|d| d.vars.iter().map(|s| s.as_str()))
        .collect();
    assert!(unknown.contains(&"DENNO_FLASH_PROT"), "{diags:?}");
    assert!(
        unknown.contains(&"DENNO_TOTALLY_UNRELATED_THING"),
        "{diags:?}"
    );
    assert!(!unknown.contains(&"DENNO_FLASH_PORT"), "既知名を unknown にした: {diags:?}");

    let typo = diags
        .iter()
        .find(|d| d.code == "env_unknown" && d.vars == vec!["DENNO_FLASH_PROT".to_string()])
        .expect("typo の env_unknown");
    assert_eq!(typo.level, Level::Warn);
    assert!(typo.message.contains("もしかして"), "{}", typo.message);
    assert!(
        typo.message.contains("DENNO_FLASH_PORT"),
        "近い名前を提案していない: {}",
        typo.message
    );
    // 値は診断に載せない
    assert!(!typo.message.contains("18098"), "値が漏れた: {}", typo.message);

    // 接頭辞欠けは doctor の一覧に無いが diag は拾う
    assert!(
        diags
            .iter()
            .any(|d| d.code == "env_unprefixed" && d.vars == vec!["ENNO_FLASH_PORT".to_string()]),
        "{diags:?}"
    );
    // `env` 一覧に入った key は必ず diag でも認識される (重複した視点の食い違い防止)
    for key in entries.keys() {
        let one = diagnose_env(&env_map(&[(key.as_str(), "x")]));
        assert!(!one.is_empty(), "doctor にだけ見えて diag が無視する key: {key}");
    }
}

// ---------------------------------------------------------------------------
// 4. 秘密値の伏せ字
// ---------------------------------------------------------------------------

const SECRET_API_KEY: &str = "sk-SENTINEL-1394-AAA";
const SECRET_API_KEYS: &str = "sk-SENTINEL-1394-BBB,sk-SENTINEL-1394-CCC";
const SECRET_HF_TOKEN: &str = "hf-SENTINEL-1394-DDD";
const SECRET_WEBHOOK_KEY: &str = "wh-SENTINEL-1394-EEE";

#[test]
fn secret_values_never_appear_in_doctor_output() {
    let envs = [
        ("DENNO_API_KEY", SECRET_API_KEY),
        ("DENNO_API_KEYS", SECRET_API_KEYS),
        ("DENNO_HF_TOKEN", SECRET_HF_TOKEN),
        ("DENNO_WEBHOOK_KEY", SECRET_WEBHOOK_KEY),
        ("DENNO_MAX_CTX", "4096"),
    ];
    let sentinels = [
        SECRET_API_KEY,
        SECRET_API_KEYS,
        "sk-SENTINEL-1394-BBB",
        "sk-SENTINEL-1394-CCC",
        SECRET_HF_TOKEN,
        SECRET_WEBHOOK_KEY,
    ];

    // JSON モード
    let o = denno(&envs, &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let raw = stdout(&o);
    for s in sentinels {
        assert!(!raw.contains(s), "秘密が stdout に漏れた: {s}\n{raw}");
        assert!(!stderr(&o).contains(s), "秘密が stderr に漏れた: {s}");
    }
    let v = parse_json(&o);
    let entries = env_entries(&v);
    for key in [
        "DENNO_API_KEY",
        "DENNO_API_KEYS",
        "DENNO_HF_TOKEN",
        "DENNO_WEBHOOK_KEY",
    ] {
        let (value, redacted) = entries
            .get(key)
            .unwrap_or_else(|| panic!("{key} が env 一覧に無い: {entries:?}"));
        assert_eq!(value, "<redacted>", "{key} の値: {value}");
        assert!(*redacted, "{key} が redacted=false");
    }
    // 秘密でない値はそのまま見える (伏せすぎて何も分からない状態にしない)
    let (ctx, redacted) = entries.get("DENNO_MAX_CTX").expect("DENNO_MAX_CTX");
    assert_eq!(ctx, "4096");
    assert!(!redacted);

    let red: Vec<&str> = v["env"]["redacted_keys"]
        .as_array()
        .expect("redacted_keys array")
        .iter()
        .map(|x| x.as_str().expect("string"))
        .collect();
    for key in [
        "DENNO_API_KEY",
        "DENNO_API_KEYS",
        "DENNO_HF_TOKEN",
        "DENNO_WEBHOOK_KEY",
    ] {
        assert!(red.contains(&key), "{key} が redacted_keys に無い: {red:?}");
    }
    assert!(!red.contains(&"DENNO_MAX_CTX"), "{red:?}");

    // テキストモードでも生値は出ない
    let t = stdout(&denno(&envs, &["doctor"]));
    for s in sentinels {
        assert!(!t.contains(s), "秘密がテキスト出力に漏れた: {s}\n{t}");
    }
    assert!(t.contains("DENNO_API_KEY=<redacted>"), "伏せ字が出ていない:\n{t}");
    assert!(t.contains("DENNO_MAX_CTX=4096"), "非秘密の値が消えた:\n{t}");

    // 診断メッセージ/行にも生値は出ない
    let diags = diagnose_env(&env_map(&envs));
    let text = diag_text(&diags);
    for s in sentinels {
        assert!(!text.contains(s), "秘密が診断に漏れた: {s}\n{text}");
    }
    assert!(text.contains("DENNO_API_KEY"), "秘密名は出してよい:\n{text}");
}

// ---------------------------------------------------------------------------
// 5. ディスクの検査対象
// ---------------------------------------------------------------------------

#[test]
fn disk_section_echoes_the_inspected_path() {
    let td = TempDir::new("disk");
    td.write("solo.gguf", b"gguf");

    // 絶対パスはそのまま反映され、実測値が入る
    let o = denno(&[], &["doctor", "--dir", td.as_str(), "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    assert_eq!(v["disk"]["path"], td.as_str(), "{}", v["disk"]);
    assert_eq!(v["model"]["dir"], td.as_str(), "{}", v["model"]);
    let total = v["disk"]["total_bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("disk.total_bytes not u64: {}", v["disk"]));
    let avail = v["disk"]["available_bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("disk.available_bytes not u64: {}", v["disk"]));
    assert!(total > 0, "total={total}");
    assert!(avail <= total, "avail {avail} > total {total}");

    // 相対指定は正規化せず、渡した文字列のまま出す
    let o2 = denno_in(td.as_path(), &[], &["doctor", "--dir", ".", "--json"]);
    assert_eq!(o2.status.code(), Some(0), "stderr:\n{}", stderr(&o2));
    let v2 = parse_json(&o2);
    assert_eq!(v2["disk"]["path"], ".", "{}", v2["disk"]);
    assert_eq!(v2["model"]["dir"], ".", "{}", v2["model"]);
    assert!(v2["model"]["ok"].as_bool().unwrap_or(false), "{}", v2["model"]);
}

// ---------------------------------------------------------------------------
// 6. 存在しない --dir はレポートを出さない
// ---------------------------------------------------------------------------

#[test]
fn nonexistent_dir_fails_without_a_report() {
    let td = TempDir::new("nodir");
    let missing = td.as_path().join("does-not-exist");
    let missing_s = missing.to_str().expect("UTF-8 path");

    for args in [
        vec!["doctor", "--dir", missing_s, "--json"],
        vec!["doctor", "--dir", missing_s],
    ] {
        let o = denno(&[], &args);
        assert_ne!(o.status.code(), Some(0), "存在しない --dir で成功した: {args:?}");
        assert!(
            stdout(&o).trim().is_empty(),
            "レポートを出してしまった: {args:?}\n{}",
            stdout(&o)
        );
        assert!(
            stderr(&o).contains("読めない"),
            "理由が stderr に無い: {args:?}\n{}",
            stderr(&o)
        );
    }
}

// ---------------------------------------------------------------------------
// 7. fd 上限の推奨は 4096 の境界で切り替わる
// ---------------------------------------------------------------------------

/// `sh` で nofile の soft 上限を `n` にしてから `denno doctor --json` を exec する。
fn doctor_with_nofile(n: u64) -> Output {
    let bin = env!("CARGO_BIN_EXE_denno");
    let script = format!("ulimit -n {n} || exit 97; exec '{bin}' doctor --json");
    Command::new("sh")
        .args(["-c", &script])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("spawn sh")
}

/// このホストの `RLIMIT_NOFILE` の hard 上限。`unlimited` は `u64::MAX`。
fn hard_nofile() -> u64 {
    let o = Command::new("sh")
        .args(["-c", "ulimit -Hn"])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("spawn sh");
    let s = String::from_utf8_lossy(&o.stdout);
    let s = s.trim();
    if s == "unlimited" {
        return u64::MAX;
    }
    s.parse()
        .unwrap_or_else(|e| panic!("unexpected ulimit -Hn output {s:?}: {e}"))
}

#[test]
fn fd_recommendation_flips_at_the_4096_boundary() {
    let hard = hard_nofile();

    // 4096 を設定できるホストでは境界の両側を確かめる
    if hard >= 4096 {
        let low = doctor_with_nofile(4095);
        assert_eq!(low.status.code(), Some(0), "stderr:\n{}", stderr(&low));
        let v = parse_json(&low);
        assert_eq!(v["fd_limits"]["soft"].as_u64(), Some(4095), "{}", v["fd_limits"]);
        assert_eq!(v["fd_limits"]["soft_unlimited"], false);
        let recs = v["recommendations"].as_array().expect("recommendations");
        assert!(
            recs.iter()
                .any(|s| s.as_str().unwrap_or("").contains("ファイル記述子")),
            "4095 で推奨が出ない: {recs:?}"
        );

        let ok = doctor_with_nofile(4096);
        assert_eq!(ok.status.code(), Some(0), "stderr:\n{}", stderr(&ok));
        let v2 = parse_json(&ok);
        assert_eq!(v2["fd_limits"]["soft"].as_u64(), Some(4096), "{}", v2["fd_limits"]);
        let recs2 = v2["recommendations"].as_array().expect("recommendations");
        assert!(
            !recs2
                .iter()
                .any(|s| s.as_str().unwrap_or("").contains("ファイル記述子")),
            "4096 で推奨が出た: {recs2:?}"
        );
        return;
    }

    // hard が 4096 未満のホストでは上げられないので、hard ちょうどで「出る」側だけ見る。
    // 偽の合格は出さず、境界の非該当を明示する。
    assert!(hard >= 3, "implausibly low hard limit: {hard}");
    let o = doctor_with_nofile(hard);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = parse_json(&o);
    assert_eq!(v["fd_limits"]["soft"].as_u64(), Some(hard), "{}", v["fd_limits"]);
    if hard < 4096 {
        let recs = v["recommendations"].as_array().expect("recommendations");
        assert!(
            recs.iter()
                .any(|s| s.as_str().unwrap_or("").contains("ファイル記述子")),
            "小さい soft 上限で推奨が出ない: {recs:?}"
        );
    }
}
