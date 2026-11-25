//! task 1064: `denno bench --compare` の CLI 契約を実バイナリで確かめる。
//!
//! `src/bench.rs` の単体試験は `compare_reports` を直接呼ぶが、ここでは
//! `CARGO_BIN_EXE_denno` を `std::process::Command` で起動し、利用者が実際に
//! 触る経路 (引数解析 → ファイル読み込み → Markdown 表 → ANSI 要約 → 終了コード)
//! を通しで検証する:
//!
//! 1. クリーン比較は終了コード 0 で Markdown 表を出し、`NO_COLOR=1` で
//!    ANSI エスケープを出さない。
//! 2. `--fail-on-regress-pct` を超える退行は非 0 で終わる。
//! 3. スループットが同じでも決定論 md5 の不一致は非 0 で終わる。
//! 4. ファイル欠け / 壊れた JSON は明確なエラーで非 0 で終わる。
//!
//! レポートは手書きの最小 JSON (省略フィールドは serde の既定値で受ける) を
//! 一時ディレクトリに置き、GPU もサーバーも使わない。

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
            "denno1064_{}_{}_{}_{}",
            tag,
            std::process::id(),
            n,
            nanos
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let p = self.path.join(name);
        std::fs::write(&p, text).expect("write fixture");
        p
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

const H1: &str = "0123456789abcdef0123456789abcdef";
const H2: &str = "ffffffffffffffffffffffffffffffff";

/// 最小限のレポート JSON。pp のスループットと md5 だけ差し替える。
fn report(pp_tok_s: f64, md5: &str) -> String {
    format!(
        r#"{{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "fixture-model",
  "rounds": 3,
  "pp": [{{"pp_tokens": 512, "depth": 0, "pp_tok_s": {pp_tok_s}}}],
  "tg": [{{"depth": 0, "gen_tokens": 128, "tg_tok_s": 50.0}}],
  "turn2": {{"depth": 2048, "speedup": 2.0}},
  "determinism": {{"md5": "{md5}", "verified": true}}
}}"#
    )
}

/// `denno` を起動する。`no_color` なら `NO_COLOR=1` を渡し、そうでなければ
/// 継承環境の `NO_COLOR` を消して色付き経路を確かめられるようにする。
fn denno(args: &[&str], no_color: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    cmd.env_remove("NO_COLOR");
    if no_color {
        cmd.env("NO_COLOR", "1");
    }
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn run_compare(base: &Path, new: &Path, extra: &[&str], no_color: bool) -> Output {
    let mut owned: Vec<String> = vec!["bench".to_string()];
    owned.extend(extra.iter().map(|s| s.to_string()));
    owned.push("--compare".to_string());
    owned.push(base.to_string_lossy().into_owned());
    owned.push(new.to_string_lossy().into_owned());
    let args: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    denno(&args, no_color)
}

// ---------------------------------------------------------------------------
// 1. クリーン比較: 終了 0 + Markdown 表 + NO_COLOR による ANSI 抑制
// ---------------------------------------------------------------------------

#[test]
fn clean_compare_exits_zero_and_prints_markdown_without_ansi() {
    let td = TempDir::new("clean");
    let base = td.write("base.json", &report(1000.0, H1));
    let new = td.write("new.json", &report(1000.0, H1));

    let o = run_compare(&base, &new, &[], true);
    assert_eq!(
        o.status.code(),
        Some(0),
        "clean compare must succeed\nstderr:\n{}",
        stderr(&o)
    );

    let t = stdout(&o);
    assert!(t.contains("### denno bench compare"), "no heading:\n{t}");
    assert!(
        t.contains("| test | metric | base | new | delta | status |"),
        "no table header:\n{t}"
    );
    assert!(t.contains("| pp512 @ d0 | pp_tok_s |"), "no pp row:\n{t}");
    assert!(
        t.contains("| determinism md5 |") && t.contains("| MATCH |"),
        "no determinism MATCH row:\n{t}"
    );
    assert!(t.contains("regressions: 0"), "no summary line:\n{t}");
    assert!(
        !t.contains('\x1b'),
        "NO_COLOR=1 must suppress ANSI escapes:\n{t:?}"
    );
}

// ---------------------------------------------------------------------------
// 1b. NO_COLOR 未設定なら同じ比較でも ANSI が出る (抑制の対照)
// ---------------------------------------------------------------------------

#[test]
fn color_is_emitted_when_no_color_is_unset() {
    let td = TempDir::new("color");
    let base = td.write("base.json", &report(1000.0, H1));
    let new = td.write("new.json", &report(800.0, H1));

    // 閾値なし: 差分は出るが退行判定はせず終了 0。
    let o = run_compare(&base, &new, &[], false);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(
        t.contains("\x1b[3"),
        "expected ANSI color when NO_COLOR is unset:\n{t:?}"
    );

    let q = run_compare(&base, &new, &[], true);
    assert!(
        !stdout(&q).contains('\x1b'),
        "NO_COLOR=1 must strip the same escapes:\n{:?}",
        stdout(&q)
    );
}

// ---------------------------------------------------------------------------
// 2. --fail-on-regress-pct を超える退行は非 0
// ---------------------------------------------------------------------------

#[test]
fn regression_beyond_threshold_exits_nonzero() {
    let td = TempDir::new("regress");
    let base = td.write("base.json", &report(1000.0, H1));
    // 1000 -> 800 は -20%。閾値 5% を超える。
    let new = td.write("new.json", &report(800.0, H1));

    let o = run_compare(&base, &new, &["--fail-on-regress-pct", "5"], true);
    assert_eq!(
        o.status.code(),
        Some(1),
        "regression must exit non-zero\nstderr:\n{}",
        stderr(&o)
    );

    // 表は終了前に出ている (失敗しても診断は読める)
    let t = stdout(&o);
    assert!(t.contains("| REGRESS |"), "no REGRESS row:\n{t}");
    assert!(t.contains("-20.00%"), "no delta in table:\n{t}");

    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(
        e.contains("regression"),
        "error should name the regression:\n{e}"
    );
}

// ---------------------------------------------------------------------------
// 2b. 閾値内の揺れは非 0 にしない
// ---------------------------------------------------------------------------

#[test]
fn small_regression_within_threshold_exits_zero() {
    let td = TempDir::new("within");
    let base = td.write("base.json", &report(1000.0, H1));
    // -2% は閾値 5% の内側。
    let new = td.write("new.json", &report(980.0, H1));

    let o = run_compare(&base, &new, &["--fail-on-regress-pct", "5"], true);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("| ok |"), "expected ok row:\n{t}");
    assert!(!t.contains("| REGRESS |"), "unexpected regression:\n{t}");
}

// ---------------------------------------------------------------------------
// 3. スループット不変でも md5 不一致は非 0 (閾値指定なし)
// ---------------------------------------------------------------------------

#[test]
fn determinism_mismatch_exits_nonzero_without_threshold() {
    let td = TempDir::new("det");
    let base = td.write("base.json", &report(1000.0, H1));
    let new = td.write("new.json", &report(1000.0, H2));

    let o = run_compare(&base, &new, &[], true);
    assert_eq!(
        o.status.code(),
        Some(1),
        "md5 mismatch must fail even with unchanged throughput\nstderr:\n{}",
        stderr(&o)
    );

    let t = stdout(&o);
    assert!(t.contains("| MISMATCH |"), "no MISMATCH row:\n{t}");
    assert!(
        t.contains("regressions: 0"),
        "throughput is unchanged, so no regressions expected:\n{t}"
    );
    assert!(t.contains("| ok |"), "throughput row should be ok:\n{t}");

    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(
        e.contains("determinism"),
        "error should point at determinism:\n{e}"
    );
}

// ---------------------------------------------------------------------------
// 4. ファイル欠け / 壊れた JSON は明確なエラーで非 0
// ---------------------------------------------------------------------------

#[test]
fn missing_file_exits_nonzero_with_clear_error() {
    let td = TempDir::new("missing");
    let base = td.write("base.json", &report(1000.0, H1));
    let absent = td.join("nope.json");

    let o = run_compare(&base, &absent, &[], true);
    assert_eq!(o.status.code(), Some(1), "missing file must fail");
    assert!(
        stdout(&o).trim().is_empty(),
        "no report should be printed when a file is unreadable:\n{}",
        stdout(&o)
    );
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(
        e.contains("nope.json"),
        "error should name the missing path:\n{e}"
    );
}

#[test]
fn invalid_json_exits_nonzero_with_parse_error() {
    let td = TempDir::new("badjson");
    let base = td.write("base.json", &report(1000.0, H1));
    let bad = td.write("bad.json", "{ this is not json");

    let o = run_compare(&base, &bad, &[], true);
    assert_eq!(o.status.code(), Some(1), "invalid JSON must fail");
    assert!(
        stdout(&o).trim().is_empty(),
        "no report should be printed for invalid JSON:\n{}",
        stdout(&o)
    );
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(
        e.contains("bad.json"),
        "error should name the bad path:\n{e}"
    );
    // パース失敗であることが読み取れる (read ではなく parse 段階)
    assert!(
        e.to_lowercase().contains("parse"),
        "error should mention parsing:\n{e}"
    );
}

// ---------------------------------------------------------------------------
// 5. --compare の引数はちょうど 2 つ
// ---------------------------------------------------------------------------

#[test]
fn compare_requires_exactly_two_paths() {
    let td = TempDir::new("argc");
    let base = td.write("base.json", &report(1000.0, H1));

    let b = base.to_string_lossy().into_owned();
    let o = denno(&["bench", "--compare", &b], true);
    assert_ne!(o.status.code(), Some(0), "one path must be rejected");
    // clap は 2 値を要求して usage エラーで終わる (独自 bail には到達しない)
    let e = stderr(&o);
    assert!(
        e.contains("--compare") || e.contains("2"),
        "unexpected error:\n{e}"
    );
}
