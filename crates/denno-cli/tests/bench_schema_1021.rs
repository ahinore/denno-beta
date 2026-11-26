//! task 1021: `denno bench --compare` のレポート schema と seed の境界を
//! 実バイナリで確かめる。
//!
//! 対象は 3 つ:
//!
//! 1. レポート JSON の読解が NaN / ゼロ / 欠損で壊れないこと。
//!    serde_json は NaN を `null` で書くので、`null` を受ける必要がある。
//!    溢れる数値リテラル (`1e400`) は f64 で inf になるため、これも畳む。
//! 2. `--compare-json` の要約 JSON が機械可読 (schema / rows / verdict /
//!    seed / determinism) であること。`-` で stdout にも出せること。
//! 3. seed の扱い。両レポートが seed を記録して食い違うなら警告、
//!    片方でも未記録 (古いレポート) なら黙る。閾値 0 でも無変化は退行にしない。
//!
//! ベンチ本体は実行しない (サーバー・GPU 不要)。

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
            "denno1021_{}_{}_{}_{}",
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

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    cmd.env_remove("NO_COLOR");
    cmd.env("NO_COLOR", "1");
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

/// `bench [extra...] --compare BASE NEW [--compare-json PATH]`
fn compare_cmd(base: &Path, new: &Path, extra: &[&str], json_out: Option<&Path>) -> Output {
    let mut owned: Vec<String> = vec!["bench".to_string()];
    owned.extend(extra.iter().map(|s| s.to_string()));
    owned.push("--compare".to_string());
    owned.push(base.to_string_lossy().into_owned());
    owned.push(new.to_string_lossy().into_owned());
    if let Some(p) = json_out {
        owned.push("--compare-json".to_string());
        owned.push(p.to_string_lossy().into_owned());
    }
    let args: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    denno(&args)
}

/// 正常値のレポート。
fn report(pp_tok_s: f64, tg_tok_s: f64, seed: Option<u64>, md5: &str) -> String {
    let seed_field = match seed {
        Some(s) => format!(",\n  \"seed\": {s}"),
        None => String::new(),
    };
    format!(
        r#"{{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "fixture-model",
  "rounds": 3{seed_field},
  "pp": [{{"pp_tokens": 512, "depth": 0,
          "total_tokens": 512, "ttft_ms": 4.0, "pp_tok_s": {pp_tok_s},
          "ttft_runs_ms": [4.0, 4.1]}}],
  "tg": [{{"depth": 0, "gen_tokens": 128, "prompt_tokens": 0,
          "ttft_ms": 10.0,
          "tpot_ms": {{"n": 2, "mean": 20.0, "median": 20.0, "min": 19.0,
                       "max": 21.0, "stddev": 0.7, "p90": 21.0, "p99": 21.0}},
          "itl_ms": {{"n": 2, "mean": 21.0, "median": 21.0, "min": 20.0,
                      "max": 22.0, "stddev": 0.7, "p90": 22.0, "p99": 22.0}},
          "tg_tok_s": {tg_tok_s}, "runs_tok_s": [{tg_tok_s}]}}],
  "turn2": {{"depth": 2048, "cold_ttft_ms": 800.0, "reuse_ttft_ms": 400.0,
             "speedup": 2.0}},
  "determinism": {{"md5": "{md5}", "verified": true}},
  "errors": []
}}"#
    )
}

/// 非有限 / 欠損だらけのレポート。`null` は JSON で合法 (f64 では値を欠く)。
/// 溢れる数値リテラル (`1e400`) は serde_json がパース時に弾くのでここでは使わない
/// (`overflowing_number_is_a_parse_error` を参照)。
fn degenerate_report() -> String {
    r#"{
  "model": "degenerate",
  "rounds": 1,
  "pp": [{"pp_tokens": 512, "depth": 0, "ttft_ms": null, "pp_tok_s": null,
          "ttft_runs_ms": [null, 3.0]}],
  "tg": [{"depth": 0, "gen_tokens": 8, "ttft_ms": null, "tg_tok_s": null,
          "tpot_ms": {"n": 1, "mean": null, "median": null, "p99": null},
          "runs_tok_s": [null]}],
  "turn2": {"depth": 0, "speedup": null},
  "determinism": {}
}"#
    .to_string()
}

// ---------------------------------------------------------------------------
// 1. null / 溢れる数値を受けて比較が落ちない
// ---------------------------------------------------------------------------

#[test]
fn compare_accepts_null_and_overflowing_metrics() {
    let td = TempDir::new("degenerate");
    let base = td.write("base.json", &degenerate_report());
    let new = td.write("new.json", &degenerate_report());

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], None);
    assert_eq!(
        o.status.code(),
        Some(0),
        "degenerate but legal JSON must compare cleanly\nstderr:\n{}",
        stderr(&o)
    );
    let t = stdout(&o);
    // 比較不能は "-" で出る (inf/NaN を表に出さない)
    assert!(t.contains("| pp512 @ d0 | pp_tok_s |"), "no pp row:\n{t}");
    assert!(
        t.contains("| - | - | - | - |") || t.contains("| 0.000 | 0.000 | - | - |"),
        "no no-value row:\n{t}"
    );
    assert!(!t.contains("inf"), "inf leaked into markdown:\n{t}");
    assert!(!t.contains("NaN"), "NaN leaked into markdown:\n{t}");
    assert!(t.contains("regressions: 0"), "unexpected regressions:\n{t}");
}

// ---------------------------------------------------------------------------
// 2. --compare-json が要約 JSON を書き出す
// ---------------------------------------------------------------------------

#[test]
fn compare_json_writes_machine_readable_summary() {
    let td = TempDir::new("json");
    let base = td.write("base.json", &report(1000.0, 50.0, Some(801), H1));
    let new = td.write("new.json", &report(800.0, 60.0, Some(801), H2));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], Some(&out_path));
    // 退行 + md5 不一致なので終了は非 0。JSON は失敗時も書かれていること。
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));

    let text = std::fs::read_to_string(&out_path).expect("summary json written");
    let v: serde_json::Value = serde_json::from_str(&text).expect("summary must be valid JSON");
    assert_eq!(v["schema"], "denno.bench.compare/1");
    assert_eq!(v["threshold_pct"], 5.0);
    assert_eq!(v["determinism"], "mismatch");
    assert_eq!(v["exit_fail"], true);
    assert_eq!(v["regressions"], 1);
    assert_eq!(v["base"]["seed"], 801);
    assert_eq!(v["new"]["seed"], 801);
    assert_eq!(v["seed_mismatch"], false);
    assert_eq!(v["base"]["md5"], H1);
    assert_eq!(v["new"]["md5"], H2);

    let rows = v["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["case"], "pp512 @ d0");
    assert_eq!(rows[0]["verdict"], "REGRESS");
    let d = rows[0]["delta_pct"].as_f64().expect("numeric delta");
    assert!((d + 20.0).abs() < 1e-9, "delta was {d}");
    assert_eq!(rows[1]["verdict"], "improve");
    assert_eq!(v["turn2"]["verdict"], "ok");
    assert!(stderr(&o).contains("summary.json"), "no wrote line:\n{}", stderr(&o));
}

#[test]
fn compare_json_dash_writes_to_stdout() {
    let td = TempDir::new("dash");
    let base = td.write("base.json", &report(1000.0, 50.0, Some(801), H1));
    let new = td.write("new.json", &report(1000.0, 50.0, Some(801), H1));

    let o = compare_cmd(&base, &new, &[], Some(Path::new("-")));
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    // stdout には markdown 表と JSON が続けて出る。JSON は行頭の `{` から始まる
    // (markdown の表は `|` 始まりなので衝突しない)。キー順は serde_json の
    // BTreeMap 既定でアルファベット順になるため、先頭キーは仮定しない。
    let t = stdout(&o);
    let start = t.find("\n{\n").expect("no JSON block on stdout") + 1;
    let v: serde_json::Value = serde_json::from_str(&t[start..]).expect("trailing JSON parses");
    assert_eq!(v["schema"], "denno.bench.compare/1");
    assert_eq!(v["exit_fail"], false);
    assert_eq!(v["determinism"], "match");
}

#[test]
fn compare_json_without_compare_is_rejected() {
    let td = TempDir::new("nocmp");
    let out_path = td.join("summary.json");
    let p = out_path.to_string_lossy().into_owned();
    let o = denno(&["bench", "--compare-json", &p]);
    assert_ne!(o.status.code(), Some(0), "--compare-json alone must fail");
    let e = stderr(&o);
    assert!(
        e.contains("--compare-json requires --compare"),
        "unexpected error:\n{e}"
    );
    assert!(!out_path.exists(), "no file may be written on rejection");
}

// ---------------------------------------------------------------------------
// 3. seed の扱い
// ---------------------------------------------------------------------------

#[test]
fn seed_mismatch_warns_but_does_not_fail() {
    let td = TempDir::new("seed");
    let base = td.write("base.json", &report(1000.0, 50.0, Some(801), H1));
    let new = td.write("new.json", &report(1000.0, 50.0, Some(802), H1));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], Some(&out_path));
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("seed differs"), "no seed warning:\n{t}");
    assert!(t.contains("801") && t.contains("802"), "seeds not named:\n{t}");

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(v["seed_mismatch"], true);
    assert_eq!(v["base"]["seed"], 801);
    assert_eq!(v["new"]["seed"], 802);
}

#[test]
fn missing_seed_is_tolerated_silently() {
    let td = TempDir::new("oldseed");
    // 古いレポートは seed を記録していない。
    let base = td.write("base.json", &report(1000.0, 50.0, None, H1));
    let new = td.write("new.json", &report(1000.0, 50.0, Some(801), H1));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &[], Some(&out_path));
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    assert!(!stdout(&o).contains("seed differs"), "must not warn on None");

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(v["seed_mismatch"], false);
    assert!(v["base"]["seed"].is_null());
    assert_eq!(v["new"]["seed"], 801);
}

#[test]
fn zero_threshold_does_not_fail_identical_reports() {
    let td = TempDir::new("zero");
    let base = td.write("base.json", &report(1000.0, 50.0, Some(801), H1));
    let new = td.write("new.json", &report(1000.0, 50.0, Some(801), H1));

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "0"], None);
    assert_eq!(
        o.status.code(),
        Some(0),
        "identical reports must not regress at threshold 0\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(t.contains("| ok |"), "no ok row:\n{t}");
    assert!(!t.contains("| REGRESS |"), "unexpected REGRESS:\n{t}");
}

// ---------------------------------------------------------------------------
// 4. 壊れた数値リテラルは明確なパースエラー
// ---------------------------------------------------------------------------

#[test]
fn nan_literal_is_a_parse_error() {
    let td = TempDir::new("nanlit");
    let base = td.write("base.json", &report(1000.0, 50.0, None, H1));
    // JS 風の NaN は JSON ではない。黙って 0 にせず、パースエラーにする。
    let bad = td.write(
        "bad.json",
        r#"{"model":"x","pp":[{"pp_tokens":512,"depth":0,"pp_tok_s":NaN}]}"#,
    );

    let o = compare_cmd(&base, &bad, &[], None);
    assert_eq!(o.status.code(), Some(1), "NaN literal must fail");
    assert!(stdout(&o).trim().is_empty(), "no table on parse error");
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(e.to_lowercase().contains("parse"), "not a parse error:\n{e}");
    assert!(e.contains("bad.json"), "path not named:\n{e}");
}

#[test]
fn overflowing_number_is_a_parse_error() {
    let td = TempDir::new("overflow");
    let base = td.write("base.json", &report(1000.0, 50.0, None, H1));
    // f64 で表現できない数値リテラル。serde_json が "number out of range" で弾く。
    let bad = td.write(
        "bad.json",
        r#"{"model":"x","pp":[{"pp_tokens":512,"depth":0,"pp_tok_s":1e400}]}"#,
    );

    let o = compare_cmd(&base, &bad, &[], None);
    assert_eq!(o.status.code(), Some(1), "overflow must fail");
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no error line:\n{e}");
    assert!(e.contains("bad.json"), "path not named:\n{e}");
    assert!(
        e.contains("number out of range"),
        "expected an out-of-range parse error:\n{e}"
    );
}

#[test]
fn empty_object_report_compares_as_all_missing() {
    let td = TempDir::new("empty");
    let base = td.write("base.json", &report(1000.0, 50.0, None, H1));
    let empty = td.write("empty.json", "{}");

    let o = compare_cmd(&base, &empty, &["--fail-on-regress-pct", "5"], None);
    assert_eq!(
        o.status.code(),
        Some(0),
        "empty report is legal (all defaults)\nstderr:\n{}",
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(t.contains("| missing |"), "rows should be missing:\n{t}");
    assert!(t.contains("regressions: 0"), "no regressions expected:\n{t}");
}
