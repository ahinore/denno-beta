//! task 1289: `denno bench --compare` のレポート読解・決定論・0 基準・
//! 巨大デルタの境界を実バイナリで確かめる。
//!
//! 対象は 4 つ:
//!
//! 1. 部分 JSON。行やトップレベルのブロックが欠けていても既定値で受け、
//!    省略された数値は 0.0 として表に出ること (黙って落ちない)。
//!    未知フィールドは前方互換のため無視されること。型違いは明白なパースエラー。
//! 2. determinism ブロックの欠落。片方/両方に無い場合と `error` 付きの場合。
//! 3. 基準 0 の除算。`delta_pct` を出さず `-` (NoValue) にし、inf/NaN を
//!    表にも JSON にも漏らさず、退行にも数えないこと。
//! 4. 巨大デルタでも Markdown 表の列数がずれないこと。
//!
//! ベンチ本体は実行しない (サーバー・GPU 不要)。レポートは手書きの最小 JSON。

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
            "denno1289_{}_{}_{}_{}",
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

/// 省略可能なフィールドはすべて埋めた「正常」レポート。
fn full_report(pp_tok_s: f64, tg_tok_s: f64, speedup: f64, md5: &str) -> String {
    format!(
        r#"{{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "fixture-model",
  "rounds": 3,
  "seed": 801,
  "created_unix": 1700000000,
  "pp": [{{"pp_tokens": 512, "depth": 0, "total_tokens": 512, "ttft_ms": 4.0,
          "pp_tok_s": {pp_tok_s}, "ttft_runs_ms": [4.0, 4.1]}}],
  "tg": [{{"depth": 0, "gen_tokens": 128, "prompt_tokens": 0, "ttft_ms": 10.0,
          "tg_tok_s": {tg_tok_s}, "runs_tok_s": [{tg_tok_s}]}}],
  "turn2": {{"depth": 2048, "cold_ttft_ms": 800.0, "reuse_ttft_ms": 400.0,
             "speedup": {speedup}}},
  "determinism": {{"tokens": 512, "gen_tokens": 16, "md5": "{md5}",
                   "verified": true, "round_md5": ["{md5}", "{md5}"]}},
  "errors": []
}}"#
    )
}

/// Markdown の表 (連続する `|` 始まりの行) を順に返す。
fn pipe_tables(text: &str) -> Vec<Vec<String>> {
    let mut tables: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.starts_with('|') {
            cur.push(line.to_string());
        } else if !cur.is_empty() {
            tables.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tables.push(cur);
    }
    tables
}

/// `| a | b |` → ["a", "b"]
fn cells(line: &str) -> Vec<String> {
    line.trim_matches('|')
        .split('|')
        .map(|c| c.trim().to_string())
        .collect()
}

/// 表の全行が同じ列数であること (巨大デルタで桁が伸びても崩れない)。
fn assert_aligned(table: &[String], expected_cols: usize, tag: &str) {
    for line in table {
        let got = cells(line).len();
        assert_eq!(
            got, expected_cols,
            "{tag}: column count changed in line {line:?} (table:\n{}\n)",
            table.join("\n")
        );
    }
}

fn assert_no_non_finite(text: &str, tag: &str) {
    assert!(!text.contains("inf"), "{tag}: inf leaked:\n{text}");
    assert!(!text.contains("NaN"), "{tag}: NaN leaked:\n{text}");
    assert!(!text.contains("nan"), "{tag}: nan leaked:\n{text}");
}

// ---------------------------------------------------------------------------
// 1. 部分 JSON
// ---------------------------------------------------------------------------

#[test]
fn partial_rows_default_to_zero_without_panicking() {
    let td = TempDir::new("partial");
    let base = td.write("base.json", &full_report(1000.0, 50.0, 2.0, H1));
    // pp / tg の行はあるが、数値フィールドも turn2 / determinism も無い。
    let new = td.write(
        "new.json",
        r#"{
  "model": "fixture-model",
  "pp": [{"pp_tokens": 512, "depth": 0}],
  "tg": [{"gen_tokens": 128, "depth": 0}]
}"#,
    );
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], Some(&out_path));
    // 省略された pp_tok_s / tg_tok_s は 0.0 なので -100% = 退行になる。
    assert_eq!(
        o.status.code(),
        Some(1),
        "partial rows default to 0 and regress\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| pp512 @ d0 | pp_tok_s | 1000.000 | 0.000 | -100.00% | REGRESS |"),
        "partial pp row not reported as 0 -> -100%:\n{t}"
    );
    assert!(
        t.contains("| tg128 @ d0 | tg_tok_s | 50.000 | 0.000 | -100.00% | REGRESS |"),
        "partial tg row not reported as 0 -> -100%:\n{t}"
    );
    // turn2 と determinism は new に無いので missing。
    assert_eq!(t.matches("| missing |").count(), 2, "expected 2 missing rows:\n{t}");
    assert!(
        t.contains("| determinism md5 | 0123456789abcdef0123456789abcdef | - | - | missing |"),
        "determinism row not marked missing:\n{t}"
    );
    assert_no_non_finite(&t, "partial");

    // 表の列数は揃っている。
    let tables = pipe_tables(&t);
    assert_eq!(tables.len(), 2, "expected 2 tables:\n{t}");
    assert_aligned(&tables[0], 6, "partial metrics");
    assert_aligned(&tables[1], 5, "partial checks");

    // 要約 JSON も同じ内容 (0.0 と -100.0、欠落は null)。
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    let rows = v["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["new"], 0.0);
    assert_eq!(rows[0]["delta_pct"], -100.0);
    assert_eq!(rows[0]["verdict"], "REGRESS");
    assert!(v["turn2"]["base"].is_number());
    assert!(v["turn2"]["new"].is_null());
    assert_eq!(v["turn2"]["verdict"], "missing");
    assert_eq!(v["determinism"], "missing");
    assert!(v["new"]["md5"].is_null());
    assert_eq!(v["exit_fail"], true);
}

#[test]
fn unknown_fields_are_ignored_for_forward_compatibility() {
    let td = TempDir::new("unknown");
    let base = td.write("base.json", &full_report(1000.0, 50.0, 2.0, H1));
    // 未来のスキーマが足したフィールド。未知フィールドで落ちてはいけない。
    let new = td.write(
        "new.json",
        r#"{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "fixture-model",
  "rounds": 3,
  "seed": 801,
  "future_schema": {"version": 2, "notes": ["a", "b"]},
  "pp": [{"pp_tokens": 512, "depth": 0, "pp_tok_s": 1000.0,
          "future_metric": 1234, "cache": {"hit": true}}],
  "tg": [{"depth": 0, "gen_tokens": 128, "tg_tok_s": 50.0, "future_metric": null}],
  "turn2": {"depth": 2048, "speedup": 2.0, "future_metric": "x"},
  "determinism": {"md5": "0123456789abcdef0123456789abcdef", "verified": true},
  "unknown_top": 42
}"#,
    );

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], None);
    assert_eq!(
        o.status.code(),
        Some(0),
        "unknown fields must be ignored\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(t.contains("regressions: 0"), "unexpected regression:\n{t}");
    assert!(t.contains("| ok |"), "no ok rows:\n{t}");
    assert!(t.contains("determinism md5 MATCH"), "no match line:\n{t}");
}

#[test]
fn wrong_field_types_are_explicit_parse_errors() {
    let td = TempDir::new("types");
    let base = td.write("base.json", &full_report(1000.0, 50.0, 2.0, H1));

    for (name, body) in [
        ("rounds.json", r#"{"model": "x", "rounds": "three"}"#),
        ("pp_shape.json", r#"{"model": "x", "pp": {"pp_tokens": 512}}"#),
    ] {
        let bad = td.write(name, body);
        let o = compare_cmd(&base, &bad, &[], None);
        assert_eq!(
            o.status.code(),
            Some(1),
            "{name} must fail\nstdout:\n{}\nstderr:\n{}",
            stdout(&o),
            stderr(&o)
        );
        assert!(
            stdout(&o).trim().is_empty(),
            "{name}: no table may be printed on a parse error"
        );
        let e = stderr(&o);
        assert!(e.contains("Error:"), "{name}: no error line:\n{e}");
        assert!(e.contains(name), "{name}: path not named:\n{e}");
        assert!(
            e.to_lowercase().contains("invalid type"),
            "{name}: expected an invalid-type parse error:\n{e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. determinism ブロックの欠落
// ---------------------------------------------------------------------------

#[test]
fn determinism_missing_in_both_reports_is_not_a_failure() {
    let td = TempDir::new("nodet");
    let body = |pp| {
        format!(
            r#"{{"model": "fixture-model", "rounds": 1,
  "pp": [{{"pp_tokens": 512, "depth": 0, "pp_tok_s": {pp}}}]}}"#
        )
    };
    let base = td.write("base.json", &body(1000.0));
    let new = td.write("new.json", &body(1000.0));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "0"], Some(&out_path));
    assert_eq!(
        o.status.code(),
        Some(0),
        "absent determinism must not fail the run\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| determinism md5 | - | - | - | missing |"),
        "determinism row should be all dashes + missing:\n{t}"
    );
    assert!(
        t.contains("determinism missing in one report"),
        "ANSI summary should say missing:\n{t}"
    );

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    assert_eq!(v["determinism"], "missing");
    assert!(v["base"]["md5"].is_null());
    assert!(v["new"]["md5"].is_null());
    assert_eq!(v["exit_fail"], false);
}

#[test]
fn determinism_error_is_treated_as_missing_not_mismatch() {
    let td = TempDir::new("detterr");
    let base = td.write(
        "base.json",
        &full_report(1000.0, 50.0, 2.0, H1).replace(
            r#""round_md5": ["0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"]"#,
            r#""round_md5": [], "error": "generation failed""#,
        ),
    );
    let new = td.write("new.json", &full_report(1000.0, 50.0, 2.0, H2));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], Some(&out_path));
    // md5 は違うが error 付きなので「値なし」。不一致として落としてはならない。
    assert_eq!(
        o.status.code(),
        Some(0),
        "determinism with error must not fail\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| determinism md5 | - | ffffffffffffffffffffffffffffffff | - | missing |"),
        "errored base determinism should render as '-' + missing:\n{t}"
    );
    assert!(
        !t.contains("MISMATCH"),
        "errored determinism must not be reported as MISMATCH:\n{t}"
    );

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    assert_eq!(v["determinism"], "missing");
    assert!(v["base"]["md5"].is_null());
    assert_eq!(v["new"]["md5"], H2);
    assert_eq!(v["exit_fail"], false);
}

#[test]
fn empty_determinism_objects_compare_as_matching_empty_md5() {
    // 現行契約: ブロックの有無だけを見て error を見る。空オブジェクトは
    // 「md5 = ""」という値として扱われ、両方空なら MATCH (欠落ではない)。
    let td = TempDir::new("detempty");
    let body = |md5_block: &str| {
        format!(
            r#"{{"model": "fixture-model", "rounds": 1,
  "pp": [{{"pp_tokens": 512, "depth": 0, "pp_tok_s": 1000.0}}],
  "determinism": {md5_block}}}"#
        )
    };
    let base = td.write("base.json", &body("{}"));
    let new = td.write("new.json", &body("{}"));

    let o = compare_cmd(&base, &new, &[], None);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(
        t.contains("| determinism md5 |  |  | - | MATCH |"),
        "empty determinism should compare as MATCH with empty hashes:\n{t}"
    );
    assert!(t.contains("determinism md5 MATCH"), "no MATCH line:\n{t}");
}

// ---------------------------------------------------------------------------
// 3. 基準 0 の除算
// ---------------------------------------------------------------------------

#[test]
fn zero_baseline_yields_no_delta_and_no_failure() {
    let td = TempDir::new("zerobase");
    let base = td.write("base.json", &full_report(0.0, 0.0, 0.0, H1));
    let new = td.write("new.json", &full_report(1000.0, 50.0, 2.0, H1));
    let out_path = td.join("summary.json");

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "0"], Some(&out_path));
    assert_eq!(
        o.status.code(),
        Some(0),
        "0 baseline must not regress or panic\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| pp512 @ d0 | pp_tok_s | 0.000 | 1000.000 | - | - |"),
        "0 baseline should have no delta:\n{t}"
    );
    assert!(
        t.contains("| tg128 @ d0 | tg_tok_s | 0.000 | 50.000 | - | - |"),
        "0 baseline tg should have no delta:\n{t}"
    );
    assert!(
        t.contains("| turn-2 KV reuse @ d2048 | 0.000 | 2.000 | - | - |"),
        "0 baseline turn2 should have no delta:\n{t}"
    );
    assert!(t.contains("regressions: 0"), "0 baseline invented a regression:\n{t}");
    assert_no_non_finite(&t, "zero-baseline");

    let tables = pipe_tables(&t);
    assert_aligned(&tables[0], 6, "zero-baseline metrics");
    assert_aligned(&tables[1], 5, "zero-baseline checks");

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    let rows = v["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2);
    for r in rows {
        assert!(r["delta_pct"].is_null(), "delta must be null, got {}", r);
        assert_eq!(r["verdict"], "-");
    }
    assert!(v["turn2"]["delta_pct"].is_null());
    assert_eq!(v["exit_fail"], false);
    // JSON テキスト自体に inf/NaN が無い。
    let text = std::fs::read_to_string(&out_path).unwrap();
    assert_no_non_finite(&text, "zero-baseline json");
}

#[test]
fn zero_baseline_and_zero_new_is_also_no_value() {
    let td = TempDir::new("zeroboth");
    let base = td.write("base.json", &full_report(0.0, 0.0, 0.0, H1));
    let new = td.write("new.json", &full_report(0.0, 0.0, 0.0, H1));

    let o = compare_cmd(&base, &new, &["--fail-on-regress-pct", "5"], None);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert_eq!(
        t.matches("| - | - |").count(),
        3,
        "all three rows should be 0 -> no delta:\n{t}"
    );
    assert!(t.contains("regressions: 0"), "no regressions expected:\n{t}");
    assert_no_non_finite(&t, "zero-both");
}

#[test]
fn overflowing_delta_folds_to_no_value() {
    // 1e-300 -> 1e300 は比が f64 の範囲を超える。inf を出さず NoValue にする。
    let td = TempDir::new("overflowdelta");
    let tiny = td.write(
        "tiny.json",
        r#"{"model": "fixture-model", "rounds": 1,
  "pp": [{"pp_tokens": 512, "depth": 0, "pp_tok_s": 1e-300}]}"#,
    );
    let huge = td.write(
        "huge.json",
        r#"{"model": "fixture-model", "rounds": 1,
  "pp": [{"pp_tokens": 512, "depth": 0, "pp_tok_s": 1e300}]}"#,
    );
    let out_path = td.join("summary.json");

    let o = compare_cmd(&tiny, &huge, &["--fail-on-regress-pct", "5"], Some(&out_path));
    assert_eq!(
        o.status.code(),
        Some(0),
        "overflowing delta must not fail\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    // 値そのものは巨大 (1e300 は 301 桁で印字される) が、比は f64 の範囲を超える
    // ので delta は `-`、判定も `-` (NoValue) になる。
    let tables = pipe_tables(&t);
    let row = cells(&tables[0][2]);
    assert_eq!(row[0], "pp512 @ d0", "unexpected row:\n{t}");
    assert_eq!(row[1], "pp_tok_s");
    assert_eq!(row[2], "0.000");
    assert_eq!(row[4], "-", "overflowing delta must be '-':\n{t}");
    assert_eq!(row[5], "-", "overflowing delta must be NoValue:\n{t}");
    assert!(t.contains("regressions: 0"), "overflow counted as regression:\n{t}");
    assert_no_non_finite(&t, "overflow-delta");

    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    assert!(v["rows"][0]["delta_pct"].is_null());
    assert_eq!(v["rows"][0]["verdict"], "-");
    assert_eq!(v["exit_fail"], false);
}

// ---------------------------------------------------------------------------
// 4. 巨大デルタの Markdown 整合
// ---------------------------------------------------------------------------

#[test]
fn large_delta_markdown_keeps_columns_aligned() {
    let td = TempDir::new("largedelta");
    let small = td.write("small.json", &full_report(1.0, 1.0, 0.001, H1));
    let large = td.write("large.json", &full_report(1e12, 1e12, 1000.0, H1));
    let out_path = td.join("summary.json");

    // 改善方向: 巨大な正のデルタ。閾値 5% では improve。
    let o = compare_cmd(&small, &large, &["--fail-on-regress-pct", "5"], Some(&out_path));
    assert_eq!(
        o.status.code(),
        Some(0),
        "large improvement must not fail\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| pp512 @ d0 | pp_tok_s | 1.000 | 1000000000000.000 | +99999999999900.00% | improve |"),
        "large positive delta not rendered as expected:\n{t}"
    );
    assert!(
        t.contains("| turn-2 KV reuse @ d2048 | 0.001 | 1000.000 | +99999900.00% | improve |"),
        "large turn2 delta not rendered as expected:\n{t}"
    );
    assert_no_non_finite(&t, "large-positive");

    let tables = pipe_tables(&t);
    assert_eq!(tables.len(), 2, "expected 2 tables:\n{t}");
    assert_aligned(&tables[0], 6, "large-positive metrics");
    assert_aligned(&tables[1], 5, "large-positive checks");

    // 退行方向: 巨大な負のデルタ。-100% に丸まり、表は崩れない。
    let o = compare_cmd(&large, &small, &["--fail-on-regress-pct", "5"], None);
    assert_eq!(
        o.status.code(),
        Some(1),
        "large regression must exit non-zero\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(
        t.contains("| pp512 @ d0 | pp_tok_s | 1000000000000.000 | 1.000 | -100.00% | REGRESS |"),
        "large negative delta not rendered as expected:\n{t}"
    );
    assert!(
        t.contains("REGRESS pp512 @ d0 pp_tok_s 1000000000000.000 -> 1.000 (-100.00%)"),
        "ANSI summary not aligned with the table:\n{t}"
    );
    assert_no_non_finite(&t, "large-negative");

    let tables = pipe_tables(&t);
    assert_aligned(&tables[0], 6, "large-negative metrics");
    assert_aligned(&tables[1], 5, "large-negative checks");

    // JSON 側も巨大値が数値のまま (文字列化や null 化をしない)。
    let o = compare_cmd(&small, &large, &[], Some(&out_path));
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("summary written"))
            .expect("valid JSON");
    assert_eq!(v["rows"][0]["base"], 1.0);
    assert_eq!(v["rows"][0]["new"], 1e12);
    assert_eq!(v["rows"][0]["delta_pct"], 99999999999900.0);
}
