//! task 1081: `denno version [--json]` の CLI 契約の試験。
//!
//! 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動して次を確かめる:
//!
//! 1. テキスト出力は固定キーの 5 行 (`denno <ver>` / `build_sha:` /
//!    `features:` / `target:` / `profile:`) で、終了コード 0。
//! 2. `--json` は stdout が単体で解析できる JSON オブジェクト 1 個で、
//!    キーは version / build_sha / features / target_arch / target_os /
//!    profile だけ。stderr は空。
//! 3. version は crate のバージョン (`CARGO_PKG_VERSION`) と一致する。
//! 4. features は cuda / xe の部分集合 (昇順)。既定ビルドでは空。
//! 5. 秘密・ホスト情報 (メール風の `@`、ループバック URL、HOME / USER の
//!    値、カレントディレクトリ) が出力に混ざらない。
//!
//! `denno version` はビルド時の定数を印字するだけで、モデルもハードウェアも
//! 環境変数も読まない。したがってこの環境で安全に起動できる。

use std::process::{Command, Output};

const KNOWN_FEATURES: [&str; 2] = ["cuda", "xe"];

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    // 出力が env に左右されないことを示すため、代表的な DENNO_* を消す。
    for k in [
        "DENNO_FLASH_DIR",
        "DENNO_FLASH_HOST",
        "DENNO_FLASH_PORT",
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

/// 出力に「ホスト固有の値」が混ざっていないこと。
fn assert_no_host_leak(text: &str, what: &str) {
    assert!(!text.contains('@'), "{what}: output contains '@': {text}");
    assert!(!text.contains("127.0.0.1"), "{what}: loopback IP leaked");
    assert!(!text.contains("localhost"), "{what}: 'localhost' leaked");
    if let Ok(home) = std::env::var("HOME") {
        if home.len() > 1 {
            assert!(!text.contains(&home), "{what}: HOME path leaked: {text}");
        }
    }
    if let Ok(user) = std::env::var("USER") {
        if user.len() > 2 {
            assert!(
                !text.contains(&format!("{user}@")),
                "{what}: user@host leaked: {text}"
            );
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cwd = cwd.to_string_lossy().to_string();
        // ビルドディレクトリは /work/... のような深いパス。短い "/" は無視。
        if cwd.len() > 3 {
            assert!(!text.contains(&cwd), "{what}: cwd leaked: {text}");
        }
    }
}

#[test]
fn text_output_has_fixed_keys() {
    let o = denno(&["version"]);
    assert!(
        o.status.success(),
        "denno version: exit {:?}, stderr={}",
        o.status.code(),
        stderr(&o)
    );
    let out = stdout(&o);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 5, "expected 5 lines, got: {out}");
    assert_eq!(
        lines[0],
        format!("denno {}", env!("CARGO_PKG_VERSION")),
        "unexpected banner: {out}"
    );
    assert!(lines[1].starts_with("build_sha: "), "got: {out}");
    assert!(!lines[1].trim_end_matches('\n').ends_with(": "), "empty sha: {out}");
    assert!(lines[2].starts_with("features: "), "got: {out}");
    assert!(lines[3].starts_with("target: "), "got: {out}");
    assert!(
        lines[4] == "profile: release" || lines[4] == "profile: debug",
        "unexpected profile line: {out}"
    );
    assert_no_host_leak(&out, "text");
}

#[test]
fn text_features_line_is_none_or_known_list() {
    let o = denno(&["version"]);
    assert!(o.status.success(), "denno version failed");
    let out = stdout(&o);
    let line = out
        .lines()
        .find(|l| l.starts_with("features: "))
        .expect("no features line");
    let value = line.trim_start_matches("features: ");
    if value == "(none)" {
        return;
    }
    let names: Vec<&str> = value.split(',').collect();
    assert!(!names.is_empty(), "empty feature list: {out}");
    for n in &names {
        assert!(KNOWN_FEATURES.contains(n), "unknown feature `{n}`: {out}");
    }
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(names, sorted, "features are not sorted/unique: {out}");
}

#[test]
fn json_output_is_single_parseable_object() {
    let o = denno(&["version", "--json"]);
    assert!(
        o.status.success(),
        "denno version --json: exit {:?}, stderr={}",
        o.status.code(),
        stderr(&o)
    );
    assert!(
        stderr(&o).trim().is_empty(),
        "--json wrote to stderr: {}",
        stderr(&o)
    );
    let out = stdout(&o);
    let v: serde_json::Value = serde_json::from_str(&out).expect("stdout is not one JSON value");
    let obj = v.as_object().expect("top-level is not an object");
    let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "build_sha",
            "features",
            "profile",
            "target_arch",
            "target_os",
            "version"
        ],
        "unexpected key set: {out}"
    );

    assert_eq!(obj["version"].as_str(), Some(env!("CARGO_PKG_VERSION")));
    let sha = obj["build_sha"].as_str().expect("build_sha not a string");
    assert!(!sha.is_empty(), "build_sha is empty");
    assert!(
        sha == "unknown" || sha.chars().all(|c| c.is_ascii_hexdigit()),
        "build_sha is not hex/unknown: {sha}"
    );
    assert_eq!(obj["target_arch"].as_str(), Some(std::env::consts::ARCH));
    assert_eq!(obj["target_os"].as_str(), Some(std::env::consts::OS));
    let profile = obj["profile"].as_str().expect("profile not a string");
    assert!(profile == "release" || profile == "debug", "profile: {profile}");
    assert_no_host_leak(&out, "json");
}

#[test]
fn json_features_are_a_known_subset() {
    let o = denno(&["version", "--json"]);
    assert!(o.status.success(), "denno version --json failed");
    let v: serde_json::Value =
        serde_json::from_str(&stdout(&o)).expect("stdout is not one JSON value");
    let arr = v["features"].as_array().expect("features is not an array");
    let names: Vec<&str> = arr.iter().map(|x| x.as_str().expect("non-string")).collect();
    for n in &names {
        assert!(KNOWN_FEATURES.contains(n), "unknown feature `{n}`");
    }
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(names, sorted, "features are not sorted/unique: {names:?}");
}

#[test]
fn json_and_text_agree_on_version_and_features() {
    let text = stdout(&denno(&["version"]));
    let json_out = stdout(&denno(&["version", "--json"]));
    let v: serde_json::Value = serde_json::from_str(&json_out).expect("bad json");

    let text_ver = text
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("denno "))
        .expect("no banner");
    assert_eq!(Some(text_ver), v["version"].as_str());

    let text_features = text
        .lines()
        .find(|l| l.starts_with("features: "))
        .and_then(|l| l.strip_prefix("features: "))
        .expect("no features line");
    let json_features: Vec<String> = v["features"]
        .as_array()
        .expect("features array")
        .iter()
        .map(|x| x.as_str().expect("string").to_string())
        .collect();
    let expected = if json_features.is_empty() {
        "(none)".to_string()
    } else {
        json_features.join(",")
    };
    assert_eq!(text_features, expected);
}

#[test]
fn version_subcommand_help_succeeds() {
    let o = denno(&["version", "--help"]);
    assert!(
        o.status.success(),
        "denno version --help: exit {:?}",
        o.status.code()
    );
    let out = stdout(&o);
    assert!(out.contains("Usage:"), "no Usage section: {out}");
    assert!(out.contains("--json"), "help does not mention --json: {out}");
}

#[test]
fn rejects_extra_positional_argument() {
    let o = denno(&["version", "extra-1081"]);
    assert!(!o.status.success(), "unexpected positional was accepted");
    assert!(
        !stderr(&o).trim().is_empty(),
        "no diagnostic on stderr for extra argument"
    );
}
