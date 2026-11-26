//! task 1224: `denno` のヘルプ表示とサブコマンド解析の煙試験。
//!
//! 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動し、次を確かめる:
//!
//! 1. `--help` が終了コード 0 / 非空 / 全サブコマンド名を列挙する。
//! 2. 各サブコマンドの `<sub> --help` が終了コード 0 / 非空。
//! 3. 不正なフラグ・未知のサブコマンドは非 0 で終わる。
//! 4. src/main.rs の `enum Commands` の全 variant が上の一覧に含まれる
//!    (新しいサブコマンドが増えたらこの試験が落ちる)。
//!
//! ヘルプは clap が組み立てるだけで、モデルもハードウェアも触らない。

use std::process::{Command, Output};

/// main.rs の `enum Commands` と 1 対 1 で対応する kebab-case 名。
/// `test_subcommands_match_main_rs` が乖離を検出する。
const SUBCOMMANDS: &[&str] = &[
    "pull",
    "models",
    "info",
    "generate",
    "serve-flash",
    "chat",
    "abliterate",
    "bench",
    "serve",
    "inspect",
    "tokenize",
    "doctor",
];

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    // ホストの DENNO_* で挙動が変わらないようにする (ヘルプには影響しないが
    // 念のため隔離する)。
    for k in [
        "DENNO_FLASH_DIR",
        "DENNO_FLASH_HOST",
        "DENNO_FLASH_PORT",
        "DENNO_EXPERT_CACHE_MB",
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

/// `--help` は stdout に出す (clap 既定)。exit 0 かつ stdout が非空。
fn assert_help_ok(o: &Output, what: &str) {
    assert!(
        o.status.success(),
        "{what}: exit was {:?}, stderr={}",
        o.status.code(),
        stderr(o)
    );
    let out = stdout(o);
    assert!(!out.trim().is_empty(), "{what}: stdout is empty");
    assert!(
        out.contains("Usage:"),
        "{what}: no Usage section in stdout: {out}"
    );
}

#[test]
fn top_level_help_lists_every_subcommand() {
    let o = denno(&["--help"]);
    assert_help_ok(&o, "denno --help");
    let out = stdout(&o);
    for sub in SUBCOMMANDS {
        assert!(
            out.contains(sub),
            "denno --help does not mention subcommand `{sub}`"
        );
    }
}

#[test]
fn every_subcommand_help_succeeds() {
    for sub in SUBCOMMANDS {
        let o = denno(&[sub, "--help"]);
        assert_help_ok(&o, &format!("denno {sub} --help"));
        let out = stdout(&o);
        assert!(
            out.contains("Usage:"),
            "denno {sub} --help: Usage section missing"
        );
    }
}

#[test]
fn invalid_top_level_flag_fails() {
    let o = denno(&["--definitely-not-a-flag-1224"]);
    assert!(!o.status.success(), "unknown flag was accepted");
    assert!(
        !stderr(&o).trim().is_empty(),
        "unknown flag produced no diagnostic on stderr"
    );
}

#[test]
fn invalid_subcommand_fails() {
    let o = denno(&["definitely-not-a-subcommand-1224"]);
    assert!(!o.status.success(), "unknown subcommand was accepted");
    assert!(
        !stderr(&o).trim().is_empty(),
        "unknown subcommand produced no diagnostic on stderr"
    );
}

#[test]
fn invalid_subcommand_flag_fails() {
    // `models` は引数を取らないので、未知フラグは必ず clap が拒否する。
    let o = denno(&["models", "--definitely-not-a-flag-1224"]);
    assert!(
        !o.status.success(),
        "unknown flag on `models` was accepted"
    );
    assert!(
        !stderr(&o).trim().is_empty(),
        "unknown flag on `models` produced no diagnostic on stderr"
    );
}

/// CamelCase の variant 名を clap 既定の kebab-case に直す。
fn kebab(variant: &str) -> String {
    let mut out = String::new();
    for (i, ch) in variant.char_indices() {
        if ch.is_ascii_uppercase() {
            if i != 0 {
                out.push('-');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// `src/main.rs` の `enum Commands { ... }` から variant 名を拾う。
/// 行頭 4 スペース + 大文字始まり + `{` または `,` で終わる行だけを見る
/// (doc コメント `///` とフィールド行は対象外)。
fn variants_in_main() -> Vec<String> {
    let src = include_str!("../src/main.rs");
    let mut in_enum = false;
    let mut found = Vec::new();
    for line in src.lines() {
        if !in_enum {
            if line.trim() == "enum Commands {" {
                in_enum = true;
            }
            continue;
        }
        if line == "}" {
            break;
        }
        let Some(rest) = line.strip_prefix("    ") else {
            continue;
        };
        if rest.starts_with(char::is_whitespace) {
            continue; // 8 スペース = フィールド
        }
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_uppercase()) {
            continue;
        }
        let after = &rest[name.len()..];
        if after.trim_start().starts_with('{') || after.trim_start().starts_with(',') {
            found.push(name);
        }
    }
    assert!(
        !found.is_empty(),
        "could not parse `enum Commands` variants from src/main.rs"
    );
    found
}

#[test]
fn subcommands_match_main_rs() {
    let mut expected: Vec<String> = variants_in_main().iter().map(|v| kebab(v)).collect();
    let mut listed: Vec<String> = SUBCOMMANDS.iter().map(|s| s.to_string()).collect();
    expected.sort();
    listed.sort();
    assert_eq!(
        listed, expected,
        "SUBCOMMANDS is out of sync with `enum Commands` in src/main.rs"
    );
}
