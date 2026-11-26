//! task 1339: サブコマンドのフラグ優先順位と不正引数の境界試験。
//!
//! 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動し、clap の解析段で
//! 終わる入力だけを与える (ポートを bind せず、GGUF も読まない)。
//! 固定する契約:
//!
//! 1. 全サブコマンドが未知フラグを終了コード 2 で拒否し、stderr に
//!    `error:` と該当フラグ名を出す (stdout は空)。
//! 2. argv 末尾で値が欠けたフラグ (`--port` など) は
//!    `a value is required` で終了コード 2。
//! 3. 必須引数の欠落は `required arguments were not provided` で終了コード 2。
//! 4. 範囲外・型不一致の数値 (`--port 70000`、`--max-body-mb 0` など) と
//!    列挙値違反 (`--log-format xml`) は `invalid value` で終了コード 2。
//! 5. 相反するフラグ (`--text` と `--stdin`) は `cannot be used with` で
//!    終了コード 2。
//! 6. 全サブコマンドで `--help` と `-h` が終了コード 0 / 非空 stdout。
//! 7. `--version` / `-V` は終了コード 0。`version` というサブコマンドは無い。
//!
//! 実測メモ: `--port 0` と `--max-tokens 0` は clap に範囲検証が無く解析を
//! 通る (`--max-body-mb` だけが `1..` の range を持つ)。`--max-tokens 0` を
//! 与えた `generate` は解析を通り、モデル未取得の診断で終了コード 1 になる。
//! `--port 0` 側は `serve` がモデルを読む経路に入るため、この試験では起動しない
//! (実測はしていない)。最後の 1 試験だけがこの「解析を通る」事実を固定する。

use std::process::{Command, Output};

/// `src/main.rs` の `enum Commands` と 1 対 1 で対応する kebab-case 名。
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

/// clap の解析エラーは 2、実行時エラーは 1。
const CLAP_ERROR: i32 = 2;

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    // ホストの DENNO_* が解析結果に影響しないよう隔離する。
    for k in [
        "DENNO_FLASH_DIR",
        "DENNO_FLASH_HOST",
        "DENNO_FLASH_PORT",
        "DENNO_EXPERT_CACHE_MB",
        "DENNO_AUTO_CONFIG",
        "DENNO_API_KEYS",
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

/// 解析エラー共通の形: 終了コード 2、stdout 空、stderr に `error:` と
/// 指定の断片。断片は利用者が次の行動を取れる文言に限る。
fn assert_clap_error(args: &[&str], needles: &[&str]) {
    let o = denno(args);
    let err = stderr(&o);
    let out = stdout(&o);
    assert_eq!(
        o.status.code(),
        Some(CLAP_ERROR),
        "`denno {}`: exit was {:?}, stderr={err}",
        args.join(" "),
        o.status.code()
    );
    assert!(
        out.is_empty(),
        "`denno {}`: parse error wrote to stdout: {out}",
        args.join(" ")
    );
    assert!(
        err.contains("error:"),
        "`denno {}`: stderr has no `error:` prefix: {err}",
        args.join(" ")
    );
    for needle in needles {
        assert!(
            err.contains(needle),
            "`denno {}`: stderr lacks `{needle}`: {err}",
            args.join(" ")
        );
    }
    // 解析エラーで panic してはならない (診断が使えなくなる)。
    assert!(
        !err.contains("panicked"),
        "`denno {}`: panicked: {err}",
        args.join(" ")
    );
}

// ------------------------------------------------ 1. 未知フラグ

#[test]
fn every_subcommand_rejects_unknown_flag() {
    for sub in SUBCOMMANDS {
        let o = denno(&[sub, "--definitely-not-a-flag-1339"]);
        let err = stderr(&o);
        assert_eq!(
            o.status.code(),
            Some(CLAP_ERROR),
            "`denno {sub} --definitely-not-a-flag-1339`: exit was {:?}, stderr={err}",
            o.status.code()
        );
        assert!(
            stdout(&o).is_empty(),
            "`denno {sub}`: unknown flag wrote to stdout"
        );
        assert!(
            err.contains("unexpected argument"),
            "`denno {sub}`: stderr lacks `unexpected argument`: {err}"
        );
        // 該当フラグ名をそのまま返す (どのフラグが悪いか特定できる)。
        assert!(
            err.contains("--definitely-not-a-flag-1339"),
            "`denno {sub}`: stderr does not name the offending flag: {err}"
        );
    }
}

#[test]
fn unknown_top_level_flag_and_subcommand_fail() {
    assert_clap_error(&["--definitely-not-a-flag-1339"], &["unexpected argument"]);
    assert_clap_error(
        &["definitely-not-a-subcommand-1339"],
        &["unrecognized subcommand"],
    );
}

// ------------------------------------------------ 2. 値の欠落

#[test]
fn missing_flag_value_at_end_of_argv_fails_cleanly() {
    // (argv, 欠けているフラグの表記) — 値が無いまま終端する形。
    let cases: &[(&[&str], &str)] = &[
        (&["serve", "--port"], "--port"),
        (&["serve", "--max-queue-depth"], "--max-queue-depth"),
        (&["serve-flash", "--port"], "--port"),
        (&["serve-flash", "--dir"], "--dir"),
        (&["bench", "--tg"], "--tg"),
        (&["bench", "--pp"], "--pp"),
        (&["chat", "--max-tokens"], "--max-tokens"),
        (&["abliterate", "--pos-file"], "--pos-file"),
        (&["doctor", "--dir"], "--dir"),
        (&["tokenize", "--dir"], "--dir"),
    ];
    for (args, flag) in cases {
        assert_clap_error(args, &["a value is required", flag]);
    }
}

// ------------------------------------------------ 3. 必須引数の欠落

#[test]
fn missing_required_arguments_fails_cleanly() {
    let cases: &[(&[&str], &str)] = &[
        (&["serve"], "<REPO>"),
        (&["pull"], "<REPO>"),
        (&["info"], "<REPO>"),
        (&["generate"], "--prompt"),
        (&["inspect"], "<PATH>"),
        (&["tokenize"], "--dir"),
        (&["abliterate"], "--pos-file"),
        (&["abliterate"], "--neg-file"),
    ];
    for (args, missing) in cases {
        assert_clap_error(
            args,
            &["required arguments were not provided", missing],
        );
    }
}

// ------------------------------------------------ 4. 範囲外 / 型不一致

#[test]
fn out_of_range_and_wrong_type_numerics_fail_cleanly() {
    let cases: &[&[&str]] = &[
        // u16 の上限超え
        &["serve", "repo/x", "--port", "70000"],
        &["serve-flash", "--port", "70000"],
        // usize の桁あふれ
        &["bench", "--tg", "99999999999999999999999999"],
        // 数値でない
        &["serve-flash", "--kv-slots", "abc"],
        &["bench", "--pp", "abc"],
        // 明示 range (`1..`) の下限割れ
        &["serve", "repo/x", "--max-body-mb", "0"],
    ];
    for args in cases {
        assert_clap_error(args, &["invalid value"]);
    }
}

#[test]
fn enum_valued_flag_rejects_unknown_variant() {
    assert_clap_error(
        &["serve", "repo/x", "--log-format", "xml"],
        &["invalid value", "possible values", "text", "json"],
    );
}

#[test]
fn negative_number_is_not_swallowed_as_a_value() {
    // `-5` は値ではなくフラグ様のトークンとして拒否される。
    assert_clap_error(&["chat", "--timeout", "-5"], &["unexpected argument"]);
}

// ------------------------------------------------ 5. 相反するフラグ

#[test]
fn conflicting_flags_are_rejected() {
    assert_clap_error(
        &[
            "tokenize",
            "--dir",
            ".",
            "--text",
            "hello",
            "--stdin",
        ],
        &["cannot be used with"],
    );
}

// ------------------------------------------------ 6. ヘルプ

#[test]
fn every_subcommand_help_succeeds_for_long_and_short() {
    for sub in SUBCOMMANDS {
        for flag in ["--help", "-h"] {
            let o = denno(&[sub, flag]);
            let out = stdout(&o);
            assert!(
                o.status.success(),
                "`denno {sub} {flag}`: exit was {:?}, stderr={}",
                o.status.code(),
                stderr(&o)
            );
            assert!(
                !out.trim().is_empty(),
                "`denno {sub} {flag}`: stdout is empty"
            );
            assert!(
                out.contains("Usage:") && out.contains(sub),
                "`denno {sub} {flag}`: usage text does not mention `{sub}`: {out}"
            );
        }
    }
}

#[test]
fn top_level_help_succeeds_for_long_and_short() {
    for flag in ["--help", "-h"] {
        let o = denno(&[flag]);
        let out = stdout(&o);
        assert!(
            o.status.success(),
            "`denno {flag}`: exit was {:?}",
            o.status.code()
        );
        for sub in SUBCOMMANDS {
            assert!(
                out.contains(sub),
                "`denno {flag}`: does not list `{sub}`"
            );
        }
    }
}

// ------------------------------------------------ 7. バージョン

#[test]
fn version_flags_succeed_and_version_subcommand_is_absent() {
    for flag in ["--version", "-V"] {
        let o = denno(&[flag]);
        let out = stdout(&o);
        assert!(
            o.status.success(),
            "`denno {flag}`: exit was {:?}",
            o.status.code()
        );
        assert!(
            out.starts_with("denno "),
            "`denno {flag}`: unexpected version banner: {out}"
        );
        assert!(
            out.trim().len() > "denno ".len(),
            "`denno {flag}`: version number is empty: {out}"
        );
    }
    // `version` はサブコマンドとしては存在しない (clap は exit 2)。
    assert_clap_error(&["version"], &["unrecognized subcommand"]);
}

// ------------------------------------------------ 解析を通る 0 値 (実測の固定)

#[test]
fn zero_max_tokens_passes_parse_then_fails_at_model_resolution() {
    // clap に範囲検証が無いので解析は通る。実サーバーや実 GGUF には触れず、
    // 未取得モデルの診断で非 0 終了することを固定する。
    let o = denno(&[
        "generate",
        "denno-edge-1339/definitely-not-cached",
        "--prompt",
        "hi",
        "--max-tokens",
        "0",
    ]);
    let err = stderr(&o);
    assert!(
        !o.status.success(),
        "zero --max-tokens was accepted all the way through"
    );
    assert!(
        !err.trim().is_empty(),
        "zero --max-tokens: no diagnostic on stderr"
    );
    // 解析段の拒否ではないこと (「解析は通る」という契約の裏取り)。
    assert!(
        !err.contains("invalid value"),
        "zero --max-tokens: rejected by clap, expectation changed: {err}"
    );
    assert!(!err.contains("panicked"), "zero --max-tokens: panicked: {err}");
}
