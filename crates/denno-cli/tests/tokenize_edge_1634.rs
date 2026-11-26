//! task 1634: `denno tokenize` の引数相互作用 (排他 / 欠落 / 診断 / ヘルプ /
//! 不正フラグ) の境界試験。
//!
//! 既存の試験と重ならない縁だけを厚くする:
//! - `tokenize_941` は基本契約 (成功経路 / 終了コード 3・4 の代表例)、
//! - `tokenize_edge_1149` は UTF-8 とバイト数の境界、
//! - `tokenize_chat_1111` はチャットプレビューの描画、
//! - `cli_flags_edge_1339` は全サブコマンド共通の clap 挙動。
//!
//! ここは tokenize 1 本に絞り、次を固定する:
//!
//! 1. `--text` / `--stdin` / `--system` / `--user` / `--messages-json` の
//!    排他行列 (同時指定はすべて終了コード 2、stderr に両方のフラグ名)。
//! 2. `--dir` / `--text` / `--stdin` / `--json` の重複指定は
//!    `cannot be used multiple times` で終了コード 2。
//! 3. 値の欠落 (`--dir` / `--text` / `--system` / `--user` / `--messages-json`
//!    を argv 末尾に置く) は `a value is required` で終了コード 2。
//! 4. `--help` / `-h` / `help tokenize` は終了コード 0 で tokenize の全フラグを
//!    列挙し、相反フラグが併記されていてもヘルプが勝つ。
//! 5. 未知フラグ (類似名の prefix / 短縮形 / トップレベルへのサブコマンド用
//!    フラグ) は `unexpected argument` で終了コード 2。`--json=1` のように
//!    真偽フラグへ値を付ける形と、余分な位置引数も終了コード 2。
//! 6. モデル不在の診断は終了コード 3 で `denno tokenize: ` 前置きの 1 行。
//! 7. 入力源の使い方エラーは GGUF を開くより前に返る (壊れた GGUF を置いても
//!    終了コード 2 のまま) — 2 が 3 に化けない。
//!
//! フィクスチャは一時ディレクトリだけを使う (実モデル / ネットワーク / GPU は
//! 触らない)。GGUF を組む必要は無い: 診断の順序は「壊れた GGUF を置いても
//! 使方エラーが勝つ」ことで確かめられる。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

const CLAP_ERROR: i32 = 2;
const LOAD_ERROR: i32 = 3;
const STDERR_PREFIX: &str = "denno tokenize: ";

// ===== 一時ディレクトリ =====

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
            "denno1634_{}_{}_{}_{}",
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

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
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

// ===== CLI 起動 =====

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    // tokenize は DENNO_* を読まないが、ホスト環境で挙動が変わらないよう隔離する。
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

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// clap の解析段で終わる入力の共通形。
fn assert_clap_error(args: &[&str], needles: &[&str]) {
    let out = denno(args);
    let err = stderr_of(&out);
    let shown = args.join(" ");
    assert_eq!(
        out.status.code(),
        Some(CLAP_ERROR),
        "`denno {shown}`: exit was {:?}, stderr={err}",
        out.status.code()
    );
    assert!(
        stdout_of(&out).is_empty(),
        "`denno {shown}`: parse error wrote to stdout: {}",
        stdout_of(&out)
    );
    assert!(
        err.contains("error:"),
        "`denno {shown}`: stderr has no `error:`: {err}"
    );
    for n in needles {
        assert!(
            err.contains(n),
            "`denno {shown}`: stderr lacks {n:?}: {err}"
        );
    }
    assert!(
        !err.contains("panicked"),
        "`denno {shown}`: panicked: {err}"
    );
}

/// tokenize::run 由来の使い方エラー (clap を通った後) の共通形。
fn assert_usage_error(out: &Output, needles: &[&str]) {
    assert_eq!(
        out.status.code(),
        Some(CLAP_ERROR),
        "usage error must be exit 2: {}",
        stderr_of(out)
    );
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(out);
    assert!(
        e.starts_with(STDERR_PREFIX),
        "stderr must be prefixed with {STDERR_PREFIX:?}: {e}"
    );
    assert!(!e.contains("panicked"), "must not panic: {e}");
    for n in needles {
        assert!(e.contains(n), "stderr lacks {n:?}: {e}");
    }
}

/// モデル / GGUF 不在の診断の共通形。
fn assert_load_error(out: &Output, needles: &[&str]) {
    assert_eq!(
        out.status.code(),
        Some(LOAD_ERROR),
        "load error must be exit 3: {}",
        stderr_of(out)
    );
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(out);
    assert!(
        e.starts_with(STDERR_PREFIX),
        "stderr must be prefixed with {STDERR_PREFIX:?}: {e}"
    );
    assert_eq!(
        e.lines().count(),
        1,
        "load diagnostic should be one line: {e}"
    );
    assert!(!e.contains("panicked"), "must not panic: {e}");
    for n in needles {
        assert!(e.contains(n), "stderr lacks {n:?}: {e}");
    }
}

/// マジックだけ正しい 4 バイトの GGUF (開けないことを確かめるためだけのもの)。
fn corrupt_gguf() -> Vec<u8> {
    b"GGUF".to_vec()
}

// ===== 1. 入力源の排他行列 =====

#[test]
fn text_and_stdin_are_mutually_exclusive() {
    assert_clap_error(
        &["tokenize", "--dir", "/nonexistent-1634", "--text", "a", "--stdin"],
        &["cannot be used with", "--text", "--stdin"],
    );
}

#[test]
fn text_conflicts_with_every_chat_input() {
    let td = TempDir::new("textchat");
    let msgs = td.join("m.json");
    let msgs = msgs.to_str().expect("utf8 path");
    for (flag, value) in [
        ("--system", "s"),
        ("--user", "u"),
        ("--messages-json", msgs),
    ] {
        assert_clap_error(
            &[
                "tokenize",
                "--dir",
                "/nonexistent-1634",
                "--text",
                "a",
                flag,
                value,
            ],
            &["cannot be used with", "--text", flag],
        );
    }
}

#[test]
fn stdin_conflicts_with_every_chat_input() {
    let td = TempDir::new("stdinchat");
    let msgs = td.join("m.json");
    let msgs = msgs.to_str().expect("utf8 path");
    for (flag, value) in [
        ("--system", "s"),
        ("--user", "u"),
        ("--messages-json", msgs),
    ] {
        assert_clap_error(
            &[
                "tokenize",
                "--dir",
                "/nonexistent-1634",
                "--stdin",
                flag,
                value,
            ],
            &["cannot be used with", "--stdin", flag],
        );
    }
}

#[test]
fn chat_inputs_conflict_with_messages_json() {
    let td = TempDir::new("chatmsg");
    let msgs = td.join("m.json");
    let msgs = msgs.to_str().expect("utf8 path");
    for (flag, value) in [("--system", "s"), ("--user", "u")] {
        assert_clap_error(
            &[
                "tokenize",
                "--dir",
                "/nonexistent-1634",
                flag,
                value,
                "--messages-json",
                msgs,
            ],
            &["cannot be used with", flag, "--messages-json"],
        );
    }
}

#[test]
fn system_and_user_together_are_not_a_conflict() {
    // この 2 つは組で使う前提 (chat 1111)。排他ではないので、使い方エラーには
    // ならずモデル読み込みまで進む (壊れた GGUF なので 3 で止まる)。
    let td = TempDir::new("sysuser");
    td.write("m.gguf", &corrupt_gguf());
    let out = denno(&[
        "tokenize",
        "--dir",
        td.as_path().to_str().expect("utf8 path"),
        "--system",
        "s",
        "--user",
        "u",
    ]);
    assert!(
        !stderr_of(&out).contains("cannot be used with"),
        "--system と --user を排他にしてはいけない: {}",
        stderr_of(&out)
    );
    assert_load_error(&out, &["GGUF"]);
}

// ===== 2. 重複指定 =====

#[test]
fn duplicated_flags_are_rejected() {
    let cases: &[(&[&str], &str)] = &[
        (
            &["tokenize", "--dir", "/a-1634", "--dir", "/b-1634", "--text", "a"],
            "--dir <DIR>",
        ),
        (
            &["tokenize", "--dir", "/a-1634", "--text", "a", "--text", "b"],
            "--text <TEXT>",
        ),
        (
            &["tokenize", "--dir", "/a-1634", "--stdin", "--stdin"],
            "--stdin",
        ),
        (
            &["tokenize", "--dir", "/a-1634", "--text", "a", "--json", "--json"],
            "--json",
        ),
    ];
    for (args, name) in cases {
        assert_clap_error(args, &["cannot be used multiple times", name]);
    }
}

// ===== 3. 値の欠落 / 必須フラグ =====

#[test]
fn missing_flag_values_are_rejected() {
    // argv の末尾で値が無い形 (次トークンが別フラグでも同じ扱い)。
    let cases: &[(&[&str], &str)] = &[
        (&["tokenize", "--dir"], "--dir <DIR>"),
        (&["tokenize", "--dir", "/x-1634", "--text"], "--text <TEXT>"),
        (
            &["tokenize", "--dir", "/x-1634", "--system"],
            "--system <SYSTEM>",
        ),
        (&["tokenize", "--dir", "/x-1634", "--user"], "--user <USER>"),
        (
            &["tokenize", "--dir", "/x-1634", "--messages-json"],
            "--messages-json <MESSAGES_JSON>",
        ),
    ];
    for (args, name) in cases {
        assert_clap_error(args, &["a value is required", name]);
    }
}

#[test]
fn bare_tokenize_reports_missing_dir() {
    assert_clap_error(
        &["tokenize"],
        &["required arguments were not provided", "--dir <DIR>"],
    );
    // 別フラグだけを渡しても --dir の欠落は同じ形で返る。
    assert_clap_error(
        &["tokenize", "--text", "a"],
        &["required arguments were not provided", "--dir <DIR>"],
    );
}

// ===== 4. ダッシュ始まりの値は = 形が必要 =====

#[test]
fn dash_leading_value_needs_equals_form() {
    // `--text -5` は -5 をフラグ様トークンとして拒否する (解析段で 2)。
    assert_clap_error(
        &["tokenize", "--dir", "/nonexistent-1634", "--text", "-5"],
        &["unexpected argument", "-5"],
    );
    // `--text=-5` なら値として受理され、モデル解決まで進む (3)。
    let out = denno(&["tokenize", "--dir", "/nonexistent-1634", "--text=-5"]);
    assert_load_error(&out, &["存在しない"]);
}

#[test]
fn empty_dir_value_is_rejected_by_the_path_parser() {
    // 空の PathBuf は値として認めない (空文字の --text とは扱いが違う)。
    assert_clap_error(
        &["tokenize", "--dir", "", "--text", "a"],
        &["a value is required", "--dir <DIR>"],
    );
}

// ===== 5. ヘルプ =====

fn assert_help_ok(args: &[&str]) {
    let out = denno(args);
    let shown = args.join(" ");
    assert!(
        out.status.success(),
        "`denno {shown}`: exit was {:?}, stderr={}",
        out.status.code(),
        stderr_of(&out)
    );
    let text = stdout_of(&out);
    assert!(!text.trim().is_empty(), "`denno {shown}`: empty help");
    assert!(
        text.contains("Usage: denno tokenize"),
        "`denno {shown}`: missing tokenize usage line: {text}"
    );
    for flag in [
        "--dir <DIR>",
        "--text <TEXT>",
        "--stdin",
        "--system <SYSTEM>",
        "--user <USER>",
        "--messages-json <MESSAGES_JSON>",
        "--json",
    ] {
        assert!(
            text.contains(flag),
            "`denno {shown}`: help does not document {flag:?}: {text}"
        );
    }
    assert!(
        text.contains("-h, --help"),
        "`denno {shown}`: short help flag is not documented: {text}"
    );
}

#[test]
fn help_lists_every_tokenize_flag() {
    assert_help_ok(&["tokenize", "--help"]);
    assert_help_ok(&["tokenize", "-h"]);
    assert_help_ok(&["help", "tokenize"]);
}

#[test]
fn help_wins_over_conflicting_flags() {
    // 相反フラグが並んでいてもヘルプ要求が優先される (終了コード 0)。
    for args in [
        &["tokenize", "--help", "--text", "a", "--stdin"][..],
        &["tokenize", "--dir", "/nonexistent-1634", "-h", "--stdin"][..],
    ] {
        let out = denno(args);
        assert!(
            out.status.success(),
            "`denno {}` must succeed: exit {:?}, stderr={}",
            args.join(" "),
            out.status.code(),
            stderr_of(&out)
        );
        assert!(
            stdout_of(&out).contains("Usage: denno tokenize"),
            "help text missing for `denno {}`",
            args.join(" ")
        );
        assert!(
            !stderr_of(&out).contains("error:"),
            "help request wrote an error: {}",
            stderr_of(&out)
        );
    }
}

// ===== 6. 不正フラグ =====

#[test]
fn unknown_flags_are_rejected() {
    // 完全一致のみ受理する (prefix 補完もしない、短縮形も無い)。
    for (args, needle) in [
        (
            &["tokenize", "--dir", "/x-1634", "--text", "a", "--bogus-1634"][..],
            "--bogus-1634",
        ),
        (
            &["tokenize", "--di", "/x-1634", "--text", "a"][..],
            "--di",
        ),
        (&["tokenize", "--dir", "/x-1634", "--text", "a", "-j"][..], "-j"),
        (&["--json", "tokenize"][..], "--json"),
    ] {
        assert_clap_error(args, &["unexpected argument", needle]);
    }
}

#[test]
fn json_flag_takes_no_value() {
    assert_clap_error(
        &["tokenize", "--dir", "/x-1634", "--text", "a", "--json=1"],
        &["unexpected value", "'1'", "--json"],
    );
}

#[test]
fn stray_positional_argument_is_rejected() {
    assert_clap_error(
        &["tokenize", "--dir", "/x-1634", "--text", "a", "extra"],
        &["unexpected argument", "extra"],
    );
}

// ===== 7. モデル不在の診断 =====

#[test]
fn missing_dir_is_a_load_error_that_echoes_the_path() {
    let td = TempDir::new("nodir");
    let p = td.join("nope-1634");
    let out = denno(&[
        "tokenize",
        "--dir",
        p.to_str().expect("utf8 path"),
        "--text",
        "a",
    ]);
    assert_load_error(&out, &["存在しない", "nope-1634"]);
}

#[test]
fn empty_and_nongguf_dirs_are_load_errors() {
    // 存在するが空 / .gguf 以外しか無いディレクトリ。
    let empty = TempDir::new("emptydir");
    let out = denno(&[
        "tokenize",
        "--dir",
        empty.as_path().to_str().expect("utf8 path"),
        "--text",
        "a",
    ]);
    assert_load_error(&out, &[".gguf"]);

    let txt = TempDir::new("txtonly");
    txt.write("readme.txt", b"not a model");
    txt.write("blob.bin", b"\x00\x01");
    let out = denno(&[
        "tokenize",
        "--dir",
        txt.as_path().to_str().expect("utf8 path"),
        "--text",
        "a",
    ]);
    assert_load_error(&out, &[".gguf"]);
}

#[test]
fn non_gguf_file_passed_as_dir_is_a_load_error() {
    // ファイルを直接渡す経路は「開けるか」で判定する (拡張子は問わない)。
    let td = TempDir::new("plainfile");
    let p = td.write("model.bin", b"definitely not a gguf");
    let out = denno(&[
        "tokenize",
        "--dir",
        p.to_str().expect("utf8 path"),
        "--text",
        "a",
    ]);
    assert_load_error(&out, &["GGUF を開けない"]);
}

// ===== 8. 使い方エラーはモデル読み込みより先 =====

#[test]
fn usage_errors_precede_model_loading() {
    // 壊れた GGUF を置いておく。使い方の誤りはこれを開く前に返るはず。
    let td = TempDir::new("precedence");
    td.write("m.gguf", &corrupt_gguf());
    let dir = td.as_path().to_str().expect("utf8 path");
    let missing_msgs = td.join("absent-1634.json");
    let missing_msgs = missing_msgs.to_str().expect("utf8 path");

    // --system 単独 (--user が無い) は 2。GGUF の読み込みに進んだら 3 になる。
    let out = denno(&["tokenize", "--dir", dir, "--system", "s"]);
    assert_usage_error(&out, &["--user"]);

    // 読めない --messages-json も 2。
    let out = denno(&["tokenize", "--dir", dir, "--messages-json", missing_msgs]);
    assert_usage_error(&out, &["--messages-json", "読めない"]);

    // 入力源そのものが無い場合も 2 (ディレクトリの存在より先に診断する)。
    let out = denno(&["tokenize", "--dir", "/nonexistent-1634"]);
    assert_usage_error(&out, &["--text", "--stdin"]);
}
