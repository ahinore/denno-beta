//! task 1284: `src/config.rs` の字句まわりの境界試験 (追補)。
//!
//! task 1001 の `config_edges_1001.rs` はエスケープ全形・行内コメント・
//! 解決順を広く押さえている。こちらは範囲を絞って次を固定する:
//!
//! 1. `\"` と `\n` の往復 (render 経由で貼り直しても同じ値になること)
//! 2. 単一引用符 (`'...'`) がリテラルであること (`\n` や `\\` を解釈しない)
//! 3. 空ファイル / コメントだけ / 空白だけのファイルが空の設定として通ること
//! 4. CRLF 改行 (`\r\n`) が LF と同じ結果になること
//! 5. 不正な UTF-8 を `load` が読めないエラーとして弾くこと
//!
//! denno-cli は bin クレートなので lib 経由で取り込めない。既存の
//! `config_906.rs` / `config_edges_1001.rs` と同じく `#[path]` で
//! `src/config.rs` をそのまま読み込む。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use config::{load, parse_str, resolve, Value};

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ。並列実行でも衝突しない名前。
fn tmpdir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "denno1284_{}_{}_{}",
        std::process::id(),
        n,
        tag
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ------------------------------------------------ エスケープ (\", \n) の往復

#[test]
fn escaped_quote_and_newline_round_trip_through_render() {
    // 設定ファイル -> 解析 -> render -> 再解析 で値が保たれること。
    let src = "name = \"a\\\"b\"\nhost = \"line1\\nline2\"\n";
    let file = parse_str(src, "t.toml").unwrap();
    assert_eq!(file.get("name"), Some(&Value::Str("a\"b".into())));
    assert_eq!(file.get("host"), Some(&Value::Str("line1\nline2".into())));

    let r = resolve(&[], &BTreeMap::new(), Some(&file)).unwrap();
    let rendered = r.render();
    // render は改行を `\n` へ、二重引用符を `\"` へ戻して 1 行に保つ。
    assert!(rendered.contains("name = \"a\\\"b\""), "{rendered}");
    assert!(rendered.contains("host = \"line1\\nline2\""), "{rendered}");

    let back = parse_str(&rendered, "roundtrip.toml").unwrap();
    assert_eq!(back.get("name"), Some(&Value::Str("a\"b".into())));
    assert_eq!(back.get("host"), Some(&Value::Str("line1\nline2".into())));
}

#[test]
fn doubled_backslash_before_quote_keeps_both() {
    // `\\\"` は「バックスラッシュ + 閉じ」ではなく「\\ のあと \"」として読む。
    let f = parse_str("name = \"x\\\\\\\"y\"\n", "t.toml").unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str("x\\\"y".into())));
}

#[test]
fn escape_only_strings_produce_the_control_characters() {
    let f = parse_str("name = \"\\n\\n\"\nhost = \"\\r\\t\"\n", "t.toml").unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str("\n\n".into())));
    assert_eq!(f.get("host"), Some(&Value::Str("\r\t".into())));
}

// ------------------------------------------------------- 単一引用符 (リテラル)

#[test]
fn single_quoted_strings_do_not_process_escapes() {
    // `\n` は 2 文字のまま、`\\` も 2 文字のまま。
    let f = parse_str(
        "name = 'a\\nb'\nhost = 'c\\\\d'\ndir = 'e\\tf'\n",
        "t.toml",
    )
    .unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str("a\\nb".into())));
    assert_eq!(f.get("host"), Some(&Value::Str("c\\\\d".into())));
    assert_eq!(f.get("dir"), Some(&Value::Str("e\\tf".into())));
}

#[test]
fn single_quoted_strings_keep_hash_and_double_quotes() {
    let f = parse_str("name = 'a # b'\nhost = 'say \"hi\"'\n", "t.toml").unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str("a # b".into())));
    assert_eq!(f.get("host"), Some(&Value::Str("say \"hi\"".into())));
}

#[test]
fn empty_single_quoted_string_is_allowed() {
    let f = parse_str("name = ''\nhost = \"\"\n", "t.toml").unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str(String::new())));
    assert_eq!(f.get("host"), Some(&Value::Str(String::new())));
}

#[test]
fn single_quoted_string_does_not_allow_a_quote_inside() {
    // `'a'b'` は `'a'` の後ろに余分な `b'` があるので弾かれる。
    let e = parse_str("name = 'a'b'\n", "t.toml").unwrap_err();
    assert!(e.contains("値の後ろに余分な文字"), "{e}");
}

// ---------------------------------------------------------- 空 / 実質空のファイル

#[test]
fn empty_file_is_an_empty_config() {
    let f = parse_str("", "empty.toml").unwrap();
    assert!(f.entries.is_empty());
    assert_eq!(f.origin, "empty.toml");
    assert_eq!(f.get("port"), None);
}

#[test]
fn whitespace_and_comment_only_files_are_empty() {
    for text in ["\n\n\n", "   \n\t\n", "# only a comment\n", "# c1\n\n  # c2\n"] {
        let f = parse_str(text, "blank.toml").unwrap();
        assert!(f.entries.is_empty(), "{text:?} から何か読めてしまった");
    }
}

#[test]
fn empty_file_resolves_to_defaults() {
    let f = parse_str("", "empty.toml").unwrap();
    let r = resolve(&[], &BTreeMap::new(), Some(&f)).unwrap();
    assert_eq!(r.string("port").as_deref(), Some("8098"));
    assert_eq!(r.string("host").as_deref(), Some("127.0.0.1"));
    assert_eq!(r.slot("dir"), None);
}

#[test]
fn loading_a_real_empty_file_succeeds() {
    let dir = tmpdir("empty");
    let path = dir.join("empty.toml");
    std::fs::write(&path, "").unwrap();
    let f = load(&path).expect("空ファイルは読めるはず");
    assert!(f.entries.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------- CRLF

#[test]
fn crlf_matches_lf_for_the_same_body() {
    let lf = "port = 9001\nhost = \"h\"\nname = 'x'\n";
    let crlf = lf.replace('\n', "\r\n");
    let a = parse_str(lf, "lf.toml").unwrap();
    let b = parse_str(&crlf, "crlf.toml").unwrap();
    assert_eq!(a.entries, b.entries);
    assert_eq!(b.get("port"), Some(&Value::Int(9001)));
    assert_eq!(b.get("host"), Some(&Value::Str("h".into())));
    assert_eq!(b.get("name"), Some(&Value::Str("x".into())));
}

#[test]
fn crlf_does_not_leak_cr_into_values_or_keys() {
    let f = parse_str("port = 9001\r\nhost = \"h\"\r\n", "t.toml").unwrap();
    // 値・キーの末尾に \r が残っていないこと。
    assert_eq!(f.get("port"), Some(&Value::Int(9001)));
    assert_eq!(f.get("host"), Some(&Value::Str("h".into())));
    assert!(f.get("host\r").is_none());
}

#[test]
fn crlf_error_line_numbers_are_correct() {
    let e = parse_str("port = 1\r\nport = 2\r\n", "t.toml").unwrap_err();
    assert!(e.contains("t.toml:2"), "{e}");
    assert!(e.contains("重複"), "{e}");
    assert!(e.contains("1 行目で既に定義"), "{e}");
}

#[test]
fn loading_a_crlf_file_succeeds() {
    let dir = tmpdir("crlf");
    let path = dir.join("c.toml");
    std::fs::write(&path, "port = 9100\r\nname = 'n'\r\n").unwrap();
    let f = load(&path).expect("CRLF ファイルは読めるはず");
    assert_eq!(f.get("port"), Some(&Value::Int(9100)));
    assert_eq!(f.get("name"), Some(&Value::Str("n".into())));
    std::fs::remove_dir_all(&dir).ok();
}

// -------------------------------------------------------------- 不正な UTF-8

#[test]
fn load_rejects_invalid_utf8() {
    let dir = tmpdir("badutf8");
    let path = dir.join("bad.toml");
    // `name = "` のあとに 0xFF (UTF-8 として不正) を置く。
    let mut bytes = b"name = \"".to_vec();
    bytes.push(0xFF);
    bytes.extend_from_slice(b"\"\n");
    std::fs::write(&path, &bytes).unwrap();

    let e = load(&path).expect_err("不正な UTF-8 は読めてはいけない");
    assert!(e.contains("読めない"), "{e}");
    assert!(e.contains("bad.toml"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_a_lone_continuation_byte() {
    let dir = tmpdir("badutf8b");
    let path = dir.join("bad2.toml");
    // 0x80 は UTF-8 の継続バイト単体なので不正。
    std::fs::write(&path, [b'p', b'o', b'r', b't', b' ', b'=', b' ', 0x80]).unwrap();
    let e = load(&path).expect_err("継続バイト単体は不正な UTF-8");
    assert!(e.contains("読めない"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_reports_a_missing_file_by_path() {
    let dir = tmpdir("missing");
    let path = dir.join("nope.toml");
    let e = load(&path).expect_err("存在しないファイルは読めない");
    assert!(e.contains("読めない"), "{e}");
    assert!(e.contains("nope.toml"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}
