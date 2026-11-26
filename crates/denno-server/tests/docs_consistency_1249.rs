//! task 1249: docs の横断整合性チェック (軽量ガード)。
//!
//! `docs/` 配下の Markdown と `docs/openapi.yaml` を走査し、コミットしては
//! いけない情報が紛れ込んでいないことを検証する:
//!
//! - LAN 私有アドレス (`192.168.x.y`, `10.0.x.y`) の生値
//! - ホスト内部の絶対パス (`/home/<name>/...`)
//!
//! ネットワークにもバイナリにも依存しない純粋なファイル走査テストなので、
//! CPU のみの環境で高速に回る。新しい docs を追加したときに、うっかり
//! 手元の環境固有の値が混ざるのを防ぐのが目的。

use std::fs;
use std::path::{Path, PathBuf};

/// 走査対象にする拡張子。docs は基本 Markdown と OpenAPI 定義だけ。
const TEXT_EXTENSIONS: &[&str] = &["md", "yaml", "yml"];

fn docs_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs")
}

/// docs 配下を再帰的に列挙し、テキスト系の拡張子だけを返す (ソート済み)。
fn collect_docs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("docs ディレクトリを読めない {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("docs エントリの読み取りに失敗");
        let path = entry.path();
        let file_type = entry.file_type().expect("docs エントリ種別の取得に失敗");
        if file_type.is_dir() {
            collect_docs(&path, out);
        } else if file_type.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase());
            if let Some(ext) = ext {
                if TEXT_EXTENSIONS.contains(&ext.as_str()) {
                    out.push(path);
                }
            }
        }
    }
    out.sort();
}

/// `prefix` の直後が IPv4 の残り 2 オクテット (`d{1,3}.d{1,3}`) になっているかを判定する。
fn has_two_octets_after(text: &str, start: usize) -> bool {
    let bytes = text.as_bytes();
    let mut i = start;
    let mut octets = 0;
    while octets < 2 {
        let begin = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == begin || i - begin > 3 {
            return false;
        }
        octets += 1;
        if octets < 2 {
            if i >= bytes.len() || bytes[i] != b'.' {
                return false;
            }
            i += 1;
        }
    }
    // 4 オクテット目 (10.0.1.2 の "2") の直後が数字やドットなら別の数値の一部。
    !(i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.'))
}

/// `offset` の直前が数字やドットなら、より長い数値の途中なので除外する。
fn starts_fresh_token(text: &str, offset: usize) -> bool {
    let bytes = text.as_bytes();
    if offset == 0 {
        return true;
    }
    let prev = bytes[offset - 1];
    !(prev.is_ascii_digit() || prev == b'.')
}

/// 1 ファイルの内容から問題箇所を集める。返り値は `行番号: 種別` の人間向け一覧。
fn scan_text(label: &str, text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let mut from = 0;
        while let Some(pos) = line[from..].find("192.168.") {
            let abs = from + pos;
            let after = abs + "192.168.".len();
            if starts_fresh_token(line, abs) && has_two_octets_after(line, after) {
                problems.push(format!("{label}:{line_no}: LAN 私有 IPv4 (192.168.*)"));
                break;
            }
            from = abs + 1;
        }
        let mut from = 0;
        while let Some(pos) = line[from..].find("10.0.") {
            let abs = from + pos;
            let after = abs + "10.0.".len();
            if starts_fresh_token(line, abs) && has_two_octets_after(line, after) {
                problems.push(format!("{label}:{line_no}: LAN 私有 IPv4 (10.0.*)"));
                break;
            }
            from = abs + 1;
        }
        if line.contains("/home/") {
            problems.push(format!("{label}:{line_no}: ホスト内部パス (/home/...)"));
        }
    }
    problems
}

#[test]
fn docs_have_no_leaked_private_addresses_or_home_paths() {
    let root = docs_root();
    assert!(
        root.is_dir(),
        "docs ディレクトリが見つからない: {}",
        root.display()
    );

    let mut files = Vec::new();
    collect_docs(&root, &mut files);
    assert!(!files.is_empty(), "docs 配下に走査対象ファイルが無い");

    // OpenAPI 定義が走査対象に入っていることを明示的に確認する
    // (ディレクトリ構成が変わって黙って対象外になるのを防ぐ)。
    let openapi = root.join("openapi.yaml");
    assert!(openapi.is_file(), "docs/openapi.yaml が見つからない");
    assert!(
        files.contains(&openapi),
        "docs/openapi.yaml が走査対象に含まれていない"
    );

    let mut problems = Vec::new();
    for path in &files {
        let label = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("docs ファイルを読めない {}: {e}", path.display()));
        problems.extend(scan_text(&label, &text));
    }

    assert!(
        problems.is_empty(),
        "docs に環境固有の情報が漏れている ({} 件):\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn scanner_detects_known_bad_samples() {
    // 走査ロジック自体の妥当性 (誤検出しないこと・検出漏れしないこと)。
    // サンプルは予約された範囲表記 (ホスト実値ではない) を使う。
    assert!(scan_text("x", "host: 192.168.0.0:8080").len() == 1);
    assert!(scan_text("x", "bind 10.0.0.0").len() == 1);
    assert!(scan_text("x", "model cached in /home/alice/models").len() == 1);

    // semver やバージョン表記 (3 オクテット) は LAN IPv4 とみなさない。
    assert!(scan_text("x", "version v10.0.1 released").is_empty());
    assert!(scan_text("x", "upgrade to 110.0.1.2 for details").is_empty());
    assert!(scan_text("x", "crate 192.168.1 is unrelated").is_empty());
    assert!(scan_text("x", "endpoint http://127.0.0.1:8080/v1").is_empty());
    assert!(scan_text("x", "workspace /srv/denno-dev5/docs").is_empty());
}
