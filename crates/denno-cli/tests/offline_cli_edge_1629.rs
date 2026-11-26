//! task 1629: `denno info` / `denno inspect` / `denno generate` / `denno pull` の
//! オフライン異常系と引数検証の境界試験。
//!
//! 実バイナリ (`CARGO_BIN_EXE_denno`) を起動するが、**ネットワークには一切
//! 触れない**。`HOME` (と `USERPROFILE`) をテスト専用の一時ディレクトリへ
//! 差し替えるので、モデルキャッシュは空のまま固定され、`pull` は引数解析の
//! 段で終わる入力しか与えない。
//!
//! 固定する契約:
//! 1. clap の引数エラーは終了コード 2、stdout は空、stderr に `error:`。
//!    (`pull` の引数欠落 / `generate` の必須引数欠落)
//! 2. 実行時エラー (存在しないパス・空ディレクトリ・壊れた GGUF・壊れた
//!    config.json・未キャッシュ) は終了コード 1。panic (101) にはならない。
//! 3. 失敗時の stdout は空 (診断は stderr だけ)。
//! 4. `generate` の値域検証 (`--no-repeat-ngram-size` / 非有限 penalty) は
//!    モデルを読む前に終了コード 1 で止まる。
//!
//! 終了コードは main の `Result` 経路 (anyhow) が 1、clap が 2、panic が 101。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const CLAP_ERROR: i32 = 2;
const RUNTIME_ERROR: i32 = 1;
const PANIC: i32 = 101;

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ (Drop で再帰削除)。`home` は子プロセスの
/// `HOME` に渡す空のキャッシュルート。
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
            "denno1629_{}_{}_{}_{}",
            tag,
            std::process::id(),
            n,
            nanos
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = self.path.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(&p, bytes).expect("write file");
        p
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

/// 隔離した環境で denno を起動する。`home` を HOME / USERPROFILE に据え、
/// ホストの `DENNO_*` と `RUST_LOG` を落として出力を決定的にする。
fn denno(home: &Path, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    cmd.env("HOME", home);
    cmd.env("USERPROFILE", home);
    for k in [
        "RUST_LOG",
        "DENNO_FLASH_DIR",
        "DENNO_FLASH_HOST",
        "DENNO_FLASH_PORT",
        "DENNO_EXPERT_CACHE_MB",
        "DENNO_AUTO_CONFIG",
        "DENNO_API_KEYS",
        "DENNO_KV_SLOTS",
        "DENNO_PREFIX_CACHE_TOKENS",
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

/// 全試験共通: panic していないこと (終了コード 101 / `panicked at` が無い)。
fn assert_no_panic(o: &Output) {
    let err = stderr(o);
    assert_ne!(
        o.status.code(),
        Some(PANIC),
        "子プロセスが panic した: stderr=\n{err}"
    );
    assert!(
        !err.contains("panicked at"),
        "stderr に panic の痕跡がある:\n{err}"
    );
}

/// clap の解析エラー共通形: 終了コード 2、stdout 空、stderr に `error:` と断片。
fn assert_clap_error(o: &Output, needles: &[&str]) {
    assert_no_panic(o);
    assert_eq!(o.status.code(), Some(CLAP_ERROR), "stderr=\n{}", stderr(o));
    assert_eq!(stdout(o), "", "clap エラーで stdout に何か出た");
    let err = stderr(o);
    assert!(err.contains("error:"), "stderr に error: が無い:\n{err}");
    for n in needles {
        assert!(err.contains(n), "stderr に {n:?} が無い:\n{err}");
    }
}

/// 実行時エラー共通形: 終了コード 1 (panic でも clap でもない)、stdout 空。
/// 診断は stderr に出る (anyhow の `Error: ...` と context の連鎖)。
fn assert_runtime_error(o: &Output, needles: &[&str]) {
    assert_no_panic(o);
    assert_eq!(
        o.status.code(),
        Some(RUNTIME_ERROR),
        "stderr=\n{}",
        stderr(o)
    );
    assert_eq!(stdout(o), "", "実行時エラーで stdout に何か出た");
    let err = stderr(o);
    assert!(!err.trim().is_empty(), "実行時エラーの stderr が空");
    for n in needles {
        assert!(err.contains(n), "stderr に {n:?} が無い:\n{err}");
    }
}

// ===== info: 存在しないパス / 空ディレクトリ / 壊れた GGUF =====

#[test]
fn info_nonexistent_path_is_clean_runtime_error() {
    let t = TempDir::new("info_missing");
    let missing = t.as_path().join("no-such-model-dir");
    let o = denno(t.as_path(), &["info", missing.to_str().unwrap()]);
    // resolve_model_dir は実在しないパスを hub cache 経路に落とし、未キャッシュで止まる。
    assert_runtime_error(&o, &["not cached", "denno pull"]);
}

#[test]
fn info_empty_directory_reports_missing_config() {
    let t = TempDir::new("info_empty");
    let empty = t.as_path().join("empty");
    std::fs::create_dir_all(&empty).expect("create empty dir");
    let o = denno(t.as_path(), &["info", empty.to_str().unwrap()]);
    assert_runtime_error(&o, &["config.json"]);
}

#[test]
fn info_corrupted_gguf_still_errors_on_config() {
    let t = TempDir::new("info_badgguf");
    let dir = t.as_path().join("bad_gguf");
    std::fs::create_dir_all(&dir).expect("create dir");
    // GGUF magic ですらないバイト列。config.json も無い。
    std::fs::write(dir.join("model.gguf"), b"NOT-A-GGUF-HEADER").expect("write gguf");
    let o = denno(t.as_path(), &["info", dir.to_str().unwrap()]);
    // GGUF から次元が取れないので、config.json 不在の Err がそのまま出る。
    assert_runtime_error(&o, &["config.json"]);
}

#[test]
fn info_corrupted_config_json_is_clean_runtime_error() {
    let t = TempDir::new("info_badcfg");
    let dir = t.as_path().join("bad_cfg");
    std::fs::create_dir_all(&dir).expect("create dir");
    std::fs::write(dir.join("config.json"), b"{ not json ").expect("write cfg");
    let o = denno(t.as_path(), &["info", dir.to_str().unwrap()]);
    assert_runtime_error(&o, &["config.json"]);
}

// ===== inspect: 存在しない / ディレクトリ / 空ファイル / 壊れた magic / 切れ =====

#[test]
fn inspect_nonexistent_path_is_clean_runtime_error() {
    let t = TempDir::new("inspect_missing");
    let missing = t.as_path().join("nope.gguf");
    let o = denno(t.as_path(), &["inspect", missing.to_str().unwrap()]);
    assert_runtime_error(&o, &["stat"]);
}

#[test]
fn inspect_directory_is_rejected() {
    let t = TempDir::new("inspect_dir");
    let dir = t.as_path().join("a_dir");
    std::fs::create_dir_all(&dir).expect("create dir");
    let o = denno(t.as_path(), &["inspect", dir.to_str().unwrap()]);
    assert_runtime_error(&o, &["ディレクトリ"]);
}

#[test]
fn inspect_empty_file_is_clean_runtime_error() {
    let t = TempDir::new("inspect_empty");
    let f = t.write("empty.gguf", b"");
    let o = denno(t.as_path(), &["inspect", f.to_str().unwrap()]);
    assert_runtime_error(&o, &[]);
}

#[test]
fn inspect_bad_magic_is_clean_runtime_error() {
    let t = TempDir::new("inspect_magic");
    let f = t.write("bad_magic.gguf", b"XXXX\x00\x00\x00\x00");
    let o = denno(t.as_path(), &["inspect", f.to_str().unwrap()]);
    assert_runtime_error(&o, &[]);
}

#[test]
fn inspect_truncated_header_is_clean_runtime_error() {
    let t = TempDir::new("inspect_trunc");
    // magic だけ (version / カウントが無い) = 途中で切れたヘッダ。
    let f = t.write("trunc.gguf", b"GGUF");
    let o = denno(t.as_path(), &["inspect", f.to_str().unwrap()]);
    assert_runtime_error(&o, &[]);
}

// ===== generate: 必須引数 / 未キャッシュ / 壊れた GGUF / 値域検証 =====

#[test]
fn generate_missing_prompt_is_clap_error() {
    let t = TempDir::new("gen_no_prompt");
    let o = denno(t.as_path(), &["generate", "nosuch/repo"]);
    // --prompt は必須 (Option ではない)。値を要求する文言で止まる。
    assert_clap_error(&o, &["--prompt"]);
}

#[test]
fn generate_missing_repo_is_clap_error() {
    let t = TempDir::new("gen_no_repo");
    let o = denno(t.as_path(), &["generate", "--prompt", "hi"]);
    assert_clap_error(&o, &["<REPO>"]);
}

#[test]
fn generate_uncached_repo_is_clean_runtime_error() {
    let t = TempDir::new("gen_uncached");
    let o = denno(
        t.as_path(),
        &["generate", "nosuch/repo", "--prompt", "hi", "--max-tokens", "1"],
    );
    // model_dir は HOME 配下の hub cache を見る。空なので not cached で止まる。
    assert_runtime_error(&o, &["not cached", "denno pull"]);
}

#[test]
fn generate_corrupted_gguf_in_cache_is_clean_runtime_error() {
    let t = TempDir::new("gen_badgguf");
    // hub cache 風のディレクトリを HOME 配下に作り、壊れた GGUF を置く。
    let dir = t
        .as_path()
        .join(".cache/denno/hub/models--corrupt--model");
    std::fs::create_dir_all(&dir).expect("create cache dir");
    std::fs::write(dir.join("config.json"), br#"{"model_type":"qwen2"}"#).expect("write cfg");
    std::fs::write(dir.join("model.gguf"), b"GGUF\x03\x00\x00\x00garbage").expect("write gguf");
    let o = denno(
        t.as_path(),
        &["generate", "corrupt/model", "--prompt", "hi", "--max-tokens", "1"],
    );
    assert_runtime_error(&o, &["モデルのロードに失敗"]);
}

#[test]
fn generate_rejects_small_no_repeat_ngram_before_loading_model() {
    let t = TempDir::new("gen_ngram_low");
    let o = denno(
        t.as_path(),
        &[
            "generate",
            "nosuch/repo",
            "--prompt",
            "hi",
            "--no-repeat-ngram-size",
            "1",
        ],
    );
    // 値域検証はモデルロードより前。未キャッシュの診断ではなく引数診断が出る。
    assert_runtime_error(&o, &["no-repeat-ngram-size", "2 以上"]);
}

#[test]
fn generate_rejects_huge_no_repeat_ngram() {
    let t = TempDir::new("gen_ngram_high");
    let o = denno(
        t.as_path(),
        &[
            "generate",
            "nosuch/repo",
            "--prompt",
            "hi",
            "--no-repeat-ngram-size",
            "4096",
        ],
    );
    assert_runtime_error(&o, &["no-repeat-ngram-size", "1024 以下"]);
}

#[test]
fn generate_rejects_non_finite_frequency_penalty() {
    let t = TempDir::new("gen_nan_penalty");
    let o = denno(
        t.as_path(),
        &[
            "generate",
            "nosuch/repo",
            "--prompt",
            "hi",
            "--frequency-penalty",
            "NaN",
        ],
    );
    assert_runtime_error(&o, &["frequency-penalty", "有限値"]);
}

// ===== pull: 引数解析だけ (ネットワークに触れない) =====

#[test]
fn pull_missing_repo_is_clap_error_without_network() {
    let t = TempDir::new("pull_no_arg");
    let o = denno(t.as_path(), &["pull"]);
    assert_clap_error(&o, &["<REPO>"]);
}

#[test]
fn pull_extra_argument_is_clap_error_without_network() {
    let t = TempDir::new("pull_extra");
    let o = denno(t.as_path(), &["pull", "a/b", "c/d"]);
    assert_clap_error(&o, &[]);
}

#[test]
fn pull_unknown_flag_is_clap_error_without_network() {
    let t = TempDir::new("pull_flag");
    let o = denno(t.as_path(), &["pull", "a/b", "--no-such-flag"]);
    assert_clap_error(&o, &["--no-such-flag"]);
}
