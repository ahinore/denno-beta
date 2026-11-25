//! task 1149: `denno tokenize` の境界 / UTF-8 まわりの追加契約試験。
//!
//! 941 (`tokenize_941.rs`) が基本契約を押さえているので、ここは重ならない
//! 縁を厚くする:
//!
//! - 4 バイト UTF-8 (絵文字) / ZWJ 連結 / 異体字セレクタ / CJK 混在の往復。
//! - `text_bytes` (バイト) と `text_chars` (スカラ値) の数え方の差。
//! - 制御文字 (改行 / タブ / CR) のテキスト表示エスケープと JSON 生値。
//! - 空入力の 3 経路 (`--text ""` / 空の `--stdin` / `--json`) が一致する。
//! - `--stdin` の末尾改行を落とさない (標準入力はバイト列そのまま)。
//! - `--json` は 1 個の JSON オブジェクトで、必須フィールドの型が揃う。
//! - GGUF 不在系の終了コード: 存在しない dir / .gguf の無い dir /
//!   .gguf でないファイルを直接渡す / 壊れた GGUF -> いずれも 3。
//! - UTF-8 でない入力 (不正バイト / 途中で切れた多バイト) -> 4。
//!
//! 合成 GGUF はトークナイザ・メタデータだけ (データ部 0 バイト)。全 256 バイトを
//! 1 バイト 1 トークンで持つ byte-level 語彙なので、どんな入力でも
//! 「バイト数 = トークン数」で割れ、復号が必ず往復する。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

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
            "denno1149_{}_{}_{}_{}",
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

    fn as_path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ===== 合成 GGUF の書き出し =====

enum Kv<'a> {
    Str(&'a str),
    StrArr(&'a [String]),
    U32Arr(&'a [u32]),
}

fn w_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;

/// GGML 型 id: F32 (データ部は読まれないので中身は 0 埋めでよい)
const T_F32: u32 = 0;

type TensorSpec = (&'static str, Vec<u64>, u32, u64);

fn gguf_bytes(kvs: &[(&str, Kv)], tensors: &[TensorSpec], data_len: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kvs.len() as u64).to_le_bytes());

    for (k, v) in kvs {
        w_str(&mut b, k);
        match v {
            Kv::Str(s) => {
                b.extend(GT_STRING.to_le_bytes());
                w_str(&mut b, s);
            }
            Kv::StrArr(items) => {
                b.extend(GT_ARRAY.to_le_bytes());
                b.extend(GT_STRING.to_le_bytes());
                b.extend((items.len() as u64).to_le_bytes());
                for it in items.iter() {
                    w_str(&mut b, it);
                }
            }
            Kv::U32Arr(items) => {
                b.extend(GT_ARRAY.to_le_bytes());
                b.extend(GT_U32.to_le_bytes());
                b.extend((items.len() as u64).to_le_bytes());
                for it in items.iter() {
                    b.extend(it.to_le_bytes());
                }
            }
        }
    }

    for (name, dims, ty, off) in tensors {
        w_str(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
    }

    let data_start = b.len().div_ceil(32) * 32;
    b.resize(data_start + data_len as usize, 0);
    b
}

// ===== byte-level 語彙 (全 256 バイトを 1 対 1) =====

/// GPT-2 / llama.cpp の byte -> unicode 写像 (全 256 バイト)。
fn byte_level_map() -> Vec<char> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(0x21u32..=0x7E);
    bs.extend(0xA1u32..=0xAD);
    bs.extend(0xAEu32..=0xFF);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = vec![' '; 256];
    for i in 0..bs.len() {
        map[bs[i] as usize] = char::from_u32(cs[i]).expect("mapped char");
    }
    map
}

/// 全 256 バイトを 1 バイト 1 トークンで持つ GGUF (マージ無し)。
fn full_byte_gguf(data_len: u64) -> Vec<u8> {
    let map = byte_level_map();
    let tokens: Vec<String> = map.iter().map(|c| c.to_string()).collect();
    let types: Vec<u32> = vec![1u32; tokens.len()];
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("tokenizer.ggml.model", Kv::Str("gpt2")),
        ("tokenizer.ggml.pre", Kv::Str("qwen35")),
        ("tokenizer.ggml.tokens", Kv::StrArr(&tokens)),
        ("tokenizer.ggml.merges", Kv::StrArr(&[])),
        ("tokenizer.ggml.token_type", Kv::U32Arr(&types)),
    ];
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![8, 4], T_F32, 0)];
    gguf_bytes(&kvs, &tensors, data_len)
}

// ===== CLI 起動 =====

fn denno() -> Command {
    Command::new(env!("CARGO_BIN_EXE_denno"))
}

fn run_text(dir: &Path, text: &str, extra: &[&str]) -> Output {
    let mut c = denno();
    c.arg("tokenize")
        .arg("--dir")
        .arg(dir)
        .arg("--text")
        .arg(text);
    for e in extra {
        c.arg(e);
    }
    c.output().expect("run denno tokenize")
}

fn run_ok(dir: &Path, text: &str, extra: &[&str]) -> Output {
    let out = run_text(dir, text, extra);
    assert!(
        out.status.success(),
        "denno tokenize failed: {}",
        stderr_of(&out)
    );
    out
}

/// `--stdin` に生バイトを流して起動する。
fn run_stdin(dir: &Path, bytes: &[u8], extra: &[&str]) -> Output {
    let mut c = denno();
    c.arg("tokenize")
        .arg("--dir")
        .arg(dir)
        .arg("--stdin");
    for e in extra {
        c.arg(e);
    }
    let mut child = c
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn denno tokenize --stdin");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(bytes)
        .expect("write stdin");
    child.wait_with_output().expect("wait")
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn json_of(out: &Output) -> serde_json::Value {
    serde_json::from_str(&stdout_of(out)).expect("stdout is a single JSON object")
}

fn fixture_dir(tag: &str, data_len: u64) -> TempDir {
    let td = TempDir::new(tag);
    td.write("m.gguf", &full_byte_gguf(data_len));
    td
}

// ===== 1. 絵文字 / 補助面の往復 =====

#[test]
fn four_byte_emoji_round_trips() {
    let td = fixture_dir("emoji", 0);
    // U+1F600 は 4 バイト / 1 スカラ値。
    let src = "😀";
    let v = json_of(&run_ok(td.as_path(), src, &["--json"]));
    assert_eq!(v["text_bytes"], 4);
    assert_eq!(v["text_chars"], 1);
    assert_eq!(v["token_count"], 4, "1 バイト 1 トークンで割れる");
    assert_eq!(v["decoded"], src, "4 バイト絵文字が往復する");
}

#[test]
fn zwj_family_emoji_round_trips() {
    let td = fixture_dir("zwj", 0);
    // 人 + ZWJ + 人 ... の連結。スカラ値は複数だが 1 グリフ。
    let src = "👨‍👩‍👧‍👦";
    let v = json_of(&run_ok(td.as_path(), src, &["--json"]));
    assert_eq!(v["decoded"], src, "ZWJ 連結がバイト単位でも往復する");
    assert_eq!(v["text_chars"], src.chars().count());
    assert_eq!(v["text_bytes"], src.len());
    assert_eq!(v["token_count"], src.len());
}

#[test]
fn variation_selector_emoji_round_trips() {
    let td = fixture_dir("vs16", 0);
    // U+2764 (ハート) + U+FE0F (異体字セレクタ)。
    let src = "\u{2764}\u{fe0f}";
    let v = json_of(&run_ok(td.as_path(), src, &["--json"]));
    assert_eq!(v["text_chars"], 2);
    assert_eq!(v["decoded"], src);
}

#[test]
fn cjk_and_emoji_mixed_counts_bytes_and_chars_separately() {
    let td = fixture_dir("mixed", 0);
    // a(1) + é(2) + 😀(4) + 日(3) = 10 バイト / 4 スカラ値。
    let src = "aé😀日";
    let v = json_of(&run_ok(td.as_path(), src, &["--json"]));
    assert_eq!(v["text_bytes"], 10);
    assert_eq!(v["text_chars"], 4);
    assert_eq!(v["token_count"], 10);
    assert_eq!(v["decoded"], src);
    assert_ne!(
        v["text_bytes"], v["text_chars"],
        "バイト数と文字数は別物として報告する"
    );
}

// ===== 2. 制御文字: JSON は生、テキスト表示はエスケープ =====

#[test]
fn control_chars_are_escaped_in_text_and_raw_in_json() {
    let td = fixture_dir("ctrl", 0);
    let src = "a\tb\nc\r";

    // JSON は生の制御文字をそのまま復号結果に載せる。
    let v = json_of(&run_ok(td.as_path(), src, &["--json"]));
    assert_eq!(v["decoded"], src);
    assert_eq!(v["text_bytes"], src.len());
    assert_eq!(v["token_count"], src.len());

    // 人が読む表示では \t \n \r が 2 文字のエスケープになる。
    let text = stdout_of(&run_ok(td.as_path(), src, &[]));
    assert!(
        text.contains("decoded: \"a\\tb\\nc\\r\""),
        "escaped decoded line missing: {text}"
    );
    // 生のタブ/改行が decoded 行に混ざっていないこと。
    assert!(!text.contains("decoded: \"a\tb"), "{text}");
}

// ===== 3. 空入力の 3 経路が一致する =====

#[test]
fn empty_input_agrees_across_text_stdin_and_json() {
    let td = fixture_dir("empty3", 0);

    let from_text_json = json_of(&run_ok(td.as_path(), "", &["--json"]));
    assert_eq!(from_text_json["token_count"], 0);
    assert_eq!(from_text_json["text_bytes"], 0);
    assert_eq!(from_text_json["text_chars"], 0);
    assert_eq!(from_text_json["tokens"].as_array().expect("tokens").len(), 0);
    assert_eq!(from_text_json["decoded"], "");

    let stdin_out = run_stdin(td.as_path(), b"", &["--json"]);
    assert!(stdin_out.status.success(), "{}", stderr_of(&stdin_out));
    let from_stdin = json_of(&stdin_out);
    assert_eq!(from_stdin["token_count"], 0);
    assert_eq!(from_stdin["tokens"], from_text_json["tokens"]);
    assert_eq!(from_stdin["decoded"], "");

    // テキスト表示も 0 トークンで落ちない。
    let text = stdout_of(&run_ok(td.as_path(), "", &[]));
    assert!(text.contains("tokens: 0"), "{text}");
    assert!(text.contains("decoded: \"\""), "{text}");
}

// ===== 4. 標準入力は末尾改行を落とさない =====

#[test]
fn stdin_preserves_trailing_newline() {
    let td = fixture_dir("trailnl", 0);
    let bytes = "ab\n".as_bytes();

    let out = run_stdin(td.as_path(), bytes, &["--json"]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let v = json_of(&out);
    assert_eq!(v["text_bytes"], 3, "末尾の改行も 1 バイトとして数える");
    assert_eq!(v["token_count"], 3);
    assert_eq!(v["decoded"], "ab\n");

    // `--text "ab"` とは別物 (改行の分だけ多い) — 入力源を取り違えない。
    let arg = json_of(&run_ok(td.as_path(), "ab", &["--json"]));
    assert_eq!(arg["token_count"], 2);
}

// ===== 5. --json の形 (1 オブジェクト / 必須フィールドの型) =====

#[test]
fn json_has_stable_field_shape() {
    let td = fixture_dir("shape", 0);
    let out = run_ok(td.as_path(), "ab", &["--json"]);
    let raw = stdout_of(&out);
    // 単一の JSON オブジェクトであること (前後にログが混ざらない)。
    assert!(raw.trim_start().starts_with('{'), "stdout: {raw}");
    assert!(raw.trim_end().ends_with('}'), "stdout: {raw}");

    let v = json_of(&out);
    let obj = v.as_object().expect("top-level object");
    for key in [
        "dir",
        "gguf",
        "vocab_size",
        "bos",
        "eos",
        "text_bytes",
        "text_chars",
        "token_count",
        "tokens",
        "decoded",
    ] {
        assert!(obj.contains_key(key), "missing field {key}: {v}");
    }
    assert!(v["dir"].is_string());
    assert!(v["gguf"].is_array());
    assert_eq!(v["gguf"].as_array().expect("gguf").len(), 1);
    assert!(v["vocab_size"].as_u64().expect("vocab_size") >= 256);
    assert!(v["token_count"].as_u64().is_some());
    assert!(v["tokens"].is_array());

    let t0 = &v["tokens"][0];
    assert!(t0["id"].as_u64().is_some(), "id must be a number: {t0}");
    assert!(t0["piece"].is_string(), "piece must be a string: {t0}");
    assert!(t0["raw"].is_string(), "raw must be a string: {t0}");
}

// ===== 6. GGUF 不在系はすべて終了コード 3 =====

fn assert_load_error(out: &Output, needle: &str) {
    assert!(!out.status.success(), "should fail: {}", stderr_of(out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert_eq!(out.status.code(), Some(3), "stderr: {}", stderr_of(out));
    let e = stderr_of(out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    assert!(e.contains(needle), "stderr should contain {needle:?}: {e}");
}

#[test]
fn nonexistent_dir_is_exit_3() {
    let td = TempDir::new("nodir");
    let p = td.as_path().join("nope");
    let out = run_text(&p, "a", &[]);
    assert_load_error(&out, "存在しない");
    assert!(stderr_of(&out).contains("nope"), "{}", stderr_of(&out));
}

#[test]
fn existing_but_empty_dir_is_exit_3() {
    // 存在するがファイルが 1 本も無いディレクトリ。
    let td = TempDir::new("emptydir");
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, ".gguf");
}

#[test]
fn dir_with_only_non_gguf_files_is_exit_3() {
    let td = TempDir::new("txtonly");
    td.write("a.txt", b"x");
    td.write("b.bin", b"y");
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, ".gguf");
}

#[test]
fn non_gguf_file_passed_directly_is_exit_3() {
    // `--dir` に .gguf でない実ファイルを直接渡すと、開けない/壊れているで 3。
    let td = TempDir::new("notgguf");
    let p = td.write("model.bin", b"definitely not a gguf");
    let out = run_text(&p, "a", &[]);
    assert_load_error(&out, "GGUF");
}

#[test]
fn truncated_gguf_is_exit_3() {
    // マジックだけ正しく、ヘッダの途中で切れているファイル。
    let td = TempDir::new("trunc");
    let mut b = full_byte_gguf(0);
    b.truncate(8);
    td.write("m.gguf", &b);
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, "GGUF");
}

// ===== 7. UTF-8 でない標準入力は終了コード 4 =====

fn assert_input_error(out: &Output) {
    assert_eq!(out.status.code(), Some(4), "stderr: {}", stderr_of(out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    let e = stderr_of(out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    assert!(e.contains("UTF-8"), "stderr should mention UTF-8: {e}");
}

#[test]
fn invalid_byte_stdin_is_exit_4() {
    let td = fixture_dir("badbyte", 0);
    let out = run_stdin(td.as_path(), &[0xff, 0xfe], &[]);
    assert_input_error(&out);
}

#[test]
fn truncated_multibyte_stdin_is_exit_4() {
    let td = fixture_dir("badmulti", 0);
    // 3 バイト文字 (日 = E6 97 A5) の先頭 2 バイトだけ。途中で切れた多バイト。
    let out = run_stdin(td.as_path(), &[0xe6, 0x97], &[]);
    assert_input_error(&out);
}

#[test]
fn isolated_continuation_byte_stdin_is_exit_4() {
    let td = fixture_dir("badcont", 0);
    // 単独の継続バイト 0x80 は UTF-8 として不正。
    let out = run_stdin(td.as_path(), &[0x80], &[]);
    assert_input_error(&out);
}
