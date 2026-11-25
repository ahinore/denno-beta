//! task 941: `denno tokenize --dir <path> [--text <str> | --stdin] [--json]` の
//! CLI 契約の試験。
//!
//! 合成の最小 GGUF (トークナイザ・メタデータだけ) を一時ディレクトリに書き、
//! 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動して確かめる:
//!
//! 1. ASCII / 日本語 / 特殊トークンの id と断片が出る。
//! 2. 日本語はバイト単位に割れても、復号 (decoded) が入力を往復する。
//! 3. 空入力は 0 トークンで成功する。
//! 4. `--stdin` は `--text` と同じ結果になる。
//! 5. 重み (テンソルのデータ部) を読まない — データ部 0 バイトでも成功する。
//! 6. 欠けたディレクトリ / .gguf の無いディレクトリ / 壊れた GGUF /
//!    トークナイザ・メタデータの無い GGUF は終了コード 3 で、理由を stderr に出す。
//! 7. `--text` も `--stdin` も無い / 両方ある は終了コード 2。
//! 8. UTF-8 でない標準入力は終了コード 4。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ (Drop で再帰削除)。
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
            "denno941_{}_{}_{}_{}",
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
    U32(u32),
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

/// `(名前, 次元, 型 id, データ部オフセット)`
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
            Kv::U32(x) => {
                b.extend(GT_U32.to_le_bytes());
                b.extend(x.to_le_bytes());
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

// ===== byte-level BPE の語彙を組む道具 =====

/// GPT-2 / llama.cpp の byte → unicode 写像 (全 256 バイト)。
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

/// 文字列を byte-level の語彙表記に直す (空白は `Ġ` になる)。
fn bl(s: &str, map: &[char]) -> String {
    s.as_bytes().iter().map(|&b| map[b as usize]).collect()
}

/// `text` のバイトをすべて単独トークンとして持つ語彙を組む (マージ無し =
/// 1 バイト 1 トークン)。日本語が複数トークンに割れることの確認に使う。
fn byte_vocab(text: &str, map: &[char]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for &b in text.as_bytes() {
        let t: String = map[b as usize].to_string();
        if !seen.contains(&t) {
            seen.push(t);
        }
    }
    seen
}

/// トークン種別 (1 = normal / 3 = control)。engine は control を added_tokens に
/// 入れるので、特殊トークンの扱いを確かめられる。
fn types_for(tokens: &[String], special: &[&str]) -> Vec<u32> {
    tokens
        .iter()
        .map(|t| {
            if special.iter().any(|s| s == t) {
                3
            } else {
                1
            }
        })
        .collect()
}

/// トークナイザ・メタデータ一式を持つ GGUF を組む。
fn tokenizer_gguf(
    tokens: &[String],
    merges: &[String],
    special: &[&str],
    data_len: u64,
) -> Vec<u8> {
    let types = types_for(tokens, special);
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("tokenizer.ggml.model", Kv::Str("gpt2")),
        ("tokenizer.ggml.pre", Kv::Str("qwen35")),
        ("tokenizer.ggml.tokens", Kv::StrArr(tokens)),
        ("tokenizer.ggml.merges", Kv::StrArr(merges)),
        ("tokenizer.ggml.token_type", Kv::U32Arr(&types)),
        ("tokenizer.ggml.bos_token_id", Kv::U32(0)),
        ("tokenizer.ggml.eos_token_id", Kv::U32(1)),
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

fn stdout_of(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn json_of(out: &Output) -> serde_json::Value {
    serde_json::from_str(&stdout_of(out)).expect("stdout is a single JSON object")
}

/// `ab` の 1 トークン (merge) + ASCII の単独 2 トークン。
fn ascii_fixture(data_len: u64) -> Vec<u8> {
    let map = byte_level_map();
    let tokens = vec![
        "<|im_end|>".to_string(),
        bl("a", &map),
        bl("b", &map),
        bl("ab", &map),
    ];
    let merges = vec![format!("{} {}", bl("a", &map), bl("b", &map))];
    tokenizer_gguf(&tokens, &merges, &["<|im_end|>"], data_len)
}

// ===== 1. ASCII =====

#[test]
fn ascii_text_reports_ids_and_pieces() {
    let td = TempDir::new("ascii");
    td.write("m.gguf", &ascii_fixture(0));

    let out = run_ok(td.as_path(), "ab", &["--json"]);
    let v = json_of(&out);
    assert_eq!(v["token_count"], 1);
    assert_eq!(v["text_bytes"], 2);
    assert_eq!(v["text_chars"], 2);
    assert_eq!(v["tokens"][0]["id"], 3);
    assert_eq!(v["tokens"][0]["piece"], "ab");
    assert_eq!(v["decoded"], "ab");
    // bos/eos は 0 / 1 を宣言している。
    assert_eq!(v["bos"], 0);
    assert_eq!(v["eos"], 1);
    assert_eq!(v["vocab_size"], 4);
    assert_eq!(v["gguf"].as_array().expect("gguf array").len(), 1);
}

#[test]
fn merge_splits_multi_char_text() {
    let td = TempDir::new("merges");
    td.write("m.gguf", &ascii_fixture(0));

    // "aab" -> "a" + "ab" (マージは "a b" だけ)
    let out = run_ok(td.as_path(), "aab", &["--json"]);
    let v = json_of(&out);
    assert_eq!(v["token_count"], 2);
    assert_eq!(v["tokens"][0]["piece"], "a");
    assert_eq!(v["tokens"][1]["piece"], "ab");
    assert_eq!(v["decoded"], "aab");

    // テキスト表示にも id と断片が出る。
    let text = stdout_of(&run_ok(td.as_path(), "aab", &[]));
    assert!(text.contains("tokens: 2"), "{text}");
    assert!(text.contains("\"ab\""), "{text}");
    assert!(text.contains("decoded: \"aab\""), "{text}");
}

// ===== 2. 日本語 =====

#[test]
fn japanese_round_trips() {
    let td = TempDir::new("ja");
    let map = byte_level_map();
    let src = "こんにちは";
    let tokens = byte_vocab(src, &map);
    td.write("m.gguf", &tokenizer_gguf(&tokens, &[], &[], 0));

    let out = run_ok(td.as_path(), src, &["--json"]);
    let v = json_of(&out);
    // マージが無いので UTF-8 のバイト数 (5 文字 × 3 バイト) だけ割れる。
    assert_eq!(v["text_chars"], 5);
    assert_eq!(v["text_bytes"], 15);
    assert_eq!(v["token_count"], 15);
    assert_eq!(v["decoded"], src, "decoded must round-trip the input");

    // どのトークンも空の断片にはならない (piece か raw が入る)。
    for t in v["tokens"].as_array().expect("tokens") {
        let piece = t["piece"].as_str().expect("piece str");
        let raw = t["raw"].as_str().expect("raw str");
        assert!(!piece.is_empty() || !raw.is_empty(), "empty token entry");
    }
}

// ===== 3. 空入力 =====

#[test]
fn empty_text_is_zero_tokens() {
    let td = TempDir::new("empty");
    td.write("m.gguf", &ascii_fixture(0));

    let out = run_ok(td.as_path(), "", &["--json"]);
    let v = json_of(&out);
    assert_eq!(v["token_count"], 0);
    assert_eq!(v["text_bytes"], 0);
    assert_eq!(v["tokens"].as_array().expect("tokens").len(), 0);
    assert_eq!(v["decoded"], "");

    let text = stdout_of(&run_ok(td.as_path(), "", &[]));
    assert!(text.contains("tokens: 0"), "{text}");
    assert!(text.contains("decoded: \"\""), "{text}");
}

// ===== 4. 特殊トークン =====

#[test]
fn special_token_is_kept_and_shown() {
    let td = TempDir::new("special");
    td.write("m.gguf", &ascii_fixture(0));

    let out = run_ok(td.as_path(), "a<|im_end|>", &["--json"]);
    let v = json_of(&out);
    assert_eq!(v["token_count"], 2);
    assert_eq!(v["tokens"][0]["piece"], "a");
    assert_eq!(v["tokens"][1]["id"], 0);
    // decode (skip_special_tokens) は空を返すので、語彙の文字列を見せる。
    assert_eq!(v["tokens"][1]["piece"], "<|im_end|>");
}

// ===== 5. 標準入力 =====

#[test]
fn stdin_matches_text_argument() {
    let td = TempDir::new("stdin");
    td.write("m.gguf", &ascii_fixture(0));

    let mut child = denno()
        .arg("tokenize")
        .arg("--dir")
        .arg(td.as_path())
        .arg("--stdin")
        .arg("--json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all("aab".as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));

    let from_stdin = json_of(&out);
    let from_arg = json_of(&run_ok(td.as_path(), "aab", &["--json"]));
    assert_eq!(from_stdin["tokens"], from_arg["tokens"]);
    assert_eq!(from_stdin["decoded"], from_arg["decoded"]);
}

// ===== 6. 重みを読まない =====

#[test]
fn header_only_gguf_works() {
    // データ部 0 バイト = 重みが 1 バイトも無いファイルでも成功する。
    let td = TempDir::new("headeronly");
    let bytes = ascii_fixture(0);
    let p = td.write("head_only.gguf", &bytes);
    assert_eq!(
        std::fs::metadata(&p).expect("stat").len() as usize,
        bytes.len()
    );

    let out = run_ok(td.as_path(), "ab", &["--json"]);
    assert_eq!(json_of(&out)["token_count"], 1);
}

// ===== 7. 壊れた入力 / 欠け =====

fn assert_load_error(out: &Output, needle: &str) {
    assert!(!out.status.success(), "should fail");
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert_eq!(out.status.code(), Some(3), "stderr: {}", stderr_of(out));
    let e = stderr_of(out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    assert!(e.contains(needle), "stderr should contain {needle:?}: {e}");
}

#[test]
fn missing_dir_fails_with_exit_3() {
    let td = TempDir::new("missing");
    let p = td.as_path().join("nope");
    let out = run_text(&p, "a", &[]);
    assert_load_error(&out, "存在しない");
    assert!(stderr_of(&out).contains("nope"), "{}", stderr_of(&out));
}

#[test]
fn dir_without_gguf_fails_with_exit_3() {
    let td = TempDir::new("nogguf");
    td.write("readme.txt", b"nothing here");
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, ".gguf");
}

#[test]
fn bad_magic_fails_with_exit_3() {
    let td = TempDir::new("badmagic");
    let mut b = ascii_fixture(0);
    b[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
    td.write("bad.gguf", &b);
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, "GGUF");
}

#[test]
fn missing_tokenizer_metadata_fails_with_exit_3() {
    // トークナイザの KV が 1 つも無い GGUF。
    let td = TempDir::new("notok");
    let kvs: Vec<(&str, Kv)> = vec![("general.architecture", Kv::Str("qwen4exp"))];
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![8, 4], T_F32, 0)];
    td.write("m.gguf", &gguf_bytes(&kvs, &tensors, 0));
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, "トークナイザ");
}

#[test]
fn unsupported_tokenizer_model_fails_with_exit_3() {
    let td = TempDir::new("unsupported");
    let map = byte_level_map();
    let tokens = vec![bl("a", &map)];
    let types = vec![1u32];
    let kvs: Vec<(&str, Kv)> = vec![
        ("tokenizer.ggml.model", Kv::Str("llama")),
        ("tokenizer.ggml.pre", Kv::Str("qwen35")),
        ("tokenizer.ggml.tokens", Kv::StrArr(&tokens)),
        ("tokenizer.ggml.merges", Kv::StrArr(&[])),
        ("tokenizer.ggml.token_type", Kv::U32Arr(&types)),
    ];
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![8, 4], T_F32, 0)];
    td.write("m.gguf", &gguf_bytes(&kvs, &tensors, 0));
    let out = run_text(td.as_path(), "a", &[]);
    assert_load_error(&out, "llama");
}

// ===== 8. 使い方の誤り =====

#[test]
fn no_input_source_is_exit_2() {
    let td = TempDir::new("nousage");
    td.write("m.gguf", &ascii_fixture(0));
    let out = denno()
        .arg("tokenize")
        .arg("--dir")
        .arg(td.as_path())
        .output()
        .expect("run");
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr_of(&out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert!(stderr_of(&out).contains("--text"), "{}", stderr_of(&out));
}

#[test]
fn text_and_stdin_together_is_exit_2() {
    let td = TempDir::new("bothusage");
    td.write("m.gguf", &ascii_fixture(0));
    let out = denno()
        .arg("tokenize")
        .arg("--dir")
        .arg(td.as_path())
        .arg("--text")
        .arg("a")
        .arg("--stdin")
        .output()
        .expect("run");
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr_of(&out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
}

// ===== 9. UTF-8 でない標準入力 =====

#[test]
fn non_utf8_stdin_is_exit_4() {
    let td = TempDir::new("badutf8");
    td.write("m.gguf", &ascii_fixture(0));

    let mut child = denno()
        .arg("tokenize")
        .arg("--dir")
        .arg(td.as_path())
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&[0xff, 0xfe])
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait");
    assert_eq!(out.status.code(), Some(4), "stderr: {}", stderr_of(&out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert!(stderr_of(&out).contains("UTF-8"), "{}", stderr_of(&out));
}

// ===== 10. 分割 GGUF のディレクトリ (パートを全部開く) =====

#[test]
fn split_dir_is_opened_by_all_parts() {
    // split.no / split.count を持つ 2 パート。トークナイザはパート 0 にだけ置く。
    let td = TempDir::new("split");
    let map = byte_level_map();
    let tokens = vec![bl("a", &map), bl("ab", &map), bl("b", &map)];
    let merges = vec![format!("{} {}", bl("a", &map), bl("b", &map))];
    let types = vec![1u32, 1, 1];

    for (no, name) in [(0u32, "m-00001-of-00002.gguf"), (1u32, "m-00002-of-00002.gguf")] {
        let mut kvs: Vec<(&str, Kv)> = vec![
            ("split.no", Kv::U32(no)),
            ("split.count", Kv::U32(2)),
            ("split.tensors.count", Kv::U32(2)),
        ];
        if no == 0 {
            kvs.push(("tokenizer.ggml.model", Kv::Str("gpt2")));
            kvs.push(("tokenizer.ggml.pre", Kv::Str("qwen35")));
            kvs.push(("tokenizer.ggml.tokens", Kv::StrArr(&tokens)));
            kvs.push(("tokenizer.ggml.merges", Kv::StrArr(&merges)));
            kvs.push(("tokenizer.ggml.token_type", Kv::U32Arr(&types)));
        }
        let tname = if no == 0 { "token_embd.weight" } else { "blk.0.attn_q.weight" };
        let tensors: Vec<TensorSpec> = vec![(tname, vec![8, 4], T_F32, 0)];
        td.write(name, &gguf_bytes(&kvs, &tensors, 0));
    }

    let out = run_ok(td.as_path(), "ab", &["--json"]);
    let v = json_of(&out);
    assert_eq!(v["token_count"], 1);
    assert_eq!(v["tokens"][0]["piece"], "ab");
    let parts = v["gguf"].as_array().expect("gguf array");
    assert_eq!(parts.len(), 2, "both shards must be listed: {v}");
    assert!(parts[0].as_str().expect("path").contains("00001"), "{v}");
    assert!(parts[1].as_str().expect("path").contains("00002"), "{v}");
}

// ===== 11. 単一ファイルを --dir に直接渡せる =====

#[test]
fn single_file_path_works() {
    let td = TempDir::new("singlefile");
    let p = td.write("m.gguf", &ascii_fixture(0));
    let out = run_ok(&p, "ab", &["--json"]);
    assert_eq!(json_of(&out)["token_count"], 1);
}
