//! task 1111: `denno tokenize` のチャットテンプレート・プレビュー
//! (`--system` / `--user` / `--messages-json`) の契約試験。
//!
//! 確かめること:
//! 1. `--system` + `--user` が GGUF の `tokenizer.chat_template` で描画され、
//!    トークン数が描画結果と一致する (JSON とテキスト表示の両方)。
//! 2. `--user` 単独では system ブロックが入らない。
//! 3. `--messages-json` の配列形 / オブジェクト形が同じ描画になる。
//! 4. 壊れた / 欠けた `--messages-json` は終了コード 2 で理由を出す。
//! 5. chat_template の無い GGUF は推測で組まず終了コード 3。
//! 6. 既存の `--text` / `--stdin` の出力は不変 (JSON に `chat` キーが無い、
//!    テキストに `prompt:` 行が無い) — tokenize_941 / tokenize_edge_1149 と
//!    同じ合成 GGUF の形を使う。
//!
//! 合成 GGUF はトークナイザ・メタデータだけ (データ部 0 バイト)。全 256 バイトを
//! 1 バイト 1 トークンで持つ byte-level 語彙なので、描画結果のバイト数が
//! そのままトークン数になり、復号が必ず往復する。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
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
            "denno1111_{}_{}_{}_{}",
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

    fn path(&self) -> &Path {
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

/// GPT-2 / llama.cpp の byte -> unicode 写像 (全 256 バイト)。
fn byte_level_map() -> Vec<char> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(0x21u32..=0x7E);
    bs.extend(0xA1u32..=0xAC);
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

/// ChatML 風のテンプレート (minijinja で描画できる最小形)。
///
/// `add_generation_prompt` を true で渡すので、最後に assistant の開始が付く。
const CHAT_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// 全 256 バイトを 1 バイト 1 トークンで持つ GGUF。`template` が true なら
/// chat_template をメタデータに足す。
fn byte_gguf(with_template: bool) -> Vec<u8> {
    let map = byte_level_map();
    let tokens: Vec<String> = map.iter().map(|c| c.to_string()).collect();
    let types: Vec<u32> = vec![1u32; tokens.len()];
    let mut kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("tokenizer.ggml.model", Kv::Str("gpt2")),
        ("tokenizer.ggml.pre", Kv::Str("qwen35")),
        ("tokenizer.ggml.tokens", Kv::StrArr(&tokens)),
        ("tokenizer.ggml.merges", Kv::StrArr(&[])),
        ("tokenizer.ggml.token_type", Kv::U32Arr(&types)),
    ];
    if with_template {
        kvs.push(("tokenizer.chat_template", Kv::Str(CHAT_TEMPLATE)));
    }
    let tensors: Vec<TensorSpec> = vec![("token_embd.weight", vec![8, 4], T_F32, 0)];
    gguf_bytes(&kvs, &tensors, 0)
}

fn fixture(tag: &str, with_template: bool) -> TempDir {
    let td = TempDir::new(tag);
    td.write("m.gguf", &byte_gguf(with_template));
    td
}

// ===== CLI 起動 =====

fn denno() -> Command {
    Command::new(env!("CARGO_BIN_EXE_denno"))
}

fn run(dir: &Path, extra: &[&str]) -> Output {
    let mut c = denno();
    c.arg("tokenize").arg("--dir").arg(dir);
    for e in extra {
        c.arg(e);
    }
    c.output().expect("run denno tokenize")
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

fn run_ok(dir: &Path, extra: &[&str]) -> Output {
    let out = run(dir, extra);
    assert!(
        out.status.success(),
        "denno tokenize failed ({}): {}",
        extra.join(" "),
        stderr_of(&out)
    );
    out
}

/// 1 ターン (system 無し) の期待描画。
fn expected_one(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// system + user の期待描画。
fn expected_sys(sys: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

// ===== 1. --system + --user =====

#[test]
fn system_and_user_render_through_chat_template() {
    let td = fixture("sysuser", true);
    let sys = "あなたは簡潔に答える。";
    let user = "こんにちは";

    let v = json_of(&run_ok(
        td.path(),
        &["--system", sys, "--user", user, "--json"],
    ));
    let rendered = expected_sys(sys, user);
    assert_eq!(v["chat"]["rendered"], rendered);
    assert_eq!(v["chat"]["system"], sys);
    assert_eq!(v["chat"]["messages"].as_array().expect("messages").len(), 1);
    assert_eq!(v["chat"]["messages"][0]["role"], "user");
    assert_eq!(v["chat"]["messages"][0]["content"], user);

    // 数える対象は描画結果 (バイト = トークン、この語彙では 1 対 1)。
    assert_eq!(v["text_bytes"].as_u64().expect("bytes"), rendered.len() as u64);
    assert_eq!(v["text_chars"].as_u64().expect("chars"), rendered.chars().count() as u64);
    assert_eq!(v["token_count"].as_u64().expect("count"), rendered.len() as u64);
    assert_eq!(v["decoded"], rendered, "描画結果が往復する");

    // 人が読む表示にも描画結果が出る (改行はエスケープ)。
    let text = stdout_of(&run_ok(td.path(), &["--system", sys, "--user", user]));
    assert!(text.contains("chat: system=yes messages=1"), "{text}");
    assert!(
        text.contains("prompt: \"<|im_start|>system\\n"),
        "prompt line missing or unescaped: {text}"
    );
    assert!(text.contains("tokens: "), "{text}");
}

#[test]
fn user_alone_has_no_system_block() {
    let td = fixture("useronly", true);
    let user = "abc";
    let v = json_of(&run_ok(td.path(), &["--user", user, "--json"]));
    assert_eq!(v["chat"]["rendered"], expected_one(user));
    assert!(v["chat"]["system"].is_null(), "system must stay null: {v}");
    assert_eq!(v["token_count"].as_u64().expect("count"), expected_one(user).len() as u64);

    let text = stdout_of(&run_ok(td.path(), &["--user", user]));
    assert!(text.contains("chat: system=no messages=1"), "{text}");
}

// ===== 2. --messages-json (配列形 / オブジェクト形) =====

#[test]
fn messages_json_array_form_matches_multiturn_rendering() {
    let td = fixture("msgarr", true);
    let p = td.write(
        "msgs.json",
        br#"[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"},{"role":"user","content":"q2"}]"#,
    );
    let v = json_of(&run_ok(
        td.path(),
        &["--messages-json", p.to_str().expect("utf8 path"), "--json"],
    ));
    let expected = "<|im_start|>user\nq1<|im_end|>\n\
                    <|im_start|>assistant\na1<|im_end|>\n\
                    <|im_start|>user\nq2<|im_end|>\n\
                    <|im_start|>assistant\n";
    assert_eq!(v["chat"]["rendered"], expected);
    assert_eq!(v["chat"]["messages"].as_array().expect("messages").len(), 3);
    assert!(v["chat"]["system"].is_null());
    assert_eq!(v["token_count"].as_u64().expect("count"), expected.len() as u64);
}

#[test]
fn messages_json_object_form_carries_system() {
    let td = fixture("msgobj", true);
    let p = td.write(
        "msgs.json",
        br#"{"system":"S","messages":[{"role":"user","content":"U"}]}"#,
    );
    let v = json_of(&run_ok(
        td.path(),
        &["--messages-json", p.to_str().expect("utf8 path"), "--json"],
    ));
    assert_eq!(v["chat"]["rendered"], expected_sys("S", "U"));
    assert_eq!(v["chat"]["system"], "S");
}

// ===== 3. --messages-json の失敗は終了コード 2 =====

fn assert_usage_error(out: &Output, needles: &[&str]) {
    assert!(!out.status.success(), "should fail: {}", stderr_of(out));
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr_of(out));
    let e = stderr_of(out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    for n in needles {
        assert!(e.contains(n), "stderr should contain {n:?}: {e}");
    }
}

#[test]
fn malformed_messages_json_is_exit_2() {
    let td = fixture("badjson", true);
    let p = td.write("bad.json", b"{not json");
    let out = run(td.path(), &["--messages-json", p.to_str().expect("utf8 path")]);
    assert_usage_error(&out, &["--messages-json", "JSON"]);
}

#[test]
fn missing_messages_json_is_exit_2() {
    let td = fixture("nojson", true);
    let missing = td.path().join("nope.json");
    let out = run(
        td.path(),
        &["--messages-json", missing.to_str().expect("utf8 path")],
    );
    assert_usage_error(&out, &["--messages-json", "読めない"]);
}

#[test]
fn wrong_message_shape_is_exit_2() {
    let td = fixture("shape", true);
    // content が文字列でない。
    let p = td.write("a.json", br#"[{"role":"user","content":3}]"#);
    let out = run(td.path(), &["--messages-json", p.to_str().expect("utf8 path")]);
    assert_usage_error(&out, &["messages[0].content"]);
}

#[test]
fn empty_messages_json_is_exit_2() {
    let td = fixture("emptyarr", true);
    let p = td.write("a.json", b"[]");
    let out = run(td.path(), &["--messages-json", p.to_str().expect("utf8 path")]);
    assert_usage_error(&out, &["空"]);
}

// ===== 4. 入力源の同時指定 / system 単独 =====

#[test]
fn user_and_text_together_is_rejected() {
    let td = fixture("both", true);
    let out = run(td.path(), &["--text", "a", "--user", "u"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr_of(&out));
    assert!(out.stdout.is_empty());
}

#[test]
fn system_without_user_is_exit_2() {
    let td = fixture("sysonly", true);
    let out = run(td.path(), &["--system", "s"]);
    assert_usage_error(&out, &["--user"]);
}

#[test]
fn messages_json_with_user_is_rejected() {
    let td = fixture("msguser", true);
    let p = td.write("a.json", br#"[{"role":"user","content":"x"}]"#);
    let out = run(
        td.path(),
        &[
            "--messages-json",
            p.to_str().expect("utf8 path"),
            "--user",
            "u",
        ],
    );
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr_of(&out));
}

// ===== 5. chat_template が無い GGUF は終了コード 3 =====

#[test]
fn gguf_without_chat_template_is_exit_3() {
    let td = fixture("notpl", false);
    let out = run(td.path(), &["--user", "u"]);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty(), "stdout must stay empty");
    assert_eq!(out.status.code(), Some(3), "stderr: {}", stderr_of(&out));
    let e = stderr_of(&out);
    assert!(!e.contains("panicked"), "must not panic: {e}");
    assert!(e.contains("chat_template"), "stderr: {e}");
}

// ===== 6. 既存の --text / --stdin は不変 =====

#[test]
fn plain_text_json_has_no_chat_key() {
    let td = fixture("compat", true);
    let v = json_of(&run_ok(td.path(), &["--text", "ab", "--json"]));
    assert_eq!(v["token_count"], 2);
    assert!(
        v.get("chat").is_none(),
        "chat key must not appear for --text: {v}"
    );

    let text = stdout_of(&run_ok(td.path(), &["--text", "ab"]));
    assert!(text.contains("tokens: 2"), "{text}");
    assert!(!text.contains("prompt:"), "prompt line must not appear: {text}");
    assert!(!text.contains("chat:"), "chat line must not appear: {text}");
}
