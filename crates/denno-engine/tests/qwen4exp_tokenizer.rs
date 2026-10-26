//! P-MOE-F6: GGUF のメタデータからトークナイザを組む経路のテスト。
//!
//! 常に走る合成 GGUF のテスト (語彙は数十〜数百件) と、実 4 分割 GGUF があるときだけ
//! 走る ignored テスト (`DENNO_Q4_DIR`) からなる。GPU / CUDA は一切通さない。
//!
//! 実 GGUF:
//!   DENNO_Q4_DIR=<4 分割 GGUF のディレクトリ> \
//!     cargo test -p denno-engine --release --test qwen4exp_tokenizer -- --ignored --nocapture

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::tok_gguf::GgufTokenizer;

const ALIGN: usize = 32;

// GGUF メタデータ型 id (gguf.h enum gguf_type)
const GT_U32: u32 = 4;
const GT_I32: u32 = 5;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;

// tokenizer.ggml.token_type の値 (llama_token_type)
const TT_NORMAL: i32 = 1;
const TT_CONTROL: i32 = 3;

// ---------------------------------------------------------------------------
// 合成 GGUF の書き出し (メタデータのみ・テンソル 0 個)
// ---------------------------------------------------------------------------

enum Kv {
    U32(u32),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::Str(s) => {
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                push_str(b, s);
            }
            Kv::ArrStr(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for s in v {
                    push_str(b, s);
                }
            }
            Kv::ArrI32(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_I32.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// テンソル 0 個の GGUF v3 を書き出す。
fn write_gguf(path: &Path, kvs: &[(String, Kv)]) {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        push_str(&mut out, k);
        v.write(&mut out);
    }
    out.resize(align_up(out.len(), ALIGN), 0);
    std::fs::write(path, out).expect("テスト GGUF の書き出し");
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_tok_gguf_{tag}_{}_{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp dir");
        TempDir(p)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// 合成語彙
// ---------------------------------------------------------------------------

/// GPT-2 / Qwen の byte-level 写像 (byte → 表示文字)。
/// 33..=126 / 161..=172 / 174..=255 は同一文字、それ以外 (0xAD を含む) は
/// バイト順に U+0100 以降へ割り当てる。
fn byte_level_char(b: u8) -> char {
    fn printable(x: u8) -> bool {
        (33..=126).contains(&x) || (161..=172).contains(&x) || (174..=255).contains(&x)
    }
    if printable(b) {
        return b as char;
    }
    let mut n = 0u32;
    for x in 0u8..=255 {
        if printable(x) {
            continue;
        }
        if x == b {
            return char::from_u32(256 + n).unwrap();
        }
        n += 1;
    }
    unreachable!()
}

/// 256 個の byte-level 単体トークン (byte_fallback に頼らず任意の UTF-8 を通せる)。
fn byte_tokens() -> (Vec<String>, Vec<i32>) {
    let mut tokens = Vec::with_capacity(256);
    let mut types = Vec::with_capacity(256);
    for b in 0u8..=255 {
        tokens.push(byte_level_char(b).to_string());
        types.push(TT_NORMAL);
    }
    (tokens, types)
}

fn id_of(tokens: &[String], t: &str) -> u32 {
    tokens
        .iter()
        .position(|x| x == t)
        .unwrap_or_else(|| panic!("語彙に {t:?} が無い")) as u32
}

/// Qwen 系 chat_template の縮小版 (構造は実物と同じ: namespace / 文字列連結 / 条件)。
const TPL_MINIJINJA: &str = concat!(
    "{%- set ns = namespace(n=0) %}",
    "{%- if tools %}{{- '<|im_start|>system\\n' }}{{- '<|im_end|>\\n' }}{%- endif %}",
    "{%- for m in messages %}",
    "{%- if m.role == 'system' %}{{- '<|im_start|>system\\n' + m.content + '<|im_end|>\\n' }}",
    "{%- elif m.role == 'user' %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}",
    "{%- endif %}{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
);

/// minijinja が描画できないテンプレート (未知のフィルタ)。断片組み立て経路に落ちる。
/// 断片経路が使われたことを判別できるよう、user ブロックに `Z` のリテラルを挟む
/// (minijinja が描画できてしまえば `Z` が本文に出るが、断片経路は出さない)。
const TPL_UNRENDERABLE: &str = concat!(
    "{%- if tools %}{{- '<|im_start|>system\\n' }}{{- '<|im_end|>\\n' }}{%- endif %}",
    "{%- for m in messages %}",
    "{%- if m.role == 'user' %}{{- '<|im_start|>' + m.role + '\\n' + 'Z'",
    " + (m.content|nosuchfilter) + '<|im_end|>\\n' }}",
    "{%- endif %}{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
);

struct Spec {
    tokens: Vec<String>,
    types: Vec<i32>,
    merges: Vec<String>,
    pre: String,
    template: Option<String>,
}

impl Spec {
    /// byte-level 256 字 + 追加トークン。特殊トークンは末尾に足す。
    fn new() -> Self {
        let (tokens, types) = byte_tokens();
        Self {
            tokens,
            types,
            merges: Vec::new(),
            pre: "qwen35".into(),
            template: None,
        }
    }

    fn push_token(mut self, t: &str, ty: i32) -> Self {
        self.tokens.push(t.to_string());
        self.types.push(ty);
        self
    }

    fn merge(mut self, m: &str) -> Self {
        self.merges.push(m.to_string());
        self
    }

    fn pre(mut self, p: &str) -> Self {
        self.pre = p.to_string();
        self
    }

    fn template(mut self, t: &str) -> Self {
        self.template = Some(t.to_string());
        self
    }

    fn write(&self, dir: &TempDir, name: &str) -> PathBuf {
        let path = dir.path(name);
        let mut kvs: Vec<(String, Kv)> = vec![
            (
                "tokenizer.ggml.model".into(),
                Kv::Str("gpt2".into()),
            ),
            ("tokenizer.ggml.pre".into(), Kv::Str(self.pre.clone())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::ArrStr(self.tokens.clone()),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Kv::ArrI32(self.types.clone()),
            ),
            (
                "tokenizer.ggml.merges".into(),
                Kv::ArrStr(self.merges.clone()),
            ),
        ];
        // bos/eos は語彙内に収まる範囲で入れる
        if let Some(t) = self.tokens.iter().position(|x| x == "<|endoftext|>") {
            kvs.push((
                "tokenizer.ggml.bos_token_id".into(),
                Kv::U32(t as u32),
            ));
        }
        if let Some(t) = self.tokens.iter().position(|x| x == "<|im_end|>") {
            kvs.push((
                "tokenizer.ggml.eos_token_id".into(),
                Kv::U32(t as u32),
            ));
        }
        if let Some(t) = &self.template {
            kvs.push(("tokenizer.chat_template".into(), Kv::Str(t.clone())));
        }
        write_gguf(&path, &kvs);
        path
    }
}

fn load(path: &Path) -> GgufTokenizer {
    let f = GgufFile::open(path).expect("合成 GGUF を開けない");
    GgufTokenizer::from_gguf(&f).expect("トークナイザを組めない")
}

// ---------------------------------------------------------------------------
// 1. encode → decode の往復
// ---------------------------------------------------------------------------

#[test]
fn encode_decode_roundtrip_ascii_japanese_space() {
    let dir = TempDir::new("roundtrip");
    // 空白込みのマージを入れて、マージ後のトークンでも往復することを見る
    let spec = Spec::new()
        .push_token("<|endoftext|>", TT_CONTROL)
        .push_token("<|im_start|>", TT_CONTROL)
        .push_token("<|im_end|>", TT_CONTROL)
        .push_token("Ġt", TT_NORMAL)
        .push_token("Ġth", TT_NORMAL)
        .push_token("Ġthe", TT_NORMAL)
        .merge("Ġ t")
        .merge("Ġt h")
        .merge("Ġth e");
    let path = spec.write(&dir, "roundtrip.gguf");
    let tok = load(&path);

    // マージが効いていること (Ġthe が 1 トークン)
    assert_eq!(tok.encode(" the").unwrap(), vec![id_of(&spec.tokens, "Ġthe")]);

    for text in [
        "hello world",
        "  leading and  double spaces  ",
        "こんにちは世界",
        "a\nb\r\nc",
        "1+1は?",
        "tab\there",
        "",
    ] {
        let ids = tok.encode(text).unwrap();
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, text, "往復で元に戻らない: {text:?} -> {ids:?}");
    }
}

// ---------------------------------------------------------------------------
// 2. 特殊トークンが 1 個の id になる (分割されない)
// ---------------------------------------------------------------------------

#[test]
fn special_token_is_a_single_id() {
    let dir = TempDir::new("special");
    let spec = Spec::new()
        .push_token("<|endoftext|>", TT_CONTROL)
        .push_token("<|im_start|>", TT_CONTROL)
        .push_token("<|im_end|>", TT_CONTROL)
        .push_token("<think>", TT_CONTROL);
    let path = spec.write(&dir, "special.gguf");
    let tok = load(&path);

    let im_start = id_of(&spec.tokens, "<|im_start|>");
    let im_end = id_of(&spec.tokens, "<|im_end|>");
    let think = id_of(&spec.tokens, "<think>");

    // 単体では 1 個の id
    assert_eq!(tok.encode("<|im_start|>").unwrap(), vec![im_start]);
    assert_eq!(tok.encode("<|im_end|>").unwrap(), vec![im_end]);
    // 地の文と混ざっても特殊トークンは 1 個 (前後の文字と結合しない)
    let ids = tok.encode("a<|im_start|>b").unwrap();
    assert_eq!(
        ids,
        vec![
            id_of(&spec.tokens, "a"),
            im_start,
            id_of(&spec.tokens, "b")
        ]
    );
    // 4 つ目の特殊トークンも単体で 1 個
    assert_eq!(tok.encode("<think>").unwrap(), vec![think]);
}

// ---------------------------------------------------------------------------
// 3. merges の適用順 (手計算と一致)
// ---------------------------------------------------------------------------

#[test]
fn merges_are_applied_in_rank_order() {
    let dir = TempDir::new("merges");

    // (a) "a b" → "ab" が先に定義され、続いて "ab c" → "abc"
    let a = Spec::new()
        .push_token("ab", TT_NORMAL)
        .push_token("abc", TT_NORMAL)
        .merge("a b")
        .merge("ab c");
    let tok_a = load(&a.write(&dir, "a.gguf"));
    let ids = tok_a.encode("abc").unwrap();
    assert_eq!(
        ids,
        vec![id_of(&a.tokens, "abc")],
        "a b → ab c の順なら 1 トークン"
    );

    // (b) "b c" → "bc" が先。bc が語彙にあるので先に結合され、a + bc になる。
    let b = Spec::new()
        .push_token("ab", TT_NORMAL)
        .push_token("abc", TT_NORMAL)
        .push_token("bc", TT_NORMAL)
        .merge("b c")
        .merge("a b");
    let tok_b = load(&b.write(&dir, "b.gguf"));
    let ids = tok_b.encode("abc").unwrap();
    assert_eq!(
        ids,
        vec![id_of(&b.tokens, "a"), id_of(&b.tokens, "bc")],
        "b c が先なら a + bc"
    );

    // (c) 結合結果が語彙に無いマージは構築時に拒否される (黙って捨てない)
    let c = Spec::new().push_token("ab", TT_NORMAL).merge("b c").merge("a b");
    let path = c.write(&dir, "c.gguf");
    let f = GgufFile::open(&path).unwrap();
    let e = GgufTokenizer::from_gguf(&f)
        .err()
        .expect("語彙に無い結合結果のマージはエラーのはず");
    assert!(format!("{e}").contains("bc"), "エラー内容: {e}");
}

// ---------------------------------------------------------------------------
// 4. 未知の pre はエラー
// ---------------------------------------------------------------------------

#[test]
fn unknown_pre_is_an_error() {
    let dir = TempDir::new("badpre");
    let spec = Spec::new().pre("qwen9");
    let path = spec.write(&dir, "badpre.gguf");
    let f = GgufFile::open(&path).unwrap();
    let e = GgufTokenizer::from_gguf(&f)
        .err()
        .expect("未知の pre はエラーのはず");
    let msg = format!("{e}");
    assert!(msg.contains("qwen9"), "エラーに pre の値が出ていない: {msg}");
}

// ---------------------------------------------------------------------------
// 5. 語彙の重複はエラー (黙って上書きしない)
// ---------------------------------------------------------------------------

#[test]
fn duplicate_vocab_is_an_error() {
    let dir = TempDir::new("dup");
    let mut spec = Spec::new();
    spec.tokens.push("dup".into());
    spec.types.push(TT_NORMAL);
    spec.tokens.push("dup".into());
    spec.types.push(TT_NORMAL);
    let path = spec.write(&dir, "dup.gguf");
    let f = GgufFile::open(&path).unwrap();
    let e = GgufTokenizer::from_gguf(&f)
        .err()
        .expect("重複語彙はエラーのはず");
    let msg = format!("{e}");
    assert!(msg.contains("dup"), "エラーに重複トークンが出ていない: {msg}");
}

// ---------------------------------------------------------------------------
// encode_chat: minijinja 経路と断片組み立て経路
// ---------------------------------------------------------------------------

#[test]
fn encode_chat_uses_minijinja_when_it_renders() {
    let dir = TempDir::new("chat");
    let spec = Spec::new()
        .push_token("<|im_start|>", TT_CONTROL)
        .push_token("<|im_end|>", TT_CONTROL)
        .template(TPL_MINIJINJA);
    let tok = load(&spec.write(&dir, "chat.gguf"));

    // decode は特殊トークンを落とすので、期待する文字列を encode した id 列と比べる
    let ids = tok.encode_chat(None, "hi").unwrap();
    assert_eq!(
        ids,
        tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
            .unwrap()
    );

    let ids = tok.encode_chat(Some("S"), "hi").unwrap();
    assert_eq!(
        ids,
        tok.encode(
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        .unwrap()
    );
}

#[test]
fn encode_chat_falls_back_to_extracted_fragments() {
    let dir = TempDir::new("chatfb");
    let spec = Spec::new()
        .push_token("<|im_start|>", TT_CONTROL)
        .push_token("<|im_end|>", TT_CONTROL)
        .template(TPL_UNRENDERABLE);
    let tok = load(&spec.write(&dir, "chatfb.gguf"));

    // minijinja は未知のフィルタで失敗する → テンプレートから取り出した断片で組む。
    // 断片経路はテンプレート中の `Z` を本文に含めない (含めば minijinja 経路の証拠)。
    let ids = tok.encode_chat(None, "hi").unwrap();
    let expected = tok
        .encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        .unwrap();
    assert_eq!(ids, expected, "断片組み立ての結果が想定と違う");
    // minijinja 経路なら本文に Z が入る (断片経路は入れない)
    assert_ne!(
        ids,
        tok.encode("<|im_start|>user\nZhi<|im_end|>\n<|im_start|>assistant\n")
            .unwrap(),
        "minijinja 経路が使われている"
    );

    let ids = tok.encode_chat(Some("S"), "hi").unwrap();
    assert_eq!(
        ids,
        tok.encode(
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        .unwrap()
    );
}

#[test]
fn encode_chat_without_template_is_an_error() {
    let dir = TempDir::new("notpl");
    let spec = Spec::new();
    let tok = load(&spec.write(&dir, "notpl.gguf"));
    assert!(tok.encode_chat(None, "hi").is_err());
}

// ---------------------------------------------------------------------------
// 実 4 分割 GGUF (ignored)
// ---------------------------------------------------------------------------

fn real_parts() -> Option<Vec<PathBuf>> {
    let dir = PathBuf::from(std::env::var("DENNO_Q4_DIR").ok()?);
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|x| x == "gguf").unwrap_or(false)
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.contains("of-00004"))
                    .unwrap_or(false)
        })
        .collect();
    v.sort();
    if v.len() != 4 {
        return None;
    }
    Some(v)
}

fn real_tokenizer() -> Option<GgufTokenizer> {
    let parts = real_parts()?;
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    Some(GgufTokenizer::from_gguf(&f).expect("実 GGUF からトークナイザを組めない"))
}

/// 6. 日本語の encode → decode 往復
#[test]
#[ignore]
fn real_encode_decode_japanese() {
    let Some(tok) = real_tokenizer() else {
        eprintln!("DENNO_Q4_DIR が無いので skip");
        return;
    };
    // 目録の実値: 語彙 248320 件
    assert_eq!(tok.vocab_size(), 248320);

    let ids = tok.encode("こんにちは").unwrap();
    eprintln!("こんにちは -> {ids:?}");
    assert!(!ids.is_empty(), "空のトークン列");
    assert!(ids.len() < 64, "id が多すぎる: {}", ids.len());
    assert_eq!(tok.decode(&ids).unwrap(), "こんにちは");
}

/// 7. encode_chat の形 (先頭が user ブロック、末尾が assistant の開始)
#[test]
#[ignore]
fn real_encode_chat_shape() {
    let Some(tok) = real_tokenizer() else {
        eprintln!("DENNO_Q4_DIR が無いので skip");
        return;
    };
    let tpl = tok.chat_template().expect("chat_template が無い");
    assert!(tpl.contains("<|im_start|>"), "テンプレートに開始マーカーが無い");

    let ids = tok.encode_chat(None, "1+1は?").unwrap();
    let text = tok.decode(&ids).unwrap();
    eprintln!("encode_chat -> {text:?}");

    let user_head = tok.encode("<|im_start|>user").unwrap();
    assert!(
        ids.starts_with(&user_head),
        "先頭が <|im_start|>user でない: {:?}",
        &ids[..ids.len().min(8)]
    );

    // 末尾: <|im_start|>assistant が現れ、その後はテンプレート由来の短い前置だけ
    let asst = tok.encode("<|im_start|>assistant").unwrap();
    let pos = (0..=ids.len().saturating_sub(asst.len()))
        .find(|&i| ids[i..i + asst.len()] == asst[..])
        .unwrap_or_else(|| panic!("<|im_start|>assistant が無い: {text:?}"));
    assert!(
        ids.len() - pos <= asst.len() + 8,
        "assistant の後に余分なトークンがある: {:?}",
        &ids[pos..]
    );
}

/// 8. eos が 248046
#[test]
#[ignore]
fn real_eos_is_248046() {
    let Some(tok) = real_tokenizer() else {
        eprintln!("DENNO_Q4_DIR が無いので skip");
        return;
    };
    // 目録の実値: eos = 248046 / bos = 248044
    assert_eq!(tok.eos(), Some(248046));
    assert_eq!(tok.bos(), Some(248044));
}
