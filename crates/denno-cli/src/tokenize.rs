//! task 941: `denno tokenize --dir <path> [--text <str> | --stdin] [--json]`。
//!
//! GGUF の **トークナイザ・メタデータだけ** を読んで文字列をトークン化する。
//! 重み (テンソルのデータ部) は 1 バイトも読まない: `GgufFile` はデータ部を
//! mmap するだけでページに触らず、ここで使うのは `tokenizer.ggml.*` の
//! 語彙 / マージ / 特殊トークン種別だけ。40 GB のモデルでも数 MiB の読み取りで
//! 終わる (task 921 の `inspect` と同じ性質)。
//!
//! 出すもの: トークン数 / トークン id / 断片 (復号した文字列) / 全体の復号結果。
//!
//! task 1111: チャットプレビュー。`--system <TEXT>` / `--user <TEXT>` か
//! `--messages-json <PATH>` を渡すと、GGUF の `tokenizer.chat_template` で
//! プロンプトを描画してから同じ形 (トークン数 / id / 断片) で出す。描画結果は
//! `chat.rendered` (JSON) と `prompt:` 行 (テキスト) に載る。テンプレートが
//! 無い / 描画できない GGUF は推測で組まずに 3 で落ちる。
//! 既存の `--text` / `--stdin` の出力 (JSON のキーと意味) は変えない:
//! `chat` キーはチャットプレビューのときだけ現れる。
//!
//! 終了コード (呼び出し側が機械的に分けられるようにする):
//! - 0: 成功 (空入力も成功 = 0 トークン)
//! - 2: 使い方の誤り (`--text` も `--stdin` も無い / 入力源の同時指定 /
//!      `--messages-json` が無い・読めない・JSON として壊れている)
//! - 3: GGUF / トークナイザを読めない (パスが無い・.gguf が無い・壊れている・
//!      chat_template が無い)
//! - 4: 標準入力が UTF-8 でない
//!
//! 数値経路 (推論) には触れない。既定の挙動は不変。

use anyhow::Result;
use denno_engine::gguf::GgufFile;
use denno_engine::tok_gguf::GgufTokenizer;
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

/// 使い方の誤り。
pub const EXIT_USAGE: i32 = 2;
/// GGUF / トークナイザを読めない。
pub const EXIT_LOAD: i32 = 3;
/// 標準入力が UTF-8 でない。
pub const EXIT_INPUT: i32 = 4;

/// `run` の失敗。終了コードと 1 行の理由を持つ (main が stderr に出して exit する)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizeError {
    pub code: i32,
    pub message: String,
}

impl TokenizeError {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn load(message: impl Into<String>) -> Self {
        Self::new(EXIT_LOAD, message)
    }

    fn usage(message: impl Into<String>) -> Self {
        Self::new(EXIT_USAGE, message)
    }
}

/// トークン 1 個。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenEntry {
    /// 語彙 id
    pub id: u32,
    /// 復号した断片。特殊トークン (decode が空になる) は語彙上の文字列を使う。
    pub piece: String,
    /// 語彙上の生の文字列 (byte-level では `Ġ` などに逃がした形)。特殊トークン以外は
    /// 復号結果と違うことがあるので診断用に残す。
    pub raw: String,
}

/// チャットプレビューの 1 メッセージ (描画に渡した形のまま)。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatMessageEntry {
    pub role: String,
    pub content: String,
}

/// task 1111: `--system` / `--user` / `--messages-json` のプレビュー情報。
///
/// `rendered` が chat_template の描画結果で、`token_count` / `tokens` /
/// `decoded` / `text_bytes` / `text_chars` はすべてこの文字列に対する値。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatPreview {
    pub system: Option<String>,
    pub messages: Vec<ChatMessageEntry>,
    /// chat_template を描画した生のプロンプト (トークン化の入力そのもの)
    pub rendered: String,
}

/// `denno tokenize` の結果。`--json` はこれをそのまま出す。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenizeReport {
    /// 指定された `--dir`
    pub dir: String,
    /// 実際に開いた GGUF (分割なら split.no 昇順)
    pub gguf: Vec<String>,
    pub vocab_size: usize,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /// 入力の長さ (バイト / 文字)
    pub text_bytes: usize,
    pub text_chars: usize,
    pub token_count: usize,
    pub tokens: Vec<TokenEntry>,
    /// トークン列全体の復号結果 (入力の往復検査に使える)
    pub decoded: String,
    /// task 1111: チャットプレビューのときだけ入る。既存の `--text` /
    /// `--stdin` ではキー自体を出さない (JSON の互換を保つ)。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatPreview>,
}

/// `--dir` から GGUF のパスを解決する。
///
/// ファイルならそれ 1 本、ディレクトリなら直下の `.gguf` を名前順に全本
/// (分割 GGUF はパートを全部そろえる必要がある)。1 本も無ければエラー。
pub fn gguf_paths(dir: &Path) -> std::result::Result<Vec<PathBuf>, TokenizeError> {
    if !dir.exists() {
        return Err(TokenizeError::load(format!("{} が存在しない", dir.display())));
    }
    if dir.is_file() {
        return Ok(vec![dir.to_path_buf()]);
    }
    let rd = std::fs::read_dir(dir)
        .map_err(|e| TokenizeError::load(format!("{} を読めない: {e}", dir.display())))?;
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| TokenizeError::load(format!("{}: {e}", dir.display())))?;
        let p = entry.path();
        if p.extension().map(|x| x == "gguf").unwrap_or(false) {
            paths.push(p);
        }
    }
    if paths.is_empty() {
        return Err(TokenizeError::load(format!(
            "{} に .gguf が 1 本も無い",
            dir.display()
        )));
    }
    paths.sort();
    Ok(paths)
}

/// GGUF からトークナイザを組む (データ部は読まない)。
pub fn load_tokenizer(paths: &[PathBuf]) -> std::result::Result<GgufTokenizer, TokenizeError> {
    let f = GgufFile::open_split(paths)
        .map_err(|e| TokenizeError::load(format!("GGUF を開けない: {}", e.0)))?;
    GgufTokenizer::from_gguf(&f)
        .map_err(|e| TokenizeError::load(format!("トークナイザを組めない: {e:#}")))
}

/// テキストをトークン化してレポートを組む (純関数)。
fn analyze(
    dir: &Path,
    paths: &[PathBuf],
    tok: &GgufTokenizer,
    text: &str,
) -> Result<TokenizeReport> {
    let ids = tok.encode(text)?;
    let mut tokens = Vec::with_capacity(ids.len());
    for &id in &ids {
        let raw = tok.token_text(id).unwrap_or_default();
        let piece = match tok.decode(&[id]) {
            Ok(s) if !s.is_empty() => s,
            // 特殊トークンは decode (skip_special_tokens) が空を返す。語彙の
            // 文字列を見せる (何が入ったか分からないと診断できない)。
            _ => raw.clone(),
        };
        tokens.push(TokenEntry { id, piece, raw });
    }
    let decoded = tok.decode(&ids)?;
    Ok(TokenizeReport {
        dir: dir.display().to_string(),
        gguf: paths.iter().map(|p| p.display().to_string()).collect(),
        vocab_size: tok.vocab_size(),
        bos: tok.bos(),
        eos: tok.eos(),
        text_bytes: text.len(),
        text_chars: text.chars().count(),
        token_count: ids.len(),
        tokens,
        decoded,
        chat: None,
    })
}

/// 制御文字を目で見える形にする (1 行 1 トークンの表示用)。
fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            c if c.is_control() => o.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn show_id(v: Option<u32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string())
}

/// 人が読む形 (1 行 1 トークン)。
pub fn render_text(r: &TokenizeReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("dir: {}\n", r.dir));
    for (i, p) in r.gguf.iter().enumerate() {
        s.push_str(&format!("gguf[{i}]: {p}\n"));
    }
    s.push_str(&format!(
        "vocab: {} (bos={} eos={})\n",
        r.vocab_size,
        show_id(r.bos),
        show_id(r.eos)
    ));
    s.push_str(&format!(
        "input: {} bytes / {} chars\n",
        r.text_bytes, r.text_chars
    ));
    if let Some(c) = &r.chat {
        s.push_str(&format!(
            "chat: system={} messages={}\n",
            if c.system.as_deref().is_some_and(|x| !x.is_empty()) {
                "yes"
            } else {
                "no"
            },
            c.messages.len()
        ));
        s.push_str(&format!("prompt: \"{}\"\n", escape(&c.rendered)));
    }
    s.push_str(&format!("tokens: {}\n", r.token_count));
    for (i, t) in r.tokens.iter().enumerate() {
        s.push_str(&format!(
            "  {:>6}  {:>8}  \"{}\"\n",
            i,
            t.id,
            escape(&t.piece)
        ));
    }
    s.push_str(&format!("decoded: \"{}\"\n", escape(&r.decoded)));
    s
}

/// `denno tokenize` の引数。入力源は「テキスト」か「チャットプレビュー」の
/// どちらか一方だけ (main 側の clap でも同じ制約を掛けている)。
#[derive(Debug, Clone)]
pub struct RunArgs<'a> {
    pub dir: &'a Path,
    /// `--text` の文字列
    pub text: Option<&'a str>,
    /// `--stdin` を読む
    pub stdin: bool,
    /// `--system` (チャットプレビュー)
    pub system: Option<&'a str>,
    /// `--user` (チャットプレビュー)
    pub user: Option<&'a str>,
    /// `--messages-json <PATH>` (チャットプレビュー)
    pub messages_json: Option<&'a Path>,
    /// `--json`
    pub json: bool,
}

/// task 1111: `--messages-json` の JSON を (system, messages) に解く。
///
/// 受ける形は 2 つ:
/// - `[{"role": "user", "content": "..."}, ...]`
/// - `{"system": "...", "messages": [...]}` (`system` は省略可)
///
/// 壊れている / 形が違うときは終了コード 2 の理由を返す (呼ぶ側がパスを前置する)。
fn parse_messages_json(
    raw: &str,
) -> std::result::Result<(Option<String>, Vec<ChatMessageEntry>), TokenizeError> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
        TokenizeError::usage(format!(
            "JSON として読めない: {e} ([{{\"role\":\"user\",\"content\":\"...\"}}] か \
             {{\"messages\":[...]}} の形で書く)"
        ))
    })?;
    let (system, arr) = match &v {
        serde_json::Value::Array(a) => (None, a),
        serde_json::Value::Object(o) => {
            let system = match o.get("system") {
                None | Some(serde_json::Value::Null) => None,
                Some(serde_json::Value::String(s)) => Some(s.clone()),
                Some(_) => {
                    return Err(TokenizeError::usage(
                        "system が文字列でない (文字列で書く)".to_string(),
                    ))
                }
            };
            let arr = o.get("messages").and_then(|m| m.as_array()).ok_or_else(|| {
                TokenizeError::usage(
                    "messages が配列でない ({\"messages\":[{\"role\":...,\"content\":...}]} の形で書く)"
                        .to_string(),
                )
            })?;
            (system, arr)
        }
        _ => {
            return Err(TokenizeError::usage(
                "配列か {\"messages\":[...]} のオブジェクトで書く".to_string(),
            ))
        }
    };
    if arr.is_empty() {
        return Err(TokenizeError::usage(
            "messages が空 (role と content を持つ要素を 1 件以上書く)".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let obj = m.as_object().ok_or_else(|| {
            TokenizeError::usage(format!("messages[{i}] がオブジェクトでない"))
        })?;
        let role = obj.get("role").and_then(|x| x.as_str()).ok_or_else(|| {
            TokenizeError::usage(format!("messages[{i}].role が文字列でない"))
        })?;
        let content = obj.get("content").and_then(|x| x.as_str()).ok_or_else(|| {
            TokenizeError::usage(format!("messages[{i}].content が文字列でない"))
        })?;
        if role.is_empty() {
            return Err(TokenizeError::usage(format!(
                "messages[{i}].role が空 (user / assistant / system などを書く)"
            )));
        }
        out.push(ChatMessageEntry {
            role: role.to_string(),
            content: content.to_string(),
        });
    }
    Ok((system, out))
}

/// 標準入力から UTF-8 を読む。
fn read_stdin() -> std::result::Result<String, TokenizeError> {
    let mut buf = Vec::new();
    std::io::stdin()
        .read_to_end(&mut buf)
        .map_err(|e| TokenizeError::new(EXIT_INPUT, format!("標準入力を読めない: {e}")))?;
    String::from_utf8(buf)
        .map_err(|_| TokenizeError::new(EXIT_INPUT, "標準入力が UTF-8 ではない".to_string()))
}

/// `run` が入力を決めた結果 (トークナイザを開く前の段階)。
enum Resolved {
    Text(String),
    Chat(Option<String>, Vec<ChatMessageEntry>),
}

/// `denno tokenize` の本体。stdout に結果を出し、終了コードを返す。
pub fn run(args: RunArgs<'_>) -> std::result::Result<(), TokenizeError> {
    let RunArgs {
        dir,
        text,
        stdin,
        system,
        user,
        messages_json,
        json,
    } = args;
    let chat_mode = system.is_some() || user.is_some() || messages_json.is_some();

    // 先に入力源を決める。使い方の誤りと --messages-json の失敗は GGUF を
    // 開く前に返す (2 が 3 に化けないようにする)。
    let resolved = if chat_mode {
        // テキスト入力とチャットプレビューは混ぜない (どちらを数えたのか
        // 分からなくなるため)。clap の conflicts と二重に守る。
        if text.is_some() || stdin {
            return Err(TokenizeError::usage(
                "--text / --stdin と --system / --user / --messages-json は同時に指定できない",
            ));
        }
        if let Some(path) = messages_json {
            if system.is_some() || user.is_some() {
                return Err(TokenizeError::usage(
                    "--messages-json と --system / --user は同時に指定できない",
                ));
            }
            let raw = std::fs::read_to_string(path).map_err(|e| {
                TokenizeError::usage(format!(
                    "--messages-json {} を読めない: {e} (UTF-8 の JSON ファイルを指定する)",
                    path.display()
                ))
            })?;
            let (system, messages) = parse_messages_json(&raw).map_err(|e| TokenizeError {
                code: e.code,
                message: format!("--messages-json {}: {}", path.display(), e.message),
            })?;
            Resolved::Chat(system, messages)
        } else {
            let Some(user) = user else {
                return Err(TokenizeError::usage(
                    "--system を使うときは --user も指定する",
                ));
            };
            Resolved::Chat(
                system.map(|s| s.to_string()),
                vec![ChatMessageEntry {
                    role: "user".to_string(),
                    content: user.to_string(),
                }],
            )
        }
    } else {
        Resolved::Text(match (text, stdin) {
            (Some(t), false) => t.to_string(),
            (None, true) => read_stdin()?,
            (Some(_), true) => {
                return Err(TokenizeError::usage(
                    "--text と --stdin は同時に指定できない",
                ))
            }
            (None, false) => {
                return Err(TokenizeError::usage(
                    "--text <str> か --stdin のどちらかを指定する",
                ))
            }
        })
    };

    let paths = gguf_paths(dir)?;
    let tok = load_tokenizer(&paths)?;
    let (input, chat) = match resolved {
        Resolved::Text(t) => (t, None),
        Resolved::Chat(system, messages) => {
            let pairs: Vec<(String, String)> = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            // 描画は engine と同じ minijinja 経路 (enable_thinking=false は
            // encode_chat / 1 ターンと同じ条件)。
            let rendered = tok
                .render_messages(system.as_deref(), &pairs, None, false)
                .ok_or_else(|| {
                    TokenizeError::load(
                        "GGUF に tokenizer.chat_template が無いか、描画に失敗した \
                         (chat_template を持つモデルを指定する)",
                    )
                })?;
            let preview = ChatPreview {
                system,
                messages,
                rendered: rendered.clone(),
            };
            (rendered, Some(preview))
        }
    };

    let mut report = analyze(dir, &paths, &tok, &input)
        .map_err(|e| TokenizeError::load(format!("トークン化に失敗: {e:#}")))?;
    report.chat = chat;
    if json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| {
            TokenizeError::load(format!("JSON にできない: {e}"))
        })?);
    } else {
        print!("{}", render_text(&report));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_makes_control_chars_visible() {
        assert_eq!(escape("a\nb"), "a\\nb");
        assert_eq!(escape("\"\\"), r#"\"\\"#);
        assert_eq!(escape("\u{1}"), "\\u{1}");
        assert_eq!(escape("こん"), "こん");
    }

    #[test]
    fn missing_dir_is_load_error() {
        let p = std::env::temp_dir().join("denno941_definitely_missing_dir");
        let e = gguf_paths(&p).unwrap_err();
        assert_eq!(e.code, EXIT_LOAD);
        assert!(e.message.contains("存在しない"), "{}", e.message);
    }

    #[test]
    fn show_id_marks_unknown() {
        assert_eq!(show_id(None), "?");
        assert_eq!(show_id(Some(0)), "0");
    }
}
