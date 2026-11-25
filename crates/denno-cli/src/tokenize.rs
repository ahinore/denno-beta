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
//! 終了コード (呼び出し側が機械的に分けられるようにする):
//! - 0: 成功 (空入力も成功 = 0 トークン)
//! - 2: 使い方の誤り (`--text` も `--stdin` も無い)
//! - 3: GGUF / トークナイザを読めない (パスが無い・.gguf が無い・壊れている)
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

/// `denno tokenize` の本体。stdout に結果を出し、終了コードを返す。
pub fn run(
    dir: &Path,
    text: Option<&str>,
    stdin: bool,
    json: bool,
) -> std::result::Result<(), TokenizeError> {
    // 入力の選び方: ちょうど 1 つ。両方/どちらも無しは使い方の誤り。
    let input: String = match (text, stdin) {
        (Some(t), false) => t.to_string(),
        (None, true) => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| TokenizeError::new(EXIT_INPUT, format!("標準入力を読めない: {e}")))?;
            String::from_utf8(buf).map_err(|_| {
                TokenizeError::new(EXIT_INPUT, "標準入力が UTF-8 ではない".to_string())
            })?
        }
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
    };

    let paths = gguf_paths(dir)?;
    let tok = load_tokenizer(&paths)?;
    let report = analyze(dir, &paths, &tok, &input)
        .map_err(|e| TokenizeError::load(format!("トークン化に失敗: {e:#}")))?;
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
