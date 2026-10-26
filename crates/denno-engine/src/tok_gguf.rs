//! P-MOE-F6: GGUF のメタデータから GPT-2 系 BPE のトークナイザを組み立てる。
//!
//! 配布物が GGUF だけで HF の `tokenizer.json` が無いモデルでもプロンプトを組める
//! ようにする経路。`tokenizer.ggml.*` (語彙 / マージ / 特殊トークン種別 / pre /
//! chat_template) を読み、`tokenizers` クレートの `Tokenizer` を JSON 経由で構築する。
//!
//! 既存の `tokenizer.rs` (HF ディレクトリ前提) には手を入れない。`DennoTokenizer` と
//! 同じ minijinja 設定で chat_template を描画するが、描画できないテンプレートのために
//! 「テンプレート文字列から ChatML の固定断片を取り出して組み立てる」経路も持つ
//! (`ChatFragments`)。

use anyhow::{anyhow, bail, Result};

use crate::gguf::{GgufFile, GgufValue};

/// Qwen 系 (qwen2 / qwen3 / qwen35) の事前分割。HF の tokenizer.json の
/// `pre_tokenizer.pretokenizers[0].pattern.Regex` と同一の文字列。
const QWEN_SPLIT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

// tokenizer.ggml.token_type の値 (gguf 仕様 / llama.cpp の llama_token_type)
const TT_UNKNOWN: u32 = 2;
const TT_CONTROL: u32 = 3;
const TT_USER_DEFINED: u32 = 4;

/// GGUF メタデータから組んだトークナイザ。
pub struct GgufTokenizer {
    tok: tokenizers::Tokenizer,
    chat_template: Option<String>,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl GgufTokenizer {
    /// GGUF のメタデータから GPT-2 系 BPE の Tokenizer を組み立てる。
    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let model = meta_str(f, "tokenizer.ggml.model")?;
        if model != "gpt2" {
            bail!("tokenizer.ggml.model = {model:?} は未対応 (gpt2 のみ)");
        }
        // pre は未知の値を既定に落とさない (分割規則が変わると id 列が変わるため)。
        let pre = meta_str(f, "tokenizer.ggml.pre")?;
        let pattern = match pre.as_str() {
            "qwen35" | "qwen2" | "qwen3" => QWEN_SPLIT_PATTERN,
            other => bail!("tokenizer.ggml.pre = {other:?} は未対応 (qwen35/qwen2/qwen3 のみ)"),
        };

        let tokens = meta_str_array(f, "tokenizer.ggml.tokens")?;
        let merges = meta_str_array(f, "tokenizer.ggml.merges")?;
        if tokens.is_empty() {
            bail!("tokenizer.ggml.tokens が空");
        }

        // 語彙: トークン文字列 → id (GGUF の並び順が id)。重複は黙って上書きしない。
        let mut vocab = serde_json::Map::with_capacity(tokens.len());
        let mut seen: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::with_capacity(tokens.len());
        for (id, t) in tokens.iter().enumerate() {
            if t.is_empty() {
                bail!("語彙 id {id} が空文字列");
            }
            if let Some(prev) = seen.insert(t.as_str(), id) {
                bail!("語彙に重複したトークン {t:?} (id {prev} と {id})");
            }
            vocab.insert(t.clone(), serde_json::json!(id));
        }

        // merges は "a b" 形式。空白で 2 分割する (piece 自体に空白は現れない:
        // byte-level BPE では空白は Ġ にエスケープされる)。
        let mut pairs: Vec<serde_json::Value> = Vec::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let (a, b) = m
                .split_once(' ')
                .ok_or_else(|| anyhow!("merges[{i}] = {m:?} に空白区切りが無い"))?;
            if a.is_empty() || b.is_empty() {
                bail!("merges[{i}] = {m:?} の片側が空");
            }
            pairs.push(serde_json::json!([a, b]));
        }

        // 特殊トークン: control / user_defined を added_tokens として登録する。
        // normalized:false が要点 (特殊トークンが事前分割でバラされない)。
        let mut added: Vec<serde_json::Value> = Vec::new();
        let mut unk: Option<String> = None;
        if let Some(v) = f.get("tokenizer.ggml.token_type") {
            let GgufValue::Arr(types) = v else {
                bail!("tokenizer.ggml.token_type が配列でない");
            };
            if types.len() != tokens.len() {
                bail!(
                    "tokenizer.ggml.token_type の要素数 {} が語彙数 {} と違う",
                    types.len(),
                    tokens.len()
                );
            }
            for (id, v) in types.iter().enumerate() {
                let ty = v
                    .as_u64()
                    .ok_or_else(|| anyhow!("token_type[{id}] が整数でない"))?
                    as u32;
                match ty {
                    TT_UNKNOWN => unk = Some(tokens[id].clone()),
                    TT_CONTROL | TT_USER_DEFINED => added.push(serde_json::json!({
                        "id": id,
                        "content": tokens[id],
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": true,
                    })),
                    _ => {}
                }
            }
        }

        let tok_json = serde_json::json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": pattern},
                        "behavior": "Isolated",
                        "invert": false,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": false,
                        "use_regex": false,
                    },
                ],
            },
            "post_processor": serde_json::Value::Null,
            // decoder も ByteLevel。tokenizers 0.21 の ByteLevel は add_prefix_space /
            // trim_offsets に serde 既定値を持たないため、decode では未使用でも
            // 3 フィールドを明示する (欠けると構築が落ちる)。
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false,
            },
            "model": {
                "type": "BPE",
                "vocab": serde_json::Value::Object(vocab),
                "merges": pairs,
                "unk_token": unk,
                "byte_fallback": true,
            },
        });
        let bytes = serde_json::to_vec(&tok_json)?;
        let tok = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| anyhow!("tokenizer の構築に失敗: {e}"))?;

        let bos = meta_token_id(f, "tokenizer.ggml.bos_token_id", tokens.len())?;
        let eos = meta_token_id(f, "tokenizer.ggml.eos_token_id", tokens.len())?;
        let chat_template = f
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Self {
            tok,
            chat_template,
            bos,
            eos,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("encode失敗: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok
            .decode(ids, true)
            .map_err(|e| anyhow!("decode失敗: {e}"))
    }

    /// 1 ターンの user メッセージを chat_template で展開してトークン列にする。
    ///
    /// `system` は空文字列なら付かない。テンプレートが無い場合は推測で組み立てず
    /// エラーにする。描画はまず minijinja (`DennoTokenizer` と同じ設定) で試し、
    /// 失敗したときだけテンプレート文字列から取り出した固定断片で組み立てる。
    ///
    /// `enable_thinking` は false を渡す (1 ターンの素の user → assistant 形状を
    /// 得るため。テンプレートが思考の指示文を system ブロックとして自動挿入するのを
    /// 避ける)。テンプレートがこの変数を持たなければ無視される。
    pub fn encode_chat(&self, system: Option<&str>, user: &str) -> Result<Vec<u32>> {
        let text = self.render_chat(system, user)?;
        self.encode(&text)
    }

    /// chat_template を描画してプロンプト文字列にする。
    fn render_chat(&self, system: Option<&str>, user: &str) -> Result<String> {
        let Some(tpl) = &self.chat_template else {
            bail!("tokenizer.chat_template が無いのでチャットを組めない");
        };
        if let Some(s) = self.render_with_minijinja(tpl, system, user) {
            return Ok(s);
        }
        let frags = ChatFragments::extract(tpl)?;
        Ok(frags.render(system, user))
    }

    /// minijinja による描画。失敗/空出力なら None (呼び出し側が断片組み立てへ落ちる)。
    fn render_with_minijinja(&self, tpl: &str, system: Option<&str>, user: &str) -> Option<String> {
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        if let Some(sys) = system {
            if !sys.is_empty() {
                msgs.push(serde_json::json!({"role": "system", "content": sys}));
            }
        }
        msgs.push(serde_json::json!({"role": "user", "content": user}));

        let token_str = |id: Option<u32>| -> serde_json::Value {
            id.and_then(|i| self.tok.id_to_token(i))
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null)
        };
        let ctx = serde_json::json!({
            "messages": msgs,
            "add_generation_prompt": true,
            "bos_token": token_str(self.bos),
            "eos_token": token_str(self.eos),
            "pad_token": serde_json::Value::Null,
            "tools": serde_json::Value::Null,
            "enable_thinking": false,
        });

        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        match env.render_str(tpl, ctx) {
            Ok(s) if !s.is_empty() => Some(s),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!("chat_template の描画に失敗、固定断片で組み立てる: {e}");
                None
            }
        }
    }

    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// GGUF の chat_template (生の Jinja 文字列)。
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// 語彙数 (added tokens 込み)。
    pub fn vocab_size(&self) -> usize {
        self.tok.get_vocab_size(true)
    }
}

fn meta_str(f: &GgufFile, key: &str) -> Result<String> {
    let v = f
        .get(key)
        .ok_or_else(|| anyhow!("メタデータ {key} が無い"))?;
    v.as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("メタデータ {key} が文字列でない"))
}

fn meta_str_array(f: &GgufFile, key: &str) -> Result<Vec<String>> {
    let v = f
        .get(key)
        .ok_or_else(|| anyhow!("メタデータ {key} が無い"))?;
    let GgufValue::Arr(items) = v else {
        return Err(anyhow!("メタデータ {key} が配列でない"));
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        out.push(
            it.as_str()
                .ok_or_else(|| anyhow!("{key}[{i}] が文字列でない"))?
                .to_string(),
        );
    }
    Ok(out)
}

fn meta_token_id(f: &GgufFile, key: &str, n_vocab: usize) -> Result<Option<u32>> {
    let Some(v) = f.get(key) else { return Ok(None) };
    let Some(raw) = v.as_u64() else {
        bail!("メタデータ {key} が整数でない");
    };
    if raw as usize >= n_vocab {
        bail!("メタデータ {key} = {raw} が語彙数 {n_vocab} の外");
    }
    Ok(Some(raw as u32))
}

/// chat_template から取り出した ChatML の固定断片。
///
/// テンプレート文字列をハードコードせず、GGUF から読んだ Jinja 文字列中の
/// 文字列リテラルから「開始マーカー / ロール名との区切り / 終了ブロック」を拾う。
struct ChatFragments {
    /// 例: `<|im_start|>`
    start: String,
    /// ロール名の後ろに付く区切り。例: `"\n"`
    sep: String,
    /// 1 ブロックを閉じる断片 (末尾の区切り込み)。例: `"<|im_end|>\n"`
    end_block: String,
}

impl ChatFragments {
    fn extract(tpl: &str) -> Result<Self> {
        let lits = jinja_string_literals(tpl);

        // 1) 「開始マーカー + ロール名 + 区切り」の形のリテラルから start と sep を取る。
        let mut start_sep: Option<(String, String)> = None;
        'outer: for lit in &lits {
            for role in ["system", "assistant", "user"] {
                let Some(i) = lit.find(role) else { continue };
                let head = &lit[..i];
                let Some(m) = special_marker(head) else {
                    continue;
                };
                if head != m {
                    continue; // マーカーの前に別の文字が付いている断片は使わない
                }
                start_sep = Some((m.to_string(), lit[i + role.len()..].to_string()));
                break 'outer;
            }
        }
        let (start, sep) = start_sep.ok_or_else(|| {
            anyhow!("chat_template から「開始マーカー + ロール名 + 区切り」を取れない")
        })?;

        // 2) 終了マーカー: start 以外で最も多く現れるマーカー。
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for lit in &lits {
            let mut rest = lit.as_str();
            while let Some(m) = special_marker(rest) {
                if m != start {
                    *counts.entry(m.to_string()).or_default() += 1;
                }
                rest = &rest[m.len()..];
                if rest.is_empty() {
                    break;
                }
            }
        }
        let end_marker = counts
            .iter()
            .max_by_key(|(_, &n)| n)
            .map(|(m, _)| m.clone())
            .ok_or_else(|| anyhow!("chat_template から終了マーカーを取れない"))?;

        // 3) 終了ブロック: 終了マーカーで始まり、区切りまで含むリテラルを優先する。
        let mut end_block = end_marker.clone();
        for lit in &lits {
            if special_marker(lit) == Some(end_marker.as_str()) && lit.len() > end_marker.len() {
                end_block = lit.clone();
                if lit[end_marker.len()..] == sep {
                    break;
                }
            }
        }

        Ok(Self {
            start,
            sep,
            end_block,
        })
    }

    fn render(&self, system: Option<&str>, user: &str) -> String {
        let mut s = String::new();
        if let Some(sys) = system {
            if !sys.is_empty() {
                s.push_str(&self.start);
                s.push_str("system");
                s.push_str(&self.sep);
                s.push_str(sys);
                s.push_str(&self.end_block);
            }
        }
        s.push_str(&self.start);
        s.push_str("user");
        s.push_str(&self.sep);
        s.push_str(user);
        s.push_str(&self.end_block);
        s.push_str(&self.start);
        s.push_str("assistant");
        s.push_str(&self.sep);
        s
    }
}

/// 文字列の先頭が `<|...|>` 形式のマーカーならその部分を返す。
fn special_marker(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("<|")?;
    let end = rest.find("|>")?;
    Some(&s[..2 + end + 2])
}

/// Jinja テンプレート中の文字列リテラルを順に取り出す (エスケープを解釈する)。
/// `{# ... #}` のコメントは飛ばす。テンプレートの意味解析はしない — 断片拾い専用。
fn jinja_string_literals(tpl: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = tpl.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if c == '{' && tpl[i..].starts_with("{#") {
            // コメント終端まで読み飛ばす
            if let Some(rel) = tpl[i + 2..].find("#}") {
                let target = i + 2 + rel + 2;
                while let Some(&(k, _)) = it.peek() {
                    if k < target {
                        it.next();
                    } else {
                        break;
                    }
                }
            }
            continue;
        }
        if c != '\'' && c != '"' {
            continue;
        }
        let quote = c;
        let mut s = String::new();
        while let Some((_, c2)) = it.next() {
            if c2 == '\\' {
                match it.next() {
                    Some((_, 'n')) => s.push('\n'),
                    Some((_, 't')) => s.push('\t'),
                    Some((_, 'r')) => s.push('\r'),
                    Some((_, '\\')) => s.push('\\'),
                    Some((_, '\'')) => s.push('\''),
                    Some((_, '"')) => s.push('"'),
                    Some((_, other)) => {
                        s.push('\\');
                        s.push(other);
                    }
                    None => break,
                }
            } else if c2 == quote {
                break;
            } else {
                s.push(c2);
            }
        }
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_extracted_with_escapes() {
        let tpl = "{# 'ignored' #}{{- '<|im_start|>system\\n' + 'x' }}{{ \"a\\tb\" }}";
        let lits = jinja_string_literals(tpl);
        assert_eq!(
            lits,
            vec![
                "<|im_start|>system\n".to_string(),
                "x".to_string(),
                "a\tb".to_string()
            ]
        );
    }

    #[test]
    fn fragments_are_taken_from_a_qwen_style_template() {
        // 実テンプレートと同じ構造の縮小版 (namespace / 条件分岐 / 文字列連結を含む)
        let tpl = concat!(
            "{%- set ns = namespace(n=0) %}",
            "{%- if tools %}{{- '<|im_start|>system\\n' }}{{- '<|im_end|>\\n' }}{%- endif %}",
            "{%- for m in messages %}",
            "{%- if m.role == \"user\" %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>' + '\\n' }}",
            "{%- endif %}{%- endfor %}",
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
        );
        let f = ChatFragments::extract(tpl).unwrap();
        assert_eq!(f.start, "<|im_start|>");
        assert_eq!(f.sep, "\n");
        assert_eq!(f.end_block, "<|im_end|>\n");
        assert_eq!(
            f.render(Some("S"), "hi"),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            f.render(None, "hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn fragments_without_role_literal_are_an_error() {
        let tpl = "{{ '<|im_start|>' + 'x' }}";
        assert!(ChatFragments::extract(tpl).is_err());
    }

    #[test]
    fn special_marker_detection() {
        assert_eq!(special_marker("<|im_end|>\n"), Some("<|im_end|>"));
        assert_eq!(special_marker("x<|im_end|>"), None);
        assert_eq!(special_marker("plain"), None);
    }
}
