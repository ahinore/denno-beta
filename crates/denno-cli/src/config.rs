//! task 906: `denno serve-flash` の設定ファイル (TOML サブセット) と解決順。
//!
//! 解決順は **CLI 引数 > `DENNO_*` 環境変数 > 設定ファイル > 既定値**。
//! このモジュールは値を「文字列 + 出所」の表として解決し、型と範囲の検査は
//! 取り出すとき (`Resolved::u16` など) に行う。ファイルの解析時には
//! 未知キー・型違い・範囲外を**行番号付き**で弾く。
//!
//! 設定ファイルは TOML のごく一部だけを受ける:
//! - `key = value` を 1 行 1 個 (セクション `[table]` は非対応)
//! - 値は `"..."` (エスケープ可) / `'...'` (リテラル) / 10 進整数
//! - `#` から行末はコメント
//!
//! 依存を増やさないため serde / toml は使わない (Cargo.lock にも無い)。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 設定として受け付けるキー (この順で `--print-config` に出る)。
pub const KEYS: [&str; 10] = [
    "dir",
    "host",
    "port",
    "max_ctx",
    "name",
    "max_tokens",
    "mtp_path",
    "kv_slots",
    "steer_recipe",
    "vram_budget",
];

/// キーの型と範囲。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Str,
    /// 10 進整数 (min..=max)。
    Int { min: u64, max: u64 },
}

/// キーごとの定義。(キー, 型, 環境変数, 既定値)
const SPEC: [(&str, Kind, Option<&str>, Option<&str>); 10] = [
    ("dir", Kind::Str, Some("DENNO_FLASH_DIR"), None),
    ("host", Kind::Str, Some("DENNO_FLASH_HOST"), Some("127.0.0.1")),
    (
        "port",
        // 0 は OS 任せのポート (従来の `--port 0` を壊さない)
        Kind::Int { min: 0, max: 65535 },
        Some("DENNO_FLASH_PORT"),
        Some("8098"),
    ),
    (
        "max_ctx",
        Kind::Int { min: 1, max: u32::MAX as u64 },
        Some("DENNO_FLASH_MAX_CTX"),
        Some("4096"),
    ),
    (
        "name",
        Kind::Str,
        Some("DENNO_FLASH_NAME"),
        Some("qwen3.8-flash-next"),
    ),
    (
        "max_tokens",
        Kind::Int { min: 1, max: u32::MAX as u64 },
        Some("DENNO_FLASH_MAX_TOKENS"),
        Some("512"),
    ),
    ("mtp_path", Kind::Str, Some("DENNO_MTP_PATH"), None),
    (
        "kv_slots",
        // serve_flash::KV_SLOTS_MAX と同じ上限 (ここは依存を持たないため数値で持つ)。
        // 0 は従来の `DENNO_KV_SLOTS=0` と同じ「未指定 (= 既定 2)」の意味。
        Kind::Int { min: 0, max: 8 },
        Some("DENNO_KV_SLOTS"),
        None,
    ),
    ("steer_recipe", Kind::Str, Some("DENNO_STEER_RECIPE"), None),
    ("vram_budget", Kind::Str, Some("DENNO_AUTO_CONFIG"), None),
];

fn spec(key: &str) -> Option<&'static (&'static str, Kind, Option<&'static str>, Option<&'static str>)> {
    SPEC.iter().find(|(k, ..)| *k == key)
}

/// キーに対応する環境変数名。
pub fn env_var(key: &str) -> Option<&'static str> {
    spec(key).and_then(|s| s.2)
}

/// キーの既定値 (無ければ `None`)。
pub fn default_value(key: &str) -> Option<&'static str> {
    spec(key).and_then(|s| s.3)
}

fn kind_of(key: &str) -> Option<Kind> {
    spec(key).map(|s| s.1)
}

fn valid_keys_hint() -> String {
    KEYS.join(", ")
}

/// 設定ファイルの 1 値。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    Int(i64),
}

impl Value {
    fn as_text(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(n) => n.to_string(),
        }
    }
}

/// 解析済みの設定ファイル。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileConfig {
    /// エラーメッセージと出所表示に使うパス
    pub origin: String,
    pub entries: BTreeMap<String, Value>,
}

impl FileConfig {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }
}

/// パスから読む。読めない / 解析できないときは人間向けの 1 行エラー。
pub fn load(path: &Path) -> Result<FileConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("config {}: 読めない: {e}", path.display()))?;
    parse_str(&text, &path.display().to_string())
}

/// 文字列から解析する。`origin` はエラー表示に使う (通常はパス)。
pub fn parse_str(text: &str, origin: &str) -> Result<FileConfig, String> {
    let mut entries: BTreeMap<String, Value> = BTreeMap::new();
    let mut seen_lines: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, raw) in text.lines().enumerate() {
        let n = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            return Err(format!(
                "{origin}:{n}: セクション ([...]) は未対応。`key = value` を並べる"
            ));
        }
        let eq = line.find('=').ok_or_else(|| {
            format!("{origin}:{n}: `key = value` の形になっていない: `{line}`")
        })?;
        let key = line[..eq].trim();
        let rest = line[eq + 1..].trim();
        if key.is_empty() {
            return Err(format!("{origin}:{n}: キーが空"));
        }
        if !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "{origin}:{n}: キー `{key}` に使える文字は英数字 / `_` / `-` だけ"
            ));
        }
        let value = parse_value(rest, origin, n)?;

        let kind = kind_of(key)
            .ok_or_else(|| unknown_key_err(origin, n, key))?;
        let value = coerce(key, kind, value, origin, n)?;
        if let Some(prev) = seen_lines.get(key) {
            return Err(format!(
                "{origin}:{n}: キー `{key}` が重複 ({} 行目で既に定義)",
                prev
            ));
        }
        seen_lines.insert(key.to_string(), n);
        entries.insert(key.to_string(), value);
    }

    Ok(FileConfig {
        origin: origin.to_string(),
        entries,
    })
}

fn unknown_key_err(origin: &str, n: usize, key: &str) -> String {
    let mut best: Vec<&str> = KEYS
        .iter()
        .copied()
        .filter(|k| k.starts_with(key) || key.starts_with(k) || near(k, key))
        .collect();
    best.sort_unstable();
    let hint = if best.is_empty() {
        String::new()
    } else {
        format!("。もしかして: {}", best.join(", "))
    };
    format!(
        "{origin}:{n}: 未知のキー `{key}`{hint}。有効なキー: {}",
        valid_keys_hint()
    )
}

/// 1 文字違い / 部分一致の雑な類似判定 (誤字のヒント用)。
fn near(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len().abs_diff(b.len()) > 2 {
        return false;
    }
    let mut diff = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            diff += 1;
            if diff > 2 {
                return false;
            }
        }
    }
    diff <= 1
}

fn parse_value(rest: &str, origin: &str, n: usize) -> Result<Value, String> {
    if rest.is_empty() {
        return Err(format!("{origin}:{n}: 値が無い"));
    }
    let bytes = rest.as_bytes();
    if bytes[0] == b'"' {
        let (s, used) = parse_basic_string(rest, origin, n)?;
        ensure_trailing(&rest[used..], origin, n)?;
        return Ok(Value::Str(s));
    }
    if bytes[0] == b'\'' {
        let close = rest[1..].find('\'').ok_or_else(|| {
            format!("{origin}:{n}: 文字列の閉じ `'` が無い")
        })?;
        let s = rest[1..1 + close].to_string();
        ensure_trailing(&rest[1 + close + 1..], origin, n)?;
        return Ok(Value::Str(s));
    }
    // コメントを落として整数を試す
    let body = match rest.find('#') {
        Some(i) => rest[..i].trim(),
        None => rest,
    };
    let digits = body.trim_start_matches(['+', '-']);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        match body.parse::<i64>() {
            Ok(v) => Ok(Value::Int(v)),
            Err(_) => Err(format!("{origin}:{n}: 整数として読めない: `{body}`")),
        }
    } else {
        Err(format!(
            "{origin}:{n}: 値 `{body}` は未対応。文字列は `\"...\"`、数値は 10 進整数で書く"
        ))
    }
}

/// `"..."` を読む。返り値は (中身, 消費したバイト数)。
fn parse_basic_string(s: &str, origin: &str, n: usize) -> Result<(String, usize), String> {
    debug_assert!(s.starts_with('"'));
    let mut out = String::new();
    let mut chars = s.char_indices().skip(1);
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Ok((out, i + 1)),
            '\\' => {
                let (_, e) = chars.next().ok_or_else(|| {
                    format!("{origin}:{n}: エスケープが行末で切れている")
                })?;
                match e {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    other => {
                        return Err(format!(
                            "{origin}:{n}: 未対応のエスケープ `\\{other}` (\\n \\t \\r \\\\ \\\" のみ)"
                        ))
                    }
                }
            }
            c => out.push(c),
        }
    }
    Err(format!("{origin}:{n}: 文字列の閉じ `\"` が無い"))
}

fn ensure_trailing(rest: &str, origin: &str, n: usize) -> Result<(), String> {
    let t = rest.trim();
    if t.is_empty() || t.starts_with('#') {
        Ok(())
    } else {
        Err(format!("{origin}:{n}: 値の後ろに余分な文字 `{t}`"))
    }
}

fn coerce(key: &str, kind: Kind, v: Value, origin: &str, n: usize) -> Result<Value, String> {
    match (kind, v) {
        (Kind::Str, Value::Str(s)) => Ok(Value::Str(s)),
        // vram_budget は `auto` か MiB の整数の両方を受ける
        (Kind::Str, Value::Int(i)) => Ok(Value::Str(i.to_string())),
        (Kind::Int { min, max }, Value::Int(i)) => {
            let u = u64::try_from(i).map_err(|_| {
                format!("{origin}:{n}: `{key}` は 0 以上 {max} 以下 (指定値 {i})")
            })?;
            if u < min || u > max {
                return Err(format!(
                    "{origin}:{n}: `{key}` の範囲は {min}..={max} (指定値 {i})"
                ));
            }
            Ok(Value::Int(i))
        }
        (Kind::Int { min, max }, Value::Str(s)) => Err(format!(
            "{origin}:{n}: `{key}` は整数 {min}..={max} で書く (指定値 `{s}`)"
        )),
    }
}

/// 値の出所。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Cli,
    Env,
    Config,
    Default,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Cli => "cli",
            Source::Env => "env",
            Source::Config => "config",
            Source::Default => "default",
        }
    }
}

/// 解決済みの 1 値。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub value: String,
    pub source: Source,
    /// env 名 / 設定ファイルのパス (cli / default では空)
    pub detail: String,
}

impl Slot {
    pub fn origin_label(&self) -> String {
        if self.detail.is_empty() {
            self.source.as_str().to_string()
        } else {
            format!("{} {}", self.source.as_str(), self.detail)
        }
    }
}

/// 全キーの解決結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    slots: BTreeMap<&'static str, Slot>,
}

impl Resolved {
    pub fn slot(&self, key: &str) -> Option<&Slot> {
        self.slots.get(key)
    }

    /// 出所だけを見る (bin では未使用、テストと将来の診断用)。
    #[allow(dead_code)]
    pub fn source(&self, key: &str) -> Option<Source> {
        self.slot(key).map(|s| s.source)
    }

    /// 生の文字列 (未設定なら `None`)。
    pub fn string(&self, key: &str) -> Option<String> {
        self.slot(key).map(|s| s.value.clone())
    }

    pub fn path(&self, key: &str) -> Option<PathBuf> {
        self.string(key).map(PathBuf::from)
    }

    /// 非負整数。型・範囲の検査はここで行う (env / CLI 由来の値のため)。
    pub fn number(&self, key: &str) -> Result<Option<u64>, String> {
        let Some(slot) = self.slot(key) else {
            return Ok(None);
        };
        let Kind::Int { min, max } = kind_of(key).unwrap_or(Kind::Int { min: 1, max: u64::MAX })
        else {
            return Err(format!("`{key}` は整数ではない"));
        };
        let raw = slot.value.trim();
        let n = raw.parse::<u64>().map_err(|_| {
            format!(
                "`{key}`: 整数 {min}..={max} を期待したが `{raw}` (出所: {})",
                slot.origin_label()
            )
        })?;
        if n < min || n > max {
            return Err(format!(
                "`{key}`: 範囲は {min}..={max} (指定値 {n}、出所: {})",
                slot.origin_label()
            ));
        }
        Ok(Some(n))
    }

    pub fn u16(&self, key: &str) -> Result<Option<u16>, String> {
        match self.number(key)? {
            Some(n) => Ok(Some(n as u16)),
            None => Ok(None),
        }
    }

    pub fn usize(&self, key: &str) -> Result<Option<usize>, String> {
        match self.number(key)? {
            Some(n) => Ok(Some(n as usize)),
            None => Ok(None),
        }
    }

    /// `--print-config` の本文。設定ファイルにそのまま貼れる TOML 風。
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("# denno serve-flash resolved config (cli > env > config > default)\n");
        for key in KEYS {
            match self.slot(key) {
                Some(slot) => {
                    let text = match kind_of(key) {
                        Some(Kind::Int { .. }) => slot.value.clone(),
                        _ => quote(&slot.value),
                    };
                    // kv_slots = 0 は実行時「未指定 (= 既定 2)」に正規化される。
                    // print-config ではその旨を併記する (値と実効値の食い違い防止)。
                    let note = if key == "kv_slots" && slot.value.trim() == "0" {
                        " (0 = 未指定 -> 既定 2)"
                    } else {
                        ""
                    };
                    out.push_str(&format!(
                        "{key} = {text}  # source: {}{note}\n",
                        slot.origin_label()
                    ));
                }
                None => out.push_str(&format!("# {key} = <unset>\n")),
            }
        }
        out
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 解決する。`cli` は**明示された**引数だけを持つ (clap の既定値は入れない)。
/// `env` はプロセスの環境変数全体でよい (既知の `DENNO_*` だけ見る)。
pub fn resolve(
    cli: &[(&'static str, String)],
    env: &BTreeMap<String, String>,
    file: Option<&FileConfig>,
) -> Result<Resolved, String> {
    let mut slots: BTreeMap<&'static str, Slot> = BTreeMap::new();
    for key in KEYS {
        let key: &'static str = key;
        if let Some((k, v)) = cli.iter().find(|(k, _)| *k == key) {
            let _ = k;
            slots.insert(
                key,
                Slot {
                    value: v.clone(),
                    source: Source::Cli,
                    detail: String::new(),
                },
            );
            continue;
        }
        if let Some(env_name) = env_var(key) {
            if let Some(v) = env.get(env_name) {
                // 空 / 空白だけの env は未設定とみなす (export DENNO_FLASH_HOST= の事故)
                if !v.trim().is_empty() {
                    slots.insert(
                        key,
                        Slot {
                            value: v.clone(),
                            source: Source::Env,
                            detail: env_name.to_string(),
                        },
                    );
                    continue;
                }
            }
        }
        if let Some(f) = file {
            if let Some(v) = f.get(key) {
                slots.insert(
                    key,
                    Slot {
                        value: v.as_text(),
                        source: Source::Config,
                        detail: f.origin.clone(),
                    },
                );
                continue;
            }
        }
        if let Some(d) = default_value(key) {
            slots.insert(
                key,
                Slot {
                    value: d.to_string(),
                    source: Source::Default,
                    detail: String::new(),
                },
            );
        }
    }
    let resolved = Resolved { slots };
    // 早期に型・範囲を検査する (print-config でも同じ検査を通す)
    for key in KEYS {
        if kind_of(key).map(|k| matches!(k, Kind::Int { .. })).unwrap_or(false) {
            resolved.number(key)?;
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parses_flat_toml_subset() {
        let f = parse_str(
            "# comment\n\ndir = \"/models/flash\"\nport = 9001\nname = 'lite'\nmax_ctx = 2048 # tail\n",
            "t.toml",
        )
        .unwrap();
        assert_eq!(f.get("dir"), Some(&Value::Str("/models/flash".into())));
        assert_eq!(f.get("port"), Some(&Value::Int(9001)));
        assert_eq!(f.get("name"), Some(&Value::Str("lite".into())));
        assert_eq!(f.get("max_ctx"), Some(&Value::Int(2048)));
    }

    #[test]
    fn parses_escapes() {
        let f = parse_str("name = \"a\\\"b\\\\c\\td\"\n", "t.toml").unwrap();
        assert_eq!(f.get("name"), Some(&Value::Str("a\"b\\c\td".into())));
    }

    #[test]
    fn rejects_unknown_key_with_hint() {
        let e = parse_str("prot = 1\n", "t.toml").unwrap_err();
        assert!(e.contains("未知のキー `prot`"), "{e}");
        assert!(e.contains("port"), "{e}");
        assert!(e.contains("t.toml:1"), "{e}");
    }

    #[test]
    fn rejects_wrong_type() {
        let e = parse_str("port = \"abc\"\n", "t.toml").unwrap_err();
        assert!(e.contains("`port` は整数"), "{e}");
        let e = parse_str("max_ctx = 2048.5\n", "t.toml").unwrap_err();
        assert!(e.contains("未対応"), "{e}");
    }

    #[test]
    fn rejects_out_of_range() {
        let e = parse_str("port = 70000\n", "t.toml").unwrap_err();
        assert!(e.contains("0..=65535"), "{e}");
        let e = parse_str("kv_slots = 9\n", "t.toml").unwrap_err();
        assert!(e.contains("0..=8"), "{e}");
    }

    #[test]
    fn rejects_duplicates_and_sections() {
        assert!(parse_str("port = 1\nport = 2\n", "t.toml")
            .unwrap_err()
            .contains("重複"));
        assert!(parse_str("[serve]\nport = 1\n", "t.toml")
            .unwrap_err()
            .contains("セクション"));
    }

    #[test]
    fn precedence_cli_env_file_default() {
        let f = parse_str("port = 9001\nhost = \"cfg-host\"\nname = \"cfg-name\"\n", "c.toml")
            .unwrap();
        let env = env_of(&[("DENNO_FLASH_PORT", "9002"), ("DENNO_FLASH_HOST", "env-host")]);
        let cli = vec![("port", "9003".to_string())];
        let r = resolve(&cli, &env, Some(&f)).unwrap();
        assert_eq!(r.string("port").as_deref(), Some("9003"));
        assert_eq!(r.source("port"), Some(Source::Cli));
        assert_eq!(r.string("host").as_deref(), Some("env-host"));
        assert_eq!(r.source("host"), Some(Source::Env));
        assert_eq!(r.string("name").as_deref(), Some("cfg-name"));
        assert_eq!(r.source("name"), Some(Source::Config));
        assert_eq!(r.string("max_ctx").as_deref(), Some("4096"));
        assert_eq!(r.source("max_ctx"), Some(Source::Default));
        assert_eq!(r.path("dir"), None);
        assert_eq!(r.source("dir"), None);
    }

    #[test]
    fn empty_env_is_unset() {
        let env = env_of(&[("DENNO_FLASH_HOST", "  ")]);
        let r = resolve(&[], &env, None).unwrap();
        assert_eq!(r.string("host").as_deref(), Some("127.0.0.1"));
        assert_eq!(r.source("host"), Some(Source::Default));
    }

    #[test]
    fn bad_env_number_is_actionable() {
        let env = env_of(&[("DENNO_FLASH_PORT", "http")]);
        let e = resolve(&[], &env, None).unwrap_err();
        assert!(e.contains("DENNO_FLASH_PORT"), "{e}");
        // 範囲外も出所付き
        let env = env_of(&[("DENNO_FLASH_PORT", "70000")]);
        let e = resolve(&[], &env, None).unwrap_err();
        assert!(e.contains("0..=65535"), "{e}");
    }

    #[test]
    fn render_marks_source_and_unset() {
        let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
        let r = resolve(&[], &env, None).unwrap();
        let s = r.render();
        assert!(s.contains("port = 9002  # source: env DENNO_FLASH_PORT"), "{s}");
        assert!(s.contains("host = \"127.0.0.1\"  # source: default"), "{s}");
        assert!(s.contains("# dir = <unset>"), "{s}");
    }

    #[test]
    fn every_key_has_a_spec() {
        for k in KEYS {
            assert!(spec(k).is_some(), "no spec for {k}");
        }
        assert_eq!(SPEC.len(), KEYS.len());
    }
}
