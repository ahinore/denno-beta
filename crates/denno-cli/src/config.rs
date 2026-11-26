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

    /// task 1001: 構造化サマリ。`--print-config --json` で出す形。
    ///
    /// キーは `KEYS` の順で 1 個ずつ、未設定なら `{"set": false, "source": "unset"}`。
    /// 設定済みなら `set` / `value` (整数キーは JSON 数値) / `source` / `detail` /
    /// `origin` / `spec` を持つ。`origin` は `render()` の `# source:` と同じ表記。
    #[allow(dead_code)]
    pub fn summary_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        for key in KEYS {
            let entry = match self.slot(key) {
                Some(slot) => {
                    let kind = kind_of(key).unwrap_or(Kind::Str);
                    let value = match kind {
                        Kind::Int { .. } => match slot.value.trim().parse::<u64>() {
                            Ok(n) => serde_json::Value::from(n),
                            Err(_) => serde_json::Value::from(slot.value.clone()),
                        },
                        Kind::Str => serde_json::Value::from(slot.value.clone()),
                    };
                    let spec = match kind {
                        Kind::Int { min, max } => {
                            serde_json::json!({"kind": "int", "min": min, "max": max})
                        }
                        Kind::Str => serde_json::json!({"kind": "str"}),
                    };
                    serde_json::json!({
                        "set": true,
                        "value": value,
                        "source": slot.source.as_str(),
                        "detail": slot.detail,
                        "origin": slot.origin_label(),
                        "spec": spec,
                    })
                }
                None => serde_json::json!({"set": false, "source": "unset"}),
            };
            obj.insert(key.to_string(), entry);
        }
        serde_json::Value::Object(obj)
    }

    /// `summary_json` を整形した本文 (テストの往復用)。
    #[allow(dead_code)]
    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(&self.summary_json()).unwrap_or_else(|_| "{}".to_string())
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

#[allow(dead_code)]
pub mod diag {
    use super::*;

    // ---------------------------------------------------------------- 環境変数の診断
    //
    // task 1066: `DENNO_*` の「矛盾」と「読まれない名前」を、値を漏らさずに列挙する
    // 純関数。`serve-flash` の起動前チェックとテストから同じ関数を呼べるよう、
    // プロセスの env もファイルも触らない (`env` は呼び手が渡す)。
    //
    // 検出するもの:
    // 1. 未知の `DENNO_*` (綴り違いは近い名前を提案する)
    // 2. 大文字小文字の取り違え / `DENNO_` 接頭辞の欠け (どちらも読まれない)
    // 3. 非推奨の名前 (`DEPRECATED_ENV`) と、その後継との値の食い違い
    // 4. 空値 (解決では未設定扱い) / 整数キーの型・範囲外
    // 5. `DENNO_KV_SLOTS=0` (未指定の旧来の書き方)、`DENNO_AUTO_CONFIG` の真偽値以外
    // 6. 出所の食い違い (CLI が env を、env が設定ファイルを黙って上書きする)
    // 7. 実効値どうしの矛盾 (max_tokens > max_ctx)
    //
    // **値は秘密名 (`is_secret_name`) なら出さない**。名前だけを出す。

    /// 診断の重大度。`Warn` は起動を止めない、`Error` は解決が失敗する。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Level {
        Warn,
        Error,
    }

    impl Level {
        pub fn as_str(self) -> &'static str {
            match self {
                Level::Warn => "warn",
                Level::Error => "error",
            }
        }
    }

    /// 診断 1 件。`code` は安定した識別子 (`env_unknown` など)、`message` は表示用。
    /// `vars` は関係する環境変数名だけで、**値は入れない**。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Diag {
        pub level: Level,
        pub code: &'static str,
        pub vars: Vec<String>,
        pub message: String,
    }

    impl Diag {
        fn new(level: Level, code: &'static str, vars: Vec<String>, message: String) -> Self {
            Diag {
                level,
                code,
                vars,
                message,
            }
        }

        /// `[warn] DENNO_X: ...` の 1 行 (stderr にそのまま出せる形)。
        pub fn line(&self) -> String {
            format!(
                "[{}] {}: {}",
                self.level.as_str(),
                self.vars.join(", "),
                self.message
            )
        }
    }

    /// serve-flash の設定として解決に使う env 名 (`SPEC` から作る)。
    pub fn known_env_vars() -> Vec<&'static str> {
        let mut v: Vec<&'static str> = SPEC.iter().filter_map(|s| s.2).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// 非推奨の env 名 → 後継の env 名。
    ///
    /// `DENNO_MAX_CTX` はこの crate のどこでも読まれない (config 解決は
    /// `DENNO_FLASH_MAX_CTX` だけを見る)。汎用名で設定したつもりで既定値のまま
    /// 動く事故があるので、後継を案内する。
    pub const DEPRECATED_ENV: [(&str, &str); 1] = [("DENNO_MAX_CTX", "DENNO_FLASH_MAX_CTX")];

    /// 値が秘密ではない「数値リミット」の名前。`*TOKEN*` の規約では伏せられるが、
    /// 中身は公開してよい上限値なので例外にする。
    ///
    /// **完全一致だけ**を例外にする (接尾辞・部分一致・大小の揺れは認めない)。
    /// 例外の綴りを少し変えた名前 (`DENNO_FLASH_MAX_TOKENS_X` など) は
    /// 従来どおり伏せる — fail-closed。
    pub const NON_SECRET_VALUE_NAMES: [&str; 2] = ["DENNO_FLASH_MAX_TOKENS", "DENNO_MAX_TOKENS"];

    /// 秘密っぽい名前か。**値は見ない** (名前だけで判定する)。
    /// doctor.rs の `is_secret_key` と同じ規約 (伏せすぎる側に倒す)。
    /// `NON_SECRET_VALUE_NAMES` の完全一致だけが例外。
    pub fn is_secret_name(name: &str) -> bool {
        if NON_SECRET_VALUE_NAMES.contains(&name) {
            return false;
        }
        let k = name.to_ascii_uppercase();
        k.contains("API_KEY")
            || k.contains("SECRET")
            || k.contains("TOKEN")
            || k.contains("PASSWORD")
            || k.contains("AUTH")
            || k.ends_with("_KEY")
            || k.ends_with("_KEYS")
    }

    /// メッセージに載せてよい値。秘密名なら常に伏せる。
    fn show_value(name: &str, value: &str) -> String {
        if is_secret_name(name) {
            "<redacted>".to_string()
        } else {
            format!("`{value}`")
        }
    }

    /// 編集距離 (バイト単位)。短い ASCII 名だけを想定する。
    fn edit_distance(a: &str, b: &str) -> usize {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for i in 1..=a.len() {
            cur[0] = i;
            for j in 1..=b.len() {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    /// `want` に近い既知の env 名を距離順で返す (誤字のヒント用)。
    fn suggest_env(want: &str) -> Vec<&'static str> {
        let mut cands: Vec<(&'static str, usize)> = known_env_vars()
            .into_iter()
            .chain(DEPRECATED_ENV.iter().map(|(old, _)| *old))
            .map(|name| (name, edit_distance(want, name)))
            .filter(|(name, d)| *name != want && *d <= 2 && *d * 3 <= name.len())
            .collect();
        cands.sort_unstable();
        cands.into_iter().map(|(n, _)| n).collect()
    }

    /// serve-flash 自身が読むが `SPEC` には無い env 名 (解決には使わない)。
    ///
    /// `serve-flash --strict-env` を env でも立てられるようにするためのゲート。
    /// `SPEC` に足すと `--print-config` と解決順の表に出てしまうので、ここに置く。
    pub const KNOWN_NON_CONFIG_ENV: [&str; 1] = ["DENNO_FLASH_STRICT_ENV"];

    /// serve-flash 起動時の「未知の `DENNO_*`」報告から外す名前か。
    ///
    /// serve-flash の設定として解決に使うのは `SPEC` の 10 本だけなので、それ以外の
    /// `DENNO_*` はすべてこの起動では読まれない。ただし denno-engine / denno-server
    /// が読むチューニング名は 600 本以上あり (`DENNO_PF_*` / `DENNO_EXPERT_*` など)、
    /// ラボの env を毎回「未知」と言うと警告で埋まる。そこで既定では次だけを黙る:
    ///
    /// - エンジン側の名前空間 (`DENNO_FLASH_` 以外) で、設定名に近くないもの
    /// - 既知の非設定名 (`KNOWN_NON_CONFIG_ENV`)
    ///
    /// `DENNO_FLASH_` 空間と、設定名に近い名前 (誤字の疑い) は残す。
    /// 値の検査は元々「既知の名前」にしか無いので、黙っても検査は落ちない。
    pub fn is_quiet_unknown(name: &str) -> bool {
        if !name.starts_with("DENNO_") {
            return false;
        }
        if KNOWN_NON_CONFIG_ENV.contains(&name) {
            return true;
        }
        if name.starts_with("DENNO_FLASH_") {
            return false;
        }
        // 秘密っぽい名前は黙らない (読まれない API キーを設定している事故を
        // 見逃さない。値は出さず名前だけ出す)。
        if is_secret_name(name) {
            return false;
        }
        if known_env_vars().contains(&name) || DEPRECATED_ENV.iter().any(|(old, _)| *old == name) {
            return false;
        }
        // 設定名に近い = 誤字の疑いがあるので報告する
        suggest_env(name).is_empty()
    }

    /// `DENNO_AUTO_CONFIG` が真偽値として解釈できる綴りか。
    /// (`denno_engine::auto_config::auto_requested` が受けるのは真側の 5 つだけ。
    /// それ以外は黙って「無効」になるので、ここで列挙して警告する。)
    fn auto_config_spelling_ok(v: &str) -> bool {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes" | "auto" | "0" | "false" | "off" | "no"
        )
    }

    /// env だけを見る診断 (ファイルも CLI も見ない純関数)。
    pub fn diagnose_env(env: &BTreeMap<String, String>) -> Vec<Diag> {
        diagnose_env_with(env, false)
    }

    /// `diagnose_env` の scoped 版。`quiet_engine` が true なら
    /// `is_quiet_unknown` の名前を `env_unknown` として報告しない
    /// (serve-flash 起動用。ラボのエンジン env で警告を埋めない)。
    pub fn diagnose_env_with(env: &BTreeMap<String, String>, quiet_engine: bool) -> Vec<Diag> {
        let known = known_env_vars();
        let mut out: Vec<Diag> = Vec::new();

        for (name, value) in env {
            let upper = name.to_ascii_uppercase();
            let exact = known.iter().any(|k| *k == name.as_str())
                || KNOWN_NON_CONFIG_ENV.contains(&name.as_str());
            let deprecated = DEPRECATED_ENV
                .iter()
                .find(|(old, _)| *old == upper.as_str());

            if let Some((old, new)) = deprecated {
                // 大小の取り違えも「読まれない」ので併せて言う
                let spelling = if *name == upper {
                    String::new()
                } else {
                    format!(" (変数名は大文字小文字を区別する。`{old}` と書く)")
                };
                out.push(Diag::new(
                    Level::Warn,
                    "env_deprecated",
                    vec![name.clone()],
                    format!(
                        "非推奨の名前。serve-flash は読まない。`{new}` を使う{spelling}"
                    ),
                ));
                continue;
            }

            // 大文字に直すと既知 = 大小の取り違え (接頭辞の検査より先に見る)
            let upper_known = known.iter().any(|k| *k == upper)
                || KNOWN_NON_CONFIG_ENV.contains(&upper.as_str());
            if !exact && upper_known {
                out.push(Diag::new(
                    Level::Warn,
                    "env_case",
                    vec![name.clone()],
                    format!("変数名は大文字小文字を区別する。`{upper}` と書く (この名前は読まれない)"),
                ));
                continue;
            }

            // `DENNO_` で始まらない名前: 既知の名前と近いときだけ拾う
            // (無関係な env を毎回警告しないため)。完全一致は上で弾いてある。
            if !name.starts_with("DENNO_") {
                let best = known
                    .iter()
                    .copied()
                    .map(|k| (edit_distance(&upper, k), k))
                    .filter(|(d, _)| *d <= 2)
                    .min();
                if let Some((_, best)) = best {
                    out.push(Diag::new(
                        Level::Warn,
                        "env_unprefixed",
                        vec![name.clone()],
                        format!("`DENNO_` 接頭辞が無いので読まれない。もしかして: `{best}`"),
                    ));
                }
                continue;
            }

            if !exact {
                // エンジン側の env (設定名に近くない `DENNO_*`) は黙る
                if quiet_engine && is_quiet_unknown(name) {
                    continue;
                }
                let hint = match suggest_env(name) {
                    v if v.is_empty() => String::new(),
                    v => format!("。もしかして: {}", v.join(", ")),
                };
                out.push(Diag::new(
                    Level::Warn,
                    "env_unknown",
                    vec![name.clone()],
                    format!("未知の `DENNO_*` は無視される{hint}"),
                ));
                continue;
            }

            // ここから既知の名前
            if value.trim().is_empty() {
                out.push(Diag::new(
                    Level::Warn,
                    "env_empty",
                    vec![name.clone()],
                    "空値は未設定として扱う (既定値 / 設定ファイルが使われる)".to_string(),
                ));
                continue;
            }

            let key = known
                .iter()
                .find(|k| **k == name.as_str())
                .and_then(|k| SPEC.iter().find(|s| s.2 == Some(*k)))
                .map(|s| s.0)
                .unwrap_or("");

            match kind_of(key) {
                Some(Kind::Int { min, max }) => {
                    let raw = value.trim();
                    match raw.parse::<u64>() {
                        Ok(n) if n >= min && n <= max => {}
                        Ok(n) => out.push(Diag::new(
                            Level::Error,
                            "env_out_of_range",
                            vec![name.clone()],
                            format!("`{key}` の範囲は {min}..={max} (指定値 {n})"),
                        )),
                        Err(_) => out.push(Diag::new(
                            Level::Error,
                            "env_not_integer",
                            vec![name.clone()],
                            format!(
                                "`{key}` は 10 進整数 {min}..={max} で書く (指定値 {})",
                                show_value(name, value)
                            ),
                        )),
                    }
                }
                _ => {}
            }

            if name == "DENNO_KV_SLOTS" && value.trim() == "0" {
                out.push(Diag::new(
                    Level::Warn,
                    "kv_slots_legacy_zero",
                    vec![name.clone()],
                    "0 は「未指定」の旧来の書き方。既定 2 が使われる (明示するなら 1..=8)".to_string(),
                ));
            }

            if name == "DENNO_AUTO_CONFIG" && !auto_config_spelling_ok(value) {
                out.push(Diag::new(
                    Level::Warn,
                    "auto_config_spelling",
                    vec![name.clone()],
                    format!(
                        "真偽値として解釈できないので「無効」になる (指定値 {})。\
                         受け付けるのは 1/true/on/yes/auto/0/false/off/no。\
                         MiB は --vram-budget で渡す",
                        show_value(name, value)
                    ),
                ));
            }
        }

        sort_diags(&mut out);
        out
    }

    /// env と、CLI の明示値と、設定ファイルを突き合わせる。
    ///
    /// `resolve()` と同じ引数を取り、**解決はしない** (副作用なし)。解決自体が
    /// 失敗する場合は `resolve_error` を 1 件足し、実効値どうしの検査は飛ばす。
    pub fn diagnose(
        cli: &[(&'static str, String)],
        env: &BTreeMap<String, String>,
        file: Option<&FileConfig>,
    ) -> Vec<Diag> {
        diagnose_with(cli, env, file, false)
    }

    /// `diagnose` の scoped 版。`quiet_engine` の意味は `diagnose_env_with` と同じ。
    pub fn diagnose_with(
        cli: &[(&'static str, String)],
        env: &BTreeMap<String, String>,
        file: Option<&FileConfig>,
        quiet_engine: bool,
    ) -> Vec<Diag> {
        let mut out = diagnose_env_with(env, quiet_engine);

        for key in KEYS {
            let cli_val = cli.iter().find(|(k, _)| *k == key).map(|(_, v)| v.trim());
            let env_name = env_var(key);
            let env_val = env_name
                .and_then(|n| env.get(n).map(|v| (n, v.trim())))
                .filter(|(_, v)| !v.is_empty());
            let file_val = file
                .and_then(|f| f.get(key))
                .map(|v| v.as_text())
                .map(|v| v.trim().to_string());

            // CLI が env を黙って上書きする
            if let (Some(c), Some((n, e))) = (cli_val, env_val) {
                if c != e {
                    out.push(Diag::new(
                        Level::Warn,
                        "env_shadowed_by_cli",
                        vec![n.to_string()],
                        format!(
                            "`{key}` は CLI が最優先。env の値 {} は使われない (実効: {})",
                            show_value(n, e),
                            show_value(n, c)
                        ),
                    ));
                }
            }
            // env が設定ファイルを黙って上書きする
            if let (Some((n, e)), Some(f)) = (env_val, file_val.as_deref()) {
                if e != f {
                    out.push(Diag::new(
                        Level::Warn,
                        "config_shadowed_by_env",
                        vec![n.to_string()],
                        format!(
                            "`{key}` は env > 設定ファイル。{origin} の値 {} は使われない (実効: {})",
                            show_value(n, f),
                            show_value(n, e),
                            origin = file.map(|f| f.origin.as_str()).unwrap_or("")
                        ),
                    ));
                }
            }
            // env が無く CLI が設定ファイルを上書きする
            if let (Some(c), None, Some(f)) = (cli_val, env_val, file_val.as_deref()) {
                if c != f {
                    out.push(Diag::new(
                        Level::Warn,
                        "config_shadowed_by_cli",
                        vec![],
                        format!(
                            "`{key}` は CLI が最優先。{origin} の値 `{f}` は使われない (実効: `{c}`)",
                            origin = file.map(|f| f.origin.as_str()).unwrap_or("")
                        ),
                    ));
                }
            }
        }

        match resolve(cli, env, file) {
            Ok(r) => value_diags(&r, &mut out),
            Err(e) => out.push(Diag::new(Level::Error, "resolve_error", vec![], e)),
        }

        sort_diags(&mut out);
        out
    }

    /// 解決済みの実効値どうしの矛盾を見る。
    fn value_diags(r: &Resolved, out: &mut Vec<Diag>) {
        // max_tokens が文脈長を超える: 文脈が埋まった時点で止まるので上限に届かない。
        if let (Ok(Some(mt)), Ok(Some(mc))) = (r.number("max_tokens"), r.number("max_ctx")) {
            if mt > mc {
                out.push(Diag::new(
                    Level::Warn,
                    "max_tokens_over_ctx",
                    vec![],
                    format!(
                        "max_tokens={mt} が max_ctx={mc} を超える。文脈が埋まった時点で止まるので {mt} には届かない"
                    ),
                ));
            }
        }
    }

    /// 出力順を安定させる (`code` → `vars`)。
    fn sort_diags(out: &mut [Diag]) {
        out.sort_by(|a, b| {
            a.code
                .cmp(b.code)
                .then_with(|| a.vars.join(",").cmp(&b.vars.join(",")))
                .then_with(|| a.message.cmp(&b.message))
        });
    }
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
