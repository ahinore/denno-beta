//! P837: OpenAI Structured Outputs (`response_format: {"type":"json_schema",...}`) の
//! スキーマ検証と、モデルへ渡すプロンプト指示の組み立て。
//!
//! このモジュールが持つのは **検証 + 決定的な文字列化** だけで、数値経路には
//! 触れない。出力の JSON 適合は次の 2 段で担保する:
//!
//! 1. 受け付ける時点でスキーマを検査する (`strict: true` では下の「厳格モード」の
//!    制約も満たすことを要求し、満たさなければ 400 — **黙って制約を捨てない**)。
//! 2. モデルへはスキーマそのもの (正規化した JSON 文字列) を system に足して渡す。
//!
//! トークンレベルの拘束デコード (logits マスク) は **行わない**。既存の
//! 拘束デコード (`toolmask.rs`) は `<tool_call>` XML スキャフォールド専用で、
//! 任意の JSON Schema から許容トークン集合を組む機構は engine に無い。ここで
//! 「strict だから token マスクしている」とは書かない (実際の適合はモデル依存で、
//! スキーマを満たさない出力も返り得る)。この限界は docs/api.md に明記する。
//!
//! 厳格モード (`strict: true`) の追加制約 (OpenAI の strict と同じ系統):
//!
//! - `type: "object"` かつ `properties` を持つノードは `additionalProperties: false`
//!   と、全 property を列挙した `required` を要求する。
//! - `oneOf` / `anyOf` / `allOf` / `not` / `if` / `then` / `else` /
//!   `patternProperties` / `$defs` / `definitions` / `prefixItems` /
//!   `unevaluatedProperties` / `unevaluatedItems` / `dependentSchemas` は非対応 (400)。
//!
//! `strict` が false/未指定のときは上記キーワードを **受理して無視する**
//! (スキーマ JSON はそのままモデルへ渡るため、指示としては効き得るが、この
//! サーバーは検査も保証もしない)。`$ref` は strict に関係なく常に 400 —
//! 参照先を解決する機構が無く、解決せずに渡すとスキーマの意味が変わるため。

use serde_json::Value;

/// スキーマ文字列の上限 (system プロンプトを押し出さないための防御)。
pub const MAX_SCHEMA_BYTES: usize = 16 * 1024;
/// スキーマのネスト上限 (再帰の深さ)。
pub const MAX_SCHEMA_DEPTH: usize = 16;

/// JSON Schema として扱える `type` の値。
const SUPPORTED_TYPES: &[&str] =
    &["object", "array", "string", "number", "integer", "boolean", "null"];

/// 厳格モードで非対応のキーワード (見つけたら 400)。
const UNSUPPORTED_STRICT_KEYWORDS: &[&str] = &[
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "$defs",
    "definitions",
    "prefixItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "dependentSchemas",
];

/// `response_format.json_schema` の検証済み内容。
#[derive(Debug, Clone)]
pub struct JsonSchemaSpec {
    /// `json_schema.name` (必須・非空)
    pub name: String,
    /// `json_schema.description` (任意)
    pub description: Option<String>,
    /// `json_schema.schema` (オブジェクト or 真偽値の JSON Schema)
    pub schema: Value,
    /// `json_schema.strict` (既定 false)
    pub strict: bool,
}

impl JsonSchemaSpec {
    /// `json_schema` オブジェクトを検証して取り出す。`param` は返すエラーの
    /// `param` フィールドに使う接頭辞 (`response_format.json_schema`)。
    pub fn parse(spec: &Value) -> Result<Self, String> {
        let obj = spec
            .as_object()
            .ok_or_else(|| "response_format.json_schema must be a JSON object".to_string())?;
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                "response_format.json_schema.name is required and must be a string".to_string()
            })?;
        if name.trim().is_empty() {
            return Err("response_format.json_schema.name must be a non-empty string".to_string());
        }
        let description = match obj.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(
                    "response_format.json_schema.description must be a string".to_string()
                )
            }
        };
        let strict = match obj.get("strict") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err("response_format.json_schema.strict must be a boolean".to_string())
            }
        };
        let schema = obj
            .get("schema")
            .ok_or_else(|| "response_format.json_schema.schema is required".to_string())?
            .clone();
        let spec = Self {
            name: name.to_string(),
            description,
            schema,
            strict,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// スキーマ本文の検査。エラー文字列はそのまま OpenAI 形式の
    /// `error.message` になる (パス付き)。
    pub fn validate(&self) -> Result<(), String> {
        if !(self.schema.is_object() || self.schema.is_boolean()) {
            return Err(
                "response_format.json_schema.schema must be a JSON object or a boolean".to_string(),
            );
        }
        let rendered = serde_json::to_string(&self.schema)
            .map_err(|e| format!("response_format.json_schema.schema is not serializable: {e}"))?;
        if rendered.len() > MAX_SCHEMA_BYTES {
            return Err(format!(
                "response_format.json_schema.schema is too large ({} bytes; max {MAX_SCHEMA_BYTES})",
                rendered.len()
            ));
        }
        walk_schema(&self.schema, "$", 0, self.strict)
    }

    /// system プロンプトへ足す固定指示。スキーマは **キー順を正規化した**
    /// JSON 文字列で埋め込む (同じスキーマ → 同じプロンプト → 同じ出力)。
    pub fn hint(&self) -> String {
        let mode = if self.strict { "strict" } else { "best-effort" };
        let mut out = format!(
            "Return a single JSON value that conforms to the following JSON Schema \
             (mode: {mode}). Output the JSON only: no markdown fences, no prose.\n\
             schema name: {}\n",
            self.name
        );
        if let Some(d) = &self.description {
            out.push_str(&format!("schema description: {d}\n"));
        }
        // serde_json::Value は BTreeMap なので、キー順は常に昇順 = 決定的。
        out.push_str(&format!(
            "schema: {}",
            serde_json::to_string(&self.schema).unwrap_or_else(|_| "{}".to_string())
        ));
        out
    }
}

/// `response_format.type` の解釈結果。
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// `{"type":"text"}` (または未指定) — プロンプト無変換
    Text,
    /// `{"type":"json_object"}` — JSON オブジェクト限定の指示を足す
    JsonObject,
    /// `{"type":"json_schema", ...}` — スキーマ指示を足す
    JsonSchema(JsonSchemaSpec),
}

/// `response_format` を検証する。`param` は外側で付けるのでここではメッセージだけ。
pub fn parse_response_format(v: &Value) -> Result<ResponseFormat, String> {
    let ty = v
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| "response_format: object form requires a string \"type\"".to_string())?;
    match ty {
        "text" => Ok(ResponseFormat::Text),
        "json_object" => Ok(ResponseFormat::JsonObject),
        "json_schema" => {
            let spec = v.get("json_schema").ok_or_else(|| {
                "response_format.json_schema is required when type is \"json_schema\"".to_string()
            })?;
            Ok(ResponseFormat::JsonSchema(JsonSchemaSpec::parse(spec)?))
        }
        other => Err(format!(
            "response_format.type: unsupported value {other:?} (text/json_object/json_schema)"
        )),
    }
}

/// スキーマ木を歩いて検査する。`path` は JSON Pointer 風 (`$.properties.a`)。
fn walk_schema(v: &Value, path: &str, depth: usize, strict: bool) -> Result<(), String> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(format!(
            "response_format.json_schema.schema is nested too deeply at {path} (max depth \
             {MAX_SCHEMA_DEPTH})"
        ));
    }
    // 真偽値スキーマ (true / false) は葉。
    let obj = match v {
        Value::Bool(_) => return Ok(()),
        Value::Object(m) => m,
        _ => {
            return Err(format!(
                "response_format.json_schema.schema: {path} must be a JSON object or a boolean"
            ))
        }
    };
    if obj.contains_key("$ref") {
        return Err(format!(
            "response_format.json_schema.schema: {path}: $ref is not supported"
        ));
    }
    if strict {
        for kw in UNSUPPORTED_STRICT_KEYWORDS {
            if obj.contains_key(*kw) {
                return Err(format!(
                    "response_format.json_schema.schema: {path}: \"{kw}\" is not supported with \
                     strict: true"
                ));
            }
        }
    }
    // type: 文字列の JSON Schema 型名のみ (配列形式は非対応)。
    match obj.get("type") {
        None | Some(Value::Null) => {}
        Some(Value::String(t)) => {
            if !SUPPORTED_TYPES.contains(&t.as_str()) {
                return Err(format!(
                    "response_format.json_schema.schema: {path}.type: unsupported type {t:?} \
                     (object/array/string/number/integer/boolean/null)"
                ));
            }
        }
        Some(_) => {
            return Err(format!(
                "response_format.json_schema.schema: {path}.type must be a string"
            ))
        }
    }
    // properties
    let properties = match obj.get("properties") {
        None | Some(Value::Null) => None,
        Some(Value::Object(m)) => Some(m),
        Some(_) => {
            return Err(format!(
                "response_format.json_schema.schema: {path}.properties must be a JSON object"
            ))
        }
    };
    if strict && obj.get("type").and_then(|t| t.as_str()) == Some("object") {
        if let Some(props) = properties {
            let addl_off = obj.get("additionalProperties") == Some(&Value::Bool(false));
            if !addl_off {
                return Err(format!(
                    "response_format.json_schema.schema: {path}: strict: true requires \
                     \"additionalProperties\": false on objects with properties"
                ));
            }
            let required: Vec<&str> = obj
                .get("required")
                .and_then(|r| r.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            for key in props.keys() {
                if !required.contains(&key.as_str()) {
                    return Err(format!(
                        "response_format.json_schema.schema: {path}: strict: true requires every \
                         property to be listed in \"required\" (missing {key:?})"
                    ));
                }
            }
        }
    }
    if let Some(props) = properties {
        for (k, sub) in props {
            walk_schema(sub, &format!("{path}.properties.{k}"), depth + 1, strict)?;
        }
    }
    // items
    match obj.get("items") {
        None | Some(Value::Null) => {}
        Some(sub @ Value::Object(_)) | Some(sub @ Value::Bool(_)) => {
            walk_schema(sub, &format!("{path}.items"), depth + 1, strict)?
        }
        Some(_) => {
            return Err(format!(
                "response_format.json_schema.schema: {path}.items must be a JSON object or a \
                 boolean (tuple form is not supported)"
            ))
        }
    }
    // additionalProperties (オブジェクト形式のみスキーマとして辿る)
    match obj.get("additionalProperties") {
        None | Some(Value::Null) => {}
        Some(sub @ Value::Object(_)) | Some(sub @ Value::Bool(_)) => {
            walk_schema(sub, &format!("{path}.additionalProperties"), depth + 1, strict)?
        }
        Some(_) => {
            return Err(format!(
                "response_format.json_schema.schema: {path}.additionalProperties must be a JSON \
                 object or a boolean"
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(schema: Value, strict: bool) -> JsonSchemaSpec {
        JsonSchemaSpec {
            name: "out".to_string(),
            description: None,
            schema,
            strict,
        }
    }

    #[test]
    fn accepts_simple_object_schema() {
        let s = json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["answer", "score"],
            "additionalProperties": false,
        });
        assert!(spec(s.clone(), true).validate().is_ok());
        assert!(spec(s, false).validate().is_ok());
    }

    #[test]
    fn boolean_schema_is_accepted() {
        assert!(spec(json!(true), true).validate().is_ok());
        assert!(spec(json!(false), false).validate().is_ok());
    }

    #[test]
    fn non_object_schema_is_rejected() {
        assert!(spec(json!("nope"), false).validate().is_err());
        assert!(spec(json!([1, 2]), false).validate().is_err());
    }

    #[test]
    fn ref_is_rejected_in_both_modes() {
        let s = json!({"type": "object", "properties": {"a": {"$ref": "#/$defs/a"}},
                       "required": ["a"], "additionalProperties": false});
        let e = spec(s.clone(), false).validate().unwrap_err();
        assert!(e.contains("$ref"), "{e}");
        assert!(spec(s, true).validate().is_err());
    }

    #[test]
    fn strict_rejects_unsupported_keywords() {
        for kw in ["oneOf", "anyOf", "allOf", "not"] {
            let mut m = serde_json::Map::new();
            m.insert("type".to_string(), json!("object"));
            m.insert(kw.to_string(), json!([{"type": "string"}]));
            let s = Value::Object(m);
            assert!(spec(s.clone(), false).validate().is_ok(), "{kw} は非 strict で受理");
            let e = spec(s, true).validate().unwrap_err();
            assert!(e.contains(kw), "{kw}: {e}");
        }
    }

    #[test]
    fn strict_requires_additional_properties_false_and_full_required() {
        // additionalProperties 未指定
        let s = json!({"type": "object", "properties": {"a": {"type": "string"}},
                       "required": ["a"]});
        assert!(spec(s.clone(), false).validate().is_ok());
        assert!(spec(s, true).validate().is_err());

        // required に載っていない property
        let s = json!({"type": "object", "properties": {"a": {"type": "string"}},
                       "required": [], "additionalProperties": false});
        let e = spec(s.clone(), true).validate().unwrap_err();
        assert!(e.contains("required"), "{e}");
        assert!(spec(s, false).validate().is_ok());
    }

    #[test]
    fn nested_unsupported_type_is_rejected() {
        let s = json!({"type": "object", "properties": {"a": {"type": "tuple"}},
                       "required": ["a"], "additionalProperties": false});
        let e = spec(s, false).validate().unwrap_err();
        assert!(e.contains("unsupported type"), "{e}");
    }

    #[test]
    fn depth_limit() {
        let mut s = json!({"type": "string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 2) {
            s = json!({"type": "array", "items": s});
        }
        let e = spec(s, false).validate().unwrap_err();
        assert!(e.contains("nested too deeply"), "{e}");
    }

    #[test]
    fn size_limit() {
        let long = "x".repeat(MAX_SCHEMA_BYTES);
        let s = json!({"type": "string", "description": long});
        let e = spec(s, false).validate().unwrap_err();
        assert!(e.contains("too large"), "{e}");
    }

    #[test]
    fn hint_is_deterministic_and_contains_schema() {
        // キー順が違っても同じ文字列になる (BTreeMap による正規化)
        let a = Value::Object(
            [
                ("type".to_string(), json!("object")),
                ("required".to_string(), json!(["z", "a"])),
            ]
            .into_iter()
            .collect(),
        );
        let b = Value::Object(
            [
                ("required".to_string(), json!(["z", "a"])),
                ("type".to_string(), json!("object")),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(spec(a, true).hint(), spec(b, true).hint());
        let h = spec(json!({"type": "string"}), true).hint();
        assert!(h.contains("\"type\":\"string\""), "{h}");
        assert!(h.contains("strict"), "{h}");
        assert!(h.contains("no markdown fences"), "{h}");
    }

    #[test]
    fn parse_requires_name_and_schema() {
        let e = JsonSchemaSpec::parse(&json!({"schema": {"type": "string"}})).unwrap_err();
        assert!(e.contains("name"), "{e}");
        let e =
            JsonSchemaSpec::parse(&json!({"name": "  ", "schema": {"type": "string"}})).unwrap_err();
        assert!(e.contains("non-empty"), "{e}");
        let e = JsonSchemaSpec::parse(&json!({"name": "n"})).unwrap_err();
        assert!(e.contains("schema is required"), "{e}");
        let e = JsonSchemaSpec::parse(&json!({"name": "n", "schema": {"type": "string"},
                                             "strict": "yes"}))
        .unwrap_err();
        assert!(e.contains("boolean"), "{e}");
        assert!(JsonSchemaSpec::parse(&json!({"name": "n", "schema": {"type": "string"}})).is_ok());
    }

    #[test]
    fn parse_response_format_variants() {
        assert!(matches!(
            parse_response_format(&json!({"type": "text"})).unwrap(),
            ResponseFormat::Text
        ));
        assert!(matches!(
            parse_response_format(&json!({"type": "json_object"})).unwrap(),
            ResponseFormat::JsonObject
        ));
        assert!(parse_response_format(&json!({"type": "json_schema"})).is_err());
        assert!(parse_response_format(&json!({"type": "yaml"})).is_err());
        assert!(parse_response_format(&json!("text")).is_err());
        let ok = parse_response_format(&json!({
            "type": "json_schema",
            "json_schema": {"name": "n", "strict": true, "schema": {"type": "string"}}
        }))
        .unwrap();
        match ok {
            ResponseFormat::JsonSchema(s) => assert!(s.strict),
            _ => panic!("json_schema にならない"),
        }
    }
}
