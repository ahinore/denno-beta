//! task 1001: `src/config.rs` の境界試験。
//!
//! 対象は TOML サブセットの字句 (エスケープ / 行内コメント / 空白 / CRLF)、
//! CLI > env > 設定ファイル > 既定値の解決順、行番号付きのエラー、そして
//! 構造化サマリ (`summary_json` / `render_json`)。
//!
//! denno-cli は bin クレートなので lib 経由で取り込めない。`config_906.rs` と
//! 同じく `#[path]` で `src/config.rs` をそのまま読み込む。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;

use config::{default_value, env_var, parse_str, resolve, Source, Value, KEYS};

fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 解析失敗を期待してメッセージを返す。
fn perr(text: &str) -> String {
    parse_str(text, "t.toml").expect_err("この本文は弾かれるはず")
}

// ------------------------------------------------------------- エスケープ

#[test]
fn escapes_all_five_forms_round_trip() {
    let f = parse_str(r#"name = "a\nb\tc\rd\\e\"f""#, "t.toml").unwrap();
    assert_eq!(
        f.get("name"),
        Some(&Value::Str("a\nb\tc\rd\\e\"f".into()))
    );
}

#[test]
fn escape_at_end_of_line_is_reported() {
    let e = perr("name = \"abc\\\n");
    assert!(e.contains("エスケープが行末で切れている"), "{e}");
}

#[test]
fn unsupported_escape_lists_the_allowed_set() {
    let e = perr("name = \"a\\qb\"\n");
    assert!(e.contains("未対応のエスケープ `\\q`"), "{e}");
    assert!(e.contains("\\n"), "{e}");
    assert!(e.contains("\\\""), "{e}");
}

#[test]
fn unterminated_strings_are_reported() {
    let e = perr("name = \"abc\n");
    assert!(e.contains("文字列の閉じ `\"` が無い"), "{e}");
    let e = perr("name = 'abc\n");
    assert!(e.contains("文字列の閉じ `'` が無い"), "{e}");
}

#[test]
fn trailing_garbage_after_a_closed_string_is_rejected() {
    let e = perr("name = \"a\" b\n");
    assert!(e.contains("値の後ろに余分な文字 `b`"), "{e}");
    let e = perr("name = 'a'b\n");
    assert!(e.contains("値の後ろに余分な文字 `b`"), "{e}");
}

#[test]
fn comment_directly_after_a_string_is_allowed() {
    let f = parse_str("host = \"h\"#note\nname = 'x'#note\n", "t.toml").unwrap();
    assert_eq!(f.get("host"), Some(&Value::Str("h".into())));
    assert_eq!(f.get("name"), Some(&Value::Str("x".into())));
}

// --------------------------------------------------------- 行内コメント

#[test]
fn hash_inside_strings_is_not_a_comment() {
    let f = parse_str(
        "name = \"a # b\"\nhost = 'c#d'\ndir = '/models/#1'\n",
        "t.toml",
    )
    .unwrap();
    assert_eq!(f.get("name"), Some(&Value::Str("a # b".into())));
    assert_eq!(f.get("host"), Some(&Value::Str("c#d".into())));
    assert_eq!(f.get("dir"), Some(&Value::Str("/models/#1".into())));
}

#[test]
fn inline_comment_after_an_int_is_stripped() {
    for (text, want) in [
        ("port = 9#tail\n", 9),
        ("port = 9 # tail\n", 9),
        ("port = 9\t#tail\n", 9),
        ("port = 9   #\n", 9),
    ] {
        let f = parse_str(text, "t.toml").unwrap();
        assert_eq!(f.get("port"), Some(&Value::Int(want)), "{text:?}");
    }
}

#[test]
fn empty_value_and_comment_only_value_are_errors() {
    let e = perr("port =\n");
    assert!(e.contains("値が無い"), "{e}");
    let e = perr("port = # c\n");
    assert!(e.contains("未対応"), "{e}");
    let e = perr("port =    \n");
    assert!(e.contains("値が無い"), "{e}");
}

// --------------------------------------------------------------- 空白

#[test]
fn tabs_spaces_blank_lines_and_crlf_are_accepted() {
    let text = "\r\n  port\t=\t9001\r\n\r\n# c\r\n\thost\t=\t\"h\" \t\r\n\r\n";
    let f = parse_str(text, "t.toml").unwrap();
    assert_eq!(f.get("port"), Some(&Value::Int(9001)));
    assert_eq!(f.get("host"), Some(&Value::Str("h".into())));
}

#[test]
fn whitespace_variants_of_the_same_key_still_collide() {
    let e = perr("port = 1\n   port\t=\t2\n");
    assert!(e.contains("キー `port` が重複"), "{e}");
    assert!(e.contains("1 行目で既に定義"), "{e}");
}

#[test]
fn key_syntax_errors_are_line_addressed() {
    let e = perr("po rt = 1\n");
    assert!(e.contains("t.toml:1"), "{e}");
    assert!(e.contains("に使える文字は英数字"), "{e}");
    let e = perr("= 1\n");
    assert!(e.contains("キーが空"), "{e}");
    let e = perr("port\n");
    assert!(e.contains("`key = value` の形になっていない"), "{e}");
    let e = perr("[serve]\nport = 1\n");
    assert!(e.contains("セクション"), "{e}");
}

#[test]
fn unknown_key_hint_fires_on_a_prefix_typo() {
    let e = perr("por = 1\n");
    assert!(e.contains("未知のキー `por`"), "{e}");
    assert!(e.contains("もしかして: port"), "{e}");
    // 似ていない綴りにはヒントを出さない
    let e = perr("zzz = 1\n");
    assert!(!e.contains("もしかして"), "{e}");
}

// --------------------------------------------------------------- 整数

#[test]
fn integers_accept_a_leading_plus_and_reject_double_signs() {
    let f = parse_str("port = +8098\n", "t.toml").unwrap();
    assert_eq!(f.get("port"), Some(&Value::Int(8098)));
    let e = perr("port = --5\n");
    assert!(e.contains("整数として読めない"), "{e}");
    let e = perr("port = 1x\n");
    assert!(e.contains("未対応"), "{e}");
    let e = perr("port = 99999999999999999999\n");
    assert!(e.contains("整数として読めない"), "{e}");
}

#[test]
fn integer_bounds_are_per_key() {
    let e = perr("port = 65536\n");
    assert!(e.contains("0..=65535"), "{e}");
    let e = perr("port = -1\n");
    assert!(e.contains("0 以上"), "{e}");
    let e = perr("max_ctx = 0\n");
    assert!(e.contains("1..=4294967295"), "{e}");
    let e = perr("max_tokens = 4294967296\n");
    assert!(e.contains("1..=4294967295"), "{e}");
    let e = perr("kv_slots = 9\n");
    assert!(e.contains("0..=8"), "{e}");
    // kv_slots = 0 は「未指定」の意味なので通す
    let f = parse_str("kv_slots = 0\n", "t.toml").unwrap();
    assert_eq!(f.get("kv_slots"), Some(&Value::Int(0)));
}

#[test]
fn string_key_accepts_a_bare_integer_as_text() {
    // vram_budget は `auto` と MiB の整数の両方を受ける
    let f = parse_str("vram_budget = 12000\nname = 5\n", "t.toml").unwrap();
    assert_eq!(f.get("vram_budget"), Some(&Value::Str("12000".into())));
    assert_eq!(f.get("name"), Some(&Value::Str("5".into())));
}

#[test]
fn int_key_rejects_a_quoted_number() {
    let e = perr("port = \"8098\"\n");
    assert!(e.contains("`port` は整数"), "{e}");
}

// ------------------------------------------------------- env 上書き

#[test]
fn env_beats_config_and_cli_beats_env() {
    let file = parse_str(
        "port = 9001\nhost = \"cfg-host\"\nname = \"cfg-name\"\n",
        "c.toml",
    )
    .unwrap();
    let env = env_of(&[
        ("DENNO_FLASH_PORT", "9002"),
        ("DENNO_FLASH_HOST", "env-host"),
    ]);
    let cli = vec![("port", "9003".to_string())];
    let r = resolve(&cli, &env, Some(&file)).unwrap();
    assert_eq!(r.string("port").as_deref(), Some("9003"));
    assert_eq!(r.source("port"), Some(Source::Cli));
    assert_eq!(r.string("host").as_deref(), Some("env-host"));
    assert_eq!(r.source("host"), Some(Source::Env));
    assert_eq!(r.string("name").as_deref(), Some("cfg-name"));
    assert_eq!(r.source("name"), Some(Source::Config));
}

#[test]
fn whitespace_only_env_falls_through_to_the_file() {
    let file = parse_str("host = \"cfg-host\"\n", "c.toml").unwrap();
    let env = env_of(&[("DENNO_FLASH_HOST", ""), ("DENNO_FLASH_PORT", "   ")]);
    let r = resolve(&[], &env, Some(&file)).unwrap();
    assert_eq!(r.string("host").as_deref(), Some("cfg-host"));
    assert_eq!(r.source("host"), Some(Source::Config));
    // 空白だけの env は未設定扱い -> 既定値
    assert_eq!(r.string("port").as_deref(), Some("8098"));
    assert_eq!(r.source("port"), Some(Source::Default));
}

#[test]
fn every_int_key_honours_its_env_var() {
    let cases = [
        ("port", "DENNO_FLASH_PORT", "9001"),
        ("max_ctx", "DENNO_FLASH_MAX_CTX", "8192"),
        ("max_tokens", "DENNO_FLASH_MAX_TOKENS", "64"),
        ("kv_slots", "DENNO_KV_SLOTS", "3"),
    ];
    for (key, name, value) in cases {
        let env = env_of(&[(name, value)]);
        let r = resolve(&[], &env, None).unwrap();
        assert_eq!(env_var(key), Some(name), "{key}");
        assert_eq!(r.string(key).as_deref(), Some(value), "{key}");
        assert_eq!(r.source(key), Some(Source::Env), "{key}");
        assert_eq!(
            r.number(key).unwrap(),
            Some(value.parse::<u64>().unwrap()),
            "{key}"
        );
    }
}

#[test]
fn env_numbers_are_trimmed_before_parsing() {
    let env = env_of(&[("DENNO_FLASH_PORT", " 9001 ")]);
    let r = resolve(&[], &env, None).unwrap();
    assert_eq!(r.u16("port").unwrap(), Some(9001));
    // CLI 由来の値も同じ経路 (両端の空白は無視)
    let cli = vec![("port", "\t9002 ".to_string())];
    let r = resolve(&cli, &BTreeMap::new(), None).unwrap();
    assert_eq!(r.u16("port").unwrap(), Some(9002));
}

#[test]
fn bad_env_numbers_name_the_variable_and_origin() {
    let env = env_of(&[("DENNO_FLASH_PORT", "http")]);
    let e = resolve(&[], &env, None).unwrap_err();
    assert!(e.contains("`port`"), "{e}");
    assert!(e.contains("http"), "{e}");
    assert!(e.contains("env DENNO_FLASH_PORT"), "{e}");
    let env = env_of(&[("DENNO_FLASH_MAX_CTX", "0")]);
    let e = resolve(&[], &env, None).unwrap_err();
    assert!(e.contains("1..=4294967295"), "{e}");
    assert!(e.contains("DENNO_FLASH_MAX_CTX"), "{e}");
}

#[test]
fn resolve_reports_nothing_for_dir_when_unset() {
    let r = resolve(&[], &BTreeMap::new(), None).unwrap();
    assert_eq!(r.path("dir"), None);
    assert_eq!(r.source("dir"), None);
    assert_eq!(r.slot("dir"), None);
    // 既定値を持つキーは必ず埋まる
    for key in ["host", "port", "max_ctx", "name", "max_tokens"] {
        assert!(default_value(key).is_some(), "{key}");
        assert!(
            r.string(key).is_some(),
            "{key} が既定値で解決されていない"
        );
    }
    // mtp_path / steer_recipe / vram_budget は既定値なし
    for key in ["mtp_path", "steer_recipe", "vram_budget"] {
        assert_eq!(default_value(key), None, "{key}");
        assert_eq!(r.slot(key), None, "{key}");
    }
}

#[test]
fn config_detail_is_the_file_origin() {
    let file = parse_str("name = \"n\"\n", "conf/x.toml").unwrap();
    let r = resolve(&[], &BTreeMap::new(), Some(&file)).unwrap();
    let slot = r.slot("name").unwrap();
    assert_eq!(slot.source, Source::Config);
    assert_eq!(slot.detail, "conf/x.toml");
    assert_eq!(slot.origin_label(), "config conf/x.toml");
}

// -------------------------------------------------- render / render_json

#[test]
fn rendered_config_parses_back_to_the_same_values() {
    let file = parse_str(
        "name = \"a b # c\"\ndir = '/m/\\\\x'\nhost = \"multi\\nline\"\nport = 9001\n",
        "c.toml",
    )
    .unwrap();
    let r = resolve(&[], &BTreeMap::new(), Some(&file)).unwrap();
    let text = r.render();
    let back = parse_str(&text, "roundtrip.toml").expect("render の出力が読み戻せない");
    for key in KEYS {
        match r.string(key) {
            Some(want) => {
                let got = back.get(key).unwrap_or_else(|| panic!("{key} が読み戻せない"));
                assert_eq!(got.as_text_for_test(), want, "{key}");
            }
            None => assert!(back.get(key).is_none(), "{key} が未設定のはず"),
        }
    }
}

#[test]
fn render_marks_source_and_escapes_strings() {
    let env = env_of(&[("DENNO_FLASH_PORT", "9002"), ("DENNO_FLASH_NAME", "a\"b")]);
    let r = resolve(&[], &env, None).unwrap();
    let s = r.render();
    assert!(s.contains("port = 9002  # source: env DENNO_FLASH_PORT"), "{s}");
    assert!(s.contains("host = \"127.0.0.1\"  # source: default"), "{s}");
    assert!(s.contains("# dir = <unset>"), "{s}");
    assert!(s.contains("name = \"a\\\"b\""), "{s}");
    // kv_slots = 0 の注記
    let env = env_of(&[("DENNO_KV_SLOTS", "0")]);
    let s = resolve(&[], &env, None).unwrap().render();
    assert!(s.contains("(0 = 未指定 -> 既定 2)"), "{s}");
}

#[test]
fn summary_json_has_every_key_and_typed_values() {
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let r = resolve(&[], &env, None).unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.render_json()).expect("JSON でない");
    let obj = v.as_object().expect("トップレベルが object でない");
    assert_eq!(obj.len(), KEYS.len());
    for key in KEYS {
        assert!(obj.contains_key(key), "{key} が無い");
    }
    // 整数キーは JSON 数値、文字列キーは文字列
    assert_eq!(obj["port"]["value"].as_u64(), Some(9002));
    assert_eq!(obj["port"]["source"].as_str(), Some("env"));
    assert_eq!(obj["port"]["detail"].as_str(), Some("DENNO_FLASH_PORT"));
    assert_eq!(obj["port"]["origin"].as_str(), Some("env DENNO_FLASH_PORT"));
    assert_eq!(obj["port"]["spec"]["kind"].as_str(), Some("int"));
    assert_eq!(obj["port"]["spec"]["min"].as_u64(), Some(0));
    assert_eq!(obj["port"]["spec"]["max"].as_u64(), Some(65535));
    assert_eq!(obj["host"]["value"].as_str(), Some("127.0.0.1"));
    assert_eq!(obj["host"]["source"].as_str(), Some("default"));
    assert_eq!(obj["host"]["spec"]["kind"].as_str(), Some("str"));
}

#[test]
fn summary_json_marks_unset_keys() {
    let r = resolve(&[], &BTreeMap::new(), None).unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.render_json()).unwrap();
    for key in ["dir", "mtp_path", "steer_recipe", "vram_budget"] {
        assert_eq!(v[key]["set"].as_bool(), Some(false), "{key}");
        assert_eq!(v[key]["source"].as_str(), Some("unset"), "{key}");
        assert!(v[key].get("value").is_none(), "{key}");
    }
    assert_eq!(v["max_ctx"]["set"].as_bool(), Some(true));
    assert_eq!(v["max_ctx"]["value"].as_u64(), Some(4096));
}

#[test]
fn summary_json_escapes_control_characters() {
    let env = env_of(&[("DENNO_FLASH_NAME", "a\"b\nc\t")]);
    let r = resolve(&[], &env, None).unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.render_json()).unwrap();
    assert_eq!(v["name"]["value"].as_str(), Some("a\"b\nc\t"));
}

#[test]
fn summary_json_agrees_with_render_for_every_key() {
    let file = parse_str(
        "dir = '/m/f'\nname = \"a b\"\nkv_slots = 0\nvram_budget = 12000\n",
        "c.toml",
    )
    .unwrap();
    let env = env_of(&[("DENNO_FLASH_PORT", "9002"), ("DENNO_KV_SLOTS", "0")]);
    let cli = vec![("max_tokens", "77".to_string())];
    let r = resolve(&cli, &env, Some(&file)).unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.render_json()).unwrap();
    for key in KEYS {
        // 未設定のキーは JSON 側に value が無い -> Resolved にも無い
        let json_text = match &v[key]["value"] {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Null => {
                assert_eq!(r.string(key), None, "{key} が JSON では未設定");
                assert_eq!(v[key]["source"].as_str(), Some("unset"), "{key}");
                continue;
            }
            other => panic!("{key} の value が数値でも文字列でもない: {other}"),
        };
        assert_eq!(
            json_text,
            r.string(key).unwrap_or_else(|| panic!("{key} が Resolved に無い")),
            "{key} の JSON と Resolved が食い違う"
        );
        assert_eq!(
            v[key]["source"].as_str(),
            Some(r.source(key).expect("source が無い").as_str()),
            "{key}"
        );
    }
    // int キーだけが JSON 数値になる
    for key in KEYS {
        let Some(spec_kind) = v[key]["spec"]["kind"].as_str() else {
            assert!(!v[key]["set"].as_bool().unwrap(), "{key}: 設定済みなのに spec が無い");
            continue;
        };
        let is_num = v[key]["value"].is_number();
        assert_eq!(spec_kind == "int", is_num, "{key} の JSON 型が kind と違う");
    }
    assert_eq!(v["max_tokens"]["value"].as_u64(), Some(77));
    assert_eq!(v["max_tokens"]["source"].as_str(), Some("cli"));
    assert_eq!(v["kv_slots"]["value"].as_u64(), Some(0));
    assert_eq!(v["vram_budget"]["value"].as_str(), Some("12000"));
}

/// `Value::as_text` は private なので試験側に同じものを持つ。
trait AsTextForTest {
    fn as_text_for_test(&self) -> String;
}

impl AsTextForTest for Value {
    fn as_text_for_test(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(n) => n.to_string(),
        }
    }
}
