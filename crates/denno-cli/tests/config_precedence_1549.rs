//! task 1549: `src/config.rs` の解決順 (`config::resolve`) の境界テスト。
//!
//! 見ているのは次の 5 点:
//!
//! 1. **解決順の全キー網羅** — `SPEC` の 10 キーすべてについて
//!    `CLI > env > 設定ファイル > 既定値` を 1 段ずつ落として確認する
//!    (`Resolved::source` / `Slot::detail` の出所も併せて見る)。
//! 2. **出所の表記** — `Slot::origin_label()` が `cli` / `env <NAME>` /
//!    `config <path>` / `default` になること。
//! 3. **整数キーの境界** — `port` / `max_ctx` / `max_tokens` / `kv_slots` の
//!    min / max がファイル・env・CLI のどこ由来でも同じ判定になること。
//! 4. **設定ファイルの誤り** — 壊れた TOML サブセットを行番号付きで弾くこと。
//! 5. **`render()` の往復** — 出力をそのまま設定ファイルとして読み直すと
//!    同じ値・同じ出所 (config) に戻ること。
//!
//! denno-cli は bin クレートなので lib 経由で取り込めない。既存の
//! `config_906.rs` / `config_diag_1066.rs` と同じく `#[path]` で
//! `src/config.rs` をそのまま読み込む。プロセスの env もファイルも触らない
//! (`parse_str` と `resolve` は引数だけを見る純関数)。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;

use config::{default_value, env_var, parse_str, resolve, Source, Value, KEYS};

const U32_MAX: u64 = u32::MAX as u64;

/// `SPEC` の整数キーと、その範囲 (src/config.rs と一致させる)。
const INT_KEYS: [(&str, u64, u64); 4] = [
    ("port", 0, 65535),
    ("max_ctx", 1, U32_MAX),
    ("max_tokens", 1, U32_MAX),
    ("kv_slots", 0, 8),
];

/// 既定値を持つキー → 既定値。持たないキーは入っていない。
const DEFAULTS: [(&str, &str); 5] = [
    ("host", "127.0.0.1"),
    ("port", "8098"),
    ("max_ctx", "4096"),
    ("name", "qwen3.8-flash-next"),
    ("max_tokens", "512"),
];

/// 設定ファイルに書いたときの値 (全 10 キー)。
const FILE_PAIRS: [(&str, &str); 10] = [
    ("dir", "/models/flash"),
    ("host", "cfg-host"),
    ("port", "9001"),
    ("max_ctx", "2048"),
    ("name", "cfg-name"),
    ("max_tokens", "256"),
    ("mtp_path", "/models/mtp.bin"),
    ("kv_slots", "3"),
    ("steer_recipe", "recipe-a"),
    ("vram_budget", "12000"),
];

/// env に書いたときの値 (全 10 キー、キー順は `KEYS` と同じ)。
const ENV_PAIRS: [(&str, &str); 10] = [
    ("DENNO_FLASH_DIR", "/env/models"),
    ("DENNO_FLASH_HOST", "env-host"),
    ("DENNO_FLASH_PORT", "9002"),
    ("DENNO_FLASH_MAX_CTX", "1024"),
    ("DENNO_FLASH_NAME", "env-name"),
    ("DENNO_FLASH_MAX_TOKENS", "128"),
    ("DENNO_MTP_PATH", "/env/mtp.bin"),
    ("DENNO_KV_SLOTS", "2"),
    ("DENNO_STEER_RECIPE", "env-recipe"),
    ("DENNO_AUTO_CONFIG", "auto"),
];

/// CLI に明示したときの値 (全 10 キー)。
const CLI_PAIRS: [(&str, &str); 10] = [
    ("dir", "/cli/models"),
    ("host", "cli-host"),
    ("port", "9003"),
    ("max_ctx", "512"),
    ("name", "cli-name"),
    ("max_tokens", "64"),
    ("mtp_path", "/cli/mtp.bin"),
    ("kv_slots", "1"),
    ("steer_recipe", "cli-recipe"),
    ("vram_budget", "auto"),
];

fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn cli_of(pairs: &[(&str, &str)]) -> Vec<(&'static str, String)> {
    pairs
        .iter()
        .map(|(k, v)| (leak(k), (*v).to_string()))
        .collect()
}

/// `cli` のキーは `&'static str` を要求される。テスト用の固定表なので
/// 一度だけ漏らして `'static` にする (プロセスは 1 テストで終わる)。
fn leak(key: &str) -> &'static str {
    KEYS.iter()
        .copied()
        .find(|k| *k == key)
        .unwrap_or_else(|| panic!("`{key}` は KEYS に無い"))
}

fn file_all() -> config::FileConfig {
    let mut body = String::new();
    for (k, v) in FILE_PAIRS {
        match v.parse::<i64>() {
            Ok(_) => body.push_str(&format!("{k} = {v}\n")),
            Err(_) => body.push_str(&format!("{k} = \"{v}\"\n")),
        }
    }
    parse_str(&body, "c.toml").expect("全キーの設定ファイル")
}

// ------------------------------------------------------- 10 キー網羅: 出所の階段

#[test]
fn every_key_has_an_env_name_and_the_expected_default_shape() {
    for key in KEYS {
        assert!(env_var(key).is_some(), "`{key}` に env 名が無い");
        let want = DEFAULTS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
        assert_eq!(default_value(key), want, "`{key}` の既定値");
    }
    // 既定値を持つキーはちょうど DEFAULTS の 5 個
    assert_eq!(
        KEYS.iter().filter(|k| default_value(k).is_some()).count(),
        DEFAULTS.len()
    );
}

#[test]
fn cli_layer_wins_for_every_key() {
    let file = file_all();
    let env = env_of(&ENV_PAIRS);
    let cli = cli_of(&CLI_PAIRS);
    let r = resolve(&cli, &env, Some(&file)).expect("解決");

    for (key, want) in CLI_PAIRS {
        let slot = r.slot(key).unwrap_or_else(|| panic!("`{key}` が無い"));
        assert_eq!(slot.value, want, "`{key}` の実効値");
        assert_eq!(slot.source, Source::Cli, "`{key}` の出所");
        assert_eq!(slot.detail, "", "`{key}` の detail は cli では空");
        assert_eq!(slot.origin_label(), "cli", "`{key}` の出所表記");
        assert_eq!(r.source(key), Some(Source::Cli));
    }
}

#[test]
fn env_layer_wins_over_file_for_every_key() {
    let file = file_all();
    let env = env_of(&ENV_PAIRS);
    let r = resolve(&[], &env, Some(&file)).expect("解決");

    for (key, (name, want)) in KEYS.iter().zip(ENV_PAIRS) {
        let key: &str = key;
        assert_eq!(env_var(key), Some(name), "`{key}` の env 名");
        let slot = r.slot(key).unwrap_or_else(|| panic!("`{key}` が無い"));
        assert_eq!(slot.value, want, "`{key}` の実効値");
        assert_eq!(slot.source, Source::Env, "`{key}` の出所");
        assert_eq!(slot.detail, name, "`{key}` の detail は env 名");
        assert_eq!(slot.origin_label(), format!("env {name}"));
    }
}

#[test]
fn file_layer_wins_over_default_for_every_key() {
    let file = file_all();
    let r = resolve(&[], &BTreeMap::new(), Some(&file)).expect("解決");

    for (key, want) in FILE_PAIRS {
        let slot = r.slot(key).unwrap_or_else(|| panic!("`{key}` が無い"));
        assert_eq!(slot.value, want, "`{key}` の実効値");
        assert_eq!(slot.source, Source::Config, "`{key}` の出所");
        assert_eq!(slot.detail, "c.toml", "`{key}` の detail はファイルの出所");
        assert_eq!(slot.origin_label(), "config c.toml");
    }
}

#[test]
fn default_layer_covers_exactly_the_documented_keys() {
    let r = resolve(&[], &BTreeMap::new(), None).expect("解決");

    for key in KEYS {
        match default_value(key) {
            Some(d) => {
                let slot = r.slot(key).unwrap_or_else(|| panic!("`{key}` が無い"));
                assert_eq!(slot.value, d, "`{key}` の既定値");
                assert_eq!(slot.source, Source::Default, "`{key}` の出所");
                assert_eq!(slot.detail, "", "`{key}` の detail は default では空");
                assert_eq!(slot.origin_label(), "default");
            }
            None => {
                assert!(r.slot(key).is_none(), "`{key}` は未設定のはず");
                assert_eq!(r.source(key), None);
                assert_eq!(r.string(key), None);
            }
        }
    }
}

#[test]
fn dropping_one_layer_at_a_time_walks_the_whole_ladder_per_key() {
    // 全キーで cli -> env -> config -> default と 1 段ずつ落とし、
    // 実効値と出所が同時に切り替わることを見る。
    let file = file_all();
    let env = env_of(&ENV_PAIRS);
    let cli = cli_of(&CLI_PAIRS);

    let r = resolve(&cli, &env, Some(&file)).expect("cli");
    for key in KEYS {
        assert_eq!(r.source(key), Some(Source::Cli), "`{key}`: {r:?}");
    }

    let r = resolve(&[], &env, Some(&file)).expect("env");
    for key in KEYS {
        assert_eq!(r.source(key), Some(Source::Env), "`{key}`: {r:?}");
    }

    let r = resolve(&[], &BTreeMap::new(), Some(&file)).expect("config");
    for key in KEYS {
        assert_eq!(r.source(key), Some(Source::Config), "`{key}`: {r:?}");
    }

    let r = resolve(&[], &BTreeMap::new(), None).expect("default");
    for key in KEYS {
        let want = if default_value(key).is_some() {
            Some(Source::Default)
        } else {
            None
        };
        assert_eq!(r.source(key), want, "`{key}`: {r:?}");
    }
}

#[test]
fn cli_value_matching_env_still_reports_cli() {
    // 値が同じでも出所は CLI (診断は「食い違い」だけを見る)。
    let env = env_of(&[("DENNO_FLASH_PORT", "9003")]);
    let cli = vec![("port", "9003".to_string())];
    let r = resolve(&cli, &env, None).expect("解決");
    assert_eq!(r.string("port").as_deref(), Some("9003"));
    assert_eq!(r.source("port"), Some(Source::Cli));
    assert_eq!(r.slot("port").unwrap().detail, "");
}

// ------------------------------------------------ env の空値 / 未知の名前の扱い

#[test]
fn blank_env_falls_through_to_file_then_default_per_key() {
    let file = file_all();
    for key in KEYS {
        let name = env_var(key).expect("env 名");
        let env = env_of(&[(name, "   ")]);
        let r = resolve(&[], &env, Some(&file)).expect("解決");
        assert_eq!(r.source(key), Some(Source::Config), "`{key}`");
        assert_eq!(r.slot(key).unwrap().value, {
            FILE_PAIRS
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| *v)
                .expect("file 値")
                .to_string()
        });

        let r = resolve(&[], &env, None).expect("解決");
        match default_value(key) {
            Some(d) => {
                assert_eq!(r.source(key), Some(Source::Default), "`{key}`");
                assert_eq!(r.slot(key).unwrap().value, d, "`{key}`");
            }
            None => assert_eq!(r.source(key), None, "`{key}` は未設定"),
        }
    }
}

#[test]
fn unknown_env_and_cli_names_are_ignored() {
    let env = env_of(&[
        ("DENNO_FLASH_PORT_X", "1"),
        ("denno_flash_port", "9009"),
        ("DENNO_NOT_A_KEY", "x"),
    ]);
    let cli = vec![("not_a_key", "x".to_string())];
    let r = resolve(&cli, &env, None).expect("解決");
    assert_eq!(r.string("port").as_deref(), Some("8098"));
    assert_eq!(r.source("port"), Some(Source::Default));
    assert!(r.slot("not_a_key").is_none());
}

#[test]
fn first_cli_occurrence_wins_for_a_repeated_key() {
    let cli = vec![("port", "9100".to_string()), ("port", "9200".to_string())];
    let r = resolve(&cli, &BTreeMap::new(), None).expect("解決");
    assert_eq!(r.string("port").as_deref(), Some("9100"));
}

#[test]
fn whitespace_env_numbers_are_trimmed_but_string_env_values_are_verbatim() {
    let env = env_of(&[("DENNO_FLASH_PORT", "  9002  "), ("DENNO_FLASH_HOST", " h ")]);
    let r = resolve(&[], &env, None).expect("解決");
    assert_eq!(r.u16("port"), Ok(Some(9002)));
    assert_eq!(r.string("host").as_deref(), Some(" h "));
}

// -------------------------------------------------------- 整数キーの境界 (file)

fn file_err(body: &str) -> String {
    parse_str(body, "b.toml").expect_err("弾かれるはず")
}

#[test]
fn port_accepts_both_ends_of_the_range_in_a_file() {
    for v in ["0", "65535"] {
        let f = parse_str(&format!("port = {v}\n"), "b.toml").expect("範囲内");
        assert_eq!(f.get("port"), Some(&Value::Int(v.parse().unwrap())), "{v}");
        let r = resolve(&[], &BTreeMap::new(), Some(&f)).expect("解決");
        assert_eq!(r.u16("port"), Ok(Some(v.parse::<u16>().unwrap())));
    }
}

#[test]
fn port_rejects_one_past_each_end_in_a_file() {
    let e = file_err("port = 65536\n");
    assert!(e.contains("`port` の範囲は 0..=65535"), "{e}");
    assert!(e.contains("65536"), "{e}");

    let e = file_err("port = -1\n");
    assert!(e.contains("`port` は 0 以上 65535 以下"), "{e}");
    assert!(e.contains("-1"), "{e}");
}

#[test]
fn max_ctx_and_max_tokens_share_the_u32_range_in_a_file() {
    for key in ["max_ctx", "max_tokens"] {
        for v in ["1", "4294967295"] {
            parse_str(&format!("{key} = {v}\n"), "b.toml")
                .unwrap_or_else(|e| panic!("`{key} = {v}` が弾かれた: {e}"));
        }
        let e = file_err(&format!("{key} = 0\n"));
        assert!(e.contains("の範囲は 1..=4294967295"), "{key}: {e}");
        assert!(e.contains("指定値 0"), "{key}: {e}");

        let e = file_err(&format!("{key} = 4294967296\n"));
        assert!(e.contains("の範囲は 1..=4294967295"), "{key}: {e}");
        assert!(e.contains("4294967296"), "{key}: {e}");
    }
}

#[test]
fn kv_slots_accepts_zero_through_eight_and_rejects_neighbours() {
    for v in ["0", "8"] {
        parse_str(&format!("kv_slots = {v}\n"), "b.toml")
            .unwrap_or_else(|e| panic!("`kv_slots = {v}` が弾かれた: {e}"));
    }
    let e = file_err("kv_slots = 9\n");
    assert!(e.contains("`kv_slots` の範囲は 0..=8"), "{e}");

    let e = file_err("kv_slots = -1\n");
    assert!(e.contains("`kv_slots` は 0 以上 8 以下"), "{e}");
}

// --------------------------------------------------------- 整数キーの境界 (env)

#[test]
fn env_accepts_both_ends_of_every_int_range() {
    for (key, min, max) in INT_KEYS {
        let name = env_var(key).expect("env 名");
        for v in [min, max] {
            let text = v.to_string();
            let env = env_of(&[(name, text.as_str())]);
            let r = resolve(&[], &env, None)
                .unwrap_or_else(|e| panic!("`{name}={v}` が弾かれた: {e}"));
            assert_eq!(r.number(key), Ok(Some(v)), "`{key}`={v}");
            assert_eq!(r.source(key), Some(Source::Env));
        }
    }
}

#[test]
fn env_rejects_one_past_each_int_range_and_names_the_origin() {
    // 下限の 1 個下 (0 が下限のキーは負値で「整数として読めない」)
    for (key, min, max) in INT_KEYS {
        let name = env_var(key).expect("env 名");
        let below = if min == 0 {
            "-1".to_string()
        } else {
            (min - 1).to_string()
        };
        let env = env_of(&[(name, &below)]);
        let e = resolve(&[], &env, None).expect_err("範囲外");
        assert!(e.contains(name), "`{key}`: {e}");
        assert!(e.contains(&format!("{min}..={max}")), "`{key}`: {e}");

        let above = (max + 1).to_string();
        let env = env_of(&[(name, &above)]);
        let e = resolve(&[], &env, None).expect_err("範囲外");
        assert!(e.contains(name), "`{key}`: {e}");
        assert!(e.contains(&above), "`{key}`: {e}");
        assert!(e.contains(&format!("{min}..={max}")), "`{key}`: {e}");
    }
}

#[test]
fn non_numeric_env_for_an_int_key_names_the_variable_and_range() {
    for (key, min, max) in INT_KEYS {
        let name = env_var(key).expect("env 名");
        let env = env_of(&[(name, "http")]);
        let e = resolve(&[], &env, None).expect_err("整数でない");
        assert!(e.contains(name), "`{key}`: {e}");
        assert!(e.contains("http"), "`{key}`: {e}");
        assert!(e.contains(&format!("{min}..={max}")), "`{key}`: {e}");
    }
}

// ---------------------------------------------- 整数キーの境界 (cli) と優先順位

#[test]
fn cli_boundaries_are_validated_even_when_env_is_valid() {
    for (key, min, max) in INT_KEYS {
        let name = env_var(key).expect("env 名");
        // 下限・上限は通る
        for v in [min, max] {
            let env = env_of(&[(name, "5")]);
            let cli = vec![(leak(key), v.to_string())];
            let r = resolve(&cli, &env, None).expect("範囲内の CLI");
            assert_eq!(r.number(key), Ok(Some(v)), "`{key}`={v}");
            assert_eq!(r.source(key), Some(Source::Cli));
        }
        // 1 個外れた CLI は env が有効でも失敗する (CLI が勝つため)
        let bad = (max + 1).to_string();
        let env = env_of(&[(name, &min.to_string())]);
        let cli = vec![(leak(key), bad.clone())];
        let e = resolve(&cli, &env, None).expect_err("CLI が範囲外");
        assert!(e.contains(&bad), "`{key}`: {e}");
        assert!(e.contains(&format!("{min}..={max}")), "`{key}`: {e}");
    }
}

#[test]
fn env_boundary_failure_is_not_masked_by_a_file_value_in_range() {
    for (key, min, max) in INT_KEYS {
        let name = env_var(key).expect("env 名");
        let f = parse_str(&format!("{key} = {min}\n"), "b.toml").expect("ファイルは範囲内");
        let env = env_of(&[(name, &(max + 1).to_string())]);
        let e = resolve(&[], &env, Some(&f)).expect_err("env が範囲外");
        assert!(e.contains(name), "`{key}`: {e}");
        assert!(e.contains(&format!("{min}..={max}")), "`{key}`: {e}");
    }
}

// ----------------------------------------------- 設定ファイルの誤り (行番号付き)

#[test]
fn a_line_without_equals_is_rejected_with_its_line_number() {
    let e = file_err("# header\ndir = \"/ok\"\nport 9001\n");
    assert!(e.contains("b.toml:3"), "{e}");
    assert!(e.contains("`key = value`"), "{e}");
}

#[test]
fn an_empty_key_is_rejected() {
    let e = file_err("port = 1\n = 2\n");
    assert!(e.contains("b.toml:2"), "{e}");
    assert!(e.contains("キーが空"), "{e}");
}

#[test]
fn keys_with_spaces_dots_or_unicode_are_rejected() {
    for (line, needle) in [
        ("max ctx = 1\n", "max ctx"),
        ("max.ctx = 1\n", "max.ctx"),
        ("ｍａｘ = 1\n", "ｍａｘ"),
    ] {
        let e = file_err(line);
        assert!(e.contains("b.toml:1"), "{line}: {e}");
        assert!(e.contains(needle), "{line}: {e}");
    }
}

#[test]
fn unterminated_strings_are_rejected_per_quote_style() {
    let e = file_err("host = \"open\n");
    assert!(e.contains("文字列の閉じ `\"` が無い"), "{e}");
    assert!(e.contains("b.toml:1"), "{e}");

    let e = file_err("host = 'open\n");
    assert!(e.contains("文字列の閉じ `'` が無い"), "{e}");
}

#[test]
fn unsupported_escape_lists_the_accepted_set() {
    let e = file_err("host = \"a\\qb\"\n");
    assert!(e.contains("未対応のエスケープ `\\q`"), "{e}");
    assert!(e.contains("\\n \\t \\r \\\\ \\\""), "{e}");
}

#[test]
fn an_escape_cut_at_end_of_line_is_reported() {
    let e = file_err("host = \"a\\\n");
    assert!(e.contains("エスケープが行末で切れている"), "{e}");
}

#[test]
fn trailing_garbage_after_a_value_is_rejected() {
    let e = file_err("port = 1 2\n");
    assert!(e.to_lowercase().contains("未対応") || e.contains("余分な文字"), "{e}");

    let e = file_err("host = \"a\" b\n");
    assert!(e.contains("余分な文字"), "{e}");
}

#[test]
fn a_missing_value_is_rejected() {
    let e = file_err("host =\n");
    assert!(e.contains("値が無い"), "{e}");
}

#[test]
fn hex_floats_and_doubled_signs_are_not_integers() {
    for body in ["port = 0x10\n", "port = 1.0\n", "port = --1\n", "port = 1_000\n"] {
        let e = file_err(body);
        assert!(e.contains("b.toml:1"), "{body}: {e}");
    }
}

#[test]
fn floats_and_hex_are_reported_as_unsupported_values() {
    for (body, needle) in [("max_ctx = 2048.5\n", "2048.5"), ("kv_slots = 0x2\n", "0x2")] {
        let e = file_err(body);
        assert!(e.contains("未対応"), "{body}: {e}");
        assert!(e.contains(needle), "{body}: {e}");
    }
}

#[test]
fn a_section_header_is_rejected_with_its_line_number() {
    let e = file_err("dir = \"/m\"\n\n[serve]\nport = 1\n");
    assert!(e.contains("b.toml:3"), "{e}");
    assert!(e.contains("セクション"), "{e}");
}

#[test]
fn a_duplicate_key_reports_the_first_definition_line() {
    let e = file_err("port = 1\nhost = \"h\"\nport = 2\n");
    assert!(e.contains("b.toml:3"), "{e}");
    assert!(e.contains("重複"), "{e}");
    assert!(e.contains("1 行目"), "{e}");
}

#[test]
fn an_unknown_key_suggests_a_near_miss_but_not_a_far_one() {
    let e = file_err("por = 9001\n");
    assert!(e.contains("未知のキー `por`"), "{e}");
    assert!(e.contains("もしかして: port"), "{e}");

    let e = file_err("zzz = 1\n");
    assert!(e.contains("未知のキー `zzz`"), "{e}");
    assert!(!e.contains("もしかして"), "{e}");
    assert!(e.contains("dir, host, port, max_ctx, name"), "{e}");
}

#[test]
fn int_keys_reject_quoted_numbers_but_accept_a_bare_comment() {
    let e = file_err("port = \"9001\"\n");
    assert!(e.contains("`port` は整数 0..=65535"), "{e}");
    assert!(e.contains("9001"), "{e}");

    let f = parse_str("port = 9001 # コメント\n", "b.toml").expect("行末コメントは通る");
    assert_eq!(f.get("port"), Some(&Value::Int(9001)));
}

#[test]
fn string_keys_accept_bare_integers_and_negative_numbers_as_text() {
    let f = parse_str("host = 8080\ndir = -5\nvram_budget = 12000\n", "b.toml")
        .expect("文字列キーは整数も文字列として受ける");
    assert_eq!(f.get("host"), Some(&Value::Str("8080".into())));
    assert_eq!(f.get("dir"), Some(&Value::Str("-5".into())));
    assert_eq!(f.get("vram_budget"), Some(&Value::Str("12000".into())));

    let r = resolve(&[], &BTreeMap::new(), Some(&f)).expect("解決");
    assert_eq!(r.string("host").as_deref(), Some("8080"));
    assert_eq!(r.source("host"), Some(Source::Config));
}

#[test]
fn an_empty_file_is_an_empty_config() {
    let f = parse_str("# only a comment\n\n   \n", "b.toml").expect("空設定");
    assert!(f.entries.is_empty(), "{:?}", f.entries);
    assert_eq!(f.origin, "b.toml");
    let r = resolve(&[], &BTreeMap::new(), Some(&f)).expect("解決");
    assert_eq!(r.source("port"), Some(Source::Default));
    assert_eq!(r.source("dir"), None);
}

// ------------------------------------------------- 型変換 (number / u16 / usize)

#[test]
fn number_returns_none_for_unset_and_errors_for_string_keys() {
    let r = resolve(&[], &BTreeMap::new(), None).expect("解決");
    assert_eq!(r.number("kv_slots"), Ok(None));
    assert_eq!(r.usize("kv_slots"), Ok(None));
    assert_eq!(r.u16("dir"), Ok(None));
    assert_eq!(r.number("does_not_exist"), Ok(None));

    let e = r.number("host").expect_err("文字列キーは整数でない");
    assert!(e.contains("`host` は整数ではない"), "{e}");
}

#[test]
fn u16_and_usize_agree_with_number_for_every_int_key() {
    let r = resolve(&[], &env_of(&ENV_PAIRS), None).expect("解決");
    for (key, _, _) in INT_KEYS {
        let n = r.number(key).expect("整数").expect("設定済み");
        assert_eq!(r.u16(key), Ok(Some(n as u16)), "`{key}` u16");
        assert_eq!(r.usize(key), Ok(Some(n as usize)), "`{key}` usize");
    }
    assert_eq!(r.u16("port"), Ok(Some(9002)));
    assert_eq!(r.usize("kv_slots"), Ok(Some(2)));
}

// ------------------------------------------------- render と summary_json の出所

#[test]
fn render_marks_the_source_of_every_key() {
    let file = file_all();
    let env = env_of(&ENV_PAIRS);
    let cli = cli_of(&CLI_PAIRS);
    let r = resolve(&cli, &env, Some(&file)).expect("解決");
    let text = r.render();
    for key in KEYS {
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("{key} =")))
            .unwrap_or_else(|| panic!("`{key}` の行が無い: {text}"));
        assert!(line.ends_with("# source: cli"), "`{key}`: {line}");
    }
    assert!(!text.contains("<unset>"), "全キー設定済み: {text}");
}

#[test]
fn render_round_trips_to_the_same_values_as_config_source() {
    let r = resolve(&cli_of(&CLI_PAIRS), &BTreeMap::new(), None).expect("解決");
    let text = r.render();
    let f = parse_str(&text, "roundtrip.toml").expect("render 出力は設定ファイルとして読める");
    let r2 = resolve(&[], &BTreeMap::new(), Some(&f)).expect("解決");
    for key in KEYS {
        assert_eq!(r2.string(key), r.string(key), "`{key}` の値が往復で変わる");
        assert_eq!(r2.source(key), Some(Source::Config), "`{key}` の出所");
        assert_eq!(r2.slot(key).unwrap().detail, "roundtrip.toml");
    }
}

#[test]
fn summary_json_reports_the_same_source_and_value_as_the_slot() {
    let file = file_all();
    let env = env_of(&ENV_PAIRS);
    let r = resolve(&[], &env, Some(&file)).expect("解決");
    let json = r.summary_json();
    let obj = json.as_object().expect("object");
    assert_eq!(obj.len(), KEYS.len());

    for key in KEYS {
        let entry = obj.get(key).unwrap_or_else(|| panic!("`{key}` が無い"));
        let slot = r.slot(key).expect("設定済み");
        assert_eq!(entry["set"], true, "`{key}`");
        assert_eq!(entry["source"].as_str(), Some(slot.source.as_str()), "`{key}`");
        assert_eq!(entry["detail"].as_str(), Some(slot.detail.as_str()), "`{key}`");
        assert_eq!(
            entry["origin"].as_str().unwrap_or_else(|| panic!("`{key}` origin")),
            slot.origin_label()
        );
        let is_int = INT_KEYS.iter().any(|(k, _, _)| *k == key);
        if is_int {
            assert_eq!(
                entry["value"].as_u64(),
                slot.value.trim().parse::<u64>().ok(),
                "`{key}` の値 (整数キーは JSON 数値)"
            );
        } else {
            assert_eq!(
                entry["value"].as_str(),
                Some(slot.value.as_str()),
                "`{key}` の値 (文字列キー)"
            );
        }
    }
}

#[test]
fn summary_json_marks_unset_keys_as_unset() {
    let r = resolve(&[], &BTreeMap::new(), None).expect("解決");
    let json = r.summary_json();
    for key in KEYS {
        let entry = &json[key];
        match default_value(key) {
            None => {
                assert_eq!(entry["set"], false, "`{key}`");
                assert_eq!(entry["source"].as_str(), Some("unset"), "`{key}`");
                assert!(entry.get("value").is_none(), "`{key}` に value がある");
            }
            Some(_) => assert_eq!(entry["source"].as_str(), Some("default"), "`{key}`"),
        }
    }
}

#[test]
fn kv_slots_zero_is_annotated_in_render_but_keeps_its_source() {
    let cli = vec![("kv_slots", "0".to_string())];
    let r = resolve(&cli, &BTreeMap::new(), None).expect("解決");
    assert_eq!(r.source("kv_slots"), Some(Source::Cli));
    let line = r
        .render()
        .lines()
        .find(|l| l.starts_with("kv_slots = 0"))
        .expect("kv_slots の行")
        .to_string();
    assert!(line.contains("# source: cli"), "{line}");
    assert!(line.contains("0 = 未指定"), "{line}");
}

// ------------------------------------------------------------ Source の表記

#[test]
fn source_and_level_strings_are_stable() {
    assert_eq!(Source::Cli.as_str(), "cli");
    assert_eq!(Source::Env.as_str(), "env");
    assert_eq!(Source::Config.as_str(), "config");
    assert_eq!(Source::Default.as_str(), "default");
}

#[test]
fn config_detail_is_the_origin_verbatim() {
    let f = parse_str("host = \"h\"\n", "some/odd path.toml").expect("設定");
    let r = resolve(&[], &BTreeMap::new(), Some(&f)).expect("解決");
    let slot = r.slot("host").expect("host");
    assert_eq!(slot.detail, "some/odd path.toml");
    assert_eq!(slot.origin_label(), "config some/odd path.toml");
}
