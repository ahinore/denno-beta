//! task 1066: `src/config.rs` の `DENNO_*` 環境変数診断 (`diagnose_env` /
//! `diagnose`) の契約。
//!
//! 見ているのは次の 4 点:
//!
//! 1. **読まれない名前**を拾う — 未知の `DENNO_*` (綴り違いは提案つき)、
//!    大小の取り違え、`DENNO_` 接頭辞の欠け、非推奨名。
//! 2. **矛盾**を拾う — CLI が env を、env が設定ファイルを黙って上書きする
//!    ケースと、実効値どうしの矛盾 (`max_tokens > max_ctx`)。
//! 3. **値の誤り**を拾う — 整数キーの型・範囲外、空値、
//!    `DENNO_KV_SLOTS=0` (未指定の旧来の書き方)、`DENNO_AUTO_CONFIG` の
//!    真偽値以外。
//! 4. **秘密を漏らさない** — `DENNO_API_KEY` などの値は診断のどこにも出ない
//!    (名前だけ出す)。
//!
//! denno-cli は bin クレートなので lib 経由で取り込めない。既存の
//! `config_906.rs` / `config_edges_1001.rs` と同じく `#[path]` で
//! `src/config.rs` をそのまま読み込む。プロセスの env は触らない
//! (診断関数は env を引数で受ける純関数)。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;

use config::diag::{diagnose, diagnose_env, known_env_vars, Level, DEPRECATED_ENV};
use config::parse_str;

fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn codes(diags: &[config::diag::Diag]) -> Vec<&'static str> {
    diags.iter().map(|d| d.code).collect()
}

fn by_code<'a>(diags: &'a [config::diag::Diag], code: &str) -> Option<&'a config::diag::Diag> {
    diags.iter().find(|d| d.code == code)
}

fn all_text(diags: &[config::diag::Diag]) -> String {
    diags
        .iter()
        .map(|d| format!("{} {}\n", d.vars.join(","), d.message))
        .collect()
}

// ------------------------------------------------------------ 既知の名前の集合

#[test]
fn known_env_vars_are_the_spec_ones_sorted_and_unique() {
    let known = known_env_vars();
    assert_eq!(
        known,
        vec![
            "DENNO_AUTO_CONFIG",
            "DENNO_FLASH_DIR",
            "DENNO_FLASH_HOST",
            "DENNO_FLASH_MAX_CTX",
            "DENNO_FLASH_MAX_TOKENS",
            "DENNO_FLASH_NAME",
            "DENNO_FLASH_PORT",
            "DENNO_KV_SLOTS",
            "DENNO_MTP_PATH",
            "DENNO_STEER_RECIPE",
        ],
        "{known:?}"
    );
    let mut sorted = known.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, known);
    let mut deduped = known.clone();
    deduped.dedup();
    assert_eq!(deduped.len(), known.len());
}

#[test]
fn clean_env_produces_no_diagnostics() {
    let env = env_of(&[
        ("DENNO_FLASH_PORT", "8098"),
        ("DENNO_FLASH_HOST", "127.0.0.1"),
        ("DENNO_KV_SLOTS", "2"),
        ("PATH", "/usr/bin"),
    ]);
    assert_eq!(diagnose_env(&env), Vec::new());
    assert_eq!(diagnose(&[], &env, None), Vec::new());
}

// --------------------------------------------------- 未知の名前 / 読まれない名前

#[test]
fn unknown_denno_var_suggests_the_near_name() {
    let env = env_of(&[("DENNO_FLASH_PROT", "8098")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_unknown").expect("env_unknown");
    assert_eq!(x.level, Level::Warn);
    assert_eq!(x.vars, vec!["DENNO_FLASH_PROT".to_string()]);
    assert!(x.message.contains("もしかして"), "{}", x.message);
    assert!(x.message.contains("DENNO_FLASH_PORT"), "{}", x.message);
    // 値は載せない (typo 名に紐づく値は信用できないため)
    assert!(!x.message.contains("8098"), "{}", x.message);
}

#[test]
fn unknown_denno_var_without_near_name_has_no_suggestion() {
    let env = env_of(&[("DENNO_TOTALLY_UNRELATED_THING", "1")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_unknown").expect("env_unknown");
    assert!(!x.message.contains("もしかして"), "{}", x.message);
}

#[test]
fn lowercase_name_is_reported_as_case_mismatch_not_unknown() {
    let env = env_of(&[("denno_flash_port", "9000")]);
    let d = diagnose_env(&env);
    assert_eq!(codes(&d), vec!["env_case"], "{d:?}");
    let x = &d[0];
    assert!(x.message.contains("DENNO_FLASH_PORT"), "{}", x.message);
    // 大小の取り違えは解決では未設定なので、範囲検査などの Error は出ない
    assert!(d.iter().all(|d| d.level == Level::Warn));
}

#[test]
fn missing_denno_prefix_is_reported() {
    let env = env_of(&[("ENNO_FLASH_PORT", "9000")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_unprefixed").expect("env_unprefixed");
    assert!(x.message.contains("DENNO_FLASH_PORT"), "{}", x.message);
}

#[test]
fn unrelated_env_does_not_trigger_unprefixed_noise() {
    // `DENNO_` で始まらず、既知名とも遠い → 何も出さない
    let env = env_of(&[("HOME", "/root"), ("LANG", "C.UTF-8"), ("TERM", "xterm")]);
    assert_eq!(diagnose_env(&env), Vec::new());
}

#[test]
fn deprecated_name_points_at_the_successor() {
    let env = env_of(&[("DENNO_MAX_CTX", "8192")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_deprecated").expect("env_deprecated");
    assert_eq!(x.vars, vec!["DENNO_MAX_CTX".to_string()]);
    assert!(x.message.contains("DENNO_FLASH_MAX_CTX"), "{}", x.message);
    // 非推奨名は「未知」として二重に数えない
    assert!(by_code(&d, "env_unknown").is_none(), "{d:?}");
}

#[test]
fn deprecated_table_entries_are_all_real_successors() {
    let known = known_env_vars();
    for (old, new) in DEPRECATED_ENV {
        assert!(!known.contains(&old), "{old} が現行名にも入っている");
        assert!(known.contains(&new), "{new} が現行名に無い");
    }
}

// --------------------------------------------------------- 値の誤り (env 単体)

#[test]
fn blank_known_value_is_unset_not_an_error() {
    let env = env_of(&[("DENNO_FLASH_HOST", "   ")]);
    let d = diagnose_env(&env);
    assert_eq!(codes(&d), vec!["env_empty"], "{d:?}");
    assert!(d.iter().all(|d| d.level == Level::Warn));
}

#[test]
fn non_integer_and_out_of_range_are_errors() {
    let env = env_of(&[("DENNO_FLASH_PORT", "http")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_not_integer").expect("env_not_integer");
    assert_eq!(x.level, Level::Error);
    assert!(x.message.contains("http"), "{}", x.message);
    assert!(x.message.contains("0..=65535"), "{}", x.message);

    let env = env_of(&[("DENNO_FLASH_PORT", "70000")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "env_out_of_range").expect("env_out_of_range");
    assert_eq!(x.level, Level::Error);
    assert!(x.message.contains("70000"), "{}", x.message);
    assert!(x.message.contains("0..=65535"), "{}", x.message);

    // 負値も整数として読めない側
    let env = env_of(&[("DENNO_KV_SLOTS", "-1")]);
    assert!(by_code(&diagnose_env(&env), "env_not_integer").is_some());
}

#[test]
fn kv_slots_zero_is_the_legacy_spelling_of_unset() {
    let env = env_of(&[("DENNO_KV_SLOTS", "0")]);
    let d = diagnose_env(&env);
    let x = by_code(&d, "kv_slots_legacy_zero").expect("kv_slots_legacy_zero");
    assert_eq!(x.level, Level::Warn);
    assert!(x.message.contains("未指定"), "{}", x.message);
    // 0 は範囲内なので Error にはしない
    assert!(by_code(&d, "env_out_of_range").is_none(), "{d:?}");

    // 1..=8 は静か
    assert_eq!(diagnose_env(&env_of(&[("DENNO_KV_SLOTS", "3")])), Vec::new());
    assert_eq!(diagnose_env(&env_of(&[("DENNO_KV_SLOTS", "8")])), Vec::new());
}

#[test]
fn auto_config_accepts_only_boolean_spellings() {
    for ok in ["1", "true", "ON", " yes ", "auto", "0", "false", "off", "no"] {
        let d = diagnose_env(&env_of(&[("DENNO_AUTO_CONFIG", ok)]));
        assert!(
            by_code(&d, "auto_config_spelling").is_none(),
            "{ok} で警告が出た: {d:?}"
        );
    }
    // 真偽値でない値は黙って「無効」になるので警告する
    for bad in ["2", "enable", "12000"] {
        let d = diagnose_env(&env_of(&[("DENNO_AUTO_CONFIG", bad)]));
        let x = by_code(&d, "auto_config_spelling")
            .unwrap_or_else(|| panic!("{bad} で警告が出ない: {d:?}"));
        assert_eq!(x.level, Level::Warn);
        assert!(x.message.contains("vram-budget"), "{}", x.message);
    }
}

// ------------------------------------------------------------- 秘密を漏らさない

#[test]
fn secret_values_never_appear_in_diagnostics() {
    let env = env_of(&[
        ("DENNO_API_KEY", "sk-live-DO-NOT-PRINT"),
        ("DENNO_HF_TOKEN", "hf_ALSO-SECRET"),
        ("DENNO_SOMETHING_PASSWORD", "pw-1234"),
        ("DENNO_FLASH_PORT", "70000"),
    ]);
    let d = diagnose_env(&env);
    let text = all_text(&d);
    for secret in ["sk-live-DO-NOT-PRINT", "hf_ALSO-SECRET", "pw-1234"] {
        assert!(!text.contains(secret), "秘密が漏れた: {text}");
    }
    // 名前は出る (何が設定されているかは分かる)
    assert!(text.contains("DENNO_API_KEY"), "{text}");
    // 値の誤りは従来どおり出る
    assert!(by_code(&d, "env_out_of_range").is_some(), "{d:?}");
}

#[test]
fn secret_name_detection_matches_doctor_convention() {
    for name in [
        "DENNO_API_KEY",
        "DENNO_API_KEYS",
        "DENNO_ADMIN_TOKEN",
        "DENNO_CLIENT_SECRET",
        "DENNO_DB_PASSWORD",
        "denno_api_key",
    ] {
        assert!(config::diag::is_secret_name(name), "{name}");
    }
    for name in ["DENNO_FLASH_PORT", "DENNO_FLASH_DIR", "DENNO_MTP_PATH"] {
        assert!(!config::diag::is_secret_name(name), "{name}");
    }
}

// ------------------------------------- 出所の食い違い (CLI / env / 設定ファイル)

#[test]
fn cli_shadowing_env_is_reported() {
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let cli = vec![("port", "9003".to_string())];
    let d = diagnose(&cli, &env, None);
    let x = by_code(&d, "env_shadowed_by_cli").expect("env_shadowed_by_cli");
    assert_eq!(x.level, Level::Warn);
    assert_eq!(x.vars, vec!["DENNO_FLASH_PORT".to_string()]);
    assert!(x.message.contains("9002"), "{}", x.message);
    assert!(x.message.contains("9003"), "{}", x.message);
    // 解決は成功し、実効値は CLI
    assert!(by_code(&d, "resolve_error").is_none(), "{d:?}");
}

#[test]
fn same_value_in_cli_and_env_is_not_a_conflict() {
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let cli = vec![("port", "9002".to_string())];
    assert_eq!(diagnose(&cli, &env, None), Vec::new());
}

#[test]
fn env_shadowing_config_file_is_reported() {
    let file = parse_str("port = 9001\nhost = \"cfg-host\"\n", "c.toml").unwrap();
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let d = diagnose(&[], &env, Some(&file));
    let x = by_code(&d, "config_shadowed_by_env").expect("config_shadowed_by_env");
    assert!(x.message.contains("9001"), "{}", x.message);
    assert!(x.message.contains("9002"), "{}", x.message);
    assert!(x.message.contains("c.toml"), "{}", x.message);
    // 一致しているキー (host は env 未設定) は黙っている
    assert_eq!(
        d.iter()
            .filter(|d| d.code == "config_shadowed_by_env")
            .count(),
        1,
        "{d:?}"
    );
}

#[test]
fn cli_shadowing_config_file_is_reported() {
    let file = parse_str("name = 'cfg-name'\n", "c.toml").unwrap();
    let cli = vec![("name", "cli-name".to_string())];
    let d = diagnose(&cli, &BTreeMap::new(), Some(&file));
    let x = by_code(&d, "config_shadowed_by_cli").expect("config_shadowed_by_cli");
    assert!(x.message.contains("cfg-name"), "{}", x.message);
    assert!(x.message.contains("cli-name"), "{}", x.message);
    assert!(x.message.contains("c.toml"), "{}", x.message);
}

#[test]
fn cli_over_env_over_file_reports_both_edges() {
    // 3 つすべてが違い、env が真ん中 → CLI>env と env>file の 2 件
    let file = parse_str("port = 9001\n", "c.toml").unwrap();
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let cli = vec![("port", "9003".to_string())];
    let d = diagnose(&cli, &env, Some(&file));
    assert!(by_code(&d, "env_shadowed_by_cli").is_some(), "{d:?}");
    assert!(by_code(&d, "config_shadowed_by_env").is_some(), "{d:?}");
    assert!(by_code(&d, "config_shadowed_by_cli").is_none(), "{d:?}");
}

// ------------------------------------------------------ 実効値どうしの矛盾

#[test]
fn max_tokens_over_ctx_is_reported_from_env() {
    let env = env_of(&[
        ("DENNO_FLASH_MAX_TOKENS", "8192"),
        ("DENNO_FLASH_MAX_CTX", "4096"),
    ]);
    let d = diagnose(&[], &env, None);
    let x = by_code(&d, "max_tokens_over_ctx").expect("max_tokens_over_ctx");
    assert_eq!(x.level, Level::Warn);
    assert!(x.message.contains("8192"), "{}", x.message);
    assert!(x.message.contains("4096"), "{}", x.message);
}

#[test]
fn max_tokens_equal_to_ctx_is_fine() {
    let env = env_of(&[
        ("DENNO_FLASH_MAX_TOKENS", "4096"),
        ("DENNO_FLASH_MAX_CTX", "4096"),
    ]);
    let d = diagnose(&[], &env, None);
    assert!(by_code(&d, "max_tokens_over_ctx").is_none(), "{d:?}");
}

#[test]
fn max_tokens_over_ctx_is_reported_from_mixed_sources() {
    // 既定 max_ctx=4096 に対し、設定ファイルで max_tokens=9999
    let file = parse_str("max_tokens = 9999\n", "c.toml").unwrap();
    let d = diagnose(&[], &BTreeMap::new(), Some(&file));
    let x = by_code(&d, "max_tokens_over_ctx").expect("max_tokens_over_ctx");
    assert!(x.message.contains("9999"), "{}", x.message);
    assert!(x.message.contains("4096"), "{}", x.message);
}

#[test]
fn resolve_failure_is_surfaced_once() {
    // 範囲外の env は Error 診断 + 解決失敗の 2 件になる
    let env = env_of(&[("DENNO_FLASH_PORT", "70000")]);
    let d = diagnose(&[], &env, None);
    assert!(by_code(&d, "env_out_of_range").is_some(), "{d:?}");
    let x = by_code(&d, "resolve_error").expect("resolve_error");
    assert_eq!(x.level, Level::Error);
    assert!(x.message.contains("0..=65535"), "{}", x.message);
    // 解決に失敗したときは実効値の検査をしない (値が決まらない)
    assert!(by_code(&d, "max_tokens_over_ctx").is_none(), "{d:?}");
}

// ------------------------------------------------------------- 出力の形と順序

#[test]
fn output_is_deterministic_and_sorted_by_code() {
    let env = env_of(&[
        ("DENNO_FLASH_PROT", "1"),
        ("DENNO_KV_SLOTS", "0"),
        ("DENNO_FLASH_PORT", "http"),
        ("DENNO_MAX_CTX", "8192"),
    ]);
    let a = diagnose_env(&env);
    let b = diagnose_env(&env);
    assert_eq!(a, b);
    let mut sorted = codes(&a);
    sorted.sort_unstable();
    assert_eq!(codes(&a), sorted, "{a:?}");
    // 4 種類の独立した問題が 4 件
    assert_eq!(
        codes(&a),
        vec![
            "env_deprecated",
            "env_not_integer",
            "env_unknown",
            "kv_slots_legacy_zero"
        ],
        "{a:?}"
    );
}

#[test]
fn line_format_is_one_shot_for_stderr() {
    let env = env_of(&[("DENNO_FLASH_PROT", "1")]);
    let d = diagnose_env(&env);
    let line = d[0].line();
    assert!(line.starts_with("[warn] DENNO_FLASH_PROT: "), "{line}");
    assert!(!line.contains('\n'), "{line}");

    let env = env_of(&[("DENNO_FLASH_PORT", "nope")]);
    let d = diagnose_env(&env);
    let line = d[0].line();
    assert!(line.starts_with("[error] DENNO_FLASH_PORT: "), "{line}");
}

#[test]
fn level_strings_are_stable() {
    assert_eq!(Level::Warn.as_str(), "warn");
    assert_eq!(Level::Error.as_str(), "error");
    assert!(Level::Warn < Level::Error);
}
