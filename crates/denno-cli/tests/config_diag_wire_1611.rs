//! task 1611: `serve-flash` 起動時の `DENNO_*` 診断 (config::diag の配線) の契約。
//!
//! 1. **エンジン変数で騒がない** — denno-engine / denno-server 側の `DENNO_*`
//!    (`DENNO_PF_*` / `DENNO_EXPERT_*` など) は既定で「未知」と報告しない。
//!    設定名に近い名前 (`DENNO_FLASH_PROT` など) は誤字の疑いとして残す。
//! 2. **値の誤りと秘密は従来どおり** — quiet でも範囲外・非整数は出る。
//!    秘密名の値は決して出ない (名前だけ)。
//! 3. **strict モード** — `--strict-env` / `DENNO_FLASH_STRICT_ENV=1` で
//!    警告が 1 件でもあれば起動しない (`--print-config` は止めない)。
//!
//! denno-cli は bin クレートなので、既存の `config_diag_1066.rs` と同じく
//! `#[path]` で `src/config.rs` を読み、CLI 側は `CARGO_BIN_EXE_denno` で確かめる。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;
use std::process::{Command, Output};

use config::diag::{
    diagnose, diagnose_env, diagnose_env_with, diagnose_with, is_quiet_unknown, Diag, Level,
    KNOWN_NON_CONFIG_ENV,
};

fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn codes(diags: &[Diag]) -> Vec<&'static str> {
    diags.iter().map(|d| d.code).collect()
}

fn by_code<'a>(diags: &'a [Diag], code: &str) -> Option<&'a Diag> {
    diags.iter().find(|d| d.code == code)
}

fn all_text(diags: &[Diag]) -> String {
    diags
        .iter()
        .map(|d| format!("{} {}\n", d.vars.join(", "), d.message))
        .collect()
}

/// ラボで実際に付く `DENNO_*` の代表 (すべて denno-engine 側が読む)。
const ENGINE_ENVS: [(&str, &str); 8] = [
    ("DENNO_PF_EARLY_ROUTE", "0"),
    ("DENNO_PF_LM_TILE", "8"),
    ("DENNO_EXPERT_CACHE_MB", "2048"),
    ("DENNO_EXPERT_IQ_GPU", "1"),
    ("DENNO_START_DROP_CACHE", "1"),
    ("DENNO_Q4LEDGER", "1"),
    ("DENNO_ATTN_KPROF", "1"),
    ("DENNO_AUTO_CONFIG", "auto"),
];

// ---------------------------------------------------------------- フィルタ (単体)

#[test]
fn engine_env_names_are_quiet_but_reported_when_unfiltered() {
    let env = env_of(&ENGINE_ENVS);
    assert_eq!(
        diagnose_env_with(&env, true),
        Vec::new(),
        "エンジン変数で警告が出た"
    );
    let loud = diagnose_env(&env);
    assert_eq!(
        loud.iter().filter(|d| d.code == "env_unknown").count(),
        7,
        "フィルタ前は unknown が出る (DENNO_AUTO_CONFIG は設定名なので除く): {loud:?}"
    );
    // quiet 側が本当に分岐した証拠 (同じ env で結果が違う)
    assert_ne!(diagnose_env(&env), diagnose_env_with(&env, true));
}

#[test]
fn quiet_filter_keeps_typos_of_config_names() {
    for name in [
        "DENNO_FLASH_PROT",
        "DENNO_FLASH_POR",
        "DENNO_FLASH_MAX_CTX_TYPO",
        "DENNO_FLASH_STRICT_EN",
    ] {
        let env = env_of(&[(name, "1")]);
        let d = diagnose_env_with(&env, true);
        assert!(
            by_code(&d, "env_unknown").is_some(),
            "{name} の警告が消えた: {d:?}"
        );
    }
    // 設定名に近いので、提案も従来どおり付く
    let d = diagnose_env_with(&env_of(&[("DENNO_FLASH_PROT", "1")]), true);
    let x = by_code(&d, "env_unknown").unwrap();
    assert!(x.message.contains("DENNO_FLASH_PORT"), "{}", x.message);
}

#[test]
fn quiet_filter_keeps_every_denno_flash_name() {
    // `DENNO_FLASH_` 空間は serve-flash の名前空間なので、設定名でなくても残す
    // (将来の設定名を黙って飲み込まない)。
    let env = env_of(&[("DENNO_FLASH_FUTURE_KNOB", "1")]);
    assert!(by_code(&diagnose_env_with(&env, true), "env_unknown").is_some());
}

#[test]
fn quiet_filter_does_not_hide_value_errors() {
    let env = env_of(&[
        ("DENNO_FLASH_PORT", "http"),
        ("DENNO_PF_LM_TILE", "8"),
    ]);
    let d = diagnose_env_with(&env, true);
    let x = by_code(&d, "env_not_integer").expect("env_not_integer");
    assert_eq!(x.level, Level::Error);
    // エンジン変数側は静か
    assert_eq!(codes(&d), vec!["env_not_integer"], "{d:?}");
}

#[test]
fn quiet_filter_keeps_deprecated_and_legacy_spellings() {
    let env = env_of(&[("DENNO_MAX_CTX", "8192"), ("DENNO_KV_SLOTS", "0")]);
    let d = diagnose_env_with(&env, true);
    assert!(by_code(&d, "env_deprecated").is_some(), "{d:?}");
    assert!(by_code(&d, "kv_slots_legacy_zero").is_some(), "{d:?}");
}

#[test]
fn strict_gate_env_name_is_quiet_and_known() {
    for name in KNOWN_NON_CONFIG_ENV {
        assert!(is_quiet_unknown(name), "{name}");
        assert_eq!(diagnose_env_with(&env_of(&[(name, "1")]), true), Vec::new(), "{name}");
        // 大文字小文字の取り違えは読まれないので従来どおり警告する
        let lower = name.to_ascii_lowercase();
        let d = diagnose_env_with(&env_of(&[(lower.as_str(), "1")]), true);
        assert_eq!(codes(&d), vec!["env_case"], "{lower}: {d:?}");
    }
}

#[test]
fn quiet_filter_is_exact_about_denno_prefix() {
    // 接頭辞の無い名前 (既知名に近い) の扱いは quiet でも変えない
    let d = diagnose_env_with(&env_of(&[("ENNO_FLASH_PORT", "1")]), true);
    assert!(by_code(&d, "env_unprefixed").is_some(), "{d:?}");
    assert!(!is_quiet_unknown("ENNO_FLASH_PORT"));
    assert!(!is_quiet_unknown("DENNO_FLASH_PORT"));
}

#[test]
fn quiet_diagnose_still_reports_source_conflicts() {
    let env = env_of(&[("DENNO_FLASH_PORT", "9002")]);
    let cli = vec![("port", "9003".to_string())];
    let d = diagnose_with(&cli, &env, None, true);
    let x = by_code(&d, "env_shadowed_by_cli").expect("env_shadowed_by_cli");
    assert!(x.message.contains("9002"), "{}", x.message);
    // フィルタ無しと同じ結果 (quiet は unknown の分類だけを変える)
    assert_eq!(d, diagnose(&cli, &env, None));
}

#[test]
fn quiet_diagnostics_never_print_secret_values() {
    let env = env_of(&[
        ("DENNO_API_KEY", "«redacted:sk-…»"),
        ("DENNO_HF_TOKEN", "hf_ALSO-SECRET"),
        ("DENNO_SOMETHING_PASSWORD", "pw-1234"),
        ("DENNO_PF_EARLY_ROUTE", "0"),
    ]);
    let d = diagnose_env_with(&env, true);
    let text = all_text(&d);
    for secret in ["«redacted:sk-…»", "hf_ALSO-SECRET", "pw-1234"] {
        assert!(!text.contains(secret), "秘密が漏れた: {text}");
    }
    // 名前は出る (何が設定されているかは分かる) — エンジン変数は出ない
    assert!(text.contains("DENNO_API_KEY"), "{text}");
    assert!(!text.contains("DENNO_PF_EARLY_ROUTE"), "{text}");
}

// ------------------------------------------------------------------ CLI (バイナリ)

const ENV_KEYS: [&str; 11] = [
    "DENNO_FLASH_DIR",
    "DENNO_FLASH_HOST",
    "DENNO_FLASH_PORT",
    "DENNO_FLASH_MAX_CTX",
    "DENNO_FLASH_NAME",
    "DENNO_FLASH_MAX_TOKENS",
    "DENNO_MTP_PATH",
    "DENNO_KV_SLOTS",
    "DENNO_STEER_RECIPE",
    "DENNO_AUTO_CONFIG",
    "DENNO_FLASH_STRICT_ENV",
];

fn denno(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    for k in ENV_KEYS {
        cmd.env_remove(k);
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

#[test]
fn cli_engine_env_is_silent_on_startup() {
    let o = denno(&["serve-flash", "--print-config"], &ENGINE_ENVS);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let e = stderr(&o);
    assert!(
        !e.contains("[flash] env:"),
        "エンジン変数で警告が出た: {e}"
    );
    assert!(stdout(&o).contains("host = "), "stdout: {}", stdout(&o));
}

#[test]
fn cli_typo_is_reported_but_does_not_block_startup() {
    let o = denno(
        &["serve-flash", "--print-config"],
        &[("DENNO_FLASH_PROT", "8098")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("[flash] env: [warn] DENNO_FLASH_PROT:"), "{e}");
    assert!(e.contains("DENNO_FLASH_PORT"), "{e}");
    // 値は載せない (typo 名に紐づく値は信用できない)
    assert!(!e.contains("8098"), "{e}");
}

#[test]
fn cli_strict_env_aborts_before_loading_the_model() {
    let o = denno(
        &["serve-flash", "--strict-env"],
        &[("DENNO_FLASH_PROT", "8098")],
    );
    assert!(!o.status.success(), "strict なのに起動した");
    let e = stderr(&o);
    assert!(e.contains("[warn] DENNO_FLASH_PROT:"), "{e}");
    assert!(e.contains("--strict-env: 警告 1 件で起動を中止した"), "{e}");
    // モデル読み込み (mtp / GGUF) の手前で止まっている
    assert!(!e.contains("mtp:"), "{e}");
}

#[test]
fn cli_strict_env_can_be_enabled_by_environment_variable() {
    let o = denno(
        &["serve-flash"],
        &[("DENNO_FLASH_STRICT_ENV", "1"), ("DENNO_FLASH_PROT", "1")],
    );
    assert!(!o.status.success(), "strict なのに起動した");
    let e = stderr(&o);
    assert!(e.contains("--strict-env: 警告 1 件で起動を中止した"), "{e}");
    // この env 自体は「未知」として二重に数えない (警告は typo の 1 件だけ)
    assert!(!e.contains("DENNO_FLASH_STRICT_ENV:"), "{e}");
}

#[test]
fn cli_strict_env_accepts_false_spellings() {
    let o = denno(
        &["serve-flash", "--print-config"],
        &[("DENNO_FLASH_STRICT_ENV", "0"), ("DENNO_FLASH_PROT", "1")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
}

#[test]
fn cli_strict_does_not_block_print_config() {
    // 設定の確認は strict でも止めない (警告は出す)。
    let o = denno(
        &["serve-flash", "--print-config", "--strict-env"],
        &[("DENNO_FLASH_PROT", "1")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    assert!(stderr(&o).contains("[warn] DENNO_FLASH_PROT:"), "{}", stderr(&o));
    assert!(stdout(&o).contains("port = 8098"), "{}", stdout(&o));
}

#[test]
fn cli_secret_value_is_never_printed() {
    let o = denno(
        &["serve-flash", "--print-config"],
        &[("DENNO_API_KEY", "«redacted:sk-live-…»")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let all = format!("{}{}", stdout(&o), stderr(&o));
    assert!(!all.contains("sk-live"), "秘密が漏れた: {all}");
    assert!(stderr(&o).contains("DENNO_API_KEY"), "{}", stderr(&o));
}

#[test]
fn cli_clean_env_has_no_diagnostic_lines() {
    let o = denno(&["serve-flash", "--print-config"], &[]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    assert!(!stderr(&o).contains("[flash] env:"), "{}", stderr(&o));
}
