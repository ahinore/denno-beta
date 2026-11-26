//! task 1496: doctor / config::diag の秘密伏せ字 allowlist。
//!
//! `*TOKEN*` を含む名前は既定で伏せるが、数値リミットとして明示的に許可した
//! 完全一致の名前だけは値を出す。許可は完全一致のみで、綴りが違えば従来どおり
//! 伏せる (fail-closed)。
//!
//! 確かめること:
//! 1. `denno doctor --json` が `DENNO_FLASH_MAX_TOKENS` / `DENNO_MAX_TOKENS` の
//!    値を出し、`redacted_keys` に載せない。
//! 2. 同じ一覧で `DENNO_API_KEY` / `DENNO_HF_TOKEN` / `DENNO_AUTH_HEADER` は
//!    伏せたまま。allowlist に似せた名前 (`DENNO_FLASH_MAX_TOKENS_X` など) も伏せる。
//! 3. テキスト表示でも数値は出て、秘密は出ない。
//! 4. `config::diag` の上書き診断が `max_tokens` の実値を載せる。
//!
//! GPU を初期化しない読み取りだけの診断なので CPU-only でも安全。

#[path = "../src/config.rs"]
mod config;

use std::collections::BTreeMap;
use std::process::{Command, Output};

use config::diag::{diagnose, is_secret_name};

fn denno(envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args);
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json(o: &Output) -> serde_json::Value {
    serde_json::from_slice(&o.stdout)
        .unwrap_or_else(|e| panic!("bad json ({e}):\n{}\nstderr:\n{}", stdout(o), stderr(o)))
}

fn entry<'a>(v: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    v["env"]["entries"]
        .as_array()
        .expect("env.entries array")
        .iter()
        .find(|e| e["key"] == key)
        .unwrap_or_else(|| panic!("{key} missing from env entries: {v}"))
}

fn redacted_keys(v: &serde_json::Value) -> Vec<String> {
    v["env"]["redacted_keys"]
        .as_array()
        .expect("redacted_keys array")
        .iter()
        .map(|x| x.as_str().expect("string").to_string())
        .collect()
}

#[test]
fn doctor_shows_numeric_limit_tokens_and_hides_secrets() {
    let envs = [
        ("DENNO_FLASH_MAX_TOKENS", "256"),
        ("DENNO_MAX_TOKENS", "512"),
        ("DENNO_API_KEY", "sk-LIVE-1496"),
        ("DENNO_HF_TOKEN", "hf_LIVE_1496"),
        ("DENNO_AUTH_HEADER", "Bearer LIVE-1496"),
        ("DENNO_FLASH_MAX_TOKENS_X", "999"),
        ("DENNO_SECRET_MAX_TOKENS", "777"),
    ];
    let o = denno(&envs, &["doctor", "--json"]);
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let v = json(&o);

    for (key, want) in [("DENNO_FLASH_MAX_TOKENS", "256"), ("DENNO_MAX_TOKENS", "512")] {
        let e = entry(&v, key);
        assert_eq!(e["value"], want, "{key}: {e}");
        assert_eq!(e["redacted"], false, "{key}: {e}");
    }

    for key in [
        "DENNO_API_KEY",
        "DENNO_HF_TOKEN",
        "DENNO_AUTH_HEADER",
        "DENNO_FLASH_MAX_TOKENS_X",
        "DENNO_SECRET_MAX_TOKENS",
    ] {
        let e = entry(&v, key);
        assert_eq!(e["value"], "<redacted>", "{key}: {e}");
        assert_eq!(e["redacted"], true, "{key}: {e}");
    }

    let red = redacted_keys(&v);
    for key in ["DENNO_FLASH_MAX_TOKENS", "DENNO_MAX_TOKENS"] {
        assert!(!red.iter().any(|k| k == key), "{key} in {red:?}");
    }
    for key in ["DENNO_API_KEY", "DENNO_HF_TOKEN", "DENNO_AUTH_HEADER"] {
        assert!(red.iter().any(|k| k == key), "{key} missing from {red:?}");
    }

    // 生の秘密も、allowlist に似せた名前の値も出力に出ない
    let raw = format!("{}{}", stdout(&o), stderr(&o));
    for leaked in ["sk-LIVE-1496", "hf_LIVE_1496", "Bearer LIVE-1496", "999", "777"] {
        assert!(!raw.contains(leaked), "leaked {leaked}:\n{raw}");
    }
}

#[test]
fn doctor_text_mode_shows_numeric_limit_value() {
    let o = denno(
        &[
            ("DENNO_FLASH_MAX_TOKENS", "321"),
            ("DENNO_API_KEY", "sk-TEXT-1496"),
        ],
        &["doctor"],
    );
    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.contains("DENNO_FLASH_MAX_TOKENS=321"), "{t}");
    assert!(t.contains("DENNO_API_KEY=<redacted>"), "{t}");
    assert!(!t.contains("sk-TEXT-1496"), "{t}");
}

#[test]
fn diag_shows_max_tokens_value_in_shadow_message() {
    let env: BTreeMap<String, String> = [
        ("DENNO_FLASH_MAX_TOKENS".to_string(), "8192".to_string()),
        ("DENNO_API_KEY".to_string(), "sk-DIAG-1496".to_string()),
    ]
    .into_iter()
    .collect();
    let cli = vec![("max_tokens", "1024".to_string())];
    let d = diagnose(&cli, &env, None);
    let text: String = d
        .iter()
        .map(|x| format!("{} {}\n", x.vars.join(","), x.message))
        .collect();
    let x = d
        .iter()
        .find(|x| x.code == "env_shadowed_by_cli")
        .unwrap_or_else(|| panic!("env_shadowed_by_cli missing: {text}"));
    assert!(x.message.contains("8192"), "{}", x.message);
    assert!(!text.contains("sk-DIAG-1496"), "{text}");
}

#[test]
fn allowlist_is_exact_match_only() {
    for name in ["DENNO_FLASH_MAX_TOKENS", "DENNO_MAX_TOKENS"] {
        assert!(!is_secret_name(name), "{name}");
    }
    for name in [
        "DENNO_FLASH_MAX_TOKENS_X",
        "DENNO_FLASH_MAX_TOKEN",
        "DENNO_MAX_TOKENS2",
        "DENNO_SECRET_MAX_TOKENS",
        "DENNO_FLASH_MAX_TOKENS_AUTH",
        "denno_flash_max_tokens",
    ] {
        assert!(is_secret_name(name), "{name}");
    }
}
