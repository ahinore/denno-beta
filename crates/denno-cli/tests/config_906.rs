//! task 906: `denno serve-flash --config` / `--print-config` の試験。
//!
//! 1) `src/config.rs` をそのまま取り込んで解析・解決の単体試験を行う
//!    (denno-cli は bin クレートなので lib 経由では取り込めない)。
//! 2) 実際のバイナリ (`CARGO_BIN_EXE_denno`) を起動し、CLI と環境変数と
//!    設定ファイルの解決順・エラーメッセージを確かめる。
//!
//! サーバーは起動しない (`--print-config` か、dir 未解決で落ちる経路のみ)。

#[path = "../src/config.rs"]
mod config;

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const ENV_KEYS: [&str; 10] = [
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
];

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ (並列実行でも衝突しない名前)。
fn tmpfile(name: &str, body: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("denno906_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// `denno` を環境変数を掃除して起動する。`envs` は明示的に与える値。
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

/// `key = value  # source: ...` の行を探して (value, source) を返す。
fn line_of(out: &str, key: &str) -> Option<(String, String)> {
    for l in out.lines() {
        let l = l.trim_start();
        if let Some(rest) = l.strip_prefix(&format!("{key} = ")) {
            let (value, src) = match rest.split_once("  # source: ") {
                Some((v, s)) => (v.to_string(), s.to_string()),
                None => (rest.to_string(), String::new()),
            };
            return Some((value, src));
        }
    }
    None
}

// ---------------------------------------------------------------- CLI 試験

#[test]
fn print_config_defaults() {
    let o = denno(&["serve-flash", "--print-config"], &[]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    assert_eq!(
        line_of(&out, "host"),
        Some(("\"127.0.0.1\"".to_string(), "default".to_string()))
    );
    assert_eq!(
        line_of(&out, "port"),
        Some(("8098".to_string(), "default".to_string()))
    );
    assert_eq!(
        line_of(&out, "max_ctx"),
        Some(("4096".to_string(), "default".to_string()))
    );
    assert_eq!(
        line_of(&out, "name"),
        Some(("\"qwen3.8-flash-next\"".to_string(), "default".to_string()))
    );
    assert_eq!(
        line_of(&out, "max_tokens"),
        Some(("512".to_string(), "default".to_string()))
    );
    assert!(out.contains("# dir = <unset>"), "{out}");
    assert!(out.contains("# kv_slots = <unset>"), "{out}");
}

#[test]
fn print_config_from_file() {
    let cfg = tmpfile(
        "flash.toml",
        "# flash server\n\
         dir = \"/models/flash\"\n\
         host = \"localhost\"\n\
         port = 9001\n\
         max_ctx = 2048\n\
         name = \"tiny\"\n\
         max_tokens = 64\n\
         mtp_path = \"/models/flash/mtp.safetensors\"\n\
         kv_slots = 4\n\
         steer_recipe = \"/models/flash/recipe.json\"\n\
         vram_budget = \"auto\"\n",
    );
    let o = denno(
        &["serve-flash", "--print-config", "--config", cfg.to_str().unwrap()],
        &[],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    let src = format!("config {}", cfg.display());
    assert_eq!(
        line_of(&out, "dir"),
        Some(("\"/models/flash\"".to_string(), src.clone()))
    );
    assert_eq!(line_of(&out, "host"), Some(("\"localhost\"".to_string(), src.clone())));
    assert_eq!(line_of(&out, "port"), Some(("9001".to_string(), src.clone())));
    assert_eq!(line_of(&out, "max_ctx"), Some(("2048".to_string(), src.clone())));
    assert_eq!(line_of(&out, "name"), Some(("\"tiny\"".to_string(), src.clone())));
    assert_eq!(line_of(&out, "max_tokens"), Some(("64".to_string(), src.clone())));
    assert_eq!(line_of(&out, "kv_slots"), Some(("4".to_string(), src.clone())));
    assert_eq!(line_of(&out, "vram_budget"), Some(("\"auto\"".to_string(), src)));
    assert!(out.contains(&format!("mtp_path = \"/models/flash/mtp.safetensors\"")), "{out}");
}

#[test]
fn precedence_cli_beats_env_beats_file_beats_default() {
    let cfg = tmpfile(
        "p.toml",
        "port = 9001\nhost = \"cfg-host\"\nname = \"cfg-name\"\nmax_ctx = 1024\n",
    );
    let envs = [
        ("DENNO_FLASH_PORT", "9002"),
        ("DENNO_FLASH_HOST", "env-host"),
        // name はファイルだけ
        ("DENNO_FLASH_MAX_TOKENS", "77"),
    ];
    let o = denno(
        &[
            "serve-flash",
            "--print-config",
            "--config",
            cfg.to_str().unwrap(),
            "--port",
            "9003",
            "--name",
            "cli-name",
        ],
        &envs,
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    // CLI > env > file
    assert_eq!(
        line_of(&out, "port"),
        Some(("9003".to_string(), "cli".to_string()))
    );
    assert_eq!(
        line_of(&out, "name"),
        Some(("\"cli-name\"".to_string(), "cli".to_string()))
    );
    // env > file
    assert_eq!(
        line_of(&out, "host"),
        Some(("\"env-host\"".to_string(), "env DENNO_FLASH_HOST".to_string()))
    );
    // file > default (max_ctx)
    assert_eq!(
        line_of(&out, "max_ctx"),
        Some((
            "1024".to_string(),
            format!("config {}", cfg.display())
        ))
    );
    // env > default
    assert_eq!(
        line_of(&out, "max_tokens"),
        Some((
            "77".to_string(),
            "env DENNO_FLASH_MAX_TOKENS".to_string()
        ))
    );
}

#[test]
fn env_only_and_file_only_paths() {
    let cfg = tmpfile("q.toml", "dir = \"/models/q\"\nport = 9100\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", cfg.to_str().unwrap()],
        &[("DENNO_FLASH_PORT", "9200")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    assert_eq!(
        line_of(&out, "port"),
        Some(("9200".to_string(), "env DENNO_FLASH_PORT".to_string()))
    );
    assert_eq!(
        line_of(&out, "dir"),
        Some(("\"/models/q\"".to_string(), format!("config {}", cfg.display())))
    );
}

#[test]
fn empty_env_value_is_treated_as_unset() {
    let o = denno(
        &["serve-flash", "--print-config"],
        &[("DENNO_FLASH_HOST", ""), ("DENNO_FLASH_PORT", "   ")],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    assert_eq!(
        line_of(&out, "host"),
        Some(("\"127.0.0.1\"".to_string(), "default".to_string()))
    );
    assert_eq!(
        line_of(&out, "port"),
        Some(("8098".to_string(), "default".to_string()))
    );
}

#[test]
fn unknown_key_is_rejected_with_hint_and_valid_keys() {
    let cfg = tmpfile("bad.toml", "dir = \"/m\"\nprot = 9000\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", cfg.to_str().unwrap()],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("未知のキー `prot`"), "{e}");
    assert!(e.contains("port"), "typo hint missing: {e}");
    assert!(e.contains("有効なキー: dir, host, port"), "{e}");
    assert!(e.contains(":2:"), "line number missing: {e}");
    assert!(stdout(&o).is_empty(), "must not print config on error");
}

#[test]
fn wrong_type_is_rejected_with_actionable_message() {
    let cfg = tmpfile("type.toml", "port = \"abc\"\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", cfg.to_str().unwrap()],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("`port` は整数 0..=65535"), "{e}");
    assert!(e.contains("abc"), "{e}");
}

#[test]
fn out_of_range_is_rejected() {
    let cfg = tmpfile("range.toml", "port = 70000\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", cfg.to_str().unwrap()],
        &[],
    );
    assert!(!o.status.success());
    assert!(stderr(&o).contains("0..=65535"), "{}", stderr(&o));
}

#[test]
fn duplicate_key_and_section_are_rejected() {
    let dup = tmpfile("dup.toml", "port = 1\nport = 2\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", dup.to_str().unwrap()],
        &[],
    );
    assert!(!o.status.success());
    assert!(stderr(&o).contains("重複"), "{}", stderr(&o));

    let sec = tmpfile("sec.toml", "[flash]\nport = 1\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", sec.to_str().unwrap()],
        &[],
    );
    assert!(!o.status.success());
    assert!(stderr(&o).contains("セクション"), "{}", stderr(&o));
}

#[test]
fn missing_config_file_is_actionable() {
    let o = denno(
        &["serve-flash", "--print-config", "--config", "/nonexistent/denno906.toml"],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("/nonexistent/denno906.toml"), "{e}");
    assert!(e.contains("読めない"), "{e}");
}

#[test]
fn bad_env_number_names_the_variable() {
    let o = denno(
        &["serve-flash", "--print-config"],
        &[("DENNO_FLASH_PORT", "http")],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("DENNO_FLASH_PORT"), "{e}");
    assert!(e.contains("整数"), "{e}");
}

#[test]
fn missing_dir_without_print_config_is_actionable() {
    let o = denno(&["serve-flash"], &[]);
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("dir"), "{e}");
    assert!(e.contains("--dir"), "{e}");
    assert!(e.contains("DENNO_FLASH_DIR"), "{e}");
}

#[test]
fn cli_dir_is_used_when_config_absent() {
    let o = denno(
        &["serve-flash", "--print-config", "--dir", "/models/from-cli"],
        &[],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    assert_eq!(
        line_of(&stdout(&o), "dir"),
        Some(("\"/models/from-cli\"".to_string(), "cli".to_string()))
    );
}

#[test]
fn invalid_file_value_fails_even_if_cli_overrides() {
    // 設定ファイルは「丸ごと」検証する。CLI が同じキーで勝つ場合でも、
    // ファイルの誤り (範囲外) は黙って無視せず落とす (設定の腐りを早期に出す)。
    let cfg = tmpfile("badport.toml", "port = 70000\n");
    let o = denno(
        &[
            "serve-flash",
            "--print-config",
            "--config",
            cfg.to_str().unwrap(),
            "--port",
            "8123",
        ],
        &[],
    );
    // ファイル側の範囲検査は解析時に行うため、ここは失敗するのが仕様。
    // 「黙って無視しない」ことを固定する (設定ファイルの誤りは常に大声で落とす)。
    assert!(!o.status.success());
    assert!(stderr(&o).contains("0..=65535"), "{}", stderr(&o));
}

#[test]
fn kv_slots_zero_means_unset_and_is_annotated() {
    // 従来の `DENNO_KV_SLOTS=0` と同じく「未指定 (= 既定 2)」に正規化される。
    let o = denno(&["serve-flash", "--print-config", "--kv-slots", "0"], &[]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let (value, src) = line_of(&stdout(&o), "kv_slots").expect("kv_slots line");
    assert_eq!(value, "0");
    assert!(src.contains("cli"), "{src}");
    assert!(src.contains("未指定"), "annotation missing: {src}");
}

// ------------------------------------------------------- config.rs 単体試験
mod unit {
    use super::config::{default_value, env_var, parse_str, resolve, Source, Value};
    use std::collections::BTreeMap;

    fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parses_all_keys() {
        let f = parse_str(
            "dir = \"/m\"\nhost = \"h\"\nport = 1\nmax_ctx = 2\nname = \"n\"\n\
             max_tokens = 3\nmtp_path = \"/t\"\nkv_slots = 4\n\
             steer_recipe = \"/r\"\nvram_budget = \"auto\"\n",
            "t",
        )
        .unwrap();
        for k in super::config::KEYS {
            assert!(f.get(k).is_some(), "missing {k}");
        }
        assert_eq!(f.get("kv_slots"), Some(&Value::Int(4)));
    }

    #[test]
    fn integer_vram_budget_is_accepted_as_text() {
        let f = parse_str("vram_budget = 24000\n", "t").unwrap();
        assert_eq!(f.get("vram_budget"), Some(&Value::Str("24000".into())));
    }

    #[test]
    fn env_names_and_defaults_are_stable() {
        assert_eq!(env_var("host"), Some("DENNO_FLASH_HOST"));
        assert_eq!(env_var("mtp_path"), Some("DENNO_MTP_PATH"));
        assert_eq!(env_var("vram_budget"), Some("DENNO_AUTO_CONFIG"));
        assert_eq!(default_value("port"), Some("8098"));
        assert_eq!(default_value("dir"), None);
    }

    #[test]
    fn resolve_records_source_and_origin() {
        let f = parse_str("name = \"file-name\"\n", "/tmp/c.toml").unwrap();
        let env = env_of(&[("DENNO_FLASH_PORT", "9300")]);
        let r = resolve(&[("name", "cli-name".to_string())], &env, Some(&f)).unwrap();
        assert_eq!(r.string("name").as_deref(), Some("cli-name"));
        assert_eq!(r.source("name"), Some(Source::Cli));
        // 設定ファイルの出所はパスを持つ
        let r = resolve(&[], &env, Some(&f)).unwrap();
        let slot = r.slot("name").unwrap();
        assert_eq!(slot.source, Source::Config);
        assert_eq!(slot.detail, "/tmp/c.toml");
        assert!(slot.origin_label().starts_with("config "));
    }

    #[test]
    fn negative_integer_is_rejected_for_unsigned_keys() {
        let e = parse_str("max_ctx = -1\n", "t").unwrap_err();
        assert!(e.contains("`max_ctx`"), "{e}");
    }

    #[test]
    fn unquoted_string_is_rejected_with_hint() {
        let e = parse_str("host = localhost\n", "t").unwrap_err();
        assert!(e.contains("未対応"), "{e}");
        assert!(e.contains("host") || e.contains("localhost"), "{e}");
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        let e = parse_str("host = \"a\" b\n", "t").unwrap_err();
        assert!(e.contains("余分な文字"), "{e}");
    }
}
