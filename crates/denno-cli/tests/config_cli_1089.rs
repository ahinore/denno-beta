//! task 1089: `denno serve-flash --config` / `--print-config` のサブプロセス試験。
//!
//! task 906 の `config_906.rs` は `src/config.rs` を直接取り込む単体試験と
//! 基本的な CLI 経路を担当する。こちらは実際のバイナリ
//! (`CARGO_BIN_EXE_denno`) だけを相手に、次を固定する:
//!
//! 1. 解決順 CLI > `DENNO_FLASH_*` > TOML > 既定値が
//!    `# source:` 注記にそのまま出ること。
//! 2. 設定ファイルの誤り (未知キー / 重複 / セクション / 範囲外) が
//!    非ゼロ終了と行番号付きメッセージになること。
//! 3. `--print-config` がポートを bind せずモデルも読まずに 0 で終わること
//!    (使用中ポートと存在しないモデル dir を与えて確認する)。

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// テストごとの一時ディレクトリ。Drop で丸ごと消す。
struct TmpDir {
    path: PathBuf,
}

impl TmpDir {
    fn new(tag: &str) -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "denno1089_{}_{}_{}",
            std::process::id(),
            n,
            tag
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn write(&self, name: &str, body: &str) -> PathBuf {
        let p = self.path.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// `denno` を DENNO_* を全部消した環境で起動する。
/// 明示する値だけ `envs` で与える (ホスト側の設定に左右されない)。
fn denno(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    for (k, _) in std::env::vars() {
        if k.starts_with("DENNO_") {
            cmd.env_remove(k);
        }
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("denno を起動できない")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn config_arg(path: &Path) -> String {
    path.to_str().expect("temp path is UTF-8").to_string()
}

/// `key = value  # source: ...` の行を key -> (value, source) に落とす。
fn annotations(out: &str) -> BTreeMap<String, (String, String)> {
    let mut m = BTreeMap::new();
    for l in out.lines() {
        let l = l.trim_start();
        let Some((key, rest)) = l.split_once(" = ") else {
            continue;
        };
        // コメントアウトされた未設定行 (`# kv_slots = <unset>`) は拾わない。
        if key.starts_with('#') || key.contains(' ') {
            continue;
        }
        let (value, source) = match rest.split_once("  # source: ") {
            Some((v, s)) => (v.to_string(), s.to_string()),
            None => continue,
        };
        m.insert(key.to_string(), (value, source));
    }
    m
}

fn assert_line(
    got: &BTreeMap<String, (String, String)>,
    key: &str,
    value: &str,
    source: &str,
) {
    let (v, s) = got
        .get(key)
        .unwrap_or_else(|| panic!("`{key}` の行が無い: {got:?}"));
    assert_eq!(v, value, "`{key}` の値");
    assert_eq!(s, source, "`{key}` の出所");
}

// ------------------------------------------------------------ 解決順

#[test]
fn precedence_ladder_is_cli_env_config_default() {
    let tmp = TmpDir::new("prec");
    let cfg = tmp.write(
        "flash.toml",
        "dir = \"/models/1089\"\n\
         host = \"cfg-host\"\n\
         port = 9101\n\
         max_ctx = 2048\n\
         name = \"cfg-name\"\n\
         max_tokens = 128\n\
         steer_recipe = \"/models/1089/recipe.json\"\n\
         vram_budget = 24000\n",
    );
    let envs = [
        ("DENNO_FLASH_HOST", "env-host"),
        ("DENNO_FLASH_PORT", "9102"),
        ("DENNO_FLASH_NAME", "env-name"),
        ("DENNO_FLASH_MAX_TOKENS", "99"),
    ];
    let o = denno(
        &[
            "serve-flash",
            "--print-config",
            "--config",
            &config_arg(&cfg),
            // 4 段すべてを同時に踏む: CLI は port / name、env は host /
            // max_tokens、config は max_ctx / dir。
            "--port",
            "9103",
            "--max-ctx",
            "1024",
            "--name",
            "cli-name",
        ],
        &envs,
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let out = stdout(&o);
    assert!(
        out.starts_with("# denno serve-flash resolved config (cli > env > config > default)\n"),
        "header missing: {out}"
    );
    let got = annotations(&out);

    // CLI > env > config > default (同じキーに 4 段が当たる形)
    assert_line(&got, "port", "9103", "cli");
    assert_line(&got, "name", "\"cli-name\"", "cli");
    // CLI が無いキーは env が config に勝つ
    assert_line(&got, "host", "\"env-host\"", "env DENNO_FLASH_HOST");
    assert_line(&got, "max_tokens", "99", "env DENNO_FLASH_MAX_TOKENS");
    // env が無いキーは config が既定値に勝つ
    let cfg_src = format!("config {}", cfg.display());
    assert_line(&got, "dir", "\"/models/1089\"", &cfg_src);
    assert_line(&got, "max_ctx", "1024", "cli");
    assert_line(&got, "steer_recipe", "\"/models/1089/recipe.json\"", &cfg_src);
    // 整数で書いた vram_budget は文字列として保持される
    assert_line(&got, "vram_budget", "\"24000\"", &cfg_src);

    // どの段にも無いキーは <unset> (既定値を持たない)
    assert!(out.contains("# kv_slots = <unset>"), "{out}");
    assert!(out.contains("# mtp_path = <unset>"), "{out}");
}

#[test]
fn defaults_are_annotated_as_default_without_config() {
    let o = denno(&["serve-flash", "--print-config"], &[]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let got = annotations(&stdout(&o));
    assert_line(&got, "host", "\"127.0.0.1\"", "default");
    assert_line(&got, "port", "8098", "default");
    assert_line(&got, "max_ctx", "4096", "default");
    assert_line(&got, "name", "\"qwen3.8-flash-next\"", "default");
    assert_line(&got, "max_tokens", "512", "default");
}

#[test]
fn config_only_values_are_annotated_with_path() {
    let tmp = TmpDir::new("cfgonly");
    let cfg = tmp.write("only.toml", "dir = \"/models/only\"\nport = 9201\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let got = annotations(&stdout(&o));
    let src = format!("config {}", cfg.display());
    assert_line(&got, "dir", "\"/models/only\"", &src);
    assert_line(&got, "port", "9201", &src);
    // ファイルに無いキーは既定値のまま
    assert_line(&got, "max_tokens", "512", "default");
}

// ------------------------------------------------------------ 設定ファイルの誤り

#[test]
fn unknown_key_typo_is_rejected_with_suggestion() {
    let tmp = TmpDir::new("typo");
    let cfg = tmp.write("typo.toml", "dir = \"/m\"\npor = 9000\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert_eq!(o.status.code(), Some(1), "stderr: {}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("未知のキー `por`"), "{e}");
    assert!(e.contains("もしかして: port"), "typo 候補が無い: {e}");
    assert!(e.contains("有効なキー: dir, host, port"), "{e}");
    assert!(e.contains(":2:"), "行番号が無い: {e}");
    // 誤った設定では解決結果を出さない
    assert!(stdout(&o).is_empty(), "stdout: {}", stdout(&o));
}

#[test]
fn duplicate_key_is_rejected_with_first_line() {
    let tmp = TmpDir::new("dup");
    let cfg = tmp.write("dup.toml", "port = 1\nport = 2\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("重複"), "{e}");
    assert!(e.contains("`port`"), "{e}");
    assert!(e.contains("1 行目で既に定義"), "{e}");
    assert!(stdout(&o).is_empty());
}

#[test]
fn section_header_is_rejected() {
    let tmp = TmpDir::new("sec");
    let cfg = tmp.write("sec.toml", "[flash]\nport = 1\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("セクション"), "{e}");
    assert!(e.contains(":1:"), "{e}");
    assert!(stdout(&o).is_empty());
}

#[test]
fn out_of_range_port_is_rejected() {
    let tmp = TmpDir::new("range");
    let cfg = tmp.write("range.toml", "port = 70000\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("`port` の範囲は 0..=65535"), "{e}");
    assert!(e.contains("70000"), "{e}");

    // 負数も同じキーの範囲検査で落ちる
    let neg = tmp.write("neg.toml", "port = -1\n");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&neg)],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("0 以上 65535 以下"), "{e}");
}

#[test]
fn missing_config_file_is_rejected() {
    let tmp = TmpDir::new("missing");
    let missing = tmp.path.join("does-not-exist.toml");
    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&missing)],
        &[],
    );
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("読めない"), "{e}");
    assert!(e.contains(&missing.display().to_string()), "{e}");
    assert!(stdout(&o).is_empty());
}

// ------------------------------------------------ print-config の副作用なし

#[test]
fn print_config_exits_zero_without_binding_port_or_loading_model() {
    // 1) テスト側で先にポートを握る。denno が bind を試みれば AddrInUse で落ちる。
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
    let busy_port = listener.local_addr().unwrap().port();
    // listener はテスト終了まで保持する (Drop まで bind したまま)。
    assert_eq!(listener.local_addr().unwrap().port(), busy_port);

    // 2) 存在しないモデル dir を指定する。読みに行けば必ず失敗する。
    let tmp = TmpDir::new("nobind");
    let missing_dir = tmp.path.join("model-does-not-exist");
    assert!(!missing_dir.exists());
    let cfg = tmp.write(
        "nobind.toml",
        &format!(
            "dir = \"{}\"\nhost = \"127.0.0.1\"\nport = {busy_port}\n",
            missing_dir.display()
        ),
    );

    let o = denno(
        &["serve-flash", "--print-config", "--config", &config_arg(&cfg)],
        &[],
    );
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    assert_eq!(o.status.code(), Some(0));

    let out = stdout(&o);
    let got = annotations(&out);
    let src = format!("config {}", cfg.display());
    assert_line(&got, "dir", &format!("\"{}\"", missing_dir.display()), &src);
    assert_line(&got, "port", &busy_port.to_string(), &src);
    // モデルも読まず、ポートも bind していないので listener は生きている。
    assert_eq!(listener.local_addr().unwrap().port(), busy_port);
}
