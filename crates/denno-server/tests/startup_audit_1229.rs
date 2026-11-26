//! task 1229: 起動サマリ整形 (`startup.rs`) と serve ログ経路の
//! **秘密の非漏洩** 監査テスト。
//!
//! 対象 (crates/denno-server/src/startup.rs / logging.rs を全読して確認):
//! - `summarize_auth(raw)` → `AuthState` (キーの **本数だけ** を持つ)
//! - `format_startup_summary(&StartupSummary)` (純関数・env も時計も読まない)
//! - `access_log_line(...)` (1 行 JSON)
//! - `NonBlockingStdout` 経由の tracing パイプライン
//!
//! 契約:
//!   (a) 生の API キー値は `AuthState` に存在しない。整形結果 (テキスト・
//!       `Debug`) のどこにも現れない。キーが改行・引用符・制御文字・非 ASCII を
//!       含んでも行構造が壊れない (ログ行の偽造ができない)。
//!   (b) キー本数が同じなら、キーの値が違っても整形結果は完全に同一。
//!   (c) tracing パイプライン (整形 → info! → MakeWriter) を通しても漏れない。
//!   (d) JSON アクセスログのフィールド集合は閉じている (ヘッダ値が乗る余地が無い)。
//!
//! 本ファイルはテスト専用。`src/` には一切触れない。

use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use denno_server::ratelimit::RateLimitConfig;
use denno_server::{
    access_log_line, format_startup_summary, summarize_auth, AuthState, CorsConfig, CorsSummary,
    NonBlockingStdout, RateLimitSummary, StartupSummary,
};
use tracing_subscriber::fmt::MakeWriter;

// ---------- 監査用の疑似キー ----------

/// 実キーらしい形の sentinel を作る (リテラルのベタ書きを避けてタグから組む)。
fn secret(tag: &str) -> String {
    format!("sk-{}-{}-{}", "live", tag, "0123456789abcdef0123456789abcdef")
}

/// 出力に現れてはならない断片 (キーの値そのもの + 特徴的な接頭辞)。
fn forbidden_needles(raw_keys: &[String]) -> Vec<String> {
    let mut needles: Vec<String> = raw_keys.iter().cloned().collect();
    needles.push("sk-live".to_string());
    needles.push("sk-".to_string());
    needles
}

fn sample<'a>() -> StartupSummary<'a> {
    StartupSummary {
        host: "127.0.0.1",
        port: 8080,
        model_id: "org/tiny-model",
        model_name: "tiny-model",
        context_tokens: 8192,
        kv_slots: 1,
        quant: "int8",
        layers: 28,
        auth: AuthState::Disabled,
        metrics_enabled: true,
        queue_depth: 32,
        request_timeout_secs: 0,
        cors: CorsSummary::LoopbackOnly,
        max_body_bytes: 16 * 1024 * 1024,
        rate_limit: RateLimitSummary::Disabled,
    }
}

fn assert_no_secrets(where_: &str, text: &str, needles: &[String]) {
    for n in needles {
        assert!(
            !text.contains(n.as_str()),
            "{where_}: 禁止断片 {n:?} が出力に現れた\n---\n{text}"
        );
    }
}

// ---------- (a) 整形結果にキー値が現れない ----------

#[test]
fn key_values_never_appear_in_formatted_summary() {
    let cases: Vec<Vec<String>> = vec![
        vec![secret("single")],
        vec![secret("a"), secret("b")],
        vec![secret("a"), secret("a"), secret("b")], // 重複
        vec![secret("with space"), secret("with  double")],
        vec!["sk-live-x\n  auth: enabled (99 key(s), values hidden)".to_string()], // 行偽造
        vec!["sk-live-\"quoted\"".to_string()],
        vec!["sk-live-\u{1b}[31mred\u{7}".to_string()],
        vec!["sk-live-日本語キー".to_string()],
        vec![format!("sk-live-{}", "z".repeat(2048))],
    ];

    for keys in cases {
        let raw = keys.join(",");
        let auth = summarize_auth(Some(&raw));
        assert_ne!(auth, AuthState::Disabled, "キーが 1 本も数えられていない");

        let out = format_startup_summary(&StartupSummary { auth, ..sample() });
        assert_no_secrets("summary", &out, &forbidden_needles(&keys));
        assert!(
            out.contains("values hidden"),
            "有効時は値が隠れている旨を出す: {out}"
        );
    }
}

#[test]
fn debug_formats_of_auth_state_and_summary_are_secret_free() {
    let a = secret("dbg-a");
    let b = secret("dbg-b");
    let auth = summarize_auth(Some(&format!("{a}, {b}")));
    let s = StartupSummary { auth, ..sample() };

    let mut text = String::new();
    let _ = write!(text, "{:?}\n{:#?}\n{:?}\n{:#?}", auth, auth, s, s);
    assert!(text.contains("key_count"), "Debug に本数が出ていない: {text}");
    assert_no_secrets("debug", &text, &forbidden_needles(&[a, b]));
}

#[test]
fn auth_state_carries_only_a_count() {
    let a = secret("payload");
    let auth = summarize_auth(Some(&a));
    assert_eq!(auth, AuthState::Enabled { key_count: 1 });

    // 判別子 + usize 1 個ぶんを超えない = 文字列/ヒープ参照を持てない。
    assert!(
        std::mem::size_of::<AuthState>() <= 2 * std::mem::size_of::<usize>(),
        "AuthState が想定より大きい (ペイロードを持ち始めた可能性): {} bytes",
        std::mem::size_of::<AuthState>()
    );
    assert_eq!(format!("{:?}", auth), "Enabled { key_count: 1 }");
    assert_eq!(format!("{:?}", AuthState::Disabled), "Disabled");
}

#[test]
fn summarize_auth_counts_but_does_not_return_values() {
    let a = secret("count-a");
    let b = secret("count-b");
    let raw = format!("  {a} , {a}, {b} ,  ");
    let auth = summarize_auth(Some(&raw));
    assert_eq!(auth, AuthState::Enabled { key_count: 2 });
    // AuthState の表現にキー値が入らない (部分文字列としても現れない)。
    let repr = format!("{:?}/{:?}", auth, auth);
    assert_no_secrets("auth-state", &repr, &forbidden_needles(&[a, b]));
}

// ---------- 行構造の堅牢性 ----------

#[test]
fn key_material_cannot_forge_extra_log_lines() {
    let base = format_startup_summary(&sample());
    let base_lines = base.lines().count();

    let injection = "sk-live-x\n  auth: enabled (99 key(s), values hidden)\n  bind: http://evil";
    let out = format_startup_summary(&StartupSummary {
        auth: summarize_auth(Some(injection)),
        ..sample()
    });

    assert_eq!(
        out.lines().count(),
        base_lines,
        "キー経由で行数が増えた (行偽造):\n{out}"
    );
    assert_eq!(
        out.lines().filter(|l| l.trim_start().starts_with("auth:")).count(),
        1,
        "auth 行が複数ある:\n{out}"
    );
    assert!(!out.contains("evil"), "キー断片が bind 行に混入:\n{out}");
    assert_eq!(out.matches('\n').count(), base.matches('\n').count());
}

#[test]
fn output_for_equal_key_counts_is_identical_regardless_of_values() {
    let one = format_startup_summary(&StartupSummary {
        auth: summarize_auth(Some(&format!("{},{}", secret("p"), secret("q")))),
        ..sample()
    });
    let two = format_startup_summary(&StartupSummary {
        auth: summarize_auth(Some(&format!("{},{}", secret("r"), secret("s")))),
        ..sample()
    });
    assert_eq!(one, two, "キー本数が同じでも値で出力が変わった (値依存の漏洩)");
}

// ---------- (b) 全設定形状の掃引で整形器が秘密を発明しない ----------

#[test]
fn no_secret_shaped_tokens_across_all_config_shapes() {
    let cors_shapes = [
        CorsSummary::Any,
        CorsSummary::LoopbackOnly,
        CorsSummary::None,
        CorsSummary::Allowlist { origins: vec!["https://a.example".into()], loopback: true },
        CorsSummary::Allowlist { origins: vec![], loopback: false },
    ];
    let rate_shapes = [
        RateLimitSummary::Disabled,
        RateLimitSummary::Enabled { rpm: 0, burst: 0 },
        RateLimitSummary::Enabled { rpm: 600, burst: 60 },
    ];
    let auth_shapes = [
        AuthState::Disabled,
        AuthState::Enabled { key_count: 0 },
        AuthState::Enabled { key_count: 7 },
    ];

    for cors in cors_shapes {
        for rate_limit in rate_shapes {
            for auth in auth_shapes {
                for metrics_enabled in [true, false] {
                    let s = StartupSummary {
                        auth,
                        cors: cors.clone(),
                        rate_limit,
                        metrics_enabled,
                        ..sample()
                    };
                    let out = format_startup_summary(&s);
                    assert!(!out.contains("sk-"), "整形器が鍵風文字列を出した:\n{out}");
                    assert!(!out.contains("DENNO_API_KEYS"), "env 名が漏れた:\n{out}");
                    assert!(!out.contains("key="), "key= 形式が漏れた:\n{out}");
                    assert!(out.ends_with('\n'), "末尾改行が無い: {out:?}");
                }
            }
        }
    }
}

// ---------- (c) tracing パイプライン経由 ----------

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
    }
}

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// serve と同じ「整形 → info! → MakeWriter」を通した出力を返す。
fn run_log_pipeline(f: impl FnOnce()) -> String {
    let cap = Capture::default();
    let writer = cap.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, f);
    cap.text()
}

#[test]
fn startup_summary_through_tracing_pipeline_is_secret_free() {
    let a = secret("pipe-a");
    let b = secret("pipe-b");
    let auth = summarize_auth(Some(&format!("{a},{b}")));
    let s = StartupSummary { auth, ..sample() };

    let logged = run_log_pipeline(|| {
        tracing::info!("listening on http://{}:{}", s.host, s.port);
        // src/lib.rs の serve 経路と同じく、整形済み文字列を 1 イベントで出す。
        tracing::info!("{}", format_startup_summary(&s));
    });

    assert!(logged.contains("denno-server startup"), "サマリがログに無い:\n{logged}");
    assert!(logged.contains("values hidden"), "本数表示が無い:\n{logged}");
    assert_no_secrets("log-pipeline", &logged, &forbidden_needles(&[a, b]));
}

#[test]
fn nonblocking_writer_passes_summary_without_secret_material() {
    // 実際の MakeWriter (有界キュー + 専用スレッド) を通す経路。
    let cap = Capture::default();
    let sink = cap.clone();
    struct SharedCapture(Capture);
    impl Write for SharedCapture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let writer = NonBlockingStdout::spawn_with(Box::new(SharedCapture(sink)), 64);
    let a = secret("queue-a");
    let s = StartupSummary {
        auth: summarize_auth(Some(&a)),
        ..sample()
    };
    let line = format_startup_summary(&s);
    writer.make_writer().write_all(line.as_bytes()).unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !cap.text().contains("denno-server startup") {
        assert!(
            std::time::Instant::now() < deadline,
            "ロガースレッドがサマリを書き出さない"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_no_secrets("nonblocking", &cap.text(), &forbidden_needles(&[a]));
    assert_eq!(writer.dropped(), 0);
}

// ---------- (d) JSON アクセスログの閉じたフィールド集合 ----------

#[test]
fn json_access_log_has_closed_field_set() {
    let line = access_log_line("POST", "/v1/messages", 200, 1.5, "req-0123456789ab");
    let v: serde_json::Value = serde_json::from_str(&line).expect("JSON として読めない");
    let obj = v.as_object().expect("JSON オブジェクトでない");

    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["duration_ms", "method", "path", "request_id", "status", "ts_ms"],
        "フィールド集合が変わった (ヘッダ/キーが乗る余地): {keys:?}"
    );
    for forbidden in ["api_key", "x-api-key", "authorization", "headers", "cookie", "env"] {
        assert!(!obj.contains_key(forbidden), "禁止フィールド {forbidden} が存在する");
    }
    assert!(!line.contains("sk-"), "アクセスログに鍵風文字列: {line}");
    assert!(!line.contains('\n'), "1 行 JSON が複数行になった: {line:?}");
}

#[test]
fn benign_inputs_never_produce_secret_shaped_output() {
    // 整形器/JSON 化は入力をそのまま出すだけ。秘密を「作らない」ことの確認。
    let cors = CorsSummary::from_config(&CorsConfig {
        allow_any: false,
        origins: vec!["https://app.example".into()],
        allow_loopback: true,
    });
    let rate = RateLimitSummary::from_config(&RateLimitConfig::new(60, Some(10)));
    let out = format_startup_summary(&StartupSummary {
        auth: AuthState::Disabled,
        cors,
        rate_limit: rate,
        ..sample()
    });
    assert_no_secrets("benign", &out, &["sk-".to_string(), "DENNO_API_KEYS".to_string()]);
    assert!(out.contains("auth:        disabled (no api key configured)"), "{out}");
}
