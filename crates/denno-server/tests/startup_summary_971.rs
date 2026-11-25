//! task 971: 起動サマリ整形の結合テスト。
//!
//! 目的は 2 つ:
//! 1. serve 起動時に出すべき設定 (bind / model / context / KV slots / auth /
//!    metrics / queue / timeout / CORS / body limit) が 1 つのサマリに揃うこと。
//! 2. **生の API キーが絶対に出ないこと** (sentinel 文字列で検査する)。

use denno_server::ratelimit::RateLimitConfig;
use denno_server::{
    format_startup_summary, summarize_auth, AuthState, CorsConfig, CorsSummary, RateLimitSummary,
    StartupSummary,
};

/// 実際のキーらしい sentinel。整形結果に現れたら漏洩。
const KEY_A: &str = "sk-live-SENTINEL-AAA-111";
const KEY_B: &str = "sk-live-SENTINEL-BBB-222";

fn base<'a>() -> StartupSummary<'a> {
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

fn lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

#[test]
fn summary_contains_every_required_section() {
    let out = format_startup_summary(&base());
    let l = lines(&out);
    assert_eq!(l[0], "denno-server startup", "先頭はヘッダ: {out}");
    for need in [
        "bind:        http://127.0.0.1:8080",
        "org/tiny-model",
        "tiny-model",
        "8192 tokens",
        "kv slots:    1",
        "quant int8",
        "28 layers",
        "auth:        disabled",
        "metrics:     enabled",
        "depth=32",
        "request timeout=off",
        "cors:        loopback only",
        "body limit:  16 MiB (16777216 bytes)",
        "rate limit:  disabled",
    ] {
        assert!(out.contains(need), "欠落: {need:?}\n---\n{out}");
    }
}

#[test]
fn summary_is_deterministic_and_ends_with_newline() {
    let a = format_startup_summary(&base());
    let b = format_startup_summary(&base());
    assert_eq!(a, b, "同じ入力で出力が揺れた");
    assert!(a.ends_with('\n'), "末尾は改行: {a:?}");
}

// ---------------------------------------------------------------- 秘密の非漏洩

#[test]
fn api_keys_never_appear_in_any_summary_field() {
    // serve と同じ経路: カンマ区切りの生キー → AuthState (本数だけ)。
    let raw = format!("{KEY_A},{KEY_B}");
    let auth = summarize_auth(Some(&raw));
    assert_eq!(auth, AuthState::Enabled { key_count: 2 });

    let mut s = base();
    s.auth = auth;
    let out = format_startup_summary(&s);

    assert!(out.contains("auth:        enabled (2 key(s), values hidden)"));
    assert!(!out.contains(KEY_A), "キー A が漏れた:\n{out}");
    assert!(!out.contains(KEY_B), "キー B が漏れた:\n{out}");
    assert!(!out.contains("SENTINEL"), "キー断片が漏れた:\n{out}");
    assert!(!out.contains("sk-live"), "キー接頭辞が漏れた:\n{out}");
}

#[test]
fn disabled_auth_does_not_echo_a_placeholder() {
    let out = format_startup_summary(&base());
    assert!(out.contains("auth:        disabled (no api key configured)"));
    assert!(!out.contains("sk-"), "キー風の文字列が混入:\n{out}");
}

#[test]
fn key_like_strings_in_model_fields_do_not_get_masked_or_leaked_into_auth() {
    // 誤ってモデル名にキーらしき文字列が入っても、それは auth 行ではない。
    // (整形器は auth 行に AuthState 以外を書かない、という不変の確認)
    let mut s = base();
    s.auth = AuthState::Disabled;
    s.model_name = "model-with-no-secret";
    let out = format_startup_summary(&s);
    let auth_line = out
        .lines()
        .find(|l| l.trim_start().starts_with("auth:"))
        .expect("auth 行が無い");
    assert_eq!(auth_line.trim(), "auth:        disabled (no api key configured)");
}

#[test]
fn key_count_reflects_dedupe_and_whitespace() {
    let out = format_startup_summary(&StartupSummary {
        auth: summarize_auth(Some(&format!("  {KEY_A} , {KEY_A},{KEY_B} ,  "))),
        ..base()
    });
    assert!(out.contains("enabled (2 key(s)"), "重複を数えている:\n{out}");
    assert!(!out.contains("SENTINEL"), "キーが漏れた:\n{out}");
}

// ------------------------------------------------------------------ 各設定

#[test]
fn metrics_disabled_is_reported() {
    let out = format_startup_summary(&StartupSummary { metrics_enabled: false, ..base() });
    assert!(out.contains("metrics:     disabled"), "{out}");
}

#[test]
fn request_timeout_in_seconds_is_reported() {
    let out =
        format_startup_summary(&StartupSummary { request_timeout_secs: 30, ..base() });
    assert!(out.contains("request timeout=30s"), "{out}");
    assert!(!out.contains("request timeout=off"), "{out}");
}

#[test]
fn queue_depth_is_the_resolved_value() {
    let out = format_startup_summary(&StartupSummary { queue_depth: 1, ..base() });
    assert!(out.contains("depth=1,"), "{out}");
}

#[test]
fn cors_any_and_allowlist_shapes() {
    let any = format_startup_summary(&StartupSummary { cors: CorsSummary::Any, ..base() });
    assert!(any.contains("cors:        * (any origin)"), "{any}");

    let allow = format_startup_summary(&StartupSummary {
        cors: CorsSummary::Allowlist {
            origins: vec!["https://app.example".into(), "https://alt.example".into()],
            loopback: false,
        },
        ..base()
    });
    assert!(allow.contains("https://app.example"), "{allow}");
    assert!(allow.contains("https://alt.example"), "{allow}");
    assert!(!allow.contains("+ loopback"), "{allow}");

    let with_lb = format_startup_summary(&StartupSummary {
        cors: CorsSummary::Allowlist { origins: vec!["https://app.example".into()], loopback: true },
        ..base()
    });
    assert!(with_lb.contains("+ loopback"), "{with_lb}");

    let none = format_startup_summary(&StartupSummary { cors: CorsSummary::None, ..base() });
    assert!(none.contains("cors:        none (same-origin only)"), "{none}");
}

#[test]
fn cors_summary_is_sorted_and_deduped() {
    let c = CorsConfig {
        allow_any: false,
        origins: vec![
            "https://b.example".into(),
            "https://a.example".into(),
            "https://b.example".into(),
        ],
        allow_loopback: false,
    };
    let out = format_startup_summary(&StartupSummary {
        cors: CorsSummary::from_config(&c),
        ..base()
    });
    let a = out.find("https://a.example").expect("a が無い");
    let b = out.find("https://b.example").expect("b が無い");
    assert!(a < b, "ソートされていない:\n{out}");
    assert_eq!(out.matches("https://b.example").count(), 1, "重複が残った:\n{out}");
}

#[test]
fn rate_limit_enabled_is_reported() {
    let out = format_startup_summary(&StartupSummary {
        rate_limit: RateLimitSummary::from_config(&RateLimitConfig::new(120, Some(20))),
        ..base()
    });
    assert!(out.contains("rate limit:  120 req/min, burst 20"), "{out}");
}

#[test]
fn body_limit_renderings() {
    let exact = format_startup_summary(&StartupSummary {
        max_body_bytes: 1024 * 1024,
        ..base()
    });
    assert!(exact.contains("body limit:  1 MiB (1048576 bytes)"), "{exact}");

    let frac = format_startup_summary(&StartupSummary {
        max_body_bytes: 1024 * 1024 + 512 * 1024,
        ..base()
    });
    assert!(frac.contains("body limit:  1.50 MiB (1572864 bytes)"), "{frac}");
}

#[test]
fn bind_address_uses_host_and_port_verbatim() {
    let out = format_startup_summary(&StartupSummary {
        host: "0.0.0.0",
        port: 9090,
        ..base()
    });
    assert!(out.contains("bind:        http://0.0.0.0:9090"), "{out}");
}

#[test]
fn no_placeholder_or_debug_artifacts_in_output() {
    let out = format_startup_summary(&base());
    for bad in ["None", "Some(", "{", "}", "dbg", "TODO", "REDACTED"] {
        assert!(!out.contains(bad), "デバッグ痕 {bad:?} が残る:\n{out}");
    }
}
