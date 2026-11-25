//! task 971: 起動サマリの整形。
//!
//! serve の起動時に「どの設定で listening したか」を 1 か所にまとめて出す。
//! 整形は **純関数** ([`format_startup_summary`]) で、env も時計も読まない。
//! 呼び出し側 (`run`) が解決済みの値を渡す。
//!
//! 秘密の方針: API キーの **値は型として持たない**。認証は
//! [`AuthState`] (無効 / 有効 + 本数) だけを渡すので、整形結果に生のキーが
//! 混入する経路が構造的に存在しない。CORS の許可オリジンとモデル ID だけは
//! 運用に必要な設定値としてそのまま出す。

use crate::ratelimit::RateLimitConfig;
use crate::CorsConfig;

/// 認証の状態。生のキーは持たない (本数だけ)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthState {
    /// キー未設定 = 従来どおり開放。
    Disabled,
    /// 1 本以上設定済み。`key_count` は実効キーの本数。
    Enabled { key_count: usize },
}

/// カンマ区切りの実効キー一覧から [`AuthState`] を作る。
/// 空白のみの要素と重複は数えない ([`crate::parse_api_keys`] と同じ規則)。
pub fn summarize_auth(raw: Option<&str>) -> AuthState {
    match raw {
        None => AuthState::Disabled,
        Some(s) => {
            let n = crate::parse_api_keys(s).len();
            if n == 0 {
                AuthState::Disabled
            } else {
                AuthState::Enabled { key_count: n }
            }
        }
    }
}

/// CORS の要約。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorsSummary {
    /// `*` (任意オリジン)。
    Any,
    /// 明示オリジンの一覧 (+ ループバック既定の有無)。
    Allowlist { origins: Vec<String>, loopback: bool },
    /// ループバックのみ。
    LoopbackOnly,
    /// クロスオリジンは全て拒否。
    None,
}

impl CorsSummary {
    /// 解決済みの [`CorsConfig`] から要約する。
    pub fn from_config(c: &CorsConfig) -> Self {
        if c.allow_any {
            return CorsSummary::Any;
        }
        if !c.origins.is_empty() {
            let mut origins = c.origins.clone();
            origins.sort();
            origins.dedup();
            return CorsSummary::Allowlist { origins, loopback: c.allow_loopback };
        }
        if c.allow_loopback {
            CorsSummary::LoopbackOnly
        } else {
            CorsSummary::None
        }
    }
}

/// レート制限の要約 (`RateLimitConfig::DISABLED` は無効)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitSummary {
    Disabled,
    Enabled { rpm: u32, burst: u32 },
}

impl RateLimitSummary {
    pub fn from_config(c: &RateLimitConfig) -> Self {
        if c.is_enabled() {
            RateLimitSummary::Enabled { rpm: c.rpm, burst: c.burst }
        } else {
            RateLimitSummary::Disabled
        }
    }
}

/// 整形に必要な解決済みの起動設定。文字列は全て借用で持つ (コピーしない)。
#[derive(Debug, Clone)]
pub struct StartupSummary<'a> {
    /// listen する host (`--host` / 既定 127.0.0.1)。
    pub host: &'a str,
    pub port: u16,
    /// pull 元の repo id (`/v1/models` の id)。
    pub model_id: &'a str,
    /// エンジンが名乗るモデル名。
    pub model_name: &'a str,
    /// コンテキスト長 (トークン)。
    pub context_tokens: usize,
    /// KV キャッシュのスロット数。
    pub kv_slots: usize,
    /// 量子化ラベル。
    pub quant: &'a str,
    /// 層数。
    pub layers: usize,
    pub auth: AuthState,
    pub metrics_enabled: bool,
    /// 実効の待ち行列上限。
    pub queue_depth: usize,
    /// 実効のリクエスト期限 (秒)。0 = 無効。
    pub request_timeout_secs: u64,
    pub cors: CorsSummary,
    /// 実効のリクエストボディ上限 (byte)。
    pub max_body_bytes: usize,
    pub rate_limit: RateLimitSummary,
}

/// ボディ上限の表示。MiB で割り切れる場合は整数、割り切れない場合は小数 2 桁。
fn body_limit_label(bytes: usize) -> String {
    const MIB: usize = 1024 * 1024;
    if bytes % MIB == 0 {
        format!("{} MiB ({} bytes)", bytes / MIB, bytes)
    } else {
        format!("{:.2} MiB ({} bytes)", bytes as f64 / MIB as f64, bytes)
    }
}

fn timeout_label(secs: u64) -> String {
    if secs == 0 {
        "off".to_string()
    } else {
        format!("{secs}s")
    }
}

/// 起動サマリを複数行の文字列へ整形する純関数。
///
/// 出力は決定的 (同じ入力なら同じ文字列)。生の API キーは引数に存在しない。
pub fn format_startup_summary(s: &StartupSummary<'_>) -> String {
    let mut out = String::new();
    out.push_str("denno-server startup\n");
    out.push_str(&format!("  bind:        http://{}:{}\n", s.host, s.port));
    out.push_str(&format!(
        "  model:       {} (id: {})\n",
        s.model_name, s.model_id
    ));
    out.push_str(&format!(
        "  engine:      {} layers, quant {}\n",
        s.layers, s.quant
    ));
    out.push_str(&format!(
        "  max context: {} tokens\n",
        s.context_tokens
    ));
    out.push_str(&format!("  kv slots:    {}\n", s.kv_slots));
    match s.auth {
        AuthState::Disabled => {
            out.push_str("  auth:        disabled (no api key configured)\n")
        }
        AuthState::Enabled { key_count } => out.push_str(&format!(
            "  auth:        enabled ({key_count} key(s), values hidden)\n"
        )),
    }
    out.push_str(&format!(
        "  metrics:     {}\n",
        if s.metrics_enabled { "enabled" } else { "disabled" }
    ));
    out.push_str(&format!(
        "  queue:       depth={}, request timeout={}\n",
        s.queue_depth,
        timeout_label(s.request_timeout_secs)
    ));
    match s.rate_limit {
        RateLimitSummary::Disabled => out.push_str("  rate limit:  disabled\n"),
        RateLimitSummary::Enabled { rpm, burst } => out.push_str(&format!(
            "  rate limit:  {rpm} req/min, burst {burst}\n"
        )),
    }
    match &s.cors {
        CorsSummary::Any => out.push_str("  cors:        * (any origin)\n"),
        CorsSummary::LoopbackOnly => out.push_str("  cors:        loopback only\n"),
        CorsSummary::None => out.push_str("  cors:        none (same-origin only)\n"),
        CorsSummary::Allowlist { origins, loopback } => {
            out.push_str(&format!("  cors:        allowlist {:?}", origins));
            if *loopback {
                out.push_str(" + loopback");
            }
            out.push('\n');
        }
    }
    out.push_str(&format!("  body limit:  {}\n", body_limit_label(s.max_body_bytes)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample<'a>() -> StartupSummary<'a> {
        StartupSummary {
            host: "127.0.0.1",
            port: 8080,
            model_id: "org/model",
            model_name: "model",
            context_tokens: 8192,
            kv_slots: 2,
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

    #[test]
    fn auth_summary_counts_unique_non_empty_keys() {
        assert_eq!(summarize_auth(None), AuthState::Disabled);
        assert_eq!(summarize_auth(Some("")), AuthState::Disabled);
        assert_eq!(summarize_auth(Some(" , , ")), AuthState::Disabled);
        assert_eq!(
            summarize_auth(Some("a, b ,a,")),
            AuthState::Enabled { key_count: 2 }
        );
    }

    #[test]
    fn body_limit_renders_integer_and_fractional_mib() {
        assert_eq!(body_limit_label(16 * 1024 * 1024), "16 MiB (16777216 bytes)");
        assert_eq!(body_limit_label(1024 * 1024 + 512 * 1024), "1.50 MiB (1572864 bytes)");
    }

    #[test]
    fn timeout_and_auth_labels() {
        assert_eq!(timeout_label(0), "off");
        assert_eq!(timeout_label(30), "30s");
    }

    #[test]
    fn formatting_is_deterministic() {
        let s = sample();
        assert_eq!(format_startup_summary(&s), format_startup_summary(&s));
    }

    #[test]
    fn cors_summary_from_config_covers_all_shapes() {
        let any = CorsConfig { allow_any: true, origins: vec![], allow_loopback: false };
        assert_eq!(CorsSummary::from_config(&any), CorsSummary::Any);
        let lb = CorsConfig { allow_any: false, origins: vec![], allow_loopback: true };
        assert_eq!(CorsSummary::from_config(&lb), CorsSummary::LoopbackOnly);
        let none = CorsConfig { allow_any: false, origins: vec![], allow_loopback: false };
        assert_eq!(CorsSummary::from_config(&none), CorsSummary::None);
        let list = CorsConfig {
            allow_any: false,
            origins: vec!["https://b.example".into(), "https://a.example".into()],
            allow_loopback: false,
        };
        assert_eq!(
            CorsSummary::from_config(&list),
            CorsSummary::Allowlist {
                origins: vec!["https://a.example".into(), "https://b.example".into()],
                loopback: false
            }
        );
    }

    #[test]
    fn rate_limit_summary_from_config() {
        assert_eq!(
            RateLimitSummary::from_config(&RateLimitConfig::DISABLED),
            RateLimitSummary::Disabled
        );
        let on = RateLimitConfig::new(60, Some(10));
        assert_eq!(
            RateLimitSummary::from_config(&on),
            RateLimitSummary::Enabled { rpm: 60, burst: 10 }
        );
    }
}
