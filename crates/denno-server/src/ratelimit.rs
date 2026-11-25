//! task 931: 任意オプトイン (opt-in) のトークンバケット式レート制限。
//!
//! 目的は「単一クライアントの暴走でキューが埋まる」のを運用側で抑えること。
//! 既定は無効 (env 未設定 = 無制限) なので、フラグを付けない起動の挙動は
//! 従来と完全に同じ (レイテンシ・出力ビットとも不変)。
//!
//! - `DENNO_RATE_LIMIT_RPM` … 1 分あたりの許可数 (0 / 未設定 / 不正 = 無効)
//! - `DENNO_RATE_LIMIT_BURST` … バケット容量 (省略時は 1 秒ぶん = rpm/60、最低 1)
//!
//! 実装は std だけで、状態は `Mutex<{tokens, last}>` の短い臨界区間で持つ
//! (非同期ロックを跨がない = lock-light)。時刻は [`std::time::Instant`] を
//! 呼び出し側から渡せるので、単体テストは実時間に依存しない。
//!
//! 制限の対象は推論エンドポイントだけ ([`is_inference_path`])。/health /
//! /ready / /metrics は明示的に素通しする ([`RATE_LIMIT_BYPASS_PATHS`])。
//! CORS プリフライト (OPTIONS) は消費しない。

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// 1 分あたりの許可数を読む env 名。0 / 未設定 / 不正 = 無効。
pub const RATE_LIMIT_RPM_ENV: &str = "DENNO_RATE_LIMIT_RPM";

/// バケット容量 (バースト) を読む env 名。省略時は `rpm / 60` (最低 1)。
pub const RATE_LIMIT_BURST_ENV: &str = "DENNO_RATE_LIMIT_BURST";

/// 429 の `error.type` (OpenAI / Anthropic 互換のレート制限種別)。
pub const RATE_LIMIT_ERROR_TYPE: &str = "rate_limit_error";

/// 429 の `error.code`。
pub const RATE_LIMIT_ERROR_CODE: &str = "rate_limit_exceeded";

/// 制限の対象にする推論エンドポイント。ここに無いパスは消費しない。
pub const RATE_LIMIT_INFERENCE_PATHS: &[&str] = &[
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/completions",
    "/api/chat",
    "/api/generate",
];

/// 明示的に素通しするパス (監視・liveness 用)。
pub const RATE_LIMIT_BYPASS_PATHS: &[&str] = &["/health", "/ready", "/metrics"];

/// `path` が制限対象の推論エンドポイントか。
///
/// 末尾スラッシュ 1 個の差は同じ扱い (`/v1/messages/` == `/v1/messages`)。
pub fn is_inference_path(path: &str) -> bool {
    let p = path.strip_suffix('/').unwrap_or(path);
    if RATE_LIMIT_BYPASS_PATHS.contains(&p) {
        return false;
    }
    RATE_LIMIT_INFERENCE_PATHS.contains(&p)
}

/// `path` が素通し対象 (/health / /ready / /metrics) か。
pub fn is_bypass_path(path: &str) -> bool {
    let p = path.strip_suffix('/').unwrap_or(path);
    RATE_LIMIT_BYPASS_PATHS.contains(&p)
}

/// レート制限の設定。`rpm == 0` は無効。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitConfig {
    /// 1 分あたりの許可数。0 = 無効。
    pub rpm: u32,
    /// バケット容量 (瞬間的に許す数)。有効時は常に 1 以上。
    pub burst: u32,
}

impl RateLimitConfig {
    /// 無効 (既定)。
    pub const DISABLED: Self = Self { rpm: 0, burst: 0 };

    /// `rpm` と任意のバーストから設定を作る。`burst` が 0 / None なら既定値。
    pub fn new(rpm: u32, burst: Option<u32>) -> Self {
        if rpm == 0 {
            return Self::DISABLED;
        }
        let burst = match burst {
            Some(b) if b > 0 => b,
            _ => default_burst(rpm),
        };
        Self { rpm, burst }
    }

    /// 文字列 (env の中身) から解決する。テストが env を触らずに使える。
    pub fn parse(rpm: Option<&str>, burst: Option<&str>) -> Self {
        let rpm = rpm.and_then(parse_u32).unwrap_or(0);
        let burst = burst.and_then(parse_u32);
        Self::new(rpm, burst)
    }

    /// プロセスの env から解決する。
    pub fn from_env() -> Self {
        let rpm = std::env::var(RATE_LIMIT_RPM_ENV).ok();
        let burst = std::env::var(RATE_LIMIT_BURST_ENV).ok();
        Self::parse(rpm.as_deref(), burst.as_deref())
    }

    pub fn is_enabled(&self) -> bool {
        self.rpm > 0 && self.burst > 0
    }

    /// 1 秒あたりに回復するトークン数。
    pub fn per_second(&self) -> f64 {
        self.rpm as f64 / 60.0
    }
}

fn parse_u32(s: &str) -> Option<u32> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        t.parse::<u32>().ok()
    }
}

/// バースト省略時の既定: 1 秒ぶん (最低 1 トークン)。
fn default_burst(rpm: u32) -> u32 {
    std::cmp::max(1, rpm / 60)
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last: Instant,
}

/// 単一バケット。`try_acquire_*` は成功で `Ok(())`、拒否で `Retry-After` 秒。
#[derive(Debug)]
pub struct TokenBucket {
    config: RateLimitConfig,
    state: Mutex<BucketState>,
}

impl TokenBucket {
    /// 有効な設定なら `Some`、`rpm == 0` なら `None`。
    pub fn new(config: RateLimitConfig) -> Option<Self> {
        Self::new_at(config, Instant::now())
    }

    /// 起点時刻を指定して作る (テスト用)。
    pub fn new_at(config: RateLimitConfig, now: Instant) -> Option<Self> {
        if !config.is_enabled() {
            return None;
        }
        Some(Self {
            config,
            state: Mutex::new(BucketState { tokens: config.burst as f64, last: now }),
        })
    }

    pub fn config(&self) -> RateLimitConfig {
        self.config
    }

    /// 現在時刻で 1 トークン取る。
    pub fn try_acquire(&self) -> Result<(), u64> {
        self.try_acquire_at(Instant::now())
    }

    /// 指定時刻で 1 トークン取る。足りなければ `Retry-After` 秒を返す。
    ///
    /// 臨界区間はここだけで、待たない (sleep しない)。
    pub fn try_acquire_at(&self, now: Instant) -> Result<(), u64> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let elapsed = now.saturating_duration_since(st.last).as_secs_f64();
        if elapsed > 0.0 {
            let cap = self.config.burst as f64;
            st.tokens = (st.tokens + elapsed * self.config.per_second()).min(cap);
            st.last = now;
        }
        if st.tokens >= 1.0 {
            st.tokens -= 1.0;
            return Ok(());
        }
        let deficit = 1.0 - st.tokens;
        let secs = (deficit / self.config.per_second()).ceil();
        // 0 秒の Retry-After はクライアントを即時リトライのループに落とすので 1 に丸める。
        let secs = if secs.is_finite() && secs >= 1.0 { secs as u64 } else { 1 };
        Err(secs)
    }

    /// 時刻 `now` 時点の残トークン (テスト・診断用。状態は変えない)。
    pub fn available_tokens(&self, now: Instant) -> f64 {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let elapsed = now.saturating_duration_since(st.last).as_secs_f64();
        let cap = self.config.burst as f64;
        let mut tokens = st.tokens;
        if elapsed > 0.0 {
            tokens = (tokens + elapsed * self.config.per_second()).min(cap);
            st.tokens = tokens;
            st.last = now;
        }
        tokens
    }
}

/// 判定結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// 通す (無効 / 対象外 / トークンあり)。
    Allow,
    /// 429 を返す。`retry_after_secs` は 1 以上。
    Deny { retry_after_secs: u64 },
}

/// ルータに 1 個だけ持たせるレート制限器。`clone` しても同じ状態を共有する。
#[derive(Debug, Clone)]
pub struct RateLimiter {
    config: RateLimitConfig,
    bucket: Option<Arc<TokenBucket>>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::disabled()
    }
}

impl RateLimiter {
    /// 制限しない。
    pub fn disabled() -> Self {
        Self { config: RateLimitConfig::DISABLED, bucket: None }
    }

    /// 設定から作る (`rpm == 0` なら無効)。
    pub fn from_config(config: RateLimitConfig) -> Self {
        match TokenBucket::new(config) {
            Some(b) => Self { config, bucket: Some(Arc::new(b)) },
            None => Self::disabled(),
        }
    }

    /// 起点時刻を指定して作る (テスト用)。
    pub fn from_config_at(config: RateLimitConfig, now: Instant) -> Self {
        match TokenBucket::new_at(config, now) {
            Some(b) => Self { config, bucket: Some(Arc::new(b)) },
            None => Self::disabled(),
        }
    }

    /// プロセスの env から作る。
    pub fn from_env() -> Self {
        Self::from_config(RateLimitConfig::from_env())
    }

    pub fn is_enabled(&self) -> bool {
        self.bucket.is_some()
    }

    pub fn config(&self) -> RateLimitConfig {
        self.config
    }

    /// `path` へのリクエスト 1 件を数える。対象外パスは消費せずに通す。
    pub fn check(&self, path: &str) -> Decision {
        let bucket = match &self.bucket {
            Some(b) if is_inference_path(path) => b,
            _ => return Decision::Allow,
        };
        match bucket.try_acquire() {
            Ok(()) => Decision::Allow,
            Err(secs) => Decision::Deny { retry_after_secs: secs },
        }
    }

    /// 起点時刻を指定する [`RateLimiter::check`] (テスト用)。
    pub fn check_at(&self, path: &str, now: Instant) -> Decision {
        let bucket = match &self.bucket {
            Some(b) if is_inference_path(path) => b,
            _ => return Decision::Allow,
        };
        match bucket.try_acquire_at(now) {
            Ok(()) => Decision::Allow,
            Err(secs) => Decision::Deny { retry_after_secs: secs },
        }
    }
}

/// 429 の応答 (OpenAI 形式の JSON 封筒 + `Retry-After`)。
///
/// ボディ: `{"error":{"message","type":"rate_limit_error","param":null,
/// "code":"rate_limit_exceeded"}}`
pub fn too_many_requests_response(retry_after_secs: u64) -> Response {
    let secs = retry_after_secs.max(1);
    let body = json!({
        "error": {
            "message": format!(
                "Rate limit exceeded: retry after {secs}s (see {RATE_LIMIT_RPM_ENV})"
            ),
            "type": RATE_LIMIT_ERROR_TYPE,
            "param": null,
            "code": RATE_LIMIT_ERROR_CODE,
        }
    });
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, v);
    }
    resp
}
