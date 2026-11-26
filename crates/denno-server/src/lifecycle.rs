//! P817: 運用の状態 (readiness / graceful drain)。
//!
//! `/ready` と停止経路が読む状態を 1 箇所に集める。どれも atomic 1 回の読みで、
//! 生成中のエンジン Mutex とは無関係 (制御プレーンを生成の直列化に巻き込まない —
//! P57 の方針と同じ)。
//!
//! 状態の意味:
//! - `ready`: モデルのロードが終わり、ワーカーが動き出した
//! - `worker_alive`: スケジューラのワーカースレッドが生きている (異常終了の検出)
//! - `draining`: SIGTERM/SIGINT を受け、新規の生成要求を 503 にしている
//!
//! `/ready` は `ready && worker_alive && !draining` のときだけ 200。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 停止時の待ちの既定値 (ms)。進行中の生成をこの時間だけ待ってから止める。
pub const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 10_000;
/// 停止時の待ちの既定値 (秒)。[`DEFAULT_DRAIN_TIMEOUT_MS`] と同じ 10 s。
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 10;
/// 停止時の待ちの上限 (秒)。暴走した env をそのまま信じない。
pub const MAX_DRAIN_TIMEOUT_SECS: u64 = 600;
/// task 1006: 停止時の待ちを **秒** で上書きする env。優先順はこの env が最上位。
pub const DRAIN_TIMEOUT_SECS_ENV: &str = "DENNO_DRAIN_TIMEOUT_SECS";
/// 停止時の待ちを **ms** で上書きする env (後方互換。secs が無いときだけ読む)。
pub const DRAIN_TIMEOUT_MS_ENV: &str = "DENNO_DRAIN_TIMEOUT_MS";
/// 待ちループのポーリング間隔 (ms)。
pub const DRAIN_POLL_MS: u64 = 50;

#[derive(Debug)]
pub struct Lifecycle {
    ready: AtomicBool,
    worker_alive: AtomicBool,
    draining: AtomicBool,
}

impl Default for Lifecycle {
    fn default() -> Self {
        // 既定は「未 ready・ワーカー生存・drain していない」。テストの fixture が
        // そのまま `/ready` の 503 経路を踏める。
        Self {
            ready: AtomicBool::new(false),
            worker_alive: AtomicBool::new(true),
            draining: AtomicBool::new(false),
        }
    }
}

impl Lifecycle {
    pub fn new(ready: bool) -> Arc<Self> {
        Arc::new(Self {
            ready: AtomicBool::new(ready),
            worker_alive: AtomicBool::new(true),
            draining: AtomicBool::new(false),
        })
    }

    /// モデルのロード完了を記録する (起動経路が 1 回だけ呼ぶ)。
    pub fn set_ready(&self, v: bool) {
        self.ready.store(v, Ordering::SeqCst);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// ワーカースレッドの生存。ワーカーは終了時に false を書く。
    pub fn worker_alive(&self) -> bool {
        self.worker_alive.load(Ordering::SeqCst)
    }

    pub fn mark_worker_dead(&self) {
        self.worker_alive.store(false, Ordering::SeqCst);
    }

    /// 停止要求を受けた (以後、新規の生成要求は 503)。
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::SeqCst);
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// 新規リクエストを受け付けてよいか (= `/ready` の 200 条件)。
    pub fn serves_requests(&self) -> bool {
        self.is_ready() && self.worker_alive() && !self.is_draining()
    }
}

/// 停止時に進行中の生成を待つ上限。優先順は
/// `DENNO_DRAIN_TIMEOUT_SECS` (秒) > `DENNO_DRAIN_TIMEOUT_MS` (ms) > 既定 10 s。
/// どちらも 600 s にクランプする。0 は「待たない」。
///
/// 純関数版 [`resolve_drain_timeout`] に env の生値を渡すだけなので、
/// テストは env を触らずに優先順とクランプを固定できる。
pub fn drain_timeout() -> Duration {
    let secs = std::env::var(DRAIN_TIMEOUT_SECS_ENV).ok();
    let ms = std::env::var(DRAIN_TIMEOUT_MS_ENV).ok();
    resolve_drain_timeout(secs.as_deref(), ms.as_deref())
}

/// task 1006: 2 つの env 文字列から実効の drain 上限を決める純関数。
///
/// - `secs` が非負整数として解析できればそれを採用 (最優先)。
/// - そうでなければ `ms` を採用。
/// - どちらも無効なら既定 [`DEFAULT_DRAIN_TIMEOUT_SECS`]。
///
/// 空白は無視し、負数・小数・空文字は「未設定」と同じ扱い。上限は
/// [`MAX_DRAIN_TIMEOUT_SECS`] 秒。0 は明示的な「待たない」として保持する。
pub fn resolve_drain_timeout(secs: Option<&str>, ms: Option<&str>) -> Duration {
    if let Some(s) = secs.and_then(parse_u64) {
        return Duration::from_secs(s.min(MAX_DRAIN_TIMEOUT_SECS));
    }
    if let Some(v) = ms.and_then(parse_u64) {
        return Duration::from_millis(v.min(MAX_DRAIN_TIMEOUT_SECS * 1_000));
    }
    Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
}

/// 空白を除いて非負整数として読む。解析できなければ None。
fn parse_u64(raw: &str) -> Option<u64> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<u64>().ok()
}

/// task 1006: 1 回のポーリング後の drain 判定。純関数なので、
/// 実時間を待たずに 3 分岐すべてを決定的に踏める。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStep {
    /// 進行中が 0 — 待たずに完了。
    Idle,
    /// まだ進行中で上限未満 — 次のポーリングへ。
    Wait,
    /// 上限に到達 — 打ち切る。
    TimedOut,
}

impl DrainStep {
    /// この判定で drain の待ちが終わるか (完了 or 打ち切り)。
    pub fn is_final(self) -> bool {
        !matches!(self, DrainStep::Wait)
    }
}

/// task 1006: 判定の 1 段。`is_idle()` が真なら常に [`DrainStep::Idle`]
/// (上限 0 でもアイドルは成功)。そうでなければ `elapsed >= timeout` で打ち切り。
pub fn drain_step(is_idle: bool, elapsed: Duration, timeout: Duration) -> DrainStep {
    if is_idle {
        DrainStep::Idle
    } else if elapsed >= timeout {
        DrainStep::TimedOut
    } else {
        DrainStep::Wait
    }
}

/// `is_idle()` が true になるか `timeout` まで待つ。true = 空になった。
///
/// `poll` 間隔で `tokio::time::sleep` するので、非同期の停止経路 (axum の
/// graceful shutdown) からそのまま await できる。判定は [`drain_step`] に
/// 集約してあり、この関数は「判定 -> 待つ」を回すだけ。
pub async fn wait_for_idle<F>(is_idle: F, timeout: Duration, poll: Duration) -> bool
where
    F: Fn() -> bool,
{
    let start = Instant::now();
    loop {
        match drain_step(is_idle(), start.elapsed(), timeout) {
            DrainStep::Idle => return true,
            DrainStep::TimedOut => return false,
            DrainStep::Wait => {}
        }
        let left = timeout.saturating_sub(start.elapsed());
        tokio::time::sleep(poll.min(left)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn readiness_requires_ready_alive_and_not_draining() {
        let l = Lifecycle::new(true);
        assert!(l.serves_requests());
        l.begin_drain();
        assert!(!l.serves_requests());
        assert!(l.is_draining());

        let l = Lifecycle::new(false);
        assert!(!l.serves_requests());
        l.set_ready(true);
        assert!(l.serves_requests());
        l.mark_worker_dead();
        assert!(!l.serves_requests());
        assert!(!l.worker_alive());
    }

    #[test]
    fn drain_timeout_default_is_bounded() {
        // env を汚さない範囲で既定値の範囲だけ確認する (値そのものは env 依存)。
        assert!(drain_timeout() <= Duration::from_secs(MAX_DRAIN_TIMEOUT_SECS));
    }

    // --- task 1006: DENNO_DRAIN_TIMEOUT_SECS の解決 (純関数・決定的) ---

    #[test]
    fn resolve_drain_timeout_prefers_secs_over_ms() {
        // secs が最優先。ms があっても無視する。
        assert_eq!(
            resolve_drain_timeout(Some("3"), Some("9000")),
            Duration::from_secs(3)
        );
        assert_eq!(
            resolve_drain_timeout(Some(" 7 "), None),
            Duration::from_secs(7)
        );
    }

    #[test]
    fn resolve_drain_timeout_falls_back_to_ms_then_default() {
        // secs が未設定 or 解析不能なら ms。
        assert_eq!(
            resolve_drain_timeout(None, Some("1500")),
            Duration::from_millis(1500)
        );
        assert_eq!(
            resolve_drain_timeout(Some("abc"), Some("250")),
            Duration::from_millis(250)
        );
        // どちらも無効なら既定 10 s。
        assert_eq!(resolve_drain_timeout(None, None), Duration::from_secs(10));
        assert_eq!(
            resolve_drain_timeout(Some(""), Some("  ")),
            Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
        );
        // 負数・小数は未設定扱い (u64 として解析できない)。
        assert_eq!(
            resolve_drain_timeout(Some("-1"), Some("2.5")),
            Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
        );
    }

    #[test]
    fn resolve_drain_timeout_clamps_and_keeps_zero() {
        // 上限 600 s にクランプ (secs 経路と ms 経路の両方)。
        assert_eq!(
            resolve_drain_timeout(Some("999999"), None),
            Duration::from_secs(MAX_DRAIN_TIMEOUT_SECS)
        );
        assert_eq!(
            resolve_drain_timeout(None, Some("999999999")),
            Duration::from_millis(MAX_DRAIN_TIMEOUT_SECS * 1_000)
        );
        // 0 は明示的な「待たない」として保持する (既定に化けない)。
        assert_eq!(resolve_drain_timeout(Some("0"), None), Duration::ZERO);
        assert_eq!(resolve_drain_timeout(None, Some("0")), Duration::ZERO);
        // u64 の上限を超える文字列は解析不能 -> ms/既定へ落ちる。
        assert_eq!(
            resolve_drain_timeout(Some("18446744073709551616"), None),
            Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
        );
    }

    #[test]
    fn drain_step_covers_all_three_branches_deterministically() {
        let timeout = Duration::from_millis(100);
        let mut idle = 0usize;
        let mut wait = 0usize;
        let mut timed_out = 0usize;
        for (is_idle, elapsed_ms, want) in [
            (true, 0u64, DrainStep::Idle),
            (true, 100, DrainStep::Idle), // 上限ちょうどでもアイドルが勝つ
            (false, 0, DrainStep::Wait),
            (false, 99, DrainStep::Wait),
            (false, 100, DrainStep::TimedOut), // 境界: >= で打ち切り
            (false, 101, DrainStep::TimedOut),
        ] {
            let got = drain_step(is_idle, Duration::from_millis(elapsed_ms), timeout);
            assert_eq!(
                got, want,
                "drain_step(idle={is_idle}, elapsed={elapsed_ms}ms)"
            );
            assert_eq!(got.is_final(), want != DrainStep::Wait);
            match got {
                DrainStep::Idle => idle += 1,
                DrainStep::Wait => wait += 1,
                DrainStep::TimedOut => timed_out += 1,
            }
        }
        // 3 分岐すべてが発火したことを数える (緑 = 分岐が踏まれた証拠)。
        assert_eq!((idle, wait, timed_out), (2, 2, 2));

        // timeout = 0 はアイドルでなければ即打ち切り。
        assert_eq!(
            drain_step(false, Duration::ZERO, Duration::ZERO),
            DrainStep::TimedOut
        );
        assert_eq!(
            drain_step(true, Duration::ZERO, Duration::ZERO),
            DrainStep::Idle
        );
    }

    #[tokio::test]
    async fn wait_for_idle_with_zero_timeout_does_not_wait() {
        let start = Instant::now();
        assert!(!wait_for_idle(|| false, Duration::ZERO, Duration::from_millis(5)).await);
        assert!(start.elapsed() < Duration::from_millis(500));
        // アイドルなら上限 0 でも成功。
        assert!(wait_for_idle(|| true, Duration::ZERO, Duration::from_millis(5)).await);
    }

    #[tokio::test]
    async fn wait_for_idle_returns_immediately_when_idle() {
        let start = Instant::now();
        assert!(wait_for_idle(|| true, Duration::from_secs(5), Duration::from_millis(10)).await);
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn wait_for_idle_times_out_when_never_idle() {
        let start = Instant::now();
        assert!(!wait_for_idle(|| false, Duration::from_millis(80), Duration::from_millis(10)).await);
        assert!(start.elapsed() >= Duration::from_millis(80));
    }

    #[tokio::test]
    async fn wait_for_idle_returns_true_when_work_finishes() {
        let n = AtomicUsize::new(3);
        let f = || {
            if n.load(Ordering::SeqCst) == 0 {
                true
            } else {
                n.fetch_sub(1, Ordering::SeqCst);
                false
            }
        };
        assert!(wait_for_idle(f, Duration::from_secs(5), Duration::from_millis(5)).await);
    }
}
