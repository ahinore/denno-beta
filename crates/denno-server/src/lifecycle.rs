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

/// 停止時に進行中の生成を待つ上限。`DENNO_DRAIN_TIMEOUT_MS` (既定 10 s、
/// 0〜600000 にクランプ) で上書きできる。0 は「待たない」。
pub fn drain_timeout() -> Duration {
    let ms = std::env::var("DENNO_DRAIN_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS)
        .min(600_000);
    Duration::from_millis(ms)
}

/// `is_idle()` が true になるか `timeout` まで待つ。true = 空になった。
///
/// `poll` 間隔で `tokio::time::sleep` するので、非同期の停止経路 (axum の
/// graceful shutdown) からそのまま await できる。
pub async fn wait_for_idle<F>(is_idle: F, timeout: Duration, poll: Duration) -> bool
where
    F: Fn() -> bool,
{
    let start = Instant::now();
    loop {
        if is_idle() {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
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
        assert!(drain_timeout() <= Duration::from_secs(600));
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
