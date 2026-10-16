//! s37: スケジューラ ワーカーのアイドル busy-spin 回帰テスト。
//!
//! 旧実装のワーカーループはブロック無しで回る設計で、active 空 + キュー空でも
//! try_recv + engine Mutex + 空 step_chat_batch を全速で回し続けていたため、
//! アイドル中の denno-scheduler スレッドが 100% CPU を恒久消費していた
//! (実機 8095 の断発ハング調査で特定・P17k コメントに「秒間数十万回」の記録)。
//! 修正後は active 空 + キュー空で rx.recv() にブロックするため、アイドル中の
//! スレッド CPU 時間は増えない。
//!
//! 検出方法: /proc/self/task/<tid>/stat の utime+stime (clock tick) をアイドル
//! 500ms の前後で読み、増分が閾値未満であることを assert する。同名スレッドの
//! 混入を防ぐため単独テストバイナリ (このファイル) にする (cargo test は
//! テストファイル単位で別プロセスになる)。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::Scheduler;

/// 指定名スレッドの utime+stime 合計 (clock tick・多くの環境で 1 tick = 10ms)。
#[cfg(target_os = "linux")]
fn thread_cpu_ticks(name: &str) -> Option<u64> {
    let dir = std::fs::read_dir("/proc/self/task").ok()?;
    for entry in dir.flatten() {
        let comm = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
        if comm.trim_end_matches('\n') == name {
            let stat = std::fs::read_to_string(entry.path().join("stat")).ok()?;
            // comm に ")" が含まれ得るため最後の ") " で分割する
            let rest = stat.rsplit_once(") ")?.1;
            let f: Vec<&str> = rest.split_whitespace().collect();
            // rest[0] は 3 番フィールド (state) → utime は 14 番 = rest[11]
            let utime: u64 = f.get(11)?.parse().ok()?;
            let stime: u64 = f.get(12)?.parse().ok()?;
            return Some(utime + stime);
        }
    }
    None
}

#[cfg(target_os = "linux")]
#[test]
fn scheduler_worker_does_not_spin_when_idle() {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    // Scheduler を生かし続ける (drop すると Sender が閉じて worker が終了する)
    let _sched = Scheduler::spawn(Arc::new(Mutex::new(engine)), 4);

    // ワーカーが recv ブロックに到達するのを待つ
    std::thread::sleep(Duration::from_millis(100));
    let t0 = thread_cpu_ticks("denno-scheduler").expect("denno-scheduler thread not found");
    std::thread::sleep(Duration::from_millis(500));
    let t1 = thread_cpu_ticks("denno-scheduler").expect("denno-scheduler thread not found");

    // 旧実装 (busy-spin): 500ms のアイドルで ~50 tick (10ms/tick) 消費。
    // 修正後 (recv ブロック): ほぼ 0。負荷揺れのマージンとして 20 tick = 200ms を許す。
    let delta = t1.saturating_sub(t0);
    assert!(
        delta < 20,
        "scheduler worker is busy-spinning while idle: +{delta} ticks in 500ms"
    );
}
