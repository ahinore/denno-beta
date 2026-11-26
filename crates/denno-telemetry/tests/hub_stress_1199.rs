//! task 1199: denno-telemetry 送出ハブのバックプレッシャ / ドロップ・Gap 計上 /
//! 複数購読者のストレステスト。
//!
//! 検証すること:
//! - [`telem::channel_cap`] ちょうどまで溜まり、超過分は `try_send` の失敗として
//!   `dropped()` に計上される (`emit` は待たない)
//! - 落ちた枚数と `Gap { lost }` の合計が一致する (relay と同じ規則で作った Gap)
//! - 遅い購読者・満杯状態でも `emit` はブロックせず、受信 + ドロップ = 送出 になる
//! - 並行 publisher で panic せず、producer ごとの順序が保たれる
//! - 複数購読者 (`recv_global_timeout` の競合) でフレームがちょうど 1 回ずつ届く
//! - 無効時はフレームが出ず、ドロップ計上にも触らない
//!
//! ハブはプロセス全体で 1 本 (static) なので、このファイルのテストは
//! [`serial`] の Mutex で直列化する。各テストは開始時に溜まりを捨て、
//! `dropped()` は常に差分で見る。

use denno_telemetry as telem;
use denno_telemetry::{ClientsGuard, Frame};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// グローバルハブを共有するテストを直列化する。
static SERIAL: Mutex<()> = Mutex::new(());

/// 直列化ロック。前のテストが panic しても (poison でも) 取得できる。
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// ハブに溜まっているフレームを全部取り出す (非ブロッキング)。
fn drain_all() -> Vec<Frame> {
    let mut out = Vec::new();
    while let Some(f) = telem::try_recv_nowait() {
        out.push(f);
    }
    out
}

/// テスト用にハブを有効化し、前のテストの残りを捨てる。
fn setup() {
    telem::set_enabled_for_tests(true);
    assert!(telem::enabled(), "テレメトリを有効化できていない");
    let _ = drain_all();
}

fn token(session: u32, token: u32, pos: u32) -> Frame {
    Frame::Token { session, token, pos }
}

/// 容量ちょうどまで溜まり、超過分だけがドロップ計上されること。
#[test]
fn channel_cap_fill_then_overflow_is_counted() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    assert!(cap >= 64, "capacity が小さすぎる: {cap}");

    let base = telem::dropped();
    for i in 0..cap {
        telem::emit(token(1, i as u32, i as u32));
    }
    assert_eq!(telem::dropped(), base, "容量ちょうどまでは落ちない");

    // 1 枚だけ溢れさせる。
    telem::emit(Frame::Gap { lost: 1 });
    assert_eq!(telem::dropped(), base + 1, "容量超過の 1 枚がドロップ計上される");

    let got = drain_all();
    assert_eq!(got.len(), cap, "受信数は容量ちょうど (溢れた 1 枚は入っていない)");
    // 溢れた Gap は捨てられ、届いた分は古い順 (単一チャネルの FIFO)。
    let poss: Vec<u32> = got
        .iter()
        .map(|f| match f {
            Frame::Token { pos, .. } => *pos,
            other => panic!("想定外のフレーム: {other:?}"),
        })
        .collect();
    assert_eq!(poss.first().copied(), Some(0));
    assert_eq!(poss.last().copied(), Some(cap as u32 - 1));
    for (a, b) in poss.iter().zip(poss.iter().skip(1)) {
        assert_eq!(*b, a + 1, "受信順が崩れた");
    }
    assert_eq!(telem::dropped(), base + 1, "ドレインでドロップ計上は増えない");
}

/// 落ちた枚数の合計と `Gap { lost }` の合計が窓ごとに一致すること。
#[test]
fn gap_frames_account_for_every_dropped_frame() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    let mut total_lost = 0u64;
    let mut total_sent = 0u64;

    // relay (flash_telemetry) と同じ規則: 窓ごとの dropped 差分を 1 枚の Gap にする。
    for window in [5usize, 17, 0, 64] {
        let _ = drain_all();
        let before = telem::dropped();
        let send = cap + window;
        for i in 0..send {
            telem::emit(token(2, i as u32, i as u32));
        }
        let lost = telem::dropped() - before;
        assert_eq!(lost, window as u64, "window {window}: 超過分だけ落ちる");
        total_lost += lost;
        total_sent += send as u64;

        // Gap はハブが勝手に作らない (購読側の責務)。ここで作って往復させる。
        let gap = Frame::Gap { lost: lost.min(u32::MAX as u64) as u32 };
        assert_eq!(gap.kind(), telem::kind::GAP);
        let dec = Frame::decode(&gap.encode()).expect("Gap は復号できること");
        assert_eq!(dec, gap);

        let got = drain_all();
        assert_eq!(got.len(), cap, "受信は容量ぶん");
        assert!(
            got.iter().all(|f| !matches!(f, Frame::Gap { .. })),
            "ハブが Gap を合成している (購読側の責務のはず)"
        );
        // 送った枚数 = 受信 + ドロップ (窓ごと)
        assert_eq!(got.len() as u64 + lost, send as u64);
    }

    assert_eq!(total_lost, 5 + 17 + 0 + 64);
    assert_eq!(total_sent, 4 * cap as u64 + total_lost, "会計が合わない");
}

/// `Gap { lost }` の境界値が encode → decode で保存されること。
#[test]
fn gap_lost_roundtrips_at_boundaries() {
    let _s = serial();
    setup();
    for lost in [0u32, 1, 127, 128, 16_383, 16_384, u32::MAX] {
        let f = Frame::Gap { lost };
        let enc = f.encode();
        // magic(4) + version(1) + kind(1) の後ろが varint。
        assert_eq!(&enc[..4], &telem::MAGIC);
        assert_eq!(enc[4], telem::VERSION);
        assert_eq!(enc[5], telem::kind::GAP);
        let dec = Frame::decode(&enc).expect("Gap の復号");
        assert_eq!(dec, f, "lost = {lost} で往復しない");
    }
    // 切り詰めた Gap は panic せず Err。
    let enc = Frame::Gap { lost: u32::MAX }.encode();
    assert!(Frame::decode(&enc[..enc.len() - 1]).is_err(), "切り詰めは Err");
}

/// 満杯状態でも `emit` は待たない (ブロックしない)。
#[test]
fn emit_never_blocks_when_channel_is_full() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    for i in 0..cap {
        telem::emit(token(3, i as u32, i as u32));
    }
    let base = telem::dropped();

    const EXTRA: u64 = 50_000;
    let t = Instant::now();
    for i in 0..EXTRA {
        telem::emit(token(3, i as u32, i as u32));
    }
    let elapsed = t.elapsed();
    assert_eq!(telem::dropped() - base, EXTRA, "満杯時の emit は全部ドロップ計上");
    assert!(
        elapsed < Duration::from_secs(2),
        "満杯時に emit が待っている (推定: {elapsed:?})"
    );
    assert_eq!(drain_all().len(), cap);
}

/// 遅い購読者: 読むそばから溢れても、受信 + ドロップ = 送出 が保たれる。
#[test]
fn slow_subscriber_conserves_frames() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    let total = cap as u64 * 3;

    let d0 = telem::dropped();
    let mut received = 0u64;
    let mut sent = 0u64;
    while sent < total {
        let burst = 100u64.min(total - sent);
        for i in 0..burst {
            telem::emit(token(4, i as u32, i as u32));
        }
        sent += burst;
        // 購読者は burst より遅い: 7 枚だけ読む。
        for _ in 0..7 {
            if telem::try_recv_nowait().is_some() {
                received += 1;
            }
        }
        std::thread::sleep(Duration::from_micros(50));
    }
    received += drain_all().len() as u64;
    let dropped = telem::dropped() - d0;

    assert!(dropped > 0, "満杯に達していない (テストが弱い)");
    assert_eq!(received + dropped, total, "受信 + ドロップ = 送出");
    assert!(received > 0, "1 枚も届いていない");
    assert_eq!(drain_all().len(), 0, "ドレイン後に残っている");
}

/// 並行 publisher: panic せず、producer ごとの順序が保たれ、会計が合う。
#[test]
fn concurrent_publishers_no_panic_and_accounting() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    const PRODUCERS: u32 = 8;
    let per: u32 = (cap as u32 * 3) / PRODUCERS;
    let total = (per * PRODUCERS) as u64;

    let d0 = telem::dropped();
    let done = Arc::new(AtomicBool::new(false));
    let got: Arc<Mutex<Vec<Frame>>> = Arc::new(Mutex::new(Vec::new()));

    std::thread::scope(|s| {
        let consumer = {
            let done = Arc::clone(&done);
            let got = Arc::clone(&got);
            s.spawn(move || {
                loop {
                    match telem::recv_global_timeout(Duration::from_millis(5)) {
                        Some(f) => got.lock().unwrap_or_else(|e| e.into_inner()).push(f),
                        None => {
                            if done.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    }
                }
            })
        };

        let mut pubs = Vec::new();
        for pid in 0..PRODUCERS {
            pubs.push(s.spawn(move || {
                for seq in 0..per {
                    telem::emit(token(100 + pid, seq, seq));
                }
            }));
        }
        for p in pubs {
            p.join().expect("publisher が panic した");
        }
        done.store(true, Ordering::Relaxed);
        consumer.join().expect("consumer が panic した");
    });

    // 最後に残りを吸い出す (consumer は done 後に None で抜けている)。
    let mut frames = Arc::try_unwrap(got)
        .ok()
        .expect("Arc が残っている")
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    frames.extend(drain_all());

    let dropped = telem::dropped() - d0;
    assert_eq!(frames.len() as u64 + dropped, total, "受信 + ドロップ = 送出");
    assert!(dropped > 0, "満杯にならなかった (テストが弱い)");

    // producer ごとの順序 (同じ session の pos は昇順)。
    let mut last: Vec<Option<u32>> = vec![None; PRODUCERS as usize];
    for f in &frames {
        if let Frame::Token { session, pos, .. } = f {
            let idx = (session - 100) as usize;
            if let Some(prev) = last[idx] {
                assert!(pos > &prev, "producer {idx} の順序が崩れた: {prev} -> {pos}");
            }
            last[idx] = Some(*pos);
        } else {
            panic!("想定外のフレーム: {f:?}");
        }
    }
}

/// 複数購読者: recv_global_timeout を奪い合っても各フレームはちょうど 1 回届く。
#[test]
fn multi_subscriber_delivers_each_frame_exactly_once() {
    let _s = serial();
    setup();
    const SUBS: usize = 4;
    const TOTAL: usize = 500;

    let clients = ClientsGuard::new(SUBS);
    assert_eq!(telem::clients(), SUBS, "購読者数が記録されていない");
    assert!(telem::has_clients(), "購読者が居るのに has_clients() が false");

    let d0 = telem::dropped();
    for i in 0..TOTAL {
        telem::emit(token(9, i as u32, i as u32));
    }
    assert_eq!(telem::dropped(), d0, "容量内なのに落ちた");

    let stop = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicUsize::new(0));
    let toks: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

    std::thread::scope(|s| {
        let mut subs = Vec::new();
        for _ in 0..SUBS {
            let stop = Arc::clone(&stop);
            let count = Arc::clone(&count);
            let toks = Arc::clone(&toks);
            subs.push(s.spawn(move || loop {
                match telem::recv_global_timeout(Duration::from_millis(5)) {
                    Some(Frame::Token { pos, .. }) => {
                        toks.lock().unwrap_or_else(|e| e.into_inner()).push(pos);
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(other) => panic!("想定外のフレーム: {other:?}"),
                    None => {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                }
            }));
        }

        // 全員が受け取り終えるまで待つ (最長 10 秒)。
        let deadline = Instant::now() + Duration::from_secs(10);
        while count.load(Ordering::Relaxed) < TOTAL {
            assert!(Instant::now() < deadline, "購読者がフレームを受け取れない");
            std::thread::sleep(Duration::from_millis(1));
        }
        stop.store(true, Ordering::Relaxed);
        for h in subs {
            h.join().expect("購読者スレッドが panic した");
        }
    });

    let got = Arc::try_unwrap(toks).ok().expect("Arc が残っている");
    let got = got.into_inner().unwrap_or_else(|e| e.into_inner());
    assert_eq!(got.len(), TOTAL, "受信枚数が合わない");
    let uniq: HashSet<u32> = got.iter().copied().collect();
    assert_eq!(uniq.len(), TOTAL, "同じフレームが二重に届いた");
    assert!(uniq.iter().all(|p| (*p as usize) < TOTAL), "範囲外の pos");
    assert_eq!(telem::dropped(), d0, "購読でドロップ計上が動いた");

    drop(clients);
    assert_eq!(telem::clients(), 0, "ClientsGuard が戻していない");
    let g0 = ClientsGuard::new(0);
    assert_eq!(telem::has_clients(), !telem::lazy_enabled(), "0 人時の判定");
    drop(g0);
}

/// 無効時はフレームが出ず、ドロップ計上にも触らない。
#[test]
fn disabled_hub_emits_nothing_and_counts_nothing() {
    let _s = serial();
    setup();
    let d0 = telem::dropped();
    telem::set_enabled_for_tests(false);
    assert!(!telem::enabled());

    let t = Instant::now();
    for i in 0..(telem::channel_cap() * 2) {
        telem::emit(token(7, i as u32, i as u32));
    }
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "無効時の emit が遅い (推定: {:?})",
        t.elapsed()
    );
    assert_eq!(telem::dropped(), d0, "無効時にドロップ計上が動いた");
    assert!(telem::try_recv_nowait().is_none(), "無効時にフレームが出た");

    telem::set_enabled_for_tests(true);
}
