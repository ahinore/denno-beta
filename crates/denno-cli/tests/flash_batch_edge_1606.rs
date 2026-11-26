//! task 1606: serve-flash の連続バッチング用スロットスケジューラ
//! (`src/flash_batch.rs`) の境界試験。
//!
//! denno-cli は bin クレートなので lib 経由では取り込めない。既存の
//! `tests/kv_radix_edge_1601.rs` と同じく `#[path]` で実ファイルをそのまま
//! 取り込み、実装そのものを叩く。
//!
//! 固定する契約:
//!
//! 1. `DENNO_FLASH_MAX_BATCH` の読み方 (既定 1、0/負/typo は 1、9 以上は 8)。
//! 2. FIFO 受付: `admit` 順に prefill が上がり、待ち位置は 0 始まりで詰まる。
//! 3. スロットのライフサイクル: complete / cancel で空いた番号の小さい順に再利用。
//! 4. キャンセルの後始末: queue 中は相対順を保って抜き、実行中は即座に空ける。
//! 5. 二重受付の拒否は状態を変えない。
//! 6. バッチ幅のクランプ: decode 列は max_batch 本以下、スロット番号の昇順。
//! 7. 差分試験: 独立に書いた単純な参照実装 (Vec のスロット + Vec の queue) と、
//!    決定的な擬似乱数で 500 操作のあいだ全状態が一致する。
//!
//! 既定 (1) では `serve_flash` がこのスケジューラを駆動しないので、ここで
//! 固定するのは状態機械の契約だけ。数値経路 (decode の出力) には触れない。

#[path = "../src/flash_batch.rs"]
mod flash_batch;

use flash_batch::{
    max_batch_from, AdmitError, BatchScheduler, CancelOutcome, SlotAssign, DEFAULT_MAX_BATCH,
    MAX_BATCH,
};

#[test]
fn env_truth_table_is_pinned() {
    // (入力, 期待値) を手で列挙する (実装の写しではなく仕様)。
    let cases: &[(Option<&str>, usize)] = &[
        (None, 1),
        (Some(""), 1),
        (Some("   "), 1),
        (Some("0"), 1),
        (Some("1"), 1),
        (Some("2"), 2),
        (Some(" 4 "), 4),
        (Some("7"), 7),
        (Some("8"), 8),
        (Some("9"), 8),
        (Some("128"), 8),
        (Some("-1"), 1),
        (Some("+3"), 3),
        (Some("two"), 1),
        (Some("2x"), 1),
    ];
    for &(input, want) in cases {
        assert_eq!(max_batch_from(input), want, "input={input:?}");
    }
    assert_eq!(DEFAULT_MAX_BATCH, 1);
    assert_eq!(MAX_BATCH, 8);
}

#[test]
fn fifo_admission_order_is_preserved() {
    let mut s = BatchScheduler::new(3);
    let ids = [40u64, 10, 30, 20, 50];
    for (pos, id) in ids.iter().enumerate() {
        assert_eq!(s.admit(*id), Ok(pos));
    }
    let mut order = Vec::new();
    for _ in 0..ids.len() {
        let p = s.plan_step();
        if let Some(a) = p.prefill {
            order.push(a.id);
        }
    }
    assert_eq!(order, ids[..3].to_vec(), "受付順 = prefill 順");
    assert_eq!(s.queued(), 2, "残り 2 本はスロット待ち");
    assert_eq!(s.position_of(20), Some(0));
    assert_eq!(s.position_of(50), Some(1));
    assert_eq!(s.active(), 3, "スロットは 3 本まで");
    assert!(s.check_invariants().is_ok());
}

#[test]
fn slot_reuse_prefers_lowest_free_index() {
    let mut s = BatchScheduler::new(4);
    for id in 0..4 {
        s.admit(id).unwrap();
        s.plan_step();
    }
    assert_eq!(s.slot_of(2), Some(2));
    assert!(s.complete(2));
    assert!(s.cancel(3).active);
    s.admit(100).unwrap();
    s.admit(101).unwrap();
    let a = s.plan_step();
    let b = s.plan_step();
    assert_eq!(a.prefill, Some(SlotAssign { id: 100, slot: 2 }));
    assert_eq!(b.prefill, Some(SlotAssign { id: 101, slot: 3 }));
    // 空きが無くなったので prefill は止まり、decode は 4 本。
    let c = s.plan_step();
    assert_eq!(c.prefill, None);
    assert_eq!(c.decode.len(), 4);
    assert_eq!(c.decode.iter().map(|x| x.slot).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    assert!(s.check_invariants().is_ok());
}

#[test]
fn cancel_keeps_relative_order_and_frees_slot() {
    let mut s = BatchScheduler::new(2);
    for id in 1..=6 {
        s.admit(id).unwrap();
    }
    s.plan_step();
    s.plan_step(); // 1, 2 が実行中
    assert_eq!(s.cancel(4), CancelOutcome { queued: true, active: false });
    assert_eq!(s.position_of(3), Some(0));
    assert_eq!(s.position_of(5), Some(1));
    assert_eq!(s.position_of(6), Some(2));

    assert_eq!(s.cancel(1), CancelOutcome { queued: false, active: true });
    assert_eq!(s.slot_of(2), Some(1), "他のスロットは動かさない");
    s.plan_step();
    assert_eq!(s.slot_of(3), Some(0), "空いた 0 番に FIFO 先頭");

    let noop = s.cancel(999);
    assert!(noop.is_noop());
    assert_eq!(s.stats().cancelled_queued, 1);
    assert_eq!(s.stats().cancelled_active, 1);
    assert!(s.check_invariants().is_ok());
}

#[test]
fn duplicate_admission_rejected_and_state_unchanged() {
    let mut s = BatchScheduler::new(2);
    s.admit(7).unwrap();
    s.plan_step();
    s.admit(8).unwrap();

    let before = (s.active(), s.queued());
    assert_eq!(s.admit(7), Err(AdmitError::Duplicate(7)), "実行中と重複");
    assert_eq!(s.admit(8), Err(AdmitError::Duplicate(8)), "queue と重複");
    assert_eq!((s.active(), s.queued()), before);
    assert_eq!(s.stats().accepted, 2);
    assert_eq!(s.stats().rejected_duplicate, 2);
    assert!(s.check_invariants().is_ok());
}

#[test]
fn decode_batch_is_bounded_and_sorted() {
    for max in 1..=MAX_BATCH {
        let mut s = BatchScheduler::new(max);
        for id in 0..48u64 {
            s.admit(id).unwrap();
        }
        let mut seen_decode = 0usize;
        for _ in 0..96 {
            let p = s.plan_step();
            assert!(p.decode.len() <= max, "max={max}");
            assert!(p.decode.len() <= MAX_BATCH);
            assert!(p.decode.windows(2).all(|w| w[0].slot < w[1].slot));
            if let Some(a) = p.prefill {
                assert_eq!(a.slot, s.slot_of(a.id).unwrap());
                assert!(
                    p.decode.iter().all(|d| d.id != a.id),
                    "prefill 中の 1 本は decode に入れない"
                );
            }
            seen_decode = seen_decode.max(p.decode.len());
            if seen_decode == max && p.prefill.is_none() {
                break;
            }
            assert!(s.check_invariants().is_ok());
        }
        assert_eq!(seen_decode, max, "max={max} の幅まで埋まる");
    }
}

/// 独立に書いた参照実装 (Vec だけ。`SlotAssign` も共有しない)。
struct Ref {
    slots: Vec<Option<u64>>,
    queue: Vec<u64>,
}

impl Ref {
    fn new(max: usize) -> Self {
        Self {
            slots: vec![None; max],
            queue: Vec::new(),
        }
    }
    fn admit(&mut self, id: u64) -> bool {
        if self.slots.contains(&Some(id)) || self.queue.contains(&id) {
            return false;
        }
        self.queue.push(id);
        true
    }
    fn plan(&mut self) -> (Option<(u64, usize)>, Vec<(u64, usize)>) {
        let mut pre = None;
        if let Some(slot) = self.slots.iter().position(|s| s.is_none()) {
            if !self.queue.is_empty() {
                let id = self.queue.remove(0);
                self.slots[slot] = Some(id);
                pre = Some((id, slot));
            }
        }
        let dec = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| s.is_some() && Some(*i) != pre.map(|p| p.1))
            .map(|(i, s)| (s.unwrap(), i))
            .collect();
        (pre, dec)
    }
    fn complete(&mut self, id: u64) -> bool {
        match self.slots.iter().position(|s| *s == Some(id)) {
            Some(i) => {
                self.slots[i] = None;
                true
            }
            None => false,
        }
    }
    fn cancel(&mut self, id: u64) -> (bool, bool) {
        if let Some(p) = self.queue.iter().position(|q| *q == id) {
            self.queue.remove(p);
            return (true, false);
        }
        match self.slots.iter().position(|s| *s == Some(id)) {
            Some(i) => {
                self.slots[i] = None;
                (false, true)
            }
            None => (false, false),
        }
    }
}

#[test]
fn differential_against_reference_state_machine() {
    let mut seed = 0x1606_1606u64;
    let mut rnd = |n: u64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) % n
    };
    for &max in &[1usize, 2, 3, 5, 8] {
        let mut s = BatchScheduler::new(max);
        let mut r = Ref::new(max);
        for step in 0..500 {
            match rnd(4) {
                0 | 1 => {
                    let id = rnd(16);
                    let got = s.admit(id).is_ok();
                    let want = r.admit(id);
                    assert_eq!(got, want, "admit id={id} step={step}");
                }
                2 => {
                    let got = s.plan_step();
                    let want = r.plan();
                    let got_pre = got.prefill.map(|a| (a.id, a.slot));
                    let got_dec: Vec<(u64, usize)> =
                        got.decode.iter().map(|a| (a.id, a.slot)).collect();
                    assert_eq!(got_pre, want.0, "prefill step={step}");
                    assert_eq!(got_dec, want.1, "decode step={step}");
                }
                3 => {
                    let id = rnd(16);
                    match rnd(2) {
                        0 => {
                            let got = s.complete(id);
                            let want = r.complete(id);
                            assert_eq!(got, want, "complete id={id} step={step}");
                        }
                        _ => {
                            let got = s.cancel(id);
                            let want = r.cancel(id);
                            assert_eq!((got.queued, got.active), want, "cancel id={id}");
                        }
                    }
                }
                _ => unreachable!(),
            }
            // 全 id について、スロットと待ち位置の見え方が参照実装と一致する。
            for id in 0..16u64 {
                assert_eq!(s.slot_of(id), r.slots.iter().position(|x| *x == Some(id)));
                assert_eq!(s.position_of(id), r.queue.iter().position(|q| *q == id));
            }
            assert_eq!(s.active(), r.slots.iter().filter(|s| s.is_some()).count());
            assert_eq!(s.queued(), r.queue.len());
            s.check_invariants().unwrap_or_else(|e| panic!("step={step}: {e}"));
        }
    }
}

#[test]
fn default_path_is_sequential_and_inert() {
    // 未設定 / 明示 1: 1 本ぶんのスロットだけを持ち、逐次経路と同じ形になる。
    for v in [None, Some("1"), Some("0"), Some("junk")] {
        let mut s = BatchScheduler::from_env(v);
        assert_eq!(s.max_batch(), 1, "v={v:?}");
        s.admit(1).unwrap();
        s.admit(2).unwrap();
        let p = s.plan_step();
        assert_eq!(p.prefill, Some(SlotAssign { id: 1, slot: 0 }));
        assert!(p.decode.is_empty());
        let q = s.plan_step();
        assert_eq!(q.prefill, None, "1 本が埋まっている間は次を上げない");
        assert_eq!(q.decode, vec![SlotAssign { id: 1, slot: 0 }]);
        assert_eq!(s.position_of(2), Some(0));
    }
}
