//! task 1606: serve-flash の連続バッチング用スロットスケジューラ (既定 off)。
//!
//! 目的:
//!
//! - serve-flash はいま 1 リクエストずつ逐次処理する (モデルは 1 つ、KV の
//!   系列状態も 1 本)。連続バッチングを入れるには、まず「どのリクエストが
//!   どのスロットで decode 中か」「新規リクエストをどの順で prefill に上げるか」
//!   「切断・中止でどのスロットをいつ空けるか」を 1 か所で決める必要がある。
//!   ここはその判断だけを持つ純粋な状態機械で、GPU もスレッドも触らない。
//! - 既定は `DENNO_FLASH_MAX_BATCH` 未設定 = 1。このとき `serve_flash` は
//!   このモジュールを参照せず、従来の逐次経路がそのまま動く (バイト一致)。
//!   2 以上で有効になったとき、次のタスクが
//!   `Qwen4ExpDecoder::forward_batch_decode(m)` の m と並び順を
//!   [`BatchScheduler::plan_step`] から取る。
//!
//! 契約 (試験 `tests/flash_batch_edge_1606.rs` が固定する):
//!
//! 1. 受付は FIFO。`admit` した順に [`BatchScheduler::plan_step`] が
//!    prefill へ上げる。割り込み・優先度・プロトコル別の差は無い。
//! 2. スロットは [`BatchScheduler::new`] の `max_batch` 本 (1..=[`MAX_BATCH`])。
//!    decode の並びは常にスロット番号の昇順で決定的。
//! 3. `complete` / `cancel` はスロットを即座に空ける (次の `plan_step` が
//!    空き番号の小さい順に再利用する)。queue 中の `cancel` は順序を保って
//!    取り除くだけで、他のリクエストの相対順は変えない。
//! 4. 同じ id の二重受付は [`AdmitError::Duplicate`] で拒否し、状態を変えない。
//! 5. prefill は 1 step に 1 本まで (denno-server の P6 と同じ「decode 優先、
//!    新規 prefill は 1 本ずつ割り込ませる」)。decode は最大 `max_batch` 本。
//!
//! 数値経路には触れない。ここが決めるのは並び順とスロットの割り当てだけで、
//! 出力の一致は呼び出し側 (`forward_batch_decode` のビット一致契約) が保証する。
//!
//! 既定 (max batch 1) では `serve_flash` から駆動されないので、bin ビルドでの
//! 未使用警告は出さない (実装は試験と次のタスクの入口として残す)。
#![allow(dead_code)]

use std::collections::VecDeque;

/// `DENNO_FLASH_MAX_BATCH` で指定できる同時実行数の上限 (forward_batch_decode の m)。
pub(crate) const MAX_BATCH: usize = 8;

/// 既定の同時実行数 (従来の逐次経路 = 1)。
pub(crate) const DEFAULT_MAX_BATCH: usize = 1;

/// 環境変数名。
pub(crate) const ENV_MAX_BATCH: &str = "DENNO_FLASH_MAX_BATCH";

/// `DENNO_FLASH_MAX_BATCH` の値を同時実行数へ変換する (純関数)。
///
/// - 未設定 / 空 / 数値でない / 0 以下 → [`DEFAULT_MAX_BATCH`] (1)。
///   ここで 1 に落ちるのは「typo で黙って別経路に切り替わる」のを避けるため。
/// - [`MAX_BATCH`] を超える値は [`MAX_BATCH`] に丸める。
///
/// 前後の空白は許す (`" 4 "` → 4)。
pub(crate) fn max_batch_from(v: Option<&str>) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(n) => n.clamp(DEFAULT_MAX_BATCH, MAX_BATCH),
        None => DEFAULT_MAX_BATCH,
    }
}

/// リクエスト id と、それが載っているスロット番号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotAssign {
    pub id: u64,
    pub slot: usize,
}

/// 1 step ぶんの割り当て。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StepPlan {
    /// この step で新規に prefill を始める 1 本 (無ければ `None`)。
    pub prefill: Option<SlotAssign>,
    /// この step で decode する列 (スロット番号の昇順 = 決定的)。
    /// 長さは最大 `max_batch`。新規 prefill の 1 本は含まない。
    pub decode: Vec<SlotAssign>,
}

/// 二重受付の拒否。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmitError {
    /// 同じ id が既に queue かスロットにある。
    Duplicate(u64),
}

/// `cancel` の結果 (監査ログと試験のため)。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CancelOutcome {
    /// queue から取り除いた。
    pub queued: bool,
    /// 実行中のスロットを空けた。
    pub active: bool,
}

impl CancelOutcome {
    /// どちらでもなかった (未知の id)。
    pub(crate) fn is_noop(self) -> bool {
        !self.queued && !self.active
    }
}

/// スケジューラの累計 (計測のみ。数値経路には触れない)。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    /// `admit` が受け付けた数 (二重受付の拒否は含まない)。
    pub accepted: u64,
    /// prefill を開始した数。
    pub started: u64,
    /// `complete` でスロットを返した数。
    pub completed: u64,
    /// queue 中に `cancel` された数。
    pub cancelled_queued: u64,
    /// 実行中に `cancel` された数。
    pub cancelled_active: u64,
    /// 二重受付として拒否した数。
    pub rejected_duplicate: u64,
    /// 同時に埋まったスロット数の最大。
    pub peak_active: usize,
}

/// 連続バッチングの受付キュー + スロット割り当て。
///
/// 状態は「`queue`: 受付順の id 列」と「`slots`: スロット番号 → 実行中の id」の
/// 2 つだけ。全操作が O(queue + max_batch) で、`max_batch <= 8` なので
/// ステップあたりのコストは無視できる。
pub(crate) struct BatchScheduler {
    max_batch: usize,
    slots: Vec<Option<u64>>,
    queue: VecDeque<u64>,
    stats: Stats,
}

impl BatchScheduler {
    /// `max_batch` は 1..=[`MAX_BATCH`] に丸める (env と同じ規則)。
    pub(crate) fn new(max_batch: usize) -> Self {
        let max_batch = max_batch.clamp(DEFAULT_MAX_BATCH, MAX_BATCH);
        Self {
            max_batch,
            slots: vec![None; max_batch],
            queue: VecDeque::new(),
            stats: Stats::default(),
        }
    }

    /// `DENNO_FLASH_MAX_BATCH` から作る。
    pub(crate) fn from_env(v: Option<&str>) -> Self {
        Self::new(max_batch_from(v))
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// 実行中 (スロット占有中) の数。
    pub(crate) fn active(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// まだ prefill していない数。
    pub(crate) fn queued(&self) -> usize {
        self.queue.len()
    }

    pub(crate) fn stats(&self) -> Stats {
        self.stats
    }

    /// 空きスロット番号のうち最小のもの。
    fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    /// 受付。queue の末尾に積み、0 始まりの待ち位置を返す。
    /// 実行中・queue 中のどちらかに同じ id があれば状態を変えずに拒否する。
    pub(crate) fn admit(&mut self, id: u64) -> Result<usize, AdmitError> {
        if self.slot_of(id).is_some() || self.queue.contains(&id) {
            self.stats.rejected_duplicate += 1;
            return Err(AdmitError::Duplicate(id));
        }
        self.queue.push_back(id);
        self.stats.accepted += 1;
        Ok(self.queue.len() - 1)
    }

    /// queue 内の待ち位置 (0 始まり)。実行中・未知なら `None`。
    pub(crate) fn position_of(&self, id: u64) -> Option<usize> {
        self.queue.iter().position(|&q| q == id)
    }

    /// 実行中のスロット番号。
    pub(crate) fn slot_of(&self, id: u64) -> Option<usize> {
        self.slots.iter().position(|s| *s == Some(id))
    }

    /// 1 step ぶんの割り当てを決める。
    ///
    /// 1. 空きスロットがあり queue が空でなければ、先頭 1 本を prefill に上げる。
    /// 2. decode 列は占有中のスロットを番号の昇順に並べたもの。
    ///
    /// 上げたばかりの 1 本は prefill が終わっていないので decode 列には入れない
    /// (次の step から decode に加わる)。
    pub(crate) fn plan_step(&mut self) -> StepPlan {
        let mut prefill = None;
        if let Some(slot) = self.free_slot() {
            if let Some(id) = self.queue.pop_front() {
                self.slots[slot] = Some(id);
                self.stats.started += 1;
                self.stats.peak_active = self.stats.peak_active.max(self.active());
                prefill = Some(SlotAssign { id, slot });
            }
        }
        // prefill を上げた 1 本はまだ decode できないので列から除く。
        let new_slot = prefill.map(|p| p.slot);
        let decode = self
            .slots
            .iter()
            .enumerate()
            .filter(|(slot, s)| s.is_some() && Some(*slot) != new_slot)
            .map(|(slot, s)| SlotAssign {
                id: s.unwrap(),
                slot,
            })
            .collect();
        StepPlan { prefill, decode }
    }

    /// 生成が終わったリクエストのスロットを返す。実行中でなければ `false`。
    pub(crate) fn complete(&mut self, id: u64) -> bool {
        match self.slot_of(id) {
            Some(slot) => {
                self.slots[slot] = None;
                self.stats.completed += 1;
                true
            }
            None => false,
        }
    }

    /// 切断・中止・期限切れの後始末。queue なら取り除き、実行中ならスロットを
    /// 空ける。どちらでもなければ何もせず [`CancelOutcome::is_noop`] を返す。
    ///
    /// queue からの取り除きは他のリクエストの相対順を変えない。
    pub(crate) fn cancel(&mut self, id: u64) -> CancelOutcome {
        if let Some(pos) = self.position_of(id) {
            self.queue.remove(pos);
            self.stats.cancelled_queued += 1;
            return CancelOutcome {
                queued: true,
                active: false,
            };
        }
        if let Some(slot) = self.slot_of(id) {
            self.slots[slot] = None;
            self.stats.cancelled_active += 1;
            return CancelOutcome {
                queued: false,
                active: true,
            };
        }
        CancelOutcome::default()
    }

    /// 何も追跡していないか。
    pub(crate) fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.active() == 0
    }

    /// 内部整合性の検査 (試験用)。壊れていれば理由を返す。
    pub(crate) fn check_invariants(&self) -> Result<(), String> {
        if self.slots.len() != self.max_batch {
            return Err(format!(
                "スロット数 {} != max_batch {}",
                self.slots.len(),
                self.max_batch
            ));
        }
        if !(DEFAULT_MAX_BATCH..=MAX_BATCH).contains(&self.max_batch) {
            return Err(format!("max_batch {} が範囲外", self.max_batch));
        }
        let mut seen: Vec<u64> = Vec::new();
        for (slot, s) in self.slots.iter().enumerate() {
            let Some(id) = *s else { continue };
            if seen.contains(&id) {
                return Err(format!("id {id} が複数スロットにある (slot {slot})"));
            }
            seen.push(id);
        }
        for (pos, id) in self.queue.iter().enumerate() {
            if seen.contains(id) {
                return Err(format!("id {id} が queue({pos}) とスロットの両方にある"));
            }
            if self.queue.iter().filter(|q| *q == id).count() > 1 {
                return Err(format!("id {id} が queue に重複"));
            }
            seen.push(*id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_batch_env_truth_table() {
        assert_eq!(max_batch_from(None), 1);
        assert_eq!(max_batch_from(Some("")), 1);
        assert_eq!(max_batch_from(Some("0")), 1);
        assert_eq!(max_batch_from(Some("1")), 1);
        assert_eq!(max_batch_from(Some(" 4 ")), 4);
        assert_eq!(max_batch_from(Some("8")), 8);
        assert_eq!(max_batch_from(Some("9")), 8);
        assert_eq!(max_batch_from(Some("64")), 8);
        assert_eq!(max_batch_from(Some("-3")), 1);
        assert_eq!(max_batch_from(Some("four")), 1);
        assert_eq!(max_batch_from(Some("2.5")), 1);
    }

    #[test]
    fn new_clamps_into_range() {
        assert_eq!(BatchScheduler::new(0).max_batch(), 1);
        assert_eq!(BatchScheduler::new(1).max_batch(), 1);
        assert_eq!(BatchScheduler::new(8).max_batch(), 8);
        assert_eq!(BatchScheduler::new(99).max_batch(), 8);
        assert!(BatchScheduler::new(3).check_invariants().is_ok());
    }

    #[test]
    fn fifo_admission_and_prefill_one_per_step() {
        let mut s = BatchScheduler::new(2);
        assert_eq!(s.admit(10), Ok(0));
        assert_eq!(s.admit(11), Ok(1));
        assert_eq!(s.admit(12), Ok(2));

        let p1 = s.plan_step();
        assert_eq!(p1.prefill, Some(SlotAssign { id: 10, slot: 0 }));
        assert!(p1.decode.is_empty(), "prefill 直後はまだ decode しない");

        let p2 = s.plan_step();
        assert_eq!(p2.prefill, Some(SlotAssign { id: 11, slot: 1 }));
        assert_eq!(p2.decode, vec![SlotAssign { id: 10, slot: 0 }]);

        // 満杯なので prefill は進まない。decode は受付順ではなくスロット順。
        let p3 = s.plan_step();
        assert_eq!(p3.prefill, None);
        assert_eq!(
            p3.decode,
            vec![
                SlotAssign { id: 10, slot: 0 },
                SlotAssign { id: 11, slot: 1 }
            ]
        );
        assert_eq!(s.queued(), 1);
        assert_eq!(s.position_of(12), Some(0));
        assert!(s.check_invariants().is_ok());
    }

    #[test]
    fn slot_lifecycle_reuses_lowest_free_slot() {
        let mut s = BatchScheduler::new(3);
        for id in 1..=3 {
            s.admit(id).unwrap();
            s.plan_step();
        }
        assert_eq!(s.active(), 3);
        assert!(s.complete(2));
        assert_eq!(s.slot_of(2), None);
        assert!(!s.complete(2), "二重 complete は無効");

        s.admit(4).unwrap();
        let p = s.plan_step();
        assert_eq!(p.prefill, Some(SlotAssign { id: 4, slot: 1 }));
        assert_eq!(s.stats().completed, 1);
        assert_eq!(s.stats().started, 4);
    }

    #[test]
    fn cancellation_removes_from_queue_without_reordering() {
        let mut s = BatchScheduler::new(1);
        s.admit(1).unwrap();
        s.plan_step(); // 1 が実行中
        s.admit(2).unwrap();
        s.admit(3).unwrap();
        s.admit(4).unwrap();

        let c = s.cancel(3);
        assert_eq!(
            c,
            CancelOutcome {
                queued: true,
                active: false
            }
        );
        assert_eq!(s.position_of(2), Some(0));
        assert_eq!(s.position_of(4), Some(1));

        assert!(s.complete(1));
        let p = s.plan_step();
        assert_eq!(p.prefill, Some(SlotAssign { id: 2, slot: 0 }));
        assert!(s.check_invariants().is_ok());
    }

    #[test]
    fn cancel_active_frees_slot_immediately() {
        let mut s = BatchScheduler::new(2);
        s.admit(7).unwrap();
        s.admit(8).unwrap();
        s.plan_step();
        s.plan_step();
        let c = s.cancel(7);
        assert_eq!(
            c,
            CancelOutcome {
                queued: false,
                active: true
            }
        );
        assert_eq!(s.active(), 1);
        assert_eq!(s.slot_of(8), Some(1), "他リクエストのスロットは動かさない");

        s.admit(9).unwrap();
        let p = s.plan_step();
        assert_eq!(p.prefill, Some(SlotAssign { id: 9, slot: 0 }));
        assert_eq!(p.decode, vec![SlotAssign { id: 8, slot: 1 }]);
        assert_eq!(
            s.cancel(999),
            CancelOutcome::default(),
            "未知の id は no-op"
        );
    }

    #[test]
    fn duplicate_admission_is_rejected_without_state_change() {
        let mut s = BatchScheduler::new(2);
        s.admit(5).unwrap();
        assert_eq!(s.admit(5), Err(AdmitError::Duplicate(5)));
        s.plan_step();
        assert_eq!(s.admit(5), Err(AdmitError::Duplicate(5)));

        let snap = (s.active(), s.queued(), s.stats());
        assert_eq!((s.active(), s.queued(), s.stats()), snap);
        assert_eq!(s.stats().rejected_duplicate, 2);
        assert_eq!(s.stats().accepted, 1);
        assert!(s.check_invariants().is_ok());
    }

    #[test]
    fn decode_batch_never_exceeds_max_batch() {
        for max in DEFAULT_MAX_BATCH..=MAX_BATCH {
            let mut s = BatchScheduler::new(max);
            for id in 0..32 {
                s.admit(id).unwrap();
            }
            for _ in 0..64 {
                let p = s.plan_step();
                assert!(p.decode.len() <= max);
                assert!(p.decode.len() <= MAX_BATCH);
                assert!(p.decode.windows(2).all(|w| w[0].slot < w[1].slot));
            }
            assert_eq!(s.active(), max);
        }
    }

    #[test]
    fn idle_after_complete_all() {
        let mut s = BatchScheduler::from_env(Some("3"));
        assert_eq!(s.max_batch(), 3);
        for id in 0..3 {
            s.admit(id).unwrap();
        }
        s.plan_step();
        s.plan_step();
        s.plan_step();
        for id in 0..3 {
            assert!(s.complete(id));
        }
        assert!(s.is_idle());
        assert_eq!(s.stats().peak_active, 3);
    }
}
