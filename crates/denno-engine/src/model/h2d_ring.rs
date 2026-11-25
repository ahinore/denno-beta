//! task 834: expert の H2D を専用コピーストリームで重ねるための**判定部分**。
//!
//! ここには CUDA を触らないものだけを置く (env の解釈・枠の巡回・再利用の判定)。
//! 実際の pinned リング / ストリーム / イベントは `crate::cuda::h2d` が持つ。
//! 分けている理由は、GPU の無いビルドでも同じ判定をテストで回せるようにするため
//! (スキルの「ゲートの補助関数を feature で切らない」と同じ方針)。
//!
//! # 何をするか
//!
//! `DENNO_EXPERT_H2D_ASYNC=1` のとき、expert の重み (gate/up/down) を
//! `cuMemcpyHtoD` (同期・null ストリーム) ではなく
//!
//! 1. pinned の枠 (`cuMemHostAlloc`) へホストで 1 回写し、
//! 2. 専用コピーストリームへ `cuMemcpyHtoDAsync` を積み、
//! 3. 完了は「その VRAM を実際に読む直前」まで待たない (`cuStreamWaitEvent`)
//!
//! という形にする。転送するバイト列は 1 ビットも変えない (数値は不変)。
//!
//! # 枠の再利用と安全
//!
//! 枠は `n` 本のリングで、`calls % n` の順に使う。使い回す枠には前の DMA が
//! まだ走っている可能性があるので、**枠へ写す前にその枠のイベントを待つ**
//! ([`needs_wait`] が真になる `calls >= n` のとき)。最初の `n` 回は新しい枠なので
//! 待たない。VRAM 側は「解放する前にイベントをホストで待つ」を
//! `crate::cuda::h2d::H2dPending` の `Drop` が担う。

/// 非同期 H2D のゲート。
pub const H2D_ASYNC_ENV: &str = "DENNO_EXPERT_H2D_ASYNC";

/// リングの枠数 (2..=16、既定 6)。
pub const H2D_RING_ENV: &str = "DENNO_EXPERT_H2D_RING_SLOTS";

/// 枠数の既定値。1 本あたり最大 8 MiB なので既定で最大 48 MiB。
pub const DEFAULT_SLOTS: usize = 6;

/// 枠数の下限 (1 にすると毎回待つので意味が無い)。
pub const MIN_SLOTS: usize = 2;

/// 枠数の上限 (pinned のホスト RAM を増やしすぎない)。
pub const MAX_SLOTS: usize = 16;

/// [`H2D_ASYNC_ENV`] の解釈 (純関数)。**既定 off** (ラボで 3% 以上を確認してから on)。
///
/// 未設定・空・`"0"` / `"false"` / `"no"` / `"off"` は off。
pub fn enabled_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// [`H2D_ASYNC_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enabled_from_value(std::env::var(H2D_ASYNC_ENV).ok().as_deref()))
}

/// [`H2D_RING_ENV`] の解釈 (純関数)。数値でない値と範囲外は丸める。
pub fn slots_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(n) => n.clamp(MIN_SLOTS, MAX_SLOTS),
        None => DEFAULT_SLOTS,
    }
}

/// リングの枠数 (プロセスで 1 回だけ env を読む)。
pub fn slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| slots_from_value(std::env::var(H2D_RING_ENV).ok().as_deref()))
}

/// この転送を非同期リングに載せてよいか (純関数)。
///
/// - ゲート off なら従来の同期経路 (数値は同じ。経路だけ)。
/// - 転送元が pinned (RAM tier) なら写さずに DMA する既存の利点を残すため同期。
/// - 枠 1 本に収まらない大きさはリングに載せない (pinned を食いすぎない)。
pub fn take_async(gate: bool, src_pinned: bool, bytes: usize, max_slot: usize) -> bool {
    gate && !src_pinned && bytes > 0 && bytes <= max_slot
}

/// 通し番号 → 枠の添字 (純関数)。
pub fn slot_index(calls: usize, n: usize) -> usize {
    let n = n.max(1);
    calls % n
}

/// この通し番号で使う枠は「前の DMA の完了を待つ」必要があるか (純関数)。
///
/// 最初の `n` 回はまだ一度も使っていない枠なので待たない。
pub fn needs_wait(calls: usize, n: usize) -> bool {
    calls >= n.max(1)
}

/// 枠の巡回の記録 (テストと診断用。CUDA 非依存)。
///
/// 実際のリングは「枠の番号」と「待つかどうか」だけをここから受け取り、
/// メモリとイベントは `crate::cuda::h2d` が持つ。
#[derive(Debug, Default)]
pub struct RingPlan {
    n: usize,
    calls: usize,
    /// `pick` が返した枠の並び (診断)。
    pub picks: Vec<usize>,
    /// 再利用でホスト待ちが要った回数。
    pub waits: usize,
}

impl RingPlan {
    pub fn new(n: usize) -> Self {
        Self {
            n: n.max(1),
            calls: 0,
            picks: Vec::new(),
            waits: 0,
        }
    }

    pub fn calls(&self) -> usize {
        self.calls
    }

    /// 次に使う枠と、使う前に前の DMA を待つ必要があるかを返す。
    pub fn pick(&mut self) -> (usize, bool) {
        let idx = slot_index(self.calls, self.n);
        let wait = needs_wait(self.calls, self.n);
        self.calls += 1;
        self.picks.push(idx);
        if wait {
            self.waits += 1;
        }
        (idx, wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn async_gate_defaults_off() {
        assert!(!enabled_from_value(None));
        assert!(!enabled_from_value(Some("")));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(Some("false")));
        assert!(!enabled_from_value(Some("off")));
        assert!(!enabled_from_value(Some("no")));
        // 明示の on だけ立つ (大文字・前後の空白も受ける)。
        assert!(enabled_from_value(Some("1")));
        assert!(enabled_from_value(Some(" true ")));
        assert!(enabled_from_value(Some("ON")));
    }

    #[test]
    fn ring_slots_are_bounded() {
        assert_eq!(slots_from_value(None), DEFAULT_SLOTS);
        assert_eq!(slots_from_value(Some("")), DEFAULT_SLOTS);
        assert_eq!(slots_from_value(Some("abc")), DEFAULT_SLOTS);
        assert_eq!(slots_from_value(Some("4")), 4);
        // 範囲外は丸める (1 本 = 毎回待つ、64 本 = pinned を食いすぎる)。
        assert_eq!(slots_from_value(Some("1")), MIN_SLOTS);
        assert_eq!(slots_from_value(Some("0")), MIN_SLOTS);
        assert_eq!(slots_from_value(Some("999")), MAX_SLOTS);
    }

    /// 枠は通し番号の剰余で、最初の 1 周は待たない / 2 周目からは待つ。
    #[test]
    fn ring_waits_only_from_the_second_lap() {
        let mut p = RingPlan::new(4);
        assert_eq!(p.pick(), (0, false));
        assert_eq!(p.pick(), (1, false));
        assert_eq!(p.pick(), (2, false));
        assert_eq!(p.pick(), (3, false));
        assert_eq!(p.pick(), (0, true), "5 本目は 1 本目の枠を使い回す");
        assert_eq!(p.pick(), (1, true));
        assert_eq!(p.waits, 2);
        assert_eq!(p.calls(), 6);
        assert_eq!(p.picks, vec![0, 1, 2, 3, 0, 1]);
    }

    /// 判定の純関数は `RingPlan` と同じ (片方だけ直す事故を防ぐ)。
    #[test]
    fn slot_helpers_agree_with_ring_plan() {
        for n in 1..=5usize {
            let mut p = RingPlan::new(n);
            for calls in 0..(n * 3) {
                let (idx, wait) = p.pick();
                assert_eq!(idx, slot_index(calls, n));
                assert_eq!(wait, needs_wait(calls, n));
            }
        }
    }

    /// 非同期リングに載せる条件 (ゲート off / pinned 源 / 大きすぎる転送は載せない)。
    #[test]
    fn take_async_truth_table() {
        let max = 8 << 20;
        assert!(take_async(true, false, 1024, max));
        // ゲート off は常に同期経路。
        assert!(!take_async(false, false, 1024, max));
        // RAM tier の pinned 源は写さずに DMA する既存の経路を残す。
        assert!(!take_async(true, true, 1024, max));
        // 空 / 枠に収まらない転送は載せない。
        assert!(!take_async(true, false, 0, max));
        assert!(!take_async(true, false, max + 1, max));
        assert!(take_async(true, false, max, max));
    }
}
