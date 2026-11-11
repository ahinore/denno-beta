//! task 254: 14k cold prefill の **ホスト台帳** (`DENNO_PF_HOSTPROF`)。
//!
//! [`crate::model::q4ledger`] は区間ごとの壁時計・発行・同期・コピーを積むが、
//! prefill の `experts` 区間の中の**ホスト仕事の内訳**は 1 本の `cpu` 列に
//! まとまっていて、何が重いのか読めない。ここではその中を項目に割る:
//!
//! | 項目 | 測る場所 |
//! |---|---|
//! | [`LOOKUP`] | expert VRAM の参照 (`peek_many` + (層, expert) の列作り) |
//! | [`PLAN`] | `by_expert` / `pick` / `entries` / `build_plan` / 行表 (`prow`) |
//! | [`STAGE_COPY`] | pinned へのバウンスコピー (`pf_copy_into_pin`) |
//! | [`STAGE_ENQ`] | `cuMemcpyHtoDAsync` + イベント記録 (H2D の積み込み) |
//! | [`PTR_TAB`] | デバイスポインタ表の組み立てと H2D (3 本/グループ) |
//! | [`UP_WAIT`] | グループの H2D 完了待ち (ホストが止まった時間) |
//! | [`SCRATCH`] | 作業域の確保と活性・表の H2D (1 層 1 回) |
//! | [`SYNC`] | `ctx.sync` (累積カーネルの完了待ち) |
//! | [`D2H`] | 累積 (`ntok x e`) の D2H |
//! | [`ADD`] | ホスト側の加重和 (`exp_pf_add_all`) |
//! | [`CPU_MISS`] | CPU に残したミス expert の計算 |
//! | [`PREFETCH`] | 選択 expert の先読みと `seen` 列 |
//! | [`OTHER`] | `experts` 区間の残差 (上のどれでもない) |
//!
//! 「露出」= その項目のホスト時間のうち GPU 仕事で隠れなかった分。
//! 区間全体では `wall - gpu_busy` で、GPU の実時間は cuda 側のイベント
//! ([`crate::cuda::tline`] の起動ペア) が測る。突き合わせは `serve_flash` が
//! 出す `[q4ledger-pf] close` 行 (wall = gpu + exposed が 1 s 以内で閉じるか)
//! と、ここの `[pf-host]` 行 (experts の中の内訳) を並べて読む。
//!
//! 積むのは診断カウンタだけ。引数・カーネル・丸め順には触れない。off のとき
//! 呼び出し側が払うのは [`on`] の atomic load 1 回。
//!
//! # ゲート
//!
//! `DENNO_PF_HOSTPROF` (既定: `DENNO_Q4_PROFILE` が立っていれば on)。
//! `0` / `false` / `no` で明示的に切れる (A/B 用)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// 区間: expert VRAM の参照 (`peek_many` と (層, expert) の列)。
pub const LOOKUP: usize = 0;
/// 区間: `by_expert` / `pick` / `entries` / `build_plan` / 行表。
pub const PLAN: usize = 1;
/// 区間: pinned へのバウンスコピー。
pub const STAGE_COPY: usize = 2;
/// 区間: H2D の積み込み (`cuMemcpyHtoDAsync` + イベント記録)。
pub const STAGE_ENQ: usize = 3;
/// 区間: デバイスポインタ表の組み立てと H2D。
pub const PTR_TAB: usize = 4;
/// 区間: グループの H2D 完了待ち。
pub const UP_WAIT: usize = 5;
/// 区間: 作業域の確保と活性・表の H2D (1 層 1 回)。
pub const SCRATCH: usize = 6;
/// 区間: `ctx.sync` (累積カーネルの完了待ち)。
pub const SYNC: usize = 7;
/// 区間: 累積の D2H。
pub const D2H: usize = 8;
/// 区間: ホスト側の加重和。
pub const ADD: usize = 9;
/// 区間: CPU に残したミス expert の計算。
pub const CPU_MISS: usize = 10;
/// 区間: 選択 expert の先読みと `seen` 列。
pub const PREFETCH: usize = 11;
/// 区間: `experts` 区間の残差 (wall − 上の和)。
pub const OTHER: usize = 12;
/// 区間の数。
pub const NSLOT: usize = 13;

/// 項目の名前 (`[pf-host]` 行の列名、[`LOOKUP`] の順)。
pub const SLOT_NAMES: [&str; NSLOT] = [
    "lookup",
    "plan",
    "stage_copy",
    "stage_enq",
    "ptrtab",
    "up_wait",
    "scratch",
    "sync",
    "d2h",
    "add",
    "cpu_miss",
    "prefetch",
    "other",
];

static NS: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
/// `experts` 区間の壁時計 (ガードが積む)。残差 `other` の分母。
static STAGE_WALL: AtomicU64 = AtomicU64::new(0);
/// ガードの入れ子の深さ (内側のガードは壁時計を積まない)。
static DEPTH: AtomicU64 = AtomicU64::new(0);

/// テストが env を立てずに計測を通すための強制フラグ (env は `OnceLock`)。
#[cfg(test)]
static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 計測をテストから強制する (`false` で env の解釈に戻る)。
#[cfg(test)]
pub fn set_for_test(v: bool) {
    FORCE.store(v, Ordering::Relaxed);
}

/// `DENNO_PF_HOSTPROF` の値の解釈 (`0` / `false` / `no` だけ off)。
fn value_on(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}

/// 計測するか。未設定なら `DENNO_Q4_PROFILE` に従う (lab の `PROFILE=1` で出る)。
#[inline]
pub fn on() -> bool {
    #[cfg(test)]
    if FORCE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("DENNO_PF_HOSTPROF") {
        Ok(v) => value_on(Some(v.as_str())),
        Err(_) => std::env::var_os("DENNO_Q4_PROFILE").is_some(),
    })
}

/// 計測が要るときだけ `Instant` を取る (off では `None` = no-op)。
#[inline]
pub fn stamp() -> Option<Instant> {
    if on() {
        Some(Instant::now())
    } else {
        None
    }
}

/// 区間 `slot` に `t` からの経過を足す (`None` は no-op)。
#[inline]
pub fn mark(t: Option<Instant>, slot: usize) {
    if let Some(t) = t {
        add_ns(slot, t.elapsed().as_nanos() as u64);
    }
}

/// 区間 `slot` に ns を直接足す (外で測った値を渡す用)。
#[inline]
pub fn add_ns(slot: usize, ns: u64) {
    if slot < NSLOT {
        NS[slot].fetch_add(ns, Ordering::Relaxed);
    }
}

/// `experts` 区間の壁時計を積むガード。drop で 1 回ぶん足す。
///
/// 入れ子 (層の中で同じ区間を開き直す) では**外側だけ**が積む。
#[must_use = "ガードを落とすと壁時計が積まれない"]
pub struct Stage(Option<Instant>);

impl Drop for Stage {
    fn drop(&mut self) {
        let Some(t) = self.0 else { return };
        if DEPTH.fetch_sub(1, Ordering::Relaxed) == 1 {
            STAGE_WALL.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// `experts` 区間の壁時計を測り始める (off では `Instant` を取らないだけ)。
pub fn stage() -> Stage {
    let t = stamp();
    if t.is_some() {
        DEPTH.fetch_add(1, Ordering::Relaxed);
    }
    Stage(t)
}

/// 区間の積算 (ns) を読む (テストとレポート用)。
pub fn ns(slot: usize) -> u64 {
    NS[slot.min(NSLOT - 1)].load(Ordering::Relaxed)
}

/// `experts` 区間の壁時計 (ns)。
pub fn stage_wall_ns() -> u64 {
    STAGE_WALL.load(Ordering::Relaxed)
}

/// 1 リクエストぶんの `[pf-host]` 行を返して積算を 0 に戻す。
///
/// 1 行目: `experts` 区間の壁時計と項目別のホスト時間 (合計 ms と割合)、
/// 残差 `other`。2 行目は無い (GPU 側は `[q4ledger-pf]` の `busy` 列)。
#[must_use]
pub fn report(tag: &str, tok: usize) -> Vec<String> {
    if !on() {
        return Vec::new();
    }
    let wall = STAGE_WALL.load(Ordering::Relaxed);
    if wall == 0 && NS.iter().all(|c| c.load(Ordering::Relaxed) == 0) {
        return Vec::new();
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    let mut sum = 0u64;
    let mut parts: Vec<String> = Vec::new();
    // `other` は残差なので、上の項目の和を出してから最後に足す。
    for s in 0..OTHER {
        let v = NS[s].load(Ordering::Relaxed);
        sum += v;
        if v > 0 {
            parts.push(format!("{}={:.1}", SLOT_NAMES[s], ms(v)));
        }
    }
    let other = wall.saturating_sub(sum);
    let mut l1 = format!(
        "[pf-host] {tag} tok={tok} experts_wall={:.1}ms",
        ms(wall)
    );
    l1.push_str(&format!(" | {}", parts.join(" ")));
    l1.push_str(&format!(
        " | other={:.1} ({:.1}% of wall)",
        ms(other),
        if wall > 0 {
            other as f64 * 100.0 / wall as f64
        } else {
            0.0
        }
    ));
    reset();
    vec![l1]
}

/// 累積を 0 に戻す。
pub fn reset() {
    for c in NS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    STAGE_WALL.store(0, Ordering::Relaxed);
    DEPTH.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// グローバルカウンタを共有するので 1 本ずつ回す。
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::model::moe_expert::moe_test_lock()
    }

    #[test]
    fn value_parsing_matches_other_gates() {
        assert!(value_on(None));
        assert!(value_on(Some("1")));
        assert!(!value_on(Some("0")));
        assert!(!value_on(Some("false")));
        assert!(!value_on(Some("no")));
        assert!(value_on(Some("yes")));
    }

    #[test]
    fn off_records_nothing() {
        let _g = lock();
        set_for_test(false);
        reset();
        assert!(stamp().is_none(), "off で Instant を取った");
        add_ns(LOOKUP, 1234);
        assert!(report("off", 1).is_empty(), "off でレポートが出た");
        reset();
    }

    #[test]
    fn stage_guard_accumulates_and_other_is_residual() {
        let _g = lock();
        set_for_test(true);
        reset();
        add_ns(LOOKUP, 1_000_000);
        add_ns(STAGE_COPY, 3_000_000);
        {
            let _s = stage();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let wall = stage_wall_ns();
        assert!(wall >= 1_000_000, "壁時計が積まれていない: {wall}");
        let lines = report("unit", 8);
        assert_eq!(lines.len(), 1, "レポートは 1 行: {lines:?}");
        assert!(lines[0].starts_with("[pf-host] unit tok=8"), "{:?}", lines[0]);
        assert!(lines[0].contains("stage_copy=3.0"), "{:?}", lines[0]);
        assert!(lines[0].contains("other="), "{:?}", lines[0]);
        // report は累積を消す
        assert_eq!(ns(LOOKUP), 0, "report のあとに累積が残った");
        assert_eq!(stage_wall_ns(), 0, "report のあとに壁時計が残った");
        set_for_test(false);
        reset();
    }

    #[test]
    fn nested_stage_guard_counts_once() {
        let _g = lock();
        set_for_test(true);
        reset();
        {
            let _a = stage();
            std::thread::sleep(std::time::Duration::from_millis(1));
            {
                let _b = stage();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let wall = stage_wall_ns();
        // 外側の 2 ms ぶんだけ。内側を足すと 3 ms を超える。
        assert!(
            wall >= 2_000_000 && wall < 3_000_000,
            "入れ子の壁時計が 1 回でない: {wall}"
        );
        set_for_test(false);
        reset();
    }
}
