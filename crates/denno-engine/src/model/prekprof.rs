//! prefill の「層の外」の区間計測 (`DENNO_PRE_KPROF=1`、既定 off)。
//!
//! [`crate::model::qwen4exp::q4prof`] は層の中 (mix / attn / moe) しか積まない。
//! P76 の実測では層合計 39.1 s に対して wall が 46.9 s で、**層の外に 7.8 s**
//! が残った。ここではその外側を区間に割って積む:
//!
//! | 区間 | 測る場所 |
//! |---|---|
//! | [`TOKENIZE`] | `serve_flash` の `encode_messages` (prefill wall の外) |
//! | [`EMBED`] | `Qwen4Exp::forward_embed` (token_embd の行デコード) |
//! | [`PLE`] | `Qwen4Exp::gather_ple` / `gather_ple_many` (表の pread) |
//! | [`MODEL`] | `serve_flash` の `build_model` (mmap と attention/KV 状態の確保) |
//! | [`LAYER`] | 層ループの wall (q4prof を立てていればその合計と比べる) |
//! | [`FINAL_NORM`] | head の mixer (`hc_mix_quant`) |
//! | [`HEAD`] | `output` の GEMV |
//! | [`SAMPLE`] | decode の `Sampler::sample` |
//!
//! PLE については表の pread 回数・行数・バイト数と、`/proc/self/stat` の
//! `minflt` / `majflt` 差分も積む。**pread は page fault を起こさない**ので、
//! 「PLE が遅い」のか「mmap 経由の重み読みが遅い」のかはこの 2 つを並べて
//! 切り分ける (前者なら reads と bytes が増えて faults は増えない)。
//!
//! 積むのは診断カウンタだけ。引数・カーネル・丸め順には触れない。off のとき
//! 呼び出し側が払うのは [`on`] の atomic load 1 回。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// 区間: prompt のトークナイズ (`serve_flash`、prefill wall の外)。
pub const TOKENIZE: usize = 0;
/// 区間: embedding の行 gather (`token_embd`)。
pub const EMBED: usize = 1;
/// 区間: PLE の n-gram 索引 → 表の pread。
pub const PLE: usize = 2;
/// 区間: モデルの組み立て (mmap / KV・SSM 状態の確保)。
pub const MODEL: usize = 3;
/// 区間: 層ループの wall (層の中身は q4prof が見る)。
pub const LAYER: usize = 4;
/// 区間: head の mixer (最終 norm 相当)。
pub const FINAL_NORM: usize = 5;
/// 区間: `output` の GEMV。
pub const HEAD: usize = 6;
/// 区間: decode のサンプリング (prefill wall の外)。
pub const SAMPLE: usize = 7;
/// 区間の数。
pub const NSLOT: usize = 8;

/// 区間の名前 (レポートの列名)。
pub const SLOT_NAMES: [&str; NSLOT] = [
    "tokenize",
    "embed",
    "ple",
    "model+kv",
    "layer",
    "final_norm",
    "head",
    "sample",
];

/// prefill wall (t1..t_prefill) に入る区間。`other = wall - この合計`。
pub const IN_WALL: [usize; 6] = [MODEL, EMBED, PLE, LAYER, FINAL_NORM, HEAD];

static NS: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
/// prefill の呼び出し回数とトークン数 (レポートの分母)。
static PREFILLS: AtomicU64 = AtomicU64::new(0);
static TOKS: AtomicU64 = AtomicU64::new(0);
/// サンプリングの呼び出し回数 (1 トークンあたりの分母)。
static SAMPLES: AtomicU64 = AtomicU64::new(0);
/// PLE 表の pread 統計。
static PLE_READS: AtomicU64 = AtomicU64::new(0);
static PLE_ROWS: AtomicU64 = AtomicU64::new(0);
static PLE_BYTES: AtomicU64 = AtomicU64::new(0);
/// PLE 区間のページフォルト差分 (`/proc/self/stat`)。
static MINFLT: AtomicU64 = AtomicU64::new(0);
static MAJFLT: AtomicU64 = AtomicU64::new(0);

/// テストが env を立てずに計測を通すための強制フラグ
/// (env は `OnceLock` なのでプロセス内で 1 回しか読めない)。
#[cfg(test)]
static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 計測をテストから強制する (`false` で env の解釈に戻る)。
#[cfg(test)]
pub fn set_for_test(v: bool) {
    FORCE.store(v, Ordering::Relaxed);
}

/// 区間計測が有効か (既定 off)。
#[inline]
pub fn on() -> bool {
    #[cfg(test)]
    if FORCE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DENNO_PRE_KPROF").is_some())
}

/// 計測が要るときだけ `Instant` を取る (既定 off では `None` = 呼び出し側は no-op)。
#[inline]
pub fn stamp() -> Option<Instant> {
    if on() {
        Some(Instant::now())
    } else {
        None
    }
}

/// すでに判定済みの条件で `Instant` を取る ([`measuring`] と組で使う)。
#[inline]
pub fn stamp_when(enabled: bool) -> Option<Instant> {
    if enabled {
        Some(Instant::now())
    } else {
        None
    }
}

/// 区間 `slot` に `t` からの経過を足す (`None` は no-op)。
#[inline]
pub fn mark(t: Option<Instant>, slot: usize) {
    if let Some(t) = t {
        NS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// 区間 `slot` に ns を直接足す (外で測った値を渡す用)。
#[inline]
pub fn add_ns(slot: usize, ns: u64) {
    NS[slot].fetch_add(ns, Ordering::Relaxed);
}

/// prefill 1 回ぶんのトークン数を数える (レポートの `tok=` は呼び出し側が渡す)。
#[inline]
pub fn note_prefill(tokens: usize) {
    if on() {
        PREFILLS.fetch_add(1, Ordering::Relaxed);
        TOKS.fetch_add(tokens as u64, Ordering::Relaxed);
    }
}

/// サンプリング 1 回 (1 トークン) を数える。
#[inline]
pub fn note_sample() {
    if on() {
        SAMPLES.fetch_add(1, Ordering::Relaxed);
    }
}

/// PLE 表の pread 1 回ぶんを積む (`rows` はその読みに含まれる行数)。
#[inline]
pub fn note_ple_pread(rows: usize, bytes: usize) {
    PLE_READS.fetch_add(1, Ordering::Relaxed);
    PLE_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    PLE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// `/proc/self/stat` の `(minflt, majflt)`。読めない環境では `None`。
pub fn faults() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // comm は括弧を含みうるので、最後の ')' 以降をフィールド 3 として数える。
    let rest = &s[s.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0] = state(3) なので minflt(10) = f[7], majflt(12) = f[9]。
    let minflt = f.get(7)?.parse().ok()?;
    let majflt = f.get(9)?.parse().ok()?;
    Some((minflt, majflt))
}

/// `before` からのフォルト差分を積む (取得できなければ何もしない)。
pub fn note_fault_delta(before: Option<(u64, u64)>) {
    let (m0, a0) = match before {
        Some(v) => v,
        None => return,
    };
    if let Some((m1, a1)) = faults() {
        MINFLT.fetch_add(m1.saturating_sub(m0), Ordering::Relaxed);
        MAJFLT.fetch_add(a1.saturating_sub(a0), Ordering::Relaxed);
    }
}

// ---- 逐次 prefill のゲート -------------------------------------------------
//
// `forward_token_at` は decode からも呼ばれる。区間計測を prefill のときだけ
// 立てるため、prefill のループの間だけスレッドローカルを true にする。
// 計測は呼び出しスレッドの上で行うので、rayon のワーカーには伝播しない
// (区間の `Instant` は層ループの外側でしか取らない)。

thread_local! {
    static IN_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 逐次 prefill の区間計測を立てるガード。drop で元に戻す。
#[must_use = "ガードを落とすと計測が止まる"]
pub struct PrefillGuard {
    prev: bool,
    active: bool,
}

impl Drop for PrefillGuard {
    fn drop(&mut self) {
        if self.active {
            IN_PREFILL.with(|c| c.set(self.prev));
        }
    }
}

/// 逐次 prefill のループの間、[`measuring`] を true にする。
pub fn enter_prefill() -> PrefillGuard {
    if !on() {
        return PrefillGuard {
            prev: false,
            active: false,
        };
    }
    let prev = IN_PREFILL.with(|c| c.replace(true));
    PrefillGuard { prev, active: true }
}

/// いま区間を積むべきか (計測が on で、逐次 prefill の中)。
#[inline]
pub fn measuring() -> bool {
    on() && IN_PREFILL.with(|c| c.get())
}

// ---- レポート ---------------------------------------------------------------

fn ns_of(slot: usize) -> u64 {
    NS[slot].load(Ordering::Relaxed)
}

/// 区間の累積 (ns)。テストとレポート用。
pub fn ns(slot: usize) -> u64 {
    ns_of(slot)
}

/// 2 行のレポートを返して累積を 0 に戻す。無効時は空 (出力ゼロ)。
///
/// 1 行目: prefill wall とその内側の区間 (`other` = wall − 内側の合計)。
/// 2 行目: prefill wall の外 (tokenize / sample) と PLE の I/O・フォルト。
#[must_use]
pub fn report(tag: &str, tok: usize, wall_ns: u64) -> Vec<String> {
    if !on() {
        return Vec::new();
    }
    let ms = |s: usize| ns_of(s) as f64 / 1e6;
    let inside: u64 = IN_WALL.iter().map(|&s| ns_of(s)).sum();
    let wall = wall_ns as f64 / 1e6;
    // inside は ns、wall は ms (f64)。ここで揃える。
    let other = wall - inside as f64 / 1e6;
    let mut l1 = format!(
        "[pre-kprof] {tag} tok={tok} prefill_calls={} wall={wall:.1}ms",
        PREFILLS.load(Ordering::Relaxed)
    );
    for s in IN_WALL {
        l1.push_str(&format!(" | {} {:.1}", SLOT_NAMES[s], ms(s)));
    }
    l1.push_str(&format!(" | other {other:.1} ms"));

    let samples = SAMPLES.load(Ordering::Relaxed);
    let sample_ms = ms(SAMPLE);
    let per = if samples > 0 {
        sample_ms / samples as f64
    } else {
        0.0
    };
    let reads = PLE_READS.load(Ordering::Relaxed);
    let rows = PLE_ROWS.load(Ordering::Relaxed);
    let bytes = PLE_BYTES.load(Ordering::Relaxed);
    let per_read = if reads > 0 {
        bytes as f64 / reads as f64
    } else {
        0.0
    };
    let l2 = format!(
        "[pre-kprof] {tag} outside-wall tokenize {:.1} | sample {:.1} ms ({per:.3} ms/tok x{samples}) | \
         ple rows={rows} reads={reads} bytes={bytes} avg_read={per_read:.0}B \
         minflt=+{} majflt=+{}",
        ms(TOKENIZE),
        sample_ms,
        MINFLT.load(Ordering::Relaxed),
        MAJFLT.load(Ordering::Relaxed)
    );
    reset();
    vec![l1, l2]
}

/// 累積を 0 に戻す。
pub fn reset() {
    for c in NS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    PREFILLS.store(0, Ordering::Relaxed);
    TOKS.store(0, Ordering::Relaxed);
    SAMPLES.store(0, Ordering::Relaxed);
    PLE_READS.store(0, Ordering::Relaxed);
    PLE_ROWS.store(0, Ordering::Relaxed);
    PLE_BYTES.store(0, Ordering::Relaxed);
    MINFLT.store(0, Ordering::Relaxed);
    MAJFLT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// グローバルカウンタを共有するので、テストは 1 本ずつ回す。
    /// (`qwen4exp` 側の区間計測テストと同じロックを使う。)
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::model::moe_expert::moe_test_lock()
    }

    #[test]
    fn off_records_nothing() {
        let _g = lock();
        set_for_test(false);
        reset();
        let t = stamp();
        assert!(t.is_none(), "off で Instant を取った");
        mark(t, EMBED);
        add_ns(PLE, 1234);
        // add_ns は呼び出し側の判断で呼ばれる (on の判定は呼び出し側の責務) が、
        // off のレポートは空で何も出さない。
        assert!(report("off", 1, 1_000_000).is_empty(), "off でレポートが出た");
        reset();
    }

    #[test]
    fn on_accumulates_and_report_resets() {
        let _g = lock();
        set_for_test(true);
        reset();
        let t = stamp();
        assert!(t.is_some(), "on で Instant が取れない");
        mark(t, EMBED);
        add_ns(PLE, 2_000_000);
        add_ns(LAYER, 5_000_000);
        note_prefill(8);
        note_ple_pread(16, 1440);
        let lines = report("unit", 8, 9_000_000);
        assert_eq!(lines.len(), 2, "レポートは 2 行: {lines:?}");
        assert!(lines[0].starts_with("[pre-kprof] unit tok=8"), "{:?}", lines[0]);
        assert!(lines[0].contains("ple 2.0"), "{:?}", lines[0]);
        assert!(lines[0].contains("layer 5.0"), "{:?}", lines[0]);
        assert!(lines[0].contains("other 2.0"), "{:?}", lines[0]);
        assert!(lines[1].contains("reads=1"), "{:?}", lines[1]);
        assert!(lines[1].contains("rows=16"), "{:?}", lines[1]);
        assert!(lines[1].contains("avg_read=1440B"), "{:?}", lines[1]);
        // report は累積を消す
        assert_eq!(ns(PLE), 0, "report のあとに累積が残った");
        assert_eq!(ns(LAYER), 0, "report のあとに累積が残った");
        set_for_test(false);
        reset();
    }

    #[test]
    fn prefill_guard_gates_measuring() {
        let _g = lock();
        set_for_test(true);
        reset();
        assert!(!measuring(), "ガードの外で measuring が true");
        {
            let _pf = enter_prefill();
            assert!(measuring(), "ガードの中なのに measuring が false");
            // 入れ子でも外側の値に戻ること
            {
                let _n = enter_prefill();
                assert!(measuring());
            }
            assert!(measuring(), "内側のガードを落としたら外側が消えた");
        }
        assert!(!measuring(), "ガードを落としても measuring が true");
        set_for_test(false);
    }

    #[test]
    fn fault_counters_are_readable_and_non_decreasing() {
        let _g = lock();
        let a = faults();
        let b = faults();
        match (a, b) {
            (Some((m0, a0)), Some((m1, a1))) => {
                assert!(m1 >= m0, "minflt が減った: {m0} -> {m1}");
                assert!(a1 >= a0, "majflt が減った: {a0} -> {a1}");
            }
            // /proc が無い環境では何も主張しない (Linux 以外)。
            _ => eprintln!("faults() が読めない環境なのでスキップ"),
        }
    }
}
