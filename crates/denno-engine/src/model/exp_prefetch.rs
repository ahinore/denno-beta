//! task 266: decode の expert 読み (SSD) を**1 トークン先に**出す。
//!
//! # なぜ要るか
//!
//! UD-Q4_K_XL (flash-next) の decode は 66 ms/token のうち **40 ms が
//! VRAM に載らない expert の読み** (46 本/token x 3.07 MB x 0.86 ms。
//! REPORT_P261.md)。読みはルータが決めた直後に**その層の中で**直列に起きるので、
//! 前の層の計算とも重ならない。1 本 0.86 ms は 3.07 MB / 3.6 GB/s で、
//! page cache に載っているときの 0.16 ms (19.4 GB/s) とは別物 = ディスク律速。
//!
//! # 何をするか
//!
//! decode のトークン先頭で、**1 トークン前の同じ層が選んだ expert** のうち
//! VRAM に載っていないものを背景スレッドへ渡し、`madvise(MADV_WILLNEED)` で
//! ページを page cache に引き込む。トークンが進む間に読むので、その層が実際に
//! ミスしたときのフォルトは page cache 命中になる (転送と H2D の道は従来のまま)。
//!
//! 予測は「直前のトークンの選択」なので、当たれば読みが消え、外れれば
//! ディスクを空回りする (捨てるだけ。値は 1 ビットも変わらない)。
//! 当たり率は [`stats_line`] の `used/sub` で読める。
//!
//! # ゲート
//!
//! `DENNO_DEC_PREF=1` のときだけ動く (**既定 off**)。off では
//! [`note_sel`] が atomic load 1 回で返り、[`begin_token`] は空を返す。
//! 調整は `DENNO_DEC_PREF_MAX` (1 トークンに積む本数、既定 64)、
//! `DENNO_DEC_PREF_THREADS` (読むスレッド数、既定 2)、
//! `DENNO_DEC_PREF_TOUCH` (1 = madvise の代わりにページを実際に触る)。
//!
//! # 触り方
//!
//! `MADV_WILLNEED` は「この範囲を先に読め」の助言で、呼び出したスレッドを
//! 長くは止めない (task 67 の実測ではフォルトで並列に読むほうが速かったが、
//! あれは**同じスレッド**で読んでいた)。ここでは専用スレッドなので、
//! 読みをメインスレッドの外へ出すこと自体が目的。
//! `DENNO_DEC_PREF_TOUCH=1` では 4 KiB ごとに 1 バイトを volatile 読みして
//! フォルトを確定させる (madvise が効かない環境の切り分け用)。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::model::exp_up_worker::Part;

/// `DENNO_DEC_PREF`: 1 トークン前の選択で expert のページを先に読む。**既定 off**。
pub const ENV: &str = "DENNO_DEC_PREF";
/// `DENNO_DEC_PREF_MAX`: 1 トークンに積む expert 本数の上限 (既定 64)。
pub const MAX_ENV: &str = "DENNO_DEC_PREF_MAX";
/// `DENNO_DEC_PREF_THREADS`: 読むスレッド数 (既定 2)。
pub const THREADS_ENV: &str = "DENNO_DEC_PREF_THREADS";
/// `DENNO_DEC_PREF_TOUCH`: 1 なら madvise ではなくページを実際に触る。
pub const TOUCH_ENV: &str = "DENNO_DEC_PREF_TOUCH";

/// 待ち行列の上限 (expert 本数)。溢れた分は積まない (結果は従来と同じ)。
pub const MAX_QUEUE: usize = 256;

/// [`ENV`] の解釈 (純関数)。`1` / `on` / `true` / `yes` だけが真 (既定 off)。
pub fn on_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on_from_value(std::env::var(ENV).ok().as_deref()))
}

fn usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// 1 トークンに積む本数。
pub fn max_per_token() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| usize_env(MAX_ENV, 64))
}

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| usize_env(THREADS_ENV, 2).min(8))
}

fn touch_mode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(TOUCH_ENV).ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

// ---------------------------------------------------------------------------
// 選択の記録 (層ごと)
// ---------------------------------------------------------------------------

/// 直前のトークンで各層が選んだ expert (index 昇順)。[`begin_token`] が返す。
static PREV: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());
/// いま走っているトークンの記録。
static CUR: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());

/// `sel` を層 `layer` の記録として積む (ゲート off では即 return)。
pub fn note_sel(layer: usize, sel: &[usize]) {
    if !enabled() {
        return;
    }
    let Ok(mut cur) = CUR.lock() else {
        return;
    };
    if cur.len() <= layer {
        cur.resize_with(layer + 1, Vec::new);
    }
    let dst = &mut cur[layer];
    dst.clear();
    dst.extend(sel.iter().map(|&e| e as u16));
    drop(cur);
    // 予測の当たりを数える (積んだ集合に入っていれば 1 本ぶん)。
    if let Ok(pf) = PF_KEYS.lock() {
        let mut n = 0u64;
        for &e in sel {
            if pf.contains(&(layer as u16, e as u16)) {
                n += 1;
            }
        }
        USED.fetch_add(n, Ordering::Relaxed);
    }
}

/// トークン先頭で呼ぶ。**直前のトークン**の選択 (層 -> expert id) を返し、
/// 記録を次のトークン用に空にする。
pub fn begin_token(n_layer: usize) -> Vec<Vec<u16>> {
    if !enabled() {
        return Vec::new();
    }
    let snap = {
        let Ok(mut cur) = CUR.lock() else {
            return Vec::new();
        };
        // いまのトークンまでに記録した内容 (= 直前のトークンの選択) を写す。
        let mut next = std::mem::take(&mut *cur);
        next.resize_with(n_layer, Vec::new);
        next.truncate(n_layer);
        *cur = vec![Vec::new(); n_layer];
        if let Ok(mut prev) = PREV.lock() {
            *prev = next.clone();
        }
        if let Ok(mut pf) = PF_KEYS.lock() {
            pf.clear();
        }
        next
    };
    TOKENS.fetch_add(1, Ordering::Relaxed);
    snap
}

// ---------------------------------------------------------------------------
// 背景スレッド
// ---------------------------------------------------------------------------

struct Pool {
    q: Mutex<VecDeque<Vec<Part>>>,
    cv: Condvar,
}

static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);

fn pool() -> &'static Arc<Pool> {
    POOL.get_or_init(|| {
        let p = Arc::new(Pool {
            q: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        });
        for i in 0..threads() {
            let p2 = Arc::clone(&p);
            let _ = std::thread::Builder::new()
                .name(format!("dec-pref-{i}"))
                .spawn(move || worker(p2));
        }
        STARTED.store(true, Ordering::Relaxed);
        p
    })
}

fn worker(p: Arc<Pool>) {
    loop {
        let job = {
            let Ok(mut q) = p.q.lock() else { return };
            loop {
                if let Some(j) = q.pop_front() {
                    break Some(j);
                }
                match p.cv.wait(q) {
                    Ok(g) => q = g,
                    Err(_) => return,
                }
            }
        };
        let Some(parts) = job else { return };
        let t = std::time::Instant::now();
        let mut bytes = 0u64;
        for part in &parts {
            let Some(sl) = part.get() else { continue };
            bytes += sl.len() as u64;
            if touch_mode() {
                touch_pages(sl);
            } else {
                crate::model::moe_expert::prefetch_range(sl);
            }
        }
        BYTES.fetch_add(bytes, Ordering::Relaxed);
        DONE.fetch_add(1, Ordering::Relaxed);
        WORK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// 4 KiB ごとに 1 バイトを読む (フォルトを確定させる)。madvise の切り分け用。
fn touch_pages(b: &[u8]) {
    const PAGE: usize = 4096;
    let mut i = 0usize;
    while i < b.len() {
        let v = unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) };
        std::hint::black_box(v);
        i += PAGE;
    }
}

/// 1 トークンぶんの job (`[gate, up, down]` の 3 Part) を積む。
///
/// 待ち行列が満杯なら**積まない** (落としても従来経路と同じ結果になる)。
/// 戻り値は積んだ本数。
pub fn submit(jobs: Vec<Vec<Part>>) -> usize {
    if !enabled() || jobs.is_empty() {
        return 0;
    }
    let p = pool();
    let Ok(mut q) = p.q.lock() else { return 0 };
    let mut n = 0usize;
    for j in jobs {
        if q.len() >= MAX_QUEUE {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        q.push_back(j);
        n += 1;
    }
    let len = q.len() as u64;
    if len > QUEUE_MAX.load(Ordering::Relaxed) {
        QUEUE_MAX.store(len, Ordering::Relaxed);
    }
    drop(q);
    p.cv.notify_all();
    SUBMITTED.fetch_add(n as u64, Ordering::Relaxed);
    n
}

/// 積んだ expert の鍵 (当たり判定用)。
static PF_KEYS: Mutex<Vec<(u16, u16)>> = Mutex::new(Vec::new());

/// 積む予定の鍵を「積んだ集合」に写す (当たり率の分母)。
pub fn note_submitted_keys(keys: &[(u16, u16)]) {
    if !enabled() {
        return;
    }
    if let Ok(mut pf) = PF_KEYS.lock() {
        pf.extend_from_slice(keys);
    }
}

// ---------------------------------------------------------------------------
// 統計 (1 リクエストぶん)
// ---------------------------------------------------------------------------

static TOKENS: AtomicU64 = AtomicU64::new(0);
static SUBMITTED: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static USED: AtomicU64 = AtomicU64::new(0);
static WORK_NS: AtomicU64 = AtomicU64::new(0);
static QUEUE_MAX: AtomicU64 = AtomicU64::new(0);

/// 1 リクエストぶんの 1 行。ゲート off / 実績 0 なら `None`。
pub fn stats_line() -> Option<String> {
    if !enabled() {
        return None;
    }
    let sub = SUBMITTED.load(Ordering::Relaxed);
    if sub == 0 {
        return None;
    }
    let tokens = TOKENS.load(Ordering::Relaxed).max(1);
    let used = USED.load(Ordering::Relaxed);
    let done = DONE.load(Ordering::Relaxed);
    let ns = WORK_NS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let gbs = if ns > 0 {
        bytes as f64 / (ns as f64 / 1e9) / 1e9
    } else {
        0.0
    };
    Some(format!(
        "[dec-pref] tokens={tokens} sub={sub} sub/tok={:.1} used={used} used_frac={:.2} \
         done={done} dropped={} bytes={:.1}MB gbs={gbs:.2} work={:.1}ms queue_max={} thr={} touch={}",
        sub as f64 / tokens as f64,
        if sub > 0 { used as f64 / sub as f64 } else { 0.0 },
        DROPPED.load(Ordering::Relaxed),
        bytes as f64 / 1e6,
        ns as f64 / 1e6,
        QUEUE_MAX.load(Ordering::Relaxed),
        threads(),
        u8::from(touch_mode())
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [
        &TOKENS, &SUBMITTED, &DONE, &DROPPED, &BYTES, &USED, &WORK_NS, &QUEUE_MAX,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// ワーカーが起動済みか (テスト・診断用)。
pub fn started() -> bool {
    STARTED.load(Ordering::Relaxed)
}

/// テスト用: 積算の現在値 (tokens, submitted, done, bytes)。
pub fn counters() -> (u64, u64, u64, u64) {
    (
        TOKENS.load(Ordering::Relaxed),
        SUBMITTED.load(Ordering::Relaxed),
        DONE.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_off_by_default() {
        assert!(!on_from_value(None));
        assert!(!on_from_value(Some("")));
        assert!(!on_from_value(Some("0")));
        assert!(!on_from_value(Some("off")));
        assert!(on_from_value(Some("1")));
        assert!(on_from_value(Some(" TRUE ")));
        assert!(on_from_value(Some("on")));
    }

    /// ゲート off では記録も積み込みもしない (1 バイトも状態が動かない)。
    #[test]
    fn off_is_inert() {
        if enabled() {
            return;
        }
        note_sel(0, &[1, 2, 3]);
        assert!(!started());
        assert!(begin_token(4).is_empty());
        assert!(stats_line().is_none());
        assert_eq!(submit(Vec::new()), 0);
    }
}
