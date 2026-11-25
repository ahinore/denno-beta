//! task 850: 層間の expert 共起 (Markov bigram) で層 L+1 の expert を先読みする。
//!
//! # なぜ要るか
//!
//! MoE の expert 選択は層をまたいで強く相関する。層 L で expert `e` が選ばれた
//! とき、層 L+1 で選ばれやすい expert はかなり絞れる。これは**同じトークン内**の
//! 話で、task 266 の `DENNO_DEC_PREF` (前の**トークン**の同じ層) とは別の信号。
//!
//! 層 L のルータが決まった直後に (層 L の expert 計算が走る前に) 層 L+1 の
//! 予測 expert のページを背景スレッドへ渡せば、層 L の計算と読みが重なる。
//!
//! # 何をするか
//!
//! オンラインの有界テーブル `(層 L, expert e) -> 上位 K 個の (層 L+1, expert f)` を
//! 数える。更新は Space-Saving (満杯なら最小カウントの枠を置換する。件数は近似)。
//! メモリは `entry 数 x K x 4 バイト` で上限つき (既定 131,072 エントリ = 2 MiB)。
//!
//! 予測は層 L の選択 `sel` の各 expert の後継スコア (カウント和) を降順に並べ、
//! 上位 `max` 本を返す。同点は expert id 昇順 (決定的)。
//!
//! # ゲート
//!
//! `DENNO_EXPERT_MARKOV_PREFETCH=1` のときだけ動く (**既定 off**)。
//! off では [`observe`] / [`predict`] が `enabled()` の atomic load 1 回で返り、
//! テーブルも確保しない。値・経路は 1 バイトも変わらない (先読みは助言だけ)。
//!
//! 調整: `DENNO_EXPERT_MARKOV_TOPK` (1 キーの後継数、既定 4、上限 8)、
//! `DENNO_EXPERT_MARKOV_MAX` (1 層で積む予測本数、既定 24)、
//! `DENNO_EXPERT_MARKOV_MAX_ENTRIES` (テーブルのエントリ上限、既定 131072)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// `DENNO_EXPERT_MARKOV_PREFETCH`: 層間共起の先読み。**既定 off**。
pub const ENV: &str = "DENNO_EXPERT_MARKOV_PREFETCH";
/// `DENNO_EXPERT_MARKOV_TOPK`: 1 キーあたりの後継枠数 (既定 4)。
pub const TOPK_ENV: &str = "DENNO_EXPERT_MARKOV_TOPK";
/// `DENNO_EXPERT_MARKOV_MAX`: 1 層で積む予測 expert 本数の上限 (既定 24)。
pub const MAX_ENV: &str = "DENNO_EXPERT_MARKOV_MAX";
/// `DENNO_EXPERT_MARKOV_MAX_ENTRIES`: テーブルのエントリ上限 (既定 131072)。
pub const CAP_ENV: &str = "DENNO_EXPERT_MARKOV_MAX_ENTRIES";

/// 1 キーあたりの後継枠のハード上限 (Slot は 4 バイトなので 8 枠 = 32 バイト/キー)。
pub const MAX_SLOTS: usize = 8;
/// 後継枠の既定値。
pub const DEFAULT_SLOTS: usize = 4;
/// 1 層で積む予測本数の既定値。
pub const DEFAULT_MAX: usize = 24;
/// エントリ上限の既定値 (4 枠 x 4 バイト = 16 バイト/キー x 131072 = 2 MiB)。
pub const DEFAULT_CAP: usize = 131_072;

/// [`ENV`] の解釈 (純関数)。`1` / `on` / `true` / `yes` だけが真 (既定 off)。
pub fn on_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on_from_value(std::env::var(ENV).ok().as_deref()))
}

/// テスト専用: ゲートを強制する (`None` で env 読みに戻す)。
#[cfg(test)]
pub fn set_enabled_for_tests(v: Option<bool>) {
    OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

fn usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// 1 キーあたりの後継枠数 (`1..=`[`MAX_SLOTS`])。
pub fn slots_per_key() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| usize_env(TOPK_ENV, DEFAULT_SLOTS).min(MAX_SLOTS))
}

/// 1 層で積む予測本数。
pub fn max_per_layer() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| usize_env(MAX_ENV, DEFAULT_MAX))
}

/// エントリ上限。
fn cap_entries() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| usize_env(CAP_ENV, DEFAULT_CAP))
}

// ---------------------------------------------------------------------------
// テーブル
// ---------------------------------------------------------------------------

/// 後継 1 枠。`c == 0` は空き。
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Slot {
    e: u16,
    c: u16,
}

/// 有界な `(層, expert) -> 後継上位 K` 表。
struct Table {
    n_layer: usize,
    n_expert: usize,
    slots: usize,
    cap: usize,
    /// `rows * slots`。`row = layer * n_expert + expert` (層優先)。
    data: Vec<Slot>,
}

impl Table {
    fn new(n_layer: usize, n_expert: usize, slots: usize, cap: usize) -> Self {
        let rows = (n_layer * n_expert).min(cap.max(1));
        Self {
            n_layer,
            n_expert,
            slots,
            cap: cap.max(1),
            data: vec![Slot::default(); rows * slots],
        }
    }

    fn rows(&self) -> usize {
        self.data.len() / self.slots
    }

    /// 行番号。上限で切れた後ろのキーは `None` (数えない)。
    fn row(&self, layer: usize, e: usize) -> Option<usize> {
        if layer >= self.n_layer || e >= self.n_expert {
            return None;
        }
        let i = layer * self.n_expert + e;
        (i < self.rows()).then_some(i)
    }

    /// 次元を広げる (古いカウントは写す)。広がらないときは何もしない。
    fn grow_to(&mut self, n_layer: usize, n_expert: usize) {
        if n_layer <= self.n_layer && n_expert <= self.n_expert {
            return;
        }
        let n_layer = n_layer.max(self.n_layer);
        let n_expert = n_expert.max(self.n_expert);
        let mut next = Table::new(n_layer, n_expert, self.slots, self.cap);
        for layer in 0..self.n_layer {
            for e in 0..self.n_expert {
                let Some(from) = self.row(layer, e) else { continue };
                let Some(to) = next.row(layer, e) else { continue };
                let a = from * self.slots;
                let b = to * next.slots;
                next.data[b..b + self.slots]
                    .copy_from_slice(&self.data[a..a + self.slots]);
            }
        }
        *self = next;
    }

    /// 遷移 `(layer, prev_e) -> next_e` を 1 回数える。
    ///
    /// 空きがあればそこへ、満杯なら **最小カウントの枠を置換** する
    /// (Space-Saving。件数は近似値)。
    fn update(&mut self, layer: usize, prev_e: usize, next_e: usize) {
        let Some(i) = self.row(layer, prev_e) else {
            return;
        };
        let base = i * self.slots;
        let seg = &mut self.data[base..base + self.slots];
        if let Some(s) = seg.iter_mut().find(|s| s.c > 0 && s.e == next_e as u16) {
            s.c = s.c.saturating_add(1);
            return;
        }
        if let Some(s) = seg.iter_mut().find(|s| s.c == 0) {
            *s = Slot {
                e: next_e as u16,
                c: 1,
            };
            return;
        }
        // 全部埋まっている: 最小カウントの枠を置換する。
        if let Some((mi, m)) = seg
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.c))
            .min_by_key(|&(_, c)| c)
        {
            seg[mi] = Slot {
                e: next_e as u16,
                c: m.saturating_add(1),
            };
        }
    }

    /// `sel` の各 expert の後継をスコア (カウント和) 降順・同点 id 昇順で返す。
    fn predict(&self, layer: usize, sel: &[usize], max: usize) -> Vec<u16> {
        if max == 0 {
            return Vec::new();
        }
        let mut acc: Vec<(u32, u16)> = Vec::with_capacity(sel.len() * self.slots);
        for &e in sel {
            let Some(i) = self.row(layer, e) else {
                continue;
            };
            let base = i * self.slots;
            for s in &self.data[base..base + self.slots] {
                if s.c > 0 {
                    acc.push((s.c as u32, s.e));
                }
            }
        }
        if acc.is_empty() {
            return Vec::new();
        }
        // 同じ後継 id のスコアを足す (id 昇順に畳んでからスコアで並べ直す)。
        acc.sort_unstable_by_key(|&(_, e)| e);
        let mut merged: Vec<(u32, u16)> = Vec::with_capacity(acc.len());
        for (c, e) in acc {
            match merged.last_mut() {
                Some(last) if last.1 == e => last.0 = last.0.saturating_add(c),
                _ => merged.push((c, e)),
            }
        }
        merged.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        merged.truncate(max);
        merged.into_iter().map(|(_, e)| e).collect()
    }
}

/// プロセス全体の状態。decode は 1 トークン 1 スレッドで層を順に呼ぶ。
struct State {
    table: Table,
    /// 直前の `observe` の (層, 選択)。同じトークン内の隣接層だけを数える。
    prev_layer: Option<usize>,
    prev_sel: Vec<u16>,
    /// 層ごとの「予測した後継」(当たり判定用)。
    pending: Vec<Vec<u16>>,
    /// 記録したトークン数 (層 0 を見た回数)。
    tokens: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// 積算 (1 リクエストぶん)
// ---------------------------------------------------------------------------

static LAYERS: AtomicU64 = AtomicU64::new(0);
static PAIRS: AtomicU64 = AtomicU64::new(0);
static PRED: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static SUB: AtomicU64 = AtomicU64::new(0);
static SKIP_RES: AtomicU64 = AtomicU64::new(0);
static SKIP_PART: AtomicU64 = AtomicU64::new(0);
static TOKENS: AtomicU64 = AtomicU64::new(0);

/// 層 `layer` の選択 `sel` を記録し、隣接層の遷移を数える。
///
/// ゲート off では何もしない。同じトークン内で `layer == prev_layer + 1` の
/// ときだけ遷移として数える (層 0 を見たらトークンの境目とみなす)。
pub fn observe(layer: usize, sel: &[usize]) {
    if !enabled() {
        return;
    }
    LAYERS.fetch_add(1, Ordering::Relaxed);
    let slots = slots_per_key();
    let cap = cap_entries();
    let Ok(mut g) = STATE.lock() else {
        return;
    };
    let n_expert = sel.iter().copied().max().map_or(1, |m| m + 1);
    if g.is_none() {
        *g = Some(State {
            table: Table::new(layer + 1, n_expert, slots, cap),
            prev_layer: None,
            prev_sel: Vec::new(),
            pending: Vec::new(),
            tokens: 0,
        });
    }
    let st = g.as_mut().expect("直前に挿入した");
    st.table.grow_to(layer + 1, n_expert);
    if layer == 0 {
        st.tokens = st.tokens.saturating_add(1);
        TOKENS.fetch_add(1, Ordering::Relaxed);
    }
    if st.pending.len() <= layer {
        st.pending.resize_with(layer + 1, Vec::new);
    }
    // 当たり: この層について前回予測した集合に、実際の選択が何本入っていたか。
    if !st.pending[layer].is_empty() {
        let mut n = 0u64;
        for &e in sel {
            if st.pending[layer].contains(&(e as u16)) {
                n += 1;
            }
        }
        HITS.fetch_add(n, Ordering::Relaxed);
        st.pending[layer].clear();
    }
    // 遷移の記録 (同じトークンの隣接層だけ)。
    if st.prev_layer == Some(layer.wrapping_sub(1)) && layer > 0 {
        let prev: Vec<usize> = st.prev_sel.iter().map(|&e| e as usize).collect();
        let mut n = 0u64;
        for &p in &prev {
            for &f in sel {
                st.table.update(layer - 1, p, f);
                n += 1;
            }
        }
        PAIRS.fetch_add(n, Ordering::Relaxed);
    }
    st.prev_layer = Some(layer);
    st.prev_sel.clear();
    st.prev_sel.extend(sel.iter().map(|&e| e as u16));
}

/// 層 `layer` の選択 `sel` から、層 `layer+1` の候補をスコア降順で `max` 本返す。
///
/// 予測は [`note_prediction`] で「層 `layer+1` の予測」として控える
/// (次に `observe(layer+1)` が来たときに当たりを数える)。
pub fn predict(layer: usize, sel: &[usize], max: usize) -> Vec<u16> {
    if !enabled() {
        return Vec::new();
    }
    let Ok(g) = STATE.lock() else {
        return Vec::new();
    };
    match g.as_ref() {
        Some(st) => st.table.predict(layer, sel, max),
        None => Vec::new(),
    }
}

/// 予測した後継を層 `layer` の控えとして積む (当たり判定の分母)。
pub fn note_prediction(layer: usize, preds: &[u16]) {
    if !enabled() || preds.is_empty() {
        return;
    }
    PRED.fetch_add(preds.len() as u64, Ordering::Relaxed);
    let Ok(mut g) = STATE.lock() else {
        return;
    };
    if let Some(st) = g.as_mut() {
        if st.pending.len() <= layer {
            st.pending.resize_with(layer + 1, Vec::new);
        }
        st.pending[layer].clear();
        st.pending[layer].extend_from_slice(preds);
    }
}

/// 背景スレッドへ実際に積んだ本数を加算する。
pub fn note_submitted(n: usize) {
    SUB.fetch_add(n as u64, Ordering::Relaxed);
}

/// 積む段で飛ばした理由 (VRAM 常駐 / mmap でない) を加算する。
pub fn note_skips(resident: u64, no_part: u64) {
    SKIP_RES.fetch_add(resident, Ordering::Relaxed);
    SKIP_PART.fetch_add(no_part, Ordering::Relaxed);
}

/// 診断: テーブルのバイト数 (上限の検査用)。
pub fn table_bytes() -> usize {
    let Ok(g) = STATE.lock() else { return 0 };
    match g.as_ref() {
        Some(st) => st.table.data.len() * std::mem::size_of::<Slot>(),
        None => 0,
    }
}

/// 1 リクエストぶんの 1 行。ゲート off / 実績 0 なら `None`。
pub fn stats_line() -> Option<String> {
    if !enabled() {
        return None;
    }
    let layers = LAYERS.load(Ordering::Relaxed);
    if layers == 0 {
        return None;
    }
    let pred = PRED.load(Ordering::Relaxed);
    let hits = HITS.load(Ordering::Relaxed);
    let sub = SUB.load(Ordering::Relaxed);
    Some(format!(
        "[exp-markov] tok={} layers={layers} pairs={} pred={pred} sub={sub} sub_frac={:.2} \
         hit={hits} hit_frac={:.2} skip_res={} skip_part={} keys={} kb={:.0}",
        TOKENS.load(Ordering::Relaxed),
        PAIRS.load(Ordering::Relaxed),
        if pred > 0 { sub as f64 / pred as f64 } else { 0.0 },
        if pred > 0 { hits as f64 / pred as f64 } else { 0.0 },
        SKIP_RES.load(Ordering::Relaxed),
        SKIP_PART.load(Ordering::Relaxed),
        table_bytes().max(1) / (slots_per_key() * std::mem::size_of::<Slot>()).max(1),
        table_bytes() as f64 / 1024.0
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。テーブルは学習を続けるので消さない。
pub fn stats_reset() {
    for c in [
        &LAYERS, &PAIRS, &PRED, &HITS, &SUB, &SKIP_RES, &SKIP_PART, &TOKENS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// テスト専用: `(layers, pairs, pred, hits, sub)` の現在値。
#[cfg(test)]
pub fn test_counters() -> (u64, u64, u64, u64, u64) {
    (
        LAYERS.load(Ordering::Relaxed),
        PAIRS.load(Ordering::Relaxed),
        PRED.load(Ordering::Relaxed),
        HITS.load(Ordering::Relaxed),
        SUB.load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// テーブルはプロセス全体で共有するので、テストは直列化する。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reset_state() {
        *STATE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        stats_reset();
    }

    #[test]
    fn gate_values() {
        assert!(!on_from_value(None));
        assert!(!on_from_value(Some("")));
        assert!(!on_from_value(Some("0")));
        assert!(!on_from_value(Some("off")));
        assert!(on_from_value(Some("1")));
        assert!(on_from_value(Some(" TRUE ")));
        assert!(on_from_value(Some("on")));
    }

    /// ゲート off では状態を 1 バイトも作らない。
    #[test]
    fn off_is_inert() {
        let _g = guard();
        reset_state();
        set_enabled_for_tests(Some(false));
        observe(0, &[1, 2]);
        assert!(predict(0, &[1, 2], 4).is_empty());
        assert_eq!(table_bytes(), 0);
        assert!(stats_line().is_none());
        set_enabled_for_tests(None);
    }

    /// 相関のある合成トレース: 層 L の expert e の後継は常に `(2*e + 1) % ne`。
    /// 学習後はほぼ全的中する (予測器が実際に学習している証拠)。
    #[test]
    fn learns_correlated_successors() {
        let _g = guard();
        reset_state();
        set_enabled_for_tests(Some(true));
        let ne = 32usize;
        let k = 4usize;
        let n_layer = 4usize;
        let succ = |e: usize| (2 * e + 1) % ne;
        let mut hit = 0u64;
        let mut total = 0u64;
        for tok in 0..60usize {
            // トークンごとに層 0 の選択を変える (多様性を作る)。
            let mut sel: Vec<usize> = (0..k).map(|i| (tok * 3 + i * 5) % ne).collect();
            sel.sort_unstable();
            for layer in 0..n_layer {
                observe(layer, &sel);
                if layer + 1 < n_layer {
                    let preds = predict(layer, &sel, 8);
                    note_prediction(layer + 1, &preds);
                    if tok >= 30 {
                        // 後継の真値が予測集合に入っているか。
                        for &e in &sel {
                            total += 1;
                            if preds.contains(&(succ(e) as u16)) {
                                hit += 1;
                            }
                        }
                    }
                }
                sel = sel.iter().map(|&e| succ(e)).collect();
                sel.sort_unstable();
                sel.dedup();
            }
        }
        assert!(total > 0, "計測区間が空");
        let frac = hit as f64 / total as f64;
        assert!(frac > 0.9, "相関トレースの的中率が低い: {hit}/{total}");
        set_enabled_for_tests(None);
    }

    /// 無相関のトレースでは的中率が高くならない (学習が「何でも当たる」わけではない)。
    #[test]
    fn uncorrelated_trace_is_not_overfit() {
        let _g = guard();
        reset_state();
        set_enabled_for_tests(Some(true));
        let ne = 64usize;
        let mut hit = 0u64;
        let mut total = 0u64;
        let mut seed = 12345u64;
        let mut next = || {
            // xorshift (テスト内の決定的な擬似乱数)。
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % ne as u64) as usize
        };
        for tok in 0..80usize {
            let mut sel: Vec<usize> = (0..4).map(|_| next()).collect();
            sel.sort_unstable();
            sel.dedup();
            for layer in 0..3usize {
                observe(layer, &sel);
                if layer + 1 < 3 {
                    let preds = predict(layer, &sel, 4);
                    note_prediction(layer + 1, &preds);
                    if tok >= 60 {
                        let mut nsel: Vec<usize> = (0..4).map(|_| next()).collect();
                        nsel.sort_unstable();
                        nsel.dedup();
                        for &e in &nsel {
                            total += 1;
                            if preds.contains(&(e as u16)) {
                                hit += 1;
                            }
                        }
                        sel = nsel;
                    }
                }
            }
        }
        assert!(total > 0);
        assert!(
            hit as f64 / total as f64 <= 0.35,
            "無相関で当たりすぎ: {hit}/{total}"
        );
        set_enabled_for_tests(None);
    }

    /// メモリは上限で切られる (多数の expert キーを入れても増え続けない)。
    #[test]
    fn memory_is_bounded() {
        let _g = guard();
        reset_state();
        set_enabled_for_tests(Some(true));
        let slots = 4usize;
        let cap = 1024usize;
        let mut t = Table::new(4, 4096, slots, cap);
        for l in 0..4usize {
            for e in 0..4096usize {
                let f = (e * 7 + 1) % 4096;
                t.update(l, e, f);
            }
        }
        assert!(t.rows() <= cap, "rows={} cap={cap}", t.rows());
        assert!(
            t.data.len() * std::mem::size_of::<Slot>() <= cap * slots * 4,
            "バイト数が上限を超えた"
        );
        // 上限の外側のキーは数えない (静かに落ちるだけ)。
        let before = t.data.clone();
        t.update(3, 4095, 1);
        assert_eq!(before, t.data);
        set_enabled_for_tests(None);
    }

    /// 1 キーの後継枠は slots 個で頭打ち、カウントは飽和する。
    #[test]
    fn slots_are_bounded_and_counts_saturate() {
        let _g = guard();
        let mut t = Table::new(1, 4, 2, 1024);
        for f in 0..10usize {
            t.update(0, 0, f % 4);
        }
        let seg = &t.data[0..2];
        assert!(seg.iter().all(|s| s.c > 0));
        // 同じ後継を飽和まで数えてもオーバーフローしない。
        let mut t2 = Table::new(1, 2, 1, 1024);
        for _ in 0..200_000 {
            t2.update(0, 0, 1);
        }
        let c = t2.data[0].c;
        assert!(c > 0);
        assert_eq!(c, c.saturating_add(1).min(u16::MAX).max(c));
    }

    /// 予測の並びは決定的 (同点は expert id 昇順)。
    #[test]
    fn prediction_order_is_deterministic() {
        let _g = guard();
        let mut t = Table::new(2, 8, 4, 1024);
        t.update(0, 1, 5);
        t.update(0, 1, 3);
        t.update(0, 2, 5);
        t.update(0, 2, 7);
        let a = t.predict(0, &[1, 2], 4);
        let b = t.predict(0, &[1, 2], 4);
        assert_eq!(a, b);
        // 5 が 2 回で先頭、同点 3/7 は id 昇順。
        assert_eq!(a, vec![5, 3, 7]);
    }

    /// 次元が広がっても古いカウントは残る。
    #[test]
    fn growth_preserves_counts() {
        let _g = guard();
        let mut t = Table::new(1, 2, 2, 1024);
        t.update(0, 0, 1);
        t.update(0, 0, 1);
        t.grow_to(3, 16);
        assert!(t.n_layer >= 3 && t.n_expert >= 16);
        assert_eq!(t.predict(0, &[0], 2), vec![1]);
    }
}
