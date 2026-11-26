//! task 874: cold prefill の選択 expert を f32 へ展開したまま再利用するキャッシュ。
//!
//! # なぜ要るか (狙い)
//!
//! 層優先のバッチ prefill は 1 チャンクぶんのトークンを層ごとにまとめて流す。
//! CPU で選択 expert の行列積を回すとき、量子ブロックは 1 行ごとに f32 へ
//! 展開されながら FMA に食われる (融合カーネル)。同じ expert が **次の
//! チャンクでも選ばれる** と、その展開をもう一度払う。
//! そこで展開済みの f32 を (層, expert) 単位で覚えておき、次のチャンクで
//! 同じ expert が来たら展開を飛ばす。
//!
//! # 形
//!
//! - `DENNO_PF_DEQUANT_CACHE_MB` (既定 0 = off)。値は **prefill 全体**の予算 (MiB)。
//! - 予算は層数で割って層ごとの上限にする。層優先 prefill では 1 チャンクが
//!   全 48 層を通るので、チャンクを跨いで残すには全層ぶんが同時に居る必要がある。
//! - 1 expert あたり gate(ff×e) + up(ff×e) + down(e×ff) の f32 =
//!   `12 * e * ff` バイト (実形 e=2560 / ff=640 で 19.66 MB)。
//! - 予算が 1 expert ぶんに満たない層ではキャッシュしない (展開だけして捨てない)。
//! - prefill の終端で全部捨てる ([`finish_prefill`])。decode のあいだは抱えない。
//!
//! # 数値
//!
//! キャッシュ経路は「展開済み f32 を f32 GEMM で回す」ので、融合カーネル
//! (量子ブロックから直接 FMA) とは **累積順が違い、ビット一致しない**。
//! 1 要素の値は `dequant_row_into` の出力をそのまま掛けるので、展開そのものは
//! 同一。差は丸めの入り方だけ (`moe_expert::gemm_rows_f32` の doc を参照)。
//! 既定 off なので、off のときの経路は 1 バイトも変わらない。
//!
//! # 統計
//!
//! `DENNO_PF_DEQUANT_CACHE_STATS=1` で prefill の終端に 1 行出す。キャッシュが
//! off でも「選択の延べ数 / 層ごとの distinct expert 数」は数えるので、
//! **再展開の重複度** (延べ / distinct) をキャッシュ無しで測れる。

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::kquant::{dequant_row_into, GgmlType};
use crate::model::moe_expert;

/// `DENNO_PF_DEQUANT_CACHE_MB`: prefill 全体のキャッシュ予算 (MiB、既定 0 = off)。
pub const ENV: &str = "DENNO_PF_DEQUANT_CACHE_MB";

/// `DENNO_PF_DEQUANT_CACHE_STATS=1` で prefill 終端に 1 行出す。
pub const STATS_ENV: &str = "DENNO_PF_DEQUANT_CACHE_STATS";

/// [`ENV`] の解釈 (純関数)。未設定・数値でない値は 0 (= off)。
pub(crate) fn cap_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s.parse::<usize>().unwrap_or(0),
        None => 0,
    }
}

/// テストから予算を差し替えるための上書き (-1 = 上書きなし)。
static TEST_CAP_BYTES: AtomicI64 = AtomicI64::new(-1);
static TEST_STATS: AtomicI64 = AtomicI64::new(-1);

fn cap_bytes() -> usize {
    let t = TEST_CAP_BYTES.load(Ordering::Relaxed);
    if t >= 0 {
        return t as usize;
    }
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        cap_mb_from_value(std::env::var(ENV).ok().as_deref()).saturating_mul(1024 * 1024)
    })
}

fn stats_on() -> bool {
    let t = TEST_STATS.load(Ordering::Relaxed);
    if t >= 0 {
        return t != 0;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var(STATS_ENV).ok().as_deref().map(str::trim) {
        Some(v) => !(v.is_empty() || v == "0" || v == "off" || v == "false" || v == "no"),
        None => false,
    })
}

/// キャッシュは有効か (予算 > 0)。off では呼び出し側が分岐 1 回で戻る。
pub(crate) fn enabled() -> bool {
    cap_bytes() > 0
}

#[cfg(test)]
pub(crate) fn set_cap_for_tests(bytes: usize) {
    TEST_CAP_BYTES.store(bytes as i64, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn clear_cap_for_tests() {
    TEST_CAP_BYTES.store(-1, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn set_stats_for_tests(on: bool) {
    TEST_STATS.store(i64::from(on), Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn clear_stats_for_tests() {
    TEST_STATS.store(-1, Ordering::Relaxed);
}

/// 1 expert ぶんの展開済み行列 (gate / up / down)。
pub(crate) struct Dq {
    pub gate: Vec<f32>, // ff 行 × e 列
    pub up: Vec<f32>,   // ff 行 × e 列
    pub down: Vec<f32>, // e 行 × ff 列
}

/// 1 expert ぶんの f32 バイト数。
pub(crate) fn entry_bytes(e: usize, ff: usize) -> usize {
    3 * e * ff * 4
}

#[derive(Default)]
struct Layer {
    map: HashMap<u16, Arc<Dq>>,
    lru: VecDeque<u16>,
    used: usize,
}

impl Layer {
    fn touch(&mut self, ei: u16) {
        if let Some(p) = self.lru.iter().position(|&x| x == ei) {
            self.lru.remove(p);
        }
        self.lru.push_back(ei);
    }

    /// 層の上限に収まるまで古い順に落とす。落とした個数を返す。
    fn evict_until(&mut self, quota: usize, need: usize) -> u64 {
        let mut n = 0;
        while self.used + need > quota {
            let Some(old) = self.lru.pop_front() else {
                break;
            };
            if let Some(d) = self.map.remove(&old) {
                self.used = self.used.saturating_sub(entry_bytes_of(&d));
                n += 1;
            }
        }
        n
    }
}

fn entry_bytes_of(d: &Dq) -> usize {
    (d.gate.len() + d.up.len() + d.down.len()) * 4
}

#[derive(Default)]
struct Counters {
    /// `note_selection` で数えた選択の延べ数
    selections: u64,
    /// 層ごとの distinct expert
    distinct: HashMap<u16, HashSet<u16>>,
    /// `layer_entries` が呼ばれた回数 (= 1 チャンク 1 層 1 回)
    layer_calls: u64,
    hits: u64,
    misses: u64,
    /// 展開した expert 数とバイト数
    fills: u64,
    fill_bytes: u64,
    evictions: u64,
}

static STATS: Mutex<Option<Counters>> = Mutex::new(None);

struct Cache {
    quota: usize,
    layers: HashMap<u16, Layer>,
}

impl Cache {
    fn layer_mut(&mut self, layer: u16) -> &mut Layer {
        self.layers.entry(layer).or_default()
    }
    fn bytes(&self) -> usize {
        self.layers.values().map(|l| l.used).sum()
    }
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
static FINISHED: AtomicBool = AtomicBool::new(true);

/// 直近の [`finish_prefill`] の 1 行 (テストが配線を確かめるため)。
#[cfg(test)]
static LAST_LINE: Mutex<Option<String>> = Mutex::new(None);

#[cfg(test)]
pub(crate) fn take_line_for_tests() -> Option<String> {
    LAST_LINE.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// prefill の入口で呼ぶ。予算と層数から層ごとの上限を決め、古い内容を捨てる。
///
/// 戻り値のガードを prefill の間だけ保持する (Drop で終端処理)。
pub(crate) fn begin_prefill(n_tokens: usize, n_layer: usize) -> Guard {
    if !enabled() && !stats_on() {
        return Guard { active: false };
    }
    let cap = cap_bytes();
    let nl = n_layer.max(1);
    let quota = cap / nl;
    FINISHED.store(false, Ordering::Relaxed);
    if enabled() {
        let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        *g = Some(Cache {
            quota,
            layers: HashMap::new(),
        });
    }
    if stats_on() {
        let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
        *s = Some(Counters::default());
        if let Some(c) = s.as_mut() {
            c.distinct.reserve(nl);
        }
        eprintln!(
            "[pf-dq-cache] begin tok={n_tokens} layers={nl} cap={} MiB quota={} MiB",
            cap / (1024 * 1024),
            quota / (1024 * 1024),
        );
    }
    Guard { active: true }
}

/// prefill の終端処理 (Drop)。統計行を出して展開済みの f32 を全部捨てる。
pub(crate) struct Guard {
    active: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(l) = finish_prefill() {
            eprintln!("{l}");
        }
    }
}

/// 予算と統計をまとめて捨てる。統計行を返す (stats off なら None)。
///
/// 2 回目以降は None (ガードの Drop と明示呼び出しが二重に出さない)。
pub(crate) fn finish_prefill() -> Option<String> {
    if FINISHED.swap(true, Ordering::Relaxed) {
        return None;
    }
    let used = {
        let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let used = g.as_ref().map(|c| c.bytes()).unwrap_or(0);
        *g = None; // f32 を即座に OS へ返す
        used
    };
    if !stats_on() {
        return None;
    }
    let s = {
        let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
        s.take().unwrap_or_default()
    };
    let distinct: usize = s.distinct.values().map(|h| h.len()).sum();
    let reuse = if distinct > 0 {
        s.selections as f64 / distinct as f64
    } else {
        0.0
    };
    Some(format!(
        "[pf-dq-cache] sel={} distinct={} reuse={reuse:.2} layer_calls={} hits={} miss={} fills={} fill={} MiB evict={} held={} MiB",
        s.selections,
        distinct,
        s.layer_calls,
        s.hits,
        s.misses,
        s.fills,
        s.fill_bytes / (1024 * 1024),
        s.evictions,
        used / (1024 * 1024),
    ))
    .map(|l| {
        #[cfg(test)]
        {
            *LAST_LINE.lock().unwrap_or_else(|e| e.into_inner()) = Some(l.clone());
        }
        l
    })
}

/// 1 層ぶんの選択を数える (stats off なら atomic 1 回)。
pub(crate) fn note_selection(layer: u16, eid: &[usize]) {
    if !stats_on() {
        return;
    }
    let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(c) = s.as_mut() else {
        return;
    };
    c.selections += eid.len() as u64;
    let set = c.distinct.entry(layer).or_default();
    for &ei in eid {
        set.insert(ei as u16);
    }
}

/// 層ぶんのキャッシュ引き当て。`eid` と同じ並びで、キャッシュできた expert だけ
/// `Some(Arc<Dq>)` を返す。
///
/// ミスはこの場で展開して入れる (層の入口で 1 回・直列)。展開できない形
/// (`row_bytes` が拒む型) や予算に 1 個も入らない層では `None` を返し、
/// 呼び出し側は従来の融合カーネル経路のままになる。
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_entries(
    layer: u16,
    e: usize,
    ff: usize,
    eid: &[usize],
    gs: &[&[u8]],
    us: &[&[u8]],
    ds: &[&[u8]],
    tys: [GgmlType; 3],
) -> Vec<Option<Arc<Dq>>> {
    let mut out: Vec<Option<Arc<Dq>>> = vec![None; eid.len()];
    if eid.is_empty() || !enabled() {
        return out;
    }
    let entry = entry_bytes(e, ff);
    let (mut hits, mut misses, mut fills, mut evictions) = (0u64, 0u64, 0u64, 0u64);
    {
        let mut g = CACHE.lock().unwrap_or_else(|err| err.into_inner());
        let Some(cache) = g.as_mut() else {
            return out;
        };
        let quota = cache.quota;
        if quota < entry {
            return out;
        }
        // 行長は型と n_in だけで決まる。全 expert 共通なので先に 1 回だけ確かめる。
        let (Ok(rb_g), Ok(rb_u), Ok(rb_d)) = (
            moe_expert::row_bytes(tys[0], e),
            moe_expert::row_bytes(tys[1], e),
            moe_expert::row_bytes(tys[2], ff),
        ) else {
            return out;
        };
        for (i, &ei) in eid.iter().enumerate() {
            let key = ei as u16;
            if let Some(v) = cache.layer_mut(layer).map.get(&key).cloned() {
                cache.layer_mut(layer).touch(key);
                hits += 1;
                out[i] = Some(v);
                continue;
            }
            misses += 1;
            let lay = cache.layer_mut(layer);
            evictions += lay.evict_until(quota, entry);
            if lay.used + entry > quota {
                continue; // 入らない (この層ではキャッシュしない)
            }
            let Some(d) = dequant_expert(e, ff, gs[i], us[i], ds[i], tys, (rb_g, rb_u, rb_d)) else {
                continue;
            };
            let arc = Arc::new(d);
            let lay = cache.layer_mut(layer);
            lay.used += entry;
            lay.map.insert(key, Arc::clone(&arc));
            lay.touch(key);
            fills += 1;
            out[i] = Some(arc);
        }
    }
    if stats_on() {
        let mut s = STATS.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(c) = s.as_mut() {
            c.layer_calls += 1;
            c.hits += hits;
            c.misses += misses;
            c.fills += fills;
            c.fill_bytes += fills * entry as u64;
            c.evictions += evictions;
        }
    }
    out
}

/// 1 expert の 3 行列を f32 へ展開する。行の展開は [`dequant_row_into`] そのもの。
fn dequant_expert(
    e: usize,
    ff: usize,
    g: &[u8],
    u: &[u8],
    d: &[u8],
    tys: [GgmlType; 3],
    rb: (usize, usize, usize),
) -> Option<Dq> {
    let (rb_g, rb_u, rb_d) = rb;
    if g.len() < ff * rb_g || u.len() < ff * rb_u || d.len() < e * rb_d {
        return None;
    }
    let mut gate = vec![0.0f32; ff * e];
    for r in 0..ff {
        dequant_row_into(
            tys[0],
            &g[r * rb_g..(r + 1) * rb_g],
            &mut gate[r * e..(r + 1) * e],
        );
    }
    let mut up = vec![0.0f32; ff * e];
    for r in 0..ff {
        dequant_row_into(
            tys[1],
            &u[r * rb_u..(r + 1) * rb_u],
            &mut up[r * e..(r + 1) * e],
        );
    }
    let mut down = vec![0.0f32; e * ff];
    for r in 0..e {
        dequant_row_into(
            tys[2],
            &d[r * rb_d..(r + 1) * rb_d],
            &mut down[r * ff..(r + 1) * ff],
        );
    }
    Some(Dq { gate, up, down })
}

/// キャッシュに入っているバイト数の合計 (テスト用)。
#[cfg(test)]
pub(crate) fn held_bytes() -> usize {
    let g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map(|c| c.bytes()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// このモジュールのテストはプロセス共有の CACHE / STATS を触るので直列化する。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 決定的な Q4_K の行列 (n_out 行 × n_in 列、1 ブロック 144 B、n_in は 256 の倍数)。
    fn q4k_mat(n_in: usize, n_out: usize, seed: u32) -> Vec<u8> {
        let rb = n_in / 256 * 144;
        let mut b = vec![0u8; rb * n_out];
        for r in 0..n_out {
            for blk in 0..rb / 144 {
                let o = r * rb + blk * 144;
                b[o..o + 2].copy_from_slice(&0x2510u16.to_le_bytes()); // d ≈ 0.02
                b[o + 2..o + 4].copy_from_slice(&0x1531u16.to_le_bytes()); // dmin ≈ 0.001
                // scales (12 B): sc / m を 0 でない値にする (0 だと重みが全部 0 になる)
                b[o + 4..o + 16].fill(0x22);
                for (j, v) in b[o + 16..o + 144].iter_mut().enumerate() {
                    *v = ((j as u32).wrapping_mul(31).wrapping_add(seed + r as u32) % 251) as u8;
                }
            }
        }
        b
    }

    /// 2 expert ぶん (gate / up は ff 行 × e 列、down は e 行 × ff 列)。
    fn expert_bytes(e: usize, ff: usize, seed: u32) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let g = vec![q4k_mat(e, ff, seed), q4k_mat(e, ff, seed + 1)];
        let u = vec![q4k_mat(e, ff, seed + 2), q4k_mat(e, ff, seed + 3)];
        let d = vec![q4k_mat(ff, e, seed + 4), q4k_mat(ff, e, seed + 5)];
        (g, u, d)
    }

    /// 予算 0 は off (env の解釈)。
    #[test]
    fn cap_mb_parse() {
        let _l = lock();
        assert_eq!(cap_mb_from_value(None), 0);
        assert_eq!(cap_mb_from_value(Some("0")), 0);
        assert_eq!(cap_mb_from_value(Some("512")), 512);
        assert_eq!(cap_mb_from_value(Some(" 64 ")), 64);
        assert_eq!(cap_mb_from_value(Some("x")), 0);
    }

    /// 引き当て → 2 回目はヒット → 展開は 1 回だけ。値は dequant の出力と一致。
    #[test]
    fn fill_then_hit_reuses_dequant() {
        let _l = lock();
        let (e, ff) = (256, 256);
        let (g, u, d) = expert_bytes(e, ff, 7);
        let gs: Vec<&[u8]> = vec![&g[0], &g[1]];
        let us: Vec<&[u8]> = vec![&u[0], &u[1]];
        let ds: Vec<&[u8]> = vec![&d[0], &d[1]];
        set_stats_for_tests(true);
        set_cap_for_tests(8 * entry_bytes(e, ff));
        let pf = begin_prefill(32, 2);
        let first = layer_entries(3, e, ff, &[0, 1], &gs, &us, &ds, [GgmlType::Q4K; 3]);
        assert!(first.iter().all(|x| x.is_some()), "2 個とも入る");
        // gate の 1 行は dequant_row_into と同じ値
        let mut want = vec![0.0f32; e];
        let rb = moe_expert::row_bytes(GgmlType::Q4K, e).unwrap();
        dequant_row_into(GgmlType::Q4K, &u[1][..rb], &mut want);
        assert_eq!(&first[1].as_ref().unwrap().up[..e], &want[..]);
        let second = layer_entries(3, e, ff, &[0, 1], &gs, &us, &ds, [GgmlType::Q4K; 3]);
        assert!(second.iter().all(|x| x.is_some()));
        assert!(Arc::ptr_eq(
            first[0].as_ref().unwrap(),
            second[0].as_ref().unwrap()
        ));
        let line = finish_prefill().expect("stats on");
        assert!(line.contains("hits=2"), "{line}");
        assert!(line.contains("fills=2"), "{line}");
        drop(pf);
        clear_cap_for_tests();
        clear_stats_for_tests();
    }

    /// 予算が 1 個ぶんに満たない層では何も返さない (展開だけして捨てない)。
    #[test]
    fn small_quota_yields_nothing() {
        let _l = lock();
        let (e, ff) = (256, 256);
        let (g, u, d) = expert_bytes(e, ff, 11);
        let gs: Vec<&[u8]> = vec![&g[0]];
        let us: Vec<&[u8]> = vec![&u[0]];
        let ds: Vec<&[u8]> = vec![&d[0]];
        set_cap_for_tests(entry_bytes(e, ff) * 4 / 3); // 48 層で 1 個に満たない
        let pf = begin_prefill(8, 48);
        let got = layer_entries(0, e, ff, &[0], &gs, &us, &ds, [GgmlType::Q4K; 3]);
        assert!(got[0].is_none(), "quota < entry では展開しない");
        assert_eq!(held_bytes(), 0);
        finish_prefill();
        drop(pf);
        clear_cap_for_tests();
    }

    /// 予算を超えたら層内 LRU で落ちる。合計は層の上限を超えない。
    #[test]
    fn lru_evicts_within_layer_quota() {
        let _l = lock();
        let (e, ff) = (256, 256);
        let (g, u, d) = expert_bytes(e, ff, 21);
        let gs: Vec<&[u8]> = vec![&g[0], &g[1]];
        let us: Vec<&[u8]> = vec![&u[0], &u[1]];
        let ds: Vec<&[u8]> = vec![&d[0], &d[1]];
        set_stats_for_tests(true);
        set_cap_for_tests(entry_bytes(e, ff) * 2); // 2 層で 1 層あたり 1 個
        let pf = begin_prefill(8, 2);
        let a = layer_entries(0, e, ff, &[0], &gs[..1], &us[..1], &ds[..1], [GgmlType::Q4K; 3]);
        assert!(a[0].is_some());
        let b = layer_entries(0, e, ff, &[1], &gs[1..], &us[1..], &ds[1..], [GgmlType::Q4K; 3]);
        assert!(b[0].is_some());
        assert_eq!(held_bytes(), entry_bytes(e, ff), "層の上限は 1 個");
        let line = finish_prefill().expect("stats on");
        assert!(line.contains("evict=1"), "{line}");
        drop(pf);
        clear_cap_for_tests();
        clear_stats_for_tests();
    }

    /// 統計だけ (キャッシュ off) でも重複度が数えられる。
    #[test]
    fn stats_count_selection_reuse() {
        let _l = lock();
        set_stats_for_tests(true);
        let pf = begin_prefill(16, 4);
        note_selection(0, &[0, 1, 2]);
        note_selection(0, &[1, 2, 3]);
        let line = finish_prefill().expect("stats on");
        assert!(line.contains("sel=6"), "{line}");
        assert!(line.contains("distinct=4"), "{line}");
        assert!(line.contains("reuse=1.50"), "{line}");
        drop(pf);
        clear_stats_for_tests();
    }

    /// キャッシュ経路の GEMM が、同じ展開値を使う素の f32 経路と一致すること。
    ///
    /// 融合カーネル (量子ブロックから直接 FMA) との差は累積順だけなので、
    /// 「展開済み f32 を 4 本の部分和で回す」こと自体が式として正しいかをここで見る。
    #[test]
    fn gemm_rows_f32_matches_scalar_on_same_rows() {
        let _l = lock();
        let (n_in, n_out, m) = (256, 64, 3);
        let raw = q4k_mat(n_in, n_out, 5);
        let rb = moe_expert::row_bytes(GgmlType::Q4K, n_in).unwrap();
        let mut rows = vec![0.0f32; n_out * n_in];
        for r in 0..n_out {
            dequant_row_into(
                GgmlType::Q4K,
                &raw[r * rb..(r + 1) * rb],
                &mut rows[r * n_in..(r + 1) * n_in],
            );
        }
        let xs: Vec<f32> = (0..m * n_in)
            .map(|i| ((i as u32).wrapping_mul(2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut got = vec![0.0f32; m * n_out];
        let mut want = vec![0.0f32; m * n_out];
        moe_expert::gemm_rows_f32(&rows, n_in, 0, n_out, &xs, m, got.as_mut_ptr(), n_out);
        // 素の f32 (F32 型 = dequant はコピー) を既存のバッチ入口で回す。
        moe_expert::gemm_rows(
            GgmlType::F32,
            n_in * 4,
            unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) },
            n_in,
            0,
            n_out,
            &xs,
            m,
            want.as_mut_ptr(),
            n_out,
            true,
        );
        let mut max_d = 0.0f32;
        let mut max_a = 0.0f32;
        for (a, b) in want.iter().zip(&got) {
            max_d = max_d.max((a - b).abs());
            max_a = max_a.max(a.abs());
        }
        assert!(max_a > 0.0, "参照が全部 0");
        assert!(max_d / max_a < 1e-5, "max|Δ|/max|ref| = {}", max_d / max_a);
    }
}
