//! task 278: 層一括 expert 経路の区間計測 (`DENNO_SEG_PROF=1`、**既定 off**)。
//!
//! # 何のために
//!
//! lab t268b の `[q4ledger]` は K 型 (UD-Q4_K_XL) の decode で 1 トークンあたり
//! `launches=143 syncs=96 h2d=235.7MB/554回 d2h=7.3MB/143回`、`loop` 行の
//! `sync=23.2ms` を出した。だが**どの区間が何 ms か**は行の粒度が粗く読めない
//! (host の `cpu` 列にミスの読みと表の構築が混ざる)。ここでは 1 層を
//! ・VRAM 参照 / 表の構築 / prologue の H2D / 起動 / 待ち / D2H / ミスの CPU 計算
//! に分け、カーネルの GPU 時間は**イベントで**測って 1 行にまとめる。
//!
//! # 使い方
//!
//! `DENNO_SEG_PROF=1` で serve を起動し、リクエストを 1 本流す。トークンの
//! 先頭 (層 0 の入口) で前のトークンぶんを 1 行出して 0 に戻す。
//!
//! ```text
//! [seg-prof] tok=128 layers=48 host_us/layer: lookup=.. table=.. h2d=../..calls ...
//!            gpu_us/layer: gate=.. up=.. silu=.. down=.. sum=.. | gbs: gate=.. up=.. down=..
//! ```
//!
//! 数値経路には触れない (カウンタは `AtomicU64`、off のときは `stamp()` が
//! `None` を返すだけで分岐 1 回)。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// 計測を有効にする env。
pub const ENV: &str = "DENNO_SEG_PROF";

/// ホスト区間: VRAM の LRU 参照 ([`crate::model::expert_vram::lookup_many`])。
pub const PH_LOOKUP: usize = 0;
/// ホスト区間: segment 表 / ポインタ表 / prologue バッファの構築。
pub const PH_TABLE: usize = 1;
/// ホスト区間: prologue の H2D (1 回)。
pub const PH_H2D: usize = 2;
/// ホスト区間: カーネルの発行。
pub const PH_LAUNCH: usize = 3;
/// ホスト区間: 待ち (イベント同期 / 文脈同期)。
pub const PH_SYNC: usize = 4;
/// ホスト区間: 結果の D2H。
pub const PH_D2H: usize = 5;
/// ホスト区間: `silu(g) * u` (旧経路はホストで回す)。
pub const PH_SILU: usize = 6;
/// ホスト区間: 重み付きの足し込み。
pub const PH_SUM: usize = 7;
/// ホスト区間: ミス expert の CPU 計算。
pub const PH_MISS: usize = 8;
/// ホスト区間: 作業域の確保。
pub const PH_ALLOC: usize = 9;
/// ホスト区間: VRAM への積み込み (旧経路の `acquire`)。
pub const PH_ACQUIRE: usize = 10;
/// ホスト区間の数。
pub const NPH: usize = 11;

/// [`NPH`] 本の名前 (レポート用)。
pub const PH_NAMES: [&str; NPH] = [
    "lookup", "table", "h2d", "launch", "sync", "d2h", "silu", "sum", "miss", "alloc", "acquire",
];

/// GPU 区間: gate の GEMV。
pub const GP_GATE: usize = 0;
/// GPU 区間: up の GEMV。
pub const GP_UP: usize = 1;
/// GPU 区間: `silu(g) * u` のカーネル。
pub const GP_SILU: usize = 2;
/// GPU 区間: down の GEMV。
pub const GP_DOWN: usize = 3;
/// GPU 区間: 重み付き和のカーネル。
pub const GP_SUM: usize = 4;
/// GPU 区間の数。
pub const NGPU: usize = 5;

/// [`NGPU`] 本の名前。
pub const GP_NAMES: [&str; NGPU] = ["gate", "up", "silu", "down", "sum"];

static PH_NS: [AtomicU64; NPH] = [const { AtomicU64::new(0) }; NPH];
static PH_CALLS: [AtomicU64; NPH] = [const { AtomicU64::new(0) }; NPH];
static GP_NS: [AtomicU64; NGPU] = [const { AtomicU64::new(0) }; NGPU];
static GP_BYTES: [AtomicU64; NGPU] = [const { AtomicU64::new(0) }; NGPU];
static LAYERS: AtomicU64 = AtomicU64::new(0);
static TOKENS: AtomicU64 = AtomicU64::new(0);
static ON: AtomicBool = AtomicBool::new(false);
static ENV_READ: AtomicBool = AtomicBool::new(false);

/// テスト用の強制 (env はプロセスで 1 回しか読めない)。
pub fn set_for_test(v: bool) {
    ON.store(v, Ordering::Relaxed);
    ENV_READ.store(true, Ordering::Relaxed);
}

/// 計測が要るか (初回だけ env を読む)。
pub fn enabled() -> bool {
    if !ENV_READ.load(Ordering::Relaxed) {
        let on = matches!(
            std::env::var(ENV).ok().as_deref().map(str::trim),
            Some("1") | Some("on") | Some("true")
        );
        ON.store(on, Ordering::Relaxed);
        ENV_READ.store(true, Ordering::Relaxed);
    }
    ON.load(Ordering::Relaxed)
}

/// 計測が要るときだけ `Instant` を取る。
#[inline]
pub fn stamp() -> Option<Instant> {
    if enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

/// ホスト区間 1 回ぶんを積む (`t` は区間の入口)。
#[inline]
pub fn add(ph: usize, t: Option<Instant>) {
    let Some(t) = t else { return };
    let ns = t.elapsed().as_nanos() as u64;
    PH_NS[ph.min(NPH - 1)].fetch_add(ns, Ordering::Relaxed);
    PH_CALLS[ph.min(NPH - 1)].fetch_add(1, Ordering::Relaxed);
}

/// 区間の入口を取らずに ns を直接積む (cuda 側で測った値を渡す)。
#[inline]
pub fn add_ns(ph: usize, ns: u64) {
    if !enabled() {
        return;
    }
    PH_NS[ph.min(NPH - 1)].fetch_add(ns, Ordering::Relaxed);
    PH_CALLS[ph.min(NPH - 1)].fetch_add(1, Ordering::Relaxed);
}

/// 呼び出し回数だけ積む (0 ns の区間)。
#[inline]
pub fn bump(ph: usize) {
    if !enabled() {
        return;
    }
    PH_CALLS[ph.min(NPH - 1)].fetch_add(1, Ordering::Relaxed);
}

/// GPU 区間 (ns) と読んだバイト数を積む。
#[inline]
pub fn add_gpu(g: usize, ns: u64, bytes: usize) {
    if !enabled() {
        return;
    }
    let g = g.min(NGPU - 1);
    GP_NS[g].fetch_add(ns, Ordering::Relaxed);
    GP_BYTES[g].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// 1 層ぶんを数える。
#[inline]
pub fn note_layer() {
    if !enabled() {
        return;
    }
    LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// 積算を 0 に戻す。
pub fn reset() {
    for c in &PH_NS {
        c.store(0, Ordering::Relaxed);
    }
    for c in &PH_CALLS {
        c.store(0, Ordering::Relaxed);
    }
    for c in &GP_NS {
        c.store(0, Ordering::Relaxed);
    }
    for c in &GP_BYTES {
        c.store(0, Ordering::Relaxed);
    }
    LAYERS.store(0, Ordering::Relaxed);
}

/// 積算の読み ([`NPH`], [`NGPU`] の生値)。
#[derive(Clone, Copy, Default, Debug)]
pub struct Snap {
    pub ph_ns: [u64; NPH],
    pub ph_calls: [u64; NPH],
    pub gp_ns: [u64; NGPU],
    pub gp_bytes: [u64; NGPU],
    pub layers: u64,
}

/// いまの積算を読む (テストと [`flush`] 用)。
pub fn snap() -> Snap {
    let mut s = Snap::default();
    for i in 0..NPH {
        s.ph_ns[i] = PH_NS[i].load(Ordering::Relaxed);
        s.ph_calls[i] = PH_CALLS[i].load(Ordering::Relaxed);
    }
    for i in 0..NGPU {
        s.gp_ns[i] = GP_NS[i].load(Ordering::Relaxed);
        s.gp_bytes[i] = GP_BYTES[i].load(Ordering::Relaxed);
    }
    s.layers = LAYERS.load(Ordering::Relaxed);
    s
}

/// [`Snap`] を 1 行にする (**純関数**。テストが突き合わせる)。
///
/// `tok` は区間の対象トークン数 (通常は積んだ層数 / 層数)。
pub fn format_line(s: &Snap, tok: u64) -> String {
    let tok = tok.max(1);
    let ly = s.layers.max(1);
    let us = |ns: u64| ns as f64 / 1e3 / ly as f64; // 1 層あたり
    let mut host = String::new();
    for i in 0..NPH {
        if s.ph_calls[i] == 0 {
            continue;
        }
        if host.is_empty() {
            host.push_str("host_us/layer:");
        }
        host.push_str(&format!(
            " {}={:.1}/{}",
            PH_NAMES[i],
            us(s.ph_ns[i]),
            s.ph_calls[i] / ly.max(1)
        ));
    }
    let mut gpu = String::new();
    for i in 0..NGPU {
        if s.gp_ns[i] == 0 && s.gp_bytes[i] == 0 {
            continue;
        }
        if gpu.is_empty() {
            gpu.push_str("gpu_us/layer:");
        }
        gpu.push_str(&format!(" {}={:.1}", GP_NAMES[i], us(s.gp_ns[i])));
    }
    let mut gbs = String::new();
    for i in 0..NGPU {
        if s.gp_bytes[i] == 0 {
            continue;
        }
        // 1 層あたりのバイト数を、その層あたりの GPU 時間で割る。
        let ns = s.gp_ns[i] as f64;
        let bytes = s.gp_bytes[i] as f64;
        let g = if ns > 0.0 {
            bytes / (ns / 1e9) / 1e9
        } else {
            0.0
        };
        if gbs.is_empty() {
            gbs.push_str("gbs:");
        }
        gbs.push_str(&format!(" {}={g:.1}", GP_NAMES[i]));
    }
    format!(
        "[seg-prof] tok={tok} layers={} {host} | {gpu} | {gbs}",
        s.layers
    )
}

/// 溜まった積算を 1 行で出して 0 に戻す (何も積んでいなければ `None`)。
pub fn flush() -> Option<String> {
    if !enabled() {
        return None;
    }
    let s = snap();
    if s.layers == 0 {
        return None;
    }
    reset();
    let tok = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    Some(format_line(&s, tok))
}

// ---------------------------------------------------------------------------
// task 498: seg 層の和 (e*4 B) の D2H を小さな pinned バウンス経由にする
// ---------------------------------------------------------------------------
//
// 動機 (lab t498b の `[seg-prof]`、UD-Q4_K_XL、decode 定常):
// `d2h=13.3-14.8 us/層` が 1 層 1 本 (= e*4 = 10 KB) の **pageable な同期 D2H** で、
// 48 層/token で 0.64-0.71 ms/token = decode 13.4 ms の 5%。pageable の D2H は
// driver が内部ステージングを取ってから写すため 0.7 GB/s しか出ない。
// 転送の実体は cuda 側 (`cuda::seg_pin_copy_d2h`) に置き、**ゲートとカウンタはここ**
// に置く (CUDA 無しのビルドでも起動ログとリクエスト終端の行が出せる)。

/// ゲート env。**既定 off** (明示的な真値だけ on)。
pub const SEG_PIN_ENV: &str = "DENNO_SEG_SUM_PIN";

/// [`SEG_PIN_ENV`] の解釈 (純関数)。
pub fn seg_sum_pin_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// seg 和の D2H を pinned バウンス経由にするか (プロセスで 1 回だけ env を読む)。
pub fn seg_sum_pin_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| seg_sum_pin_from_value(std::env::var(SEG_PIN_ENV).ok().as_deref()))
}

/// 発火した回数 (pinned 経由で写せた) と、従来の pageable へ落ちた回数。
static SEG_PIN_CALLS: AtomicU64 = AtomicU64::new(0);
static SEG_PIN_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// 発火を 1 回積む (cuda 側から呼ぶ)。
pub fn note_seg_pin_call() {
    SEG_PIN_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// 従来経路へ落ちたことを 1 回積む (cuda 側から呼ぶ)。
pub fn note_seg_pin_fallback() {
    SEG_PIN_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// (発火, 落ちた) の回数。
pub fn seg_pin_counts() -> (u64, u64) {
    (
        SEG_PIN_CALLS.load(Ordering::Relaxed),
        SEG_PIN_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// テスト用: 積算を 0 に戻す。
#[cfg(test)]
pub fn seg_pin_reset() {
    SEG_PIN_CALLS.store(0, Ordering::Relaxed);
    SEG_PIN_FALLBACKS.store(0, Ordering::Relaxed);
}

/// 起動ログ用: ゲートの状態 (COND の env が届いたかを確かめる)。
pub fn seg_sum_pin_boot_line() -> String {
    format!(
        "[flash] seg sum pin: {} ({SEG_PIN_ENV})",
        if seg_sum_pin_on() { "on" } else { "off" }
    )
}

/// リクエスト終端の診断行 (ゲート on のときだけ)。`calls` が 0 なら**分岐が
/// 発火していない**ので、そのときは採否の判断をしない (fail closed の証拠)。
pub fn seg_sum_pin_report() -> Option<String> {
    if !seg_sum_pin_on() {
        return None;
    }
    let (calls, fallbacks) = seg_pin_counts();
    Some(format!(
        "[seg-pin] calls={calls} fallbacks={fallbacks} ({SEG_PIN_ENV})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// グローバルの積算を使うのでテストは直列化する。
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// task 498: `DENNO_SEG_SUM_PIN` の真理値表。**既定は off**。
    #[test]
    fn t498_seg_sum_pin_truth_table() {
        assert!(!seg_sum_pin_from_value(None), "未設定は既定 off");
        assert!(!seg_sum_pin_from_value(Some("")));
        assert!(!seg_sum_pin_from_value(Some("0")));
        assert!(!seg_sum_pin_from_value(Some("off")));
        assert!(!seg_sum_pin_from_value(Some(" no ")));
        assert!(!seg_sum_pin_from_value(Some("FALSE")));
        assert!(seg_sum_pin_from_value(Some("1")));
        assert!(seg_sum_pin_from_value(Some("on")));
        assert!(seg_sum_pin_from_value(Some(" true ")));
        assert!(seg_sum_pin_from_value(Some("YES")));
    }

    /// task 498: 発火 / 落ちた回数が別々に積まれる。
    #[test]
    fn t498_seg_pin_counts_accumulate() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        seg_pin_reset();
        note_seg_pin_call();
        note_seg_pin_call();
        note_seg_pin_fallback();
        assert_eq!(seg_pin_counts(), (2, 1));
        seg_pin_reset();
        assert_eq!(seg_pin_counts(), (0, 0));
    }

    /// off のときは何も積まない (既定の経路にカウンタが乗らない)。
    #[test]
    fn off_records_nothing() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_for_test(false);
        reset();
        // 呼び出し側は `stamp()` が None なら積まない (これが off の保証)。
        add(PH_SYNC, stamp());
        add_gpu(GP_GATE, 1_000, 1_000);
        note_layer();
        assert_eq!(snap().layers, 0);
        assert_eq!(snap().ph_calls[PH_SYNC], 0);
        assert!(flush().is_none());
    }

    /// on のときは区間ごとに積み、行に ms ではなく us/層で出る。
    #[test]
    fn on_accumulates_and_formats() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_for_test(true);
        reset();
        // 2 層ぶん: sync 100 us/層、gate 50 us/層 (10 MB/層 -> 200 GB/s)
        for _ in 0..2 {
            add_ns(PH_SYNC, 100_000);
            add_gpu(GP_GATE, 50_000, 10_000_000);
            note_layer();
        }
        let s = snap();
        assert_eq!(s.layers, 2);
        assert_eq!(s.ph_ns[PH_SYNC], 200_000);
        assert_eq!(s.ph_calls[PH_SYNC], 2);
        let line = format_line(&s, 1);
        assert!(line.contains("sync=100.0"), "行: {line}");
        assert!(line.contains("gate=50.0"), "行: {line}");
        assert!(line.contains("gate=200.0"), "GB/s が 200 でない: {line}");
        let flushed = flush().expect("行");
        assert!(flushed.starts_with("[seg-prof] tok=1 layers=2"));
        assert_eq!(snap().layers, 0, "flush で 0 に戻る");
        set_for_test(false);
    }
}
