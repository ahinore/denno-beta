//! task 149: デバイス確保の**タグ付き台帳**と、カード実容量からの expert 予算。
//!
//! # なぜ要るか
//!
//! 実測 (lab t147a, swift15-abl IQ3_S) では、expert を VRAM に載せない条件でも
//! プロセスの VRAM が約 7.2 GB あった。`[flash] vram budget` の 1 行は
//! `weights=445 MiB` としか言っておらず、残りの数 GB が何なのかを誰も説明できて
//! いなかった。ここでは `DevBuffer` の確保・解放を全部 1 本の台帳に通し、
//! 「どのタグが何本・何 MiB 持っているか」を 1 表で出す。
//!
//! # 契約
//!
//! - [`note_alloc`] / [`note_free`] は**必ず対**で呼ぶ (`DevBuffer` の Drop から)。
//! - 合計 (`LIVE_BYTES`) と expert 分 (`LIVE_EXPERTS`) は**常に**数える。
//!   予算 (expert が他に譲る量) は計測 OFF でも要るため。
//! - タグ別の内訳と表の印刷は `DENNO_Q4_PROFILE` があるときだけ
//!   (計測の副作用を本番経路に足さない)。
//! - 数値・カーネル引数・発行順には一切触らない (数えるだけ)。
//!
//! # 表の読み方
//!
//! `residual = デバイスの使用量 - 台帳の合計` で、これは
//! CUDA context / JIT されたカーネルのコード / 断片化 の合計にあたる
//! (ドライバはコードの実バイト数を公開しないので、ここでは差分として出す)。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// デバイス常駐の dense 重み (gguf レジストリ)
pub const TAG_WEIGHTS: &str = "w:gguf";
/// f32 lm_head のレジストリ (s49)
pub const TAG_LMHEAD: &str = "w:lm_head";
/// router f32 のレジストリ
pub const TAG_ROUTER: &str = "w:router";
/// hc mix の f32 展開済み重み (task 110)
pub const TAG_MIX_F32: &str = "w:mix_f32";
/// GDN の重み (層ごとに差し替える 3 本)
pub const TAG_GDN_W: &str = "w:gdn";
/// expert の LRU (decode のヒット/ミス)
pub const TAG_EXPERTS: &str = "experts";
/// prefill のパック済み expert 重み (grouped GEMM)
pub const TAG_GROUPED_PACKED: &str = "grouped packed";
/// grouped GEMM の入出力スクラッチ
pub const TAG_GROUPED: &str = "grouped scratch";
/// prefill の層一括 GEMV (segment) スクラッチ
pub const TAG_SEG: &str = "seg scratch";
/// `moe_gemv_dev` の入出力スクラッチ
pub const TAG_GEMV: &str = "gemv scratch";
/// GDN スキャンの作業領域
pub const TAG_GDN_SC: &str = "gdn scratch";
/// KV ミラー (attention の K/V 複製)
pub const TAG_KV_MIRROR: &str = "kv mirror";
/// q4 attention の K/V ミラー
pub const TAG_ATTN_MIRROR: &str = "q4attn mirror";
/// GDN 状態のデバイスミラー
pub const TAG_GDN_MIRROR: &str = "gdn mirror";
/// prefill スクラッチの arena チャンク
pub const TAG_ARENA: &str = "scratch arena";
/// デバイス常駐チェーンの入出力 (arena が使えないときの個別確保)
pub const TAG_CHAIN: &str = "chain scratch";
/// GEMM (int8 / q4) の一時バッファ
pub const TAG_GEMM: &str = "gemm scratch";
/// タグを付けていない確保 (表の `other` 行)
pub const TAG_OTHER: &str = "other";

/// 合計 (常時)
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
/// expert LRU が持っているバイト数 (常時)
static LIVE_EXPERTS: AtomicUsize = AtomicUsize::new(0);

/// タグ別の内訳 (`DENNO_Q4_PROFILE` のときだけ更新)
#[derive(Debug, Default, Clone)]
struct Ledger {
    map: HashMap<&'static str, (usize, usize)>,
}

static LEDGER: OnceLock<Mutex<Ledger>> = OnceLock::new();

/// ロード済みカーネルモジュール (名前, PTX のバイト数)。JIT 後のコード量は
/// ドライバが公開しないので、ここでは PTX の大きさだけを「推定の材料」として持つ。
static MODULES: OnceLock<Mutex<Vec<(&'static str, usize)>>> = OnceLock::new();

fn ledger() -> &'static Mutex<Ledger> {
    LEDGER.get_or_init(|| Mutex::new(Ledger::default()))
}

fn modules() -> &'static Mutex<Vec<(&'static str, usize)>> {
    MODULES.get_or_init(|| Mutex::new(Vec::new()))
}

/// `DENNO_Q4_PROFILE` があるか (値は見ない。q4ledger と同じ解釈)。
pub fn audit_on() -> bool {
    #[cfg(test)]
    if let Some(v) = tests::force_enabled() {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DENNO_Q4_PROFILE").is_some())
}

/// 1 本ぶんの確保を記帳する。
pub fn note_alloc(tag: &'static str, bytes: usize) {
    LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    if tag == TAG_EXPERTS {
        LIVE_EXPERTS.fetch_add(bytes, Ordering::Relaxed);
    }
    if !audit_on() {
        return;
    }
    if let Ok(mut g) = ledger().lock() {
        let e = g.map.entry(tag).or_insert((0, 0));
        e.0 = e.0.saturating_add(1);
        e.1 = e.1.saturating_add(bytes);
    }
}

/// 1 本ぶんの解放を記帳する (`note_alloc` と対)。
pub fn note_free(tag: &'static str, bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes.min(LIVE_BYTES.load(Ordering::Relaxed)), Ordering::Relaxed);
    if tag == TAG_EXPERTS {
        LIVE_EXPERTS.fetch_sub(
            bytes.min(LIVE_EXPERTS.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }
    if !audit_on() {
        return;
    }
    if let Ok(mut g) = ledger().lock() {
        if let Some(e) = g.map.get_mut(tag) {
            e.0 = e.0.saturating_sub(1);
            e.1 = e.1.saturating_sub(bytes);
        }
    }
}

/// 台帳の合計 (常時更新)。
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// expert LRU が持っているバイト数 (常時更新)。
pub fn experts_live_bytes() -> usize {
    LIVE_EXPERTS.load(Ordering::Relaxed)
}

/// タグ別の内訳 (バイト数の大きい順)。計測 OFF なら空。
pub fn ledger_rows() -> Vec<(&'static str, usize, usize)> {
    if !audit_on() {
        return Vec::new();
    }
    let Ok(g) = ledger().lock() else {
        return Vec::new();
    };
    let mut v: Vec<(&'static str, usize, usize)> = g
        .map
        .iter()
        .filter(|(_, (_, b))| *b > 0)
        .map(|(k, (c, b))| (*k, *c, *b))
        .collect();
    v.sort_unstable_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));
    v
}

/// モジュール 1 本を記帳する (PTX のバイト数)。
pub fn note_module(name: &'static str, ptx_bytes: usize) {
    if !audit_on() {
        return;
    }
    if let Ok(mut g) = modules().lock() {
        if !g.iter().any(|(n, _)| *n == name) {
            g.push((name, ptx_bytes));
        }
    }
}

/// モジュールの本数と PTX の合計バイト数。
pub fn module_stats() -> (usize, usize) {
    match modules().lock() {
        Ok(g) => (g.len(), g.iter().map(|(_, b)| *b).sum()),
        Err(_) => (0, 0),
    }
}

/// MiB (切り上げ。0 は 0 のまま)。
fn mib(bytes: usize) -> usize {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(1 << 20)
    }
}

/// 1 表ぶんの行を組む (純関数。GPU 無しで検証できる)。
///
/// `dev` は `(空き MiB, 総 MiB)`。`None` は「ドライバに聞けなかった」。
pub fn table_lines(
    reason: &str,
    dev: Option<(usize, usize)>,
    rows: &[(&'static str, usize, usize)],
) -> Vec<String> {
    let total: usize = rows.iter().map(|(_, _, b)| *b).sum();
    let n: usize = rows.iter().map(|(_, c, _)| *c).sum();
    let mut out = Vec::new();
    match dev {
        Some((free, tot)) => {
            let used = tot.saturating_sub(free) * (1 << 20);
            let residual = used.saturating_sub(total);
            out.push(format!(
                "[mem-audit] {reason}: dev used {} MiB / free {} MiB / total {} MiB, ledger {} MiB ({} 本), residual {} MiB (context + JIT + 断片)",
                mib(used),
                free,
                tot,
                mib(total),
                n,
                mib(residual)
            ));
        }
        None => {
            out.push(format!(
                "[mem-audit] {reason}: dev 不明, ledger {} MiB ({} 本)",
                mib(total),
                n
            ));
        }
    }
    for (tag, c, b) in rows {
        out.push(format!("[mem-audit]   {tag:<16} {c:>6} 本 {b:>8} MiB", b = mib(*b)));
    }
    out.push(format!("[mem-audit]   {:<16} {n:>6} 本 {t:>8} MiB", "sum", t = mib(total)));
    out
}

/// 表を 1 回出す (`DENNO_Q4_PROFILE` が無ければ何もしない)。
pub fn print_table(reason: &str, dev: Option<(usize, usize)>) {
    if !audit_on() {
        return;
    }
    let rows = ledger_rows();
    for line in table_lines(reason, dev, &rows) {
        tracing::info!("{line}");
    }
    let (n, bytes) = module_stats();
    tracing::info!(
        "[mem-audit]   modules {n} 本 (PTX {} MiB。JIT 後のコード量は未計測)",
        mib(bytes)
    );
}

static REQ_N: AtomicUsize = AtomicUsize::new(0);
static LONG_DONE: AtomicBool = AtomicBool::new(false);

/// このリクエストの後で表を出すか (純関数)。
///
/// 出すのは **1 本目の後** と、**長い文脈 (`ctx >= long_ctx`) の最初の 1 回**
/// だけ (`long_done` が既に真なら出さない)。毎リクエスト出すとログが埋まる。
pub fn should_print_after(n: usize, ctx: usize, long_ctx: usize, long_done: bool) -> bool {
    n == 1 || (ctx >= long_ctx && !long_done)
}

/// リクエスト完了時に呼ぶ。**1 本目の後**と、**長い文脈 (>= `long_ctx`) の
/// 最初の 1 回**だけ表を出す (毎リクエスト出すとログが埋まる)。
pub fn request_end(ctx_tokens: usize, long_ctx: usize, dev: Option<(usize, usize)>) {
    if !audit_on() {
        return;
    }
    let n = REQ_N.fetch_add(1, Ordering::Relaxed) + 1;
    let long_done = LONG_DONE.load(Ordering::Relaxed);
    if !should_print_after(n, ctx_tokens, long_ctx, long_done) {
        return;
    }
    if ctx_tokens >= long_ctx {
        LONG_DONE.store(true, Ordering::Relaxed);
    }
    print_table(
        &format!("request #{n} の後 (ctx={ctx_tokens})"),
        dev,
    );
}

// ---------------------------------------------------------------------------
// task 149: カード実容量からの expert 予算
// ---------------------------------------------------------------------------

/// `DENNO_VRAM_RESERVE_MB` の既定 (MiB)。CUDA context + JIT + 断片の余裕。
pub const VRAM_RESERVE_MB_DEFAULT: u64 = 1024;

/// `DENNO_VRAM_BUDGET_CARD` の解釈 (未設定 = `default_on`)。
pub fn parse_card_budget(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}

/// カード予算が有効か。既定は採用時に true (env で切り戻せる)。
pub fn card_budget_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_card_budget(
            std::env::var("DENNO_VRAM_BUDGET_CARD").ok().as_deref(),
            CARD_BUDGET_DEFAULT,
        )
    })
}

/// 既定値。lab の結果で採否を決めてから切り替える。
pub const CARD_BUDGET_DEFAULT: bool = true;

/// `DENNO_VRAM_RESERVE_MB` (MiB)。
pub fn vram_reserve_mb() -> u64 {
    static MB: OnceLock<u64> = OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("DENNO_VRAM_RESERVE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(VRAM_RESERVE_MB_DEFAULT)
    })
}

/// expert の実効上限 (純関数)。
///
/// `total` はカードの総バイト数、`others` は expert 以外が**いま**使っている
/// バイト数 ([`live_bytes`] - [`experts_live_bytes`])、`reserve` は余裕。
/// `env_cap` が 0 のときは「無制限」ではなくカード側の上限になる (これが
/// 「予算を本物にする」という変更の本体)。`env_cap > 0` なら小さい方を採る。
pub fn card_expert_cap(env_cap: usize, total: usize, reserve: usize, others: usize) -> usize {
    let card = total.saturating_sub(reserve).saturating_sub(others);
    if env_cap == 0 {
        card
    } else {
        env_cap.min(card)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI8;
    use std::sync::Mutex as StdMutex;

    static SERIAL: StdMutex<()> = StdMutex::new(());

    /// `audit_on()` を env を触らずに差し替えるためのフック
    /// (env を書き換えると並走テストの解釈を揺らすため)。
    static FORCE: AtomicI8 = AtomicI8::new(-1);

    pub(super) fn force_enabled() -> Option<bool> {
        match FORCE.load(Ordering::Relaxed) {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        }
    }

    fn set_force(v: Option<bool>) {
        FORCE.store(
            match v {
                Some(true) => 1,
                Some(false) => 0,
                None => -1,
            },
            Ordering::Relaxed,
        );
    }

    fn fresh() -> std::sync::MutexGuard<'static, ()> {
        let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(mut l) = ledger().lock() {
            l.map.clear();
        }
        LIVE_BYTES.store(0, Ordering::Relaxed);
        LIVE_EXPERTS.store(0, Ordering::Relaxed);
        g
    }

    #[test]
    fn alloc_free_are_symmetric() {
        let _g = fresh();
        note_alloc(TAG_WEIGHTS, 1000);
        note_alloc(TAG_EXPERTS, 4000);
        assert_eq!(live_bytes(), 5000);
        assert_eq!(experts_live_bytes(), 4000);
        note_free(TAG_WEIGHTS, 1000);
        note_free(TAG_EXPERTS, 4000);
        assert_eq!(live_bytes(), 0);
        assert_eq!(experts_live_bytes(), 0);
    }

    #[test]
    fn over_free_never_wraps() {
        let _g = fresh();
        note_alloc(TAG_KV_MIRROR, 10);
        note_free(TAG_KV_MIRROR, 99);
        assert_eq!(live_bytes(), 0, "引きすぎても 0 で止まる");
        assert_eq!(experts_live_bytes(), 0);
    }

    #[test]
    fn rows_sort_by_bytes_and_skip_zero() {
        let _g = fresh();
        set_force(Some(true));
        note_alloc(TAG_WEIGHTS, 300);
        note_alloc(TAG_ARENA, 0); // 合計 0 のタグは行に出さない
        note_alloc(TAG_MIX_F32, 500);
        let rows = ledger_rows();
        set_force(None);
        assert_eq!(rows.len(), 2, "0 バイトのタグは落とす: {rows:?}");
        assert_eq!(rows[0].0, TAG_MIX_F32, "大きい順");
        assert_eq!(rows[1].0, TAG_WEIGHTS);
        let lines = table_lines("t", Some((100, 200)), &rows);
        assert!(lines[0].contains("dev used 100 MiB"), "{lines:?}");
        // 800 B は MiB に切り上げて 1 MiB。
        assert!(lines[0].contains("ledger 1 MiB (2 本)"), "{:?}", lines[0]);
        assert!(lines[0].contains("residual"), "{:?}", lines[0]);
        assert!(lines[1].contains(TAG_MIX_F32), "{lines:?}");
        assert!(lines[2].contains(TAG_WEIGHTS), "{lines:?}");
        assert!(lines[3].contains("sum"), "{lines:?}");
        assert!(lines[3].contains("1 MiB"), "{:?}", lines[3]);
    }

    /// 計測 OFF ではタグ別の内訳を作らない (合計だけは常に数える)。
    #[test]
    fn off_keeps_totals_but_no_rows() {
        let _g = fresh();
        set_force(Some(false));
        note_alloc(TAG_WEIGHTS, 100);
        assert!(ledger_rows().is_empty());
        assert_eq!(live_bytes(), 100, "予算用の合計は常時");
        set_force(None);
    }

    /// 表を出すタイミングは「1 本目の後」と「長い文脈の最初の 1 回」だけ。
    #[test]
    fn table_is_printed_once_for_first_and_once_for_long() {
        assert!(should_print_after(1, 20, 4096, false), "1 本目は必ず");
        assert!(!should_print_after(2, 20, 4096, false), "短い 2 本目は出さない");
        assert!(should_print_after(3, 4096, 4096, false), "長い文脈の最初");
        assert!(!should_print_after(4, 8000, 4096, true), "2 回目以降は出さない");
    }

    /// 表の印刷は計測 OFF では何もしない (パニックしないことも見る)。
    #[test]
    fn print_table_is_a_no_op_when_off() {
        let _g = fresh();
        set_force(Some(false));
        print_table("off", Some((1, 2)));
        request_end(10_000, 4096, Some((1, 2)));
        set_force(None);
    }

    #[test]
    fn table_without_device_still_prints_ledger() {
        let rows = vec![(TAG_ARENA, 1usize, 1 << 20)];
        let lines = table_lines("t", None, &rows);
        assert!(lines[0].contains("dev 不明"), "{lines:?}");
        assert!(lines[0].contains("ledger 1 MiB"), "{lines:?}");
    }

    #[test]
    fn card_cap_yields_to_other_residents() {
        let total = 32_110usize << 20;
        let reserve = 1024 << 20;
        // 他が 9 GiB 使っている: 22,000 MiB の要求はカード側で切られる。
        let others = 9 * 1024usize << 20;
        let cap = card_expert_cap(22_000 << 20, total, reserve, others);
        assert_eq!(cap, total - reserve - others);
        assert!(cap < (22_000 << 20), "env 予算より小さい");
        // 他が 3 GiB なら env 予算がそのまま通る。
        let cap2 = card_expert_cap(22_000 << 20, total, reserve, 3 * 1024usize << 20);
        assert_eq!(cap2, 22_000 << 20);
        // env 0 (無制限) でもカード側で必ず有限になる。
        let cap3 = card_expert_cap(0, total, reserve, others);
        assert_eq!(cap3, total - reserve - others);
        // 他が総量を超えても 0 で止まる (アンダーフローしない)。
        assert_eq!(card_expert_cap(0, total, reserve, total * 2), 0);
    }

    #[test]
    fn card_budget_env_parsing() {
        assert!(parse_card_budget(None, true));
        assert!(!parse_card_budget(None, false));
        assert!(parse_card_budget(Some("1"), false));
        assert!(parse_card_budget(Some(" on "), false));
        assert!(!parse_card_budget(Some("0"), true));
        assert!(!parse_card_budget(Some("off"), true));
        assert!(parse_card_budget(Some("なにか"), true), "読めない値は既定");
    }

    #[test]
    fn mib_rounds_up() {
        assert_eq!(mib(0), 0);
        assert_eq!(mib(1), 1);
        assert_eq!(mib(1 << 20), 1);
        assert_eq!(mib((1 << 20) + 1), 2);
    }
}
