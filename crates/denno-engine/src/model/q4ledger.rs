//! task 138: decode 1 トークンの GPU 往復台帳 (`DENNO_Q4_PROFILE`)。
//!
//! decode の時間は「カーネルの計算」より「ホスト ↔ GPU の往復」が支配的な
//! 区間がある (mix / router / shared / full / ssm)。ここでは区間ごとに
//!
//! - 壁時計 (`wall`)
//! - カーネル発行の回数と発行そのものの時間 (`launch`)
//! - `cuCtxSynchronize` の回数と待ち時間 (`sync`)。空の sync の実測値を
//!   差し引いた残りを `gpu` (カーネル実行 + 転送の完了待ち) として出す
//! - H2D / D2H の回数とバイト数 (`h2d` / `d2h`)
//! - 上記のどれでもない残り (`cpu` = ホスト側の要素演算・確保など)
//!
//! を積む。区間の切れ目はモデル側が [`enter`] のガードで付ける
//! (`run_layer` の mix / attn、`moe_ffn_quant_impl` の router / shared)。
//! 発行・同期・コピーの実測は cuda 側の 1 箇所 (`launch_async_on` / `sync` /
//! コピーの trampoline) から入るので、区間の中の**全部**が数えられる。
//!
//! 観測だけを行う: カーネル・引数・発行順・コピーの順序は 1 つも変えない。
//! 既定 off では [`stamp`] が `None` を返し、[`note_launch`] などは
//! `enabled()` の分岐 1 回で戻る。
//!
//! 出力は `serve_flash::prof_flash_q4` が decode のリクエスト終端で 1 回
//! ([`line`])、prefill の終端で 1 回 ([`line_prefill`])。decode は **1 トークン
//! あたり**、prefill は **合計と割合**で出す (task 145)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// 区間: hc mix (attention 前 / ffn 前 / 最終 norm)。
pub const MIX: usize = 0;
/// 区間: MoE の router (gate_inp の GEMV → softmax → top-k)。
pub const ROUTER: usize = 1;
/// 区間: shared expert (gate/up の重ね打ち + down + gate_inp)。
pub const SHARED: usize = 2;
/// 区間: full attention 層の attention。
pub const FULL: usize = 3;
/// 区間: ssm 層の attention (qkv / gate / GDN / out)。
pub const SSM: usize = 4;
/// 区間: 選択 expert の本体 (task 145: prefill の expert を分離)。
///
/// prefill のバッチ経路では「重みの前置き (prefetch) を除いた、選択 expert の
/// 寄与計算」= `moe_experts_flat` / `moe_experts_legacy` / grouped GEMM の区間。
pub const EXPERTS: usize = 5;
/// 区間: 上のどれでもない (embed / ple / expert 本体 / head / サンプラ)。
///
/// prefill では層ループの外側 (embed / PLE / 最終 mixer / head) と、層の中で
/// まだ区間に割っていない部分がここに入る。`forward_prefill_inner` が全体を
/// この区間で包むので、prefill では「全項目の和 = prefill の壁時計」になる。
pub const OTHER: usize = 6;
/// 区間の数。
pub const N: usize = 7;
/// 区間の名前 ([`MIX`] の順)。
pub const NAMES: [&str; N] = ["mix", "router", "shared", "full", "ssm", "experts", "other"];

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // q4prof と同じ解釈 (値は見ない。`DENNO_Q4_PROFILE=0` でも有効)。
    *ON.get_or_init(|| std::env::var_os("DENNO_Q4_PROFILE").is_some())
}

/// テストが env を立てずに台帳を通すための強制フラグ (env は `OnceLock`)。
#[cfg(test)]
static FORCE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 台帳をテストから強制する (`false` で env の解釈に戻る)。
#[cfg(test)]
pub fn set_for_test(v: bool) {
    FORCE_ON.store(v, Ordering::Relaxed);
}

/// 計測するか (テストの強制を含む)。
#[inline]
pub fn enabled() -> bool {
    #[cfg(test)]
    if FORCE_ON.load(Ordering::Relaxed) {
        return true;
    }
    on()
}

/// 1 区間ぶんの積算器 (スレッドを跨いでも読めるよう atomic)。
struct Acc {
    wall_ns: AtomicU64,
    launches: AtomicU64,
    launch_ns: AtomicU64,
    syncs: AtomicU64,
    sync_ns: AtomicU64,
    h2d_calls: AtomicU64,
    h2d_bytes: AtomicU64,
    h2d_ns: AtomicU64,
    d2h_calls: AtomicU64,
    d2h_bytes: AtomicU64,
    d2h_ns: AtomicU64,
}

impl Acc {
    const fn new() -> Self {
        Self {
            wall_ns: AtomicU64::new(0),
            launches: AtomicU64::new(0),
            launch_ns: AtomicU64::new(0),
            syncs: AtomicU64::new(0),
            sync_ns: AtomicU64::new(0),
            h2d_calls: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
            h2d_ns: AtomicU64::new(0),
            d2h_calls: AtomicU64::new(0),
            d2h_bytes: AtomicU64::new(0),
            d2h_ns: AtomicU64::new(0),
        }
    }
}

static ROWS: [Acc; N] = [const { Acc::new() }; N];

/// 空の sync に近い待ち時間 (全 sync の最小値)。`u64::MAX` = まだ無い。
static SYNC_MIN_NS: AtomicU64 = AtomicU64::new(u64::MAX);

/// prefill 1 回ぶんの壁時計 (ns)。[`total`] のガードが積む。
///
/// 区間のガードは**入れ子を許す** (外側は内側を含む) ので、区間の和は壁時計に
/// ならない。prefill は「全体の壁時計 − 名前付き区間の和」を `other` として
/// 出したいので、全体だけ別に 1 本で測る。
static PREFILL_WALL_NS: AtomicU64 = AtomicU64::new(0);

/// prefill 全体の壁時計を積むガード ([`total`])。
pub struct Total(Option<Instant>);

impl Drop for Total {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            PREFILL_WALL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// prefill 全体の壁時計 (ns)。まだ測っていなければ 0。
pub fn prefill_wall_ns() -> u64 {
    PREFILL_WALL_NS.load(Ordering::Relaxed)
}

/// prefill 全体を 1 本で測る (off では `Instant` を取らないだけ)。
pub fn total() -> Total {
    Total(if enabled() { Some(Instant::now()) } else { None })
}

thread_local! {
    /// いま開いている区間。ネストはガードが前の値を復元する。
    static CUR: std::cell::Cell<usize> = const { std::cell::Cell::new(OTHER) };
}

/// 区間を開く。戻り値の drop で壁時計を積み、前の区間へ戻す。
///
/// off のときは `Instant` を取らず、[`CUR`] にも触らない (no-op)。
pub fn enter(item: usize) -> Scope {
    if !enabled() || item >= N {
        return Scope::none();
    }
    let prev = CUR.with(|c| c.replace(item));
    Scope {
        prev,
        item,
        t: Some(Instant::now()),
    }
}

/// [`enter`] が返すガード。
pub struct Scope {
    prev: usize,
    item: usize,
    t: Option<Instant>,
}

impl Scope {
    fn none() -> Self {
        Self {
            prev: OTHER,
            item: OTHER,
            t: None,
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(t) = self.t else { return };
        ROWS[self.item]
            .wall_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CUR.with(|c| c.set(self.prev));
    }
}

/// 計測が要るときだけ `Instant` を取る (既定 off では `None`)。
#[inline]
pub fn stamp() -> Option<Instant> {
    if enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline]
fn cur() -> usize {
    CUR.with(|c| c.get())
}

/// カーネル 1 発を数える (`t` は発行前の `Instant`)。
#[inline]
pub fn note_launch(t: Option<Instant>) {
    let Some(t) = t else { return };
    let r = &ROWS[cur()];
    r.launches.fetch_add(1, Ordering::Relaxed);
    r.launch_ns
        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// `cuCtxSynchronize` 1 回を数える (`t` は同期前の `Instant`)。
#[inline]
pub fn note_sync(t: Option<Instant>) {
    let Some(t) = t else { return };
    let ns = t.elapsed().as_nanos() as u64;
    let r = &ROWS[cur()];
    r.syncs.fetch_add(1, Ordering::Relaxed);
    r.sync_ns.fetch_add(ns, Ordering::Relaxed);
    // 最小値 (空に近い sync = ドライバ往復そのもの) を残す。
    SYNC_MIN_NS.fetch_min(ns, Ordering::Relaxed);
}

/// H2D 1 回を数える。`t` はコピー呼び出しの前の `Instant`。
#[inline]
pub fn note_h2d(bytes: usize, t: Option<Instant>) {
    let Some(t) = t else { return };
    let r = &ROWS[cur()];
    r.h2d_calls.fetch_add(1, Ordering::Relaxed);
    r.h2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    r.h2d_ns
        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// D2H 1 回を数える。`t` はコピー呼び出しの前の `Instant`。
#[inline]
pub fn note_d2h(bytes: usize, t: Option<Instant>) {
    let Some(t) = t else { return };
    let r = &ROWS[cur()];
    r.d2h_calls.fetch_add(1, Ordering::Relaxed);
    r.d2h_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    r.d2h_ns
        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// 1 区間ぶんの読み取り値。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Row {
    pub wall_ns: u64,
    pub launches: u64,
    pub launch_ns: u64,
    pub syncs: u64,
    pub sync_ns: u64,
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub h2d_ns: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
    pub d2h_ns: u64,
}

fn load(r: &Acc) -> Row {
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    Row {
        wall_ns: g(&r.wall_ns),
        launches: g(&r.launches),
        launch_ns: g(&r.launch_ns),
        syncs: g(&r.syncs),
        sync_ns: g(&r.sync_ns),
        h2d_calls: g(&r.h2d_calls),
        h2d_bytes: g(&r.h2d_bytes),
        h2d_ns: g(&r.h2d_ns),
        d2h_calls: g(&r.d2h_calls),
        d2h_bytes: g(&r.d2h_bytes),
        d2h_ns: g(&r.d2h_ns),
    }
}

/// 区間 `item` の積算を読む (テストと [`line`] 用)。
pub fn row(item: usize) -> Row {
    load(&ROWS[item.min(N - 1)])
}

/// 空の sync に近い待ち時間 (ns)。まだ 1 回も同期していなければ `None`。
pub fn sync_floor_ns() -> Option<u64> {
    match SYNC_MIN_NS.load(Ordering::Relaxed) {
        u64::MAX => None,
        v => Some(v),
    }
}

/// 台帳を 1 つの文字列群にする (`tok` = その窓の decode トークン数)。
///
/// 値は 1 トークンあたり。無効時は空 (出力ゼロ)。
pub fn line(tok: usize) -> Vec<String> {
    if !enabled() {
        return Vec::new();
    }
    let d = tok.max(1) as f64;
    let floor_ns = sync_floor_ns().unwrap_or(0);
    let floor_us = floor_ns as f64 / 1e3;

    let rows: Vec<Row> = (0..N).map(row).collect();
    let t = |f: fn(&Row) -> u64| -> u64 { rows.iter().map(f).sum() };
    let total_wall = t(|r| r.wall_ns) as f64 / 1e6 / d;
    let total_sync = t(|r| r.sync_ns) as f64 / 1e6 / d;
    let total_launch = t(|r| r.launches);
    let total_syncs = t(|r| r.syncs);
    let total_h2d: u64 = t(|r| r.h2d_bytes);
    let total_d2h: u64 = t(|r| r.d2h_bytes);

    let mut out = vec![format!(
        "[q4ledger] tok={tok} total wall={total_wall:.2}ms sync={total_sync:.2}ms \
         launches={:.1}/tok syncs={:.1}/tok h2d={:.2}MB d2h={:.2}MB sync_floor={floor_us:.1}us",
        total_launch as f64 / d,
        total_syncs as f64 / d,
        total_h2d as f64 / 1e6 / d,
        total_d2h as f64 / 1e6 / d,
    )];
    out.push(
        "[q4ledger]   item wall launch/cnt sync/cnt gpu cpu h2d d2h  (ms/tok・MB/tok)"
            .to_string(),
    );
    for (i, r) in rows.iter().enumerate() {
        if r.wall_ns == 0 && r.launches == 0 && r.syncs == 0 {
            continue;
        }
        let ms = |ns: u64| ns as f64 / 1e6 / d;
        // 空の sync (ドライバ往復) を引いた残り = カーネル実行 + 転送の完了待ち。
        let gpu_ns = r.sync_ns.saturating_sub(r.syncs.saturating_mul(floor_ns));
        let busy_ns = r.launch_ns + r.sync_ns + r.h2d_ns + r.d2h_ns;
        let cpu_ns = r.wall_ns.saturating_sub(busy_ns);
        out.push(format!(
            "[q4ledger]   {:<6} wall={:.2} launch={:.2}/{:.0} sync={:.2}/{:.0} \
             gpu={:.2} cpu={:.2} h2d={:.2}MB/{:.0} d2h={:.2}MB/{:.0}",
            NAMES[i],
            ms(r.wall_ns),
            ms(r.launch_ns),
            r.launches as f64 / d,
            ms(r.sync_ns),
            r.syncs as f64 / d,
            ms(gpu_ns),
            ms(cpu_ns),
            r.h2d_bytes as f64 / 1e6 / d,
            r.h2d_calls as f64 / d,
            r.d2h_bytes as f64 / 1e6 / d,
            r.d2h_calls as f64 / d,
        ));
    }
    out
}

/// prefill 1 回ぶんの台帳 (`tok` = その prefill のトークン数)。
///
/// [`line`] は decode の 1 トークンあたりに割った値だが、prefill は
/// 「1 チャンク = 数百トークン」で区間の内訳を**合計 ms** で読みたいので
/// こちらは合計と割合を出す。`forward_prefill_inner` が全体を [`OTHER`] で
/// 包むので、全項目の `wall` の和 = prefill の壁時計 (層の外も含む)。
///
/// 列: `wall` (ms) / 割合 / `launch` (発行そのものの時間と回数) /
/// `sync` (待ち時間と回数) / `gpu` (sync から空 sync 分を引いた残り) /
/// `cpu` (wall - launch - sync - コピー) / `h2d` `d2h` (MB と回数)。
pub fn line_prefill(tok: usize) -> Vec<String> {
    if !enabled() {
        return Vec::new();
    }
    let floor_ns = sync_floor_ns().unwrap_or(0);
    let rows: Vec<Row> = (0..N).map(row).collect();
    // 名前付きの区間 (other は残差として出す)。
    let named = [MIX, ROUTER, SHARED, FULL, SSM, EXPERTS];
    let sum_named: u64 = named.iter().map(|&i| rows[i].wall_ns).sum();
    let measured = PREFILL_WALL_NS.load(Ordering::Relaxed);
    let total_wall = if measured == 0 { sum_named } else { measured };
    let other_ns = total_wall.saturating_sub(sum_named);
    let t = |f: fn(&Row) -> u64| -> u64 { rows.iter().map(f).sum() };
    let total_launch = t(|r| r.launches);
    let total_syncs = t(|r| r.syncs);
    let total_h2d: u64 = t(|r| r.h2d_bytes);
    let total_d2h: u64 = t(|r| r.d2h_bytes);
    let d = total_wall.max(1) as f64;

    let mut out = vec![format!(
        "[q4ledger-pf] tok={tok} wall={:.1}ms launches={total_launch} syncs={total_syncs} \
         h2d={:.1}MB d2h={:.1}MB sync_floor={:.1}us",
        total_wall as f64 / 1e6,
        total_h2d as f64 / 1e6,
        total_d2h as f64 / 1e6,
        floor_ns as f64 / 1e3,
    )];
    out.push(
        "[q4ledger-pf]   item   wall_ms  share  launch_ms/cnt  sync_ms/cnt  gpu_ms  cpu_ms  \
         h2d_MB/cnt  d2h_MB/cnt"
            .to_string(),
    );
    let ms = |ns: u64| ns as f64 / 1e6;
    let mut push_row = |name: &str, wall_ns: u64, r: Option<&Row>| {
        let zero = Row::default();
        let r = r.unwrap_or(&zero);
        let gpu_ns = r.sync_ns.saturating_sub(r.syncs.saturating_mul(floor_ns));
        let busy_ns = r.launch_ns + r.sync_ns + r.h2d_ns + r.d2h_ns;
        // `other` (残差) は launch/sync/コピーを持たないので、cpu は残差そのもの。
        let cpu_ns = if name == NAMES[OTHER] {
            wall_ns
        } else {
            wall_ns.saturating_sub(busy_ns)
        };
        out.push(format!(
            "[q4ledger-pf]   {:<7} {:>8.1} {:>5.1}%  {:>8.2}/{:<6} {:>8.2}/{:<6} {:>7.1} {:>7.1}  \
             {:>8.2}/{:<6} {:>8.2}/{:<6}",
            name,
            ms(wall_ns),
            wall_ns as f64 * 100.0 / d,
            ms(r.launch_ns),
            r.launches,
            ms(r.sync_ns),
            r.syncs,
            ms(gpu_ns),
            ms(cpu_ns),
            r.h2d_bytes as f64 / 1e6,
            r.h2d_calls,
            r.d2h_bytes as f64 / 1e6,
            r.d2h_calls,
        ));
    };
    for &i in named.iter() {
        if rows[i].wall_ns == 0 && rows[i].launches == 0 && rows[i].syncs == 0 {
            continue;
        }
        push_row(NAMES[i], rows[i].wall_ns, Some(&rows[i]));
    }
    // 残差 (層の外 = embed / PLE / 最終 mixer / head / 未計測)。
    push_row(NAMES[OTHER], other_ns, Some(&rows[OTHER]));
    out
}

/// 積算を 0 に戻す (リクエスト境界。`serve_flash` が [`line`] の直後に呼ぶ)。
pub fn reset() {
    for r in ROWS.iter() {
        for c in [
            &r.wall_ns,
            &r.launches,
            &r.launch_ns,
            &r.syncs,
            &r.sync_ns,
            &r.h2d_calls,
            &r.h2d_bytes,
            &r.h2d_ns,
            &r.d2h_calls,
            &r.d2h_bytes,
            &r.d2h_ns,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
    SYNC_MIN_NS.store(u64::MAX, Ordering::Relaxed);
    PREFILL_WALL_NS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 積算はプロセス共通なので 1 本ずつ回す (moe と同じロック)。
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::model::moe_expert::moe_test_lock()
    }

    /// off では Instant も取らず、[`line`] も出ない (既定 off の契約)。
    #[test]
    fn off_is_noop() {
        let _g = lock();
        set_for_test(false);
        reset();
        {
            let _s = enter(MIX);
            note_launch(stamp());
            note_sync(stamp());
            note_h2d(1024, stamp());
            note_d2h(512, stamp());
        }
        assert_eq!(row(MIX), Row::default(), "off なのに積まれた");
        assert!(line(8).is_empty(), "off でレポートが出た");
        assert_eq!(sync_floor_ns(), None);
    }

    /// 区間ごとに分かれ、ネストは内側に付いて外側へ戻る。
    #[test]
    fn scopes_are_per_item_and_nest() {
        let _g = lock();
        set_for_test(true);
        reset();
        {
            let _outer = enter(SSM);
            note_launch(stamp());
            note_h2d(4000, stamp());
            {
                let _inner = enter(SHARED);
                note_launch(stamp());
                note_launch(stamp());
                note_d2h(800, stamp());
                note_sync(stamp());
            }
            // ネストから戻ったら、また ssm に積まれる。
            note_launch(stamp());
        }
        let ssm = row(SSM);
        let sh = row(SHARED);
        assert_eq!(ssm.launches, 2, "ssm の launch 数: {ssm:?}");
        assert_eq!(ssm.h2d_bytes, 4000);
        assert_eq!(sh.launches, 2, "shared の launch 数: {sh:?}");
        assert_eq!(sh.d2h_bytes, 800);
        assert_eq!(sh.syncs, 1);
        assert!(ssm.wall_ns > 0 && sh.wall_ns > 0, "wall が積まれていない");
        assert_eq!(row(MIX), Row::default());
        reset();
        set_for_test(false);
    }

    /// 出力は 1 トークンあたりで、全 item の名前と列が揃う。
    #[test]
    fn line_is_per_token_and_has_all_columns() {
        let _g = lock();
        set_for_test(true);
        reset();
        // 2 トークンぶんに見立てた値: mix に launch 200 回・sync 100 回・4 MB H2D。
        {
            let _s = enter(MIX);
            for _ in 0..200 {
                note_launch(Some(Instant::now()));
            }
            for _ in 0..100 {
                note_sync(Some(Instant::now()));
            }
            for _ in 0..2 {
                note_h2d(2_000_000, Some(Instant::now()));
            }
        }
        let lines = line(2);
        assert!(!lines.is_empty(), "on なのにレポートが出ない");
        let head = &lines[0];
        assert!(head.starts_with("[q4ledger] tok=2"), "{head}");
        assert!(head.contains("sync_floor="), "{head}");
        // launch は 100/tok。
        assert!(head.contains("launches=100.0/tok"), "{head}");
        let mix_line = lines
            .iter()
            .find(|l| l.contains(" mix "))
            .expect("mix の行が無い");
        assert!(mix_line.contains("h2d=2.00MB/1"), "{mix_line}");
        assert!(mix_line.contains("wall="), "{mix_line}");
        reset();
        set_for_test(false);
    }

    /// 最小 sync は 0 に戻る (窓ごとに取り直す)。
    #[test]
    fn sync_floor_resets() {
        let _g = lock();
        set_for_test(true);
        reset();
        note_sync(Some(Instant::now()));
        assert!(sync_floor_ns().is_some());
        reset();
        assert_eq!(sync_floor_ns(), None);
        set_for_test(false);
    }

    /// prefill の台帳は合計 ms と割合で、experts が other と別に出る (task 145)。
    #[test]
    fn prefill_line_totals_and_experts() {
        let _g = lock();
        set_for_test(true);
        reset();
        // 全体の外側 (other) の中で experts が動いた形を作る。
        {
            let _all = enter(OTHER);
            {
                let _e = enter(EXPERTS);
                for _ in 0..4 {
                    note_launch(Some(Instant::now()));
                }
            }
            {
                let _m = enter(MIX);
                for _ in 0..2 {
                    note_launch(Some(Instant::now()));
                }
            }
        }
        let lines = line_prefill(512);
        assert!(!lines.is_empty(), "on なのに prefill の台帳が出ない");
        let head = &lines[0];
        assert!(head.starts_with("[q4ledger-pf] tok=512"), "{head}");
        assert!(head.contains("wall="), "{head}");
        assert!(head.contains("sync_floor="), "{head}");
        let exp = lines
            .iter()
            .find(|l| l.contains(" experts "))
            .expect("experts の行が無い");
        assert!(exp.contains("%"), "{exp}");
        // other は外側のガードなので、内側 (experts / mix) の時間を含まない。
        let other = lines
            .iter()
            .find(|l| l.contains(" other "))
            .expect("other の行が無い");
        assert!(other.contains("%"), "{other}");
        assert!(other.contains("experts") == false, "{other}");
        // 全項目の wall の和 = 全体 (other が残りを吸う)。
        let sum: u64 = (0..N).map(|i| row(i).wall_ns).sum();
        let total: u64 = lines[0]
            .split("wall=")
            .nth(1)
            .and_then(|s| s.split("ms").next())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| (v * 1e6) as u64)
            .expect("wall の値が読めない");
        assert!(
            sum.abs_diff(total) < 1_000_000,
            "項目の和 {sum} と全体 {total} が合わない"
        );
        reset();
        set_for_test(false);
    }
}
