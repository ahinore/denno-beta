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
/// 区間: 選択 expert の寄与計算 (task 145)。decode と prefill の両方で立つ。
///
/// prefill のバッチ経路では「重みの前置き (prefetch) を除いた、選択 expert の
/// 寄与計算」= `moe_experts_flat` / `moe_experts_legacy` / grouped GEMM の区間。
/// decode では CPU ミスぶんと [`EXPERT`] (IQ の層一括 GEMV) の**両方**を含む
/// (IQ の GPU 区間はこの内側に入る)。台帳の名前は `experts`。
pub const EXPERTS: usize = 5;
/// 区間: 上のどれでもない (embed / ple / 層の外 / head / サンプラ)。
///
/// prefill では層ループの外側 (embed / PLE / 最終 mixer / head) と、層の中で
/// まだ区間に割っていない部分がここに入る。`forward_prefill_inner` が全体を
/// この区間で包むので、prefill では「全項目の和 = prefill の壁時計」になる。
pub const OTHER: usize = 6;
/// 区間: IQ 系 expert の層一括 GEMV (`kernels_exp_iq`)。
///
/// 「1 層あたり何 us か」を台帳から直接読むための区間。ミスして CPU で計算した
/// expert はこの区間の外 (呼び出し側のループ) なので、**GPU で回したぶんだけ**が
/// ここに積まれる。decode の IQ 経路だけが通り、decode では [`EXPERTS`] の
/// 内側 (ネスト) に入る。台帳の名前は `exp_iq`。
pub const EXPERT: usize = 7;
/// task 153: 層の外の head (最終 head GEMV と logits の D2H)。
///
/// `forward_token_at` の末尾だけが通る。台帳の名前は `head`。
pub const HEAD: usize = 8;
/// task 153: 層ごとの活性タップ (`ActTap::on_layer` = LayerAct フレームの組み立て)。
///
/// decode の層ループの中で 1 層 1 回呼ばれる。台帳の名前は `tap`。
pub const TAP: usize = 9;
/// task 153: サンプラ 1 回 (`Sampler::sample`)。名前は `sample`。
pub const SAMPLE: usize = 10;
/// task 153: テレメトリ用の top-8 softmax (`topk_softmax`)。名前は `topk`。
pub const TOPK: usize = 11;
/// task 153: 出力トークンの切り出しと本文への反映 (`emit_confirmed`)。名前は `detok`。
pub const DETOK: usize = 12;
/// task 153: テレメトリのフレーム送出 (`Frame::Step` / `Token` / `TopK` / `KvUsage`)。
pub const TELEM: usize = 13;
/// task 153: engine の 1 ステップ全体 (`forward_token_at` + 構造テレメトリ)。
///
/// 層の中の名前付き区間に入らなかった時間 (hooks / arch など) は
/// この行の残差として出る。台帳の名前は `step`。
pub const STEP: usize = 14;
/// task 153: 層の出口の `hc_combine` (hyper connection の足し込み)。
///
/// 1 層 2 回 (attn の後 / ffn の後)。40 KB の確保と 10240 要素の写しがある。
pub const COMBINE: usize = 15;
/// task 153: 層ごとの役割 RMS と構造テレメトリ (`emit_role_act`)。
pub const ROLES: usize = 16;
/// task 155: 層の中の**フック呼び出し** (`hook("hc_mixed_attn", ...)` など 7 回/層)。
///
/// 購読者が居ないときは呼び出し自体を飛ばす (`DENNO_HOOK_SKIP` 既定 on) ので、
/// 通常は 0 付近になる。台帳の名前は `hooks`。
pub const HOOKS: usize = 17;
/// task 155: `q4prof::add` (3 回/層) の区間。名前は `qprof`。
pub const QPROF: usize = 18;
/// task 155: `arch_emit_token` (構造テレメトリの組み立て) の区間。名前は `arch`。
pub const ARCH: usize = 19;
/// task 155: 層ループの端 (層の入口の役割リセット / 出口の `LayerOut` 組み立てと
/// `on_layer` 呼び出し) と、MoE の中で router と experts の間に落ちる部分
/// (使用回数の記録 / ゲート判定 / 作業域の確保)。名前は `loop`。
pub const LOOP: usize = 20;
/// 区間の数。
pub const N: usize = 21;
/// 区間の名前 ([`MIX`] の順)。`experts` (選択 expert の寄与全体) と
/// `exp_iq` (task 144 の IQ 層一括 GEMV、decode では `experts` の内側) は別の行。
pub const NAMES: [&str; N] = [
    "mix", "router", "shared", "full", "ssm", "experts", "other", "exp_iq", "head", "tap",
    "sample", "topk", "detok", "telem", "step", "combine", "roles", "hooks", "qprof", "arch",
    "loop",
];

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
        pf_span_end();
    }
}

/// prefill 全体の壁時計 (ns)。まだ測っていなければ 0。
pub fn prefill_wall_ns() -> u64 {
    PREFILL_WALL_NS.load(Ordering::Relaxed)
}

/// prefill 全体を 1 本で測る (off では `Instant` を取らないだけ)。
pub fn total() -> Total {
    let t = if enabled() {
        Some(Instant::now())
    } else {
        None
    };
    // task 254: ホスト台帳が on なら、区間の入口/出口と起動ペアにイベントを
    // 打って GPU の実時間を積む (decode の `DENNO_Q4_TLINE` とは別の窓)。
    if t.is_some() {
        pf_span_begin();
    }
    Total(t)
}

// ---- task 254: prefill の GPU 時間 (イベント) -----------------------------
//
// ホスト台帳 (`DENNO_PF_HOSTPROF`) のときだけ、prefill の窓で
// - 区間 (`enter`) の入口/出口に 1 本ずつ
// - カーネル起動の前後に 1 本ずつ ([`crate::cuda::tline`] の `launch_*`)
// を打つ。起動ペアの差が **GPU の実時間** (アイドルを含まない) で、区間の
// `wall` (ホスト) から引いた残りが「露出したホスト時間」になる。

/// prefill の窓が開いているか (イベントを打つかの分岐 1 回)。
static PF_SPAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// prefill の窓が開いているか。
#[inline]
pub fn pf_span_active() -> bool {
    PF_SPAN.load(Ordering::Relaxed)
}

/// prefill の窓を開ける (`q4ledger::total` が 1 回)。
fn pf_span_begin() {
    if !crate::model::pf_host::on() {
        return;
    }
    PF_SPAN.store(true, Ordering::Relaxed);
    pf_span_prepare();
}

/// 窓を閉じる (イベントは [`line_prefill`] が読んでから捨てる)。
fn pf_span_end() {
    PF_SPAN.store(false, Ordering::Relaxed);
}

/// 窓の入口の用意 (cuda 無しは何もしない)。
#[cfg(feature = "cuda")]
#[inline]
fn pf_span_prepare() {
    crate::cuda::tline::prepare_prefill();
}

/// 窓の入口の用意 (cuda 無しは何もしない)。
#[cfg(not(feature = "cuda"))]
#[inline]
fn pf_span_prepare() {}

/// task 254: 区間ごとの GPU 実時間 (ms、起動ペアの和)。窓が閉じていれば `None`。
#[cfg(feature = "cuda")]
fn pf_busy_ms() -> Option<Vec<f64>> {
    crate::cuda::tline::item_busy_ms(N)
}

/// task 254: cuda 無しは常に `None`。
#[cfg(not(feature = "cuda"))]
fn pf_busy_ms() -> Option<Vec<f64>> {
    None
}

/// task 254: イベントの記録を捨てる (cuda 無しは何もしない)。
#[cfg(feature = "cuda")]
fn pf_span_clear() {
    crate::cuda::tline::reset();
}

/// task 254: cuda 無しは何もしない。
#[cfg(not(feature = "cuda"))]
fn pf_span_clear() {}

thread_local! {
    /// いま開いている区間。ネストはガードが前の値を復元する。
    static CUR: std::cell::Cell<usize> = const { std::cell::Cell::new(OTHER) };
}

/// いま開いている区間 (イベント時系列が起動を区間に割り当てるのに使う)。
#[inline]
pub fn current() -> usize {
    CUR.with(|c| c.get())
}

/// task 153: CUDA イベントの時系列を取るか (`DENNO_Q4_TLINE` があるとき)。
fn tline_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DENNO_Q4_TLINE").is_some())
}

/// 時系列の記録が開いているか (decode の窓の間だけ真)。
static TLINE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 時系列が記録中か (イベント 1 本ぶんの分岐)。
#[inline]
pub fn tline_active() -> bool {
    TLINE_ACTIVE.load(Ordering::Relaxed)
}

/// decode の窓を開ける (`serve_flash` が prefill の台帳を出した直後に 1 回)。
pub fn tline_begin() {
    if enabled() && tline_on() {
        TLINE_ACTIVE.store(true, Ordering::Relaxed);
        tline_prepare();
    }
}

/// 窓の入口の用意 (cuda 無しは何もしない)。
#[cfg(feature = "cuda")]
#[inline]
fn tline_prepare() {
    crate::cuda::tline::prepare();
}

/// 窓の入口の用意 (cuda 無しは何もしない)。
#[cfg(not(feature = "cuda"))]
#[inline]
fn tline_prepare() {}

/// decode の窓を閉じる (decode の台帳を出す直前に 1 回)。
pub fn tline_end() {
    TLINE_ACTIVE.store(false, Ordering::Relaxed);
}

/// task 173: GDN prefill 連鎖の段ごとの GPU 時間 (`DENNO_GDN_PF_KPROF=1`) の 1 行。
///
/// 計測器は cuda.rs 側にあるので、cuda feature の有無をここで吸収する
/// (denno-cli は feature を持たないため、cuda 無しでは常に空)。
#[cfg(feature = "cuda")]
pub fn gdn_pf_kprof_lines() -> Vec<String> {
    crate::cuda::gdn_pf_kprof::lines()
}

/// task 173: cuda 無しは常に空。
#[cfg(not(feature = "cuda"))]
pub fn gdn_pf_kprof_lines() -> Vec<String> {
    Vec::new()
}

/// task 187: decode の IQ expert 段の層内 GPU 時間 (イベント計測) の 1 行。
///
/// 計測器は cuda.rs 側にあるので、cuda feature の有無をここで吸収する。
#[cfg(feature = "cuda")]
pub fn dec_exp_kprof_lines() -> Vec<String> {
    crate::cuda::dec_exp_kprof::lines()
}

/// task 187: cuda 無しは常に空。
#[cfg(not(feature = "cuda"))]
pub fn dec_exp_kprof_lines() -> Vec<String> {
    Vec::new()
}

/// task 187: 積算を 0 に戻す (cuda 無しは何もしない)。
#[cfg(feature = "cuda")]
pub fn dec_exp_kprof_reset() {
    crate::cuda::dec_exp_kprof::reset()
}

/// task 187: cuda 無しは何もしない。
#[cfg(not(feature = "cuda"))]
pub fn dec_exp_kprof_reset() {}

/// task 173: 計測の積算を 0 に戻す (cuda 無しは何もしない)。
#[cfg(feature = "cuda")]
pub fn gdn_pf_kprof_reset() {
    crate::cuda::gdn_pf_kprof::reset()
}

/// task 173: cuda 無しは何もしない。
#[cfg(not(feature = "cuda"))]
pub fn gdn_pf_kprof_reset() {}

/// task 189: prefill の mix 1 呼び出しの内訳 (`DENNO_PF_MIX_KPROF=1`) の 1 行。
///
/// 計測器は cuda.rs 側にあるので、cuda feature の有無をここで吸収する。
#[cfg(feature = "cuda")]
pub fn pf_mix_kprof_lines() -> Vec<String> {
    crate::cuda::pf_mix_kprof::lines()
}

/// task 189: cuda 無しは常に空。
#[cfg(not(feature = "cuda"))]
pub fn pf_mix_kprof_lines() -> Vec<String> {
    Vec::new()
}

/// task 189: cuda 無しは何もしない。
#[cfg(feature = "cuda")]
pub fn pf_mix_kprof_reset() {
    crate::cuda::pf_mix_kprof::reset()
}

/// task 189: cuda 無しは何もしない。
#[cfg(not(feature = "cuda"))]
pub fn pf_mix_kprof_reset() {}

/// 区間を開く。戻り値の drop で壁時計を積み、前の区間へ戻す。
///
/// off のときは `Instant` を取らず、[`CUR`] にも触らない (no-op)。
/// task 153: `DENNO_Q4_TLINE` のときは入口でイベントを 1 本打つ。
pub fn enter(item: usize) -> Scope {
    if !enabled() || item >= N {
        return Scope::none();
    }
    let prev = CUR.with(|c| c.replace(item));
    Scope {
        prev,
        item,
        t: Some(Instant::now()),
        ev: tline_row(item).then(tline_ev_begin).flatten(),
    }
}

/// task 155: その行が時系列のイベントを打つか。
///
/// task 155 で足した診断行 (`hooks` / `qprof` / `arch` / `loop`) は GPU 仕事を
/// 持たないので**イベントを打たない**。イベントのプール (60 万本) は 128 トークンの
/// 窓で 7 割方埋まっており (t153c で ~35 万)、診断行の span (実測 ~290/tok) を足すと
/// 窓の途中で記録が止まって以後の `[q4tline]` が出なくなる (t155b で実際に起きた)。
#[inline]
fn tline_row(item: usize) -> bool {
    item < HOOKS
}

/// 区間の入口イベント (時系列 off / cuda 無しでは `None`)。
#[cfg(feature = "cuda")]
#[inline]
fn tline_ev_begin() -> Option<u32> {
    crate::cuda::tline::span_begin()
}

/// 区間の入口イベント (cuda 無しは常に `None`)。
#[cfg(not(feature = "cuda"))]
#[inline]
fn tline_ev_begin() -> Option<u32> {
    None
}

/// 区間の出口イベントを打って span を積む。
#[cfg(feature = "cuda")]
#[inline]
fn tline_ev_end(item: usize, a: u32) {
    crate::cuda::tline::span_end(item, a);
}

/// 区間の出口 (cuda 無しは何もしない)。
#[cfg(not(feature = "cuda"))]
#[inline]
fn tline_ev_end(item: usize, a: u32) {
    let _ = (item, a);
}

/// [`enter`] が返すガード。
pub struct Scope {
    prev: usize,
    item: usize,
    t: Option<Instant>,
    /// task 153: 入口イベントの通し番号 (時系列 off は `None`)。
    ev: Option<u32>,
}

impl Scope {
    fn none() -> Self {
        Self {
            prev: OTHER,
            item: OTHER,
            t: None,
            ev: None,
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(t) = self.t else { return };
        ROWS[self.item]
            .wall_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(a) = self.ev {
            tline_ev_end(self.item, a);
        }
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
    sub_bytes(true, bytes);
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
    sub_bytes(false, bytes);
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

/// prefill の「cpu」列 (wall - launch - sync - copy) の中身を名前で割るための
/// 副区間 (task 154)。区間は**親** (`MIX` / `SSM` / `FULL` / `OTHER` / ...) ごとに
/// 積むので、「mix の 18.7 s のうち norm がいくつで gating がいくつか」が読める。
///
/// 親は [`sub`] を開いた時点の [`CUR`] で決まる。ネストは許すが、親子の関係は
/// 持たない (合計は親の wall を超えうる)。[`line_prefill`] は親ごとに 1 行で出す。
pub const SUB_NORM: usize = 0;
/// 残差の合成 (`hc_combine`) と mix の要素演算 (`hc_lo_apply` / `hc_mixed` / inject)。
pub const SUB_ELEM: usize = 1;
/// ホスト側の行列積 (GPU に載らなかったバッチ GEMV・F32/F16/BF16 の経路)。
pub const SUB_GEMM_CPU: usize = 2;
/// 重みの f32 展開 (上の gemm_cpu の内側にも入る)。
pub const SUB_DEQUANT: usize = 3;
/// GDN の depthwise conv1d + silu (状態の末尾コピーを含む)。
pub const SUB_CONV: usize = 4;
/// GDN のデルタ則スキャン + q/k の L2 正規化 + 出力の gated RMS norm。
pub const SUB_SCAN: usize = 5;
/// GDN の beta / alpha 変換 (`gate_table`)。
pub const SUB_GATE: usize = 6;
/// PLE の注入 (gather した行の射影・RMS・conv・silu)。
pub const SUB_PLE: usize = 7;
/// 埋め込み行のデコード。
pub const SUB_EMBED: usize = 8;
/// レイアウト変換 (行の flatten / transpose / 連結コピー)。
pub const SUB_LAYOUT: usize = 9;
/// ルータの softmax / top-k (ホスト)。
pub const SUB_SOFTMAX: usize = 10;
/// task 167: ssm 層の qkv (q/k/v) 投影 (バッチ GEMM)。
pub const SUB_PF_QKV: usize = 11;
/// task 167: ssm 層の beta/alpha 投影。
pub const SUB_PF_GATE: usize = 12;
/// task 167: GDN コア (GPU スキャン or CPU スキャン)。
pub const SUB_PF_SCAN: usize = 13;
/// task 167: ssm 層の出力投影 (out_proj)。
pub const SUB_PF_OUT: usize = 14;
/// task 187: decode の expert 段の VRAM 参照 + segment 表の組み立て。
pub const SUB_EXP_LOOKUP: usize = 15;
/// task 187: decode の expert 段の作業域の確保 (`h` / `hs`)。
pub const SUB_EXP_ALLOC: usize = 16;
/// task 187: decode の expert ミスぶんの CPU 計算。
pub const SUB_EXP_MISS: usize = 17;
/// task 187: decode の expert 寄与のホスト足し込み (`out += w*h` / shared の足し込み)。
pub const SUB_EXP_SUM: usize = 18;
/// task 187: decode のミス expert の上げ直し (ページフォルト + H2D)。
pub const SUB_EXP_UPLOAD: usize = 19;
/// task 187: shared expert の gate/up の enqueue (router 期に積むぶん)。
pub const SUB_SH_BEGIN: usize = 20;
/// task 187: shared expert の join の**待ち**。
pub const SUB_SH_WAIT: usize = 21;
/// task 187: shared expert の join の**写し**。
pub const SUB_SH_COPY: usize = 22;
/// task 187: shared expert の silu / 積 / 作業域の確保。
pub const SUB_SH_ELEM: usize = 23;
/// task 187: shared expert の down の enqueue。
pub const SUB_SH_SDOWN: usize = 24;
/// task 187: shared expert の `gate_inp_shexp` (F32・1 行) の GEMV。
pub const SUB_SH_SG: usize = 25;
/// task 187: shared expert の出力の足し込み (`out += sh*sgw`) と RMS。
pub const SUB_SH_ADD: usize = 26;
/// task 187: decode の IQ expert 段の prologue (x / ポインタ表 / segment 表の H2D)。
pub const SUB_EXP_PRO: usize = 27;
/// task 254: `other` の中の combine (`pf_combine` の H2D + カーネル + 同期)。
pub const SUB_COMBINE: usize = 28;
/// task 254: `other` の中の mix の D2H (`pf_take_mixed`。層ごとに 14 MB)。
pub const SUB_MIXD2H: usize = 29;
/// 副区間の数。
pub const SUB_N: usize = 30;
/// 副区間の名前 ([`SUB_NORM`] の順)。
pub const SUB_NAMES: [&str; SUB_N] = [
    "norm",
    "elem",
    "gemm_cpu",
    "dequant",
    "conv",
    "scan",
    "gate",
    "ple",
    "embed",
    "layout",
    "softmax",
    "qkv_pf",
    "beta_pf",
    "scan_pf",
    "out_pf",
    "exp_lookup",
    "exp_alloc",
    "exp_miss",
    "exp_sum",
    "exp_upload",
    "sh_begin",
    "sh_wait",
    "sh_copy",
    "sh_elem",
    "sh_sdown",
    "sh_sg",
    "sh_add",
    "exp_pro",
    "combine",
    "mixd2h",
];

/// 親 × 副区間の壁時計 (ns)。
static SUBS: [[AtomicU64; SUB_N]; N] = [const { [const { AtomicU64::new(0) }; SUB_N] }; N];

// task 167: 副区間の中で起きた H2D / D2H を**その副区間にも**積む。
// 台帳の行 (`ROWS`) は親の区間しか持たないので、「ssm の 105 GB のうち
// どの段が何 MB 動かしたか」が読めなかった。転送の trampoline は
// `note_h2d` / `note_d2h` に集約されているので、ここで副区間の
// カウンタにも同じ値を足す (off は thread_local の Cell 読み 1 回)。
static SUB_H2D_BYTES: [[AtomicU64; SUB_N]; N] = [const { [const { AtomicU64::new(0) }; SUB_N] }; N];
static SUB_D2H_BYTES: [[AtomicU64; SUB_N]; N] = [const { [const { AtomicU64::new(0) }; SUB_N] }; N];
static SUB_H2D_CALLS: [[AtomicU64; SUB_N]; N] = [const { [const { AtomicU64::new(0) }; SUB_N] }; N];
static SUB_D2H_CALLS: [[AtomicU64; SUB_N]; N] = [const { [const { AtomicU64::new(0) }; SUB_N] }; N];

/// 副区間を開いていないことを表す番兵。
const NO_SUB: usize = usize::MAX;

thread_local! {
    /// いま開いている副区間 (ネストは「内側が勝つ」= 最も内側に積む)。
    static CUR_SUB: std::cell::Cell<usize> = const { std::cell::Cell::new(NO_SUB) };
}

#[inline]
fn sub_bytes(is_h2d: bool, bytes: usize) {
    let item = CUR_SUB.with(|c| c.get());
    if item == NO_SUB {
        return;
    }
    let row = cur().min(N - 1);
    let item = item.min(SUB_N - 1);
    let (b, c) = if is_h2d {
        (&SUB_H2D_BYTES, &SUB_H2D_CALLS)
    } else {
        (&SUB_D2H_BYTES, &SUB_D2H_CALLS)
    };
    b[row][item].fetch_add(bytes as u64, Ordering::Relaxed);
    c[row][item].fetch_add(1, Ordering::Relaxed);
}

/// 副区間 `item` の転送 (バイト, 回数) を読む。`is_h2d` で向きを選ぶ。
pub fn sub_bytes_of(parent: usize, item: usize, is_h2d: bool) -> (u64, u64) {
    let row = parent.min(N - 1);
    let item = item.min(SUB_N - 1);
    if is_h2d {
        (
            SUB_H2D_BYTES[row][item].load(Ordering::Relaxed),
            SUB_H2D_CALLS[row][item].load(Ordering::Relaxed),
        )
    } else {
        (
            SUB_D2H_BYTES[row][item].load(Ordering::Relaxed),
            SUB_D2H_CALLS[row][item].load(Ordering::Relaxed),
        )
    }
}

/// 副区間を開く。`drop` で親の行に壁時計を積む (off では何もしない)。
pub fn sub(item: usize) -> SubScope {
    if !enabled() || item >= SUB_N {
        return SubScope {
            parent: 0,
            item: 0,
            t: None,
            prev_sub: NO_SUB,
        };
    }
    let prev_sub = CUR_SUB.with(|c| c.replace(item));
    SubScope {
        parent: cur(),
        item,
        t: Some(Instant::now()),
        prev_sub,
    }
}

/// [`sub`] が返すガード。
pub struct SubScope {
    parent: usize,
    item: usize,
    t: Option<Instant>,
    prev_sub: usize,
}

impl Drop for SubScope {
    fn drop(&mut self) {
        if self.prev_sub != NO_SUB || self.t.is_some() {
            CUR_SUB.with(|c| c.set(self.prev_sub));
        }
        let Some(t) = self.t else { return };
        SUBS[self.parent][self.item].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// 親 `parent` の副区間 `item` の積算 (ns)。テストと [`line_prefill`] 用。
pub fn sub_ns(parent: usize, item: usize) -> u64 {
    SUBS[parent.min(N - 1)][item.min(SUB_N - 1)].load(Ordering::Relaxed)
}

/// 台帳を 1 つの文字列群にする (`tok` = その窓の decode トークン数、
/// `wall_ns` = decode ループ全体の壁時計 = リクエストのループ入口から
/// 最後のトークンを出し終えるまで)。
///
/// 値は 1 トークンあたり。無効時は空 (出力ゼロ)。
pub fn line(tok: usize, wall_ns: u64) -> Vec<String> {
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
    // task 187: decode の副区間 (親ごとの内訳)。`cpu` 列 (wall - launch - sync -
    // コピー) の中身を名前で割るためのもの。0 でないものだけ出し、`*` は
    // 「他の副区間の内側にも入る」もの。
    let nested = [SUB_DEQUANT];
    let sub_ms = |ns: u64| ns as f64 / 1e6 / d;
    for &p in &[MIX, ROUTER, SHARED, FULL, SSM, EXPERTS, HEAD, OTHER] {
        let mut parts: Vec<String> = Vec::new();
        let mut sum = 0u64;
        for i in 0..SUB_N {
            let ns = SUBS[p][i].load(Ordering::Relaxed);
            if ns == 0 {
                continue;
            }
            sum += ns;
            let mark = if nested.contains(&i) { "*" } else { "" };
            parts.push(format!("{}{}={:.2}", SUB_NAMES[i], mark, sub_ms(ns)));
        }
        if parts.is_empty() {
            continue;
        }
        out.push(format!(
            "[q4ledger]   sub {:<7} sum={:.2} of {:.2}ms/tok  {}",
            NAMES[p],
            sub_ms(sum),
            sub_ms(rows[p].wall_ns),
            parts.join(" ")
        ));
    }
    // task 153: イベント時系列 (GPU の span / busy / idle)。`DENNO_Q4_TLINE` のときだけ。
    //
    // 台帳の行は**入れ子を外側にも積む** (外側の wall は内側を含む) ので、全部を
    // 足すと二重計上になる。ここでは「トップレベルの区間」(= serve_flash の
    // decode ループが開く区間) だけで `items` を作り、残りを `outside` とする。
    // 起動の対応 (busy / launches) と sync/コピーの待ちは**最内側**に付くので、
    // 全行の和が正しい総量になる。
    #[cfg(feature = "cuda")]
    if tline_on() && wall_ns > 0 {
        if let Some(tl) = crate::cuda::tline::items(N) {
            const TOP: [usize; 5] = [STEP, SAMPLE, TOPK, DETOK, TELEM];
            let total_ms = wall_ns as f64 / 1e6 / d;
            let items_ns: u64 = TOP.iter().map(|&i| rows[i].wall_ns).sum();
            let items_ms = items_ns as f64 / 1e6 / d;
            let wait_ms = t(|r| r.sync_ns + r.h2d_ns + r.d2h_ns) as f64 / 1e6 / d;
            let span_ms: f64 = TOP.iter().map(|&i| tl[i].span_ms).sum::<f64>() / d;
            let busy_ms: f64 = tl.iter().map(|i| i.busy_ms).sum::<f64>() / d;
            let n_launch: u64 = tl.iter().map(|i| i.launches).sum();
            out.push(format!(
                "[q4tline] tok={tok} token={total_ms:.2}ms items={items_ms:.2}ms outside={:.2}ms \
                 | gpu_span={span_ms:.2}ms gpu_busy={busy_ms:.2}ms gpu_idle={:.2}ms \
                 host_wait={wait_ms:.2}ms host_busy={:.2}ms launches={}",
                total_ms - items_ms,
                (span_ms - busy_ms).max(0.0),
                (total_ms - wait_ms).max(0.0),
                n_launch as f64 / d,
            ));
            out.push(
                "[q4tline]   item host wait cpu gpu_span gpu_busy gpu_idle n k  (ms/tok・行は入れ子を含む)"
                    .to_string(),
            );
            for (i, r) in rows.iter().enumerate() {
                let it = tl[i];
                if r.wall_ns == 0 && it.spans == 0 && it.launches == 0 {
                    continue;
                }
                let ms = |ns: u64| ns as f64 / 1e6 / d;
                let wait = ms(r.sync_ns + r.h2d_ns + r.d2h_ns);
                let busy_host = ms(
                    r.wall_ns
                        .saturating_sub(r.launch_ns + r.sync_ns + r.h2d_ns + r.d2h_ns),
                );
                let span = it.span_ms / d;
                let busy = it.busy_ms / d;
                out.push(format!(
                    "[q4tline]   {:<6} host={:.2} wait={wait:.2} cpu={busy_host:.2} \
                     gpu_span={span:.2} gpu_busy={busy:.2} gpu_idle={:.2} n={:<5} k={:<5}",
                    NAMES[i],
                    ms(r.wall_ns),
                    (span - busy).max(0.0),
                    it.spans,
                    it.launches,
                ));
            }
        }
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
    // task 254: GPU の実時間 (起動ペアのイベント) と、そこから出る露出ホスト時間。
    let busy = pf_busy_ms();
    if busy.is_some() {
        out.push(
            "[q4ledger-pf]   item   wall_ms  share  launch_ms/cnt  sync_ms/cnt  gpu_ms  cpu_ms  \
             h2d_MB/cnt  d2h_MB/cnt  busy_ms  exposed_ms"
                .to_string(),
        );
    } else {
        out.push(
            "[q4ledger-pf]   item   wall_ms  share  launch_ms/cnt  sync_ms/cnt  gpu_ms  cpu_ms  \
             h2d_MB/cnt  d2h_MB/cnt"
                .to_string(),
        );
    }
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
        let mut line = format!(
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
        );
        if let Some(b) = busy.as_ref() {
            let g = b.get(NAMES.iter().position(|n| *n == name).unwrap_or(0)).copied().unwrap_or(0.0);
            line.push_str(&format!("  {g:>7.1} {:>10.1}", (ms(wall_ns) - g).max(0.0)));
        }
        out.push(line);
    };
    for &i in named.iter() {
        if rows[i].wall_ns == 0 && rows[i].launches == 0 && rows[i].syncs == 0 {
            continue;
        }
        push_row(NAMES[i], rows[i].wall_ns, Some(&rows[i]));
    }
    // task 148: `exp_iq` は `experts` の**内側**なので合計には入れない (二重計上
    // になる)。prefill でも「GPU の起動・同期・コピー」を読めるように 1 行だけ足す。
    if rows[EXPERT].wall_ns > 0 || rows[EXPERT].launches > 0 || rows[EXPERT].syncs > 0 {
        push_row(NAMES[EXPERT], rows[EXPERT].wall_ns, Some(&rows[EXPERT]));
    }
    // 残差 (層の外 = embed / PLE / 最終 mixer / head / 未計測)。
    push_row(NAMES[OTHER], other_ns, Some(&rows[OTHER]));
    // task 154: 親ごとの副区間 (cpu = wall - launch - sync - copy の中身)。
    // 0 でないものだけ出し、`*` は「他の副区間の内側にも入る」もの (dequant は
    // gemm_cpu の内側、grms は scan の内側)。合計は親の wall を超えうる。
    let nested = [SUB_DEQUANT];
    for &p in &[MIX, ROUTER, SHARED, FULL, SSM, EXPERTS, OTHER] {
        let mut parts: Vec<String> = Vec::new();
        let mut sum = 0u64;
        for i in 0..SUB_N {
            let ns = SUBS[p][i].load(Ordering::Relaxed);
            if ns == 0 {
                continue;
            }
            sum += ns;
            let mark = if nested.contains(&i) { "*" } else { "" };
            parts.push(format!("{}{}={:.1}", SUB_NAMES[i], mark, ms(ns)));
        }
        if parts.is_empty() {
            continue;
        }
        out.push(format!(
            "[q4ledger-pf]   sub {:<7} sum={:>8.1} of {:>8.1}ms  {}",
            NAMES[p],
            ms(sum),
            ms(rows[p].wall_ns),
            parts.join(" ")
        ));
    }
    // task 167: ssm 層の中の 4 段 (qkv 投影 / beta/alpha 投影 / GDN コア / 出力投影)
    // ごとの壁時計と転送量。`sub` の行は名前だけなので、ここでは
    // 「どの段が何 MB 動かしたか」を独立の行で出す (段の和は ssm の wall 以下)。
    let steps = [SUB_PF_QKV, SUB_PF_GATE, SUB_PF_SCAN, SUB_PF_OUT];
    let any = steps
        .iter()
        .any(|&i| sub_ns(SSM, i) > 0 || sub_bytes_of(SSM, i, true).0 > 0);
    if any {
        out.push(
            "[q4ledger-pf]   ssm step       wall_ms   h2d_MB/cnt     d2h_MB/cnt".to_string(),
        );
        for &i in steps.iter() {
            let (hu, hc) = sub_bytes_of(SSM, i, true);
            let (du, dc) = sub_bytes_of(SSM, i, false);
            out.push(format!(
                "[q4ledger-pf]   ssm {:<9} {:>9.1}   {:>8.2}/{:<6} {:>8.2}/{:<6}",
                SUB_NAMES[i],
                ms(sub_ns(SSM, i)),
                hu as f64 / 1e6,
                hc,
                du as f64 / 1e6,
                dc,
            ));
        }
    }
    // task 254: 突き合わせ。wall = GPU の実時間 + 露出したホスト時間 (+ 層の外)
    // が 1 s 以内で閉じるか。`gpu_busy` は起動ペアのイベントの和 (アイドルを
    // 含まない)、`exposed_host` は区間ごとの `max(0, wall - busy)` の和。
    if let Some(b) = busy.as_ref() {
        let mut gpu_sum = 0.0f64;
        let mut exposed = 0.0f64;
        for &i in named.iter().chain([OTHER].iter()) {
            let w = if i == OTHER {
                ms(other_ns)
            } else {
                ms(rows[i].wall_ns)
            };
            let g = b.get(i).copied().unwrap_or(0.0);
            gpu_sum += g;
            exposed += (w - g).max(0.0);
        }
        let wall_ms = ms(total_wall);
        let sum = gpu_sum + exposed;
        out.push(format!(
            "[q4ledger-pf] close wall={wall_ms:.1}ms gpu_busy={gpu_sum:.1}ms \
             exposed_host={exposed:.1}ms sum={sum:.1}ms delta={:.1}ms",
            wall_ms - sum
        ));
        // `other` (残差) の中身: 名前付きでない区間の wall。`qprof` などは
        // 他の区間の内側にも入るので、和は `other` を超えうる。
        let mut rest: Vec<(usize, u64)> = (0..N)
            .filter(|&i| !named.contains(&i) && i != EXPERT && i != OTHER && rows[i].wall_ns > 0)
            .map(|i| (i, rows[i].wall_ns))
            .collect();
        rest.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        if !rest.is_empty() {
            let parts: Vec<String> = rest
                .iter()
                .map(|&(i, v)| format!("{}={:.1}", NAMES[i], ms(v)))
                .collect();
            out.push(format!(
                "[q4ledger-pf]   rest (other の中身、qprof 等は入れ子) {}",
                parts.join(" ")
            ));
        }
        pf_span_clear();
    }
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
    #[cfg(feature = "cuda")]
    crate::cuda::tline::reset();
    for row in SUBS.iter() {
        for c in row.iter() {
            c.store(0, Ordering::Relaxed);
        }
    }
    for grid in [&SUB_H2D_BYTES, &SUB_D2H_BYTES, &SUB_H2D_CALLS, &SUB_D2H_CALLS] {
        for row in grid.iter() {
            for c in row.iter() {
                c.store(0, Ordering::Relaxed);
            }
        }
    }
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
        assert!(line(8, 0).is_empty(), "off でレポートが出た");
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
        let lines = line(2, 3_000_000);
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

    /// task 167: 副区間の中の転送は、その副区間にも積まれる (親の行だけだと
    /// 「ssm の 105 GB のうちどの段が何 MB か」が読めない)。副区間の外の転送は
    /// 副区間に積まない。ネストは内側にだけ積む。
    #[test]
    fn sub_scope_counts_copies() {
        let _g = lock();
        set_for_test(true);
        reset();
        let _l = enter(SSM);
        note_h2d(1000, stamp());
        note_d2h(400, stamp());
        {
            let _s = sub(SUB_PF_QKV);
            note_h2d(7000, stamp());
            note_d2h(3000, stamp());
            {
                let _i = sub(SUB_PF_OUT);
                note_d2h(11, stamp());
            }
            note_d2h(13, stamp());
        }
        note_d2h(21, stamp());
        assert_eq!(sub_bytes_of(SSM, SUB_PF_QKV, true), (7000, 1));
        assert_eq!(sub_bytes_of(SSM, SUB_PF_QKV, false), (3013, 2));
        assert_eq!(sub_bytes_of(SSM, SUB_PF_OUT, false), (11, 1));
        assert_eq!(sub_bytes_of(SSM, SUB_PF_SCAN, true), (0, 0));
        // 親の行は副区間に関係なく全部を数える (従来どおり)。
        let r = row(SSM);
        assert_eq!(r.h2d_bytes, 8000);
        assert_eq!(r.d2h_bytes, 400 + 3000 + 11 + 13 + 21);
        reset();
        assert_eq!(sub_bytes_of(SSM, SUB_PF_QKV, true), (0, 0), "reset で消えない");
        set_for_test(false);
    }

    /// task 187: decode の `line` も副区間の行を出す (cpu 列の中身を名前で割る)。
    ///
    /// 親は `experts` / `shared` の 2 本。0 の副区間は出さない。
    #[test]
    fn decode_line_prints_expert_and_shared_subs() {
        let _g = lock();
        set_for_test(true);
        reset();
        {
            let _e = enter(EXPERTS);
            {
                let _s = sub(SUB_EXP_LOOKUP);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            {
                let _s = sub(SUB_EXP_SUM);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        {
            let _h = enter(SHARED);
            let _s = sub(SUB_SH_WAIT);
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let lines = line(2, 20_000_000);
        let exp = lines
            .iter()
            .find(|l| l.contains("sub experts"))
            .expect("experts の副区間の行が無い");
        assert!(exp.contains("exp_lookup="), "{exp}");
        assert!(exp.contains("exp_sum="), "{exp}");
        let sh = lines
            .iter()
            .find(|l| l.contains("sub shared"))
            .expect("shared の副区間の行が無い");
        assert!(sh.contains("sh_wait="), "{sh}");
        // 積んでいない副区間は出ない (列が増えない)。
        assert!(!exp.contains("exp_miss="), "{exp}");
        reset();
        set_for_test(false);
    }

    /// task 154: 副区間は親ごとに積まれ、`line_prefill` が親ごとに 1 行出す。
    /// off では 1 ns も積まない (既定 off の契約)。
    #[test]
    fn sub_buckets_are_per_parent_and_printed() {
        let _g = lock();
        set_for_test(true);
        reset();
        {
            let _m = enter(MIX);
            {
                let _s = sub(SUB_NORM);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            {
                let _s = sub(SUB_ELEM);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        {
            let _o = enter(OTHER);
            let _s = sub(SUB_ELEM);
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        assert!(sub_ns(MIX, SUB_NORM) > 0, "mix/norm が積まれていない");
        assert!(sub_ns(MIX, SUB_ELEM) > 0, "mix/elem が積まれていない");
        assert!(sub_ns(OTHER, SUB_ELEM) > 0, "other/elem が積まれていない");
        assert_eq!(sub_ns(OTHER, SUB_NORM), 0, "親が混ざっている");
        let lines = line_prefill(512);
        assert!(
            lines.iter().any(|l| l.contains("sub mix") && l.contains("norm=")),
            "mix の副区間の行が無い: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("sub other") && l.contains("elem=")),
            "other の副区間の行が無い"
        );
        reset();
        assert_eq!(sub_ns(MIX, SUB_NORM), 0, "reset で消えない");
        // off では積まない。
        set_for_test(false);
        {
            let _m = enter(MIX);
            let _s = sub(SUB_NORM);
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        assert_eq!(sub_ns(MIX, SUB_NORM), 0, "off なのに積まれた");
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
