//! 計測 (q4prof / mix_kprof / attn_kprof / gdn_gpu_win / headprof / pf_ple_kprof)。
//!
//! task 668: `src/model/qwen4exp.rs` から純粋に切り出した (ロジック変更なし)。

use super::*;

/// task 195: PLE 注入の段ごとの内訳 (`DENNO_PF_PLE_KPROF=1`・既定 off)。
///
/// 台帳の `sub other ple` は「PLE 注入ぜんぶ」の 1 本なので、射影 (GPU f32 GEMM) と
/// トークンごとの 4 パス (dot / gated / rms+conv / out) のどちらが重いかが
/// 分からなかった (REPORT_P189 の表 2 は表の読みだけを割っている)。
/// ここでは各段の**ホスト時計**を積んで、チャンクごとの累積を 1 行で出す。
/// 有効化しても数値経路は 1 命令も変わらない (計測だけ)。
pub mod pf_ple_kprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// 段 (この順で出す)。
    pub const FLAT: usize = 0;
    pub const GEMM: usize = 1;
    pub const TOK: usize = 2;
    pub const RMS: usize = 3;
    pub const DOT: usize = 4;
    pub const GATED: usize = 5;
    pub const RMSC: usize = 6;
    pub const CONV: usize = 7;
    pub const OUT: usize = 8;
    pub const HIST: usize = 9;
    pub const ALL: usize = 10;
    pub const N: usize = 11;
    pub const NAMES: [&str; N] = [
        "flat", "gemm", "tok", "rms", "dot", "gated", "rmsc", "conv", "out", "hist", "all",
    ];

    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CNT: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);

    /// 計測するか (`DENNO_PF_PLE_KPROF` があるとき)。
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_PF_PLE_KPROF").is_some())
    }

    /// 段の入口 (off なら `None` を返すだけ)。
    pub fn t0() -> Option<std::time::Instant> {
        on().then(std::time::Instant::now)
    }

    /// 段の経過を積む。
    pub fn add(i: usize, t: Option<std::time::Instant>) {
        let Some(t) = t else { return };
        if i >= N {
            return;
        }
        NS[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// チャンク 1 回を数える。
    pub fn note_call(rows: usize) {
        if !on() {
            return;
        }
        CNT.fetch_add(1, Ordering::Relaxed);
        ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// 累積を 1 行で出す (on でなければ何もしない)。
    pub fn emit() {
        if !on() {
            return;
        }
        let calls = CNT.load(Ordering::Relaxed);
        if calls == 0 {
            return;
        }
        let mut s = format!(
            "[pf-ple-kprof] calls={calls} rows={} (合計 ms / 1 トークン平均 us)",
            ROWS.load(Ordering::Relaxed)
        );
        let rows = ROWS.load(Ordering::Relaxed).max(1);
        for i in 0..N {
            let ns = NS[i].load(Ordering::Relaxed);
            s.push_str(&format!(
                " | {} {:.1}/{:.2}",
                NAMES[i],
                ns as f64 / 1e6,
                ns as f64 / 1e3 / rows as f64
            ));
        }
        eprintln!("{s}");
    }

    /// 積算を 0 に戻す (テスト用)。
    pub fn reset() {
        for i in 0..N {
            NS[i].store(0, Ordering::Relaxed);
        }
        CNT.store(0, Ordering::Relaxed);
        ROWS.store(0, Ordering::Relaxed);
    }
}

/// 層内フェーズ別の累積時間 (ns)。`DENNO_Q4_PROFILE=1` のときだけ加算する。
///
/// 27B の `DENNO_PROFILE` はこのモデルには配線されていないので、Flash-Next 用に
/// 独立した軽量なカウンタを置く。GEMV の中ではなく層の区切りで測るため、
/// 有効化しても各フェーズの内訳は見えるが内側の粒度は無い。
pub mod q4prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static PLE: AtomicU64 = AtomicU64::new(0);
    pub static MIX: AtomicU64 = AtomicU64::new(0);
    pub static ATTN: AtomicU64 = AtomicU64::new(0);
    pub static MOE: AtomicU64 = AtomicU64::new(0);
    pub static N_LAYER: AtomicU64 = AtomicU64::new(0);
    /// attention の内訳 (SSM 層: qkv 投影 / gate 投影 / GDN 本体 / 出力投影、
    /// full attention 層: まとめて FULL)。
    pub static A_QKV: AtomicU64 = AtomicU64::new(0);
    pub static A_GATE: AtomicU64 = AtomicU64::new(0);
    pub static A_GDN: AtomicU64 = AtomicU64::new(0);
    pub static A_OUT: AtomicU64 = AtomicU64::new(0);
    pub static A_FULL: AtomicU64 = AtomicU64::new(0);

    // ---- decode の「層の外」内訳 (task 132) --------------------------------
    //
    // decode の壁時計 (serve が測る) から層実行 (D_LAYER) を引いた残りを、
    // それぞれ実際に取ったタイマーで割る。合計から逆算した値を区間として
    // 出さない (差は `unexplained` の 1 列だけ)。

    /// 区間: embedding の行 gather (`Qwen4Exp::forward_embed`)。
    pub const D_EMBED: usize = 0;
    /// 区間: PLE の n-gram 索引 → 表の pread (`Qwen4Exp::gather_ple`)。
    pub const D_PLE: usize = 1;
    /// 区間: 層ループの wall (中の mix/attn/moe は [`MIX`] / [`ATTN`] / [`MOE`])。
    pub const D_LAYER: usize = 2;
    /// 区間: head の mixer (最終 norm 相当、`hc_mix_quant`)。
    pub const D_FINAL: usize = 3;
    /// 区間: `output` の GEMV。
    pub const D_HEAD: usize = 4;
    /// 区間: decode のサンプリング (`Sampler::sample`)。
    pub const D_SAMPLE: usize = 5;
    /// 区間: detokenize (`tok.decode`)。
    pub const D_DETOK: usize = 6;
    /// 区間: SSE への書き出し (`emit` の呼び出し)。
    pub const D_EMIT: usize = 7;
    /// 区間: step の入口 (`tokens.push` と窓の `ctx` の組み立て)。
    pub const D_STEP: usize = 8;
    /// 区間: 構造テレメトリの送出 (`arch_emit_token`)。
    pub const D_ARCH: usize = 9;
    /// 区間: テレメトリのフレーム送出 (Step / Token / KvUsage)。
    pub const D_TELEM: usize = 10;
    /// 区間: 選ぶ前の分布の top-k (テレメトリ用、`topk_softmax`)。
    pub const D_TOPK: usize = 11;
    /// 区間の数。
    pub const D_NSLOT: usize = 12;
    /// 区間の名前 (`[q4prof] decode outside` の列名、[`D_EMBED`] の順)。
    pub const D_NAMES: [&str; D_NSLOT] = [
        "embed",
        "ple",
        "layer",
        "final_norm",
        "head",
        "sample",
        "detok",
        "emit",
        "step",
        "arch",
        "telem",
        "topk",
    ];

    static D_NS: [AtomicU64; D_NSLOT] = [const { AtomicU64::new(0) }; D_NSLOT];

    fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_Q4_PROFILE").is_some())
    }

    /// `DENNO_PF_ATT_DBG=1` で attention の層別内訳を取る (既定 off)。
    pub fn att_dbg() -> bool {
        static D: OnceLock<bool> = OnceLock::new();
        if *D.get_or_init(|| std::env::var_os("DENNO_PF_ATT_DBG").is_some()) {
            return true;
        }
        #[cfg(test)]
        {
            return FORCE_ATT_DBG.load(Ordering::Relaxed);
        }
        #[allow(unreachable_code)]
        false
    }

    /// テストが env を立てずに層別内訳の経路を通すための強制フラグ
    /// (env は OnceLock なので プロセス内で 1 回しか読めない)。
    #[cfg(test)]
    static FORCE_ATT_DBG: AtomicBool = AtomicBool::new(false);

    /// テストが env を立てずに [`enabled`] の経路を通すための強制フラグ
    /// (task 132 の層の外内訳を --lib で発火させる)。
    #[cfg(test)]
    static FORCE_ON: AtomicBool = AtomicBool::new(false);

    /// 計測をテストから強制する (`false` で env の解釈に戻る)。
    #[cfg(test)]
    pub fn set_for_test(v: bool) {
        FORCE_ON.store(v, Ordering::Relaxed);
    }

    /// 層別内訳をテストから強制する (`None` で env の値に戻す)。
    #[cfg(test)]
    pub fn set_att_dbg_for_test(v: bool) {
        FORCE_ATT_DBG.store(v, Ordering::Relaxed);
    }

    /// 時間を取るか (Q4_PROFILE か PF_ATT_DBG のどちらかが有効)。
    #[inline]
    pub fn timed() -> bool {
        // テストの強制フラグを立てた後に古い値を掴まないようキャッシュしない
        // (`enabled` / `att_dbg` はどちらも OnceLock + atomic load 1 回)。
        enabled() || att_dbg()
    }

    // ---- 層別内訳 (DENNO_PF_ATT_DBG) -------------------------------------
    //
    // prefill の層 0 / 23 / 47 だけを段別に積む。1 層につき 1 行で出すので、
    // 「どの層のどの段が重いか」が実機のログだけで読める。既定 off。

    /// 内訳を取る層 (48 層の 0 = ssm / 23 = ssm / 47 = full attention)。
    pub const DBG_LAYERS: [usize; 3] = [0, 23, 47];
    /// 段の数 (下の ST_* の個数)。
    pub const NSTAGE: usize = 7;
    /// 段: qkv 射影 (full は q/k/v、ssm は qkv の 1 本)
    pub const ST_QKV: usize = 0;
    /// 段: ssm の gate / beta / alpha 射影
    pub const ST_GATE: usize = 1;
    /// 段: ssm の GDN 本体 (conv + scan + gated norm)
    pub const ST_GDN: usize = 2;
    /// 段: full の qk-norm + rope + KV push
    pub const ST_ROPE: usize = 3;
    /// 段: full の attend (スコア + softmax + PV)
    pub const ST_ATTEND: usize = 4;
    /// 段: full の出力ゲート
    pub const ST_GMUL: usize = 5;
    /// 段: 出力射影 (full / ssm 共通)
    pub const ST_OUT: usize = 6;

    static LAYER_NS: [AtomicU64; 3 * NSTAGE] = [const { AtomicU64::new(0) }; 3 * NSTAGE];
    static LAYER_NT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    fn dbg_slot(layer: usize) -> Option<usize> {
        DBG_LAYERS.iter().position(|&l| l == layer)
    }

    /// 計測が要るときだけ `Instant` を取る (既定 off では `None` = 呼び出し側は no-op)。
    #[inline]
    pub fn stamp() -> Option<std::time::Instant> {
        if timed() {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// 区間 `t` をグローバル束 `c` と層別束の両方へ足す。
    #[inline]
    pub fn mark(t: Option<std::time::Instant>, c: &AtomicU64, layer: usize, stage: usize) {
        if let Some(t) = t {
            let ns = t.elapsed().as_nanos() as u64;
            if enabled() {
                c.fetch_add(ns, Ordering::Relaxed);
            }
            mark_l_ns(layer, stage, ns);
        }
    }

    /// 区間 `t` を層別束だけへ足す (グローバル束を持たない段用)。
    #[inline]
    pub fn mark_l(t: Option<std::time::Instant>, layer: usize, stage: usize) {
        if let Some(t) = t {
            mark_l_ns(layer, stage, t.elapsed().as_nanos() as u64);
        }
    }

    /// task 167: 外で測った ns をグローバル束と層別束の両方へ足す
    /// (デバイス連鎖は 1 呼び出しの中で段を測るので、`Instant` の外側では
    /// 段の内訳が作れない)。
    #[inline]
    pub fn add_ns(c: &AtomicU64, layer: usize, stage: usize, ns: u64) {
        if !timed() {
            return;
        }
        if enabled() {
            c.fetch_add(ns, Ordering::Relaxed);
        }
        mark_l_ns(layer, stage, ns);
    }

    #[inline]
    fn mark_l_ns(layer: usize, stage: usize, ns: u64) {
        if !att_dbg() {
            return;
        }
        if let Some(slot) = dbg_slot(layer) {
            LAYER_NS[slot * NSTAGE + stage].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// 層別内訳の分母 (その層を回したトークン数)。
    #[inline]
    pub fn bump_nt(layer: usize, nt: usize) {
        if !att_dbg() {
            return;
        }
        if let Some(slot) = dbg_slot(layer) {
            LAYER_NT[slot].fetch_add(nt as u64, Ordering::Relaxed);
        }
    }

    /// 層別内訳の 1 行 (積んでいない層は `None`)。
    ///
    /// `is_full` はその層が full attention かどうか (段名の出し分けに使う)。
    pub fn layer_line(layer: usize, is_full: bool) -> Option<String> {
        let slot = dbg_slot(layer)?;
        let nt = LAYER_NT[slot].load(Ordering::Relaxed);
        if nt == 0 {
            return None;
        }
        let ms = |stage: usize| LAYER_NS[slot * NSTAGE + stage].load(Ordering::Relaxed) as f64 / 1e6;
        let per = |stage: usize| {
            let v = ms(stage);
            if nt == 0 {
                0.0
            } else {
                v * 1e3 / nt as f64
            }
        };
        let kind = if is_full { "full" } else { "ssm " };
        if is_full {
            Some(format!(
                "[pf-att] layer {layer} ({kind}) qkv {:.2} / rope {:.2} / attend {:.2} / gmul {:.2} / out {:.2} ms \
                 (nt {nt}, per-token {:.3} / {:.3} / {:.3} / {:.3} / {:.3} ms)",
                ms(ST_QKV),
                ms(ST_ROPE),
                ms(ST_ATTEND),
                ms(ST_GMUL),
                ms(ST_OUT),
                per(ST_QKV),
                per(ST_ROPE),
                per(ST_ATTEND),
                per(ST_GMUL),
                per(ST_OUT),
            ))
        } else {
            Some(format!(
                "[pf-att] layer {layer} ({kind}) qkv {:.2} / gate {:.2} / gdn {:.2} / out {:.2} ms \
                 (nt {nt}, per-token {:.3} / {:.3} / {:.3} / {:.3} ms)",
                ms(ST_QKV),
                ms(ST_GATE),
                ms(ST_GDN),
                ms(ST_OUT),
                per(ST_QKV),
                per(ST_GATE),
                per(ST_GDN),
                per(ST_OUT),
            ))
        }
    }

    #[inline]
    pub fn enabled() -> bool {
        #[cfg(test)]
        if FORCE_ON.load(Ordering::Relaxed) {
            return true;
        }
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(on)
    }

    #[inline]
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        if enabled() {
            c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// (ple, mix, attn, moe, 層数) をミリ秒で返す。
    pub fn dump() -> (f64, f64, f64, f64, u64) {
        let f = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        (
            f(&PLE),
            f(&MIX),
            f(&ATTN),
            f(&MOE),
            N_LAYER.load(Ordering::Relaxed),
        )
    }

    /// attention の内訳をミリ秒で返す (qkv, gate, gdn, out, full)。
    pub fn dump_attn() -> (f64, f64, f64, f64, f64) {
        let f = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        (f(&A_QKV), f(&A_GATE), f(&A_GDN), f(&A_OUT), f(&A_FULL))
    }

    /// task 133: full attention の GPU 実行内訳 (`cuda::q4attn_kprof` の 1 行)。
    /// cuda feature 無しのビルドは常に `None` (GPU 経路が無い)。
    #[cfg(feature = "cuda")]
    pub fn attn_gpu_line() -> Option<String> {
        crate::cuda::q4attn_kprof::line()
    }
    #[cfg(not(feature = "cuda"))]
    pub fn attn_gpu_line() -> Option<String> {
        None
    }

    /// task 133: [`attn_gpu_line`] の累計を 0 に戻す。
    pub fn attn_gpu_reset() {
        #[cfg(feature = "cuda")]
        crate::cuda::q4attn_kprof::reset();
    }

    // ---- decode の層の外 (task 132) --------------------------------------

    /// 層の外の区間用の `Instant`。計測が要るときだけ取る (既定 off では `None`)。
    #[inline]
    pub fn dstamp() -> Option<std::time::Instant> {
        if enabled() {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// 層の外の区間 `slot` に `t` からの経過を足す (`None` は no-op)。
    #[inline]
    pub fn dmark(slot: usize, t: Option<std::time::Instant>) {
        if let Some(t) = t {
            dadd_ns(slot, t.elapsed().as_nanos() as u64);
        }
    }

    /// 層の外の区間 `slot` に ns を直接足す (外で測った値を渡す用)。
    #[inline]
    pub fn dadd_ns(slot: usize, ns: u64) {
        if enabled() {
            D_NS[slot].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// 層の外の区間 `slot` の累積 (ns)。テストとレポート用。
    #[inline]
    pub fn dns(slot: usize) -> u64 {
        D_NS[slot].load(Ordering::Relaxed)
    }

    /// decode の層の外の内訳 1 行。`total_ns` は decode の壁時計 (`serve_flash`
    /// が測る t2..t_decode)。区間はすべて個別のタイマーで、合計との差だけを
    /// `unexplained` として出す。無効時は `None` (出力ゼロ)。
    pub fn decode_outside_line(tok: usize, total_ns: u64) -> Option<String> {
        if !enabled() {
            return None;
        }
        let ms = |s: usize| dns(s) as f64 / 1e6;
        let inside: u64 = D_NS.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let total = total_ns as f64 / 1e6;
        let un = total - inside as f64 / 1e6;
        let d = tok.max(1) as f64;
        let mut s = format!(
            "[q4prof] decode outside tok={tok} total={total:.1}ms layer={:.1}",
            ms(D_LAYER)
        );
        for i in 0..D_NSLOT {
            if i == D_LAYER {
                continue;
            }
            s.push_str(&format!(" | {} {:.1}", D_NAMES[i], ms(i)));
        }
        s.push_str(&format!(
            " | unexplained {un:.1}ms ({:.3} ms/tok, layer {:.3} ms/tok)",
            un / d,
            ms(D_LAYER) / d
        ));
        Some(s)
    }

    pub fn reset() {
        for c in [&PLE, &MIX, &ATTN, &MOE, &A_QKV, &A_GATE, &A_GDN, &A_OUT, &A_FULL] {
            c.store(0, Ordering::Relaxed);
        }
        N_LAYER.store(0, Ordering::Relaxed);
        for c in D_NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        for c in LAYER_NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        for c in LAYER_NT.iter() {
            c.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// グローバルカウンタを共有するので、テストは 1 本ずつ回す
        /// (`prekprof` / `mix_kprof` と同じロック)。
        fn lock() -> std::sync::MutexGuard<'static, ()> {
            crate::model::moe_expert::moe_test_lock()
        }

        /// 無効時は Instant も取らず、レポートも出ない (既定 off の契約)。
        #[test]
        fn decode_outside_off_is_noop() {
            let _g = lock();
            set_for_test(false);
            reset();
            assert!(dstamp().is_none(), "off で Instant を取った");
            dmark(D_EMBED, Some(std::time::Instant::now()));
            dadd_ns(D_HEAD, 1_000_000);
            assert_eq!(dns(D_HEAD), 0, "off なのに積まれた");
            assert!(
                decode_outside_line(4, 10_000_000).is_none(),
                "off でレポートが出た"
            );
            reset();
        }

        /// 区間はそれぞれ独立に積まれ、合計との差だけが unexplained になる。
        /// (配線そのものは呼び出し側 — ここでは行の組み立てと無効時の
        /// no-op を固定する。)
        #[test]
        fn decode_outside_line_reports_slots_and_residual() {
            let _g = lock();
            set_for_test(true);
            reset();
            dadd_ns(D_EMBED, 1_000_000);
            dadd_ns(D_LAYER, 6_000_000);
            dadd_ns(D_SAMPLE, 500_000);
            let line = decode_outside_line(2, 10_000_000).expect("on なのにレポートが出ない");
            assert!(line.starts_with("[q4prof] decode outside tok=2"), "{line}");
            assert!(line.contains("total=10.0ms"), "{line}");
            assert!(line.contains("layer=6.0"), "{line}");
            assert!(line.contains("embed 1.0"), "{line}");
            assert!(line.contains("sample 0.5"), "{line}");
            // 10 - 7.5 = 2.5 ms が説明できない分。
            assert!(line.contains("unexplained 2.5ms"), "{line}");
            // 列が欠けない (全スロットの名前が出る)。
            for name in D_NAMES {
                assert!(line.contains(name), "{name} が行に無い: {line}");
            }
            // ms/tok はトークン数で割る (2.5 ms / 2 tok = 1.25)。
            assert!(line.contains("1.250 ms/tok"), "{line}");
            reset();
            set_for_test(false);
        }
    }
}

/// `hc_mix_quant` の内側の区間別累積時間 (ns)。`DENNO_MIX_KPROF=1` のときだけ
/// `Instant` を取り、区間ごとに atomic へ足す。既定 off では [`stamp`] が `None`
/// を返すだけで、演算順序にも確保にも触れない。
///
/// decode は 1 トークンで mix を `2 * n_layer + 1` 回呼ぶので、リクエスト末尾で
/// [`line`] を出して [`reset`] すると「1 トークンぶんの内訳」が 1 行で読める
/// (`serve_flash::prof_flash_q4` がそうする)。
pub mod mix_kprof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// 区間: grouped RMS 正規化 (入力の整形)。
    pub const GN: usize = 0;
    /// 区間: `hc_down` の GEMV (`hc_dim -> low_rank`)。
    pub const DOWN: usize = 1;
    /// 区間: `hc_lo_apply` (`1/hc` と silu)。
    pub const LO: usize = 2;
    /// 区間: `hc_up` の GEMV (`low_rank -> hc_dim`)。
    pub const UP: usize = 3;
    /// 区間: `hc_mixed` (ストリームごとの sigmoid と加算と `1/hc`)。
    pub const MIXED: usize = 4;
    /// 区間: `hc_inject` の GEMV (`hc_dim -> hc`)。
    pub const INJ: usize = 5;
    /// 区間: `hc_inject_apply` (`1/hc`)。
    pub const INJ_APPLY: usize = 6;
    /// 区間: `hc_mix_quant` 全体。上の合計との差が確保などの「その他」。
    pub const TOTAL: usize = 7;
    /// 区間の数。
    pub const N: usize = 8;

    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: AtomicU64 = AtomicU64::new(0);

    /// テストが env を立てずに区間計測を通すための強制フラグ
    /// (env は `OnceLock` なのでプロセス内で 1 回しか読めない)。
    #[cfg(test)]
    static FORCE: AtomicBool = AtomicBool::new(false);

    /// 区間計測をテストから強制する (`false` で env の解釈に戻る)。
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
        *ON.get_or_init(|| std::env::var_os("DENNO_MIX_KPROF").is_some())
    }

    /// 計測が要るときだけ `Instant` を取る (既定 off では `None` = 呼び出し側は no-op)。
    #[inline]
    pub fn stamp() -> Option<std::time::Instant> {
        if on() {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// 区間 `slot` に `t` からの経過を足す。
    #[inline]
    pub fn mark(t: Option<std::time::Instant>, slot: usize) {
        if let Some(t) = t {
            NS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// `hc_mix_quant` の呼び出し回数 (1 行の分母)。
    #[inline]
    pub fn bump_call() {
        if on() {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn ns(slot: usize) -> u64 {
        NS[slot].load(Ordering::Relaxed)
    }

    /// 区間の累積 (ns)。`reset` してから 1 トークン回すと、そのトークンの内訳になる。
    pub fn ns_of(slot: usize) -> u64 {
        ns(slot)
    }

    /// 内訳 1 行。無効時は `None` (出力ゼロ)。
    pub fn line() -> Option<String> {
        if !on() {
            return None;
        }
        let calls = CALLS.load(Ordering::Relaxed);
        let tot = ns(TOTAL).max(1);
        let pct = |slot: usize| ns(slot) as f64 * 100.0 / tot as f64;
        let staged: u64 = [GN, DOWN, LO, UP, MIXED, INJ, INJ_APPLY]
            .iter()
            .map(|&s| ns(s))
            .sum();
        let other = ns(TOTAL).saturating_sub(staged);
        let mut s = format!(
            "[mix-kprof] calls={calls} total={} ns | gn {} ns {:.1}% | down {} ns {:.1}% | \
             lo {} ns {:.1}% | up {} ns {:.1}% | mixed {} ns {:.1}% | inj {} ns {:.1}% | \
             inj_apply {} ns {:.1}% | other {} ns {:.1}%",
            ns(TOTAL),
            ns(GN),
            pct(GN),
            ns(DOWN),
            pct(DOWN),
            ns(LO),
            pct(LO),
            ns(UP),
            pct(UP),
            ns(MIXED),
            pct(MIXED),
            ns(INJ),
            pct(INJ),
            ns(INJ_APPLY),
            pct(INJ_APPLY),
            other,
            other as f64 * 100.0 / tot as f64
        );
        // P84: 行並列 region の実態 (発火の有無・チャンク数・実効並列度)。
        for i in 0..rowpar::NL {
            s.push_str(&rowpar::line_part(i));
        }
        // task 88: down / up GEMV の実行先内訳 (calls / ms / gpu / ty)。
        s.push_str(&gemv::line_part());
        Some(s)
    }

    /// 累積を 0 に戻す。
    pub fn reset() {
        for c in NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        CALLS.store(0, Ordering::Relaxed);
        rowpar::reset();
        gemv::reset();
    }

    /// P84: hc mix の行並列 region の常時カウンタ ([`on`] と同じ env ゲート)。
    ///
    /// [`crate::model::moe_expert`] の `for_each_row_chunk` が region のたびに積む。
    /// どの GEMV の region かは mix 側が thread-local のラベルで示す
    /// ([`rowpar::set_label`]、`0` は未設定 = 何も積まない)。expert GEMV など
    /// ラベルの立っていない呼び出しは混ざらない。
    ///
    /// 実機で読みたいこと (task 84):
    /// - P79b の行並列が本番で発火しているか (`fired` / `regions`)
    /// - 1 呼び出しで rayon が何チャンク (= 最大スレッド数) に割ったか
    ///   (`chunks` / `fired`)
    /// - 実効並列度 (`comp` / `wall`。1 のままなら 1 スレッドでしか回っていない)、
    ///   fork-join の固定費は 1 スレッドのときの `wall - comp` で読む
    ///
    /// task 96: wall の内訳を skew 3 項目で分離する (発火 region のみ)。
    /// - `t_first=`: region 先頭 -> 最初のチャンク完了。1 チャンクの平均時間
    ///   (`comp` / `chunks`) と等しければ全スレッドが即座に仕事を得ている。
    ///   大きく上回るなら fork/ディスパッチ待ちが実在する。
    /// - `spread=`: 最初の完了 -> 最後の完了。`t_first` + `spread` = wall。
    ///   チャンク実時間のばらつき (メモリ競合での伸び) と後発走行がここに出る。
    /// - `tstart=`: チャンク開始オフセットのチャンク平均。0 に近ければ
    ///   スレッドは仕事を待っていなかった (不足しているのは走行の重なり)。
    pub mod rowpar {
        use std::sync::atomic::{AtomicU64, Ordering};

        /// ラベル: 未設定 (何も積まない)。
        pub const NONE: u8 = 0;
        /// ラベル: `hc_down` の GEMV。
        pub const L_DOWN: u8 = 1;
        /// ラベル: `hc_up` の GEMV。
        pub const L_UP: u8 = 2;
        /// ラベルの数 (カウンタ配列の長さ)。
        pub const NL: usize = 2;
        /// 行の名前 (レポート用、[`L_DOWN`] / [`L_UP`] の順)。
        pub const NAME: [&str; NL] = ["down", "up"];

        static REGIONS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        static FIRED: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        static CHUNKS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        static WALL_NS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        static COMP_NS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        /// task 93: 発火した region で `rayon::current_num_threads()` が返した値の合計
        /// (発火 1 件あたりの平均を `th=` で出す。逐次 region は積まない)。
        /// 実機の「プール スレッド数 > 物理コア数」(SMT の取り合い) を判別する。
        static TH_SUM: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        /// task 96: 発火 region の「最初のチャンク完了オフセット」の合計
        /// (region 先頭基準。[`FIRED`] で割ると `t_first=`)。
        static FIRST_END_SUM: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        /// task 96: 発火 region の全チャンク開始オフセットの合計
        /// ([`CHUNKS`] で割ると `tstart=` = スレッドが仕事を得るまでの待ちの平均)。
        static START_SUM: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
        /// task 96: 発火 region の wall の合計 ([`FIRED`] で割った値と
        /// [`FIRST_END_SUM`] から `spread=` を出す。[`WALL_NS`] は逐次 region も
        /// 含むので発火だけの平均には使えない)。
        static FIRED_WALL_NS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];

        thread_local! {
            pub(crate) static LBL: std::cell::Cell<u8> = const { std::cell::Cell::new(NONE) };
        }

        /// このスレッドの GEMV ラベルを立てる (off のときは no-op)。
        #[inline]
        pub fn set_label(l: u8) {
            if super::on() {
                LBL.with(|c| c.set(l));
            }
        }

        /// 現在のラベル (off なら常に [`NONE`])。
        #[inline]
        pub fn label() -> u8 {
            if super::on() {
                LBL.with(|c| c.get())
            } else {
                NONE
            }
        }

        /// region を 1 件積む。`fired` は rayon で割れたか、`chunks` は分割チャンク数
        /// (逐次は 1)、`wall_ns` は region の壁時計、`comp_ns` はチャンク内の純計算
        /// (`f`) の合計 (スレッド時間の合計)。`th` は発火 region の
        /// `rayon::current_num_threads()` (逐次は 0 = 積まない)。
        ///
        /// task 96: `first_end_ns` は発火 region の「最初のチャンク完了オフセット」
        /// (region 先頭基準)、`start_sum_ns` は全チャンク開始オフセットの和。
        /// **fired のときだけ**積む (逐次呼び出し側は 0 を渡す。逐次は単一チャンクで
        /// 開始オフセット 0・完了 = wall なので skew の情報を持たない)。
        pub fn record(
            l: u8,
            fired: bool,
            chunks: usize,
            wall_ns: u64,
            comp_ns: u64,
            th: usize,
            first_end_ns: u64,
            start_sum_ns: u64,
        ) {
            let i = (l as usize).saturating_sub(1).min(NL - 1);
            REGIONS[i].fetch_add(1, Ordering::Relaxed);
            if fired {
                FIRED[i].fetch_add(1, Ordering::Relaxed);
                CHUNKS[i].fetch_add(chunks as u64, Ordering::Relaxed);
                TH_SUM[i].fetch_add(th as u64, Ordering::Relaxed);
                FIRST_END_SUM[i].fetch_add(first_end_ns, Ordering::Relaxed);
                START_SUM[i].fetch_add(start_sum_ns, Ordering::Relaxed);
                FIRED_WALL_NS[i].fetch_add(wall_ns, Ordering::Relaxed);
            }
            WALL_NS[i].fetch_add(wall_ns, Ordering::Relaxed);
            COMP_NS[i].fetch_add(comp_ns, Ordering::Relaxed);
        }

        /// (regions, fired, chunks, wall_ns, comp_ns)。テスト用。
        pub fn stats(i: usize) -> (u64, u64, u64, u64, u64) {
            let i = i.min(NL - 1);
            (
                REGIONS[i].load(Ordering::Relaxed),
                FIRED[i].load(Ordering::Relaxed),
                CHUNKS[i].load(Ordering::Relaxed),
                WALL_NS[i].load(Ordering::Relaxed),
                COMP_NS[i].load(Ordering::Relaxed),
            )
        }

        /// task 96: (first_end_sum, start_sum, fired_wall_ns)。発火 region だけの
        /// skew 累計。テスト用 (本番の表示は [`line_part`])。
        pub fn skew(i: usize) -> (u64, u64, u64) {
            let i = i.min(NL - 1);
            (
                FIRST_END_SUM[i].load(Ordering::Relaxed),
                START_SUM[i].load(Ordering::Relaxed),
                FIRED_WALL_NS[i].load(Ordering::Relaxed),
            )
        }

        /// カウンタを 0 に戻す ([`super::reset`] から呼ぶ)。
        pub fn reset() {
            for i in 0..NL {
                REGIONS[i].store(0, Ordering::Relaxed);
                FIRED[i].store(0, Ordering::Relaxed);
                CHUNKS[i].store(0, Ordering::Relaxed);
                WALL_NS[i].store(0, Ordering::Relaxed);
                COMP_NS[i].store(0, Ordering::Relaxed);
                TH_SUM[i].store(0, Ordering::Relaxed);
                FIRST_END_SUM[i].store(0, Ordering::Relaxed);
                START_SUM[i].store(0, Ordering::Relaxed);
                FIRED_WALL_NS[i].store(0, Ordering::Relaxed);
            }
        }

        /// 1 ラベルぶんの追記 (`[mix-kprof]` 行の末尾に付ける)。
        pub fn line_part(i: usize) -> String {
            let (regions, fired, chunks, wall_ns, comp_ns) = stats(i);
            let name = NAME[i];
            if regions == 0 {
                return format!(" | {name} regions=0");
            }
            let per = |v: u64| v as f64 / regions as f64 / 1e3;
            let par_eff = if wall_ns > 0 {
                comp_ns as f64 / wall_ns as f64
            } else {
                1.0
            };
            // task 93: th = 発火 region の rayon スレッド数の平均 (逐次 region は積まない)。
            // chunks と並べて見ると「プール スレッド数 > chunks (物理コア数)」の
            // SMT の取り合いが読める。
            let th = if fired > 0 {
                TH_SUM[i].load(Ordering::Relaxed) as f64 / fired as f64
            } else {
                0.0
            };
            let mut s = format!(
                " | {name} regions={regions} fired={fired} chunks={:.1} th={th:.1} wall={:.0}us \
                 comp={:.0}us par_eff={par_eff:.1}",
                chunks as f64 / regions as f64,
                per(wall_ns),
                per(comp_ns),
            );
            // task 96: wall の内訳 (発火 region のみ。逐次は単一チャンクなので
            // skew を持たない)。
            // - t_first: region 先頭 -> 最初のチャンク完了。comp/chunks (1 チャンクの
            //   平均時間) と比べると fork/ディスパッチ待ちが読める (等しければ
            //   全スレッドが即座に仕事を得て並列走行している)。
            // - spread: 最初の完了 -> 最後の完了 (= wall)。完了の散らばり。
            //   t_first が理想に近く spread が大きいときは、チャンクは即時開始だが
            //   一部が長く走っている (メモリ競合での伸び or 後発走行)。
            // - tstart: チャンク開始オフセットのチャンク平均。スレッドが仕事を
            //   得るまでの待ちの平均 (0 に近ければ待ち無し)。
            if fired > 0 {
                let fired_f = fired as f64;
                let t_first_us = FIRST_END_SUM[i].load(Ordering::Relaxed) as f64 / fired_f / 1e3;
                let fired_wall_us = FIRED_WALL_NS[i].load(Ordering::Relaxed) as f64 / fired_f / 1e3;
                let spread_us = (fired_wall_us - t_first_us).max(0.0);
                let chunks_total = CHUNKS[i].load(Ordering::Relaxed);
                let tstart_us = if chunks_total > 0 {
                    START_SUM[i].load(Ordering::Relaxed) as f64 / chunks_total as f64 / 1e3
                } else {
                    0.0
                };
                s.push_str(&format!(
                    " t_first={t_first_us:.0}us spread={spread_us:.0}us tstart={tstart_us:.0}us"
                ));
            }
            s
        }
    }

    /// task 88: down / up GEMV 1 発ぶんの実行先内訳 ([`attn_kprof`] と同じ形式、
    /// `[mix-kprof]` の 1 行に統合)。
    ///
    /// hc down/up はモデルごとに量子化型が違う (本番 Swift-1.5 IQ3_S の GGUF
    /// ヘッダ実読では Q8_0 でない型。P84/P86 の 2 行カーネルは Q8_0 専門の門番を
    /// 持つので一度も発火しなかった = regions=0 の最終的な正体)。
    /// 実機で次のことを区別するために積む:
    /// - `calls` / `ms`: down と up の GEMV 1 発ずつの発数と時間 (区間計測
    ///   [`DOWN`] / [`UP`] と同じ `Instant` を再利用するので追加の取得は不要)
    /// - `gpu`: GEMV 直後の [`crate::cuda::last_linear_ran_on_gpu`] を数える
    ///   (cuda 無しのビルドでは常に 0)。decode 経路は P86 の two_row バイパスで
    ///   GPU を通らないはずなので、gpu が積まれていたら別経路 (classic / batch)
    ///   が GPU に落ちている印になる
    /// - `ty`: 呼び出し側が渡した [`GgmlType`] の id (ggml.h の enum 値)。
    ///   型が混ざるプロセスでは最後に積んだ 1 個 (hc down/up はプロセスで 1 種)
    ///
    /// task 96: `inj` ([`hc_mix_quant_fast`] / [`hc_mix_quant_ref`] の inject GEMV,
    /// k=hc_dim / n=hc の小行列) も同じ形式で積む。hc テンソルの目録上の型
    /// (`hc_attn_inject.weight` は F32) は down/up (目録 Q8_0、実機 BF16) で
    /// 外れていた実績があるので、実型と融合カーネルの要否を実機で判定するための
    /// 出力。BF16 なら [`crate::model::moe_expert::expert_gemv_bf16`] (task 92 の
    /// 融合カーネル) に今日の経路のまま乗る (`gemv_label` -> `gemv_impl_par` の
    /// BF16 アーム)。`DENNO_BF16_FAST=0` で ms が動けば融合カーネルだった証拠。
    pub mod gemv {
        use std::sync::atomic::{AtomicU64, Ordering};

        static DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
        static DOWN_NS: AtomicU64 = AtomicU64::new(0);
        static DOWN_GPU: AtomicU64 = AtomicU64::new(0);
        static UP_CALLS: AtomicU64 = AtomicU64::new(0);
        static UP_NS: AtomicU64 = AtomicU64::new(0);
        static UP_GPU: AtomicU64 = AtomicU64::new(0);
        static INJ_CALLS: AtomicU64 = AtomicU64::new(0);
        static INJ_NS: AtomicU64 = AtomicU64::new(0);
        static INJ_GPU: AtomicU64 = AtomicU64::new(0);
        /// 最後に積んだ型の id ([`GgmlType::from_id`] で戻す。0 は「未設定」を兼ねる。
        /// ggml には id 0 = F32 があるので、未設定の表示は `from_id` 前の 0 チェック)。
        static DOWN_TY_ID: AtomicU64 = AtomicU64::new(0);
        static UP_TY_ID: AtomicU64 = AtomicU64::new(0);
        static INJ_TY_ID: AtomicU64 = AtomicU64::new(0);

        /// 直前の GEMV が GPU で走りきったか (cuda 無しのビルドは常に false)。
        #[cfg(feature = "cuda")]
        fn ran_on_gpu() -> bool {
            crate::cuda::last_linear_ran_on_gpu()
        }
        #[cfg(not(feature = "cuda"))]
        fn ran_on_gpu() -> bool {
            false
        }

        fn ty_name(id: &AtomicU64) -> &'static str {
            let v = id.load(Ordering::Relaxed);
            // 0 は未積算 (note がまだ一度も走っていないときの初期値)。
            // hc down/up は量子化型のテンソルなので F32 (id 0) との区別はしない。
            if v == 0 {
                return "-";
            }
            crate::kquant::GgmlType::from_id(v as u32)
                .map(|t| t.name())
                .unwrap_or("-")
        }

        /// `hc_down` の GEMV 1 発。`t` は GEMV 直前に取った `Instant` の
        /// [`super::stamp`] の戻り値 (区間計測 [`super::DOWN`] と共有する。off の
        /// ときは呼び出し側が `None` を持つので note も no-op)。
        #[inline]
        pub fn note_down(t: Option<std::time::Instant>, ty: crate::kquant::GgmlType) {
            if !super::on() {
                return;
            }
            let Some(t) = t else { return };
            DOWN_CALLS.fetch_add(1, Ordering::Relaxed);
            DOWN_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if ran_on_gpu() {
                DOWN_GPU.fetch_add(1, Ordering::Relaxed);
            }
            DOWN_TY_ID.store(ty.id() as u64, Ordering::Relaxed);
        }

        /// `hc_up` の GEMV 1 発 ([`note_down`] と同じ規約)。
        #[inline]
        pub fn note_up(t: Option<std::time::Instant>, ty: crate::kquant::GgmlType) {
            if !super::on() {
                return;
            }
            let Some(t) = t else { return };
            UP_CALLS.fetch_add(1, Ordering::Relaxed);
            UP_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if ran_on_gpu() {
                UP_GPU.fetch_add(1, Ordering::Relaxed);
            }
            UP_TY_ID.store(ty.id() as u64, Ordering::Relaxed);
        }

        /// `hc_inject` の GEMV 1 発 (task 96。[`note_down`] と同じ規約)。
        #[inline]
        pub fn note_inj(t: Option<std::time::Instant>, ty: crate::kquant::GgmlType) {
            if !super::on() {
                return;
            }
            let Some(t) = t else { return };
            INJ_CALLS.fetch_add(1, Ordering::Relaxed);
            INJ_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if ran_on_gpu() {
                INJ_GPU.fetch_add(1, Ordering::Relaxed);
            }
            INJ_TY_ID.store(ty.id() as u64, Ordering::Relaxed);
        }

        fn slot(
            calls: &AtomicU64,
            ns: &AtomicU64,
            gpu: &AtomicU64,
            ty_id: &AtomicU64,
            name: &str,
            s: &mut String,
        ) {
            let c = calls.load(Ordering::Relaxed);
            let n = ns.load(Ordering::Relaxed);
            let g = gpu.load(Ordering::Relaxed);
            let pct = if c == 0 {
                0.0
            } else {
                g as f64 * 100.0 / c as f64
            };
            let tn = ty_name(ty_id);
            s.push_str(&format!(
                " | gemv {name} calls={c} ms={:.1} gpu={g} ({pct:.1}%) ty={tn}",
                n as f64 / 1e6
            ));
        }

        /// `[mix-kprof]` 行の末尾に付ける 3 区間ぶん (down / up / inj)。
        pub fn line_part() -> String {
            let mut s = String::new();
            slot(&DOWN_CALLS, &DOWN_NS, &DOWN_GPU, &DOWN_TY_ID, "down", &mut s);
            slot(&UP_CALLS, &UP_NS, &UP_GPU, &UP_TY_ID, "up", &mut s);
            slot(&INJ_CALLS, &INJ_NS, &INJ_GPU, &INJ_TY_ID, "inj", &mut s);
            s
        }

        /// カウンタを 0 に戻す ([`super::reset`] から呼ぶ)。
        pub fn reset() {
            for c in [
                &DOWN_CALLS,
                &DOWN_NS,
                &DOWN_GPU,
                &UP_CALLS,
                &UP_NS,
                &UP_GPU,
                &INJ_CALLS,
                &INJ_NS,
                &INJ_GPU,
            ] {
                c.store(0, Ordering::Relaxed);
            }
            DOWN_TY_ID.store(0, Ordering::Relaxed);
            UP_TY_ID.store(0, Ordering::Relaxed);
            INJ_TY_ID.store(0, Ordering::Relaxed);
        }
    }
}

/// task 87: decode の ssm 層 attention 投影 (qkv / gate / out GEMV) の経路内訳
/// 1 行カウンタ。`DENNO_ATTN_KPROF=1` のときだけ積む (既定 off は分岐 1 回・
/// `Instant` も追加で取らない — 呼び出し側が既存の q4prof 用に作った `Instant`
/// をそのまま渡す)。
///
/// 実測 (`[q4prof] ... attn qkv=616.4 gate=118.3 gdn=288.7 out=362.9 full=0.0`,
/// 検証機, 1116 in / 48 out) の qkv / gate / out は
/// [`crate::model::qwen4exp_ssm::SsmLayer::forward_token`] の 3 区間の合計
/// (ssm 層 36 × decode トークン数) で、1 呼び出し = qkv+z 連結 GEMV 1 発 /
/// beta+alpha 連結 GEMV 1 発 / ssm_out GEMV 1 発。区間内は GEMV 1 発のみなので、
/// GEMV 直後に [`crate::cuda::last_linear_ran_on_gpu`] を読めば
/// **その 1 発の実行先** (GPU intercept か CPU) が取る (GPU 経路は同期 GEMV =
/// 帰ってきた時点で計算済み)。cuda feature 無しのビルドでは常に CPU 扱い。
///
/// full 層 12 本は task 132 から `A_FULL` (decode も計上) が持つので、ここには
/// 積まない — qkv / gate / gdn / out + full が実測行の attn 合計になる。
///
/// 出力は `serve_flash::prof_flash_q4` が prefill / decode の区切りで 1 行出して
/// [`reset`] する (`mix_kprof` と同じ規約・`DENNO_Q4_PROFILE` とは独立に
/// `DENNO_ATTN_KPROF=1` だけで読める)。
///
/// ```text
/// [attn-kprof] qkv calls=1728 ms=616.4 gpu=1728 (100.0%) | gate calls=1728 ms=118.3 gpu=0 (0.0%) | out calls=1728 ms=362.9 gpu=1728 (100.0%)
/// ```
pub mod attn_kprof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;

    static QKV_CALLS: AtomicU64 = AtomicU64::new(0);
    static QKV_NS: AtomicU64 = AtomicU64::new(0);
    static QKV_GPU: AtomicU64 = AtomicU64::new(0);
    static GATE_CALLS: AtomicU64 = AtomicU64::new(0);
    static GATE_NS: AtomicU64 = AtomicU64::new(0);
    static GATE_GPU: AtomicU64 = AtomicU64::new(0);
    static OUT_CALLS: AtomicU64 = AtomicU64::new(0);
    static OUT_NS: AtomicU64 = AtomicU64::new(0);
    static OUT_GPU: AtomicU64 = AtomicU64::new(0);

    fn on() -> bool {
        #[cfg(test)]
        if FORCE.load(Ordering::Relaxed) {
            return true;
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_ATTN_KPROF").is_some())
    }

    /// テストが env を立てずに区間計測を通すための強制フラグ
    /// (env は OnceLock なのでプロセス内で 1 回しか読めない)。
    #[cfg(test)]
    static FORCE: AtomicBool = AtomicBool::new(false);

    /// 区間計測をテストから強制する (`false` で env の解釈に戻す)。
    #[cfg(test)]
    pub fn set_for_test(v: bool) {
        FORCE.store(v, Ordering::Relaxed);
    }

    /// 直前の GEMV が GPU で走りきったか (cuda feature 無しのビルドは常に false)。
    #[cfg(feature = "cuda")]
    fn ran_on_gpu() -> bool {
        crate::cuda::last_linear_ran_on_gpu()
    }
    #[cfg(not(feature = "cuda"))]
    fn ran_on_gpu() -> bool {
        false
    }

    /// GEMV 1 発ぶんを積む。`t` は呼び出し直前に取った `Instant` (q4prof と共有)。
    #[inline]
    fn note(t: std::time::Instant, calls: &AtomicU64, ns: &AtomicU64, gpu: &AtomicU64) {
        if !on() {
            return;
        }
        calls.fetch_add(1, Ordering::Relaxed);
        ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if ran_on_gpu() {
            gpu.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// qkv+z 連結 GEMV (`attn_qkv` + `attn_gate`) 1 発。
    #[inline]
    pub fn note_qkv(t: std::time::Instant) {
        note(t, &QKV_CALLS, &QKV_NS, &QKV_GPU);
    }
    /// beta+alpha 連結 GEMV (`ssm_beta` + `ssm_alpha`) 1 発。
    #[inline]
    pub fn note_gate(t: std::time::Instant) {
        note(t, &GATE_CALLS, &GATE_NS, &GATE_GPU);
    }
    /// ssm_out GEMV 1 発。
    #[inline]
    pub fn note_out(t: std::time::Instant) {
        note(t, &OUT_CALLS, &OUT_NS, &OUT_GPU);
    }

    fn slot(calls: &AtomicU64, ns: &AtomicU64, gpu: &AtomicU64, name: &str, s: &mut String) {
        let c = calls.load(Ordering::Relaxed);
        let n = ns.load(Ordering::Relaxed);
        let g = gpu.load(Ordering::Relaxed);
        let pct = if c == 0 {
            0.0
        } else {
            g as f64 * 100.0 / c as f64
        };
        s.push_str(&format!(
            " | {name} calls={c} ms={:.1} gpu={g} ({pct:.1}%)",
            n as f64 / 1e6
        ));
    }

    /// 内訳 1 行。無効時は `None` (出力ゼロ)。
    pub fn line() -> Option<String> {
        if !on() {
            return None;
        }
        let mut s = String::from("[attn-kprof]");
        slot(&QKV_CALLS, &QKV_NS, &QKV_GPU, "qkv", &mut s);
        slot(&GATE_CALLS, &GATE_NS, &GATE_GPU, "gate", &mut s);
        slot(&OUT_CALLS, &OUT_NS, &OUT_GPU, "out", &mut s);
        Some(s)
    }

    /// 累積を 0 に戻す。
    pub fn reset() {
        for c in [
            &QKV_CALLS,
            &QKV_NS,
            &QKV_GPU,
            &GATE_CALLS,
            &GATE_NS,
            &GATE_GPU,
            &OUT_CALLS,
            &OUT_NS,
            &OUT_GPU,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// task 128: GDN (qwen4exp) の decode 窓。
///
/// [`Qwen4Exp::forward_token_at`] が `nt == 1` (decode) のときだけ開き、
/// `SsmLayer::forward_token` がこの中でだけ GPU submission
/// (`crate::cuda::gdn_decode_submit`) を使う。バッチ prefill / spec verify の
/// 複数トークン経路は窓の外なので、従来どおり CPU で state を進める。
///
/// 窓はスレッドローカル + ネスト可 (内側の drop で外側の状態に戻す)。env とは
/// 無関係 (実際に GPU を使うかは `DENNO_GDN_GPU_SCAN` が決める)。
pub mod gdn_gpu_win {
    use std::cell::Cell;

    thread_local! {
        pub(crate) static OPEN: Cell<bool> = Cell::new(false);
        pub(crate) static TOK: Cell<u64> = Cell::new(0);
    }

    /// 窓が開いているか (SsmLayer が 1 トークンごとに見る)。
    pub fn active() -> bool {
        OPEN.with(|c| c.get())
    }

    /// 開いた decode 窓の通し番号 (1 から)。`DENNO_GDN_GPU_CHECK` が
    /// 「最初の数トークンだけ」比較するためのカウンタ。
    pub fn token_index() -> u64 {
        TOK.with(|c| c.get())
    }

    /// `decode` の状態で窓を開き、drop で呼び出し前の状態へ戻すガードを返す。
    pub fn window(decode: bool) -> GdnGpuWinGuard {
        let prev = OPEN.with(|c| c.replace(decode));
        if decode {
            TOK.with(|c| c.set(c.get() + 1));
        }
        GdnGpuWinGuard(prev)
    }

    /// [`window`] のガード。drop で窓の状態を呼び出し前に戻す。
    pub struct GdnGpuWinGuard(pub(crate) bool);

    impl Drop for GdnGpuWinGuard {
        fn drop(&mut self) {
            OPEN.with(|c| c.set(self.0));
        }
    }

    /// テストが窓を直接開け閉めする (guard を使わない経路の検証用)。
    #[cfg(test)]
    pub fn set_window_for_test(v: bool) {
        OPEN.with(|c| c.set(v));
    }
}

/// output head (`lm_head`) の**層の外**の内訳 (`DENNO_HEAD_PROF=1`、既定 off)。
///
/// [`q4prof`] の層合計は層の中しか積まないので、decode で 1 トークンあたり 1 回だけ
/// 回る head (`output` の GEMV と head の mixer) は「層の外」として残る。ここでは
/// その 2 個を 1 呼び出し単位で積み、**呼び出しごとに 1 行**を stderr に出す:
///
/// ```text
/// [head] calls=118 total=4123.437 per_call=34944.1us threads=16 gbps=9.5
/// [head-mix] calls=118 total=214.002 per_call=1813.6us threads=16
/// ```
///
/// `per_call` は 1 呼び出しあたりの µs、`gbps` は重み (`output.weight`) の実バイトを
/// 累積時間で割った実効帯域。**積むのは診断カウンタだけで、GEMV の引数・カーネル・
/// 丸め順は変えない** (off のとき呼び出し側が払うのは [`stamp`] の atomic load 1 回)。
pub mod headprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// 区間: `output` の GEMV (語彙方向の 151936 行)
    pub const GEMV: usize = 0;
    /// 区間: head の mixer (`hc_mix_quant` = grouped RMS + down/up + silu + 混ぜ)
    pub const MIX: usize = 1;
    /// 区間の数
    pub const N: usize = 2;

    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    /// 区間ごとの累積バイト数 (`gbps` の分子。mixer は 0 のまま)。
    static BYTES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    /// テストが env を立てずに計測を強制するためのフラグ。
    #[cfg(test)]
    static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// 計測をテストから強制する (`false` で env の解釈に戻る)。
    #[cfg(test)]
    pub fn set_for_test(v: bool) {
        FORCE.store(v, Ordering::Relaxed);
    }

    /// `DENNO_HEAD_PROF` が立っているか (既定 off)。
    pub fn on() -> bool {
        #[cfg(test)]
        if FORCE.load(Ordering::Relaxed) {
            return true;
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_HEAD_PROF").is_some())
    }

    /// 計測が要るときだけ `Instant` を取る (既定 off では `None` = 呼び出し側は no-op)。
    #[inline]
    pub fn stamp() -> Option<std::time::Instant> {
        if on() {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// 区間 `slot` を 1 呼び出しとして積み、累積の 1 行を stderr に出す。
    ///
    /// `bytes` はこの呼び出しで読んだ重みのバイト数 (`gbps` 用。mixer は 0)。
    /// `t` が `None` (off) のときは数えるのも出すのも行わない。
    #[inline]
    pub fn mark(slot: usize, t: Option<std::time::Instant>, bytes: u64) {
        let Some(t) = t else { return };
        if slot >= N {
            return;
        }
        NS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[slot].fetch_add(1, Ordering::Relaxed);
        if bytes > 0 {
            BYTES[slot].fetch_add(bytes, Ordering::Relaxed);
        }
        if let Some(l) = line(slot) {
            eprintln!("{l}");
        }
    }

    /// 区間の累積 1 行。無効時・未計測時は `None`。
    pub fn line(slot: usize) -> Option<String> {
        if !on() || slot >= N {
            return None;
        }
        let calls = CALLS[slot].load(Ordering::Relaxed);
        if calls == 0 {
            return None;
        }
        let ns = NS[slot].load(Ordering::Relaxed);
        let name = match slot {
            GEMV => "head",
            _ => "head-mix",
        };
        let threads = rayon::current_num_threads().max(1);
        let mut s = format!(
            "[{name}] calls={calls} total={:.3} per_call={:.1}us threads={threads}",
            ns as f64 / 1e6,
            ns as f64 / calls as f64 / 1e3
        );
        let bytes = BYTES[slot].load(Ordering::Relaxed);
        if bytes > 0 && ns > 0 {
            s.push_str(&format!(
                " gbps={:.1}",
                bytes as f64 / (ns as f64 / 1e9) / 1e9
            ));
        }
        Some(s)
    }

    /// 累積を 0 に戻す。
    pub fn reset() {
        for c in NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        for c in CALLS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        for c in BYTES.iter() {
            c.store(0, Ordering::Relaxed);
        }
    }
}
