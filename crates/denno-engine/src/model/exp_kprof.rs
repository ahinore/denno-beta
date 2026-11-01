//! expert GEMV の区間別プロファイル (`DENNO_EXP_KPROF=1`、既定 off)。
//!
//! decode は 1 トークンで 48 層 × 10 expert × 3 行列 = 1,440 回の GEMV を回し、
//! 1.47 GB の重みを読む。どこに時間が落ちているかを「行の選択 (gather) / 量子ブロックの
//! 展開 (dequant) / 積和 (fma) / 書き戻し (writeback) / 活性 (silu) / 足し込み (scatter)」に
//! 分けて積む。**読み取り専用のカウンタ**で、数値経路には触れない。
//!
//! - 層レベルの区間 ([`GATHER`] / [`SILU`] / [`SCAT`]) は `Instant` で測る
//!   (`qwen4exp` の平坦化経路が呼ぶ)。
//! - カーネル内の区間 ([`DEQ`] / [`FMA`] / [`WB`]) は `rdtsc` で測る
//!   (`moe_expert` の `*_prof` カーネルが積む)。rdtsc は 1 回 20 サイクル程度なので
//!   **prof を立てたときだけ**カーネルが 1.5〜2 倍遅くなる。比率を見るためのもの。
//! - 行列 (gate / up / down) ごとに分けるのは、量子型が違う (Q4_K / Q5_1) ため。
//!   どの型が何 GB/s 出ているかが 1 行で読める。
//!
//! 使い方: `DENNO_EXP_KPROF=1` で serve-flash を起動し、リクエストを 1 本流すと
//! `serve_flash::prof_flash_q4` が decode 終端 (リクエストの終端) で [`line`] を
//! 1 回出して [`reset`] する (prefill 終端では積むだけにして、decode (m=1) と
//! prefill (m>1) を 1 つのレポートに分けて載せる。task 95)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// 行列の種別: gate (`ffn_gate_exps`、Q4_K)。
pub const GATE: usize = 0;
/// 行列の種別: up (`ffn_up_exps`、Q4_K)。
pub const UP: usize = 1;
/// 行列の種別: down (`ffn_down_exps`、Q5_1 / Q8_0)。
pub const DOWN: usize = 2;
/// 行列の種別が分からないとき (層の外から呼ばれた等)。
pub const OTHER: usize = 3;
/// 行列の種別の数 (レポートに出す本数)。
pub const NMAT: usize = 3;
/// カウンタの本数 (`OTHER` の分を 1 本余分に持つ)。
pub const NSLOT: usize = 4;
/// 行列の名前 (レポート用)。
pub const MAT_NAME: [&str; NMAT] = ["gate", "up", "down"];

/// カーネル内の区間: 量子ブロックの展開 (scale/min・nibble 抽出・int→f32・スケール乗算)。
pub const DEQ: usize = 0;
/// カーネル内の区間: x のロードと積和。
pub const FMA: usize = 1;
/// カーネル内の区間: 横加算と出力へのストア。
pub const WB: usize = 2;
/// カーネル内の区間の数。
pub const NK: usize = 3;

/// 層レベルの区間: 行の選択 (job 構築と expert 範囲の切り出し)。
pub const GATHER: usize = 0;
/// 層レベルの区間: `silu(gate) * up`。
pub const SILU: usize = 1;
/// 層レベルの区間: 重み付きの足し込み (`out[j] += pe * h[j]`)。
pub const SCAT: usize = 2;
/// 層レベルの区間: expert 寄与の計算全体 (壁時計。GB/s の分母)。
pub const TOTAL: usize = 3;
/// 層レベルの区間の数。
pub const NL: usize = 4;
/// 位相: decode (m=1 の expert GEMV。1 トークン step とバッチでない prefill)。
pub const PH_DECODE: usize = 0;
/// 位相: prefill (バッチ GEMM。`moe_expert::gemm_rows` / `gemm_rows_vnni` 経路)。
pub const PH_PREFILL: usize = 1;
/// 位相の数。
pub const NPH: usize = 2;
/// 位相の名前 (レポート用。d = decode / p = prefill)。
pub const PH_NAME: [&str; NPH] = ["d", "p"];
/// rdtsc を取る間隔 (ブロック数)。
/// 絶対値はこの倍数で戻す ([`line`] が掛ける)。
pub const SAMPLE_EVERY: usize = 8;

static CALLS: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
static BYTES: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
static ROWS: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
static CALL_NS: [AtomicU64; NSLOT] = [const { AtomicU64::new(0) }; NSLOT];
static K_TICKS: [[AtomicU64; NK]; NSLOT] =
    [const { [const { AtomicU64::new(0) }; NK] }; NSLOT];
static L_NS: [AtomicU64; NL] = [const { AtomicU64::new(0) }; NL];
static LAYERS: AtomicU64 = AtomicU64::new(0);

/// テストが env を立てずに計測経路を通すための強制フラグ
/// (env は `OnceLock` なのでプロセス内で 1 回しか読めない)。
#[cfg(test)]
static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 計測をテストから強制する (`false` で env の解釈に戻る)。
#[cfg(test)]
pub fn set_for_test(v: bool) {
    FORCE.store(v, Ordering::Relaxed);
}

/// 計測が有効か (既定 off)。`DENNO_EXP_KPROF` が立っているときだけ true。
#[inline]
pub fn on() -> bool {
    #[cfg(test)]
    if FORCE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DENNO_EXP_KPROF").is_some())
}

thread_local! {
    /// このスレッドが今どの行列を回しているか (job が呼び出しの前に立てる)。
    static CUR: std::cell::Cell<usize> = const { std::cell::Cell::new(OTHER) };
}

/// このスレッドの「今の行列」を立てる (job の先頭で 1 回)。
#[inline]
pub fn set_mat(m: usize) {
    if on() {
        CUR.with(|c| c.set(m));
    }
}

/// このスレッドの「今の行列」。
#[inline]
pub fn cur_mat() -> usize {
    CUR.with(|c| c.get())
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

/// 層レベルの区間 `slot` に `t` からの経過を足す。
#[inline]
pub fn mark(t: Option<std::time::Instant>, slot: usize) {
    if let Some(t) = t {
        L_NS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// 1 層ぶんの expert 寄与を回したことを数える (レポートの分母)。
#[inline]
pub fn bump_layer() {
    if on() {
        LAYERS.fetch_add(1, Ordering::Relaxed);
    }
}

/// GEMV 1 回ぶんの実測 (呼び出し全体の時間・読んだ重みのバイト・行数)。
///
/// `bytes` は **実際に読んだ重みのバイト数** (`blocks.len()` をそのまま渡す)。
#[inline]
pub fn record_call(mat: usize, bytes: usize, rows: usize, d: std::time::Duration) {
    let mat = mat.min(NSLOT - 1);
    CALLS[mat].fetch_add(1, Ordering::Relaxed);
    BYTES[mat].fetch_add(bytes as u64, Ordering::Relaxed);
    ROWS[mat].fetch_add(rows as u64, Ordering::Relaxed);
    CALL_NS[mat].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}

/// カーネル内の区間 `slot` に `ticks` (rdtsc) を足す。
#[inline]
pub fn add_ticks(mat: usize, slot: usize, ticks: u64) {
    if mat < NSLOT && slot < NK {
        K_TICKS[mat][slot].fetch_add(ticks, Ordering::Relaxed);
    }
}

/// rdtsc の 1 µs あたりの刻み。較正は `line()` から 1 回だけ走る。
#[cfg(target_arch = "x86_64")]
fn ticks_per_us() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let r0 = unsafe { core::arch::x86_64::_rdtsc() };
        while t0.elapsed().as_micros() < 500 {
            std::hint::spin_loop();
        }
        let r1 = unsafe { core::arch::x86_64::_rdtsc() };
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if us <= 0.0 {
            1.0
        } else {
            (r1.wrapping_sub(r0)) as f64 / us
        }
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn ticks_per_us() -> f64 {
    0.0
}

fn ticks_to_ns(ticks: u64) -> f64 {
    let f = ticks_per_us();
    if f <= 0.0 {
        0.0
    } else {
        ticks as f64 / f * 1e3
    }
}

/// 区間の累積 (ns)。テスト用。
pub fn layer_ns(slot: usize) -> u64 {
    L_NS[slot].load(Ordering::Relaxed)
}

/// カーネル区間の累積 (ns 換算)。テスト用。
pub fn kernel_ns(mat: usize, slot: usize) -> f64 {
    let mat = mat.min(NSLOT - 1);
    ticks_to_ns(K_TICKS[mat][slot].load(Ordering::Relaxed)) * SAMPLE_EVERY as f64
}

/// 行列 `mat` の呼び出し回数・バイト・行数。テスト用。
pub fn mat_stats(mat: usize) -> (u64, u64, u64, u64) {
    let mat = mat.min(NSLOT - 1);
    (
        CALLS[mat].load(Ordering::Relaxed),
        BYTES[mat].load(Ordering::Relaxed),
        ROWS[mat].load(Ordering::Relaxed),
        CALL_NS[mat].load(Ordering::Relaxed),
    )
}

/// 内訳を 1 + `NMAT` 行の文字列にする。無効時は `None` (出力ゼロ)。
pub fn line() -> Option<String> {
    if !on() {
        return None;
    }
    let layers = LAYERS.load(Ordering::Relaxed);
    let mut calls = 0u64;
    let mut bytes = 0u64;
    let mut ns = 0u64;
    for m in 0..NMAT {
        let (c, b, _, n) = mat_stats(m);
        calls += c;
        bytes += b;
        ns += n;
    }
    let mb = bytes as f64 / 1e6;
    // 壁時計 (層の区間) と「スレッド合計」の比 = 実効並列度。GB/s は壁時計を分母にする。
    let wall = L_NS[TOTAL].load(Ordering::Relaxed);
    let par = if wall > 0 {
        (ns as f64 / wall as f64).max(1.0)
    } else {
        1.0
    };
    let total_ms = wall as f64 / 1e6;
    let gbs = |b: u64, n: u64| {
        if n == 0 {
            0.0
        } else {
            b as f64 / (n as f64 / 1e9) / 1e9
        }
    };
    let lns = |slot: usize| L_NS[slot].load(Ordering::Relaxed) as f64 / 1e3;
    let mut s = format!(
        "[exp-kprof] layers={layers} calls={calls} bytes={mb:.1} MB total={total_ms:.1} ms \
         ({:.1} GB/s) | cpu_sum {:.1} ms (par x{par:.1}) | gather {:.0} us | silu {:.0} us \
         | scatter {:.0} us",
        gbs(bytes, wall),
        ns as f64 / 1e6,
        lns(GATHER),
        lns(SILU),
        lns(SCAT)
    );
    for m in 0..NMAT {
        let (c, b, rows, n) = mat_stats(m);
        if c == 0 {
            continue;
        }
        let t = |slot: usize| kernel_ns(m, slot) / 1e6;
        let (deq, fma, wb) = (t(DEQ), t(FMA), t(WB));
        let kt = deq + fma + wb;
        let pct = |v: f64| if kt > 0.0 { v * 100.0 / kt } else { 0.0 };
        s.push_str(&format!(
            "\n[exp-kprof]  {} calls={c} rows={rows} {:.1} MB {:.1} GB/s | \
             dequant {deq:.1} ms {:.1}% | fma {fma:.1} ms {:.1}% | writeback {wb:.1} ms {:.1}%",
            MAT_NAME[m],
            b as f64 / 1e6,
            gbs(b, (n as f64 / par) as u64),
            pct(deq),
            pct(fma),
            pct(wb)
        ));
    }
    // P93: (行列, 型) ごとの内訳 (非ゼロだけ。カーネル実行先と m 分布を含む)。
    s.push_str(&ty_lines());
    Some(s)
}

/// カウンタを 0 に戻す (`line()` を出したあとに呼ぶ)。
pub fn reset() {
    for m in 0..NSLOT {
        CALLS[m].store(0, Ordering::Relaxed);
        BYTES[m].store(0, Ordering::Relaxed);
        ROWS[m].store(0, Ordering::Relaxed);
        CALL_NS[m].store(0, Ordering::Relaxed);
        for slot in 0..NK {
            K_TICKS[m][slot].store(0, Ordering::Relaxed);
        }
    }
    for slot in 0..NL {
        L_NS[slot].store(0, Ordering::Relaxed);
    }
    LAYERS.store(0, Ordering::Relaxed);
    reset_ty();
}

// ---- 型別 / カーネル別 / m 分布の内訳 (P93) --------------------------------
//
// 本番モデルの expert は層ごとに型が混在している (gate/up: Q4_K / Q6_K / Q5_K /
// IQ4_XS / IQ3_XXS、down: IQ4_NL / Q2_0)。行列ごとの集計だけでは「どの型がどの
// カーネルを通って何 ms 使ったか」が読めないので、(行列, 型) ごとに
// 呼び出し数 / バイト / ns / m (1 呼び出しの行数) 分布 / カーネル実行先を積む。
// こちらも読み取り専用のカウンタで、数値経路には触れない。

/// 型別内訳のスロット数。型 id は ggml 値そのまま (実機で確認されている最大は 42)。
pub const NTY: usize = 48;
/// m (1 呼び出しの行数) 分布のビン数。上限は `M_BIN_MAX` ([1, 2, 4, 8, 16, 32, 64, 65+])。
pub const NMB: usize = 8;
/// m 分布のビン上限 (`rows <= max` の最初のビンに入れる。末尾は 65+)。
pub const M_BIN_MAX: [u64; NMB] = [1, 2, 4, 8, 16, 32, 64, u64::MAX];
/// m 分布のビン名 (レポート用)。
pub const M_BIN_NAME: [&str; NMB] = ["1", "2", "4", "8", "16", "32", "64", "65+"];

/// カーネル実行先: スカラー (1 行 dequant + [`crate::model::moe_expert::dot_seq`])。
pub const K_SCALAR: usize = 0;
/// カーネル実行先: IQ 系 v2 (8 要素グループを 1 本ずつ)。
pub const K_IQ_V2: usize = 1;
/// カーネル実行先: IQ 系 v3 (副ブロック 32 要素をまとめて)。
pub const K_IQ_V3: usize = 2;
/// カーネル実行先: IQ4_XS 融合 (P89)。
pub const K_IQ4XS: usize = 3;
/// カーネル実行先: Q4_K 融合。
pub const K_Q4K: usize = 4;
/// カーネル実行先: Q6_K 融合 (P93)。
pub const K_Q6K: usize = 5;
/// カーネル実行先: Q5_K 融合 (P93)。
pub const K_Q5K: usize = 6;
/// カーネル実行先: Q5_1 融合。
pub const K_Q5_1: usize = 7;
/// カーネル実行先: Q8_0 (expert は classic)。
pub const K_Q8_0: usize = 8;
/// カーネル実行先: VNNI (int8 活性、`DENNO_GEMV_VNNI=1` のときだけ)。
pub const K_VNNI: usize = 9;
/// カーネル実行先: prefill の gemm_iq v1 (従来カーネル、4 トークン使い回し)。
pub const K_GIQ_V1: usize = 10;
/// カーネル実行先: prefill の AVX-512 gemm_iq_z (P85)。
pub const K_GIQ_Z: usize = 11;
/// カーネル実行先: Q2_0 融合 (task 95)。
pub const K_Q2_0: usize = 12;
/// カーネル実行先: IQ3_S バッチ融合 (task 97)。
pub const K_IQ3S: usize = 13;
/// カーネル実行先: IQ4_NL 融合 m=1 (task 98)。
pub const K_IQ4NL_M1: usize = 13;
/// カーネル実行先の種別数。
pub const NKER: usize = 15;
/// カーネル実行先の名前 (レポート用)。
pub const KERNEL_NAME: [&str; NKER] = [
    "scalar", "iq_v2", "iq_v3", "iq4xs", "q4k", "q6k", "q5k", "q5_1", "q8_0", "vnni", "giq_v1",
    "giq_z", "q2_0", "iq3s", "iq4nl_m1",
];
/// 融合カーネル (型専用の AVX2 GEMV/GEMM) のスロット ([] の分類用)。
const KER_FUSED: [usize; 9] = [
    K_IQ4XS,
    K_Q4K,
    K_Q6K,
    K_Q5K,
    K_Q5_1,
    K_Q8_0,
    K_Q2_0,
    K_IQ3S,
    K_IQ4NL_M1,
];
/// gemv_iq 系 (IQ 系ジェネリックカーネル、decode の v2/v3 と prefill の v1/z) のスロット。
const KER_IQ: [usize; 4] = [K_IQ_V2, K_IQ_V3, K_GIQ_V1, K_GIQ_Z];

/// カーネル実行先の累積から経路の粗分類を出す (fused / gemv_iq / scalar / vnni)。
///
/// カウントが最大の族を採用する。同点のときは scalar > gemv_iq > vnni > fused の順で
/// 残す (遅い側を優先して目に付くように。1 つの (位相, 行列, 型) が混在するのは
/// env で経路を切り替えたときくらい)。
fn path_of(ker: &[u64; NKER]) -> &'static str {
    let fused: u64 = KER_FUSED.iter().map(|&k| ker[k]).sum();
    let iq: u64 = KER_IQ.iter().map(|&k| ker[k]).sum();
    let mut best: (&'static str, u64) = ("-", 0);
    for (name, v) in [
        ("scalar", ker[K_SCALAR]),
        ("gemv_iq", iq),
        ("vnni", ker[K_VNNI]),
        ("fused", fused),
    ] {
        if v > best.1 {
            best = (name, v);
        }
    }
    best.0
}

static TY_CALLS: [AtomicU64; NPH * NSLOT * NTY] =
    [const { AtomicU64::new(0) }; NPH * NSLOT * NTY];
static TY_BYTES: [AtomicU64; NPH * NSLOT * NTY] =
    [const { AtomicU64::new(0) }; NPH * NSLOT * NTY];
static TY_ROWS: [AtomicU64; NPH * NSLOT * NTY] =
    [const { AtomicU64::new(0) }; NPH * NSLOT * NTY];
static TY_NS: [AtomicU64; NPH * NSLOT * NTY] = [const { AtomicU64::new(0) }; NPH * NSLOT * NTY];
static TY_MBIN: [AtomicU64; NPH * NSLOT * NTY * NMB] =
    [const { AtomicU64::new(0) }; NPH * NSLOT * NTY * NMB];
static TY_KER: [AtomicU64; NPH * NSLOT * NTY * NKER] =
    [const { AtomicU64::new(0) }; NPH * NSLOT * NTY * NKER];

#[inline]
fn ty_slot(ph: usize, mat: usize, ty_id: u32) -> usize {
    (ph.min(NPH - 1) * NSLOT + mat.min(NSLOT - 1)) * NTY + (ty_id as usize).min(NTY - 1)
}

/// 行数を m 分布のビン番号にする。
fn m_bin(rows: usize) -> usize {
    let r = rows as u64;
    for (i, max) in M_BIN_MAX.iter().enumerate() {
        if r <= *max {
            return i;
        }
    }
    NMB - 1
}

/// (行列, 型) ごとの decode GEMV 1 回ぶんの実測 ([`record_call`] の型別版)。
///
/// 位相は decode 固定。prefill (バッチ GEMM) は [`record_call_ph`] を使う。
#[inline]
pub fn record_call_ty(mat: usize, ty_id: u32, bytes: usize, rows: usize, d: std::time::Duration) {
    record_call_ph(PH_DECODE, mat, ty_id, bytes, rows, 1, d);
}

/// (位相, 行列, 型) ごとの GEMV/GEMM 1 回ぶんの実測。
///
/// `m` は 1 呼び出しのトークン数 (decode は常に 1)。m 分布のビンは decode では
/// 行チャンク幅 `rows`、prefill ではトークン数 `m` で振る (decode は「1 呼び出しが
/// どの行粒度で切られたか」、prefill は「1 expert に何トークン集まったか」を見る)。
#[inline]
pub fn record_call_ph(
    ph: usize,
    mat: usize,
    ty_id: u32,
    bytes: usize,
    rows: usize,
    m: usize,
    d: std::time::Duration,
) {
    if !on() {
        return;
    }
    let s = ty_slot(ph, mat, ty_id);
    TY_CALLS[s].fetch_add(1, Ordering::Relaxed);
    TY_BYTES[s].fetch_add(bytes as u64, Ordering::Relaxed);
    TY_ROWS[s].fetch_add(rows as u64, Ordering::Relaxed);
    TY_NS[s].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    let bin_src = if ph == PH_DECODE { rows } else { m };
    TY_MBIN[s * NMB + m_bin(bin_src)].fetch_add(1, Ordering::Relaxed);
}

/// カーネル実行先を 1 回数える (分岐の直後で呼ぶ)。位相は呼び出し側が指定する。
#[inline]
pub fn bump_kernel(ph: usize, mat: usize, ty_id: u32, ker: usize) {
    if on() {
        TY_KER[ty_slot(ph, mat, ty_id) * NKER + ker.min(NKER - 1)].fetch_add(1, Ordering::Relaxed);
    }
}

/// 型 id の表示名 ([`crate::kquant::GgmlType`] に無い id は `ty{n}`)。
fn ty_name(ty_id: usize) -> String {
    match crate::kquant::GgmlType::from_id(ty_id as u32) {
        Some(t) => t.name().to_string(),
        None => format!("ty{ty_id}"),
    }
}

/// (位相, 行列, 型) ごとの累積。テスト用。
pub fn ty_stats(ph: usize, mat: usize, ty_id: u32) -> (u64, u64, u64, u64) {
    let s = ty_slot(ph, mat, ty_id);
    (
        TY_CALLS[s].load(Ordering::Relaxed),
        TY_BYTES[s].load(Ordering::Relaxed),
        TY_ROWS[s].load(Ordering::Relaxed),
        TY_NS[s].load(Ordering::Relaxed),
    )
}

/// (位相, 行列, 型) のカーネル実行先の累積。テスト用。
pub fn ty_kernel(ph: usize, mat: usize, ty_id: u32, ker: usize) -> u64 {
    TY_KER[ty_slot(ph, mat, ty_id) * NKER + ker.min(NKER - 1)].load(Ordering::Relaxed)
}

/// (位相, 行列, 型) の m 分布。テスト用。
pub fn ty_mbin(ph: usize, mat: usize, ty_id: u32, bin: usize) -> u64 {
    TY_MBIN[ty_slot(ph, mat, ty_id) * NMB + bin.min(NMB - 1)].load(Ordering::Relaxed)
}

fn reset_ty() {
    for s in 0..NPH * NSLOT * NTY {
        TY_CALLS[s].store(0, Ordering::Relaxed);
        TY_BYTES[s].store(0, Ordering::Relaxed);
        TY_ROWS[s].store(0, Ordering::Relaxed);
        TY_NS[s].store(0, Ordering::Relaxed);
        for b in 0..NMB {
            TY_MBIN[s * NMB + b].store(0, Ordering::Relaxed);
        }
        for k in 0..NKER {
            TY_KER[s * NKER + k].store(0, Ordering::Relaxed);
        }
    }
}

/// 型別内訳を [`line`] の後ろに付ける (非ゼロの (位相, 行列, 型) だけ)。
///
/// 位相ごとに合計 1 行 + 明細 (呼び出し数 / 行数 / MB / ms / 経路 / カーネル実行先 /
/// m 分布)。経路は [`path_of`] の粗分類 (fused / gemv_iq / scalar / vnni)。
fn ty_lines() -> String {
    let mut s = String::new();
    for ph in 0..NPH {
        let mut body = String::new();
        let (mut tot_c, mut tot_ns) = (0u64, 0u64);
        for m in 0..NSLOT {
            for ty in 0..NTY {
                let (c, b, rows, ns) = ty_stats(ph, m, ty as u32);
                if c == 0 {
                    continue;
                }
                tot_c += c;
                tot_ns += ns;
                let mat_name = if m < NMAT { MAT_NAME[m] } else { "other" };
                let mut ker = String::new();
                for k in 0..NKER {
                    let v = TY_KER[ty_slot(ph, m, ty as u32) * NKER + k].load(Ordering::Relaxed);
                    if v > 0 {
                        if !ker.is_empty() {
                            ker.push('/');
                        }
                        ker.push_str(&format!("{}={v}", KERNEL_NAME[k]));
                    }
                }
                if ker.is_empty() {
                    ker.push('-');
                }
                let mut mbin = String::new();
                for bin in 0..NMB {
                    let v = ty_mbin(ph, m, ty as u32, bin);
                    if v > 0 {
                        if !mbin.is_empty() {
                            mbin.push('/');
                        }
                        mbin.push_str(&format!("{}:{}", M_BIN_NAME[bin], v));
                    }
                }
                body.push_str(&format!(
                    "\n[exp-kprof]  {} {} {:6} calls={c} rows={rows} {:.1} MB {:.1} ms path={} ker {ker} | m {mbin}",
                    PH_NAME[ph],
                    mat_name,
                    ty_name(ty),
                    b as f64 / 1e6,
                    ns as f64 / 1e6,
                    path_of(&ker_counts(ph, m, ty as u32)),
                ));
            }
        }
        if tot_c > 0 {
            s.push_str(&format!(
                "\n[exp-kprof]  {} total calls={tot_c} {:.1} ms",
                PH_NAME[ph],
                tot_ns as f64 / 1e6
            ));
            s.push_str(&body);
        }
    }
    s
}

/// (位相, 行列, 型) のカーネル実行先の累積を配列にする ([`path_of`] の引数)。
fn ker_counts(ph: usize, mat: usize, ty_id: u32) -> [u64; NKER] {
    let base = ty_slot(ph, mat, ty_id) * NKER;
    let mut a = [0u64; NKER];
    for (k, v) in a.iter_mut().enumerate() {
        *v = TY_KER[base + k].load(Ordering::Relaxed);
    }
    a
}
