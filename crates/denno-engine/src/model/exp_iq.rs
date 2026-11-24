//! task 144: IQ 系の量子型 expert を GPU で回すための入口 (GPU 非依存の部分)。
//!
//! # なぜ要るか
//!
//! P39/P43 の VRAM 常駐経路は `cuda::gguf_row_bytes` が返す **Q8_0 / Q5_1 / Q4_K /
//! Q5_K の 4 型**しか GPU で読めない。本番 (swift15 IQ3_S) の expert は
//! gate/up = IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS、down = IQ4_NL / Q2_0 なので、
//! `expert_vram::acquire` の `moe_gemv_dev_ready` が **型の判定で false** になり、
//! 1 本も VRAM に載らなかった (実測: VRAM 使用 0 MiB / ヒット 0 / 全 480 expert が
//! CPU で 99〜108 ms/tok)。`f_moe` (PTX のモジュール) は正常で、原因は型。
//!
//! ここでは新しいカーネル (`kernels_exp_iq.cu`) が読める型の判定・起動形・
//! ゲートの解釈を持つ。cuda.rs は `feature = "cuda"` のときしかコンパイルされない
//! ので、**GPU 無しで検証できる部分をこちらに置く**。
//!
//! # カーネルの契約 (cuda 側と共有)
//!
//! - 1 行 = 1 warp。32 レーンが 8 要素ずつ (IQ4_NL / Q2_0 は 1 レーン = 1 ブロック)
//!   受け持ち、`__shfl_down_sync` の固定順で足す。
//! - ブロック = 4 warp (4 行)、共有メモリ無し。
//! - 累積の順序は CPU と違う (task 144 はビット一致を要求しない)。**決定性**は
//!   起動形が同じなら保たれる。
//!
//! [`dequant_mirror`] / [`dot_mirror`] はカーネルと同じ索引・同じ累積順を Rust で
//! 書き下したもの。GPU の無いコンテナでも「要素 → バイト」の対応が正しいことを
//! `kquant::dequant_row` とのビット一致で確かめられる。
//!
//! # ラボ実測 (task 144, swift15-abl IQ3_S, 14 T, 8098 同居)
//!
//! `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=14000 DENNO_EXPERT_GPU_MIN_HIT=0`,
//! SCENARIO=depth, commit 9c8fe9b5d (t144e):
//!
//! | cond | decode ms/tok | prefill s | md5 |
//! | --- | --- | --- | --- |
//! | base | 74 | 1.39 / 108.01 | 基準 |
//! | on | 63 | 1.38 / 107.93 | SAME |
//!
//! リクエスト別 (base -> on): a1 74->63, b 75->73, a2 74->58 (ヒット率 0.889),
//! c2000_cold 76->72, c2000_turn2 76->66。VRAM は 1 リクエストで 14 GB まで埋まる。
//! 1 層あたり GPU wall 151 us / kernel 76 us (9.3 expert)。**GATE 1 の 400 us は
//! 2.6 倍の余裕で通る**。上げ直しは pageable H2D で 0.27〜0.31 ms/本
//! (pinned staging は 0.34〜0.42 で負け)。既定 off のまま。
//!
//! # ラボ実測 (task 147: int-auto にマージ後、8098 停止で GPU 単独)
//!
//! SCENARIO=depth8k, commit 47c0e5128 (t147a / t147b)。`DENNO_EXPERT_IQ_GPU=1
//! DENNO_EXPERT_GPU_MIN_HIT=0`、production env (`DENNO_GPU_WEIGHT_MB=12000` なので
//! 予算は `DENNO_EXPERT_VRAM_MB + 12000` MiB)。decode は**温まった後**の値
//! (28 tok は 2 本目のリクエスト):
//!
//! | cond | decode ms/tok 28 / 3,954 / 14,139 | hit rate | VRAM 実測 |
//! | --- | --- | --- | --- |
//! | base | 74 / 76 / 83 | - | - |
//! | v14 | 58 / 68 / 66 | 0.911 / 0.848 / 0.915 | 13,999 MiB |
//! | v20 | 47 / 58 / 57 | 0.990 / 0.922 / 0.963 | 19,999 MiB |
//! | v22 | 47 / 55 / 58 | 0.990 / 0.944 / 0.972 | 21,997 MiB |
//! | v24 | 47 / 55 / 139 | - (14k prefill で OOM) | 22,735 MiB で停止 |
//!
//! 24,000 と 26,000 は 14,018 トークンの prefill 中に `cuMemAlloc` が失敗し
//! (空き 9 MiB / 総 32,110 MiB)、以後 q4 attention まで CPU に落ちて
//! 132〜140 ms/tok・md5 DIFF になった。22,000 は 14k で 9 MiB まで埋まり、
//! リクエスト境界で 0 byte の上げ直しが 1 回失敗して以後アップロードが止まる
//! (decode は 57〜58 ms/tok のまま)。20,000 は 2 ラウンドとも確保失敗なし。
//! **production には 20,000 を推奨** (14k で headroom 2 GB、decode は 22,000 と
//! 同等、3,954 で 5% 遅いだけ)。md5 は base と全条件で SAME (24k/26k の 14k のみ DIFF)。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::kquant::GgmlType;

/// `DENNO_EXPERT_IQ_GPU=1`: IQ 系 expert を VRAM 常駐 + GPU GEMV で回す。既定 off。
pub const EXPERT_IQ_GPU_ENV: &str = "DENNO_EXPERT_IQ_GPU";

/// `DENNO_EXPERT_IQ_UPLOAD`: 1 トークンあたりに VRAM へ上げ直す expert の本数の上限。
///
/// ミスした expert は CPU で計算し、**トークンを止めない範囲**で後から VRAM へ
/// 上げる。0 で「上げ直さない」(= 最初に載ったものだけがヒットする)。
pub const EXPERT_IQ_UPLOAD_ENV: &str = "DENNO_EXPERT_IQ_UPLOAD";

/// `DENNO_EXPERT_IQ_UPLOAD_FULL`: LRU が埋まっているときの 1 トークンあたりの上限。
///
/// 埋まっている (= 追い出しが起きる) 状態では、ミスの挿入は「その expert がまた
/// 来るか」の賭けになる。lab t144d の実測では 48 本/token を入れ続けると 1 本
/// 0.27 ms の H2D が 7 ms/token になり、GPU の節約 (約 25 ms) を一部食っていた。
pub const EXPERT_IQ_UPLOAD_FULL_ENV: &str = "DENNO_EXPERT_IQ_UPLOAD_FULL";

/// [`EXPERT_IQ_UPLOAD_FULL_ENV`] の既定値。
pub const DEFAULT_UPLOAD_WHEN_FULL: usize = 16;

/// `DENNO_EXPERT_IQ_MIN_HITS`: 1 層を GPU で回すのに要るヒット本数 (既定 4)。
///
/// 1 層 1 同期の実測が 74 us、1 expert あたりの CPU 計算が約 81 us・GPU が約 7 us
/// なので、4 本未満では同期の方が高い。0 で常に GPU。
pub const EXPERT_IQ_MIN_HITS_ENV: &str = "DENNO_EXPERT_IQ_MIN_HITS";

/// [`EXPERT_IQ_MIN_HITS_ENV`] の既定値。
pub const DEFAULT_MIN_HITS_PER_LAYER: usize = 4;

/// 1 層を GPU で回すのに要るヒット本数 (プロセスで 1 回だけ env を読む)。
pub fn min_hits_per_layer() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var(EXPERT_IQ_MIN_HITS_ENV).ok() {
        Some(s) if !s.trim().is_empty() => s.trim().parse().unwrap_or(DEFAULT_MIN_HITS_PER_LAYER),
        _ => DEFAULT_MIN_HITS_PER_LAYER,
    })
}

/// [`EXPERT_IQ_UPLOAD_ENV`] の既定値 (1 トークンあたり、LRU に空きがあるとき)。
///
/// 1 本 = 約 2.05 MB、H2D は実測 0.27 ms (pageable)。空きがある間は詰め込む
/// (lab t144d: 8 本/token では 1 ラウンド (約 530 decode トークン) でヒット率が
/// 0.5 までしか上がらず decode 83 ms、48 本/token なら 0.88 まで上がって 68 ms)。
pub const DEFAULT_UPLOAD_PER_TOKEN: usize = 48;

/// カーネルのスロット数 (型の数)。
pub const SLOTS: usize = 6;

/// `DENNO_EXPERT_IQ_GPU` の解釈 (純関数)。`"1"` / `"true"` / `"on"` だけが真。
pub fn iq_gpu_from_value(v: Option<&str>) -> bool {
    crate::model::expert_vram::flag_on(v)
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn iq_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| iq_gpu_from_value(std::env::var(EXPERT_IQ_GPU_ENV).ok().as_deref()))
}

/// task 190: `DENNO_EXP_IQ_V2` — 層一括 GEMV を v2 カーネル
/// (`kernels_exp_iq_v2.ptx`) で回す。**既定 on** (lab t190c で採否 yes。
/// decode 短/3.9k/14k で -4.3〜-4.8%、prefill ±1%、md5 SAME)。
///
/// v2 は v1 と同じ segment 表・同じ起動形で、中の実装だけが違う
/// (ベクタロード / 次ブロックの先読み / アキュムレータ 4 本 / 表を shared へ /
/// IQ4_NL・Q2_0 のレーン分割)。PTX が無いときは v1 へ落ちる (fail closed)。
/// 明示的な偽値 `0` / `off` / `false` / `no` だけ切る。
pub const EXPERT_IQ_V2_ENV: &str = "DENNO_EXP_IQ_V2";

/// [`EXPERT_IQ_V2_ENV`] の解釈 (純関数)。既定 on。
pub fn iq_v2_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// v2 カーネルを使うか (プロセスで 1 回だけ env を読む)。
pub fn iq_v2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| iq_v2_from_value(std::env::var(EXPERT_IQ_V2_ENV).ok().as_deref()))
}

/// task 275: `DENNO_EXP_IQ_V3` — IQ2_S / IQ3_S のグリッド表を詰めた v3 カーネル
/// (`kernels_exp_iq_v3.ptx`) で回す。**既定 off** (採否は lab で決める)。
///
/// v3 は v2 と同じ起動形・同じ segment 表で、変わるのは shared に置く表だけ:
/// IQ2_S は 8 KB の u64 を 2 bit x 8 に詰めた 2 KB、IQ3_S は 4 KB の u64 のうち
/// 使われている下位 32 bit だけの 2 KB (上位 32 bit は全 0 で誰も読まない)。
/// 表の値と積の順は v2 と同じなので**値は v2 と一致する** (lab の md5 で確認)。
/// PTX が無いときは v2 のまま (fail closed = 性能だけのゲート)。
pub const EXPERT_IQ_V3_ENV: &str = "DENNO_EXP_IQ_V3";

/// [`EXPERT_IQ_V3_ENV`] の解釈 (純関数)。`"1"` / `"true"` / `"on"` だけが真。
pub fn iq_v3_from_value(v: Option<&str>) -> bool {
    // tasks 269+275 adopted by the operator (2026-11-13): default on; only an
    // explicit false value (0 / off / false / no) turns it off.
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// v3 カーネルを使うか (プロセスで 1 回だけ env を読む)。
pub fn iq_v3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| iq_v3_from_value(std::env::var(EXPERT_IQ_V3_ENV).ok().as_deref()))
}

/// v3 が差し替えるスロット (`kernels_exp_iq_v3.cu` にあるカーネルの型)。
pub const IQ_V3_SLOTS: [usize; 2] = [0, 2];

// ---------------------------------------------------------------------------
// task 356: expert 段の寄与 D2H を専用ストリームへ (`DENNO_DEC_EXP_D2H_STREAM`)
// ---------------------------------------------------------------------------
//
// IQ expert の層一括経路は「カーネルを積む -> 非同期 D2H -> CPU ミス -> 同期」で
// GPU と CPU を重ねる。ところが D2H を **null ストリーム** に出しているので、
// 小さいコピー (和モードは e = 2,560 要素 = 10 KB) では driver がコピーの完了を
// 呼び出しの中で待つ経路に入ることがある。その待ちは CPU ミスの**前**に直列に
// 挟まるので、重ね打ちが丸ごと効かなくなる (lab t352a の台帳: experts 段
// 5.39 ms/tok に対し GPU 3.07 + ミス 2.78 で、重なりは 8% しか無い)。
//
// 専用の非ブロッキングストリームへ出すと、コピーは「この層のカーネルの完了」を
// イベントで待つだけになり、呼び出しはホストを止めない。待ちは CPU ミスの後へ
// 回す。**デバイスへ渡るバイト列・カーネル・発行順は 1 つも変わらない**。
pub const DEC_D2H_STREAM_ENV: &str = "DENNO_DEC_EXP_D2H_STREAM";

/// [`DEC_D2H_STREAM_ENV`] の解釈 (純関数)。**既定 off**。
///
/// 明示的な真値 (`1` / `on` / `true` / `yes`) だけ on にする (A/B 用のゲート)。
pub fn dec_d2h_stream_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(
            s.to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => false,
    }
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn dec_d2h_stream_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dec_d2h_stream_from_value(std::env::var(DEC_D2H_STREAM_ENV).ok().as_deref()))
}

/// 専用ストリームへ出したコピーの本数 (**分岐が発火した証拠**)。
static D2H_STREAM_COPIES: AtomicU64 = AtomicU64::new(0);
/// 専用ストリームのコピーを待った合計時間 (ns)。
static D2H_STREAM_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// 専用ストリームを作れず従来経路へ落ちた回数。
static D2H_STREAM_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// コピーを 1 本、専用ストリームへ出した。
pub fn note_d2h_stream_copy() {
    D2H_STREAM_COPIES.fetch_add(1, Ordering::Relaxed);
}

/// 専用ストリームのコピーを待った (`ns` = 待ち時間)。
pub fn note_d2h_stream_wait(ns: u64) {
    D2H_STREAM_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 専用ストリームを作れず従来経路へ落ちた。
pub fn note_d2h_stream_fallback() {
    D2H_STREAM_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行 (ゲート on でもコピーが 0 なら `None`)。
pub fn d2h_stream_line() -> Option<String> {
    d2h_stream_line_with(dec_d2h_stream_enabled())
}

/// [`d2h_stream_line`] の本体 (ゲートの真偽を引数で受ける = テストで両側を踏める)。
fn d2h_stream_line_with(on: bool) -> Option<String> {
    if !on {
        return None;
    }
    let copies = D2H_STREAM_COPIES.load(Ordering::Relaxed);
    if copies == 0 && D2H_STREAM_FALLBACKS.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let wait_ms = D2H_STREAM_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let us = if copies > 0 {
        wait_ms * 1e3 / copies as f64
    } else {
        0.0
    };
    Some(format!(
        "[dec-exp-d2h] stream=1 copies={copies} fallback={} wait={wait_ms:.2}ms us/copy={us:.1}",
        D2H_STREAM_FALLBACKS.load(Ordering::Relaxed)
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn d2h_stream_reset() {
    for c in [&D2H_STREAM_COPIES, &D2H_STREAM_WAIT_NS, &D2H_STREAM_FALLBACKS] {
        c.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// task 356: 和の出力を **pinned のホストバッファへ直接書く** (`DENNO_DEC_EXP_SUM_ZC`)
// ---------------------------------------------------------------------------
//
// lab t356a の実測 (IQ3_S、depth8k、hit_rate 0.99、decode 14.8 ms/tok) では
// experts 段が 1 層 71.7 us で、そのうち GPU の鎖が 54 us:
// gate+up 25.9 / silu 3.9 / down 16.8 / sum 5.0 / **d2h 5.5** (us/層)。
// d2h は `dno_dec_exp_sum` の e = 2,560 要素 (10 KB) を VRAM から pinned へ
// 写すだけで、DMA の起動待ちがそのまま層の鎖に乗る。t356a で専用ストリームへ
// 出しても decode は変わらなかった (ホストが結局その完了を待つため)。
//
// ならば**コピーそのものを消す**: 和のカーネルの出力ポインタに pinned のホスト
// バッファのアドレスを渡す (UVA なのでデバイスから同じアドレスで書ける)。カーネルが
// ホストへ直接書くので、D2H も、その完了待ちも要らない。値は同じ (同じカーネル・
// 同じ式・同じ並び)。ストリームの同期は従来どおり 1 回で、その後ホストが
// バッファを読む (カーネル完了 = 書き込みが見える)。
//
// lab t356b (2 ラウンド、交互、4 COND) の実測:
// - `[dec-exp-kprof]` の 1 層あたり: base = gate 23.2 / up 4.2 / silu 4.9 /
//   down 15.4 / d2h 10.0 / sum 5.2 (計 62.9) -> 本ゲート = d2h 1.3 / sum 7.2
//   (計 56.8)。**d2h -8.7 us の代わりに sum +2.0 us** (ホストへの 10 KB 書き) で
//   正味 -5.4〜-6.0 us/層。
// - `[q4ledger] experts` (ms/tok、同じ request 同士の比較):
//   28 tok 窓 base 3.41/3.48・base2 3.46/3.44 -> **3.23/3.20 (-6.7%)**、
//   3929 tok 窓 4.36/4.19・4.21/4.19 -> 4.03/3.99 (-5.4%)、
//   3826 tok 窓 5.39/4.85・4.86/4.87 -> 4.63/4.69 (-4.2%)。
//   2 ラウンドとも同じ向き。md5 は全 COND SAME、errors 0。
// - decode 中央値は 16.8 (base/base2) -> 16.7 (本ゲート) で、段の -6.7% が
//   トークンでは -1.4% 相当 (きれいな窓の step 13.80 -> 13.61)。
pub const SUM_ZC_ENV: &str = "DENNO_DEC_EXP_SUM_ZC";

/// [`SUM_ZC_ENV`] の解釈 (純関数)。**既定 on** (lab t356b で採否 yes)。
///
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
pub fn sum_zc_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn sum_zc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| sum_zc_from_value(std::env::var(SUM_ZC_ENV).ok().as_deref()))
}

/// 和を pinned のホストへ直接書いた層の数 (**分岐が発火した証拠**)。
static SUM_ZC_LAYERS: AtomicU64 = AtomicU64::new(0);
/// pinned を確保できず従来の D2H 経路へ落ちた層の数。
static SUM_ZC_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// 和をホストへ直接書いた層を 1 つ数える。
pub fn note_sum_zc_layer() {
    SUM_ZC_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// pinned を確保できず従来経路へ落ちた層を 1 つ数える。
pub fn note_sum_zc_fallback() {
    SUM_ZC_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行 (ゲート on でも 0 層なら `None`)。
pub fn sum_zc_line() -> Option<String> {
    sum_zc_line_with(sum_zc_enabled())
}

/// [`sum_zc_line`] の本体 (ゲートの真偽を引数で受ける = テストで両側を踏める)。
fn sum_zc_line_with(on: bool) -> Option<String> {
    if !on {
        return None;
    }
    let layers = SUM_ZC_LAYERS.load(Ordering::Relaxed);
    let fb = SUM_ZC_FALLBACKS.load(Ordering::Relaxed);
    if layers == 0 && fb == 0 {
        return None;
    }
    Some(format!("[dec-exp-zc] sum_zc_layers={layers} fallback={fb}"))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn sum_zc_reset() {
    for c in [&SUM_ZC_LAYERS, &SUM_ZC_FALLBACKS] {
        c.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// task 388: ミスの CPU 計算を GPU の起動と重ねる (`DENNO_DEC_MISS_SPAWN`)
// ---------------------------------------------------------------------------
//
// いまの `iq_layer_run` は「カーネルを全部積む -> `cpu_miss` (ミスの CPU GEMV)
// -> 同期」の順で、CPU の仕事は**起動が終わってから**始まる。起動 (1 層 4 本 =
// 実測 5-6 us) の裏へ回せば、そのぶん層が縮む。
//
// `DENNO_DEC_MISS_SPAWN=1` では `cpu_miss` を rayon のプールへ先に流し、
// 呼び出しスレッドは起動と同期だけを行う (scoped なので借用はそのまま)。
// 値は変えない (同じカーネル・同じ順序・同じ足し込み)。
//
// `DENNO_DEC_MISS_SERIAL=1` は**診断用**: 同期と控えの写しを先に済ませてから
// CPU ミスを回す (= 重ね打ちを完全に切る)。既定との差が「いま重ね打ちが
// 効いている量」になる。
//
// 既定は spawn **on** (lab t388b: experts 段 -4.5% / -6.9% の 2 ラウンド、
// decode 15.8 -> 15.6、14k 後 17.2 -> 15.3、md5 全 SAME、errors 0)。
pub const DEC_MISS_SPAWN_ENV: &str = "DENNO_DEC_MISS_SPAWN";
pub const DEC_MISS_SERIAL_ENV: &str = "DENNO_DEC_MISS_SERIAL";

/// 真理値の解釈 (純関数)。明示的な真値だけ on (既定 off)。
fn on_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(
            s.to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => false,
    }
}

/// [`DEC_MISS_SPAWN_ENV`] の解釈 (純関数)。**既定 on** (lab t388b で採否 yes)。
///
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
pub fn miss_spawn_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// `DENNO_DEC_MISS_SPAWN` が有効か (プロセスで 1 回だけ env を読む)。
pub fn miss_spawn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| miss_spawn_from_value(std::env::var(DEC_MISS_SPAWN_ENV).ok().as_deref()))
}

/// `DENNO_DEC_MISS_SERIAL` が有効か (プロセスで 1 回だけ env を読む)。
pub fn miss_serial_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_from_value(std::env::var(DEC_MISS_SERIAL_ENV).ok().as_deref()))
}

/// 分岐が発火した層数 (spawn / serial)。**緑 = 経路が通った証拠ではない**ので、
/// 発火を数えて 1 行で出す。
static MISS_SPAWN_LAYERS: AtomicU64 = AtomicU64::new(0);
static MISS_SERIAL_LAYERS: AtomicU64 = AtomicU64::new(0);

/// spawn 経路を 1 層ぶん数える。
pub fn note_miss_spawn_layer() {
    MISS_SPAWN_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// serial 経路を 1 層ぶん数える。
pub fn note_miss_serial_layer() {
    MISS_SERIAL_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行 (どちらのゲートも off、または 0 層なら `None`)。
pub fn miss_spawn_line() -> Option<String> {
    miss_spawn_line_with(
        miss_spawn_enabled(),
        miss_serial_enabled(),
        MISS_SPAWN_LAYERS.load(Ordering::Relaxed),
        MISS_SERIAL_LAYERS.load(Ordering::Relaxed),
    )
}

/// [`miss_spawn_line`] の本体 (テストで両側を踏めるように引数で受ける)。
fn miss_spawn_line_with(on: bool, serial: bool, spawn: u64, ser: u64) -> Option<String> {
    if !on && !serial {
        return None;
    }
    if spawn == 0 && ser == 0 {
        return None;
    }
    Some(format!(
        "[dec-miss-spawn] spawn_layers={spawn} serial_layers={ser}"
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn miss_spawn_reset() {
    for c in [&MISS_SPAWN_LAYERS, &MISS_SERIAL_LAYERS] {
        c.store(0, Ordering::Relaxed);
    }
}

/// [`EXPERT_IQ_UPLOAD_ENV`] の解釈 (純関数)。未設定・数値でない値は既定値。
pub fn upload_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_UPLOAD_PER_TOKEN),
        _ => DEFAULT_UPLOAD_PER_TOKEN,
    }
}

/// 1 トークンあたりの上げ直し上限 (プロセスで 1 回だけ env を読む)。
pub fn upload_per_token() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| upload_from_value(std::env::var(EXPERT_IQ_UPLOAD_ENV).ok().as_deref()))
}

/// LRU が埋まっているときの上限 (プロセスで 1 回だけ env を読む)。
pub fn upload_when_full() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var(EXPERT_IQ_UPLOAD_FULL_ENV).ok() {
            Some(s) if !s.trim().is_empty() => {
                s.trim().parse().unwrap_or(DEFAULT_UPLOAD_WHEN_FULL)
            }
            _ => DEFAULT_UPLOAD_WHEN_FULL,
        }
    })
}

/// 型 → カーネルのスロット。未対応は `None` (呼び出し側は CPU へ = fail closed)。
pub fn slot(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::IQ2S => Some(0),
        GgmlType::IQ3Xxs => Some(1),
        GgmlType::IQ3S => Some(2),
        GgmlType::IQ4Xs => Some(3),
        GgmlType::IQ4Nl => Some(4),
        GgmlType::Q2_0 => Some(5),
        _ => None,
    }
}

/// この型を新しいカーネルで読めるか。
pub fn supported(ty: GgmlType) -> bool {
    slot(ty).is_some()
}

/// カーネル名 (PTX の `.visible .entry`)。`cuModuleGetFunction` に渡す形。
pub fn kernel_name(slot: usize) -> Option<&'static str> {
    Some(match slot {
        0 => "dno_exp_gemv_iq2s",
        1 => "dno_exp_gemv_iq3xxs",
        2 => "dno_exp_gemv_iq3s",
        3 => "dno_exp_gemv_iq4xs",
        4 => "dno_exp_gemv_iq4nl",
        5 => "dno_exp_gemv_q2_0",
        _ => return None,
    })
}

/// silu カーネルの名前。
pub const SILU_KERNEL: &str = "dno_exp_silu_mul";

/// 1 行あたりのバイト数。ブロックサイズで割り切れない k は `None`。
///
/// `moe_expert::row_bytes` と同じ式だが、こちらは型の許可制を持たない純関数
/// (`(k / blck) * type_size`)。
pub fn row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if bs == 0 || ts == 0 || k == 0 || k % bs != 0 {
        return None;
    }
    Some((k / bs) * ts)
}

/// カーネルの起動形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

/// 層一括 GEMV の起動形 (純関数)。`None` = この形では起動しない。
///
/// 行は warp 単位なので `n_out` は 4 で切り上げる (端数 warp は `row >= n_out` で
/// 何も書かない)。`nseg` は grid.y、上限は 65535。
pub fn launch_plan(nseg: usize, n_out: usize) -> Option<IqLaunch> {
    const ROWS: usize = 4; // kernels_exp_iq.cu の EI_ROWS
    if nseg == 0 || n_out == 0 {
        return None;
    }
    let tiles = n_out.div_ceil(ROWS);
    if nseg > 65535 || tiles > 65535 {
        return None;
    }
    Some(IqLaunch {
        grid: (tiles as u32, nseg as u32, 1),
        block: (32 * ROWS as u32, 1, 1),
    })
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 1 リクエストぶんの観測 (task 144 の REPORT 用)
// ---------------------------------------------------------------------------
//
// ヒット率は「層で参照した expert のうち VRAM に載っていた割合」。GPU に倒した
// トークンだけでなく CPU のトークンも数える (載っていないと永久に上がらないため、
// 上げ直しの効果を同じ物差しで見たい)。台帳 (`DENNO_Q4_PROFILE`) とは独立に、
// リクエスト終端で 1 行だけ出す。

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static GPU_LAYERS: AtomicU64 = AtomicU64::new(0);
static GPU_EXPERTS: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんの参照を積む (`lookups` = 選択 expert 数、`hits` = 載っていた数)。
pub fn note_lookup(lookups: u32, hits: u32) {
    LOOKUPS.fetch_add(lookups as u64, Ordering::Relaxed);
    HITS.fetch_add(hits as u64, Ordering::Relaxed);
}

/// 層一括 GEMV を 1 層ぶん回したことを積む (`n` = GPU で回した expert 数)。
pub fn note_gpu_layer(n: u32) {
    GPU_LAYERS.fetch_add(1, Ordering::Relaxed);
    GPU_EXPERTS.fetch_add(n as u64, Ordering::Relaxed);
}

/// task 261: 層一括 (旧 4 型) 経路が 1 層を GPU で完了した回数。
static BATCH_OK: AtomicU64 = AtomicU64::new(0);

/// task 261: 層一括経路が expert を載せられず per-expert 経路へ落ちた回数。
static BATCH_BAIL: AtomicU64 = AtomicU64::new(0);

/// 層一括経路が 1 層を回し切った。
pub fn note_batch_ok() {
    BATCH_OK.fetch_add(1, Ordering::Relaxed);
}

/// 層一括経路が途中で諦めた (1 本でも VRAM に載らなかった)。
pub fn note_batch_bail() {
    BATCH_BAIL.fetch_add(1, Ordering::Relaxed);
}

/// 上げ直し (H2D) を 1 本ぶん積む。
pub fn note_upload() {
    UPLOADS.fetch_add(1, Ordering::Relaxed);
}

/// 上げ直しに使ったホスト時間 (ns)。acquire = ページフォルト + H2D。
static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

/// 上げ直しのホスト時間を積む。
pub fn note_upload_ns(ns: u64) {
    UPLOAD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行。ゲートが off / 参照 0 件なら `None`。
pub fn stats_line() -> Option<String> {
    if !iq_gpu_enabled() {
        return None;
    }
    let l = LOOKUPS.load(Ordering::Relaxed);
    if l == 0 {
        return None;
    }
    let h = HITS.load(Ordering::Relaxed);
    let layers = GPU_LAYERS.load(Ordering::Relaxed);
    let exps = GPU_EXPERTS.load(Ordering::Relaxed);
    let up = UPLOADS.load(Ordering::Relaxed);
    let vram = match crate::model::expert_vram::budget_usage() {
        Some((used, pinned, cap)) => format!(
            "vram={}MiB/{}MiB pin={}MiB",
            used >> 20,
            cap >> 20,
            pinned >> 20
        ),
        None => "vram=-".to_string(),
    };
    let up_ms = UPLOAD_NS.load(Ordering::Relaxed) as f64 / 1e6;
    // task 261: 載らなかった理由の内訳 (`type` = カーネルが無い型 / `budget` =
    // 予算に 1 本が入らない / `alloc` = 実メモリの確保に失敗)。
    #[cfg(feature = "cuda")]
    let (fb, fb_ty, fb_budget, fb_alloc) = crate::model::expert_vram::fallback_breakdown();
    #[cfg(not(feature = "cuda"))]
    let (fb, fb_ty, fb_budget, fb_alloc) = (0u64, 0u64, 0u64, 0u64);
    // task 794: 載せ替え方針と、頻度ゲートで載せなかった回数 (発火の証拠)。
    let pol = crate::model::expert_vram::policy_stats()
        .map(|(p, rej, _)| format!("policy={} adm_rej={rej}", p.as_str()))
        .unwrap_or_else(|| "policy=-".to_string());
    // task 261: 旧 4 型 (Q4_K / Q5_K / Q5_1 / Q8_0) の層一括経路の当たり外れ。
    let batch_ok = BATCH_OK.load(Ordering::Relaxed);
    let batch_bail = BATCH_BAIL.load(Ordering::Relaxed);
    Some(format!(
        "[exp-iq] lookups={l} hits={h} hit_rate={:.3} gpu_layers={layers} gpu_experts={exps} uploads={up} up_ms={up_ms:.1} batch_ok={batch_ok} batch_bail={batch_bail} fb={fb}(type={fb_ty} budget={fb_budget} alloc={fb_alloc}) {pol} {vram}",
        h as f64 / l as f64
    ))
}

/// task 271: 直前のリクエストの参照数と VRAM ヒット数。
///
/// `stats_line` と同じ積算を、整形せずに返す。`/v1/experts/stats` が
/// 「直前のリクエストの tier 別ヒット率」を出すのに使う (リクエスト終端で
/// [`stats_reset`] が 0 に戻す前に写す)。ゲート off でも 0/0 を返すだけ。
pub fn lookups_hits() -> (u64, u64) {
    (
        LOOKUPS.load(Ordering::Relaxed),
        HITS.load(Ordering::Relaxed),
    )
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [
        &LOOKUPS,
        &HITS,
        &GPU_LAYERS,
        &GPU_EXPERTS,
        &UPLOADS,
        &UPLOAD_NS,
        &MISS_LAYERS,
        &MISS_N,
        &MISS_NS,
        &UPLOAD_LAYERS,
        &UPLOAD_N,
        &UPLOAD_LAYERS_NS,
        &BATCH_OK,
        &BATCH_BAIL,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    for h in 0..=MAX_HIT_BUCKET {
        MH_LAYERS[h].store(0, Ordering::Relaxed);
        MH_MISS[h].store(0, Ordering::Relaxed);
        MH_MISS_NS[h].store(0, Ordering::Relaxed);
        MH_UP[h].store(0, Ordering::Relaxed);
        MH_UP_NS[h].store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// task 190: ヒット本数ごとの「ミス CPU 計算」と「上げ直し」
// ---------------------------------------------------------------------------
//
// `exp_miss` / `exp_upload` は台帳では 1 本の数字しか出ないので、
// 「ヒット率が変わるとどう変わるか」が読めない。ここでは 1 層ぶんの実測を
// ヒット本数 (GPU で回した expert 数) のバケットへ積み、リクエスト終端に
// 1 行で出す。計測は `DENNO_Q4_PROFILE` のときだけ (Instant 2 回/層)。
//
// 列: ヒット本数 / 層数 / ミス本数 / ミス CPU の合計 ms / 1 トークンあたり /
//     上げ直した本数 / 上げ直しの合計 ms / 1 本あたり us。

/// ヒット本数のバケット上限 (n_expert_used = 10 まで)。
pub const MAX_HIT_BUCKET: usize = 10;

static MH_LAYERS: [AtomicU64; MAX_HIT_BUCKET + 1] =
    [const { AtomicU64::new(0) }; MAX_HIT_BUCKET + 1];
static MH_MISS: [AtomicU64; MAX_HIT_BUCKET + 1] =
    [const { AtomicU64::new(0) }; MAX_HIT_BUCKET + 1];
static MH_MISS_NS: [AtomicU64; MAX_HIT_BUCKET + 1] =
    [const { AtomicU64::new(0) }; MAX_HIT_BUCKET + 1];
static MH_UP: [AtomicU64; MAX_HIT_BUCKET + 1] =
    [const { AtomicU64::new(0) }; MAX_HIT_BUCKET + 1];
static MH_UP_NS: [AtomicU64; MAX_HIT_BUCKET + 1] =
    [const { AtomicU64::new(0) }; MAX_HIT_BUCKET + 1];

/// 直近の層ぶんの合計 (行を出すときの分母に使う)。
static MISS_LAYERS: AtomicU64 = AtomicU64::new(0);
static MISS_N: AtomicU64 = AtomicU64::new(0);
static MISS_NS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_LAYERS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_N: AtomicU64 = AtomicU64::new(0);
static UPLOAD_LAYERS_NS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんのミス CPU 計算を積む (`hits` = GPU で回した本数)。
pub fn note_layer_miss(hits: u32, n_miss: u32, ns: u64) {
    if !crate::model::q4ledger::enabled() {
        return;
    }
    let h = (hits as usize).min(MAX_HIT_BUCKET);
    MH_LAYERS[h].fetch_add(1, Ordering::Relaxed);
    MH_MISS[h].fetch_add(n_miss as u64, Ordering::Relaxed);
    MH_MISS_NS[h].fetch_add(ns, Ordering::Relaxed);
    MISS_LAYERS.fetch_add(1, Ordering::Relaxed);
    MISS_N.fetch_add(n_miss as u64, Ordering::Relaxed);
    MISS_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 層ぶんの上げ直し (ページフォルト + H2D) を積む。
pub fn note_layer_upload(hits: u32, n_up: u32, ns: u64) {
    if !crate::model::q4ledger::enabled() {
        return;
    }
    let h = (hits as usize).min(MAX_HIT_BUCKET);
    MH_UP[h].fetch_add(n_up as u64, Ordering::Relaxed);
    MH_UP_NS[h].fetch_add(ns, Ordering::Relaxed);
    UPLOAD_LAYERS.fetch_add(1, Ordering::Relaxed);
    UPLOAD_N.fetch_add(n_up as u64, Ordering::Relaxed);
    UPLOAD_LAYERS_NS.fetch_add(ns, Ordering::Relaxed);
}

/// task 190: ヒット本数ごとの表 (PROFILE のときだけ 1 行)。
///
/// `tok` は窓のトークン数 (128)。「1 トークンあたり」の列に使う。
pub fn hit_lines(tok: u64) -> Vec<String> {
    if !crate::model::q4ledger::enabled() || !iq_gpu_enabled() {
        return Vec::new();
    }
    let layers = MISS_LAYERS.load(Ordering::Relaxed);
    if layers == 0 {
        return Vec::new();
    }
    let tok = tok.max(1);
    let mut s = String::from(
        "[exp-iq-hit] ヒット本数ごと (gpu expert 数): 層 ミス ミスCPUms/tok 上げ本 上げms/tok us/本",
    );
    for h in 0..=MAX_HIT_BUCKET {
        let n = MH_LAYERS[h].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let miss = MH_MISS[h].load(Ordering::Relaxed);
        let miss_ms = MH_MISS_NS[h].load(Ordering::Relaxed) as f64 / 1e6;
        let up = MH_UP[h].load(Ordering::Relaxed);
        let up_ms = MH_UP_NS[h].load(Ordering::Relaxed) as f64 / 1e6;
        let us_up = if up > 0 {
            up_ms * 1e3 / up as f64
        } else {
            0.0
        };
        s.push_str(&format!(
            " | h{h} {n}層 {miss}本 {:.3} {up}本 {:.3} {us_up:.0}us",
            miss_ms / tok as f64,
            up_ms / tok as f64
        ));
    }
    s.push_str(&format!(
        " || 計 {layers}層 ミス{}本/{}層 {:.3}ms/tok 上げ{}本 {:.3}ms/tok",
        MISS_N.load(Ordering::Relaxed),
        MISS_LAYERS.load(Ordering::Relaxed),
        MISS_NS.load(Ordering::Relaxed) as f64 / 1e6 / tok as f64,
        UPLOAD_N.load(Ordering::Relaxed),
        UPLOAD_LAYERS_NS.load(Ordering::Relaxed) as f64 / 1e6 / tok as f64
    ));
    vec![s]
}

// カーネルのミラー (GPU 無しで索引と累積順を確かめるため)
// ---------------------------------------------------------------------------

/// カーネルが 1 レーンぶん読む要素数 (256 要素ブロックの型)。
const LANE_ELEMS: usize = 8;

/// カーネルと同じ手順で 1 ブロック (`blk`) の 256 要素 (IQ4_NL は 32、Q2_0 は 64) を
/// `out_block` に書く。`lane` は 0..32。
///
/// 返り値は書いた要素数 (`None` ならこの型はミラー未対応)。
#[allow(clippy::too_many_arguments)]
fn mirror_block(ty: GgmlType, blk: &[u8], lane: usize, out_block: &mut [f32]) -> Option<usize> {
    use crate::kquant_tables as t;
    match ty {
        GgmlType::IQ2S => {
            // d(2) + qs[32] + signs[32] + qh[8] + scales[8]
            let d = f16(blk, 0);
            let ib32 = lane >> 2;
            let l4 = lane & 3;
            let qh = blk[66 + ib32] as u32;
            let idx = blk[2 + lane] as usize | (((qh << (8 - 2 * l4)) & 0x300) as usize);
            let sg = blk[34 + lane];
            let sc = blk[74 + ib32];
            let nib = if l4 & 2 != 0 { sc >> 4 } else { sc & 0x0f };
            let db = d * (0.5 + nib as f32) * 0.25;
            let grid = t::IQ2S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                let gv = grid[j] as f32;
                let sgn = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                out_block[lane * LANE_ELEMS + j] = db * gv * sgn;
            }
            Some(256)
        }
        GgmlType::IQ3Xxs => {
            let d = f16(blk, 0);
            let ib32 = lane >> 2;
            let l4 = lane & 3;
            let aux = u32::from_le_bytes([
                blk[66 + ib32 * 4],
                blk[67 + ib32 * 4],
                blk[68 + ib32 * 4],
                blk[69 + ib32 * 4],
            ]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            let signs = t::KSIGNS_IQ2XS[((aux >> (7 * l4)) & 127) as usize];
            let g1 = t::IQ3XXS_GRID[blk[2 + ib32 * 8 + 2 * l4] as usize].to_le_bytes();
            let g2 = t::IQ3XXS_GRID[blk[2 + ib32 * 8 + 2 * l4 + 1] as usize].to_le_bytes();
            for j in 0..4 {
                let s0 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & (1 << (j + 4)) != 0 {
                    -1.0
                } else {
                    1.0
                };
                out_block[lane * LANE_ELEMS + j] = db * g1[j] as f32 * s0;
                out_block[lane * LANE_ELEMS + 4 + j] = db * g2[j] as f32 * s1;
            }
            Some(256)
        }
        GgmlType::IQ3S => {
            let d = f16(blk, 0);
            let g = lane >> 2;
            let l4 = lane & 3;
            let sc = blk[106 + (g >> 1)];
            let nib = if g & 1 != 0 { sc >> 4 } else { sc & 0x0f };
            let db = d * (1.0 + 2.0 * nib as f32);
            let qh = blk[66 + g] as u32;
            let idx1 = blk[2 + g * 8 + 2 * l4] as usize | (((qh << (8 - 2 * l4)) & 256) as usize);
            let idx2 =
                blk[2 + g * 8 + 2 * l4 + 1] as usize | (((qh << (7 - 2 * l4)) & 256) as usize);
            let signs = blk[74 + g * 4 + l4];
            let g1 = t::IQ3S_GRID[idx1].to_le_bytes();
            let g2 = t::IQ3S_GRID[idx2].to_le_bytes();
            for j in 0..4 {
                let s0 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & (1 << (j + 4)) != 0 {
                    -1.0
                } else {
                    1.0
                };
                out_block[lane * LANE_ELEMS + j] = db * g1[j] as f32 * s0;
                out_block[lane * LANE_ELEMS + 4 + j] = db * g2[j] as f32 * s1;
            }
            Some(256)
        }
        GgmlType::IQ4Xs => {
            let d = f16(blk, 0);
            let g = lane >> 2;
            let h4 = lane & 3;
            let half = h4 >> 1;
            let jbase = (h4 & 1) * 8;
            let sh = blk[2] as u32 | ((blk[3] as u32) << 8);
            let sl = blk[4 + (g >> 1)] as u32;
            let ls = ((sl >> (4 * (g & 1))) & 0x0f) | (((sh >> (2 * g)) & 3) << 4);
            let dl = d * (ls as i32 - 32) as f32;
            for j in 0..8 {
                let byte = blk[8 + g * 16 + jbase + j];
                let q = if half != 0 { byte >> 4 } else { byte & 0x0f } as usize;
                let gv = t::KVALUES_IQ4NL[q] as f32;
                out_block[lane * LANE_ELEMS + j] = dl * gv;
            }
            Some(256)
        }
        GgmlType::IQ4Nl => {
            // レーン 1 本 = 1 ブロック (32 要素)。`out_block` は 32 要素。
            let d = f16(blk, 0);
            for j in 0..16 {
                let lo = (blk[2 + j] & 0x0f) as usize;
                let hi = (blk[2 + j] >> 4) as usize;
                out_block[j] = d * t::KVALUES_IQ4NL[lo] as f32;
                out_block[16 + j] = d * t::KVALUES_IQ4NL[hi] as f32;
            }
            Some(32)
        }
        GgmlType::Q2_0 => {
            // レーン 1 本 = 1 ブロック (64 要素)。
            let d = f16(blk, 0);
            for j in 0..64 {
                let q = ((blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 3) as i32;
                out_block[j] = (q - 1) as f32 * d;
            }
            Some(64)
        }
        _ => None,
    }
}

/// 行のバイト列を 1 行ぶんデコードする (カーネルと同じ索引)。
///
/// `out.len()` は `k`。要素の並びは [`crate::kquant::dequant_row`] と一致する
/// (テストがビット一致で確かめる)。
pub fn dequant_mirror(ty: GgmlType, row: &[u8], k: usize, out: &mut [f32]) -> bool {
    let Some(bs) = mirror_block_size(ty) else {
        return false;
    };
    if out.len() != k || k % bs != 0 {
        return false;
    }
    let nb = k / bs;
    let mut block = vec![0.0f32; bs];
    for b in 0..nb {
        let bytes = &row[b * ty.type_size()..(b + 1) * ty.type_size()];
        if bs == 256 {
            // 32 レーンが 8 要素ずつ (lane * 8 + j の位置へ書く)。
            for lane in 0..32 {
                mirror_block(ty, bytes, lane, &mut block).expect("mirror");
            }
        } else {
            // 1 レーン = 1 ブロック。ブロック内の並びをそのまま使う。
            mirror_block(ty, bytes, 0, &mut block).expect("mirror");
        }
        out[b * bs..(b + 1) * bs].copy_from_slice(&block);
    }
    true
}

/// 1 ブロックの要素数 (カーネルのレーン割り当ての単位)。
pub fn mirror_block_size(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::IQ2S | GgmlType::IQ3Xxs | GgmlType::IQ3S | GgmlType::IQ4Xs => Some(256),
        GgmlType::IQ4Nl => Some(32),
        GgmlType::Q2_0 => Some(64),
        _ => None,
    }
}

/// カーネルの累積順そのままで 1 行の内積を取る (テスト用)。
///
/// レーンごとに自分の要素を昇順に足し、最後に `__shfl_down_sync` と同じ木で束ねる。
pub fn dot_mirror(ty: GgmlType, row: &[u8], k: usize, x: &[f32]) -> Option<f32> {
    let bs = mirror_block_size(ty)?;
    if x.len() != k || k % bs != 0 {
        return None;
    }
    let nb = k / bs;
    let mut lane_acc = [0.0f32; 32];
    let mut block = vec![0.0f32; bs];
    for b in 0..nb {
        let bytes = &row[b * ty.type_size()..(b + 1) * ty.type_size()];
        let xb = &x[b * bs..(b + 1) * bs];
        if bs == 256 {
            for lane in 0..32 {
                mirror_block(ty, bytes, lane, &mut block).expect("mirror");
                for j in 0..8 {
                    lane_acc[lane] += block[lane * 8 + j] * xb[lane * 8 + j];
                }
            }
        } else {
            mirror_block(ty, bytes, 0, &mut block).expect("mirror");
            // 1 レーン = 1 ブロック: レーン b % 32 がこのブロックを持つ。
            let lane = b % 32;
            for j in 0..bs {
                lane_acc[lane] += block[j] * xb[j];
            }
        }
    }
    // __shfl_down_sync と同じ木 (同時更新)。
    let mut a = lane_acc;
    let mut off = 16;
    while off > 0 {
        for l in 0..off {
            a[l] += a[l + off];
        }
        off >>= 1;
    }
    Some(a[0])
}

/// blk の先頭 2 バイトを f16 として f32 にする (`kquant` の `f16_to_f32` と同じ式)。
fn f16(blk: &[u8], off: usize) -> f32 {
    let h = u16::from_le_bytes([blk[off], blk[off + 1]]);
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquant;

    /// 決定的な擬似乱数 (外部クレートを使わない)。
    fn fill(seed: u64, buf: &mut [u8]) {
        let mut s = seed | 1;
        for b in buf.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 24) as u8;
        }
    }

    fn types() -> Vec<GgmlType> {
        vec![
            GgmlType::IQ2S,
            GgmlType::IQ3Xxs,
            GgmlType::IQ3S,
            GgmlType::IQ4Xs,
            GgmlType::IQ4Nl,
            GgmlType::Q2_0,
        ]
    }

    /// task 356: `DENNO_DEC_EXP_D2H_STREAM` は**既定 off** で、明示的な真値だけ on。
    ///
    /// lab の COND は `VAR=value` をそのまま env にするので、`0` / 空 / 未知の値で
    /// 黙って on になると A/B が壊れる。ここで解釈を固定する。
    #[test]
    fn dec_d2h_stream_gate_defaults_off() {
        assert!(!dec_d2h_stream_from_value(None), "未設定は off");
        for v in ["", "0", "off", "false", "no", "2", "abc", " ON?"] {
            assert!(!dec_d2h_stream_from_value(Some(v)), "{v:?} は off");
        }
        for v in ["1", "on", "true", "yes", " ON ", "True"] {
            assert!(dec_d2h_stream_from_value(Some(v)), "{v:?} は on");
        }
    }

    /// task 356: 積算が 0 のときは行を出さず、コピーを数えると 1 行になる
    /// (ゲート on でも「分岐が発火したか」を数字で読めるようにする)。
    #[test]
    fn dec_d2h_stream_line_counts_copies() {
        d2h_stream_reset();
        assert!(d2h_stream_line_with(true).is_none(), "0 本なら行を出さない");
        note_d2h_stream_copy();
        assert!(
            d2h_stream_line_with(false).is_none(),
            "ゲート off は数えていても行を出さない"
        );
        let line = d2h_stream_line_with(true).expect("ゲート on なら行が出る");
        assert!(line.contains("copies=1"), "{line}");
        note_d2h_stream_wait(2_000_000);
        note_d2h_stream_fallback();
        let line = d2h_stream_line_with(true).expect("行が出る");
        assert!(line.contains("wait=2.00ms"), "{line}");
        assert!(line.contains("fallback=1"), "{line}");
        assert!(line.contains("us/copy=2000.0"), "{line}");
        d2h_stream_reset();
        assert!(d2h_stream_line_with(true).is_none(), "0 に戻したら行を出さない");
    }

    /// task 356: `DENNO_DEC_EXP_SUM_ZC` は**既定 on** (lab t356b で採否 yes)。
    /// 明示的な偽値だけ切る (lab の COND は `VAR=value` をそのまま env にする)。
    #[test]
    fn sum_zc_gate_defaults_on() {
        assert!(sum_zc_from_value(None), "未設定は on");
        for v in ["0", "off", "false", "no", " OFF ", "False"] {
            assert!(!sum_zc_from_value(Some(v)), "{v:?} は off");
        }
        for v in ["", "1", "on", "true", "yes", "2", "abc"] {
            assert!(sum_zc_from_value(Some(v)), "{v:?} は on (偽値だけ切る)");
        }
    }

    /// task 356: 和の ZC の積算 (0 層なら行を出さない / 数えると出る)。
    #[test]
    fn sum_zc_line_counts_layers() {
        sum_zc_reset();
        assert!(sum_zc_line_with(true).is_none(), "0 層なら行を出さない");
        note_sum_zc_layer();
        note_sum_zc_layer();
        assert!(
            sum_zc_line_with(false).is_none(),
            "ゲート off は数えていても行を出さない"
        );
        let line = sum_zc_line_with(true).expect("ゲート on なら行が出る");
        assert!(line.contains("sum_zc_layers=2"), "{line}");
        note_sum_zc_fallback();
        let line = sum_zc_line_with(true).expect("行が出る");
        assert!(line.contains("fallback=1"), "{line}");
        sum_zc_reset();
        assert!(sum_zc_line_with(true).is_none(), "0 に戻したら行を出さない");
    }

    /// task 388: `DENNO_DEC_MISS_SPAWN` は**既定 on** (lab t388b で採否 yes)、
    /// 明示的な偽値だけ切る。`DENNO_DEC_MISS_SERIAL` は既定 off (診断用)。
    #[test]
    fn miss_spawn_gates_parse_explicitly() {
        assert!(miss_spawn_from_value(None), "既定 on");
        assert!(miss_spawn_from_value(Some("")), "空は on");
        for v in ["0", "off", "false", "no", " OFF ", "False"] {
            assert!(!miss_spawn_from_value(Some(v)), "{v:?} は off");
        }
        for v in ["1", "on", "true", "yes", "2", "abc"] {
            assert!(miss_spawn_from_value(Some(v)), "{v:?} は on (偽値だけ切る)");
        }
        // serial は診断用なので既定 off (明示的な真値だけ on)。
        assert!(!on_from_value(None));
        assert!(!on_from_value(Some("0")));
        assert!(on_from_value(Some("1")));
        assert!(on_from_value(Some("TRUE")));
    }

    /// task 388: 発火した層数を数えて 1 行で出す (ゲート off / 0 層なら出さない)。
    #[test]
    fn miss_spawn_line_counts_fired_layers() {
        miss_spawn_reset();
        assert!(miss_spawn_line_with(true, true, 0, 0).is_none(), "0 層なら出さない");
        note_miss_spawn_layer();
        note_miss_spawn_layer();
        note_miss_serial_layer();
        let spawn = MISS_SPAWN_LAYERS.load(Ordering::Relaxed);
        let ser = MISS_SERIAL_LAYERS.load(Ordering::Relaxed);
        assert_eq!((spawn, ser), (2, 1), "数える");
        let line = miss_spawn_line_with(true, false, spawn, ser).expect("ゲート on なら出る");
        assert!(line.contains("spawn_layers=2"), "{line}");
        assert!(line.contains("serial_layers=1"), "{line}");
        assert!(
            miss_spawn_line_with(false, false, spawn, ser).is_none(),
            "ゲート off は数えていても出さない"
        );
        miss_spawn_reset();
        assert_eq!(MISS_SPAWN_LAYERS.load(Ordering::Relaxed), 0);
        assert_eq!(MISS_SERIAL_LAYERS.load(Ordering::Relaxed), 0);
    }


    /// ミラーの索引が `kquant::dequant_row` とビット一致する (全 6 型 x 実形の k)。
    #[test]
    fn mirror_matches_dequant_row() {
        for ty in types() {
            let ts = ty.type_size();
            for k in [256usize, 640, 2560, 5120] {
                if k % ty.blck_size() != 0 {
                    continue;
                }
                let nblk = k / ty.blck_size();
                let mut row = vec![0u8; nblk * ts];
                fill(
                    0x9e37_79b9_7f4a_7c15u64.wrapping_mul(k as u64 + 1),
                    &mut row,
                );
                // f16 のスケールが inf/NaN になると比較が意味を失うので指数部を抑える。
                for b in 0..nblk {
                    row[b * ts + 1] &= 0x3b; // 指数の上位ビットを落とす (|exp| を小さく)
                }
                let want = kquant::dequant_row(ty, &row, k);
                let mut got = vec![0.0f32; k];
                assert!(
                    dequant_mirror(ty, &row, k, &mut got),
                    "{ty:?} k={k}: mirror 未対応"
                );
                for i in 0..k {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "{ty:?} k={k} 要素 {i}: mirror と dequant_row がずれた"
                    );
                }
                assert!(
                    got.iter().any(|v| *v != 0.0),
                    "{ty:?}: 全部 0 では比較にならない"
                );
            }
        }
    }

    /// ミラーの内積が逐次 f32 内積と相対 1e-5 以内 (累積順が違うだけ)。
    #[test]
    fn dot_mirror_is_close_to_sequential_dot() {
        for ty in types() {
            let k = 2560usize;
            let ts = ty.type_size();
            let nblk = k / ty.blck_size();
            let mut row = vec![0u8; nblk * ts];
            fill(0x1234_5678_9abc_def1, &mut row);
            for b in 0..nblk {
                row[b * ts + 1] &= 0x3b;
            }
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
                .collect();
            let w = kquant::dequant_row(ty, &row, k);
            let mut seq = 0.0f32;
            for i in 0..k {
                seq += w[i] * x[i];
            }
            let got = dot_mirror(ty, &row, k, &x).expect("dot");
            let denom = seq.abs().max(1e-6);
            assert!(
                (got - seq).abs() / denom < 1e-5,
                "{ty:?}: mirror {got} と逐次 {seq} が離れすぎ"
            );
        }
    }

    /// スロット・行長・起動形の基本。
    #[test]
    fn slots_and_launch_plan() {
        assert_eq!(slot(GgmlType::IQ2S), Some(0));
        assert_eq!(slot(GgmlType::Q2_0), Some(5));
        assert_eq!(slot(GgmlType::Q4K), None, "既存 4 型は IQ 経路の担当外");
        assert!(kernel_name(0).unwrap().starts_with("dno_exp_gemv_"));
        assert!(kernel_name(SLOTS).is_none());
        // gate (k=2560 IQ2_S): 820 B/行、down (k=640 IQ4_NL): 360 B/行
        assert_eq!(row_bytes(GgmlType::IQ2S, 2560), Some(820));
        assert_eq!(row_bytes(GgmlType::IQ4Nl, 640), Some(360));
        assert_eq!(row_bytes(GgmlType::Q2_0, 640), Some(180));
        assert_eq!(row_bytes(GgmlType::IQ2S, 640), None, "256 で割り切れない");
        // gate/up: 640 行 -> 160 タイル、down: 2560 行 -> 640 タイル
        let p = launch_plan(10, 640).expect("plan");
        assert_eq!(p.grid, (160, 10, 1));
        assert_eq!(p.block, (128, 1, 1));
        let q = launch_plan(10, 2560).expect("plan");
        assert_eq!(q.grid, (640, 10, 1));
        assert!(launch_plan(0, 640).is_none());
        assert!(launch_plan(10, 0).is_none());
    }

    /// ホスト側が読む名前が .cu のカーネル名と一致する (名前の食い違いは
    /// 実機でしか分からないので、ソースを読んで突き合わせる)。
    #[test]
    fn kernel_names_exist_in_cu_source() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let cu = std::fs::read_to_string(dir.join("kernels_exp_iq.cu")).expect(".cu が読めない");
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert!(
                cu.contains(&format!("__global__ void {name}(")),
                "{name} が kernels_exp_iq.cu に無い"
            );
        }
        assert!(
            cu.contains(&format!("__global__ void {SILU_KERNEL}(")),
            "{SILU_KERNEL} が kernels_exp_iq.cu に無い"
        );
        // PTX の生成先 (build_ptx.sh の exp_iq) と cuda.rs が読む名前が同じであること。
        let sh = std::fs::read_to_string(dir.join("build_ptx.sh")).expect("build_ptx.sh");
        assert!(
            sh.contains("exp_iq)       echo \"kernels_exp_iq.cu"),
            "build_ptx.sh に exp_iq の規則が無い"
        );
    }

    /// コミット済みの PTX が .cu と同じ引数の数を持つ (ABI の取り違えは実機で
    /// しか分からないので、PTX の `.param` 行を数えて固定する)。
    #[test]
    fn committed_ptx_has_expected_params() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq.ptx")) else {
            // PTX 未生成のツリーでは何も言えない (カーネル無し = CPU へ落ちるだけ)。
            return;
        };
        let count = |name: &str| -> Option<usize> {
            let head = format!(".visible .entry {name}(");
            let i = ptx.find(&head)?;
            let rest = &ptx[i + head.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            Some(rest[..end].matches(".param").count())
        };
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert_eq!(count(name), Some(8), "{name} の引数の数が .cu と違う");
        }
        assert_eq!(count(SILU_KERNEL), Some(4), "{SILU_KERNEL} の引数の数");
    }

    /// task 190: v2 ゲートは**既定 on** (`DENNO_EXP_IQ_V2`、lab t190c で採用)。
    /// PTX があれば引数の数も v1 と同じ 8 / silu は 4 であることを固定する。
    #[test]
    fn v2_gate_and_ptx_params() {
        assert!(iq_v2_from_value(None));
        assert!(iq_v2_from_value(Some("1")));
        assert!(iq_v2_from_value(Some("on")));
        assert!(iq_v2_from_value(Some(" ON ")));
        assert!(!iq_v2_from_value(Some("0")));
        assert!(!iq_v2_from_value(Some("off")));
        assert!(!iq_v2_from_value(Some("false")));
        assert!(!iq_v2_from_value(Some("no")));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq_v2.ptx")) else {
            return;
        };
        let count = |name: &str| -> Option<usize> {
            let head = format!(".visible .entry {name}(");
            let i = ptx.find(&head)?;
            let rest = &ptx[i + head.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            Some(rest[..end].matches(".param").count())
        };
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert_eq!(count(name), Some(8), "v2: {name} の引数の数が .cu と違う");
        }
        assert_eq!(count(SILU_KERNEL), Some(4), "v2: {SILU_KERNEL} の引数の数");
    }

    /// task 275: v3 ゲートは**既定 off** (`DENNO_EXP_IQ_V3`)。名前は .cu に在り、
    /// v2 と同じ 8 引数であること (PTX があれば引数の数も見る)。
    #[test]
    fn v3_gate_and_kernel_names() {
        // adopted 2026-11-13: default on, explicit false values turn it off.
        assert!(iq_v3_from_value(None));
        assert!(iq_v3_from_value(Some("")));
        assert!(iq_v3_from_value(Some("1")));
        assert!(iq_v3_from_value(Some("on")));
        assert!(iq_v3_from_value(Some(" true ")));
        assert!(!iq_v3_from_value(Some("0")));
        assert!(!iq_v3_from_value(Some("off")));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let cu = std::fs::read_to_string(dir.join("kernels_exp_iq_v3.cu")).expect(".cu が読めない");
        for i in IQ_V3_SLOTS {
            let name = kernel_name(i).expect("slot");
            let full = format!("{name}_v3");
            assert!(cu.contains(&format!("__global__ void {full}(")), "{full} が .cu に無い");
        }
        // v3 が差し替えないスロットは v2 の名前のまま (取り違えの防止)。
        for i in 0..SLOTS {
            if IQ_V3_SLOTS.contains(&i) {
                continue;
            }
            let name = kernel_name(i).expect("slot");
            assert!(!cu.contains(&format!("__global__ void {name}_v3(")), "{name} は v3 の担当外");
        }
        // PTX があれば引数 8 本 (v1/v2 と同じ segment 表)。
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq_v3.ptx")) else {
            return;
        };
        for i in IQ_V3_SLOTS {
            let name = format!("{}_v3", kernel_name(i).expect("slot"));
            let head = format!(".visible .entry {name}(");
            let j = ptx.find(&head).expect("v3 PTX に kernel が無い");
            let rest = &ptx[j + head.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            assert_eq!(rest[..end].matches(".param").count(), 8, "{name} の引数の数");
        }
    }

    /// .h の C 配列を数値列にする (0x.. / 10 進 / 接尾辞 u ull を食う)。
    fn header_values(src: &str, name: &str) -> Vec<u64> {
        let head = format!("{name}[");
        let i = src.find(&head).unwrap_or_else(|| panic!("{name} が .h に無い"));
        let open = src[i..].find('{').expect("{ が無い") + i;
        let close = src[open..].find('}').expect("} が無い") + open;
        let body = &src[open + 1..close];
        let mut out = Vec::new();
        for tok in body.split(',') {
            let t = tok.trim().trim_end_matches("ull").trim_end_matches('u').trim();
            if t.is_empty() {
                continue;
            }
            let v = if let Some(h) = t.strip_prefix("0x") {
                u64::from_str_radix(h, 16).expect("hex")
            } else {
                t.parse::<u64>().expect("dec")
            };
            out.push(v);
        }
        out
    }

    /// task 275: 詰めた表 (`dno_iq2s_grid2` / `dno_iq3s_grid32`) が元の表と
    /// **同じ値**を与えること。GPU 無しで確かめられる部分をここで固定する。
    #[test]
    fn compact_tables_match_originals() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let h = std::fs::read_to_string(dir.join("kernels_exp_iq_tables.h")).expect(".h が読めない");
        const LUT: [u64; 3] = [8, 25, 43];
        let g = header_values(&h, "dno_iq2s_grid2");
        let full = header_values(&h, "dno_iq2s_grid");
        assert_eq!(g.len(), 1024);
        assert_eq!(full.len(), 1024);
        for i in 0..1024 {
            for j in 0..8 {
                let code = (g[i] >> (2 * j)) & 3;
                let want = (full[i] >> (8 * j)) & 0xff;
                assert_ne!(code, 3, "iq2s: code 3 が使われている (entry {i} j {j})");
                assert_eq!(LUT[code as usize], want, "iq2s: entry {i} j {j}");
            }
        }
        let g32 = header_values(&h, "dno_iq3s_grid32");
        let full3 = header_values(&h, "dno_iq3s_grid");
        assert_eq!(g32.len(), 512);
        assert_eq!(full3.len(), 512);
        for i in 0..512 {
            assert_eq!(full3[i] >> 32, 0, "iq3s: 上位 32 bit が 0 でない (entry {i})");
            assert_eq!(g32[i], full3[i] & 0xffff_ffff, "iq3s: entry {i}");
        }
    }

    /// task 190: ヒット本数ごとの表はゲート off では空。
    #[test]
    fn hit_lines_are_silent_when_off() {
        stats_reset();
        if !crate::model::q4ledger::enabled() {
            assert!(hit_lines(128).is_empty());
        }
        note_layer_miss(9, 1, 1000);
        note_layer_upload(9, 1, 1000);
        stats_reset();
        assert_eq!(MISS_N.load(Ordering::Relaxed), 0);
        assert_eq!(UPLOAD_N.load(Ordering::Relaxed), 0);
    }

    /// ゲートは既定 off、`DENNO_EXPERT_IQ_UPLOAD` の既定は 8。
    #[test]
    fn env_parsing() {
        assert!(!iq_gpu_from_value(None));
        assert!(iq_gpu_from_value(Some("1")));
        assert!(iq_gpu_from_value(Some(" on ")));
        assert!(!iq_gpu_from_value(Some("0")));
        assert_eq!(upload_from_value(None), DEFAULT_UPLOAD_PER_TOKEN);
        assert_eq!(upload_from_value(Some("")), DEFAULT_UPLOAD_PER_TOKEN);
        assert_eq!(upload_from_value(Some("0")), 0);
        assert_eq!(upload_from_value(Some("24")), 24);
        assert_eq!(upload_from_value(Some("x")), DEFAULT_UPLOAD_PER_TOKEN);
    }
}
