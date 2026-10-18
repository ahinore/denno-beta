//! 軽量セルフプロファイラ (P7.5)。
//!
//! `DENNO_PROFILE=1` を付けた場合のみ有効になり、推論の各フェーズの
//! 累積実時間と呼び出し回数を記録する。無効時は分岐 1 回のオーバーヘッド
//! もなく(コンパイラが消す)、数値経路には一切触れないため
//! exactness (ビット一致) は完全不変。
//!
//! リクエスト完了時に server 側から `summary()` を INFO ログ出力する。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const EMBED: usize = 0; // トークン→埋め込み lookup
pub const RMSNORM: usize = 1; // 各層の RMSNorm (2回/層) + 最終 (FINAL_NORM は別)
pub const ROPE: usize = 2; // RoPE 回転 (q, k)
pub const QKV: usize = 3; // Q/K/V 投影 (線形層)
pub const ATTN: usize = 4; // attention 本体 (KV書き込み、scores、softmax、probs@V)
pub const OPROJ: usize = 5; // 出力投影
pub const GATE_UP: usize = 6; // MLP gate/up 投影
pub const SWIGLU: usize = 7; // SiLU + 要素積
pub const DOWN: usize = 8; // MLP down 投影
pub const FINAL_NORM: usize = 9; // 最終 RMSNorm
pub const LM_HEAD: usize = 10; // 最終 logits 投影 (vocab 次元 matvec)
pub const SAMPLER: usize = 11; // サンプリング
pub const DETOK: usize = 12; // トークン→テキスト デコード (ストリーミング差分用)
pub const QKNORM: usize = 13; // QK-Norm (Qwen3 のみ。q/k の head_dim 単位 RMSNorm)
pub const D2H: usize = 14; // s49: デバイス→ホスト戻し (chain の logits download + staged 状態フラッシュ)
/// s54: d2h_state の細分 (倍率 10)。生 id (D2H, D2H+1, …) は変更しない —
/// telemetry の OpSummary / denno-view は生 id で扱うため、細分は
/// `summary()` のテキスト出力と telemetry の op 記録 (telemetry_add_op) だけに
/// 掛ける (バイト互換維持)。ON 無効時は分岐 1 回で従来どおり D2H に積む。
///
/// s54 の実測 (9B int8・decode 232 step・PROFILE): s49 の `d2h_state` バケットは
/// **step あたり 2 call** = ①`chain_download(CH_LOGITS)` の sync (ステップ全体の
/// GPU 実行完了待ちを吸収する支配項) ②`chain_q35_kv_stage_flush` (ステップ末尾の
/// ホスト反映。`chain_q35_fullattn` が層ごとに出す staged async D2H は非同期
/// 発行のみで同期点を持たないため、計上は ①に集まる)。full 層の k/v 行
/// (16 層 × 2 × kv_dim 4B = 128KB/step) は ①の sync が完了を待つ転送であり、
/// その分が decode の律速に効く — ①の内訳値を D2H_LOGITS / D2H_KV_STAGE /
/// D2H_KV_FLUSH で分離して実測するための id。
pub const D2H_LOGITS: usize = 10 * D2H; // logits の chain_download (sync 待ちを吸収)
pub const D2H_KV_STAGE: usize = 10 * D2H + 1; // full 層 k/v 行の staged async D2H (per 層)
pub const D2H_KV_FLUSH: usize = 10 * D2H + 2; // ステップ末尾の staged → ホスト反映
pub const D2H_GDN_FLUSH: usize = 10 * D2H + 3; // 遅延 GDN 状態の回復 (prefill 冒頭)
pub const D2H_KV_DELAYED_FLUSH: usize = 10 * D2H + 4; // 遅延 KV 行の回復 (将来用)
pub const D2H_ATTN_PF: usize = 10 * D2H + 5; // prefill attention 出力の D2H
pub const D2H_N: usize = 6; // 細分バケット数 (D2H_LOGITS .. D2H_ATTN_PF)
pub const N_OPS: usize = 15;

const NAMES: [&str; N_OPS] = [
    "embed", "rmsnorm", "rope", "qkv_proj", "attn_core", "o_proj", "gate_up_proj", "swiglu",
    "down_proj", "final_norm", "lm_head", "sampler", "detok", "qk_norm", "d2h_state",
];

/// s54: d2h_state の細分名 (id は 10*D2H + i)。
const D2H_NAMES: [&str; D2H_N] = [
    "d2h_logits",
    "d2h_kv_stage",
    "d2h_kv_flush",
    "d2h_gdn_flush",
    "d2h_kv_dflush",
    "d2h_attn_pf",
];

static NS: [AtomicU64; N_OPS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; N_OPS]
};
static CALLS: [AtomicU64; N_OPS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; N_OPS]
};
/// s54: d2h_state の細分累積 (id は 10*D2H + i → 添字 i)。
static D2H_SUB_NS: [AtomicU64; D2H_N] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; D2H_N]
};
static D2H_SUB_CALLS: [AtomicU64; D2H_N] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; D2H_N]
};

static ON: OnceLock<bool> = OnceLock::new();

pub fn enabled() -> bool {
    *ON.get_or_init(|| std::env::var("DENNO_PROFILE").map(|v| v != "0").unwrap_or(false))
}

/// `f` の実行時間を op に計上して結果を返す。
/// (P13: テレメトリ有効時は op 時間を telemetry ハブにも累積する。
/// どちらも計測のみで数値経路には触れないため exactness は不変)
pub fn time<T>(op: usize, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        // s54: d2h の細分 id は NS の範囲外 — 生 id で受ける (下の summary が集約)
        let slot = if op < N_OPS { op } else { D2H };
        NS[slot].fetch_add(ns, Ordering::Relaxed);
        CALLS[slot].fetch_add(1, Ordering::Relaxed);
        if op >= N_OPS {
            D2H_SUB_NS[op - D2H_LOGITS].fetch_add(ns, Ordering::Relaxed);
            D2H_SUB_CALLS[op - D2H_LOGITS].fetch_add(1, Ordering::Relaxed);
        }
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
    }
    r
}

/// `time` の層付き版 (P14: LayerRun フレーム用)。層ループ内の op に使う。
/// op 時間は OpSummary 累積に加算され、さらに telemetry の層バッファに
/// 実行順で記録される (`telemetry_layer_begin` 〜 `telemetry_layer_flush` 間)。
/// layer 引数は呼び出し側の文脈明示用 (記録は層バッファ側が担当する)。
pub fn time_layer<T>(op: usize, _layer: usize, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        NS[op].fetch_add(ns, Ordering::Relaxed);
        CALLS[op].fetch_add(1, Ordering::Relaxed);
    }
    if tel_on {
        crate::telemetry_add_op(op, ns);
        denno_telemetry::record_layer_op(op, ns);
    }
    r
}

/// s54: d2h 細分の計測ゲート (prof 有効 or テレメトリ有効)。無効時は分岐 1 回で
/// 生 id 経路と完全に同じ (数値経路に接触しない)。
#[inline]
pub fn op_on() -> bool {
    enabled() || crate::telemetry_on()
}

/// s54: `op_on()` が偽なら `f` を素通しする `time` (計測の分岐を 1 箇所に閉じる)。
#[inline]
pub fn time_if_on<T>(op: usize, f: impl FnOnce() -> T) -> T {
    if op_on() {
        return time(op, f);
    }
    f()
}

pub fn reset() {
    for i in 0..N_OPS {
        NS[i].store(0, Ordering::Relaxed);
        CALLS[i].store(0, Ordering::Relaxed);
    }
    for i in 0..D2H_N {
        D2H_SUB_NS[i].store(0, Ordering::Relaxed);
        D2H_SUB_CALLS[i].store(0, Ordering::Relaxed);
    }
}

/// 整形済みサマリ (合計時間降順、累積%) を返す。
/// s54: d2h_state の直後にその細分行をインデントして出す (呼び出し回数の内訳が
/// 「毎ステップ 2 call」の正体を特定するための実測値)。
pub fn summary() -> String {
    let rows: Vec<(usize, u64, u64)> = (0..N_OPS)
        .map(|i| (i, NS[i].load(Ordering::Relaxed), CALLS[i].load(Ordering::Relaxed)))
        .filter(|(_, ns, _)| *ns > 0)
        .collect();
    let total: u64 = rows.iter().map(|(_, ns, _)| ns).sum();
    let mut sorted = rows;
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let mut s = String::from("PROFILE (cumulative, decode+prefill):\n");
    s.push_str(&format!(
        "  {:<14} {:>10} {:>8} {:>7}\n",
        "op", "ms", "calls", "cum%"
    ));
    let mut acc = 0u64;
    for (i, ns, calls) in sorted {
        acc += ns;
        s.push_str(&format!(
            "  {:<14} {:>10.1} {:>8} {:>6.1}%\n",
            NAMES[i],
            ns as f64 / 1e6,
            calls,
            acc as f64 / total.max(1) as f64 * 100.0
        ));
        if i == D2H {
            for k in 0..D2H_N {
                let sns = D2H_SUB_NS[k].load(Ordering::Relaxed);
                let sc = D2H_SUB_CALLS[k].load(Ordering::Relaxed);
                if sns == 0 && sc == 0 {
                    continue;
                }
                s.push_str(&format!(
                    "    {:<12} {:>10.1} {:>8} {:>6.1}%\n",
                    D2H_NAMES[k],
                    sns as f64 / 1e6,
                    sc,
                    sns as f64 / total.max(1) as f64 * 100.0
                ));
            }
        }
    }
    s.push_str(&format!("  TOTAL {:.1} ms\n", total as f64 / 1e6));
    s
}
