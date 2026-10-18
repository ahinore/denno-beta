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
pub const D2H_N: usize = 9; // 細分バケット数 (D2H_LOGITS .. D2H_ATTN_PF_HELD)
/// s55 (T1 step2): `d2h_logits` (= 従来 D2H_LOGITS 1 本) を「待ち」と「コピー」に
/// 分離するための id。decode 経路で logits の D2H を発行する直前に追加 sync を
/// 1 回だけ打ち (prof 有効時のみ)、その所要を WAIT に積む。従来の D2H_LOGITS
/// (連続区間) は COPY に置き換える — WAIT + COPY = 従来の D2H_LOGITS と一致する。
/// 最初のステップだけは logits D2H の前に既存 sync (lm_head 側) が無い状態から
/// 測ってしまうため、その 1 点は `HELD` に別計上する (WAIT の平均を汚さない)。
pub const D2H_LOGITS_WAIT: usize = 10 * D2H + 6; // 追加 sync のみ (GPU 側未完了処理の待ち)
pub const D2H_LOGITS_COPY: usize = 10 * D2H + 7; // 従来 D2H_LOGITS 区間 (従来計測)
pub const D2H_LOGITS_PF_HELD: usize = 10 * D2H + 8; // prefill 後 1 点目の WAIT (集計から除外)
pub const N_OPS: usize = 15;

/// s56 (T2 項目0): decode chain の logits D2H 直前に打つ追加 sync を、どちらの
/// 細分バケットに計上するか決める純関数 (GPU 不要・テスト可能)。
///
/// 従来は `cache.len() > pos0` で判定していたが、この chain は decode 1 トークン
/// だけを受け (`forward` の先頭で `tokens.len() > 1` は prefill へ分岐)、成功時に
/// `cache.advance(1)` するため **`pos0 + 1 == cache.len()` が常に成り立つ**。
/// つまり従来条件は decode では常に false になり、待ち時間が集計外の
/// `D2H_LOGITS_PF_HELD` に全部入って `d2h_logits_wait` 行が 1 度も出なかった
/// (T1 step2 の実測で判明)。
///
/// 正しい判定: 1 点目 = 「直前が prefill で、ここより前に GPU 同期点が無い」点。
/// chain に入る時点の `pos0` は直前の prefill で kv 長が進んだ後の値 = プロンプト長
/// (= キャッシュ起点が 0) か、multi-turn の prefix 再利用時は `pos0 - base_len`
/// (base ≤ プロンプト長)。どちらも `pos0 > 0` で、2 点目以降は前ステップの
/// logits D2H (COPY) が同期点になっている。よって「cache が空から進んだ (= pos0 == 0)
/// のでない限り 1 点目」= `pos0 > 0` かつ「cache が pos0 より先へ進んでいない」
/// (`cache.len() <= pos0 + 1`) を PF_HELD、それ以外 (2 点目以降、または cache が
/// 既に pos0 を超えて進んでいる) を WAIT とする。
///
/// なお `pos0 == 0` は chain では起きない (prefill が最低 1 トークン消費する) が、
/// 起きた場合は「同期点が無い」ので PF_HELD 側に倒す (平均を汚さない安全側)。
#[inline]
pub fn d2h_logits_split_op(pos0: usize, cache_len: usize) -> usize {
    if pos0 == 0 || cache_len <= pos0 + 1 {
        D2H_LOGITS_PF_HELD
    } else {
        D2H_LOGITS_WAIT
    }
}

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
    // s55: d2h_logits を「待ち」と「コピー」に分離 (合計は従来 d2h_logits と一致)
    "d2h_logits_wait",
    "d2h_logits_copy",
    "d2h_logits_pfhold",
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

#[cfg(test)]
mod tests {
    //! s55: d2h 細分 id の整合 (生 id 不変・細分は N_OPS の外・summary が 3 行を出す)。
    //! 時刻値は環境依存なので検証しない — 検証するのは id の算術と集約だけ。
    use super::*;

    /// 生 id と telemetry のバイト表現に触れていないこと (細分を足しても不変)。
    #[test]
    fn d2h_subsplit_ids_are_outside_raw_range() {
        assert_eq!(D2H, 14);
        assert_eq!(N_OPS, 15);
        let ids = [
            D2H_LOGITS,
            D2H_KV_STAGE,
            D2H_KV_FLUSH,
            D2H_GDN_FLUSH,
            D2H_KV_DELAYED_FLUSH,
            D2H_ATTN_PF,
            D2H_LOGITS_WAIT,
            D2H_LOGITS_COPY,
            D2H_LOGITS_PF_HELD,
        ];
        assert_eq!(ids.len(), D2H_N, "D2H_N は細分 id の総数と一致");
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, D2H_LOGITS + i, "細分 id は連番 (10*D2H + i)");
            assert!(id >= N_OPS, "細分 id は生 id の範囲外: {id}");
        }
    }

    /// s56 (T2 項目0): decode chain のラベル分岐の算術。
    ///
    /// chain は decode 1 トークンのみを受け、成功時に `cache.advance(1)` するので
    /// 入口では常に `pos0 + 1 == cache.len()`。従来の `cache.len() > pos0` は
    /// この不変条件により **常に false** となり、追加 sync の待ちが全部
    /// 集計外の pfhold に落ちていた (= wait 行が 1 度も出ない)。
    /// ここでは chain の呼び出し位置 (1 点目 / 2 点目以降) を引数に持たせず
    /// `(pos0, cache_len)` だけで判定が成立することを固定する:
    /// - chain 入口の不変条件 (len == pos0 + 1) では pfhold (集計外 = 1 点目扱い)
    /// - cache が pos0 より先へ進んでいる (= prefill 直後ではない) なら wait
    /// - pos0 == 0 (chain では起きない) も同期点が無いので pfhold 側
    #[test]
    fn d2h_logits_split_op_marks_first_decode_step_only() {
        // 通常: prompt 512 トークン prefill 後の 1 点目 (pos0 = 512, len = 513)
        assert_eq!(d2h_logits_split_op(512, 513), D2H_LOGITS_PF_HELD);
        // chain 入口の不変条件 (advance 前): 常に pfhold 側 (= 1 点目)
        for &p in &[1usize, 2, 511, 512, 4095] {
            assert_eq!(
                d2h_logits_split_op(p, p + 1),
                D2H_LOGITS_PF_HELD,
                "chain 入口 (len == pos0+1) は 1 点目扱い: pos0={p}"
            );
        }
        // 従来条件 `len > pos0` が真になる状況: cache が pos0 を超えて先行
        // = 前ステップの sync が既にある → wait に計上
        assert_eq!(d2h_logits_split_op(512, 514), D2H_LOGITS_WAIT);
        assert_eq!(d2h_logits_split_op(512, 600), D2H_LOGITS_WAIT);
        // 空キャッシュ (chain では起きない) は同期点が無いので pfhold 側
        assert_eq!(d2h_logits_split_op(0, 1), D2H_LOGITS_PF_HELD);
        assert_eq!(d2h_logits_split_op(0, 0), D2H_LOGITS_PF_HELD);
        // 返す id は必ず logits 細分 3 本のどれかで、生 id (N_OPS 未満) を汚さない
        for &(p, l) in &[(1usize, 2usize), (7, 7), (7, 8), (7, 9), (4096, 5000)] {
            let op = d2h_logits_split_op(p, l);
            assert!(
                [D2H_LOGITS_WAIT, D2H_LOGITS_COPY, D2H_LOGITS_PF_HELD].contains(&op),
                "細分 id 以外を返した: {op}"
            );
            assert!(op >= N_OPS, "生 id の範囲外であること: {op}");
        }
    }
    /// s55: logits の 3 分割が summary に 3 行 (呼び出し回数の内訳) として出ること。
    /// 直接叩く (整形はコード上 enabled() と無関係)。
    #[test]
    fn logits_split_is_summarized_per_bucket() {
        for (op, calls) in [
            (D2H_LOGITS_WAIT, 3u64),
            (D2H_LOGITS_COPY, 3),
            (D2H_LOGITS_PF_HELD, 1),
        ] {
            D2H_SUB_NS[op - D2H_LOGITS].fetch_add(1000, Ordering::Relaxed);
            D2H_SUB_CALLS[op - D2H_LOGITS].fetch_add(calls, Ordering::Relaxed);
        }
        NS[D2H].fetch_add(7000, Ordering::Relaxed);
        CALLS[D2H].fetch_add(7, Ordering::Relaxed);
        let s = summary();
        for (name, calls) in [
            ("d2h_logits_wait", "3"),
            ("d2h_logits_copy", "3"),
            ("d2h_logits_pfhold", "1"),
        ] {
            let cols: Vec<&str> = s
                .lines()
                .find(|l| l.trim_start().starts_with(name))
                .unwrap_or_else(|| panic!("summary に行が無い: {name}\n{s}"))
                .split_whitespace()
                .collect();
            assert_eq!(cols.len(), 4, "op/ms/calls/cum% の 4 列: {name}\n{s}");
            assert_eq!(cols[2], calls, "呼び出し回数: {name}\n{s}");
        }
        // 従来の単独行 (d2h_logits) は COPY への置換後は出ない
        assert!(
            !s.lines()
                .any(|l| l.split_whitespace().next() == Some("d2h_logits")),
            "旧 d2h_logits 単独行が出ている\n{s}"
        );
    }
}
