//! prefill 経路 (forward_prefill* / 常駐連鎖 / expert 先読み)。
//!
//! task 668: `src/model/qwen4exp.rs` から純粋に切り出した (ロジック変更なし)。

use super::*;

/// task 266: `DENNO_MIX_QCHAIN` — 量子型 (Q8_0 / Q5_1 / Q4_K / Q5_K) の mix を
/// **down → silu → up → hc_mixed の 1 同期**に畳む。既定 off。
///
/// UD-Q4_K_XL の mix は Q8_0 で、`DENNO_MIX_QUANT_GPU` の逐次経路は 1 呼び出し
/// 2 往復 (194 起動 / 194 同期 per token) だった。ここでは中間をデバイスに
/// 残したまま 4 本積む。値は CPU とビット一致 (silu は glibc expf の逐語再現)。
thread_local! {
    /// task 282: prefill の中か (`forward_prefill_inner` が立てる)。
    ///
    /// このモデルの prefill は **1 トークンずつ** `hc_mix_quant_fast` を通るので、
    /// 量子型 mix の fused chain を prefill でも使うと、chain の同期が prefill の
    /// GPU 仕事の後ろに並んで **prefill が 6〜10% 遅くなる** (lab t282a/t282b:
    /// 0.80-0.84 s → 0.89-0.94 s)。decode だけに限ると prefill は 1 バイトも
    /// 変わらない (従来のホスト経路)。
    pub(crate) static IN_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// いま prefill の中か (mix の fused chain の門番)。
pub(crate) fn in_prefill_seq() -> bool {
    IN_PREFILL.with(|c| c.get())
}

/// prefill の入口で立てるガード (Drop で元に戻す)。
pub(crate) struct PrefillSeqGuard(pub(crate) bool);

/// prefill の中に入ったことを記録する。
pub(crate) fn enter_prefill_seq() -> PrefillSeqGuard {
    PrefillSeqGuard(IN_PREFILL.with(|c| c.replace(true)))
}

/// task 126: 短い prefill の hc mix を **1 トークン経路** (decode と同じ
/// [`hc_mix_quant`]) で回すしきい値 (トークン数)。既定 [`PREFILL_MIX_SEQ_MAX`]。
///
/// prefill のバッチ経路 ([`hc_mix_quant_batch`] → [`gemv_batch`]) は、行数が
/// [`GEMV_BATCH_MIN_ROWS`] 未満の `hc_down` (320 行) を
/// [`gemv_rows_pertoken`] に落とす。そこは 1 行 1 トークンごとに
/// `vec![0.0f32; k]` を確保して行全体を f32 へ開き直すので、28 トークンでは
/// 同じ重みを 28 回デコードし、確保も 8960 回/層になる。`hc_up` (10240 行) も
/// ブロック 1 回デコードのまま行並列に乗らない (実測 `regions=0`)。
/// 1 層あたりの実測 (このコンテナ 16 スレッド、BF16 実形
/// hc_dim=10240 / low_rank=320、`bench_mix_batch_real_shape`):
///
/// | nt | batch | 1 トークン経路 | 比 |
/// |---|---|---|---|
/// | 28 | 11.6 ms | 3.2 ms | batch が 3.6 倍遅い |
/// | 64 | 9.7 ms | 7.1 ms | batch が 1.4 倍遅い |
/// | 128 | 11.9 ms | 24.4 ms | batch が 2.0 倍速い |
///
/// 交点は 64〜128 の間。このコンテナ (16 スレッド) の実測なので、既定は
/// 余裕を見て 64。
///
/// 実機 (GPU ラボ t126b, swift15-abl, 128 tok decode / 28・33 tok prefill,
/// int-auto の production env): prefill 2.20-2.24 s (28 tok) →
/// 1.40-1.46 s、2.69-2.71 s (33 tok) → 1.63-1.65 s (-35%)。
/// per-layer mix は 0.864 → 0.278 ms/トークン。decode 84 → 84 ms/tok (+0.6%)、
/// 出力 md5 は base と同一 (`DENNO_MIX_GPU=1` でも値は変わらなかった)。
pub(crate) const PREFILL_MIX_SEQ_MAX: usize = 64;

/// `DENNO_PREFILL_MIX_SEQ` の解釈。**既定 on** (t126b の実測で採択)、
/// `0` / `off` / `false` で従来のバッチ経路に戻る (`shared_overlap_from_value`
/// と同じ規約)。CPU ではバッチ経路とビット一致する。
pub(crate) fn prefill_mix_seq_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// `DENNO_PREFILL_MIX_SEQ_MAX` の解釈。未設定・数値でないときは
/// [`PREFILL_MIX_SEQ_MAX`]。`0` は「常にバッチ経路」(ゲート off と同じ)。
pub(crate) fn prefill_mix_seq_max_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(PREFILL_MIX_SEQ_MAX)
}

/// task 126: テスト用上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
pub(crate) static PREFILL_MIX_SEQ_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// task 126: テスト用のしきい値上書き (0 = env を見る)。
pub(crate) static PREFILL_MIX_SEQ_MAX_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 126: テスト専用: ゲートの on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_prefill_mix_seq_for_test(v: Option<bool>) {
    PREFILL_MIX_SEQ_OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// task 126: テスト専用: しきい値を強制する (`0` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_prefill_mix_seq_max_for_test(v: usize) {
    PREFILL_MIX_SEQ_MAX_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// task 126: 短い prefill の mix を 1 トークン経路で回してよいか
/// (`DENNO_PREFILL_MIX_SEQ`、既定 off)。
pub(crate) fn prefill_mix_seq_enabled() -> bool {
    match PREFILL_MIX_SEQ_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        prefill_mix_seq_from_value(std::env::var("DENNO_PREFILL_MIX_SEQ").ok().as_deref())
    })
}

/// task 126: 1 トークン経路に切り替えるトークン数の上限。
pub(crate) fn prefill_mix_seq_max() -> usize {
    let f = PREFILL_MIX_SEQ_MAX_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if f != 0 {
        return f;
    }
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        prefill_mix_seq_max_from_value(std::env::var("DENNO_PREFILL_MIX_SEQ_MAX").ok().as_deref())
    })
}

/// task 126: `nt` トークンのバッチ mix を 1 トークン経路へ振るか (純粋関数)。
pub(crate) fn prefill_mix_seq_for(nt: usize) -> bool {
    nt > 1 && nt <= prefill_mix_seq_max() && prefill_mix_seq_enabled()
}

/// task 309: prefill の選択 expert を **GPU の m 行カーネルで回してよいか**。
///
/// IQ 系は従来どおり ([`iq_expert_mode`])。K 系 (Q4_K / Q5_K / Q6_K / Q5_1 /
/// Q8_0) は `DENNO_PF_EXP_K=1` のときだけ、かつ**ストリーミング経路が使える
/// ときだけ**真にする (K 系のカーネルはステージング領域の m 行カーネルにしか
/// 無い。ストリーミングを切ると 1 本ずつ VRAM 常駐の expert だけを回す形になり、
/// ミスが CPU に残って遅い)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pf_expert_mode(lw: &LayerW) -> bool {
    if iq_expert_mode(lw) {
        return true;
    }
    pf_expert_mode_from(
        crate::model::exp_pf::k_enabled(),
        crate::model::exp_pf::stream_enabled(),
        [
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
        ]
        .iter()
        .all(|t| crate::model::exp_pf::slot(*t).is_some()),
    )
}

/// [`pf_expert_mode`] の純関数 (テスト用)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pf_expert_mode_from(k_gate: bool, stream: bool, k_types_ok: bool) -> bool {
    k_gate && stream && k_types_ok
}

/// task 318: `DENNO_PF_LM_K_DEV` の値の解釈 (純関数)。**既定 on** (t318c で採用)。
///
/// on のとき、層優先のデバイス MoE (`moe_ffn_quant_batch_lm_dev_at`) の前提判定が
/// `iq_expert_mode` ではなく `pf_expert_mode` になる。K 系 (UD-Q4_K_XL) は
/// これが無いと全層がホストの mixed 経路へ落ち、mixed を丸ごと往復する。
/// IQ 系では `pf_expert_mode` が `iq_expert_mode` を包含するので判定は変わらない。
pub(crate) fn pf_lm_k_dev_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => true,
        Some(s) => !matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
    }
}

/// ゲートの実値 (1 回だけ読む)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pf_lm_k_dev_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_lm_k_dev_from_value(std::env::var("DENNO_PF_LM_K_DEV").ok().as_deref())
    })
}

/// バッチ GEMV が 1 行ぶんを計算するときの入口。
///
/// **同じ量子型でも入口によってカーネルが違う**ので、呼び出し側に合わせる:
/// - [`GgufLinear::gemv`] は `moe_expert::expert_gemv` を通る。Q4_K / Q5_K だけが
///   融合カーネル、Q5_1 / Q8_0 はスカラー。ただしこのモデルのテンソルは
///   [`linear`] が `with_fast_q8(true)` で組むので、**Q8_0 だけ**は
///   `expert_gemv_q8_fast` (AVX2 融合) になる。
/// - [`ExpertW::gemv`] は `moe_expert::expert_gemv_any` を通る。Q5_1 / Q8_0 も
///   融合カーネル (AVX2+FMA) になる。
///
/// ここを取り違えると、同じバイト列・同じ `x` でも Q5_1 / Q8_0 で 1 ulp ずれる
/// (2026-10-27 に実際に踏んだ)。
#[derive(Clone, Copy)]
pub(crate) enum GemvKernel {
    /// [`GgufLinear::gemv`] と同じ入口。`fast_q8` は層の設定
    /// ([`GgufLinear::fast_q8`]) をそのまま渡す (Q8_0 で融合カーネルに載るかが
    /// 変わる)。`from_slice` で組んだ層を比べるときは、その層も同じ設定に
    /// しておくこと。
    Linear { fast_q8: bool },
    /// [`ExpertW::gemv`] と同じ入口 (`expert_gemv_any`)
    Expert,
}

/// バッチ GEMV が 1 回の呼び出しで触る `x` の上限 (バイト)。
///
/// ブロックを 1 回だけデコードして全トークンへ配る形は、1 行あたり
/// `nt * k * 4` バイトの `x` を読む。これが L2 (≈1 MB) を超えると行ごとの
/// 読み直しでキャッシュを落とし、逐次版より遅くなる (k=10240 / nt=64 で
/// 31 ms/chunk、逐次版 1.7 ms)。256 KB に切ると 1 呼び出しの `x` が L2 に
/// 収まり、重みのデコード回数も nt 回から nt/T 回に減る。
pub(crate) const GEMV_BATCH_TILE_BYTES: usize = 256 * 1024;

/// ブロック 1 回デコード版に載せる最小の行数。
///
/// 行数が少ない行列は、重みを nt 回デコードし直す代わりに行を 16 分割できる
/// 旧経路のほうが速い (実測 nt=64: n=640 で x0.88、n=1024 で x0.96、
/// n=2560 で x1.23〜1.47)。しきい値はその境目。
pub(crate) const GEMV_BATCH_MIN_ROWS: usize = 2048;

/// 量子化重みのバッチ GEMV: `out[t * n + r] = dot(row_r, x_t)`。
///
/// 行 `r` について `acc[t]` (t = 0..nt) を 0 から始め、ブロックを先頭から順に
/// **1 回だけ** f32 へ開いて、その場で全トークンへ配る。1 行ぶんの計算は
/// `kern` が指す入口 (= 逐次版と同じカーネル) なので、Q4_K の融合カーネル
/// (AVX2+FMA) を含めてビット一致する。行は互いに独立で、`x` に依存する前計算も
/// 同じ `x` から作られる。
///
/// 速くなる根拠は演算量ではなく**重みのデコードと読み込み**: 逐次版はトークン
/// ごとに同じ行を最初から読み直すが、ここでは重みのバイト列を 1 チャンクあたり
/// 1 回しか読まない。27B の expert は 1 トークンで 487 MB を触るので、ここが支配的。
pub(crate) fn gemv_rows_batch(
    kern: GemvKernel,
    ty: GgmlType,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    let _sub = q4ledger::sub(q4ledger::SUB_GEMM_CPU);
    if nt == 0 {
        return Ok(());
    }
    if xs.len() != nt * k {
        return Err(one(format!(
            "gemv_batch: x の長さ {} が nt({nt}) * k({k}) と違う",
            xs.len()
        )));
    }
    if outs.len() != nt * n {
        return Err(one(format!(
            "gemv_batch: out の長さ {} が nt({nt}) * n({n}) と違う",
            outs.len()
        )));
    }
    if n == 0 || bytes.len() % n != 0 {
        return Err(one(format!(
            "gemv_batch: 行数 {n} でバイト数 {} を割り切れない",
            bytes.len()
        )));
    }
    match kern {
        // 1 行版 (`GgufLinear::gemv`) と同じカーネルに振る。同じ型でも入口で
        // カーネルが違うので、ここを取り違えると 1 ulp ずれる ([`GemvKernel`]):
        //   Q8_0 + `fast_q8` → `expert_gemv_q8_fast` (AVX2+FMA 融合)
        //   Q8_0 (`fast_q8` 無し) / Q5_1 → `expert_gemv` のスカラー経路
        //   Q4_K / Q5_K → `expert_gemv` (AVX2+FMA があれば融合、無ければスカラー。
        //   Q5_K は P93 の融合カーネルで、`DENNO_Q5K_FAST=0` でスカラーに戻る)
        // どちらのバッチ経路も「重みのブロックを 1 回だけデコードし、その場で
        // 全トークンへ配る」ので、重みのバイト列は 1 チャンクあたり 1 回しか読まない。
        // トークンごとの累積順は 1 行版と同一 = ビット一致。
        //
        // 2026-10-28: 以前は行ごとに `expert_gemv*` を nt 回呼んでいた (行優先)。
        // トークンを外側にしていたので、同じ重み行を nt 回デコードし直していた。
        GemvKernel::Linear { fast_q8 } => {
            let scalar_only = ty == GgmlType::Q5_1 || (ty == GgmlType::Q8_0 && !fast_q8);
            // 行数が少ない形はバッチ版が行並列を使えない (gemm 側の門番は
            // `PAR_MIN_ROWS`) うえ、重みのデコードし直しより行分割の効きが大きい。
            // その場合は行を自分で分ける旧経路の方が速い (実測: k=10240 / n=320 で
            // 旧 1.7 ms/chunk、バッチ版 8.6 ms。n=640 / 1024 も旧がわずかに速い)。
            if !gemv_batch_blockdecode(n, nt) {
                gemv_rows_pertoken(ty, fast_q8, bytes, k, n, xs, nt, outs)?;
            } else {
                #[cfg(test)]
                GEMV_BATCH_BLOCKDECODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // トークンをまとめて渡すと、1 行の計算で触る x が nt * k * 4 バイトに
                // なる。k=10240 / nt=64 だと 2.6 MB で L2 に載らず、行ごとに
                // 読み直すたびにキャッシュを落とす (実測: 31 ms/chunk、逐次版 1.7 ms)。
                // そこで 1 呼び出しの x を L2 に収まる枚数へ切り、重みのデコード回数を
                // nt 回から nt/T 回に減らす。トークンは独立なので値は変わらない。
                let t_max = (GEMV_BATCH_TILE_BYTES / (4 * k.max(1))).max(1);
                let mut done = 0usize;
                while done < nt {
                    let t = t_max.min(nt - done);
                    let res = if scalar_only {
                        moe_expert::expert_gemm_scalar(
                            ty,
                            bytes,
                            k,
                            n,
                            &xs[done * k..(done + t) * k],
                            t,
                            &mut outs[done * n..(done + t) * n],
                        )
                    } else {
                        moe_expert::expert_gemm_any(
                            ty,
                            bytes,
                            k,
                            n,
                            &xs[done * k..(done + t) * k],
                            t,
                            &mut outs[done * n..(done + t) * n],
                        )
                    };
                    res.map_err(|err| one(format!("gemv_batch: {err}")))?;
                    done += t;
                }
            }
        }
        // expert の重みは型ごとの融合カーネルを使う。1 行版 `expert_gemv_any` と
        // ビット一致することを H3 側のテストで確認済み。
        GemvKernel::Expert => {
            moe_expert::expert_gemm_any(ty, bytes, k, n, xs, nt, outs)
                .map_err(|err| one(format!("gemv_batch: {err}")))?;
        }
    }
    #[cfg(test)]
    GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// 量子型のバッチ GEMV を「ブロック 1 回デコード」で回すか (純粋関数)。
///
/// 行数が [`GEMV_BATCH_MIN_ROWS`] 未満の行列は、重みを nt 回デコードし直す
/// 代わりに行を 16 分割できる旧経路 ([`gemv_rows_pertoken`]) の方が速いので
/// そちらへ振る (実測値は定数の doc)。境目はテストで固定する。
pub(crate) fn gemv_batch_blockdecode(n: usize, nt: usize) -> bool {
    nt > 0 && n >= GEMV_BATCH_MIN_ROWS
}

/// 旧経路: 行ごとに `expert_gemv*` を `nt` 回呼び、行方向だけ並列にする。
///
/// バッチ版 ([`gemv_rows_batch`] の `Linear`) は行並列を gemm 側
/// ([`moe_expert::PAR_MIN_ROWS`]) に任せるので、行数がその門番より少ない形では
/// 1 スレッドに落ちる。行数が少ない行列はむしろこちらが速い (実測: k=10240 /
/// n=320 / nt=64 で 1.7 ms/chunk、バッチ版 8.6 ms)。重み行は nt 回デコードするが、
/// x は 1 トークンずつ L1 に載る。
///
/// 数値: 1 行 1 トークンの計算は [`GgufLinear::gemv`] と同じカーネル・同じ引数
/// (`n_out = 1`) なので、`fast_q8` の扱いさえ合わせればビット一致する。
pub(crate) fn gemv_rows_pertoken(
    ty: GgmlType,
    fast_q8: bool,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    let par = n >= 64 && k >= 256;
    let rb = bytes.len() / n;
    // 行ごとに書くと出力がトークン優先の並びなので、いったん行優先で作って
    // 最後に詰め替える (n * nt * 4 バイト。640x8 で 20 KB)。
    let mut tmp = vec![0f32; n * nt];
    let run_row = |r: usize, o: &mut [f32]| -> Result<()> {
        let row = &bytes[r * rb..(r + 1) * rb];
        for t in 0..nt {
            let x = &xs[t * k..(t + 1) * k];
            let slot = &mut o[t..t + 1];
            let res = if ty == GgmlType::Q8_0 && fast_q8 {
                moe_expert::expert_gemv_q8_fast(row, k, 1, x, slot)
            } else {
                moe_expert::expert_gemv(ty, row, k, 1, x, slot)
            };
            res.map_err(|err| one(format!("gemv_batch: 行 {r} トークン {t}: {err}")))?;
        }
        Ok(())
    };
    #[cfg(feature = "par")]
    if par {
        use rayon::prelude::*;
        tmp.par_chunks_mut(nt)
            .enumerate()
            .try_for_each(|(r, o)| run_row(r, o))?;
    } else {
        for (r, o) in tmp.chunks_mut(nt).enumerate() {
            run_row(r, o)?;
        }
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = par;
        for (r, o) in tmp.chunks_mut(nt).enumerate() {
            run_row(r, o)?;
        }
    }
    for t in 0..nt {
        for r in 0..n {
            outs[t * n + r] = tmp[r * nt + t];
        }
    }
    Ok(())
}

/// F32 行列積のトークンブロック幅 (1 ブロックのトークン数)。
///
/// 32 にすると 1 行の内側ループが 32 レーン (AVX2 で 4 本、AVX-512 で 2 本の
/// ベクタ) になり、重み 1 要素の読み出しで 32 MAC を回せる。ルータ
/// (`ffn_gate_inp` 2560→512, F32) の重みの再読み出しが「トークンごと」= nt 回から
/// nt/32 回へ減る (実測は REPORT_P33.md)。
pub(crate) const ROUTER_TB: usize = 32;

/// [`ROUTER_TB`] 本のトークンと同時に回す行数。
///
/// 2 × 32 = 64 f32 = 512 bit ベクタ 4 本で、アキュムレータがレジスタに載る。
/// 実測 (`moe_router_microbench`, 実形 nt=512 / n=512 / k=2560, 同一実行内 A/B):
/// RB=1 で 3.9〜4.6 ms、RB=2 で 2.5〜3.0 ms、RB=4 で 2.6 ms (8 スレッドでは 4.2)、
/// RB=8 で 4.1 ms。RB=4 以上はレジスタが溢れて遅くなる。
pub(crate) const ROUTER_RB: usize = 2;

/// [`ROUTER_TB`] 本のトークンを列優先へ詰め替える (`src` はトークン優先)。
///
/// `dst[(i * tb + t)] = src[t * k + i]`。8×8 タイルで回すと、読み書きとも
/// 連続 f32 8 本 (= 1 ベクタ) になる。
pub(crate) fn transpose_token_block(src: &[f32], dst: &mut [f32], k: usize, tb: usize) {
    const T: usize = 8;
    let kt = k / T * T;
    let tt = tb / T * T;
    for t0 in (0..tt).step_by(T) {
        for i0 in (0..kt).step_by(T) {
            // tile[i][t] (i = 列内の行方向、t = トークン)
            let mut tile = [[0f32; T]; T];
            for r in 0..T {
                let s = &src[(t0 + r) * k + i0..(t0 + r) * k + i0 + T];
                for (c, &v) in s.iter().enumerate() {
                    tile[c][r] = v;
                }
            }
            for c in 0..T {
                dst[(i0 + c) * tb + t0..(i0 + c) * tb + t0 + T].copy_from_slice(&tile[c]);
            }
        }
        // k の端数
        for t in t0..t0 + T {
            for i in kt..k {
                dst[i * tb + t] = src[t * k + i];
            }
        }
    }
    // トークンの端数
    for t in tt..tb {
        for i in 0..k {
            dst[i * tb + t] = src[t * k + i];
        }
    }
}

/// [`ROUTER_TB`] 本のトークン × 全行の内積を、行 [`ROUTER_RB`] 本ずつの
/// アキュムレータで回す (`xt` は列優先、`out` は `[ROUTER_TB][n]` のトークン優先)。
///
/// 1 要素の演算列は [`gemv_f32_blocked`] の doc を参照 (i 昇順の f32 逐次加算)。
pub(crate) fn router_block_kernel(wf: &[f32], k: usize, n: usize, xt: &[f32], out: &mut [f32]) {
    let mut r0 = 0usize;
    while r0 < n {
        let rb = (n - r0).min(ROUTER_RB);
        let mut acc = [[0f32; ROUTER_TB]; ROUTER_RB];
        for i in 0..k {
            let xv = &xt[i * ROUTER_TB..i * ROUTER_TB + ROUTER_TB];
            for r in 0..rb {
                let w = wf[(r0 + r) * k + i];
                let a = &mut acc[r];
                for t in 0..ROUTER_TB {
                    a[t] += w * xv[t];
                }
            }
        }
        for (t, o) in out.chunks_exact_mut(n).enumerate() {
            for r in 0..rb {
                o[r0 + r] = acc[r][t];
            }
        }
        r0 += rb;
    }
}

/// F32 行列積のトークンブロック版: `out[t*n + r] = dot(row_r, x_t)`。
///
/// # 速くなる理由
///
/// 従来は「トークンごとに重み全行を読む」ので、重み 5.2 MB (2560×512 F32) を
/// `nt` 回読んでいた (512 トークンで 2.7 GB/層)。ここでは 32 トークンをまとめて
/// 1 行の内積を 32 レーンで回すので、重みの読み出しが `nt/32` 回 = 16 回
/// (84 MB/層) になる。
///
/// # 数値 (ビット一致の根拠)
///
/// 1 要素は `acc = 0.0; for i in 0..k { acc += w[r*k + i] * x_t[i] }` の i 昇順
/// 逐次で、行を f32 に開いてから同じ順で内積する従来経路 ([`gemv_f32_pertoken`])
/// と**同じ演算列**。トークン 32 本と行 [`ROUTER_RB`] 本は互いに独立なので、同時に
/// 回しても丸めの順序は変わらない (mul と add は融合しない: Rust は fast-math を
/// 付けない)。端数 (nt % [`ROUTER_TB`]、n % [`ROUTER_RB`]) は同じ関数の中で
/// 1 本ずつ処理する。
pub(crate) fn gemv_f32_blocked(
    wf: &[f32],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) {
    let nb = nt / ROUTER_TB;
    // 1. トークンを 32 本ずつ列優先へ詰め替える (ブロックごとに連続配置)。
    let mut xt = vec![0f32; nb * ROUTER_TB * k];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        xt.par_chunks_mut(ROUTER_TB * k)
            .enumerate()
            .for_each(|(b, dst)| {
                let src = &xs[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k];
                transpose_token_block(src, dst, k, ROUTER_TB);
            });
    }
    #[cfg(not(feature = "par"))]
    for b in 0..nb {
        let src = &xs[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k];
        transpose_token_block(src, &mut xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], k, ROUTER_TB);
    }
    // 2. ブロック × 全行 (ブロックは独立なので並列化しても値は変わらない)。
    let blocks: Vec<&mut [f32]> = outs[..nb * ROUTER_TB * n]
        .chunks_mut(ROUTER_TB * n)
        .collect();
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        blocks.into_par_iter().enumerate().for_each(|(b, o)| {
            router_block_kernel(wf, k, n, &xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], o);
        });
    }
    #[cfg(not(feature = "par"))]
    for (b, o) in blocks.into_iter().enumerate() {
        router_block_kernel(wf, k, n, &xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], o);
    }
    // 3. 端数のトークン (nt % 32) は従来どおり 1 トークンずつ。
    for t in nb * ROUTER_TB..nt {
        let x = &xs[t * k..(t + 1) * k];
        let o = &mut outs[t * n..(t + 1) * n];
        for r in 0..n {
            let row = &wf[r * k..(r + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            o[r] = acc;
        }
    }
}

/// `cuda` feature の有無で GPU GEMM の入口を切り替える (feature 無しは常に false)。
///
/// `gemv_batch` が `crate::cuda` を直接参照すると、既定ビルド (cuda なし) で
/// `E0433` になる (cd23ec8fc 以降、`--lib` がコンパイルできなかった)。
/// 入口をここに閉じ込めて、feature 無しでもこのファイルがコンパイルできるようにする。
#[cfg(feature = "cuda")]
pub(crate) fn gpu_gemm_into(w: &GgufLinear, xs: &[f32], nt: usize, outs: &mut [f32]) -> bool {
    crate::cuda::maybe_gpu_gguf_gemm_into(
        w.uid(),
        w.ty(),
        w.blocks().as_slice(),
        w.k(),
        w.n(),
        xs,
        nt,
        outs,
    )
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn gpu_gemm_into(_w: &GgufLinear, _xs: &[f32], _nt: usize, _outs: &mut [f32]) -> bool {
    false
}

/// [`GgufLinear`] のバッチ GEMV (`nt` トークン × 1 行列)。
///
/// 量子型は [`gemv_rows_batch`] (逐次版と同じカーネル)、
/// 無量子型 (F32/F16/BF16) は [`GgufLinear::gemv`] と同じ「行を f32 に開いて
/// 列昇順に逐次加算」を、行ごとに 1 回だけ展開してトークンぶん使い回す。
///
/// 注意: `--features cuda` で重みが VRAM に常駐している場合、逐次版
/// ([`GgufLinear::gemv`]) は GPU カーネルで計算するため、ここ (CPU 参照) とは
/// 一致しない。バッチ経路は現状 CPU のみ (GPU 常駐時は `DENNO_PREFILL_BATCH` を
/// 立てないこと)。
///
/// 2026-10-27: GPU 常駐の重みをバッチでも使う版 (`grid.y` = トークンのカーネル) を
/// 実装して A/B したが、802 トークンで **GPU 版のほうが遅かった** (mix 0.30 → 0.72,
/// attn 1.04 → 1.20, moe 1.70 → 1.75 ms/層)。呼び出しごとの H2D + 起動 + 同期 +
/// D2H が、この行列サイズ (k=320〜10240) では計算量を上回る。CPU 参照のままにする。
pub(crate) fn gemv_batch(w: &GgufLinear, xs: &[f32], nt: usize, outs: &mut [f32]) -> Result<()> {
    gemv_batch_inner(w, xs, nt, outs, false)
}

/// [`gemv_batch`] の **CPU 固定**版 (task 154)。
///
/// 量子型でも GPU GEMM を試さず、逐次版と同じカーネル・同じ加算順で回す。
/// トークン方向にだけ並列化するので、1 トークンずつ `gemv` を呼ぶのと
/// **ビット一致**する。PLE の射影 (チャンクでまとめたいが数値を動かしたくない)
/// に使う。
pub(crate) fn gemv_batch_cpu(
    w: &GgufLinear,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    gemv_batch_inner(w, xs, nt, outs, true)
}

pub(crate) fn gemv_batch_inner(
    w: &GgufLinear,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
    cpu_only: bool,
) -> Result<()> {
    let (k, n) = (w.k(), w.n());
    match w.ty() {
        // task 858: GPU GEMM を試す型は [`crate::quant_matrix::DENSE_GPU`] が唯一の
        // 定義 (decode の GEMV と同じ 4 型)。表とここがずれると、GPU を試みて必ず
        // 落ちる (= 静かに CPU へ戻る) か、逆に回せない型を投げることになる。
        // `DENNO_GPU_GEMM=0` で切れる。false なら CPU バッチへ落ちる。
        // cfg は `gpu_gemm_into` 側に閉じている (`cuda` feature 無しでは常に false)。
        ty if crate::quant_matrix::dense_gpu_supported(ty) => {
            // P-PF-GB: prefill のバッチ投影は GPU GEMM を先に試す。
            // 数値契約は prefill のみ解除済み (FMA 融合 + 加算順が違う)。
            if !cpu_only && gpu_gemm_into(w, xs, nt, outs) {
                #[cfg(test)]
                GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            gemv_rows_batch(
                GemvKernel::Linear {
                    fast_q8: w.fast_q8(),
                },
                w.ty(),
                w.blocks().as_slice(),
                k,
                n,
                xs,
                nt,
                outs,
            )
        }
        // 2026-10-31: 融合カーネル/GPU GEMM を持たない量子型 (IQ4_XS / IQ3_S /
        // IQ2_S / Q2_0 / IQ4_NL 等) は、無量子型と同じ「行を f32 に開いて
        // トークンブロックで使い回す」経路へ流す。値は逐次版と同じ規則
        // (行内は i 昇順の逐次加算) で、型ごとの分岐を増やさない。
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
            let _sub = q4ledger::sub(q4ledger::SUB_GEMM_CPU);
            if nt == 0 {
                return Ok(());
            }
            if xs.len() != nt * k {
                return Err(one(format!(
                    "gemv_batch: x の長さ {} が nt({nt}) * k({k}) と違う",
                    xs.len()
                )));
            }
            if outs.len() != nt * n {
                return Err(one(format!(
                    "gemv_batch: out の長さ {} が nt({nt}) * n({n}) と違う",
                    outs.len()
                )));
            }
            let bytes = w.blocks().as_slice();
            if n == 0 || bytes.len() % n != 0 {
                return Err(one(format!(
                    "gemv_batch: 行数 {n} でバイト数 {} を割り切れない",
                    bytes.len()
                )));
            }
            let rb = bytes.len() / n;
            let ty = w.ty();
            // P33: ルータ級の形 (32 トークン × 2 ブロック以上) はトークンブロック版。
            // 重みの再読み出しが nt 回 → nt/32 回になる。値は同じ (doc 参照)。
            if nt / ROUTER_TB >= 2 && n >= ROUTER_RB {
                let wf = dequant_rows_f32(ty, bytes, rb, k, n);
                gemv_f32_blocked(&wf, k, n, xs, nt, outs);
                #[cfg(test)]
                GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                #[cfg(test)]
                GEMV_BATCH_BLOCKED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            gemv_f32_pertoken(ty, bytes, rb, k, n, xs, nt, outs);
            #[cfg(test)]
            GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        // 2026-10-31: 融合カーネル/GPU GEMM を持たない量子型 (IQ4_XS / IQ3_S /
        // IQ2_S / Q2_0 / IQ4_NL 等) は、無量子型と同じ「行を f32 に開いて
        // トークンブロックで使い回す」経路へ流す (型ごとの分岐を増やさない)。
        ty if crate::kquant::dequant_supported(ty) => {
            let _sub = q4ledger::sub(q4ledger::SUB_GEMM_CPU);
            if nt == 0 {
                return Ok(());
            }
            if xs.len() != nt * k || outs.len() != nt * n {
                return Err(one(format!(
                    "gemv_batch: x/out の長さが nt({nt}) * k({k}) / n({n}) と違う"
                )));
            }
            let bytes = w.blocks().as_slice();
            if n == 0 || bytes.len() % n != 0 {
                return Err(one(format!(
                    "gemv_batch: 行数 {n} でバイト数 {} を割り切れない",
                    bytes.len()
                )));
            }
            let rb = bytes.len() / n;
            let wf = dequant_rows_f32(ty, bytes, rb, k, n);
            gemv_f32_blocked(&wf, k, n, xs, nt, outs);
            #[cfg(test)]
            GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            #[cfg(test)]
            GEMV_BATCH_BLOCKED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        other => Err(one(format!("gemv_batch: 未対応の量子型 {other:?}"))),
    }
}

/// 重みを 1 行ずつ f32 へ開く (行は独立なので rayon で分割してよい)。
pub(crate) fn dequant_rows_f32(ty: GgmlType, bytes: &[u8], rb: usize, k: usize, n: usize) -> Vec<f32> {
    let _sub = q4ledger::sub(q4ledger::SUB_DEQUANT);
    let mut wf = vec![0.0f32; n * k];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        wf.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
            crate::kquant::dequant_row_into(ty, &bytes[r * rb..(r + 1) * rb], row);
        });
    }
    #[cfg(not(feature = "par"))]
    for r in 0..n {
        crate::kquant::dequant_row_into(ty, &bytes[r * rb..(r + 1) * rb], &mut wf[r * k..(r + 1) * k]);
    }
    wf
}

/// 従来経路: 行を 1 回だけ f32 に開き、1 トークンずつ全行の内積を回す
/// (並列化するのはトークン方向だけ)。
///
/// 1 要素の累積は `acc += row[i] * x[i]` の i 昇順逐次なので、逐次版とビット一致する。
/// P33 でルータ級の形は [`gemv_f32_blocked`] に置き換えたが、形が小さいとき
/// (nt < 2 ブロック / n < [`ROUTER_RB`]) はこちらが速いので残してある。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_f32_pertoken(
    ty: GgmlType,
    bytes: &[u8],
    rb: usize,
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) {
    debug_assert_eq!(outs.len(), nt * n, "gemv_f32_pertoken: out の長さ");
    let wf = dequant_rows_f32(ty, bytes, rb, k, n);
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        outs.par_chunks_mut(n).enumerate().for_each(|(t, o)| {
            let x = &xs[t * k..(t + 1) * k];
            for r in 0..n {
                let row = &wf[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                o[r] = acc;
            }
        });
    }
    #[cfg(not(feature = "par"))]
    {
        for t in 0..nt {
            let x = &xs[t * k..(t + 1) * k];
            for r in 0..n {
                let row = &wf[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                outs[t * n + r] = acc;
            }
        }
    }
}

/// [`ExpertW::gemv`] のバッチ版 (1 expert × `nt` トークン)。
///
/// [`ExpertW::slice`] が返す 1 expert ぶんのバイト列を [`gemv_rows_batch`] に渡す。
/// 逐次版と同じ [`moe_expert::expert_gemv_any`] を 1 行ずつ通るのでビット一致する
/// (`GgufLinear::gemv` とは Q5_1 / Q8_0 で入口が違う点に注意 → [`GemvKernel`])。
pub(crate) fn expert_gemv_batch(
    w: &ExpertW,
    e: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    let s = w.slice(e)?;
    gemv_rows_batch(
        GemvKernel::Expert,
        w.ty,
        s,
        w.n_in,
        w.n_out,
        xs,
        nt,
        outs,
    )
    .map_err(|err| one(format!("{}: expert {e} (batch): {err}", w.name)))
}

/// 行のベクタを 1 本の平坦な配列に連結する ([`gemv_batch`] の入力形式)。
pub(crate) fn flatten_rows(rows: &[Vec<f32>], width: usize) -> Result<Vec<f32>> {
    let _sub = q4ledger::sub(q4ledger::SUB_LAYOUT);
    let mut out = Vec::with_capacity(rows.len() * width);
    for (t, r) in rows.iter().enumerate() {
        if r.len() != width {
            return Err(one(format!(
                "flatten_rows: 行 {t} の長さ {} が {width} と違う",
                r.len()
            )));
        }
        out.extend_from_slice(r);
    }
    Ok(out)
}

/// テスト用: 層優先バッチ prefill / バッチ GEMV が実際に走った回数。
/// 「緑 = 経路が発動した証拠」ではないので、分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static PREFILL_BATCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 200: 層優先 prefill で平坦な attn 入口が何層走ったか (テスト用)。
#[cfg(test)]
pub(crate) static PF_ATTN_FLAT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 200: 同じく従来の行入口 (`Vec<Vec<f32>>`) が何層走ったか (テスト用)。
#[cfg(test)]
pub(crate) static PF_ATTN_ROWS_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) static GEMV_BATCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: 量子型のバッチ GEMV が**ブロック 1 回デコード経路**に入った回数。
/// 行数が少ない形は旧経路 (行ごとに nt 回デコード) に落ちるので、どちらを通ったかを
/// 値の一致だけからは判定できない。分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static GEMV_BATCH_BLOCKDECODE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: 無量子型 (F32/F16/BF16) のバッチ GEMV が**トークンブロック経路**
/// ([`gemv_f32_blocked`]) に入った回数。形が小さいと従来の 1 トークン経路へ
/// 落ちるので、分岐の通過をここで数える (緑 = 経路が発動した証拠ではない)。
#[cfg(test)]
pub(crate) static GEMV_BATCH_BLOCKED_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: PLE のまとめ引き ([`Qwen4Exp::gather_ple_many`]) が走った回数。
/// 経路の通過を数える (緑 = 経路が発動した証拠ではない)。
#[cfg(test)]
pub(crate) static PLE_BATCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `DENNO_PREFILL_BATCH` の値の解釈 (純粋関数)。既定 (未設定) は無効。
///
/// `"1"` のときだけ有効。`"0"`・空文字・その他は無効 (既定 off を崩さない)。
pub(crate) fn prefill_batch_from_value(v: Option<&str>) -> bool {
    matches!(v, Some("1"))
}

/// 層優先バッチ prefill が有効か (`DENNO_PREFILL_BATCH`, 既定 off)。プロセスで 1 回だけ読む。
pub(crate) fn prefill_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| prefill_batch_from_value(std::env::var("DENNO_PREFILL_BATCH").ok().as_deref()))
}

/// 1 チャンクのトークン数 (`DENNO_PREFILL_BATCH_CHUNK`, 既定 8, 1..=512)。
///
/// チャンクを大きくするほど重みの再読み込みは減るが、残差 `[nt][hc_dim]` と
/// `hc_mix` の中間がそのぶん増える (27B の hc_dim は 20480 f32 = 80 KiB/トークン)。
///
/// 2026-10-28: 上限を 64 から 512 へ。Flash-Next は 512 expert / top-10 なので、
/// chunk 64 では 1 チャンクで 365 体ぶんの expert 重みを読む (1 体あたり 1.4
/// トークンしか使わない)。512 まで上げると 1 体あたり 10 トークンになり、
/// トークンあたりの expert 読み出しが 5.7 分の 1 になる。
///
/// 2026-10-30: 既定を 8 から **512** へ (上の実測に合わせる)。`DENNO_PREFILL_BATCH_CHUNK`
/// で下げられるが、512 未満は expert の読み直しが増えるだけなので通常は触らない。
/// 中間バッファが増える (hc_mix の 512 トークンぶん) ので、RAM が厳しい環境だけ下げる。
pub(crate) fn prefill_batch_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        // task 197: `DENNO_PF_CHUNK` があれば最優先 (層ループのチャンク上書き)。
        // ストリーミング経路は 1 層 1 チャンクでミス expert を上げるので、チャンクを
        // 大きくすると同じ重みを上げる回数が減る (台帳の最大項目が消える)。
        if let Some(n) = crate::model::exp_pf::chunk_override() {
            return n;
        }
        // task 162: ストリーミング経路では層ループのチャンクを大きく取れる
        // (ミス expert の上げ直しが 1 層 1 回に減る)。既定は従来どおり 512。
        // task 165: ストリーミング時は既定を 1024 にする (lab t162c の最良。
        // 上げ専用スレッドを足した版でも 1024 を既定にする)。
        let (hi, dflt) = if crate::model::exp_pf::stream_enabled() {
            (65536, crate::model::exp_pf::DEFAULT_PREFILL_CHUNK_STREAM)
        } else {
            (512, 512)
        };
        std::env::var("DENNO_PREFILL_BATCH_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(1, hi))
            .unwrap_or(dflt)
    })
}

/// task 239: このプロンプトのチャンク長。
///
/// `DENNO_PF_CHUNK_BALANCE=1` のとき、ストリーミング経路のチャンクを
/// 「上限 [`crate::model::exp_pf::CHUNK_BALANCE_CAP`] の下でチャンク数を最小にする」
/// 均等割りにする。ミス expert は 1 チャンク 1 回上げるので、チャンク数が
/// そのまま転送量に効く (固定 1536 では 3,222 トークンが 3 枚になり、同じ
/// 重みを 3 回上げていた)。
///
/// `DENNO_PF_CHUNK` の明示指定があるときは均等割りしない (既存の上書きを尊重)。
pub(crate) fn prefill_chunk_for(n: usize) -> usize {
    let base = prefill_batch_chunk();
    // 明示指定の判定は生の env で行う (`chunk_override()` は未設定でも
    // `Some(既定)` を返すので、それで判定するとゲートが発火しない)。
    let explicit = crate::model::exp_pf::chunk_override_from_value(
        std::env::var(crate::model::exp_pf::PF_CHUNK_ENV)
            .ok()
            .as_deref(),
    )
    .is_some();
    crate::model::exp_pf::chunk_for_prefill(
        n,
        base,
        crate::model::exp_pf::CHUNK_BALANCE_CAP,
        crate::model::exp_pf::chunk_balance(),
        crate::model::exp_pf::chunk_balance_long(),
        crate::model::exp_pf::stream_enabled(),
        explicit,
    )
}

// ---------------------------------------------------------------------------
// task 154: prefill の GPU 常駐 (`DENNO_PF_RESIDENT`)
// ---------------------------------------------------------------------------
//
// 動機 (lab t151c の 14,018 トークン prefill の台帳): h2d 111 GB / d2h 168 GB /
// sync 36,208 回。host 側 (cpu 列) が約 150 s で、内訳は GDN コア (ssm)、
// hc mix の要素演算、hc combine、PLE、埋め込み。ここでは
//
// - `DENNO_PF_GDN_GPU`: ssm 層の GDN コアを GPU の sigmoid スキャンで回す
// - `DENNO_PF_MIX_DEV`: hc mix (norm → down → lo → up → mixed → inject) と
//   hc combine をデバイス上の 1 連鎖で回し、活性をチャンク中 GPU に置く
//
// の 2 つを 1 つの master ゲートで束ねる。**既定 on** (lab t154e で採用。
// `DENNO_PF_RESIDENT=0` で従来経路に戻る)。

/// 副ゲートの解釈 (純関数)。明示的な真値だけ on (未設定は master を見る)。
pub(crate) fn pf_on_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        // task 181: lab t181b で 3 本 + 小さい N + ミラー + D2H の組が目標
        // (-8% / -8%) を満たしたので **既定 on**。明示的な偽値で切る。
        None => true,
    }
}

/// task 154: master ゲート (`DENNO_PF_RESIDENT`) の解釈 (純関数)。**既定 on**。
///
/// `0` / `off` / `false` / `no` のときだけ無効。未設定と空文字は既定 (on)。
/// lab t154e (base = このゲート off) の実測: cold prefill 3,826 tok
/// 63.05 -> 49.52 s (-21.5%)、14,018 tok 210.39 -> 153.69 s (-26.9%)、decode
/// 54 -> 53 ms/tok。`DENNO_PF_RESIDENT=0` で従来経路に戻せる。
pub(crate) fn pf_resident_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
    }
}

/// task 154: prefill を GPU 常駐にするか (`DENNO_PF_RESIDENT`、既定 on)。
pub(crate) fn pf_resident_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_resident_from_value(std::env::var("DENNO_PF_RESIDENT").ok().as_deref())
    })
}

/// 副ゲートの上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。テスト用。
pub(crate) static PF_SUB_OVERRIDE: [std::sync::atomic::AtomicU8; 2] =
    [const { std::sync::atomic::AtomicU8::new(0) }; 2];

/// 副ゲートの番号: 0 = GDN、1 = mix/combine。
pub(crate) const PF_SUB_GDN: usize = 0;

/// 副ゲートの番号: mix/combine。
pub(crate) const PF_SUB_MIX: usize = 1;

/// task 154: テスト専用: 副ゲートの on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf_sub_for_test(which: usize, v: Option<bool>) {
    PF_SUB_OVERRIDE[which].store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub(crate) fn pf_sub_on(which: usize, env: &str) -> bool {
    match PF_SUB_OVERRIDE[which].load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    match std::env::var(env) {
        Ok(v) => pf_on_from_value(Some(&v)),
        Err(_) => pf_resident_on(),
    }
}

/// ssm 層の GDN コアを GPU スキャンで回すか (`DENNO_PF_GDN_GPU`、未設定は master)。
pub(crate) fn pf_gdn_gpu_on() -> bool {
    pf_sub_on(PF_SUB_GDN, "DENNO_PF_GDN_GPU")
}

/// task 167: ssm 層の prefill を「投影 → GDN → out_proj」のデバイス連鎖で回すか
/// (`DENNO_GDN_PF_CHAIN`、**既定 on**)。`0` / `off` / `false` で従来経路。
///
/// 連鎖は GPU スキャン (`pf_gdn_gpu_on`) と同じ前提の上でだけ動く。ゲートを
/// 別に持つのは、数値の契約 (丸め順は同じ・中間をホストへ落とさない) と
/// 経路の切り戻しを独立にできるようにするため。
///
/// 採用は lab t167c (2026-11-07): 14,018 の cold prefill -8.7%、3,826 で -8.0%、
/// decode 全 depth と 19 トークン next turn prefill は不変、md5 は全リクエスト
/// 一致。ssm の h2d 53.6 → 5.6 GB、d2h 53.9 → 8.5 GB、同期 10,798 → 3,024 回。
pub(crate) fn pf_gdn_chain_on() -> bool {
    #[cfg(test)]
    {
        match PF_GDN_CHAIN_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_gdn_chain_from_value(std::env::var("DENNO_GDN_PF_CHAIN").ok().as_deref())
    })
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
#[cfg(test)]
pub(crate) static PF_GDN_CHAIN_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) fn set_pf_gdn_chain_for_test(v: Option<bool>) {
    PF_GDN_CHAIN_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// task 167: 連鎖のゲート値の解釈 (純関数)。**未設定は on** (lab t167c で採用)、
/// 明示的な偽値 (`0` / `off` / `false` / `no`。前後の空白と大小文字は無視) だけ off。
/// 他の master ゲート ([`pf_resident_from_value`]) と同じ規約に揃える。
pub(crate) fn pf_gdn_chain_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// hc mix と hc combine をデバイス連鎖で回すか (`DENNO_PF_MIX_DEV`、未設定は master)。
pub(crate) fn pf_mix_dev_on() -> bool {
    pf_sub_on(PF_SUB_MIX, "DENNO_PF_MIX_DEV")
}

// ---------------------------------------------------------------------------
// task 172: prefill の stage をデバイスで閉じるゲート (3 本)
// ---------------------------------------------------------------------------
//
// どれも **既定 off**。lab で採否を決めたら既定 on にする (採否は task の
// 規則: cold prefill 3,826 / 14,018 の両方が 10% 以上速い・他が 2% 超悪化しない・
// text が正常)。off のときは 1 回の env 読み (OnceLock) だけで従来経路。

/// テストが env を立てずに 3 本のゲートを強制するための上書き
/// (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。番号は [`PF2_GATE_FULL`] など。
pub(crate) static PF2_FORCE: [std::sync::atomic::AtomicU8; 3] = [const { std::sync::atomic::AtomicU8::new(0) }; 3];

/// task 172: ゲート番号 (full)。
pub(crate) const PF2_GATE_FULL: usize = 0;

/// task 172: ゲート番号 (shared)。
pub(crate) const PF2_GATE_SHARED: usize = 1;

/// task 172: ゲート番号 (ple)。
pub(crate) const PF2_GATE_PLE: usize = 2;

/// task 172: テスト専用: ゲートの on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf2_for_test(which: usize, v: Option<bool>) {
    PF2_FORCE[which.min(2)].store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub(crate) fn pf2_forced(which: usize) -> Option<bool> {
    match PF2_FORCE[which.min(2)].load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

/// task 172: `DENNO_PF_FULL_DEV` の解釈 (純関数)。明示的な真値だけ on。
pub(crate) fn pf_full_dev_from_value(v: Option<&str>) -> bool {
    pf_on_from_value(v)
}

/// task 172: full attention 層の 1 チャンクを「入力 1 H2D / 出力 1 D2H」で
/// 閉じるか (`DENNO_PF_FULL_DEV`、既定 off)。
///
/// task 175 (lab t175b) の実測: int-auto + br172 + br176 の既定 (base) に対し
/// cold prefill 3,826 tok 17.76 → 17.41 s (-2.0%)、14,018 tok 60.05 → 57.73 s
/// (-3.9%)、decode と 19 トークン next turn は ±2% 以内、errors 0・text sane。
/// 目標 (-10%) に届かないので **off のまま**。理由は REPORT_P175.md。
pub(crate) fn pf_full_dev_on() -> bool {
    if let Some(v) = pf2_forced(PF2_GATE_FULL) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_full_dev_from_value(std::env::var("DENNO_PF_FULL_DEV").ok().as_deref()))
}

/// task 183: `DENNO_PF_FULL_FUSED` の解釈 (純関数)。明示的な偽値だけ off。
pub(crate) fn pf_full_fused_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}

/// task 183: full 層の 1 チャンクをデバイス連鎖で閉じ、**融合済み qkv は 1 回の
/// GEMM** で撃つか (`DENNO_PF_FULL_FUSED`、既定 off)。
///
/// このゲートは「full 層をデバイス連鎖で閉じる」ところまで含む
/// (br172 の `DENNO_PF_FULL_DEV` と同じ連鎖。融合していない層も連鎖に載る)。
/// 融合層は q|gate|k|v を 1 本のまま撃ち、q/k/v を行オフセットで読むので、
/// 射影出力 (12 層で 8.96 GB) の D2H が消える。
///
/// **既定 on** (lab t183a で採用)。lab t183a (depth8k) の実測:
/// cold prefill 3,826 tok 14.65 → 13.81 s (-6.0%)、14,018 tok 51.58 → 46.64 s
/// (-9.6%)、decode は全点 ±0%、errors 0、text sane。台帳の full は
/// wall 6,944 → 3,609 ms、cpu 2,933 → 316 ms、d2h 10.68 GB → 2.41 GB
/// (h2d 7.7 → 1.8 GB)。md5 は射影の丸めで変わる (契約外)。
pub(crate) fn pf_full_fused_on() -> bool {
    #[cfg(test)]
    {
        match PF_FULL_FUSED_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_full_fused_from_value(std::env::var("DENNO_PF_FULL_FUSED").ok().as_deref())
    })
}

#[cfg(test)]
pub(crate) static PF_FULL_FUSED_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) fn set_pf_full_fused_for_test(v: Option<bool>) {
    PF_FULL_FUSED_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

// ---------------------------------------------------------------------------
// task 200: 層優先 prefill の活性を平坦なまま attn へ渡す
// ---------------------------------------------------------------------------
//
// 動機 (REPORT_P198.md の 14,018 prefill 台帳): 層ループは mix の出力を
// `mixed_flat` (`nt × n_embd`) で受け取り、`pf_combine` にも平坦な配列で
// 渡している。その間に attn の口だけが `Vec<Vec<f32>>` なので、
// 1 層チャンクあたり
//
//   mixed_flat → curs (nt 個の Vec) → 実装内で平坦に戻す → 出力 (nt 個の Vec)
//   → flatten_rows で平坦に戻す
//
// と 4 回のコピー + 2 * nt 個のアロケーションが乗っていた。台帳の
// `other` の未計測分 (0.10 ms/tok) と `layout` (245 ms) がこれにあたる。
// 平坦な入口 [`LayerAttn::forward_batch_flat`] を通すと往復が消える。
// 数値は同じ (コピーの有無だけ)。lab t200a の flat で 14,018 -5.6% /
// 3,826 -2.9% だったので **既定 on** (明示的な偽値で切る)。

/// task 200: `DENNO_PF_ATTN_FLAT` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが
/// off。**未設定は on** (lab t200b の flatdev が目標内なので既定にした)。
pub(crate) fn pf_attn_flat_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
    }
}

/// task 200: 層優先 prefill で attn へ平坦な活性をそのまま渡すか
/// (`DENNO_PF_ATTN_FLAT`、**既定 on**)。
pub(crate) fn pf_attn_flat_on() -> bool {
    #[cfg(test)]
    {
        match PF_ATTN_FLAT_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_attn_flat_from_value(std::env::var("DENNO_PF_ATTN_FLAT").ok().as_deref()))
}

#[cfg(test)]
pub(crate) static PF_ATTN_FLAT_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) fn set_pf_attn_flat_for_test(v: Option<bool>) {
    PF_ATTN_FLAT_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

// ---------------------------------------------------------------------------
// task 200: attn の出力をデバイスに残したまま combine する
// ---------------------------------------------------------------------------
//
// 動機 (lab t200a の 14,018 / 3,826 prefill 台帳): `other` の残差は 1,278 ms /
// 334 ms で、中身はほぼ `pf_combine` の H2D (1 層チャンク 2 回 × nt*n_embd) だった。
// ssm 層の attention は `DENNO_GDN_PF_CHAIN` の連鎖が**デバイスで**出力を作って
// いるのに、いったん D2H してから combine でもう一度 H2D している。
// 連鎖の出力バッファをそのまま combine に読ませれば往復が消える。
// 数値は同じ (combine は同じカーネル・同じ入力)。lab t200b の flatdev で
// 14,018 -4.9% / 3,826 -4.8% (base2 比) だったので **既定 on**。

/// task 200: `DENNO_PF_ATTN_DEVRES` の解釈 (純関数)。未設定は on
/// (lab t200b の flatdev が目標内なので既定にした)。明示的な偽値だけ off。
pub(crate) fn pf_attn_devres_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
    }
}

/// task 200: ssm 連鎖の出力をデバイスに残し、直後の combine にそのまま読ませるか
/// (`DENNO_PF_ATTN_DEVRES`、**既定 on**)。
pub(crate) fn pf_attn_devres_on() -> bool {
    #[cfg(test)]
    {
        match PF_ATTN_DEVRES_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_attn_devres_from_value(std::env::var("DENNO_PF_ATTN_DEVRES").ok().as_deref())
    })
}

#[cfg(test)]
pub(crate) static PF_ATTN_DEVRES_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) fn set_pf_attn_devres_for_test(v: Option<bool>) {
    PF_ATTN_DEVRES_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

// ---------------------------------------------------------------------------
// task 308: full 層の出力をデバイスに残す (`DENNO_PF_FULL_DEVRES`)
// ---------------------------------------------------------------------------
//
// full 層の連鎖 (`DENNO_PF_FULL_FUSED`、既定 on) は出力投影までデバイスで
// 回しているのに、最後に `[nt][e]` を D2H し、直後の combine が同じ `[nt][e]`
// を H2D し直している (3,826 トークン・14 層で 1 往復 1.1 GB)。
// `DENNO_PF_ATTN_DEVRES` は ssm 連鎖の出力にしか効いていなかったので、
// full 層にも同じ道を通す。値は変わらない (combine は同じカーネル・同じ入力)。
// lab t308a/t308b で目標内だったので**既定 on** (切り戻しは `=0`)。

/// task 308: `DENNO_PF_FULL_DEVRES` の解釈 (純関数)。明示的な偽値だけ off。
pub(crate) fn pf_full_devres_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
        None => true,
    }
}

/// task 308: full 層の出力をデバイスに残すか (`DENNO_PF_FULL_DEVRES`、**既定 on**)。
///
/// 効くのは `DENNO_PF_ATTN_DEVRES` (既定 on) と平坦な活性の層優先 / 常駐経路
/// だけ。`roles` を取るときは呼び出し側が要求を立てない (ホストの行が要る)。
///
/// lab t308b (depth8k): 3,826 cold は 1 ビットも変わらない (md5 全出力 SAME、
/// t308a の `full` 単独 COND で確認)。台帳の h2d / d2h が 14k で
/// 41,636.8 / 12,987.8 MB → 39,914.3 / 11,265.2 MB (-1.72 GB ずつ)。
pub(crate) fn pf_full_devres_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_full_devres_from_value(std::env::var("DENNO_PF_FULL_DEVRES").ok().as_deref())
    })
}

/// task 172: `DENNO_PF_SHARED_DEV` の解釈 (純関数)。明示的な真値だけ on。
pub(crate) fn pf_shared_dev_from_value(v: Option<&str>) -> bool {
    pf_on_from_value(v)
}

/// task 172: shared expert の 3 射影を 1 チャンクぶんデバイスで回すか
/// (`DENNO_PF_SHARED_DEV`、既定 off)。
///
/// task 175 (lab t175b) の実測: 同じ base に対し cold prefill 3,826 tok
/// 17.76 → 17.35 s (-2.3%)、14,018 tok 60.05 → 57.50 s (-4.2%)、出力は
/// base と md5 一致 (丸めも同じ)、errors 0。目標 (-10%) に届かないので
/// **off のまま**。理由は REPORT_P175.md。
pub(crate) fn pf_shared_dev_on() -> bool {
    if let Some(v) = pf2_forced(PF2_GATE_SHARED) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_shared_dev_from_value(std::env::var("DENNO_PF_SHARED_DEV").ok().as_deref()))
}

/// task 172: `DENNO_PF_PLE_GPU` の解釈 (純関数)。明示的な真値だけ on。
pub(crate) fn pf_ple_gpu_from_value(v: Option<&str>) -> bool {
    pf_on_from_value(v)
}

/// task 172: PLE の key / value 射影を GPU で回すか (`DENNO_PF_PLE_GPU`、既定 off)。
pub(crate) fn pf_ple_gpu_on() -> bool {
    if let Some(v) = pf2_forced(PF2_GATE_PLE) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_ple_gpu_from_value(std::env::var("DENNO_PF_PLE_GPU").ok().as_deref()))
}

// ---------------------------------------------------------------------------
// task 182: prefill の router / PLE 射影 (F32) を GPU で回すゲート (2 本)
// ---------------------------------------------------------------------------
//
// どちらも **明示的な真値だけ on** (既定 off)。`pf_on_from_value` は task 181
// 以降「未設定 = on」なので、ここでは使わない (lab の COND で明示的に on に
// したときだけ効かせ、採否が決まるまで本番の既定挙動を変えない)。

/// task 182: `DENNO_PF_ROUTER_GPU` / `DENNO_PF_PLE_F32_GPU` の解釈 (純関数)。
///
/// task 182 の lab t182b (combo vs off) が cold prefill 3,826 tok -8.5%、
/// 14,018 tok -9.3%、decode と 19 トークン next turn は ±2% 以内 (10 ms の差)、
/// errors 0・text sane だったので **既定 on**。明示的な偽値だけ off。
pub(crate) fn pf182_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
        None => true,
    }
}

/// task 182: 実際に GPU で回ったことを lab のログに 1 回だけ出す
/// (分岐が発火した証拠。off のときは何も出ない)。
pub(crate) static PF182_LOG_ROUTER: std::sync::Once = std::sync::Once::new();

pub(crate) static PF182_LOG_PLE_F32: std::sync::Once = std::sync::Once::new();

pub(crate) static PF182_LOG_PLE_TY: std::sync::Once = std::sync::Once::new();

/// テストが env を立てずに 2 本のゲートを強制するための上書き
/// (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
pub(crate) static PF182_FORCE: [std::sync::atomic::AtomicU8; 2] =
    [const { std::sync::atomic::AtomicU8::new(0) }; 2];

/// task 182: ゲート番号 (router)。
pub(crate) const PF182_GATE_ROUTER: usize = 0;

/// task 182: ゲート番号 (PLE の F32 射影)。
pub(crate) const PF182_GATE_PLE_F32: usize = 1;

pub(crate) fn pf182_forced(which: usize) -> Option<bool> {
    match PF182_FORCE[which.min(1)].load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

/// テスト専用: ゲートの on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf182_for_test(which: usize, v: Option<bool>) {
    PF182_FORCE[which.min(1)].store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// task 182: prefill の router (`ffn_gate_inp`、F32) を GPU の f32 GEMM で
/// 回すか (`DENNO_PF_ROUTER_GPU`、**既定 on**。`=0` で CPU バッチに戻る)。
///
/// lab t182a (3,826 tok): router 1,334 -> 380 ms、prefill -8.0%。
pub(crate) fn pf_router_gpu_on() -> bool {
    if let Some(v) = pf182_forced(PF182_GATE_ROUTER) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf182_from_value(std::env::var("DENNO_PF_ROUTER_GPU").ok().as_deref()))
}

/// task 182: PLE の key / value 射影 (F32 重み) を GPU の f32 GEMM で回すか
/// (`DENNO_PF_PLE_F32_GPU`、**既定 on**。`=0` で CPU バッチに戻る)。
pub(crate) fn pf_ple_f32_gpu_on() -> bool {
    if let Some(v) = pf182_forced(PF182_GATE_PLE_F32) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf182_from_value(std::env::var("DENNO_PF_PLE_F32_GPU").ok().as_deref()))
}

/// task 189: `DENNO_PF_ROUTER_SM` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn pf_router_sm_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}

/// task 189: router の softmax と top-k を「1 行 1 rayon タスク」ではなく
/// 「トークンをワーカー数で割ったブロック」で回す (`DENNO_PF_ROUTER_SM`、既定 off)。
///
/// `par_chunks_mut(ne)` は 1 行 (ne=512 要素) ごとに 1 タスクを作るので、
/// 1 チャンク 512 トークン × 48 層 × 28 チャンク = 68.8 万タスクになる。
/// 行は独立なので割り方だけを変えれば値はビット一致する。
pub(crate) fn pf_router_sm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_router_sm_from_value(std::env::var("DENNO_PF_ROUTER_SM").ok().as_deref()))
}

/// task 195: `DENNO_PF_ROUTER_FAST` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// **既定 on** (lab t195a: -1.1%/-0.5%、選択規則は従来と同一 = md5 SAME)。
pub(crate) fn pf_router_fast_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}

/// task 195: prefill router の top-k を挿入法で回す (`DENNO_PF_ROUTER_FAST`)。
///
/// 選択規則は従来と同一なので値はビット一致する (変わるのは比較の回数だけ)。
/// task 198 で既定 on。
pub(crate) fn pf_router_fast_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_router_fast_from_value(std::env::var("DENNO_PF_ROUTER_FAST").ok().as_deref())
    })
}

/// task 308: `DENNO_PF_LM_ROUTER_DEV` の解釈 (純関数)。明示的な偽値だけ off。
///
/// 層優先 prefill (既定経路) の MoE は、router の logits を GPU の f32 GEMM で
/// 作ってから `[nt][n_expert]` を**ホストへ降ろし**、softmax と top-k を
/// rayon で回していた。デバイスのカーネル (`dno_router_softmax_topk`、task 225)
/// は既にあるので、タイル全トークンを 1 回で回して選んだ k 個だけを降ろす。
pub(crate) fn pf_lm_router_dev_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
        None => true,
    }
}

/// task 308: 層優先 prefill の router + top-k をデバイスでまとめて回すか
/// (`DENNO_PF_LM_ROUTER_DEV`、**既定 on**。lab t308b で採用)。
///
/// lab t308b (depth8k、3,826 cold、2 ラウンド目の同一 COND): base 3.13 s →
/// both 2.94 s (-6.1%)。台帳の d2h は 14k で 12,987.8 → 11,666.3 MB
/// (probs の降ろしが消えた)。md5 は softmax の和の結合順だけで変わる。
pub(crate) fn pf_lm_router_dev_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_lm_router_dev_from_value(std::env::var("DENNO_PF_LM_ROUTER_DEV").ok().as_deref())
    })
}

/// task 182: router を GPU に載せる最小トークン数 (これ未満は CPU バッチ)。
///
/// 19 トークンの next turn prefill はここで CPU のまま (= 従来とビット一致) に
/// なる。64 トークンでも CPU は約 12 ms、GPU は固定費 1 ms 未満なので境界は 64。
pub(crate) const PF_ROUTER_GPU_MIN_TOK: usize = 64;

/// task 182: PLE の F32 射影を GPU に載せる最小トークン数。
pub(crate) const PF_PLE_F32_GPU_MIN_TOK: usize = 64;

/// `DENNO_PLE_BATCH` の値の解釈 (純粋関数)。**既定 on**。
///
/// `0` / `false` / `off` のときだけ無効 (トークンごとに 1 回ずつ引く旧経路)。
/// どちらの経路も logits はビット一致する (行のデコードは同じ関数)。
pub(crate) fn ple_batch_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => {
            !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off"))
        }
    }
}

/// PLE の行 gather をチャンクぶんまとめて引くか (`DENNO_PLE_BATCH`, 既定 on)。
///
/// まとめ引きは「行 index を全トークン分集める → 昇順・重複排除 → 1 行 1 回読む」
/// だけなので、値はトークンごとの gather とビット一致する。逃げ道として
/// `DENNO_PLE_BATCH=0` で旧経路に戻せる。
pub(crate) fn ple_batch_enabled() -> bool {
    #[cfg(test)]
    {
        match PLE_BATCH_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => return false,
            1 => return true,
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ple_batch_from_value(std::env::var("DENNO_PLE_BATCH").ok().as_deref()))
}

/// テストが env を立てずに PLE のまとめ引きを on/off するための強制値
/// (env は `OnceLock` なのでプロセス内で 1 回しか読めない)。`2` = 未設定。
#[cfg(test)]
pub(crate) static PLE_BATCH_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// テストから [`ple_batch_enabled`] を固定する (`None` で env の解釈に戻る)。
#[cfg(test)]
pub(crate) fn set_ple_batch_for_test(v: Option<bool>) {
    let raw = match v {
        None => 2u8,
        Some(false) => 0,
        Some(true) => 1,
    };
    PLE_BATCH_FORCE.store(raw, std::sync::atomic::Ordering::Relaxed);
}

/// task 148/151: prefill の選択 expert を **m 行カーネル**で GPU に回す。
///
/// ヒット (VRAM 常駐) の expert だけを `kernels_exp_iq_pf` で回し、ミスは
/// [`moe_experts_flat`] で CPU に回す。戻り値:
///
/// - `Ok(Some(hits))` — `out` に全 expert の寄与が入っている (GPU + CPU ミス)
/// - `Ok(None)` — GPU を使わなかった。**`out` は触っていない** (呼び出し側が
///   従来の CPU 経路を回す)
/// - `Err` — CPU ミスの計算そのものが失敗した
///
/// 足し込みの順序 (task 151 で修正): トークンごとに **expert index 昇順で 1 本
/// ずつ** `out[t*e+d] += w * h[d]` する ([`exp_pf_add_all`])。ミスもヒットも同じ
/// 列に並べて足すので、**和の順序が VRAM のヒット率に依存しない**。
///
/// 以前 (t148b/t148c) は「ミス全部を flat が `out` へ → GPU の加重和 1 回を
/// `out` へ」だった。丸めの入り方がヒット/ミスの内訳で変わるため、同じ入力でも
/// VRAM の中身が違えば最終ビットが変わり、greedy の 1 トークンが入れ替わった
/// (`a1=0f63edea` / `a2=f0a01ec1`)。この形なら CPU だけの経路とビット一致する。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_experts_iq_pf_batch(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    nt: usize,
    e: usize,
    ff: usize,
    out: &mut [f32],
) -> Result<Option<u32>> {
    use crate::model::expert_cache::ExpertKey;
    use crate::model::expert_vram::ExpertShape;

    if by_expert.is_empty()
        || nt == 0
        || !crate::model::exp_pf::pf_enabled()
        || !pf_expert_mode(lw)
    {
        return Ok(None);
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // 行のある expert だけ参照する (1 チャンクで全 256 本が普通)。
    let t_lookup = crate::model::pf_host::stamp();
    let mut keys: Vec<ExpertKey> = Vec::with_capacity(by_expert.len());
    let mut ids: Vec<usize> = Vec::with_capacity(by_expert.len());
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        keys.push(ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        });
        ids.push(ei);
    }
    if keys.is_empty() {
        return Ok(None);
    }
    // **触らない**参照 (`get_mut` で LRU を塗り替えると、その後の decode の
    // ヒット率が落ちる: lab t148c の 14k で 0.966 → 0.866)。
    let got = expert_vram::peek_many(&keys, shapes)
        .map_err(|err| one(format!("層 {}: expert VRAM 参照: {err:#}", lw.il)))?;
    let mut pick = vec![false; by_expert.len()];
    let mut devs: Vec<std::sync::Arc<expert_vram::ExpertDev>> = Vec::new();
    // task 162: expert 番号 → 常駐 (ストリーミングの表を組むのに使う)。
    let mut resident: Vec<Option<std::sync::Arc<expert_vram::ExpertDev>>> =
        vec![None; by_expert.len()];
    for (i, g) in got.iter().enumerate() {
        if let Some(d) = g {
            pick[ids[i]] = true;
            resident[ids[i]] = Some(std::sync::Arc::clone(d));
            devs.push(std::sync::Arc::clone(d));
        }
    }
    let hits = devs.len() as u32;
    crate::model::exp_pf::note_lookup(keys.len() as u32, hits);
    crate::model::pf_host::mark(t_lookup, crate::model::pf_host::LOOKUP);
    // task 162: ミスをステージングへ上げ、行のある expert を**全部** GPU で回す。
    // ヒット数に関係なく試す (ミスを載せられるならヒット 0 でも CPU より速い)。
    // 載せられないときだけ従来の経路へ落ちる (`out` は触られていない)。
    if crate::model::exp_pf::stream_enabled() && nt >= crate::model::exp_pf::stream_min_tok() {
        {
            static ONCE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[exp-pf-stream] 有効: min_tok={} group={} pin={} staging={}MiB                      (これ未満は従来経路)",
                    crate::model::exp_pf::stream_min_tok(),
                    crate::model::exp_pf::stream_group(),
                    crate::model::exp_pf::stream_pin(),
                    crate::model::exp_pf::stream_bytes() >> 20,
                );
            }
        }
        // task 309: 行の少ない expert は CPU に残す (重み 1 本 3 MB の転送が
        // 計算より高い)。K 系のときだけ (IQ 系の既存の挙動は変えない)。
        // task 419: IQ 系にも同じ床を置けるようにした (`DENNO_PF_EXP_IQ_MIN_ROWS`。
        // 既定 0 = 従来どおり IQ は全部上げる)。既定モデルの expert は IQ 系の
        // 混在なので、行が少ない expert の 2 MB を上げるか CPU で回すかを選べる。
        let iq_mode = iq_expert_mode(lw);
        let min_rows = if iq_mode {
            crate::model::exp_pf::iq_min_rows()
        } else {
            crate::model::exp_pf::k_min_rows()
        };
        let floor_on = if iq_mode {
            min_rows > 0
        } else {
            crate::model::exp_pf::k_enabled() && min_rows > 0
        };
        let split = floor_on
            && by_expert.iter().any(|p| {
                !p.is_empty() && !crate::model::exp_pf::k_expert_pays(p.len(), min_rows)
            });
        if split {
            let mut gpu_be: Vec<Vec<(usize, usize)>> = Vec::with_capacity(by_expert.len());
            let mut cpu_be: Vec<Vec<(usize, usize)>> = Vec::with_capacity(by_expert.len());
            for picks in by_expert.iter() {
                if !picks.is_empty() && !crate::model::exp_pf::k_expert_pays(picks.len(), min_rows) {
                    gpu_be.push(Vec::new());
                    cpu_be.push(picks.clone());
                } else {
                    gpu_be.push(picks.clone());
                    cpu_be.push(Vec::new());
                }
            }
            // GPU 側 (行のある expert が 1 本でもあれば回す)。戻り値は GPU に
            // 回した expert 数で、`out` は GPU の加重和で**上書き**される。
            let mut ran = 0u32;
            let mut ok = true;
            if gpu_be.iter().any(|p| !p.is_empty()) {
                match iq_pf_stream_layer(
                    lw,
                    &gpu_be,
                    &resident,
                    weights,
                    k,
                    PfStreamTgt::Host { xs, out },
                    nt,
                    e,
                    ff,
                )? {
                    Some(h) => ran = h,
                    None => ok = false,
                }
            }
            if ok {
                // CPU 側の寄与を上に足す (flat は `out` へ加算する)。
                moe_experts_flat(
                    lw,
                    &cpu_be,
                    weights,
                    k,
                    xs,
                    e,
                    ff,
                    MOE_FLAT_ROWS,
                    out,
                    moe_expert::vnni_enabled(),
                    None,
                )?;
                return Ok(Some(ran));
            }
        }
        if let Some(gpu) = iq_pf_stream_layer(
            lw,
            by_expert,
            &resident,
            weights,
            k,
            PfStreamTgt::Host { xs, out },
            nt,
            e,
            ff,
        )? {
            return Ok(Some(gpu));
        }
    }
    if (hits as usize) < crate::model::exp_pf::min_hits_per_layer() {
        crate::model::exp_pf::note_skip_layer();
        return Ok(None);
    }
    // segment 表 (expert 昇順・行はトークン昇順)。devs も同じ順で積んである。
    let plan = crate::model::exp_pf::build_plan(by_expert, &pick);
    if plan.total_rows == 0 || plan.nseg() != devs.len() {
        return Ok(None);
    }
    let ptr_table = |m: ExpertMatrix| -> Vec<u64> { devs.iter().map(|d| d.ptr(m)).collect() };
    let (pg, pu, pd) = (
        ptr_table(ExpertMatrix::Gate),
        ptr_table(ExpertMatrix::Up),
        ptr_table(ExpertMatrix::Down),
    );
    // ミスは CPU の flat 経路で計算する。cuda 側は GPU を**起動した直後**に
    // これを呼ぶので、カーネルが走っている間に CPU が進む
    // (層あたり GPU ~6 ms / CPU ~57 ms)。
    let mut miss_be: Vec<Vec<(usize, usize)>> = Vec::with_capacity(by_expert.len());
    for (ei, picks) in by_expert.iter().enumerate() {
        if pick[ei] {
            miss_be.push(Vec::new());
        } else {
            miss_be.push(picks.clone());
        }
    }
    // 足し込みの表: トークンごとに (重みの添字, 行番号, ヒットか) を
    // **expert index 昇順**で並べる (flat の scatter と同じ順序)。
    // 行番号はヒットなら GPU の h、ミスなら flat が返す h の添字。
    let mut hoff = vec![usize::MAX; by_expert.len()];
    for (s, &ei) in plan.experts.iter().enumerate() {
        hoff[ei] = plan.seg[s * 4 + 1] as usize;
    }
    let mut moff = vec![usize::MAX; by_expert.len()];
    {
        let mut off = 0usize;
        for (ei, picks) in miss_be.iter().enumerate() {
            if picks.is_empty() {
                continue;
            }
            moff[ei] = off;
            off += picks.len();
        }
    }
    let mut order: Vec<Vec<(u32, u32, bool)>> = vec![Vec::new(); nt];
    for (ei, picks) in by_expert.iter().enumerate() {
        let (ho, mo) = (hoff[ei], moff[ei]);
        for (idx, &(t, j)) in picks.iter().enumerate() {
            let wi = (t * k + j) as u32;
            if ho != usize::MAX {
                order[t].push((wi, (ho + idx) as u32, true));
            } else {
                order[t].push((wi, (mo + idx) as u32, false));
            }
        }
    }
    // GPU の行 (total_rows × e) と flat が返すミス行。どちらも層をまたいで
    // バッファを使い回す (毎層 30 MB を確保するとページフォルトだけで数 ms)。
    let mut hit_h = PF_H.with(|c| std::mem::take(&mut *c.borrow_mut()));
    hit_h.resize(plan.total_rows * e, 0.0);
    let mut miss_h = PF_MH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let t_gpu = std::time::Instant::now();
    let ran = {
        let mut cpu_miss = || -> anyhow::Result<()> {
            // capture 付き: `out` は触らず、h を `miss_h` に返す
            // (足し込みは GPU の行と一緒にこの層の後半で 1 回だけ行う)。
            moe_experts_flat(
                lw,
                &miss_be,
                weights,
                k,
                xs,
                e,
                ff,
                MOE_FLAT_ROWS,
                out,
                moe_expert::vnni_enabled(),
                Some(&mut miss_h),
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))
        };
        match crate::cuda::moe_experts_iq_pf_layer(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &pg,
            &pu,
            &pd,
            &plan.seg,
            &plan.rows,
            xs,
            e,
            ff,
            e,
            &mut hit_h,
            &mut cpu_miss,
        ) {
            Ok(v) => v,
            Err(err) => {
                // cuda 側は「CPU ミスの計算そのものが落ちた」ときだけ Err を返す。
                // 隠さずに層を落とす。
                PF_H.with(|c| *c.borrow_mut() = std::mem::take(&mut hit_h));
                PF_MH.with(|c| *c.borrow_mut() = std::mem::take(&mut miss_h));
                return Err(one(format!("層 {}: IQ prefill: {err:#}", lw.il)));
            }
        }
    };
    if !ran {
        // GPU を起動しなかった = `out` は 1 要素も書かれていない (fail closed)。
        PF_H.with(|c| *c.borrow_mut() = std::mem::take(&mut hit_h));
        PF_MH.with(|c| *c.borrow_mut() = std::mem::take(&mut miss_h));
        return Ok(None);
    }
    let gpu_ns = t_gpu.elapsed().as_nanos() as u64;
    crate::model::exp_pf::note_layer(plan.total_rows as u32, hits, gpu_ns);
    if std::env::var_os("DENNO_EXP_PF_DBG").is_some() {
        // 層ごとの内訳 (lab で「1 チャンク 1 層あたり何 ms か」を直接読むため)。
        let miss_rows = by_expert
            .iter()
            .filter(|p| !p.is_empty())
            .map(|p| p.len())
            .sum::<usize>()
            - plan.total_rows;
        eprintln!(
            "[exp-pf-dbg] layer={} nt={} nseg={} rows={} hits={} gpu_ms={:.3} \
             d2h_MB={:.2} miss_rows={}",
            lw.il,
            nt,
            plan.nseg(),
            plan.total_rows,
            hits,
            gpu_ns as f64 / 1e6,
            (plan.total_rows * e * 4) as f64 / 1e6,
            miss_rows,
        );
    }

    // 2. ヒット (GPU) とミス (CPU) を **expert index 昇順で 1 本ずつ**足す。
    //    この順序は flat の scatter と同じなので、CPU だけの経路とビット一致する。
    let t_add = std::time::Instant::now();
    exp_pf_add_all(out, &order, &hit_h, &miss_h, weights, e);
    let add_ns = t_add.elapsed().as_nanos() as u64;
    crate::model::exp_pf::note_add_ns(add_ns);
    crate::model::pf_host::add_ns(crate::model::pf_host::ADD, add_ns);

    // 3. 診断 (`DENNO_EXPERT_GPU_CHECK=1`): GPU の行を CPU で引き直して比べる。
    if crate::model::expert_vram::check_enabled() && crate::model::exp_pf::claim_check_layer() {
        let c = iq_pf_check_layer(lw, &plan, xs, e, ff, &hit_h);
        if c.rows > 0 {
            eprintln!(
                "[exp-pf-check] layer={} rows={} max_abs={:.3e} max_rel={:.3e} argmax={}/{}",
                lw.il, c.rows, c.max_abs, c.max_rel, c.argmax, c.rows
            );
        }
        crate::model::exp_pf::note_check(c);
    }

    // 4. ミスを次のチャンクのために上げる (空きがある間だけ・上限つき)。
    iq_pf_warm_misses(lw, by_expert, &pick, e, ff, p.n_layer);
    PF_H.with(|c| *c.borrow_mut() = std::mem::take(&mut hit_h));
    PF_MH.with(|c| *c.borrow_mut() = std::mem::take(&mut miss_h));
    Ok(Some(hits))
}

/// task 233: prefill expert ストリーミングの入出力。
///
/// `Dev` は層優先 prefill 用で、入力 (ffn mix の出力) も出力 (この層の結果) も
/// デバイスに置いたままにする。`sh` は共有 expert の出力の行 (`0` なら無し)。
#[cfg(feature = "cuda")]
pub(crate) enum PfStreamTgt<'a> {
    Host {
        xs: &'a [f32],
        out: &'a mut [f32],
    },
    Dev {
        xs: u64,
        out: u64,
        sh: u64,
        sh_w: &'a [f32],
    },
}

/// task 162: ミス expert をステージング領域へ上げ、**行のある expert を全部**
/// m 行カーネルで GPU に回す。
///
/// 上げ直しと計算の重なりは [`crate::cuda::moe_experts_iq_pf_layer_stream`] が
/// 持つ (グループごとのダブルバッファ)。ここは「どの expert が常駐で、どれが
/// ミスか」を plan 順に並べ、行表を作るだけ。
///
/// 戻り値 `Some(本数)` = `out` に全 expert の寄与が入っている。
/// `None` = この経路を使わなかった (`out` は触っていない)。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn iq_pf_stream_layer(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    resident: &[Option<std::sync::Arc<crate::model::expert_vram::ExpertDev>>],
    weights: &[f32],
    k: usize,
    tgt: PfStreamTgt<'_>,
    nt: usize,
    e: usize,
    ff: usize,
) -> Result<Option<u32>> {
    let ne = by_expert.len();
    if ne == 0 || nt == 0 || k == 0 || weights.len() < nt * k {
        crate::cuda::pf_stream_why_set("args");
        return Ok(None);
    }
    // task 227: 転送経路の素の速度を 1 回だけ測る (診断。既定 off)。
    // ここに来るのは streaming が実際に走るときだけなので、GPU 文脈も出来ている。
    if crate::model::exp_pf::dma_probe_on() {
        crate::cuda::pf_dma_probe(lw.gate_exps.full());
    }
    // task 254: entries の組み立て + plan + 行表 (層ごとのホスト仕事)。
    let t_plan = crate::model::pf_host::stamp();
    // 行のある expert を **expert 昇順**に並べる (plan と同じ順)。
    let mut pick = vec![false; ne];
    // task 260: RAM tier に載っている expert は mmap ではなく tier の区間から
    // 上げる (冷えた page cache でも disk を読まない)。参照は `entries` が借りる
    // ので、entries を作る前に全部引いて、そのあと触らない。
    let mut ram: Vec<Option<[std::sync::Arc<crate::model::expert_cache::Region>; 3]>> =
        vec![None; ne];
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        if resident.get(ei).and_then(|r| r.as_ref()).is_some() {
            continue;
        }
        ram[ei] = ram_regions(lw, ei, crate::model::expert_ram::adapt_enabled());
    }
    let mut entries: Vec<crate::cuda::PfStreamEntry<'_>> = Vec::with_capacity(ne);
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        pick[ei] = true;
        match resident.get(ei).and_then(|r| r.as_ref()) {
            Some(d) => {
                entries.push(crate::cuda::PfStreamEntry {
                    expert: ei as u16,
                    dev: Some([
                        d.ptr(ExpertMatrix::Gate),
                        d.ptr(ExpertMatrix::Up),
                        d.ptr(ExpertMatrix::Down),
                    ]),
                    host: None,
                    src: None,
                });
            }
            None => match &ram[ei] {
                Some(r) => entries.push(crate::cuda::PfStreamEntry {
                    expert: ei as u16,
                    dev: None,
                    host: Some([r[0].as_slice(), r[1].as_slice(), r[2].as_slice()]),
                    src: None,
                }),
                None => {
                    let host = [
                        lw.gate_exps.slice(ei)?,
                        lw.up_exps.slice(ei)?,
                        lw.down_exps.slice(ei)?,
                    ];
                    // task 325: `DENNO_PF_PREAD` 用の (パート, 絶対オフセット)。
                    // `slice` と同じ `ExpertSlice` から出るので範囲は必ず一致する。
                    let src = match (
                        lw.gate_exps.slice_abs(ei),
                        lw.up_exps.slice_abs(ei),
                        lw.down_exps.slice_abs(ei),
                    ) {
                        (Ok((ga, _)), Ok((ua, _)), Ok((da, _))) => Some([
                            (lw.gate_exps.file as u32, ga),
                            (lw.up_exps.file as u32, ua),
                            (lw.down_exps.file as u32, da),
                        ]),
                        _ => None,
                    };
                    entries.push(crate::cuda::PfStreamEntry {
                        expert: ei as u16,
                        dev: None,
                        host: Some(host),
                        src,
                    });
                }
            },
        }
    }
    if entries.is_empty() {
        crate::cuda::pf_stream_why_set("entries");
        return Ok(None);
    }
    // task 510: この層で「選択した expert 全体」と「VRAM 常駐から回せたぶん」を
    // 積む。`up_MB` はステージングへ上げたぶんしか数えないので、常駐が実際に
    // 効いているかはこの台帳でしか読めない。
    {
        let per = (|| -> Option<usize> {
            let rg = crate::model::exp_pf::row_bytes(lw.gate_exps.ty, e)?;
            let ru = crate::model::exp_pf::row_bytes(lw.up_exps.ty, e)?;
            let rd = crate::model::exp_pf::row_bytes(lw.down_exps.ty, ff)?;
            Some(rg * ff + ru * ff + rd * e)
        })();
        let n_all = entries.len();
        let n_res = entries.iter().filter(|en| en.dev.is_some()).count();
        let per = per.unwrap_or(0);
        crate::model::exp_pf::note_stream_residency(n_res, n_res * per, n_all, n_all * per);
    }
    let plan = crate::model::exp_pf::build_plan(by_expert, &pick);
    if plan.total_rows == 0 || plan.nseg() != entries.len() {
        crate::cuda::pf_stream_why_set("plan");
        return Ok(None);
    }
    // 全 expert が GPU に載るので、行表に穴は無い (穴があれば従来経路へ)。
    let Some(prow) = crate::model::exp_pf::row_table_full(by_expert, &plan, k, nt) else {
        crate::cuda::pf_stream_why_set("row_table");
        return Ok(None);
    };
    crate::model::pf_host::mark(t_plan, crate::model::pf_host::PLAN);
    // task 333: `serial` = 上げスレッドを使わずこのスレッドで写す (再試行用)。
    let mut tgt = tgt;
    let mut run = |tgt: &mut PfStreamTgt<'_>, serial: bool| match tgt {
        PfStreamTgt::Host { xs, out } => crate::cuda::moe_experts_iq_pf_layer_stream(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &entries,
            &plan.seg,
            &plan.rows,
            xs,
            e,
            ff,
            e,
            nt,
            k,
            &prow,
            &weights[..nt * k],
            crate::model::exp_pf::stream_group(),
            out,
            serial,
        ),
        PfStreamTgt::Dev {
            xs,
            out,
            sh,
            sh_w,
        } => crate::cuda::moe_experts_iq_pf_layer_stream_dev(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &entries,
            &plan.seg,
            &plan.rows,
            *xs,
            e,
            ff,
            e,
            nt,
            k,
            &prow,
            &weights[..nt * k],
            crate::model::exp_pf::stream_group(),
            *sh,
            sh_w,
            *out,
            serial,
        ),
    };
    let mut r = run(&mut tgt, false);
    // task 333: 上げスレッドが居ない / 表の対応が合わない、で落ちたときは
    // **1 回だけ** 写しをこのスレッドへ落としてやり直す (従来は層ごと CPU へ
    // 落ちていた)。数値経路は同じカーネル・同じ順序で、変わるのは転送の担い手だけ。
    if matches!(r, Ok(None)) {
        let why = crate::cuda::pf_stream_why();
        if crate::model::exp_pf::stream_retry() && crate::model::exp_pf::stream_retry_pays(why) {
            crate::model::exp_pf::note_stream_retry();
            r = run(&mut tgt, true);
        }
    }
    r.map_err(|err| one(format!("層 {}: expert streaming: {err:#}", lw.il)))
}

/// task 151: ヒット (GPU) とミス (CPU) の行を、**トークンごとに expert index
/// 昇順で 1 本ずつ** `out` へ足す。
///
/// `order[t]` は (重みの添字, h の行番号, ヒットか) を expert index 昇順で並べた
/// もの。`out` はゼロ初期化済みで、トークンは独立なので並列化しても値は変わらない。
///
/// これが「丸めが VRAM のヒット率に依存しない」ための中心で、ミスとヒットを
/// 別々に足す (以前の形) と和の順序が内訳で変わってしまう。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn exp_pf_add_all(
    out: &mut [f32],
    order: &[Vec<(u32, u32, bool)>],
    hit_h: &[f32],
    miss_h: &[f32],
    weights: &[f32],
    e: usize,
) {
    let add_one = |(t, o): (usize, &mut [f32])| {
        for &(wi, row, hit) in &order[t] {
            let pe = weights[wi as usize];
            let src = if hit { hit_h } else { miss_h };
            let hh = &src[row as usize * e..(row as usize + 1) * e];
            for d in 0..e {
                o[d] += pe * hh[d];
            }
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(e.max(1)).enumerate().for_each(add_one);
    }
    #[cfg(not(feature = "par"))]
    {
        out.chunks_mut(e.max(1)).enumerate().for_each(add_one);
    }
}

thread_local! {
    /// GPU の行 (total_rows × e)。層ごとに作り直さない (ページフォルトを避ける)。
    pub(crate) static PF_H: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// CPU ミスの行 (ミスした行数 × e)。flat が capture で埋める。
    pub(crate) static PF_MH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// task 151: prefill の GPU 行を CPU で引き直して比べる
/// (`DENNO_EXPERT_GPU_CHECK=1`。診断専用・既定 off)。
///
/// 1 層あたり先頭から [`crate::model::exp_pf::check_rows`] 行だけ、decode と
/// 同じ 1 行ずつの経路 (`gemv` + [`silu_mul`]) で計算し、GPU の行と比べる。
/// 丸めの差 (max_abs が 1e-3 相対くらいまで) は正常、それより大きい差はバグ。
#[cfg(feature = "cuda")]
pub(crate) fn iq_pf_check_layer(
    lw: &LayerW,
    plan: &crate::model::exp_pf::PfPlan,
    xs: &[f32],
    e: usize,
    ff: usize,
    hit_h: &[f32],
) -> crate::model::exp_pf::PfCheck {
    let mut left = crate::model::exp_pf::check_rows();
    let (mut g, mut u) = (vec![0.0f32; ff], vec![0.0f32; ff]);
    let mut act = vec![0.0f32; ff];
    let mut hc = vec![0.0f32; e];
    let mut acc = crate::model::exp_pf::PfCheck {
        layers: 1,
        ..Default::default()
    };
    for (s, &ei) in plan.experts.iter().enumerate() {
        if left == 0 {
            break;
        }
        let m = plan.seg[s * 4] as usize;
        let ro = plan.seg[s * 4 + 1] as usize;
        for idx in 0..m.min(left) {
            left -= 1;
            let t = plan.rows[ro + idx] as usize;
            let x = &xs[t * e..(t + 1) * e];
            if lw.gate_exps.gemv(ei, x, &mut g).is_err() || lw.up_exps.gemv(ei, x, &mut u).is_err()
            {
                continue;
            }
            for r in 0..ff {
                act[r] = silu_mul(g[r], u[r]);
            }
            if lw.down_exps.gemv(ei, &act, &mut hc).is_err() {
                continue;
            }
            let gpu_row = &hit_h[(ro + idx) * e..(ro + idx + 1) * e];
            let (mut ma, mut mr) = (0.0f32, 0.0f32);
            for i in 0..e {
                let d = (gpu_row[i] - hc[i]).abs();
                if d > ma {
                    ma = d;
                }
                let rel = d / hc[i].abs().max(1e-6);
                if rel > mr {
                    mr = rel;
                }
            }
            if argmax_of(gpu_row) == argmax_of(&hc) {
                acc.argmax += 1;
            }
            acc.rows += 1;
            if ma > acc.max_abs {
                acc.max_abs = ma;
            }
            if mr > acc.max_rel {
                acc.max_rel = mr;
            }
        }
    }
    acc
}

/// 最初の最大値の添字 (NaN は無視 = 比較で負ける)。
#[cfg(feature = "cuda")]
pub(crate) fn argmax_of(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// task 148: prefill のミス expert を **チャンク予算つき**で VRAM へ上げる。
///
/// 上げるのは「このチャンクで行数が多い」ミスだけ (既定 8 行以上)。短い prefill
/// (19 トークン) では 1 expert あたり平均 0.7 行なので 1 本も上がらない。
///
/// **LRU に空きがある間だけ**動く。埋まっているときの上げ直しは「1 本上げるために
/// 1 本追い出す」で、次にその expert を使うチャンスは同じだけ減る (追い出された
/// 方は次のチャンクでミスになる)。差し引きゼロに H2D の時間 (実測 0.27 ms/本) が
/// 乗るだけなので、やらない。
#[cfg(feature = "cuda")]
pub(crate) fn iq_pf_warm_misses(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    hit: &[bool],
    e: usize,
    ff: usize,
    n_layer: usize,
) {
    use crate::model::expert_cache::ExpertKey;
    use crate::model::expert_vram::ExpertShape;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// このチャンクでまだ上げられる本数。
    static LEFT: AtomicUsize = AtomicUsize::new(0);
    if lw.il == 0 {
        // prefill のチャンク先頭 (層 0) で予算を戻す。
        LEFT.store(crate::model::exp_pf::upload_budget_experts(), Ordering::Relaxed);
    }
    let budget = LEFT.load(Ordering::Relaxed);
    if budget == 0 {
        return;
    }
    // 空きが無ければ何もしない (上のコメント)。
    let free = match expert_vram::cache_usage() {
        Some((used, cap)) => cap.saturating_sub(used) > (32 << 20),
        None => false,
    };
    if !free {
        return;
    }
    let min_rows = crate::model::exp_pf::upload_min_rows();
    let mut cand: Vec<(usize, usize)> = Vec::new();
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() || hit.get(ei).copied().unwrap_or(false) {
            continue;
        }
        if picks.len() >= min_rows {
            cand.push((picks.len(), ei));
        }
    }
    if cand.is_empty() {
        return;
    }
    // 行の多い順 (同数は expert 昇順)。
    cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    let nl = n_layer.max(1);
    let quota = (budget / nl).max(1).min(budget);
    let mut left = budget;
    for &(_, ei) in cand.iter().take(quota) {
        let key = ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        };
        let t = std::time::Instant::now();
        // task 260: 転送元は RAM tier → mmap の順 (`acquire_expert`)。
        let got = acquire_expert(lw, ei, key, shapes);
        match got {
            Ok(Some(_)) => {
                crate::model::exp_pf::note_upload_ns(t.elapsed().as_nanos() as u64);
                left = left.saturating_sub(1);
            }
            // 確保失敗・型未対応はこのチャンクでは諦める (リクエストは落とさない)。
            _ => break,
        }
    }
    LEFT.store(left, Ordering::Relaxed);
}

/// [`moe_ffn_quant`] のバッチ版 (`nt` トークン)。
///
/// ルータ / shared expert / 選択 expert の行列積を [`gemv_batch`] に置き換える。
/// 同じ expert を選んだトークンはまとめて GEMV する。
///
/// 数値: トークンごとの softmax → top-k → 再正規化 → **index 昇順**の重み付き加算、
/// という順序をそのまま保つ。1 トークンの `h_e` は (行, トークン) ごとに逐次版と
/// 同じカーネルで計算され、足し込みは index 昇順の逐次なので f32 でビット一致する。
///
/// キャッシュ経路 (`cache: Some`) は [`ExpertCacheSet`] が `&mut` でスロットを
/// 共有するためバッチ化しない (トークンごとに逐次版へ落とす)。
pub(crate) fn moe_ffn_quant_batch(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&ExpertCacheSet>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_batch_with(p, lw, xs, nt, cache, moe_schedule())
}

/// P54: 役割別 RMS (チャンク末尾のトークン) も取る [`moe_ffn_quant_batch`]。
///
/// `emit_router` は P56 の RouterTopK を送るか (チャンク末尾の 1 トークンだけ)。
pub(crate) fn moe_ffn_quant_batch_tapped(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&ExpertCacheSet>,
    roles: Option<&mut RoleRms>,
    emit_router: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_batch_full(
        p,
        lw,
        xs,
        nt,
        cache,
        moe_schedule(),
        moe_expert::vnni_enabled(),
        roles,
        emit_router,
        iv,
    )
}

/// task 189: router のソフトマックスの行割り。
///
/// `Some(n)` は「トークンを n ブロックに割って 1 タスク 1 ブロック」、
/// `None` は従来の「1 行 1 タスク」。行は独立なので値はどちらでも同じ。
pub(crate) fn router_sm_block() -> Option<usize> {
    #[cfg(feature = "par")]
    {
        if pf_router_sm_on() {
            return Some(rayon::current_num_threads().max(1));
        }
    }
    None
}

/// top-k の再正規化 (選択 index 昇順に逐次加算した和で割る)。
pub(crate) fn router_renorm(pr: &[f32], sel: &[usize], w: &mut [f32]) {
    let mut sum = 0.0f32;
    for &i in sel {
        sum += pr[i];
    }
    // task 217: 介入で支配的な expert を切ると、残りの確率が softmax の
    // アンダーフローで全部 0 になりうる (0/0 = NaN)。そのときだけ一様に分ける。
    if sum.is_finite() && sum > 0.0 {
        for (j, &i) in sel.iter().enumerate() {
            w[j] = pr[i] / sum;
        }
    } else {
        let u = 1.0f32 / sel.len().max(1) as f32;
        for x in w.iter_mut() {
            *x = u;
        }
    }
}

/// task 189: 行ごとの softmax (block が `Some` ならトークンをブロックで割る)。
pub(crate) fn router_softmax_rows(probs: &mut [f32], ne: usize, block: Option<usize>) {
    let nt = probs.len() / ne.max(1);
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        match block {
            Some(n) => {
                let per = nt.div_ceil(n.max(1)).max(1);
                probs.par_chunks_mut(per * ne).for_each(|c| {
                    for row in c.chunks_mut(ne) {
                        softmax(row);
                    }
                });
            }
            None => probs.par_chunks_mut(ne).for_each(softmax),
        }
        return;
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = block;
        for t in 0..nt {
            softmax(&mut probs[t * ne..(t + 1) * ne]);
        }
    }
}

/// task 189: 行ごとの top-k + 再正規化 (block が `Some` ならトークンを割る)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn router_topk_rows(
    probs: &[f32],
    ne: usize,
    k: usize,
    sels: &mut [usize],
    weights: &mut [f32],
    block: Option<usize>,
    fast: bool,
    mask: Option<&[u64]>,
) {
    let nt = probs.len() / ne.max(1);
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        match block {
            Some(n) => {
                let per = nt.div_ceil(n.max(1)).max(1);
                sels.par_chunks_mut(per * k)
                    .zip(weights.par_chunks_mut(per * k))
                    .enumerate()
                    .for_each_init(Vec::<u32>::new, |buf, (ci, (s, w))| {
                        let base = ci * per;
                        for j in 0..s.len() / k {
                            let t = base + j;
                            let pr = &probs[t * ne..(t + 1) * ne];
                            let (sj, wj) =
                                (&mut s[j * k..(j + 1) * k], &mut w[j * k..(j + 1) * k]);
                            // task 195: `DENNO_PF_ROUTER_FAST` は挿入法 (結果は同一)。
                            // task 217: 無効 expert は候補から外す (mask が None なら従来)。
                            if mask.is_some() {
                                select_topk_into_mask(pr, k, mask, buf, sj);
                            } else if fast {
                                select_topk_into_fast(pr, k, sj);
                            } else {
                                select_topk_into(pr, k, buf, sj);
                            }
                            router_renorm(pr, sj, wj);
                        }
                    });
            }
            None => {
                sels.par_chunks_mut(k)
                    .zip(weights.par_chunks_mut(k))
                    .enumerate()
                    .for_each_init(Vec::<u32>::new, |buf, (t, (s, w))| {
                        let pr = &probs[t * ne..(t + 1) * ne];
                        if mask.is_some() {
                            select_topk_into_mask(pr, k, mask, buf, s);
                        } else if fast {
                            select_topk_into_fast(pr, k, s);
                        } else {
                            select_topk_into(pr, k, buf, s);
                        }
                        router_renorm(pr, s, w);
                    });
            }
        }
        return;
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = (block, fast);
        let mut buf: Vec<u32> = Vec::new();
        for t in 0..nt {
            let pr = &probs[t * ne..(t + 1) * ne];
            select_topk_into_mask(pr, k, mask, &mut buf, &mut sels[t * k..(t + 1) * k]);
            router_renorm(pr, &sels[t * k..(t + 1) * k], &mut weights[t * k..(t + 1) * k]);
        }
    }
}

#[cfg(test)]
mod router_sm_tests {
    use super::*;

    /// task 189: 行の割り方 (1 行 1 タスク / n ブロック) を変えても
    /// softmax・top-k・再正規化の値がビット一致すること。
    ///
    /// 端数が出る形 (nt=7, ne=8, k=3, ブロック数 3) にして、最後の切れ端と
    /// `base` の計算も通す。同値の選択規則 (小さい index が先) も固定する。
    #[test]
    fn blocked_rows_match_per_row() {
        let (nt, ne, k) = (7usize, 8usize, 3usize);
        let xs: Vec<f32> = (0..nt * ne)
            .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
            .collect();

        let mut a = xs.clone();
        router_softmax_rows(&mut a, ne, None);
        let mut b = xs.clone();
        router_softmax_rows(&mut b, ne, Some(3));
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "softmax 要素 {i}");
        }
        // 各行の和は 1 (softmax が実際に効いている)。
        for t in 0..nt {
            let s: f32 = a[t * ne..(t + 1) * ne].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "行 {t} の和 {s}");
        }

        let mut s1 = vec![0usize; nt * k];
        let mut w1 = vec![0f32; nt * k];
        router_topk_rows(&a, ne, k, &mut s1, &mut w1, None, false, None);
        let mut s2 = vec![0usize; nt * k];
        let mut w2 = vec![0f32; nt * k];
        router_topk_rows(&a, ne, k, &mut s2, &mut w2, Some(3), false, None);
        assert_eq!(s1, s2, "top-k の選択が割り方で変わった");
        for i in 0..w1.len() {
            assert_eq!(w1[i].to_bits(), w2[i].to_bits(), "重み 要素 {i}");
        }

        // 同値だらけの行: 小さい index から k 個 (task 182 の規則)。
        let flat = vec![1.0f32; ne];
        let mut s3 = vec![0usize; k];
        let mut w3 = vec![0f32; k];
        router_topk_rows(&flat, ne, k, &mut s3, &mut w3, None, false, None);
        assert_eq!(s3, vec![0, 1, 2], "同値のとき小さい index が先でない");
        for v in w3 {
            assert!((v - 1.0 / 3.0).abs() < 1e-6, "再正規化の重み {v}");
        }
    }

    /// task 189: `DENNO_PF_ROUTER_SM` は明示的な真値だけ on (既定 off)。
    #[test]
    fn router_sm_gate_truth_table() {
        assert!(!pf_router_sm_from_value(None), "未設定は既定 off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!pf_router_sm_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(pf_router_sm_from_value(Some(v)), "{v:?} が off のまま");
        }
    }
}

/// task 162: prefill の終端でステージング領域を必ず返すためのガード。
///
/// decode のあいだステージング (既定 512 MiB) を抱えたままにすると、expert の
/// LRU がその分だけ狭くなる。`forward_prefill_inner` の入口で作って Drop で返す。
#[cfg(feature = "cuda")]
pub(crate) struct PfStreamGuard;

/// [`moe_ffn_quant_batch`] の本体。expert ループの並べ方 (`sched`) を外から渡せる
/// 形にしてある (テストと A/B 計測で両方の経路を同じ入力に通すため)。
pub(crate) fn moe_ffn_quant_batch_with(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&ExpertCacheSet>,
    sched: MoeSchedule,
) -> Result<Vec<f32>> {
    moe_ffn_quant_batch_opts(p, lw, xs, nt, cache, sched, moe_expert::vnni_enabled())
}

/// [`moe_ffn_quant_batch_with`] の本体。VNNI 経路の可否 (`vnni`) も外から渡せる
/// 形にしてある (テストが同じ入力で両方の経路を通して比べるため)。
pub(crate) fn moe_ffn_quant_batch_opts(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&ExpertCacheSet>,
    sched: MoeSchedule,
    vnni: bool,
) -> Result<Vec<f32>> {
    moe_ffn_quant_batch_full(
        p,
        lw,
        xs,
        nt,
        cache,
        sched,
        vnni,
        None,
        true,
        IvView::none(),
    )
}

/// [`moe_ffn_quant_batch_opts`] の本体 (+ P54 の役割別 RMS / P56 の RouterTopK)。
///
/// `roles` が `Some` のときだけ、**チャンク末尾のトークン**の router 確率和と
/// shared expert の RMS を書く (prefill の観測はチャンク末尾だけ、の仕様に合わせる)。
/// `emit_router` も同じく**チャンク末尾の 1 トークンだけ**送る。
pub(crate) fn moe_ffn_quant_batch_full(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&ExpertCacheSet>,
    sched: MoeSchedule,
    vnni: bool,
    mut roles: Option<&mut RoleRms>,
    emit_router: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    let e = p.n_embd;
    let ne = p.n_expert;
    let k = p.n_expert_used;
    let ff = p.n_ff_exp;
    if nt == 0 {
        return Ok(Vec::new());
    }
    if xs.len() != nt * e {
        return Err(one(format!(
            "moe_ffn_quant (batch): x の長さ {} が nt({nt}) * n_embd({e}) と違う",
            xs.len()
        )));
    }
    let mut out = Vec::with_capacity(nt * e);
    if let Some(c) = cache {
        // P56: RouterTopK はチャンク末尾の 1 トークンだけ送る (キャッシュ経路は
        // トークンごとに逐次版へ落ちるので、末尾以外は emit_router=false)。
        for t in 0..nt {
            // P54: 役割別 RMS はチャンク末尾のトークンだけ取る。
            // P56: RouterTopK も同じく末尾の 1 トークンだけ送る。
            let rr: Option<&mut RoleRms> =
                if t + 1 == nt { roles.as_deref_mut() } else { None };
            let er = emit_router && t + 1 == nt;
            out.extend_from_slice(&moe_ffn_quant_full(
                p,
                lw,
                &xs[t * e..(t + 1) * e],
                Some(c),
                moe_par_enabled(),
                expert_route(),
                rr,
                er,
                iv,
            )?);
        }
        return Ok(out);
    }

    // 1. ルータ (バッチ)
    // task 145: 台帳の区間 (router)。gate_inp の GEMV から top-k の確定まで。
    let _l_router = q4ledger::enter(q4ledger::ROUTER);
    let t_router = std::time::Instant::now();
    // task 217: 無効 expert は top-k の候補から外す (層に無効が無ければ None)。
    let expert_mask = iv.expert_mask(lw.il);
    // ホスト経路のときだけ確保する (デバイス経路では logits を降ろさない)。
    let mut probs: Vec<f32> = Vec::new();
    let mut sels: Vec<usize> = vec![0; nt * k];
    let mut weights: Vec<f32> = vec![0.0; nt * k];
    // task 154: 台帳の副区間 (softmax + top-k)。
    let _sub_sm = q4ledger::sub(q4ledger::SUB_SOFTMAX);
    // task 225: router の softmax と top-k をデバイスで回す (`DENNO_PF_ROUTER_DEV`)。
    // logits もデバイスで作り、選んだ k 個の index と重みだけを受け取る。
    // 使えないときは下の従来経路 (ホスト) がそのまま走る。
    #[cfg(feature = "cuda")]
    let mut router_tail_sum: Option<f32> = None;
    #[cfg(feature = "cuda")]
    {
        let dev_ok = crate::cuda::pf_router_dev_on()
            && lw.gate_inp.k() == e
            && expert_mask.is_none()
            && nt >= PF_ROUTER_GPU_MIN_TOK
            && crate::cuda::pf_router_dev_ok(nt, ne, k);
        if dev_ok {
            let mut sel32 = vec![0u32; nt * k];
            let mut rowsum = vec![0.0f32; nt];
            let ok = match crate::cuda::pf_mixed_dev_in_ptr(xs.as_ptr(), nt, e) {
                Some(dp) => crate::cuda::pf_router_dev_topk(
                    lw.gate_inp.uid(),
                    lw.gate_inp.blocks().as_slice(),
                    lw.gate_inp.k(),
                    lw.gate_inp.n(),
                    Some(dp),
                    None,
                    nt,
                    k,
                    &mut sel32,
                    &mut weights,
                    &mut rowsum,
                    GgmlType::F32,
                ),
                None => crate::cuda::pf_router_dev_topk(
                    lw.gate_inp.uid(),
                    lw.gate_inp.blocks().as_slice(),
                    lw.gate_inp.k(),
                    lw.gate_inp.n(),
                    None,
                    Some(xs),
                    nt,
                    k,
                    &mut sel32,
                    &mut weights,
                    &mut rowsum,
                    GgmlType::F32,
                ),
            };
            if ok {
                for (i, v) in sel32.iter().enumerate() {
                    sels[i] = *v as usize;
                }
                router_tail_sum = rowsum.last().copied();
            }
        }
    }
    #[cfg(feature = "cuda")]
    let router_dev_done = router_tail_sum.is_some();
    #[cfg(not(feature = "cuda"))]
    let router_dev_done = false;
    if !router_dev_done {
        probs = vec![0.0f32; nt * ne];
        // task 182: router (`ffn_gate_inp`、F32) を GPU の f32 GEMM で回す
        // (`DENNO_PF_ROUTER_GPU=1` のときだけ。失敗・前提外は CPU バッチへ)。
        #[cfg(feature = "cuda")]
        let dev_router_in = if lw.gate_inp.k() == e {
            crate::cuda::pf_mixed_dev_in_ptr(xs.as_ptr(), nt, e)
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let gpu_router = pf_router_gpu_on()
            && nt >= PF_ROUTER_GPU_MIN_TOK
            && match dev_router_in {
                // task 209: 入力が直近の mix の出力なら、デバイスの `mixed` を直接読む
                // (`DENNO_PF_MIXDEVIN`)。値はホストから上げるのと同じバイト列。
                Some(dp) => crate::cuda::maybe_gpu_f32_gemm_batch_dev_into(
                    lw.gate_inp.uid(),
                    lw.gate_inp.blocks().as_slice(),
                    lw.gate_inp.k(),
                    lw.gate_inp.n(),
                    dp,
                    nt,
                    &mut probs,
                    GgmlType::F32,
                ),
                None => crate::cuda::maybe_gpu_f32_gemm_batch_into(
                    lw.gate_inp.uid(),
                    lw.gate_inp.blocks().as_slice(),
                    lw.gate_inp.k(),
                    lw.gate_inp.n(),
                    xs,
                    nt,
                    &mut probs,
                    GgmlType::F32,
                ),
            };
        #[cfg(not(feature = "cuda"))]
        let gpu_router = false;
        if gpu_router {
            PF182_LOG_ROUTER.call_once(|| {
                eprintln!("[pf182] prefill router: GPU f32 GEMM (DENNO_PF_ROUTER_GPU)")
            });
        }
        if !gpu_router {
            gemv_batch(&lw.gate_inp, xs, nt, &mut probs)
                .map_err(|err| one(format!("層 {}: ffn_gate_inp (batch): {err}", lw.il)))?;
        }

        // 2. トークンごとの softmax (行は独立なので並列化しても値は変わらない)
        // task 189: `DENNO_PF_ROUTER_SM` のときは行ごとのタスク生成をやめる。
        let sm_block = router_sm_block();
        router_softmax_rows(&mut probs, ne, sm_block);

        // 3. top-k (index 昇順) + 再正規化 (選択 index 昇順に逐次加算した和で割る)。
        //    P33: トークンごとの `Vec` を作らず、平らな配列へ直接書く (1 層あたり
        //    1,024 回の確保が 2 回になる)。選択規則は [`select_topk_into`] が
        //    [`select_topk`] と同じ (同値は小さい index) なので値は変わらない。
        router_topk_rows(
            &probs,
            ne,
            k,
            &mut sels,
            &mut weights,
            sm_block,
            pf_router_fast_on(),
            expert_mask,
        );
    }
    drop(_sub_sm);
    moe_sub_add(0, t_router.elapsed());
    // 区間を閉じる (experts の時間を router に二重計上しない)。
    drop(_l_router);
    // task 263: prefill の選択も数える (8 トークンに 1 回 = 重み 1/8)。
    // decode の履歴を長いプロンプトで塗り潰さないための重みで、コストは
    // `(nt/8) * k` 回の fetch_add (ロックなし・確保なし)。
    crate::model::exp_stats::observe_batch(p.n_layer, p.n_expert, lw.il, &sels, k, nt);
    // task 246: 末尾トークンの選択を控える (prefill の終端で LRU を温め直すため)。
    // ゲート off なら `note_tail_selection` が即座に戻る。
    crate::model::exp_pf::note_tail_selection(lw.il, nt, k, &sels);
    // P56: RouterTopK は**チャンク末尾の 1 トークンだけ**送る。
    //
    // 理由: 全トークンを送ると 1 プロンプトで nt × n_layer 枚 (512 トークン ×
    // 48 層 = 24,576 枚) になり、テレメトリのチャネル (4096) が溢れて
    // Gap だらけになる。ビューアが描くのは「その層が今どの expert を使うか」で、
    // チャンク内の代表 1 トークンで足りる (`TapTokens::Last` と同じ規則)。
    // ここは非キャッシュ経路なので、全トークンの sels / weights が手元にある。
    if emit_router {
        let tail = (nt - 1) * k;
        emit_router_topk(lw.il, &sels[tail..tail + k], &weights[tail..tail + k]);
        // task 220: チャンク末尾のトークンで、介入に消された expert を送る (既定 off)。
        if let Some(m) = expert_mask {
            let row = &probs[(nt - 1) * ne..nt * ne];
            emit_expert_skip(lw.il, row, k, &sels[tail..tail + k], m);
        }
    }

    // task 217: routed expert を切った層は expert の選択後処理・重み読み・寄与計算を
    // 丸ごと飛ばす (out は 0 のまま = 残差に何も足さない)。router 自体は走らせる
    // (shared の係数や telemetry が同じ経路で要るため)。
    let mut out = vec![0.0f32; nt * e];
    // task 145: 台帳の区間 (experts)。選択 expert の寄与計算だけ。
    // task 217: 切っていても区間の開閉は同じ位置で行う (skip 時は 0 秒)。
    let t_experts = std::time::Instant::now();
    let _l_experts = q4ledger::enter(q4ledger::EXPERTS);
    // task 254: `experts` 区間の中のホスト仕事を項目別に割るための壁時計。
    let _pf_host_stage = crate::model::pf_host::stage();
    let skip_routed = iv.block_off(lw.il, IvBlock::Moe);
    let skip_shared = iv.block_off(lw.il, IvBlock::Shared);
    if !skip_routed {
        // 4. 選択 expert の重みを先に読み始める (逐次版と同じ expert を同じ順で)
        //    同じ expert は 1 層で何度も選ばれる。madvise は 1 expert 1 回で足りるので、
        //    トークン×k 回 (512 × 10 = 5,120 回/層) ではなく distinct な expert の数 (86 回/層)
        //    だけ呼ぶ。選択順は変わらないので数値には影響しない。
        let t_pref = crate::model::pf_host::stamp();
        {
            let mut seen = vec![false; ne];
            for &ei in &sels {
                if !seen[ei] {
                    seen[ei] = true;
                    lw.gate_exps.prefetch(ei);
                    lw.up_exps.prefetch(ei);
                    lw.down_exps.prefetch(ei);
                }
            }
        }
        crate::model::pf_host::mark(t_pref, crate::model::pf_host::PREFETCH);

        // 5. expert ごとのトークン一覧 (`picks`) と件数 `m` を 1 パスで確定する。
        //    sels は index 昇順の distinct な expert なので、legacy が
        //    `position()` で作っていた順 (トークン昇順, 同トークン内は j 昇順) と一致する。
        //    件数を先に数えて容量を確保するので、push のたびの再確保は無い。
        let mut counts = vec![0usize; ne];
        for &ei in &sels {
            counts[ei] += 1;
        }
        let t_build = crate::model::pf_host::stamp();
        let mut by_expert: Vec<Vec<(usize, usize)>> = counts
            .iter()
            .map(|&c| Vec::with_capacity(c))
            .collect();
        for t in 0..nt {
            for j in 0..k {
                by_expert[sels[t * k + j]].push((t, j));
            }
        }
        crate::model::pf_host::mark(t_build, crate::model::pf_host::PLAN);
        let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
        let flat = match sched {
            MoeSchedule::Flat => true,
            MoeSchedule::Legacy => false,
            MoeSchedule::Auto => m_max > MOE_FLAT_MIN_M,
        };
        if std::env::var_os("DENNO_MOE_M_DEBUG").is_some() {
            let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
            let sum_m: usize = by_expert.iter().map(|p| p.len()).sum();
            eprintln!(
                "[moe-m] nt={nt} ne={ne} k={k} nonempty={nonempty} sum_m={sum_m} mean={:.2} max={m_max} flat={flat}",
                sum_m as f32 / nonempty.max(1) as f32
            );
        }
        // 5.4 (P40): expert FFN を GPU の grouped GEMM にまとめる。
        //      `DENNO_MOE_GROUPED_GPU=1` のときだけ入る (既定 off)。off のときは
        //      この分岐に入らないので、従来の flat / legacy 経路は 1 バイトも変わらない。
        //      false = GPU が引き受けなかった (env / CUDA / PTX / 形のどれかが欠けた)。
        let mut experts_done = false;
        if crate::model::moe_grouped_gpu::enabled() {
            /// P40: `ExpertW` (mmap のスライス) を grouped GEMM の入力へ。
            fn src<'a>(w: &'a ExpertW) -> crate::model::moe_grouped_gpu::ExpertSource<'a> {
                crate::model::moe_grouped_gpu::ExpertSource {
                    blocks: w.blocks.as_slice(),
                    ty: w.ty,
                    n_in: w.n_in,
                    n_out: w.n_out,
                    n_expert: w.n_expert,
                }
            }
            // 失敗したら CPU 経路へ落とす (この経路は実験的で、既定 off。
            // ここで層を落とすと prefill 全体が落ちるため fail closed にする)。
            match crate::model::moe_grouped_gpu::try_experts(
                lw.il as u64,
                src(&lw.gate_exps),
                src(&lw.up_exps),
                src(&lw.down_exps),
                &by_expert,
                &weights,
                k,
                xs,
                nt,
                e,
                ff,
                &mut out,
            ) {
                Ok(v) => experts_done = v,
                // P42: 層ごとに 1 行出すと実機で 48 行になる。詳細は最初の 1 回だけ
                // 出し、残りは prefill 末尾の集計 1 行に畳む。
                Err(err) => crate::model::moe_grouped_gpu::note_failure(lw.il, &err),
            }
        }
        // task 148: prefill の選択 expert を m 行カーネルで GPU に回す
        // (`DENNO_EXPERT_IQ_GPU=1` と `DENNO_EXPERT_IQ_GPU_PF=1` の両方が on のときだけ)。
        // ヒットだけ GPU、ミスはこの下の flat (CPU)。GPU を使わない/失敗したら `None`
        // が返り、`out` は触られていないので従来の経路がそのまま走る (fail closed)。
        #[cfg(feature = "cuda")]
        if !experts_done {
            match moe_experts_iq_pf_batch(p, lw, &by_expert, &weights, k, xs, nt, e, ff, &mut out) {
                Ok(Some(_)) => experts_done = true,
                Ok(None) => {}
                Err(err) => return Err(err),
            }
        }
        if !experts_done {
            if flat {
                moe_experts_flat(
                    lw,
                    &by_expert,
                    &weights,
                    k,
                    xs,
                    e,
                    ff,
                    MOE_FLAT_ROWS,
                    &mut out,
                    vnni,
                    None,
                )?;
            } else {
                moe_experts_legacy(lw, &by_expert, &weights, k, xs, e, ff, &mut out)?;
            }
        }

    }

    // 5. shared expert (バッチ)。係数は sigmoid(gate_inp_shexp @ x)
    // task 822: 天井の切り分け用に、選択 expert の積和回数と壁時計を積む
    // (`DENNO_PF_ROOFLINE` のときだけ。off は分岐 1 回)。
    if !skip_routed {
        crate::model::qwen4exp::roofline::note(
            lw.il,
            nt,
            nt as u64 * k as u64 * 3 * e as u64 * ff as u64,
            t_experts.elapsed().as_nanos() as u64,
        );
    }
    moe_sub_add(1, t_experts.elapsed());
    // 区間を閉じる (shared の時間を experts に二重計上しない)。
    drop(_l_experts);
    // task 254: ホスト台帳の `experts` 壁時計もここで閉じる (shared を含めない)。
    drop(_pf_host_stage);
    if !skip_shared {
        // task 145: 台帳の区間 (shared)。
        let _l_shared = q4ledger::enter(q4ledger::SHARED);
        let t_shared = std::time::Instant::now();
        let sff = lw.gate_shexp.n();
        // task 209: shared の出力はデバイスからの D2H の宛先なので pinned に置く
        // (`DENNO_PF_PIN`)。中身は下のどちらの経路も全要素を書く。
        let mut sh_buf = PfHostBuf::new(nt * e);
        let sh = sh_buf.as_mut_slice();
        // task 172: 3 射影 (gate / up / down) を 1 チャンクぶんデバイスで回す
        // (活性を GPU に置いたまま。silu と積は elem の swiglu カーネル)。
        // 使えない形・環境では false が返り、下の従来経路がそのまま走る。
        let mut shared_dev = false;
        if pf_shared_dev_on() {
            shared_dev = Qwen4Exp::pf_shared_dev_run(lw, xs, nt, e, sff, sh);
        }
        if !shared_dev {
            let mut sact = vec![0.0f32; nt * sff];
            gemv_batch(&lw.gate_shexp, xs, nt, &mut sact)
                .map_err(|err| one(format!("層 {}: ffn_gate_shexp (batch): {err}", lw.il)))?;
            silu_inplace(&mut sact);
            let mut sup = vec![0.0f32; nt * sff];
            gemv_batch(&lw.up_shexp, xs, nt, &mut sup)
                .map_err(|err| one(format!("層 {}: ffn_up_shexp (batch): {err}", lw.il)))?;
            for r in 0..nt * sff {
                sact[r] *= sup[r];
            }
            gemv_batch(&lw.down_shexp, &sact, nt, sh)
                .map_err(|err| one(format!("層 {}: ffn_down_shexp (batch): {err}", lw.il)))?;
        }
        let gn = lw.gate_inp_shexp.n();
        let mut sg = vec![0.0f32; nt * gn];
        gemv_batch(&lw.gate_inp_shexp, xs, nt, &mut sg)
            .map_err(|err| one(format!("層 {}: ffn_gate_inp_shexp (batch): {err}", lw.il)))?;
        for t in 0..nt {
            let sgw = hyper::sigmoid(sg[t * gn]);
            for d in 0..e {
                out[t * e + d] += sh[t * e + d] * sgw;
            }
        }
        // P54: shared expert の出力そのものの RMS (切っていないときだけ)。
        if let Some(r) = roles.as_deref_mut() {
            let t = nt - 1;
            r.shex = rms_of(&sh[t * e..(t + 1) * e]);
        }
        moe_sub_add(2, t_shared.elapsed());
    }

    // P54: チャンク末尾のトークンの router 確率和 (= 選択 expert の確率の和)。
    // shared を切っていても probs / sels は手元にあるので読むだけ。
    // task 225: デバイス経路ではカーネルが同じ和 (index 昇順の逐次和) を返す。
    if let Some(r) = roles.as_deref_mut() {
        let t = nt - 1;
        #[cfg(feature = "cuda")]
        let dev_sum = router_tail_sum;
        #[cfg(not(feature = "cuda"))]
        let dev_sum: Option<f32> = None;
        let s = match dev_sum {
            Some(v) => v,
            None => {
                let mut s = 0.0f32;
                for j in 0..k {
                    s += probs[t * ne + sels[t * k + j]];
                }
                s
            }
        };
        r.router = s;
    }

    Ok(out)
}

/// task 242: 層優先 prefill が作業域のために追い出した expert を VRAM へ戻す。
///
/// `keys` は `expert_vram::emergency_free_keys` が返した**古い順**。同じ順に
/// `acquire_soft` で入れ直すと、相対的な新しさ (LRU の並び) も元に近づく。
/// 失敗しても致命的ではない (次の decode が `acquire` で上げ直す) ので、
/// `acquire_soft` (ラッチしない方) を使う。
///
/// task 394: `DENNO_PF_RESTORE_MT=1` のときは [`pf_restore_experts_mt`] へ回す
/// (1 本ずつの同期 H2D を N 本で重ねる)。`DENNO_Q4_PROFILE=1` のときは
/// 1 回ぶんの実測を `[pf-restore]` に出す。
#[cfg(feature = "cuda")]
pub(crate) fn pf_restore_experts(
    layers: &[LayerW],
    keys: &[crate::model::expert_cache::ExpertKey],
) -> (usize, usize) {
    let t = std::time::Instant::now();
    let mt = crate::model::exp_pf::pf_restore_mt();
    // task 565: `DENNO_PF_RESTORE_NOSYNC` なら 1 本ごとに待たず、終端で 1 回待つ。
    let nosync = crate::model::exp_pf::pf_restore_nosync();
    if nosync {
        crate::cuda::restore_defer_begin();
    }
    let (n, bytes) = if mt && keys.len() >= 2 {
        pf_restore_experts_mt(layers, keys, nosync)
    } else {
        pf_restore_experts_seq(layers, keys, nosync)
    };
    if nosync {
        crate::cuda::restore_defer_end();
    }
    if crate::model::pf_host::on() && !keys.is_empty() {
        eprintln!(
            "[pf-restore] {} 本 / {} MiB を戻した ({} 本は戻せず) {} ms mt={} threads={}",
            n,
            bytes >> 20,
            keys.len().saturating_sub(n),
            t.elapsed().as_millis(),
            u8::from(mt),
            crate::model::exp_pf::pf_restore_threads()
        );
    }
    (n, bytes)
}

/// task 394: その層の expert 重み全体 (gate / up / down) をページキャッシュへ
/// 先読みする (`DENNO_PF_LAYER_PRE`)。戻り値は投げたバイト数。
///
/// 範囲は [`ExpertW::slice_abs`] (mmap 経路の `slice` と同じ `ExpertSlice`) から
/// 作るので、あとで `pread` が読む範囲と必ず一致する。1 層あたり 3 本
/// (gate / up / down) の `readahead(2)` を投げるだけで、待たない。
/// VRAM もステージングも増やさず、中身は 1 バイトも変えない。
#[cfg(feature = "cuda")]
pub(crate) fn pf_layer_prefetch(lw: &LayerW) -> usize {
    let mut total = 0usize;
    let mut calls = 0usize;
    for w in [&lw.gate_exps, &lw.up_exps, &lw.down_exps] {
        let Some(last) = w.n_expert.checked_sub(1) else {
            continue;
        };
        let (Ok((off0, _)), Ok((off1, len1))) = (w.slice_abs(0), w.slice_abs(last)) else {
            continue;
        };
        let end = off1.saturating_add(len1 as u64);
        if end <= off0 {
            continue;
        }
        let n =
            crate::model::exp_pf::readahead_part(w.file as u32, off0, (end - off0) as usize);
        if n > 0 {
            total += n;
            calls += 1;
        }
    }
    if total > 0 {
        crate::model::exp_pf::note_layer_pre(total, calls);
    }
    total
}

/// task 470: 1 チャンク目の routing を**実際に解いて**、選ばれた expert の重みを
/// ページキャッシュへ先読みする (`DENNO_PF_EARLY_ROUTE`)。
///
/// # なぜ要るか
///
/// 層優先 prefill は「全チャンクの attention → 層で 1 回の MoE」の順なので、
/// routing が決まるまでディスクが遊んでいる (t470a の b1: 3,826 で 1 層 9.5 ms /
/// 14k で 46 ms、48 層で 0.46 s / 2.2 s)。1 チャンク目の attention が終わった
/// 時点でそのチャンクの MoE 入力はデバイスに確定しているので、そこで routing を
/// 解けば **残りのチャンクの attention が走っているあいだ**に expert の読みを
/// 始められる。
///
/// task 394 の `DENNO_PF_LAYER_PRE` は「層の expert は全部要る」前提で全数
/// (512 本) を投げ、実際は 296 本 (58%) しか使わないので 1.73 倍を読んで負けた。
/// ここは**そのチャンクの実際の routing** なので、集合は層全体の部分集合 =
/// 無駄読み 0。
///
/// # どこへ読むか (task 474)
///
/// 読んだバイトは**捨てずに** RAM の置き場 ([`crate::model::exp_pf::rc_plan`]) へ
/// 置く。層の MoE の写し (`pf_copy_into_pin`) は同じ (パート, 絶対オフセット,
/// 長さ) が置き場にあればディスクを読まずに memcpy する。task 470 の
/// `readahead(2)` / 「pread して捨てる」はページキャッシュに残らず、投げた
/// バイトの 11-28% しか `copy_s` が縮まなかった (lab t470b-d)。
///
/// # 何をしないか
///
/// - VRAM に載っている expert は飛ばす (MoE も上げ直さない)。
/// - RAM tier に載っている expert (`ram_peek`) は飛ばす (MoE は mmap から写す
///   ので置き場を使わない。lab t474c で 14.2 GB が全部無駄になった)。
/// - MoE がデバイス経路に乗らない層 (mask 付き / 形が合わない / stream の床未満)
///   は何もしない。
/// - **残りの窓が 4 チャンク未満の層は何もしない** ([`crate::model::exp_pf::early_route_window_ok`]。
///   3,826 の窓 9.5 ms に 42 MB の先読みが収まらず ±0 だった)。
/// - 値は 1 バイトも変えない。router の結果は先読みにしか使わず、MoE は従来どおり
///   全トークンで routing を解き直す。
///
/// 戻り値は (先読みした expert 本数, 投げたバイト数, 呼び出し回数)。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf_early_route_prefetch(
    lw: &LayerW,
    nt0: usize,
    nt_all: usize,
    e: usize,
    k: usize,
    ff: usize,
    ne: usize,
    iv: IvView<'_>,
) -> Option<(usize, usize, usize)> {
    use crate::model::expert_cache::ExpertKey;
    use crate::model::expert_vram::ExpertShape;
    if !crate::model::exp_pf::early_route_on() {
        return None;
    }
    // task 482: 窓が足りない層は、この先の routing (GPU の router GEMV + D2H) も
    // VRAM 参照も置き場の計画も**まるごと**要らない (この関数は最後に `None` を
    // 返して何もしない)。以前はその判定が後ろにあり、3,826 (2 チャンク) でも
    // 48 層ぶん払っていた (lab t482a: `sub other pf_er=72.6/90.1 ms`)。
    // `DENNO_PF_ER_EARLY_EXIT=0` で task 474 と同じ順序へ戻せる。
    //
    // task 486: 窓の判定を 2 本に分ける。
    // - `wide` = task 474 の規則 (残り 4 チャンク以上)。14k の経路は不変。
    // - `narrow` = 残り 1〜3 チャンク。`DENNO_PF_ER_SMALL` のときだけ、しかも
    //   ストリームの実測が「ディスクから読んでいる」層だけ通す
    //   (既定モデルのように page cache / RAM tier から来る層では、先読みは
    //   1 バイトも得をせず router の同期だけ増える)。
    // どちらも安い判定なので、ここ (router の前) で決める。
    let wide = crate::model::exp_pf::early_route_window_ok(nt0, nt_all);
    let (rate_bytes, rate_ns) = crate::model::exp_pf::stream_copy_rate();
    let narrow = crate::model::exp_pf::er_small_pass(
        wide,
        crate::model::exp_pf::er_small_on(),
        nt0,
        nt_all,
        rate_bytes,
        rate_ns,
    );
    if crate::model::exp_pf::er_early_exit_on() && !(wide || narrow) {
        return None;
    }
    // task 609: 窓が広くても、ストリームの実測が page cache 速度なら先読みしない
    // (`DENNO_PF_ER_PAGE_HIT`、既定 off)。理由は `exp_pf::ER_PAGE_HIT_ENV` の doc
    // (14k で 18.3 GB を page cache から 5 往復で運んでいた)。
    // ここは router の同期より前なので、止める層は router の GPU 仕事も払わない
    // (task 482 が窓の判定を前へ出したのと同じ理由)。
    if wide
        && crate::model::exp_pf::er_page_hit_on()
        && crate::model::exp_pf::er_stream_page_hit(rate_bytes, rate_ns)
    {
        crate::model::exp_pf::note_er_page_hit();
        return None;
    }
    // 窓が無い (1 チャンクで終わる) / 形が合わない層は何もしない。
    if nt0 == 0 || nt0 >= nt_all || k == 0 || ne == 0 || lw.gate_inp.k() != e {
        return None;
    }
    // mask 付きの層はデバイス router に乗らない (MoE 側も同じ条件で落ちる)。
    if iv.expert_mask(lw.il).is_some() {
        return None;
    }
    // task 474: `DENNO_PF_PREAD=0` では MoE の写しが mmap からの memcpy になり
    // (`src: None`)、置き場を 1 バイトも使わない。先読みしない。
    if !crate::model::exp_pf::pread_on() {
        return None;
    }
    let expert_gpu = if pf_lm_k_dev_on() {
        pf_expert_mode(lw)
    } else {
        iq_expert_mode(lw)
    };
    if !lm_dev_moe_ok(
        nt_all,
        crate::model::exp_pf::pf_enabled(),
        expert_gpu,
        crate::model::exp_pf::stream_enabled(),
        crate::model::exp_pf::stream_min_tok(),
        pf_router_gpu_on(),
        true,
    ) {
        return None;
    }
    // 1 チャンク目の MoE 入力 (attention のあとの残差) はデバイスにある。
    let xs0 = crate::cuda::pf_lm_mixed_ptr(0, nt0, e)?;
    if !crate::cuda::pf_router_dev_ok(nt0, ne, k) {
        return None;
    }
    let mut sel32 = vec![0u32; nt0 * k];
    let mut wts = vec![0.0f32; nt0 * k];
    let mut rowsum = vec![0.0f32; nt0];
    if !crate::cuda::pf_router_dev_topk(
        lw.gate_inp.uid(),
        lw.gate_inp.blocks().as_slice(),
        lw.gate_inp.k(),
        lw.gate_inp.n(),
        Some(xs0),
        None,
        nt0,
        k,
        &mut sel32,
        &mut wts,
        &mut rowsum,
        GgmlType::F32,
    ) {
        return None;
    }
    let mut seen = vec![false; ne];
    let mut nex_pick = 0usize;
    for &s in &sel32 {
        let ei = s as usize;
        if ei < ne && !seen[ei] {
            seen[ei] = true;
            nex_pick += 1;
        }
    }
    if nex_pick == 0 {
        return None;
    }
    let keys: Vec<ExpertKey> = (0..ne)
        .filter(|&ei| seen[ei])
        .map(|ei| ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        })
        .collect();
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // VRAM に載っている expert は飛ばす (読んでも MoE は上げ直さない)。
    // 参照だけ (LRU を塗り替えない)。失敗したら全部先読みする側へ倒す。
    let resident: Vec<bool> = match crate::model::expert_vram::peek_many(&keys, shapes) {
        Ok(v) => v.iter().map(|g| g.is_some()).collect(),
        Err(_) => vec![false; keys.len()],
    };
    // task 490: 窓の広い層は別の予算を使える (`DENNO_PF_ER_KB_WIDE`)。
    // 未設定 (0) なら従来どおり `DENNO_PF_EARLY_ROUTE_KB`。狭い窓 (`narrow`)
    // は常に従来の値 (3,826 の窓は 8.4 ms で、増やすと写しと帯域を奪い合う)。
    let kb = crate::model::exp_pf::er_route_kb(
        wide,
        crate::model::exp_pf::early_route_kb(),
        crate::model::exp_pf::er_kb_wide(),
    );
    let budget = crate::model::exp_pf::early_route_budget(
        crate::model::exp_pf::early_route_mb(),
        kb,
        nt_all - nt0,
    );
    // expert 昇順 = MoE が読む順。上限まで集めたら止める (ディスクの遊びは
    // 残りチャンクの attention の時間ぶんしかない)。
    // 1.1 窓が短すぎる層は何もしない。
    //
    // lab t474a の実測: 3,826 (1 チャンク = 1,913 トークンの窓) では 1 層 42 MB の
    // 先読みに 11.3 ms かかり、窓の 9.5 ms に収まらない。はみ出した分は写しと
    // 帯域を奪い合うだけで、壁は b 9.34/9.30 s 対 rc 8.96/10.04 s = **±0**
    // (ディスクの揺れの内側) だった。14k (残り 6 チャンク = 62 ms の窓) では
    // 1 層 287 MB が 51 ms で収まり、壁は b 15.20 / b2 13.88 s 対 rc 12.56 s。
    // したがって「残りの窓が 4 チャンク以上」のときだけ先読みする
    // (`nt0` = 1 チャンクのトークン数)。
    // task 474 の窓の下限 (残り 4 チャンク)。task 486 は狭い窓 (`narrow`) も
    // 同じ判断で通すので、ここでは「どちらでもない」ときだけ降りる。
    if !(wide || narrow) {
        return None;
    }
    let mut runs: Vec<(u32, u64, usize)> = Vec::new();
    let mut bytes = 0usize;
    let mut calls = 0usize;
    let mut nex = 0usize;
    for (i, key) in keys.iter().enumerate() {
        if resident[i] {
            continue;
        }
        let ei = key.expert as usize;
        let mut one = 0usize;
        let mut add: Vec<(u32, u64, usize)> = Vec::with_capacity(3);
        for w in [&lw.gate_exps, &lw.up_exps, &lw.down_exps] {
            let Ok((off, len)) = w.slice_abs(ei) else {
                continue;
            };
            one += len;
            add.push((w.file as u32, off, len));
        }
        if one == 0 {
            continue;
        }
        // task 474: RAM tier に載っている expert は MoE が mmap から写す
        // (`src: None` = `pread` を使わない) ので、先読みしても 1 バイトも
        // 使われない。t474c の既定モデル (IQ3_XXS) は全 expert が tier から
        // 来ていて、先読み 14.2 GB が全部無駄になった (`ra_pct=0`)。
        if ram_peek(lw, ei).is_some() {
            continue;
        }
        calls += add.len();
        runs.extend(add);
        bytes += one;
        nex += 1;
        if bytes >= budget {
            break;
        }
    }
    if nex == 0 {
        return None;
    }
    // task 474: 読んだバイトは**捨てずに** RAM の置き場へ置く。MoE の写し
    // (`pf_copy_into_pin`) は同じ (パート, 絶対オフセット, 長さ) が置き場に
    // あればディスクを読まずにそこから memcpy する (値はバイト単位で同じ)。
    //
    // task 470 は同じ窓を `readahead(2)` と「pread して捨てる」で埋めようとして
    // 負けた: 投げたバイトの 11-28% しか `copy_s` が縮まなかった (ページキャッシュ
    // に残らない)。置き場をプロセスが持てばその取りこぼしが無い。
    let Some((buf, jobs)) = crate::model::exp_pf::rc_plan(&runs) else {
        return None;
    };
    let h = std::thread::Builder::new()
        .name("pf-er-ram".to_string())
        .spawn(move || {
            use rayon::prelude::*;
            let t = std::time::Instant::now();
            let nt = crate::model::exp_pf::rc_threads();
            let run = || {
                jobs.par_iter().with_max_len(1).for_each(|&(dst, file, off, len)| {
                    // 窓が終わったら (MoE が始まったら) そこで止める。
                    if crate::model::exp_pf::rc_stopped() {
                        return;
                    }
                    // SAFETY: `dst` は `rc_layout` が割り当てた互いに重ならない
                    // 範囲で、置き場はプロセスの終わりまで有効。
                    let p = unsafe { buf.ptr().add(dst as usize) };
                    if unsafe {
                        crate::model::exp_pf::read_part_ptr(file, off, p, len as usize)
                    } {
                        crate::model::exp_pf::rc_commit_one(file, off, len as usize, dst);
                    }
                });
            };
            match crate::model::exp_pf::rc_pool(nt) {
                Some(pool) => pool.install(run),
                None => run(),
            }
            crate::model::exp_pf::note_rc_read_ns(t.elapsed().as_nanos() as u64);
        });
    match h {
        Ok(h) => crate::model::exp_pf::rc_worker_set(h),
        Err(_) => return None,
    }
    crate::model::exp_pf::note_early_route(nex, bytes, calls);
    Some((nex, bytes, calls))
}

/// task 394: 戻しの 1 本ぶん。`deferred` なら転送を LRU のロックの外で行う
/// ([`crate::model::expert_vram::acquire_deferred`])。
///
/// 戻り値は載せたバイト数 (`None` = この 1 本は載せられなかった)。VRAM へ
/// 渡るバイト列は 1 本ずつの経路と同じ (ここは行列の切り出しと形の判定だけを
/// 行い、転送は [`crate::model::expert_vram`] の同じ関数に任せる)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_restore_one(
    layers: &[LayerW],
    key: crate::model::expert_cache::ExpertKey,
    deferred: bool,
) -> Option<usize> {
    use crate::model::expert_vram::{self, ExpertShape};
    let lw = layers.get(key.layer as usize)?;
    let ei = key.expert as usize;
    let (g, u, d) = (&lw.gate_exps, &lw.up_exps, &lw.down_exps);
    let shapes = [
        ExpertShape {
            ty: g.ty,
            k: g.n_in,
            n: g.n_out,
        },
        ExpertShape {
            ty: u.ty,
            k: u.n_in,
            n: u.n_out,
        },
        ExpertShape {
            ty: d.ty,
            k: d.n_in,
            n: d.n_out,
        },
    ];
    let want = shapes.iter().try_fold(0usize, |a, s| s.bytes().map(|b| a + b)).ok()?;
    let load = || Ok([g.slice(ei)?, u.slice(ei)?, d.slice(ei)?]);
    let got = if deferred {
        expert_vram::acquire_deferred(key, shapes, load)
    } else {
        expert_vram::acquire_soft(key, shapes, load)
    };
    match got {
        Ok(Some(_)) => Some(want),
        _ => None,
    }
}

/// task 394: 戻しを 1 本ずつ直列に行う (従来の経路)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_restore_experts_seq(
    layers: &[LayerW],
    keys: &[crate::model::expert_cache::ExpertKey],
    nosync: bool,
) -> (usize, usize) {
    let mut n = 0usize;
    let mut bytes = 0usize;
    for key in keys {
        if let Some(want) = pf_restore_one(layers, *key, false) {
            n += 1;
            bytes += want;
        }
    }
    if nosync {
        // 直列でもリングは遅延で積むので、ここで 1 回待つ。
        crate::cuda::restore_defer_flush_thread();
    }
    (n, bytes)
}

/// task 394: 戻しを N 本のスレッドで行う (`DENNO_PF_RESTORE_MT`)。
///
/// 1 本の戻しは「gate/up/down の 3 回の同期 H2D + スロット確保」で、待ちが
/// 直列に並ぶ (lab t390b の 14k: 1,057 本で約 1.27 s = 1 本 1.2 ms)。ここでは
/// [`crate::model::expert_vram::acquire_deferred`] (転送を LRU のロックの外で
/// 行う版) を専用プールの N 本で回し、同期の待ちを重ねる。
///
/// **CUDA の current context はスレッドローカル**なので、プールの各スレッドで
/// 1 回 [`crate::cuda::make_current_on_this_thread`] を呼ぶ (呼ばないと最初の
/// 確保が error 201 で落ち、載せられないまま静かに終わる)。H2D も専用
/// ストリームへ出す ([`crate::cuda::enable_thread_upload_stream`]) ので、
/// メインスレッドのカーネルと直列にならない。デバイスへ渡るバイト列は
/// 1 本ずつの経路と同一で、変わるのは順序と重なりだけ。
///
/// プールを作れない / context を張れない / `par` 無しのビルドでは従来経路へ
/// 落ちる (fail closed)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_restore_experts_mt(
    layers: &[LayerW],
    keys: &[crate::model::expert_cache::ExpertKey],
    nosync: bool,
) -> (usize, usize) {
    #[cfg(feature = "par")]
    {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        static READY: AtomicBool = AtomicBool::new(false);
        let nt = crate::model::exp_pf::pf_restore_threads()
            .min(keys.len())
            .max(1);
        if let Some(pool) = pf_restore_pool(nt) {
            if !READY.load(Ordering::Relaxed) {
                // プールの全スレッドで 1 回ずつ context と H2D ストリームを張る。
                let ok = pool.broadcast(|_| {
                    crate::cuda::make_current_on_this_thread()
                        && crate::cuda::enable_thread_upload_stream()
                });
                if !ok.iter().any(|b| *b) {
                    return pf_restore_experts_seq(layers, keys, nosync);
                }
                READY.store(true, Ordering::Relaxed);
            }
            use rayon::prelude::*;
            let n = AtomicUsize::new(0);
            let bytes = AtomicUsize::new(0);
            // task 565: 遅延モードでは、各スレッドが積んだリングの DMA を
            // プール全スレッドで 1 回ずつ待つ (待ちは戻しの終端で 1 回)。
            if nosync {
                pool.broadcast(|_| crate::cuda::restore_defer_begin());
            }
            pool.install(|| {
                keys.par_iter().for_each(|k| {
                    if let Some(want) = pf_restore_one(layers, *k, true) {
                        n.fetch_add(1, Ordering::Relaxed);
                        bytes.fetch_add(want, Ordering::Relaxed);
                    }
                });
            });
            if nosync {
                pool.broadcast(|_| crate::cuda::restore_defer_flush_thread());
                crate::cuda::restore_defer_flush_thread();
            }
            return (n.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed));
        }
    }
    pf_restore_experts_seq(layers, keys, nosync)
}

/// task 394: 戻し専用の rayon プール (スレッド数ごとに 1 個だけ作る)。
///
/// スレッド局所の pinned バウンスと H2D ストリームをそのまま使い回せるので、
/// 2 回目以降の戻しはプールの作成も初期化も要らない。
#[cfg(all(feature = "cuda", feature = "par"))]
pub(crate) fn pf_restore_pool(nt: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    static NT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let first = NT.swap(nt, std::sync::atomic::Ordering::Relaxed) == 0;
    if !first {
        // 2 本目以降の違う本数は無視する (プロセスで 1 個)。
        return POOL.get().and_then(|p| p.as_ref());
    }
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .thread_name(|i| format!("pf-restore-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// task 233: 層優先 prefill の MoE (活性はデバイスに置いたまま)。
///
/// 入力 (`pf_mix` の ffn 出口の出力) と出力 (この層の MoE の結果) は、どちらも
/// [`crate::cuda::pf_lm_ensure`] が用意したデバイスの置き場にある。ホストへ
/// 戻るのは router の確率 (`nt x n_expert`、softmax / top-k のため) だけ。
/// 共有 expert の出力は MoE の作業域の末尾の行へ直接書き、行の加重和
/// ([`crate::cuda::moe_experts_iq_pf_layer_stream_dev`]) の**最後の 1 項**にする。
///
/// `Ok(false)` = この経路を使わなかった (置き場は触っていない)。呼び出し側は
/// ホスト版 [`moe_ffn_quant_batch_tapped`] へ落ちる。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_quant_batch_lm_dev(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    nt: usize,
    roles: Option<&mut RoleRms>,
    emit_router: bool,
    iv: IvView<'_>,
) -> Result<bool> {
    moe_ffn_quant_batch_lm_dev_at(p, lw, 0, nt, nt, roles, emit_router, iv)
}

/// task 250: 層優先 MoE の**トークンタイル版**。`tok0` 行目から `nt` 行だけを
/// 受ける (`nt_all` は置き場の全行数)。`emit_last` は層の末尾トークンの
/// router 記録 / 役割 RMS を出すか (最後のタイルだけ true)。
///
/// `h` などの作業域はタイルの行数で取るので、14k でも 1.5 GB の連続確保が要らない。
#[allow(clippy::too_many_arguments)]
/// task 257: dev MoE を諦めた理由 (診断。層ループの外で 1 行だけ出す)。
#[cfg(feature = "cuda")]
thread_local! {
    pub(crate) static LM_DECLINE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// task 257: 理由を控えて `false` を返す (呼び出し側が 1 行だけ出す)。
#[cfg(feature = "cuda")]
pub(crate) fn lm_decline(reason: &'static str) -> bool {
    LM_DECLINE.with(|c| c.set(reason));
    false
}

/// task 257: 直前の dev MoE が諦めた理由 (空 = 無い)。
#[cfg(feature = "cuda")]
pub(crate) fn lm_decline_reason() -> &'static str {
    LM_DECLINE.with(|c| c.get())
}

/// task 333: `理由: stream` のときの内訳 (ストリーミングが落ちた門)。
///
/// task 328 は `理由: stream` だけで、ステージングの確保なのか上げスレッドなのかが
/// 分からなかった。空 = 内訳なし (stream 以外の理由)。
#[cfg(feature = "cuda")]
pub(crate) fn lm_decline_detail() -> String {
    let why = crate::cuda::pf_stream_why();
    if lm_decline_reason() == "stream" && !why.is_empty() {
        format!(" / stream: {why}")
    } else {
        String::new()
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_quant_batch_lm_dev_at(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    tok0: usize,
    nt_all: usize,
    nt: usize,
    mut roles: Option<&mut RoleRms>,
    emit_last: bool,
    iv: IvView<'_>,
) -> Result<bool> {
    use crate::model::expert_cache::ExpertKey;
    use crate::model::expert_vram::ExpertShape;

    let e = p.n_embd;
    let ne = p.n_expert;
    let k = p.n_expert_used;
    let ff = p.n_ff_exp;
    // 診断 (`DENNO_PF_LM_DBG=1`): どの段が時間を使ったかを層ごとに 1 行。
    let dbg = std::env::var_os("DENNO_PF_LM_DBG").is_some();
    let t_dbg = std::time::Instant::now();
    let mut ph = [0f64; 6];
    let mut mark = |i: usize| {
        if dbg {
            ph[i] = t_dbg.elapsed().as_secs_f64() * 1e3;
        }
    };
    // task 318: 層優先のデバイス MoE は IQ 系だけでなく **K 系 (UD-Q4_K_XL) でも
    // 回せる**。ここが `iq_expert_mode` のままだと UD は全 48 層が
    // `moe_ffn_quant_batch_tapped` (ホストの mixed 経路) へ落ち、層ごとに
    // mixed を丸ごと往復していた (lab t318a: 14k で 143.5 MB x 2 x 48 層 =
    // 13.8 GB、`[xfer] item=other sub=-`)。expert の m 行カーネルは
    // ストリーミング経路のものなので、`pf_expert_mode` (IQ or K) が正しい判定。
    let expert_gpu = if pf_lm_k_dev_on() {
        pf_expert_mode(lw)
    } else {
        iq_expert_mode(lw)
    };
    if !lm_dev_moe_ok(
        nt,
        crate::model::exp_pf::pf_enabled(),
        expert_gpu,
        crate::model::exp_pf::stream_enabled(),
        crate::model::exp_pf::stream_min_tok(),
        pf_router_gpu_on(),
        lw.gate_inp.k() == e,
    ) {
        return Ok(lm_decline("gate"));
    }
    // task 318: `start = tok0` から **このタイルの `nt` 行** ぶんの先頭を取る。
    // ここが `nt_all` だと 2 枚目以降のタイルで「置き場の末尾を越える」ので
    // `None` になり、14k のようなタイル分割時に dev MoE が丸ごと落ちていた
    // (lab t318b: 14k が `理由: mixed ptr` で全層フォールバックし、ストリームが
    //  98 GB → 147 GB に増えた)。
    let Some(xs_dev) = crate::cuda::pf_lm_mixed_ptr(tok0, nt, e) else {
        return Ok(lm_decline("mixed ptr"));
    };
    let Some(out_base) = crate::cuda::pf_lm_ffn_ptr(nt_all, e) else {
        return Ok(lm_decline("ffn ptr"));
    };
    // 出力もこのタイルの先頭へずらす (`out` は nt_all 行の置き場)。
    let out_dev = out_base + (tok0 * e * 4) as u64;
    // routed expert を丸ごと切った層はホスト経路へ (共有だけを書く形は作らない)。
    let skip_routed = iv.block_off(lw.il, IvBlock::Moe);
    let skip_shared = iv.block_off(lw.il, IvBlock::Shared);
    if skip_routed {
        return Ok(lm_decline("skip_routed"));
    }
    // 1. ルータ (入力をデバイスから直接読む)。
    let mut sels: Vec<usize> = vec![0; nt * k];
    let mut weights: Vec<f32> = vec![0.0; nt * k];
    let expert_mask = iv.expert_mask(lw.il);
    // task 308: このタイル全体の router (logits → softmax → top-k → 再正規化) を
    // デバイスで 1 回にまとめる (`DENNO_PF_LM_ROUTER_DEV`)。降りるのは選んだ
    // k 個の index と重み、行ごとの選択和だけ (probs の `[nt][n_expert]` を
    // ホストへ戻さない)。カーネルは task 225 のもの (`kernels_router225`)。
    #[cfg(feature = "cuda")]
    let mut router_rowsum: Option<f32> = None;
    #[cfg(feature = "cuda")]
    {
        if pf_lm_router_dev_on()
            && expert_mask.is_none()
            && nt >= PF_ROUTER_GPU_MIN_TOK
            && crate::cuda::pf_router_dev_ok(nt, ne, k)
        {
            let mut sel32 = vec![0u32; nt * k];
            let mut rowsum = vec![0.0f32; nt];
            if crate::cuda::pf_router_dev_topk(
                lw.gate_inp.uid(),
                lw.gate_inp.blocks().as_slice(),
                lw.gate_inp.k(),
                lw.gate_inp.n(),
                Some(xs_dev),
                None,
                nt,
                k,
                &mut sel32,
                &mut weights,
                &mut rowsum,
                GgmlType::F32,
            ) {
                for (i, v) in sel32.iter().enumerate() {
                    sels[i] = *v as usize;
                }
                router_rowsum = rowsum.last().copied();
                mark(0);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "[pf308] 層優先 router: logits + softmax + top-k をデバイスで 1 回 \
                         (DENNO_PF_LM_ROUTER_DEV)"
                    )
                });
            }
        }
    }
    #[cfg(feature = "cuda")]
    let router_dev_done = router_rowsum.is_some();
    #[cfg(not(feature = "cuda"))]
    let router_dev_done = false;
    // ホスト経路の probs (デバイス経路では空のまま)。
    let mut probs: Vec<f32> = Vec::new();
    if !router_dev_done {
        probs = vec![0.0f32; nt * ne];
        if !crate::cuda::maybe_gpu_f32_gemm_batch_dev_into(
            lw.gate_inp.uid(),
            lw.gate_inp.blocks().as_slice(),
            lw.gate_inp.k(),
            lw.gate_inp.n(),
            xs_dev,
            nt,
            &mut probs,
            GgmlType::F32,
        ) {
            return Ok(lm_decline("router gemm"));
        }
        mark(0);
        let sm_block = router_sm_block();
        router_softmax_rows(&mut probs, ne, sm_block);
        router_topk_rows(
            &mut probs,
            ne,
            k,
            &mut sels,
            &mut weights,
            sm_block,
            pf_router_fast_on(),
            expert_mask,
        );
    }
    mark(1);
    if emit_last {
        let tail = (nt - 1) * k;
        emit_router_topk(lw.il, &sels[tail..tail + k], &weights[tail..tail + k]);
        if let Some(m) = expert_mask {
            let row = &probs[(nt - 1) * ne..nt * ne];
            emit_expert_skip(lw.il, row, k, &sels[tail..tail + k], m);
        }
    }
    // 2. 行のある expert と行表 (共有 expert の行を h の末尾に足すので先に組む)。
    let mut counts = vec![0usize; ne];
    for &ei in &sels {
        counts[ei] += 1;
    }
    let mut by_expert: Vec<Vec<(usize, usize)>> = counts
        .iter()
        .map(|&c| Vec::with_capacity(c))
        .collect();
    for t in 0..nt {
        for j in 0..k {
            by_expert[sels[t * k + j]].push((t, j));
        }
    }
    let pick: Vec<bool> = by_expert.iter().map(|p| !p.is_empty()).collect();
    let plan = crate::model::exp_pf::build_plan(&by_expert, &pick);
    if plan.total_rows == 0 {
        return Ok(lm_decline("no rows"));
    }
    let total_rows = plan.total_rows;
    mark(2);
    // 3. shared expert (入力も出力もデバイス)。出力は加重和の最後の項になる。
    let mut sh_dev = 0u64;
    let mut sh_w: Vec<f32> = Vec::new();
    if !skip_shared {
        let sff = lw.gate_shexp.n();
        let gn = lw.gate_inp_shexp.n();
        if sff == 0 || gn == 0 {
            return Ok(lm_decline("shared shape"));
        }
        let mut sg = vec![0.0f32; nt * gn];
        if !crate::cuda::maybe_gpu_f32_gemm_batch_dev_into(
            lw.gate_inp_shexp.uid(),
            lw.gate_inp_shexp.blocks().as_slice(),
            lw.gate_inp_shexp.k(),
            gn,
            xs_dev,
            nt,
            &mut sg,
            GgmlType::F32,
        ) {
            return Ok(lm_decline("shared router"));
        }
        sh_w = (0..nt).map(|t| hyper::sigmoid(sg[t * gn])).collect();
        let Some(tail) = crate::cuda::pf_moe_shared_tail(total_rows, nt, e) else {
            return Ok(lm_decline("shared tail"));
        };
        if !crate::cuda::pf_shared_dev_to(
            xs_dev,
            nt,
            e,
            sff,
            Qwen4Exp::pf_lin(&lw.gate_shexp),
            Qwen4Exp::pf_lin(&lw.up_shexp),
            Qwen4Exp::pf_lin(&lw.down_shexp),
            tail,
        ) {
            return Ok(lm_decline("shared dev"));
        }
        sh_dev = tail;
    }
    mark(3);
    // 4. routed experts (入力も出力もデバイス)。
    {
        // 同じ expert は 1 層で何度も選ばれる。madvise は 1 expert 1 回で足りる。
        let mut seen = vec![false; ne];
        for &ei in &sels {
            if !seen[ei] {
                seen[ei] = true;
                lw.gate_exps.prefetch(ei);
                lw.up_exps.prefetch(ei);
                lw.down_exps.prefetch(ei);
            }
        }
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: e,
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    let mut keys: Vec<ExpertKey> = Vec::with_capacity(ne);
    let mut ids: Vec<usize> = Vec::with_capacity(ne);
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        keys.push(ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        });
        ids.push(ei);
    }
    // **触らない**参照 (`get_mut` で LRU を塗り替えると decode のヒット率が落ちる)。
    let got = crate::model::expert_vram::peek_many(&keys, shapes)
        .map_err(|err| one(format!("層 {}: expert VRAM 参照: {err:#}", lw.il)))?;
    let mut resident: Vec<Option<std::sync::Arc<crate::model::expert_vram::ExpertDev>>> =
        vec![None; ne];
    let mut hits = 0u32;
    for (i, g) in got.iter().enumerate() {
        if let Some(dv) = g {
            resident[ids[i]] = Some(std::sync::Arc::clone(dv));
            hits += 1;
        }
    }
    crate::model::exp_pf::note_lookup(keys.len() as u32, hits);
    let sh_w_ref: &[f32] = if sh_dev != 0 { sh_w.as_slice() } else { &[] };
    match iq_pf_stream_layer(
        lw,
        &by_expert,
        &resident,
        &weights,
        k,
        PfStreamTgt::Dev {
            xs: xs_dev,
            out: out_dev,
            sh: sh_dev,
            sh_w: sh_w_ref,
        },
        nt,
        e,
        ff,
    )? {
        Some(_) => {}
        None => return Ok(lm_decline("stream")),
    }
    mark(4);
    // 5. 役割 RMS (チャンク末尾のトークン)。
    if let Some(r) = roles.as_deref_mut() {
        let t = nt - 1;
        let mut s = 0.0f32;
        // task 308: デバイス経路は選択和をカーネルから受け取る
        // (`rowsum` = 選んだ k 個の index 昇順の逐次和。ホストの下の和と同じ)。
        #[cfg(feature = "cuda")]
        let dev_sum = router_rowsum;
        #[cfg(not(feature = "cuda"))]
        let dev_sum: Option<f32> = None;
        match dev_sum {
            Some(v) => s = v,
            None => {
                for j in 0..k {
                    s += probs[t * ne + sels[t * k + j]];
                }
            }
        }
        r.router = s;
        if sh_dev != 0 {
            let mut row = vec![0.0f32; e];
            if crate::cuda::pf_moe_h_row(total_rows + t, e, &mut row) {
                r.shex = rms_of(&row);
            }
        }
    }
    mark(5);
    if dbg {
        let st = crate::cuda::pf_stream_times_ms();
        let ex = crate::cuda::pf_stream_extra_ms();
        let sum: f64 = st.iter().sum::<f64>() + ex[0] + ex[1];
        // task 296: `prep` (スロット準備) と `pin` (pinned の確保し直し) を足した。
        // `resid` = experts 段のうち上のどれでもない時間 (層の外の仕事)。
        eprintln!(
            "[pf-lm-dbg] stream pump={:.1} wait={:.1} table={:.1} launch={:.1} accum={:.1} \
             prep={:.1} pin={:.1}ms x{} sum={:.1} resid={:.1} ms",
            st[0], st[1], st[2], st[3], st[4], ex[0], ex[1], ex[2] as u64, sum,
            ((ph[4] - ph[3]) - sum).max(0.0)
        );
        eprintln!(
            "[pf-lm-dbg] layer={} nt={} router={:.1} topk={:.1} plan={:.1} shared={:.1} \
             experts={:.1} roles={:.1} total={:.1} ms",
            lw.il,
            nt,
            ph[0],
            ph[1] - ph[0],
            ph[2] - ph[1],
            ph[3] - ph[2],
            ph[4] - ph[3],
            ph[5] - ph[4],
            t_dbg.elapsed().as_secs_f64() * 1e3
        );
    }
    // 分岐が実際に発火した証拠を lab のログに 1 回だけ出す。
    {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[pf-lm-dev] 層優先: 活性はデバイスに置いたまま (router / shared / expert / combine)"
            );
        }
    }
    Ok(true)
}

/// task 233: 層優先 prefill のデバイス MoE を使う前提 (純関数・テスト用)。
///
/// - router はデバイス入力を読むので `DENNO_PF_ROUTER_GPU` が要る。
/// - expert はストリーミング経路 (ステージング) 前提 (`DENNO_PF_STREAM`)。
/// - `gate_inp` の入力幅が `n_embd` と一致すること (デバイスの mixed を読むため)。
///
/// どれか欠けたら `false` = 呼び出し側はホスト経路へ落ちる (値は同じ)。
pub(crate) fn lm_dev_moe_ok(
    nt: usize,
    pf: bool,
    iq_expert: bool,
    stream: bool,
    stream_min_tok: usize,
    router_gpu: bool,
    gate_in_k_is_e: bool,
) -> bool {
    nt > 0
        && pf
        && iq_expert
        && stream
        && nt >= stream_min_tok
        && router_gpu
        && gate_in_k_is_e
}

/// task 420: **全 MoE 層**が tensor core の grouped GEMM に乗るか (純関数)。
///
/// 実行時の層ごとの選択は 2 段で、IQ の形が合えば IQ 版、合わなければ K 版。
/// IQ の形が合うのに PTX が無い層は **K へ落ちない** (`tc_kernels = None`) ので、
/// 「全層 IQ」と「全層 K (IQ の形はどこも合わない)」を別々に評価する。
///
/// `per_layer` は層ごとの `(IQ の形が合う, K の形が合う)`、`iq` / `k` は
/// `(ゲートが on, PTX が読めた)`。1 層でも乗らない層があれば `false` =
/// 呼び出し側は g / u の作業域を取る (安全側。取り過ぎても遅いだけで壊れない)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pf_pool_tc_all(
    per_layer: &[(bool, bool)],
    iq: (bool, bool),
    k: (bool, bool),
) -> bool {
    if per_layer.is_empty() {
        return false;
    }
    let iq_all = iq.0 && iq.1 && per_layer.iter().all(|&(i, _)| i);
    let k_all = k.0 && k.1 && per_layer.iter().all(|&(i, kk)| !i && kk);
    iq_all || k_all
}

#[cfg(test)]
mod lm_dev_tests {
    use super::*;

    /// task 420: 全層 TC の真理値表。
    ///
    /// K だけのモデル (UD-Q4_K_XL) は K で乗る。IQ と K が混在したら、実行時に
    /// 層ごとに別のカーネルになるので g / u を取る (安全側)。
    #[test]
    fn pf_pool_tc_all_truth_table() {
        let k_model = [(false, true), (false, true)];
        let iq_model = [(true, true), (true, true)];
        let mixed = [(false, true), (true, true)];
        let iq = (true, true);
        let k = (true, true);
        assert!(pf_pool_tc_all(&k_model, iq, k), "K だけのモデルは K で乗る");
        assert!(pf_pool_tc_all(&iq_model, iq, k), "IQ のモデルは IQ で乗る");
        assert!(
            !pf_pool_tc_all(&mixed, iq, k),
            "IQ と K が混在したら g / u を取る"
        );
        assert!(!pf_pool_tc_all(&k_model, iq, (true, false)), "K の PTX が無い");
        assert!(!pf_pool_tc_all(&k_model, iq, (false, true)), "K のゲートが off");
        assert!(
            !pf_pool_tc_all(&iq_model, (true, false), k),
            "IQ の形で PTX が無い層は K へ落ちない (実行時と同じ)"
        );
        assert!(
            !pf_pool_tc_all(&iq_model, (false, true), k),
            "IQ のゲートが off なら IQ の形の層は乗らない"
        );
        assert!(!pf_pool_tc_all(&[], iq, k), "層が無い");
        assert!(
            pf_pool_tc_all(&[(true, false)], iq, k),
            "IQ の形だけ合っていても、IQ が使えれば TC に乗る"
        );
    }

    /// task 233: 層優先のデバイス MoE の前提 (真理値表)。
    ///
    /// 1 つでも欠けたら `false` = 呼び出し側はホスト経路へ落ちる (値は同じ)。
    /// task 318: `DENNO_PF_LM_K_DEV` の真理値表 (t318c で採用、既定 on)。
    #[test]
    fn pf_lm_k_dev_truth_table() {
        assert!(pf_lm_k_dev_from_value(None));
        assert!(pf_lm_k_dev_from_value(Some("")));
        assert!(!pf_lm_k_dev_from_value(Some("0")));
        assert!(!pf_lm_k_dev_from_value(Some(" off ")));
        assert!(!pf_lm_k_dev_from_value(Some("FALSE")));
        assert!(!pf_lm_k_dev_from_value(Some("no")));
        assert!(pf_lm_k_dev_from_value(Some("1")));
        assert!(pf_lm_k_dev_from_value(Some("on")));
        assert!(pf_lm_k_dev_from_value(Some("true")));
        assert!(pf_lm_k_dev_from_value(Some("yes")));
    }

    #[test]
    fn lm_dev_moe_gate_truth_table() {
        assert!(lm_dev_moe_ok(4096, true, true, true, 512, true, true));
        assert!(!lm_dev_moe_ok(0, true, true, true, 512, true, true), "nt=0");
        assert!(
            !lm_dev_moe_ok(4096, false, true, true, 512, true, true),
            "pf off"
        );
        assert!(
            !lm_dev_moe_ok(4096, true, false, true, 512, true, true),
            "iq expert off"
        );
        assert!(
            !lm_dev_moe_ok(4096, true, true, false, 512, true, true),
            "stream off"
        );
        assert!(
            !lm_dev_moe_ok(511, true, true, true, 512, true, true),
            "nt < min"
        );
        assert!(
            !lm_dev_moe_ok(4096, true, true, true, 512, false, true),
            "router が CPU"
        );
        assert!(
            !lm_dev_moe_ok(4096, true, true, true, 512, true, false),
            "gate_inp の幅が違う"
        );
        // 境界 (nt == min) は通す。
        assert!(lm_dev_moe_ok(512, true, true, true, 512, true, true));
    }

    /// task 309: K 系の prefill expert を GPU に回す前提 (真理値表)。
    ///
    /// ゲート・ストリーミング・型の 3 つが揃ったときだけ真。
    #[test]
    fn pf_expert_mode_k_truth_table() {
        assert!(pf_expert_mode_from(true, true, true));
        assert!(!pf_expert_mode_from(false, true, true), "ゲート off");
        assert!(!pf_expert_mode_from(true, false, true), "ストリーミング off");
        assert!(
            !pf_expert_mode_from(true, true, false),
            "型が K 系のカーネルに無い"
        );
        // task 309: lab t309b で採用して既定 on。
        assert!(
            crate::model::exp_pf::DEFAULT_K_ON,
            "既定 off に戻っている"
        );
    }
}
impl Qwen4Exp {
    /// プロンプト全体を 1 回で処理して**最後のトークンの logits** を返す。
    ///
    /// `DENNO_PREFILL_BATCH=1` のときだけ層優先のバッチ経路を使う (既定 off)。
    /// off のとき、または attention が層優先の順序に耐えない実装のときは、
    /// decoder と同じ「1 トークンずつ [`Qwen4Exp::forward_token`]」に落ちる。
    /// どちらの経路も出力はビット一致する (バッチ経路は行列積を
    /// [`gemv_batch`] = 行優先・同一カーネルに置き換えるだけ)。
    pub fn forward_prefill(&mut self, ctx: &[u32]) -> Result<Vec<f32>> {
        self.forward_prefill_with(ctx, prefill_batch_enabled())
    }

    /// [`Qwen4Exp::forward_prefill`] の切替を引数で受ける版 (テスト用)。
    ///
    /// `batch == false` は既存の逐次経路そのもの (1 ビットも変えない)。
    pub fn forward_prefill_with(&mut self, ctx: &[u32], batch: bool) -> Result<Vec<f32>> {
        self.forward_prefill_inner(ctx, batch, TapTokens::Last, None, None)
    }

    /// P53: 層の出口観測フック付きの prefill。
    ///
    /// `tokens` で観測するトークンを選ぶ。`TapTokens::Last` は**チャンクの
    /// 最後のトークンだけ**を流す (層ループがチャンクを跨いでも 1 チャンク 1 回)。
    ///
    /// フックは層の出口 (ffn の `hc_combine` の後) で呼ばれる。読むだけなので
    /// logits は `forward_prefill_with` とビット一致する。
    pub fn forward_prefill_tapped<F>(
        &mut self,
        ctx: &[u32],
        batch: bool,
        tokens: TapTokens,
        on_layer: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&LayerOut<'_>),
    {
        self.forward_prefill_tapped_roles(ctx, batch, tokens, on_layer, None)
    }

    /// P54: 役割別 RMS も取る [`Qwen4Exp::forward_prefill_tapped`]。
    ///
    /// `roles` が `Some` のときだけ層の中で RMS を取る。観測するトークンは
    /// `tokens` に従う (`TapTokens::Last` ならチャンク末尾だけ)。
    pub fn forward_prefill_tapped_roles<F>(
        &mut self,
        ctx: &[u32],
        batch: bool,
        tokens: TapTokens,
        mut on_layer: F,
        roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&LayerOut<'_>),
    {
        let mut hook = |o: &LayerOut<'_>| -> Result<()> {
            on_layer(o);
            Ok(())
        };
        self.forward_prefill_inner(ctx, batch, tokens, Some(&mut hook), roles)
    }

    pub(crate) fn forward_prefill_inner(
        &mut self,
        ctx: &[u32],
        batch: bool,
        tokens: TapTokens,
        on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>> {
        // P78: 層の外の区間計測の分母 (prefill 回数とトークン数)。
        prekprof::note_prefill(ctx.len());
        // task 145: prefill の台帳。全体の壁時計を 1 本で測り、層の中の区間
        // (mix / attn / router / shared / experts) は内側のガードで積む。
        // `other` は「全体 − 名前付き区間の和」の残差として `line_prefill` が出す。
        let _pf_total = q4ledger::total();
        // task 162: この prefill が終わったらステージング領域を VRAM へ返す
        // (エラーで抜けても Drop で戻る)。decode の間は抱えない。
        #[cfg(feature = "cuda")]
        let _pf_stream_guard = PfStreamGuard;
        // task 270: チャンクを上げると prefill の作業域 (mix の res / scratch +
        // MoE の行バッファ + staging) がそのぶん増える。`DENNO_PREFILL_WORK_MB`
        // が指定されていれば、足りない分を expert の LRU から**チャンク単位で
        // 先に**空けておく。途中の確保失敗で staging が取れず、streaming が
        // 黙って落ちる (14k が 25 → 39 s になった t270a) のを防ぐ。drop で release。
        #[cfg(feature = "cuda")]
        let _pf_work = crate::model::expert_vram::reserve_work_area(
            crate::model::expert_vram::prefill_work_bytes(),
        );
        // task 282: この prefill の間は mix の fused chain を使わない (decode 専用)。
        let _pf_seq_guard = enter_prefill_seq();
        let layer_major = batch && self.attn.layer_major_safe();
        let logits = if layer_major {
            self.forward_prefill_batch_tapped(ctx, prefill_chunk_for(ctx.len()), tokens, on_layer, roles)?
        } else {
            self.prefill_seq(ctx, tokens, on_layer, roles)?
        };
        // task 154: prefill の GPU スキャンが持つ層状態のバッファ (約 385 MB) を
        // VRAM へ返す。ホスト状態は毎回書き戻してあるので、次はホストから seed する。
        #[cfg(feature = "cuda")]
        if pf_gdn_gpu_on() {
            crate::cuda::gdn_state_release();
        }
        // task 246: prefill のストリーミング経路は LRU を**読むだけ** (peek) なので、
        // 末尾チャンクの選択で LRU を触り直して decode の作業集合を先頭へ寄せる
        // (H2D なし)。層優先バッチのときだけ (逐次 prefill は自分で `acquire` する)。
        #[cfg(feature = "cuda")]
        if layer_major {
            self.pf_reseed_decode_hot();
        }
        // P58: 1 リクエスト 1 回の階層統計 (prefill 終了時)。arch off なら即座に戻る。
        self.arch_emit_prefill_done();
        Ok(logits)
    }

    /// task 246: prefill の終端で、末尾トークンが選んだ expert を LRU の先頭へ
    /// 寄せ直す (H2D なし)。
    ///
    /// prefill のストリーミング経路 ([`crate::model::exp_pf::stream_enabled`]) は
    /// VRAM 常駐を **peek するだけ** で LRU を触らない (lab t148c: 触ると 14k の
    /// decode ヒット率が 0.966 → 0.866)。その副作用として、チャンクを均等割りに
    /// すると「末尾の小さいチャンクが従来経路に落ちて LRU を温める」効果まで
    /// 消え、decode のヒット率が 1.7 pp 落ちる (REPORT_P239 の bal)。
    ///
    /// ここで控えておいた選択 (`exp_pf::note_tail_selection`) を、従来経路と
    /// 同じ順序 (トークン昇順 → 層昇順 → j 昇順) で [`expert_vram::touch_many_count`]
    /// に渡す。載っていない鍵は飛ばすので **転送は起きない** (LRU の順序だけ)。
    /// 発火と当たり数は `[pf-reseed]` の 1 行で数える。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_reseed_decode_hot(&self) {
        let t0 = std::time::Instant::now();
        let (keys, ntok) = crate::model::exp_pf::take_dec_hot_keys();
        if keys.is_empty() {
            return;
        }
        let ek: Vec<ExpertKey> = keys
            .iter()
            .map(|&(layer, expert)| ExpertKey { layer, expert })
            .collect();
        let hit = expert_vram::touch_many_count(&ek);
        eprintln!(
            "[pf-reseed] tok={} keys={} resident={} ({:.1}%) in {:.2}ms",
            ntok,
            ek.len(),
            hit,
            100.0 * hit as f64 / ek.len().max(1) as f64,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }

    /// 逐次 prefill (= decoder と同じ経路)。既存の [`Qwen4Exp::forward_token_at`] を
    /// トークン数だけ呼ぶだけで、数値経路には触らない。
    pub(crate) fn prefill_seq(
        &mut self,
        ctx: &[u32],
        tokens: TapTokens,
        mut on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        mut roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>> {
        if ctx.is_empty() {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        let nt = ctx.len();
        let mut logits = Vec::new();
        // P78: このループの間だけ `forward_token_at` の層の外の区間を積む
        // (decode の `forward_token_at` は同じ関数なので、ガードで切り分ける)。
        let _pf = prekprof::enter_prefill();
        for i in 0..nt {
            let want = match tokens {
                TapTokens::All => true,
                TapTokens::Last => i + 1 == nt,
            };
            // `on_layer` は `&mut dyn` なので、ループのたびに短い寿命で借り直す
            // (`as_deref_mut` だと外側の寿命に固定されて 2 回目で E0499 になる)。
            let hook: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>> = match &mut on_layer {
                Some(f) if want => Some(&mut **f),
                _ => None,
            };
            let rr: Option<&mut RoleRms> = match &mut roles {
                Some(r) if want => Some(&mut **r),
                _ => None,
            };
            logits = self.forward_token_at(&ctx[..=i], i, nt, true, hook, rr)?;
        }
        Ok(logits)
    }

    /// [`Qwen4Exp::forward_prefill_batch_tapped`] のフック無し版 (テスト用)。
    pub(crate) fn forward_prefill_batch(&mut self, ctx: &[u32], chunk: usize) -> Result<Vec<f32>> {
        self.forward_prefill_batch_tapped(ctx, chunk, TapTokens::Last, None, None)
    }

    /// 層優先のバッチ prefill。`ctx` を `chunk` トークンずつに切り、チャンク内では
    /// 「層 0 の全トークン → 層 1 の全トークン → ...」の順に回す。
    ///
    /// チャンクを跨いで持ち越す残差は無い (各トークンの層 0 の入力は埋め込みだけで、
    /// 残差は層の中で閉じる)。attention の KV / GDN 状態は `attn` の中に
    /// **トークン順**で積まれるので、層優先でも順序は保たれる
    /// (`LayerAttn::layer_major_safe` が true の実装だけがここへ来る)。
    ///
    /// 数値: 1 トークンぶんの演算は逐次版と同じ関数・同じ順序で、行列積だけを
    /// [`gemv_batch`] に置き換える。したがって f32 でビット一致する。
    pub(crate) fn forward_prefill_batch_tapped(
        &mut self,
        ctx: &[u32],
        chunk: usize,
        tokens: TapTokens,
        on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>> {
        // P109: 従来経路は「最後の 1 位置だけ」。本体 (`forward_prefill_batch_core`)
        // に collect_all=false で渡すだけで、呼び出し規約も数値も不変。
        let mut all =
            self.forward_prefill_batch_core(ctx, chunk, tokens, on_layer, roles, false, 0)?;
        Ok(all
            .pop()
            .expect("collect_all=false は最後の位置の logits を返す"))
    }

    /// P109: 検証列を 1 回の forward で回し、**検証トークンごとの logits** を返す。
    ///
    /// n-gram 投機的デコード ([`crate::model::qwen4exp_decode::Qwen4ExpDecoder::spec_verify`])
    /// の検証用。`ctx` は PLE 索引の窓ぶんの履歴 (`n_hist` 個) + 検証列で、
    /// **食うのは `ctx[n_hist..]` だけ** (履歴は窓を作るためだけで、状態は進まない)。
    /// 戻り値 `[i]` は `ctx[n_hist..][..=i]` を食った直後の logits
    /// (位置 i が次の 1 トークンを予測する)。`layer_major_safe` な実装は
    /// 1 チャンクの層優先バッチで回し、そうでない実装は逐次で集める
    /// (同じトークンを同じ順で食う)。
    ///
    /// 数値: 層優先バッチは prefill と同じ `gemv_batch` (P-PF-GB、FMA 融合 +
    /// 加算順が違う) を通るので、逐次 decode ([`Self::forward_token_at`]) とは
    /// **ビット一致しない** (prefill の数値契約は解除済み。ULP 差で greedy が
    /// 反転することは稀)。
    pub fn forward_verify(&mut self, ctx: &[u32], n_hist: usize) -> Result<Vec<Vec<f32>>> {
        if ctx.len() <= n_hist {
            return Err(one("forward_verify: 検証列が空 (ctx が履歴と同じ長さ)"));
        }
        if self.attn.layer_major_safe() {
            self.forward_prefill_batch_core(
                ctx,
                ctx.len() - n_hist,
                TapTokens::Last,
                None,
                None,
                true,
                n_hist,
            )
        } else {
            let nt = ctx.len() - n_hist;
            let mut out = Vec::with_capacity(nt);
            for i in 0..nt {
                out.push(self.forward_token_at(&ctx[n_hist..][..=i], i, nt, true, None, None)?);
            }
            Ok(out)
        }
    }

/// task 154: 常駐経路を使う最小トークン数 (これ未満は従来の CPU バッチ)。
///
/// 28 トークンの prefill は mix の要素演算が数百 us で、GEMM 7 発 + 1 同期 +
/// 往復の固定費と釣り合わない。境界は task 145 のバッチ mix GPU と同じ 64。
pub(crate) const PF_RESIDENT_MIN_TOK: usize = 64;

/// task 788: 最初のチャンクで CUDA context を用意するか (純関数)。
///
/// `cuModuleLoadData` は current context を要求する。常駐経路に載る大きさ
/// (>= [`PF_RESIDENT_MIN_TOK`]) の最初のチャンクだけ、PTX を読む前に
/// `CudaCtx` を作って current にする。境界は常駐経路の判定と同じ 1 か所で決める。
#[cfg(feature = "cuda")]
pub(crate) fn pf_init_ctx_first(first_nt: usize) -> bool {
    first_nt >= Self::PF_RESIDENT_MIN_TOK
}

/// task 154: 常駐経路を採らなかった理由を 1 回だけ出す (既定は黙って従来経路)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_note_decline(reason: &str) {
    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        // tracing はこのモジュールのログを拾わない設定があるため eprintln にする。
        eprintln!("[pf] resident を見送り: {reason} → 従来経路");
    }
}

/// task 154: 常駐連鎖の失敗を要求エラーにする (途中で従来経路へ落ちると
/// attention の KV/GDN 状態が二重に進むため、黙って切り替えない)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_bail(msg: &str) -> Result<()> {
    Err(one(format!("prefill 常駐 (DENNO_PF_RESIDENT): {msg}")))
}

/// task 154: `GgufLinear` を常駐連鎖の行列記述へ。
#[cfg(feature = "cuda")]
pub(crate) fn pf_lin(w: &GgufLinear) -> crate::cuda::PfLin<'_> {
    crate::cuda::PfLin {
        uid: w.uid(),
        ty: w.ty(),
        blocks: w.blocks().as_slice(),
        k: w.k(),
        n: w.n(),
    }
}

/// task 172: shared expert の 3 射影をデバイスで回す (`cuda` feature 無しは常に false)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_shared_dev_run(lw: &LayerW, xs: &[f32], nt: usize, e: usize, sff: usize, sh: &mut [f32]) -> bool {
    // task 209: 入力が直近の `pf_mix` の出力そのものなら、H2D をせずに
    // デバイスの `mixed` を読む (`DENNO_PF_MIXDEVIN`)。
    if let Some(dp) = crate::cuda::pf_mixed_dev_in_ptr(xs.as_ptr(), nt, e) {
        return crate::cuda::pf_shared_dev(
            dp,
            nt,
            e,
            sff,
            Self::pf_lin(&lw.gate_shexp),
            Self::pf_lin(&lw.up_shexp),
            Self::pf_lin(&lw.down_shexp),
            sh,
        );
    }
    crate::cuda::pf_shared(
        xs,
        nt,
        e,
        sff,
        Self::pf_lin(&lw.gate_shexp),
        Self::pf_lin(&lw.up_shexp),
        Self::pf_lin(&lw.down_shexp),
        sh,
    )
}

/// task 172: `cuda` feature 無しの shared 連鎖 (常に従来経路)。
#[cfg(not(feature = "cuda"))]
pub(crate) fn pf_shared_dev_run(
    _lw: &LayerW,
    _xs: &[f32],
    _nt: usize,
    _e: usize,
    _sff: usize,
    _sh: &mut [f32],
) -> bool {
    false
}

/// task 172: PLE の key / value 射影をデバイスで回す (`cuda` feature 無しは常に false)。
#[cfg(feature = "cuda")]
pub(crate) fn pf_ple_dev_run(
    pw: &PleW,
    flat: &[f32],
    nt: usize,
    e: usize,
    key: &mut [f32],
    value: &mut [f32],
) -> bool {
    crate::cuda::pf_ple_proj(flat, nt, e, Self::pf_lin(&pw.key), Self::pf_lin(&pw.value), key, value)
}

/// task 172: `cuda` feature 無しの PLE 射影 (常に従来経路)。
#[cfg(not(feature = "cuda"))]
pub(crate) fn pf_ple_dev_run(
    _pw: &PleW,
    _flat: &[f32],
    _nt: usize,
    _e: usize,
    _key: &mut [f32],
    _value: &mut [f32],
) -> bool {
    false
}

/// task 154: この層の hc mix 6 本が常駐連鎖で回せるか (形と型)。
///
/// 連鎖が使う GEMM は BF16 (model の mix 本体) と F32 (inject 級) の 2 つ。
/// 1 本でも条件を外れたら層を進める前に従来経路へ戻す (途中で落ちない)。
pub(crate) fn pf_layer_ready(lw: &LayerW, e: usize, hc: usize, lr: usize) -> bool {
    let hc_dim = hc * e;
    Self::pf_lin_ok(&lw.hc_attn_down, hc_dim, lr)
        && Self::pf_lin_ok(&lw.hc_attn_up, lr, hc_dim)
        && Self::pf_lin_ok(&lw.hc_attn_inject, hc_dim, hc)
        && Self::pf_lin_ok(&lw.hc_ffn_down, hc_dim, lr)
        && Self::pf_lin_ok(&lw.hc_ffn_up, lr, hc_dim)
        && Self::pf_lin_ok(&lw.hc_ffn_inject, hc_dim, hc)
}

/// task 154: 常駐連鎖が 1 本の行列を扱えるか (型と形)。
///
/// 実モデルでは mix の down / up が BF16、inject が F32 か Q8_0 (requant) で来る。
/// ここを狭くしすぎると**層を 1 つも進めずに**従来経路へ落ちる (lab t154b/t154c で
/// 実際に踏んだ) ので、pf_weight が常駐できる型をそのまま並べる。
pub(crate) fn pf_lin_ok(w: &GgufLinear, k: usize, n: usize) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(
        w.ty(),
        T::Bf16 | T::F32 | T::Q8_0 | T::Q5_1 | T::Q4K | T::Q5K
    ) && w.k() == k
        && w.n() == n
}

    /// task 154: デバイス常駐の prefill 本体 (`DENNO_PF_RESIDENT` の mix 側)。
    ///
    /// 従来の [`Self::forward_prefill_batch_core`] との違いは hc mix と hc combine
    /// だけ:
    ///
    /// - チャンクの残差 `res` は開始時に 1 回だけ H2D し、層をまたいでデバイスに置く
    /// - mix 1 段 (norm → down GEMM → lo → up GEMM → mixed → inject) をデバイスの
    ///   1 連鎖で回し、ホストへは `mixed` (`[nt][n_embd]`) だけ落とす
    /// - hc combine (`res += block_out * 2*sigmoid(inject)`) もデバイスで回す
    ///
    /// CPU が計算するもの (attention / MoE / PLE) は従来どおりで、その入出力だけが
    /// 往復する。数値は `DENNO_MIX_BATCH_GPU=1` の既存経路と同じ (同じ GEMM カーネル・
    /// 同じ要素演算の順序)。
    ///
    /// 実行できないときは **層を 1 つも進めずに** `Ok(None)` を返す (呼び出し側が
    /// 従来経路をそのまま走らせる)。開始後に失敗したら `Err` にする — attention の
    /// KV/GDN 状態は既に進んでいるので、黙って従来経路へ落ちると状態が二重に進む。

    /// task 230: full 層の attn mix を D2H せず、直後の qkv にデバイスの
    /// `mixed` をそのまま読ませるか。
    ///
    /// 条件は「ゲート on・平坦な活性 (トークンごとの Vec を作らない)・full 層・
    /// qkv のデバイス連鎖が受ける行数」。ssm 層 (GDN がホストを読む) と MoE は
    /// 従来どおりホストへ降ろす。
    pub(crate) fn pf_mix_keep_layer(flat_attn: bool, nt: usize, kind: Option<LayerKind>) -> bool {
        #[cfg(feature = "cuda")]
        {
            // `DENNO_PF_MIXDEVIN=0` のときは qkv がデバイスを読まないので、
            // D2H を飛ばしてはいけない (ホストが古いバイト列になる)。
            crate::cuda::pf_dev_in_on()
                && flat_attn
                && nt >= 64
                && match kind {
                    // task 230: full 層は直後の qkv (`pf_full_qkv` / v3) が
                    // デバイスの `mixed` を直接読める。
                    Some(LayerKind::FullAttn) => crate::cuda::pf_mix_keep_on(),
                    // task 317: ssm 層は直後の GDN 連鎖 (`gdn_pf_chain`) が
                    // デバイスの `mixed` を読む (`DENNO_PF_MIXKEEP_SSM`)。
                    // 連鎖自体が off の環境では D2H を飛ばさない (飛ばすと
                    // 復元の D2H が 1 本増えるだけ)。連鎖に乗らない層は
                    // `forward_batch_flat` が復元してからホスト経路で計算する。
                    Some(LayerKind::Ssm) => {
                        crate::cuda::pf_mix_keep_ssm_on() && pf_gdn_chain_on()
                    }
                    None => false,
                }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (flat_attn, nt, kind);
            false
        }
    }

    /// task 200: 層優先 prefill の 1 層ぶんの attn 呼び出し。
    ///
    /// `flat_attn` が true なら平坦な入口 ([`LayerAttn::forward_batch_flat`]) を
    /// 通り、`mixed_flat` をそのまま渡して平坦な出力を返す (トークンごとの
    /// `Vec<f32>` を作らない)。false なら従来の行入口
    /// ([`LayerAttn::forward_batch`]) + [`flatten_rows`]。
    /// 戻り値はどちらも平坦な活性 (`nt` 行 × `e`)、値はビット一致する。
    pub(crate) fn prefill_attn_chunk(
        attn: &mut dyn LayerAttn,
        il: usize,
        mixed_flat: &[f32],
        curs: &[Vec<f32>],
        nt: usize,
        e: usize,
        flat_attn: bool,
        allow_dev: bool,
    ) -> Result<Vec<f32>> {
        #[cfg(test)]
        if flat_attn {
            PF_ATTN_FLAT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            PF_ATTN_ROWS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if flat_attn {
            let f = attn.forward_batch_flat(il, mixed_flat, nt)?;
            // 空 = 実装が層出力をデバイスに残した (呼び出し側が combine で読む)。
            if f.is_empty() && allow_dev {
                return Ok(f);
            }
            if f.len() != nt * e {
                return Err(one(format!(
                    "層 {il}: attention の出力長 {} が nt {nt} * n_embd {e} と違う",
                    f.len()
                )));
            }
            return Ok(f);
        }
        let outs = attn.forward_batch(il, curs)?;
        for o in &outs {
            if o.len() != e {
                return Err(one(format!(
                    "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                    o.len()
                )));
            }
        }
        if outs.len() != nt {
            return Err(one(format!(
                "層 {il}: attention の出力本数 {} が nt {nt} と違う",
                outs.len()
            )));
        }
        flatten_rows(&outs, e)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_prefill_batch_resident(
        &mut self,
        ctx: &[u32],
        chunk: usize,
        tokens: TapTokens,
        mut on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        mut roles: Option<&mut RoleRms>,
        iv: IvView<'_>,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let hp = self.hp();
        let e = self.p.n_embd;
        let hc = hp.hc;
        let lr = hp.low_rank;
        let hc_dim = hc * e;
        let n_layer = self.p.n_layer;
        let chunk = chunk.max(1);
        if ctx.is_empty() {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        // 形と重みの前提は**層を進める前に**全部見る (途中で落ちると状態が進む)。
        if hc == 0 || e == 0 || lr == 0 || hc_dim > (1 << 20) {
            Self::pf_note_decline("形が常駐連鎖の前提外");
            return Ok(None);
        }
        for il in 0..n_layer {
            if !Self::pf_layer_ready(&self.layers[il], e, hc, lr) {
                Self::pf_note_decline("層の重みが常駐連鎖の前提外 (型/形)");
            return Ok(None);
            }
        }
        let first_nt = chunk.min(ctx.len());
        // 短いプロンプトは従来の CPU 経路のほうが速い (GEMM の起動と往復が勝つ)。
        // 判定は「このリクエストの最初のチャンク」で 1 回だけ行う。
        if first_nt < Self::PF_RESIDENT_MIN_TOK {
            Self::pf_note_decline("64 トークン未満");
            return Ok(None);
        }
        // task 227: 層優先 (`DENNO_PF_LM_LAYER=1`)。1 チャンクに収まるプロンプトは
        // チャンク優先と同じ結果になるので、そのまま下へ落とす。
        if crate::model::exp_pf::lm_layer_on() && ctx.len() > first_nt {
            let hook: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>> = match &mut on_layer {
                Some(f) => Some(&mut **f),
                None => None,
            };
            if let Some(v) = self.forward_prefill_batch_resident_lm(
                ctx,
                chunk,
                tokens,
                hook,
                roles.as_deref_mut(),
                iv,
            )? {
                return Ok(Some(v));
            }
        }
        // task 257/265: 作業域のプール (この prefill のあいだ expert から借りる)。
        // チャンク優先は 1 チャンクぶんの MoE なので、タイル = 最初のチャンク。
        // ガードはこの関数のスコープに置く (早期撤退でも終端で返る)。
        let _pool = self.pf_pool_begin_for(first_nt, first_nt, first_nt, hc, lr);
        if !crate::cuda::pf_begin(first_nt, hc, e, lr) {
            Self::pf_note_decline("作業域/PTX/GPU の前提外");
            return Ok(None);
        }
        // task 218: 記録 / ステアリングもデバイスで閉じられるか**層を進める前に**見る。
        // steer の PTX が無い・確保できないときは常駐連鎖を諦め、呼び出し側が
        // ホスト経路で記録 / 適用する (値は同じ、速度だけ落ちる)。
        let steer_state = steer::snapshot();
        let steer_apply = steer_state.apply.as_ref();
        let steer_rec = steer::rec_on();
        // task 849: 重ね掛け (rules) は 1 本の合成ベクトルに畳めない (規則ごとに
        // 層範囲 / alpha / mode が違う)。GPU の常駐経路は使わずホスト経路へ落とす
        // (値は同じ、速度だけ落ちる)。単一規則は従来どおり。
        let steer_dev_apply = steer_apply
            .map(|a| a.is_single_rule() && a.from < n_layer && a.to > 0)
            .unwrap_or(false);
        if steer_rec && !crate::cuda::pf_steer_acc_ensure(n_layer, hc, e) {
            crate::cuda::pf_release();
            Self::pf_note_decline("steer の加算域が使えない");
            return Ok(None);
        }
        if steer_dev_apply {
            let a = steer_apply.expect("steer_dev_apply は apply があるときだけ真");
            if !crate::cuda::pf_steer_apply_begin(&a.v, n_layer, hc, e) {
                crate::cuda::pf_release();
                Self::pf_note_decline("steer の v をデバイスへ置けない");
                return Ok(None);
            }
        }
        // task 209: `DENNO_PF_PIN` のときは mix の出力 (mix の D2H の宛先、
        // router と shared の H2D の元) を pinned ホストメモリに置く。同じ
        // バイト列がページャブル経由より速く動く (値は 1 bit も変わらない)。
        // 1 リクエスト 1 本 (チャンクで使い回す)。
        let mut mixed_buf = PfHostBuf::new(first_nt * e);
        // task 209: 「このホスト配列はデバイスの mixed と同一」の記録は
        // この関数を出るときに必ず消す (非常駐経路が古い記録を拾わない)。
        let _mixed_mark = PfMixedMarkGuard;
        // P78: チャンクぶんの PLE は行単位でまとめて引く (従来と同じ)。
        let ple_batch = ple_batch_enabled();
        // task 254: 次チャンクの PLE 行を、このチャンクの層ループの裏で読む。
        let ple_pre = ple_batch && crate::ple::pf_ple_pre_on();
        // task 349: `DENNO_PF_PLE_PIPE` では 1 チャンク先ではなく**プロンプト
        // 全体**の gather を 1 本の生産スレッドで流す。冷えた page cache では
        // gather が 2.7 s かかり、1 チャンク (0.46 s) には隠れないため。
        let mut ple_pipe_rx: Option<std::sync::mpsc::Receiver<Result<Vec<Option<Vec<f32>>>>>> =
            None;
        if ple_pre && crate::model::exp_pf::ple_pipe() && ctx.len() > chunk {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<Option<Vec<f32>>>>>(2);
            let c: Vec<u32> = ctx.to_vec();
            let rt = self.ple.as_ref().map(std::sync::Arc::clone);
            let ch = chunk;
            // 先読みは専用プールで回す (層ループの rayon を奪わない)。
            // プールは 1 回だけ作る (チャンクごとに作るとスレッドの起動だけで重い)。
            std::thread::spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(PLE_PRE_THREADS)
                    .build()
                    .ok();
                let mut s = 0usize;
                while s < c.len() {
                    let e = (s + ch).min(c.len());
                    let r = match pool.as_ref() {
                        Some(p) => p.install(|| gather_ple_many_rt(rt.as_deref(), &c[..e], s)),
                        None => gather_ple_many_rt(rt.as_deref(), &c[..e], s),
                    };
                    let bad = r.is_err();
                    // 受け手が居なくなったら (prefill が途中で失敗した) 止まる。
                    if tx.send(r).is_err() {
                        break;
                    }
                    if bad {
                        break;
                    }
                    s = e;
                }
            });
            ple_pipe_rx = Some(rx);
        }
        // 先読みの結果 (次チャンクぶん)。1 チャンク目は先読みできない。
        let mut ples_next: Option<Vec<Option<Vec<f32>>>> = None;
        let mut last: Option<HyperStreams> = None;
        let mut start = 0usize;
        while start < ctx.len() {
            let end = (start + chunk).min(ctx.len());
            let nt = end - start;
            let mut ress: Vec<HyperStreams> = Vec::with_capacity(nt);
            let t_emb = prekprof::stamp();
            for i in start..end {
                ress.push(self.forward_embed(&ctx[..=i], &hp)?);
            }
            prekprof::mark(t_emb, prekprof::EMBED);
            let t_ple = prekprof::stamp();
            let ples: Vec<Option<Vec<f32>>> = if let Some(rx) = ple_pipe_rx.as_ref() {
                // 生産スレッドはチャンク順に流すので、受けるだけで順序が合う。
                rx.recv()
                    .map_err(|_| one("PLE のパイプラインが止まった"))?
                    ?
            } else if ple_batch {
                match ples_next.take() {
                    // 前のチャンクの層ループの裏で読み終わっている。
                    Some(v) => v,
                    None => self.gather_ple_many(&ctx[..end], start)?,
                }
            } else {
                let mut v: Vec<Option<Vec<f32>>> = Vec::with_capacity(nt);
                for i in start..end {
                    v.push(self.gather_ple(&ctx[..=i])?);
                }
                v
            };
            prekprof::mark(t_ple, prekprof::PLE);
            // 残差をデバイスへ (チャンクで 1 回)。
            let mut res_flat: Vec<f32> = Vec::with_capacity(nt * hc_dim);
            for r in &ress {
                res_flat.extend_from_slice(&r.data);
            }
            if !crate::cuda::pf_upload_res(&res_flat) {
                Self::pf_bail("res の H2D に失敗")?;
            }
            let mixed_flat: &mut [f32] = &mut mixed_buf.as_mut_slice()[..nt * e];
            // task 254: 次チャンクの PLE 行 gather を、このチャンクの層ループ
            // (GPU) の裏で走らせる。最終チャンクでは起動しない。
            // task 349: パイプライン (`DENNO_PF_PLE_PIPE`) のときは生産スレッドが
            // 全チャンクを受け持つので、ここでは起こさない (二重に読まない)。
            let pre = if ple_pre && ple_pipe_rx.is_none() && end < ctx.len() {
                PlePrefetch::spawn(
                    self.ple.as_ref().map(std::sync::Arc::clone),
                    ctx,
                    end,
                    (end + chunk).min(ctx.len()),
                )
            } else {
                PlePrefetch::none()
            };
            let t_lay = prekprof::stamp();
            for il in 0..n_layer {
                let lw = &self.layers[il];
                if q4prof::enabled() {
                    q4prof::N_LAYER.fetch_add(nt as u64, std::sync::atomic::Ordering::Relaxed);
                }
                if let Some(d) = roles.as_deref_mut() {
                    *d = RoleRms::default();
                }
                // task 218: ステアリング (層の入口 = 層 `il-1` の出口と同じ点)。
                if steer_dev_apply && steer_apply.map(|a| a.covers(il)).unwrap_or(false) {
                    let a = steer_apply.expect("covers が真なら apply がある");
                    let mode = if a.mode == steer::Mode::Project { 1u8 } else { 0u8 };
                    if !crate::cuda::pf_steer_apply(nt, hc, e, il, mode, a.alpha) {
                        Self::pf_bail("steer の適用に失敗")?;
                    }
                }

                // 3.1 PLE 注入 (層 1 のみ)。conv 履歴はトークン順に進める。
                // ホストで回すので、この層だけ res を往復する。
                if let Some(pw) = &lw.ple {
                    // task 300: `DENNO_PF_PLE_REUSE` では res の往復バッファを使い回す。
                    let (mut back, back_reused) = pf_ple_back_take(nt * hc_dim);
                    if !crate::cuda::pf_download_res(&mut back) {
                        Self::pf_bail("PLE 層の res の D2H に失敗")?;
                    }
                    // task 154: 射影 (key / value) はチャンクでまとめて回す。
                    // 残り (RMS / gating / conv) はトークン順で、値は 1 個ずつ
                    // 回すのとビット一致 (gemv_batch_cpu は逐次版と同じカーネル)。
                    let mut rows: Vec<&[f32]> = Vec::with_capacity(nt);
                    for t in 0..nt {
                        rows.push(
                            ples[t]
                                .as_deref()
                                .ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?,
                        );
                    }
                    let mut contrib = 0.0f32;
                    let want = roles.is_some();
                    let hist = self.ple_conv_state[il]
                        .as_mut()
                        .ok_or_else(|| one(format!("層 {il}: PLE の conv 履歴が無い")))?;
                    // task 195: `DENNO_PF_PLE_FAST` は平坦な res をその場で更新する
                    // (トークンごとの `HyperStreams` の確保と書き戻しをしない)。
                    // トークンが多いときは 2 段に分けて並列に回す。
                    if pf_ple_fast_on() {
                        if nt >= PLE_PAR_MIN_TOK {
                            ple_inject_batch_par(
                                &self.p,
                                pw,
                                &rows,
                                &mut back,
                                &hp,
                                hist,
                                want.then_some(&mut contrib),
                            )?;
                        } else {
                            ple_inject_batch_flat(
                                &self.p,
                                pw,
                                &rows,
                                &mut back,
                                &hp,
                                hist,
                                want.then_some(&mut contrib),
                            )?;
                        }
                    } else {
                        let mut streams: Vec<HyperStreams> = (0..nt)
                            .map(|t| HyperStreams {
                                data: back[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                                hc,
                                n_embd: e,
                            })
                            .collect();
                        ple_inject_batch(
                            &self.p,
                            pw,
                            &rows,
                            &mut streams,
                            &hp,
                            hist,
                            want.then_some(&mut contrib),
                        )?;
                        for t in 0..nt {
                            back[t * hc_dim..(t + 1) * hc_dim]
                                .copy_from_slice(&streams[t].data);
                        }
                    }
                    if want {
                        if let Some(r) = roles.as_deref_mut() {
                            r.ple = contrib;
                        }
                    }
                    if !crate::cuda::pf_upload_res(&back) {
                        Self::pf_bail("PLE 層の res の H2D に失敗")?;
                    }
                    // task 300: 往復バッファを返す。
                    pf_ple_back_return(back, back_reused);
                }

                // 3.2-3.4 attention 入口 (デバイス mix) → attn (CPU) → combine (デバイス)
                let t_mix = std::time::Instant::now();
                // task 200: `DENNO_PF_ATTN_FLAT` は活性を平坦なまま attn へ渡し、
                // 平坦なまま受け取る (トークンごとの Vec を作らない)。
                let flat_attn = pf_attn_flat_on();
                {
                    let _l = q4ledger::enter(q4ledger::MIX);
                    if !crate::cuda::pf_mix(
                        &lw.hc_attn_norm,
                        Self::pf_lin(&lw.hc_attn_down),
                        Self::pf_lin(&lw.hc_attn_up),
                        Self::pf_lin(&lw.hc_attn_inject),
                        nt,
                        hc,
                        e,
                        lr,
                        hp.eps,
                    ) {
                        Self::pf_bail("attn mix の連鎖に失敗")?;
                    }
                    // task 230: full 層は直後の qkv がデバイスの `mixed` を直接
                    // 読めるので、この mix では D2H しない (`DENNO_PF_MIXKEEP`)。
                    // ホストが要る経路 (ssm / 非常駐 / 小さい nt) では従来どおり。
                    if Self::pf_mix_keep_layer(flat_attn, nt, self.p.layer_kind(il).ok()) {
                        crate::cuda::pf_mixed_keep_dev(mixed_flat.as_ptr(), nt, e);
                    } else {
                        if !crate::cuda::pf_take_mixed(mixed_flat) {
                            Self::pf_bail("attn mix の mixed D2H に失敗")?;
                        }
                        // task 209: デバイスの `mixed` とこのホスト配列は同じ内容
                        // (`DENNO_PF_MIXDEVIN` の router / shared が読む)。
                        crate::cuda::pf_mixed_mark(mixed_flat.as_ptr(), nt, e);
                    }
                }
                q4prof::add(&q4prof::MIX, t_mix);
                // task 200: `DENNO_PF_ATTN_DEVRES` は ssm 連鎖の層出力をデバイスに
                // 残し、直後の combine にそのまま読ませる (roles を取るときはホストが要る)。
                let attn_dev = flat_attn && pf_attn_devres_on() && roles.is_none();
                let curs: Vec<Vec<f32>> = if flat_attn {
                    Vec::new()
                } else {
                    mixed_flat.chunks_exact(e).map(|c| c.to_vec()).collect()
                };
                let t_attn = std::time::Instant::now();
                crate::cuda::pf_attn_dev_clear();
                crate::cuda::pf_attn_dev_request(attn_dev);
                let flat_outs = {
                    let item = if q4ledger::enabled() {
                        match self.p.layer_kind(il) {
                            Ok(LayerKind::FullAttn) => q4ledger::FULL,
                            _ => q4ledger::SSM,
                        }
                    } else {
                        q4ledger::OTHER
                    };
                    let _l = q4ledger::enter(item);
                    let r = Self::prefill_attn_chunk(
                        self.attn.as_mut(),
                        il,
                        mixed_flat,
                        &curs,
                        nt,
                        e,
                        flat_attn,
                        attn_dev,
                    );
                    crate::cuda::pf_attn_dev_request(false);
                    r?
                };
                q4prof::add(&q4prof::ATTN, t_attn);
                // 空の出力 = 層出力はデバイスに残っている (combine が直接読む)。
                let dev_out = flat_outs.is_empty();
                if let Some(r) = roles.as_deref_mut() {
                    if !dev_out {
                        let t = nt - 1;
                        let v = rms_of(&flat_outs[t * e..(t + 1) * e]);
                        if matches!(self.p.layer_kind(il), Ok(LayerKind::FullAttn)) {
                            r.attn = v;
                        } else {
                            r.gdn = v;
                        }
                    }
                }
                let combined = if dev_out {
                    crate::cuda::pf_combine_dev(nt, hc, e)
                } else {
                    crate::cuda::pf_combine(&flat_outs, nt, hc, e)
                };
                if !combined {
                    Self::pf_bail("attn の combine に失敗")?;
                }

                // 3.5-3.7 ffn 入口 (デバイス mix) → MoE (CPU/GPU) → combine (デバイス)
                let t_mix = std::time::Instant::now();
                {
                    let _l = q4ledger::enter(q4ledger::MIX);
                    if !crate::cuda::pf_mix(
                        &lw.hc_ffn_norm,
                        Self::pf_lin(&lw.hc_ffn_down),
                        Self::pf_lin(&lw.hc_ffn_up),
                        Self::pf_lin(&lw.hc_ffn_inject),
                        nt,
                        hc,
                        e,
                        lr,
                        hp.eps,
                    ) {
                        Self::pf_bail("ffn mix の連鎖に失敗")?;
                    }
                    if !crate::cuda::pf_take_mixed(mixed_flat) {
                        Self::pf_bail("ffn mix の mixed D2H に失敗")?;
                    }
                    crate::cuda::pf_mixed_mark(mixed_flat.as_ptr(), nt, e);
                }
                q4prof::add(&q4prof::MIX, t_mix);
                let t_moe = std::time::Instant::now();
                let ffn = moe_ffn_quant_batch_tapped(
                    &self.p,
                    lw,
                    &mixed_flat,
                    nt,
                    self.expert_cache.as_ref(),
                    roles.as_deref_mut(),
                    true,
                    iv,
                )?;
                q4prof::add(&q4prof::MOE, t_moe);
                if ffn.len() != nt * e {
                    return Err(one(format!(
                        "層 {il}: MoE の出力長 {} が nt {nt} * n_embd {e} と違う",
                        ffn.len()
                    )));
                }
                if !crate::cuda::pf_combine(&ffn, nt, hc, e) {
                    Self::pf_bail("ffn の combine に失敗")?;
                }
                // task 218: 層の出口の残差をデバイスで加算する (D2H しない)。
                // task 829: LastPromptToken は最終チャンクの最後の行だけを加算する。
                if steer_rec {
                    let last_only = steer::last_only();
                    let final_chunk = end == ctx.len();
                    if !last_only || final_chunk {
                        let ok = if last_only {
                            crate::cuda::pf_steer_acc_last(nt, hc, e, il)
                        } else {
                            crate::cuda::pf_steer_acc(nt, hc, e, il)
                        };
                        if !ok {
                            Self::pf_bail("steer の記録に失敗")?;
                        }
                    }
                }
                if end == ctx.len() {
                    if let Some(r) = roles.as_deref() {
                        emit_role_act(il, r);
                    }
                }
                // 層の出口フック (TapTokens::Last のみ): 残差 1 行だけ D2H して渡す。
                if let Some(h) = on_layer.as_deref_mut() {
                    let t = nt - 1;
                    let mut row = vec![0.0f32; hc_dim];
                    if !crate::cuda::pf_download_res_row(t, hc_dim, &mut row) {
                        Self::pf_bail("層の出口フック用の res D2H に失敗")?;
                    }
                    h(&LayerOut {
                        il,
                        t,
                        nt,
                        n_embd: e,
                        res: &row,
                        roles: roles.as_deref(),
                    })?;
                }
            }
            if steer_rec {
                // task 829: LastPromptToken は最終チャンクの 1 トークンだけ数える。
                if steer::last_only() {
                    if end == ctx.len() {
                        steer::note_dev_tokens(1);
                    }
                } else {
                    steer::note_dev_tokens(nt as u64);
                }
            }
            prekprof::mark(t_lay, prekprof::LAYER);
            // 最終チャンクだけ、最終 mixer 用に残差をホストへ戻す。
            if end == ctx.len() {
                if !crate::cuda::pf_download_res(&mut res_flat) {
                    Self::pf_bail("最終チャンクの res の D2H に失敗")?;
                }
                ress.clear();
                for t in 0..nt {
                    ress.push(HyperStreams {
                        data: res_flat[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                        hc,
                        n_embd: e,
                    });
                }
                last = ress.pop();
            }
            // task 254: 先読みを回収する (ここまでで層ループの裏に隠れている)。
            if pre.is_some() {
                ples_next = Some(pre.join()?);
            }
            start = end;
        }
        crate::cuda::pf_release();
        // task 218: デバイスの加算域をホストへ降ろす (加算したスレッドで 1 回)。
        if steer_rec {
            steer::merge_device();
        }
        let res = last.ok_or_else(|| one("ctx が空 (現在のトークンが無い)"))?;
        let t_mix = headprof::stamp();
        let t_fin = prekprof::stamp();
        let cur = hc_mix_quant(
            &res,
            &self.head.norm,
            &self.head.down,
            &self.head.up,
            None,
            &hp,
        )?
        .0;
        headprof::mark(headprof::MIX, t_mix, 0);
        prekprof::mark(t_fin, prekprof::FINAL_NORM);
        let mut logits = vec![0.0f32; self.head.output.n()];
        let t_head = prekprof::stamp();
        self.head_logits_into(&cur, &mut logits)?;
        prekprof::mark(t_head, prekprof::HEAD);
        Ok(Some(vec![logits]))
    }

    /// task 257/265: このリクエストの形で prefill 作業域のプール区画を組む。
    ///
    /// task 265: プールは **この prefill のあいだだけ** expert から借りる。
    /// 戻り値のガードを prefill の関数スコープに置くと、終端 (早期撤退も含む) で
    /// 返る。借りられないときは `owned == false` で、確保は従来の
    /// `alloc_tagged` に落ちる (fail open。リクエストは失敗させない)。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_pool_begin_for(
        &self,
        nt_all: usize,
        tile: usize,
        chunk: usize,
        hc: usize,
        lr: usize,
    ) -> crate::cuda::PfPoolGuard {
        let need = self.pf_pool_need_for(nt_all, tile, chunk, hc, lr);
        crate::cuda::pf_pool_acquire(&need)
    }

    /// task 420: 層ごとの TC 適格性 `(IQ の形が合う, K の形が合う)` を集める。
    ///
    /// 実行時の選択 (`cuda` のストリーミング MoE の `tc_kernels`) と**同じ 2 段**で
    /// 見る: IQ の形が合えば IQ 版、合わなければ K 版 (IQ の形が合って PTX が無い
    /// 層は K へ落ちない)。判定は層の型だけで、PTX の有無は [`pf_pool_tc_all`] が受ける。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_pool_tc_layers(&self, e: usize, ff: usize) -> Vec<(bool, bool)> {
        self.layers
            .iter()
            .map(|lw| {
                let (g, u, d) = (lw.gate_exps.ty, lw.up_exps.ty, lw.down_exps.ty);
                (
                    crate::cuda::tc_shapes_ok(
                        crate::model::exp_pf::slot(g),
                        crate::model::exp_pf::slot(d),
                        g == u,
                        e,
                        ff,
                        ff,
                        e,
                    ),
                    crate::cuda::ktc_shapes_ok(
                        crate::model::exp_pf::slot(g),
                        crate::model::exp_pf::slot(d),
                        g == u,
                        e,
                        ff,
                        ff,
                        e,
                    ),
                )
            })
            .collect()
    }

    /// task 279: このリクエストの形 (チャンク / タイル) からレイアウトの材料を組む。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_pool_need_for(
        &self,
        nt_all: usize,
        tile: usize,
        chunk: usize,
        hc: usize,
        lr: usize,
    ) -> crate::model::pf_pool::Need {
        let e = self.p.n_embd;
        let ff = self.p.n_ff_exp;
        // tensor core の grouped GEMM がこのモデルで実際に使えるか。
        // 使えないときは GEMV 経路になり g / u の作業域が要るので、その分も取る。
        let lw0 = &self.layers[0];
        let (ty_g, ty_u, ty_d) = (lw0.gate_exps.ty, lw0.up_exps.ty, lw0.down_exps.ty);
        // task 420: 実行時の TC の選択は層ごとに IQ → K の 2 段。K 系
        // (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0) は task 416 から既定 on なので、
        // 計画側も K を見ないと **使わない g / u を予約してしまう**
        // (UD-Q4_K_XL の 14k で 686 MiB。1 パスのタイルが上限に収まらず 2 パスに
        // なり、expert の pread が 98.5 GB = 1 パスの 2 倍になっていた)。
        let tc = if crate::model::exp_pf::pool_k_plan_on() {
            !crate::model::exp_pf::lm_group_local()
                && pf_pool_tc_all(
                    &self.pf_pool_tc_layers(e, ff),
                    (
                        crate::model::exp_pf::tc_exp_on(),
                        crate::cuda::exp_tc_ready(),
                    ),
                    (
                        crate::model::exp_pf::tc_exp_k_on(),
                        crate::cuda::exp_ktc_ready(),
                    ),
                )
        } else {
            crate::model::exp_pf::tc_exp_on()
                && !crate::model::exp_pf::lm_group_local()
                && crate::cuda::exp_tc_ready()
                && crate::cuda::tc_shapes_ok(
                    crate::model::exp_pf::slot(ty_g),
                    crate::model::exp_pf::slot(ty_d),
                    ty_g == ty_u,
                    e,
                    ff,
                    ff,
                    e,
                )
        };
        crate::model::pf_pool::Need {
            nt_all,
            tile,
            k: self.p.n_expert_used,
            e,
            ff,
            sff: lw0.gate_shexp.n(),
            hc,
            lr,
            chunk: chunk.max(1),
            ne: self.p.n_expert,
            tc,
            stage_bytes: crate::model::exp_pf::stream_bytes(),
            stage_slots: crate::cuda::PF_STAGE_SLOTS,
        }
    }

    /// task 279: 借りられる量から (チャンク, タイル) を選ぶ (純関数の決定 + 上限)。
    ///
    /// `DENNO_PF_POOL_BIG=1` のとき、パス数 (= expert 重みの H2D 量) が減る順に
    /// 候補を試し、[`crate::model::pf_pool::PF_POOL_BIG_MB_ENV`] に収まる最初の組を
    /// 返す。ゲート off / 候補なし / 上限が増えないときは従来の形を返す
    /// (呼び出し側はそのまま `pf_pool_acquire_capped` に渡す)。
    ///
    /// 戻り値の 3 番目は「大きい形を選んだか」。true のとき呼び出し側は、借りた
    /// 作業域のぶんに追い出した expert を prefill の終端で LRU へ戻す。
    ///
    /// 4 番目は「借りたあとに残す余白」(task 290)。プールの確保がこの余白を
    /// 残すように、借りる前に expert を追い出す。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_pool_pick(
        &self,
        nt_all: usize,
        chunk: usize,
        hc: usize,
        lr: usize,
    ) -> (crate::model::pf_pool::Need, usize, bool, usize) {
        let base_tile = crate::model::exp_pf::lm_tile(nt_all);
        let base = self.pf_pool_need_for(nt_all, base_tile, chunk.min(nt_all), hc, lr);
        let cap = crate::model::pf_pool::pf_pool_bytes();
        let reserve = self.pf_pool_reserve_for(nt_all);
        if !crate::model::pf_pool::pf_pool_big_on() {
            return (base, cap, false, reserve);
        }
        let big_cap = crate::model::pf_pool::pf_pool_big_bytes();
        if big_cap <= cap {
            return (base, cap, false, reserve);
        }
        let max_chunk = crate::model::pf_pool::big_chunk_max();
        // 2,048 以上のチャンクは GDN prefill の連鎖の上限撤廃 (task 270 の
        // `DENNO_PF_CHAIN_BIG`) が前提。無いまま大きいチャンクを使うと ssm が
        // 連鎖の外へ落ちる (task 197: 3.96 → 24.7 s)。
        let max_chunk = if crate::cuda::chain_big_on() {
            max_chunk
        } else {
            max_chunk.min(crate::model::exp_pf::CHUNK_BALANCE_CAP)
        };
        // task 290: 借りてよい上限は**実空きから**決める。expert の LRU を
        // 追い出して作れる分は「余白を残したうえで」使う。上限を固定値
        // (4,864 MiB) にすると、モデルが大きいときにカードを 100% まで使い切り、
        // prefill の後半の数 MiB の確保が断片化で落ちる (lab t286c の
        // UD-Q4_K_XL 14k: 空き 3 MiB で 3 MiB の確保が error 2)。
        let free = crate::cuda::dev_mem_mb().map(|(f, _)| (f as usize) << 20);
        let evictable = crate::model::expert_vram::resident_bytes().min(big_cap);
        let budget = crate::model::pf_pool::pool_budget(big_cap, free, evictable, reserve);
        eprintln!(
            "[pf-pool-big] 上限 (実空き {} MiB + 追い出せる expert {} MiB - 余白 {} MiB = {} MiB / 固定上限 {} MiB)",
            free.map(|b| b >> 20).unwrap_or(0),
            evictable >> 20,
            reserve >> 20,
            budget >> 20,
            big_cap >> 20
        );
        match crate::model::pf_pool::choose_big(&base, budget, max_chunk) {
            Some((c, t, l)) if c != base.chunk || t != base.tile => {
                let passes = nt_all.div_ceil(t.max(1));
                let base_passes = nt_all.div_ceil(base.tile.max(1));
                eprintln!(
                    "[pf-pool-big] 形を上げた (チャンク {} → {} / タイル {} → {} = {} → {} パス / 作業域 {} MiB)",
                    base.chunk,
                    c,
                    base.tile,
                    t,
                    base_passes,
                    passes,
                    l.total >> 20
                );
                let need = crate::model::pf_pool::Need {
                    chunk: c,
                    tile: t,
                    ..base
                };
                (need, big_cap, true, reserve)
            }
            _ => (base, cap, false, reserve),
        }
    }

    /// task 290: プールのあとに残す余白 (バイト)。
    ///
    /// 層優先の dev MoE が使えるモデルは、作業域がプールの中に収まるので余白は
    /// 小さくてよい。使えないモデル (expert の型が IQ カーネル非対応 =
    /// UD-Q4_K_XL など) はホスト経路がステージングを自前で確保するので、
    /// 余白を広く取ってパス数を増やす (lab t286c の 14k は約 2.3 GB 足りなかった)。
    #[cfg(feature = "cuda")]
    pub(crate) fn pf_pool_reserve_for(&self, nt_all: usize) -> usize {
        let lw0 = &self.layers[0];
        // task 420: task 318 で層優先の dev MoE の前提は `pf_expert_mode` (IQ or K)
        // になったが、ここは `iq_expert_mode` のままだった。UD-Q4_K_XL は dev MoE を
        // 使うのに「使えない」と判定され、余白が CPU 経路の 3,072 MiB になっていた
        // (dev 経路は 1,024 MiB)。余白 2,048 MiB の差が、14k の 1 パスを潰していた。
        let expert_mode = if crate::model::exp_pf::pool_k_plan_on() {
            pf_expert_mode(lw0)
        } else {
            iq_expert_mode(lw0)
        };
        let dev_ok = lm_dev_moe_ok(
            nt_all,
            crate::model::exp_pf::pf_enabled(),
            expert_mode,
            crate::model::exp_pf::stream_enabled(),
            crate::model::exp_pf::stream_min_tok(),
            pf_router_gpu_on(),
            lw0.gate_inp.k() == self.p.n_embd,
        );
        if dev_ok {
            crate::model::pf_pool::pf_pool_reserve_bytes()
        } else {
            crate::model::pf_pool::pf_pool_reserve_cpu_bytes()
        }
    }

    /// task 227: 層優先の常駐 prefill (`DENNO_PF_LM_LAYER=1`)。
    ///
    /// 部品はチャンク優先の常駐連鎖と同じで、順序だけが違う。層 L について
    /// 1. 全チャンクの mix / attn / ssm を回して ffn の入力を集め (`mixed_all`)、
    /// 2. **MoE を層で 1 回**だけ回し、
    /// 3. 全チャンクの combine を回す。
    ///
    /// これで expert 重みは 1 プロンプトで 1 回しか H2D されない (チャンク優先は
    /// チャンク数ぶん = 14k で 9 パス、158.5 GB)。MoE の入力は
    /// `n_tokens x e` がホストに載り、残差は層のあいだデバイスの退避域に置く
    /// (D2D。ホスト経由だと 14k で 110 GB 往復するので t170a の二の舞になる)。
    ///
    /// 前提が足りなければ `Ok(None)` を返し、呼び出し側がチャンク優先へ落ちる。
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_prefill_batch_resident_lm(
        &mut self,
        ctx: &[u32],
        chunk: usize,
        tokens: TapTokens,
        mut on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        mut roles: Option<&mut RoleRms>,
        iv: IvView<'_>,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let hp = self.hp();
        let e = self.p.n_embd;
        let hc = hp.hc;
        let lr = hp.low_rank;
        let hc_dim = hc * e;
        let n_layer = self.p.n_layer;
        let chunk = chunk.max(1);
        let nt_all = ctx.len();
        // 1 チャンクならチャンク優先と同じ (層優先にする意味が無い)。
        if nt_all.div_ceil(chunk) < 2 {
            return Ok(None);
        }
        // task 257/265: 作業域のプールをこのリクエストの形で組む (長い prefill の
        // あいだだけ expert から借りる)。ガードはこの関数のスコープに置くので、
        // どの早期撤退でも終端で返る。
        // task 279: 借りられる量から (チャンク, タイル) を選ぶ。**大きい形は
        // プールを実際に借りられたときだけ**使う (借りられないと作業域が組めず、
        // t270a のように streaming が黙って落ちる)。落ちたら従来の形へ。
        let (need, pool_cap, big, reserve) = self.pf_pool_pick(nt_all, chunk, hc, lr);
        let _pool = crate::cuda::pf_pool_acquire_capped(&need, pool_cap, reserve);
        let big = big && _pool.is_held();
        let (chunk, tile) = if _pool.is_held() {
            (need.chunk.max(1), need.tile)
        } else {
            (chunk, crate::model::exp_pf::lm_tile(nt_all))
        };
        let ranges: Vec<(usize, usize)> = (0..nt_all)
            .step_by(chunk)
            .map(|s| (s, (s + chunk).min(nt_all)))
            .collect();
        if !crate::cuda::pf_begin(chunk.min(nt_all), hc, e, lr) {
            Self::pf_note_decline("作業域/PTX/GPU の前提外");
            return Ok(None);
        }
        // 全チャンクの残差の退避域 (D2D)。
        if !crate::cuda::pf_res_park_ensure(ranges.len(), chunk, hc, e) {
            crate::cuda::pf_release();
            Self::pf_note_decline("残差の退避域を確保できない");
            return Ok(None);
        }
        let steer_state = steer::snapshot();
        let steer_apply = steer_state.apply.as_ref();
        let steer_rec = steer::rec_on();
        // task 849: 重ね掛け (rules) は 1 本の合成ベクトルに畳めない (規則ごとに
        // 層範囲 / alpha / mode が違う)。GPU の常駐経路は使わずホスト経路へ落とす
        // (値は同じ、速度だけ落ちる)。単一規則は従来どおり。
        let steer_dev_apply = steer_apply
            .map(|a| a.is_single_rule() && a.from < n_layer && a.to > 0)
            .unwrap_or(false);
        if steer_rec && !crate::cuda::pf_steer_acc_ensure(n_layer, hc, e) {
            crate::cuda::pf_release();
            Self::pf_note_decline("steer の加算域が使えない");
            return Ok(None);
        }
        if steer_dev_apply {
            let a = steer_apply.expect("steer_dev_apply は apply があるときだけ真");
            if !crate::cuda::pf_steer_apply_begin(&a.v, n_layer, hc, e) {
                crate::cuda::pf_release();
                Self::pf_note_decline("steer の v をデバイスへ置けない");
                return Ok(None);
            }
        }
        let _mixed_mark = PfMixedMarkGuard;
        // task 233: 層優先は全トークンの活性をデバイスに置くので、置き場
        // (mixed + ffn + inject) と MoE の作業域 (h / g / u / act) のぶんだけ
        // 先に空ける。**足りない分だけ**追い出す (task 227 の 6 GB 固定は
        // 14k 後の decode を 36 ms/tok まで落とした)。
        // task 242: 追い出した鍵を控えておき、prefill の終端で戻す
        // (戻さないと次の decode のヒット率が 0.75 → 0.25 に落ちる)。
        let mut evicted: Vec<crate::model::expert_cache::ExpertKey> = Vec::new();
        // task 250: 作業域の外部予約 (`DENNO_PF_LM_RESERVE=1`)。層ループの間だけ持つ。
        let mut reserve_guard: Option<crate::model::expert_vram::ExternalReserve> = None;
        // task 419: 層ループ入口の確保と追い出し階段を分けて数える (診断)。
        let sub_setup = q4ledger::sub(q4ledger::SUB_PF_SETUP);
        {
            let k = self.p.n_expert_used;
            let ff = self.p.n_ff_exp;
            // task 250: tensor core 経路は gate/up を融合して act に直接書くので、
            // 作業域は act 1 枚で足りる (g/u は取らない)。見積りも 3 → 1 に落とす
            // (過大に見積もると、要らない expert まで追い出して decode を冷やす)。
            let act_terms = if crate::model::exp_pf::tc_exp_on() {
                1
            } else {
                3
            };
            // task 250: MoE をトークンで割るときは h / act もその行数で足りる。
            // task 279: 行数は選んだタイル (プールのレイアウトと同じ値) を使う。
            let tile_rows = tile.saturating_mul(k);
            let act_rows = if act_terms == 1 {
                tile_rows
            } else if crate::model::exp_pf::lm_group_local() {
                tile_rows.div_ceil(4).max(1)
            } else {
                tile_rows
            };
            let need = nt_all * e * 8
                + ranges.len() * chunk * hc * 4
                + (tile_rows + nt_all) * e * 4
                + act_terms * act_rows * ff * 4
                + (64usize << 20);
            // lab t242b: この `need` を満たしても、14k では `h` (1,505 MiB) の
            // 確保が「空き 1,243 MiB」で失敗し、`DevBuffer::alloc_tagged` の
            // 追い出し階段が **expert の LRU を全部捨てる** (39,755 MiB)。
            // expert は slab のチャンクが空になるまで実メモリを返さないので、
            // 少しずつ追い出しても 1.5 GB の連続領域は空かない。
            // task 250: `DENNO_PF_LM_TILE` で MoE をトークン分割すると `h` が
            // 1/分割数になり、この連続確保が通る (下の tile_rows)。
            // task 250: 作業域を expert 予算からの**外部予約**として取る版
            // (`DENNO_PF_LM_RESERVE=1`)。予約は prefill の終端で返すので、LRU の
            // 上限が作業域のぶんだけ下がって戻る (確保のたびに追い出し階段が
            // 走って上限が縮んだままになるのを避ける)。
            if crate::model::exp_pf::lm_reserve_on() {
                if let Some(r) = crate::model::expert_vram::reserve_external(need) {
                    eprintln!(
                        "[pf-lm] 作業域 {} MiB を予算から予約した (会計={})",
                        need >> 20,
                        r.is_guarded()
                    );
                    reserve_guard = Some(r);
                }
            }
            if reserve_guard.is_none() {
                let reserve = 1024usize << 20;
                let short = match crate::cuda::dev_mem_mb_cached() {
                    Some((free_mb, _)) => {
                        need.saturating_sub((free_mb << 20).saturating_sub(reserve))
                    }
                    None => need,
                };
                let want = short.min(6usize << 30);
                if want > (64 << 20) {
                    let (freed, keys) = crate::model::expert_vram::emergency_free_keys(want);
                    if freed > 0 {
                        eprintln!(
                            "[pf-lm] expert を {} MiB 追い出して活性と作業域を空けた (必要 {} MiB)",
                            freed >> 20,
                            need >> 20
                        );
                    }
                    evicted = keys;
                }
            }
        }
        if !crate::cuda::pf_lm_ensure(nt_all, e, hc, chunk) {
            crate::cuda::pf_release();
            Self::pf_note_decline("活性の置き場を確保できない");
            return Ok(None);
        }
        drop(sub_setup);
        let ple_batch = ple_batch_enabled();
        // 埋め込みと PLE は層ループの外で 1 回だけ (全トークン)。
        // task 415: 埋め込みは**行独立**なので N 本で引ける (`DENNO_PF_EMBED_PAR`)。
        // 1 本あたり 2,100 B のランダム読みなので、表がページキャッシュから落ちて
        // いると 1 トークン 100 us の待ちが直列に並ぶ (層ループの前 = GPU は遊ぶ)。
        let embed_par = crate::model::exp_pf::embed_par();
        let mut res_all: Vec<f32> = vec![0.0; nt_all * hc_dim];
        // task 349: `DENNO_PF_PLE_PIPE` では埋め込み (直列) と gather を分け、
        // gather を広い専用プールで引く。冷えた page cache では 1 行 90 B の
        // ランダム pread が IOPS 律速 (14k で 1,848 ms、wall の 20%) なので、
        // queue を深くする。読む行・綴じ方は同じ。
        let ple_wide = ple_batch && crate::model::exp_pf::ple_pipe();
        // task 415: `DENNO_PF_EMBED_OVL` では gather を背景スレッドへ出し、その裏で
        // 埋め込みを引く (どちらも同じファイルの待ちなので重ねられる)。
        let ple_ovl = ple_wide && embed_par > 1 && crate::model::exp_pf::embed_ovl();
        // task 423: gather を**チャンクごとに**受け取れる形にする。背景スレッドは
        // チャンクの順に送り、受け手は層ループの PLE 層で 1 チャンクずつ受ける
        // (`DENNO_PF_PLE_LJOIN`)。off のときは層ループの前に全部受ける (従来と同じ)。
        type PleChunk = Result<Vec<Option<Vec<f32>>>>;
        let mut ple_rx: Option<std::sync::mpsc::Receiver<PleChunk>> = None;
        let ljoin = ple_ovl && crate::model::exp_pf::ple_ljoin();
        if ple_ovl {
            let c: Vec<u32> = ctx.to_vec();
            let rt = self.ple.as_ref().map(std::sync::Arc::clone);
            let chunk_list: Vec<(usize, usize)> = ranges.to_vec();
            let threads = crate::model::exp_pf::ple_threads().max(1);
            let (tx, rx) = std::sync::mpsc::channel::<PleChunk>();
            ple_rx = Some(rx);
            // ハンドルは持たない (受け手はチャネルで待つ。落ちたら送信が失敗して止まる)。
            let _ple_join = std::thread::spawn(move || {
                let run = || {
                    for &(start, end) in &chunk_list {
                        let r = gather_ple_many_rt(rt.as_deref(), &c[..end], start);
                        let bad = r.is_err();
                        // 受け手が居ない (早期撤退) なら止める。
                        if tx.send(r).is_err() || bad {
                            return;
                        }
                    }
                };
                // 専用プールの本数がそのまま同時に飛ぶ pread の数になる (task 357)。
                match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                    Ok(p) => p.install(run),
                    Err(_) => run(),
                }
            });
        }
        let mut ples_all: Vec<Vec<Option<Vec<f32>>>> = Vec::with_capacity(ranges.len());
        for &(start, end) in &ranges {
            let t_emb = prekprof::stamp();
            if embed_par > 1 {
                // 行が独立なので N 本で引く (値は直列の `forward_embed` と一致)。
                crate::model::linear_gguf::embed_rows_into(
                    &self.tok_embd,
                    &ctx[start..end],
                    hc,
                    &mut res_all[start * hc_dim..end * hc_dim],
                    embed_par,
                )
                .map_err(|e| one(format!("埋め込みの並列引きに失敗: {e:#}")))?;
            } else {
                for i in start..end {
                    let r = self.forward_embed(&ctx[..=i], &hp)?;
                    let o = start * hc_dim + (i - start) * hc_dim;
                    res_all[o..o + hc_dim].copy_from_slice(&r.data);
                }
            }
            prekprof::mark(t_emb, prekprof::EMBED);
            if !ple_wide {
                let t_ple = prekprof::stamp();
                let ples: Vec<Option<Vec<f32>>> = if ple_batch {
                    self.gather_ple_many(&ctx[..end], start)?
                } else {
                    let mut v: Vec<Option<Vec<f32>>> = Vec::with_capacity(end - start);
                    for i in start..end {
                        v.push(self.gather_ple(&ctx[..=i])?);
                    }
                    v
                };
                prekprof::mark(t_ple, prekprof::PLE);
                ples_all.push(ples);
            }
        }
        if ple_wide && !ljoin {
            // gather を全部受けてから層ループへ (従来と同じ)。
            // `ple_par_parts` が `DENNO_PF_PLE_THREADS` (既定 128) と
            // `current_num_threads()` の小さいほうを見る (task 357 / 419)。
            let t_ple = prekprof::stamp();
            match ple_rx.take() {
                // task 415: 背景で読み終わっている (埋め込みと重ねた)。
                Some(rx) => {
                    for _ in 0..ranges.len() {
                        ples_all
                            .push(rx.recv().map_err(|_| one("PLE の先読みスレッドが落ちた"))??);
                    }
                }
                None => {
                    let ple_threads = crate::model::exp_pf::ple_threads().max(1);
                    let rt = self.ple.as_ref().map(std::sync::Arc::clone);
                    let chunk_list: Vec<(usize, usize)> = ranges.to_vec();
                    let run = || -> Result<Vec<Vec<Option<Vec<f32>>>>> {
                        let mut out: Vec<Vec<Option<Vec<f32>>>> =
                            Vec::with_capacity(chunk_list.len());
                        for &(start, end) in &chunk_list {
                            out.push(gather_ple_many_rt(rt.as_deref(), &ctx[..end], start)?);
                        }
                        Ok(out)
                    };
                    ples_all = match rayon::ThreadPoolBuilder::new()
                        .num_threads(ple_threads)
                        .build()
                    {
                        Ok(p) => p.install(run)?,
                        Err(_) => run()?,
                    };
                }
            }
            prekprof::mark(t_ple, prekprof::PLE);
        }
        // task 233: デバイス経路を諦めたときだけ要るホストの控え (普段は確保しない)。
        let mut mixed_buf: Option<PfHostBuf> = None;
        // 診断: ホスト経路へ落ちた層の数 (`DENNO_PF_LM_DBG=1` で 1 行)。
        let mut lm_fallback = 0usize;
        // task 233: attention の入口は CPU が読むので、チャンクぶんだけホストへ戻す。
        let mut attn_buf = PfHostBuf::new(chunk * e);
        // task 394: 次の層の expert 重みを先にページキャッシュへ載せる
        // (`DENNO_PF_LAYER_PRE`)。層優先の MoE は routing が決まるまで読めないので、
        // attention (phase 1) のあいだディスクが遊ぶ。3,826 の実測では 1 層 =
        // 296 本 (層の全数) が要るので、予測ではなく全数を 1 層先に読む。
        let layer_pre_on = crate::model::exp_pf::layer_pre();
        if layer_pre_on {
            pf_layer_prefetch(&self.layers[0]);
        }
        for il in 0..n_layer {
            let lw = &self.layers[il];
            if layer_pre_on && il + 1 < n_layer {
                pf_layer_prefetch(&self.layers[il + 1]);
            }
            if q4prof::enabled() {
                q4prof::N_LAYER.fetch_add(nt_all as u64, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(d) = roles.as_deref_mut() {
                *d = RoleRms::default();
            }
            let t_lay = prekprof::stamp();
            // ---- 1. 全チャンクの mix / attn / ssm ----
            // task 461: 直前のチャンクの PLE をデバイスで回したか (履歴の上げ直しと
            // 同期の回数を減らすため。層ごとにリセットする)。
            let mut ple_dev_chain = false;
            for (ci, &(start, end)) in ranges.iter().enumerate() {
                let nt = end - start;
                // task 419: 1 回転のうち名前付き区間に入らない時間を測る (診断)。
                let t_iter = q4ledger::iter_begin();
                // task 482: `other` の残差 (14k で 672 ms が名前の無い区間) を割る。
                let _s_itop = q4ledger::sub(q4ledger::SUB_PF_ITTOP);
                // 残差をデバイスへ (層 0 はホストの初期値、それ以降は退避域から)。
                // task 553: `iter_top` のうち残差の復帰を分けて数える。
                {
                    let _s_ires = q4ledger::sub(q4ledger::SUB_PF_ITRES);
                    if il == 0 {
                        if !crate::cuda::pf_upload_res(&res_all[start * hc_dim..end * hc_dim]) {
                            Self::pf_bail("res の H2D に失敗")?;
                        }
                    } else if !crate::cuda::pf_res_park_load(ci, nt, hc, e) {
                        Self::pf_bail("res の退避域からの復帰に失敗")?;
                    }
                }
                // task 218: ステアリング (層の入口)。
                if steer_dev_apply && steer_apply.map(|a| a.covers(il)).unwrap_or(false) {
                    let a = steer_apply.expect("covers が真なら apply がある");
                    let mode = if a.mode == steer::Mode::Project { 1u8 } else { 0u8 };
                    if !crate::cuda::pf_steer_apply(nt, hc, e, il, mode, a.alpha) {
                        Self::pf_bail("steer の適用に失敗")?;
                    }
                }
                // 1.1 PLE 注入 (層 1 のみ)。ホストで回すので、この層だけ往復する。
                if let Some(pw) = &lw.ple {
                    // task 423: `DENNO_PF_PLE_LJOIN` では背景の gather を**この層で
                    // 1 チャンクずつ**受ける (層 0 の 1 層ぶんが gather と重なる)。
                    // 待ちは台帳の `ple` に積む。チャネルはチャンク順なので ci と揃う。
                    if let Some(rx) = ple_rx.as_ref() {
                        let t_ple = prekprof::stamp();
                        // task 553: `iter_top` のうち PLE の gather 待ちを分けて数える。
                        let chunk_rows = {
                            let _s_iple = q4ledger::sub(q4ledger::SUB_PF_ITPLE);
                            rx.recv().map_err(|_| one("PLE の先読みスレッドが落ちた"))??
                        };
                        prekprof::mark(t_ple, prekprof::PLE);
                        ples_all.push(chunk_rows);
                    }
                    // task 461: PLE の注入をデバイスで閉じる (`DENNO_PF_PLE_DEV`)。
                    //
                    // 従来は res を D2H → ホストでトークンループ → H2D していた
                    // (14k で 1 層チャンクあたり res 82 MB x 2 + key/value の D2H)。
                    // res / key / value は元からデバイスにあるので、2 本の要素
                    // カーネル (gate / conv+silu) で閉じる。形が外れたら従来経路。
                    let mut ple_dev_done = false;
                    if crate::cuda::pf_ple_dev_on() {
                        let ples = &ples_all[ci];
                        let kern = pw.kern;
                        let dil = self.p.ple_ngram_size;
                        let hist_len = kern.saturating_sub(1) * dil;
                        let rows_ok = ples.len() >= nt
                            && (0..nt).all(|t| {
                                ples[t]
                                    .as_deref()
                                    .map(|r| r.len() == e)
                                    .unwrap_or(false)
                            });
                        if rows_ok && crate::cuda::pf_ple_dev_ok(nt, hc, e, kern, dil, hist_len) {
                            // task 505: 平坦化バッファを使い回す (`DENNO_PF_PLE_FLAT_REUSE`)。
                            // 従来は層チャンクごとに `Vec::with_capacity` で新しい
                            // (nt * e * 4 B = 14k で 21 MB) を確保していたので、
                            // 初回タッチのフォルトと munmap を層チャンクの数だけ払った。
                            let reuse = pf_ple_flat_reuse_on();
                            let mut flat = if reuse {
                                PF_PLE_FLAT.with(|c| {
                                    let mut v = c.borrow_mut();
                                    let mut b = std::mem::take(&mut *v);
                                    b.clear();
                                    if b.capacity() < nt * e {
                                        b.reserve(nt * e - b.capacity());
                                    }
                                    b
                                })
                            } else {
                                Vec::with_capacity(nt * e)
                            };
                            for t in 0..nt {
                                flat.extend_from_slice(ples[t].as_deref().expect("rows_ok"));
                            }
                            // 寄与 RMS (役割テレメトリ) は層の最後のチャンクの値だけが
                            // 残る (`r.ple` はチャンクごとに上書き) ので、そこでだけ取る。
                            let last_chunk = ci + 1 == ranges.len();
                            let want = roles.is_some() && last_chunk;
                            let hist = self.ple_conv_state[il]
                                .as_mut()
                                .ok_or_else(|| one(format!("層 {il}: PLE の conv 履歴が無い")))?;
                            let args = crate::cuda::PfPleDevArgs {
                                flat: &flat,
                                nt,
                                hc,
                                e,
                                key: Self::pf_lin(&pw.key),
                                value: Self::pf_lin(&pw.value),
                                gk: &pw.norm_key[..],
                                gq: &pw.norm_query[..],
                                gc: &pw.norm_conv[..],
                                wt: &pw.conv1d_t[..],
                                hist_host: hist.as_mut_slice(),
                                hist_len,
                                upload_hist: !ple_dev_chain,
                                flush_hist: last_chunk,
                                kern,
                                dil,
                                li: il,
                                eps: hp.eps,
                                want,
                            };
                            match crate::cuda::pf_ple_dev_run(args) {
                                Ok(Some(out)) => {
                                    if want {
                                        if let Some(r) = roles.as_deref_mut() {
                                            r.ple = out.contrib_rms;
                                        }
                                    }
                                    ple_dev_done = true;
                                    ple_dev_chain = true;
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    return Err(one(format!(
                                        "層 {il}: PLE のデバイス経路: {e}"
                                    )))
                                }
                            }
                            // task 505: バッファを返す (次の層チャンクが同じ容量を使い回す)。
                            if reuse {
                                PF_PLE_FLAT.with(|c| {
                                    let mut v = c.borrow_mut();
                                    if v.capacity() < flat.capacity() {
                                        *v = flat;
                                    }
                                });
                            }
                        }
                    }
                    if ple_dev_done {
                        // res はデバイスで更新済み (ホストの控えは触らない)。
                    } else {
                    let (mut back, back_reused) = pf_ple_back_take(nt * hc_dim);
                    if !crate::cuda::pf_download_res(&mut back) {
                        Self::pf_bail("PLE 層の res の D2H に失敗")?;
                    }
                    let ples = &ples_all[ci];
                    let mut rows: Vec<&[f32]> = Vec::with_capacity(nt);
                    for t in 0..nt {
                        rows.push(
                            ples[t]
                                .as_deref()
                                .ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?,
                        );
                    }
                    let mut contrib = 0.0f32;
                    let want = roles.is_some();
                    let hist = self.ple_conv_state[il]
                        .as_mut()
                        .ok_or_else(|| one(format!("層 {il}: PLE の conv 履歴が無い")))?;
                    if pf_ple_fast_on() {
                        if nt >= PLE_PAR_MIN_TOK {
                            ple_inject_batch_par(
                                &self.p,
                                pw,
                                &rows,
                                &mut back,
                                &hp,
                                hist,
                                want.then_some(&mut contrib),
                            )?;
                        } else {
                            ple_inject_batch_flat(
                                &self.p,
                                pw,
                                &rows,
                                &mut back,
                                &hp,
                                hist,
                                want.then_some(&mut contrib),
                            )?;
                        }
                    } else {
                        let mut streams: Vec<HyperStreams> = (0..nt)
                            .map(|t| HyperStreams {
                                data: back[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                                hc,
                                n_embd: e,
                            })
                            .collect();
                        ple_inject_batch(
                            &self.p,
                            pw,
                            &rows,
                            &mut streams,
                            &hp,
                            hist,
                            want.then_some(&mut contrib),
                        )?;
                        for t in 0..nt {
                            back[t * hc_dim..(t + 1) * hc_dim]
                                .copy_from_slice(&streams[t].data);
                        }
                    }
                    if want {
                        if let Some(r) = roles.as_deref_mut() {
                            r.ple = contrib;
                        }
                    }
                    if !crate::cuda::pf_upload_res(&back) {
                        Self::pf_bail("PLE 層の res の H2D に失敗")?;
                    }
                    // task 300: 往復バッファを返す。
                    pf_ple_back_return(back, back_reused);
                    }
                }
                // 1.2 attention 入口 (デバイス mix) → attn → combine
                drop(_s_itop);
                let t_mix = std::time::Instant::now();
                let flat_attn = pf_attn_flat_on();
                {
                    let _l = q4ledger::enter(q4ledger::MIX);
                    if !crate::cuda::pf_mix(
                        &lw.hc_attn_norm,
                        Self::pf_lin(&lw.hc_attn_down),
                        Self::pf_lin(&lw.hc_attn_up),
                        Self::pf_lin(&lw.hc_attn_inject),
                        nt,
                        hc,
                        e,
                        lr,
                        hp.eps,
                    ) {
                        Self::pf_bail("attn mix の連鎖に失敗")?;
                    }
                    let mixed = &mut attn_buf.as_mut_slice()[..nt * e];
                    // task 317: full 層 (task 230) と ssm 層 (`MIXKEEP_SSM`) は
                    // 直後の attn がデバイスの `mixed` を読めるので D2H しない。
                    if Self::pf_mix_keep_layer(flat_attn, nt, self.p.layer_kind(il).ok()) {
                        crate::cuda::pf_mixed_keep_dev(mixed.as_ptr(), nt, e);
                    } else {
                        if !crate::cuda::pf_take_mixed(mixed) {
                            Self::pf_bail("attn mix の mixed D2H に失敗")?;
                        }
                        crate::cuda::pf_mixed_mark(mixed.as_ptr(), nt, e);
                    }
                }
                q4prof::add(&q4prof::MIX, t_mix);
                let mixed_attn: &[f32] = &attn_buf.as_slice()[..nt * e];
                // task 419: attn 入口の切り替え (誰の区間でもない) を分けて数える。
                let sub_attnpre = q4ledger::sub(q4ledger::SUB_PF_ATTPRE);
                let attn_dev = flat_attn && pf_attn_devres_on() && roles.is_none();
                // task 482: `curs` の組み立て (トークンごとの `Vec` 確保) は
                // 平坦化していないときだけ走る。`other` の残差の候補。
                let _s_apre = q4ledger::sub(q4ledger::SUB_PF_ITAPRE);
                let curs: Vec<Vec<f32>> = if flat_attn {
                    Vec::new()
                } else {
                    mixed_attn.chunks_exact(e).map(|c| c.to_vec()).collect()
                };
                drop(_s_apre);
                let t_attn = std::time::Instant::now();
                crate::cuda::pf_attn_dev_clear();
                crate::cuda::pf_attn_dev_request(attn_dev);
                drop(sub_attnpre);
                let flat_outs = {
                    let item = if q4ledger::enabled() {
                        match self.p.layer_kind(il) {
                            Ok(LayerKind::FullAttn) => q4ledger::FULL,
                            _ => q4ledger::SSM,
                        }
                    } else {
                        q4ledger::OTHER
                    };
                    let _l = q4ledger::enter(item);
                    // task 379: 層の最終チャンクかを GDN 連鎖へ伝える
                    // (`DENNO_PF_GDN_WB_LAST` は最終チャンクだけ状態を書き戻す)。
                    // 直後に必ず真へ戻す (途中で失敗しても次の層が拾わない)。
                    crate::cuda::pf_gdn_last_chunk_set(end == nt_all);
                    let r = Self::prefill_attn_chunk(
                        self.attn.as_mut(),
                        il,
                        mixed_attn,
                        &curs,
                        nt,
                        e,
                        flat_attn,
                        attn_dev,
                    );
                    crate::cuda::pf_gdn_last_chunk_reset();
                    crate::cuda::pf_attn_dev_request(false);
                    r?
                };
                q4prof::add(&q4prof::ATTN, t_attn);
                let dev_out = flat_outs.is_empty();
                if let Some(r) = roles.as_deref_mut() {
                    if !dev_out && end == nt_all {
                        let t = nt - 1;
                        let v = rms_of(&flat_outs[t * e..(t + 1) * e]);
                        if matches!(self.p.layer_kind(il), Ok(LayerKind::FullAttn)) {
                            r.attn = v;
                        } else {
                            r.gdn = v;
                        }
                    }
                }
                let combined = if dev_out {
                    crate::cuda::pf_combine_dev(nt, hc, e)
                } else {
                    crate::cuda::pf_combine(&flat_outs, nt, hc, e)
                };
                if !combined {
                    Self::pf_bail("attn の combine に失敗")?;
                }
                // 1.3 ffn 入口 (デバイス mix) まで。MoE は層でまとめて回す。
                let t_mix = std::time::Instant::now();
                {
                    let _l = q4ledger::enter(q4ledger::MIX);
                    if !crate::cuda::pf_mix(
                        &lw.hc_ffn_norm,
                        Self::pf_lin(&lw.hc_ffn_down),
                        Self::pf_lin(&lw.hc_ffn_up),
                        Self::pf_lin(&lw.hc_ffn_inject),
                        nt,
                        hc,
                        e,
                        lr,
                        hp.eps,
                    ) {
                        Self::pf_bail("ffn mix の連鎖に失敗")?;
                    }
                    // task 233: ホストへ戻さず、置き場 (全トークンぶん) へ D2D で写す。
                    if !crate::cuda::pf_lm_park_mixed(start, nt, e) {
                        Self::pf_bail("ffn mix の mixed を置き場へ写せない")?;
                    }
                    // task 233: inject もチャンクごとに退避する (combine が読む。
                    // task 227 は最後のチャンクの inject を全チャンクに使っていた)。
                    if !crate::cuda::pf_lm_inj_save(ci, nt, hc) {
                        Self::pf_bail("ffn mix の inject を退避できない")?;
                    }
                }
                q4prof::add(&q4prof::MIX, t_mix);
                // 残差を退避域へ (次のチャンクが res を使う)。
                if !crate::cuda::pf_res_park_save(ci, nt, hc, e) {
                    Self::pf_bail("res の退避に失敗")?;
                }
                // task 470: 1 チャンク目の routing をここで解き、残りのチャンクの
                // attention が走るあいだに expert の重みを先へ読む
                // (`DENNO_PF_EARLY_ROUTE`、既定 off)。この時点で `mixed` の
                // 先頭 `nt` 行は MoE 入力として確定している (pf_lm_park_mixed)。
                #[cfg(feature = "cuda")]
                if ci == 0 {
                    // task 482: 早期 routing の先読みは名前付き区間の外で走る
                    // (14k で層 1 回 = 48 回)。`other` の残差の最有力候補。
                    let _s_er = q4ledger::sub(q4ledger::SUB_PF_ER);
                    let _ = pf_early_route_prefetch(
                        lw,
                        nt,
                        nt_all,
                        e,
                        self.p.n_expert_used,
                        self.p.n_ff_exp,
                        self.p.n_expert,
                        iv,
                    );
                }
                q4ledger::iter_end(t_iter);
            }
            // task 474: ここから先はディスクが主役なので、先読みの実行役は止める
            // (窓 = 残りチャンクの attention は終わった)。
            crate::model::exp_pf::rc_stop();
            // ---- 2. 層で 1 回の MoE (expert 重みはここでしか運ばれない) ----
            let t_moe = std::time::Instant::now();
            // task 233: 入力も出力もデバイスの置き場のまま回す。ホストへ戻るのは
            // router の確率だけ。前提が欠けたら置き場をホストへ戻して従来経路へ。
            // task 250: トークンタイル (`DENNO_PF_LM_TILE`)。MoE の作業域
            // (h) はタイルの行数で取るので、14k でも 1.5 GB の連続確保が要らない。
            // 分割しないときは 1 回 (従来と同じ経路)。
            // task 279: プールが借りられたときは選んだタイル (パス数が最小) を使う。
            let mut ffn_dev = true;
            // task 318: 台帳の `experts` 区間に載せる。フォールバック
            // (`moe_ffn_quant_batch_tapped`) は内部で同じ区間を開くので、ここだけ
            // 無いとデバイス経路の MoE が `other` に混ざって読めなくなる。
            if tile >= nt_all {
                let _l = q4ledger::enter(q4ledger::EXPERTS);
                ffn_dev = moe_ffn_quant_batch_lm_dev(
                    &self.p,
                    lw,
                    nt_all,
                    roles.as_deref_mut(),
                    true,
                    iv,
                )?;
            } else {
                let mut t0 = 0usize;
                while t0 < nt_all {
                    let nt = tile.min(nt_all - t0);
                    let last = t0 + nt == nt_all;
                    let ok = {
                        let _l = q4ledger::enter(q4ledger::EXPERTS);
                        moe_ffn_quant_batch_lm_dev_at(
                            &self.p,
                            lw,
                            t0,
                            nt_all,
                            nt,
                            roles.as_deref_mut(),
                            last,
                            iv,
                        )?
                    };
                    if !ok {
                        ffn_dev = false;
                        break;
                    }
                    t0 += nt;
                }
            }
            if !ffn_dev {
                // task 257: フォールバックは**タイル単位**で回す。nt_all で一気に回すと
                // 作業域 (14k で h 1.5 GB) がプールに収まらず、確保失敗 → ストリーム
                // 停止 → 以後すべて CPU で 209 s になる (lab t257a の実測)。
                if lm_fallback == 0 {
                    eprintln!(
                        "[pf-lm] 層 {il}: dev MoE を諦めた (理由: {}{}) → CPU (タイル {} 単位)",
                        lm_decline_reason(),
                        lm_decline_detail(),
                        tile
                    );
                }
                lm_fallback += 1;
                let buf = mixed_buf.get_or_insert_with(|| PfHostBuf::new(nt_all * e));
                if !crate::cuda::pf_lm_take_mixed(&mut buf.as_mut_slice()[..nt_all * e]) {
                    Self::pf_bail("mixed の D2H に失敗")?;
                }
                let mixed_all: &[f32] = &buf.as_slice()[..nt_all * e];
                let mut ffn_all = vec![0.0f32; nt_all * e];
                let mut t0 = 0usize;
                while t0 < nt_all {
                    let nt = tile.min(nt_all - t0);
                    let last = t0 + nt == nt_all;
                    let ffn = moe_ffn_quant_batch_tapped(
                        &self.p,
                        lw,
                        &mixed_all[t0 * e..(t0 + nt) * e],
                        nt,
                        self.expert_cache.as_ref(),
                        roles.as_deref_mut(),
                        last,
                        iv,
                    )?;
                    if ffn.len() != nt * e {
                        return Err(one(format!(
                            "層 {il}: MoE の出力長 {} が nt {nt} * n_embd {e} と違う",
                            ffn.len()
                        )));
                    }
                    ffn_all[t0 * e..(t0 + nt) * e].copy_from_slice(&ffn);
                    t0 += nt;
                }
                if !crate::cuda::pf_lm_upload_ffn(&ffn_all) {
                    Self::pf_bail("MoE の出力を置き場へ上げられない")?;
                }
            }
            q4prof::add(&q4prof::MOE, t_moe);
            // ---- 3. 全チャンクの combine ----
            for (ci, &(start, end)) in ranges.iter().enumerate() {
                let nt = end - start;
                if !crate::cuda::pf_res_park_load(ci, nt, hc, e) {
                    Self::pf_bail("res の退避域からの復帰に失敗")?;
                }
                // task 233: このチャンクの inject を戻してから、MoE の出力
                // (デバイスの置き場) をそのまま combine する。
                if !crate::cuda::pf_lm_inj_load(ci, nt, hc) {
                    Self::pf_bail("ffn mix の inject を復帰できない")?;
                }
                let base = crate::cuda::pf_lm_ffn_ptr(nt_all, e)
                    .ok_or_else(|| one("MoE の出力の置き場が無い"))?;
                if !crate::cuda::pf_combine_at(base + (start * e * 4) as u64, nt, hc, e) {
                    Self::pf_bail("ffn の combine に失敗")?;
                }
                // task 218: このチャンクの残差をデバイスで加算する。
                // task 829: LastPromptToken は最終チャンクの最後の行だけ。
                if steer_rec {
                    let last_only = steer::last_only();
                    let final_chunk = end == nt_all;
                    if !last_only || final_chunk {
                        let ok = if last_only {
                            crate::cuda::pf_steer_acc_last(nt, hc, e, il)
                        } else {
                            crate::cuda::pf_steer_acc(nt, hc, e, il)
                        };
                        if !ok {
                            Self::pf_bail("steer の記録に失敗")?;
                        }
                        if last_only {
                            steer::note_dev_tokens(1);
                        } else {
                            steer::note_dev_tokens(nt as u64);
                        }
                    }
                }
                if end == nt_all {
                    if let Some(r) = roles.as_deref() {
                        emit_role_act(il, r);
                    }
                    if let Some(h) = on_layer.as_deref_mut() {
                        let t = nt - 1;
                        let mut row = vec![0.0f32; hc_dim];
                        if !crate::cuda::pf_download_res_row(t, hc_dim, &mut row) {
                            Self::pf_bail("層の出口フック用の res D2H に失敗")?;
                        }
                        h(&LayerOut {
                            il,
                            t,
                            nt,
                            n_embd: e,
                            res: &row,
                            roles: roles.as_deref(),
                        })?;
                    }
                }
                if !crate::cuda::pf_res_park_save(ci, nt, hc, e) {
                    Self::pf_bail("res の退避に失敗")?;
                }
            }
            prekprof::mark(t_lay, prekprof::LAYER);
        }
        // task 553: 層ループの終端で、後ろへ回した KV の書き戻しを降ろす
        // (最後の層チャンクの保留を decode や次のリクエストへ持ち越さない)。
        self.attn.as_mut().flush_deferred()?;
        // 最終チャンクの残差をホストへ (最終 mixer 用)。
        let (lstart, lend) = *ranges.last().expect("ranges は空でない");
        let lnt = lend - lstart;
        let mut res_flat = vec![0.0f32; lnt * hc_dim];
        if !crate::cuda::pf_download_res(&mut res_flat) {
            Self::pf_bail("最終チャンクの res の D2H に失敗")?;
        }
        if std::env::var_os("DENNO_PF_LM_DBG").is_some() {
            eprintln!("[pf-lm-dbg] layers={n_layer} fallback={lm_fallback}");
        }
        // task 233: 置き場と MoE の作業域 (14k で合計 約 3.5 GB) を decode に返す。
        // task 419: 後片付け (解放・返却・追い出した expert の戻し) を分けて数える。
        let sub_td = q4ledger::sub(q4ledger::SUB_PF_TEARDOWN);
        crate::cuda::pf_lm_release();
        crate::cuda::pf_scratch_drop();
        crate::cuda::pf_release();
        // task 265: 借りた作業域を expert に返す (ここでカードが expert に戻る)。
        drop(_pool);
        // task 279: 大きい形で借りたときは、その確保が追い出した expert を LRU へ
        // 戻す (借りた作業域だけでなく、代わりに追い出した在庫も返す)。
        // MoE はもう完了している (末尾の D2H が同期) ので、t265b のように
        // 実行中のカーネルと衝突しない。本数は借りた量に比例して有界。
        // task 290: 借りられなかったときも、余白のために先に空けた expert が
        // 残っているので戻す (借りた/借りないに関わらず同じ経路で返す)。
        let keys = crate::cuda::pf_pool_take_evicted();
        if !keys.is_empty() {
            let (n, bytes) = pf_restore_experts(&self.layers, &keys);
            eprintln!(
                "[pf-pool] {}追い出した expert を {} 本 ({} MiB) 戻した ({} 本は戻せず)",
                if big { "大きい形で" } else { "余白の確保で" },
                n,
                bytes >> 20,
                keys.len().saturating_sub(n)
            );
        }
        // task 250: 作業域の外部予約を返す (LRU の上限がここで戻る)。
        drop(reserve_guard);
        // task 242: 作業域のために追い出した expert を戻す (prefill の終端)。
        // ここまで decode の LRU は 2 GB 以上痩せていて、次の decode のヒット率が
        // 0.75 → 0.25 に落ちる (lab t233a)。追い出した順 (古い順) に戻すので、
        // 相対的な新しさも元に近づく。1 本 2 MB × 約 1,100 本で 0.1 秒程度。
        if !evicted.is_empty() && crate::model::exp_pf::lm_reload_evicted() {
            let (n, bytes) = pf_restore_experts(&self.layers, &evicted);
            if n > 0 {
                eprintln!(
                    "[pf-lm] 追い出した expert を {} 本 ({} MiB) 戻した ({} 本は戻せず)",
                    n,
                    bytes >> 20,
                    evicted.len().saturating_sub(n)
                );
            }
        }
        if steer_rec {
            steer::merge_device();
        }
        drop(sub_td);
        let mut ress: Vec<HyperStreams> = Vec::with_capacity(lnt);
        for t in 0..lnt {
            ress.push(HyperStreams {
                data: res_flat[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                hc,
                n_embd: e,
            });
        }
        let res = ress.pop().ok_or_else(|| one("ctx が空 (現在のトークンが無い)"))?;
        let t_mix = headprof::stamp();
        let t_fin = prekprof::stamp();
        let cur = hc_mix_quant(
            &res,
            &self.head.norm,
            &self.head.down,
            &self.head.up,
            None,
            &hp,
        )?
        .0;
        headprof::mark(headprof::MIX, t_mix, 0);
        prekprof::mark(t_fin, prekprof::FINAL_NORM);
        let mut logits = vec![0.0f32; self.head.output.n()];
        let t_head = prekprof::stamp();
        self.head_logits_into(&cur, &mut logits)?;
        prekprof::mark(t_head, prekprof::HEAD);
        Ok(Some(vec![logits]))
    }

    /// cuda feature 無しでは常に従来経路 (`Ok(None)`)。
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn forward_prefill_batch_resident(
        &mut self,
        _ctx: &[u32],
        _chunk: usize,
        _tokens: TapTokens,
        _on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        _roles: Option<&mut RoleRms>,
        _iv: IvView<'_>,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(None)
    }

    /// 層優先のバッチ prefill の本体 (P109 で `collect_all` を追加)。
    ///
    /// `collect_all == true` は全位置の logits を返す (検証列の照合用)。
    /// `false` は従来どおり最後の 1 位置だけ (既存の呼び出し側はすべてこちら)。
    /// それ以外の引数と処理順序は P108 までの
    /// [`Self::forward_prefill_batch_tapped`] とまったく同じ (ビット不変)。
    pub(crate) fn forward_prefill_batch_core(
        &mut self,
        ctx: &[u32],
        chunk: usize,
        tokens: TapTokens,
        mut on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        roles: Option<&mut RoleRms>,
        collect_all: bool,
        hist: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // P109: `hist` は「ctx の先頭 hist 個は PLE の窓を作るためだけで、
        // 食わない」ことを示す (検証列の照合用。既存の呼び出し側は 0)。
        if ctx.len() <= hist {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        #[cfg(test)]
        PREFILL_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // P58: arch テレメトリが有効なら、呼び出し側が roles を渡していなくても
        // 役割 RMS を取る (RoleAct を送るため)。読むだけなので logits は変わらない。
        let mut auto_roles = RoleRms::default();
        let mut roles: Option<&mut RoleRms> = if roles.is_none() && arch_frames_enabled() {
            Some(&mut auto_roles)
        } else {
            roles
        };
        let hp = self.hp();
        let e = self.p.n_embd;
        let n_layer = self.p.n_layer;
        let chunk = chunk.max(1);
        // task 788: 常駐経路 (>= PF_RESIDENT_MIN_TOK) に載る大きさの最初のチャンクは、
        // **どの PTX ローダより先に** CUDA context を用意して current にする。
        // 冷えた起動直後は `shared_ctx()` がまだ None で、prefill の
        // `cuModuleLoadData` (pf_kernels / tc_gemm_kernels / exp_tc_kernels /
        // pf_elem_kernels) が CUDA_ERROR_INVALID_CONTEXT で失敗し、その None が
        // `OnceLock` に焼き付いてサーバの寿命ぶん GPU prefill が死んでいた。
        // 小さい prefill は従来どおり decode 側の遅延初期化に任せる。
        #[cfg(feature = "cuda")]
        {
            let first = chunk.min(ctx.len().saturating_sub(hist));
            if Self::pf_init_ctx_first(first) {
                let _ = crate::cuda::make_current_on_this_thread();
            }
        }
        // task 217: 介入設定は **prefill 1 回の先頭で 1 回だけ**読む。
        let iv_cfg = intervene::snapshot();
        let iv = IvView::of(&iv_cfg);
        // task 218: 残差の平均の記録 / ステアリング (ホスト経路)。
        let steer_state = steer::snapshot();
        let steer_apply = steer_state.apply.as_ref();
        let steer_rec = steer::rec_on();
        // task 154: hc mix / combine のデバイス常駐 (`DENNO_PF_RESIDENT`)。
        // 観測フック付き (`on_layer`) と全位置 logits (`collect_all`) は
        // 従来経路だけが受ける (常駐経路は層の出口をホストへ出さない)。
        // 層の出口フックは `TapTokens::Last` だけ受ける (残差 1 行を D2H する)。
        let pf_hook_ok = on_layer.is_none() || matches!(tokens, TapTokens::Last);
        if pf_mix_dev_on() {
            // 入口の状態を 1 回だけ出す (台帳だけでは「不発」の理由が読めないため)。
            static ONCE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[pf] resident 入口: hook={} all={} ctx={} chunk={} gate={}",
                    on_layer.is_some(),
                    matches!(tokens, TapTokens::All),
                    ctx.len(),
                    chunk,
                    pf_hook_ok
                );
            }
        }
        // task 217: 常駐連鎖 (mix / combine / attn / PLE をデバイスで閉じる) は、
        //   そのどれかを切る設定のときだけ使わない。expert / shared の無効化は
        //   ホスト側の MoE で処理できるので連鎖のまま回す (prefill の速度を保つ)。
        if pf_mix_dev_on() && pf_hook_ok && !collect_all && iv.resident_safe() {
            // `as_deref_mut` だと外側の寿命に固定されて 2 回目で E0499 になる
            // (`prefill_seq` と同じく短い寿命で借り直す)。
            let hook: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>> = match &mut on_layer {
                Some(f) => Some(&mut **f),
                None => None,
            };
            if let Some(out) = self.forward_prefill_batch_resident(
                ctx,
                chunk,
                tokens,
                hook,
                roles.as_deref_mut(),
                iv,
            )? {
                return Ok(out);
            }
        }
        let mut last: Option<HyperStreams> = None;
        // P109: collect_all は全位置の最終残差を保持する (検証列の各位置の logits)。
        let mut all_ress: Vec<HyperStreams> = Vec::new();
        let mut start = hist;
        if start >= ctx.len() {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        // mix の要素演算の作業域 (実形で xn/gate 20 MB ずつ)。層ループの外で 1 回だけ持つ。
        let mut mx = MixScratch::default();
        // P78: チャンクぶんの PLE を行単位でまとめて引くか (既定 on)。
        let ple_batch = ple_batch_enabled();
        while start < ctx.len() {
            let end = (start + chunk).min(ctx.len());
            let nt = end - start;
            let mut ress: Vec<HyperStreams> = Vec::with_capacity(nt);
            // P78: embedding と PLE は層の外。区間を分けて積む。
            let t_emb = prekprof::stamp();
            for i in start..end {
                ress.push(self.forward_embed(&ctx[..=i], &hp)?);
            }
            prekprof::mark(t_emb, prekprof::EMBED);
            let t_ple = prekprof::stamp();
            let f0 = if t_ple.is_some() {
                prekprof::faults()
            } else {
                None
            };
            // P78: PLE の行 gather はチャンクぶんまとめて引く (既定 on、
            // `DENNO_PLE_BATCH=0` で 1 トークンずつの旧経路)。値はビット一致。
            let ples: Vec<Option<Vec<f32>>> = if ple_batch {
                self.gather_ple_many(&ctx[..end], start)?
            } else {
                let mut v: Vec<Option<Vec<f32>>> = Vec::with_capacity(nt);
                for i in start..end {
                    v.push(self.gather_ple(&ctx[..=i])?);
                }
                v
            };
            prekprof::mark(t_ple, prekprof::PLE);
            if t_ple.is_some() {
                prekprof::note_fault_delta(f0);
            }

            let t_lay = prekprof::stamp();
            for il in 0..n_layer {
                let lw = &self.layers[il];
                if q4prof::enabled() {
                    q4prof::N_LAYER.fetch_add(nt as u64, std::sync::atomic::Ordering::Relaxed);
                }
                // P54: 役割別 RMS は層ごとに 0 から取り直す (観測しないときは None)。
                if let Some(d) = roles.as_deref_mut() {
                    *d = RoleRms::default();
                }

                // task 218: ステアリングは層の入口 (層 `il-1` の出口と同じ点)。
                if let Some(a) = steer_apply {
                    for t in 0..nt {
                        a.apply_one(il, &mut ress[t].data);
                    }
                }

                // 3.1 PLE 注入 (層 1 のみ)。conv 履歴はトークン順に進める。
                // task 217: `ple` block を切った層では注入を計算しない。
                if let Some(pw) = &lw.ple {
                    if !iv.block_off(il, IvBlock::Ple) {
                        for t in 0..nt {
                            let pe = ples[t]
                                .as_ref()
                                .ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?;
                            let hist = self.ple_conv_state[il]
                                .as_mut()
                                .ok_or_else(|| one(format!("層 {il}: PLE の conv 履歴が無い")))?;
                            // P54: 寄与の RMS はチャンク末尾のトークンだけ取る。
                            let mut contrib = 0.0f32;
                            let want = roles.is_some() && t + 1 == nt;
                            ress[t] = ple_inject(
                                &self.p,
                                pw,
                                pe,
                                &ress[t],
                                &hp,
                                hist,
                                want.then_some(&mut contrib),
                            )?;
                            if want {
                                if let Some(r) = roles.as_deref_mut() {
                                    r.ple = contrib;
                                }
                            }
                        }
                    }
                }

                // task 217: `attn` block を切った層は mix / attn / combine を飛ばす。
                if !iv.block_off(il, IvBlock::Attn) {
                    let t_mix = std::time::Instant::now();
                    let (curs, injs) = {
                        // task 145: 台帳の区間 (mix)。
                        let _l = q4ledger::enter(q4ledger::MIX);
                        hc_mix_quant_batch(
                            &ress,
                            &lw.hc_attn_norm,
                            &lw.hc_attn_down,
                            &lw.hc_attn_up,
                            Some(&lw.hc_attn_inject),
                            &hp,
                            &mut mx,
                        )?
                    };
                    q4prof::add(&q4prof::MIX, t_mix);
                    let t_attn = std::time::Instant::now();
                    // 投影をチャンクでまとめる実装 (ssm 層) はここで GPU 往復を減らす。
                    let outs = {
                        // task 145: 台帳の区間。層種別で full / ssm に分ける。
                        let item = if q4ledger::enabled() {
                            match self.p.layer_kind(il) {
                                Ok(LayerKind::FullAttn) => q4ledger::FULL,
                                _ => q4ledger::SSM,
                            }
                        } else {
                            q4ledger::OTHER
                        };
                        let _l = q4ledger::enter(item);
                        self.attn.forward_batch(il, &curs)?
                    };
                    for o in &outs {
                        if o.len() != e {
                            return Err(one(format!(
                                "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                                o.len()
                            )));
                        }
                    }
                    q4prof::add(&q4prof::ATTN, t_attn);
                    if outs.len() != nt {
                        return Err(one(format!(
                            "層 {il}: attention の出力本数 {} が nt {nt} と違う",
                            outs.len()
                        )));
                    }
                    let flat_outs = flatten_rows(&outs, e)?;
                    // P54: attn / gdn の RMS はチャンク末尾のトークンの出力で取る。
                    if let Some(r) = roles.as_deref_mut() {
                        let t = nt - 1;
                        let v = rms_of(&flat_outs[t * e..(t + 1) * e]);
                        if matches!(self.p.layer_kind(il), Ok(LayerKind::FullAttn)) {
                            r.attn = v;
                        } else {
                            r.gdn = v;
                        }
                    }
                    hyper::hc_combine_batch_masked(
                        &mut ress,
                        &flat_outs,
                        &injs,
                        &hp,
                        iv.stream_keep(il, hp.hc),
                    );

                }

                // task 217: routed と shared の両方を切った層は mix / combine も飛ばす。
                let skip_routed = iv.block_off(il, IvBlock::Moe);
                let skip_shared = iv.block_off(il, IvBlock::Shared);
                if !(skip_routed && skip_shared) {
                    let t_mix = std::time::Instant::now();
                    let (curs, injs) = {
                        // task 145: 台帳の区間 (mix)。
                        let _l = q4ledger::enter(q4ledger::MIX);
                        hc_mix_quant_batch(
                            &ress,
                            &lw.hc_ffn_norm,
                            &lw.hc_ffn_down,
                            &lw.hc_ffn_up,
                            Some(&lw.hc_ffn_inject),
                            &hp,
                            &mut mx,
                        )?
                    };
                    q4prof::add(&q4prof::MIX, t_mix);
                    let flat = flatten_rows(&curs, e)?;
                    let t_moe = std::time::Instant::now();
                    let ffn = moe_ffn_quant_batch_tapped(
                        &self.p,
                        lw,
                        &flat,
                        nt,
                        self.expert_cache.as_ref(),
                        roles.as_deref_mut(),
                        true,
                        iv,
                    )?;
                    q4prof::add(&q4prof::MOE, t_moe);
                    if ffn.len() != nt * e {
                        return Err(one(format!(
                            "層 {il}: MoE の出力長 {} が nt {nt} * n_embd {e} と違う",
                            ffn.len()
                        )));
                    }
                    hyper::hc_combine_batch_masked(
                        &mut ress,
                        &ffn,
                        &injs,
                        &hp,
                        iv.stream_keep(il, hp.hc),
                    );

                }

                // task 218: 層の出口 (2 つの combine を終えた後) の残差を記録する。
                // task 829: LastPromptToken は最終チャンクの最後の行だけ。
                if steer_rec {
                    let rows: Vec<&[f32]> = (0..nt).map(|t| ress[t].data.as_slice()).collect();
                    steer::acc_host_rows(il, &rows, end == ctx.len());
                }

                // P58: 層の役割別活性 (kind 13)。prefill は**最後のチャンクの末尾**
                // トークンだけ送る (チャンクは既定 8 トークンなので、全チャンクで
                // 送ると 層数 × チャンク数 枚になってチャネル (4096) が溢れる)。
                // RoleRms 自体もチャンク末尾のトークンのぶんしか持たない。
                if end == ctx.len() {
                    if let Some(r) = roles.as_deref() {
                        emit_role_act(il, r);
                    }
                }

                // P53: 層の出口 (残差を足した後) を観測フックに渡す。読むだけ。
                // P54: 役割別 RMS はチャンク末尾のトークンのぶんしか無いので、
                // 他のトークンには載せない (嘘の値を送らない)。
                if let Some(h) = on_layer.as_deref_mut() {
                    match tokens {
                        TapTokens::All => {
                            for t in 0..nt {
                                let is_last = t + 1 == nt;
                                h(&LayerOut {
                                    il,
                                    t,
                                    nt,
                                    n_embd: e,
                                    res: &ress[t].data,
                                    roles: if is_last { roles.as_deref() } else { None },
                                })?;
                            }
                        }
                        TapTokens::Last => {
                            let t = nt - 1;
                            h(&LayerOut {
                                il,
                                t,
                                nt,
                                n_embd: e,
                                res: &ress[t].data,
                                roles: roles.as_deref(),
                            })?;
                        }
                    }
                }
            }
            prekprof::mark(t_lay, prekprof::LAYER);

            // 次のチャンクは自分の埋め込みから始まる (残差の持ち越しは無い)。
            if collect_all {
                all_ress.append(&mut ress);
            } else {
                last = ress.pop();
            }
            start = end;
        }

        // P42: grouped GPU 経路の失敗は「最初の 1 行 + ここの集計 1 行」に畳む
        // (0 件なら何も出さない)。
        crate::model::moe_grouped_gpu::flush_failures();

        // P49: `DENNO_PF_ATT_DBG=1` のときだけ attn の層別内訳を 1 層 1 行で出す
        // (層 0 / 23 = ssm、47 = full attention)。既定 off では何もしない。
        if q4prof::att_dbg() {
            for &l in q4prof::DBG_LAYERS.iter() {
                let is_full = matches!(self.p.layer_kind(l), Ok(LayerKind::FullAttn));
                if let Some(line) = q4prof::layer_line(l, is_full) {
                    eprintln!("{line}");
                }
            }
        }

        // 4. 最終 mixer (output_norm の代わり) + 5. logits
        if collect_all {
            // P109: 全位置の logits (検証列の照合用)。演算は最後の 1 位置と
            // 同じ (hc_mix_quant + head GEMV) を位置の数だけ繰り返す。
            let mut out = Vec::with_capacity(all_ress.len());
            for res in &all_ress {
                let t_fin = prekprof::stamp();
                let cur = hc_mix_quant(res, &self.head.norm, &self.head.down, &self.head.up, None, &hp)?.0;
                prekprof::mark(t_fin, prekprof::FINAL_NORM);
                let mut logits = vec![0.0f32; self.head.output.n()];
                let t_head = prekprof::stamp();
                self.head_logits_into(&cur, &mut logits)?;
                prekprof::mark(t_head, prekprof::HEAD);
                out.push(logits);
            }
            return Ok(out);
        }
        let res = last.ok_or_else(|| one("ctx が空 (現在のトークンが無い)"))?;
        let t_mix = headprof::stamp();
        let t_fin = prekprof::stamp();
        let cur = hc_mix_quant(
            &res,
            &self.head.norm,
            &self.head.down,
            &self.head.up,
            None,
            &hp,
        )?
        .0;
        headprof::mark(headprof::MIX, t_mix, 0);
        prekprof::mark(t_fin, prekprof::FINAL_NORM);
        let mut logits = vec![0.0f32; self.head.output.n()];
        let t_head = prekprof::stamp();
        self.head_logits_into(&cur, &mut logits)?;
        prekprof::mark(t_head, prekprof::HEAD);
        // P109: 本体は全位置の形で返す (従来経路は最後の 1 位置だけ)。
        Ok(vec![logits])
    }
}


#[cfg(all(test, feature = "cuda"))]
mod p788_tests {
    use super::*;

    /// task 788 (バグ 1): 冷えた起動直後の最初のリクエストで、どの plen が
    /// 「PTX を読む前に context を作る」側に入るか。境目は 64。
    /// 512 / 506 / 64 は入り、63 / 28 は入らない (小さい prefill は従来どおり
    /// decode 側の遅延初期化に任せる)。
    #[test]
    fn pf788_ctx_init_boundary() {
        for nt in [64usize, 65, 67, 127, 128, 129, 506, 512, 1536, 8192] {
            assert!(
                Qwen4Exp::pf_init_ctx_first(nt),
                "nt={nt} が従来経路に落ちている"
            );
        }
        for nt in [0usize, 1, 28, 63] {
            assert!(
                !Qwen4Exp::pf_init_ctx_first(nt),
                "nt={nt} で context を作っている"
            );
        }
        // 常駐経路の判定と同じ境界であること (2 か所でずれない)。
        assert!(Qwen4Exp::pf_init_ctx_first(Qwen4Exp::PF_RESIDENT_MIN_TOK));
        assert!(!Qwen4Exp::pf_init_ctx_first(
            Qwen4Exp::PF_RESIDENT_MIN_TOK - 1
        ));
    }
}
