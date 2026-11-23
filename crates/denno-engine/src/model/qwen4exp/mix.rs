//! hc mix (mix の GEMM・要素演算) と PLE 注入。
//!
//! task 668: `src/model/qwen4exp.rs` から純粋に切り出した (ロジック変更なし)。

use super::*;

/// `DENNO_MIX_FAST` の解釈。**既定 on**。`0` / `false` / `off` のときだけ旧経路。
///
/// 旧経路は呼び出しごとに `vec![0.0; ..]` を確保し、`hc_mixed` を 1 スレッドで
/// 回す。新経路は作業域を使い回し、`hc_mixed` を `d` 方向に rayon で分ける
/// (どちらも演算順序は同じ = ビット一致)。
pub(crate) fn mix_fast_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => {
            !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off"))
        }
    }
}

/// `DENNO_MIX_FAST` を hc mix に適用するか (既定 on)。
pub(crate) fn mix_fast_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mix_fast_from_value(std::env::var("DENNO_MIX_FAST").ok().as_deref()))
}

/// `DENNO_HC_MIX_FAST` の解釈。**既定 on**。`0` / `false` / `off` のときだけ
/// P84 より前の経路 (classic カーネル + 従来の入口) に戻る。
///
/// P84 は `hc_down` / `hc_up` の Q8_0 融合カーネルを 2 行同時版
/// ([`moe_expert::expert_gemv_q8_fast_2row`]) に切り替える (値はビット一致、
/// 行ごとの累積順序は不変)。実機で退行したときの逃げ道。
pub(crate) fn hc_mix_fast_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mix_fast_from_value(std::env::var("DENNO_HC_MIX_FAST").ok().as_deref()))
}

/// task 110: `DENNO_MIX_GPU` の解釈。**既定 off**。`1` / `on` / `true` / `yes`
/// (`trim` + 小文字化。`router_gpu_from_value` と同一規約) のとき hc mix の
/// BF16 down / up を GPU (`cuda::maybe_gpu_mix_bf16_into` = `dno_matvec_f32`
/// 流用・task 121 から長い行 k >= n は並列リダクション版 `dno_matvec_f32_r`)
/// で回す。失敗時は cuda 側が warn 1 行で CPU に落ちる (短い行の値はビット一致)。
pub(crate) fn mix_gpu_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => false,
    }
}

/// task 110: hc mix GPU 経路のテスト用上書き (0 = env を見る / 1 = 強制 off /
/// 2 = 強制 on)。`ROUTER_GPU_OVERRIDE` と同じ 3 点セット。
pub(crate) static MIX_GPU_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// task 110: hc mix の down / up を GPU で回してよいか (`DENNO_MIX_GPU`、既定 off)。
/// プロセスで 1 回だけ読む。GPU 無しビルドでは呼び出し側が cfg(feature = "cuda")
/// の下にあるので dead になるが、判定関数そのものはテストから同じ形で叩ける。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn mix_gpu_enabled() -> bool {
    match MIX_GPU_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mix_gpu_from_value(std::env::var("DENNO_MIX_GPU").ok().as_deref()))
}

/// task 261: `DENNO_MIX_QUANT_GPU` — hc mix の down / up を **旧来の 4 型**
/// (Q8_0 / Q5_1 / Q4_K / Q5_K) でも GPU で回す。既定 off。
///
/// swift15 (mix = BF16) は既存の BF16 経路 (`DENNO_MIX_GPU` + chain) がそのまま
/// 使える。UD-Q4_K_XL のような **Q8_0 の mix** を持つモデルでは、このゲートが
/// 無いと mix が丸ごと CPU に落ちる (実測は REPORT_P261.md)。カーネルは
/// dense 重みレジストリ + `moe_gemv_dev` / `moe_gemm_dev` をそのまま使う。
pub const MIX_QUANT_GPU_ENV: &str = "DENNO_MIX_QUANT_GPU";

/// [`MIX_QUANT_GPU_ENV`] の解釈 (純関数)。`1` / `true` / `on` だけが真。
pub fn mix_quant_gpu_from_value(v: Option<&str>) -> bool {
    crate::model::expert_vram::flag_on(v)
}

/// mix の量子型を GPU で回すか (プロセスで 1 回だけ env を読む)。既定 off。
pub(crate) fn mix_quant_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_quant_gpu_from_value(std::env::var(MIX_QUANT_GPU_ENV).ok().as_deref())
    })
}

pub const MIX_QCHAIN_ENV: &str = "DENNO_MIX_QCHAIN";

/// [`MIX_QCHAIN_ENV`] の解釈 (純関数)。**既定 on** (task 282 で採用)。
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけが off。
pub fn mix_qchain_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// 量子型 mix の fused chain を使うか (プロセスで 1 回だけ env を読む)。既定 on。
pub(crate) fn mix_qchain_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_qchain_from_value(std::env::var(MIX_QCHAIN_ENV).ok().as_deref())
    })
}

/// gate_audit の台帳用 (既定 on のゲート)。`DENNO_MIX_QCHAIN` が有効か。
pub fn mix_qchain_gate_on() -> bool {
    mix_qchain_enabled()
}

// ===========================================================================
// task 484: hc inject の GEMV を「行数 4 の F32 FMA カーネル」にする
// ===========================================================================
//
// lab t484a (現 int-auto、既定モデル、`[dec-kprof-avg]` の mix band) で、
// mix のホスト仕事は **mix_inj 16.1 us/回** が最大だった (`mix_norm` 3.3、
// `mix_h2d_pack` 0.8)。97 呼び出し/token なので 1.56 ms/token = decode 11.1 の
// 14%。中身は `hc_inject` の GEMV で、実形は **F32 の 4 行 x 10,240**
// (`gemv inj ... ty=-` は F32 = id 0。`[mix-kprof]` の `inj 137 ms / 9506 回`)。
// 従来は 1 行ずつの逐次和 (`gemv_scalar_body`) で、dev の実測でも 16.7 us。
//
// 4 行を同時に回して x の 1 ブロックを全行で共有する FMA 版は 1.00 us で、
// 差は 2.3e-6 (|ref| 1.0) = 丸めだけ。lab t484b (base vs inj4、2 ラウンド交互) は
// 短い decode 11.1 -> 10.8 (-2.7%)、c2000_cold 13.4 -> 13.0 (-3.0%)、
// c2000_turn2 12.2 -> 11.8 (-3.3%)、c8000_cold 13.0 -> 13.1 (+0.8%、ノイズ幅)、
// prefill 0.58 -> 0.57、md5 全 SAME、errors 0 だったので **既定 on** にした。
// 切り戻しは `DENNO_MIX_INJ_R4=0`。
//
// lab の COND が新しい env 名を落とす回があるので、既存の `DENNO_MIX_QK` の値に
// `inj4` が含まれていても on にする (逃げ道。値の解釈は dn7 などのまま)。

/// [`MIX_INJ_R4_ENV`] の解釈 (純関数)。**既定 on** (task 484 で採否 yes)。
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけが off。
/// `qk` は `DENNO_MIX_QK` の生値で、値に `inj4` が入っていれば on
/// (lab の COND が新しい env 名を落とす回の逃げ道。明示 off より強い)。
pub fn mix_inj_r4_from_value(v: Option<&str>, qk: Option<&str>) -> bool {
    if qk
        .map(|s| s.to_ascii_lowercase().contains("inj4"))
        .unwrap_or(false)
    {
        return true;
    }
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// task 484: hc inject を 4 行同時の F32 FMA GEMV で回すか (**既定 on**)。
pub fn mix_inj_r4_on() -> bool {
    #[cfg(test)]
    if let Some(v) = MIX_INJ_R4_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_inj_r4_from_value(
            std::env::var(MIX_INJ_R4_ENV).ok().as_deref(),
            std::env::var("DENNO_MIX_QK").ok().as_deref(),
        )
    })
}

/// task 484: ゲートの env 名。
pub const MIX_INJ_R4_ENV: &str = "DENNO_MIX_INJ_R4";

#[cfg(test)]
thread_local! {
    pub(crate) static MIX_INJ_R4_FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// テスト用: ゲートの強制 (`None` = env の解釈)。
#[cfg(test)]
pub(crate) fn mix_inj_r4_set_for_test(v: Option<bool>) {
    MIX_INJ_R4_FORCE.with(|c| c.set(v));
}

/// task 484: 4 行 FMA 経路で inject を計算した回数 (発火の証拠)。
pub(crate) static MIX_INJ_R4_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`MIX_INJ_R4_HITS`] を読む (`[mix-qchain]` の `inj4=` に出る)。
pub fn mix_inj_r4_hits() -> u64 {
    MIX_INJ_R4_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// 計数を 0 に戻す (リクエスト終端)。
pub fn mix_inj_r4_reset() {
    MIX_INJ_R4_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// task 266: mix が呼ばれた回数 (ゲート on のときだけ数える。1 プロセス通算)。
pub(crate) static MIX_QCHAIN_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 266: 量子型 mix の fused chain が走った回数 (1 プロセス通算)。
/// ゲート on で 0 なら「分岐に入っていない」ことが分かる (診断行 [mix-qchain])。
pub(crate) static MIX_QCHAIN_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 266: `[mix-qchain]` の 1 行。ゲート off / 0 回なら `None`。
pub fn mix_qchain_line() -> Option<String> {
    if !mix_qchain_enabled() {
        return None;
    }
    let n = MIX_QCHAIN_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let calls = MIX_QCHAIN_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    // task 282: ゲートが既定 on になったので、**1 度も呼ばれていない**モデル
    // (mix が BF16 の既定モデルなど) では行を出さない (ログを 1 バイトも変えない)。
    if calls == 0 {
        return None;
    }
    // task 282: 高速 GEMV (`DENNO_MIX_QK`) が実際に撃たれた回数も出す。
    #[cfg(feature = "cuda")]
    let qk = crate::cuda::mix_qk_hits();
    #[cfg(not(feature = "cuda"))]
    let qk = 0u64;
    // task 364: ルータ GEMV を鎖の末尾に積んだ回数 (task 307 の
    // `DENNO_ROUTER_IN_MIX` を Q8_0 の qchain にも広げた)。0 なら載っていない。
    #[cfg(feature = "cuda")]
    let rtr = crate::cuda::router_in_mix_qchain_hits();
    #[cfg(not(feature = "cuda"))]
    let rtr = 0u64;
    // task 418: down を新カーネル (1 ブロック 1 行) で撃った回数。0 なら v5。
    #[cfg(feature = "cuda")]
    let dn5 = crate::cuda::mix_dn5_hits();
    #[cfg(not(feature = "cuda"))]
    let dn5 = 0u64;
    // task 422: down を「1 ブロック rows 行」の dn6 で撃った回数。0 なら dn5/v5。
    #[cfg(feature = "cuda")]
    let dn6 = crate::cuda::mix_dn6_hits();
    #[cfg(not(feature = "cuda"))]
    let dn6 = 0u64;
    // task 427: dn6 を mix 以外 (full 層の qkv/wo = f6、head = h6) で撃った回数。
    #[cfg(feature = "cuda")]
    let (f6, h6) = crate::cuda::dn6_dec_hits();
    #[cfg(not(feature = "cuda"))]
    let (f6, h6) = (0u64, 0u64);
    // task 496: GDN の qkv をレーン詰めの dn8 で撃った回数 (0 なら dn6/v5)。
    #[cfg(feature = "cuda")]
    let dn8 = crate::cuda::mix_dn8_hits();
    #[cfg(not(feature = "cuda"))]
    let dn8 = 0u64;
    // task 431: down を 1 対 4 レーンの dn7 で撃った回数 (0 なら dn6/dn5/v5)。
    #[cfg(feature = "cuda")]
    let dn7 = crate::cuda::mix_dn7_hits();
    #[cfg(not(feature = "cuda"))]
    let dn7 = 0u64;
    // task 504: dn7 を mix 以外 (full 層 = f7、GDN の out = o7) で撃った回数。
    #[cfg(feature = "cuda")]
    let (f7, o7) = crate::cuda::dn7_dec_hits();
    #[cfg(not(feature = "cuda"))]
    let (f7, o7) = (0u64, 0u64);
    // task 504: head を dn7 で撃った回数。
    #[cfg(feature = "cuda")]
    let h7 = crate::cuda::dn7_head_hits();
    #[cfg(not(feature = "cuda"))]
    let h7 = 0u64;
    // task 431: xn の H2D を別ストリームへ出した回数 (`DENNO_MIX_ZC=stream`)。
    // 0 なら従来の null ストリーム (発火の証拠)。
    #[cfg(feature = "cuda")]
    let h2s = crate::cuda::mix_h2s_hits();
    #[cfg(not(feature = "cuda"))]
    let h2s = 0u64;
    // task 484: inject を 4 行同時の F32 FMA GEMV で計算した回数
    // (`DENNO_MIX_INJ_R4`)。0 なら従来の 1 行ずつの逐次和。
    let inj4 = mix_inj_r4_hits();
    Some(format!(
        "[mix-qchain] on calls={calls} hits={n} frac={:.3} qk={qk} rtr={rtr} dn5={dn5} dn6={dn6} dn7={dn7} dn8={dn8} f6={f6} h6={h6} f7={f7} o7={o7} h7={h7} h2s={h2s} inj4={inj4}",
        if calls > 0 { n as f64 / calls as f64 } else { 0.0 }
    ))
}

/// task 266: 集計を 0 に戻す (リクエスト終端)。
pub fn mix_qchain_reset() {
    MIX_QCHAIN_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    MIX_QCHAIN_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    // task 282: 高速 GEMV の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_qk_reset();
    // task 418: dn5 の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_dn5_reset();
    // task 422: dn6 の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_dn6_reset();
    // task 496: dn8 の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_dn8_reset();
    // task 427: full 層 / head の dn6 計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::dn6_dec_reset();
    // task 364: ルータを鎖に積んだ回数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::router_in_mix_qchain_reset();
    // task 431: 別ストリーム H2D の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_h2s_reset();
    // task 431: dn7 の計数も同じ区切りで 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::mix_dn7_reset();
    // task 504: dn7 を full 層 / GDN の out で撃った計数も 0 に戻す。
    #[cfg(feature = "cuda")]
    crate::cuda::dn7_dec_reset();
    // task 484: inject の 4 行 FMA 経路の計数も同じ区切りで 0 に戻す。
    mix_inj_r4_reset();
}

#[cfg(test)]
mod mix_qchain_tests {
    use super::*;

    /// task 484: `DENNO_MIX_INJ_R4` の真理値表。**既定 on** (lab t484b で採否 yes) で、
    /// 明示的な偽値だけ off。新しい env 名が lab の COND で落ちる回の逃げ道
    /// (`DENNO_MIX_QK` の値に `inj4`) も固定する。
    #[test]
    fn mix_inj_r4_gate_truth_table() {
        assert!(mix_inj_r4_from_value(None, None), "未設定は既定 on");
        assert!(mix_inj_r4_from_value(Some(""), None), "空は既定 on");
        assert!(!mix_inj_r4_from_value(Some("0"), None));
        assert!(!mix_inj_r4_from_value(Some(" off "), None));
        assert!(!mix_inj_r4_from_value(Some("false"), None));
        assert!(mix_inj_r4_from_value(Some("1"), None));
        assert!(mix_inj_r4_from_value(Some("on"), None));
        assert!(mix_inj_r4_from_value(Some("R4"), None));
        // 逃げ道: 既存 env の値に inj4 (値の解釈は dn7 などのまま)。
        assert!(mix_inj_r4_from_value(None, Some("dn7,inj4")));
        assert!(mix_inj_r4_from_value(None, Some("INJ4")));
        assert!(mix_inj_r4_from_value(None, Some("dn7")), "既定 on なので dn7 でも on");
        assert!(mix_inj_r4_from_value(Some("0"), Some("dn7,inj4")), "逃げ道は明示 off でも効く");
    }

    /// task 484: ゲート on で 4 行 FMA 経路が**実際に発火**し、値は従来の
    /// 逐次和と同じ (差は丸めだけ) ことを固定する。hc inject の実形に近い
    /// 形 (k = 10,240 は重いので k = 1,024 / hc = 4) で見る。
    #[test]
    fn mix_inj_r4_fires_and_matches_classic() {
        if !crate::model::moe_expert::q4k_fused_supported() {
            eprintln!("skip mix_inj_r4_fires_and_matches_classic: AVX2/FMA が無い");
            return;
        }
        let k = 1024usize;
        for n in [1usize, 2, 3, 4] {
            let mut blocks = Vec::with_capacity(k * n * 4);
            for r in 0..n {
                for i in 0..k {
                    let v = (((i * 31 + r * 7) % 251) as f32 - 125.0) * 0.001;
                    blocks.extend_from_slice(&v.to_le_bytes());
                }
            }
            let w = GgufLinear::new(
                crate::kquant::GgmlType::F32,
                k,
                n,
                crate::quant::WeightBlocks::from_vec(blocks),
            )
            .expect("GgufLinear");
            let x: Vec<f32> = (0..k)
                .map(|i| (((i * 37) % 89) as f32 - 44.0) * 0.013)
                .collect();
            let mut classic = vec![0.0f32; n];
            w.gemv(&x, &mut classic).expect("classic");

            let before = mix_inj_r4_hits();
            let mut fast = vec![0.0f32; n];
            mix_inj_r4_set_for_test(Some(true));
            let fired = inject_gemv_r4(&w, &x, &mut fast);
            mix_inj_r4_set_for_test(None);
            assert!(fired, "n={n}: 行 FMA 経路が使われていない (発火の証拠が無い)");
            assert!(
                mix_inj_r4_hits() > before,
                "n={n}: 計数が増えていない (発火の証拠が無い)"
            );
            for i in 0..n {
                let d = (classic[i] as f64 - fast[i] as f64).abs();
                let tol = 1e-4 + 1e-5 * classic[i].abs() as f64;
                assert!(
                    d <= tol,
                    "n={n} 行 {i}: classic {} vs FMA {} (差 {d:.3e} > {tol:.3e})",
                    classic[i],
                    fast[i]
                );
            }
            // ゲート off なら何も書かない (従来経路がそのまま走る)。
            let mut off = vec![7.0f32; n];
            mix_inj_r4_set_for_test(Some(false));
            assert!(!inject_gemv_r4(&w, &x, &mut off), "ゲート off で発火した");
            mix_inj_r4_set_for_test(None);
            assert_eq!(off, vec![7.0f32; n], "ゲート off なのに書いた");
        }
        // 形が合わない (n > 4) は何も書かずに false = 従来経路。
        let (k5, n5) = (1024usize, 5usize);
        let blocks5 = vec![0u8; k5 * n5 * 4];
        let w5 = GgufLinear::new(
            crate::kquant::GgmlType::F32,
            k5,
            n5,
            crate::quant::WeightBlocks::from_vec(blocks5),
        )
        .expect("GgufLinear");
        let x5 = vec![0.5f32; k5];
        let mut o5 = vec![7.0f32; n5];
        mix_inj_r4_set_for_test(Some(true));
        assert!(!inject_gemv_r4(&w5, &x5, &mut o5), "n=5 で発火した");
        mix_inj_r4_set_for_test(None);
        assert_eq!(o5, vec![7.0f32; n5], "n=5 なのに書いた");
    }

    /// task 282: ゲートは既定 on。明示的な偽値だけ off (task 266 では既定 off だった)。
    #[test]
    fn qchain_gate_values() {
        assert!(mix_qchain_from_value(None), "未設定は on (既定)");
        assert!(mix_qchain_from_value(Some("")));
        assert!(mix_qchain_from_value(Some("1")));
        assert!(mix_qchain_from_value(Some(" TRUE ")));
        assert!(mix_qchain_from_value(Some("on")));
        assert!(!mix_qchain_from_value(Some("0")));
        assert!(!mix_qchain_from_value(Some("off")));
        assert!(!mix_qchain_from_value(Some("false")));
        assert!(!mix_qchain_from_value(Some(" NO ")));
    }

    /// task 266: fused chain は **ブロック量子の 4 型だけ**。
    /// BF16 (既定モデルの mix) / F32 / IQ では 1 度も入らない。
    #[test]
    fn qchain_types_are_the_block_four() {
        use crate::kquant::GgmlType as T;
        for t in [T::Q8_0, T::Q5_1, T::Q4K, T::Q5K] {
            assert!(quant_gpu_ty_ok(t), "{t:?} は GPU で回せるはず");
        }
        for t in [T::Bf16, T::F32, T::F16, T::IQ3S, T::IQ2S, T::IQ4Nl, T::Q2_0] {
            assert!(!quant_gpu_ty_ok(t), "{t:?} は CPU 経路のままのはず");
        }
    }

    /// 呼び出しが 0 のときは行を出さない (既定 on でもログを変えない)。
    /// ゲートが既定 on になったぶん、既定モデル (mix は BF16) では
    /// `hc_mix_quant_fast` が 1 度も呼ばれないので行が出ない。
    #[test]
    fn qchain_line_is_quiet_without_calls() {
        mix_qchain_reset();
        assert!(mix_qchain_line().is_none(), "呼び出し 0 では行を出さない");
    }

    /// task 282: prefill のガードが立っている間は chain を使わない (decode 専用)。
    /// prefill で chain を使うと prefill が 6〜10% 遅れる (lab t282a/t282b)。
    #[test]
    fn prefill_guard_turns_the_qchain_gate_off() {
        assert!(!in_prefill_seq(), "既定は prefill の外");
        {
            let _g = enter_prefill_seq();
            assert!(in_prefill_seq(), "ガードの中は prefill");
            // 門番の式と同じ形 (mix_gpu / 型はここでは true 扱い)。
            let will = mix_qchain_enabled() && !in_prefill_seq();
            assert!(!will, "prefill の中では chain を撃たない");
        }
        assert!(!in_prefill_seq(), "Drop で戻る");
    }
}

/// task 138: `DENNO_MIX_CHAIN` の値を解釈する (純関数・テスト用)。
/// **既定 on** で、明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
pub(crate) fn mix_chain_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// task 138: hc mix の down → lo_apply → up を 1 往復に畳む経路
/// (`DENNO_MIX_CHAIN`、**既定 on**。`=0` で従来の 2 往復に戻る)。
/// `DENNO_MIX_GPU` が on のときだけ意味を持つ。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn mix_chain_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mix_chain_from_value(std::env::var("DENNO_MIX_CHAIN").ok().as_deref()))
}

/// task 158/164: `DENNO_MIX_INJ_EARLY` の値解釈 (純関数・テスト用)。**既定 off**。
///
/// task 158 では既定 on で lab に投入したが、base 条件に含めたため単独の差は
/// 取れていない。task 164 では int-auto の既定を変えないため off から始める
/// (on は明示的な真値だけ)。
pub fn mix_inj_early_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// task 158: `hc_inject` の GEMV + 係数を、down/up の GPU chain を積む**前**に
/// 計算するか (`DENNO_MIX_INJ_EARLY`)。
///
/// inject は `xn` だけで決まるので、chain の完了を待ってから計算すると、その間
/// GPU が遊ぶ (mix は 1 トークンに 97 回)。値は変わらない (入力も演算順序も同じ)。
/// lab t158a では base 条件に含めたため単独の差は取れていないが、全リクエストで
/// md5 は SAME。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn mix_inj_early_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_inj_early_from_value(std::env::var("DENNO_MIX_INJ_EARLY").ok().as_deref())
    })
}

/// task 145: `DENNO_MIX_BATCH_GPU` の解釈 (純関数)。**既定 on**。
/// 明示的な偽値 (`0` / `off` / `false` / `no`) のときだけ切る。
/// on のとき、prefill のバッチ mix (BF16 の down / up) を GPU の
/// `dno_gemm_f32_mt` に載せる。
///
/// ラボの実測 (swift15-abl IQ3_S, 14T, production env):
/// - t145b (depth, PTX=elem): c2000 cold 108.68 → 91.91 s (-15.4%)、
///   next turn 0.98 → 0.97 s、短文 1.35 → 1.35 s、decode 78 → 76 ms/tok、
///   md5 全一致。
/// - t145c (depth8k, 既定 on vs off): c2000 cold 107.87 → 89.00 s、
///   c8000 cold 366.49 → 328.64 s、next turn 0.98/1.04 → 0.97/1.04 s、
///   decode 74-83 ms/tok で不変、md5 全一致・errors 0。
/// - 台帳の mix 区間は 27.0 → 8.6 s (3,826 tok) / 74.1 → 31.0 s (14,018 tok)。
pub(crate) fn mix_batch_gpu_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// task 145: テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
pub(crate) static MIX_BATCH_GPU_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// task 145: prefill のバッチ mix を GPU で回してよいか (既定 off)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn mix_batch_gpu_enabled() -> bool {
    match MIX_BATCH_GPU_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_batch_gpu_from_value(std::env::var("DENNO_MIX_BATCH_GPU").ok().as_deref())
    })
}

/// task 145: バッチ mix の GPU 経路を試す最小トークン数。
///
/// チャンク 8/16 のような小さい塊では 1 呼び出しあたりの H2D + 起動 + 同期 +
/// D2H が計算量を上回る (decode 側の `DENNO_MIX_GPU` は 1 トークン経路、
/// こちらは「層優先バッチ」専用)。既定チャンクは 512 なので通常は通る。
pub(crate) const MIX_BATCH_GPU_MIN_TOK: usize = 64;

/// task 145: 1 呼び出しの積 (nt × n × k) の下限。inject (n=hc=4) のような
/// 小さい行列を GPU に載せないための門番。
pub(crate) const MIX_BATCH_GPU_MIN_MAC: usize = 4_000_000;

/// task 145: バッチ mix を GPU に載せるかの判定 (純関数・GPU 無しで検証できる)。
///
/// 条件: ゲート on / BF16 / nt >= [`MIX_BATCH_GPU_MIN_TOK`] /
/// 積 nt×n×k >= [`MIX_BATCH_GPU_MIN_MAC`]。
pub(crate) fn mix_batch_gpu_ok(
    ty: crate::kquant::GgmlType,
    k: usize,
    n: usize,
    nt: usize,
    enabled: bool,
) -> bool {
    enabled
        && ty == crate::kquant::GgmlType::Bf16
        && nt >= MIX_BATCH_GPU_MIN_TOK
        && nt.saturating_mul(n).saturating_mul(k) >= MIX_BATCH_GPU_MIN_MAC
}

/// task 145: prefill のバッチ mix の 1 行列を GPU で回す (BF16 のときだけ)。
///
/// `true` = GPU が計算した (out は GPU の結果で置き換え済み)。`false` は
/// 呼び出し側が CPU の [`gemv_batch`] へ落ちること。
#[cfg(feature = "cuda")]
pub(crate) fn mix_batch_gpu(w: &GgufLinear, xs: &[f32], nt: usize, outs: &mut [f32]) -> bool {
    if mix_batch_gpu_ok(w.ty(), w.k(), w.n(), nt, mix_batch_gpu_enabled()) {
        return crate::cuda::maybe_gpu_mix_bf16_batch_into(
            w.uid(),
            w.blocks().as_slice(),
            w.k(),
            w.n(),
            xs,
            nt,
            outs,
        );
    }
    // task 261: 旧来の 4 型 (UD-Q4_K_XL の mix は Q8_0) の prefill は **GPU にしない**。
    // lab t261b の実測で、この形 (down は k=10240/n=320、up は k=320/n=10240) の
    // `moe_gemm_dev` は CPU のバッチ GEMV より遅かった (mix 0.145 -> 0.248 ms/layer、
    // prefill +15%)。decode の GEMV だけを `DENNO_MIX_QUANT_GPU` で載せる。
    false
}

/// cuda feature 無しでは常に CPU。
#[cfg(not(feature = "cuda"))]
pub(crate) fn mix_batch_gpu(w: &GgufLinear, _xs: &[f32], nt: usize, _outs: &mut [f32]) -> bool {
    let _ = mix_batch_gpu_ok(w.ty(), w.k(), w.n(), nt, mix_batch_gpu_enabled());
    false
}

/// head (`output`) の GEMV を**行 (語彙方向) で並列化する**門番 (行数)。
///
/// `GgufLinear::gemv_label` の汎用の門番は 512 行 (`PAR_MIN_ROWS`) なので、実形
/// (151936 行) では既に行並列に入る。ここは head 専用にしきい値を明示するための
/// もので、**門番が 512 に届かない形でも head だけは割る**。行は独立で、1 行の
/// 計算順序 (`dot` の累積順) は分割に依らないので、値を変えずに分割だけ変わる。
pub const HEAD_ROW_PAR_MIN: usize = 8;

/// `DENNO_HEAD_PAR` の解釈。**既定 on**。`0` / `false` / `off` のときだけ 0
/// (= 汎用の門番に任せる = 従来の `gemv_label`)。
pub(crate) fn head_row_par_min_from(v: Option<&str>) -> usize {
    match v {
        Some(s) if s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off") => {
            0
        }
        _ => HEAD_ROW_PAR_MIN,
    }
}

/// `DENNO_HEAD_PAR` をプロセスで 1 回だけ読む (逃げ道 `DENNO_HEAD_PAR=0`)。
pub(crate) fn head_row_par_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| head_row_par_min_from(std::env::var("DENNO_HEAD_PAR").ok().as_deref()))
}

/// `hc_*_down` (320 行 × `hc_dim`) の GEMV を並列化する門番 (行数)。
///
/// 既定の門番は 512 行なので 320 行は 1 スレッドで回っていた (実形の CPU 計測で
/// mix の約 50%)。行は独立で各要素の加算順は行内で完結するので、行を分けても
/// 値はビット一致する。マイクロベンチ ([`tests/head_gemv_bench`] と同じ流儀) が
/// 本番入口を通すので `pub`。
pub const HC_DOWN_ROW_PAR_MIN: usize = 64;

/// `hc_mix` を**量子化重み**で計算する (`hyper::hc_mix` と同じ丸め順序)。
///
/// `inject` が `None` なら inject は 0 で返す (head の mixer は inject を
/// 持たない: 参照実装は `w_inject` / `w_down_inject` とも `nullptr`)。
///
/// `DENNO_MIX_FAST` (既定 on) で [`hc_mix_quant_fast`]、off で
/// [`hc_mix_quant_ref`] を通る。どちらも同じ値 (ビット一致) を返す。
pub(crate) fn hc_mix_quant(
    res: &HyperStreams,
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if mix_fast_enabled() {
        return hc_mix_quant_fast(res, norm, down, up, inject, hp, false);
    }
    hc_mix_quant_ref(res, norm, down, up, inject, hp)
}

/// task 464: [`hc_mix_quant`] の「出力をホストで読まない」版 (ssm 層の attn mix 用)。
///
/// `DENNO_MIX_NOWAIT` が on で mix 鎖が走ったとき、鎖は `mixed` をホストへ降ろさず
/// (D2H 無し)、イベントも待たない。返る `mixed` は **空の `Vec`** で、入力は
/// デバイス側の `dm` を直後の GDN submission がそのまま読む。鎖が走らない・
/// ゲートが off のときは従来どおり値が返るので、呼び出し側は空かどうかだけ見れば
/// よい。
pub(crate) fn hc_mix_quant_defer(
    res: &HyperStreams,
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    hc_mix_quant_fast(res, norm, down, up, inject, hp, cfg!(feature = "cuda"))
}

/// [`hc_mix_quant`] の旧経路 (A/B と `DENNO_MIX_FAST=0` 用)。
///
/// `DENNO_MIX_KPROF=1` のときだけ区間ごとの時間を積む (既定 off では
/// `stamp()` が `None` を返すだけで、確保も演算も従来どおり)。
pub(crate) fn hc_mix_quant_ref(
    res: &HyperStreams,
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let t_all = mix_kprof::stamp();

    let t = mix_kprof::stamp();
    // task 368: norm は**高速経路と同じ関数**を通す。`DENNO_HC_NORM_QUAD` (既定 on)
    // は二乗和の丸め順を変えるので、ここだけ `grouped_rms_norm` (逐次和) のままだと
    // 「fast と ref がビット一致する」という既存の契約が崩れる。値の意味は同じで、
    // 変わるのは丸め順だけ (ゲート `=0` で両方とも逐次和に戻る)。
    let mut xn = vec![0.0f32; hp.hc * hp.n_embd];
    hyper::grouped_rms_norm_batch_into(std::slice::from_ref(res), norm, &mut xn, hp);
    mix_kprof::mark(t, mix_kprof::GN);

    let t = mix_kprof::stamp();
    let mut lo = vec![0.0f32; down.n()];
    mix_kprof::rowpar::set_label(mix_kprof::rowpar::L_DOWN);
    down.gemv_label("hc_down", &xn, &mut lo)
        .map_err(|err| one(format!("hc_mix down: {err}")))?;
    // task 88: down GEMV 1 発の実行先と dtype (区間計測と同じ `Instant` を再利用)。
    mix_kprof::gemv::note_down(t, down.ty());
    mix_kprof::rowpar::set_label(mix_kprof::rowpar::L_UP);
    mix_kprof::mark(t, mix_kprof::DOWN);

    let t = mix_kprof::stamp();
    hyper::hc_lo_apply(&mut lo, hp);
    mix_kprof::mark(t, mix_kprof::LO);

    let t = mix_kprof::stamp();
    let mut gate = vec![0.0f32; up.n()];
    up.gemv_label("hc_up", &lo, &mut gate)
        .map_err(|err| one(format!("hc_mix up: {err}")))?;
    mix_kprof::gemv::note_up(t, up.ty());
    mix_kprof::rowpar::set_label(mix_kprof::rowpar::NONE);
    mix_kprof::mark(t, mix_kprof::UP);

    let t = mix_kprof::stamp();
    let mixed = hyper::hc_mixed(&xn, &gate, hp);
    mix_kprof::mark(t, mix_kprof::MIXED);

    let inj = match inject {
        Some(w) => {
            let mut v = vec![0.0f32; w.n()];
            let t = mix_kprof::stamp();
            w.gemv_label("hc_inject", &xn, &mut v)
                .map_err(|err| one(format!("hc_mix inject: {err}")))?;
            // task 96: inject GEMV 1 発の実行先と dtype (down/up と同じ `Instant`)。
            mix_kprof::gemv::note_inj(t, w.ty());
            mix_kprof::mark(t, mix_kprof::INJ);
            let t = mix_kprof::stamp();
            hyper::hc_inject_apply(&mut v, hp);
            mix_kprof::mark(t, mix_kprof::INJ_APPLY);
            v
        }
        None => vec![0.0f32; hp.hc],
    };
    mix_kprof::mark(t_all, mix_kprof::TOTAL);
    mix_kprof::bump_call();
    Ok((mixed, inj))
}

/// [`hc_mix_quant_batch`] が使う作業域 (grow-only)。
///
/// 実形 (nt=512 / hc_dim=10240) では `xn` と `gate` が 20 MB ずつ、`mixed` が
/// 5 MB ある。呼び出しごとに `vec![0.0; ..]` で確保すると、書き込み時の
/// first-touch のページフォルトだけで 2 ms 前後乗る (実測:
/// `crate::model::hyper::tests::bench_mix_call_real_shape` で
/// grouped_rms_norm_batch 3.02 → 1.05 ms)。層ループの外で 1 回だけ持ち、
/// チャンクを跨いで使い回す。
#[derive(Default)]
pub(crate) struct MixScratch {
    pub(crate) xn: Vec<f32>,
    pub(crate) lo: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) mixed: Vec<f32>,
}

thread_local! {
    /// [`hc_mix_quant_fast`] が使う 1 トークンぶんの作業域 (grow-only)。
    ///
    /// decode は 1 トークンで mix を `2 * n_layer + 1` 回呼ぶ。呼び出しごとに
    /// `xn` / `gate` を `vec![0.0; 10240]` (40 KB ずつ) で確保すると、ゼロ埋めと
    /// 解放だけで毎回 ~90 KB を触る。スレッドを跨がず、次の呼び出しが全要素を
    /// 書くので、残っていても値は変わらない (長さは必ず `[..n]` で切る)。
    pub(crate) static MIX_SCRATCH: std::cell::RefCell<MixScratch> = const {
        std::cell::RefCell::new(MixScratch {
            xn: Vec::new(),
            lo: Vec::new(),
            gate: Vec::new(),
            mixed: Vec::new(),
        })
    };
}

/// テスト用: [`hc_mix_quant_fast`] が呼ばれた回数。
/// 「既定 on の分岐が実際に通った」ことを数えるためのもの。
#[cfg(test)]
pub(crate) static MIX_FAST_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// [`hc_mix_quant`] の高速経路 (`DENNO_MIX_FAST` 既定 on)。
///
/// 旧経路との違いは 2 点だけで、どちらも**値はビット一致**する:
/// 1. 作業域 (`xn` / `lo` / `gate`) をスレッドローカルで使い回す
///    (毎回の `vec![0.0; ..]` とゼロ埋めを消す)。
/// 2. `hc_mixed` を `d` 方向に rayon で分ける ([`hyper::hc_mixed_into_par`])。
///    `d` は独立で、`s` の昇順の加算は 1 つの `d` の中で完結する。
/// 3. task 106: 合成 (`hc_mixed_into_par`) は戻り値の `Vec` に直接書く
///    (scratch へ書いて `to_vec()` で写す 10 KB の 2 パスを 1 パスに)。
/// task 261: mix の 1 行列 (m=1) を GPU で回す。
///
/// BF16 は従来の [`crate::cuda::maybe_gpu_mix_bf16_into`] (f32 展開 /
/// `dno_matvec_bf16`)、旧来の 4 型は dense 重みレジストリ + `moe_gemv_dev`
/// ([`crate::cuda::maybe_gpu_gguf_gemv_into`]、`DENNO_MIX_QUANT_GPU` のときだけ)。
/// どちらも不発なら `false` を返し、呼び出し側が CPU 経路へ落ちる。
#[cfg(feature = "cuda")]
pub(crate) fn mix_gemv_gpu(w: &GgufLinear, x: &[f32], out: &mut [f32]) -> bool {
    if !mix_gpu_enabled() {
        return false;
    }
    if w.ty() == crate::kquant::GgmlType::Bf16 {
        return crate::cuda::maybe_gpu_mix_bf16_into(
            w.uid(),
            w.blocks().as_slice(),
            w.k(),
            w.n(),
            x,
            out,
        );
    }
    if mix_quant_gpu_enabled() && quant_gpu_ty_ok(w.ty()) {
        return crate::cuda::maybe_gpu_gguf_gemv_into(
            w.uid(),
            w.ty(),
            w.blocks().as_slice(),
            w.k(),
            w.n(),
            x,
            out,
        );
    }
    false
}

pub(crate) fn hc_mix_quant_fast(
    res: &HyperStreams,
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
    // task 464: 量子型の鎖に「出力をホストへ降ろさず待ちも省く」を許すか。
    defer: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    #[cfg(not(feature = "cuda"))]
    let _ = defer;
    #[cfg(test)]
    MIX_FAST_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // task 266: 分岐の分母 (ゲート on のときだけ数える)。
    if mix_qchain_enabled() {
        MIX_QCHAIN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    let hc_dim = hp.hc * hp.n_embd;
    let e = hp.n_embd;
    let t_all = mix_kprof::stamp();
    // task 184: norm と inject をデバイスで回す経路 (`DENNO_MIX_DEC_DEV`)。
    // 通ればホストの mix_norm / mix_inj と down/up の chain を丸ごと置き換え、
    // mixed[e] と inj[hc] だけを受け取る (同期 1 回)。縮約がツリーになるので
    // 丸め順は従来と変わる。失敗したら下のホスト経路へそのまま落ちる。
    #[cfg(feature = "cuda")]
    if crate::cuda::mix_dec_dev_enabled()
        && down.ty() == crate::kquant::GgmlType::Bf16
        && up.ty() == crate::kquant::GgmlType::Bf16
        && inject
            .map(|w| w.ty() == crate::kquant::GgmlType::Bf16)
            .unwrap_or(true)
    {
        let mut m = vec![0.0f32; e];
        let mut ij = vec![0.0f32; hp.hc];
        // task 216: inj をホストで先に作る (`DENNO_DEC_MIX_INJHOST`)。
        // GPU の鎖 (norm → down → up → mixed) が走っている間に CPU で
        // grouped RMS + inj GEMV を済ませる。GPU の inj カーネル 1 本と、その
        // 分の待ちが消える (ホストは同期で遊んでいるので重ねられる)。
        // 値は GPU の inj と 1 ビットも変わらない契約 (同じ xn・同じ k 昇順)。
        let inj_host: Option<Vec<f32>> = match inject {
            Some(w)
                if crate::cuda::dec_mix_injhost_on()
                    && matches!(
                        w.ty(),
                        crate::kquant::GgmlType::F32
                            | crate::kquant::GgmlType::F16
                            | crate::kquant::GgmlType::Bf16
                    ) =>
            {
                let mut v = vec![0.0f32; w.n()];
                MIX_SCRATCH.with(|cell| -> Result<()> {
                    let mut b = cell.borrow_mut();
                    let xnb = MixScratch::take(&mut b.xn, hc_dim);
                    hyper::grouped_rms_norm_batch_into(std::slice::from_ref(res), norm, xnb, hp);
                    w.gemv_label("hc_inject", xnb, &mut v)
                        .map_err(|err| one(format!("hc_mix inject (host): {err}")))
                })?;
                hyper::hc_inject_apply(&mut v, hp);
                Some(v)
            }
            _ => None,
        };
        let inj_arg = if inj_host.is_some() {
            None
        } else {
            inject.map(|w| (w.uid(), w.k(), w.n(), w.blocks().as_slice()))
        };
        if crate::cuda::maybe_gpu_mix_bf16_chain3_into(
            down.uid(),
            down.blocks().as_slice(),
            down.k(),
            down.n(),
            up.uid(),
            up.blocks().as_slice(),
            up.k(),
            up.n(),
            inj_arg,
            norm,
            hp.eps,
            1.0f32 / hp.hc as f32,
            e,
            hp.hc,
            &res.data,
            &mut m,
            &mut ij,
        ) {
            if let Some(v) = inj_host {
                ij = v;
            } else {
                hyper::hc_inject_apply(&mut ij, hp);
            }
            mix_kprof::mark(t_all, mix_kprof::TOTAL);
            mix_kprof::bump_call();
            return Ok((m, ij));
        }
    }
    MIX_SCRATCH.with(|cell| {
        // task 179: mix 1 回の内訳 (`DENNO_GDN_DEC_KPROF` の平均表)。窓は
        // ホスト仕事 (norm / inject / 詰め替え) を含めるため、ここで開く。
        #[cfg(feature = "cuda")]
        let _kp_win = crate::cuda::dec_kprof::window(
            crate::cuda::dec_kprof::MIX,
            crate::cuda::dec_kprof::cur_layer(),
            crate::cuda::dec_kprof::cur_pos(),
            0,
        );
        // 計測 off では closure をそのまま呼ぶだけ。
        macro_rules! kp_step {
            ($name:expr, $e:expr) => {{
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::dec_kprof::step($name, || $e)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    $e
                }
            }};
        }
        let mut b = cell.borrow_mut();
        let MixScratch { xn, lo, gate, .. } = &mut *b;

        let xn = MixScratch::take(xn, hc_dim);
        let t = mix_kprof::stamp();
        kp_step!(
            "mix_norm",
            hyper::grouped_rms_norm_batch_into(std::slice::from_ref(res), norm, xn, hp)
        );
        mix_kprof::mark(t, mix_kprof::GN);

        // task 158: inject (`hc_inject` の GEMV + 係数) は **xn だけで決まる**。
        // CPU 型 (F32 / F16 / BF16 = `gemv_impl` が GPU に奪わない型) なら
        // down/up の GPU chain を積む前に済ませる。以前は chain の同期が返ってから
        // 計算していたため、その間 GPU が遊んでいた (97 呼び出し/token)。
        // 量子型は GPU 経路を持ち順序を変えると同期が増えるので従来位置のまま。
        let inj_is_cpu = |w: &GgufLinear| {
            matches!(
                w.ty(),
                crate::kquant::GgmlType::F32
                    | crate::kquant::GgmlType::F16
                    | crate::kquant::GgmlType::Bf16
            )
        };
        let do_inj = |w: &GgufLinear| -> Result<Vec<f32>> {
            let mut v = vec![0.0f32; w.n()];
            let t = mix_kprof::stamp();
            // task 484: inject の実形 (hc 行 x hc_dim の F32) は 1 行ずつの逐次和で
            // 16.7 us/回かかっていた (lab t484a の `[dec-kprof-avg]` mix_inj 16.1 us、
            // 97 呼び出し/token で 1.56 ms/token)。行を同時に回して x を共有する
            // FMA 版は 0.98 us (`DENNO_MIX_INJ_R4`、既定 off)。
            if !inject_gemv_r4(w, xn, &mut v) {
                kp_step!("mix_inj", w.gemv_label("hc_inject", xn, &mut v))
                    .map_err(|err| one(format!("hc_mix inject: {err}")))?;
            }
            // task 96: inject GEMV 1 発の実行先と dtype (down/up と同じ `Instant`)。
            mix_kprof::gemv::note_inj(t, w.ty());
            mix_kprof::mark(t, mix_kprof::INJ);
            let t = mix_kprof::stamp();
            hyper::hc_inject_apply(&mut v, hp);
            mix_kprof::mark(t, mix_kprof::INJ_APPLY);
            Ok(v)
        };
        // task 282: 量子型の fused chain が走る見込みのときは、inject の GEMV を
        // 「鎖を積んだあと・待つ前」へ回す (ホスト仕事が GPU の待ちに隠れる。
        // 鎖が不発なら下の通常位置で計算し直す)。
        let qchain_will = mix_gpu_enabled()
            && mix_qchain_enabled()
            && !in_prefill_seq()
            && quant_gpu_ty_ok(down.ty())
            && quant_gpu_ty_ok(up.ty());
        let mut inj_early = match inject {
            Some(w) if mix_inj_early_enabled() && inj_is_cpu(w) && !qchain_will => {
                Some(do_inj(w)?)
            }
            _ => None,
        };

        let lo = MixScratch::take(lo, down.n());
        let gate = MixScratch::take(gate, up.n());
        let t = mix_kprof::stamp();
        mix_kprof::rowpar::set_label(mix_kprof::rowpar::L_DOWN);
        let hc_fast = hc_mix_fast_enabled();
        // task 150: down → lo_apply → up → **hc_mixed** まで GPU で閉じる経路
        // (`DENNO_DECODE_CHAIN2` + `DENNO_DECODE_CHAIN2_MIX`、両方 BF16 のとき)。
        // 従来 (task 138 の chain) は up の出力 gate[hc*e] を D2H してホストで
        // sigmoid 4 本 x e の加算を回していた (実測 34 us/回 x 97 回/token)。
        // ここでは dno_hc_mixed_f32 が同じ丸め順で mixed[e] を作るので、D2H は
        // 1/hc になり、ホストの計算は無くなる (値はビット一致)。
        let mut chain2_mixed: Option<Vec<f32>> = None;
        #[cfg(feature = "cuda")]
        if mix_gpu_enabled()
            && mix_chain_enabled()
            && decode_chain2_mix_enabled()
            && down.ty() == crate::kquant::GgmlType::Bf16
            && up.ty() == crate::kquant::GgmlType::Bf16
        {
            let mut m = kp_step!("mix_alloc", vec![0.0f32; e]);
            if crate::cuda::maybe_gpu_mix_bf16_chain2_into(
                down.uid(),
                down.blocks().as_slice(),
                down.k(),
                down.n(),
                up.uid(),
                up.blocks().as_slice(),
                up.k(),
                up.n(),
                1.0f32 / hp.hc as f32,
                e,
                xn,
                &mut m,
            ) {
                chain2_mixed = Some(m);
            }
        }
        // task 266: 量子型 (UD-Q4_K_XL の mix は Q8_0) の fused chain。
        // down → silu(v*inv_hc) → up → hc_mixed を 1 回の同期で閉じる
        // (`DENNO_MIX_QCHAIN=1` のときだけ)。不発なら下の 2 往復経路へ落ちる。
        #[cfg(feature = "cuda")]
        if chain2_mixed.is_none()
            && mix_gpu_enabled()
            && mix_qchain_enabled()
            && !in_prefill_seq()
            && quant_gpu_ty_ok(down.ty())
            && quant_gpu_ty_ok(up.ty())
        {
            let mut m = kp_step!("mix_alloc", vec![0.0f32; e]);
            // task 282: inject の GEMV は鎖を積んだあと・待つ前にホストで回す
            // (GPU が遊んでいる間に済むので待ちに隠れる。値は従来と同じ)。
            let mut inj_hidden: Option<Vec<f32>> = None;
            let inj_already = inj_early.is_some();
            let mut inj_hook = || -> anyhow::Result<()> {
                if let (Some(w), false) = (inject, inj_already) {
                    inj_hidden = Some(do_inj(w).map_err(|e| anyhow::anyhow!("{e}"))?);
                }
                Ok(())
            };
            if crate::cuda::maybe_gpu_mix_q_chain_into(
                down.uid(),
                down.ty(),
                down.blocks().as_slice(),
                down.k(),
                down.n(),
                up.uid(),
                up.ty(),
                up.blocks().as_slice(),
                up.k(),
                up.n(),
                1.0f32 / hp.hc as f32,
                e,
                hp.hc,
                xn,
                &mut m,
                Some(&mut inj_hook),
                defer,
            ) {
                MIX_QCHAIN_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // task 464: defer した回は `m` がホストへ降りていない (中身は
                // 不定)。空の `Vec` を返して「入力はデバイス側だけ」を伝える。
                chain2_mixed = Some(if crate::cuda::mix_take_deferred() {
                    Vec::new()
                } else {
                    m
                });
                if inj_hidden.is_some() {
                    inj_early = inj_hidden;
                }
            }
        }
        let gpu_chain2 = chain2_mixed.is_some();
        // task 138: down → lo_apply → up を 1 回の同期に畳む経路
        // (`DENNO_MIX_GPU` on・既定 on の `DENNO_MIX_CHAIN`・両方 BF16 のとき)。
        // 丸めは従来の GPU 経路と同じ (silu は dno_expf = glibc expf のビット一致
        // 実装) なので、通れば logits は従来とビット一致する。不発なら下の
        // 2 往復経路へそのまま落ちる。
        #[cfg(feature = "cuda")]
        let gpu_chain = !gpu_chain2
            && mix_gpu_enabled()
            && mix_chain_enabled()
            && down.ty() == crate::kquant::GgmlType::Bf16
            && up.ty() == crate::kquant::GgmlType::Bf16
            && crate::cuda::maybe_gpu_mix_bf16_chain_into(
                down.uid(),
                down.blocks().as_slice(),
                down.k(),
                down.n(),
                up.uid(),
                up.blocks().as_slice(),
                up.k(),
                up.n(),
                1.0f32 / hp.hc as f32,
                xn,
                gate,
            );
        #[cfg(not(feature = "cuda"))]
        let gpu_chain = false;
        if gpu_chain2 {
            // chain2 は down/up/hc_mixed を 1 回の同期で閉じる (内訳は chain と同じ
            // 規則で down に積む)。
            mix_kprof::gemv::note_down(t, down.ty());
            mix_kprof::gemv::note_up(t, up.ty());
            mix_kprof::rowpar::set_label(mix_kprof::rowpar::NONE);
            mix_kprof::mark(t, mix_kprof::DOWN);
        } else if gpu_chain {
            // 内訳は down/up の 2 区間に同じ区間を積む (合計は chain 1 回のまま)。
            mix_kprof::gemv::note_down(t, down.ty());
            mix_kprof::gemv::note_up(t, up.ty());
            mix_kprof::rowpar::set_label(mix_kprof::rowpar::NONE);
            mix_kprof::mark(t, mix_kprof::DOWN);
        } else {
            // task 110: `DENNO_MIX_GPU=1` かつ BF16 のときだけ down GEMV を GPU に奪う
            // (カーネルは dno_matvec_f32 流用・1 行 = 1 ブロック・k 昇順直列加算 = CPU
            // 参照とビット一致。task 121 から長い行 k >= n (= down 実形) は s169 の
            // 並列リダクション版 dno_matvec_f32_r (block 256・固定木縮約・決定論) に
            // 切り替わるためビット一致しない: 許容 絶対 1e-4 + 相対 1e-5)。false の
            // ときは従来どおり CPU 経路 (cuda 側が warn を 1 行だけ出す)。GPU 経路が
            // 成功すると `last_linear_ran_on_gpu` が立つので下の `note_down` が gpu= を
            // 積む ([mix-kprof] の内訳で見える)。
            #[cfg(feature = "cuda")]
            let gpu_down = mix_gemv_gpu(down, xn, lo);
            #[cfg(not(feature = "cuda"))]
            let gpu_down = false;
            if !gpu_down {
                if hc_fast {
                    down.gemv_label_par_2row("hc_down", xn, lo, HC_DOWN_ROW_PAR_MIN)
                } else {
                    down.gemv_label_par("hc_down", xn, lo, HC_DOWN_ROW_PAR_MIN)
                }
                .map_err(|err| one(format!("hc_mix down: {err}")))?;
            }
            // task 88: down GEMV 1 発の実行先と dtype (区間計測と同じ `Instant` を再利用)。
            mix_kprof::gemv::note_down(t, down.ty());
            mix_kprof::rowpar::set_label(mix_kprof::rowpar::L_UP);
            mix_kprof::mark(t, mix_kprof::DOWN);

            let t = mix_kprof::stamp();
            hyper::hc_lo_apply(lo, hp);
            mix_kprof::mark(t, mix_kprof::LO);

            let t = mix_kprof::stamp();
            // task 110: up GEMV も down と同じ (`DENNO_MIX_GPU=1` かつ BF16 のときだけ
            // GPU に奪う。入力は hc_lo_apply 済みの `lo`、成功で `note_up` が gpu= を積む)。
            // up 実形 (k=320 < n=10240) は短い行なので task 121 でも逐次版のまま
            // (ビット一致)。
            #[cfg(feature = "cuda")]
            let gpu_up = mix_gemv_gpu(up, lo, gate);
            #[cfg(not(feature = "cuda"))]
            let gpu_up = false;
            if !gpu_up {
                if hc_fast {
                    up.gemv_label_par_2row("hc_up", lo, gate, 0)
                } else {
                    up.gemv_label("hc_up", lo, gate)
                }
                .map_err(|err| one(format!("hc_mix up: {err}")))?;
            }
            mix_kprof::gemv::note_up(t, up.ty());
            mix_kprof::rowpar::set_label(mix_kprof::rowpar::NONE);
            mix_kprof::mark(t, mix_kprof::UP);
        }

        let t = mix_kprof::stamp();
        // task 106: 合成は戻り値に直接書く (scratch へ書いて `to_vec()` で写す
        // 旧形から、10 KB の写しを消す。同じ [`hyper::hc_mixed_into_par`] なので
        // 値はビット一致)。`mixed` scratch は batch 経路 (`hc_mix_quant_batch`) 専用。
        // task 150: chain2 が通ったときは GPU が mixed まで作っている (ホストの
        // hc_mixed は回さない)。
        let mut out = match chain2_mixed {
            Some(m) => m,
            None => {
                let mut out = vec![0.0f32; e];
                hyper::hc_mixed_into_par(xn, gate, &mut out, hp);
                out
            }
        };
        mix_kprof::mark(t, mix_kprof::MIXED);

        // task 158: inject は CPU 型なら chain の前に計算済み (`inj_early`)。
        // 量子型 (GPU 経路) は従来どおりここで計算する。
        let inj = match inj_early.take() {
            Some(v) => v,
            None => match inject {
                Some(w) => do_inj(w)?,
                None => vec![0.0f32; hp.hc],
            },
        };
        mix_kprof::mark(t_all, mix_kprof::TOTAL);
        mix_kprof::bump_call();
        Ok((out, inj))
    })
}

/// [`hc_mix_quant`] のバッチ版 (`nt` トークンぶんまとめて行列積)。
///
/// 行列積 (`down` / `up` / `inject`) を [`gemv_batch`] に、要素演算
/// (grouped RMS 正規化・`hc_lo_apply` / `hc_mixed` / `hc_inject_apply`) を
/// `hyper::*_batch` に置き換える。要素演算は要素ごとに独立で、トークン内の
/// 加算順序も逐次版と同じなので、1 トークンぶんの出力は [`hc_mix_quant`] と
/// ビット一致する (`inject` は `[nt][hc]` の連結で返す)。
///
/// **nt == 1 (decode) は逐次 fast 経路 ([`hc_mix_quant_fast`]) に委譲する**
/// (task 86c)。1 トークンぶんの出力は上記の通り逐次版とビット一致するので、
/// 委譲で値は動かない。batch 経路の [`gemv_batch`] は nt=1 では classic カーネルを
/// 1 行ずつ回すだけで (`gemv_rows_pertoken` / `expert_gemm_any` m=1)、2 行カーネルも
/// 行並列の region 計測も持たない。委譲先は P84/P86 でチューニングした入口そのもの
/// (2 行カーネル + `HC_DOWN_ROW_PAR_MIN` の行並列 + scratch 使い回し) で、
/// `DENNO_HC_MIX_FAST=0` で旧経路 (GPU intercept に落ちる入口) に戻るのも
/// 逐次経路と同じ。
///
/// 要素演算の出力は `scratch` に書く (戻り値で返す `mixed` / `inject` だけ確保する)。
/// nt == 1 の委譲時は `scratch` を使わない (委譲先がスレッドローカルの
/// [`MIX_SCRATCH`] を使う)。
pub(crate) fn hc_mix_quant_batch(
    ress: &[HyperStreams],
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
    scratch: &mut MixScratch,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let nt = ress.len();
    let e = hp.n_embd;
    let hc_dim = hp.hc * hp.n_embd;
    for (t, r) in ress.iter().enumerate() {
        if r.data.len() != hc_dim {
            return Err(one(format!(
                "hc_mix (batch): トークン {t} の res 長 {} が hc_dim {hc_dim} と違う",
                r.data.len()
            )));
        }
    }
    // 形の検査は nt によらず先に済ませる (nt == 1 の委譲時もエラーの文言を揃える。
    // 委譲先の逐次 fast は low_rank / hc_dim の検査を持たない)。
    let dn = down.n();
    if dn != hp.low_rank {
        return Err(one(format!(
            "hc_mix down (batch): 行数 {dn} が low_rank {} と違う",
            hp.low_rank
        )));
    }
    let un = up.n();
    if un != hc_dim {
        return Err(one(format!(
            "hc_mix up (batch): 行数 {un} が hc_dim {hc_dim} と違う"
        )));
    }
    if let Some(w) = inject {
        let wn = w.n();
        if wn != hp.hc {
            return Err(one(format!(
                "hc_mix inject (batch): 行数 {wn} が hc {} と違う",
                hp.hc
            )));
        }
    }
    if nt == 1 {
        let (mixed, inj) = hc_mix_quant_fast(&ress[0], norm, down, up, inject, hp, false)?;
        return Ok((vec![mixed], inj));
    }
    // task 126: 短い prefill は 1 トークン経路のほうが速い (しきい値の根拠は
    // [`PREFILL_MIX_SEQ_MAX`]。既定 off)。値は CPU では batch 経路とビット一致
    // (テスト `prefill_mix_seq_matches_batch`)、`DENNO_MIX_GPU=1` のときは
    // decode と同じ GPU 経路になるので丸めだけが変わる (許容 1e-4 + 1e-5)。
    if prefill_mix_seq_for(nt) {
        let mut mixed = Vec::with_capacity(nt);
        let mut injs = Vec::with_capacity(nt * hp.hc);
        for r in ress.iter() {
            let (m, i) = hc_mix_quant(r, norm, down, up, inject, hp)?;
            mixed.push(m);
            injs.extend_from_slice(&i);
        }
        return Ok((mixed, injs));
    }
    // task 126: batch 経路も `[mix-kprof]` に内訳を積む (`DENNO_MIX_KPROF=1` のときだけ
    // `Instant` を取る。off は `stamp()` が `None` を返すだけ)。prefill の mix は
    // ここを通るので、区間計測が無いと「down が重いのか up が重いのか」が読めなかった。
    let t_all = mix_kprof::stamp();
    let xn_flat = MixScratch::take(&mut scratch.xn, nt * hc_dim);
    let t = mix_kprof::stamp();
    hyper::grouped_rms_norm_batch_into(ress, norm, xn_flat, hp);
    mix_kprof::mark(t, mix_kprof::GN);

    let lo = MixScratch::take(&mut scratch.lo, nt * dn);
    let t = mix_kprof::stamp();
    // task 145: BF16 の down / up は GPU の m トークン GEMM を先に試す
    // (`DENNO_MIX_BATCH_GPU=1` のときだけ。失敗・不発なら CPU バッチへ)。
    if !mix_batch_gpu(down, xn_flat, nt, lo) {
        gemv_batch(down, xn_flat, nt, lo)
            .map_err(|err| one(format!("hc_mix down (batch): {err}")))?;
    }
    mix_kprof::mark(t, mix_kprof::DOWN);
    let t = mix_kprof::stamp();
    hyper::hc_lo_apply_batch(lo, nt, hp);
    mix_kprof::mark(t, mix_kprof::LO);

    let gate = MixScratch::take(&mut scratch.gate, nt * un);
    let t = mix_kprof::stamp();
    if !mix_batch_gpu(up, lo, nt, gate) {
        gemv_batch(up, lo, nt, gate).map_err(|err| one(format!("hc_mix up (batch): {err}")))?;
    }
    mix_kprof::mark(t, mix_kprof::UP);
    let mixed_flat = MixScratch::take(&mut scratch.mixed, nt * e);
    let t = mix_kprof::stamp();
    hyper::hc_mixed_batch_into(xn_flat, gate, nt, mixed_flat, hp);
    mix_kprof::mark(t, mix_kprof::MIXED);
    let mut mixed = Vec::with_capacity(nt);
    for t in 0..nt {
        mixed.push(mixed_flat[t * e..(t + 1) * e].to_vec());
    }

    let injs = match inject {
        Some(w) => {
            let wn = w.n();
            if wn != hp.hc {
                return Err(one(format!(
                    "hc_mix inject (batch): 行数 {wn} が hc {} と違う",
                    hp.hc
                )));
            }
            let mut v = vec![0.0f32; nt * wn];
            let t = mix_kprof::stamp();
            gemv_batch(w, xn_flat, nt, &mut v)
                .map_err(|err| one(format!("hc_mix inject (batch): {err}")))?;
            mix_kprof::mark(t, mix_kprof::INJ);
            let t = mix_kprof::stamp();
            hyper::hc_inject_apply_batch(&mut v, nt, hp);
            mix_kprof::mark(t, mix_kprof::INJ_APPLY);
            v
        }
        None => vec![0.0f32; nt * hp.hc],
    };
    mix_kprof::mark(t_all, mix_kprof::TOTAL);
    mix_kprof::bump_call();
    Ok((mixed, injs))
}

/// PLE 注入 (層 `ple.layers` のみ)。`res_hc` を置き換えて返す。
///
/// 参照実装 `build_ple` (1467-1555 行) の順序:
/// 1. `ple_emb` (16 head × 160 = 2560。head の順に連結) を `ple_key` / `ple_value` に通す
/// 2. `key` を `ple_norm_key` で、`res_hc` を `ple_norm_query` で grouped RMS
/// 3. ストリームごとの内積 / sqrt(n_embd) → `sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))`
/// 4. `gated[s] = value * gate[s]` (value は hc 本へ broadcast)
/// 5. `gated` を `ple_norm_conv` で grouped RMS → depthwise 因果 conv → silu
/// 6. `res_hc + (gated + conv_out)` (この足し方の順序も参照実装どおり)
///
/// **F1 の制限**: conv の履歴は 0 で埋める。`t = 0` で効くのは遅延 0 の項
/// (`k = kern - 1`) だけ。履歴バッファは F2 で入れる。
pub(crate) fn ple_inject(
    p: &Qwen4ExpParams,
    pw: &PleW,
    ple_emb: &[f32],
    res: &HyperStreams,
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
    contrib_rms: Option<&mut f32>,
) -> Result<HyperStreams> {
    // task 154: 台帳の副区間 (PLE 注入)。
    let _sub = q4ledger::sub(q4ledger::SUB_PLE);
    let mut key = vec![0.0f32; pw.key.n()];
    pw.key
        .gemv_label("ple", ple_emb, &mut key)
        .map_err(|err| one(format!("ple_key: {err}")))?;
    let mut value = vec![0.0f32; pw.value.n()];
    pw.value
        .gemv_label("ple", ple_emb, &mut value)
        .map_err(|err| one(format!("ple_value: {err}")))?;
    ple_inject_from_kv(p, pw, &key, &value, res, hp, hist_buf, contrib_rms)
}

/// task 195: `DENNO_PF_PLE_FAST` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// **既定 on** (lab t195a/t195b: 単独で -1.4%、値はビット一致)。
pub(crate) fn pf_ple_fast_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}

/// task 195: 平坦な res をその場で更新する PLE 注入を使うか
/// (`DENNO_PF_PLE_FAST`。task 198 で既定 on)。
pub(crate) fn pf_ple_fast_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_ple_fast_from_value(std::env::var("DENNO_PF_PLE_FAST").ok().as_deref()))
}

/// task 195: 並列版 (`ple_inject_batch_par`) を使う最小トークン数。
///
/// 1 チャンク = 512 トークン (実形) なので常に並列版。次のターンの 19 トークン
/// のような短い prefill は rayon の起動のほうが高いので逐次平坦版にする。
pub(crate) const PLE_PAR_MIN_TOK: usize = 32;

/// task 195: PLE 射影 (`key` / `value`) を 1 チャンクぶん回す。
///
/// 経路は task 172 / 182 と同じ: `DENNO_PF_PLE_GPU` (量子型) → だめなら
/// `DENNO_PF_PLE_F32_GPU` (f32 / bf16 / f16) → だめなら CPU のバッチ GEMV。
pub(crate) fn ple_project_kv(
    pw: &PleW,
    flat: &[f32],
    nt: usize,
    e: usize,
    key: &mut [f32],
    value: &mut [f32],
    t: Option<std::time::Instant>,
) -> Result<()> {
    let hc_dim = pw.key.n();
    pf_ple_kprof::add(pf_ple_kprof::GEMM, t);
    // task 172: 2 射影 (key / value) を GPU で回す (`DENNO_PF_PLE_GPU`)。
    // 表の pread は SSD なので CPU のまま (gather は呼び出し側)。
    let mut ple_dev = false;
    if pf_ple_gpu_on() {
        ple_dev = Qwen4Exp::pf_ple_dev_run(pw, flat, nt, e, key, value);
    }
    // task 182: 実モデルの ple_key / ple_value は **F32** (目録は Q8_0) なので、
    // 上の量子型専用経路は形の段階で断られる (t172b/t172c)。F32 は同じ
    // `dno_gemm_f32_mt` のバッチ GEMM に載せる (`DENNO_PF_PLE_F32_GPU`)。
    // 片方でも GPU に載らなければ両方 CPU で回し直す (混在させない)。
    #[cfg(feature = "cuda")]
    if !ple_dev && pf_ple_f32_gpu_on() && nt >= PF_PLE_F32_GPU_MIN_TOK {
        // 実ファイルの型は F32 か BF16 (目録は Q8_0)。BF16 は f32 に開いてから
        // 同じカーネルに載せる (VRAM は f32 で 2 倍になるが 105 MB 程度)。
        let f32_ok = matches!(pw.key.ty(), GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16)
            && matches!(pw.value.ty(), GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16);
        PF182_LOG_PLE_TY.call_once(|| {
            eprintln!(
                "[pf182] ple key ty={:?} value ty={:?} (f32_ok={f32_ok})",
                pw.key.ty(),
                pw.value.ty()
            )
        });
        if f32_ok {
            let k_ok = crate::cuda::maybe_gpu_f32_gemm_batch_into(
                pw.key.uid(),
                pw.key.blocks().as_slice(),
                pw.key.k(),
                pw.key.n(),
                flat,
                nt,
                key,
                pw.key.ty(),
            );
            let v_ok = crate::cuda::maybe_gpu_f32_gemm_batch_into(
                pw.value.uid(),
                pw.value.blocks().as_slice(),
                pw.value.k(),
                pw.value.n(),
                flat,
                nt,
                value,
                pw.value.ty(),
            );
            ple_dev = k_ok && v_ok;
            if ple_dev {
                PF182_LOG_PLE_F32.call_once(|| {
                    eprintln!("[pf182] PLE key/value: GPU f32 GEMM (DENNO_PF_PLE_F32_GPU)")
                });
            }
        }
    }
    if !ple_dev {
        gemv_batch_cpu(&pw.key, flat, nt, key)
            .map_err(|err| one(format!("ple_key (batch): {err}")))?;
        gemv_batch_cpu(&pw.value, flat, nt, value)
            .map_err(|err| one(format!("ple_value (batch): {err}")))?;
    }
    let _ = hc_dim;
    Ok(())
}

/// task 300: `DENNO_PF_PLE_REUSE` — PLE の作業域を grow-only で使い回す (既定 on)。
///
/// 従来は PLE 層のたびに `back` (`nt × hc_dim`、14k のチャンクで 82 MB) と
/// `key` / `value` (82 + 20 MB) を `vec![0.0; n]` で作り直し、さらに
/// `pf_download_res` が同じ大きさの一時 Vec をもう 1 本作って写していた。
/// 32 MB を超える確保は glibc が mmap で出し入れするので、チャンクごとに
/// 数万ページのマイナーフォルトが戻る (`[exp-ram-probe] prefill minflt=58 万`)。
/// 3 本とも**全要素を書き直す**ので、使い回してもデバイスへ渡るバイト列は同じ。
pub const PLE_REUSE_ENV: &str = "DENNO_PF_PLE_REUSE";

/// [`PLE_REUSE_ENV`] の解釈 (未設定・空は [`DEFAULT_PLE_REUSE`])。
pub(crate) fn pf_ple_reuse_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => DEFAULT_PLE_REUSE,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// PLE の作業域を使い回すか (プロセスで 1 回だけ env を読む)。
pub(crate) fn pf_ple_reuse_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_ple_reuse_from_value(std::env::var(PLE_REUSE_ENV).ok().as_deref()))
}

/// task 505: `DENNO_PF_PLE_FLAT_REUSE` — PLE のデバイス経路が作る `flat`
/// (チャンクの全トークンの行を詰めた `nt × e`、14k で 21 MB) を使い回す。
///
/// 従来は層チャンクごとに `Vec::with_capacity` で新しく確保していたので、
/// 初回タッチのフォルトと開放を 14k で 14 回 (PLE 層は 48 層中 2 層 x 7 チャンク)
/// 払っていた。`DENNO_PF_PLE_REUSE` と同じ「全要素を書き直す」形なので、
/// 使い回してもデバイスへ渡るバイト列は同じ。既定は測って採るまで **off**。
pub const PLE_FLAT_REUSE_ENV: &str = "DENNO_PF_PLE_FLAT_REUSE";

/// [`PLE_FLAT_REUSE_ENV`] の解釈 (明示的な真値だけ on)。
pub(crate) fn pf_ple_flat_reuse_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// `flat` を使い回すか (プロセスで 1 回だけ env を読む)。
pub(crate) fn pf_ple_flat_reuse_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_ple_flat_reuse_from_value(std::env::var(PLE_FLAT_REUSE_ENV).ok().as_deref())
    })
}

/// task 393: `DENNO_PF_PLE_PIN` — PLE 層の res 往復バッファを pinned にする。
///
/// prefill は PLE 層 (このモデルは 48 層中 2 層) で残差を **ホストへ降ろして
/// PLE を注入し、上げ直す**。14k のチャンクは 1 件 82 MB で、
/// `[q4ledger-pf] sub other` の実測 (t402b b2、14k) は
/// `res_up 201.0 ms [h=1148.4MB/14 d=645.3MB/8]` / `ple 202.4 ms [d=717.7MB/14]`。
/// pageable の同期 H2D/D2H はこのホストで 3.5-5.7 GB/s しか出ないので、
/// バッファ自体を pinned (`PfHostBuf`) にすれば DMA がそのまま速くなる
/// (追加の写しを挟まない。task 318 の `DENNO_PF_PIN_STAGE` は逆に
/// 「pinned へ写してから同期コピー」で写しを 1 回増やしていた)。
pub const PLE_PIN_ENV: &str = "DENNO_PF_PLE_PIN";

/// [`PLE_PIN_ENV`] の解釈 (未設定・空は [`DEFAULT_PLE_PIN`])。
pub(crate) fn pf_ple_pin_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => DEFAULT_PLE_PIN,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// PLE の res 往復バッファを pinned にするか (プロセスで 1 回だけ env を読む)。
pub(crate) fn pf_ple_pin_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_ple_pin_from_value(std::env::var(PLE_PIN_ENV).ok().as_deref()))
}

/// task 393: pinned 版のバッファ型 (cuda feature が無いときは `Vec` で代替する)。
///
/// prefill の GPU 経路自体が cuda feature の内側なので、feature 無しでは
/// このバッファは使われない。それでも `cargo build --workspace` が通るように
/// 同じ口 (`new` / `len` / `as_slice` / `as_mut_slice`) だけを持つ器を置く。
#[cfg(feature = "cuda")]
pub(crate) type PlePinBuf = crate::cuda::PfHostBuf;

#[cfg(not(feature = "cuda"))]
pub(crate) struct PlePinBuf {
    pub(crate) v: Vec<f32>,
}

thread_local! {
    /// task 300: PLE の res 往復バッファ (`nt × hc_dim`)。grow-only。
    pub(crate) static PLE_BACK_BUF: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// task 393: 同じバッファの pinned 版 (`DENNO_PF_PLE_PIN`)。grow-only。
    pub(crate) static PLE_BACK_PIN: std::cell::RefCell<Option<PlePinBuf>> =
        const { std::cell::RefCell::new(None) };
    /// task 300: PLE の射影バッファ (`key` / `value`)。grow-only。
    pub(crate) static PLE_KV_BUF: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    /// task 505: PLE の `flat` (チャンクの全トークンの行を詰めた `nt × e`)。
    /// `DENNO_PF_PLE_FLAT_REUSE` のとき grow-only で使い回す。
    pub(crate) static PF_PLE_FLAT: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// task 393: PLE の res 往復バッファ (スライスとして長さ `len` ちょうど).
///
/// pinned のときは `PfHostBuf` (使い回し) を、そうでなければ `Vec` を持つ。
/// どちらも `Deref` で `[f32]` になるので、呼び出し側は今までどおり
/// `&mut back` / `&back` / `back[a..b]` と書ける。
pub(crate) struct PleBack {
    pub(crate) pin: Option<PlePinBuf>,
    pub(crate) v: Vec<f32>,
    pub(crate) len: usize,
}

/// task 300: PLE の res バッファを取る。2 つ目の戻り値が「使い回しか」。
pub(crate) fn pf_ple_back_take(len: usize) -> (PleBack, bool) {
    pf_ple_back_take_with(len, pf_ple_reuse_on())
}

/// task 300: [`pf_ple_back_take`] の実体 (テストから直接叩けるように分けた)。
///
/// task 393: [`pf_ple_pin_on`] のときは pinned の 1 本を使い回す
/// (`PfHostBuf::new` はスレッド局所の空きリストから借りるので、2 回目以降は
/// 確保もしない)。返る長さは常に `len` ちょうど。
pub(crate) fn pf_ple_back_take_with(len: usize, reuse: bool) -> (PleBack, bool) {
    pf_ple_back_take_pin(len, reuse, pf_ple_pin_on())
}

/// [`pf_ple_back_take_with`] の本体 (`pinned` を引数にして CPU テストから
/// 両方の枝を通せるようにした)。
pub(crate) fn pf_ple_back_take_pin(len: usize, reuse: bool, pinned: bool) -> (PleBack, bool) {
    if !reuse || len == 0 {
        return (
            PleBack {
                pin: None,
                v: vec![0.0f32; len],
                len,
            },
            false,
        );
    }
    if pinned {
        let cur = PLE_BACK_PIN.with(|c| c.borrow_mut().take());
        let buf = match cur {
            Some(b) if b.len() >= len => b,
            _ => PlePinBuf::new(len),
        };
        return (
            PleBack {
                pin: Some(buf),
                v: Vec::new(),
                len,
            },
            true,
        );
    }
    let mut v = PLE_BACK_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if v.len() < len {
        v.resize(len, 0.0);
    }
    v.truncate(len);
    (
        PleBack {
            pin: None,
            v,
            len,
        },
        true,
    )
}

/// task 300: 使い回した res バッファを返す。
pub(crate) fn pf_ple_back_return(b: PleBack, reused: bool) {
    if !reused {
        return;
    }
    let PleBack { pin, v, len: _ } = b;
    match pin {
        Some(buf) => PLE_BACK_PIN.with(|c| *c.borrow_mut() = Some(buf)),
        None => PLE_BACK_BUF.with(|c| *c.borrow_mut() = v),
    }
}

/// task 300: PLE の `key` / `value` を取る (`len_k` / `len_v` に切り詰める)。
pub(crate) fn pf_ple_kv_take(len_k: usize, len_v: usize) -> ((Vec<f32>, Vec<f32>), bool) {
    pf_ple_kv_take_with(len_k, len_v, pf_ple_reuse_on())
}

/// task 300: [`pf_ple_kv_take`] の実体 (テストから直接叩けるように分けた)。
pub(crate) fn pf_ple_kv_take_with(
    len_k: usize,
    len_v: usize,
    reuse: bool,
) -> ((Vec<f32>, Vec<f32>), bool) {
    if !reuse || len_k == 0 {
        return ((vec![0.0f32; len_k], vec![0.0f32; len_v]), false);
    }
    let (mut k, mut v) = PLE_KV_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if k.len() < len_k {
        k.resize(len_k, 0.0);
    }
    if v.len() < len_v {
        v.resize(len_v, 0.0);
    }
    k.truncate(len_k);
    v.truncate(len_v);
    ((k, v), true)
}

/// task 300: 使い回した `key` / `value` を返す。
pub(crate) fn pf_ple_kv_return(kv: (Vec<f32>, Vec<f32>), reused: bool) {
    if reused {
        PLE_KV_BUF.with(|c| *c.borrow_mut() = kv);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_inject_batch(
    p: &Qwen4ExpParams,
    pw: &PleW,
    ple_rows: &[&[f32]],
    ress: &mut [HyperStreams],
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
    mut last_rms: Option<&mut f32>,
) -> Result<()> {
    let nt = ple_rows.len();
    if nt == 0 || ress.len() != nt {
        return Err(one(format!(
            "ple_inject_batch: トークン数が違う (rows={nt} res={})",
            ress.len()
        )));
    }
    let _sub = q4ledger::sub(q4ledger::SUB_PLE);
    let t_all = pf_ple_kprof::t0();
    let e = p.n_embd;
    let hc_dim = p.hc_dim();
    let kdim = ple_rows.first().map(|r| r.len()).unwrap_or(0);
    if ple_rows.iter().any(|r| r.len() != kdim) {
        return Err(one("ple_inject_batch: ple_emb の長さが揃っていない"));
    }
    let t_flat = pf_ple_kprof::t0();
    let mut flat = Vec::with_capacity(nt * kdim);
    for r in ple_rows {
        flat.extend_from_slice(r);
    }
    pf_ple_kprof::add(pf_ple_kprof::FLAT, t_flat);
    let mut key = vec![0.0f32; nt * hc_dim];
    let mut value = vec![0.0f32; nt * e];
    ple_project_kv(pw, &flat, nt, e, &mut key, &mut value, pf_ple_kprof::t0())?;
    let t_tok = pf_ple_kprof::t0();
    for t in 0..nt {
        let want = last_rms.is_some() && t + 1 == nt;
        let mut contrib = 0.0f32;
        let out = ple_inject_from_kv(
            p,
            pw,
            &key[t * hc_dim..(t + 1) * hc_dim],
            &value[t * e..(t + 1) * e],
            &ress[t],
            hp,
            hist_buf,
            want.then_some(&mut contrib),
        )?;
        if want {
            if let Some(d) = last_rms.as_deref_mut() {
                *d = contrib;
            }
        }
        ress[t] = out;
    }
    pf_ple_kprof::add(pf_ple_kprof::TOK, t_tok);
    pf_ple_kprof::add(pf_ple_kprof::ALL, t_all);
    pf_ple_kprof::note_call(nt);
    pf_ple_kprof::emit();
    Ok(())
}

/// task 195: `DENNO_PF_PLE_FAST` のときの PLE 注入 (平坦な res をその場で更新)。
///
/// 従来の [`ple_inject_batch`] (トークンごとに `HyperStreams` の Vec を作って
/// 呼び出し側が書き戻す) との違いは**出力の持ち方だけ**:
/// - トークンごとの `Vec` 確保 / `key`/`value` の `to_vec` / 書き戻しコピーをしない
/// - conv 履歴をリングで進める (従来は 1 トークンごとに hist*hc_dim 要素を
///   `copy_within` でずらしていた = 実形で 1 トークン 327 KB)
///
/// 値は [`ple_inject_batch`] とビット一致する (`ple_inject_into` は同じ関数で、
/// 履歴の意味も同じ。チャンク終端で線形に戻す)。
pub(crate) fn ple_inject_batch_flat(
    p: &Qwen4ExpParams,
    pw: &PleW,
    ple_rows: &[&[f32]],
    res_flat: &mut [f32],
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
    mut last_rms: Option<&mut f32>,
) -> Result<()> {
    let nt = ple_rows.len();
    let e = p.n_embd;
    let hc_dim = p.hc_dim();
    if nt == 0 || res_flat.len() != nt * hc_dim {
        return Err(one(format!(
            "ple_inject_batch_flat: 形が違う (rows={nt} res={} hc_dim={hc_dim})",
            res_flat.len()
        )));
    }
    let _sub = q4ledger::sub(q4ledger::SUB_PLE);
    let t_all = pf_ple_kprof::t0();
    let kdim = ple_rows.first().map(|r| r.len()).unwrap_or(0);
    if ple_rows.iter().any(|r| r.len() != kdim) {
        return Err(one("ple_inject_batch_flat: ple_emb の長さが揃っていない"));
    }
    let t_flat = pf_ple_kprof::t0();
    let mut flat = Vec::with_capacity(nt * kdim);
    for r in ple_rows {
        flat.extend_from_slice(r);
    }
    pf_ple_kprof::add(pf_ple_kprof::FLAT, t_flat);
    // task 300: `DENNO_PF_PLE_REUSE` では射影バッファを使い回す (全要素を書き直す)。
    let ((mut key, mut value), kv_reused) = pf_ple_kv_take(nt * hc_dim, nt * e);
    ple_project_kv(pw, &flat, nt, e, &mut key, &mut value, pf_ple_kprof::t0())?;
    let t_tok = pf_ple_kprof::t0();
    let mut sc = PleInjScratch::take(hc_dim);
    let mut head = 0usize;
    let mut r = Ok(());
    for t in 0..nt {
        let want = last_rms.is_some() && t + 1 == nt;
        let mut contrib = 0.0f32;
        let rt = ple_inject_into(
            p,
            pw,
            &key[t * hc_dim..(t + 1) * hc_dim],
            &value[t * e..(t + 1) * e],
            &mut res_flat[t * hc_dim..(t + 1) * hc_dim],
            hp,
            hist_buf,
            &mut head,
            true,
            want.then_some(&mut contrib),
            &mut sc,
        );
        if let Err(err) = rt {
            r = Err(err);
            break;
        }
        if want {
            if let Some(d) = last_rms.as_deref_mut() {
                *d = contrib;
            }
        }
    }
    sc.put();
    // task 300: 射影バッファを返す。
    pf_ple_kv_return((key, value), kv_reused);
    pf_ple_kprof::add(pf_ple_kprof::TOK, t_tok);
    // チャンク終端で履歴を線形 (先頭が最古) に戻す。他の経路 (decode の 1 トークン
    // 注入) は線形前提なので、ここで戻さないと経路をまたいだときに履歴がずれる。
    if r.is_ok() {
        let t_hist = pf_ple_kprof::t0();
        let mut tmp = Vec::new();
        ple_hist_to_linear(hist_buf, head, hc_dim, &mut tmp);
        pf_ple_kprof::add(pf_ple_kprof::HIST, t_hist);
    }
    pf_ple_kprof::add(pf_ple_kprof::ALL, t_all);
    pf_ple_kprof::note_call(nt);
    pf_ple_kprof::emit();
    r
}

/// task 195: チャンク内のトークンを並列に回す PLE 注入 (`DENNO_PF_PLE_FAST`)。
///
/// lab t195a の `[pf-ple-kprof]` (14,018 + 3,826 トークン) で、PLE 注入 1,163 ms の
/// 内訳は conv 489 / rms 171 / rmsc 83 / dot 75 / gated 11 / out 12 ms だった。
/// どれも**トークンごとに独立**なので (conv は「前のトークンの x」を読むだけで、
/// x は先に全トークンぶん作れる)、2 段に分けて rayon で並列化する:
///
/// 1. 射影 (key / value) → 全トークンの x (= grouped RMS 後の gated) を作る
/// 2. 全トークンの conv + 加算 (x は手元にあるので履歴待ちが無い)
///
/// 履歴は `win = [直前の hist 行][このチャンクの x]` として先頭に連結するので、
/// conv は win の線形添字だけで引ける。値は逐次版とビット一致する
/// (1 トークンあたりの演算列を変えない)。
pub(crate) fn ple_inject_batch_par(
    p: &Qwen4ExpParams,
    pw: &PleW,
    ple_rows: &[&[f32]],
    res_flat: &mut [f32],
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
    mut last_rms: Option<&mut f32>,
) -> Result<()> {
    let nt = ple_rows.len();
    let e = p.n_embd;
    let hc = p.n_hc;
    let hc_dim = p.hc_dim();
    if nt == 0 || res_flat.len() != nt * hc_dim {
        return Err(one(format!(
            "ple_inject_batch_par: 形が違う (rows={nt} res={} hc_dim={hc_dim})",
            res_flat.len()
        )));
    }
    let kern = pw.kern;
    let dil = p.ple_ngram_size;
    let hist = (kern - 1) * dil;
    if pw.conv1d_t.len() != kern * hc_dim {
        return Err(one(format!(
            "ple_inject_batch_par: conv1d_t の長さ {} が kern({kern}) * hc_dim({hc_dim}) と違う",
            pw.conv1d_t.len()
        )));
    }
    if hist_buf.len() != hist * hc_dim {
        return Err(one(format!(
            "ple_inject_batch_par: 履歴の長さ {} が hist({hist}) * hc_dim({hc_dim}) と違う",
            hist_buf.len()
        )));
    }
    let _sub = q4ledger::sub(q4ledger::SUB_PLE);
    let t_all = pf_ple_kprof::t0();
    let kdim = ple_rows.first().map(|r| r.len()).unwrap_or(0);
    if ple_rows.iter().any(|r| r.len() != kdim) {
        return Err(one("ple_inject_batch_par: ple_emb の長さが揃っていない"));
    }
    let t_flat = pf_ple_kprof::t0();
    let mut flat = Vec::with_capacity(nt * kdim);
    for r in ple_rows {
        flat.extend_from_slice(r);
    }
    pf_ple_kprof::add(pf_ple_kprof::FLAT, t_flat);
    // task 300: `DENNO_PF_PLE_REUSE` では射影バッファを使い回す (全要素を書き直す)。
    let ((mut key, mut value), kv_reused) = pf_ple_kv_take(nt * hc_dim, nt * e);
    ple_project_kv(pw, &flat, nt, e, &mut key, &mut value, pf_ple_kprof::t0())?;

    // 作業域 (grow-only): win = (hist + nt) 行、gated = nt 行。
    let (mut win, mut gated) = PLE_PAR_SCRATCH.with(|c| {
        let mut b = c.borrow_mut();
        (std::mem::take(&mut b.0), std::mem::take(&mut b.1))
    });
    let need_win = (hist + nt) * hc_dim;
    if win.len() < need_win {
        win.resize(need_win, 0.0);
    }
    if gated.len() < nt * hc_dim {
        gated.resize(nt * hc_dim, 0.0);
    }
    win[..hist * hc_dim].copy_from_slice(hist_buf);

    let t_tok = pf_ple_kprof::t0();
    let want_rms = last_rms.is_some();
    // 段 1: x (= grouped RMS 後の gated) を全トークンぶん作る。
    {
        let nrm_k: &[f32] = &pw.norm_key;
        let nrm_q: &[f32] = &pw.norm_query;
        let nrm_c: &[f32] = &pw.norm_conv;
        let inv_sqrt_e = 1.0f32 / (e as f32).sqrt();
        let gated_all: &mut [f32] = &mut gated[..nt * hc_dim];
        let x_all: &mut [f32] = &mut win[hist * hc_dim..(hist + nt) * hc_dim];
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            gated_all
                .par_chunks_mut(hc_dim)
                .zip(x_all.par_chunks_mut(hc_dim))
                .enumerate()
                .for_each_init(
                    || (vec![0.0f32; hc_dim], vec![0.0f32; hc_dim]),
                    |(key_n, query_n), (t, (grow, nx))| {
                        let krow = &key[t * hc_dim..(t + 1) * hc_dim];
                        let rrow = &res_flat[t * hc_dim..(t + 1) * hc_dim];
                        hyper::grouped_rms_norm_into(krow, nrm_k, hp, key_n);
                        hyper::grouped_rms_norm_into(rrow, nrm_q, hp, query_n);
                        // 2. ストリームごとの内積 → 符号付き平方根 → sigmoid
                        let mut gate = [0.0f32; 8];
                        for s in 0..hc {
                            let mut acc = 0.0f32;
                            for i in 0..e {
                                let j = s * e + i;
                                acc += key_n[j] * query_n[j];
                            }
                            let sv = acc * inv_sqrt_e;
                            let mag = sv.abs().max(1e-6).sqrt();
                            let sign = if sv > 0.0 {
                                1.0f32
                            } else if sv < 0.0 {
                                -1.0f32
                            } else {
                                0.0f32
                            };
                            gate[s] = hyper::sigmoid(sign * mag);
                        }
                        // 3. gated = value * gate
                        let vrow = &value[t * e..(t + 1) * e];
                        for (s, &g) in gate[..hc].iter().enumerate() {
                            for d in 0..e {
                                grow[s * e + d] = vrow[d] * g;
                            }
                        }
                        // 4. grouped RMS → x
                        hyper::grouped_rms_norm_into(grow, nrm_c, hp, nx);
                    },
                );
        }
        #[cfg(not(feature = "par"))]
        {
            let mut key_n = vec![0.0f32; hc_dim];
            let mut query_n = vec![0.0f32; hc_dim];
            for t in 0..nt {
                let krow = &key[t * hc_dim..(t + 1) * hc_dim];
                let rrow = &res_flat[t * hc_dim..(t + 1) * hc_dim];
                hyper::grouped_rms_norm_into(krow, nrm_k, hp, &mut key_n);
                hyper::grouped_rms_norm_into(rrow, nrm_q, hp, &mut query_n);
                let mut gate = [0.0f32; 8];
                for s in 0..hc {
                    let mut acc = 0.0f32;
                    for i in 0..e {
                        let j = s * e + i;
                        acc += key_n[j] * query_n[j];
                    }
                    let sv = acc * inv_sqrt_e;
                    let mag = sv.abs().max(1e-6).sqrt();
                    let sign = if sv > 0.0 {
                        1.0f32
                    } else if sv < 0.0 {
                        -1.0f32
                    } else {
                        0.0f32
                    };
                    gate[s] = hyper::sigmoid(sign * mag);
                }
                let grow = &mut gated_all[t * hc_dim..(t + 1) * hc_dim];
                let vrow = &value[t * e..(t + 1) * e];
                for (s, &g) in gate[..hc].iter().enumerate() {
                    for d in 0..e {
                        grow[s * e + d] = vrow[d] * g;
                    }
                }
                let nx = &mut x_all[t * hc_dim..(t + 1) * hc_dim];
                hyper::grouped_rms_norm_into(grow, nrm_c, hp, nx);
            }
        }
    }
    // 段 2: conv + silu + 加算。最後のトークンだけ寄与 RMS も取る。
    {
        let win_all: &[f32] = &win[..(hist + nt) * hc_dim];
        let gated_all: &[f32] = &gated[..nt * hc_dim];
        let wt: &[f32] = &pw.conv1d_t;
        let par_nt = if want_rms { nt - 1 } else { nt };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            res_flat[..par_nt * hc_dim]
                .par_chunks_mut(hc_dim)
                .enumerate()
                .for_each_init(
                    || vec![0.0f32; hc_dim],
                    |conv, (t, r)| {
                        ple_conv_silu_win(
                            conv.as_mut_slice(),
                            win_all,
                            hist + t,
                            wt,
                            kern,
                            dil,
                            hc_dim,
                        );
                        let _ = ple_add_token(
                            r,
                            &gated_all[t * hc_dim..(t + 1) * hc_dim],
                            conv.as_slice(),
                            false,
                        );
                    },
                );
        }
        #[cfg(not(feature = "par"))]
        {
            let mut conv = vec![0.0f32; hc_dim];
            for t in 0..par_nt {
                ple_conv_silu_win(&mut conv, win_all, hist + t, wt, kern, dil, hc_dim);
                let _ = ple_add_token(
                    &mut res_flat[t * hc_dim..(t + 1) * hc_dim],
                    &gated_all[t * hc_dim..(t + 1) * hc_dim],
                    &conv,
                    false,
                );
            }
        }
        if want_rms {
            let t = nt - 1;
            let mut conv = vec![0.0f32; hc_dim];
            ple_conv_silu_win(&mut conv, win_all, hist + t, wt, kern, dil, hc_dim);
            let sq = ple_add_token(
                &mut res_flat[t * hc_dim..(t + 1) * hc_dim],
                &gated_all[t * hc_dim..(t + 1) * hc_dim],
                &conv,
                true,
            );
            if let Some(d) = last_rms.as_deref_mut() {
                *d = if hc_dim == 0 {
                    0.0
                } else {
                    (sq / hc_dim as f32).sqrt()
                };
            }
        }
    }
    // 履歴を進める: 新しい履歴 = [直前 ++ このチャンク] の末尾 hist 行。
    if hist > 0 {
        hist_buf.copy_from_slice(&win[nt * hc_dim..(nt + hist) * hc_dim]);
    }
    pf_ple_kprof::add(pf_ple_kprof::TOK, t_tok);
    // 作業域を返す。
    PLE_PAR_SCRATCH.with(|c| {
        let mut b = c.borrow_mut();
        b.0 = win;
        b.1 = gated;
    });
    // task 300: 射影バッファを返す。
    pf_ple_kv_return((key, value), kv_reused);
    pf_ple_kprof::add(pf_ple_kprof::ALL, t_all);
    pf_ple_kprof::note_call(nt);
    pf_ple_kprof::emit();
    Ok(())
}

thread_local! {
    /// task 195: 並列 PLE の作業域 (win, gated)。grow-only。
    pub(crate) static PLE_PAR_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// task 195: 並列 PLE の加算 (res += gated + conv)。`want_rms` のときだけ
/// 寄与の二乗和を返す (逐次版と同じ順序 = i 昇順で `add*add` を足す)。
#[inline]
pub(crate) fn ple_add_token(res: &mut [f32], gated: &[f32], conv: &[f32], want_rms: bool) -> f32 {
    let mut sq = 0.0f32;
    for i in 0..res.len() {
        let add = gated[i] + conv[i];
        if want_rms {
            sq += add * add;
        }
        res[i] = res[i] + add;
    }
    sq
}

/// task 195: conv + silu の窓版。`win` は `[履歴 hist 行][今のチャンク nt 行]` を
/// 連結したもので、`t_off` が今のトークンの行 (hist + t)。
/// `back` トークン前の x は `win[(t_off - back) * hc_dim ..]` で引ける。
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_conv_silu_win(
    out: &mut [f32],
    win: &[f32],
    t_off: usize,
    wt: &[f32],
    kern: usize,
    dil: usize,
    hc_dim: usize,
) {
    assert_eq!(out.len(), hc_dim, "conv の出力長は hc_dim");
    assert!(t_off < win.len() / hc_dim, "conv の窓の添字が範囲外");
    for v in out.iter_mut() {
        *v = 0.0f32;
    }
    for k in 0..kern {
        let back = (kern - 1 - k) * dil;
        let src = &win[(t_off - back) * hc_dim..][..hc_dim];
        let wk = &wt[k * hc_dim..(k + 1) * hc_dim];
        for c in 0..hc_dim {
            out[c] += wk[c] * src[c];
        }
    }
    for v in out.iter_mut() {
        *v = hyper::silu(*v);
    }
}

/// task 189: PLE 注入の per-token 作業域 (grow-only)。
///
/// 実形は hc_dim = 10,240 (40 KB) で、`ple_inject_from_kv` は 1 トークンあたり
/// key_n / query_n / gated / その clone / conv / 中間 rmsnorm で約 280 KB を
/// 確保していた。14,336 トークン (14k プロンプト) では 4 GB ぶんの確保と
/// 初回タッチのフォールトになり、実測で ple 1,541 ms の大半がこれだった。
#[derive(Default)]
pub(crate) struct PleInjScratch {
    pub(crate) key_n: Vec<f32>,
    pub(crate) query_n: Vec<f32>,
    pub(crate) gated: Vec<f32>,
    pub(crate) nx: Vec<f32>,
    pub(crate) conv: Vec<f32>,
}

thread_local! {
    pub(crate) static PLE_INJ_SCRATCH: std::cell::RefCell<PleInjScratch> =
        std::cell::RefCell::new(PleInjScratch::default());
}

/// task 154: 射影済みの key / value から PLE の残り (RMS → gating → conv → 加算)。
///
/// [`ple_inject`] の GEMV をチャンクでまとめるために切り出した。値は
/// 1 トークンずつ回すのとビット一致する (射影は行独立)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_inject_from_kv(
    p: &Qwen4ExpParams,
    pw: &PleW,
    key: &[f32],
    value: &[f32],
    res: &HyperStreams,
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
    contrib_rms: Option<&mut f32>,
) -> Result<HyperStreams> {
    let hc_dim = p.hc_dim();
    // task 195: 出力は res の複製をその場で更新して作る (値は従来と同じ)。
    let mut out = res.data.clone();
    let mut head = 0usize;
    // task 189: `DENNO_PF_PLE_PAR` のときだけ作業域を使い回す。
    // どちらの経路も同じ関数 (`ple_inject_into`) を通るので値は同一。
    if crate::ple::pf_ple_par_on() {
        let mut sc = PleInjScratch::take(hc_dim);
        let r = ple_inject_into(
            p,
            pw,
            key,
            value,
            &mut out,
            hp,
            hist_buf,
            &mut head,
            false,
            contrib_rms,
            &mut sc,
        );
        sc.put();
        r?;
    } else {
        let mut sc = PleInjScratch::zeros(hc_dim);
        ple_inject_into(
            p,
            pw,
            key,
            value,
            &mut out,
            hp,
            hist_buf,
            &mut head,
            false,
            contrib_rms,
            &mut sc,
        )?;
    }
    Ok(HyperStreams {
        data: out,
        hc: res.hc,
        n_embd: res.n_embd,
    })
}

/// PLE の手順 5: depthwise 因果 conv → silu。
///
/// `out[c] = silu( Σ_k w[c*kern + k] * x[c, t - (kern-1-k)*dil] )`。
/// 参照実装は履歴を `padded = [hist + n_tokens]` として前置きするので、`back`
/// トークン前の値は `hist_buf` の `(hist - back)` 番目のスロット。
///
/// task 189: k を外側・c を内側に置く。各 c の加算は k 昇順のままで c は独立
/// なので値はビット一致する。旧ループは内側に `back == 0` の分岐があり、c 方向
/// にベクトル化できなかった (実形 kern 4 x hc_dim 10,240 x 14,336 トークン
/// = 5.9 億回が ple 1,152 ms の主因)。
pub(crate) fn ple_conv_silu(
    out: &mut [f32],
    x: &[f32],
    hist_buf: &[f32],
    head: usize,
    w: &[f32],
    kern: usize,
    dil: usize,
    hc_dim: usize,
) {
    assert_eq!(out.len(), hc_dim, "conv の出力長は hc_dim");
    assert_eq!(x.len(), hc_dim, "conv の入力長は hc_dim");
    assert_eq!(w.len(), kern * hc_dim, "conv の重み長は kern*hc_dim");
    let hist = (kern - 1) * dil;
    assert_eq!(hist_buf.len(), hist * hc_dim, "conv の履歴長は hist*hc_dim");
    assert!(hist == 0 || head < hist, "履歴の ring head が範囲外 (head={head} hist={hist})");
    for v in out.iter_mut() {
        *v = 0.0f32;
    }
    for k in 0..kern {
        let back = (kern - 1 - k) * dil;
        // back == 0 の項だけは履歴ではなく現在の x。履歴は ring で、`head` が最古の
        // スロット。head = 0 のときは従来の線形配列 (末尾が最新) と同じ添字になる。
        let src: &[f32] = if back == 0 {
            x
        } else {
            let slot = (head + hist - back) % hist;
            &hist_buf[slot * hc_dim..][..hc_dim]
        };
        for c in 0..hc_dim {
            out[c] += w[c * kern + k] * src[c];
        }
    }
    for v in out.iter_mut() {
        *v = hyper::silu(*v);
    }
}

/// task 195: [`ple_conv_silu`] の **kern 優先の重み**版 (`wt[k*hc_dim + c]`)。
///
/// 演算は同じ (k 昇順に `out[c] += wt * src`、最後に silu) なので [`ple_conv_silu`]
/// とビット一致する。重みの並びだけが違うので、内側のループが連続読みになり
/// ベクトル化できる (実測 27 us/トークン → 数 us)。
pub(crate) fn ple_conv_silu_t(
    out: &mut [f32],
    x: &[f32],
    hist_buf: &[f32],
    head: usize,
    wt: &[f32],
    kern: usize,
    dil: usize,
    hc_dim: usize,
) {
    assert_eq!(out.len(), hc_dim, "conv の出力長は hc_dim");
    assert_eq!(x.len(), hc_dim, "conv の入力長は hc_dim");
    assert_eq!(wt.len(), kern * hc_dim, "conv の重み長は kern*hc_dim");
    let hist = (kern - 1) * dil;
    assert_eq!(hist_buf.len(), hist * hc_dim, "conv の履歴長は hist*hc_dim");
    assert!(
        hist == 0 || head < hist,
        "履歴の ring head が範囲外 (head={head} hist={hist})"
    );
    for v in out.iter_mut() {
        *v = 0.0f32;
    }
    for k in 0..kern {
        let back = (kern - 1 - k) * dil;
        let src: &[f32] = if back == 0 {
            x
        } else {
            let slot = (head + hist - back) % hist;
            &hist_buf[slot * hc_dim..][..hc_dim]
        };
        let wk = &wt[k * hc_dim..(k + 1) * hc_dim];
        for c in 0..hc_dim {
            out[c] += wk[c] * src[c];
        }
    }
    for v in out.iter_mut() {
        *v = hyper::silu(*v);
    }
}

/// 履歴を 1 トークン進める (線形版: 先頭を捨てて末尾に今の x を入れる)。
///
/// head は常に 0 のまま。1 トークンずつ回す経路 (decode / 従来の prefill) 用。
pub(crate) fn ple_hist_push_shift(hist_buf: &mut [f32], x: &[f32], hc_dim: usize) {
    let hist = hist_buf.len() / hc_dim;
    if hist == 0 {
        return;
    }
    hist_buf.copy_within(hc_dim.., 0);
    hist_buf[(hist - 1) * hc_dim..].copy_from_slice(x);
}

/// 履歴を 1 トークン進める (リング版: 最古のスロットを上書きして head を進める)。
///
/// 1 トークンあたり hc_dim 個のコピーだけで済む (線形版は hist*hc_dim 個ずつ
/// ずらす)。チャンクの終端で [`ple_hist_to_linear`] により線形へ戻すので、
/// 経路をまたいでも履歴の意味は変わらない。
pub(crate) fn ple_hist_push_ring(hist_buf: &mut [f32], head: &mut usize, x: &[f32], hc_dim: usize) {
    let hist = hist_buf.len() / hc_dim;
    if hist == 0 {
        return;
    }
    let slot = *head % hist;
    hist_buf[slot * hc_dim..(slot + 1) * hc_dim].copy_from_slice(x);
    *head = (slot + 1) % hist;
}

/// リングの履歴を線形 (先頭が最古・末尾が最新) に戻す。head = 0 なら何もしない。
pub(crate) fn ple_hist_to_linear(hist_buf: &mut [f32], head: usize, hc_dim: usize, tmp: &mut Vec<f32>) {
    let hist = hist_buf.len() / hc_dim;
    let head = head % hist.max(1);
    if hist == 0 || head == 0 {
        return;
    }
    tmp.clear();
    tmp.extend_from_slice(hist_buf);
    for j in 0..hist {
        let src = ((head + j) % hist) * hc_dim;
        tmp[j * hc_dim..(j + 1) * hc_dim].copy_from_slice(&hist_buf[src..src + hc_dim]);
    }
    hist_buf.copy_from_slice(&tmp[..hist * hc_dim]);
}

/// task 189: [`ple_conv_silu`] の参照実装 (旧ループ: c 外側・k 内側・分岐あり)。
/// テストでビット一致を固定するためだけに置く。
#[cfg(test)]
pub(crate) fn ple_conv_silu_ref(
    x: &[f32],
    hist_buf: &[f32],
    w: &[f32],
    kern: usize,
    dil: usize,
    hc_dim: usize,
) -> Vec<f32> {
    let hist = (kern - 1) * dil;
    let mut v = vec![0.0f32; hc_dim];
    for c in 0..hc_dim {
        let mut acc = 0.0f32;
        for k in 0..kern {
            let back = (kern - 1 - k) * dil;
            let s = if back == 0 {
                x[c]
            } else {
                hist_buf[(hist - back) * hc_dim + c]
            };
            acc += w[c * kern + k] * s;
        }
        v[c] = hyper::silu(acc);
    }
    v
}

/// [`ple_inject_from_kv`] の本体。作業域は呼び出し側が持つ。
///
/// task 195: 出力を確保せず `res` を**その場で**書き換える (`res[i] = res[i] + add`
/// は i 昇順で、i 番目を読んでから i 番目に書くので従来の
/// `out.push(res.data[i] + add)` と同じ値)。`ring` は履歴の進め方:
/// `true` = リング (head を進める) / `false` = 線形 (先頭を詰めて head は 0 のまま)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_inject_into(
    p: &Qwen4ExpParams,
    pw: &PleW,
    key: &[f32],
    value: &[f32],
    res: &mut [f32],
    hp: &HyperParams,
    hist_buf: &mut [f32],
    hist_head: &mut usize,
    ring: bool,
    mut contrib_rms: Option<&mut f32>,
    sc: &mut PleInjScratch,
) -> Result<()> {
    let e = p.n_embd;
    let hc = p.n_hc;
    let hc_dim = p.hc_dim();
    if key.len() != hc_dim {
        return Err(one(format!(
            "ple_key: 出力長 {} が hc_dim {hc_dim} と違う",
            key.len()
        )));
    }
    if value.len() != e {
        return Err(one(format!(
            "ple_value: 出力長 {} が n_embd {e} と違う",
            value.len()
        )));
    }
    if res.len() != hc_dim {
        return Err(one(format!(
            "ple_inject: res の長さ {} が hc_dim {hc_dim} と違う",
            res.len()
        )));
    }

    // 2. grouped RMS (key は [hc][n_embd] と見る)
    // task 189: 中間の rmsnorm もスレッドの作業域へ直接書く (確保しない)。
    let t_rms = pf_ple_kprof::t0();
    hyper::grouped_rms_norm_into(key, &pw.norm_key, hp, &mut sc.key_n[..hc_dim]);
    hyper::grouped_rms_norm_into(res, &pw.norm_query, hp, &mut sc.query_n[..hc_dim]);
    pf_ple_kprof::add(pf_ple_kprof::RMS, t_rms);
    let key_n = &sc.key_n[..hc_dim];
    let query_n = &sc.query_n[..hc_dim];

    // 3. ストリームごとの内積 → 符号付き平方根 → sigmoid
    let t_dot = pf_ple_kprof::t0();
    let inv_sqrt_e = 1.0f32 / (e as f32).sqrt();
    let mut gate = [0.0f32; 8];
    if hc > gate.len() {
        return Err(one(format!("ple_inject: hc {hc} が上限を超えた")));
    }
    for s in 0..hc {
        let mut acc = 0.0f32;
        for i in 0..e {
            let j = s * e + i;
            acc += key_n[j] * query_n[j];
        }
        let sv = acc * inv_sqrt_e;
        let mag = sv.abs().max(1e-6).sqrt();
        let sign = if sv > 0.0 {
            1.0f32
        } else if sv < 0.0 {
            -1.0f32
        } else {
            0.0f32
        };
        gate[s] = hyper::sigmoid(sign * mag);
    }
    pf_ple_kprof::add(pf_ple_kprof::DOT, t_dot);

    // 4. gated[s][d] = value[d] * gate[s]
    let t_gated = pf_ple_kprof::t0();
    for (s, &g) in gate[..hc].iter().enumerate() {
        for d in 0..e {
            sc.gated[s * e + d] = value[d] * g;
        }
    }
    pf_ple_kprof::add(pf_ple_kprof::GATED, t_gated);

    // 5. grouped RMS → depthwise 因果 conv → silu
    let t_rmsc = pf_ple_kprof::t0();
    let gated = &sc.gated[..hc_dim];
    hyper::grouped_rms_norm_into(gated, &pw.norm_conv, hp, &mut sc.nx[..hc_dim]);
    pf_ple_kprof::add(pf_ple_kprof::RMSC, t_rmsc);
    let x = &sc.nx[..hc_dim];
    let kern = pw.kern;
    if pw.conv1d.len() != kern * hc_dim {
        return Err(one(format!(
            "ple_conv1d: 長さ {} が kern({kern}) * hc_dim({hc_dim}) と違う",
            pw.conv1d.len()
        )));
    }
    let dil = p.ple_ngram_size;
    let hist = (kern - 1) * dil;
    if hist_buf.len() != hist * hc_dim {
        return Err(one(format!(
            "ple_conv_state: 長さ {} が hist({hist}) * hc_dim({hc_dim}) と違う",
            hist_buf.len()
        )));
    }
    let t_conv = pf_ple_kprof::t0();
    // task 195: `DENNO_PF_PLE_FAST` は kern 優先の重みで回す (値は同じ)。
    if pf_ple_fast_on() && pw.conv1d_t.len() == kern * hc_dim {
        ple_conv_silu_t(
            &mut sc.conv[..hc_dim],
            x,
            hist_buf,
            if ring { *hist_head } else { 0 },
            &pw.conv1d_t,
            kern,
            dil,
            hc_dim,
        );
    } else {
        ple_conv_silu(
            &mut sc.conv[..hc_dim],
            x,
            hist_buf,
            if ring { *hist_head } else { 0 },
            &pw.conv1d,
            kern,
            dil,
            hc_dim,
        );
    }
    pf_ple_kprof::add(pf_ple_kprof::CONV, t_conv);
    // 履歴を 1 トークン進める (ring = 上書き + head 前進 / 線形 = 先頭を詰める)。
    if hist > 0 {
        if ring {
            ple_hist_push_ring(hist_buf, hist_head, x, hc_dim);
        } else {
            ple_hist_push_shift(hist_buf, x, hc_dim);
        }
    }

    // 6. res + (gated + conv)
    let t_out = pf_ple_kprof::t0();
    let conv = &sc.conv[..hc_dim];
    let gated = &sc.gated[..hc_dim];
    let want_rms = contrib_rms.is_some();
    let mut sq = 0.0f32;
    for i in 0..hc_dim {
        let add = gated[i] + conv[i];
        if want_rms {
            sq += add * add;
        }
        res[i] = res[i] + add;
    }
    pf_ple_kprof::add(pf_ple_kprof::OUT, t_out);
    // P54: 寄与の RMS (`out - res` = gated + conv)。読み取りのみ。
    if let Some(dst) = contrib_rms.as_deref_mut() {
        *dst = if hc_dim == 0 {
            0.0
        } else {
            (sq / hc_dim as f32).sqrt()
        };
    }
    let _ = (hc, e);
    Ok(())
}
