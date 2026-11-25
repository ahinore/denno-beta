//! MoE (expert 選択・VRAM 常駐・batch GEMV・expert 貢献)。
//!
//! task 668: `src/model/qwen4exp.rs` から純粋に切り出した (ロジック変更なし)。

use super::*;

/// task 281/285: `DENNO_EXP_SCRATCH` の解釈 (純関数・テスト用)。
///
/// **既定 on** (task 285)。on では、sumdev 経路 (既定 on) が 1 度も読まない `h`
/// (`nseg * e` = 実形で 40 KB) のゼロ埋めを毎層やらない。値は変わらない
/// (h を使うのは sumdev が不成立のときだけで、その直前に確保する)。
/// 実測は lab t281a/t281b: 48 層で 1.9 MB/token のゼロ埋めが消え、短い decode は
/// -0.4% (ノイズ床 0.4% と同程度)、md5 SAME。明示的な偽値だけ切る。
pub(crate) fn exp_scratch_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// [`exp_scratch_from_value`] をプロセスで 1 回だけ読む。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn exp_scratch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| exp_scratch_from_value(std::env::var("DENNO_EXP_SCRATCH").ok().as_deref()))
}

/// MoE FFN (参照実装 `build_layer_ffn` 1240-1285 行)。
///
/// - ルータ: `ffn_gate_inp @ x` → softmax → top-k (`n_expert_used`) →
///   選択分の和で再正規化 (`norm_w = true`)。
/// - expert: `down( silu(gate(x)) * up(x) )` を選択 index 昇順に
///   `w_e` を掛けて積む。行列は expert 単位で切って GEMV する。
/// - shared expert: `sigmoid(ffn_gate_inp_shexp @ x)` を掛けて最後に 1 回加算。
///
/// `silu` は dense 経路 (`crate::tensor::silu_inplace`) と同じ式を使う。
///
/// `cache` が `Some` のときだけ expert の生バイトを [`ExpertCacheSet`] から取る
/// (P-MOE-G1)。`None` は従来どおり mmap のスライス + `prefetch()` で、バイト列も
/// 呼び出し順も変えない。**どちらの経路も同じ [`moe_expert::expert_gemv_any`] を
/// 通る**ので、同じバイト列に対しては出力がビット一致する。
///
/// P-MOE-G5: 選択 expert の寄与 `h_e` は互いに独立なので、`DENNO_MOE_PAR` (既定 1) が
/// 有効で選択 expert が 2 個以上のときだけ rayon で並列に計算する。**重み付きの
/// 足し込みは index 昇順の逐次**のままなので、丸めの順序は従来と変わらない。
pub(crate) fn moe_ffn_quant(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_full(
        p,
        lw,
        x,
        cache,
        moe_par_enabled(),
        expert_route(),
        None,
        true,
        IvView::none(),
    )
}

/// P56: 1 層ぶんのルータ結果を [`denno_telemetry::Frame::RouterTopK`] として送る。
///
/// `DENNO_TELEMETRY` と `DENNO_TELEMETRY_ARCH` の両方が立っているときだけ
/// フレームを作る (どちらもプロセスで 1 回だけ env を読む。既定はどちらも off)。
/// **probs / sel / weights を読むだけ**なので logits も出力も変わらない。
/// 満杯時はハブが捨てる (推論は待たない)。
///
/// expert index と重みはルータの結果そのものなので、コストは top-k (≤10) 個の
/// コピーと並べ替えだけ (フレームの並びは重みの降順)。
pub(crate) fn emit_router_topk(layer: usize, sel: &[usize], weights: &[f32]) {
    if !denno_telemetry::enabled() || !denno_telemetry::arch_enabled() {
        return;
    }
    denno_telemetry::emit(denno_telemetry::router_topk_frame(layer as u16, sel, weights));
}

/// task 220: `DENNO_TELEM_EXPERT_SKIP` のゲート (既定 off)。
///
/// env はプロセスで 1 回だけ読む (AtomicI8: -1 = 未読 / 0 = off / 1 = on)。
/// **off のときは [`select_topk`] を回さない**ので、余分な計算も確保もしない
/// (分岐 1 個だけ)。テストは [`set_expert_skip_for_tests`] で上書きする。
pub(crate) static EXPERT_SKIP_GATE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

pub(crate) fn expert_skip_enabled() -> bool {
    let cached = EXPERT_SKIP_GATE.load(std::sync::atomic::Ordering::Relaxed);
    if cached >= 0 {
        return cached == 1;
    }
    let on = match std::env::var("DENNO_TELEM_EXPERT_SKIP").ok().as_deref() {
        Some("1") | Some("true") | Some("yes") => true,
        _ => false,
    };
    EXPERT_SKIP_GATE.store(on as i8, std::sync::atomic::Ordering::Relaxed);
    on
}

/// task 220: テストから [`EXPERT_SKIP_GATE`] を上書きする (`None` で env 読みに戻す)。
#[cfg(test)]
pub(crate) fn set_expert_skip_for_tests(v: Option<bool>) {
    let raw = match v {
        None => -1i8,
        Some(false) => 0,
        Some(true) => 1,
    };
    EXPERT_SKIP_GATE.store(raw, std::sync::atomic::Ordering::Relaxed);
}

/// task 220: 「ルータが選びたかったが、介入で無効化されていた expert」を送る。
///
/// - `probs`: softmax 後の全 expert の確率 (マスク前)。
/// - `k`: ルータの top-k 数 (無効化が無ければ選ばれていた集合を求める)。
/// - `sel`: 実際に使った expert (マスク後・index 昇順)。
/// - `mask`: その層の無効ビット列 ([`IvView::expert_mask`] の戻り値)。
///
/// `DENNO_TELEMETRY` / `DENNO_TELEMETRY_ARCH` / `DENNO_TELEM_EXPERT_SKIP` が
/// すべて立っているときだけ計算する (既定 off)。読むだけなので logits は変わらない。
pub(crate) fn emit_expert_skip(layer: usize, probs: &[f32], k: usize, sel: &[usize], mask: &[u64]) {
    if !denno_telemetry::enabled() || !denno_telemetry::arch_enabled() || !expert_skip_enabled() {
        return;
    }
    let want = select_topk(probs, k);
    let skipped: Vec<usize> = want
        .into_iter()
        .filter(|&e| mask.get(e / 64).map(|w| (w >> (e % 64)) & 1 != 0).unwrap_or(false))
        .filter(|e| !sel.contains(e))
        .collect();
    if let Some(fr) = denno_telemetry::expert_skip_frame(layer as u16, &skipped) {
        denno_telemetry::emit(fr);
    }
}

/// P58: 構造テレメトリ (RoleAct / ExpertResidency / TierStats) の送出が有効か。
///
/// RouterTopK と同じく `DENNO_TELEMETRY` と `DENNO_TELEMETRY_ARCH` の両方が要る
/// (どちらもプロセスで 1 回だけ env を読む。既定はどちらも off)。off のときは
/// フレームを作らないので、送出バイトも確保もゼロ。
pub(crate) fn arch_frames_enabled() -> bool {
    // task 155: 購読者 (WS クライアント) が 1 つも居ないときは、フレームを
    // 組み立てない (役割 RMS も取らない)。読む相手が居る場合の出力は不変。
    denno_telemetry::enabled() && denno_telemetry::arch_enabled() && denno_telemetry::has_clients()
}

/// P58: 1 層ぶんの役割別活性 (kind 13) を送る。
///
/// 値は `[attn, gdn, router, shex, ple]` の生値を**層内の最大役割**で割った 0..1
/// ([`denno_telemetry::role_act_frame`] が正規化する)。層の種別で使わない役割
/// (ssm 層の attn / full attention 層の gdn / PLE の無い層の ple) は 0.0 になる。
/// 読むだけなので logits は変わらない。
pub(crate) fn emit_role_act(layer: usize, roles: &RoleRms) {
    if !arch_frames_enabled() {
        return;
    }
    denno_telemetry::emit(denno_telemetry::role_act_frame(layer as u16, roles.as_array()));
}

/// P54: 役割別 RMS (router の top-k 確率和 / shared expert の RMS) も取る
/// [`moe_ffn_quant`]。
///
/// `roles` が `Some` のときだけ 2 つの値を書く。どちらも**すでに計算した値**を
/// 読むだけで、追加の行列積も確保もしない (top-k 和は再正規化の分母をそのまま使う)。
///
/// `emit_router` は P56 の RouterTopK を送るか (どちらも読むだけの観測)。
pub(crate) fn moe_ffn_quant_tapped(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    roles: Option<&mut RoleRms>,
    emit_router: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_full(
        p,
        lw,
        x,
        cache,
        moe_par_enabled(),
        expert_route(),
        roles,
        emit_router,
        iv,
    )
}

/// P39: expert の計算経路 (`DENNO_EXPERT_GPU` + `DENNO_EXPERT_VRAM_MB`)。プロセスで 1 回だけ。
///
/// 既定 (両方とも未設定) は [`Route::Cpu`] で、従来の `expert_gemv_any` だけを通る。
/// 決定と LRU の初期化をここで 1 回だけ行う (decode のたびに env を読まない)。
pub(crate) fn expert_route() -> Route {
    static R: std::sync::OnceLock<Route> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let mb = expert_vram::vram_mb_from_value(
            std::env::var(expert_vram::EXPERT_VRAM_MB_ENV).ok().as_deref(),
        );
        let gpu = std::env::var(expert_vram::EXPERT_GPU_ENV).ok();
        // task 144: IQ 系の型 (本番の expert) を GPU で回すゲート。既定 off。
        let iq = crate::model::exp_iq::iq_gpu_enabled();
        let route = expert_vram::resolve_route(
            gpu.as_deref(),
            mb,
            expert_vram::gpu_available(),
            iq,
        );
        match route {
            Route::Vram => {
                #[cfg(feature = "cuda")]
                {
                    // task 265: prefill の作業域は **起動時には取らない**。長い
                    // prefill の入口で expert から借り、終端で返す
                    // (`pf_pool_acquire` / `PfPoolGuard`)。予算は削らないので、
                    // 起動直後の decode と短いプロンプトは base と同一になる。
                    expert_vram::init(mb.saturating_mul(1024 * 1024));
                    if crate::model::pf_pool::pf_pool_on() {
                        eprintln!(
                            "[pf-pool] 長い prefill (>= {} tok) のときだけ expert から \
                             作業域を借りる (上限 {} MiB / 終端で返す)",
                            crate::model::pf_pool::pf_pool_min_tok(),
                            crate::model::pf_pool::pf_pool_bytes() >> 20
                        );
                    }
                    let cap_mb = mb;
                    let min_hit = expert_min_hit();
                    eprintln!(
                        "[flash] expert GPU: ON (VRAM {} MiB, 既定の CPU 経路は使わない / \
                         ヒット率 < {} のトークンは CPU: DENNO_EXPERT_GPU_MIN_HIT / \
                         IQ ゲート {} / 上げ直し {}/tok)",
                        cap_mb,
                        min_hit,
                        if iq { "on" } else { "off" },
                        crate::model::exp_iq::upload_per_token()
                    );
                }
            }
            Route::Cpu => {
                // 「GPU を要求したのに CPU で回っている」を取りこぼさないように 1 行出す
                // (未設定なら何も出さない = 従来のログと同一)。
                if expert_vram::flag_on(gpu.as_deref()) || iq {
                    eprintln!(
                        "[flash] expert GPU: OFF (DENNO_EXPERT_GPU={:?} / DENNO_EXPERT_IQ_GPU={} / VRAM {} MiB / cuda={}) → CPU 経路",
                        gpu.as_deref().unwrap_or(""),
                        iq,
                        mb,
                        expert_vram::gpu_available()
                    );
                }
            }
        }
        route
    })
}

/// P45: `DENNO_EXPERT_GPU_MIN_HIT` (既定 0.5)。プロセスで 1 回だけ読む。
///
/// ゲート本体 ([`expert_vram::ExpertGate`]) は GPU に依存しないので、この層も
/// `cuda` feature で切らない (CPU だけのビルドでも判定式をテストできる)。
pub(crate) fn expert_min_hit() -> f32 {
    static H: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        expert_vram::min_hit_from_value(
            std::env::var(expert_vram::EXPERT_GPU_MIN_HIT_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// P45: 直近のヒット率ゲート (プロセスで 1 個)。
///
/// `Route::Cpu` (既定) と `DENNO_EXPERT_GPU_MIN_HIT=0` では**作らない**。
/// その場合 [`expert_gate_decide`] は常に GPU を返すので、P39 までの経路と
/// 1 バイトも変わらない (env を読む以外に何もしない)。
pub(crate) fn expert_gate() -> Option<&'static std::sync::Mutex<expert_vram::ExpertGate>> {
    static G: std::sync::OnceLock<Option<std::sync::Mutex<expert_vram::ExpertGate>>> =
        std::sync::OnceLock::new();
    G.get_or_init(|| {
        if expert_route() != Route::Vram {
            return None;
        }
        let min_hit = expert_min_hit();
        if min_hit <= 0.0 {
            return None;
        }
        // task 830: 載せ替えが tinylfu なら、頻度ゲート自身が「冷たい expert を
        // 載せない」ことで thrash を防いでいる。その下のヒット率は構造的に
        // 0.5 を割るので、P45 の降格は当たっている expert まで CPU に落とすだけ
        // (lab t794a 27.3 対 t794b 16.5 ms/tok)。既定で迂回する。
        let policy = expert_vram::policy()
            .unwrap_or_else(crate::model::cache_policy::policy_from_env);
        let bypass = expert_vram::gate_bypass(
            min_hit,
            policy,
            std::env::var(expert_vram::EXPERT_GATE_TINYLFU_ENV)
                .ok()
                .as_deref(),
        );
        if bypass {
            eprintln!(
                "[flash] expert GPU gate: 載せ替え {} なので P45 のヒット率降格を使わない \
                 (常時 GPU。DENNO_EXPERT_GATE_TINYLFU=0 で従来に戻る)",
                policy.as_str()
            );
        }
        Some(std::sync::Mutex::new(
            expert_vram::ExpertGate::new(expert_vram::GateConfig {
                min_hit,
                ..Default::default()
            })
            .with_bypass(bypass),
        ))
    })
    .as_ref()
}

/// task 830: リクエスト終端の 1 行 (ゲートが無ければ `None`)。
///
/// 内訳を 0 に戻すので、リクエスト単位の数として読める。
pub(crate) fn expert_gate_line() -> Option<String> {
    let g = expert_gate()?;
    let mut s = g.lock().ok()?;
    let line = s.stats_line();
    s.reset_decision_counts();
    Some(line)
}

/// expert 1 本 (gate / up / down) を VRAM へ積むときのバイト数。
///
/// `None` は「型・形から読めない」。呼び出し側は「ミス時の H2D 量が読めない」と
/// みなして CPU に倒す (fail-safe)。
pub(crate) fn expert_h2d_bytes_one(lw: &LayerW, e: usize, ff: usize) -> Option<u64> {
    let rb = |ty, k| {
        crate::model::moe_expert::row_bytes(ty, k)
            .ok()
            .map(|b| b as u64)
    };
    let g = rb(lw.gate_exps.ty, e)? * ff as u64;
    let u = rb(lw.up_exps.ty, e)? * ff as u64;
    let d = rb(lw.down_exps.ty, ff)? * e as u64;
    Some(g + u + d)
}

/// P45: このトークンを GPU で回すか。`layer` はログ用。
///
/// ゲートが無効 (CPU 経路 / `DENNO_EXPERT_GPU_MIN_HIT=0`) なら常に GPU =
/// P39 までの挙動。ロックが壊れているときも GPU に倒す (リクエストを落とさない)。
pub(crate) fn expert_gate_decide(
    layer: usize,
    n_sel: usize,
    per_expert: Option<u64>,
) -> expert_vram::GateDecision {
    gate_decide(expert_gate(), layer, n_sel, per_expert)
}

/// P45: 1 層ぶんの実測 (ヒット数 / expert 参照数) をゲートへ積む。
///
/// サンプルへの確定はトークンの先頭 ([`ExpertGate::decide`] を `layer == 0` で
/// 呼んだとき) に行うので、ここでは足すだけ。
pub(crate) fn expert_gate_record(hits: u32, lookups: u32) {
    let Some(g) = expert_gate() else {
        return;
    };
    if let Ok(mut s) = g.lock() {
        s.record(hits, lookups);
    }
}

/// `DENNO_MOE_PAR` の値の解釈 (純粋関数)。既定 (未設定) は有効、`"0"` のときだけ無効。
///
/// env を読む部分から切り離してあるのは、テストが環境変数を触らずに
/// 両方の分岐を確かめられるようにするため。
pub(crate) fn moe_par_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0"))
}

/// expert ループの並列化が有効か (`DENNO_MOE_PAR`, 既定 1)。プロセスで 1 回だけ読む。
pub(crate) fn moe_par_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_par_from_value(std::env::var("DENNO_MOE_PAR").ok().as_deref()))
}

/// P64: 1 層ぶんの expert ループを **1 つの rayon 並列領域**にまとめる新経路の
/// ゲート (`DENNO_EXPERT_LOOP`, 既定 off)。プロセスで 1 回だけ読む。
///
/// off のときの経路は従来と同一 (`moe_ffn_quant_full_opts` の `loop_on=false`)。
pub(crate) fn expert_loop_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = expert_loop_from_value(std::env::var("DENNO_EXPERT_LOOP").ok().as_deref());
        if on {
            // 1 回だけ出す (毎層の eprintln は decode の邪魔になる)。
            eprintln!("[flash] expert loop: ON (1 層 1 並列領域 / DENNO_EXPERT_LOOP)");
        }
        on
    })
}

/// `DENNO_EXPERT_LOOP` の値解釈。
///
/// **既定 (未設定・空) は off**。`1` / `on` / `true` のときだけ on にする
/// (`DENNO_MOE_PAR` のような「0 以外は on」にしない。誤設定で新経路に入ると
/// 比較の基準が無くなるため)。
pub(crate) fn expert_loop_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("on") | Some("ON") | Some("true")
    )
}

/// P74: 平坦化経路のテスト用上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
///
/// `CPU_GEMV_PAR_OVERRIDE` / `ROUTER_GPU_OVERRIDE` と同じ流儀で `OnceLock` より
/// 優先する。プロセス全体で 1 個なので、触るテストは `moe_test_guard()` を取る。
pub(crate) static EXPERT_FLAT_DECODE_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// P70: decode の expert 寄与を **expert 横断の行タスク**に平坦化する新経路の
/// ゲート (`DENNO_EXPERT_FLAT_DECODE`, 既定 on = P74)。プロセスで 1 回だけ読む。
///
/// off のときの経路は従来と同一 ([`expert_contribs_par_into`] = expert 1 本を
/// 1 タスクとして並列化)。on のときは「expert × 行列 × 行チャンク」を 1 本の
/// タスク列にして 2 位相 (gate+up → down) で回す。
pub(crate) fn expert_flat_decode_enabled() -> bool {
    match EXPERT_FLAT_DECODE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on =
            expert_flat_decode_from_value(std::env::var("DENNO_EXPERT_FLAT_DECODE").ok().as_deref());
        if on {
            // 1 回だけ出す (毎層の eprintln は decode の邪魔になる)。
            eprintln!(
                "[flash] expert flat decode: ON (expert 横断の行タスク / \
                 DENNO_EXPERT_FLAT_DECODE)"
            );
        }
        on
    })
}

/// `DENNO_EXPERT_FLAT_DECODE` の値解釈 (純粋関数)。**既定は on**。
///
/// main の実測 (2026-10-31 12:40 JST、v3 + gdn 並列 + attn GQA + fused gemv の上で
/// 118 tok を 3 ラン): off 106.0 -> on **103.8 ms/token**、wall 16.48 -> 16.21 s、
/// md5 は `3b77a3b996fd` で不変。`decode_flat_matches_expert_parallel` が
/// 従来経路とのビット一致を固定しているので既定を on にした。
/// 逃げ道は `0` / `off` / `false` / `no` (それ以外・未設定は on)。
pub(crate) fn expert_flat_decode_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0") | Some("off") | Some("false") | Some("no") => false,
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// task 153: VRAM ミス expert の寄与を行まで平坦化して並列化する
// ---------------------------------------------------------------------------
//
// IQ の層一括 GEMV (task 144) が on のとき、VRAM に載っていない expert だけを
// CPU で計算する。ヒット率が高い (実測 0.95〜0.99) のでミスは 1 層あたり 0〜3 本に
// なり、`expert_contribs_par_into` (1 expert = 1 rayon タスク) では**1 スレッド
// しか働かない**。実測 (lab t152c def23): experts の cpu が 14k で 5.6 ms/token、
// 3.9k で 8.9 ms/token (台帳の最大のホスト項)。
//
// ここではミス集合を行チャンクまで割って 1 本のタスク列にする
// ([`expert_contribs_flat_decode_into`] の再利用)。1 行の計算は
// `gemv_rows_seq` = `gemv_seq` と同じカーネル・同じ順なので**ビット一致**する。

/// task 153: ミス expert の寄与を行並列にするか (`DENNO_EXP_MISS_FLAT`)。
///
/// **既定 on** (`0` / `off` / `false` / `no` で従来の expert 単位の並列へ戻る)。
pub(crate) fn expert_miss_flat_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0") | Some("off") | Some("false") | Some("no") => false,
        _ => true,
    }
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
pub(crate) static EXPERT_MISS_FLAT_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// task 153: [`expert_miss_flat_from_value`] のゲート (プロセスで 1 回だけ env を読む)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn expert_miss_flat_enabled() -> bool {
    match EXPERT_MISS_FLAT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        expert_miss_flat_from_value(std::env::var("DENNO_EXP_MISS_FLAT").ok().as_deref())
    })
}

/// task 67: 選択 expert の mmap 先読み (`MADV_WILLNEED`) の on/off
/// (`DENNO_EXPERT_PREFETCH`)。**既定 off** (task 67 の実測で on が遅かった)。
///
/// 数値には影響しない (ページを先に page cache へ読むだけ)。ただし
/// `MADV_WILLNEED` は 1 呼び出しぶん (921 KB) の readahead を**呼び出しスレッドで
/// 同期的に**起こすので、8 スレッドがフォルトで並列に読む従来経路より遅い。
/// dev コンテナの実測は `REPORT_task67.md`: 温 page cache で +8 %、
/// 冷 page cache (fadvise で落とした後) で +7 %。
/// `DENNO_EXPERT_PREFETCH=1` で従来 (常時先読み) に戻せる。
pub(crate) static EXPERT_PREFETCH_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// `DENNO_EXPERT_PREFETCH` の値解釈 (純粋関数)。既定 off、`1` / `on` / `true` で on。
pub(crate) fn expert_prefetch_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("on") | Some("ON") | Some("true")
    )
}

pub(crate) fn expert_prefetch_enabled() -> bool {
    match EXPERT_PREFETCH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        expert_prefetch_from_value(std::env::var("DENNO_EXPERT_PREFETCH").ok().as_deref())
    })
}

/// task 67: expert テンソルの mmap に `MADV_HUGEPAGE` を付けるか
/// (`DENNO_EXPERT_HUGEPAGE`)。**既定 off**。
///
/// dev コンテナ (kernel 6.17 / overlayfs) では read-only のファイル写像に対する
/// THP が効かず (`MADV_COLLAPSE` が EINVAL・`FilePmdMapped` が 0)、実測の効果が
/// 無かったので既定 off のままにした (`REPORT_task67.md`)。RO のファイル THP を
/// サポートするカーネルでは `=1` で試せる。
pub(crate) static EXPERT_HUGEPAGE_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// `DENNO_EXPERT_HUGEPAGE` の値解釈 (純粋関数)。既定 off、
/// `1` / `on` / `true` のときだけ on (`DENNO_EXPERT_LOOP` と同じ厳しさ)。
pub(crate) fn expert_hugepage_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("on") | Some("ON") | Some("true")
    )
}

pub(crate) fn expert_hugepage_enabled() -> bool {
    match EXPERT_HUGEPAGE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        expert_hugepage_from_value(std::env::var("DENNO_EXPERT_HUGEPAGE").ok().as_deref())
    })
}

// ===========================================================================
// task 120: decode (m=1) の expert 剪定ゲートと重み統計。
//
// 方針 (オペレータ 2026-11-03): 速度をビット一致より優先できるが、剪定は丸めでなく
// 挙動を変えるので、**測定した品質数字が無い限り既定は off のまま**。ここで作るのは
// スイッチと測定道具だけ。ゲートが全部 off のときは出力が親経路と to_bits 一致する。
// ===========================================================================

/// `DENNO_EXPERT_TOPK` の値解釈 (純粋関数)。
///
/// top-k のうち確率上位 `n` 個だけ残す。有効範囲は `1 <= n <= k`
/// (`k` = `expert_used_count`)。**未設定 / 空 / 非数 / 範囲外は無効 = off**。
/// `n == k` は有効だが切る相手が無い (経路も値も親経路と同じ)。
pub(crate) fn expert_topk_from_value(v: Option<&str>, k: usize) -> Option<usize> {
    let s = v.map(str::trim).filter(|s| !s.is_empty())?;
    let n: usize = s.parse().ok()?;
    if (1..=k).contains(&n) {
        Some(n)
    } else {
        None
    }
}

/// `DENNO_EXPERT_MIN_W` の値解釈 (純粋関数)。
///
/// 再正規化後の重みが `x` 未満の expert を落とす。**0 / 未設定 / 空 / 負 /
/// NaN / inf / 非数は無効 = off**。有限の正数だけ有効 (1.0 は「最高重みの
/// 1 個を常に残す」の意味で有効)。
pub(crate) fn expert_min_w_from_value(v: Option<&str>) -> Option<f32> {
    let s = v.map(str::trim).filter(|s| !s.is_empty())?;
    let x: f32 = s.parse().ok()?;
    if x.is_finite() && x > 0.0 {
        Some(x)
    } else {
        None
    }
}

/// `DENNO_EXPERT_W_STATS` の値解釈 (純粋関数)。`DENNO_EXPERT_LOOP` と同じ厳しさ
/// (`1` / `on` / `ON` / `true` だけ on)。
pub(crate) fn expert_w_stats_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("on") | Some("ON") | Some("true")
    )
}

/// 測定 harness 専用の上書き (env より優先)。品質 harness
/// (`tests/expert_prune_quality.rs`) が 1 プロセスで設定を切り替えるためのもの。
/// `EXPERT_FLAT_DECODE_OVERRIDE` と同じ流儀。
pub(crate) static EXPERT_PRUNE_OVERRIDE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) static EXPERT_PRUNE_TOPK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) static EXPERT_PRUNE_MIN_W: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) static EXPERT_PRUNE_W_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 測定 harness が切り替える設定の 1 つぶん。
///
/// - `topk`: `DENNO_EXPERT_TOPK` 相当 (`None` = off)
/// - `min_w`: `DENNO_EXPERT_MIN_W` 相当 (`None` = off)
/// - `w_stats`: `DENNO_EXPERT_W_STATS` 相当
///
/// env で k (n_expert_used) 超過の topk を渡しても剪定は起こらない
/// (残る相手が無いので no-op)。値の妥当性は env の純粋関数と同じ規則で
/// 適用側が解釈する。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertPruneMeasurement {
    pub topk: Option<usize>,
    pub min_w: Option<f32>,
    pub w_stats: bool,
}

/// 現在の剪定ゲート (`topk`, `min_w`)。プロセスで 1 回だけ env を読む。
///
/// 上書きが効いているときは env を見ない。`k` は呼び出し側の `n_expert_used`
/// で、env 値の範囲判定に使う (キャッシュは生の解釈ではなく k 込みの結果。
/// 1 プロセス 1 モデルが前提の既存ゲートと同じ前提)。
#[inline]
pub(crate) fn expert_prune_cfg(k: usize) -> (Option<usize>, Option<f32>) {
    use std::sync::atomic::Ordering;
    if EXPERT_PRUNE_OVERRIDE.load(Ordering::Relaxed) {
        let t = EXPERT_PRUNE_TOPK.load(Ordering::Relaxed);
        let w = EXPERT_PRUNE_MIN_W.load(Ordering::Relaxed);
        return (
            (t != 0).then(|| t as usize),
            // min_w は有限の正数だけ保存するので bits != 0 が「あり」を意味する。
            (w != 0).then(|| f32::from_bits(w)),
        );
    }
    static ENV: std::sync::OnceLock<(Option<usize>, Option<f32>)> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        (
            expert_topk_from_value(
                std::env::var("DENNO_EXPERT_TOPK").ok().as_deref(),
                k,
            ),
            expert_min_w_from_value(std::env::var("DENNO_EXPERT_MIN_W").ok().as_deref()),
        )
    })
}

/// 重み統計の収集が on か。off では呼び出し側は分岐 1 回で終わる。
#[inline]
pub(crate) fn expert_w_stats_on() -> bool {
    use std::sync::atomic::Ordering;
    if EXPERT_PRUNE_OVERRIDE.load(Ordering::Relaxed) {
        return EXPERT_PRUNE_W_STATS.load(Ordering::Relaxed);
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        expert_w_stats_from_value(std::env::var("DENNO_EXPERT_W_STATS").ok().as_deref())
    })
}

/// 重みの rank 別平均を取る深さ (rank 1..=10)。
pub(crate) const EXPERT_W_RANKS: usize = 10;

/// 1 リクエストぶんの重み統計の蓄積。
///
/// `calls` = 層呼び出し数 (`moe_ffn_quant_impl` 1 回 = decode 1 トークンの 1 層)。
/// `rank_sum[r]` / `rank_n[r]` は「その層呼び出しの再正規化後の重みを降順に並べた
/// ときの rank r (1 始まり) の値」の和と、rank r を持つ層呼び出し数 (expert を
/// r 個より多く残した呼び出しだけが rank r を持つ)。
#[derive(Debug)]
pub(crate) struct ExpertWAccum {
    pub(crate) calls: u64,
    pub(crate) experts_total: u64,
    pub(crate) rank_sum: [f64; EXPERT_W_RANKS],
    pub(crate) rank_n: [u64; EXPERT_W_RANKS],
}

pub(crate) static EXPERT_W_ACCUM: std::sync::Mutex<ExpertWAccum> = std::sync::Mutex::new(ExpertWAccum {
    calls: 0,
    experts_total: 0,
    rank_sum: [0.0; EXPERT_W_RANKS],
    rank_n: [0; EXPERT_W_RANKS],
});

/// 1 層呼び出しぶん記録する。`weights_desc` は再正規化後の重みの降順
/// (`moe_ffn_quant_impl` のゲート適用後の列)。
pub(crate) fn expert_w_stats_record(weights_desc: &[f32], n_run: usize) {
    if !expert_w_stats_on() {
        return;
    }
    let mut a = EXPERT_W_ACCUM
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    a.record(weights_desc, n_run);
}

/// 蓄積済み統計の読み取り専用スナップショット (測定 harness 用)。
#[derive(Clone, Debug)]
pub struct ExpertWStatsSnapshot {
    pub(crate) calls: u64,
    pub(crate) experts_total: u64,
    pub(crate) rank_sum: [f64; EXPERT_W_RANKS],
    pub(crate) rank_n: [u64; EXPERT_W_RANKS],
}

/// serve のリクエスト終端から呼ぶ: `DENNO_EXPERT_W_STATS=1` (または測定 harness
/// の上書き) で統計が溜まっていれば 1 行出して蓄積を消す。off では分岐 1 回で
/// 何もしない。蓄積が空 (decode を 1 歩も走っていない要求) のときも出さない。
pub fn expert_w_stats_request_end() {
    if !expert_w_stats_on() {
        return;
    }
    if let Some(s) = take_expert_w_stats() {
        eprintln!("[WSTATS] {}", s.line());
    }
}

/// task 67: CPU expert GEMV の作業域 (`act` / `up` / `h`) をスレッドローカルで
/// 使い回すか (`DENNO_EXPERT_SCRATCH`)。**既定 off**。
///
/// dev コンテナの実測では効果が雑音に埋もれた (1 トークンの確保は 7.4 MB で、
/// glibc のアリーナ再利用が効くため)。`=1` で使い回し経路になる。数値は変わらない
/// (使う前に必ずゼロ埋めする)。詳細は [`with_expert_scratch`]。
pub(crate) static EXPERT_SCRATCH_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// `DENNO_EXPERT_SCRATCH` の値解釈 (純粋関数)。既定 off、`1` / `on` / `true` で on。
pub(crate) fn expert_scratch_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("on") | Some("ON") | Some("true")
    )
}

pub(crate) fn expert_scratch_enabled() -> bool {
    match EXPERT_SCRATCH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        expert_scratch_from_value(std::env::var("DENNO_EXPERT_SCRATCH").ok().as_deref())
    })
}

/// テスト用: task 67 の各ゲートが実際に通った回数 (分岐が発動した証拠)。
#[cfg(test)]
pub(crate) static EXPERT_PREFETCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: `MADV_HUGEPAGE` を投げたテンソル数。
#[cfg(test)]
pub(crate) static EXPERT_HUGEPAGE_ADVISES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: スレッドローカルの作業域を使い回した回数 (確保し直した回数ではない)。
#[cfg(test)]
pub(crate) static EXPERT_SCRATCH_REUSES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// [`moe_ffn_quant`] の本体。`par` は expert ループの並列化 on/off
/// (env を読まずに両経路を比べられるよう引数で受ける)。
///
/// テスト専用: 本番の入口は [`moe_ffn_quant_with`] (経路も引数で受ける)。
#[cfg(test)]
pub(crate) fn moe_ffn_quant_with_par(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    par: bool,
) -> Result<Vec<f32>> {
    // 既存のテスト用入口は「CPU 経路」を固定する (env の状態に依存させない)。
    moe_ffn_quant_with(p, lw, x, cache, par, Route::Cpu)
}

/// [`moe_ffn_quant`] の本体。`par` は expert ループの並列化 on/off、
/// `route` は expert の計算経路 (P39)。
///
/// P45: `Route::Vram` でも、[`expert_gate_decide`] が低ヒット率と判断したトークンは
/// CPU で計算する (`use_gpu == false`)。CPU と GPU は同じバイト列に同じ式を通すので、
/// どのトークンがどちらに落ちても最終出力は CPU 経路と一致する (実測 max_abs 0.000e0)。
///
/// テスト専用: 本番の入口は [`moe_ffn_quant`] / [`moe_ffn_quant_tapped`]
/// (RouterTopK を送るかと役割 RMS を取るかを引数で受ける)。
#[cfg(test)]
pub(crate) fn moe_ffn_quant_with(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    par: bool,
    route: Route,
) -> Result<Vec<f32>> {
    // 既存のテスト用入口は**従来経路を固定**する (`DENNO_EXPERT_LOOP` に依存させない)。
    // P64 の新経路は [`moe_ffn_quant_full_opts`] を直接呼ぶテストで確かめる。
    moe_ffn_quant_full_opts(p, lw, x, cache, par, route, None, true, false, IvView::none())
}

/// [`moe_ffn_quant_with`] の本体 (+ P54 の役割別 RMS / P56 の RouterTopK)。
///
/// `emit_router` が false のときは RouterTopK を送らない (prefill の非末尾
/// トークン)。役割 RMS と RouterTopK はどちらも読むだけなので、数値経路は
/// どちらの引数にも依らない。
pub(crate) fn moe_ffn_quant_full(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    par: bool,
    route: Route,
    roles: Option<&mut RoleRms>,
    emit_router: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_full_opts(
        p,
        lw,
        x,
        cache,
        par,
        route,
        roles,
        emit_router,
        expert_loop_enabled(),
        iv,
    )
}

/// [`moe_ffn_quant_full`] の本体。`loop_on` は P64 の新経路 (1 層 1 並列領域) を
/// env を読まずに選ぶための引数。
///
/// **`loop_on == false` のときの経路は従来と同一** (下の `if loop_new` が丸ごと
/// 飛ばされるだけで、`else if parallel` 以降の文は 1 行も変えていない)。
pub(crate) fn moe_ffn_quant_full_opts(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    par: bool,
    route: Route,
    roles: Option<&mut RoleRms>,
    emit_router: bool,
    loop_on: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_impl(
        p,
        lw,
        x,
        cache,
        par,
        route,
        roles,
        emit_router,
        loop_on,
        expert_flat_decode_enabled(),
        iv,
    )
}

// ---------------------------------------------------------------------------
// task 124: shared expert の gate/up を router/expert の CPU 仕事と重ねる
// ---------------------------------------------------------------------------
//
// 実測 (GPU ラボ t124a、swift15-abl IQ3_S 128 tok、14T、DENNO_MIX_GPU=1):
// decode off 86 -> on 84 ms/tok (-2.9%)、md5 は全条件で SAME
// (a:82ab0c06 / b:624e0d4d)、prefill 2.25 -> 2.25 s、errors 0。
// RAYON_NUM_THREADS=13 では 83 ms/tok (-4.0% vs off の 14T)。
// [q4prof] decode の shared は 826.7 -> 308.4 ms/128tok (gate/up が CPU 仕事の
// 裏に隠れた)。router は 956.2 -> 1142.3 = +1.45 ms/tok になるが、これは
// 積むときの発行コスト (~30 us/層) を router の区間が抱えるため (wall time は
// 上の decode の差がすべて)。

/// `DENNO_SHARED_OVERLAP` の生値 → 重ね打ちの on/off (純関数)。**既定 on**、
/// `0` / `off` / `false` で従来の順序 (router → expert → shared) に戻る。
pub(crate) fn shared_overlap_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// shared expert の重ね打ちを使うか (`DENNO_SHARED_OVERLAP`、既定 on)。
///
/// プロセスで 1 回だけ env を読む。off のときの追加コストは atomic load 1 回。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn shared_overlap_enabled() -> bool {
    match SHARED_OVERLAP_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        shared_overlap_from_value(std::env::var("DENNO_SHARED_OVERLAP").ok().as_deref())
    })
}

/// task 124: shared expert の gate/up を先に GPU へ積むハンドル。
///
/// `cuda` feature の有無はこの中で吸収する (`begin` は常に「積まなかった」を
/// 返すので、feature 無しでも呼び出し側のコードは同じ形でコンパイルできる)。
pub(crate) struct SharedExpertOverlap {
    #[cfg(feature = "cuda")]
    pub(crate) pending: crate::cuda::SharedPending,
    /// 積んだときの gate/up の出力長 (join の検査用。0 = 積んでいない)
    pub(crate) lens: [usize; 2],
    /// task 187: down までデバイスで積んだか (`DENNO_DEC_SHARED_DEV`)。
    pub(crate) full: bool,
    /// down の出力長 (full のときだけ意味がある)。
    pub(crate) down_n: usize,
}

/// task 155: `DENNO_SHARED_TAIL` の値解釈 (純関数・テスト用)。**既定 on**、
/// `0` / `off` / `false` で従来の順序 (down → gate_inp の 2 往復) に戻る。
pub(crate) fn shared_tail_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// task 155: shared expert の **down** をサイドストリームへ先に積むか
/// (`DENNO_SHARED_TAIL`、既定 on)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn shared_tail_enabled() -> bool {
    match SHARED_TAIL_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| shared_tail_from_value(std::env::var("DENNO_SHARED_TAIL").ok().as_deref()))
}

/// task 155: shared expert の **down** を先に GPU (サイドストリーム) へ積むハンドル。
///
/// 狙いは層あたりの往復を 1 回減らすこと: down (n_embd 行) を非同期で積み、
/// その完了を待つ間に `gate_inp_shexp` (1 行) を null ストリームで回す。
/// 積めなかったとき (型が未対応 / GPU 不発 / 予算外) は `finish` が `false` を
/// 返すので、呼び出し側は従来どおり同期の `gemv_label` に落ちること。
///
/// 実測 (GPU ラボ t155a、14,139、plain): shared 2.55 → 2.32 ms/tok (-0.23)、
/// null ストリームの sync 1.12 → 0。**md5 は SAME** (選ばれた GEMV カーネルは
/// CPU の逐次加算と同じ値だった)。
pub(crate) struct SharedDownOverlap {
    #[cfg(feature = "cuda")]
    pub(crate) pending: crate::cuda::SharedPending,
    /// 期待する出力長 (join の検査用。0 = 積んでいない)
    pub(crate) n: usize,
}

/// [`moe_ffn_quant_full_opts`] の本体。`flat_decode` は P70 の expert 横断平坦化
/// (env を読まずにテストから選べるように引数で受ける)。
///
/// **`flat_decode == false` のときの経路は従来と同一** (下の `if flat_new` が
/// 丸ごと飛ばされるだけで、`else if parallel` 以降の文は 1 行も変えていない)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_quant_impl(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&ExpertCacheSet>,
    par: bool,
    route: Route,
    mut roles: Option<&mut RoleRms>,
    emit_router: bool,
    loop_on: bool,
    flat_decode: bool,
    iv: IvView<'_>,
) -> Result<Vec<f32>> {
    let t_router = std::time::Instant::now();
    // task 278: 層 0 の入口 = トークンの境目。`DENNO_SEG_PROF` の積算を 1 行で出す
    // (off は分岐 1 回で `enabled()` の atomic load だけ)。
    if lw.il == 0 {
        if let Some(line) = crate::model::seg_prof::flush() {
            eprintln!("{line}");
        }
    }
    let e = p.n_embd;
    let ne = p.n_expert;
    let k = p.n_expert_used;
    let ff = p.n_ff_exp;

    // task 192: `DENNO_DEC_ROUTER_EARLY` — ルータ GEMV を共有 expert の enqueue
    // より**先に**積む。ルータの同期は自分の GEMV の完了待ち (lab t187c で
    // 1 層 12.7 us) なので、積むのが早いほど待ちが短くなる。数値は 1 ビットも
    // 変わらない (同じカーネル・同じ引数・同じ入力)。
    // decode だけに効かせる (prefill は `gdn_gpu_win` が閉じている)。
    #[cfg(feature = "cuda")]
    let router_early = crate::cuda::dec_router_early_on()
        && gdn_gpu_win::active()
        && lw.gate_inp.router_gpu_ok()
        && crate::cuda::maybe_gpu_router_gemv_begin(
            lw.gate_inp.uid(),
            lw.gate_inp.blocks().as_slice(),
            lw.gate_inp.k(),
            lw.gate_inp.n(),
            x,
        );
    #[cfg(not(feature = "cuda"))]
    let router_early = false;

    // task 124: shared expert の gate/up は `x` だけで決まるので、router と
    // 選択 expert の CPU 仕事を始める前に GPU へ積む (join は下の 5. で 1 回)。
    // task 138: この発行は台帳では shared に数える。
    // task 217: shared expert を切った層では gate/up の先積みもしない。
    let mut shed_ov = if iv.block_off(lw.il, IvBlock::Shared) {
        SharedExpertOverlap::none()
    } else {
        let _l = q4ledger::enter(q4ledger::SHARED);
        // task 187: 台帳の副区間 (shared gate/up の enqueue)。
        let _s = q4ledger::sub(q4ledger::SUB_SH_BEGIN);
        SharedExpertOverlap::begin(lw, x)
    };

    // task 138: 台帳の区間 (router)。gate_inp の GEMV から top-k の確定まで。
    let _l_router = q4ledger::enter(q4ledger::ROUTER);
    // task 192: router 1 回の操作別内訳 (`DENNO_GDN_DEC_KPROF` の平均表)。
    // 何を待ったかは `sync` 行の host に出る。
    #[cfg(feature = "cuda")]
    let _kp_router = crate::cuda::dec_kprof::window(
        crate::cuda::dec_kprof::ROUTER,
        lw.il,
        crate::cuda::dec_kprof::cur_pos(),
        0,
    );

    // 1. ルータ
    let mut probs = vec![0.0f32; ne];
    #[allow(unused_mut)]
    let mut router_done = false;
    #[cfg(feature = "cuda")]
    if router_early {
        router_done = crate::cuda::maybe_gpu_router_gemv_finish(&mut probs);
    }
    if !router_done {
        lw.gate_inp
            .gemv_label("gate_inp", x, &mut probs)
            .map_err(|err| one(format!("層 {}: ffn_gate_inp: {err}", lw.il)))?;
    }
    #[cfg(feature = "cuda")]
    let t_softmax = std::time::Instant::now();
    softmax(&mut probs);
    #[cfg(feature = "cuda")]
    crate::cuda::dec_kprof::note_since("softmax", t_softmax);

    // 2. top-k (index 昇順で返る)
    #[cfg(feature = "cuda")]
    let t_topk = std::time::Instant::now();
    // task 217: 無効 expert は候補から外す (設定が無ければ従来と 1 ビット同じ)。
    // 残った expert から top-k を選び、下の再正規化は選んだ集合の和で割る
    // = 「残りだけで正規化する」というルータの通常の規則になる。
    let expert_mask_il = iv.expert_mask(lw.il);
    let mut sel = match expert_mask_il {
        Some(m) => select_topk_mask(&probs, k, m),
        None => {
            // task 285: decode (1 行) の top-k は挿入法 (`select_topk_into_fast`)。
            // 選択規則 (値の降順・同値は小さい index・出力は index 昇順) は
            // `select_nth_unstable_by` 版と同一なので値は 1 ビットも変わらない。
            // ne=512 / k=8 の相対マイクロベンチ (task 195) は 4.2 -> 0.68 us/行。
            if dec_router_topk_fast_on() {
                #[cfg(test)]
                DEC_ROUTER_TOPK_FAST_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mut out = vec![0usize; k];
                select_topk_into_fast(&probs, k, &mut out);
                out
            } else {
                select_topk(&probs, k)
            }
        }
    };
    #[cfg(feature = "cuda")]
    crate::cuda::dec_kprof::note_since("topk", t_topk);

    // task 120: decode (m=1) の expert 剪定ゲート (**既定 off**)。
    //
    // - `DENNO_EXPERT_TOPK=n`: top-k のうち確率上位 n 個だけ残す (1 <= n <= k。
    //   範囲外の値は無効 = off)。n == k は切る相手が無いので何もしない。
    // - `DENNO_EXPERT_MIN_W=x`: 再正規化後の重みが x 未満の expert を落とす
    //   (0 / 未設定 / 無効値は off)。**最低 1 個は必ず残す** (最も重い expert)。
    //
    // 適用順: TOPK で切る → 残りで再正規化 → その重みで MIN_W を判定して落とす
    // → 残りでもう一度再正規化。落とす相手が 1 個も無いときは 2 回目の再正規化を
    // **丸ごと飛ばす** (重みの和が 1.0 に丸まるとは限らないので、やり直すと親経路
    // からビットが動く)。残す集合は常に index 昇順に並べ替えるので、蓄積の丸め
    // 順序 (index 昇順の逐次) は従来と同じ。ゲートが 1 つも効いていないときは
    // 下の if に入らないので、sel / weights / sum は親経路と 1 ビットも違わない。
    let (topk_n, min_w_x) = expert_prune_cfg(k);
    if let Some(n) = topk_n {
        if n < sel.len() {
            // 確率上位 n 個を残す。安定ソートなので同点は元の並び (index 昇順) の
            // 先頭側が残る = select_topk と同じ「同点は小さい index」の規則。
            let mut by_prob = sel.clone();
            by_prob.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            by_prob.truncate(n);
            by_prob.sort_unstable();
            sel = by_prob;
        }
    }

    // 3. 再正規化 (選択 index 昇順に逐次加算した和で割る)
    let k_run = sel.len();
    let mut weights = vec![0.0f32; k_run];
    let mut sum = 0.0f32;
    for &i in &sel {
        sum += probs[i];
    }
    // task 217: 選んだ expert の確率が全部 0 (softmax のアンダーフロー) だと
    // 0/0 = NaN になる。介入で支配的な expert を切ると起こりうるので、
    // そのときだけ一様に分ける (通常の経路では和が正なので 1 ビットも変わらない)。
    if sum.is_finite() && sum > 0.0 {
        for (t, &i) in sel.iter().enumerate() {
            weights[t] = probs[i] / sum;
        }
    } else {
        let u = 1.0f32 / k_run as f32;
        for w in weights.iter_mut() {
            *w = u;
        }
    }
    if let Some(x) = min_w_x {
        // 重みが x 未満の expert を落とす (最低 1 個は残す)。kept は昇順走査の
        // filter なので index 昇順のまま。
        let kept: Vec<usize> = (0..k_run).filter(|&t| weights[t] >= x).collect();
        if kept.len() < k_run {
            let kept: Vec<usize> = if kept.is_empty() {
                // 全員が x 未満: 最も重い 1 個だけ残す (同点は先に見た小さい index)。
                let mut best = 0usize;
                for t in 1..k_run {
                    if weights[t] > weights[best] {
                        best = t;
                    }
                }
                vec![best]
            } else {
                kept
            };
            let mut sum2 = 0.0f32;
            for &t in &kept {
                sum2 += weights[t];
            }
            let renorm: Vec<f32> = kept.iter().map(|&t| weights[t] / sum2).collect();
            sel = kept;
            weights = renorm;
        }
    }
    // P54: router の活性 = **実際に足し込む expert** の確率和。剪定が無いときは
    // 従来どおり再正規化の分母そのもの (同じ集合を同じ順序で足した和 = 同じビット
    // 列)。剪定で集合が変わったときだけ残った expert で足し直す (役割テレメと
    // RouterTopK は実際に使った expert を示す)。
    let router_sum = if sel.len() == k {
        sum
    } else {
        let mut s = 0.0f32;
        for &i in &sel {
            s += probs[i];
        }
        s
    };
    // P54: router の活性は「実際に足し込む expert の確率和」。剪定が無いときは
    // 再正規化の分母 `sum` そのもの (追加の走査は無い)。スカラーなので RMS は値そのもの。
    if let Some(r) = roles.as_deref_mut() {
        r.router = router_sum;
    }
    // task 120: `DENNO_EXPERT_W_STATS=1` (または測定 harness の上書き) のときだけ、
    // この層呼び出しの再正規化後の重み (降順) と実行 expert 数を積む。off では
    // この分岐で終わる (蓄積も確保も無い)。リクエスト終端で 1 行にまとめて出す
    // ([`expert_w_stats_request_end`])。
    if expert_w_stats_on() {
        let mut desc = weights.clone();
        desc.sort_by(|a, b| b.total_cmp(a));
        expert_w_stats_record(&desc, sel.len());
    }

    // P56: ルータ結果 (index 昇順の expert と再正規化後の重み) を 1 フレームで送る。
    // 送るのは decode (1 呼び出し = 1 トークン) と、prefill のチャンク末尾の
    // トークンだけ (`emit_router` は呼び出し側が決める)。
    if emit_router {
        emit_router_topk(lw.il, &sel, &weights);
        // task 220: 介入で切った expert のうち、無効化が無ければ top-k に入っていたもの
        // (既定 off。off はゲートの分岐 1 個で、余分な top-k は回さない)。
        if let Some(m) = expert_mask_il {
            emit_expert_skip(lw.il, &probs, k, &sel, m);
        }
    }

    // task 266: 次のトークンの先読み用に、この層の選択を記録する
    // (ゲート off では atomic load 1 回。値には触れない)。
    crate::model::exp_prefetch::note_sel(lw.il, &sel);
    // task 862: decode の選択を page hint の「温かい作業集合」に足す
    // (`DENNO_EXPERT_PAGE_HINT` のときだけ。off は atomic load 1 回)。
    // 逐次 prefill もこの関数を通るので、prefill の中では数えない。
    if !prefill::in_prefill_seq() {
        crate::model::exp_page_hint::note_decode(lw.il, &sel);
    }

    // task 850: 層間の共起表を更新し、層 L+1 の予測 expert を背景スレッドへ渡す
    // (`DENNO_EXPERT_MARKOV_PREFETCH=1` のときだけ。off は atomic load 1 回)。
    crate::model::qwen4exp::markov_note_and_prefetch(lw.il, &sel);

    // task 138: router の区間はここまで (expert の CPU 仕事は台帳の外 = other)。
    drop(_l_router);
    // task 192: kprof の窓もここで閉じる (expert 段の操作を混ぜない)。
    #[cfg(feature = "cuda")]
    drop(_kp_router);

    // task 155: router と experts の間の下ごしらえ (使用回数の記録 / ゲート判定 /
    // 出力の確保 / 経路フラグ) を台帳の `loop` 行に分ける。以前は `step` の残差に
    // 紛れていた部分。
    let _l_loop = q4ledger::enter(q4ledger::LOOP);

    // task 146: (層, expert) の使用回数を積む (起動時の事前ロード用)。
    // ゲート off では `stats_path()` の atomic load 1 回で返る。
    // decode 1 トークン = 48 層 x 10 本 = 480 回の `fetch_add` (Relaxed)。
    crate::model::exp_stats::observe_layer(p.n_layer, p.n_expert, lw.il, &sel);

    // task 217: routed expert を切った層は選択後の先読み・寄与計算・GPU 経路を
    // 丸ごと飛ばす (out は 0 のまま)。
    let mut out = vec![0.0f32; e];
    let t_experts = std::time::Instant::now();
    // task 145: 台帳の区間 (experts)。選択 expert の寄与計算だけ。
    let _l_experts = q4ledger::enter(q4ledger::EXPERTS);
    if !iv.block_off(lw.il, IvBlock::Moe) {
        // 3.5 選択 expert の重みを先に読み始める (I/O と計算を重ねる)。
        //     1 トークンで 48 層 × 11 行列 × 921 KB を触るので、フォルト待ちが支配的になる。
        //     キャッシュ経路は pread で必要ぶんだけ読むので、mmap のページフォルトを
        //     先に起こす意味が無い (`prefetch()` 自体は無効化も変更もしない)。
        //
        //     task 67: **先読みは既定で off** (`DENNO_EXPERT_PREFETCH=1` で on)。
        //     `MADV_WILLNEED` は readahead を呼び出しスレッドで同期的に起こすので、
        //     8 スレッドがフォルトで並列に読むほうが速い (実測 +6-8 %)。
        //
        //     P39: VRAM 常駐経路 (`vram_mode`) では pread キャッシュを**使わない**
        //     (置き場所を 2 か所に持たない)。ヒットした expert は mmap を触らないので
        //     prefetch も投げない。ミスした expert は H2D のために同じページを
        //     順に読むので、先読みの有無で総 I/O は変わらない。
        //
        //     P45: `vram_mode` でも、直近のヒット率がしきい値未満のトークンは CPU で
        //     計算する (`use_gpu == false`)。そのトークンは mmap を直接読むので
        //     prefetch は投げる (RAM キャッシュは vram_mode では使わない = P39 と同じ)。
        let vram_mode = route == Route::Vram;
        let use_gpu = vram_mode && {
            let per = expert_h2d_bytes_one(lw, e, ff);
            expert_gate_decide(lw.il, sel.len(), per).uses_gpu()
        };
        if vram_mode && cache.is_some() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "[flash] DENNO_EXPERT_VRAM_MB を優先: DENNO_EXPERT_CACHE_MB (RAM の pread キャッシュ) は使わない"
                );
            });
        }
        let cache = if vram_mode { None } else { cache };
        if cache.is_none() && !use_gpu {
            prefetch_selected(lw, &sel);
        }

        // 4. 選択 expert を index 昇順に積む
        //
        // P-MOE-G5: 各 expert の寄与 `h_e` は x だけで決まる独立な計算なので並列化できる。
        // ただし **重み付きの足し込みは index 昇順の逐次** のままにする (並列に足すと
        // 丸めの順序が変わる)。並列化するのは `h_e` の計算だけ。
        //
        // P38: キャッシュ経路も並列化する。`ExpertCacheSet` は `&self` で共有でき、
        // ヒットは `Arc` を clone するだけ・ミスの pread はロックの外なので、旧実装のように
        // 「キャッシュを有効にすると 10 expert が逐次になる」ことがない。
        #[cfg(feature = "par")]
        let parallel = par && sel.len() > 1 && !use_gpu;
        #[cfg(not(feature = "par"))]
        let parallel = {
            // par 機能なしでは行並列も expert 並列も無い (従来と同じ逐次経路)。
            let _ = par;
            false
        };
        // P64: 新経路 (1 層 1 並列領域)。rayon が要るので `par` 機能が無いビルドと、
        // GPU 経路 (`use_gpu`) では使わない (GPU の層一括・per-expert 経路は従来のまま)。
        #[cfg(feature = "par")]
        let loop_new = loop_on && !use_gpu;
        #[cfg(not(feature = "par"))]
        let loop_new = {
            let _ = loop_on;
            false
        };

        // P70: expert 寄与の expert 横断平坦化。**mmap 経路だけ**に使う (キャッシュ経路は
        // expert 単位の pread が前提で、行チャンクに切ると同じ expert を何度も読む)。
        #[cfg(feature = "par")]
        let flat_new = flat_decode && !use_gpu && cache.is_none();
        #[cfg(not(feature = "par"))]
        let flat_new = {
            let _ = flat_decode;
            false
        };

        // task 155: `loop` 区間はここまで (experts の区間とは重ねない)。
        drop(_l_loop);
        moe_sub_add(0, t_router.elapsed());

        if loop_new {
            // P64: 1 層ぶんの寄与を 1 つの並列領域で計算し、足し込みは index 昇順の
            // 逐次 (下の `else if parallel` と 1 要素ずつ同じ順序で足す)。
            #[cfg(feature = "par")]
            {
                let hs = expert_contribs_loop(lw, &sel, x, e, ff, cache)?;
                for (t, &_ei) in sel.iter().enumerate() {
                    let pe = weights[t];
                    let hh = &hs[t * e..(t + 1) * e];
                    for j in 0..e {
                        out[j] += pe * hh[j];
                    }
                }
            }
        } else if parallel {
            #[cfg(feature = "par")]
            {
                // P70: expert 寄与を expert 横断の行タスクに平坦化する経路 (既定 off)。
                // 足し込みは下の 2 経路と同一の「index 昇順の逐次」なので、
                // 行分割の有無・タスクの割り当てに依らず出力はビット一致する。
                if flat_new {
                    let mut hs = vec![0.0f32; sel.len() * e];
                    expert_contribs_flat_decode_into(lw, &sel, x, e, ff, &mut hs, DECODE_FLAT_ROWS)?;
                    let t_scat = crate::model::exp_kprof::stamp();
                    for (t, _) in sel.iter().enumerate() {
                        let pe = weights[t];
                        let hh = &hs[t * e..(t + 1) * e];
                        for j in 0..e {
                            out[j] += pe * hh[j];
                        }
                    }
                    crate::model::exp_kprof::mark(t_scat, crate::model::exp_kprof::SCAT);
                } else if expert_scratch_enabled() {
                    let mut hs = vec![0.0f32; sel.len() * e];
                    expert_contribs_par_into(lw, &sel, x, e, ff, cache, &mut hs)?;
                    for (t, _) in sel.iter().enumerate() {
                        let pe = weights[t];
                        let hh = &hs[t * e..(t + 1) * e];
                        for j in 0..e {
                            out[j] += pe * hh[j];
                        }
                    }
                } else {
                    let hs = expert_contribs_par(lw, &sel, x, e, ff, cache)?;
                    for (t, &_ei) in sel.iter().enumerate() {
                        let pe = weights[t];
                        let hh = &hs[t];
                        for j in 0..e {
                            out[j] += pe * hh[j];
                        }
                    }
                }
            }
        } else {
            // task 67: 作業域 (act / up / h) はスレッドローカルで使い回す (既定)。
            // `DENNO_EXPERT_SCRATCH=0` のときは従来どおり呼び出しごとに確保する。
            // P45: このトークンで VRAM に載っていた expert の本数 (ゲートのサンプル)。
            let mut gpu_hits = 0u32;
            // P43: まず層一括経路 (gate 1 + up 1 + down 1 = 3 起動 / 2 同期) を試す。
            // `Some(hs)` は「選択 expert 全員の寄与が揃った」という意味で、そのときは
            // per-expert のループを回さない。足し込みは下の per-expert 経路と同一の
            // 「index 昇順の逐次」なので、丸めの順序は変わらない。
            // P45: ゲートが CPU と決めたトークンは層一括経路も呼ばない (`use_gpu` が false)。
            // task 144: IQ 型の層一括経路 (ヒットだけ GPU / ミスは CPU)。ゲートが on で
            // 3 行列とも IQ カーネルが読める型のときだけこの経路にする。
            // **旧い層一括経路は呼ばない**: そちらは `acquire` でミスを同期 H2D するので、
            // 呼ぶと毎層 20 MB を上げてから型不一致で CPU に落ちる (二重の無駄)。
            #[cfg(feature = "cuda")]
            let iq_mode = crate::model::exp_iq::iq_gpu_enabled() && iq_expert_mode(lw);
            #[cfg(not(feature = "cuda"))]
            let iq_mode = false;
            // task 278: K 型の層を 1 待ちで閉じる新経路 (`DENNO_SEG_LAYER`)。
            #[cfg(feature = "cuda")]
            let seg_mode = !iq_mode && crate::model::expert_batch::seg_layer_enabled();
            #[cfg(not(feature = "cuda"))]
            let seg_mode = false;

            // task 190 (`DENNO_DEC_MISS_FAST`): ミスの上げ直しを**層の前**に移す。
            // 上げた本はこの層の lookup でヒットになるので、CPU で計算せず GPU で回せる
            // (page fault + CPU GEMV の 1 本ぶんが消える)。予算・担当層の回転・
            // 1 本 2 MB の転送は従来と同じ (`iq_warm_misses` を前へ動かすだけ)。
            #[cfg(feature = "cuda")]
            let miss_fast_now = vram_mode && iq_mode && crate::model::exp_miss_up::fast_enabled();
            #[cfg(not(feature = "cuda"))]
            let miss_fast_now = false;
            #[cfg(feature = "cuda")]
            if miss_fast_now {
                // task 187: 台帳の副区間 (ミス expert の上げ直し = ページフォルト + H2D)。
                let _s = q4ledger::sub(q4ledger::SUB_EXP_UPLOAD);
                iq_warm_misses(lw, &sel, x.len(), e, ff, p.n_layer);
            }

            // task 187: ヒットの寄与をデバイスで和にする (`DENNO_DEC_EXP_SUMDEV`)。
            // PTX が無い / 形が合わないときは `moe_experts_iq_gpu_batched` が従来経路へ落ちる。
            #[cfg(feature = "cuda")]
            let sumdev = dec_exp_sumdev_enabled();
            #[cfg(not(feature = "cuda"))]
            let sumdev = {
                let _ = ();
                false
            };

            #[cfg(feature = "cuda")]
            let iq_batched = if use_gpu && iq_mode {
                // task 158: ミスを GPU で計算する経路 (`DENNO_DEC_MISS_UP`)。
                // 不発 (`None`) なら従来の「ヒットだけ GPU / ミスは CPU」へ落ちる。
                match moe_experts_iq_gpu_miss_up(lw, &sel, x, e, ff)? {
                    Some(v) => Some(IqBatchOut::Contribs(v.0, v.1)),
                    None => moe_experts_iq_gpu_batched(lw, &sel, &weights, x, e, ff, sumdev)?,
                }
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let iq_batched: Option<IqBatchOut> = None;

            // task 278: K 型の新経路。ヒットだけ GPU・ミスは CPU・和はデバイス。
            // 1 本でも載っていなければ `Contribs(全て None)` で呼び出し側が CPU へ。
            #[cfg(feature = "cuda")]
            let seg_batched: Option<IqBatchOut> = if use_gpu && seg_mode {
                moe_experts_seg_gpu_batched(lw, &sel, &weights, x, e, ff)?
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let seg_batched: Option<IqBatchOut> = None;
            // task 278: IQ 経路と seg 経路は同じ `IqBatchOut` を返すので、後段の
            // 足し込みを共有する (どちらも高々 1 つしか `Some` にならない)。
            let fast_batched = iq_batched.or(seg_batched);

            let batched = if use_gpu && !iq_mode && fast_batched.is_none() {
                moe_experts_gpu_batched(lw, &sel, x, e, ff)?
            } else {
                None
            };
            if let Some(IqBatchOut::Sum {
                sum,
                miss_pos,
                miss_h,
                hits,
            }) = fast_batched
            {
                // task 187: ヒットぶんはデバイスで和にしてある (重み付き・選択順)。
                // ミスだけホストで足す (ミスの無い層では従来の足し込みとビット一致)。
                gpu_hits = hits;
                out.copy_from_slice(&sum[..e]);
                if !miss_pos.is_empty() {
                    let _s_sum = q4ledger::sub(q4ledger::SUB_EXP_SUM);
                    for (i, &t) in miss_pos.iter().enumerate() {
                        let pe = weights[t];
                        let v = &miss_h[i * e..(i + 1) * e];
                        for j in 0..e {
                            out[j] += pe * v[j];
                        }
                    }
                }
            } else if let Some(IqBatchOut::Contribs(hs, hits)) = fast_batched {
                // GPU で回せた expert はその寄与、回せなかった expert は CPU で 3 行列。
                // 足し込みは今までと同じ **expert index 昇順の逐次** (同じ位置で同じ重み)。
                //
                // `vram_mode` では `parallel` が false になる (P39 の設計) ので、ミスを
                // そのまま逐次で計算すると base の並列経路より遅い。ミスだけを取り出して
                // **base と同じ expert 並列** (`expert_contribs_par`) で計算する。
                gpu_hits = hits;
                let miss_pos: Vec<usize> = (0..sel.len()).filter(|&t| hs[t].is_none()).collect();
                #[cfg(feature = "par")]
                let miss_h: Vec<Vec<f32>> = if miss_pos.is_empty() {
                    Vec::new()
                } else {
                    let ids: Vec<usize> = miss_pos.iter().map(|&t| sel[t]).collect();
                    // task 187: 台帳の副区間 (ミスの CPU 計算)。
                    let _s = q4ledger::sub(q4ledger::SUB_EXP_MISS);
                    expert_contribs_par(lw, &ids, x, e, ff, None)?
                };
                #[cfg(not(feature = "par"))]
                let miss_h: Vec<Vec<f32>> = {
                    let mut v = Vec::with_capacity(miss_pos.len());
                    let mut act = vec![0.0f32; ff];
                    let mut up = vec![0.0f32; ff];
                    let mut h = vec![0.0f32; e];
                    for &t in &miss_pos {
                        let ei = sel[t];
                        lw.gate_exps.gemv(ei, x, &mut act)?;
                        silu_inplace(&mut act);
                        lw.up_exps.gemv(ei, x, &mut up)?;
                        for r in 0..ff {
                            act[r] *= up[r];
                        }
                        lw.down_exps.gemv(ei, &act, &mut h)?;
                        v.push(h.clone());
                    }
                    v
                };
                let mut mi = 0usize;
                // task 187: 台帳の副区間 (host の重み付き足し込み)。
                let _s_sum = q4ledger::sub(q4ledger::SUB_EXP_SUM);
                for (t, &_ei) in sel.iter().enumerate() {
                    let pe = weights[t];
                    let v: &[f32] = match hs[t].as_deref() {
                        Some(v) => {
                            debug_assert_eq!(v.len(), e, "GPU の寄与長が e と違う");
                            v
                        }
                        None => {
                            let v = &miss_h[mi];
                            mi += 1;
                            v
                        }
                    };
                    for j in 0..e {
                        out[j] += pe * v[j];
                    }
                }
            } else if let Some((hs, hits)) = batched {
                // P45: 層一括経路も「何本が載っていたか」を返す (ゲートの 1 サンプル)。
                gpu_hits = hits;
                debug_assert_eq!(hs.len(), sel.len(), "寄与の本数が選択 expert と違う");
                // task 187: 台帳の副区間 (host の重み付き足し込み)。
                let _s_sum = q4ledger::sub(q4ledger::SUB_EXP_SUM);
                for (t, _) in sel.iter().enumerate() {
                    let pe = weights[t];
                    let hh = &hs[t];
                    for j in 0..e {
                        out[j] += pe * hh[j];
                    }
                }
            } else {
                with_expert_scratch(ff, e, |s| -> Result<()> {
                    let act = &mut s.act[..ff];
                    let up = &mut s.up[..ff];
                    let h = &mut s.h[..e];
                    for (t, &ei) in sel.iter().enumerate() {
                        // P39: VRAM 常駐経路。`None` = 載らない/未対応なので CPU へ落ちる。
                        // P45: `Some(true)` はヒット (積まずに回せた)、`Some(false)` はミス。
                        #[cfg(feature = "cuda")]
                        let gpu = if use_gpu {
                            moe_expert_gpu(lw, ei, x, act, up, h)?
                        } else {
                            None
                        };
                        #[cfg(not(feature = "cuda"))]
                        let gpu: Option<bool> = None;
                        if gpu == Some(true) {
                            gpu_hits += 1;
                        }

                        if gpu.is_none() {
                            match cache {
                                // 従来経路: mmap のスライスを直接 GEMV する。
                                None => {
                                    lw.gate_exps.gemv(ei, x, act)?;
                                    silu_inplace(act);
                                    lw.up_exps.gemv(ei, x, up)?;
                                    for r in 0..ff {
                                        act[r] *= up[r];
                                    }
                                    lw.down_exps.gemv(ei, act, h)?;
                                }
                                // P-MOE-G1: SSD pread + LRU スロット経路。同じ expert は読み直さない。
                                Some(c) => {
                                    let key = ExpertKey {
                                        layer: lw.il as u16,
                                        expert: ei as u16,
                                    };
                                    expert_gemv_cached(c, ExpertMatrix::Gate, &lw.gate_exps, key, x, act)?;
                                    silu_inplace(act);
                                    expert_gemv_cached(c, ExpertMatrix::Up, &lw.up_exps, key, x, up)?;
                                    for r in 0..ff {
                                        act[r] *= up[r];
                                    }
                                    expert_gemv_cached(c, ExpertMatrix::Down, &lw.down_exps, key, act, h)?;
                                }
                            }
                        } else {
                            debug_assert!(use_gpu, "GPU 経路が use_gpu=false で走った");
                        }
                        let pe = weights[t];
                        for j in 0..e {
                            out[j] += pe * h[j];
                        }
                    }
                    Ok(())
                })?;
            }
            // P45: 1 トークンぶんの実測をゲートへ積む (CPU のトークンはサンプルにしない)。
            if use_gpu {
                expert_gate_record(gpu_hits, sel.len() as u32);
            }
            // task 144: ミスした expert を上限つきで VRAM へ戻す (次のトークンのため)。
            // GPU に倒したトークンだけでなく CPU のトークンでも積む (載せないと
            // ヒット率が永久に 0 のままになる)。
            #[cfg(feature = "cuda")]
            if vram_mode && (iq_mode || seg_mode) && !miss_fast_now {
                // task 187: 台帳の副区間 (ミス expert の上げ直し = ページフォルト + H2D)。
                let _s = q4ledger::sub(q4ledger::SUB_EXP_UPLOAD);
                iq_warm_misses(lw, &sel, x.len(), e, ff, p.n_layer);
            }
        }

        // P39: `DENNO_EXPERT_GPU_CHECK=1` の比較結果 (層ごとに 1 行)。
        #[cfg(feature = "cuda")]
        if use_gpu {
            if let Some(c) = expert_vram::take_check() {
                // P43: 予算は prefill のパック済み重みと共有なので、`pinned` も出す
                // (`vram=LRU使用+外部予約/上限`)。
                let (used, pinned, cap) = match expert_vram::budget_usage() {
                    Some((u, p, c)) => (u, p, c),
                    None => {
                        let (u, c) = expert_vram::cache_usage().unwrap_or((0, 0));
                        (u, 0, c)
                    }
                };
                eprintln!(
                    "[flash] expert_gpu_check layer={} cmp={} argmax={}/{} max_abs={:.3e} max_rel={:.3e} \
                     vram={}/{} MiB (pinned={} MiB) cpu_fallback={}",
                    lw.il,
                    c.n,
                    c.argmax,
                    c.n,
                    c.max_abs,
                    c.max_rel,
                    (used + pinned) >> 20,
                    cap >> 20,
                    pinned >> 20,
                    expert_vram::fallbacks()
                );
            }
        }

    }

    // 5. shared expert (係数は sigmoid(gate_inp_shexp @ x))
    moe_sub_add(1, t_experts.elapsed());
    // 区間を閉じる (shared の時間を experts に二重計上しない)。
    drop(_l_experts);
    if !iv.block_off(lw.il, IvBlock::Shared) {
        // task 138: 台帳の区間 (shared)。join・down・gate_inp の GEMV を含む。
        let _l_shared = q4ledger::enter(q4ledger::SHARED);
        let t_shared = std::time::Instant::now();
        let sff = lw.gate_shexp.n();
        let mut sh = vec![0.0f32; e];
        let mut sg = vec![0.0f32; lw.gate_inp_shexp.n()];
        // task 187: `DENNO_DEC_SHARED_DEV` では gate → up → silu*up → down を router の
        // 時点で 1 本のサイドストリームに積んである。ここでは down の出力を join する
        // だけ (ホストの silu/mul・down の再発行・待ちが消える)。
        let dev_shared = if shed_ov.has_down() {
            let _s = q4ledger::sub(q4ledger::SUB_SH_WAIT);
            shed_ov.finish_shared(&mut sh)
        } else {
            false
        };
        if !dev_shared {
            // 従来 (task 124 / 155): gate/up をホストで受け、silu/mul を回し、
            // down をサイドストリームへ先に積んでから gate_inp を回す。
            // task 187: 台帳の副区間 (作業域の確保)。
            let s_elem = q4ledger::sub(q4ledger::SUB_SH_ELEM);
            let mut sact = vec![0.0f32; sff];
            let mut sup = vec![0.0f32; sff];
            drop(s_elem);
            // task 124: router の前に積んだ gate/up をここで回収する (間に合っていれば
            // 待ちはほぼ 0)。取れなければ従来どおりこの場で同期して計算する。
            // 従来の文の順序 (gate → silu → up → 積) は `!overlapped` 側でそのまま。
            let overlapped = shed_ov.finish(&mut sact, &mut sup);
            if !overlapped {
                lw.gate_shexp
                    .gemv_label("shared_gate", x, &mut sact)
                    .map_err(|err| one(format!("層 {}: ffn_gate_shexp: {err}", lw.il)))?;
            }
            {
                // task 187: 台帳の副区間 (silu と積)。
                let _s = q4ledger::sub(q4ledger::SUB_SH_ELEM);
                silu_inplace(&mut sact);
                if !overlapped {
                    lw.up_shexp
                        .gemv_label("shared_up", x, &mut sup)
                        .map_err(|err| one(format!("層 {}: ffn_up_shexp: {err}", lw.il)))?;
                }
                for r in 0..sff {
                    sact[r] *= sup[r];
                }
            }
            // task 155: down は**サイドストリーム**に先に積み、その完了を待つ間に
            // gate_inp (null ストリーム) を回す (層あたり 2 往復 → 1 往復)。
            // 積めない型/環境では `finish` が false を返すので従来の同期経路に戻る。
            let mut sdown = {
                let _s = q4ledger::sub(q4ledger::SUB_SH_SDOWN);
                SharedDownOverlap::begin(lw, &sact, e)
            };
            {
                // task 187: 台帳の副区間 (F32・1 行の GEMV)。
                let _s = q4ledger::sub(q4ledger::SUB_SH_SG);
                lw.gate_inp_shexp
                    .gemv_label("shared_gate_inp", x, &mut sg)
                    .map_err(|err| one(format!("層 {}: ffn_gate_inp_shexp: {err}", lw.il)))?;
            }
            if !sdown.finish(&mut sh) {
                lw.down_shexp
                    .gemv_label("shared_down", &sact, &mut sh)
                    .map_err(|err| one(format!("層 {}: ffn_down_shexp: {err}", lw.il)))?;
            }
        } else {
            // 係数 `sigmoid(gate_inp_shexp @ x)` はホストで要る (1 行の GEMV)。
            let _s = q4ledger::sub(q4ledger::SUB_SH_SG);
            lw.gate_inp_shexp
                .gemv_label("shared_gate_inp", x, &mut sg)
                .map_err(|err| one(format!("層 {}: ffn_gate_inp_shexp: {err}", lw.il)))?;
        }
        let sgw = hyper::sigmoid(sg[0]);
        {
            // task 187: 台帳の副区間 (shared 出力の足し込み + RMS)。
            let _s = q4ledger::sub(q4ledger::SUB_SH_ADD);
            for j in 0..e {
                out[j] += sh[j] * sgw;
            }
            // P54: shared expert の出力そのもの (`ffn_down_shexp` の出力) の RMS。
            // 係数 `sgw` は掛ける前の値を見る (expert の寄与の大きさを測るため)。
            if let Some(r) = roles.as_deref_mut() {
                r.shex = rms_of(&sh);
            }
        }
        moe_sub_add(2, t_shared.elapsed());
    }

    Ok(out)
}

/// P39: expert 1 本ぶんの FFN (`down(silu(gate(x)) * up(x))`) を **VRAM 常駐の重み**で回す。
///
/// 戻り値 (P45 で `bool` から変更):
/// - `Some(true)` — GPU で計算した。重みは**すでに VRAM に載っていた** (ヒット)。
/// - `Some(false)` — GPU で計算した。重みは今回 VRAM へ積んだ (ミス)。
/// - `None` — この expert は GPU に載せられない (型・カーネル・上限)。呼び出し側は
///   従来の CPU 経路へ落ちること。
///
/// `act` / `up` / `h` は **GPU で計算したときだけ**書き換える (途中で失敗して
/// 半端な値を残さない)。ヒット率は `Some(..)` の割合としてゲートに積まれる。
#[cfg(feature = "cuda")]
pub(crate) fn moe_expert_gpu(
    lw: &LayerW,
    ei: usize,
    x: &[f32],
    act: &mut [f32],
    up: &mut [f32],
    h: &mut [f32],
) -> Result<Option<bool>> {
    use crate::model::expert_vram::ExpertShape;

    // task 144: IQ 系の型は層一括経路 (`kernels_exp_iq`) だけが読める。per-expert の
    // 旧経路 (`kernels_moe.ptx`) は 4 型 (Q8_0/Q5_1/Q4_K/Q5_K) しか読めないので、
    // VRAM に載っていても `dev.gemv` が false を返してエラーになる。CPU へ落とす
    // (呼び出し側は `gpu.is_none()` を見て CPU 経路を回す)。
    if crate::model::exp_iq::iq_gpu_enabled()
        && [
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
        ]
        .iter()
        .any(|t| crate::model::exp_iq::supported(*t))
    {
        return Ok(None);
    }

    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x.len(),
            n: act.len(),
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x.len(),
            n: up.len(),
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: act.len(),
            n: h.len(),
        },
    ];
    let key = ExpertKey {
        layer: lw.il as u16,
        expert: ei as u16,
    };
    // ミスのときだけ mmap のスライスを要求する (ヒットでページフォルトを起こさない)。
    // task 260: 転送元は RAM tier → mmap の順 (`acquire_expert`)。
    let got = acquire_expert(lw, ei, key, shapes)
        .map_err(|e| one(format!("層 {}: expert {ei}: VRAM 経路: {e:#}", lw.il)))?;
    let Some(got) = got else {
        // task 261: 載らなかった 1 本 (理由は expert_vram 側の fb 内訳が出す)。
        crate::model::exp_iq::note_lookup(1, 0);
        return Ok(None);
    };
    let hit = got.hit;
    // task 261: 層一括を使わない経路でもヒット率を数える。
    crate::model::exp_iq::note_lookup(1, u32::from(hit));
    let dev = got.dev;

    let ctx = |e: anyhow::Error, m: ExpertMatrix| {
        one(format!(
            "層 {}: expert {ei}: {} の GPU GEMV: {e:#}",
            lw.il,
            m.name()
        ))
    };
    dev.gemv(ExpertMatrix::Gate, x, act)
        .map_err(|e| ctx(e, ExpertMatrix::Gate))?;
    silu_inplace(act);
    dev.gemv(ExpertMatrix::Up, x, up)
        .map_err(|e| ctx(e, ExpertMatrix::Up))?;
    for r in 0..act.len() {
        act[r] *= up[r];
    }
    dev.gemv(ExpertMatrix::Down, act, h)
        .map_err(|e| ctx(e, ExpertMatrix::Down))?;

    // DENNO_EXPERT_GPU_CHECK=1: 同じ入力で CPU の 3 本を回し、最終寄与 h を比べる。
    // (層ごとに 1 行だけ出す。decode 全体では層数ぶんの行になる。)
    if expert_vram::check_enabled() {
        let mut act_c = vec![0.0f32; act.len()];
        let mut up_c = vec![0.0f32; up.len()];
        let mut h_c = vec![0.0f32; h.len()];
        lw.gate_exps.gemv(ei, x, &mut act_c)?;
        silu_inplace(&mut act_c);
        lw.up_exps.gemv(ei, x, &mut up_c)?;
        for r in 0..act_c.len() {
            act_c[r] *= up_c[r];
        }
        lw.down_exps.gemv(ei, &act_c, &mut h_c)?;
        expert_vram::note_check(h, &h_c);
    }
    Ok(Some(hit))
}

/// P43: 選択 expert **全員** の寄与を「3 起動 / 2 同期」でまとめて計算する。
///
/// `sel` の全 expert が VRAM に載り、カーネル (PTX) と型が対応しているときだけ
/// `Ok(Some((hs, hits)))` を返す (`hs[t]` は `sel[t]` の寄与 h、長さ `e`。
/// `hits` は「すでに VRAM に載っていて H2D しなかった本数」= P45 のゲートのサンプル)。
/// 1 本でも載らない / カーネルが無い / 型が未対応なら `Ok(None)` を返し、
/// 呼び出し側は per-expert 経路 ([`moe_expert_gpu`]) へ落ちる (fail closed)。
///
/// 起動計画 (`DENNO_EXPERT_BATCH` / segment 表の作り方) は
/// [`crate::model::expert_batch`] にある。数値は per-expert 経路とビット一致する
/// (同じ dequant・同じ i 昇順の累積・同じ silu の式)。
#[cfg(feature = "cuda")]
pub(crate) fn moe_experts_gpu_batched(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
) -> Result<Option<(Vec<Vec<f32>>, u32)>> {
    use crate::model::expert_batch as eb;
    use crate::model::expert_vram::ExpertShape;

    if sel.is_empty() || !eb::batch_enabled() {
        return Ok(None);
    }
    let n = sel.len();
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // task 266: ミスする expert の 3 行列を、H2D の**前**に背景スレッドで
    // 並列に読ませる (`DENNO_DEC_MISS_READ=1` のときだけ)。メインスレッドは
    // そのまま下の acquire (H2D) を回すが、同じページのフォルトはページロックで
    // 1 回に畳まれるので二重読みにはならない。値・経路は 1 バイトも変わらない。
    #[cfg(feature = "cuda")]
    if crate::model::exp_prefetch::miss_read_enabled() {
        let mut parts: Vec<Vec<crate::model::exp_up_worker::Part>> = Vec::with_capacity(n);
        for &ei in sel {
            let key = ExpertKey {
                layer: lw.il as u16,
                expert: ei as u16,
            };
            // 載っている expert は触らない (ヒットは mmap を読まない)。
            if matches!(expert_vram::lookup(key, shapes), Ok(Some(_))) {
                continue;
            }
            let part = |w: &ExpertW| w.slice_mapped(ei).ok().flatten();
            if let (Some(g), Some(u), Some(d)) = (
                part(&lw.gate_exps),
                part(&lw.up_exps),
                part(&lw.down_exps),
            ) {
                parts.push(vec![g, u, d]);
            }
        }
        crate::model::exp_prefetch::submit_misses(parts);
    }
    // 1. 全 expert を VRAM に載せる (1 本でも載らなければ per-expert 経路へ)。
    //    P45: 載っていた本数 (`hit`) を数え、ゲートの 1 サンプルにする。
    let mut devs = Vec::with_capacity(n);
    let mut ids: Vec<u16> = Vec::with_capacity(n);
    let mut hits = 0u32;
    for &ei in sel {
        let key = ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        };
        // task 260: 転送元は RAM tier → mmap の順 (`acquire_expert`)。
        let got = acquire_expert(lw, ei, key, shapes)
            .map_err(|err| one(format!("層 {}: expert {ei}: VRAM 経路 (層一括): {err:#}", lw.il)))?;
        let Some(a) = got else {
            // task 261: 1 本でも載らないと層ごと per-expert 経路へ落ちる。回数を残す
            // (「なぜ GPU で回らなかったか」を [exp-iq] の 1 行で読むため)。
            crate::model::exp_iq::note_batch_bail();
            return Ok(None);
        };
        if a.hit {
            hits += 1;
        }
        devs.push(a.dev);
        ids.push(ei as u16);
    }
    // task 261: 層一括経路のヒット率を [exp-iq] に積む (従来は IQ 経路だけが
    // 数えていたので、旧 4 型のモデルではヒット率が読めなかった)。
    crate::model::exp_iq::note_lookup(sel.len() as u32, hits);
    crate::model::exp_iq::note_batch_ok();
    let ptr_table = |m: ExpertMatrix| -> Vec<u64> { devs.iter().map(|d| d.ptr(m)).collect() };
    let (pg, pu, pd) = (
        ptr_table(ExpertMatrix::Gate),
        ptr_table(ExpertMatrix::Up),
        ptr_table(ExpertMatrix::Down),
    );

    // 2. gate + up (2 起動 / 1 同期)
    let mut g = vec![0.0f32; n * ff];
    let mut up = vec![0.0f32; n * ff];
    let seg_gu = eb::seg_table_shared_x(&ids, ff);
    let ran = crate::cuda::moe_gemv_seg_pair(
        lw.gate_exps.ty,
        lw.up_exps.ty,
        &pg,
        &pu,
        &seg_gu,
        1,
        x,
        x.len(),
        ff,
        &mut g,
        &mut up,
    )
    .map_err(|err| one(format!("層 {}: 層一括 GEMV (gate+up): {err:#}", lw.il)))?;
    if !ran {
        return Ok(None);
    }
    // 3. act = silu(g) * u (per-expert 経路と同じ式・同じ順)
    let mut act = vec![0.0f32; n * ff];
    eb::silu_mul(&g, &up, &mut act);
    drop(g);
    drop(up);

    // 4. down (1 起動 / 1 同期)
    let mut h = vec![0.0f32; n * e];
    let seg_d = eb::seg_table_per_act(&ids, ff, e);
    let ran = crate::cuda::moe_gemv_seg_one(lw.down_exps.ty, &pd, &seg_d, 1, &act, ff, e, &mut h)
        .map_err(|err| one(format!("層 {}: 層一括 GEMV (down): {err:#}", lw.il)))?;
    if !ran {
        return Ok(None);
    }
    let hs: Vec<Vec<f32>> = h.chunks_exact(e).map(|c| c.to_vec()).collect();

    // 5. DENNO_EXPERT_GPU_CHECK=1: per-expert 経路と同じ比較 (層ごとに 1 行)。
    if expert_vram::check_enabled() {
        let mut act_c = vec![0.0f32; ff];
        let mut up_c = vec![0.0f32; ff];
        let mut h_c = vec![0.0f32; e];
        for (t, &ei) in sel.iter().enumerate() {
            lw.gate_exps.gemv(ei, x, &mut act_c)?;
            silu_inplace(&mut act_c);
            lw.up_exps.gemv(ei, x, &mut up_c)?;
            for r in 0..ff {
                act_c[r] *= up_c[r];
            }
            lw.down_exps.gemv(ei, &act_c, &mut h_c)?;
            expert_vram::note_check(&hs[t], &h_c);
        }
    }
    Ok(Some((hs, hits)))
}

/// task 278: K 型 (Q4_K / Q5_K / Q5_1 / Q8_0) の層を **1 待ち**で回す新経路。
///
/// 旧 [`moe_experts_gpu_batched`] との違い:
///
/// - ミス expert を VRAM へ**上げない**。ヒットだけを GPU の segment GEMV で回し、
///   ミスは GPU の裏で CPU が計算する (8 スレッド)。1 本 3 MB の H2D と
///   ホスト側の写し (1 本 0.25 ms 級) が消える。
/// - `silu(g) * u` と重み付き和をデバイスで回し、ホストへは `e` 要素だけ返す。
/// - 待ちは null ストリームのイベント 1 本 (文脈同期は背景の上げ直しまで待つ)。
///
/// 戻り値は [`IqBatchOut::Sum`] (ヒットぶんはデバイスで和、ミスは `miss_h`)。
/// ヒットが 1 本も無ければ `Contribs(全て None)` = 呼び出し側が全部 CPU で回す。
/// カーネル / PTX / 形が合わないときは `Ok(None)` = 旧経路へ (fail closed)。
#[cfg(feature = "cuda")]
pub(crate) fn moe_experts_seg_gpu_batched(
    lw: &LayerW,
    sel: &[usize],
    weights: &[f32],
    x: &[f32],
    e: usize,
    ff: usize,
) -> Result<Option<IqBatchOut>> {
    use crate::model::expert_vram::ExpertShape;

    if sel.is_empty() || weights.len() != sel.len() {
        return Ok(None);
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // 1. 載っている expert だけ集める (H2D しない)。
    let mut devs = Vec::with_capacity(sel.len());
    let mut ids: Vec<u16> = Vec::with_capacity(sel.len());
    let mut pos: Vec<usize> = Vec::with_capacity(sel.len());
    let _t_lookup = crate::model::seg_prof::stamp();
    let keys: Vec<ExpertKey> = sel
        .iter()
        .map(|&ei| ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        })
        .collect();
    let got = expert_vram::lookup_many(&keys, shapes)
        .map_err(|err| one(format!("層 {}: expert VRAM 参照 (seg): {err:#}", lw.il)))?;
    for (t, g) in got.into_iter().enumerate() {
        if let Some(dev) = g {
            devs.push(dev);
            ids.push(sel[t] as u16);
            pos.push(t);
        }
    }
    crate::model::seg_prof::add(crate::model::seg_prof::PH_LOOKUP, _t_lookup);
    if pos.is_empty() {
        // VRAM が初期化されていない (CUDA 無し / 予算 0) ときは旧経路へ。
        // 従来のフォールバック契約 (CPU の per-expert 経路) をそのまま通す。
        if expert_vram::cache_usage().is_none() {
            return Ok(None);
        }
        // VRAM はあるが 1 本も載っていない層は CPU で回す (ここで上げ直すと
        // 1 層 30 MB の H2D になる)。足し込みの口は IQ 経路と共通。
        return Ok(Some(IqBatchOut::Contribs(vec![None; sel.len()], 0)));
    }
    let hits = devs.len() as u32;

    // 2. prologue (表 + 入力) を組む。
    let _t_tbl = crate::model::seg_prof::stamp();
    let ptr_table = |m: ExpertMatrix| -> Vec<u64> { devs.iter().map(|d| d.ptr(m)).collect() };
    let (pg, pu, pd) = (
        ptr_table(ExpertMatrix::Gate),
        ptr_table(ExpertMatrix::Up),
        ptr_table(ExpertMatrix::Down),
    );
    let seg_gu = crate::model::expert_batch::seg_table_shared_x(&ids, ff);
    let seg_d = crate::model::expert_batch::seg_table_per_act(&ids, ff, e);
    let hw: Vec<f32> = pos.iter().map(|&t| weights[t]).collect();
    crate::model::seg_prof::add(crate::model::seg_prof::PH_TABLE, _t_tbl);

    // 3. ミスの位置と id (CPU 計算は GPU の裏で走らせる)。
    let mut is_hit = vec![false; sel.len()];
    for &t in pos.iter() {
        is_hit[t] = true;
    }
    let miss_pos: Vec<usize> = (0..sel.len()).filter(|&t| !is_hit[t]).collect();
    let miss_ids: Vec<usize> = miss_pos.iter().map(|&t| sel[t]).collect();
    let mut miss_h = vec![0.0f32; miss_pos.len() * e];

    // 4. gate / up / silu / down / 和 を 1 待ちで回す。
    let mut sum = vec![0.0f32; e];
    let _t_alloc = crate::model::seg_prof::stamp();
    let ran = crate::cuda::moe_seg_layer_sum(
        lw.gate_exps.ty,
        lw.up_exps.ty,
        lw.down_exps.ty,
        &pg,
        &pu,
        &pd,
        &seg_gu,
        &seg_d,
        x,
        x.len(),
        ff,
        e,
        &hw,
        &mut sum,
        || {
            let _s = q4ledger::sub(q4ledger::SUB_EXP_MISS);
            // ミスは base と同じ 8 スレッド経路で計算する (1 expert を 1 スレッドで
            // 逐次に回すと 1 本 0.15 ms がそのまま層に乗る)。
            expert_miss_contribs_into(lw, &miss_ids, x, e, ff, &mut miss_h)
                .map_err(anyhow::Error::from)
        },
    )
    .map_err(|err| one(format!("層 {}: seg 層一括 GEMV (sum): {err:#}", lw.il)))?;
    crate::model::seg_prof::add(crate::model::seg_prof::PH_ALLOC, _t_alloc);
    if !ran {
        return Ok(None);
    }
    // 5. `DENNO_EXPERT_GPU_CHECK=1`: 全 expert を CPU で回した値と、この経路の
    //    出力 (デバイスの和 + ホストのミス) を層ごとに 1 行で比べる。
    //    新しい足し込み順 (ヒットを先に全部、ミスを後) になるので、
    //    ミスの無い層ではビット一致、有る層では丸め差だけが出るはず。
    if expert_vram::check_enabled() {
        let mut got = sum.clone();
        for (i, &t) in miss_pos.iter().enumerate() {
            let pe = weights[t];
            let v = &miss_h[i * e..(i + 1) * e];
            for j in 0..e {
                got[j] += pe * v[j];
            }
        }
        let mut want = vec![0.0f32; e];
        let mut act = vec![0.0f32; ff];
        let mut up = vec![0.0f32; ff];
        let mut h = vec![0.0f32; e];
        for (t, &ei) in sel.iter().enumerate() {
            lw.gate_exps.gemv(ei, x, &mut act)?;
            silu_inplace(&mut act);
            lw.up_exps.gemv(ei, x, &mut up)?;
            for r in 0..ff {
                act[r] *= up[r];
            }
            lw.down_exps.gemv(ei, &act, &mut h)?;
            let pe = weights[t];
            for j in 0..e {
                want[j] += pe * h[j];
            }
        }
        expert_vram::note_check(&got, &want);
    }
    Ok(Some(IqBatchOut::Sum {
        sum,
        miss_pos,
        miss_h,
        hits,
    }))
}

/// task 278: 新経路のゲートが off / cuda 無しのときの入口。
#[cfg(not(feature = "cuda"))]
pub(crate) fn moe_experts_seg_gpu_batched(
    _lw: &LayerW,
    _sel: &[usize],
    _weights: &[f32],
    _x: &[f32],
    _e: usize,
    _ff: usize,
) -> Result<Option<IqBatchOut>> {
    Ok(None)
}

/// task 144: この層の 3 行列が全部 IQ カーネルで読める型か (かつゲートが on)。
///
/// 型はテンソル単位で決まるので層ごとに同じ値になるが、層ごとに呼んで安い。
#[cfg(not(feature = "cuda"))]
pub(crate) fn iq_expert_mode(_lw: &LayerW) -> bool {
    false
}

#[cfg(feature = "cuda")]
pub(crate) fn iq_expert_mode(lw: &LayerW) -> bool {
    crate::model::exp_iq::iq_gpu_enabled()
        && [
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
        ]
        .iter()
        .all(|t| crate::model::exp_iq::supported(*t))
}

/// task 150: ミスした expert の寄与を `dst` (`sel.len() * e`) へ直接書く。
///
/// 1 expert の計算は [`expert_contribs_par`] と同じ [`ExpertW::gemv_seq`] を同じ順で
/// 通るので、寄与はビット一致する (足し込みは呼び出し側が index 昇順の逐次)。
/// `par` 無しのビルドでは逐次で同じものを書く。
#[cfg(feature = "cuda")]
pub(crate) fn expert_miss_contribs_into(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(dst.len(), sel.len() * e, "dst は sel.len()*e");
    // task 260: RAM tier があればバイト列はそこから取る (mmap を触らない =
    // ページフォルトも disk も無い)。ミスは pread で埋めるので、同じ expert が
    // 次に来たときは RAM から出す。tier の値は mmap と同じバイト列
    // (`ExpertW::slice_abs` と `ExpertW::slice` が同じ範囲) なので数値は変わらない。
    let ram = crate::model::expert_ram::get();
    #[cfg(feature = "par")]
    {
        if let Some(c) = ram.as_deref() {
            if !sel.is_empty() {
                return expert_contribs_par_into(lw, sel, x, e, ff, Some(c), dst);
            }
        }
        // task 153: ミスは 1 層 0〜3 本なので expert 単位の並列では 1 スレッドしか
        // 働かない。行まで平坦化して全スレッドへ配る (値は gemv_rows_seq =
        // gemv_seq とビット一致)。`DENNO_EXP_MISS_FLAT=0` で従来経路へ戻る。
        if expert_miss_flat_enabled() {
            if sel.is_empty() {
                return Ok(());
            }
            return expert_contribs_flat_decode_into(lw, sel, x, e, ff, dst, miss_flat_rows());
        }
        expert_contribs_par_into(lw, sel, x, e, ff, None, dst)
    }
    #[cfg(not(feature = "par"))]
    {
        if let Some(c) = ram.as_deref() {
            let mut act = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            for (t, &ei) in sel.iter().enumerate() {
                let h = &mut dst[t * e..(t + 1) * e];
                expert_gemv_cached_seq(c, ExpertMatrix::Gate, &lw.gate_exps, lw.il, ei, x, &mut act)?;
                silu_inplace(&mut act);
                expert_gemv_cached_seq(c, ExpertMatrix::Up, &lw.up_exps, lw.il, ei, x, &mut up)?;
                for r in 0..ff {
                    act[r] *= up[r];
                }
                expert_gemv_cached_seq(c, ExpertMatrix::Down, &lw.down_exps, lw.il, ei, &act, h)?;
            }
            return Ok(());
        }
        // par 無しのビルドでも同じ寄与 (1 expert = gemv_seq を 3 回)。
        let mut act = vec![0.0f32; ff];
        let mut up = vec![0.0f32; ff];
        for (t, &ei) in sel.iter().enumerate() {
            let h = &mut dst[t * e..(t + 1) * e];
            lw.gate_exps.gemv_seq(ei, x, &mut act)?;
            silu_inplace(&mut act);
            lw.up_exps.gemv_seq(ei, x, &mut up)?;
            for r in 0..ff {
                act[r] *= up[r];
            }
            lw.down_exps.gemv_seq(ei, &act, h)?;
        }
        Ok(())
    }
}

/// task 150: ミスの寄与を「何本ぶん書くか」(`overlap` でなければ 0)。
///
/// `miss_h` の確保長と [`scatter_contribs`] に渡す表を**同じ値**から作るための
/// 唯一の入口 (片方だけ `miss_pos.len()` を使うと長さ検査で落ちる。lab t150a で
/// CHAIN2=0 の全リクエストがこれで落ちた)。
#[cfg(feature = "cuda")]
pub(crate) fn miss_slots(overlap: bool, n_miss: usize) -> usize {
    if overlap {
        n_miss
    } else {
        0
    }
}

/// task 150: GPU のヒット (`pos` / `h`) と CPU ミス (`miss_pos` / `miss_h`) の寄与を
/// `sel` の並び (`hs`) へ戻す。
///
/// 位置と寄与の対応が 1 つずれると静かに別の expert の値を足すことになるので、
/// 長さと範囲を検査して落とす (どちらも呼び出し側の作った表なので、ここが最後の砦)。
#[cfg(feature = "cuda")]
pub(crate) fn scatter_contribs(
    hs: &mut [Option<Vec<f32>>],
    pos: &[usize],
    h: &[f32],
    miss_pos: &[usize],
    miss_h: &[f32],
    e: usize,
) -> Result<()> {
    if e == 0 || h.len() != pos.len() * e || miss_h.len() != miss_pos.len() * e {
        return Err(one(format!(
            "expert 寄与の長さが合わない (h {} / pos {} / miss {} / miss_pos {} / e {e})",
            h.len(),
            pos.len(),
            miss_h.len(),
            miss_pos.len()
        )));
    }
    for (i, &t) in pos.iter().enumerate() {
        let slot = hs
            .get_mut(t)
            .ok_or_else(|| one(format!("hit 位置 {t} が hs の範囲外")))?;
        *slot = Some(h[i * e..(i + 1) * e].to_vec());
    }
    for (i, &t) in miss_pos.iter().enumerate() {
        let slot = hs
            .get_mut(t)
            .ok_or_else(|| one(format!("miss 位置 {t} が hs の範囲外")))?;
        *slot = Some(miss_h[i * e..(i + 1) * e].to_vec());
    }
    Ok(())
}

/// task 144: IQ 系の型を **ヒットだけ GPU / ミスは CPU** で回す層一括経路。
///
/// # 何が違うか
///
/// P43 の [`moe_experts_gpu_batched`] は「全 expert が VRAM に載らなければ `None`」
/// だった (ミスは同期 H2D で積んでいた)。本番の VRAM 予算 (14〜22 GB) では 1 本が
/// 約 2.05 MB でも全 24,576 本 (48 層 x 512) は入らないので、**ミスを GPU の待ちに
/// しない**方が速い。ここでは
///
/// - 載っている expert だけを segment 表に載せ、`kernels_exp_iq` で 1 層 4 起動 +
///   1 同期で回す (gate/up/silu/down)。
/// - 載っていない expert は `None` を返し、呼び出し側が CPU で計算する
///   (合計の順序は今と同じ expert index 昇順の逐次)。
/// - ミスの H2D は [`iq_warm_misses`] がトークンあたりの上限つきで行う。
///
/// 戻り値: `Some((hs, hits))` の `hs[t]` は `sel[t]` の寄与 (GPU で回したものだけ
/// `Some`)。GPU のカーネルが無い / 型が未対応 / 1 本も載っていないなら `None`
/// (呼び出し側は per-expert = CPU 経路へ)。
/// task 187: IQ 層一括経路の戻り値。
///
/// - [`IqBatchOut::Contribs`] は従来 (task 144) の形: `sel` の並びの寄与。
/// - [`IqBatchOut::Sum`] は `DENNO_DEC_EXP_SUMDEV` の形: ヒットぶんを**デバイスで**
///   重み付き和にした `e` 要素と、ミスの位置・寄与 (ホストで足す)。
pub(crate) enum IqBatchOut {
    Contribs(Vec<Option<Vec<f32>>>, u32),
    Sum {
        sum: Vec<f32>,
        miss_pos: Vec<usize>,
        miss_h: Vec<f32>,
        hits: u32,
    },
}

#[cfg(feature = "cuda")]
pub(crate) fn moe_experts_iq_gpu_batched(
    lw: &LayerW,
    sel: &[usize],
    weights: &[f32],
    x: &[f32],
    e: usize,
    ff: usize,
    sumdev: bool,
) -> Result<Option<IqBatchOut>> {
    use crate::model::expert_vram::ExpertShape;

    if sel.is_empty() {
        return Ok(None);
    }
    // task 164: `DENNO_DEC_MISS_PROBE=1` なら 1 回だけ H2D の実効レートを測る
    // (miss_up のゲートとは独立。計測は decode の最初の層で 1 回だけ)。
    if crate::model::exp_miss_up::probe_enabled() {
        let src = lw.gate_exps.slice(sel[0]).ok();
        crate::cuda::dec_miss_probe_once(2, src);
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // 1. 載っている expert だけ集める (H2D しない)。
    let mut devs = Vec::with_capacity(sel.len());
    let mut ids: Vec<u16> = Vec::with_capacity(sel.len());
    let mut pos: Vec<usize> = Vec::with_capacity(sel.len());
    // task 187: 台帳の副区間 (VRAM 参照)。`experts` の cpu 列の中身。
    let s_lookup = q4ledger::sub(q4ledger::SUB_EXP_LOOKUP);
    if crate::model::exp_up_worker::enabled() {
        // task 214: 背景の上げ直しが転送中は LRU のロックが 1 本 0.16 ms 埋まる。
        // 1 層 10 回のロック取得を 1 回に寄せる (`lookup_many` は `lookup` と
        // 同じ `get_mut` を順に呼ぶだけ = 触り方も順序も同じ)。
        let keys: Vec<ExpertKey> = sel
            .iter()
            .map(|&ei| ExpertKey {
                layer: lw.il as u16,
                expert: ei as u16,
            })
            .collect();
        let got = expert_vram::lookup_many(&keys, shapes)
            .map_err(|err| one(format!("層 {}: expert VRAM 参照: {err:#}", lw.il)))?;
        for (t, g) in got.into_iter().enumerate() {
            if let Some(dev) = g {
                devs.push(dev);
                ids.push(sel[t] as u16);
                pos.push(t);
            }
        }
    } else {
        for (t, &ei) in sel.iter().enumerate() {
            let key = ExpertKey {
                layer: lw.il as u16,
                expert: ei as u16,
            };
            let got = expert_vram::lookup(key, shapes)
                .map_err(|err| one(format!("層 {}: expert {ei}: VRAM 参照: {err:#}", lw.il)))?;
            if let Some(dev) = got {
                devs.push(dev);
                ids.push(ei as u16);
                pos.push(t);
            }
        }
    }
    drop(s_lookup);
    if devs.is_empty() {
        return Ok(None);
    }
    let hits = devs.len() as u32;
    // task 144: 1 層 1 同期は実測 74 us (ledger t144d)。ヒットが数本しか無い層を
    // GPU で回すと、節約 (1 本 0.07 ms) より同期の方が高くつく。閾値未満は
    // **全 expert を CPU 並列**へ回す (呼び出し側は `None` を見て CPU を回す)。
    let min_hits = crate::model::exp_iq::min_hits_per_layer();
    if (hits as usize) < min_hits {
        return Ok(Some(IqBatchOut::Contribs(vec![None; sel.len()], 0)));
    }
    crate::model::exp_iq::note_gpu_layer(hits);

    // 2. gate/up (2 起動) -> silu -> down (1 起動) -> 同期 1 回 (cuda 側)。
    let ptr_table = |m: ExpertMatrix| -> Vec<u64> { devs.iter().map(|d| d.ptr(m)).collect() };
    let (pg, pu, pd) = (
        ptr_table(ExpertMatrix::Gate),
        ptr_table(ExpertMatrix::Up),
        ptr_table(ExpertMatrix::Down),
    );
    let seg_gu = crate::model::expert_batch::seg_table_shared_x(&ids, ff);
    let seg_d = crate::model::expert_batch::seg_table_per_act(&ids, ff, e);
    let s_alloc = q4ledger::sub(q4ledger::SUB_EXP_ALLOC);
    // task 281: `DENNO_EXP_SCRATCH` — sumdev (既定 on) の経路は `h` を 1 度も読まない
    // ので、40 KB のゼロ埋めを毎層まるごと捨てない (48 層で 1.9 MB/token)。
    // 値は変わらない (h を使うのは !sumdev のときだけ。使う直前に確保する)。
    let scratch_on = exp_scratch_enabled();
    let mut h: Vec<f32> = if scratch_on {
        Vec::new()
    } else {
        vec![0.0f32; devs.len() * e]
    };
    drop(s_alloc);
    // task 150 (`DENNO_DECODE_CHAIN2_EXP`): ミス (VRAM に載っていない expert) の寄与を
    // GPU の完了前に CPU で計算し、同期をその裏へ隠す。ミスの位置と id を先に作り、
    // overlap 経路ではコールバックの中で `miss_h` へ書かせる (足し込みは従来どおり
    // 呼び出し側が index 昇順の逐次)。
    let overlap = decode_chain2_exp_enabled();
    let mut is_hit = vec![false; sel.len()];
    for &t in pos.iter() {
        is_hit[t] = true;
    }
    let miss_pos: Vec<usize> = (0..sel.len()).filter(|&t| !is_hit[t]).collect();
    let miss_ids: Vec<usize> = miss_pos.iter().map(|&t| sel[t]).collect();
    // ミスの寄与を書き込む先は overlap のときだけ。確保長と scatter に渡す表は
    // 必ず同じ値から作る ([`miss_slots`])。
    let n_miss_used = miss_slots(overlap, miss_pos.len());
    let miss_slice: &[usize] = &miss_pos[..n_miss_used];
    let mut miss_h = vec![0.0f32; n_miss_used * e];
    // task 187: `DENNO_DEC_EXP_SUMDEV` ではヒットの寄与をデバイスで和にして
    // `e` 要素だけ D2H する (ミスはホストで足す)。重みは選択順 (pos の順) に詰める。
    if sumdev {
        let hw: Vec<f32> = pos.iter().map(|&t| weights[t]).collect();
        let mut sum = vec![0.0f32; e];
        // ミスの寄与は常にこの関数の中で計算して返す (overlap の有無に依らない)。
        let mut miss_s = vec![0.0f32; miss_pos.len() * e];
        let ran = crate::cuda::moe_experts_iq_layer_sum_overlap(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &pg,
            &pu,
            &pd,
            &seg_gu,
            &seg_d,
            x,
            x.len(),
            ff,
            e,
            &hw,
            &mut sum,
            || {
                // task 187: 台帳の副区間 (ミスの CPU 計算。GPU の裏で走る)。
                let _s = q4ledger::sub(q4ledger::SUB_EXP_MISS);
                // task 190: ヒット本数ごとの表 (PROFILE のときだけ Instant 2 回)。
                let t = crate::model::q4ledger::stamp();
                let r = expert_miss_contribs_into(lw, &miss_ids, x, e, ff, &mut miss_s)
                    .map_err(anyhow::Error::from);
                if let Some(t) = t {
                    crate::model::exp_iq::note_layer_miss(
                        hits,
                        miss_ids.len() as u32,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                r
            },
        )
        .map_err(|err| one(format!("層 {}: IQ 層一括 GEMV (sum): {err:#}", lw.il)))?;
        if ran {
            return Ok(Some(IqBatchOut::Sum {
                sum,
                miss_pos,
                miss_h: miss_s,
                hits,
            }));
        }
        // 和のカーネル (PTX) が無い / 形が合わないときは**従来の寄与経路**へ落ちる
        // (per-expert 経路へは落とさない: そちらは IQ 型を CPU で回すので遅い)。
    }
    // task 281: `DENNO_EXP_SCRATCH` では `h` を遅延確保する (sumdev の経路は
    // ここまで来ない = 40 KB のゼロ埋めをしない)。値は従来と同じ。
    if scratch_on && h.len() < devs.len() * e {
        h = vec![0.0f32; devs.len() * e];
    }
    let ran = if overlap {
        // 台帳の exp_iq 区間は取らない (同期が CPU ミスの後ろへ移るので、GPU の
        // 待ちだけを切り出せない)。起動と同期は `experts` 行に入る。
        crate::cuda::moe_experts_iq_layer_overlap(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &pg,
            &pu,
            &pd,
            &seg_gu,
            &seg_d,
            x,
            x.len(),
            ff,
            e,
            &mut h,
            || {
                // task 187: 台帳の副区間 (ミスの CPU 計算。GPU の裏で走る)。
                let _s = q4ledger::sub(q4ledger::SUB_EXP_MISS);
                // task 190: ヒット本数ごとの表 (PROFILE のときだけ Instant 2 回)。
                let t = crate::model::q4ledger::stamp();
                let r = expert_miss_contribs_into(lw, &miss_ids, x, e, ff, &mut miss_h)
                    .map_err(anyhow::Error::from);
                if let Some(t) = t {
                    crate::model::exp_iq::note_layer_miss(
                        hits,
                        miss_ids.len() as u32,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                r
            },
        )
    } else {
        // task 138 の台帳に「1 層あたりの GPU 時間」を積む (DENNO_Q4_PROFILE=1 のときだけ)。
        let _l = crate::model::q4ledger::enter(crate::model::q4ledger::EXPERT);
        crate::cuda::moe_experts_iq_layer(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &pg,
            &pu,
            &pd,
            &seg_gu,
            &seg_d,
            x,
            x.len(),
            ff,
            e,
            &mut h,
        )
    }
    .map_err(|err| one(format!("層 {}: IQ 層一括 GEMV: {err:#}", lw.il)))?;
    if !ran {
        return Ok(None);
    }

    let s_alloc = q4ledger::sub(q4ledger::SUB_EXP_ALLOC);
    let mut hs: Vec<Option<Vec<f32>>> = vec![None; sel.len()];
    // task 150: overlap のときはミスも CPU で計算済みなので、両方をここで揃える
    // (呼び出し側は `None` を見ない = CPU を二度回さない)。
    scatter_contribs(&mut hs, &pos, &h, miss_slice, &miss_h, e)?;
    drop(s_alloc);

    // 3. DENNO_EXPERT_GPU_CHECK=1: CPU の 3 行列と突き合わせる (層ごとに 1 行)。
    if expert_vram::check_enabled() {
        let mut act_c = vec![0.0f32; ff];
        let mut up_c = vec![0.0f32; ff];
        let mut h_c = vec![0.0f32; e];
        for (i, &t) in pos.iter().enumerate() {
            let ei = sel[t];
            lw.gate_exps.gemv(ei, x, &mut act_c)?;
            silu_inplace(&mut act_c);
            lw.up_exps.gemv(ei, x, &mut up_c)?;
            for r in 0..ff {
                act_c[r] *= up_c[r];
            }
            lw.down_exps.gemv(ei, &act_c, &mut h_c)?;
            expert_vram::note_check(&h[i * e..(i + 1) * e], &h_c);
        }
    }
    Ok(Some(IqBatchOut::Contribs(hs, hits)))
}

/// task 158: ミス expert を GPU で計算する decode 専用の入口。
///
/// 2 つの変種をゲート [`crate::model::exp_miss_up::ENV`] で選ぶ:
///
/// - `1` (scratch): ミスの重みを pinned の bounce buffer へ写してから専用の
///   デバイス scratch へ非同期 H2D し、ヒットと**同じカーネル**で回す。VRAM の LRU は
///   触らない。写し (mmap → pinned) はヒットのカーネル発行の後に行うので、GPU の
///   ヒット計算と重なる。
/// - `2` (cache): `expert_vram::acquire` の通常方針 (LRU・予算・追い出し) で
///   キャッシュへ挿入してから回す。以後のトークンではヒットになる。
///
/// 戻り値は [`moe_experts_iq_gpu_batched`] と同じ形 (`hs` は `sel` の並び、
/// `hits` は VRAM に載っていた本数 = ゲートのサンプル)。`Ok(None)` は
/// 「この経路ではやらない」= 呼び出し側は従来経路 (CPU ミス + 重ね打ち) へ。
///
/// fail closed: scratch の確保失敗 / カーネル不発 / 型未対応 / ミスが
/// [`crate::model::exp_miss_up::MAX_MISS`] 超 / ヒットが
/// [`crate::model::exp_iq::min_hits_per_layer`] 未満 のいずれでも `None`。
#[cfg(feature = "cuda")]
pub(crate) fn moe_experts_iq_gpu_miss_up(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
) -> Result<Option<(Vec<Option<Vec<f32>>>, u32)>> {
    use crate::model::exp_miss_up as mu;
    use crate::model::expert_vram::ExpertShape;

    if !mu::enabled() || sel.is_empty() {
        return Ok(None);
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x.len(),
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    // 1. 載っている expert を集める (H2D しない)。
    let mut devs = Vec::with_capacity(sel.len());
    let mut pos: Vec<usize> = Vec::with_capacity(sel.len());
    let mut miss_pos: Vec<usize> = Vec::new();
    for (t, &ei) in sel.iter().enumerate() {
        let key = ExpertKey {
            layer: lw.il as u16,
            expert: ei as u16,
        };
        let got = expert_vram::lookup(key, shapes)
            .map_err(|err| one(format!("層 {}: expert {ei}: VRAM 参照: {err:#}", lw.il)))?;
        match got {
            Some(dev) => {
                devs.push(dev);
                pos.push(t);
            }
            None => miss_pos.push(t),
        }
    }
    let n_hit = devs.len();
    let n_miss = miss_pos.len();
    // ヒットが閾値未満の層は従来経路 (層一括 = CPU へ落ちる) に任せる。
    // ミスが scratch の本数を超える層も同じ (cold の 1 本目など)。
    if n_hit == 0
        || n_hit < crate::model::exp_iq::min_hits_per_layer()
        || n_miss > mu::MAX_MISS
    {
        return Ok(None);
    }
    let ids_of = |ps: &[usize]| -> Vec<u16> { ps.iter().map(|&t| sel[t] as u16).collect() };
    let ptrs = |m: ExpertMatrix| -> Vec<u64> { devs.iter().map(|d| d.ptr(m)).collect() };
    // 全ヒットの層と `MODE_CACHE` は「全 expert が VRAM に載った状態で 1 回の
    // 層一括」。全ヒットの層をここで処理するのは、呼び出し側に二度目の lookup を
    // させないため (定常状態では大半の層がミス 0 本になる)。
    if n_miss == 0 || mu::mode() == mu::MODE_CACHE {
        let mut pg = ptrs(ExpertMatrix::Gate);
        let mut pu = ptrs(ExpertMatrix::Up);
        let mut pd = ptrs(ExpertMatrix::Down);
        for &t in &miss_pos {
            let ei = sel[t];
            let key = ExpertKey {
                layer: lw.il as u16,
                expert: ei as u16,
            };
            // ミスは `acquire` (通常方針: LRU・予算・追い出し) でキャッシュへ挿入。
            // task 260: 転送元は RAM tier → mmap の順 (`acquire_expert`)。
            let got = acquire_expert(lw, ei, key, shapes).map_err(|err| {
                one(format!(
                    "層 {}: expert {ei}: VRAM 経路 (miss up): {err:#}",
                    lw.il
                ))
            })?;
            let Some(a) = got else {
                return Ok(None);
            };
            pg.push(a.dev.ptr(ExpertMatrix::Gate));
            pu.push(a.dev.ptr(ExpertMatrix::Up));
            pd.push(a.dev.ptr(ExpertMatrix::Down));
        }
        let ids: Vec<u16> = sel.iter().map(|&ei| ei as u16).collect();
        let seg_gu = crate::model::expert_batch::seg_table_shared_x(&ids, ff);
        let seg_d = crate::model::expert_batch::seg_table_per_act(&ids, ff, e);
        let mut h = vec![0.0f32; sel.len() * e];
        let _l = crate::model::q4ledger::enter(crate::model::q4ledger::EXPERT);
        let ran = crate::cuda::moe_experts_iq_layer(
            lw.gate_exps.ty,
            lw.up_exps.ty,
            lw.down_exps.ty,
            &pg,
            &pu,
            &pd,
            &seg_gu,
            &seg_d,
            x,
            x.len(),
            ff,
            e,
            &mut h,
        )
        .map_err(|err| one(format!("層 {}: ミス GPU 経路 (全ヒット/挿入): {err:#}", lw.il)))?;
        if !ran {
            return Ok(None);
        }
        let hs: Vec<Option<Vec<f32>>> = h.chunks_exact(e).map(|c| Some(c.to_vec())).collect();
        crate::model::exp_iq::note_gpu_layer(sel.len() as u32);
        return Ok(Some((hs, n_hit as u32)));
    }

    // 2b. scratch 経路: ミスだけ pinned → デバイス scratch へ非同期 H2D。
    let mut miss_bytes = Vec::with_capacity(n_miss);
    for &t in &miss_pos {
        let ei = sel[t];
        miss_bytes.push(crate::cuda::MissExpertBytes {
            gate: lw.gate_exps.slice(ei)?,
            up: lw.up_exps.slice(ei)?,
            down: lw.down_exps.slice(ei)?,
        });
    }
    let pg_hit: Vec<u64> = devs.iter().map(|d| d.ptr(ExpertMatrix::Gate)).collect();
    let pu_hit: Vec<u64> = devs.iter().map(|d| d.ptr(ExpertMatrix::Up)).collect();
    let pd_hit: Vec<u64> = devs.iter().map(|d| d.ptr(ExpertMatrix::Down)).collect();
    let ids_hit = ids_of(&pos);
    let ids_miss = ids_of(&miss_pos);
    let mut h: Vec<f32> = Vec::new();
    let ran = crate::cuda::moe_experts_iq_layer_miss_up(
        lw.gate_exps.ty,
        lw.up_exps.ty,
        lw.down_exps.ty,
        &pg_hit,
        &pu_hit,
        &pd_hit,
        &ids_hit,
        &miss_bytes,
        &ids_miss,
        x,
        x.len(),
        ff,
        e,
        &mut h,
    )
    .map_err(|err| one(format!("層 {}: ミス GPU 経路 (scratch): {err:#}", lw.il)))?;
    if !ran {
        return Ok(None);
    }
    if h.len() != (n_hit + n_miss) * e {
        return Err(one(format!(
            "層 {}: ミス GPU 経路の寄与長 {} が (ヒット {} + ミス {}) * e {} と違う",
            lw.il,
            h.len(),
            n_hit,
            n_miss,
            e
        )));
    }
    let mut hs: Vec<Option<Vec<f32>>> = vec![None; sel.len()];
    for (i, &t) in pos.iter().enumerate() {
        hs[t] = Some(h[i * e..(i + 1) * e].to_vec());
    }
    let base = n_hit * e;
    for (j, &t) in miss_pos.iter().enumerate() {
        hs[t] = Some(h[base + j * e..base + (j + 1) * e].to_vec());
    }
    crate::model::exp_iq::note_gpu_layer((n_hit + n_miss) as u32);
    Ok(Some((hs, n_hit as u32)))
}

/// task 146: 全層の expert 3 行列の合計バイト数 (使用回数ファイルのモデル同一性)。
///
/// テンソルのヘッダだけを見るので mmap のページは触らない。名前・層数が変わった
/// モデルの記録を取り違えないための値。
pub fn expert_identity_bytes(f: &GgufFile, p: &Qwen4ExpParams) -> u64 {
    let mut total = 0u64;
    for il in 0..p.n_layer {
        for m in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
            let name = format!("blk.{il}.{m}");
            if let Some((_, _, len)) = f.tensor_mapping_parts(&name) {
                total = total.saturating_add(len as u64);
            }
        }
    }
    total
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn moe_experts_gpu_batched(
    _lw: &LayerW,
    _sel: &[usize],
    _x: &[f32],
    _e: usize,
    _ff: usize,
) -> Result<Option<(Vec<Vec<f32>>, u32)>> {
    Ok(None)
}

/// MoE フェーズの内訳を測るための累積カウンタ (ns)。0 = router、1 = 選択 expert、2 = shared。
///
/// `moe` のフェーズ時間が 110〜124 ms と大きく、どこに落ちているか分からなかったので
/// 入れた。常時計測する (1 層あたり `Instant::now()` が 4 回で、無視できる)。
pub(crate) static MOE_SUB_NS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

pub(crate) fn moe_sub_add(idx: usize, d: std::time::Duration) {
    MOE_SUB_NS[idx].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `moe` の内訳 (router / 選択 expert / shared expert) を ms で返し、カウンタを 0 に戻す。
pub fn moe_sub_take() -> (f64, f64, f64) {
    use std::sync::atomic::Ordering;
    let mut v = [0f64; 3];
    for (i, slot) in MOE_SUB_NS.iter().enumerate() {
        v[i] = slot.swap(0, Ordering::Relaxed) as f64 / 1e6;
    }
    (v[0], v[1], v[2])
}

/// flat スケジューラの 1 行チャンク幅 (行数)。
///
/// 1 タスクの仕事は `MOE_FLAT_ROWS × n_in × m` MAC。実形 (n_in=2560, m=60) なら
/// 64 行で約 9.8 MMAC。1 層のタスク数は expert 数 × (n_ff/64 + n_embd/64) で、
/// 86 expert / n_ff 640 / n_embd 2560 なら 860 + 3,440 = 4,300 本。16 スレッドで
/// 269 本ずつなので、最後の 1 本の偏りは 1 タスク = 0.4% 程度。
///
/// 実測 (`moe_flat_schedule_microbench`, m=5..120 / 16 スレッド): 32 行 220 ms・
/// 64 行 210 ms・128 行 212 ms・256 行 213 ms。64〜256 は測定誤差 (±2%) の範囲で
/// 同等なので、偏りの小さい 64 を採る。
pub(crate) const MOE_FLAT_ROWS: usize = 64;

/// flat 経路へ切り替える閾値 (1 expert あたりの最大トークン数 `m_max`)。
///
/// 実測 (`moe_flat_schedule_microbench`, REPORT_P30.md) では **m = 1 でも flat が
/// 勝つ** (legacy の 10.5 ms/層 → flat 7.2 ms。1 層あたりの fork-join が 258 回 →
/// 2 回 + gather/scatter の 2 回になるため)。よって既定は閾値 0 = 常に flat。
/// legacy は `DENNO_MOE_FLAT=0` と、ビット一致の参照用テストに残してある。
pub(crate) const MOE_FLAT_MIN_M: usize = 0;

/// 選択 expert のループの並べ方。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MoeSchedule {
    /// `m_max` の閾値 ([`MOE_FLAT_MIN_M`]) で選ぶ (既定)
    Auto,
    /// 層全体の (expert, 行列, 行チャンク) を 1 本のタスク列にして 1 回だけ並列化する
    Flat,
    /// expert ごと × 行列ごとに 1 回ずつ rayon を立てる (従来の形)
    Legacy,
}

/// `DENNO_MOE_FLAT` の解釈 (純粋関数)。未設定・その他の値は [`MoeSchedule::Auto`]。
pub(crate) fn moe_schedule_from_value(v: Option<&str>) -> MoeSchedule {
    match v {
        Some("1") => MoeSchedule::Flat,
        Some("0") => MoeSchedule::Legacy,
        _ => MoeSchedule::Auto,
    }
}

/// expert ループの並べ方 (既定 Auto)。プロセスで 1 回だけ読む。
pub(crate) fn moe_schedule() -> MoeSchedule {
    static ON: std::sync::OnceLock<MoeSchedule> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_schedule_from_value(std::env::var("DENNO_MOE_FLAT").ok().as_deref()))
}

/// テスト用: flat 経路 / legacy 経路が実際に走った回数。
/// 「緑 = その経路が発動した証拠」ではないので、分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static MOE_FLAT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) static MOE_LEGACY_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// task 104: token 分割が実際に 2 以上に割った expert の数 (テスト用カウンタ)。
/// 「テストが緑」=「分割が発動した」ではないので、分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static MOE_TOKSPLIT_EXPERTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// flat スケジューラの token 分割の既定値。
///
/// 1 expert の m トークンを最大 `MAX` 個のチャンクに割る。m が expert 間で偏る
/// (実機プロファイル: mean 20-60 / max 492) ので、m の大きい expert の重みを
/// スケジューラが細かく配れるようにする。既定の最大 4 は「2〜4 分割」の中央で、
/// `MIN = 64` 未満の expert は割らない (小さすぎるタスクは rayon の任せると
/// オーバーヘッドが勝つ)。どちらも env で実機 A/B できる。
pub(crate) const MOE_TOK_SPLIT_MAX_DEF: usize = 4;

pub(crate) const MOE_TOK_SPLIT_MIN_DEF: usize = 64;

/// `DENNO_MOE_TOKSPLIT` の解釈 (純粋関数)。1 expert の最大 token チャンク数。
/// 未設定・数値でない値は既定。0 は無効、上限は 64 (それ以上はタスク表が無駄に太る)。
pub(crate) fn moe_tok_split_max_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(MOE_TOK_SPLIT_MAX_DEF)
        .min(64)
}

/// `DENNO_MOE_TOKSPLIT_MIN` の解釈 (純粋関数)。分割する最小の m。0 は「しきい値なし」。
pub(crate) fn moe_tok_split_min_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(MOE_TOK_SPLIT_MIN_DEF)
}

/// `DENNO_MOE_TOKSPLIT` をプロセスで 1 回だけ読む。
pub(crate) fn moe_tok_split_max() -> usize {
    #[cfg(test)]
    {
        let v = TOK_SPLIT_MAX_OVR.load(std::sync::atomic::Ordering::Relaxed);
        if v != usize::MAX {
            return v;
        }
    }
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        moe_tok_split_max_from_value(std::env::var("DENNO_MOE_TOKSPLIT").ok().as_deref())
    })
}

/// `DENNO_MOE_TOKSPLIT_MIN` をプロセスで 1 回だけ読む。
pub(crate) fn moe_tok_split_min() -> usize {
    #[cfg(test)]
    {
        let v = TOK_SPLIT_MIN_OVR.load(std::sync::atomic::Ordering::Relaxed);
        if v != usize::MAX {
            return v;
        }
    }
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        moe_tok_split_min_from_value(std::env::var("DENNO_MOE_TOKSPLIT_MIN").ok().as_deref())
    })
}

/// 1 expert の token チャンク数 (純粋関数)。
///
/// - `max_c <= 1` は常に 1 (無効)。
/// - `m < min_m` は 1 (割らない。`min_m == 0` はしきい値なし = 全 expert が対象)。
/// - それ以外は「1 チャンク `min_m` トークン以上」で割れる数 `m.div_ceil(min_m)` と
///   `max_c` の小さい方。
///
/// チャンク長は呼び出し側 ([`moe_push_tasks`] / gather / [`moe_quant_all`]) が
/// `m / c` + 端数で均等に切るので、どの分割数でも `[0, m)` を 1 度ずつ覆う。
pub(crate) fn moe_split_n(m: usize, max_c: usize, min_m: usize) -> usize {
    if max_c <= 1 || m <= 1 {
        return 1;
    }
    if min_m == 0 || m >= min_m {
        let by_len = if min_m == 0 { m } else { m.div_ceil(min_m) };
        return max_c.min(by_len).max(1);
    }
    1
}

/// flat スケジューラの 1 タスク (1 expert × token チャンク `[t0, t1)` × 行チャンク `[a, b)`)。
///
/// 行列の別 (gate / up / down) はフィールドに持たず、タスク列を 2 本に分ける。
/// gate と up は同じ入力 `xin` を読むので 1 タスクにまとめ、silu の積までその場で
/// 書く (3 回目のバリアを作らないため)。`pos` は active な expert の並び (`eid`) の添字。
///
/// task 104: token チャンクを追加した。m が expert 間で極端に偏る (実機: mean 20-60 /
/// max 492) ため、行チャンクだけだと 1 expert の全 m が行チャンク数個のタスクに付随し、
/// m 最大の expert が重い。トークンは互いに独立 (出力の 1 要素 = 1 行 × 1 トークン) なので、
/// 分割しても各要素の計算列は変わらずビット一致する。
#[derive(Clone, Copy)]
pub(crate) struct MoeTask {
    pub(crate) pos: u16,
    pub(crate) t0: u16,
    pub(crate) t1: u16,
    pub(crate) a: u16,
    pub(crate) b: u16,
}

/// 1 行チャンクのタスクを作る (u16 に収まらない形は黙って切らずにエラーで落とす)。
pub(crate) fn moe_task(pos: usize, t0: usize, t1: usize, a: usize, b: usize) -> Result<MoeTask> {
    let conv = |v: usize, what: &str| {
        u16::try_from(v).map_err(|_| one(format!("flat スケジューラ: {what} {v} が u16 を超える")))
    };
    Ok(MoeTask {
        pos: conv(pos, "expert 添字")?,
        t0: conv(t0, "トークン")?,
        t1: conv(t1, "トークン")?,
        a: conv(a, "行")?,
        b: conv(b, "行")?,
    })
}

/// token チャンク `[t0, t1)` × `n_out` 行を `rows` 行ずつのタスクへ切って `out` に積む。
pub(crate) fn moe_row_chunks(
    pos: usize,
    t0: usize,
    t1: usize,
    n_out: usize,
    rows: usize,
    out: &mut Vec<MoeTask>,
) -> Result<()> {
    let mut a = 0usize;
    while a < n_out {
        let b = (a + rows).min(n_out);
        out.push(moe_task(pos, t0, t1, a, b)?);
        a = b;
    }
    Ok(())
}

/// 1 expert ぶんのタスク (token チャンク × 行チャンク) を全部 `out` に積む。
///
/// token チャンク数は [`moe_split_n`] で決め、各チャンクの長さはできるだけ均等
/// (差 1 以下) に切る。チャンクは `[0, m)` を隙間なく 1 度ずつ覆うので、
/// 全タスクの和は従来 (チャンク 1 個) と同じ領域になる。
pub(crate) fn moe_push_tasks(
    pos: usize,
    m: usize,
    n_out: usize,
    rows: usize,
    max_c: usize,
    min_m: usize,
    out: &mut Vec<MoeTask>,
) -> Result<()> {
    let c = moe_split_n(m, max_c, min_m);
    #[cfg(test)]
    if c > 1 {
        MOE_TOKSPLIT_EXPERTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let (base, rem) = (m / c, m % c);
    let mut t0 = 0usize;
    for i in 0..c {
        let t1 = t0 + base + usize::from(i < rem);
        moe_row_chunks(pos, t0, t1, n_out, rows, out)?;
        t0 = t1;
    }
    Ok(())
}

/// `silu(v) * u`。`silu_inplace` と同じ式 (`v / (1 + exp(-v))`) の後に積を取る。
#[inline]
pub(crate) fn silu_mul(v: f32, u: f32) -> f32 {
    v / (1.0 + (-v).exp()) * u
}

/// flat スケジューラの作業バッファ。**層をまたいで使い回す**。
///
/// 層ごとに `vec![0.0; n]` で確保すると、初回タッチのページフォルトを毎層払う
/// (実形 m=64 で 4 本合計 140 MB ≈ 35,000 ページ。実測は REPORT_P30.md)。
/// 中身は「読む前に必ず全部書く」のでゼロ初期化には依存しない
/// (`gemm_rows` は行チャンクの全要素を書き、gather は `xin` を全部埋める)。
#[derive(Default)]
pub(crate) struct MoeScratch {
    pub(crate) xin: Vec<f32>,
    pub(crate) act: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) h: Vec<f32>,
    /// VNNI 経路の活性 (int8) とスケール・32 要素ごとの和。位相 A と B で使い回す
    /// (`quant_all` が長さを合わせる)。
    pub(crate) q8: Vec<i8>,
    pub(crate) qsc: Vec<f32>,
    pub(crate) qgs: Vec<f32>,
}

/// 活性 `src` (`tot × n_in`) を expert ごとに int8 へ量子化する (`q8` / `qsc` / `qgs`)。
///
/// expert の並びは gather と同じ (`off_x[pos] = その expert の先頭トークン × n_in`)。
/// 量子化の粒度は重みの型のブロック長 (`blk`)。
///
/// task 104: タスクは (expert, token チャンク) に切る。expert 単位のままだと
/// m 最大の expert が 1 タスクになり、この位相の straggler になる (量子化は
/// 1 expert あたり `m × n_in` 要素の走査なので、m の偏りがそのまま偏りになる)。
/// `quant_act_i8` はトークンごとに独立 (外側ループがトークン) なので、
/// 分割しても 1 ビットも変わらない。
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_quant_all(
    src: &[f32],
    ms: &[usize],
    off_x: &[usize],
    n_in: usize,
    blk: usize,
    tok_max: usize,
    tok_min: usize,
    q8: &mut Vec<i8>,
    qsc: &mut Vec<f32>,
    qgs: &mut Vec<f32>,
) {
    let tot: usize = ms.iter().sum();
    let nb = n_in / blk;
    let ng = n_in / 32;
    q8.resize(tot * n_in, 0);
    qsc.resize(tot * nb, 0.0);
    qgs.resize(tot * ng, 0.0);

    // タスク表 (pos, チャンク先頭トークン, チャンク長)。expert 内は連続で、
    // 全チャンクの長さの和が `tot` になる (3 バッファの分割と同じ順序)。
    let mut specs: Vec<(usize, usize, usize)> = Vec::new();
    for (pos, &m) in ms.iter().enumerate() {
        let c = moe_split_n(m, tok_max, tok_min);
        let (base, rem) = (m / c, m % c);
        let mut t0 = 0usize;
        for i in 0..c {
            let len = base + usize::from(i < rem);
            specs.push((pos, t0, len));
            t0 += len;
        }
    }

    // 3 本のバッファをタスク順に**非重複**で切る (task i が書くのは part i だけ)。
    let mut pq: Vec<&mut [i8]> = Vec::with_capacity(specs.len());
    let mut rest: &mut [i8] = &mut q8[..];
    for &(_, _, len) in &specs {
        let (h, t) = rest.split_at_mut(len * n_in);
        pq.push(h);
        rest = t;
    }
    let mut ps: Vec<&mut [f32]> = Vec::with_capacity(specs.len());
    let mut rest: &mut [f32] = &mut qsc[..];
    for &(_, _, len) in &specs {
        let (h, t) = rest.split_at_mut(len * nb);
        ps.push(h);
        rest = t;
    }
    let mut pg: Vec<&mut [f32]> = Vec::with_capacity(specs.len());
    let mut rest: &mut [f32] = &mut qgs[..];
    for &(_, _, len) in &specs {
        let (h, t) = rest.split_at_mut(len * ng);
        pg.push(h);
        rest = t;
    }
    let parts: Vec<(&mut [i8], &mut [f32], &mut [f32])> = pq
        .into_iter()
        .zip(ps)
        .zip(pg)
        .map(|((a, b), c)| (a, b, c))
        .collect();
    // チャンクごとに独立なので、どのスレッドが担当しても値は変わらない。
    let one = |(i, (q, sc, gs)): (usize, (&mut [i8], &mut [f32], &mut [f32]))| {
        let (pos, t0, len) = specs[i];
        let o = off_x[pos] + t0 * n_in;
        let x = &src[o..o + len * n_in];
        moe_expert::quant_act_i8(x, len, n_in, blk, q, sc, gs).expect("活性の量子化 (VNNI)")
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        parts.into_par_iter().enumerate().for_each(one);
    }
    #[cfg(not(feature = "par"))]
    {
        parts.into_iter().enumerate().for_each(one);
    }
}

/// task 291: タイル全体の活性 (`nt × n_in`、トークン順) を 1 回だけ int8 に量子化する。
///
/// gather をしない経路 (`DENNO_MOE_FLAT_IDX=1`) 用。トークンごとに独立なので、
/// 行をいくつに割っても値は変わらない ([`moe_expert::quant_act_i8`] の外側ループは
/// トークン)。量子化の結果は 1 トークンにつき 1 回だけ計算される
/// (gather 経路は expert の数だけ同じ行を量子化し直していた)。
pub(crate) fn moe_quant_tile(
    src: &[f32],
    nt: usize,
    n_in: usize,
    blk: usize,
    q8: &mut Vec<i8>,
    qsc: &mut Vec<f32>,
    qgs: &mut Vec<f32>,
) {
    let nb = n_in / blk;
    let ng = n_in / 32;
    q8.resize(nt * n_in, 0);
    qsc.resize(nt * nb, 0.0);
    qgs.resize(nt * ng, 0.0);
    if nt == 0 {
        return;
    }
    let one = |(i, ((q, sc), gs)): (usize, ((&mut [i8], &mut [f32]), &mut [f32]))| {
        let rows = q.len() / n_in;
        let t0 = i * rows;
        moe_expert::quant_act_i8(
            &src[t0 * n_in..(t0 + rows) * n_in],
            rows,
            n_in,
            blk,
            q,
            sc,
            gs,
        )
        .expect("活性の量子化 (タイル)");
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        let per = nt.div_ceil(rayon::current_num_threads().max(1)).max(1);
        q8.par_chunks_mut(per * n_in)
            .zip(qsc.par_chunks_mut(per * nb))
            .zip(qgs.par_chunks_mut(per * ng))
            .enumerate()
            .for_each(one);
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = &one;
        let rows = nt;
        moe_expert::quant_act_i8(src, rows, n_in, blk, q8, qsc, qgs)
            .expect("活性の量子化 (タイル)");
    }
}

/// task 291: 選択 expert のバッチ GEMM で gather を省く (`DENNO_MOE_FLAT_IDX`、既定 off)。
///
/// on のとき、位相 A (gate/up) の活性は**タイル全体を 1 回だけ量子化**し、
/// カーネルが expert ごとのトークン索引 (`VnniAct::tid`) で引く。
/// 値は gather 経路と同じ (行ごとの量子化は行に依存しない) なので、
/// 経路を切り替えても出力はビット一致する。
pub(crate) fn moe_flat_idx_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(std::env::var("DENNO_MOE_FLAT_IDX").ok().as_deref(), Some("1"))
    })
}

/// タスク表の `ahead` 個先のタスクが触る重みの行頭を先読みする。
///
/// expert の重みは毎層 DRAM から cold で来る (実測 1 層あたり 86〜255 expert × 約 3 MB)。
/// 1 つ先ではなく 2〜4 個先まで出す。行の残りは HW プリフェッチャに任せる。
pub(crate) fn moe_prefetch_task_ahead(table: &[MoeTask], i: usize, ahead: usize, blocks: &[&[u8]], rb: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let Some(t) = table.get(i + ahead) else { return };
        let Some(b) = blocks.get(t.pos as usize) else {
            return;
        };
        for r in t.a as usize..t.b as usize {
            // SAFETY: `r < n_out` で `b` は n_out 行ぶんのバイト列 (呼び出し側の契約)。
            let p = unsafe { b.as_ptr().add(r * rb) };
            unsafe {
                _mm_prefetch(p as *const i8, _MM_HINT_T0);
                _mm_prefetch(p.add(64) as *const i8, _MM_HINT_T0);
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (table, i, ahead, blocks, rb);
    }
}

/// 使い回し用の置き場 (プロセスで 1 つ)。使用中は `None` にして貸し出す。
pub(crate) static MOE_SCRATCH: std::sync::Mutex<Option<MoeScratch>> = std::sync::Mutex::new(None);

/// 借りる。使用中 (同時に 2 本走っている) なら新しく確保する = 壊れない。
pub(crate) fn moe_scratch_take() -> MoeScratch {
    MOE_SCRATCH
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default()
}

/// 返す。大きすぎるものは捨てる (巨大な層のあとメモリを抱え込まない)。
///
/// task 291: 上限を `DENNO_MOE_SCRATCH_KEEP_MB` で動かせるようにした (**既定 2048**)。
///
/// 実形 (tot = 3826 × 8 = 30,608、e 2560 / ff 640) の作業域は 782 MB で、旧固定値
/// 512 MiB では毎層捨てられ、次の層で `vec![0.0; ...]` のゼロ埋めを払い直していた
/// (lab t291a の実測 149 ms/層、prefill 全体で 7 s)。`=2048` にすると 48 層で
/// 使い回して UD-Q4_K_XL の 3,826 cold prefill が -20.7% になる (lab t291a:
/// 50.85 -> 40.30 s)。task 291 では target 30% に届かないため既定 off で置いたが、
/// task 297 で target 20% に対して採否 yes になり **既定 2048** にした。
/// `=512` で従来 (毎層捨てる) に戻せる。
pub(crate) fn moe_scratch_keep_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        moe_scratch_keep_mb_from_value(std::env::var("DENNO_MOE_SCRATCH_KEEP_MB").ok().as_deref())
            .saturating_mul(1024 * 1024)
    })
}

/// task 297: `DENNO_MOE_SCRATCH_KEEP_MB` の解釈 (純関数)。未設定・数値でない値は
/// 既定 2048 MiB。数値はそのまま (0 で毎層捨てる)。
pub(crate) fn moe_scratch_keep_mb_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(MOE_SCRATCH_KEEP_MB_DEFAULT)
}

/// task 297: 作業域を使い回す上限の既定 (MiB)。
pub(crate) const MOE_SCRATCH_KEEP_MB_DEFAULT: usize = 2048;

pub(crate) fn moe_scratch_put(s: MoeScratch) {
    let mut g = MOE_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    *g = if s.bytes() <= moe_scratch_keep_bytes() {
        Some(s)
    } else {
        None
    };
}

/// task 297: prefill の終端で作業域を OS へ返す (持ち越さない)。
///
/// 使い回しが効くのは **1 回の prefill の中** (48 層 × 同じチャンク) だけで、
/// prefill が終わったあとの decode には要らない。終端で持ち越すと実形で 782 MB を
/// decode のあいだ抱えたままになり、長いプロンプトのあとの decode でページキャッシュを
/// 圧迫する (lab t297b の既定モデル: warm short decode が +12%)。
/// `MOE_SCRATCH` は `None` に戻すだけで、次の prefill は先頭で 1 回だけ確保し直す。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn moe_scratch_release() {
    let mut g = MOE_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    *g = None;
}

/// 選択 expert の FFN を「層全体の (expert, 行列, 行チャンク) を 1 本のタスク列に
/// して 1 回だけ並列化する」形で計算し、`out` に重み付きで足し込む (呼び出し側で
/// ゼロ初期化済み)。
///
/// task 151: `capture` が `Some` のときは **`out` を触らず**、expert ごとの
/// 出力行 `h` をその Vec へ返す (`tot × e`)。prefill の GPU 経路が
/// 「ヒット (GPU) とミス (CPU) を expert index 昇順で 1 本ずつ足す」ために使う。
/// 受け渡しは swap なので、層ごとの再確保は起きない。
///
/// 従来の [`moe_experts_legacy`] は expert 1 体につき gate / up / down で 3 回
/// rayon の fork-join をしていた (86 expert × 3 × 48 層 = 12,384 回/chunk)。
/// ここでは 1 層あたり「gate+up+silu 積」の 1 パスと「down」の 1 パスの計 2 回に減らす。
/// down の入力は gate/up の結果 (silu の積) なので、この 2 パスは分けざるをえない。
///
/// # 数値 (ビット一致の根拠)
///
/// - 1 要素の計算は `moe_expert::expert_gemm_any` と**同じ関数** (`moe_expert::gemm_rows`)
///   を同じ引数 (同じ `n_in` / `m` / 行) で呼ぶ。行チャンクは `[0, n_out)` の分割で、
///   行の中の累積順は行内で閉じるので、どのスレッドがどの行を計算しても値は変わらない。
/// - task 104: タスクは (expert, token チャンク, 行チャンク) に切る。出力の 1 要素は
///   「1 行 × 1 トークン」で独立、1 要素の累積順は行のブロック昇順でトークンの
///   分割に依らないので、token を割っても 1 ビットも変わらない
///   (量子化 [`moe_quant_all`] と gather もトークンごとに独立)。
/// - silu の積は `v / (1 + (-v).exp()) * u` で、legacy の `silu_inplace` + `*=` と同順。
/// - 足し込みは expert index 昇順 → トークン昇順の逐次 (legacy と同順)。
/// - タスク表は **m の降順** に並べるが、タスクは (expert, token チャンク, 行チャンク)
///   ごとに独立な領域を書くだけなので順序は値に影響しない (gather / scatter は eid 昇順)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_experts_flat(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    e: usize,
    ff: usize,
    rows: usize,
    out: &mut [f32],
    vnni: bool,
    capture: Option<&mut Vec<f32>>,
) -> Result<()> {
    #[cfg(test)]
    MOE_FLAT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // 1. expert の並びと、expert ごとの領域の先頭 (f32 要素) を確定する。
    let mut eid: Vec<usize> = Vec::new();
    let mut ms: Vec<usize> = Vec::new();
    let mut off_x: Vec<usize> = Vec::new();
    let mut off_a: Vec<usize> = Vec::new();
    let mut off_h: Vec<usize> = Vec::new();
    let mut tot = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        eid.push(ei);
        ms.push(picks.len());
        off_x.push(tot * e);
        off_a.push(tot * ff);
        off_h.push(tot * e);
        tot += picks.len();
    }
    let n = eid.len();
    if n == 0 {
        return Ok(());
    }
    // P85: kprof に expert 数を積む (0 トークンの expert はここで落ちるので
    // 呼び出し側で数える。off のときは no-op)。
    crate::model::gemm_iq_kprof::record_experts(n, by_expert.len());

    // フェーズ内訳 (`DENNO_MOE_FLAT_PHASE=1` のときだけ stderr へ出す)。
    let ph = std::env::var_os("DENNO_MOE_FLAT_PHASE").is_some();
    let mut ph_t = std::time::Instant::now();
    macro_rules! lap {
        ($name:expr) => {
            if ph {
                eprintln!("[moe-flat] {:.1} ms {}", ph_t.elapsed().as_secs_f64() * 1e3, $name);
                ph_t = std::time::Instant::now();
            }
        };
    }

    // 2. 作業バッファ (層をまたいで使い回す) を必要な長さに合わせる。
    //    task 291: 索引付き経路では gather をしないので `xin` を取らない。
    let (g, u, d) = (&lw.gate_exps, &lw.up_exps, &lw.down_exps);
    let (gty, uty, dty) = (g.ty, u.ty, d.ty);
    let idx_mode = moe_flat_idx_on()
        && vnni
        && moe_expert::vnni_supported()
        && moe_expert::vnni_act_block(gty).is_some()
        && moe_expert::vnni_act_block(gty) == moe_expert::vnni_act_block(uty);
    let mut sc = moe_scratch_take();
    sc.resize(tot, e, ff, !idx_mode);
    lap!("resize");
    let MoeScratch {
        mut xin,
        mut act,
        mut up,
        mut h,
        mut q8,
        mut qsc,
        mut qgs,
    } = sc;
    // task 151: `capture` があれば、呼び出し側のバッファを h として使う
    // (結果は swap で返す)。長さは resize 済みなので両方 `tot * e`。
    let mut cap = capture;
    if let Some(v) = cap.as_deref_mut() {
        v.resize(tot * e, 0.0);
        std::mem::swap(&mut h, v);
    }

    // 3. gather: expert ごとにトークンを詰め直す (legacy と同じ順序 = トークン昇順)。
    //    task 104: m の大きい expert は token チャンクに割って複数タスクにする
    //    (expert 単位のままだと m 最大の expert が gather の straggler になる)。
    //    expert ごとに独立なコピーなので、分割しても値は変わらない。
    let (tok_max, tok_min) = (moe_tok_split_max(), moe_tok_split_min());
    // task 291: 索引付き経路は gather の代わりに「expert → タイル内トークン番号」の
    // 表だけを作る (活性はトークン順のまま、カーネルが索引で引く)。
    let mut ids: Vec<u32> = Vec::new();
    let mut off_t: Vec<usize> = Vec::new();
    if idx_mode {
        ids.reserve(tot);
        off_t.reserve(n);
        for pos in 0..n {
            off_t.push(ids.len());
            for &(t, _) in &by_expert[eid[pos]] {
                ids.push(t as u32);
            }
        }
        debug_assert_eq!(ids.len(), tot);
    } else {
    {
        // (pos, チャンクの先頭トークン, 長さ)。expert 内のチャンクは連続。
        let mut gspec: Vec<(usize, usize, usize)> = Vec::new();
        for (pos, &m) in ms.iter().enumerate() {
            let c = moe_split_n(m, tok_max, tok_min);
            let (base, rem) = (m / c, m % c);
            let mut t0 = 0usize;
            for i in 0..c {
                let len = base + usize::from(i < rem);
                gspec.push((pos, t0, len));
                t0 += len;
            }
        }
        let mut parts: Vec<&mut [f32]> = Vec::with_capacity(gspec.len());
        let mut rest: &mut [f32] = &mut xin[..];
        for &(_, _, len) in &gspec {
            let (head, tail) = rest.split_at_mut(len * e);
            parts.push(head);
            rest = tail;
        }
        let gather_one = |(gi, dst): (usize, &mut [f32])| {
            let (pos, t0, len) = gspec[gi];
            let ei = eid[pos];
            for (idx, &(t, _)) in by_expert[ei].iter().skip(t0).take(len).enumerate() {
                dst[idx * e..(idx + 1) * e].copy_from_slice(&xs[t * e..(t + 1) * e]);
            }
        };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            parts.into_par_iter().enumerate().for_each(gather_one);
        }
        #[cfg(not(feature = "par"))]
        parts.into_iter().enumerate().for_each(gather_one);
    }
    }
    lap!("gather");

    // 4. 行列・入力・出力長の検証 (1 expert 1 行列につき 1 度だけ) とバイト列の切り出し
    let rb_g = moe_expert::row_bytes(gty, e).map_err(|err| one(format!("{}: {err}", g.name)))?;
    let rb_u = moe_expert::row_bytes(uty, e).map_err(|err| one(format!("{}: {err}", u.name)))?;
    let rb_d = moe_expert::row_bytes(dty, ff).map_err(|err| one(format!("{}: {err}", d.name)))?;
    let mut gs: Vec<&[u8]> = Vec::with_capacity(n);
    let mut us: Vec<&[u8]> = Vec::with_capacity(n);
    let mut ds: Vec<&[u8]> = Vec::with_capacity(n);
    for pos in 0..n {
        let m = ms[pos];
        let ei = eid[pos];
        let ad = &act[off_a[pos]..off_a[pos] + m * ff];
        let bg = g.slice(ei)?;
        let bu = u.slice(ei)?;
        let bd = d.slice(ei)?;
        if idx_mode {
            // task 291: 活性はタイル全体 (トークン順) を索引で引くので、行の
            // 連続スライスは無い。形状だけを検証する。
            moe_expert::check_gemm_matrix_shape(gty, bg, e, ff, m)
                .map_err(|err| one(format!("{}: expert {ei}: {err}", g.name)))?;
            moe_expert::check_gemm_matrix_shape(uty, bu, e, ff, m)
                .map_err(|err| one(format!("{}: expert {ei}: {err}", u.name)))?;
        } else {
            let xa = &xin[off_x[pos]..off_x[pos] + m * e];
            moe_expert::check_gemm_matrix(gty, bg, e, ff, xa, m)
                .map_err(|err| one(format!("{}: expert {ei}: {err}", g.name)))?;
            moe_expert::check_gemm_matrix(uty, bu, e, ff, xa, m)
                .map_err(|err| one(format!("{}: expert {ei}: {err}", u.name)))?;
        }
        moe_expert::check_gemm_matrix(dty, bd, ff, e, ad, m)
            .map_err(|err| one(format!("{}: expert {ei}: {err}", d.name)))?;
        gs.push(bg);
        us.push(bu);
        ds.push(bd);
    }
    lap!("validate+slices+task_table");

    // 5. タスク表 (gate+up+silu 積 / down)。**m の降順** に並べる
    //    (m の大きい expert から先にタスク列へ入るので、rayon の分割で
    //    重い仕事が先に始まる)。タスクは独立な領域を書くので順序は値に影響しない。
    //    task 104: 1 expert のタスクは (token チャンク × 行チャンク) に切る
    //    ([`moe_push_tasks`])。m 最大の expert の 1 タスクの重みが 1/C になり、
    //    行チャンクだけの分割より末尾の偏りが小さい。
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ms[b].cmp(&ms[a]).then_with(|| a.cmp(&b)));
    let mut ta: Vec<MoeTask> = Vec::new();
    let mut tb: Vec<MoeTask> = Vec::new();
    let rows = rows.max(1);
    for &pos in &order {
        moe_push_tasks(pos, ms[pos], ff, rows, tok_max, tok_min, &mut ta)?;
        moe_push_tasks(pos, ms[pos], e, rows, tok_max, tok_min, &mut tb)?;
    }

    // SAFETY (act / up / h をタスク間で共有してよい根拠):
    // - タスク (pos, [t0,t1), [a,b)) が書くのは expert `pos` の領域の
    //   `t * n_out + [a, b)` (t0 <= t < t1) だけで、token チャンク × 行チャンクは
    //   タスク間で重ならない (token も行も `[0, m)` / `[0, n_out)` の分割)。
    // - 領域の大きさは `tot * ff` / `tot * e` で、添字は必ずその内側。
    // - `moe_expert::gemm_rows` は `m * stride` 要素以上を書き、行チャンクの外へは書かない。
    // よって異なるタスクが同じ要素へ書くことはない (data race なし)。
    let act_p = moe_expert::OutPtr::new(act.as_mut_ptr());
    let up_p = moe_expert::OutPtr::new(up.as_mut_ptr());
    let h_p = moe_expert::OutPtr::new(h.as_mut_ptr());

    // 5.5 VNNI (AVX-512 `vpdpbusd`): 活性を expert ごとに 1 回だけ int8 へ量子化する。
    //     粒度は重みの型のブロック長 (Q4_K 256 / Q5_1 32)。gate と up は同じ `xin` を
    //     読むので、位相 A の量子化は 1 回で両方に使える (型のブロック長が同じときだけ)。
    //     `DENNO_VNNI=0` と、VNNI 非対応 CPU では `blk_a`/`blk_b` が `None` になり
    //     従来の f32 経路 ([`moe_expert::gemm_rows`]) のまま動く。
    let blk_g = moe_expert::vnni_act_block(gty);
    let blk_u = moe_expert::vnni_act_block(uty);
    let blk_d = moe_expert::vnni_act_block(dty);
    let blk_a = if !vnni {
        None
    } else if blk_g.is_some() {
        blk_g
    } else if blk_u.is_some() {
        blk_u
    } else {
        None
    };
    let use_g = blk_a.is_some() && blk_g == blk_a;
    let use_u = blk_a.is_some() && blk_u == blk_a;
    let use_d = vnni && blk_d.is_some();
    // task 291: 索引付き経路ではタイル全体を 1 回だけ量子化する (gather 無し)。
    let nt = out.len() / e.max(1);
    // expert ごとの先頭 (ブロック単位)。`off_x` は f32 要素単位なので割って使う。
    let mut off_s: Vec<usize> = Vec::with_capacity(n);
    let mut off_g: Vec<usize> = Vec::with_capacity(n);
    if let Some(blk) = blk_a {
        if idx_mode {
            moe_quant_tile(&xs, nt, e, blk, &mut q8, &mut qsc, &mut qgs);
        } else {
            let (nb, ng) = (e / blk, e / 32);
            let (mut so, mut go) = (0usize, 0usize);
            for &m in ms.iter() {
                off_s.push(so);
                off_g.push(go);
                so += m * nb;
                go += m * ng;
            }
            moe_quant_all(&xin, &ms, &off_x, e, blk, tok_max, tok_min, &mut q8, &mut qsc, &mut qgs);
        }
    }
    let nb_a = blk_a.map(|b| e / b).unwrap_or(0);
    let ng_a = e / 32;

    // 6. gate+up+silu 積 (行チャンクごとに 1 タスク)
    //    索引付き経路で VNNI が受け取らなかったら記録して層を落とす (値が変わるため)。
    let idx_bad = std::sync::atomic::AtomicBool::new(false);
    let run_a = |(i, t): (usize, &MoeTask)| {
        let pos = t.pos as usize;
        let (a, b) = (t.a as usize, t.b as usize);
        let (t0, t1) = (t.t0 as usize, t.t1 as usize);
        let m = t1 - t0;
        let x: &[f32] = if idx_mode {
            &[]
        } else {
            &xin[off_x[pos] + t0 * e..off_x[pos] + t1 * e]
        };
        let dst_a = act_p.add(off_a[pos] + t0 * ff + a);
        let dst_u = up_p.add(off_a[pos] + t0 * ff + a);
        // 2〜4 個先のタスクの重みを先読みする (層のタスク表は expert ごとに cold)。
        moe_prefetch_task_ahead(&ta, i, 3, &gs, rb_g);
        moe_prefetch_task_ahead(&ta, i, 3, &us, rb_u);
        // 前提 (安全に呼べる条件): `[a, b)` は [0, n_out) の分割の 1 つで、領域は
        // `m * stride` 要素以上。`gemm_rows` 自身は unsafe fn ではないが、生ポインタの
        // 指す先の条件はこのモジュールのタスク分割が保証する (上の SAFETY を参照)。
        // VNNI を使わない位相では空の活性を渡す (使わないので中身は見られない)。
        let mut act_g = moe_expert::VnniAct {
            q: &[],
            sc: &[],
            gs: &[],
            m,
            n_in: e,
            blk: 0,
            tid: &[],
        };
        if let Some(blk) = blk_a {
            act_g = if idx_mode {
                // task 291: 活性はタイル順のまま。この expert のトークンを索引で引く。
                moe_expert::VnniAct {
                    q: &q8[..nt * e],
                    sc: &qsc[..nt * nb_a],
                    gs: &qgs[..nt * ng_a],
                    m,
                    n_in: e,
                    blk,
                    tid: &ids[off_t[pos] + t0..off_t[pos] + t1],
                }
            } else {
                moe_expert::VnniAct {
                    q: &q8[off_x[pos] + t0 * e..off_x[pos] + t1 * e],
                    sc: &qsc[off_s[pos] + t0 * nb_a..off_s[pos] + t1 * nb_a],
                    gs: &qgs[off_g[pos] + t0 * ng_a..off_g[pos] + t1 * ng_a],
                    m,
                    n_in: e,
                    blk,
                    tid: &[],
                }
            };
        }
        // task 95: `DENNO_EXP_KPROF=1` のとき、バッチ GEMM の型別記録が行列を
        // 知る必要がある。off では `set_mat` が分岐 1 回で戻る。
        for (mat, (ty, rb, w, dst, use_vnni)) in [
            (
                crate::model::exp_kprof::GATE,
                (gty, rb_g, gs[pos], dst_a.get(), use_g),
            ),
            (
                crate::model::exp_kprof::UP,
                (uty, rb_u, us[pos], dst_u.get(), use_u),
            ),
        ] {
            crate::model::exp_kprof::set_mat(mat);
            let used =
                use_vnni && moe_expert::gemm_rows_vnni(ty, w, e, a, b, &act_g, dst, ff);
            if !used {
                if idx_mode {
                    // 索引付き経路は活性の行が連続でないので f32 へ落とせない。
                    // 前提は呼び出し前に検査済みなので、ここへ来たら層を落とす。
                    idx_bad.store(true, std::sync::atomic::Ordering::Relaxed);
                } else {
                    moe_expert::gemm_rows(ty, rb, w, e, a, b, x, m, dst, ff, true);
                }
            }
        }
        // act の行 [a, b) を silu(act) * up で置き換える (このタスクが書いた行だけ)。
        let pa = act_p.get();
        let pu = up_p.get();
        for idx in t0..t1 {
            let base = off_a[pos] + idx * ff;
            for r in a..b {
                // SAFETY: base + r はこのタスクが上で書いた範囲 (他タスクと重ならない)。
                unsafe {
                    let v = *pa.add(base + r);
                    *pa.add(base + r) = silu_mul(v, *pu.add(base + r));
                }
            }
        }
    };

    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        ta.par_iter().enumerate().for_each(run_a);
    }
    #[cfg(not(feature = "par"))]
    {
        ta.iter().enumerate().for_each(run_a);
    }
    if idx_bad.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(one(format!(
            "層 {}: 索引付き VNNI 経路の前提が崩れた (gemm_rows_vnni が受取拒否)",
            lw.il
        )));
    }
    lap!("phaseA");

    // 6.5 位相 B の活性 (silu の積) を int8 に量子化する (down は Q5_1 = 32 要素)。
    let (nb_b, off_sb, off_gb) = if use_d {
        let blk = blk_d.unwrap_or(32);
        let (nb, ng) = (ff / blk, ff / 32);
        let mut offs: Vec<usize> = Vec::with_capacity(n);
        let mut offg: Vec<usize> = Vec::with_capacity(n);
        let (mut so, mut go) = (0usize, 0usize);
        for &m in ms.iter() {
            offs.push(so);
            offg.push(go);
            so += m * nb;
            go += m * ng;
        }
        let act_r: &[f32] = &act;
        moe_quant_all(act_r, &ms, &off_a, ff, blk, tok_max, tok_min, &mut q8, &mut qsc, &mut qgs);
        (nb, offs, offg)
    } else {
        (0, Vec::new(), Vec::new())
    };

    // 7. down (act は前のパスで確定しているので読むだけ)
    let act_r: &[f32] = &act;
    let run_b = |(i, t): (usize, &MoeTask)| {
        let pos = t.pos as usize;
        let (a, b) = (t.a as usize, t.b as usize);
        let (t0, t1) = (t.t0 as usize, t.t1 as usize);
        let m = t1 - t0;
        let x = &act_r[off_a[pos] + t0 * ff..off_a[pos] + t1 * ff];
        let dst = h_p.add(off_h[pos] + t0 * e + a);
        crate::model::exp_kprof::set_mat(crate::model::exp_kprof::DOWN);
        moe_prefetch_task_ahead(&tb, i, 3, &ds, rb_d);
        // 前提は run_a と同じ (`[a, b)` は [0, n_out) の分割の 1 つ、領域は `m * stride` 以上)。
        if use_d {
            let act_d = moe_expert::VnniAct {
                q: &q8[off_a[pos] + t0 * ff..off_a[pos] + t1 * ff],
                sc: &qsc[off_sb[pos] + t0 * nb_b..off_sb[pos] + t1 * nb_b],
                gs: &qgs[off_gb[pos] + t0 * (ff / 32)..off_gb[pos] + t1 * (ff / 32)],
                m,
                n_in: ff,
                tid: &[],
                blk: blk_d.unwrap_or(32),
            };
            if moe_expert::gemm_rows_vnni(dty, ds[pos], ff, a, b, &act_d, dst.get(), e) {
                return;
            }
        }
        moe_expert::gemm_rows(dty, rb_d, ds[pos], ff, a, b, x, m, dst.get(), e, true);
    };

    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        tb.par_iter().enumerate().for_each(run_b);
    }
    #[cfg(not(feature = "par"))]
    {
        tb.iter().enumerate().for_each(run_b);
    }
    lap!("phaseB");

    // 8. 重み付きの足し込み。トークンごとに書く先が違う (`out[t*e..]`) ので
    //    トークン単位で並列化する。1 トークンの中は expert 昇順の逐次 = legacy と同順。
    //    `capture` のときは足さない (呼び出し側がヒットと混ぜて 1 回だけ足す)。
    if cap.is_none() {
        let nt = out.len() / e.max(1);
        // (pos, expert 内の添字, 重みの添字)。pos 昇順に積むので index 昇順が保たれる。
        let mut by_token: Vec<Vec<(u16, usize, u16)>> = vec![Vec::new(); nt];
        for (pos, &ei) in eid.iter().enumerate() {
            let pos_u16 = moe_task(pos, 0, 0, 0, 0)?.pos;
            for (idx, &(t, j)) in by_expert[ei].iter().enumerate() {
                by_token[t].push((pos_u16, idx, j as u16));
            }
        }
        let mut parts: Vec<&mut [f32]> = out.chunks_mut(e.max(1)).collect();
        let scatter_one = |(t, o): (usize, &mut [f32])| {
            for &(pos, idx, j) in &by_token[t] {
                let pos = pos as usize;
                let pe = weights[t * k + j as usize];
                let hh = &h[off_h[pos] + idx * e..off_h[pos] + (idx + 1) * e];
                for d in 0..e {
                    o[d] += pe * hh[d];
                }
            }
        };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            parts.into_par_iter().enumerate().for_each(scatter_one);
        }
        #[cfg(not(feature = "par"))]
        parts.into_iter().enumerate().for_each(scatter_one);
    }
    lap!("scatter");
    if let Some(v) = cap.as_deref_mut() {
        // 結果を呼び出し側へ、flat のバッファを scratch へ戻す。
        std::mem::swap(&mut h, v);
    }
    moe_scratch_put(MoeScratch {
        xin,
        act,
        up,
        h,
        q8,
        qsc,
        qgs,
    });
    Ok(())
}

/// 選択 expert の FFN を expert ごとに回す (従来の形)。
///
/// expert 1 体につき gate / up / down で 3 回 rayon の fork-join をする。
/// `m == 1` の expert は 1 トークン用の入口 ([`ExpertW::gemv`]) を通る。
/// `m_max` が小さい層 ([`MOE_FLAT_MIN_M`] 以下) ではこちらが速い
/// (`moe_flat_schedule_microbench` の実測)。
///
/// 数値: 各 expert の `h_e` は (行, トークン) ごとに逐次版と同じカーネルで計算し、
/// 足し込みは expert index 昇順 → トークン昇順の逐次。`moe_ffn_quant` と一致する。
pub(crate) fn moe_experts_legacy(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    e: usize,
    ff: usize,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(test)]
    MOE_LEGACY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    for ei in 0..by_expert.len() {
        let picks = &by_expert[ei];
        if picks.is_empty() {
            continue;
        }
        let m = picks.len();
        let mut xin = Vec::with_capacity(m * e);
        for &(t, _) in picks {
            xin.extend_from_slice(&xs[t * e..(t + 1) * e]);
        }
        let mut act = vec![0.0f32; m * ff];
        let mut up = vec![0.0f32; m * ff];
        let mut h = vec![0.0f32; m * e];
        if m == 1 {
            // 1 トークンなら逐次版とまったく同じ入口を使う (行並列も従来どおり)。
            lw.gate_exps.gemv(ei, &xin, &mut act)?;
            silu_inplace(&mut act);
            lw.up_exps.gemv(ei, &xin, &mut up)?;
            for r in 0..m * ff {
                act[r] *= up[r];
            }
            lw.down_exps.gemv(ei, &act, &mut h)?;
        } else {
            expert_gemv_batch(&lw.gate_exps, ei, &xin, m, &mut act)?;
            silu_inplace(&mut act);
            expert_gemv_batch(&lw.up_exps, ei, &xin, m, &mut up)?;
            for r in 0..m * ff {
                act[r] *= up[r];
            }
            expert_gemv_batch(&lw.down_exps, ei, &act, m, &mut h)?;
        }
        // 重み付きの足し込みはトークンごとに逐次 (index 昇順のまま)
        for (idx, &(t, j)) in picks.iter().enumerate() {
            let pe = weights[t * k + j];
            let hh = &h[idx * e..(idx + 1) * e];
            for d in 0..e {
                out[t * e + d] += pe * hh[d];
            }
        }
    }
    Ok(())
}

/// テスト用: 並列の寄与計算を何回走らせたか (P-MOE-G5 の分岐が実際に通ったことの証拠)。
#[cfg(test)]
pub(crate) static PAR_CONTRIB_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: 従来の並列経路 [`expert_contribs_par`] が**並列領域**を立てた回数
/// (層 1 回 = 1)。P64 の「1 層 1 回」の比較対象。
#[cfg(test)]
pub(crate) static PAR_CONTRIB_REGIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: 新経路 [`expert_contribs_loop`] が**並列領域**を立てた回数
/// (層 1 回 = 1)。P64 の「1 層 1 回」の証拠。
#[cfg(test)]
pub(crate) static EXPERT_LOOP_REGIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: P70 の平坦化経路 ([`expert_contribs_flat_decode_into`]) が
/// **層 1 回ぶんの計算**を回した回数 (層 1 回 = 1)。
#[cfg(test)]
pub(crate) static FLAT_DECODE_REGIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// テスト用: 平坦化経路が積んだ**行タスク**の本数
/// (`expert × 行列 × 行チャンク`。実形 10 expert / 64 行なら 600/層)。
#[cfg(test)]
pub(crate) static FLAT_DECODE_TASKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 選択 expert の寄与 `h_e` (長さ `e = n_embd`) を rayon で並列に計算する (P-MOE-G5)。
///
/// 返り値の `hs[t]` は `sel[t]` の寄与で、**足し込みは呼び出し側が index 昇順に
/// 逐次行う** (丸めの順序を変えないため。ここで足してはいけない)。
///
/// 内側は [`ExpertW::gemv_seq`] (行並列なし) を使う。外側が expert 並列なので、
/// 内側でも行を分けるとスレッドが二重に立つ。行分割の有無は各要素の計算順序を
/// 変えないので、逐次版とビット一致する。
#[cfg(feature = "par")]
pub(crate) fn expert_contribs_par(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
    cache: Option<&ExpertCacheSet>,
) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    #[cfg(test)]
    PAR_CONTRIB_REGIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    sel.par_iter()
        .map(|&ei| {
            let mut act = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            let mut h = vec![0.0f32; e];
            match cache {
                // 従来経路: mmap のスライスを直接 GEMV する (行並列なし)。
                None => {
                    lw.gate_exps.gemv_seq(ei, x, &mut act)?;
                    silu_inplace(&mut act);
                    lw.up_exps.gemv_seq(ei, x, &mut up)?;
                    for r in 0..ff {
                        act[r] *= up[r];
                    }
                    lw.down_exps.gemv_seq(ei, &act, &mut h)?;
                }
                // P38: 常駐キャッシュ経路。バイト列の出所だけが違い、GEMV の入口は
                // mmap 経路の `gemv_seq` と同じ `expert_gemv_any_seq`。
                Some(c) => {
                    expert_gemv_cached_seq(
                        c,
                        ExpertMatrix::Gate,
                        &lw.gate_exps,
                        lw.il,
                        ei,
                        x,
                        &mut act,
                    )?;
                    silu_inplace(&mut act);
                    expert_gemv_cached_seq(
                        c,
                        ExpertMatrix::Up,
                        &lw.up_exps,
                        lw.il,
                        ei,
                        x,
                        &mut up,
                    )?;
                    for r in 0..ff {
                        act[r] *= up[r];
                    }
                    expert_gemv_cached_seq(
                        c,
                        ExpertMatrix::Down,
                        &lw.down_exps,
                        lw.il,
                        ei,
                        &act,
                        &mut h,
                    )?;
                }
            }
            #[cfg(test)]
            PAR_CONTRIB_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(h)
        })
        .collect()
}

/// task 67: CPU expert GEMV の作業域 (`act` / `up` / `h`)。
///
/// 27B の形 (e = 2560 / ff = 640) では 1 expert あたり act + up = 5 KB、h = 10 KB。
/// 従来は expert ごとに `vec![0.0; ..]` を確保していた (1 トークン = 48 層 ×
/// top-10 × 3 本 ≈ 7.4 MB の確保)。**使う前に必ずゼロ埋めする**ので、前の expert の
/// 値が残っていても出力は変わらない。
///
/// P64 の新経路 ([`expert_contribs_loop`]) は `act` / `up` だけを使う
/// (寄与の出力は呼び出し側のバッファへ直接書く)。
#[derive(Default)]
pub(crate) struct ExpertScratch {
    pub(crate) act: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) h: Vec<f32>,
}

thread_local! {
    /// スレッドごとに 1 本。rayon のワーカーも OS スレッドなので、層をまたいで
    /// 同じバッファを使い回す (スレッドをまたいで共有しない = データ競合しない)。
    pub(crate) static EXPERT_SCRATCH: std::cell::RefCell<ExpertScratch> =
        std::cell::RefCell::new(ExpertScratch::default());
}

/// task 67: [`expert_contribs_par`] の**確保なし**版。寄与を `hs[t * e + j]` へ直接書く。
///
/// 1 expert の計算は従来と同じ [`ExpertW::gemv_seq`] を同じ順で通すので、寄与は
/// 従来経路と**ビット一致**する。足し込みは呼び出し側が index 昇順に逐次行う
/// (ここでは足さない)。作業域は [`with_expert_scratch`] から借りるので、expert
/// ごとの `Vec` 確保が無い (`hs` は呼び出し側が 1 回だけ確保する)。
#[cfg(feature = "par")]
pub(crate) fn expert_contribs_par_into(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
    cache: Option<&ExpertCacheSet>,
    hs: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;
    #[cfg(test)]
    PAR_CONTRIB_REGIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    hs.par_chunks_mut(e)
        .zip(sel.par_iter())
        .try_for_each(|(h, &ei)| -> Result<()> {
            with_expert_scratch(ff, e, |s| -> Result<()> {
                let act = &mut s.act[..ff];
                let up = &mut s.up[..ff];
                match cache {
                    // 従来経路: mmap のスライスを直接 GEMV する (行並列なし)。
                    None => {
                        lw.gate_exps.gemv_seq(ei, x, act)?;
                        silu_inplace(act);
                        lw.up_exps.gemv_seq(ei, x, up)?;
                        for r in 0..ff {
                            act[r] *= up[r];
                        }
                        lw.down_exps.gemv_seq(ei, act, h)?;
                    }
                    // 常駐キャッシュ経路。バイト列の出所だけが違う。
                    Some(c) => {
                        expert_gemv_cached_seq(c, ExpertMatrix::Gate, &lw.gate_exps, lw.il, ei, x, act)?;
                        silu_inplace(act);
                        expert_gemv_cached_seq(c, ExpertMatrix::Up, &lw.up_exps, lw.il, ei, x, up)?;
                        for r in 0..ff {
                            act[r] *= up[r];
                        }
                        expert_gemv_cached_seq(c, ExpertMatrix::Down, &lw.down_exps, lw.il, ei, act, h)?;
                    }
                }
                #[cfg(test)]
                PAR_CONTRIB_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })
        })
}

/// 1 層ぶんの expert 寄与を **expert 横断の行タスク**に平坦化して計算する
/// (P70, `DENNO_EXPERT_FLAT_DECODE`)。結果は `hs[t * e + j]` に書く
/// (`hs` は `sel.len() * e` 要素)。
///
/// # なぜ平坦化するか
///
/// 従来の [`expert_contribs_par_into`] は **1 expert = 1 タスク**で、10 expert を
/// スレッド数 (8〜16) で割ると端数が出る (10/16 = 1 巡 + 6 スレッド遊び)。
/// ここでは「expert × 行列 × 行チャンク」を 1 本のタスク列にするので、
/// 重みの読み出しが全スレッドに均等に割り当たる。位相は 2 つ:
///
/// 1. gate / up の全行 (入力はどちらも `x`)
/// 2. down の全行 (入力は `silu(gate) * up`)
///
/// down は自分の expert の `act` 全要素を要るので、位相をまたぐバリアが 1 本要る。
///
/// # ビット一致の根拠
///
/// 1 行の計算は [`ExpertW::gemv_rows_seq`] が [`ExpertW::gemv_seq`] と同じ行
/// カーネルを同じ行順で通す (行は独立で、行チャンクの境界は各要素の積和に
/// 影響しない)。`silu` と `act *= up` も従来と同じ式・同じ順で、足し込みは
/// 呼び出し側が index 昇順に逐次行う。
#[cfg(feature = "par")]
pub(crate) fn expert_contribs_flat_decode_into(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
    hs: &mut [f32],
    rows: usize,
) -> Result<()> {
    use rayon::prelude::*;
    if rows == 0 {
        return Err(one("expert 平坦化: 行チャンク幅が 0".to_string()));
    }
    if hs.len() != sel.len() * e {
        return Err(one(format!(
            "expert 平坦化: hs の長さ {} が sel.len() * e = {} と違う",
            hs.len(),
            sel.len() * e
        )));
    }
    #[cfg(test)]
    FLAT_DECODE_REGIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // `DENNO_EXP_KPROF=1` のときだけ区間を積む (既定 off では `stamp()` が None)。
    use crate::model::exp_kprof as kp;
    kp::bump_layer();
    let t_phase = kp::stamp();
    let t_gather1 = kp::stamp();

    // gate / up の中間結果を 1 本ずつ確保する (sel.len() * ff 要素。実形で 25 KB)。
    let mut act = vec![0.0f32; sel.len() * ff];
    let mut up = vec![0.0f32; sel.len() * ff];

    /// 1 タスク = 1 行チャンク。`mat` は 0 = gate / 1 = up。
    struct Job<'a> {
        ei: usize,
        r0: usize,
        r1: usize,
        out: &'a mut [f32],
        mat: u8,
    }

    // 位相 1: gate と up の行を 1 本のタスク列にする (出力は act / up)。
    let mut jobs: Vec<Job> = Vec::new();
    {
        let mut ar: &mut [f32] = &mut act;
        let mut ur: &mut [f32] = &mut up;
        for &ei in sel {
            let mut r = 0;
            while r < ff {
                let n = rows.min(ff - r);
                let (a, arest) = ar.split_at_mut(n);
                ar = arest;
                let (u, urest) = ur.split_at_mut(n);
                ur = urest;
                jobs.push(Job {
                    ei,
                    r0: r,
                    r1: r + n,
                    out: a,
                    mat: 0,
                });
                jobs.push(Job {
                    ei,
                    r0: r,
                    r1: r + n,
                    out: u,
                    mat: 1,
                });
                r += n;
            }
        }
    }
    #[cfg(test)]
    FLAT_DECODE_TASKS.fetch_add(jobs.len(), std::sync::atomic::Ordering::Relaxed);
    kp::mark(t_gather1, kp::GATHER);
    jobs.into_par_iter().try_for_each(|j| -> Result<()> {
        kp::set_mat(if j.mat == 0 { kp::GATE } else { kp::UP });
        if j.mat == 0 {
            lw.gate_exps.gemv_rows_seq(j.ei, j.r0, j.r1, x, j.out)
        } else {
            lw.up_exps.gemv_rows_seq(j.ei, j.r0, j.r1, x, j.out)
        }
    })?;

    // silu(act) * up。expert ごとに従来と同じ順で適用する (要素ごとの演算なので
    // expert 間の順序は値に影響しない)。
    let t_silu = kp::stamp();
    for t in 0..sel.len() {
        let a = &mut act[t * ff..(t + 1) * ff];
        silu_inplace(a);
        let u = &up[t * ff..(t + 1) * ff];
        for r in 0..ff {
            a[r] *= u[r];
        }
    }
    kp::mark(t_silu, kp::SILU);

    /// 1 タスク = down の 1 行チャンク (入力は expert ごとの `act`)。
    struct DownJob<'a> {
        ei: usize,
        r0: usize,
        r1: usize,
        inp: &'a [f32],
        out: &'a mut [f32],
    }

    // 位相 2: down の行。出力は最終寄与 `hs` へ直接書く (expert ごとの Vec を作らない)。
    let t_gather2 = kp::stamp();
    let mut djobs: Vec<DownJob> = Vec::new();
    {
        let mut hr: &mut [f32] = hs;
        for (t, &ei) in sel.iter().enumerate() {
            let inp = &act[t * ff..(t + 1) * ff];
            let mut r = 0;
            while r < e {
                let n = rows.min(e - r);
                let (h, hrest) = hr.split_at_mut(n);
                hr = hrest;
                djobs.push(DownJob {
                    ei,
                    r0: r,
                    r1: r + n,
                    inp,
                    out: h,
                });
                r += n;
            }
        }
    }
    #[cfg(test)]
    FLAT_DECODE_TASKS.fetch_add(djobs.len(), std::sync::atomic::Ordering::Relaxed);
    kp::mark(t_gather2, kp::GATHER);
    djobs.into_par_iter().try_for_each(|j| -> Result<()> {
        kp::set_mat(kp::DOWN);
        lw.down_exps
            .gemv_rows_seq(j.ei, j.r0, j.r1, j.inp, j.out)
    })?;
    kp::mark(t_phase, kp::TOTAL);
    Ok(())
}

/// P70: 平坦化経路の 1 タスクあたりの行数。32 行未満はタスクのオーバーヘッドが
/// 勝ち、128 行を超えると端数 (10 expert × 行数) でスレッドが遊ぶ。
/// 実形 (e = 2560 / ff = 640) の実測は `bench_decode_flat` を参照。
pub(crate) const DECODE_FLAT_ROWS: usize = 64;

/// task 361: ミス経路の 1 タスクあたりの行数 (`DENNO_EXP_MISS_ROWS`、既定 16)。
///
/// `DENNO_EXP_MISS_FLAT` (task 153) の平坦化経路は 1 層のミス (0〜3 本) を
/// 行チャンクに割って全スレッドへ配る。固定 64 行だとミス 1 本の層で
/// gate/up が 20 タスク / 14 スレッド = 2 巡 (6 スレッド遊び)、down が 40 タスク
/// = 3 巡になり、末尾の巡が遊ぶ。16 行にすると gate/up 80 / down 160 タスクに
/// なって末尾の偏りが消える (行は独立なので値はビット一致)。
///
/// lab t361b (IQ3_S、depth8k、ミス CPU 時間 `[exp-iq-hit]` の ms/tok、2 ラウンド):
/// 定常窓 (レイヤ 6117 / ミス 5338) で 2.820 -> 2.810 (r1) / 2.864 -> 2.784 (r2)。
/// 窓ごとの比の中央値は r1 **-5.8%** / r2 **-2.4%** (ミスが 1 本以下の窓で
/// -2〜-12%、ミス 3 本の窓では 0%)。decode 中央値は 16.6 -> 16.7 (+0.2% = 分解能)、
/// prefill +0.9%、md5 は全 COND SAME、errors 0。
/// `=auto` (位相の全行数 / スレッド数) も測ったが、定常窓で **+4.0% (r1) /
/// +3.9% (r2)**、ミス 3 本/層の窓で +7.3 / +7.8% と悪化した (大きい塊ほど
/// 1 タスクの末尾待ちが効く) ので、その経路は撤去した。
/// `=64` で元に戻る (切り戻し)。
#[cfg_attr(not(feature = "par"), allow(dead_code))]
pub(crate) const MISS_FLAT_ROWS: usize = 16;

/// task 361: `DENNO_EXP_MISS_ROWS` の解釈 (純関数)。数字だけを受け付け、
/// 未設定は [`MISS_FLAT_ROWS`]、壊れた値も既定へ倒す。
#[cfg_attr(not(feature = "par"), allow(dead_code))]
pub(crate) fn miss_flat_rows_from_value(v: Option<&str>) -> usize {
    match v.map(|s| s.trim()) {
        None => MISS_FLAT_ROWS,
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => MISS_FLAT_ROWS,
        },
    }
}

/// task 361: [`miss_flat_rows_from_value`] をプロセスで 1 回だけ env から読む。
#[cfg_attr(not(feature = "par"), allow(dead_code))]
pub(crate) fn miss_flat_rows() -> usize {
    static ON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ON.get_or_init(|| miss_flat_rows_from_value(std::env::var("DENNO_EXP_MISS_ROWS").ok().as_deref()))
}

/// task 361: 起動行用。ミス経路の行幅を 1 行にする (ラボの COND の env が
/// サーバに届いたかを起動ログで確かめるため。env は `OnceLock` で 1 回だけ読む)。
pub fn exp_miss_rows_report() -> String {
    let v = miss_flat_rows();
    if v == MISS_FLAT_ROWS {
        format!("{v} (既定)")
    } else {
        format!("{v} (DENNO_EXP_MISS_ROWS)")
    }
}

/// P64: expert 1 本の寄与 `h = down(silu(gate(x)) * up(x))` を `h` に書く。
///
/// 従来の並列経路 ([`expert_contribs_par`]) の 1 expert ぶんと**同じ呼び出し列**:
/// `gemv_seq` → `silu_inplace` → `gemv_seq` → 積 → `gemv_seq`
/// (キャッシュ経路は `expert_gemv_cached_seq` × 3)。入口が同じなので、
/// 同じ `x` に対してビット一致する。
#[cfg(feature = "par")]
pub(crate) fn expert_contrib_one(
    lw: &LayerW,
    cache: Option<&ExpertCacheSet>,
    ei: usize,
    x: &[f32],
    s: &mut ExpertScratch,
    h: &mut [f32],
) -> Result<()> {
    let ExpertScratch { act, up, .. } = s;
    match cache {
        // 従来経路: mmap のスライスを直接 GEMV する (行並列なし)。
        None => {
            let kp = crate::model::exp_kprof::set_mat;
            kp(crate::model::exp_kprof::GATE);
            lw.gate_exps.gemv_seq(ei, x, act)?;
            silu_inplace(act);
            kp(crate::model::exp_kprof::UP);
            lw.up_exps.gemv_seq(ei, x, up)?;
            for r in 0..act.len() {
                act[r] *= up[r];
            }
            kp(crate::model::exp_kprof::DOWN);
            lw.down_exps.gemv_seq(ei, act, h)?;
        }
        // 常駐キャッシュ経路。バイト列の出所だけが違う。
        Some(c) => {
            expert_gemv_cached_seq(c, ExpertMatrix::Gate, &lw.gate_exps, lw.il, ei, x, act)?;
            silu_inplace(act);
            expert_gemv_cached_seq(c, ExpertMatrix::Up, &lw.up_exps, lw.il, ei, x, up)?;
            for r in 0..act.len() {
                act[r] *= up[r];
            }
            expert_gemv_cached_seq(c, ExpertMatrix::Down, &lw.down_exps, lw.il, ei, act, h)?;
        }
    }
    Ok(())
}

/// P64: 1 層ぶんの expert 寄与を**1 つの rayon 並列領域**で計算する新経路。
///
/// 返り値は `hs[t * e + j]` = `sel[t]` の寄与の `j` 番目 (expert ごとの `Vec` は
/// 作らない)。**足し込みは呼び出し側が index 昇順に逐次行う** (丸めの順序を
/// 変えないため。ここで足してはいけない)。
///
/// 形:
/// - 並列単位は expert (`sel` の各要素)。1 ジョブが gate / up / down の 3 行列を
///   順に回す (**行列ごとに並列領域を立てない**)。
/// - スクラッチ (`act` / `up`) は `map_init` の初期化でスレッドごとに 1 回だけ
///   確保し、そのスレッドが担当する以降の expert で使い回す。
/// - 寄与のバッファ `hs` は並列領域の外で 1 回だけ確保する。
///
/// 数値: 1 expert の計算は [`expert_contrib_one`] が [`ExpertW::gemv_seq`]
/// (= `expert_gemv_any_seq`) で回す。従来の並列経路 ([`expert_contribs_par`]) と
/// 同じカーネル・同じ行順で、行分割の有無は各要素の計算順序を変えないので
/// **旧経路とビット一致**する。
#[cfg(feature = "par")]
pub(crate) fn expert_contribs_loop(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
    cache: Option<&ExpertCacheSet>,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;
    #[cfg(test)]
    EXPERT_LOOP_REGIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut hs = vec![0.0f32; sel.len() * e];
    hs.par_chunks_mut(e)
        .zip(sel.par_iter())
        .map_init(
            || ExpertScratch::new(ff),
            |s, (h, &ei)| -> Result<()> { expert_contrib_one(lw, cache, ei, x, s, h) },
        )
        .collect::<Result<Vec<()>>>()?;
    Ok(hs)
}

/// キャッシュ経路の 1 expert GEMV (P-MOE-G1)。
///
/// `w` はオフセット・量子型・形状の出所で、バイト列は `c` から取る。
/// 使う範囲は [`ExpertW::slice_abs`] = mmap 経路の [`ExpertW::slice`] と同じ
/// [`ExpertSlice`] なので、2 経路が読むバイト列は同一になる。
pub(crate) fn expert_gemv_cached(
    c: &ExpertCacheSet,
    m: ExpertMatrix,
    w: &ExpertW,
    key: ExpertKey,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (off, len) = w.slice_abs(key.expert as usize)?;
    // task 260: `peek_only` (RAM tier) は**ヒットだけ**を引き、ミスは mmap へ落ちる
    // (pread で埋めない)。従来の `DENNO_EXPERT_CACHE_MB` はミスで埋める。
    let tier = if c.peek_only() {
        c.peek(m, key, len)
    } else {
        Some(
            c.get(m, w.file, key, off, len)
                .map_err(|err| one(format!("{}: expert {} (cache): {err}", w.name, key.expert)))?,
        )
    };
    let bytes: &[u8] = match &tier {
        Some(r) => r.as_slice(),
        None => w.slice(key.expert as usize)?,
    };
    // mmap 経路と同じゲートを通す (同じバイト列なら同じカーネル・同じ丸め順)。
    if moe_expert::expert_gemv_any_vnni(w.ty, bytes, w.n_in, w.n_out, x, out)
        .map_err(|err| one(format!("{}: expert {} (cache): {err}", w.name, key.expert)))?
    {
        return Ok(());
    }
    moe_expert::expert_gemv_any(w.ty, bytes, w.n_in, w.n_out, x, out)
        .map_err(|err| one(format!("{}: expert {} (cache): {err}", w.name, key.expert)))
}

/// [`expert_gemv_cached`] の**行並列なし**版 (expert 並列の内側で使う)。
///
/// mmap 経路の [`ExpertW::gemv_seq`] と同じ `expert_gemv_any_seq` を通るので、
/// 同じバイト列・同じ `x` に対してビット一致する。
pub(crate) fn expert_gemv_cached_seq(
    c: &ExpertCacheSet,
    m: ExpertMatrix,
    w: &ExpertW,
    layer: usize,
    ei: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let key = ExpertKey {
        layer: layer as u16,
        expert: ei as u16,
    };
    let (off, len) = w.slice_abs(ei)?;
    let tier = if c.peek_only() {
        c.peek(m, key, len)
    } else {
        Some(
            c.get(m, w.file, key, off, len)
                .map_err(|err| one(format!("{}: expert {ei} (cache): {err}", w.name)))?,
        )
    };
    let bytes: &[u8] = match &tier {
        Some(r) => r.as_slice(),
        None => w.slice(ei)?,
    };
    if moe_expert::expert_gemv_any_vnni_seq(w.ty, bytes, w.n_in, w.n_out, x, out)
        .map_err(|err| one(format!("{}: expert {ei} (cache): {err}", w.name)))?
    {
        return Ok(());
    }
    moe_expert::expert_gemv_any_seq(w.ty, bytes, w.n_in, w.n_out, x, out)
        .map_err(|err| one(format!("{}: expert {ei} (cache): {err}", w.name)))
}
impl Qwen4Exp {
    /// P54: expert の置き場所の統計 `(RAM 常駐 expert 数, SSD 読みの累計)`。
    ///
    /// `DENNO_EXPERT_CACHE_MB` が 0 (既定) のときは `(0, 0)`。VRAM の使用量は
    /// [`expert_vram::budget_usage`] から取る (こちらはキャッシュを持たない)。
    pub fn expert_tier_stats(&self) -> (u32, u32) {
        match &self.expert_cache {
            Some(c) => (
                c.used_experts().min(u32::MAX as usize) as u32,
                c.total_reads().min(u32::MAX as u64) as u32,
            ),
            None => (0, 0),
        }
    }
}
