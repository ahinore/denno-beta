//! GDN decode と prefill
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;

pub(crate) const PTX_GDN: &[u8] = include_bytes!("../../kernels/kernels_gdn.ptx");


/// task 137: `dno_gdn_scan_sig` の ABI (引数の個数) が `dno_gdn_scan` と
/// 同じであること。両者はゲート式 1 行以外が同一のカーネルなので、引数の
/// 個数も同じでなければならない (現状 23)。
///
/// 個数が食い違う PTX を cuLaunchKernel で起動すると、足りない引数は**不定**の
/// まま読まれ、エラーも NaN も出ずに別の行を書く。ホストの起動コードと PTX が
/// ずれたとき (PTX 再生成の取り違え) に黙って壊れないよう、モジュールロード時に
/// ここで弾いて CPU 経路へ落とす。GPU 無しでテストできる
/// (`gdn_scan_sig_abi_matches_scan_in_committed_ptx`)。
pub(crate) fn gdn_scan_sig_abi_ok(ptx: &str) -> bool {
    match (
        ptx_entry_params(ptx, "dno_gdn_scan"),
        ptx_entry_params(ptx, "dno_gdn_scan_sig"),
    ) {
        (Some(base), Some(sig)) => sig == base,
        _ => false,
    }
}


/// task 177: GDN decode submission の GEMV だけカーネルを差し替える
/// (`DENNO_GDN_DEC_GEMV=v2|v3|fp16`。未設定・空・`0`/`off` は従来 =
/// [`gemv_kernel_mode`])。
///
/// 動機 (lab t177a の `[dec-kprof]` 実測、pos=3827 の ssm 層 1 本):
/// `dno_gemv_q8_0_v3` (grid=1280, block=32) が 42.5 us / 27.9 MB = 656 GB/s、
/// `dno_gemv_q4k_v3` (grid=768, block=32) が 31.8 us / 8.9 MB = 278 GB/s、
/// `dno_gemv_q8_0_v3` (grid=320, block=32) が 50.9 us / 16.7 MB = 328 GB/s。
/// v3 は 1 行を `lanes` 本で分けるので **スレッド数が n_out x lanes で固定**
/// (out 投影は 2,560 x 4 = 10,240 スレッド = 5090 の 170 SM に 1.9 warp/SM)。
/// v2 (`kernels_moe.cu`、4 行/ブロック・256 スレッド・行を 64 本で shared へ
/// 積んでから 1 本が畳む) は **n_out x 64 スレッド**で、同じ形でも一桁多い。
/// どちらが速いかは形に依るので、既定は従来のまま env で切り替える。
/// `DENNO_GDN_DEC_GEMV` の解釈 (純関数。真理値表はテストで固定)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GdnDecGemv {
    /// 未設定・空・`0`/`off` = 従来 (`gemv_kernel_mode()`)。
    Default,
    V2,
    V3,
    Fp16,
    /// v4 (レーン数可変。`kernels_moe_v4.cu`)。
    V4,
    /// v5 (ベクタロード + 複数アキュムレータ + 先読み。`kernels_moe_v5.cu`)。
    V5,
    /// task 422: dn6 (`kernels_mix_dn6.cu` = 1 ブロック rows 行 / 1 レーン 1 対)。
    /// Q8_0 だけ。形が合わない単位は v5 へ落ちる (CPU へは落とさない)。
    Dn6,
    /// task 496: dn8 (`kernels_mix_dn8.cu` = 1 ブロック rows * np 個の (行, 対))。
    /// np が 32 の倍数でない形 (= dn6 でレーンが遊ぶ形) だけ dn8 へ、
    /// それ以外の単位は dn6 のまま。
    Dn8,
}


/// task 184: v5 を既定にするか。
///
/// lab t184a (depth8k) で採否 yes: 128 トークンの生 decode 秒が
/// 3.39/2.93 -> 3.25/2.82 (短)、3.05 -> 2.93 (3,965)、3.07 -> 2.96 (14,157)
/// = -3.6..-4.1%。kprof の 1 発あたりは gate Q4_K 36.3 -> 22.1 us、
/// Q8_0 (qkv の一部 + out) 40.8 -> 30.2 us。**Q5_K の qkv は v5 が無い**ので
/// その層は v4 のまま (53.6 us で不変)。戻すのは `DENNO_GDN_DEC_GEMV=v4`。
const GDN_DEC_GEMV5_DEFAULT: bool = true;


/// task 496: GDN の qkv/out の既定を dn8 (レーン詰め + x を shared) にする。
///
/// lab t496c (c042a09f6、depth8k、既定モデル、2 ラウンド) の同じ run の対で、
/// base ssm 3.07/3.08 -> `dn8ar8` 2.94/2.95 (-4.2%、t496a/t496b の
/// rows=8 = -3.4% と合わせて 3 run 連続)、短い decode 10.8 -> 10.6 (-1.4〜-1.9%)、
/// md5 は全 6 文脈 SAME、errors 0。14k 後と prefill は不変 (±1% 以内)。
/// 切り戻しは `DENNO_GDN_DEC_GEMV=dn6` (dn8 をやめる) / `=dn8` (out は dn6)。
const GDN_DEC_GEMV_DN8_DEFAULT: bool = true;


/// task 422: GDN の qkv/out の既定を dn6 (1 ブロック 2 行 / 1 レーン 1 対) にする。
///
/// lab t422c (8aed5bd7b、depth8k、既定モデル) の同じ run の対で、`DENNO_MIX_QK`
/// の既定 (dn6) に重ねて **ssm 段 3.40 (base / base2 とも) -> 2.75 (-19%)**、
/// 短い decode 12.5 -> 12.0 (-4.0%)、14k 後 14.2 -> 13.8、md5 は全 SAME。
/// ただしこの値は **1 ラウンド (t422c) だけ**の実測で、rows=6 は t422b で悪化
/// (+2.1%、ssm 3.39 -> 3.65) だった。カーネル自体 (mix の down) は t422b/t422c の
/// 2 ラウンドで採っている。戻すのは `DENNO_GDN_DEC_GEMV=v5`。
const GDN_DEC_GEMV_DN6_DEFAULT: bool = true;


/// `DENNO_GDN_DEC_GEMV` の解釈 (未設定・空・`0`/`off` は既定 = v4)。
pub(crate) fn parse_gdn_dec_gemv(raw: Option<&str>) -> GdnDecGemv {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("v2") => GdnDecGemv::V2,
        Some("v3") => GdnDecGemv::V3,
        Some("fp16") => GdnDecGemv::Fp16,
        Some("v4") => GdnDecGemv::V4,
        Some("v5") => GdnDecGemv::V5,
        // task 422: dn6 (1 ブロック rows 行 / 1 レーン 1 対)。rows は
        // `gdn_dn6_rows_from_value` が決める (既定 2。t422c で rows=2 が最良)。
        Some("dn6") | Some("d6") | Some("dn6r1") | Some("dn6r2") | Some("dn6r4")
        | Some("dn6r6") | Some("d6r1") | Some("d6r2") | Some("d6r4") | Some("d6r6") => {
            GdnDecGemv::Dn6
        }
        // task 496: dn8 (レーン詰め版)。rows は `gdn_dn8_rows_from_value`。
        Some("dn8") | Some("d8") | Some("dn8r2") | Some("dn8r4") | Some("dn8r6")
        | Some("dn8r8") | Some("d8r2") | Some("d8r4") | Some("d8r6") | Some("d8r8")
        | Some("dn8r12") | Some("dn8r16") | Some("d8r12") | Some("d8r16") | Some("dn8a")
        | Some("d8a") | Some("dn8ar2") | Some("dn8ar4") | Some("dn8ar6") | Some("dn8ar8")
        | Some("dn8ar12") | Some("dn8ar16") | Some("d8ar2") | Some("d8ar4")
        | Some("d8ar6") | Some("d8ar8") | Some("d8ar12") | Some("d8ar16") => {
            GdnDecGemv::Dn8
        }
        // task 504: `dn8*` に加えて out 投影だけ dn7 (1 対 = 4 レーン) で撃つ値。
        // `o7` で終わる値は qkv = dn8 / out = dn7 (`gdn_out_dn7_from_value`)。
        Some("dn8o7") | Some("d8o7") | Some("dn8ao7") | Some("d8ao7") => GdnDecGemv::Dn8,
        // 明示 off は「v5 以前の既定」= v4 (task 179 の既定)。
        Some("0") | Some("off") | Some("false") | Some("no") => GdnDecGemv::V4,
        _ => {
            // task 496: 既定は dn8 (lab t496c: ssm 3.07 -> 2.94 (-4.2%)、短い
            // decode 10.8 -> 10.6、md5 SAME、errors 0。np が 32 の倍数の形や
            // 形が合わない単位は dn6 / v5 が受け皿)。切り戻しは
            // `DENNO_GDN_DEC_GEMV=dn6` (dn8 をやめる) / `=dn8` (out は dn6)。
            if GDN_DEC_GEMV_DN8_DEFAULT {
                GdnDecGemv::Dn8
            } else if GDN_DEC_GEMV_DN6_DEFAULT {
                GdnDecGemv::Dn6
            } else if GDN_DEC_GEMV5_DEFAULT {
                GdnDecGemv::V5
            } else {
                GdnDecGemv::Default
            }
        }
    }
}


/// [`parse_gdn_dec_gemv`] を env から 1 回だけ読む。
fn gdn_dec_gemv_choice() -> GdnDecGemv {
    static C: OnceLock<GdnDecGemv> = OnceLock::new();
    *C.get_or_init(|| parse_gdn_dec_gemv(std::env::var("DENNO_GDN_DEC_GEMV").ok().as_deref()))
}


/// GDN decode の submission が使う GEMV モード (ゲートが無ければ従来)。
pub(crate) fn gdn_dec_gemv_mode_eff() -> GemvKernelMode {
    match gdn_dec_gemv_choice() {
        GdnDecGemv::V2 => GemvKernelMode::V2,
        GdnDecGemv::V3 => GemvKernelMode::V3,
        GdnDecGemv::Fp16 => GemvKernelMode::Fp16,
        _ => gemv_kernel_mode(),
    }
}


/// v4 を選んでいるか。
///
/// task 179: env 未設定の既定を v4 にした (lab t177c: decode -1.8%・md5 SAME。
/// 形が合わない単位は v3 へ落ちる)。従来へ戻すのは `DENNO_GDN_DEC_GEMV=v3`
/// (または `v2` / `fp16` / `default`)。
pub(crate) fn gdn_dec_gemv_v4_on() -> bool {
    match gdn_dec_gemv_choice() {
        // task 184: v5 の形が合わない単位は v4 へ落とす (CPU へは落とさない)。
        GdnDecGemv::V4 | GdnDecGemv::V5 | GdnDecGemv::Dn6 | GdnDecGemv::Dn8
        | GdnDecGemv::Default => true,
        GdnDecGemv::V2 | GdnDecGemv::V3 | GdnDecGemv::Fp16 => false,
    }
}


/// task 184: v5 を選んでいるか。
///
/// task 422: dn6 のときも true (dn6 の形に合わない単位・型は v5 の計画で撃つ。
/// 撃ち分けは `gdn_dn6_plan` を見る起動側で行う)。
pub(crate) fn gdn_dec_gemv_v5_on() -> bool {
    matches!(
        gdn_dec_gemv_choice(),
        GdnDecGemv::V5 | GdnDecGemv::Dn6 | GdnDecGemv::Dn8
    )
}


/// task 422: dn6 を選んでいるか (`DENNO_GDN_DEC_GEMV=dn6`)。
///
/// task 496: `dn8` のときも true (dn8 の形に合わない単位 = np が 32 の倍数の形は
/// dn6 で撃つ)。
pub(crate) fn gdn_dec_gemv_dn6_on() -> bool {
    dn6_alive_for(gdn_dec_gemv_choice())
}


/// dn8 を選んでいるか (`DENNO_GDN_DEC_GEMV=dn8`)。
pub(crate) fn gdn_dec_gemv_dn8_on() -> bool {
    dn8_on_for(gdn_dec_gemv_choice())
}


/// task 496: dn8 の rows を env から決める (純関数)。
///
/// `dn8` = 4 行 (np = 40 の形で items = 160 = 5 warp、埋まり 100%)、
/// `dn8r2` / `dn8r6` / `dn8r8` で行数を変える (A/B 用)。
pub fn gdn_dn8_rows_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("dn8r2") | Some("d8r2") | Some("dn8ar2") | Some("d8ar2") => 2,
        Some("dn8r6") | Some("d8r6") | Some("dn8ar6") | Some("d8ar6") => 6,
        Some("dn8r8") | Some("d8r8") | Some("dn8ar8") | Some("d8ar8") => 8,
        Some("dn8r12") | Some("d8r12") | Some("dn8ar12") | Some("d8ar12") => 12,
        Some("dn8r16") | Some("d8r16") | Some("dn8ar16") | Some("d8ar16") => 16,
        // lab t496c (既定モデル、depth8k) で rows=8 が最良 (ssm 3.07 -> 2.97)。
        _ => 8,
    }
}


/// `dn8a*` (= np が 32 の倍数の形にも dn8 を使う) か (純関数)。
pub fn gdn_dn8_all_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("dn8a") | Some("d8a") => true,
        // task 504: out だけ dn7 で撃つ値でも、dn7 が撃てない形では dn8 を
        // 受け皿にする (dn6 へ落とさない)。
        Some("dn8o7") | Some("d8o7") => true,
        Some(s) => s.starts_with("dn8a") || s.starts_with("d8a"),
        // lab t496c の `d8ar8` (out も dn8、rows=4) が ssm 2.94 = 既定より -4.2%。
        // 未設定 (既定) は out も dn8。`DENNO_GDN_DEC_GEMV=dn8` で out は dn6 へ。
        None => true,
    }
}


/// GDN の dn8 の rows (env から 1 回だけ読む)。
fn gdn_dn8_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        gdn_dn8_rows_from_value(std::env::var("DENNO_GDN_DEC_GEMV").ok().as_deref())
    })
}


/// np が 32 の倍数の形にも dn8 を使うか (env から 1 回だけ読む)。
fn gdn_dn8_all() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| {
        gdn_dn8_all_from_value(std::env::var("DENNO_GDN_DEC_GEMV").ok().as_deref())
    })
}


/// task 496: GDN の qkv/out を dn8 で撃つときの計画。
///
/// Q8_0 で、ゲートが dn8 のときだけ。np が 32 の倍数の形 (out = np 96 など) は
/// `mix_dn8_plan` が `None` を返すので dn6 のまま。
fn gdn_dn8_plan(
    ty: crate::kquant::GgmlType,
    k: usize,
    n: usize,
    want_rows: usize,
) -> Option<MixDn8Plan> {
    if ty != crate::kquant::GgmlType::Q8_0
        || !gdn_dec_gemv_dn8_on()
        || mix_dn8_kernels().is_none()
    {
        return None;
    }
    mix_dn8_plan_with(k, n, want_rows, gdn_dn8_all())
}


/// out 投影を dn8 で撃つときの rows (4 固定)。
///
/// out は np = 96 で dn6 でもレーンは 100% なので、狙いは x (y ベクトル
/// 6,144 本 = 24 KB) を 1 ブロック 1 回にすること。rows=4 で
/// 1,280 -> 640 ブロック = x のグローバル読みが 31 MB -> 15.5 MB。block は
/// 384 (レジスタ上限 640 の内側)。
const GDN_DN8_OUT_ROWS: usize = 4;


/// task 422: GDN の qkv/out 用の rows (純関数。既定 6 = block 960)。
pub fn gdn_dn6_rows_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("dn6r1") | Some("d6r1") => 1,
        Some("dn6r4") | Some("d6r4") => 4,
        Some("dn6r6") | Some("d6r6") => 6,
        // task 422: 既定 (未設定 / dn6 / それ以外) は 2 行 (t422c で rows=2 が最良)。
        _ => 2,
    }
}


/// GDN の dn6 の rows (env から 1 回だけ読む。層 x トークンごとに引かない)。
fn gdn_dn6_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        gdn_dn6_rows_from_value(std::env::var("DENNO_GDN_DEC_GEMV").ok().as_deref())
    })
}


/// task 422: GDN の qkv/out を dn6 で撃つときの `(warps, rows)`。
///
/// Q8_0 だけで、ゲートが dn6 のときだけ。形が合わなければ `None`
/// (呼び出し側は v5 の `gdn_launch_gemv_dev` へ)。
fn gdn_dn6_plan(ty: crate::kquant::GgmlType, k: usize, n: usize) -> Option<(usize, usize)> {
    if ty != crate::kquant::GgmlType::Q8_0
        || !gdn_dec_gemv_dn6_on()
        || mix_dn6_kernels().is_none()
    {
        return None;
    }
    mix_dn6_plan(k, n, gdn_dn6_rows())
}


/// task 192: `DENNO_GDN_DEC_GEMV_Q5K` の解釈 (純関数。真理値表はテストで固定)。
///
/// v5 を選んでいるときに Q5_K だけ v5 の新カーネル (`kernels_moe_v5_q5k.cu`) を
/// 使うか。既定 on。**明示的な偽値だけが off** (未設定・空は既定)。
pub(crate) fn parse_gdn_dec_gemv_q5k(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 192: Q5_K の v5 カーネルを使うか (env から 1 回だけ読む)。
///
/// A/B 用の細かいゲート。`DENNO_GDN_DEC_GEMV=v4` にすると Q8_0 / Q4_K も
/// まとめて v4 へ戻るので、Q5_K だけを切り替えるために分けてある。
pub(crate) fn gdn_dec_gemv_q5k_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_gdn_dec_gemv_q5k(std::env::var("DENNO_GDN_DEC_GEMV_Q5K").ok().as_deref())
    })
}


/// `DENNO_GDN_DEC_GEMV_LANES` の解釈 (純関数)。既定 32、4..32 に丸める。
pub(crate) fn parse_gdn_dec_v4_lanes(raw: Option<&str>) -> usize {
    match raw.map(|s| s.trim().parse::<usize>()) {
        Some(Ok(v)) => v.clamp(4, 32),
        _ => 32,
    }
}


/// v4 のレーン上限 (env から 1 回だけ読む)。
pub(crate) fn gdn_dec_v4_lanes_cap() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| {
        parse_gdn_dec_v4_lanes(std::env::var("DENNO_GDN_DEC_GEMV_LANES").ok().as_deref())
    })
}


/// s67: GDN スキャン (chunked) の作業領域。**層をまたいで再利用する** (grow-only)。
///
/// 動機 (s64/s66 の計測): prefill の `gdn_scan` 3.15 s のうち **alloc 1.04 s** は
/// 層ごとに 16 本 (計 ≈180MB) を `DevBuffer::from_slice` で作り直していた分で、
/// 形はモデルで固定 (n はリクエスト内で一定) なのに毎回 cuMemAlloc/cuMemFree
/// していた。確保を初回だけにし、以降は `copy_from` で上書きする。
///
/// 数値は不変 — バッファの寿命が延びるだけで、書き込む内容・カーネル引数・
/// 演算順は 1 つも変えない (scheduler ワーカーは単一スレッドなので thread_local
/// で競合しない・`SCRATCH` と同じ規約)。
pub(crate) struct GdnScratch {
    q: Option<DevBuffer>,        // [n, conv_dim]
    z: Option<DevBuffer>,        // [n, vd]
    beta: Option<DevBuffer>,     // [n, nv]
    alpha: Option<DevBuffer>,    // [n, nv]
    conv_in: Option<DevBuffer>,  // [conv_dim, cs]
    ssm_in: Option<DevBuffer>,   // [nv, dk, dv]
    wc: Option<DevBuffer>,       // 重み 4 本 (層ごとに差し替え)
    wa: Option<DevBuffer>,
    wd: Option<DevBuffer>,
    wn: Option<DevBuffer>,
    w: Option<DevBuffer>,     // [n_chunks, nv, dk, dv] chunk ローカル遷移
    m: Option<DevBuffer>,     // [n_chunks, nv, dk, dk]
    sinit: Option<DevBuffer>, // [n_chunks, nv, dk, dv]
    y: Option<DevBuffer>,     // [n, vd]
    conv_out: Option<DevBuffer>,
    ssm_out: Option<DevBuffer>,
}


/// カーネル引数用の生ポインタ束 (バッファ本体は thread_local が保持し続ける)。
#[derive(Clone, Copy)]
pub(crate) struct GdnPtrs {
    q: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wc: CUdeviceptr,
    wa: CUdeviceptr,
    wd: CUdeviceptr,
    wn: CUdeviceptr,
    w: CUdeviceptr,
    m: CUdeviceptr,
    sinit: CUdeviceptr,
    y: CUdeviceptr,
    conv_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
}


#[derive(Clone, Copy)]
pub(crate) enum GdnBuf {
    Y,
    ConvOut,
    SsmOut,
}


// ---------------------------------------------------------------------------
// task 393: GDN の重み 4 本 (conv / a / dt / norm) を層ごとに上げ直さない
// (`DENNO_PF_GDN_WSTAT`)
// ---------------------------------------------------------------------------
//
// 層優先 prefill は **層ごとに全チャンク** を回す (task 250) のに、
// `gdn_pf_chain` は呼ばれるたびに重み 4 本を `DevBuffer::copy_from` で
// 上げ直していた。`copy_from` は pageable の **同期** H2D で、
// **呼び出し時点でコンテキスト全体の完了を待つ** (task 369 が D2H で見つけた
// のと同じ性質)。つまりチャンクごとに GPU のキューを吐き切っていた。
//
// 実測 (lab t393a の b2、14k): `[q4ledger-pf]` の ssm は
// wall 2,460.3 / busy 1,558.2 / exposed 902.1 ms で、CPU は 28 ms しかない。
// 36 層 x 7 チャンク x 4 本 = 1,008 回の H2D が 0.9 ms ずつ待っている計算になる
// (3,826 は 36 層 x 2 チャンク x 4 = 288 回で exposed 26.7 ms = 0.09 ms/回。
// 待ちの長さはキューの深さに比例する)。
//
// 同じ層の重みは 4 本とも**同じバイト列**なので、置き場が同じままなら
// 上げ直さなくてよい。値は 1 bit も変わらない。

/// task 393: `DENNO_PF_GDN_WSTAT` の env 名。
pub const GDN_WSTAT_ENV: &str = "DENNO_PF_GDN_WSTAT";


/// 未設定のときの既定 (**off のまま**)。
///
/// 実測 (lab t393b、depth8k): ゲートは発火した (ssm の H2D 159.18 MB / 1,080 回
/// -> 123.60 MB / 216 回 = 36 層 x 6 チャンク x 4 本を飛ばした) が、ssm 段の壁は
/// 14k 2,459.8 -> 2,433.8 ms (-1.1%)、3,826 454.9 -> 452.3 ms (-0.6%) だけで、
/// 壁は -0.1% / -0.0%。**同期 H2D の待ちは ssm の露出の正体ではなかった**
/// (露出 900 ms は GPU の待ち)。採否規則 (段で 2% 以上) に届かないので off。
pub const DEFAULT_GDN_WSTAT: bool = false;


/// [`GDN_WSTAT_ENV`] の解釈 (未設定・空は [`DEFAULT_GDN_WSTAT`])。
pub(crate) fn gdn_wstat_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => DEFAULT_GDN_WSTAT,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}


/// 重みの上げ直しを飛ばすか (プロセスで 1 回だけ env を読む)。
pub fn gdn_wstat_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_wstat_from_value(std::env::var(GDN_WSTAT_ENV).ok().as_deref()))
}


thread_local! {
    /// 直前に 4 本を上げた組み合わせ: (層, 転送元 4 本の (ptr, 要素数), 置き場 4 本の ptr)。
    /// 一致したら上げ直さない。3 つの呼び出し口 (層優先 / チャンク優先 / decode) が
    /// 同じ置き場を共有するので、鍵も 1 本で持つ。
    pub(crate) static GDN_WSTAT_KEY: std::cell::Cell<Option<[usize; 13]>> =
        const { std::cell::Cell::new(None) };
}


/// task 393: 上げ直しを飛ばしてよいかの鍵 (純関数)。
///
/// 「同じ層の、同じバイト列 (転送元のポインタと要素数) を、同じ置き場へ」だけを
/// 一致と見る。置き場が `grow` で差し替わればポインタが変わるので必ず上げ直す。
pub(crate) fn gdn_wstat_key(li: usize, src: [(usize, usize); 4], dst: [usize; 4]) -> [usize; 13] {
    let mut key = [0usize; 13];
    key[0] = li;
    for i in 0..4 {
        key[1 + i * 2] = src[i].0;
        key[2 + i * 2] = src[i].1;
        key[9 + i] = dst[i];
    }
    key
}


/// task 393: 重み 4 本 (`wc` / `wa` / `wd` / `wn` の順) をスクラッチへ上げる。
///
/// [`gdn_wstat_on`] が on で、直前に同じ層の同じバイト列を同じ置き場へ
/// 上げていれば 4 本とも飛ばす (飛ばした回数は台帳に出さない — 効果は
/// `[q4ledger-pf]` の ssm exposed で読む)。
fn gdn_put_weights(s: &GdnScratch, li: usize, src: [&[f32]; 4]) -> Result<()> {
    let dst = [
        s.wc.as_ref().map(|b| b.ptr() as usize).unwrap_or(0),
        s.wa.as_ref().map(|b| b.ptr() as usize).unwrap_or(0),
        s.wd.as_ref().map(|b| b.ptr() as usize).unwrap_or(0),
        s.wn.as_ref().map(|b| b.ptr() as usize).unwrap_or(0),
    ];
    let key = gdn_wstat_key(
        li,
        [
            (src[0].as_ptr() as usize, src[0].len()),
            (src[1].as_ptr() as usize, src[1].len()),
            (src[2].as_ptr() as usize, src[2].len()),
            (src[3].as_ptr() as usize, src[3].len()),
        ],
        dst,
    );
    if gdn_wstat_on() && GDN_WSTAT_KEY.with(|c| c.get()) == Some(key) {
        return Ok(());
    }
    // 置き場が無い (未確保) のは呼び出し側のバグ。従来どおり panic させる。
    s.wc.as_ref().expect("GdnScratch: 未確保").copy_from(src[0])?;
    s.wa.as_ref().expect("GdnScratch: 未確保").copy_from(src[1])?;
    s.wd.as_ref().expect("GdnScratch: 未確保").copy_from(src[2])?;
    s.wn.as_ref().expect("GdnScratch: 未確保").copy_from(src[3])?;
    GDN_WSTAT_KEY.with(|c| c.set(Some(key)));
    Ok(())
}


// ---------------------------------------------------------------------------
// s89 (P1-1): GDN スキャンのデバイス常駐状態
//
// 動機: s88 の計測で prefill の `d2h_state` が大きいのは GDN 状態の往復
// (`ssm_out` の D2H + 次層の `ssm_in` の H2D・1 層 2MB ずつ) を含むため。
// ここでは「層のスキャン状態の正本」をデバイスに置き、ホスト側の
// `KvCache::gdn` は**実行境界 (forward の切れ目) でだけ**一致させる。
//
// s90 の状態 (重要・偽らない):
//   この構造体は **まだどの経路からも使われていない**。tl の `GDN_STATE` に
//   確保・seed・無効化の関数は用意してあるが、`gdn_scan_chunked_impl` は
//   (i) 2 launch 経路が現状の forward では成立しない (→ gdn/scan_schedule.rs)
//   (ii) y をデバイスに残す oproj 経路 (s72) は 2 層目以降の conv 入力を
//        ホストから捨てる形になっており、ホスト状態が層ごとにしか確定しない
//   の 2 点により、従来どおり層ごとに ssm を H2D / D2H している。
//   `f_gdn_chunk_scan_hoist` のカーネルと Optional 取得は「古い PTX でも起動を
//   壊さない」性質ごと残してある (Next: 上記 (i)(ii) を解いて配線する)。
//
// 無効化の契約 (将来 配線したときに効くもの — 先に用意しておく):
//   ホストの `KvCache::gdn` を書き換える経路 (セッション reset / エントリ復元 /
//   デコードチェーンの遅延フラッシュ / CPU フォールバック) は
//   [`gdn_scan_state_invalidate`] を呼ぶ。呼ばれたら次の実行でホストから
//   再 seed する (数値不変・余分な H2D が 1 回増えるだけ)。**古い正本を読むと
//   出力が変わる**ので、新しい書き換え経路を足すときは必ずここを通すこと。
// ---------------------------------------------------------------------------
pub(crate) struct GdnStateCache {
    pub(crate) conv: Vec<Option<DevBuffer>>, // [layer] conv [conv_dim, cs] (デバイス正本)
    conv_len: usize,              // conv の要素数
    pub(crate) cur_kv: Option<u64>,          // この正本が属する KV キャッシュ ID (None = 未使用)
    pub(crate) ssm: Vec<Option<DevBuffer>>, // [layer] ssm [nv, dk, dv] (デバイス正本)
    pub(crate) valid: Vec<bool>,            // [layer] = ホスト KvCache::gdn と一致している
    /// task 1577: 連結プール (`DENNO_CUDA_GDN_POOL`, 既定 off)。Some のとき
    /// `conv` / `ssm` の各層はこのプールの**非所有ビュー**になる (drop で解放しない)。
    pool: Option<GdnPool>,
    srows: Option<DevBuffer>,    // [n, nv, dk, dv] hoist 経路の境界行
    srows_n: usize,              // srows がいま表現している n (0 = 未初期化)
    rows: usize,                 // 行数 (= nk*dk = vd)
    cols: usize,                 // 列数 (= dk*dv)
}


/// task 1577: 全層の GDN 状態を 1 本に詰めた連続領域。
///
/// P1372 の L2 persisting 窓はストリームに 1 本しか張れないため、層ごとに
/// 別々に確保された状態では 1 層しか固定できない。ここを 1 本にすれば
/// プール全体 (36-48 層で約 380-510 MiB) を 1 本の窓で persisting にできる
/// (窓の上限は `MAX_ACCESS_POLICY_WINDOW_SIZE` で clamp される)。
struct GdnPool {
    buf: DevBuffer,
    /// 収容できる層数 (倍々で伸ばす)。
    cap_layers: usize,
    conv_len: usize,
    rows: usize,
    cols: usize,
}


// ---------------------------------------------------------------------------
// task 1577: 連結 GDN 状態プールの純関数 (`DENNO_CUDA_GDN_POOL`, 既定 off)
//
// レイアウト (層 li の先頭 = pool + li * stride):
//   [conv (conv_len*4 を 256 B に切り上げ)] [ssm (rows*cols*4 を 256 B に切り上げ)]
// 層数は最初の decode で 1 層ずつ増えるので、容量は倍々で伸ばし、伸ばすときは
// 旧領域の中身を同期 D2D で先頭へ写す (値と `valid` の意味は不変)。
// ---------------------------------------------------------------------------

/// プール内のスロット境界。cuMemcpyDtoD の宛先整列と conv/ssm の取り違え防止。
pub(crate) const GDN_POOL_ALIGN: usize = 256;

/// `bytes` を [`GDN_POOL_ALIGN`] の倍数に切り上げる (純関数)。
pub(crate) fn gdn_pool_align_up(bytes: usize) -> usize {
    bytes.saturating_add(GDN_POOL_ALIGN - 1) & !(GDN_POOL_ALIGN - 1)
}

/// 層 1 本のレイアウト `(conv スロット, ssm スロット, 層ストライド)`。
/// どちらかが 0 要素になる形は None (プールに載せず従来経路へ落とす)。
pub(crate) fn gdn_pool_layout(
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Option<(usize, usize, usize)> {
    const F: usize = std::mem::size_of::<f32>();
    let conv = conv_len.checked_mul(F).filter(|v| *v > 0)?;
    let ssm = rows.checked_mul(cols)?.checked_mul(F).filter(|v| *v > 0)?;
    let conv = gdn_pool_align_up(conv);
    let ssm = gdn_pool_align_up(ssm);
    Some((conv, ssm, conv.checked_add(ssm)?))
}

/// プールの総バイト数 (純関数)。
pub(crate) fn gdn_pool_bytes(
    layers: usize,
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Option<usize> {
    let (_, _, stride) = gdn_pool_layout(conv_len, rows, cols)?;
    layers.checked_mul(stride).filter(|v| *v > 0)
}

/// 必要な層数を満たす容量 (純関数)。最小 8 層から倍々 (8, 16, 32, ...)。
pub(crate) fn gdn_pool_grow_cap(cap: usize, need: usize) -> usize {
    let mut c = cap.max(8);
    while c < need {
        c = c.saturating_mul(2);
    }
    c
}

/// `DENNO_CUDA_GDN_POOL` の真偽表 (純関数)。1/on/true/yes だけが有効。
pub(crate) fn parse_gdn_pool(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// プールを使うか (env は 1 回だけ読む。未設定・空・その他の値は off)。
fn gdn_pool_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gdn_pool(std::env::var("DENNO_CUDA_GDN_POOL").ok().as_deref()))
}


impl GdnStateCache {
    pub(crate) const fn empty() -> Self {
        Self {
            conv: Vec::new(),
            conv_len: 0,
            cur_kv: None,
            ssm: Vec::new(),
            valid: Vec::new(),
            pool: None,
            srows: None,
            srows_n: 0,
            rows: 0,
            cols: 0,
        }
    }
    fn ensure_layers(&mut self, n_layers: usize) {
        // `Option<DevBuffer>` は Clone でない (DevBuffer は RAII の所有権型) →
        // resize_with で伸ばす。縮む場合は末尾を落とす (drop = cuMemFree)。
        if self.ssm.len() < n_layers {
            self.ssm.resize_with(n_layers, || None);
            self.conv.resize_with(n_layers, || None);
        } else {
            self.ssm.truncate(n_layers);
            self.conv.truncate(n_layers);
        }
        self.valid.resize(n_layers, false);
    }
    /// 全層を「ホストが正」に戻す (次の実行で再 seed)。ホストの GDN 状態を
    /// 書き換える経路 (セッション reset / エントリ復元 / デコードチェーンの
    /// 遅延フラッシュ / CPU フォールバック) は必ずこれを呼ぶ。
    pub(crate) fn invalidate(&mut self) {
        for v in self.valid.iter_mut() {
            *v = false;
        }
        self.srows_n = 0;
    }
    /// 層ベクタを `li` まで伸ばす (層数は初回利用時に確定する)。
    fn touch_layer(&mut self, li: usize) {
        if self.ssm.len() <= li {
            let n = li + 1;
            self.ssm.resize_with(n, || None);
            self.conv.resize_with(n, || None);
            self.valid.resize(n, false);
        }
    }
    /// 層 1 本の確保。形が変わったら全層を無効化する (別モデル/別形状)。
    pub(crate) fn ensure_layer(&mut self, li: usize, rows: usize, cols: usize) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        self.touch_layer(li);
        if (self.rows != 0 && (self.rows != rows || self.cols != cols)) || self.rows != rows {
            self.invalidate();
            self.srows_n = 0;
            if self.pool.is_some() {
                self.drop_pool();
            }
        }
        self.rows = rows;
        self.cols = cols;
        // task 1577: プール有効時は ssm スロットも ensure_conv で張る
        // (conv 長がそろってからでないと 1 層ぶんのストライドが決まらない)。
        if gdn_pool_on() {
            return Ok(());
        }
        GdnScratch::grow(&mut self.ssm[li], rows * cols * F)?;
        Ok(())
    }
    /// 層 1 本の conv 状態を確保。
    pub(crate) fn ensure_conv(&mut self, li: usize, len: usize) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        self.touch_layer(li);
        self.conv_len = len;
        // task 1577: 連結プール (既定 off)。載らない形なら従来の層別確保へ落ちる。
        if gdn_pool_on() {
            if self.ensure_pool(li + 1, len, self.rows, self.cols)? {
                return Ok(());
            }
            // 載らない形 (conv か ssm が 0 要素) では ensure_layer も確保を
            // 見送っているので、ここで従来どおり層別に確保してから戻る。
            let ssm_bytes = self.rows * self.cols * F;
            if self.ssm[li].is_none() && ssm_bytes > 0 {
                GdnScratch::grow(&mut self.ssm[li], ssm_bytes)?;
            }
        }
        GdnScratch::grow(&mut self.conv[li], len * F)?;
        Ok(())
    }

    /// task 1577: プールを畳む。`conv` / `ssm` のビューは生ポインタを持たない
    /// ように先に捨てる (所有者を消す前に参照を残さない)。
    fn drop_pool(&mut self) {
        self.pool = None;
        for v in self.conv.iter_mut() {
            *v = None;
        }
        for v in self.ssm.iter_mut() {
            *v = None;
        }
    }

    /// task 1577: プールを (必要なら伸ばして) 確保し、層ビューを張り直す。
    ///
    /// 戻り値 true = プールに載った (`conv[li]` / `ssm[li]` がプール内ビュー)。
    /// false = 形が 0 で載せられない (呼び出し側が従来の層別確保へ落ちる)。
    fn ensure_pool(&mut self, need: usize, conv_len: usize, rows: usize, cols: usize) -> Result<bool> {
        let Some((cb, sb, stride)) = gdn_pool_layout(conv_len, rows, cols) else {
            return Ok(false);
        };
        let (cap, same_shape) = match &self.pool {
            Some(p) => (
                p.cap_layers,
                p.conv_len == conv_len && p.rows == rows && p.cols == cols,
            ),
            None => (0, false),
        };
        if same_shape && cap >= need {
            self.bind_pool_views();
            return Ok(true);
        }
        let cap = gdn_pool_grow_cap(cap, need);
        if self.pool.is_some() && !same_shape {
            // 形が変わった: 旧領域の中身は新しいストライドへ写せない →
            // 次の層利用でホストから seed し直す (`valid` を落とす)。
            self.invalidate();
        }
        let bytes = cap
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("gdn pool size overflow"))?;
        let buf = DevBuffer::alloc_tagged(bytes, crate::mem_audit::TAG_GDN_MIRROR)?;
        if same_shape {
            // 容量だけ伸ばす: 旧領域の中身 (valid な層の状態) を先頭へ写す。
            // 同期 D2D なので戻った時点で写し終わっている (旧領域はこの後 drop)。
            let mut copied = false;
            if let Some(old) = &self.pool {
                let n = old.buf.bytes().min(bytes);
                if n > 0 {
                    if let Ok(l) = lib() {
                        copied = unsafe { (l.memcpy_dtod)(buf.ptr(), old.buf.ptr(), n) } == CUDA_SUCCESS;
                    }
                } else {
                    copied = true;
                }
            }
            if !copied {
                // 写せないなら中身は信用しない (次にその層を使うときホストから seed)。
                for v in self.valid.iter_mut() {
                    *v = false;
                }
            }
        }
        // 旧ビューは新領域とは別人なので、まず全部落としてから差し替える。
        for v in self.conv.iter_mut() {
            *v = None;
        }
        for v in self.ssm.iter_mut() {
            *v = None;
        }
        let moved = self.pool.as_ref().map(|p| p.buf.ptr()).unwrap_or(0);
        self.pool = Some(GdnPool {
            buf,
            cap_layers: cap,
            conv_len,
            rows,
            cols,
        });
        self.bind_pool_views();
        let base = self.pool.as_ref().expect("pool 直前に確保").buf.ptr();
        tracing::info!(
            "[gdn-pool] alloc {} MiB ({} layers x {} B = conv {} B + ssm {} B)",
            bytes / (1024 * 1024),
            cap,
            stride,
            cb,
            sb
        );
        // task 1577: 連続領域 1 本を L2 persisting の窓にする
        // (DENNO_CUDA_L2_PERSIST=1 のときだけ。それ以外は no-op)。
        // 領域が動いた (再確保) なら古い窓を捨ててから張り直す。
        if moved != 0 && moved != base {
            l2::reset();
        }
        if l2::persist_range_budgeted(base, bytes) {
            tracing::info!(
                "[gdn-pool] L2 persisting window base=0x{base:x} (pool {} MiB, budget で clamp)",
                bytes / (1024 * 1024)
            );
        }
        Ok(true)
    }

    /// task 1577: プール内の各層の conv / ssm ビューを (再) 設定する。
    fn bind_pool_views(&mut self) {
        let (cap, conv_len, rows, cols, base) = match &self.pool {
            Some(p) => (p.cap_layers, p.conv_len, p.rows, p.cols, p.buf.ptr()),
            None => return,
        };
        let Some((cb, sb, stride)) = gdn_pool_layout(conv_len, rows, cols) else {
            return;
        };
        if self.conv.len() < cap {
            self.conv.resize_with(cap, || None);
            self.ssm.resize_with(cap, || None);
            self.valid.resize(cap, false);
        }
        for li in 0..cap {
            let cptr = base + (li * stride) as CUdeviceptr;
            if self.conv[li].as_ref().map(|b| b.ptr()) != Some(cptr) {
                self.conv[li] = Some(DevBuffer::view(cptr, cb));
            }
            let sptr = cptr + cb as CUdeviceptr;
            if self.ssm[li].as_ref().map(|b| b.ptr()) != Some(sptr) {
                self.ssm[li] = Some(DevBuffer::view(sptr, sb));
            }
        }
    }
    /// hoist 経路の境界行バッファ [n, nv, dk, dv] を確保 (n が同じなら再利用)。
    fn ensure_srows(&mut self, n: usize, rows: usize, cols: usize) -> Result<CUdeviceptr> {
        const F: usize = std::mem::size_of::<f32>();
        let bytes = n
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .and_then(|v| v.checked_mul(F))
            .ok_or_else(|| anyhow!("srows size overflow"))?;
        GdnScratch::grow(&mut self.srows, bytes)?;
        if self.srows_n != n {
            self.srows_n = n;
        }
        Ok(self.srows.as_ref().expect("srows 未確保").ptr())
    }
}


impl GdnScratch {
    pub(crate) const fn empty() -> Self {
        Self {
            q: None,
            z: None,
            beta: None,
            alpha: None,
            conv_in: None,
            ssm_in: None,
            wc: None,
            wa: None,
            wd: None,
            wn: None,
            w: None,
            m: None,
            sinit: None,
            y: None,
            conv_out: None,
            ssm_out: None,
        }
    }
    pub(crate) fn grow(cap: &mut Option<DevBuffer>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes()).unwrap_or(0) < bytes {
            *cap = Some(DevBuffer::alloc(bytes.max(1))?);
        }
        Ok(())
    }
    /// task 167: デバイス連鎖が必要とする分だけを確保する。
    ///
    /// 連鎖は q / z / beta / alpha / y を自前のバッファに持つので、
    /// [`Self::ensure`] が確保する約 90 MB (nt=1024 の実形) を無駄に取らない。
    fn ensure_chain(
        &mut self,
        conv_dim: usize,
        nv: usize,
        dk: usize,
        dv: usize,
        cs: usize,
    ) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        Self::grow(&mut self.wc, conv_dim * 4 * F)?;
        Self::grow(&mut self.wa, nv * F)?;
        Self::grow(&mut self.wd, nv * F)?;
        Self::grow(&mut self.wn, dv * F)?;
        Self::grow(&mut self.conv_out, conv_dim * cs * F)?;
        Self::grow(&mut self.ssm_out, nv * dk * dv * F)?;
        Ok(())
    }

    /// task 167: 連鎖が要るポインタだけ ([`Self::ensure_chain`] が見る 6 本)。
    /// `ptrs()` は q/z/beta/alpha/... まで全部を要求するので連鎖では使えない。
    fn chain_ptrs(&self) -> [CUdeviceptr; 6] {
        let p = |b: &Option<DevBuffer>| b.as_ref().expect("GdnScratch: 未確保").ptr();
        [
            p(&self.wc),
            p(&self.wa),
            p(&self.wd),
            p(&self.wn),
            p(&self.conv_out),
            p(&self.ssm_out),
        ]
    }

    /// 形が変わらない限り何もしない (prefill の 24 層は全て同じ形)。
    pub(crate) fn ensure(
        &mut self,
        n: usize,
        conv_dim: usize,
        vd: usize,
        nv: usize,
        dk: usize,
        dv: usize,
        cs: usize,
        n_chunks: usize,
    ) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        Self::grow(&mut self.q, n * conv_dim * F)?;
        Self::grow(&mut self.z, n * vd * F)?;
        Self::grow(&mut self.beta, n * nv * F)?;
        Self::grow(&mut self.alpha, n * nv * F)?;
        Self::grow(&mut self.conv_in, conv_dim * cs * F)?;
        Self::grow(&mut self.ssm_in, nv * dk * dv * F)?;
        Self::grow(&mut self.wc, conv_dim * 4 * F)?; // d_conv == 4 (chunked 前提)
        Self::grow(&mut self.wa, nv * F)?;
        Self::grow(&mut self.wd, nv * F)?;
        Self::grow(&mut self.wn, dv * F)?;
        Self::grow(&mut self.w, n_chunks * nv * dk * dv * F)?;
        Self::grow(&mut self.m, n_chunks * nv * dk * dk * F)?;
        Self::grow(&mut self.sinit, n_chunks * nv * dk * dv * F)?;
        Self::grow(&mut self.y, n * vd * F)?;
        Self::grow(&mut self.conv_out, conv_dim * cs * F)?;
        Self::grow(&mut self.ssm_out, nv * dk * dv * F)?;
        Ok(())
    }
    pub(crate) fn ptrs(&self) -> GdnPtrs {
        let p = |b: &Option<DevBuffer>| b.as_ref().expect("GdnScratch: 未確保").ptr();
        GdnPtrs {
            q: p(&self.q),
            z: p(&self.z),
            beta: p(&self.beta),
            alpha: p(&self.alpha),
            conv_in: p(&self.conv_in),
            ssm_in: p(&self.ssm_in),
            wc: p(&self.wc),
            wa: p(&self.wa),
            wd: p(&self.wd),
            wn: p(&self.wn),
            w: p(&self.w),
            m: p(&self.m),
            sinit: p(&self.sinit),
            y: p(&self.y),
            conv_out: p(&self.conv_out),
            ssm_out: p(&self.ssm_out),
        }
    }
    pub(crate) fn buf(&self, which: GdnBuf) -> &DevBuffer {
        let b = match which {
            GdnBuf::Y => &self.y,
            GdnBuf::ConvOut => &self.conv_out,
            GdnBuf::SsmOut => &self.ssm_out,
        };
        b.as_ref().expect("GdnScratch: 未確保")
    }
}


thread_local! {
    pub(crate) static GDN_SCRATCH: std::cell::RefCell<GdnScratch> =
        std::cell::RefCell::new(GdnScratch::empty());
    /// s89 (P1-1): GDN スキャン状態のデバイス正本 (scheduler ワーカーは単一
    /// スレッドなので thread_local で競合しない・`GDN_SCRATCH` と同じ規約)。
    pub(crate) static GDN_STATE: std::cell::RefCell<GdnStateCache> =
        std::cell::RefCell::new(GdnStateCache::empty());
}


/// s89 (P1-1): GDN スキャン状態のデバイス正本を「ホストが正」に戻す。
///
/// ホストの `KvCache::gdn` を書き換える経路 (セッション reset / エントリ復元 /
/// デコードチェーンの遅延フラッシュ / CPU フォールバック) は、その後にこの
/// 関数を呼ぶ。呼ばれた層は次の prefill でホストから再 seed する (数値不変・
/// 余分な H2D が 1 回増えるだけ)。**古い正本を読むと出力が変わる**ので、
/// 新しい書き換え経路を足すときは必ずここを通すこと。
pub fn gdn_scan_state_invalidate() {
    GDN_STATE.with(|c| c.borrow_mut().invalidate());
}


/// task 154: prefill の GPU スキャンが確保した層状態のバッファを VRAM へ返す。
///
/// 36 層ぶん (conv 172 KB + ssm 10.5 MB) で約 385 MB。prefill が終わったら
/// ホスト側の状態が正本なので、デバイス側を持ち続ける理由が無い (expert の
/// カード予算を圧迫して decode のヒット率を落とす)。次に使うときは
/// `gdn_state_begin` がホストから seed し直す。
pub fn gdn_state_release() {
    GDN_STATE.with(|c| {
        let mut c = c.borrow_mut();
        c.invalidate();
        c.ensure_layers(0);
        // task 1577: 連結プールも返す (VRAM を解放するのがこの関数の目的)。
        c.drop_pool();
        c.cur_kv = None;
        c.rows = 0;
        c.cols = 0;
        c.conv_len = 0;
    });
}


// s131: デバイス正本を使うか。`DENNO_GDN_DEVSTATE=0` で従来動作 (毎回ホストと
// 往復) に戻す。切り分け用で、既定は有効。
fn gdn_devstate() -> bool {
    !matches!(
        std::env::var("DENNO_GDN_DEVSTATE").ok().as_deref(),
        Some("0")
    )
}


thread_local! {
    /// task 379: 「いま回しているチャンク層は層の最終チャンクか」。
    ///
    /// 層優先 prefill のループが attn の直前で立て、直後に真へ戻す。既定は真
    /// (どこから呼ばれても従来どおり書き戻す)。[`GDN_WB_LAST_ENV`] が on の
    /// ときだけ読む。
    ///
    /// [`GDN_WB_LAST_ENV`]: crate::model::exp_pf::GDN_WB_LAST_ENV
    pub(crate) static PF_GDN_LAST_CHUNK: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}


/// task 379: このチャンク層が層の最終チャンクか (既定 = 真)。
pub fn pf_gdn_last_chunk() -> bool {
    PF_GDN_LAST_CHUNK.with(|c| c.get())
}


/// task 379: 「このチャンク層は層の最終チャンクか」を立てる。
///
/// 呼び出し側 (層優先 prefill) は attn の直後に必ず [`pf_gdn_last_chunk_reset`]
/// で真へ戻すこと (次の層の最初のチャンクが古い偽を拾わないため)。
pub fn pf_gdn_last_chunk_set(v: bool) {
    PF_GDN_LAST_CHUNK.with(|c| c.set(v));
}


/// task 379: [`pf_gdn_last_chunk_set`] を既定 (真) に戻す。
pub fn pf_gdn_last_chunk_reset() {
    PF_GDN_LAST_CHUNK.with(|c| c.set(true));
}


/// task 379: このチャンク層の書き戻しを飛ばしてよいか。
///
/// `DENNO_PF_GDN_WB_LAST=1` かつ最終チャンク以外、かつデバイスが正本
/// (`DENNO_GDN_DEVSTATE=0` の切り分け時は毎チャンクの書き戻しが要る)、かつ
/// 診断のトークン逐次 (`DENNO_DEBUG_GDN_DUMP_LAYER` がこの層を指す) でないとき
/// だけ真。診断時は連鎖に乗らずホスト経路がホスト控えを読むので、古い控えを
/// 渡さない。
pub fn pf_gdn_last_chunk_skippable(li: usize) -> bool {
    crate::model::exp_pf::gdn_wb_last_on()
        && gdn_devstate()
        && !pf_gdn_last_chunk()
        && !crate::gdn::gdn_dbg_enabled_for(li as i64)
}


thread_local! {
    /// task 379: [`gdn_state_restore_host`] の受け皿 (層ごとに使い回す)。
    pub(crate) static GDN_RESTORE_STAGE: std::cell::RefCell<GdnHostStage> =
        std::cell::RefCell::new(GdnHostStage::new());
}


/// task 379: デバイス正本の層状態をホストの配列へ戻す (`valid[li]` は落とす)。
///
/// `DENNO_PF_GDN_WB_LAST` で途中チャンクの書き戻しを飛ばしている木で、そのチャンク
/// が連鎖に乗れずホスト経路へ落ちたときにだけ使う (古い控えで計算しないため)。
/// 戻せたら真。デバイス正本が無い (未確定 / 別 kv) ときは何もせず偽。
pub fn gdn_state_restore_host(kv: u64, li: usize, conv: &mut [f32], ssm: &mut [f32]) -> bool {
    GDN_RESTORE_STAGE.with(|cell| {
        let mut st = cell.borrow_mut();
        match gdn_state_take(kv, li, &mut st) {
            Some((c, s)) if c.len() == conv.len() && s.len() == ssm.len() => {
                conv.copy_from_slice(c);
                ssm.copy_from_slice(s);
                true
            }
            _ => false,
        }
    })
}


thread_local! {
    /// task 397: 状態の書き戻しを受ける pinned の受け皿 (層ごとに使い回す)。
    pub(crate) static GDN_WB_STAGE: std::cell::RefCell<Option<PinMem>> =
        const { std::cell::RefCell::new(None) };
}


/// task 397: 状態の書き戻しを `cuMemcpyDtoHAsync` + 自ストリームの待ちで行う。
///
/// `cuMemcpyDtoH` (= [`DevBuffer::to_slice`]) はコンテキスト全体の完了を待つので、
/// mix の別ストリームと expert の上げストリームの残りまで待ってしまう。
/// pinned へ非同期で落としてから [`pf_sync_own_stream`] で null ストリームだけを
/// 待てば、**ホストへ載るバイト列は同じ**で待ちの範囲だけが狭くなる。
/// 戻り値 `false` は「この経路を使わなかった」(呼び出し側は従来の同期コピーへ)。
fn gdn_wb_async_copy(
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    conv_ptr: CUdeviceptr,
    ssm_ptr: CUdeviceptr,
) -> Result<bool> {
    let cb = std::mem::size_of_val(conv_st);
    let sb = std::mem::size_of_val(ssm_st);
    let bytes = cb + sb;
    if bytes == 0 {
        return Ok(false);
    }
    GDN_WB_STAGE.with(|cell| -> Result<bool> {
        let mut g = cell.borrow_mut();
        let need = match g.as_ref() {
            Some(p) => p.bytes < bytes,
            None => true,
        };
        if need {
            // 古い 1 本を落としてから取り直す (pinned を二重に持たない)。
            *g = None;
            match PinMem::alloc(bytes) {
                Ok(p) => *g = Some(p),
                Err(_) => return Ok(false),
            }
        }
        let pin = g.as_ref().expect("確保済み").ptr;
        let lib = lib()?;
        unsafe {
            if check(
                (lib.memcpy_dtoh_async)(pin, conv_ptr, cb, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (gdn conv)",
            )
            .is_err()
            {
                return Ok(false);
            }
            if check(
                (lib.memcpy_dtoh_async)(
                    (pin as *mut u8).add(cb) as *mut c_void,
                    ssm_ptr,
                    sb,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (gdn ssm)",
            )
            .is_err()
            {
                return Ok(false);
            }
        }
        // 自ストリーム (null) の完了だけを待つ。失敗したら呼び出し側が
        // 同期コピーで丸ごとやり直す (待たないまま読まない)。
        if pf_sync_own_stream().is_err() {
            return Ok(false);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(pin as *const u8, conv_st.as_mut_ptr() as *mut u8, cb);
            std::ptr::copy_nonoverlapping(
                (pin as *const u8).add(cb),
                ssm_st.as_mut_ptr() as *mut u8,
                sb,
            );
        }
        Ok(true)
    })
}


// 従来動作のホスト書き戻し (デバイス正本を使わないときだけ)。
fn gdn_host_writeback(conv_st: &mut [f32], ssm_st: &mut [f32]) -> Result<()> {
    GDN_SCRATCH.with(|cell| -> Result<()> {
        let s = cell.borrow();
        // task 397: `DENNO_PF_GDN_WB_ASYNC` は非同期 + 自ストリームの待ち。
        // 失敗 (pinned が取れない / 転送エラー) は下の同期経路でやり直す。
        if crate::model::exp_pf::gdn_wb_async_on() {
            let cp = s.buf(GdnBuf::ConvOut).ptr();
            let sp = s.buf(GdnBuf::SsmOut).ptr();
            if gdn_wb_async_copy(conv_st, ssm_st, cp, sp)? {
                return Ok(());
            }
        }
        if crate::model::exp_pf::gdn_wb_direct_on() {
            // task 334: `to_vec` は 1.6 MB の Vec を確保して D2H したあと
            // `copy_from_slice` でもう 1 回写す (毎チャンク 2 本 x 48 層 x 7)。
            // `to_slice` は呼び出し側の配列へ直接 D2H する (値は同じバイト列)。
            s.buf(GdnBuf::ConvOut).to_slice(conv_st)?;
            s.buf(GdnBuf::SsmOut).to_slice(ssm_st)?;
        } else {
            let cv = s.buf(GdnBuf::ConvOut).to_vec::<f32>(conv_st.len())?;
            let sm = s.buf(GdnBuf::SsmOut).to_vec::<f32>(ssm_st.len())?;
            conv_st.copy_from_slice(&cv);
            ssm_st.copy_from_slice(&sm);
        }
        Ok(())
    })
}


// s131: デバイス正本の入り口。層の状態が未確定ならホストから 1 回だけ seed し、
// 以後はデバイスの値をそのままカーネルへ渡す。
fn gdn_state_begin(
    kv: u64,
    li: usize,
    conv: &[f32],
    ssm: &[f32],
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    GDN_STATE.with(|cell| -> Result<(CUdeviceptr, CUdeviceptr)> {
        let mut c = cell.borrow_mut();
        // 別スロット/別セッションの状態を混ぜない (層番号だけでは足りない)。
        if c.cur_kv != Some(kv) {
            c.invalidate();
            c.cur_kv = Some(kv);
        }
        let devstate = gdn_devstate();
        c.ensure_layer(li, rows, cols)?;
        c.ensure_conv(li, conv_len)?;
        if !c.valid[li] || !devstate {
            c.conv[li].as_ref().unwrap().copy_from(conv)?;
            c.ssm[li].as_ref().unwrap().copy_from(ssm)?;
            c.valid[li] = devstate;
        }
        let pc = c.conv[li].as_ref().unwrap().ptr();
        let ps = c.ssm[li].as_ref().unwrap().ptr();
        // task 1577: GDN のデバイス正本を L2 persisting にする (既定 off)。
        // プール有効時は ensure_conv がプール全体を 1 本の窓で登録済みで、
        // ここに来る時点で窓はプール (より大きい) なので driver は呼ばれない。
        let pooled = c.pool.is_some();
        if !pooled {
            l2::persist_range(pc, conv_len * 4);
            l2::persist_range(ps, rows * cols * 4);
        }
        Ok((pc, ps))
    })
}


// s131: スキャン結果をデバイス正本へ D2D で戻す (ホストへは落とさない)。
fn gdn_state_commit(
    li: usize,
    conv_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let (pc, ps) = GDN_STATE
        .with(|cell| -> Option<(CUdeviceptr, CUdeviceptr)> {
            let c = cell.borrow();
            Some((c.conv.get(li)?.as_ref()?.ptr(), c.ssm.get(li)?.as_ref()?.ptr()))
        })
        .ok_or_else(|| anyhow!("gdn_state_commit: 状態バッファ未確保 (layer {li})"))?;
    let lib = lib()?;
    unsafe {
        check(
            (lib.memcpy_dtod_async)(pc, conv_out, conv_len * 4, std::ptr::null_mut()),
            "cuMemcpyDtoDAsync (gdn conv)",
        )?;
        check(
            (lib.memcpy_dtod_async)(ps, ssm_out, rows * cols * 4, std::ptr::null_mut()),
            "cuMemcpyDtoDAsync (gdn ssm)",
        )?;
    }
    Ok(())
}


/// s131: デバイス正本をホストへ戻す (層指定)。戻した層は無効化するので、
/// 次にその層を使うときはホストから 1 回 seed し直す。
/// ホストの `KvCache::gdn` を読む前に必ず通すこと。
///
/// s141 (Task B7): 返す 2 本は **thread_local のホスト一時領域の借り物**。
/// 従来は層ごとに `to_vec` (= `Vec::with_capacity`) で 2 本確保していたため、
/// 27B q4 の decode 1 ステップで 48 層 x 2 本 = 96 本 / 564 KiB の確保と解放を
/// 毎回払っていた。転送・内容は同一 (同じ D2H を同じ長さで 1 回) で、変わるのは
/// 「毎回確保するか、伸ばすだけの 1 本を使い回すか」だけ。
/// 借り物なので、**次の呼び出しで中身が変わる** — 呼び出し側はその場で読み終わること
/// (`KvCache::flush_gdn_device_state` はその場で copy_from_slice する)。
pub fn gdn_state_take<'a>(kv: u64, li: usize, reuse: &'a mut GdnHostStage) -> Option<(&'a [f32], &'a [f32])> {
    let (pc, ps, conv_len, ssm_len) =
        GDN_STATE.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr, usize, usize)> {
            let mut c = cell.borrow_mut();
            if c.cur_kv != Some(kv) || li >= c.valid.len() || !c.valid[li] {
                return None;
            }
            let (conv_len, rows, cols) = (c.conv_len, c.rows, c.cols);
            let pc = c.conv[li].as_ref()?.ptr();
            let ps = c.ssm[li].as_ref()?.ptr();
            // 戻した層は無効化 (従来どおり・成功時のみ)。
            c.valid[li] = false;
            Some((pc, ps, conv_len, rows * cols))
        })?;
    // 借り物を 2 本同時に返すので、転送は 2 回とも先に済ませる。
    // conv / ssm は別々の領域なので、2 本の借用は同時に成立する。
    let conv = d2h_raw(&mut reuse.conv, pc, conv_len)?;
    let ssm = d2h_raw(&mut reuse.ssm, ps, ssm_len)?;
    Some((conv, ssm))
}


/// s141 (Task B7): `gdn_state_take` に渡す呼び出し側所有の受け皿 1 組。
/// 呼び出し元のループの外で 1 回だけ作り、層ごとに使い回す (grow-only)。
/// 中身は「前の層の値」なので、**返り値をその場で読み終えること**。
pub struct GdnHostStage {
    pub(crate) conv: Vec<f32>,
    pub(crate) ssm: Vec<f32>,
}


impl GdnHostStage {
    /// 空の受け皿 (初回の D2H で必要な分だけ伸びる)。
    pub fn new() -> Self {
        Self { conv: Vec::new(), ssm: Vec::new() }
    }
    /// 確保済みの要素数 (テストが「ステップ数に対して増えない」ことを見る口)。
    pub fn capacities(&self) -> (usize, usize) {
        (self.conv.capacity(), self.ssm.capacity())
    }
}


impl Default for GdnHostStage {
    fn default() -> Self {
        Self::new()
    }
}


/// s141b (Task B7c): デバイス正本の 1 層を「デバイス正本から読む」ために取り出す。
///
/// s131 の `gdn_state_take` はホストへ D2H する版 (ホストの `KvCache::gdn` を
/// 復帰させる用途)。こちらは decode chain の seed 用で、返すのは移し元の
/// デバイスポインタだけ — ホストを経由しないので層あたり ~3.2 MiB の PCIe 往復が
/// 消える。返す値は (conv ポインタ, ssm ポインタ)。
///
/// 取り出した層は**無効化しない** (正本の値はそのまま有効 — 読むだけ)。ただし
/// 読み手は「ミラーへコピーした後」に scan カーネルがその面を更新するので、正本と
/// ミラーの内容は今後ずれる → 戻した層は valid=false にする (従来の
/// `gdn_state_take` と同じ規則: 次に scan 経路で使うときは必ず seed し直す)。
pub(crate) fn gdn_state_take_dev(
    kv: u64,
    li: usize,
    c: &mut GdnStateCache,
) -> Option<(CUdeviceptr, CUdeviceptr)> {
    if c.cur_kv != Some(kv) || li >= c.valid.len() || !c.valid[li] {
        return None;
    }
    let pc = c.conv.get(li)?.as_ref()?.ptr();
    let ps = c.ssm.get(li)?.as_ref()?.ptr();
    c.valid[li] = false;
    Some((pc, ps))
}


/// s141b (Task B7c): 遅延フラッシュの「D2H が要るか」の判定 (純関数 — テストが
/// 実物を通す)。
///
/// 規則: **ホストが古い可能性がある層** (= `stale`) と **位置が合っている層**
/// (= `pos == Some(target)` / `Some(target+1)`) の両方を満たす層だけ D2H する。
///
/// - stale でない層はホストが既に最新 (このステップで面 0 を上から書いた) →
///   読み直さなくてよい。これが「何も要らないステップでホストバイトをコピー
///   しない」の実体。
/// - stale でも位置が合っていない層 (chain 未走 / 不整合) は従来どおり判定へ回し、
///   呼び出し側がバグ時 None を返す経路をそのまま踏ませる。
///
/// 戻り値: (D2H するか, 位置が合っているか)。
pub(crate) fn gdn_flush_step(is_gdn_layer: bool, stale: bool, target: usize, pos: Option<usize>) -> (bool, bool) {
    if !is_gdn_layer {
        return (false, false);
    }
    // 成功ステップ後: 位置は target (= cache.len()) に一致し、その面は**書き込んだ
    // 状態** = ホストが遅れている層だけが読み直しの対象。
    if pos == Some(target) {
        return (stale, true);
    }
    // 層途中で失敗したステップ: 位置は target+1 で、書いた面は「次」の状態。
    // 現在の状態 (target) は反対の面にあり、その面は前回の seed 直後に書かれた
    // = **ホストと一致している** (= stale が偽) ので、stale の層は読んでも
    // 意味が無い (古いホストを書き戻すだけ)。一律で「読まない」。
    if pos == Some(target + 1) {
        return (false, true);
    }
    (false, false)
}


/// s239 (P36): GDN pass 1 を 4 レーン版で回すか (`DENNO_GDN_LANES=4` で有効)。
///
/// 4 レーン版は W/M をレジスタ常駐にしてローカルアクセス命令を 1/4 にするが、
/// 内積が 4 部分和の木縮約になるため **旧版と bit 一致しない** (GDN は bit 一致
/// 不要契約・決定論のみ)。アンカー v6_* は GDN 層を通るので、既定にする際は
/// 再基準化が必要。よって既定は旧カーネル (off) とし、A/B 用の入口だけ用意する。
fn gdn_lanes4() -> bool {
    static L4: OnceLock<bool> = OnceLock::new();
    *L4.get_or_init(|| {
        std::env::var("DENNO_GDN_LANES")
            .ok()
            .map(|v| v.trim() == "4")
            .unwrap_or(false)
    })
}


/// P55: GDN pass 1 の状態配列 (W/M) を共有メモリへ置く版を使うか
/// (`DENNO_GDN_PRE_SM=1` で有効・既定 off)。
///
/// 数値は旧版と **ビット同一** (格納先がローカルメモリから共有メモリに変わる
/// だけで、値は f32 のまま・読み書きの順序/回数/丸めは同一)。既定 off の理由は
/// 数値ではなく occupancy で、dk=128 では 1 ブロックの smem が 129 KiB になり
/// 1 block/SM まで落ちる。P36 (4 レーン版) が同じ理由で利得を打ち消したので、
/// 実機の `[GDN_KPROF]` と cold A/B で確かめるまで既定にはしない。
fn gdn_pre_sm_enabled() -> bool {
    static SM: OnceLock<bool> = OnceLock::new();
    *SM.get_or_init(|| {
        std::env::var("DENNO_GDN_PRE_SM")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
    })
}


/// P55: 共有メモリ版 pass 1 の動的 smem 既定上限 (48 KiB)。`cuFuncSetAttribute`
/// による opt-in に失敗した場合の値で、dk=128 の必要量 (129 KiB) を満たさない
/// ためゲートで旧カーネルへ落ちる。
pub(crate) const GDN_PRE_SM_SMEM_DEFAULT: usize = 48 * 1024;


/// P55: 共有メモリ版 pass 1 が載る最大 dk。必要量 = (2*dk*dk + 2*dk + 3)*4 B
/// なので dk=128 で 132,108 B (opt-in 上限 164 KiB / 227 KiB の範囲)、
/// dk=192 では 296,652 B になり opt-in 上限を超える。27B の GDN は dk=dv=128
/// (`linear_key_head_dim`・kernel_invariants の実形状表)。dk > 128 の形は
/// 従来カーネルで回る (このゲートはそのための上限)。
const GDN_PRE_SM_MAX_DK: usize = 128;


/// P55: 共有メモリ版 pass 1 の動的 smem 必要量 (bytes)。
/// カーネル側のレイアウト W[dk][dk] | M[dk][dk] | k[dk] | v[dk] | scalars[3] と
/// 同じ式 (kernel_invariants の `gdn_kernel_shared_layout_matches_host_smem` が
/// カーネル側の宣言と対で見る)。
const fn gdn_pre_sm_smem_bytes(dk: usize) -> usize {
    (2 * dk * dk + 2 * dk + 3) * 4
}


/// P55: 共有メモリ版 pass 1 を選ぶ条件 (純粋関数・GPU 不要で検証できる)。
///
/// `env_on` = `DENNO_GDN_PRE_SM=1` / `has_fn` = PTX からカーネルを取得できたか /
/// `smem` = 必要量 / `limit` = デバイスの opt-in 上限 (失敗時は 48 KiB)。
/// 1 つでも欠けたら false = 旧カーネル。カーネル不在・opt-in 失敗・dk 過大の
/// どれでも「静かに従来経路」になることが契約。
pub(crate) fn gdn_pre_sm_gate(
    env_on: bool,
    has_fn: bool,
    dk: usize,
    dv: usize,
    smem: usize,
    limit: usize,
) -> bool {
    env_on && has_fn && dk == dv && (1..=GDN_PRE_SM_MAX_DK).contains(&dk) && smem <= limit
}


// ---------------------------------------------------------------------------
// task 504: dn7 を mix の down 以外 (GDN の out 投影 / full 層の qkv・wo) へ
// ---------------------------------------------------------------------------
//
// P500 の profile: ssm の out 投影 (n_in 6,144 / n_out 2,560) は dn8 で 16.7 MB
// を 24.8 us = 673 GB/s、full 層の qkv/wo は dn6 で 6.96 MB を 23.4 us =
// 298 GB/s しか出ない。どちらも np が 96 / 40 で、dn8/dn6 は **1 行の和を 1
// レーンが np 本直列**に足す形 (依存鎖 96 / 40 段) なので、latency で止まる。
// dn7 (`kernels_mix_dn7.cu`) は 1 対を 4 レーンに割って 16 要素ずつ積むので
// 依存鎖が 1/4 になる (mix の down では 14.0 -> 12.1 us)。同じ PTX をそのまま
// 撃つだけで、新しい `.cu` は要らない。
//
// 値は丸め順だけが変わる (16 要素ずつ 4 レーンに分けて足す)。ゲート:
//   - GDN の out: `DENNO_GDN_DEC_GEMV=dn8ao7` (qkv は dn8 のまま)
//   - full 層:    `DENNO_FULL_DEC_FAST=dn7` (qkv と wo の両方)
// 既定はどちらも従来どおり (off)。

/// GDN の out 投影だけ dn7 で撃つか (純関数)。
///
/// task 504 (lab t504b で採否): **未設定が既定 on** (ssm 2.93 -> 2.87 = -2.1%、
/// 2 ラウンド同じ、md5 SAME)。
/// - 未設定: true = out を dn7
/// - `dn8ao7` / `dn8o7` / `d8o7` / `d8ao7`: true (明示 on)
/// - それ以外の明示値 (`dn8ar4` / `dn8a` / `dn8` / `dn6` / `v5` / ...): false
///   = その値が決める従来のカーネル。**厳密な切り戻しは `dn8ar4`**
///   (out = dn8 rows 4、t496c の既定)。
pub fn gdn_out_dn7_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim).map(str::to_ascii_lowercase) {
        Some(s) => s.ends_with("o7"),
        None => true,
    }
}


/// [`gdn_out_dn7_from_value`] をプロセスで 1 回だけ読む。
pub(crate) fn gdn_out_dn7_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        gdn_out_dn7_from_value(std::env::var("DENNO_GDN_DEC_GEMV").ok().as_deref())
    })
}


/// C4 (qwen35): GDN 状態 (conv + SSM) のデバイスミラー 1 層分。
/// dno_gdn_scan は in/out に別ポインタを要求する (nv > nk のとき複数ブロックが
/// 同一 conv チャネルを共有するため in==out エイリアスは不正) — ping-pong 2 面で
/// 「cur を読み、1-cur に書き、cur を交換」する。
pub(crate) struct GdnDev {
    pub(crate) conv: [Option<DevBuffer>; 2], // 各 [conv_dim, cs] f32
    pub(crate) ssm: [Option<DevBuffer>; 2],  // 各 [nv, dk, dv] f32
    pub(crate) cur: usize,
}


impl GdnDev {
    pub(crate) fn bytes(&self) -> usize {
        self.conv
            .iter()
            .chain(self.ssm.iter())
            .map(|b| b.as_ref().map(|b| b.bytes()).unwrap_or(0))
            .sum()
    }
}


/// C4: qwen35 GDN 層の decode スキャン (デバイス常駐)。
/// 前提: CH_H に rmsnorm 出力、CH_QKV に [qkv (conv_dim) | z (vd)] がある
/// (chain_linear_grouped 直後)。beta/alpha は本関数が dno_matvec_f32 で
/// CH_QKV テール [(conv_dim+vd)..] へ出力する。GDN 状態ミラー (ping-pong) を
/// seed/更新し、dno_gdn_scan (既存カーネル・grid=(nv,), block=(dk,)) を起動 —
/// y を CH_ATT に書く。新状態 (conv/ssm) は s50 からミラーの書き込み面に留め、
/// ステップ毎のホスト D2H はしない (遅延フラッシュ — 契約と回復手順は
/// chain_q35_gdn_flush_pending)。失敗時 None (forward 全体を従来経路でやり直し)。
#[allow(clippy::too_many_arguments)]
pub fn chain_q35_gdn_scan(
    mid: u64,
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &mut crate::kv_cache::KvCache,
    pos0: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    beta_w: &[f32],
    alpha_w: &[f32],
    // s174 (P16): [beta | alpha] の行連結 [2*nv, e] (行順 [beta 全行 | alpha 全行])。
    // `Some` かつ形状が合うときだけ beta/alpha を 1 発に畳む (`None` は従来の 2 発)。
    ba_w: Option<&[f32]>,
    e: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    // 新カーネル (dno_matvec_f32) の存在確認を早めに行う (ctl_f32 内でも失敗するが
    // ここで弾く方が意図が明確)。
    if q35_fns().is_none() {
        return None;
    }
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv; // == qkv_w.n (GDN カーネルの qkv 行幅)
    let vd = nv * dv;
    let t_total = pos0 + 1;
    if dk == 0 || dv == 0 || dk != dv || d_conv != 4 || !(1..=256).contains(&dk) {
        return None;
    }
    if wconv.len() != conv_dim * d_conv || wa.len() != nv || wdt.len() != nv || wnorm.len() != dv {
        return None;
    }
    if beta_w.len() != e * nv || alpha_w.len() != e * nv {
        return None;
    }
    let width = conv_dim + vd + 2 * nv;
    ok(chain_ensure_buf(CH_QKV, width * 4))?;
    ok(chain_ensure_buf(CH_ATT, vd * 4))?;

    // beta/alpha 制御投影 (fp32 matvec・CH_QKV テールへ)
    // s174 (P16): 連結重み [beta | alpha] があれば **1 発** (grid=(2*nv))。
    // 同じ入力 CH_H・同じ形状・隣接発行で、出力もテールの連続領域なので
    // 1 発に畳める (行独立な逐次縮約 = ビット一致)。無ければ従来の 2 発。
    match ba_w {
        Some(w) if w.len() == e * 2 * nv && 2 * nv <= 65535 => {
            chain_q35_ctl_f32_ba(mid, li, w, e, nv, conv_dim + vd)?
        }
        _ => {
            chain_q35_ctl_f32(mid, li, Q35_TAG_BETA, beta_w, e, nv, conv_dim + vd)?;
            chain_q35_ctl_f32(mid, li, Q35_TAG_ALPHA, alpha_w, e, nv, conv_dim + vd + nv)?;
        }
    }

    // s141b (Task B7c): seed の入力は **デバイス正本** から直接 D2D で取る。
    // 以前は「正本 → ホスト (D2H) → ミラー (H2D)」と、同一 GPU 内のコピーを
    // わざわざ PCIe 経由で往復させていた (層ごとに ~3.2 MiB の D2H)。
    // 27B q4 の decode では `with_mirror_ext` の cap 成長時に全層の gdn_pos が
    // 落ちるため、この往復が 48 層 x 2 本ずつ (= gdn_pf_alloc の実体) 発生していた。
    //
    // 規則 (所有者はこう変わる):
    // - デバイスミラー = decode 中の権威 (従来どおり)
    // - ホスト `KvCache::gdn` = **遅延リフレッシュ対象**。mirror.gdn_stale が真の間は
    //   ホストが古い可能性がある。ホストを読む前に必ず
    //   chain_q35_gdn_flush_pending (= mirror.gdn_pending を Some にしておく) か
    //   KvCache::flush_gdn_device_state を通すこと。
    // - 正本が無効な層 (valid[li] == false = ホストが正) だけは従来どおりホストから
    //   H2D する。ホストはその時点で正 (何も捨てていない)。
    //
    // 失敗時の扱い: 正本から読めなかった (None) 場合は従来どおりホストから seed
    // する。これは s141 までの「正本が無効な層 = ホストが正」と同じ状態で、
    // 正本が無効になるのは必ずホストが正に戻された後 (gdn_scan_state_invalidate は
    // ホスト書き換え経路が呼ぶ) なので、古いホストを読むことはない。
    let devseed = GDN_STATE.with(|cell| {
        let mut c = cell.borrow_mut();
        gdn_state_take_dev(cache.kv_id(), li, &mut c)
    });
    let (conv_in, ssm_in, conv_out, ssm_out) =
        with_mirror_ext(key, li, n_layers, t_total, 0, cache.epoch(), false, |m, _| {
            if m.gdn[li].is_none() {
                let mut g = GdnDev {
                    conv: [None, None],
                    ssm: [None, None],
                    cur: 0,
                };
                ok(Scratch::ensure_tagged(
                    &mut g.conv[0],
                    conv_dim * cs * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.conv[1],
                    conv_dim * cs * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.ssm[0],
                    nv * dk * dv * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.ssm[1],
                    nv * dk * dv * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                m.gdn[li] = Some(g);
            }
            let gd = m.gdn[li].as_mut().unwrap();
            // seed: この層のミラーが pos0 に同期していなければ状態を面 0 へ入れる
            if m.gdn_pos[li] != Some(pos0) {
                // s131/s141b: 正本 (デバイス) が有効ならそこから D2D で取る。
                // 無効 (= valid[li] == false = ホストが正) のときだけホストから。
                match devseed {
                    Some((pc, ps)) => {
                        ok(gd.conv[0].as_ref().unwrap().copy_from_dev(pc, conv_dim * cs))?;
                        ok(gd.ssm[0].as_ref().unwrap().copy_from_dev(ps, nv * dk * dv))?;
                        // 正本から入れた = **ホストは更新していない**。ホストが古い
                        // 可能性は残るので遅延フラグは立てたままにする (ここで落とすと、
                        // 後で flush がこの層を飛ばして古いホストを正として使ってしまう)。
                        m.gdn_stale[li] = true;
                    }
                    None => {
                        // 正本が無効 = ホストが正。ホストを読んで面 0 を書くので
                        // ミラーとホストは一致する → 遅延フラグを落とす。
                        let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
                        ok(gd.conv[0].as_ref().unwrap().copy_from(conv_h))?;
                        ok(gd.ssm[0].as_ref().unwrap().copy_from(ssm_h))?;
                        m.gdn_stale[li] = false;
                    }
                }
                gd.cur = 0;
            }
            let cur = gd.cur;
            let nxt = 1 - cur;
            Some((
                gd.conv[cur].as_ref().unwrap().ptr(),
                gd.ssm[cur].as_ref().unwrap().ptr(),
                gd.conv[nxt].as_ref().unwrap().ptr(),
                gd.ssm[nxt].as_ref().unwrap().ptr(),
            ))
        })?;

    // dno_gdn_scan 起動 (qkv/z は CH_QKV・beta/alpha はテール・y は CH_ATT)
    let (mut pq, mut pz, mut pb, mut pa, mut pci, mut psi) = CHAIN.with(
        |cell| -> Option<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            let base = s.bufs[CH_QKV].as_ref()?.ptr();
            Some((
                base,
                base + (conv_dim * 4) as u64,
                base + ((conv_dim + vd) * 4) as u64,
                base + ((conv_dim + vd + nv) * 4) as u64,
                conv_in,
                ssm_in,
            ))
        },
    )?;
    let pwc = q35_w_dev(mid, li, Q35_TAG_CONV, wconv)?;
    let pwa = q35_w_dev(mid, li, Q35_TAG_WA, wa)?;
    let pwd = q35_w_dev(mid, li, Q35_TAG_WDT, wdt)?;
    let pwn = q35_w_dev(mid, li, Q35_TAG_WNORM, wnorm)?;
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (pwc, pwa, pwd, pwn);
    let (mut py, mut pco, mut pso) = (
        CHAIN.with(|cell| -> Option<CUdeviceptr> {
            let s = cell.borrow();
            Some(s.bufs[CH_ATT].as_ref()?.ptr())
        })?,
        conv_out,
        ssm_out,
    );
    let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
        1 as c_int, nk as c_int, nv as c_int, dk as c_int, dv as c_int,
    );
    let (mut cdi, mut dci, mut csi) = (conv_dim as c_int, d_conv as c_int, cs as c_int);
    let (mut epsi, mut qsi) = (eps, q_scale);
    let mut args = [
        &mut pq as *mut _ as *mut c_void,
        &mut pz as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut psi as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pwn as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut dci as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
        &mut qsi as *mut _ as *mut c_void,
    ];
    let shmem = ((3 * dk + dv + 5) * 4) as u32;
    ok(unsafe {
        // dno_gdn_scan は既存カーネル (kernels_gdn.ptx・ctx.f_gdn に取得済み) を
        // そのまま使う — 新 PTX は不要 (kernels_elem 側の新カーネルのみ人間再生成)。
        ctx.launch_async(ctx.f_gdn, (nv as u32, 1, 1), (dk as u32, 1, 1), shmem, &mut args)
    })?;

    // s50: 新状態 (ping-pong の書き込み面) はホストへ戻さない (遅延フラッシュ)。
    // 旧実装はここで conv/ssm を pinned へ staged async D2H し、sync 後の
    // chain_q35_gdn_stage_flush でホストへ反映していた。ステップ毎 D2H は
    // 9B で ~75MB/step = decode の支配項だったため、ミラーを権威のままにして
    // 「ホスト状態が実際に必要になる時点」まで転送を遅らせる。

    // ミラー位置の確定 (launch 成功後 — 失敗時は GPU_BROKEN で chain 停止のため
    // 誤更新は残らない)。ping-pong 面を交換し、この層の新状態を遅延 pending に
    // 載せる。層途中で chain が失敗した場合は掃けた層だけ pending になる — flush 側は
    // gdn_pos と cache.len() の整合で掃けた面だけを回復できる。
    with_mirror_ext(key, li, n_layers, t_total, 0, cache.epoch(), false, |m, _| {
        m.gdn_pos[li] = Some(pos0 + 1);
        m.gdn_pending = Some(pos0 + 1);
        // s141b (Task B7c): この層の新状態は書き込み面にだけある (ホストは古い)。
        // 以後この層の seed はホストではなくデバイス正本から取る。
        m.gdn_stale[li] = true;
        if let Some(gd) = m.gdn[li].as_mut() {
            gd.cur = 1 - gd.cur;
        }
        Some(())
    })
}


/// s50: 遅延 GDN 状態フラッシュ。decode chain は GDN の新状態 (conv/ssm) をステップ
/// 毎にホストへ戻さず GPU ミラー (ping-pong の cur 面) を権威のままにする
/// (chain_q35_gdn_scan の doc 参照 — 旧実装のステップ毎 D2H は 9B で ~75MB/step =
/// decode の支配項)。本関数は「ホスト状態が実際に必要になる時点」—
/// ①次の prefill の冒頭 (forward_prefill はホスト状態を正として読み/書く)
/// ②decode chain から従来 CPU 経路へのフォールバック時 (forward) —
/// に呼ばれ、ミラーの現行面を同期 D2H でホストキャッシュへ回復する。
///
/// 契約:
/// - ミラーの内容は各 decode ステップ末尾の chain_download (sync) で確定済み
///   (ステップ末尾にステップ後の非同期カーネルを残さない現行設計) → 同期 D2H
///   (DevBuffer::to_vec) は確定値を読む。転送は bit そのまま (演算なし) のため、
///   フラッシュ後のホスト状態は「毎ステップフラッシュした場合」と bit 一致。
/// - 位置整合: 対象位置 = cache.len()。gdn_pos[li] == target → cur 面、
///   gdn_pos[li] == target+1 → 1-cur 面 (直前の chain ステップが層途中で失敗した
///   ケースの ping-pong 入力面 = state at target)。gdn_pos[li] == None → その層は
///   chain で 1 度も走っていない (ホスト状態が最新) → スキップ。それ以外は不整合
///   (バグのみ) → pending 破棄のうえ None (呼び出し側は bail で要求を落とす =
///   古い状態で黙って継続するより安全)。
/// - epoch 不一致 (reset/restore でホストが正に戻った) → pending 破棄で no-op
///   (前会話のデータは使わない — ホストが正)。
/// - pending 無し (通常の prefill 毎・fp32 モード等) は no-op (登録簿参照のみ)。
/// - ミラーロック保持中に D2H を実施する (解放すると LRU エビクトが走り得て
///   読み出し元の DevBuffer が解放される競合になる — with_mirror 内 H2D と同一方針)。
/// 戻り値: None = ホスト状態を回復できなかった (呼び出し側は継続してはならない)。
pub fn chain_q35_gdn_flush_pending(cache: &mut crate::kv_cache::KvCache) -> Option<()> {
    let key = cache.kv_id();
    let epoch = cache.epoch();
    let target = cache.len();
    let mirrors = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut ms = ok(mirrors.lock().map_err(|e| anyhow!("{e}")))?;
    let Some(idx) = ms.iter().position(|(k, _, _)| *k == key) else {
        return Some(()); // ミラー未作成 = 遅延も存在し得ない
    };
    // s90 (P1-1): ホストの GDN 状態をこれから書き換える (遅延分の回復) →
    // デバイス常駐の正本は古くなるので先に捨てる。早めに捨てても失うものは無い
    // (次の実行でホストから再 seed するだけ) ので、分岐の前に 1 回で置く。
    gdn_scan_state_invalidate();
    // s141 (Task B7): D2H の受け皿は層ループの**外**で 1 回だけ作り、層ごとに
    // 使い回す (grow-only)。従来は層ごとに `to_vec` で 2 本確保していた。
    let mut stage = GdnHostStage::new();
    let m = &mut ms[idx].1;
    if m.gdn_pending.is_none() {
        return Some(()); // 遅延なし
    }
    if m.epoch != epoch {
        m.gdn_pending = None;
        return Some(());
    }
    for li in 0..m.gdn.len() {
        let Some(gd) = m.gdn[li].as_ref() else {
            continue; // GDN 層ではない (full attention 層)
        };
        // s141b (Task B7c): ホストが既に最新の層 (gdn_stale = false) は D2H しない。
        // 判定は純関数 `gdn_flush_step` (テストが実物を通す)。
        let (need, aligned) = gdn_flush_step(true, m.gdn_stale[li], target, m.gdn_pos[li]);
        if !need {
            if m.gdn_stale[li] && !aligned {
                // 古い可能性があり、かつ位置が合っていない → 従来どおり不整合扱い
                m.gdn_pending = None;
                return None;
            }
            continue;
        }
        let face = match m.gdn_pos[li] {
            Some(p) if p == target => gd.cur,
            Some(p) if p == target + 1 => 1 - gd.cur,
            _ => {
                // 不整合 (正常系では起こらない — バグのみ)
                m.gdn_pending = None;
                return None;
            }
        };
        let (Some(cb), Some(sb)) = (&gd.conv[face], &gd.ssm[face]) else {
            m.gdn_pending = None;
            return None;
        };
        // s65: 発行元の件数分類 (GDN 状態の回復 = conv + ssm の 2 本)
        // s88 (Step 1): この層の conv + ssm の回復区間を site の時間としても積む
        // (表示専用)。件数は従来どおり 2 件 (conv / ssm の 2 本)。
        let t_site = if crate::prof::d2h_src_on() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        let conv = match d2h_raw(&mut stage.conv, cb.ptr(), cb.bytes() / 4) {
            Some(v) => v,
            None => {
                m.gdn_pending = None;
                return None;
            }
        };
        let ssm = match d2h_raw(&mut stage.ssm, sb.ptr(), sb.bytes() / 4) {
            Some(v) => v,
            None => {
                m.gdn_pending = None;
                return None;
            }
        };
        if let Some(t) = t_site {
            crate::prof::add_d2h_src_ns_only(
                crate::prof::D2H_SRC_GDN_STATE,
                t.elapsed().as_nanos() as u64,
            );
        }
        let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
        if conv_h.len() < conv.len() || ssm_h.len() < ssm.len() {
            m.gdn_pending = None;
            return None;
        }
        conv_h[..conv.len()].copy_from_slice(conv);
        ssm_h[..ssm.len()].copy_from_slice(ssm);
        // s141b (Task B7c): この層のホストは今デバイスと一致した → 遅延フラグを落とす。
        // 落とさないと、次の flush で同じ層を読み直してしまう (無駄な D2H)。
        m.gdn_stale[li] = false;
    }
    m.gdn_pending = None;
    Some(())
}


// ---------------------------------------------------------------------------
// P19-A1: GDN 逐次 AR スキャン (prefill バッチ)
// ---------------------------------------------------------------------------

/// GDN prefill バッチスキャン (レイヤー 1 回のカーネル launch)。
///
/// - 入力: 投影済み qkv/z/beta/alpha の全トークン分行 + レイヤー状態 (conv/ssm)
/// - 出力: y [n × value_dim] (gated RMS norm 済み・out_w 投影前)。状態は in-place 更新。
/// - GDN は bit 一致不要 (CPU 逐次経路と数値は僅かに異なる・契約は決定論のみ)。
/// - GPU 無効/未初期化/実行時エラー時は None (呼び出し側は CPU 逐次にフォールバック)。
/// - 前提: head_k_dim == head_v_dim かつ d_conv == 4 (Qwen3.5 は常に成立)。
/// - A1-r2: n >= GDN_CHUNK_MIN_TOKENS の長 prefill は 3-pass chunked カーネルに
///   ルーティング (chunked は GPU 専用・CPU 経路の契約は不変)。n == 1 は decode
///   入口 (qwen35 の gdn_scan から呼ばれる)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_batch(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
) -> Option<Vec<f32>> {
    gdn_scan_batch_inner(
        qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv, d_conv,
        eps, q_scale, li, kv, false,
    )
}


/// task 154: qwen4exp (出力ゲートが **sigmoid**) の prefill 用スキャン。
///
/// [`gdn_scan_batch`] との違いは 1 点だけ: 単発カーネルを sigmoid 版
/// (`dno_gdn_scan_sig`、task 137 で decode 用に追加済み) にする。
/// **chunked 経路 (silu 固定) は受けない**: n >= [`GDN_CHUNK_MIN_TOKENS`] では
/// `None` を返して呼び出し側が CPU へ落ちる (prefill のチャンクは既定 512 なので
/// 通常は単発経路に入る)。
///
/// 数値: 同じ CPU 参照 (`gdn::forward_batch_core_gate`) を GPU で回すだけで、
/// 加算順はカーネル側の固定順 (t 昇順・i 昇順) なので決定論。CPU とは
/// conv/L2/デルタ則の丸めが揃わず 1 ulp 単位でずれうる (decode で既に既定 on)。
///
/// 前提: `dk == dv && dk <= 256 && d_conv == 4` (`dno_gdn_scan_sig` のレジスタ
/// 配列 `Scol[256]` と、列 thread を q/k/v で共有する実装の制約)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_batch_sig(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
) -> Option<Vec<f32>> {
    // 形状の前提はカーネル側の制約 (受けないときは CPU へ)。
    // 短い列は CPU のほうが速い: 1 回のカーネル起動 + 層状態の書き戻し
    // (10.7 MB/層) が、19 トークン程度の CPU スキャンより高い (lab t154e:
    // next turn の prefill が +5.6% になった)。境界は常駐 mix と同じ 64。
    if !gdn_sig_accepts(n, nk, nv, dk, dv, d_conv) {
        return None;
    }
    gdn_scan_batch_inner(
        qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv, d_conv,
        eps, q_scale, li, kv, true,
    )
}


/// task 154: シグモイド版スキャンが受ける形か (純関数。GPU を呼ばない)。
///
/// - `n < `[`GDN_SIG_MIN_TOKENS`]: 短い列は CPU のほうが速い (1 回のカーネル起動と
///   層状態の書き戻し 10.7 MB/層が、19 トークン程度の CPU スキャンより高い。
///   lab t154e: next turn の prefill が +5.6% になった)。境界は常駐 mix と同じ 64。
/// - `n >= `[`GDN_CHUNK_MIN_TOKENS`]: 長い列は 3-pass の chunked 経路が受ける。
/// - 残りは `dno_gdn_scan_sig` のレジスタ配列 (`Scol[256]`) と列共有の制約。
pub(crate) fn gdn_sig_accepts(n: usize, nk: usize, nv: usize, dk: usize, dv: usize, d_conv: usize) -> bool {
    n >= GDN_SIG_MIN_TOKENS
        && n < GDN_CHUNK_MIN_TOKENS
        && nk != 0
        && nv != 0
        && dk == dv
        && dk <= 256
        && d_conv == 4
}


/// [`gdn_scan_batch`] / [`gdn_scan_batch_sig`] の共通本体 (`sig` でカーネルを選ぶ)。
#[allow(clippy::too_many_arguments)]
fn gdn_scan_batch_inner(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
    sig: bool,
) -> Option<Vec<f32>> {
    if !cuda_enabled() {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, GDN scan using CPU: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    // task 154: sigmoid 版は単発カーネルだけ (chunked は silu 固定)。
    let f_scan = if sig {
        match ctx.f_gdn_scan_sig {
            Some(f) => f,
            // 旧 PTX (sig 未収録) では黙って CPU へ。ラッチはしない。
            None => return None,
        }
    } else {
        ctx.f_gdn
    };
    // A1-r2: 長 prefill は chunked 3 カーネルへ。chunked は dk==dv / d_conv==4
    // 前提のカーネルのため、非成立時は従来の単発カーネルにフォールバック。
    let chunked = !sig && n >= GDN_CHUNK_MIN_TOKENS && dk == dv && d_conv == 4;
    let result = if chunked {
        gdn_scan_chunked_impl(
            ctx,
            GdnScanSrc::Host { qkv, z, beta, alpha },
            conv_st,
            ssm_st,
            wconv,
            wa,
            wdt,
            wnorm,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            None,
            li,
            kv,
        )
    } else {
        gdn_scan_impl(
            ctx, f_scan, sig, qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk,
            nv, dk, dv, d_conv, eps, q_scale, li, kv,
        )
    };
    match result {
        Ok(y) => Some(y),
        Err(e) => {
            tracing::error!("CUDA GDN scan failed, using CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}


/// task 154: シグモイド版スキャンを受ける最小トークン数。
///
/// これ未満は CPU スキャンのほうが速い (起動 + 層状態の D2H が固定費)。
pub(crate) const GDN_SIG_MIN_TOKENS: usize = 64;


/// task 162: シグモイド版スキャンを 1 回の起動で回す最大トークン数
/// (`DENNO_PF_GDN_SUB`、既定 1024)。
///
/// `dno_gdn_scan_sig` は `n < GDN_CHUNK_MIN_TOKENS (2048)` の単発経路しか無い
/// (chunked は silu 固定) ので、prefill のチャンクを 2048 以上にすると CPU
/// スキャンへ落ちる (lab t162a の `pchunk`: ssm の `scan` が 9.4 s)。ここで
/// 2048 未満に切って順に呼ぶ。状態はカーネルが更新して毎回ホストへ戻るので、
/// 切っても値は 1 ビットも変わらない (トークン順の逐次は同じ)。
/// task 162: 単発スキャン (`dno_gdn_scan_sig`) が受けられる n の上限。
///
/// これ以上は chunked 経路 (silu 固定) の担当で、sigmoid 版は受けない。
pub fn gdn_sub_limit() -> usize {
    GDN_CHUNK_MIN_TOKENS
}


/// task 162: シグモイド版スキャンの最小トークン数 (これ未満は CPU のほうが速い)。
pub fn gdn_sig_min() -> usize {
    GDN_SIG_MIN_TOKENS
}


/// task 162: `nt` トークンのスキャンを何回に切るか (純関数)。
///
/// - `nt < gdn_sub_limit()` なら 1 回 (従来どおり)。
/// - それ以上は `gdn_sig_sub_chunk()` ずつ。**末尾が最小トークン数
///   ([`gdn_sig_min`]) を切る場合だけ**、最後の 2 つを併合する (小さい n は
///   カーネルが受けず `None` になるため)。
///
/// 返り値は `(start, end)` の列で、連続・重複なし・合計 `nt`。
pub fn gdn_sub_bounds(nt: usize) -> Vec<(usize, usize)> {
    if nt == 0 {
        return Vec::new();
    }
    // `dno_gdn_scan_sig` は n < GDN_CHUNK_MIN_TOKENS のときだけ受ける。
    if nt < gdn_sub_limit() {
        return vec![(0, nt)];
    }
    let sub = gdn_sig_sub_chunk().clamp(1, gdn_sub_limit() - 1);
    let min = gdn_sig_min();
    let mut out = Vec::new();
    let mut t0 = 0usize;
    while t0 < nt {
        let left = nt - t0;
        let end = if left <= sub {
            nt
        } else if left - sub < min {
            // 端数が小さすぎる → この回で飲み込む (sub + min <= 上限)。
            nt
        } else {
            t0 + sub
        };
        out.push((t0, end));
        t0 = end;
    }
    out
}


pub fn gdn_sig_sub_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DENNO_PF_GDN_SUB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|n| n.clamp(GDN_SIG_MIN_TOKENS, GDN_CHUNK_MIN_TOKENS - 1))
            .unwrap_or(1024)
    })
}


/// 実装 (同期コピー + 1 launch + 1 同期。prefill はリクエストあたり層数回しか
/// 呼ばれないためピン留めパイプライン化の恩恵は小さい)。
#[allow(clippy::too_many_arguments)]
fn gdn_scan_impl(
    ctx: &CudaCtx,
    f_scan: CUfunction,
    force_writeback: bool,
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
) -> Result<Vec<f32>> {
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    debug_assert_eq!(qkv.len(), n * conv_dim);
    debug_assert_eq!(z.len(), n * vd);
    debug_assert_eq!(beta.len(), n * nv);
    debug_assert_eq!(alpha.len(), n * nv);
    debug_assert_eq!(conv_st.len(), conv_dim * cs);
    debug_assert_eq!(ssm_st.len(), nv * dk * dv);

    ctx.make_current()?;
    // s64 (T5b L1 Step 1 計測): この関数の全体は呼び出し側 (qwen35.rs) の
    // `time_layer_n_ev(ATT_GDN_SCAN)` = `attn_gdn` / `gdn_scan` バケットで計時される。
    // ここではスキャン 1 call の中身を「確保 + H2D / launch / sync / D2H」の 4 区間に
    // 割り、既存バケットを壊さずに **表示専用の細分** (`D2H_GDN_PF_ALLOC` ..
    // `D2H_GDN_PF_D2H`) へ積む。各区間の経過時間は次の区間の入口で読むので、
    // 細分の合計≒外側バケット値になる。計測器は `prof::Seq` (prof off は素透し
    // で時刻読みすらしない = 数値経路・実行時間とも不変)。
    //
    // 区間の切り方: 確保 + 入力 H2D (重み 4 本を含む) を 1 区間にまとめた理由は、
    // 重み (`wconv` 等) が層ごとに固定で `gdn_scan_batch` の引数にはバッファが
    // 無い (= 呼び出しごとに H2D し直している) ことを確定させるため。
    let mut q = crate::prof::Seq::begin(crate::prof::D2H_GDN_PF_ALLOC, Instant::now());
    // s128 (decode): 13 本の作業領域を chunked 経路と同じ thread_local (GDN_SCRATCH)
    // から借りる。従来は層ごと・token ごとに 13 本を確保して解放しており、
    // 27B の decode で 1 token あたり約 38 ms (48 層 x 0.80 ms) を捨てていた。
    // 中身は従来どおり copy_from で全要素を上書きするので状態は同一。
    let dp = GDN_SCRATCH.with(|cell| -> Result<GdnPtrs> {
        let mut s = cell.borrow_mut();
        s.ensure(n, conv_dim, vd, nv, dk, dv, cs, 1)?;
        // conv_out / ssm_out は「書かれない領域」を残さないため従来どおりゼロで初期化
        s.buf(GdnBuf::ConvOut).fill_zero()?;
        s.buf(GdnBuf::SsmOut).fill_zero()?;
        s.q.as_ref().unwrap().copy_from(qkv)?;
        s.z.as_ref().unwrap().copy_from(z)?;
        s.beta.as_ref().unwrap().copy_from(beta)?;
        s.alpha.as_ref().unwrap().copy_from(alpha)?;
        gdn_put_weights(&s, li, [wconv, wa, wdt, wnorm])?;
        Ok(s.ptrs())
    })?;
    // s131: conv / ssm の正本はデバイス。未確定のときだけホストから 1 回 seed。
    let (pst_conv, pst_ssm) =
        gdn_state_begin(kv, li, conv_st, ssm_st, conv_dim * cs, nv, dk * dv)?;
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    let (mut pq, mut pz, mut pb, mut pa) = (dp.q, dp.z, dp.beta, dp.alpha);
    let (mut pci, mut psi) = (pst_conv, pst_ssm);
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (dp.wc, dp.wa, dp.wd, dp.wn);
    let (mut py, mut pco, mut pso) = (dp.y, dp.conv_out, dp.ssm_out);
    let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
        n as c_int, nk as c_int, nv as c_int, dk as c_int, dv as c_int,
    );
    let (mut cdi, mut dci, mut csi) = (conv_dim as c_int, d_conv as c_int, cs as c_int);
    let (mut epsi, mut qsi) = (eps, q_scale);
    let mut args = [
        &mut pq as *mut _ as *mut c_void,
        &mut pz as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut psi as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pwn as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut dci as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
        &mut qsi as *mut _ as *mut c_void,
    ];
    let shmem = ((3 * dk + dv + 5) * 4) as u32;
    unsafe {
        ctx.launch_async(f_scan, (nv as u32, 1, 1), (dk as u32, 1, 1), shmem, &mut args)?;
    }
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);

    let y = GDN_SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let s = cell.borrow();
        s.buf(GdnBuf::Y).to_vec::<f32>(n * vd)
    })?;
    // s131: 状態はホストへ落とさずデバイス正本へ D2D で戻す。
    gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
    // task 154: prefill の sigmoid 経路は毎回ホストへ書き戻す。prefill のあとに
    // decode (別ストア) や CPU 経路がホスト state を seed にするので、デバイス正本の
    // ままだと 1 チャンク前の状態から走り出す。10.7 MB/層チャンクの D2H を払う。
    if force_writeback || !gdn_devstate() {
        gdn_host_writeback(conv_st, ssm_st)?;
    }
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    q.end();
    Ok(y)
}


/// 連鎖の作業域 (thread_local, grow-only)。`x`/`o` は [nt, e]、
/// `qkv` は [nt, conv_dim]、`z` と `y` は [nt, vd]、`beta`/`alpha` は [nt, nv]。
pub(crate) struct GdnPfChainBufs {
    x: Option<DevBuffer>,
    o: Option<DevBuffer>,
    qkv: Option<DevBuffer>,
    z: Option<DevBuffer>,
    y: Option<DevBuffer>,
    beta: Option<DevBuffer>,
    alpha: Option<DevBuffer>,
    nt: usize,
    e: usize,
    conv_dim: usize,
    vd: usize,
    nv: usize,
}


impl GdnPfChainBufs {
    pub(crate) const fn empty() -> Self {
        Self {
            x: None,
            o: None,
            qkv: None,
            z: None,
            y: None,
            beta: None,
            alpha: None,
            nt: 0,
            e: 0,
            conv_dim: 0,
            vd: 0,
            nv: 0,
        }
    }
    /// 形が変わったら全バッファを grow し直す (prefill の 1 リクエストは
    /// 同じ形のチャンクが続くので、2 チャンク目以降は何もしない)。
    pub(crate) fn ensure(
        &mut self,
        nt: usize,
        e: usize,
        conv_dim: usize,
        vd: usize,
        nv: usize,
    ) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        if self.nt >= nt
            && self.e == e
            && self.conv_dim == conv_dim
            && self.vd == vd
            && self.nv == nv
            && self.x.is_some()
        {
            return Ok(());
        }
        let grow = |cap: &mut Option<DevBuffer>, bytes: usize| -> Result<()> {
            if cap.as_ref().map(|b| b.bytes()).unwrap_or(0) < bytes {
                *cap = Some(DevBuffer::alloc(bytes.max(1))?);
            }
            Ok(())
        };
        grow(&mut self.x, nt * e * F)?;
        grow(&mut self.o, nt * e * F)?;
        grow(&mut self.qkv, nt * conv_dim * F)?;
        grow(&mut self.z, nt * vd * F)?;
        grow(&mut self.y, nt * vd * F)?;
        grow(&mut self.beta, nt * nv * F)?;
        grow(&mut self.alpha, nt * nv * F)?;
        self.nt = nt;
        self.e = e;
        self.conv_dim = conv_dim;
        self.vd = vd;
        self.nv = nv;
        Ok(())
    }
    pub(crate) fn buf(&self, which: usize) -> CUdeviceptr {
        let b = match which {
            0 => &self.x,
            1 => &self.o,
            2 => &self.qkv,
            3 => &self.z,
            4 => &self.y,
            5 => &self.beta,
            _ => &self.alpha,
        };
        b.as_ref().expect("GdnPfChainBufs: 未確保").ptr()
    }
}


thread_local! {
    pub(crate) static GDN_PF_CHAIN_BUF: std::cell::RefCell<GdnPfChainBufs> =
        const { std::cell::RefCell::new(GdnPfChainBufs::empty()) };
}


/// task 167: 連鎖の前提 (GPU を触らない純関数)。false なら従来経路へ。
///
/// 形は「ssm 層 1 本ぶんの投影 3 本とスキャン」がそのままつながることを要求する:
/// qkv: [e → 2*nk*dk + nv*dv] / gate: [e → nv*dv] / out: [nv*dv → e]。
#[allow(clippy::too_many_arguments)]
pub fn gdn_pf_chain_ready(
    wq: &PfChainQkv<'_>,
    wout: &PfChainW<'_>,
    nt: usize,
    e: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> bool {
    cuda_enabled() && gdn_pf_chain_shape_ok(wq, wout, nt, e, nk, nv, dk, dv, d_conv)
}


/// [`gdn_pf_chain_ready`] の形だけの判定 (GPU を触らない純関数)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_pf_chain_shape_ok(
    wq: &PfChainQkv<'_>,
    wout: &PfChainW<'_>,
    nt: usize,
    e: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> bool {
    use crate::kquant::GgmlType as T;
    if nt == 0 || e == 0 || nk == 0 || nv == 0 {
        return false;
    }
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    // 単発の sigmoid スキャンが受ける範囲 (decode と同じ境界) と、task 270 の
    // 大きい nt (`DENNO_PF_CHAIN_BIG` + 上限の無いスキャンの形) の判定。
    // task 197 ではここが 2048 で切れて連鎖が外れ、ssm が 3.96 → 24.7 s に落ちた
    // (チャンク 2048 が +39.6%)。v1/v2/v3 のスキャンには長さの上限が無い。
    if !chain_nt_ok(nt, nk, nv, dk, dv, d_conv, conv_dim, chain_big_on()) {
        return false;
    }
    if dk != dv || dk > 256 || d_conv != 4 {
        return false;
    }
    let ok_ty = |w: &PfChainW<'_>| matches!(w.ty, T::Q8_0 | T::Q5_1 | T::Q4K | T::Q5K);
    let pair_ok = if wq.is_fused() {
        // 連結済み: 1 本の行列の行範囲として qkv と gate が並んでいること。
        wq.qkv.k == e
            && wq.qkv.n == conv_dim + vd
            && wq.gate.ty == wq.qkv.ty
            && wq.gate.k == e
            && wq.gate.n == conv_dim + vd
            && wq.qkv_row + conv_dim <= wq.qkv.n
            && wq.gate_row + vd <= wq.gate.n
            && wq.qkv_row != wq.gate_row
    } else {
        // 別々: それぞれが単独の行列で、行オフセットは無い。
        wq.qkv.k == e
            && wq.qkv.n == conv_dim
            && wq.gate.k == e
            && wq.gate.n == vd
            && wq.qkv_row == 0
            && wq.gate_row == 0
    };
    pair_ok && wout.k == vd && wout.n == e && ok_ty(&wq.qkv) && ok_ty(&wq.gate) && ok_ty(wout)
}


/// task 167: デバイス入出力の GGUF GEMM を 1 発だけ撃つ (コピーもしない)。
/// カーネル・引数・グリッドは [`CudaCtx::moe_gemm_dev`] と同一。
// ---------------------------------------------------------------------------
// task 407: GDN 連鎖の投影 GEMM を br169 (tc, bf16 テンソルコア) 側で撃つ
// (`DENNO_PF_GDN_TC1`、**lab t407c で採ったので既定 on**)
// ---------------------------------------------------------------------------
//
// 既定モデル (UD-IQ3_XXS) の ssm 層の投影は Q8_0 で、`gguf_gemm_dev_launch` は
// まず v2 カーネル (tc2, 128x128 タイル・2 段プリフェッチ) を選ぶ。lab t407c で
// `DENNO_PF_TC_GEMM2=0` にして br169 (tc, 128x64) へ落とすと、同じ run の対で
// 3,826 cold の `ssm` busy が **371.4 / 371.2 -> 360.7 / 360.8 (-2.9 / -2.8%)**、
// 内訳は qkv_gemm 71.9 -> 66.6 ms (-7.4%)、z_gemm 46.1 -> 40.7 ms (-11.7%)、
// out_gemm 43.1 -> 43.2 (不変)、scan 193.5 -> 193.5 (不変)。`full` busy も
// 88.9 -> 87.1 と一緒に下がった。decode 15.5 -> 15.4、**md5 は全 COND SAME**、
// errors 0、text sane。14k は qkv 251.2 -> 243.9 / z 158.6 -> 152.7 で、
// ssm busy は 1,330.0 -> 1,318.0 (-0.9%。
//
// task 176 は tc2 を「qkv_pf 6,268 -> 579 ms」で採っているが、それは旧既定
// モデル (swift15-abl-IQ3_S) の測定で、いまの既定モデルでは投影が Q8_0・
// 形も違う。**モデル全体の既定は動かさず** (`DENNO_PF_TC_GEMM2` は既定 on の
// まま)、GDN 連鎖の投影 2 本だけこのゲートで tc に落とす。
pub(crate) fn gdn_pf_tc1_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


#[cfg(test)]
static GDN_PF_TC1_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_gdn_pf_tc1_for_test(v: Option<bool>) {
    GDN_PF_TC1_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// GDN 連鎖の投影を tc (br169) 側で撃つか (**既定 on**、`=0` で tc2 に戻る)。
pub(crate) fn gdn_pf_tc1_on() -> bool {
    #[cfg(test)]
    {
        match GDN_PF_TC1_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_tc1_from_value(std::env::var("DENNO_PF_GDN_TC1").ok().as_deref()))
}


// ---------------------------------------------------------------------------
// task 173: prefill の GDN スキャンを「状態レジスタ常駐 + 4 レーン分割」へ
// ---------------------------------------------------------------------------
//
// 既存 `dno_gdn_scan_sig` (gdn.cu) は 1 ブロック = dk スレッドで、状態の列
// (dk = 128 float = 1 KiB) をスレッドローカルの配列に持つ。作業集合 128 KiB が
// L1 に収まらず毎トークン L2 へ落ちるため、lab t169b の実測は 10.7 us/token
// (1 層 1 チャンク 512 トークンで 5.5 ms = 14k prefill の 5.5 s)。
//
// `kernels_gdn_pf.cu` の `dno_gdn_pf_scan128` は同じ演算を 512 スレッド
// (128 列 × 4 レーン) で回し、状態をレジスタに置く (ローカルメモリ往復なし)。
// 要素ごとの更新は同一の丸め、縮約 (q/k ノルム・skv・o・出力 RMS) だけが
// 4 本の部分和の結合になる (丸め順の差のみ)。
//
// PTX: `bash build_ptx.sh gdn_pf` (lab は `PTX=gdn_pf` で再生成する)。
// このコンテナには nvcc が無いので生成は GPU 側。

/// 新カーネルの 1 ヘッド次元 (`kernels_gdn_pf.cu` の PF_DK と一致)。
const GDN_PF_DK: usize = 128;

/// 新カーネルのスレッド数 (`PF_DK * PF_LANES` = 128 * 4)。
const GDN_PF_SCAN_THREADS: usize = 512;


/// `DENNO_GDN_PF_FAST` の値の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// 既定 on: lab t174b (SHA 99807fa7d、新ホスト、SCENARIO=depth8k、PROFILE=1) で採用。
/// cold prefill 3,826 tok 20.55 -> 19.74 s (-4.0%)、14,018 tok 71.70 -> 66.80 s (-6.8%)、
/// decode は 1 トークンあたりの合計で -0.4 〜 +0.4% (median の ±1 ms は量子化)、
/// base COND は errors 0 のまま。md5 DIFF は縮約順の差 (仕様)。
pub(crate) fn gdn_pf_fast_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 新しいスキャンカーネルを使うか (既定 on)。
pub(crate) fn gdn_pf_fast_on() -> bool {
    #[cfg(test)]
    {
        match PF_FAST_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_fast_from_value(std::env::var("DENNO_GDN_PF_FAST").ok().as_deref()))
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf_fast_for_test(v: Option<bool>) {
    PF_FAST_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// 新しいスキャンカーネルが受ける形か (純関数。カーネルの前提と同じ)。
pub(crate) fn gdn_pf_fast_eligible(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    nt >= 1
        && nk >= 1
        && nv >= 1
        && dk == GDN_PF_DK
        && dv == GDN_PF_DK
        && d_conv == 4
        && conv_dim == 2 * nk * dk + nv * dv
}


#[cfg(test)]
mod gdn_pf_fast_tests {
    use super::*;

    /// ゲートの真理値表 (既定は on。lab t174b で採用)。
    #[test]
    fn gate_truth_table() {
        assert!(gdn_pf_fast_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!gdn_pf_fast_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", "anything"] {
            assert!(gdn_pf_fast_from_value(Some(v)), "{v}");
        }
    }

    /// 形のゲート: 実モデル (nk=16 / nv=48 / dk=dv=128) だけを受ける。
    #[test]
    fn shape_gate() {
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert_eq!(conv_dim, 10240);
        assert!(gdn_pf_fast_eligible(1, 16, 48, 128, 128, 4, conv_dim));
        assert!(gdn_pf_fast_eligible(512, 16, 48, 128, 128, 4, conv_dim));
        assert!(gdn_pf_fast_eligible(2048, 16, 48, 128, 128, 4, conv_dim));
        // dk != 128 / d_conv != 4 / conv_dim 不一致 / nt=0 は受けない。
        assert!(!gdn_pf_fast_eligible(512, 16, 48, 64, 64, 4, 2 * 16 * 64 + 48 * 64));
        assert!(!gdn_pf_fast_eligible(512, 16, 48, 128, 128, 3, conv_dim));
        assert!(!gdn_pf_fast_eligible(512, 16, 48, 128, 128, 4, conv_dim + 1));
        assert!(!gdn_pf_fast_eligible(0, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf_fast_eligible(512, 0, 48, 128, 128, 4, conv_dim));
    }

    /// テスト用の強制スイッチが env 解決を上書きする (戻し忘れない)。
    #[test]
    fn force_switch_overrides_env() {
        set_gdn_pf_fast_for_test(Some(true));
        assert!(gdn_pf_fast_on());
        set_gdn_pf_fast_for_test(Some(false));
        assert!(!gdn_pf_fast_on());
        set_gdn_pf_fast_for_test(None);
    }
}


/// 新しいスキャンカーネルのモジュール (PTX は実行時読み込み)。
pub(crate) struct GdnPfKernels {
    #[allow(dead_code)]
    module: CUmodule,
    scan128: CUfunction,
}


unsafe impl Send for GdnPfKernels {}

unsafe impl Sync for GdnPfKernels {}


static GDN_PF_K: OnceLock<Option<GdnPfKernels>> = OnceLock::new();


/// PTX の置き場所。`DENNO_GDN_PF_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gdn_pf.ptx`。
fn gdn_pf_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GDN_PF_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn gdn_pf_kernels() -> Option<&'static GdnPfKernels> {
    GDN_PF_K
        .get_or_init(|| {
            let path = gdn_pf_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[gdn-pf] PTX が読めない ({}) : {e} → 従来のスキャンへ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0);
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[gdn-pf] cuModuleLoadData 失敗 → 従来のスキャンへ");
                    return None;
                }
                let mut scan128: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(
                    &mut scan128,
                    module,
                    b"dno_gdn_pf_scan128\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
                {
                    eprintln!("[gdn-pf] PTX に dno_gdn_pf_scan128 が無い → 従来のスキャンへ");
                    return None;
                }
                log_kernel_attr(lib, scan128, "dno_gdn_pf_scan128");
                Some(GdnPfKernels { module, scan128 })
            }
        })
        .as_ref()
}


/// task 173: レジスタ常駐スキャンを 1 回撃つ (引数はカーネルの並び)。
///
/// 起動サイトは `fns.scan128` なので、kernel_invariants の起動サイト表が
/// カーネル名 (`dno_gdn_pf_scan128`) と block 形状 (512) に解決できる。
#[allow(clippy::too_many_arguments)]
fn gdn_pf_scan128_launch(
    ctx: &CudaCtx,
    fns: &GdnPfKernels,
    qkv: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_st_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wconv: CUdeviceptr,
    wa: CUdeviceptr,
    wdt: CUdeviceptr,
    wnorm: CUdeviceptr,
    y_out: CUdeviceptr,
    conv_st_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    n: usize,
    nk: usize,
    nv: usize,
    conv_dim: usize,
    eps: f32,
    q_scale: f32,
) -> Result<()> {
    let (mut pq, mut pzz, mut pbb, mut paa) = (qkv, z, beta, alpha);
    let (mut pci, mut psi) = (conv_st_in, ssm_in);
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (wconv, wa, wdt, wnorm);
    let (mut pyy, mut pco, mut pso) = (y_out, conv_st_out, ssm_out);
    let (mut ni, mut nki, mut nvi) = (n as c_int, nk as c_int, nv as c_int);
    let (mut cdi, mut epsi, mut qsi) = (conv_dim as c_int, eps, q_scale);
    let mut args = [
        &mut pq as *mut _ as *mut c_void,
        &mut pzz as *mut _ as *mut c_void,
        &mut pbb as *mut _ as *mut c_void,
        &mut paa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut psi as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pwn as *mut _ as *mut c_void,
        &mut pyy as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
        &mut qsi as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            fns.scan128,
            (nv as u32, 1, 1),
            (GDN_PF_SCAN_THREADS as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


// ---------------------------------------------------------------------------
// task 186: prefill GDN スキャン v2 (`kernels_gdn_pf2.cu`、DENNO_PF_SCAN_FAST)
// ---------------------------------------------------------------------------
//
// `dno_gdn_pf_scan128` は 1 ブロック = 1 ヘッド (128 列) で 48 ブロックしかなく、
// 1 トークンあたり barrier 3 回と単一スレッドの超越関数を払う。lab t183a の実測は
// 5.53 us/トークン (16.3 GB/s, 0.85 TFLOP/s)。v2 は
//   - 1 ブロック = 64 列 (2 ブロックで 1 ヘッド) → 96 ブロック
//   - conv の窓をレーンごとに 1 チャネルずつ分散し、3 段リングで 2 トークン先を
//     先に埋める (barrier は 1 トークン 1 回)
//   - 出力の gated RMS norm は別カーネル `dno_gdn_pf_norm128`
// で回す。値は `dno_gdn_pf_scan128` とビット一致 (同じ状態の持ち方・同じ縮約の木・
// 同じ式) なので、採否は速度だけで決まる。PTX は
// `kernels/kernels_gdn_pf2.ptx` を実行時に読む (無ければ従来へ fail closed)。

/// v2 スキャンのスレッド数 (カーネルの PF2_THREADS)。
pub(crate) const GDN_PF2_THREADS: usize = 256;

/// v2 スキャンの 1 ブロックあたりの列数 (カーネルの PF2_COLS)。
pub(crate) const GDN_PF2_COLS: usize = 64;

/// RMS norm カーネルのスレッド数 (カーネルの PF2_NORM_THREADS)。
pub(crate) const GDN_PF2_NORM_THREADS: usize = 128;


/// task 186: `kernels_gdn_pf2.cu` の関数一式。
pub(crate) struct GdnPf2Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    scan64: CUfunction,
    norm: CUfunction,
}


// 生ハンドル (GdnPfKernels と同じ規約)。
unsafe impl Send for GdnPf2Kernels {}

unsafe impl Sync for GdnPf2Kernels {}


/// PTX の置き場所。`DENNO_GDN_PF2_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gdn_pf2.ptx`。
fn gdn_pf2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GDN_PF2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf2.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn gdn_pf2_kernels() -> Option<&'static GdnPf2Kernels> {
    static K: OnceLock<Option<GdnPf2Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = gdn_pf2_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[gdn-pf2] PTX が読めない ({}) : {e} → 従来のスキャンへ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[gdn-pf2] cuModuleLoadData 失敗 → 従来のスキャンへ");
                return None;
            }
            let mut scan64: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut scan64,
                module,
                b"dno_gdn_pf_scan64\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[gdn-pf2] PTX に dno_gdn_pf_scan64 が無い → 従来のスキャンへ");
                return None;
            }
            let mut norm: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut norm,
                module,
                b"dno_gdn_pf_norm128\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[gdn-pf2] PTX に dno_gdn_pf_norm128 が無い → 従来のスキャンへ");
                return None;
            }
            log_kernel_attr(lib, scan64, "dno_gdn_pf_scan64");
            log_kernel_attr(lib, norm, "dno_gdn_pf_norm128");
            Some(GdnPf2Kernels {
                module,
                scan64,
                norm,
            })
        }
    })
    .as_ref()
}


/// `DENNO_PF_SCAN_FAST` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// task 189: **既定 on**。lab t187c で scan_pf 2,853 -> 1,295 ms (md5 SAME)、
/// wall -3.2% を確認済み (task 186 では目標未達だったため off のままだった)。
/// `DENNO_PF_SCAN_FAST=0` で従来の `gdn_pf` スキャンに戻る。
pub(crate) fn gdn_pf2_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf2_for_test(v: Option<bool>) {
    PF2_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// v2 スキャンを使うか (既定 on。`DENNO_PF_SCAN_FAST=0` で off)。
pub(crate) fn gdn_pf2_on() -> bool {
    #[cfg(test)]
    {
        match PF2_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf2_from_value(std::env::var("DENNO_PF_SCAN_FAST").ok().as_deref()))
}


/// v2 スキャンが受ける形か (純関数。カーネルの前提と同じ)。
pub(crate) fn gdn_pf2_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    nt >= 1
        && nk >= 1
        && nv >= 1
        && dk == GDN_PF_DK
        && dv == GDN_PF_DK
        && d_conv == 4
        && conv_dim == 2 * nk * dk + nv * dv
}


/// task 186: v2 スキャンを撃つ (scan64 → norm128 の 2 起動)。引数の並びは
/// カーネルの宣言順。`o_out` と `y_out` は同じバッファでよい (norm は in-place)。
#[allow(clippy::too_many_arguments)]
fn gdn_pf2_launch(
    ctx: &CudaCtx,
    fns: &GdnPf2Kernels,
    qkv: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_st_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wconv: CUdeviceptr,
    wa: CUdeviceptr,
    wdt: CUdeviceptr,
    wnorm: CUdeviceptr,
    y_out: CUdeviceptr,
    conv_st_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    n: usize,
    nk: usize,
    nv: usize,
    conv_dim: usize,
    eps: f32,
    q_scale: f32,
) -> Result<()> {
    {
        let (mut pq, mut pb, mut pa) = (qkv, beta, alpha);
        let (mut pci, mut psi) = (conv_st_in, ssm_in);
        let (mut pwc, mut pwa, mut pwd) = (wconv, wa, wdt);
        let (mut py, mut pco, mut pso) = (y_out, conv_st_out, ssm_out);
        let (mut ni, mut nki, mut nvi) = (n as c_int, nk as c_int, nv as c_int);
        let (mut cdi, mut epsi, mut qsi) = (conv_dim as c_int, eps, q_scale);
        let mut args = [
            &mut pq as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut pci as *mut _ as *mut c_void,
            &mut psi as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void,
            &mut pwa as *mut _ as *mut c_void,
            &mut pwd as *mut _ as *mut c_void,
            &mut py as *mut _ as *mut c_void,
            &mut pco as *mut _ as *mut c_void,
            &mut pso as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nki as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut cdi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
            &mut qsi as *mut _ as *mut c_void,
        ];
        // 1 ヘッドを 2 ブロックが分担する (64 列ずつ)。
        unsafe {
            ctx.launch_async(
                fns.scan64,
                (nv as u32, 2, 1),
                (GDN_PF2_THREADS as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    {
        let (mut py, mut pz, mut pwn) = (y_out, z, wnorm);
        let (mut ni, mut nvi) = (n as c_int, nv as c_int);
        let mut epsi = eps;
        let mut args = [
            &mut py as *mut _ as *mut c_void,
            &mut pz as *mut _ as *mut c_void,
            &mut pwn as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
        ];
        let groups = n.saturating_mul(nv);
        let per_block = GDN_PF2_NORM_THREADS / 4;
        let grid = (groups.div_ceil(per_block)) as u32;
        unsafe {
            ctx.launch_async(
                fns.norm,
                (grid, 1, 1),
                (GDN_PF2_NORM_THREADS as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    Ok(())
}


#[cfg(test)]
mod gdn_pf2_tests {
    use super::*;

    /// ゲートの真理値表 (既定 off。lab の結果で on にする)。
    #[test]
    fn gdn_pf2_gate_truth_table() {
        assert!(gdn_pf2_from_value(None), "未設定は既定 on (task 189)");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!gdn_pf2_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf2_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    /// 形のゲートは scan128 と同じ前提 (dk=dv=128 / d_conv=4 / conv_dim 一致)。
    #[test]
    fn gdn_pf2_ok_checks_shape() {
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf2_ok(1, 16, 48, 128, 128, 4, conv_dim));
        assert!(gdn_pf2_ok(512, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf2_ok(512, 16, 48, 64, 64, 4, 2 * 16 * 64 + 48 * 64));
        assert!(!gdn_pf2_ok(512, 16, 48, 128, 128, 3, conv_dim));
        assert!(!gdn_pf2_ok(512, 16, 48, 128, 128, 4, conv_dim + 1));
        assert!(!gdn_pf2_ok(0, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf2_ok(512, 0, 48, 128, 128, 4, conv_dim));
    }

    /// 強制スイッチ (テスト用) が効く。
    #[test]
    fn gdn_pf2_force_switch() {
        set_gdn_pf2_for_test(Some(true));
        assert!(gdn_pf2_on());
        set_gdn_pf2_for_test(Some(false));
        assert!(!gdn_pf2_on());
        set_gdn_pf2_for_test(None);
        // task 189: 既定を on に反転した (lab t187c: scan_pf 2,853 -> 1,295 ms)。
        assert!(gdn_pf2_on(), "env 未設定の既定は on");
    }
}


// ---------------------------------------------------------------------------
// task 225: prefill GDN スキャン v3 (ヘッドあたり 4 ブロック x 8 レーン)
// (`DENNO_PF_SCAN3`、task 228 から既定 on)
// ---------------------------------------------------------------------------
//
// lab t222c: `ssm scan_pf` は 1,470.3 ms = 2.9 us/トークン。v2
// (`dno_gdn_pf_scan64`) は 1 ブロック = 64 列で 96 ブロックしかなく (170 SM)、
// 8 ワープ/ブロックなので 4.5 ワープ/SM。1 トークンの依存鎖 (32 行の skv 和 →
// 32 行の更新 + o 和) が latency にそのまま出る。
// v3 は 1 ブロック = 32 列 x 8 レーン = 256 スレッド (1 レーン 16 行) で
// 192 ブロック。リングを 4 段にして t+3 のグローバル読みを t の計算の前に発行し、
// silu とリング書きを計算の後に回す。値は v2 と丸め順だけが違う (縮約の木が
// 8 レーン)。

/// v3 スキャンの 1 ブロックあたりの列数 (カーネルの PF3_COLS)。
pub(crate) const GDN_PF3_COLS: usize = 32;

/// v3 スキャンのスレッド数 (カーネルの PF3_THREADS)。
pub(crate) const GDN_PF3_THREADS: usize = 256;

/// v3 スキャンが 1 ヘッドを割るブロック数 (128 列 / 32 列)。
pub(crate) const GDN_PF3_BLOCKS: u32 = 4;


pub(crate) struct GdnPf3Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    scan32: CUfunction,
    /// 出力の gated RMS norm は v2 の PTX のものを使う (同じ演算列)。
    norm: CUfunction,
}


unsafe impl Send for GdnPf3Kernels {}

unsafe impl Sync for GdnPf3Kernels {}


/// PTX の置き場所 (`DENNO_GDN_PF3_PTX` があればそれ)。
fn gdn_pf3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GDN_PF3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf3.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。norm は v2 の PTX から取る。
fn gdn_pf3_kernels() -> Option<&'static GdnPf3Kernels> {
    static K: OnceLock<Option<GdnPf3Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = gdn_pf3_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[gdn-pf3] PTX が読めない ({}) : {e} → v2 のスキャンへ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[gdn-pf3] cuModuleLoadData 失敗 → v2 のスキャンへ");
                return None;
            }
            let mut scan32: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut scan32,
                module,
                b"dno_gdn_pf_scan32\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[gdn-pf3] PTX に dno_gdn_pf_scan32 が無い → v2 のスキャンへ");
                return None;
            }
            log_kernel_attr(lib, scan32, "dno_gdn_pf_scan32");
            // 出力の norm は v2 の PTX (同じ演算列) をそのまま使う。
            let norm = gdn_pf2_kernels()?.norm;
            Some(GdnPf3Kernels {
                module,
                scan32,
                norm,
            })
        }
    })
    .as_ref()
}


/// `DENNO_PF_SCAN3` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn gdn_pf3_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        // task 228: task 225 の lab t225c で scan_pf 1,469.8 → 1,310.4 ms (14k) /
        // decode・text 無影響の実測が出たので既定 on。切り戻しは `DENNO_PF_SCAN3=0`。
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf3_for_test(v: Option<bool>) {
    PF3_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// v3 スキャンを使うか (task 228 から既定 on)。
pub(crate) fn gdn_pf3_on() -> bool {
    #[cfg(test)]
    {
        match PF3_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf3_from_value(std::env::var("DENNO_PF_SCAN3").ok().as_deref()))
}


/// v3 スキャンが受ける形か (純関数。カーネルの前提と同じ)。
pub(crate) fn gdn_pf3_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    gdn_pf2_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


/// task 225: v3 スキャンを撃つ (scan32 → v2 の norm の 2 起動)。引数の並びは
/// カーネルの宣言順で、v2 の [`gdn_pf2_launch`] と同じ。
#[allow(clippy::too_many_arguments)]
fn gdn_pf3_launch(
    ctx: &CudaCtx,
    fns: &GdnPf3Kernels,
    qkv: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_st_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wconv: CUdeviceptr,
    wa: CUdeviceptr,
    wdt: CUdeviceptr,
    wnorm: CUdeviceptr,
    y_out: CUdeviceptr,
    conv_st_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    n: usize,
    nk: usize,
    nv: usize,
    conv_dim: usize,
    eps: f32,
    q_scale: f32,
) -> Result<()> {
    {
        let (mut pq, mut pb, mut pa) = (qkv, beta, alpha);
        let (mut pci, mut psi) = (conv_st_in, ssm_in);
        let (mut pwc, mut pwa, mut pwd) = (wconv, wa, wdt);
        let (mut py, mut pco, mut pso) = (y_out, conv_st_out, ssm_out);
        let (mut ni, mut nki, mut nvi) = (n as c_int, nk as c_int, nv as c_int);
        let (mut cdi, mut epsi, mut qsi) = (conv_dim as c_int, eps, q_scale);
        let mut args = [
            &mut pq as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut pci as *mut _ as *mut c_void,
            &mut psi as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void,
            &mut pwa as *mut _ as *mut c_void,
            &mut pwd as *mut _ as *mut c_void,
            &mut py as *mut _ as *mut c_void,
            &mut pco as *mut _ as *mut c_void,
            &mut pso as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nki as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut cdi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
            &mut qsi as *mut _ as *mut c_void,
        ];
        // 1 ヘッドを 4 ブロックが分担する (32 列ずつ)。
        unsafe {
            ctx.launch_async(
                fns.scan32,
                (nv as u32, GDN_PF3_BLOCKS, 1),
                (GDN_PF3_THREADS as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    {
        let (mut py, mut pz, mut pwn) = (y_out, z, wnorm);
        let (mut ni, mut nvi) = (n as c_int, nv as c_int);
        let mut epsi = eps;
        let mut args = [
            &mut py as *mut _ as *mut c_void,
            &mut pz as *mut _ as *mut c_void,
            &mut pwn as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
        ];
        let groups = n.saturating_mul(nv);
        let per_block = GDN_PF2_NORM_THREADS / 4;
        let grid = (groups.div_ceil(per_block)) as u32;
        unsafe {
            ctx.launch_async(
                fns.norm,
                (grid, 1, 1),
                (GDN_PF2_NORM_THREADS as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// task 228: prefill の GDN スキャン WY (UT) 形式チャンク版
// (`DENNO_PF_SCAN_WY`、既定 off。初回ラウンドは PTX= で生成)
// ---------------------------------------------------------------------------
//
// 1 ブロック = 1 ヘッド。32 トークンを 1 チャンクにして、チャンク内を
//    A[t][j] = (k_t·k_j) * Π_{m=j+1..t} eg_m
//    D_t = b_t (v_t - Λ_t (k_tᵀ S_0) - Σ_{j<t} A[t][j] D_j)
//    o_t = Λ_t (q_tᵀ S_0) + Σ_{j≤t} (q_t·k_j) Π_{m=j+1..t} eg_m D_j
//    S_new = Λ_last S_0 + Σ_j (Λ_last/Λ_j) k_j ⊗ D_j
// の行列積 + 前進代入で回す (直列なのはチャンク間だけ)。
// CPU 参照は `crate::gdn::wy::wy_head_scan`、値の裏付けはそこのテスト。
//
// 共有メモリは 91,012 B (状態 64 KB + 作業域 C = 8 本ぶん) なので、
// cuFuncSetAttribute で opt-in してから起動する。lab t228a の実機 (RTX 5090,
// sm_120) の opt-in 上限は 99 KB で、C = 32 のときの 176,644 B は通らず
// v3 へ落ちていた (t228a の wy COND は v3 を測っていた)。上限は
// `attn_pf_dynamic_smem_limit()` (CudaCtx 生成時のデバイス問い合わせ) を使い、
// 足りないときはカーネルを掴まない (v3 → v2 → v1 → CPU へ fail closed)。

pub(crate) const GDN_PF_WY_THREADS: usize = 512;

/// カーネルの動的共有メモリ (float 数。`kernels_gdn_pf_wy.cu` の WY_TOTAL)。
pub(crate) const GDN_PF_WY_SMEM_FLOATS: usize = 22753;

pub(crate) const GDN_PF_WY_SMEM_BYTES: usize = GDN_PF_WY_SMEM_FLOATS * 4;

/// lab t228a の実機が許した opt-in 上限 (99 KB)。これ以上は起動できない。
pub(crate) const GDN_PF_WY_SMEM_LIMIT: usize = 99 * 1024;


pub(crate) struct GdnPfWyKernels {
    #[allow(dead_code)]
    module: CUmodule,
    scan: CUfunction,
    norm: CUfunction,
}


unsafe impl Send for GdnPfWyKernels {}

unsafe impl Sync for GdnPfWyKernels {}


fn gdn_pf_wy_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy.ptx"),
    }
}


fn gdn_pf_wy_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = gdn_pf_wy_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[gdn-pf-wy] PTX が読めない ({}) : {e} → v3 のスキャンへ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[gdn-pf-wy] cuModuleLoadData 失敗 → v3 のスキャンへ");
                return None;
            }
            let mut scan: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut scan,
                module,
                b"dno_gdn_pf_wy32\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[gdn-pf-wy] PTX に dno_gdn_pf_wy32 が無い → v3 のスキャンへ");
                return None;
            }
            // 動的 smem の opt-in (48 KB 既定を超えるため必須)。デバイスの
            // opt-in 上限 (CudaCtx 生成時に問い合わせ済み) を先に見て、
            // 足りないときは理由を出して掴まない (t228a: 176,644 B が 99 KB の
            // 上限に弾かれて、黙って v3 が測られていた)。
            let limit = attn_pf_dynamic_smem_limit().min(GDN_PF_WY_SMEM_LIMIT);
            if GDN_PF_WY_SMEM_BYTES > limit {
                eprintln!(
                    "[gdn-pf-wy] 動的 smem {} B がデバイスの上限 {limit} B を超える → v3 のスキャンへ",
                    GDN_PF_WY_SMEM_BYTES
                );
                return None;
            }
            let ok = match lib.func_set_attribute {
                Some(fn_attr) => {
                    (fn_attr)(
                        scan,
                        CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        GDN_PF_WY_SMEM_BYTES as c_int,
                    ) == 0
                }
                None => false,
            };
            if !ok {
                eprintln!(
                    "[gdn-pf-wy] 動的 smem {GDN_PF_WY_SMEM_BYTES} B の opt-in に失敗 → v3 のスキャンへ"
                );
                return None;
            }
            log_kernel_attr(lib, scan, "dno_gdn_pf_wy32");
            // 出力の gated RMS norm は v2 の PTX (同じ演算列) をそのまま使う。
            let norm = gdn_pf2_kernels()?.norm;
            Some(GdnPfWyKernels { module, scan, norm })
        }
    })
    .as_ref()
}


// ---------------------------------------------------------------------------
// task 311: WY スキャンの v 列 4 分割版 (`DENNO_PF_SCAN_WY4`)
// ---------------------------------------------------------------------------
//
// `dno_gdn_pf_wy32` は 1 ブロック = 1 ヘッドなので 48 ブロックしか立たず
// (170 SM)、C = 8 では v3 より 11% 遅かった (t231a: scan_pf 392.8 vs 357.0 ms)。
// `dno_gdn_pf_wy4` は v 列を 4 分割して 192 ブロックにする。状態は
// dk x 32 列 = 16 KB で、共有メモリは 29,572 B (48 KB 既定の内側なので opt-in
// 不要だが、実機の上限は念のため見る)。数値は wy32 と要素ごとに同じ順序。
pub(crate) const GDN_PF_WY4_THREADS: usize = 256;

/// v 列の分割数 (`blockIdx.y`)。dk = 128 を 32 列ずつに切る。
pub(crate) const GDN_PF_WY4_BLOCKS: usize = 4;

/// 1 ブロックが持つ v 列数 (カーネルの `WY4_DV`)。
pub(crate) const GDN_PF_WY4_DV: usize = 32;

/// カーネルの動的共有メモリ (float 数。`kernels_gdn_pf_wy4.cu` の WY4_TOTAL)。
pub(crate) const GDN_PF_WY4_SMEM_FLOATS: usize = 7393;

pub(crate) const GDN_PF_WY4_SMEM_BYTES: usize = GDN_PF_WY4_SMEM_FLOATS * 4;


/// WY 系スキャン (wy32 / wy4) の PTX を読んでカーネルを掴む。
///
/// 3 本とも同じ 17 引数なので、ホストの起動 (`gdn_pf_wy_launch`) と norm を
/// 共有する。違うのは PTX のパス・entry 名・動的共有メモリ・起動形状だけ。
/// `require_attr` は「`func_set_attribute` が無い/失敗したら降りる」か
/// (wy32 = true、48 KB 既定を超えるため必須。wy4 = false)。
fn gdn_wy_load(
    path: std::path::PathBuf,
    entry: &'static [u8],
    smem_bytes: usize,
    tag: &'static str,
    smem_limit: usize,
    require_attr: bool,
) -> Option<GdnPfWyKernels> {
    let mut image = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "{tag} PTX が読めない ({}) : {e} → v3 のスキャンへ",
                path.display()
            );
            return None;
        }
    };
    image.push(0);
    let lib = lib().ok()?;
    let mut module: CUmodule = std::ptr::null_mut();
    unsafe {
        if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
            eprintln!("{tag} cuModuleLoadData 失敗 → v3 のスキャンへ");
            return None;
        }
        let name = String::from_utf8_lossy(&entry[..entry.len() - 1]).to_string();
        let mut scan: CUfunction = std::ptr::null_mut();
        if (lib.module_get_function)(&mut scan, module, entry.as_ptr() as *const c_char)
            != CUDA_SUCCESS
        {
            eprintln!("{tag} PTX に {name} が無い → v3 のスキャンへ");
            return None;
        }
        // 実機の opt-in 上限 (CudaCtx 生成時に問い合わせ済み) を先に見る。
        let limit = attn_pf_dynamic_smem_limit().min(smem_limit);
        if smem_bytes > limit {
            eprintln!(
                "{tag} 動的 smem {smem_bytes} B がデバイスの上限 {limit} B を超える → v3 のスキャンへ"
            );
            return None;
        }
        match lib.func_set_attribute {
            Some(fn_attr) => {
                if (fn_attr)(
                    scan,
                    CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes as c_int,
                ) != 0
                {
                    eprintln!("{tag} 動的 smem {smem_bytes} B の opt-in に失敗 → v3 のスキャンへ");
                    return None;
                }
            }
            None => {
                if require_attr {
                    eprintln!("{tag} func_set_attribute が無い → v3 のスキャンへ");
                    return None;
                }
            }
        }
        log_kernel_attr(lib, scan, &name);
        // 出力の gated RMS norm は v2 の PTX (同じ演算列) をそのまま使う。
        let norm = gdn_pf2_kernels()?.norm;
        Some(GdnPfWyKernels { module, scan, norm })
    }
}


fn gdn_pf_wy4_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY4_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy4.ptx"),
    }
}


fn gdn_pf_wy4_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy4_ptx_path(),
            b"dno_gdn_pf_wy4\0",
            GDN_PF_WY4_SMEM_BYTES,
            "[gdn-pf-wy4]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


// ---------------------------------------------------------------------------
// task 397: WY スキャンの従属連鎖を短くした版 (`DENNO_PF_SCAN_WY5`)
// ---------------------------------------------------------------------------
//
// 起動形 (256 スレッド x (nv, 4) ブロック) も共有メモリ (29,572 B) も wy4 と
// 同じで、**カーネルの中の従属連鎖だけ**が違う (A/G をワープ並列・P/QP を 4
// アキュムレータ・conv の入力を先読み)。よってゲートは env 1 本で、落ちれば
// wy4 へ戻る (wy4 が既定 on)。
pub(crate) const GDN_PF_WY5_SMEM_FLOATS: usize = GDN_PF_WY4_SMEM_FLOATS;

pub(crate) const GDN_PF_WY5_SMEM_BYTES: usize = GDN_PF_WY5_SMEM_FLOATS * 4;


fn gdn_pf_wy5_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY5_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy5.ptx"),
    }
}


fn gdn_pf_wy5_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy5_ptx_path(),
            b"dno_gdn_pf_wy5\0",
            GDN_PF_WY5_SMEM_BYTES,
            "[gdn-pf-wy5]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf_wy5_for_test(v: Option<bool>) {
    WY5_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_SCAN_WY5` の解釈 (純関数)。**lab t397b で採ったので既定 on**。
///
/// lab 実測 (既定モデル、depth8k、PROFILE=1、`[q4ledger-pf]` の `ssm` 段の
/// `busy`): 3,826 cold 433.3/433.1 (b1/b2) -> **370.9** (wy5) = -14.4%、
/// 14k cold 1,554.5/1,557.9 -> **1,330.9** = -14.5%。decode は 15.5-15.9 ms/tok で
/// 不変、errors 0、text sane。md5 は turn2 の 2 本だけ DIFF (A/G と P/QP の
/// 和の木を変えた丸め差。cold の text は同じ)。切り戻しは `DENNO_PF_SCAN_WY5=0`。
pub(crate) fn gdn_pf_wy5_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 従属連鎖を短くした WY スキャンを使うか (既定 on)。
pub(crate) fn gdn_pf_wy5_on() -> bool {
    #[cfg(test)]
    {
        match WY5_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy5_from_value(std::env::var("DENNO_PF_SCAN_WY5").ok().as_deref()))
}


/// 従属連鎖を短くした WY スキャンが受ける形か (wy4 と同じ前提)。
pub(crate) fn gdn_pf_wy5_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    gdn_pf_wy4_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


// ---------------------------------------------------------------------------
// task 407: WY スキャンのバリア削減 (wy7) と P/QP の S 読み直し削減 (wy7q)
// ---------------------------------------------------------------------------
//
// lab t407a で kprof を直したら 14k cold の `ssm` busy 1,331 ms の内訳が読めた:
// scan 707.1 / qkv_gemm 264.9 / z_gemm 158.4 / out_gemm 152.5 / ba_gemm 60.1。
// 同じ run の wy6 (1 ブロック 512 スレッド) / wy6p は scan 1,024.4 / 1,071.6 ms と
// **+45 / +51%** で、占有率を上げる方向は外れ (256 スレッドが位相の幅と一致している)。
//
// - wy7: バリアだけを 1 チャンク 8 -> 6 本に減らす。(2) と (3)、(7) と (8) は
//   互いに別の配列しか触らない。**値は wy5 と 1 bit も変わらない**。
// - wy7q: wy7 + (5) P/QP の r 方向をワープで分ける (S の読み直し 8 回 -> 1 回、
//   1 レーンのロード 256 -> 152)。和の木が変わるので**丸めだけ**違う。
pub(crate) const GDN_PF_WY7_SMEM_FLOATS: usize = GDN_PF_WY4_SMEM_FLOATS;

pub(crate) const GDN_PF_WY7_SMEM_BYTES: usize = GDN_PF_WY7_SMEM_FLOATS * 4;

/// wy7q は P/QP の部分和 (2 x 8 ワープ x C x DV = 4,096 float) を足す。
pub(crate) const GDN_PF_WY7Q_PART_FLOATS: usize = 2 * 8 * 8 * 32;

pub(crate) const GDN_PF_WY7Q_SMEM_FLOATS: usize =
    GDN_PF_WY7_SMEM_FLOATS + GDN_PF_WY7Q_PART_FLOATS;

pub(crate) const GDN_PF_WY7Q_SMEM_BYTES: usize = GDN_PF_WY7Q_SMEM_FLOATS * 4;


fn gdn_pf_wy7_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY7_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy7.ptx"),
    }
}


fn gdn_pf_wy7_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy7_ptx_path(),
            b"dno_gdn_pf_wy7\0",
            GDN_PF_WY7_SMEM_BYTES,
            "[gdn-pf-wy7]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


fn gdn_pf_wy7q_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY7Q_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy7q.ptx"),
    }
}


fn gdn_pf_wy7q_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy7q_ptx_path(),
            b"dno_gdn_pf_wy7q\0",
            GDN_PF_WY7Q_SMEM_BYTES,
            "[gdn-pf-wy7q]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


#[cfg(test)]
pub(crate) fn set_gdn_pf_wy7_for_test(v: Option<bool>) {
    WY7_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


#[cfg(test)]
pub(crate) fn set_gdn_pf_wy7q_for_test(v: Option<bool>) {
    WY7Q_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_SCAN_WY7` の解釈 (純関数)。**既定 off**。
pub(crate) fn gdn_pf_wy7_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => false,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// `DENNO_PF_SCAN_WY7Q` の解釈 (純関数)。**既定 off**。
pub(crate) fn gdn_pf_wy7q_from_value(v: Option<&str>) -> bool {
    gdn_pf_wy7_from_value(v)
}


pub(crate) fn gdn_pf_wy7_on() -> bool {
    #[cfg(test)]
    {
        match WY7_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy7_from_value(std::env::var("DENNO_PF_SCAN_WY7").ok().as_deref()))
}


pub(crate) fn gdn_pf_wy7q_on() -> bool {
    #[cfg(test)]
    {
        match WY7Q_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy7q_from_value(std::env::var("DENNO_PF_SCAN_WY7Q").ok().as_deref()))
}


/// wy7 / wy7q が受ける形か (wy4 / wy5 と同じ前提)。
pub(crate) fn gdn_pf_wy7_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    gdn_pf_wy5_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


// ---------------------------------------------------------------------------
// task 407: WY スキャンの占有率を上げた版 (`DENNO_PF_SCAN_WY6` / `DENNO_PF_SCAN_WY6P`)
// ---------------------------------------------------------------------------
//
// 形 (v 列 4 分割 = (nv, 4) ブロック) も共有メモリ (7,393 float = 29,572 B) も
// wy5 と同じで、**1 ブロックのスレッド数だけ** 256 -> 512 にした。170 SM に対して
// 192 ブロック = 1 ブロック/SM だと 8 ワープ/SM (64 の 12.5%) しか立たず、
// 共有メモリ経由の従属連鎖を隠せないため (P397 §6-2 の「まだ余白がある」)。
//
// - wy6: 演算列も wy5 と同一 (値も同一)。状態の更新が 1 スレッド 16 -> 8 本、
//   A/G が 8 -> 16 ワープになる。
// - wy6p: wy6 に加えて (5) P/QP の r 方向 (dk=128) を隣り合う 2 レーンで
//   半分ずつ持ち、`__shfl_xor_sync(...,1)` で足す。和の木が変わるので
//   **丸めだけ** wy6 と違う (決定論的)。
pub(crate) const GDN_PF_WY6_SMEM_FLOATS: usize = GDN_PF_WY4_SMEM_FLOATS;

pub(crate) const GDN_PF_WY6_SMEM_BYTES: usize = GDN_PF_WY6_SMEM_FLOATS * 4;

pub(crate) const GDN_PF_WY6_THREADS: usize = 512;

pub(crate) const GDN_PF_WY6_BLOCKS: usize = GDN_PF_WY4_BLOCKS;


fn gdn_pf_wy6_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY6_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy6.ptx"),
    }
}


fn gdn_pf_wy6_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy6_ptx_path(),
            b"dno_gdn_pf_wy6\0",
            GDN_PF_WY6_SMEM_BYTES,
            "[gdn-pf-wy6]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


fn gdn_pf_wy6p_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SCAN_WY6P_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_pf_wy6p.ptx"),
    }
}


fn gdn_pf_wy6p_kernels() -> Option<&'static GdnPfWyKernels> {
    static K: OnceLock<Option<GdnPfWyKernels>> = OnceLock::new();
    K.get_or_init(|| {
        gdn_wy_load(
            gdn_pf_wy6p_ptx_path(),
            b"dno_gdn_pf_wy6p\0",
            GDN_PF_WY6_SMEM_BYTES,
            "[gdn-pf-wy6p]",
            GDN_PF_WY_SMEM_LIMIT,
            false,
        )
    })
    .as_ref()
}


#[cfg(test)]
pub(crate) fn set_gdn_pf_wy6_for_test(v: Option<bool>) {
    WY6_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


#[cfg(test)]
pub(crate) fn set_gdn_pf_wy6p_for_test(v: Option<bool>) {
    WY6P_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_SCAN_WY6` の解釈 (純関数)。**既定 off** (lab で採否を決める)。
pub(crate) fn gdn_pf_wy6_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => false,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// `DENNO_PF_SCAN_WY6P` の解釈 (純関数)。**既定 off**。
pub(crate) fn gdn_pf_wy6p_from_value(v: Option<&str>) -> bool {
    gdn_pf_wy6_from_value(v)
}


pub(crate) fn gdn_pf_wy6_on() -> bool {
    #[cfg(test)]
    {
        match WY6_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy6_from_value(std::env::var("DENNO_PF_SCAN_WY6").ok().as_deref()))
}


pub(crate) fn gdn_pf_wy6p_on() -> bool {
    #[cfg(test)]
    {
        match WY6P_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy6p_from_value(std::env::var("DENNO_PF_SCAN_WY6P").ok().as_deref()))
}


/// wy6 / wy6p が受ける形か (wy4 / wy5 と同じ前提)。
pub(crate) fn gdn_pf_wy6_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    gdn_pf_wy5_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf_wy4_for_test(v: Option<bool>) {
    WY4_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_SCAN_WY4` の解釈 (純関数)。**task 317 で既定 on**。
///
/// task 311: wy4 単独では 3,826 -2.9% / 14k -4.3% で目標 -5% に届かず既定 off
/// だった。task 317 (t317a) で ssm の mix D2H 除去 (`DENNO_PF_MIXKEEP_SSM`) と
/// 重ねると 3,826 cold 2.945 → 2.78 s (-5.4%、2 ラウンドとも同値) になり、
/// 採択して既定 on にした。scan_pf は 359 → 256 ms。
pub(crate) fn gdn_pf_wy4_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// v 列 4 分割の WY スキャンを使うか (既定 off)。
pub(crate) fn gdn_pf_wy4_on() -> bool {
    #[cfg(test)]
    {
        match WY4_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy4_from_value(std::env::var("DENNO_PF_SCAN_WY4").ok().as_deref()))
}


/// v 列 4 分割の前提 (純関数)。dk = dv = 128 / d_conv = 4 / dv が 4 で割れる。
pub(crate) fn gdn_pf_wy4_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    gdn_pf2_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
        && dv == GDN_PF_WY4_BLOCKS * GDN_PF_WY4_DV
}


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_gdn_pf_wy_for_test(v: Option<bool>) {
    WY_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_SCAN_WY` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn gdn_pf_wy_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        // task 228: 初回は既定 off (lab で採否を決める)。
        // task 231 (t231a): C = 8 で起動できるようにしたうえで測ったが、
        // scan_pf が v3 より 11% 遅かった (3826: 392.8 vs 357.0 ms /
        // 14018: 1458.6 vs 1311.7 ms)。14k cold prefill も +1.3% なので
        // 既定 off のまま。切り戻し不要 (env = 1 で試せる)。
        None => false,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// WY スキャンを使うか (既定 off。t231a で v3 より遅いと分かった)。
pub(crate) fn gdn_pf_wy_on() -> bool {
    #[cfg(test)]
    {
        match WY_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_pf_wy_from_value(std::env::var("DENNO_PF_SCAN_WY").ok().as_deref()))
}


/// WY スキャンが受ける形か (純関数。カーネルの前提と同じ)。
pub(crate) fn gdn_pf_wy_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
) -> bool {
    // 共有メモリは dk = dv = 128 固定 (WY_DK)。v2/v3 と同じ前提に揃える。
    gdn_pf2_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


/// task 228: WY スキャンを撃つ (wy32 → v2 の norm の 2 起動)。引数の並びは
/// カーネルの宣言順で、v2/v3 の launch と同じ (17 個)。
///
/// task 311: `vblocks` / `threads` / `smem_bytes` を引数にした。wy32 は
/// (1, 512, 91,012)、wy4 は (4, 256, 29,572)。カーネルの引数は同じなので
/// 起動形状だけが違う。
#[allow(clippy::too_many_arguments)]
fn gdn_pf_wy_launch(
    ctx: &CudaCtx,
    fns: &GdnPfWyKernels,
    qkv: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_st_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wconv: CUdeviceptr,
    wa: CUdeviceptr,
    wdt: CUdeviceptr,
    wnorm: CUdeviceptr,
    y_out: CUdeviceptr,
    conv_st_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    n: usize,
    nk: usize,
    nv: usize,
    conv_dim: usize,
    eps: f32,
    q_scale: f32,
    vblocks: usize,
    threads: usize,
    smem_bytes: u32,
) -> Result<()> {
    {
        let (mut pq, mut pb, mut pa) = (qkv, beta, alpha);
        let (mut pci, mut psi) = (conv_st_in, ssm_in);
        let (mut pwc, mut pwa, mut pwd) = (wconv, wa, wdt);
        let (mut py, mut pco, mut pso) = (y_out, conv_st_out, ssm_out);
        let (mut ni, mut nki, mut nvi) = (n as c_int, nk as c_int, nv as c_int);
        let (mut cdi, mut epsi, mut qsi) = (conv_dim as c_int, eps, q_scale);
        let mut args = [
            &mut pq as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut pci as *mut _ as *mut c_void,
            &mut psi as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void,
            &mut pwa as *mut _ as *mut c_void,
            &mut pwd as *mut _ as *mut c_void,
            &mut py as *mut _ as *mut c_void,
            &mut pco as *mut _ as *mut c_void,
            &mut pso as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nki as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut cdi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
            &mut qsi as *mut _ as *mut c_void,
        ];
        // 1 ブロック = (1 ヘッド, v 列の 1/vblocks)。wy32 は 1 ヘッド全体。
        unsafe {
            ctx.launch_async(
                fns.scan,
                (nv as u32, vblocks as u32, 1),
                (threads as u32, 1, 1),
                smem_bytes,
                &mut args,
            )?;
        }
    }
    // 以降は v2 と同じ norm (gated RMS norm) の 1 起動。
    {
        let (mut py, mut pz, mut pwn) = (y_out, z, wnorm);
        let (mut ni, mut nvi) = (n as c_int, nv as c_int);
        let mut epsi = eps;
        let mut args = [
            &mut py as *mut _ as *mut c_void,
            &mut pz as *mut _ as *mut c_void,
            &mut pwn as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
        ];
        let groups = n.saturating_mul(nv);
        let per_block = GDN_PF2_NORM_THREADS / 4;
        let grid = (groups.div_ceil(per_block)) as u32;
        unsafe {
            ctx.launch_async(
                fns.norm,
                (grid, 1, 1),
                (GDN_PF2_NORM_THREADS as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    Ok(())
}


/// task 173: レジスタ常駐スキャンを「ホスト入力 → ホスト出力」で 1 回だけ回す
/// (検算用。prefill の本線は連鎖 [`gdn_pf_chain`] の中から呼ぶ)。
///
/// 戻り値 `None` は「GPU・PTX・形の前提が無い」(呼び出し側はスキップしてよい)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_pf_scan128_host(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
) -> Option<Vec<f32>> {
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    if !gdn_pf_fast_eligible(n, nk, nv, dk, dv, d_conv, conv_dim)
        || qkv.len() != n * conv_dim
        || z.len() != n * vd
        || beta.len() != n * nv
        || alpha.len() != n * nv
        || conv_st.len() != conv_dim * cs
        || ssm_st.len() != nv * dk * dv
        || wconv.len() != conv_dim * d_conv
        || wa.len() != nv
        || wdt.len() != nv
        || wnorm.len() != dv
    {
        return None;
    }
    if !cuda_enabled() {
        return None;
    }
    let ctx = shared_ctx()?;
    if ctx.make_current().is_err() {
        return None;
    }
    let fns = gdn_pf_kernels()?;
    let d_qkv = DevBuffer::from_slice(qkv).ok()?;
    let d_z = DevBuffer::from_slice(z).ok()?;
    let d_beta = DevBuffer::from_slice(beta).ok()?;
    let d_alpha = DevBuffer::from_slice(alpha).ok()?;
    let d_cst = DevBuffer::from_slice(conv_st).ok()?;
    let d_sst = DevBuffer::from_slice(ssm_st).ok()?;
    let d_wc = DevBuffer::from_slice(wconv).ok()?;
    let d_wa = DevBuffer::from_slice(wa).ok()?;
    let d_wd = DevBuffer::from_slice(wdt).ok()?;
    let d_wn = DevBuffer::from_slice(wnorm).ok()?;
    let d_y = DevBuffer::alloc(n * vd * 4).ok()?;
    let d_co = DevBuffer::alloc(conv_dim * cs * 4).ok()?;
    let d_so = DevBuffer::alloc(nv * dk * dv * 4).ok()?;
    gdn_pf_scan128_launch(
        &ctx,
        fns,
        d_qkv.ptr(),
        d_z.ptr(),
        d_beta.ptr(),
        d_alpha.ptr(),
        d_cst.ptr(),
        d_sst.ptr(),
        d_wc.ptr(),
        d_wa.ptr(),
        d_wd.ptr(),
        d_wn.ptr(),
        d_y.ptr(),
        d_co.ptr(),
        d_so.ptr(),
        n,
        nk,
        nv,
        conv_dim,
        eps,
        q_scale,
    )
    .ok()?;
    ctx.sync().ok()?;
    let y = d_y.to_vec::<f32>(n * vd).ok()?;
    let co = d_co.to_vec::<f32>(conv_dim * cs).ok()?;
    let so = d_so.to_vec::<f32>(nv * dk * dv).ok()?;
    conv_st.copy_from_slice(&co);
    ssm_st.copy_from_slice(&so);
    Some(y)
}


// ---------------------------------------------------------------------------
// task 173: 連鎖の段ごとの GPU 時間 (`DENNO_GDN_PF_KPROF=1`・既定 off)
// ---------------------------------------------------------------------------
//
// 台帳の段 (`qkv_pf` / `scan_pf` / `out_pf`) は「発行 + 同期の待ち」を含む壁時計
// なので、**待ち**が混ざると中身が読めない (expert の別ストリームが走っている
// 間の `cuCtxSynchronize` はその完了も待つ)。ここでは段の前後に CUDA イベントを
// 打ち、`cuEventElapsedTime` で「その段の GPU 時間」だけを積む。段の壁時計との
// 差が「他人の仕事を待っていた時間」になる。
pub mod gdn_pf_kprof {
    use super::*;

    /// 段の番号 (この順で 1 行に出す)。
    pub const X_H2D: usize = 0;
    pub const QKV_GEMM: usize = 1;
    pub const Z_GEMM: usize = 2;
    pub const SCAN: usize = 3;
    pub const OUT_GEMM: usize = 4;
    pub const OUT_D2H: usize = 5;
    pub const ST_D2H: usize = 6;
    /// task 225: beta/alpha 射影 (デバイス版)。
    pub const BA_GEMM: usize = 7;
    pub const N: usize = 8;
    pub const NAMES: [&str; N] = [
        "x_h2d", "qkv_gemm", "z_gemm", "scan", "out_gemm", "out_d2h", "st_d2h", "ba_gemm",
    ];

    pub(crate) static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    pub(crate) static CNT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    /// 計測するか (`DENNO_GDN_PF_KPROF` があるとき)。
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_GDN_PF_KPROF").is_some())
    }

    /// 積算を 0 に戻す。
    pub fn reset() {
        for i in 0..N {
            NS[i].store(0, Ordering::Relaxed);
            CNT[i].store(0, Ordering::Relaxed);
        }
        open_reset();
    }

    /// 1 行にまとめる (1 本も打っていなければ `None`)。
    pub fn lines() -> Vec<String> {
        if !on() {
            return Vec::new();
        }
        let cnt: u64 = CNT.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        if cnt == 0 {
            return Vec::new();
        }
        let mut s = String::from("[gdn-pf-kprof] 段ごとの GPU 時間 (イベント計測)");
        for i in 0..N {
            let ns = NS[i].load(Ordering::Relaxed);
            let n = CNT[i].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            s.push_str(&format!(
                " | {} {:.1}ms/{}",
                NAMES[i],
                ns as f64 / 1e6,
                n
            ));
        }
        vec![s]
    }

    pub(crate) fn add_ns(i: usize, ns: u64) {
        if i >= N {
            return;
        }
        NS[i].fetch_add(ns, Ordering::Relaxed);
        CNT[i].fetch_add(1, Ordering::Relaxed);
    }

    /// イベントのプール (スレッドローカル)。収集のたびに先頭から使い直す。
    pub(crate) struct Pool {
        evs: Vec<CUevent>,
        used: usize,
        dead: bool,
    }

    thread_local! {
        static P: std::cell::RefCell<Pool> = const {
            std::cell::RefCell::new(Pool { evs: Vec::new(), used: 0, dead: false })
        };
        /// 段ごとの入口イベント番号。
        static OPEN: std::cell::RefCell<[Option<u32>; N]> = const {
            std::cell::RefCell::new([None; N])
        };
        /// 収集待ちの (段, 入口, 出口)。
        static PEND: std::cell::RefCell<Vec<(u8, u32, u32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    pub(crate) fn mark() -> Option<u32> {
        P.with(|c| {
            let mut p = c.borrow_mut();
            if p.dead {
                return None;
            }
            if p.used >= p.evs.len() {
                let Some(create) = lib().ok().and_then(|l| l.event_create) else {
                    p.dead = true;
                    return None;
                };
                let mut ev: CUevent = std::ptr::null_mut();
                // CU_EVENT_DEFAULT (0) = 計時有効。
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    p.dead = true;
                    return None;
                }
                p.evs.push(ev);
            }
            let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
                p.dead = true;
                return None;
            };
            let ev = p.evs[p.used];
            if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
                p.dead = true;
                return None;
            }
            let i = p.used as u32;
            p.used += 1;
            Some(i)
        })
    }

    pub(crate) fn open_reset() {
        OPEN.with(|c| {
            let mut o = c.borrow_mut();
            for v in o.iter_mut() {
                *v = None;
            }
        });
        PEND.with(|c| c.borrow_mut().clear());
        P.with(|c| c.borrow_mut().used = 0);
    }

    /// 段の入口 (on でなければ 1 分岐)。
    pub fn begin(item: usize) {
        if !on() || item >= N {
            return;
        }
        let a = mark();
        OPEN.with(|c| c.borrow_mut()[item] = a);
    }

    /// 段の出口。
    pub fn end(item: usize) {
        if !on() || item >= N {
            return;
        }
        let a = OPEN.with(|c| c.borrow_mut()[item].take());
        let Some(a) = a else { return };
        if let Some(b) = mark() {
            PEND.with(|c| c.borrow_mut().push((item as u8, a, b)));
        }
    }

    /// 積算する (**同期の後**に呼ぶ。イベントが完了している必要がある)。
    ///
    /// task 407: まだ GPU が通っていない段 (イベントが未完了 = `CUDA_ERROR_NOT_READY`)
    /// は**捨てずに次回へ回す**。捨てると、同期を省いた経路
    /// (`DENNO_PF_GDN_STATE_LAST` で書き戻しを飛ばしたチャンクなど) の段だけが
    /// 数から抜けて合計が小さく出る (P397 §6-1 の「scan が 36 回しか出ない」)。
    pub fn collect() {
        if !on() {
            return;
        }
        let Some(f) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            return;
        };
        let pend = PEND.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if pend.is_empty() {
            return;
        }
        let mut keep: Vec<(u8, u32, u32)> = Vec::new();
        P.with(|c| {
            let p = c.borrow();
            for (item, a, b) in pend {
                let (Some(&ea), Some(&eb)) = (p.evs.get(a as usize), p.evs.get(b as usize))
                else {
                    continue;
                };
                let mut ms = 0.0f32;
                if unsafe { f(&mut ms, ea, eb) } == CUDA_SUCCESS && ms.is_finite() && ms >= 0.0 {
                    add_ns(item as usize, (ms as f64 * 1e6) as u64);
                } else {
                    keep.push((item, a, b));
                }
            }
        });
        if keep.is_empty() {
            // イベントは先頭から使い直す (プールは数十本で足りる)。
            P.with(|c| c.borrow_mut().used = 0);
        } else {
            PEND.with(|c| c.borrow_mut().extend(keep));
        }
    }
}


// ===========================================================================
// task 187: decode の IQ expert 段 (1 層 4 起動 + D2H) の GPU 時間をイベントで
// 段ごとに測る。`DENNO_Q4_PROFILE` のときだけ動く (台帳と同じ窓)。
//
// 目的 (task 187 の表): 「experts の gpu 3.3 ms は expert 数とカーネルごとに
// いくらか」を実測で割る。ヒット本数ごとの層数も一緒に積む。
// ===========================================================================
pub mod dec_exp_kprof {
    use super::*;

    /// 段の番号 (この順で 1 行に出す)。
    pub const GATE: usize = 0;
    pub const UP: usize = 1;
    pub const SILU: usize = 2;
    pub const DOWN: usize = 3;
    pub const D2H: usize = 4;
    /// task 187: デバイス和 (`dno_dec_exp_sum`)。
    pub const SUM: usize = 5;
    pub const N: usize = 6;
    pub const NAMES: [&str; N] = ["gate", "up", "silu", "down", "d2h", "sum"];
    /// ヒット本数の上限 (n_expert_used = 10 まで数える)。
    pub const MAX_HITS: usize = 10;

    pub(crate) static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    pub(crate) static CNT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    /// ヒット本数ごとの層数と合計 ns。
    static BUCKET_N: [AtomicU64; MAX_HITS + 1] = [const { AtomicU64::new(0) }; MAX_HITS + 1];
    static BUCKET_NS: [AtomicU64; MAX_HITS + 1] = [const { AtomicU64::new(0) }; MAX_HITS + 1];

    // ------------------------------------------------------------------
    // task 190: 型 (gate/up/down のスロット) × 行数 の表
    //
    // 「expert カーネルが IQ の型と行数でいくらか」を出すための積算。1 層 =
    // gate / up / silu / down の 4 起動なので、層の合計 ns を (型, 行数) に
    // 積む。行数は 1 層の起動が扱う出力行の合計 (`nseg*(ff + e)` に相当)。
    // ------------------------------------------------------------------

    /// (slot_g, slot_u, slot_d) の組み合わせ数。
    pub const FKIND: usize = crate::model::exp_iq::SLOTS * crate::model::exp_iq::SLOTS
        * crate::model::exp_iq::SLOTS;
    /// 行数のバケット数 (2 の冪で 1,000 行を下限にする)。
    pub const FROW: usize = 8;

    /// 1 セルぶんの積算。
    struct KAcc {
        calls: AtomicU64,
        experts: AtomicU64,
        bytes: AtomicU64,
        ns: AtomicU64,
    }

    static KFMT: [KAcc; FKIND] = [const {
        KAcc {
            calls: AtomicU64::new(0),
            experts: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            ns: AtomicU64::new(0),
        }
    }; FKIND];
    static KROW: [KAcc; FROW] = [const {
        KAcc {
            calls: AtomicU64::new(0),
            experts: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            ns: AtomicU64::new(0),
        }
    }; FROW];

    thread_local! {
        /// この層の (slot_g*36 + slot_u*6 + slot_d)。
        static KIND: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
        static K_NSEG: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static K_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static K_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// 層の (型, 行数, 重みバイト) を控える ([`begin_layer`] の直後に呼ぶ)。
    ///
    /// `rows_gu` = gate/up の出力行 (通常 ff)、`rows_d` = down の出力行 (通常 e)。
    #[allow(clippy::too_many_arguments)]
    pub fn note_kind(
        slot_g: usize,
        slot_u: usize,
        slot_d: usize,
        nseg: usize,
        rows_gu: usize,
        rows_d: usize,
        bytes: u64,
    ) {
        if !on() {
            return;
        }
        let slots = crate::model::exp_iq::SLOTS;
        let kind = (slot_g.min(slots - 1)) * slots * slots
            + (slot_u.min(slots - 1)) * slots
            + slot_d.min(slots - 1);
        KIND.with(|c| c.set(kind));
        K_NSEG.with(|c| c.set(nseg as u64));
        K_ROWS.with(|c| c.set((nseg * (rows_gu + rows_d)) as u64));
        K_BYTES.with(|c| c.set(bytes));
    }

    /// 行数のバケット (1,000 行 = 0、2,000 = 1、4,000 = 2、... 128,000 以上 = 7)。
    fn row_bucket(rows: u64) -> usize {
        let mut b = 0usize;
        let mut lim = 2_000u64;
        while b + 1 < FROW && rows >= lim {
            b += 1;
            lim *= 2;
        }
        b
    }

    /// 計測するか (台帳と同じ `DENNO_Q4_PROFILE`。テストの強制も含む)。
    pub fn on() -> bool {
        crate::model::q4ledger::enabled()
    }

    /// 積算を 0 に戻す。
    pub fn reset() {
        for i in 0..N {
            NS[i].store(0, Ordering::Relaxed);
            CNT[i].store(0, Ordering::Relaxed);
        }
        for i in 0..=MAX_HITS {
            BUCKET_N[i].store(0, Ordering::Relaxed);
            BUCKET_NS[i].store(0, Ordering::Relaxed);
        }
        for a in KFMT.iter().chain(KROW.iter()) {
            a.calls.store(0, Ordering::Relaxed);
            a.experts.store(0, Ordering::Relaxed);
            a.bytes.store(0, Ordering::Relaxed);
            a.ns.store(0, Ordering::Relaxed);
        }
        KIND.with(|c| c.set(usize::MAX));
        POOL_DEAD.with(|d| d.set(false));
    }

    /// 1 行にまとめる (1 本も打っていなければ `None`)。
    pub fn lines() -> Vec<String> {
        if !on() {
            return Vec::new();
        }
        let cnt: u64 = CNT.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        if cnt == 0 {
            return Vec::new();
        }
        let mut s = String::from("[dec-exp-kprof] 層内の GPU 時間 (イベント計測)");
        for i in 0..N {
            let ns = NS[i].load(Ordering::Relaxed);
            let n = CNT[i].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            s.push_str(&format!(
                " | {} {:.2}ms/{:.0} ({:.1}us/層)",
                NAMES[i],
                ns as f64 / 1e6,
                n,
                ns as f64 / 1e3 / n as f64
            ));
        }
        let mut out = vec![s];
        let mut b = String::from("[dec-exp-kprof] ヒット本数ごと (層数 / 合計 ms / 1 層 us)");
        for h in 0..=MAX_HITS {
            let n = BUCKET_N[h].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let ns = BUCKET_NS[h].load(Ordering::Relaxed);
            b.push_str(&format!(
                " | h{h} {n}層 {:.2}ms {:.1}us",
                ns as f64 / 1e6,
                ns as f64 / 1e3 / n as f64
            ));
        }
        out.push(b);
        // task 190: 型 (gate/up/down) ごとの表。
        let mut f = String::from("[exp-iq-kprof] 型ごと (g/u/d): calls experts rows bytes us/expert GB/s");
        let mut any = false;
        for (k, a) in KFMT.iter().enumerate() {
            let calls = a.calls.load(Ordering::Relaxed);
            if calls == 0 {
                continue;
            }
            any = true;
            let slots = crate::model::exp_iq::SLOTS;
            let (g, u, d) = (k / (slots * slots), (k / slots) % slots, k % slots);
            let experts = a.experts.load(Ordering::Relaxed);
            let bytes = a.bytes.load(Ordering::Relaxed);
            let ns = a.ns.load(Ordering::Relaxed);
            let us_ex = if experts > 0 {
                ns as f64 / 1e3 / experts as f64
            } else {
                0.0
            };
            let gbs = if ns > 0 {
                bytes as f64 / ns as f64
            } else {
                0.0
            };
            f.push_str(&format!(
                " | {g}/{u}/{d} {calls}層 {experts}本 {:.1}MB {us_ex:.1}us/本 {gbs:.0}GB/s",
                bytes as f64 / 1e6
            ));
        }
        if any {
            out.push(f);
        }
        // task 190: 行数バケットごとの表 (1 層 = gate/up + down の出力行の合計)。
        let mut r = String::from("[exp-iq-kprof] 行数ごと: rows calls experts us/launch us/expert GB/s");
        let mut any = false;
        for (i, a) in KROW.iter().enumerate() {
            let calls = a.calls.load(Ordering::Relaxed);
            if calls == 0 {
                continue;
            }
            any = true;
            let experts = a.experts.load(Ordering::Relaxed);
            let bytes = a.bytes.load(Ordering::Relaxed);
            let ns = a.ns.load(Ordering::Relaxed);
            let lo = if i == 0 { 0u64 } else { 2_000u64 << (i - 1) };
            let us_call = ns as f64 / 1e3 / calls as f64;
            let us_ex = if experts > 0 {
                ns as f64 / 1e3 / experts as f64
            } else {
                0.0
            };
            let gbs = if ns > 0 {
                bytes as f64 / ns as f64
            } else {
                0.0
            };
            r.push_str(&format!(
                " | >={lo} {calls}層 {experts}本 {us_call:.1}us {us_ex:.1}us/本 {gbs:.0}GB/s"
            ));
        }
        if any {
            out.push(r);
        }
        out
    }

    thread_local! {
        /// イベントのプール。層ごとに先頭から使い直す。
        static POOL: std::cell::RefCell<Vec<CUevent>> = const {
            std::cell::RefCell::new(Vec::new())
        };
        static POOL_USED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// 直前のイベント (段の開始点) と、収集待ちの (段, 入口, 出口)。
        static LAST: std::cell::Cell<Option<CUevent>> = const { std::cell::Cell::new(None) };
        static PEND: std::cell::RefCell<Vec<(usize, CUevent, CUevent)>> = const {
            std::cell::RefCell::new(Vec::new())
        };
        /// この層のヒット本数 (end で束へ積む)。
        static HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    thread_local! {
        static POOL_DEAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// 層の入口。`hits` は GPU で回した expert 数。
    pub fn begin_layer(hits: u32) {
        if !on() {
            return;
        }
        HITS.with(|c| c.set(hits));
        POOL_USED.with(|c| c.set(0));
        PEND.with(|c| c.borrow_mut().clear());
        let ev = record();
        LAST.with(|c| c.set(ev));
    }

    /// 段 `item` の終わり (イベントを 1 本打って差を**収集待ち**に積む)。
    ///
    /// 経過は [`end_layer`] で読む: overlap 経路では D2H のイベントを同期の
    /// **前**に打つので、ここで `cuEventElapsedTime` を呼ぶと未完了で失敗する。
    pub fn mark(item: usize) {
        if !on() || item >= N {
            return;
        }
        let Some(prev) = LAST.with(|c| c.get()) else {
            return;
        };
        let Some(cur) = record() else { return };
        PEND.with(|c| c.borrow_mut().push((item, prev, cur)));
        LAST.with(|c| c.set(Some(cur)));
    }

    /// 層の出口 (**同期の後**に呼ぶ)。
    pub fn end_layer() {
        if !on() {
            return;
        }
        LAST.with(|c| c.set(None));
        let Some(f) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            PEND.with(|c| c.borrow_mut().clear());
            return;
        };
        let pend = PEND.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut acc = [0u64; N];
        let mut sum = 0u64;
        for (item, a, b) in pend {
            let mut ms = 0.0f32;
            if unsafe { f(&mut ms, a, b) } == CUDA_SUCCESS && ms.is_finite() && ms >= 0.0 {
                let ns = (ms as f64 * 1e6) as u64;
                acc[item] += ns;
                sum += ns;
            }
        }
        for i in 0..N {
            if acc[i] == 0 {
                continue;
            }
            NS[i].fetch_add(acc[i], Ordering::Relaxed);
            CNT[i].fetch_add(1, Ordering::Relaxed);
        }
        let h = (HITS.with(|c| c.get()) as usize).min(MAX_HITS);
        BUCKET_N[h].fetch_add(1, Ordering::Relaxed);
        BUCKET_NS[h].fetch_add(sum, Ordering::Relaxed);
        // task 190: gate + up + silu + down の 4 段だけを「expert カーネル」として
        // (型, 行数) に積む (D2H と和は別物なので入れない)。
        let gemv_ns = acc[GATE] + acc[UP] + acc[SILU] + acc[DOWN];
        let kind = KIND.with(|c| c.get());
        if kind != usize::MAX && gemv_ns > 0 {
            let nseg = K_NSEG.with(|c| c.get());
            let rows = K_ROWS.with(|c| c.get());
            let bytes = K_BYTES.with(|c| c.get());
            let cell = &KFMT[kind.min(FKIND - 1)];
            cell.calls.fetch_add(1, Ordering::Relaxed);
            cell.experts.fetch_add(nseg, Ordering::Relaxed);
            cell.bytes.fetch_add(bytes, Ordering::Relaxed);
            cell.ns.fetch_add(gemv_ns, Ordering::Relaxed);
            let cell = &KROW[row_bucket(rows).min(FROW - 1)];
            cell.calls.fetch_add(1, Ordering::Relaxed);
            cell.experts.fetch_add(nseg, Ordering::Relaxed);
            cell.bytes.fetch_add(bytes, Ordering::Relaxed);
            cell.ns.fetch_add(gemv_ns, Ordering::Relaxed);
        }
    }

    /// イベントを 1 本打つ (プールから取る)。
    pub(crate) fn record() -> Option<CUevent> {
        if !on() || POOL_DEAD.with(|d| d.get()) {
            return None;
        }
        let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
            POOL_DEAD.with(|d| d.set(true));
            return None;
        };
        POOL.with(|c| {
            let mut p = c.borrow_mut();
            let i = POOL_USED.with(|u| u.get());
            if i >= p.len() {
                let Some(create) = lib().ok().and_then(|l| l.event_create) else {
                    POOL_DEAD.with(|d| d.set(true));
                    return None;
                };
                let mut ev: CUevent = std::ptr::null_mut();
                // CU_EVENT_DEFAULT (0) = 計時有効。
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    POOL_DEAD.with(|d| d.set(true));
                    return None;
                }
                p.push(ev);
            }
            POOL_USED.with(|u| u.set(i + 1));
            let ev = p[i];
            if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
                POOL_DEAD.with(|d| d.set(true));
                return None;
            }
            Some(ev)
        })
    }
}


/// task 167: ssm 層 1 チャンクを「qkv 投影 → z 投影 → GDN スキャン → out_proj」の
/// デバイス連鎖で回す。成功したら `out` に層の出力 [nt, e] を書く。
///
/// 戻り値 `Ok(false)` は「何もしていない」= 呼び出し側は従来経路へ落ちてよい。
/// スキャン起動後の失敗は `Err` (状態が進んでいるので要求を落とすこと)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_pf_chain(
    wq: PfChainQkv<'_>,
    wout: PfChainW<'_>,
    flat: &[f32],
    beta: &[f32],
    alpha: &[f32],
    // task 186: Some のとき beta/alpha は **デバイスで作る** (`DENNO_PF_SSM_GEMM_GPU`)。
    // そのとき `beta` / `alpha` は空でよい (ホスト側の計算を省く)。
    beta_alpha: Option<PfBetaAlpha<'_>>,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
    out: &mut [f32],
    // task 200: `true` を返したときは層出力を**デバイスに残した** (`out` は未書き込み)。
    // 直後の `pf_combine_dev` がそのまま読む (`DENNO_PF_ATTN_DEVRES`)。
    kept_dev: &mut bool,
    phase_ns: &mut [u64; 3],
) -> Result<bool> {
    let e = wq.qkv.k;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    let cs = d_conv - 1;
    if !gdn_pf_chain_ready(&wq, &wout, nt, e, nk, nv, dk, dv, d_conv) {
        return Ok(false);
    }
    // `DENNO_GPU_GEMM=0` は従来経路でも GEMM を CPU に落とす。連鎖も同じ扱いにする
    // (ゲート off のときの木を 1 バイトも変えない)。
    if !gpu_gemm_enabled() {
        return Ok(false);
    }
    let ba_dev_req = beta_alpha.is_some() && pf_ba_dev_on();
    if flat.len() != nt * e
        || (ba_dev_req && (!beta.is_empty() || !alpha.is_empty()))
        || (!ba_dev_req && (beta.len() != nt * nv || alpha.len() != nt * nv))
        || out.len() != nt * e
        || conv_st.len() != conv_dim * cs
        || ssm_st.len() != nv * dk * dv
        || wconv.len() != conv_dim * d_conv
        || wa.len() != nv
        || wdt.len() != nv
        || wnorm.len() != dv
    {
        return Ok(false);
    }
    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    if let Err(err) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, GDN prefill chain skipped: {err:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return Ok(false);
    }
    let Some(f_scan) = ctx.f_gdn_scan_sig else {
        // 旧 PTX (sig 未収録) では黙って従来経路へ (ラッチはしない)。
        return Ok(false);
    };
    // task 270: スキャンカーネルを**投影より前**に選ぶ。`nt >= 2048` では単発
    // sigmoid (`f_scan`) の前提外なので、v1/v2/v3/WY のどれも撃てなければ
    // まだ何も書いていないここで降りる (呼び出し側は従来経路で計算し直す)。
    // 既定の順は task 173/186/225/228/311 のまま: WY8 (opt-in) → WY4 (opt-in)
    // → WY (opt-in) → v3 → v2 → v1。
    // task 407: wy7q / wy7 (バリア削減 + P/QP の S 読み直し削減) が先、次に wy6p /
    // wy6 (占有率。t407a で +45% と分かったので既定 off のまま A/B 用)、
    // 既定は wy5 (wy4 も既定 on のまま後ろに残す)。
    let pf_scan_wy7q = if gdn_pf_wy7q_on() && gdn_pf_wy7_ok(nt, nk, nv, dk, dv, d_conv, conv_dim) {
        gdn_pf_wy7q_kernels()
    } else {
        None
    };
    let pf_scan_wy7 = if pf_scan_wy7q.is_none()
        && gdn_pf_wy7_on()
        && gdn_pf_wy7_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy7_kernels()
    } else {
        None
    };
    let pf_scan_wy6p = if pf_scan_wy7q.is_none()
        && pf_scan_wy7.is_none()
        && gdn_pf_wy6p_on()
        && gdn_pf_wy6_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy6p_kernels()
    } else {
        None
    };
    let pf_scan_wy6 = if pf_scan_wy6p.is_none()
        && gdn_pf_wy6_on()
        && gdn_pf_wy6_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy6_kernels()
    } else {
        None
    };
    let pf_scan_wy5 = if pf_scan_wy6p.is_none()
        && pf_scan_wy6.is_none()
        && gdn_pf_wy5_on()
        && gdn_pf_wy5_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy5_kernels()
    } else {
        None
    };
    let pf_scan_wy4 = if pf_scan_wy5.is_none()
        && gdn_pf_wy4_on()
        && gdn_pf_wy4_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy4_kernels()
    } else {
        None
    };
    let pf_scan_wy = if pf_scan_wy4.is_none()
        && gdn_pf_wy_on()
        && gdn_pf_wy_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_wy_kernels()
    } else {
        None
    };
    let pf_scan_fast3 = if pf_scan_wy.is_none()
        && gdn_pf3_on()
        && gdn_pf3_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf3_kernels()
    } else {
        None
    };
    let pf_scan_fast2 = if pf_scan_fast3.is_none()
        && gdn_pf2_on()
        && gdn_pf2_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf2_kernels()
    } else {
        None
    };
    let pf_scan_fast = if pf_scan_fast2.is_none()
        && gdn_pf_fast_on()
        && gdn_pf_fast_eligible(nt, nk, nv, dk, dv, d_conv, conv_dim)
    {
        gdn_pf_kernels()
    } else {
        None
    };
    if nt >= GDN_CHUNK_MIN_TOKENS
        && pf_scan_wy7q.is_none()
        && pf_scan_wy7.is_none()
        && pf_scan_wy6p.is_none()
        && pf_scan_wy6.is_none()
        && pf_scan_wy5.is_none()
        && pf_scan_wy4.is_none()
        && pf_scan_wy.is_none()
        && pf_scan_fast3.is_none()
        && pf_scan_fast2.is_none()
        && pf_scan_fast.is_none()
    {
        // 状態は 1 つも触っていない (scratch のゼロ埋めだけ)。
        return Ok(false);
    }
    // 重みは 3 本とも「デバイスに載る」ことを先に確かめる。ここまでの失敗は
    // 状態を 1 つも触っていないので `Ok(false)` で従来経路に戻せる。
    let (wqkv, wgate, wo) = match (
        weights_for_gguf(wq.qkv.uid, wq.qkv.ty, wq.qkv.blocks, wq.qkv.k, wq.qkv.n),
        weights_for_gguf(wq.gate.uid, wq.gate.ty, wq.gate.blocks, wq.gate.k, wq.gate.n),
        weights_for_gguf(wout.uid, wout.ty, wout.blocks, wout.k, wout.n),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        (a, b, c) => {
            for r in [a.err(), b.err(), c.err()].into_iter().flatten() {
                tracing::error!("GDN prefill chain: 重みの登録に失敗: {r:#}");
            }
            return Ok(false);
        }
    };
    // 連結済みのときは qkv と gate が同じ行列を指す。行オフセットを
    // バイトに直して GEMM の重み先頭をずらす (カーネルは `w + row*row_bytes`
    // で行を読むので、部分行列をそのまま渡せる)。
    let Some(rb_qkv) = gguf_row_bytes(wq.qkv.ty, e) else {
        return Ok(false);
    };
    let qkv_woff = wq.qkv_row * rb_qkv;
    let gate_woff = wq.gate_row * rb_qkv;
    // ここから先はデバイスの作業域だけを触る (モデルの状態は不変)。
    let ptrs = GDN_SCRATCH.with(|cell| -> Result<[CUdeviceptr; 6]> {
        let mut s = cell.borrow_mut();
        s.ensure_chain(conv_dim, nv, dk, dv, cs)?;
        // スキャンカーネルは「書かない領域」を残すので従来どおりゼロ埋め。
        s.buf(GdnBuf::ConvOut).fill_zero()?;
        s.buf(GdnBuf::SsmOut).fill_zero()?;
        gdn_put_weights(&s, li, [wconv, wa, wdt, wnorm])?;
        Ok(s.chain_ptrs())
    })?;
    // task 186: beta/alpha をデバイスで作る場合の重み常駐 (失敗は None → ホスト経路)。
    // task 228: 元の型を問わず f32 へ展開して常駐する (`pf_ba_weight_dev`)。
    // 判定は呼び出し側と同じ純関数 [`pf_ba_w_check`] を使う。
    let ba_dev = if ba_dev_req {
        beta_alpha.and_then(|ba| {
            let w = ba.w;
            // task 225: alpha が別の行列 (未連結) のときは 2 本を別々に常駐する。
            let wa = match ba.alpha_w {
                Some(aw) => aw,
                None => w,
            };
            if let Some(why) = pf_ba_w_check(w.ty, w.k, w.n, &ba.beta_rows, nv, e)
                .or_else(|| pf_ba_w_check(wa.ty, wa.k, wa.n, &ba.alpha_rows, nv, e))
            {
                tracing::warn!("GDN prefill chain: beta/alpha をデバイスに置けない: {why}");
                return None;
            }
            let (p, rb) = pf_ba_weight_dev(w.uid, w.ty, w.blocks, w.k, w.n)?;
            let (pa, rba) = if ba.alpha_w.is_some() {
                pf_ba_weight_dev(wa.uid, wa.ty, wa.blocks, wa.k, wa.n)?
            } else {
                (p, rb)
            };
            Some((
                p + (ba.beta_rows.start * rb) as CUdeviceptr,
                pa + (ba.alpha_rows.start * rba) as CUdeviceptr,
            ))
        })
    } else {
        None
    };
    // ホスト側の beta/alpha を作らせたのにデバイス版が使えない場合は、ここで
    // 連鎖を諦める (状態は未着手。呼び出し側が従来経路で作り直す)。空のスライスで
    // H2D すると古い中身のままスキャンに入ってしまうので、必ず落とす。
    if ba_dev_req && ba_dev.is_none() {
        return Ok(false);
    }
    // task 317: 直前の attn mix を D2H していなければ (`DENNO_PF_MIXKEEP_SSM`)、
    // デバイスの `mixed` をそのまま入力にする。`b.x` への H2D 1 本 (nt × e × 4 B)
    // が消えるだけで、値は同じバイト列。
    let xdev = if pf_mix_keep_ssm_on() {
        pf_mixed_dev_in_ptr(flat.as_ptr(), nt, e)
    } else {
        None
    };
    // 記録は「デバイスだけが正」なのに dev-in が取れない (作業域の形が合わない等)
    // ときは、古いホストのバイト列を上げないように先に降ろす。降ろせなければ
    // 何も書いていないここで要求を落とす (誤った値で走らせない)。
    if xdev.is_none() && pf_mixed_dev_only(flat.as_ptr(), nt, e) {
        if !pf_mixed_restore_host(flat.as_ptr() as *mut f32, nt, e) {
            bail!("GDN 連鎖: D2H していない mixed の復元に失敗 (li={li})");
        }
    }
    gdn_pf_kprof::begin(gdn_pf_kprof::X_H2D);
    let ptr = GDN_PF_CHAIN_BUF.with(|cell| -> Result<[CUdeviceptr; 7]> {
        let mut b = cell.borrow_mut();
        b.ensure(nt, e, conv_dim, vd, nv)?;
        if xdev.is_none() {
            b.x.as_ref().unwrap().copy_from(flat)?;
        }
        if ba_dev.is_none() {
            b.beta.as_ref().unwrap().copy_from(beta)?;
            b.alpha.as_ref().unwrap().copy_from(alpha)?;
        }
        Ok([
            xdev.unwrap_or_else(|| b.buf(0)),
            b.buf(1),
            b.buf(2),
            b.buf(3),
            b.buf(4),
            b.buf(5),
            b.buf(6),
        ])
    })?;
    gdn_pf_kprof::end(gdn_pf_kprof::X_H2D);
    let (px, po, pqkv, pz, py, pb, pa) = (ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6]);

    // task 169: 3 本の投影がすべてテンソルコア GEMM の前提を満たすときだけ、
    // 1 起動でチャンク全体 (最大 TC_MAX_TOKENS 行) を回す。満たさないときは
    // 従来どおり 128 トークンずつ (旧カーネルの上限)。
    // task 176: v2 カーネルは Q4_K / Q5_K も扱えるので、判定は
    // `pf_gemm_fast_ok` (v2 → br169 の順) に任せる。
    let tc_ready = pf_gemm_fast_ok(wq.qkv.ty, nt, e, conv_dim, pf_tc_min_m())
        && pf_gemm_fast_ok(wq.gate.ty, nt, e, vd, pf_tc_min_m())
        && pf_gemm_fast_ok(wout.ty, nt, vd, e, pf_tc_min_m());
    let gemm_tile = if tc_ready { TC_MAX_TOKENS } else { 128 };

    // ---- qkv / z 投影 (デバイス → デバイス) --------------------------------
    {
        let _s = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_PF_QKV);
        let t_ph = Instant::now();
        let mut t0 = 0usize;
        while t0 < nt {
            let m = (nt - t0).min(gemm_tile);
            let xo = px + (t0 * e * 4) as CUdeviceptr;
            let qo = pqkv + (t0 * conv_dim * 4) as CUdeviceptr;
            let zo = pz + (t0 * vd * 4) as CUdeviceptr;
            for (idx, (w, wty, n_out, o)) in [
                (
                    wqkv.buf.ptr() + qkv_woff as CUdeviceptr,
                    wq.qkv.ty,
                    conv_dim,
                    qo,
                ),
                (
                    wgate.buf.ptr() + gate_woff as CUdeviceptr,
                    wq.gate.ty,
                    vd,
                    zo,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                // task 173: 段ごとの GPU 時間 (イベント)。qkv と z を分けて積む。
                let kp = if idx == 0 {
                    gdn_pf_kprof::QKV_GEMM
                } else {
                    gdn_pf_kprof::Z_GEMM
                };
                gdn_pf_kprof::begin(kp);
                let r = gguf_gemm_dev_launch(&ctx, wty, w, xo, o, m, e, n_out, gdn_pf_tc1_on());
                gdn_pf_kprof::end(kp);
                match r {
                    Ok(true) => {}
                    Ok(false) => return Ok(false),
                    Err(err) => {
                        tracing::error!("GDN prefill chain: 投影の起動に失敗: {err:#}");
                        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Ok(false);
                    }
                }
            }
            t0 += m;
        }
        // task 186: beta/alpha (F32) を同じ x から作る。入力も出力もデバイスに
        // あるので転送は 0。無い/形が違うときは連鎖ごと従来経路へ (状態は未着手)。
        if let Some((wbeta, walpha)) = ba_dev {
            // task 225: 専用 GEMM (`DENNO_PF_BA_FAST`) が撃てればそちら。
            // 撃てないときは従来の `dno_gemm_f32_mt` (lab t183a: ホストより遅い)。
            gdn_pf_kprof::begin(gdn_pf_kprof::BA_GEMM);
            let mut ba_fast_done = false;
            if pf_ba_fast_on() {
                ba_fast_done = pf_ba_gemm_dev_launch(ctx, wbeta, px, pb, nt, e, nv)?
                    && pf_ba_gemm_dev_launch(ctx, walpha, px, pa, nt, e, nv)?;
            }
            if !ba_fast_done {
                let Some(f32g) = gemm_f32_mt_fn(ctx) else {
                    return Ok(false);
                };
                for (w, o) in [(wbeta, pb), (walpha, pa)] {
                    let (mut pw, mut pxx, mut poo) = (w, px, o);
                    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, e as c_int, nv as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut pxx as *mut _ as *mut c_void,
                        &mut poo as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut k_c as *mut _ as *mut c_void,
                        &mut n_c as *mut _ as *mut c_void,
                    ];
                    let grid = (
                        ((nv + GEMM_MT_ROWS - 1) / GEMM_MT_ROWS) as u32,
                        ((nt + GEMM_MT_TOKS - 1) / GEMM_MT_TOKS) as u32,
                        1,
                    );
                    unsafe {
                        ctx.launch_async(f32g, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)?;
                    }
                }
            }
            gdn_pf_kprof::end(gdn_pf_kprof::BA_GEMM);
        }
        // 段ごとの壁時計を意味のある値にするため、ここで 1 回だけ待つ
        // (1 チャンク = 1 段 40 ms 級なので、同期の固定費は無視できる)。
        // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (次は同じストリーム)。
        pf_chain_sync(&ctx)?;
        phase_ns[0] = t_ph.elapsed().as_nanos() as u64;
    }

    // ---- GDN スキャン (状態はデバイス正本) ----------------------------------
    let (pst_conv, pst_ssm) =
        gdn_state_begin(kv, li, conv_st, ssm_st, conv_dim * cs, nv, dk * dv)?;
    // スキャンカーネルの選択 (`pf_scan_wy` / `pf_scan_fast3` / `pf_scan_fast2` /
    // `pf_scan_fast`) は task 270 で**投影より前**へ移した (大きい nt で撃てる
    // カーネルが無いときに、状態を進めずに降りるため)。
    {
        let _s = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_PF_SCAN);
        let t_ph = Instant::now();
        if let Some(fns) = pf_scan_wy7q {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY6_BLOCKS, GDN_PF_WY4_THREADS, GDN_PF_WY7Q_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy7 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY6_BLOCKS, GDN_PF_WY4_THREADS, GDN_PF_WY7_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy6p {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY6_BLOCKS, GDN_PF_WY6_THREADS, GDN_PF_WY6_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy6 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY6_BLOCKS, GDN_PF_WY6_THREADS, GDN_PF_WY6_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy5 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY4_BLOCKS, GDN_PF_WY4_THREADS, GDN_PF_WY5_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy4 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                GDN_PF_WY4_BLOCKS, GDN_PF_WY4_THREADS, GDN_PF_WY4_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_wy {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_wy_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
                1, GDN_PF_WY_THREADS, GDN_PF_WY_SMEM_BYTES as u32,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_fast3 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf3_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_fast2 {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf2_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else if let Some(fns) = pf_scan_fast {
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            gdn_pf_scan128_launch(
                &ctx, fns, pqkv, pz, pb, pa, pst_conv, pst_ssm, ptrs[0], ptrs[1], ptrs[2],
                ptrs[3], py, ptrs[4], ptrs[5], nt, nk, nv, conv_dim, eps, q_scale,
            )?;
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        } else {
            let (mut pq, mut pzz, mut pbb, mut paa) = (pqkv, pz, pb, pa);
            let (mut pci, mut psi) = (pst_conv, pst_ssm);
            let (mut pwc, mut pwa, mut pwd, mut pwn) = (ptrs[0], ptrs[1], ptrs[2], ptrs[3]);
            let (mut pyy, mut pco, mut pso) = (py, ptrs[4], ptrs[5]);
            let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
                nt as c_int,
                nk as c_int,
                nv as c_int,
                dk as c_int,
                dv as c_int,
            );
            let (mut cdi, mut dci, mut csi) = (conv_dim as c_int, d_conv as c_int, cs as c_int);
            let (mut epsi, mut qsi) = (eps, q_scale);
            let mut args = [
                &mut pq as *mut _ as *mut c_void,
                &mut pzz as *mut _ as *mut c_void,
                &mut pbb as *mut _ as *mut c_void,
                &mut paa as *mut _ as *mut c_void,
                &mut pci as *mut _ as *mut c_void,
                &mut psi as *mut _ as *mut c_void,
                &mut pwc as *mut _ as *mut c_void,
                &mut pwa as *mut _ as *mut c_void,
                &mut pwd as *mut _ as *mut c_void,
                &mut pwn as *mut _ as *mut c_void,
                &mut pyy as *mut _ as *mut c_void,
                &mut pco as *mut _ as *mut c_void,
                &mut pso as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
                &mut nki as *mut _ as *mut c_void,
                &mut nvi as *mut _ as *mut c_void,
                &mut dki as *mut _ as *mut c_void,
                &mut dvi as *mut _ as *mut c_void,
                &mut cdi as *mut _ as *mut c_void,
                &mut dci as *mut _ as *mut c_void,
                &mut csi as *mut _ as *mut c_void,
                &mut epsi as *mut _ as *mut c_void,
                &mut qsi as *mut _ as *mut c_void,
            ];
            let shmem = ((3 * dk + dv + 5) * 4) as u32;
            gdn_pf_kprof::begin(gdn_pf_kprof::SCAN);
            unsafe {
                ctx.launch_async(
                    f_scan,
                    (nv as u32, 1, 1),
                    (dk as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
            gdn_pf_kprof::end(gdn_pf_kprof::SCAN);
        }
        // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (out_proj は同じストリーム)。
        pf_chain_sync(&ctx)?;
        phase_ns[1] = t_ph.elapsed().as_nanos() as u64;
    }

    // ---- out_proj (デバイス → デバイス) ------------------------------------
    let t_out = Instant::now();
    {
        let _s = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_PF_OUT);
        let mut t0 = 0usize;
        while t0 < nt {
            let m = (nt - t0).min(gemm_tile);
            let yo = py + (t0 * vd * 4) as CUdeviceptr;
            let oo = po + (t0 * e * 4) as CUdeviceptr;
            gdn_pf_kprof::begin(gdn_pf_kprof::OUT_GEMM);
            let r = gguf_gemm_dev_launch(&ctx, wout.ty, wo.buf.ptr(), yo, oo, m, vd, e, gdn_pf_tc1_on());
            gdn_pf_kprof::end(gdn_pf_kprof::OUT_GEMM);
            match r {
                Ok(true) => {}
                Ok(false) => {
                    // スキャンは既に走っている → 状態が進んでいる。落とす。
                    GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
                    bail!("GDN prefill chain: out_proj の形がカーネルの前提外 (スキャン後)");
                }
                Err(err) => {
                    GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Err(err.context("GDN prefill chain: out_proj の起動に失敗"));
                }
            }
            t0 += m;
        }
        // out_proj の完了待ちもこの段に含める (台帳の行を「発行だけ」にしない)。
        // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (ホストは同期 D2H で読む)。
        pf_chain_sync(&ctx)?;
    }
    phase_ns[2] = t_out.elapsed().as_nanos() as u64;

    // ---- 層の出力と状態をホストへ ------------------------------------------
    // task 200: `DENNO_PF_ATTN_DEVRES` のときは層出力をデバイスに残し、直後の
    // combine にそのまま読ませる (D2H + H2D の往復を消す)。数値は同じ。
    *kept_dev = pf_attn_dev_requested() && crate::model::qwen4exp::pf_attn_devres_on();
    if *kept_dev {
        pf_attn_dev_store(po, nt, e);
    } else {
        gdn_pf_kprof::begin(gdn_pf_kprof::OUT_D2H);
        let got: Vec<f32> = GDN_PF_CHAIN_BUF.with(|cell| -> Result<Vec<f32>> {
            let b = cell.borrow();
            b.o.as_ref().expect("連鎖の出力").to_vec::<f32>(nt * e)
        })?;
        gdn_pf_kprof::end(gdn_pf_kprof::OUT_D2H);
        out[..nt * e].copy_from_slice(&got);
    }
    gdn_state_commit(li, ptrs[4], ptrs[5], conv_dim * cs, nv, dk * dv)?;
    // prefill のあとに decode (別ストア) や CPU 経路がホスト state を seed に
    // するので、状態は毎回ホストへも戻す (従来の sig 経路と同じ)。
    // task 379: `DENNO_PF_GDN_WB_LAST` では層の最終チャンクだけ戻す (途中チャンクの
    // ホスト控えは次チャンクが読まない。値は同じ)。飛ばした回数を数える。
    if pf_gdn_last_chunk_skippable(li) {
        crate::model::exp_pf::note_gdn_wb_skip();
    } else {
        gdn_pf_kprof::begin(gdn_pf_kprof::ST_D2H);
        gdn_host_writeback(conv_st, ssm_st)?;
        gdn_pf_kprof::end(gdn_pf_kprof::ST_D2H);
    }
    // task 173: 段ごとの GPU 時間を回収する (この時点で全イベントが完了している)。
    gdn_pf_kprof::collect();
    Ok(true)
}


// ---------------------------------------------------------------------------
// A1-r2: GDN chunked scan (長 prefill 用・3-pass 構成)
// ---------------------------------------------------------------------------

/// chunked ルーティングの閾値 (n がこれ以上で 3-pass chunked カーネルを使用)。
/// 実際の chunk 分割は [`crate::gdn::scan_schedule::plan`] (CPU 参照と共有する
/// 唯一の定義) が返す。
pub(crate) const GDN_CHUNK_MIN_TOKENS: usize = crate::gdn::scan_schedule::CHUNK_MIN_TOKENS;


/// s72: GDN スキャンの y をデバイス上に残したまま out_proj までチェーンで実行し、
/// 結果だけを host へ落とす。従来は y (n×vd) を D2H し、次の out_proj で再び H2D
/// していた (1 層あたり 2×47MB)。
///
/// 演算は従来と同一 (同じ `f_quant` + `f_gemm`・同一 int8 累積) なので出力は
/// ビット一致する。conv/ssm 状態の更新はスキャンカーネルが行うため、スキャン開始後に
/// 失敗するとフォールバックでは状態を巻き戻せない。よって前提条件 (重みのアップロード・
/// バッファ確保・ctx) はすべてスキャン前に確認し、以降の失敗は GPU 障害として扱う。
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_batch_oproj(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    o: &crate::quant::Quant8Linear,
    out: &mut [f32],
    li: usize,
    kv: u64,
) -> Option<()> {
    if !cuda_enabled() || n < GDN_CHUNK_MIN_TOKENS || dk != dv || d_conv != 4 {
        return None;
    }
    let vd = nv.checked_mul(dv)?;
    if o.k != vd || out.len() != n.checked_mul(o.n)? {
        return None;
    }
    let ctx = chain_ctx()?;
    if ctx.make_current().is_err() {
        return None;
    }
    ok(chain_ensure_buf(CH_PFH, n.checked_mul(vd)?.checked_mul(4)?))?;
    ok(chain_ensure_buf(CH_O, n.checked_mul(o.n)?.checked_mul(4)?))?;
    ok(weights_for_group(&[o.uid], &[o], o.k, o.n))?;
    let yptr = CHAIN.with(|cell| cell.borrow().bufs[CH_PFH].as_ref().map(|b| b.ptr()))?;

    if let Err(e) = gdn_scan_chunked_impl(
        ctx,
        GdnScanSrc::Host { qkv, z, beta, alpha },
        conv_st,
        ssm_st,
        wconv,
        wa,
        wdt,
        wnorm,
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        Some(yptr),
        li,
        kv,
    ) {
        tracing::error!("GDN scan (device y) failed: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    if chain_gemm_grouped_pf(&[o], CH_PFH, CH_O, n).is_none() {
        tracing::error!("GDN out_proj (chain) failed");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let bytes = out.len().checked_mul(4)?;
    let optr = CHAIN.with(|cell| cell.borrow().bufs[CH_O].as_ref().map(|b| b.ptr()))?;
    let lib = lib().ok()?;
    unsafe {
        ok(check(
            (lib.memcpy_dtoh_async)(
                out.as_mut_ptr() as *mut c_void,
                optr,
                bytes,
                std::ptr::null_mut(),
            ),
            "cuMemcpyDtoHAsync (gdn oproj)",
        ))?;
    }
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    ctx.sync().ok()?;
    Some(())
}


/// per-op 経路の行タイル (この単位で H2D + GEMM を回す)。
///
/// s168 (P7): 実際の dispatch は `DENNO_Q4_M_TILE` の実効値 ([`q4_m_tile`]) を使う。
/// この定数は**入口ゲート** ([`gdn_prefill_q4_geom`] / [`pf_res_gate`]) が前提にする
/// 歴史的なタイル幅 (1024) で、ゲートは端数タイルの帯域だけを見る保守的な判定
/// なので、実効タイルがこれと違っても「受けない」側にしか倒れない
/// (受けた場合は [`q4_group_rows`] が端数を直前のタイルへ併合する)。
pub(crate) const GDN_PREFILL_TILE: usize = 1024;


/// `chain_gdn_prefill_q4` が使う寸法 (入口条件を通ったときだけ作られる)。
pub(crate) struct GdnPrefillQ4Geom<'a> {
    qw: &'a crate::quant::Q4Linear,
    zw: &'a crate::quant::Q4Linear,
    ow: &'a crate::quant::Q4Linear,
    pub(crate) e: usize,
    pub(crate) conv_dim: usize,
    pub(crate) vd: usize,
    pub(crate) cs: usize,
}


/// `chain_gdn_prefill_q4` の入口条件 (重み種別・形・寸法・端数タイル) を判定し、
/// 以降で使う寸法を返す。**純粋関数** — CUDA もグローバル状態も触らないので、
/// フォールバック契約 (None になる条件) を GPU 無しのホストでテストできる。
/// 本番経路はこの戻り値だけを見て判断する (条件の写しを持たない)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_prefill_q4_geom<'a>(
    qkv_w: &'a crate::quant::Linear,
    z_w: &'a crate::quant::Linear,
    o_w: &'a crate::quant::Linear,
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> Option<GdnPrefillQ4Geom<'a>> {
    use crate::quant::Linear;
    let (qw, zw, ow) = match (qkv_w, z_w, o_w) {
        (Linear::Q4(a), Linear::Q4(b), Linear::Q4(c)) => (a, b, c),
        _ => return None,
    };
    // チャンク経路の前提 (gdn_scan_batch と同じ) + 形状。
    if n < GDN_CHUNK_MIN_TOKENS || nk == 0 || nv == 0 || dk == 0 || dv == 0 {
        return None;
    }
    if dk != dv || d_conv != 4 {
        return None;
    }
    let cs = d_conv - 1;
    let e = qw.k;
    let conv_dim = nk.checked_mul(dk)?.checked_mul(2)?.checked_add(nv.checked_mul(dv)?)?;
    let vd = nv.checked_mul(dv)?;
    // Q4 GEMM の入口条件 (maybe_gpu_linear_q4_into と同じ k%32==0) と重み形状。
    if e == 0 || e % 32 != 0 || vd % 32 != 0 {
        return None;
    }
    if qw.n != conv_dim || zw.k != e || zw.n != vd || ow.k != vd || ow.n != e {
        return None;
    }
    // per-op 経路は 1024 行タイルで、端数タイルが m <= max_m() だと GEMV になる。
    // そこだけ演算列が変わるので、その形では発動しない (従来経路へ落とす)。
    let rem = n % GDN_PREFILL_TILE;
    if rem != 0 && rem <= max_m() {
        return None;
    }
    Some(GdnPrefillQ4Geom { qw, zw, ow, e, conv_dim, vd, cs })
}


/// s164 (P1): GDN 層の prefill を **q4 重みで** デバイス常駐に通す。
///
/// 旧経路 (q4・`forward_prefill` の GDN 分岐) は 1 層あたり
///   qkv: D2H n*conv_dim*4 → H2D 同量 / z: D2H n*vd*4 → H2D 同量 /
///   y: D2H n*vd*4 → H2D 同量 / o_proj: D2H n*e*4
/// を往復していた (27B・n=13000 で 1 層 約 2.0 GB)。新経路は
///   H2D x (n*e*4) + H2D beta/alpha (n*nv*4 x2) + D2H o (n*e*4)
/// だけで、qkv / z / y はデバイス上に留まる。
///
/// ビット一致の根拠:
/// - qkv / z / o の投影は per-op 経路 (`q4_dispatch_tile`) と同一カーネル
///   (`dno_gemm_q4_0`)・同一引数。行ブロック (64 行) は互いに独立で、1 要素の
///   i32 累積順は k 昇順のまま変わらないため、grid の行ブロック数が増えても
///   出力は 1 要素ずつ同一。x のアップロードを 1024 行タイルに割るのも同じ理由で
///   数値を変えない。
///   (per-op 経路は m <= max_m() のタイルを GEMV で回すため、その帯域に
///   入る端数行がある n では発動しない → None)
/// - GDN スキャンは **n を分割しない**。chunk 分割 `plan(n)`・grid・引数・
///   カーネルとも従来の `gdn_scan_batch` と同一で、変わるのは入力 4 本の
///   出所 (`GdnScanSrc::Dev`) だけ。chunk 境界は状態の持ち越しで決まるので
///   行タイルに割ると plan が変わり数値が変わる — ここは触らない。
/// - beta / alpha は従来どおり呼び出し側がホストで計算した値をそのまま使う
///   (値も契約も不変)。
///
/// 状態 (conv/ssm) の契約は従来と同じ: `gdn_scan_chunked_impl` が
/// `gdn_state_begin` / `gdn_state_commit` / (devstate 無効時のみ)
/// `gdn_host_writeback` を通す。スキャン開始後に失敗すると状態を巻き戻せない
/// ため、前提条件はすべてスキャン前に確認する (失敗は GPU 障害として扱う)。
#[allow(clippy::too_many_arguments)]
pub fn chain_gdn_prefill_q4(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &[f32],
    alpha: &[f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    x: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    out: &mut [f32],
    li: usize,
    kv: u64,
) -> Option<()> {
    let e = gate_dims_in(qkv_w)?;
    if x.len() < n.checked_mul(e)? || out.len() < n.checked_mul(e)? {
        return None;
    }
    chain_gdn_prefill_q4_impl(
        qkv_w,
        z_w,
        o_w,
        beta,
        alpha,
        wconv,
        wa,
        wdt,
        wnorm,
        PfSrc::Host(x),
        PfSink::Host(out),
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        conv_st,
        ssm_st,
        li,
        kv,
        PfGdnPhase::All,
    )
}


/// s167 (P6): [`chain_gdn_prefill_q4_impl`] の実行位相。
///
/// GDN 層の prefill は「qkv/z GEMM (h2 だけに依存)」→「beta/alpha H2D → chunk
/// scan → o_proj (beta/alpha にも依存)」の 2 段に分かれており、beta/alpha は
/// ホストが h2 から計算する。h2 の D2H を専用コピーストリームに載せるとき、
/// 前段 (Gemm) だけを先に流してその間にコピーを進め、join 後に後段 (Rest) を
/// 流す。`All` は従来どおり 1 回で通す (フラグ OFF の経路はこれ)。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PfGdnPhase {
    /// 従来どおり全段 (qkv/z GEMM → beta/alpha H2D → scan → o_proj → D2H)
    All,
    /// qkv/z GEMM だけ (beta/alpha を待たない)
    Gemm,
    /// beta/alpha H2D → scan → o_proj (と Host 出力の D2H)
    Rest,
}


/// [`chain_gdn_prefill_q4`] の本体 (入出力の出所だけを差し替える)。
#[allow(clippy::too_many_arguments)]
fn chain_gdn_prefill_q4_impl(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &[f32],
    alpha: &[f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    src: PfSrc<'_>,
    mut sink: PfSink<'_>,
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    li: usize,
    kv: u64,
    phase: PfGdnPhase,
) -> Option<()> {
    // 形・重み種別の検査は gdn_prefill_q4_geom (純粋・GPU 不要) に集約してある。
    // フォールバック契約は tests::b20_stage1_tests::gdn_prefill_q4_geom_contract で固定する。
    // デバイスの準備 (cuda_enabled 等) は形が合ってから。
    let g = gdn_prefill_q4_geom(qkv_w, z_w, o_w, n, nk, nv, dk, dv, d_conv)?;
    let (qw, zw, ow) = (g.qw, g.zw, g.ow);
    let (e, conv_dim, vd, cs) = (g.e, g.conv_dim, g.vd, g.cs);
    // s168 (P7): 行タイルは `DENNO_Q4_M_TILE` の実効値。端数は直前のタイルへ併合
    // するので、CH_PFH (入力のステージ) は 1 タイルの最大行数で確保する。
    let tile = q4_m_tile();
    let cap = q4_tile_max_rows(tile).min(n);
    let nv_tot = n.checked_mul(nv)?;
    if beta.len() < nv_tot || alpha.len() < nv_tot {
        return None;
    }
    if wconv.len() != conv_dim.checked_mul(d_conv)? || wa.len() != nv || wdt.len() != nv {
        return None;
    }
    if wnorm.len() != dv {
        return None;
    }
    if let PfSrc::Host(x) = src {
        if x.len() < n.checked_mul(e)? {
            return None;
        }
    }
    if let PfSink::Host(out) = &sink {
        if out.len() < n.checked_mul(e)? {
            return None;
        }
    }
    if conv_st.len() != conv_dim.checked_mul(cs)? || ssm_st.len() != nv.checked_mul(dk)?.checked_mul(dv)? {
        return None;
    }
    // Slot 入力は CH_PFH の先頭 n 行をそのまま読むので、行数ぶんの領域を要求する
    // (rmsnorm が ensure 済みなら no-op。足りなければ None → 呼び出し側がフォールバック)。
    if matches!(src, PfSrc::Slot) {
        ok(chain_ensure_buf(CH_PFH, n.checked_mul(e)?.checked_mul(4)?))?;
    }
    // ここまでで「形が合っている」ことが確定した → デバイスの準備に入る。
    if !cuda_enabled()
        || !gemm_enabled()
        || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
    {
        return None;
    }
    let ctx = chain_ctx()?;
    let gw_qkv = weights_for_q4(qw.uid, &qw.blocks, qw.k, qw.n).ok()?;
    let gw_z = weights_for_q4(zw.uid, &zw.blocks, zw.k, zw.n).ok()?;
    let gw_o = weights_for_q4(ow.uid, &ow.blocks, ow.k, ow.n).ok()?;
    let (n_chunks, _) = crate::gdn::scan_schedule::plan(n);
    // GDN スクラッチを先に確保し、qkv / z の GEMM 出力を**そのまま**スキャンの
    // 入力にする (中間バッファを増やさない)。impl 側の ensure は同形なので no-op。
    ok(GDN_SCRATCH.with(|cell| {
        cell.borrow_mut().ensure(n, conv_dim, vd, nv, dk, dv, cs, n_chunks)
    }))?;
    let (pq, pz, py, pb, pa) =
        GDN_SCRATCH.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            Some((
                s.q.as_ref()?.ptr(),
                s.z.as_ref()?.ptr(),
                s.y.as_ref()?.ptr(),
                s.beta.as_ref()?.ptr(),
                s.alpha.as_ref()?.ptr(),
            ))
        })?;
    ok(chain_ensure_buf(CH_PFH, cap.checked_mul(e)?.checked_mul(4)?))?;
    ok(chain_ensure_buf(CH_O, n.checked_mul(e)?.checked_mul(4)?))?;

    // 1) qkv / z 投影: 入力を 1 タイル (`DENNO_Q4_M_TILE`・端数は直前へ併合) ずつ
    //    用意し、GEMM は出力の行オフセットへ書く
    //    (per-op 経路の 1024 行タイルと同じ形。H2D と GEMM は同一ストリームで
    //    順序が保たれるので、タイル間の sync は要らない)。Host はタイルごとに
    //    H2D、Slot (残差常駐経路) は CH_PFH の該当行をそのまま読む — 転送だけが
    //    消え、カーネル・引数は同一。
    //    s167 (P6): 位相 Rest では再実行しない (位相 Gemm の呼び出しが済ませている。
    //    q/z は GDN_SCRATCH に残っており、間に GDN_SCRATCH を触る処理は無い)。
    if phase != PfGdnPhase::Rest {
        let mut r0 = 0usize;
        while r0 < n {
            let mr = q4_group_rows(r0, n, tile);
            debug_assert!(mr <= cap, "tile group {mr} > cap {cap}");
            CHAIN.with(|cell| -> Option<()> {
                let s = cell.borrow();
                let pbuf = s.bufs[CH_PFH].as_ref()?.ptr();
                let pin = match &src {
                    PfSrc::Host(x) => {
                        let lib = lib().ok()?;
                        // H2D x タイル → 同一ストリームの GEMM (順序はストリームが保証する)
                        unsafe {
                            ok(check(
                                (lib.memcpy_htod_async)(
                                    pbuf,
                                    x[r0 * e..].as_ptr() as *const c_void,
                                    mr * e * 4,
                                    std::ptr::null_mut(),
                                ),
                                "cuMemcpyHtoDAsync (gdn prefill q4 x)",
                            ))?;
                        }
                        pbuf
                    }
                    PfSrc::Slot => pbuf + (r0 * e * 4) as u64,
                };
                crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
                    chain_gemm_q4_pf(gw_qkv, pin, pq + (r0 * conv_dim * 4) as u64, mr, e, conv_dim)?;
                    chain_gemm_q4_pf(gw_z, pin, pz + (r0 * vd * 4) as u64, mr, e, vd)
                })
            })?;
            r0 += mr;
        }
    }
    // s167 (P6): 位相 Gemm はここで終わる (呼び出し側が h2 のコピーを join してから
    // 位相 Rest を呼ぶ)。beta/alpha の H2D もスキャンも o_proj も走らせない。
    if phase == PfGdnPhase::Gemm {
        return Some(());
    }

    // 2) beta / alpha (ホストで計算済み・従来と同じ値) をデバイスへ。
    //    2*n*nv*4 は 27B/13k でも 5 MB なので、ここは往復を消さない。
    GDN_SCRATCH.with(|cell| -> Option<()> {
        let s = cell.borrow();
        ok(s.beta.as_ref()?.copy_from(&beta[..nv_tot]))?;
        ok(s.alpha.as_ref()?.copy_from(&alpha[..nv_tot]))?;
        Some(())
    })?;

    // 3) GDN chunk scan (3 launch・入力はすべてデバイス、y もデバイスに残す)。
    //    s173 (P15): pass 間のホスト待ちは無い (同期は最後の 1 回だけ)。
    pf_phase(PF_PH_GDN_SCAN);
    crate::prof::time_layer(crate::prof::ATT_GDN_SCAN, li, || -> Option<()> {
        ok(gdn_scan_chunked_impl(
            ctx,
            GdnScanSrc::Dev { qkv: pq, z: pz, beta: pb, alpha: pa },
            conv_st,
            ssm_st,
            wconv,
            wa,
            wdt,
            wnorm,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            Some(py),
            li,
            kv,
        )
        .map(|_| ()))
    })?;

    // 4) o_proj (y はデバイス上) → 5) D2H は Host 出力のときだけ最後の 1 回。
    pf_phase(PF_PH_GDN_OPROJ);
    let po = CHAIN.with(|cell| cell.borrow().bufs[CH_O].as_ref().map(|b| b.ptr()))?;
    crate::prof::time_layer(crate::prof::OPROJ, li, || -> Option<()> {
        chain_gemm_q4_pf(gw_o, py, po, n, vd, e)
    })?;
    // Slot 出力 (残差常駐経路) は CH_O に残したまま、次の rmsnorm の proj 引数として
    // 読ませる — D2H も sync もしない (同じストリームなので順序は保たれる)。
    if let PfSink::Host(out) = &mut sink {
        CHAIN.with(|cell| -> Option<()> {
            let buf = cell.borrow().bufs[CH_O].as_ref()?.ptr();
            let lib = lib().ok()?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        n * e * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (gdn prefill q4 o)",
                ))
            }
        })?;
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        ok(ctx.sync())?;
    }
    Some(())
}

pub const PF_PH_GDN_PROJ: usize = 1;

pub const PF_PH_GDN_SCAN: usize = 2;

pub const PF_PH_GDN_OPROJ: usize = 3;


/// s167 (P6): GDN 層の prefill を、h2 の D2H と qkv/z GEMM の**オーバーラップ付き**
/// で通す版。`DENNO_PF_OVERLAP=1` かつデバイス準備が成功したときだけ
///   位相 Gemm → join → `host_ctrl` (beta/alpha) → 位相 Rest
/// の順で流し、それ以外は従来と**同一の順序**
///   同期 D2H → `host_ctrl` → (GEMM → beta/alpha H2D → scan → o_proj)
/// を通す。`host_ctrl` は h2 を受け取って beta/alpha を書くホスト区間で、呼び出し
/// 側は従来と同じ関数 (`apply_bt_rows_into`) を同じ引数で呼ぶ。
#[allow(clippy::too_many_arguments)]
pub fn chain_gdn_prefill_q4_dev_ovl<F>(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &mut [f32],
    alpha: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    li: usize,
    kv: u64,
    host_ctrl: F,
) -> Option<()>
where
    F: FnOnce(&[f32], &mut [f32], &mut [f32]) -> Option<()>,
{
    // 前提 (形・重み種別・n の帯域) はコピーを発行する前に確定させる — ここで
    // 落ちる形ではコピーを 1 バイトも発行しない (未 join のコピーを残さない)。
    let g = gdn_prefill_q4_geom(qkv_w, z_w, o_w, n, nk, nv, dk, dv, d_conv)?;
    let e = g.e;
    let n_e = n.checked_mul(e)?;
    // s169 (P8): 位相の境界は**発行グループの先頭**。ここから「h2 の D2H (同期 or
    // join) → ホストの beta/alpha → qkv/z GEMM」までが gdn_proj の区間になる。
    pf_phase(PF_PH_GDN_PROJ);
    let want = pf_overlap_flag();
    let begun = want && chain_pf_h2_begin(CH_PFH, 0, n_e).is_some();
    if pf_h2_path(want, n, e, begun) == PfH2Path::CopyStream {
        // 位相 Gemm を先に流す (h2 を待たない)。この間コピーストリームが h2 を運ぶ。
        // 既定ストリームは非ブロッキングのコピーストリームと暗黙同期しないので、
        // GEMM がコピーを待つことはない。
        if chain_gdn_prefill_q4_impl(
            qkv_w,
            z_w,
            o_w,
            beta,
            alpha,
            wconv,
            wa,
            wdt,
            wnorm,
            PfSrc::Slot,
            PfSink::Slot,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            conv_st,
            ssm_st,
            li,
            kv,
            PfGdnPhase::Gemm,
        )
        .is_none()
        {
            // 位相 Gemm の失敗 = デバイス障害 (従来のチェーン失敗と同じ扱い)。
            // 未 join のコピーは残さない。
            let _ = chain_pf_h2_join();
            return None;
        }
        // join: ホストが h2 を読む直前の唯一の同期点。
        if chain_pf_h2_join().is_some() {
            // pinned の h2 をそのまま渡す (Vec への写しをしない)。安全性: pin_h2 は
            // thread_local の grow-only バッファで、join 済み = 全要素がコピーで
            // 埋まっている。CHAIN の借用は保持しない (保持したままホスト区間を走ら
            // せると、内側の CHAIN アクセスが RefCell で panic する)。
            let (p, cap) = CHAIN.with(|cell| {
                cell.borrow()
                    .pin_h2
                    .as_ref()
                    .map(|p| (p.ptr as *const f32, p.bytes / std::mem::size_of::<f32>()))
            })?;
            if cap < n_e {
                return None;
            }
            let h2: &[f32] = unsafe { std::slice::from_raw_parts(p, n_e) };
            host_ctrl(h2, beta, alpha)?;
            return chain_gdn_prefill_q4_impl(
                qkv_w,
                z_w,
                o_w,
                beta,
                alpha,
                wconv,
                wa,
                wdt,
                wnorm,
                PfSrc::Slot,
                PfSink::Slot,
                n,
                nk,
                nv,
                dk,
                dv,
                d_conv,
                eps,
                q_scale,
                conv_st,
                ssm_st,
                li,
                kv,
                PfGdnPhase::Rest,
            );
        }
        // join だけ失敗した (driver 障害)。q/z は既にデバイスにあるので、h2 を
        // 従来の同期 D2H で取り直して残りを流す — 値は同一。
        let h2 = chain_download_at(CH_PFH, 0, n_e)?;
        host_ctrl(&h2, beta, alpha)?;
        return chain_gdn_prefill_q4_impl(
            qkv_w,
            z_w,
            o_w,
            beta,
            alpha,
            wconv,
            wa,
            wdt,
            wnorm,
            PfSrc::Slot,
            PfSink::Slot,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            conv_st,
            ssm_st,
            li,
            kv,
            PfGdnPhase::Rest,
        );
    }
    // 従来と同一の順序: 同期 D2H → ホスト計算 → GEMM → beta/alpha H2D → scan → o_proj
    let h2 = chain_download_at(CH_PFH, 0, n_e)?;
    host_ctrl(&h2, beta, alpha)?;
    chain_gdn_prefill_q4_impl(
        qkv_w,
        z_w,
        o_w,
        beta,
        alpha,
        wconv,
        wa,
        wdt,
        wnorm,
        PfSrc::Slot,
        PfSink::Slot,
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        conv_st,
        ssm_st,
        li,
        kv,
        PfGdnPhase::All,
    )
}


/// chunked scan の入力 4 本 (qkv / z / beta / alpha) の出所。
///
/// s164 (P1): 従来は必ずホスト slice で、`GDN_SCRATCH` へ H2D してから
/// カーネルへ渡していた。`Dev` は「既にデバイスに居る」場合にそのポインタを
/// そのまま使う — カーネルの引数・grid・演算列は 1 つも変わらず、消えるのは
/// 転送だけ (q4 prefill チェーン `chain_gdn_prefill_q4` が使う)。
#[derive(Clone, Copy)]
pub(crate) enum GdnScanSrc<'a> {
    Host {
        qkv: &'a [f32],
        z: &'a [f32],
        beta: &'a [f32],
        alpha: &'a [f32],
    },
    Dev {
        qkv: CUdeviceptr,
        z: CUdeviceptr,
        beta: CUdeviceptr,
        alpha: CUdeviceptr,
    },
}


// ---------------------------------------------------------------------------
// s240 (P40): GDN chunked 3 カーネルの実時間 (`DENNO_GDN_KPROF=1`・既定 OFF)
//
// 背景: 27B q4 prefill の `sl_lay_gdn_scan` 相は 6,965 ms (19.4%) だが、
// `DENNO_PROFILE` の `gdn_pf_*` では `sync` (13,774.9 ms / 2 リクエスト) まで
// しか割れず、「pass 1/2/3 のどれが実行時間を持っているか」は未計測だった。
// ここでは 3 pass の発行位置にだけイベントを打ち、既存の同期 (3 pass 後の
// 1 回・`cuda.rs` の `ctx.sync()`) が済んだ直後に `cuEventElapsedTime` で
// 3 区間を読む。
//
// s240 (P43): 出力点は **prefill 終了時** (`pf_timing_read`) が本線。
// P40 は dump (`prof::summary`) とスレッド終了フックにしか出しておらず、
// `scripts/prod_serve_27b.sh` が `unset DENNO_PROFILE` する本番構成では
// そのどちらも走らないため、計測できていても 1 行も見えなかった。
// 行には `miss=<n>` (捨てた call 数) を必ず出し、env ON なら
// calls=0 でも行が出る (静かな no-op を構造的に禁止)。
//
// 数値経路を触らない根拠:
// - 追加するのは `cuEventRecord` / `cuEventElapsedTime` の 2 API だけで、
//   launch の関数・引数・grid/block・発行順・同期の位置は 1 つも変えない
//   (イベントは 3 カーネルと同じ既定ストリームに打つので、ストリームに
//    挿入されるのは「時刻を記録するだけの操作」)。
// - 積算は表示専用の static にしか書かない (forward の戻り値・状態・
//   アンカー値には一切流れない)。
// - `DENNO_GDN_KPROF` が未設定なら `kprof_begin()` が None を返すだけで、
//   イベント生成も計測も起きない (各サイト分岐 1 回)。`DENNO_PROFILE` /
//   `DENNO_PF_DBG` の出力は env 未設定時に 1 バイトも変わらない。
// - 区間の意味は「カーネル実行 + 前のカーネルとの発行の隙間」。pass 間で
//   ホストが待たないので隙間は小さいが 0 ではない (数字はそのまま出す)。
// ---------------------------------------------------------------------------

/// `DENNO_GDN_KPROF=1` のときだけ 3 カーネルの実時間を測る (既定 OFF)。
pub(crate) fn gdn_kprof_on() -> bool {
    // テストから env を介さずに切り替える (`prof::enabled` と同じ流儀)。
    #[cfg(test)]
    if let Some(v) = kprof_tests::force_on() {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_GDN_KPROF").ok().as_deref() == Some("1"))
}


/// 積算の表示行 (`DENNO_GDN_KPROF` 未設定・未計測なら None)。
/// dump (`prof::summary`) と終了時の 1 行がこれを使う。
pub fn gdn_kprof_line() -> Option<String> {
    if !gdn_kprof_on() {
        return None;
    }
    kprof_acc().line()
}


/// 区間を読めずに捨てた call 数 (診断用。表示行には入れない)。
pub fn gdn_kprof_miss() -> u64 {
    KPROF_MISS.load(Ordering::Relaxed)
}


/// 積算を stderr へ 1 行だけ出す (未設定・未計測なら何もしない)。
pub fn gdn_kprof_report() {
    if let Some(l) = gdn_kprof_line() {
        eprintln!("{l}");
    }
}


/// prefill 終了時の 1 行 (s240/P43)。**サーバ経路の唯一の観測点**。
///
/// P40 は dump (`prof::summary()`) と終了フックにしか行を出していなかったため、
/// `scripts/prod_serve_27b.sh` が `unset DENNO_PROFILE` する本番構成では
/// `prof::enabled()` が偽 → `prof::summary()` が呼ばれず、ワーカースレッドも
/// 終了しないので **1 行も出なかった** (計測できていても見えない)。
/// ここは `pf_timing_read()` (prefill 末尾・最終行 D2H の後) から呼ぶ:
/// 読むのは表示専用の static だけなので**同期は 1 つも足さない**。
///
/// dump 経路 (`DENNO_PROFILE`) が生きているときは `prof::summary()` が同じ行を
/// 出すので、ここでは出さない (二重行の回避)。`DENNO_GDN_KPROF` 未設定なら
/// 先頭の分岐 1 回で戻る (イベント生成も出力も無し)。
pub fn gdn_kprof_report_at_prefill_end() {
    if !gdn_kprof_on() {
        return;
    }
    if let Some(l) = kprof_prefill_line(&kprof_acc(), crate::prof::enabled()) {
        eprintln!("{l}");
    }
}


/// A1-r2: GDN chunked scan 実装 (3 カーネル・長 prefill 用)。
///
/// CPU 参照: [`crate::gdn::scan_chunked_core`]。3-pass 構成:
/// - K1 `dno_gdn_chunk_pre` (chunk 並列): ローカル遷移 W[ci]・M[ci]
/// - K2 `dno_gdn_chunk_state` (chunk 直列): chunk 先頭状態 Sinit の直列伝播
/// - K3 `dno_gdn_chunk_scan` (chunk 並列): Sinit からの逐次 AR スキャン → y・
///   最終状態 (最終 chunk ブロックが書き戻す)
///
/// 決定論: 各カーネルとも launch 構成固定・昇順逐次和・atomic/ワープ縮約不使用
/// (gdn.cu の方針どおり)。演算は単発カーネル `dno_gdn_scan` と同一で、chunk
/// 境界でのみ状態を持ち越す。前提: dk == dv・d_conv == 4 (呼び出し側で保証)。
///
/// s173 (P15): **pass 間でホストを止めない**。3 pass はすべて既定ストリームへ
/// 発行するので、pass 2 が pass 1 の W/M を読み、pass 3 が pass 2 の Sinit を
/// 読む順序はストリームが保証する (同一ストリーム上の発行は FIFO 実行)。
/// s245 (P45): 同期点は「ホストが結果を読む経路」だけにした:
/// - デバイス出力 (`y_dst = Some`・prefill 残差常駐): **待たない**。y は後続の
///   o_proj が同じストリームで読み、状態の D2D 書き戻しも同じストリームなので
///   FIFO で足りる。従来の 1 回は「3 pass の実行時間を `DENNO_PROFILE` の
///   `gdn_pf_sync` に写す」ためだけに残していた (表示専用) — P45 で外した。
/// - ホスト出力 (`y_dst = None`) は `to_vec` の `cuMemcpyDtoH` (同期) で y を
///   読むので、その手前で 1 回待つ (従来どおり)。
/// 1 層あたり: launch 3 (不変) / 同期 0 (デバイス出力) または 1 (ホスト出力)。
/// カーネル・引数・grid・chunk 分割・発行順は 1 つも変えないので出力ビットは不変。
#[allow(clippy::too_many_arguments)]
fn gdn_scan_chunked_impl(
    ctx: &CudaCtx,
    src: GdnScanSrc<'_>,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    y_dst: Option<CUdeviceptr>,
    li: usize,
    kv: u64,
) -> Result<Vec<f32>> {
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    if let GdnScanSrc::Host { qkv, z, beta, alpha } = &src {
        debug_assert_eq!(qkv.len(), n * conv_dim);
        debug_assert_eq!(z.len(), n * vd);
        debug_assert_eq!(beta.len(), n * nv);
        debug_assert_eq!(alpha.len(), n * nv);
    }
    debug_assert_eq!(conv_st.len(), conv_dim * cs);
    debug_assert_eq!(ssm_st.len(), nv * dk * dv);
    debug_assert_eq!(dk, dv, "chunked scan は dk == dv 前提");
    debug_assert_eq!(d_conv, 4, "chunked scan は d_conv == 4 前提");

    let (n_chunks, chunk_len) = crate::gdn::scan_schedule::plan(n);

    // s90 (P1-1): 2 launch 経路 (`f_gdn_chunk_scan_hoist`) の入り口判定は
    // `crate::gdn::scan_schedule::route_hoist` の 1 か所。この forward は
    // 「3 launch 経路に対して別 launch の書き手」を保証できない
    // (gdn/scan_schedule.rs のモジュール冒頭) ので、route_hoist(.., false) = false
    // が既定 → 下の 3 pass が走る。将来 常駐を実装したときに、ここで
    // route_hoist(.., true) 側の分岐を足す (今は分岐を持たない = 死んだ
    // パスを疑似的に走らせない)。
    if crate::gdn::scan_schedule::route_hoist(n, nv, dk, dv, false) {
        unreachable!("s90: route_hoist(.., false) は常に偽 (既定は 3 launch)");
    }
    // 帳簿の健全性 (debug のみ): 境界行の書き手は読み手より前の chunk であること。
    // 「別 launch の書き手」になるのはこの条件が満たされたうえで前の launch が
    // 実際に存在する場合だけなので、ここでは順序関係だけを見る。
    debug_assert!(
        crate::gdn::scan_schedule::read_write_conflicts(n, chunk_len, n_chunks)
            .iter()
            .all(|(reader, writer, _)| writer < reader),
        "s90: 境界行の書き手が読み手より後になっている"
    );

    ctx.make_current()?;
    // s64 (T5b L1 Step 1 計測): `gdn_scan_impl` と同一の 4 区間細分 (表示専用)。
    // s173 (P15): 区間は「確保/アップロード (alloc) → 3 pass の発行 (launch) →
    // 同期 1 回の待ち (sync) → 状態の書き戻し・y の読み出し (d2h)」の 4 つ。
    // pass 間で待たないので、3 pass の実行時間は最後の sync にまとめて乗る。
    let mut q = crate::prof::Seq::begin(crate::prof::D2H_GDN_PF_ALLOC, Instant::now());
    // s67: 16 本の作業領域は層をまたいで再利用する (確保は初回のみ・形が同じ間は
    // 何もしない)。中身は毎回 copy_from で上書きするので従来と同じ状態になる。
    // w/m/sinit/y は各 pass が全要素を書くためゼロ埋め不要 (旧実装は
    // `from_slice(&vec![0.0; …])` でゼロを H2D していた — その転送も消える)。
    // conv_out/ssm_out だけは「最終 chunk ブロックのみ書き込み」なので従来どおり
    // ゼロで初期化する (未書込部は未使用・挙動を変えない)。
    let dp = GDN_SCRATCH.with(|cell| -> Result<GdnPtrs> {
        let mut s = cell.borrow_mut();
        s.ensure(n, conv_dim, vd, nv, dk, dv, cs, n_chunks)?;
        let p = s.ptrs();
        s.buf(GdnBuf::ConvOut).fill_zero()?;
        s.buf(GdnBuf::SsmOut).fill_zero()?;
        // s164: ホスト入力のときだけ H2D する (Dev は呼び出し側が既に置いている)。
        if let GdnScanSrc::Host { qkv, z, beta, alpha } = &src {
            s.q.as_ref().unwrap().copy_from(qkv)?;
            s.z.as_ref().unwrap().copy_from(z)?;
            s.beta.as_ref().unwrap().copy_from(beta)?;
            s.alpha.as_ref().unwrap().copy_from(alpha)?;
        }
        gdn_put_weights(&s, li, [wconv, wa, wdt, wnorm])?;
        Ok(p)
    })?;
    // Dev のときは入力 4 本のポインタだけ差し替える (他はスクラッチのまま)。
    let dp = match src {
        GdnScanSrc::Dev { qkv, z, beta, alpha } => GdnPtrs { q: qkv, z, beta, alpha, ..dp },
        GdnScanSrc::Host { .. } => dp,
    };
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    // ---- pass 1: chunk 並列でローカル遷移 (W[ci], M[ci]) --------------------
    let (pst_conv, pst_ssm) =
        gdn_state_begin(kv, li, conv_st, ssm_st, conv_dim * cs, nv, dk * dv)?;
    let (mut pq, mut pb, mut pa) = (dp.q, dp.beta, dp.alpha);
    let (mut pci, mut pwc, mut pwa, mut pwd) = (pst_conv, dp.wc, dp.wa, dp.wd);
    let (mut pw, mut pm) = (dp.w, dp.m);
    let (mut ni, mut cli, mut nki, mut nvi, mut dki, mut dvi) = (
        n as c_int,
        chunk_len as c_int,
        nk as c_int,
        nv as c_int,
        dk as c_int,
        dv as c_int,
    );
    let (mut cdi, mut csi, mut epsi) = (conv_dim as c_int, cs as c_int, eps);
    let mut args_pre = [
        &mut pq as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pw as *mut _ as *mut c_void,
        &mut pm as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut cli as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
    ];
    let shmem_pre = ((2 * dk + 3) * 4) as u32;
    // s239 (P36): 4 レーン版 (DK=128 専用) は `DENNO_GDN_LANES=4` のときだけ。
    // block は列 dk × 4 レーン。数値は旧版と bit 一致しない (木縮約)。
    // P55: 共有メモリ版 (`DENNO_GDN_PRE_SM=1`・dk <= 128・opt-in 成功) は
    // 4 レーン版の**後ろ**に置く。4 レーン版が PTX にあり `DENNO_GDN_LANES=4` が
    // 立っていればそちらが勝ち、無ければ共有メモリ版のゲートへ進む
    // (両方 opt-in なので優先順位を明示しておく)。数値経路はどの枝でも
    // 下の launch 1 本だけで、カーネル・引数・grid・発行順は共通
    // (block と shmem だけが違う)。
    let smem_pre_sm = gdn_pre_sm_smem_bytes(dk);
    let (f_pre, blk_pre, shm_pre) = match (gdn_lanes4(), ctx.f_gdn_chunk_pre_l4) {
        (true, Some(f)) if dk == 128 && dv == 128 => (f, (dk as u32) * 4, shmem_pre),
        _ => {
            let use_sm = gdn_pre_sm_gate(
                gdn_pre_sm_enabled(),
                ctx.f_gdn_chunk_pre_sm.is_some(),
                dk,
                dv,
                smem_pre_sm,
                ctx.gdn_pre_sm_dyn_smem,
            );
            match (use_sm, ctx.f_gdn_chunk_pre_sm) {
                (true, Some(f)) => (f, dk as u32, smem_pre_sm as u32),
                _ => (ctx.f_gdn_chunk_pre, dk as u32, shmem_pre),
            }
        }
    };
    // s240 (P40): pass 1 の実時間 (`DENNO_GDN_KPROF=1` のときだけ・既定 OFF)。
    let mut kp = kprof_begin();
    kprof_mark(&mut kp, 0);
    unsafe {
        ctx.launch_async(
            f_pre,
            (n_chunks as u32, nv as u32, 1),
            (blk_pre, 1, 1),
            shm_pre,
            &mut args_pre,
        )?;
    }
    kprof_mark(&mut kp, 1);
    // s173 (P15): pass 1 → pass 2 は同一ストリームの発行順で足りる (pass 2 は
    // pass 1 が書いた W/M を読むが、既定ストリームは FIFO なので待つ必要が無い)。
    // 従来ここにあった `ctx.sync()` (cuCtxSynchronize = デバイス全体の drain) は
    // 削除した — 数値・発行順は不変で、消えたのはホストの待ちだけ。

    // ---- pass 2: chunk 先頭状態 Sinit の直列伝播 ----------------------------
    let (mut psi, mut pm2, mut pw2, mut psinit) = (pst_ssm, dp.m, dp.w, dp.sinit);
    let (mut nc2, mut nv2, mut dk2, mut dv2) =
        (n_chunks as c_int, nv as c_int, dk as c_int, dv as c_int);
    let mut args_state = [
        &mut psi as *mut _ as *mut c_void,
        &mut pm2 as *mut _ as *mut c_void,
        &mut pw2 as *mut _ as *mut c_void,
        &mut psinit as *mut _ as *mut c_void,
        &mut nc2 as *mut _ as *mut c_void,
        &mut nv2 as *mut _ as *mut c_void,
        &mut dk2 as *mut _ as *mut c_void,
        &mut dv2 as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            ctx.f_gdn_chunk_state,
            (nv as u32, 1, 1),
            (dk as u32, 1, 1),
            0,
            &mut args_state,
        )?;
    }
    kprof_mark(&mut kp, 2);
    // s173 (P15): pass 2 → pass 3 も同じ理由で待たない (pass 3 は pass 2 の
    // Sinit を読むが、発行順で足りる)。

    // ---- pass 3: Sinit[ci] からの逐次 AR スキャン (chunk 並列) ---------------
    let (mut pq3, mut pz3, mut pb3, mut pa3) = (dp.q, dp.z, dp.beta, dp.alpha);
    // s131: pass 3 の conv 入力もデバイス正本 (スクラッチの conv_in はもう書かない)。
    let (mut pci3, mut psinit3) = (pci, dp.sinit);
    let (mut pwc3, mut pwa3, mut pwd3, mut pwn3) = (dp.wc, dp.wa, dp.wd, dp.wn);
    // s72: y_dst があれば y はチェーンスロットへ直接書く (host へ落とさない)。
    let (mut py, mut pco, mut pso) = (y_dst.unwrap_or(dp.y), dp.conv_out, dp.ssm_out);
    let (mut ni3, mut cli3) = (n as c_int, chunk_len as c_int);
    let (mut nki3, mut nvi3, mut dki3, mut dvi3) =
        (nk as c_int, nv as c_int, dk as c_int, dv as c_int);
    let (mut cdi3, mut csi3, mut epsi3, mut qsi3) =
        (conv_dim as c_int, cs as c_int, eps, q_scale);
    let mut args_scan = [
        &mut pq3 as *mut _ as *mut c_void,
        &mut pz3 as *mut _ as *mut c_void,
        &mut pb3 as *mut _ as *mut c_void,
        &mut pa3 as *mut _ as *mut c_void,
        &mut pci3 as *mut _ as *mut c_void,
        &mut psinit3 as *mut _ as *mut c_void,
        &mut pwc3 as *mut _ as *mut c_void,
        &mut pwa3 as *mut _ as *mut c_void,
        &mut pwd3 as *mut _ as *mut c_void,
        &mut pwn3 as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni3 as *mut _ as *mut c_void,
        &mut cli3 as *mut _ as *mut c_void,
        &mut nki3 as *mut _ as *mut c_void,
        &mut nvi3 as *mut _ as *mut c_void,
        &mut dki3 as *mut _ as *mut c_void,
        &mut dvi3 as *mut _ as *mut c_void,
        &mut cdi3 as *mut _ as *mut c_void,
        &mut csi3 as *mut _ as *mut c_void,
        &mut epsi3 as *mut _ as *mut c_void,
        &mut qsi3 as *mut _ as *mut c_void,
    ];
    let shmem_scan = ((3 * dk + dv + 5) * 4) as u32;
    unsafe {
        ctx.launch_async(
            ctx.f_gdn_chunk_scan,
            (n_chunks as u32, nv as u32, 1),
            (dk as u32, 1, 1),
            shmem_scan,
            &mut args_scan,
        )?;
    }
    kprof_mark(&mut kp, 3);
    // s245 (P45): y をデバイスに残す経路 (`y_dst = Some` = prefill 残差常駐) では
    // **ホストが結果を読まない**ので待たない。3 pass は同一ストリームへ発行順に並び、
    // pass 3 が書いた y を後続の o_proj が同じストリームで読み、状態の D2D 書き戻しと
    // 次の層の beta/alpha H2D (cuMemcpyHtoD = ホスト同期コピー) も同じストリームなので、
    // バッファ再利用の順序もストリームが保証する (s173/P15 で pass 間の 2 回を外したのと
    // 同じ根拠)。数値経路・発行順は 1 つも変えない。
    if y_dst.is_some() {
        // s72: y はデバイス上に残す (呼び出し側がチェーンで消費する)。
        // s131: 状態もデバイス正本へ D2D で戻す (ホストへは落とさない)。
        q.end();
        // s245 (P45): 3 区間の読み出しは次の既存の同期 (次の層の kprof_begin /
        // prefill 末尾の pf_timing_read) へ回す。そこで未完了なら finish が miss に
        // 数える (P43 の規則) — 計測のために同期を足さない。
        if let Some(k) = kp.take() {
            kprof_defer(k);
        }
        gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
        return Ok(Vec::new());
    }
    // ホスト出力 (`y_dst = None`) は y を `to_vec` (同期 D2H) で読むので 1 回待つ。
    // この 1 回は数値のため (従来どおり) で、細分の意味も変えない。
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);
    // s240 (P40): 既存の同期が済んだ直後に 3 区間を読む (待ちは足さない)。
    // `q.lap(D2H)` の後に置くのは、3 回の `cuEventElapsedTime` (数 µs) を
    // 既存の `gdn_pf_sync` の数字に混ぜないため (d2h は表示専用の細分)。
    if let Some(k) = kp.take() {
        k.finish();
    }
    let y = GDN_SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let s = cell.borrow();
        s.buf(GdnBuf::Y).to_vec::<f32>(n * vd)
    })?;
    // s131: 状態はホストへ落とさずデバイス正本へ D2D で戻す。
    gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    q.end();
    Ok(y)
}


#[cfg(test)]
mod p55_gdn_pre_sm_tests {
    //! P55: pass 1 の状態配列を共有メモリへ置く版 (`dno_gdn_chunk_pre_sm`) を
    //! **選ぶゲート**の契約テスト (GPU 不要)。守るのは「カーネル不在 (PTX 未再
    //! 生成) / opt-in 失敗 / dk 過大 / dk != dv / env off のどれでも静かに旧
    //! カーネルへ落ちる」= fail-closed と、必要 smem が 48 KiB 既定を超えること。
    //! 数値の一致と速度は実機 (main) の担当。
    use super::*;

    /// dk=128 (qwen35 の GDN 実形状) の必要量。48 KiB 既定では載らず、
    /// A100 (164 KiB) / H100 (227 KiB) の opt-in 上限には収まること。
    #[test]
    fn gdn_pre_sm_smem_bytes_need_optin() {
        let need = gdn_pre_sm_smem_bytes(128);
        assert_eq!(need, (2 * 128 * 128 + 2 * 128 + 3) * 4);
        assert_eq!(need, 132_108);
        assert!(need > GDN_PRE_SM_SMEM_DEFAULT, "48 KiB 既定では載らない");
        assert!(need <= 164 * 1024, "A100 の opt-in 上限を超える");
        assert!(need <= 227 * 1024, "H100 の opt-in 上限を超える");
        // dk=64 は 48 KiB 既定に収まる (opt-in 不要の形が存在すること)。
        assert!(gdn_pre_sm_smem_bytes(64) <= GDN_PRE_SM_SMEM_DEFAULT);
        // dk=256 は opt-in 上限でも載らない (ゲートで落ちる側)。
        assert!(gdn_pre_sm_smem_bytes(256) > 227 * 1024);
    }

    /// ゲートの真理値表: 4 条件 (env / カーネル有無 / 形 / smem 上限) の
    /// どれか 1 つが欠けたら false = 旧カーネル。
    #[test]
    fn gdn_pre_sm_gate_contract() {
        let (dk, dv) = (128usize, 128usize);
        let need = gdn_pre_sm_smem_bytes(dk);
        let optin = 227 * 1024;
        assert!(gdn_pre_sm_gate(true, true, dk, dv, need, optin));
        // env off (既定) では常に旧カーネル
        assert!(!gdn_pre_sm_gate(false, true, dk, dv, need, optin));
        // PTX にカーネルが無い (未再生成) → 旧カーネル
        assert!(!gdn_pre_sm_gate(true, false, dk, dv, need, optin));
        // opt-in 失敗 (上限が 48 KiB 既定のまま) → 旧カーネル
        assert!(!gdn_pre_sm_gate(true, true, dk, dv, need, GDN_PRE_SM_SMEM_DEFAULT));
        // dk 過大 (共有メモリに載らない) → 旧カーネル
        assert!(!gdn_pre_sm_gate(
            true,
            true,
            256,
            256,
            gdn_pre_sm_smem_bytes(256),
            227 * 1024
        ));
        assert!(!gdn_pre_sm_gate(true, true, GDN_PRE_SM_MAX_DK + 1, GDN_PRE_SM_MAX_DK + 1, 0, optin));
        // dk != dv (chunked scan の前提外) → 旧カーネル
        assert!(!gdn_pre_sm_gate(true, true, 128, 64, need, optin));
        // dk = 0 は範囲外
        assert!(!gdn_pre_sm_gate(true, true, 0, 0, 0, optin));
        // dk=64 は 48 KiB 既定でも true (opt-in 不要の形)
        assert!(gdn_pre_sm_gate(
            true,
            true,
            64,
            64,
            gdn_pre_sm_smem_bytes(64),
            GDN_PRE_SM_SMEM_DEFAULT
        ));
    }
}



// ---------------------------------------------------------------------------
// task 128: GDN (qwen4exp) decode の GPU 常駐スキャン (DENNO_GDN_GPU_SCAN)
// ---------------------------------------------------------------------------
//
// br114 (a4a0c617f) の設計を int-auto の API (`CudaCtx::plan_moe_gemv` +
// `launch_async`) へ移植したもの。1 トークン 1 層ぶんの
//   [qkv(+z) の GEMV] → [dno_gdn_scan_sig] → [attn_out の GEMV]
// を 1 つの submission にまとめ、層の状態 (conv / ssm) をデバイス常駐にする
// (デバイスが正本。ホスト `SsmLayer::state` は 1 トークン以上古い)。
//
// バッファの中身 (token 1 / token 2 の各段階。REPORT_P128.md と対):
//   x      [n_embd]            層入力 (ホスト→H2D、毎トークン)
//   qkv    [qkv_total]         [qkv 行 | z 行] の 1 本 (GEMV 出力。毎トークン)
//   ba     [2*nv]              [beta | alpha] (CPU F32 GEMV → H2D、毎トークン)
//   conv   [conv_dim][cs]      cs = d_conv-1 = 3。**ping-pong 2 面**。
//                              q/k チャネルは nk < nv のとき複数の v-head が
//                              同じチャネルを読むので in-place にできない。
//   ssm    [nv][dk][dv]        1 面。各スレッドが自分の列を先頭で全読みして
//                              から書くので in == out で安全。
//   y      [nv*dv]             scan 出力 (gate 済み)
//   out    [n_embd]            attn_out GEMV の出力 (= 層出力)
//
// ホスト state との契約:
//   (a) デバイスが正の間ホストは古い。CPU 経路が state を読む前に必ず
//       [`gdn_decode_state_take`] (同期 D2H) で回復する。
//   (b) ホスト側で state を書き換えたら (reset / spec_restore) 必ず
//       [`gdn_decode_state_invalidate`] でデバイスを無効化する。
//       **task 128 の修正点**: br114 は reset でこれを落としていた。モデル
//       使い回し (`DENNO_FLASH_MODEL_REUSE=1`) で前置きが変わると reset が走り、
//       次の prefill の take が **前のリクエストの状態** をホストへ戻すため、
//       全トークンが同じ壊れ方をする (安定 md5 の `!!!!`)。
//   (c) ホストが正のときだけシード (H2D) する。
//
// 失敗はすべて fail-closed (`out: None` → 呼び出し側は CPU 経路)。
// scan を発行した後の失敗だけは状態を回復して返し、回復も失敗したら poisoned
// (呼び出し側は要求を落とす)。
//
// **task 137 で解消**: このカーネル `dno_gdn_scan_sig` (gdn.cu の sigmoid
// 出力ゲート版) を gdn.cu へ追加し、`kernels_gdn.ptx` を再生成した
// (再生成前の PTX は gdn.cu より古く、P55 の `dno_gdn_chunk_pre_sm` も
// 未収録だった)。旧 PTX を読んだ場合だけ `CudaCtx::f_gdn_scan_sig` が None に
// なり、この経路は発動しない (計測を 1 つも積まずに CPU 経路)。
// **task 140 のラボ (t140a) で確認**: 今の gdn.cu から nvcc で再生成した
// `kernels_gdn.ptx` は、コミット済みのものと byte 一致 (9965 行 /
// md5 b580964cfe78ad8d10beab3b7682ea3c)。つまり t137c が使った PTX は
// 正しい再生成物であり、当時の 700 は PTX ではなくホスト側のバグだった。
//
// **task 137 のラボ (t137c) で分かった残件**: 経路は発動し、数値も合う
// (tok=1 の層 0..9 で maxabs out <= 1.8e-7 / conv 0 / ssm <= 2.4e-7、出力
// テキストの md5 は CPU と同一) が、**最初の decode トークンの直後の
// submission で CUDA error 700** になり、ラッチで以後 CPU に落ちる。
//
// **task 140 で原因を特定**: 700 は `qkv_gemv[0]` (li=41 tok=1) で起きていた。
// `gdn_qkv_units` が GEMV の出力行オフセットに `FusedLinear::range(i).start` を
// 使っていたのが誤りで、連結済みの qkv_gate では単位 = 連結バッファ全体
// (先頭から書く) なのに、gate が qkv より前に置かれた層 (写しなし連結が
// ファイル順を `ranges` に保存するため `range(0).start = 6144`) では
// 16384 行を 6144 行目から書いて **e.qkv の終端を 24 KiB 越えて**いた
// (ラボのダンプで li=13/38/41/46 が該当)。同じ層が 41 番目で初めて
// 未マップの VA に当たり 700 → ラッチ、以後 129% 遅い状態になっていた。
// 修正後 (t140b/t140c) は errors 0・md5 SAME・テキスト正常で、
// 短 decode -1.9%、2k 文脈 -4.2..-4.8% (t140c の VERDICT)。
// ただしタスクの採否目標 (短 decode 3% 以上) に届かないので**既定 off のまま**。
//
// **task 142 で既定 on にした**: 上の目標 (短 decode 3%) は「利得が文脈長で
// 伸びる」ことを知る前に置かれたものだった。文脈 2k / 8k で目標を取り直し、
// int-auto 065cf20 (task 141 まで) に br140 をマージした t142a で
// 短 -5.1% / c2000 next -5.6% / c8000 next -6.7% / prefill 2% 以内 /
// errors 0 / md5 6 本 SAME を確認した (詳細は [`gdn_gpu_scan_on`] の doc)。

/// task 142 のラボ (t142a) で採否目標を満たしたので**既定 on**。
/// `DENNO_GDN_GPU_SCAN=0|off|false|no` で off に戻せる (未設定・空は on)。
///
/// t142a (int-auto 065cf20 + task 141 までのマージ、depth8k、PROFILE=1、2 ラウンド交互):
/// decode は短 -5.1%、c2000 next turn -5.6%、c8000 next turn -6.7%、prefill は
/// -0.7%..+1.5% (全て 2% 以内)、errors 0、md5 は 6 本とも SAME。q4prof の gdn 項目は
/// on で +116 ms/128tok (短)・+103 (c2000)・+12 (c8000) と増えるが、これは GPU scan の
/// 待ちが attn.gate に移り、qkv/out の GEMV が同じ submission に入るため (層 wall は
/// 短 -4.2%、c2000 -4.5%、c8000 -6.9%)。
///
/// t142b (既定 on 版の確認、def vs `DENNO_GDN_GPU_SCAN=0`): def が全 5 点で off 以上
/// (短 -3.8%、c2000_cold -3.7%、c2000_turn2 -1.3%、c8000 -5.6%)、errors 0、md5 SAME。
/// c2000_turn2 の 1.3% は 2 ラウンド目が同値でノイズ帯 (<2%)。prefill は
/// -0.4..-2.0% で t142a と符号が逆 (ノイズ)。
///
/// off のときは `SsmLayer::forward_token` / `forward_batch` / `reset_state` の分岐
/// (この static read) 以外に何も変わらない。テストは
/// [`gdn_gpu_scan_set_for_test`] で env 無しに on/off を強制できる
/// (env はプロセスで 1 回しか読めないため)。
pub fn gdn_gpu_scan_on() -> bool {
    #[cfg(test)]
    match GDN_GPU_SCAN_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gdn_gpu_scan(std::env::var("DENNO_GDN_GPU_SCAN").ok().as_deref()))
}


/// `DENNO_GDN_GPU_SCAN` の生値 → on/off (純関数)。**未設定は on** (task 142 で採用)。
/// 明示的な `0|off|false|no` (前後空白と大小文字は無視) だけ off。
fn parse_gdn_gpu_scan(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "off" | "false" | "no")
        }
    }
}


/// テスト強制の値 (0 = env の解釈、1 = 強制 on、2 = 強制 off)。
#[cfg(test)]
static GDN_GPU_SCAN_FORCE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);


/// テストが env を立てずにゲートを強制する (`true` = 強制 on、`false` = 強制 off)。
///
/// 既定が on になったので、off 経路を見るテストは `false` で明示的に落とす。
#[cfg(test)]
pub fn gdn_gpu_scan_set_for_test(v: bool) {
    GDN_GPU_SCAN_FORCE.store(if v { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
}


/// テストが強制を解除して env の解釈 (既定 on) に戻す。
#[cfg(test)]
pub fn gdn_gpu_scan_reset_force_for_test() {
    GDN_GPU_SCAN_FORCE.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// CPU vs GPU の状態突き合わせ (`DENNO_GDN_GPU_CHECK=1`。診断専用・既定 off)。
///
/// on のとき [`gdn_decode_submit`] は層出力に加えて **scan 後の conv/ssm** を
/// D2H して返す (毎層 1 往復 = 遅い。診断専用なので既定 off)。比較そのものは
/// 呼び出し側 (`SsmLayer::forward_token`) が CPU 参照と行い、最初に違った層と
/// 要素を 1 行だけ出す。
pub fn gdn_gpu_check_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("DENNO_GDN_GPU_CHECK").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    })
}


/// task 140 の診断 (`DENNO_GDN_SUBMIT_SYNC=1`。既定 off)。
///
/// on のとき [`gdn_decode_submit`] は submission の**各段の直後**に
/// `cuCtxSynchronize` を挟み、失敗したらその段の名前をエラーに付ける。
/// error 700 は「違反した launch の後、次にホストと同期する API が返す」ため、
/// 段ごとに同期しないとどの launch が壊れたかを切り分けられない (task 137 の
/// t137c は「scan 発行前」までしか絞れなかった)。**診断専用**で、on のときは
/// 段数ぶん同期するので decode は大幅に遅くなる。数値経路は変えない。
pub fn gdn_submit_sync_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gdn_submit_sync(std::env::var("DENNO_GDN_SUBMIT_SYNC").ok().as_deref()))
}


/// `DENNO_GDN_SUBMIT_SYNC` の生値 → on/off (純関数。真理値表はテストで固定)。
fn parse_gdn_submit_sync(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("on") | Some("true"))
}


/// task 140 の診断 (`DENNO_GDN_SUBMIT_DUMP=1`。既定 off)。
///
/// on のとき [`gdn_decode_submit`] は **decode 窓の 1 トークン目だけ**、層ごとに
/// 1 行の plan ダンプ (型・k/n・行バイト・実バイト数・行オフセット・起動形状・
/// ポインタ) を出す。t140a のラボでは `qkv_gemv[0]` が li=41 で
/// error 700 になったが、同じ重み・同じ形状は intercept 経路で正常に走っており、
/// 引数のどこが違うかをログから確定できなかった。診断専用 (数値経路は不変)。
pub fn gdn_submit_dump_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gdn_submit_dump(std::env::var("DENNO_GDN_SUBMIT_DUMP").ok().as_deref()))
}


/// `DENNO_GDN_SUBMIT_DUMP` の生値 → on/off (純関数)。
fn parse_gdn_submit_dump(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("on") | Some("true"))
}


/// 状態ストアの上限 (エントリ = 1 層ぶん ≈ 3.3 MiB: conv 2x120 KiB + ssm 3.0 MiB
/// + x/qkv/ba/y/out 約 100 KiB)。72 = 36 層モデル 2 本ぶん。
/// 追い出しは最古の挿入順 (Vec の先頭)。`SsmLayer` は Drop を持たないので、
/// 解放済みモデルのエントリはこの上限まで残る (上限到達で先頭から回収)。
const GDN_DECODE_STORE_MAX: usize = 72;


/// ストアの鍵。`nonce` は `SsmLayer` インスタンスごとの通し番号
/// (構築時に 1 回発行) — アドレス再利用 (ABA) で別モデルのデバイス状態を
/// 流用しないための identity。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnDecodeKey {
    nonce: u64,
    li: u32,
}


/// 層 1 本ぶんのデバイス常駐バッファ (lazy に 1 回だけ確保)。
pub(crate) struct GdnDecodeLayerDev {
    /// conv 状態 [conv_dim * cs] の ping-pong (両面とも同じレイアウト)。
    conv: [DevBuffer; 2],
    /// ssm 状態 [nv * dk * dv] (in-place 更新)。
    ssm: DevBuffer,
    /// 現在の読み出し面 (書き込み面は 1-cur)。
    cur: usize,
    /// 小さい f32 重み (構築時に 1 回だけ上げる)。
    wconv: DevBuffer,
    wa: DevBuffer,
    wdt: DevBuffer,
    wnorm: DevBuffer,
    /// 層入力 cur [n_embd] と beta/alpha [2 * nv] を**1 本に連結**した staging。
    ///
    /// task 140: 以前は `x` と `ba` の 2 本で、層ごとに同期 H2D を 3 回
    /// (x / beta / alpha) 積んでいた。cuMemcpyHtoD は既定ストリームの完了を待つ
    /// ので、36 層 × 3 回の待ちが submission の主要コストだった (実測 t140b:
    /// gdn bucket 6.9 ms/tok)。1 本にまとめて H2D を 1 回にする (転送する値・
    /// 並び・カーネルへ渡すポインタは同一)。
    xba: DevBuffer,
    /// qkv_gate の全出力 ([qkv 行 + z 行]。ユニットは行範囲へ書く)。
    qkv: DevBuffer,
    /// scan 出力 y [vd]。
    y: DevBuffer,
    /// attn_out GEMV の出力 [n_embd]。
    out: DevBuffer,
    /// 寸法 (初回確定。以後の要求が違ったら fail-closed)。
    n_embd: usize,
    qkv_total: usize,
    vd: usize,
    conv_len: usize,
    ssm_len: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    nk: usize,
    conv_dim: usize,
    d_conv: usize,
    /// true = デバイスが正 (ホスト state は 1 トークン以上古い)。
    device_authoritative: bool,
}


static GDN_DECODE_STORE: std::sync::Mutex<Vec<(GdnDecodeKey, GdnDecodeLayerDev)>> =
    std::sync::Mutex::new(Vec::new());


// task 140: 層ごとの H2D staging (cur | beta | alpha を 1 本に詰める)。
// grow-only で使い回す (36 層 × 128 トークンで数万回呼ばれるため、毎回の
// Vec 確保を避ける)。中身は毎回 clear してから詰め直す。
thread_local! {
    pub(crate) static GDN_STAGE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}


/// `gdn_decode_submit` の入力 1 行分 (GGUF ブロックのままの重み 1 本)。
///
/// cuda.rs は [`crate::model::linear_gguf::GgufLinear`] を import しない
/// (`maybe_gpu_gguf_gemv_into` と同じ分離)。呼び出し側が uid / ty / ブロック /
/// k / n を写して渡す。
pub struct GdnGgufUnit<'a> {
    pub uid: u64,
    pub ty: crate::kquant::GgmlType,
    pub blocks: &'a [u8],
    pub k: usize,
    pub n: usize,
}


// ---- task 269: beta/alpha 制御投影をデバイスで回す -------------------------
//
// 背景 (lab t269a の `[q4prof]`): decode の ssm 層は
// 「beta/alpha の F32 GEMV を **ホストで** 計算 → qkv GEMV 以降を GPU に 1 発」
// という順で、ホストの GEMV が終わるまで GPU が遊んでいる。実測は
// `A_GATE − A_GDN` = 0.785 ms/token (36 層 × 21.8 us) で、これは
// 983 KB/層 の重みを RAM から読む帯域 (48 GB/s) にほぼ等しい = ホストでは
// これ以上縮まない。GPU 側は同じ重みが VRAM に載るので 3-5 us で済む。
//
// 数値: `dno_matvec_f32_r` (固定の木で縮約) を使うので、CPU の逐次/FMA 経路
// とは丸め順が違う (**ビット一致しない**)。決定論は保つ。`DENNO_GDN_BA_GPU=0`
// で従来のホスト計算へ戻る。
//
// task 285: **既定 on** にした (lab t269c で短い decode -3.4%、14k の次 +2.6%、
// md5 DIFF = 意図した丸め順の変更、errors 0、text 健全。t285a で再測)。
// 当時 (task 269) は目標 8% に届かず既定 off のまま残っていたが、その後の
// 265/271/275/278/279/282 で他の段が縮み、同じ 0.785 ms/token の相対価値が
// 上がった。切り戻しは `DENNO_GDN_BA_GPU=0`。

/// task 269: `DENNO_GDN_BA_GPU` の解釈 (純関数)。**既定 on** (task 285)。
///
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけ off。未設定・空・その他の
/// 値は on (取り違えを増やさないため、真偽の判定はこの 4 語だけ)。
pub fn parse_gdn_ba_gpu(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no") | Some("")
    )
}


/// task 269: beta/alpha を GPU で計算するか (`DENNO_GDN_BA_GPU`・既定 off)。
pub fn gdn_ba_gpu_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gdn_ba_gpu(std::env::var("DENNO_GDN_BA_GPU").ok().as_deref()))
}


/// task 900: `DENNO_GDN_BA_ORDER_FIX` の解釈 (純関数)。**既定 on**。
///
/// decode の submission は ba の出力を `xba` の beta/alpha 節域へ書く。連結済み
/// (`FusedLinear::concat`) でデバイスが 1 発の matvec を撃つとき、その出力は
/// **連結バッファの座標** (= 写しなし連結では GGUF のファイル順) に並ぶ。実モデルは
/// ファイル上 alpha が先 (`range(0)=nv..2nv`, `range(1)=0..nv`) なので、従来の
/// 「beta は先頭・alpha は nv 行目」という固定の読み方では beta と alpha が
/// 入れ替わって scan に渡る (P864)。on にすると scan の読み出し位置を
/// `range(i).start` にする。明示的な偽値 (`0`/`off`/`false`/`no`/`""`) だけ off (切り戻し用)。
pub fn parse_gdn_ba_order_fix(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no") | Some("")
    )
}


/// テストが env を立てずにゲートを強制する (1 = on、2 = off、0 = env の解釈)。
#[cfg(test)]
static GDN_BA_ORDER_FIX_FORCE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);


/// task 900: デバイス ba の行順補正を掛けるか (`DENNO_GDN_BA_ORDER_FIX`・既定 on)。
pub fn gdn_ba_order_fix_on() -> bool {
    #[cfg(test)]
    match GDN_BA_ORDER_FIX_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_gdn_ba_order_fix(std::env::var(GDN_BA_ORDER_FIX_ENV).ok().as_deref())
    })
}


/// [`GDN_BA_ORDER_FIX_ENV`] の名前。
pub const GDN_BA_ORDER_FIX_ENV: &str = "DENNO_GDN_BA_ORDER_FIX";


/// テストが強制を解除して env の解釈 (既定 on) に戻す。
#[cfg(test)]
pub fn gdn_ba_order_fix_reset_force_for_test() {
    GDN_BA_ORDER_FIX_FORCE.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// テストがゲートを強制する (`Some(true)` = on、`Some(false)` = off、`None` = 解除)。
#[cfg(test)]
pub fn gdn_ba_order_fix_set_for_test(v: Option<bool>) {
    GDN_BA_ORDER_FIX_FORCE.store(
        match v {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// task 269: beta/alpha の重み型がデバイス経路で扱えるか (純関数)。
pub fn gdn_ba_ty_ok(ty: crate::kquant::GgmlType) -> bool {
    matches!(
        ty,
        crate::kquant::GgmlType::F32 | crate::kquant::GgmlType::Bf16
    )
}


/// task 269: beta/alpha を 1 発の matvec で `x_dev` → `out_dev` へ。
///
/// task 392: `stream` を渡すとそのストリームへ出す (重ね打ち経路)。
/// `stream == None` は従来と同じ (null ストリーム)。
fn gdn_launch_ba_on(
    ctx: &CudaCtx,
    stream: Option<CUstream>,
    unit: &GdnGgufUnit<'_>,
    x_dev: CUdeviceptr,
    out_dev: CUdeviceptr,
) -> Result<()> {
    use crate::kquant::GgmlType as T;
    if !gdn_ba_ty_ok(unit.ty) || unit.n == 0 || unit.n > 65535 {
        bail!("gdn ba: 型/行数が想定外 ({:?}, n={})", unit.ty, unit.n);
    }
    let w = ba_gpu_weight(unit.uid, unit.ty, unit.blocks, unit.k, unit.n)
        .ok_or_else(|| anyhow!("gdn ba: 重みを VRAM に置けない (uid={})", unit.uid))?;
    // 並列縮約版 (1 行 = 1 ブロック・256 スレッド)。BF16 は型そのままの
    // カーネル (BF16 -> f32 は正確)、F32 は f32 版。
    let (f, thr) = match unit.ty {
        T::Bf16 => (
            matvec_bf16_r_fn(ctx).ok_or_else(|| anyhow!("gdn ba: dno_matvec_bf16_r が無い"))?,
            256u32,
        ),
        _ => match matvec_f32_r_fn(ctx) {
            Some(fr) => (fr, 256u32),
            None => match matvec_f32_fn(ctx) {
                Some(f) => (f, 128u32),
                None => bail!("gdn ba: dno_matvec_f32 が PTX に無い"),
            },
        },
    };
    let (mut ph, mut pw, mut po) = (x_dev, w, out_dev);
    let (mut kc, mut nc) = (unit.k as c_int, unit.n as c_int);
    let mut args = [
        &mut ph as *mut _ as *mut c_void,
        &mut pw as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut kc as *mut _ as *mut c_void,
        &mut nc as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async_on(
            stream.unwrap_or(std::ptr::null_mut()),
            f,
            (unit.n as u32, 1, 1),
            (thr, 1, 1),
            0,
            &mut args,
        )?
    };
    Ok(())
}


// ---------------------------------------------------------------------------
// task 392: beta/alpha の matvec を qkv GEMV と同じ時間帯に走らせる
// (`DENNO_GDN_BA_PAR`)
// ---------------------------------------------------------------------------
//
// GDN の submission は「xba H2D -> ba matvec (beta/alpha) -> qkv GEMV -> scan
// -> out GEMV -> D2H」の 1 本の鎖で、ba は **x だけで決まる** (qkv とは独立)。
// lab t392a の `[dec-kprof-avg]` (gdn、1 層 1 回) は ba 6.0 us / qkv 35.2 us /
// scan 13.2 us / out 35.2 us で、ba は qkv の前に直列に並んでいるだけ = 1 層
// 6 us が丸ごと遊び。ba を専用ストリームへ出し、scan の前にイベントで join
// すると、カーネルも引数も入力も変えずに鎖が 6 us/層 短くなる (36 層 =
// 0.22 ms/token)。値はビット一致 (同じカーネル・同じ入力・同じ順序)。
//
// 失敗したら (ドライバのシンボルが無い / 生成に失敗) 従来の直列経路へ落ちる。

/// `DENNO_GDN_BA_PAR`: ba matvec を専用ストリームで qkv と重ねる。
pub const GDN_BA_PAR_ENV: &str = "DENNO_GDN_BA_PAR";


/// [`GDN_BA_PAR_ENV`] の解釈 (純関数)。**既定 off**。
///
/// 明示的な真値 (`1` / `on` / `true` / `yes`) のときだけ重ねる。
/// 採否が出たら既定を here で切り替える。
pub(crate) fn gdn_ba_par_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// `DENNO_GDN_BA_PAR` が有効か (プロセスで 1 回だけ env を読む)。
fn gdn_ba_par_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_ba_par_from_value(std::env::var(GDN_BA_PAR_ENV).ok().as_deref()))
}


/// 重ね打ちが効いた層数 / 用意に失敗した回数。
static GDN_BA_PAR_LAYERS: AtomicU64 = AtomicU64::new(0);

static GDN_BA_PAR_FALLBACK: AtomicU64 = AtomicU64::new(0);


/// 重ね打ちが成立した層を 1 つ数える。
fn note_gdn_ba_par_layer() {
    GDN_BA_PAR_LAYERS.fetch_add(1, Ordering::Relaxed);
}


/// 専用ストリームを用意できなかった回数を 1 つ数える。
fn note_gdn_ba_par_fallback() {
    GDN_BA_PAR_FALLBACK.fetch_add(1, Ordering::Relaxed);
}


/// 1 リクエストぶんの 1 行 (ゲート on で 1 層も通らなければ `None`)。
///
/// **緑 = 経路が通った証拠ではない**ので、層数を数えて 1 行で出す。
pub fn gdn_ba_par_line() -> Option<String> {
    gdn_ba_par_line_with(
        gdn_ba_par_on(),
        GDN_BA_PAR_LAYERS.load(Ordering::Relaxed),
        GDN_BA_PAR_FALLBACK.load(Ordering::Relaxed),
    )
}


/// [`gdn_ba_par_line`] の本体 (テストで両側を踏めるように引数で受ける)。
pub(crate) fn gdn_ba_par_line_with(on: bool, layers: u64, fallback: u64) -> Option<String> {
    if !on || (layers == 0 && fallback == 0) {
        return None;
    }
    Some(format!(
        "[gdn-ba-par] par_layers={layers} fallback={fallback}"
    ))
}


/// 積算を 0 に戻す (リクエスト終端)。
pub fn gdn_ba_par_reset() {
    GDN_BA_PAR_LAYERS.store(0, Ordering::Relaxed);
    GDN_BA_PAR_FALLBACK.store(0, Ordering::Relaxed);
}


/// 専用ストリームとイベント 2 本 (スレッドローカル。作るのは 1 回だけ)。
pub(crate) struct GdnBaParStream {
    stream: CUstream,
    /// null ストリーム (xba H2D の後) に打つ。
    ready: CUevent,
    /// 専用ストリームの終点に打つ。
    done: CUevent,
}


thread_local! {
    pub(crate) static GDN_BA_PAR_STREAM: std::cell::RefCell<Option<GdnBaParStream>> =
        const { std::cell::RefCell::new(None) };
}


/// 専用ストリームを用意する (無ければ作る)。作れなければ `false`。
fn gdn_ba_par_ensure() -> bool {
    GDN_BA_PAR_STREAM.with(|c| {
        if c.borrow().is_some() {
            return true;
        }
        let Ok(l) = lib() else {
            return false;
        };
        let (Some(create_stream), Some(create_event)) = (l.stream_create, l.event_create) else {
            return false;
        };
        if l.stream_wait_event.is_none() || l.event_record.is_none() {
            return false;
        }
        let mut st: CUstream = std::ptr::null_mut();
        if unsafe { create_stream(&mut st, CU_STREAM_NON_BLOCKING_BAPAR) } != CUDA_SUCCESS {
            return false;
        }
        let mut ready: CUevent = std::ptr::null_mut();
        let mut done: CUevent = std::ptr::null_mut();
        let ok = unsafe { create_event(&mut ready, CU_EVENT_DISABLE_TIMING_BAPAR) } == CUDA_SUCCESS
            && unsafe { create_event(&mut done, CU_EVENT_DISABLE_TIMING_BAPAR) } == CUDA_SUCCESS;
        if !ok {
            return false;
        }
        *c.borrow_mut() = Some(GdnBaParStream {
            stream: st,
            ready,
            done,
        });
        true
    })
}


/// この層ぶんの重ね打ちを用意する。戻り値は専用ストリーム (join は
/// [`gdn_ba_par_mark_done`] / [`gdn_ba_par_join`] で行う)。
///
/// xba の H2D は**同期コピー**なので、この時点で null ストリームは空。
/// そこへ `ready` を打ち、専用ストリームに待たせる = ba は H2D の後に走る。
fn gdn_ba_par_arm() -> Option<CUstream> {
    if !gdn_ba_par_ensure() {
        note_gdn_ba_par_fallback();
        return None;
    }
    let Ok(l) = lib() else {
        note_gdn_ba_par_fallback();
        return None;
    };
    let (Some(rec), Some(wait_ev)) = (l.event_record, l.stream_wait_event) else {
        note_gdn_ba_par_fallback();
        return None;
    };
    GDN_BA_PAR_STREAM.with(|c| {
        let b = c.borrow();
        let Some(p) = b.as_ref() else {
            return None;
        };
        if unsafe { rec(p.ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            note_gdn_ba_par_fallback();
            return None;
        }
        if unsafe { wait_ev(p.stream, p.ready, 0) } != CUDA_SUCCESS {
            note_gdn_ba_par_fallback();
            return None;
        }
        Some(p.stream)
    })
}


/// 専用ストリームの終点に `done` を打つ (ba を積み終わった直後に呼ぶ)。
fn gdn_ba_par_mark_done() -> Result<()> {
    let l = lib()?;
    let Some(rec) = l.event_record else {
        bail!("gdn ba par: event_record が無い");
    };
    GDN_BA_PAR_STREAM.with(|c| {
        let b = c.borrow();
        let Some(p) = b.as_ref() else {
            bail!("gdn ba par: 専用ストリームが無い");
        };
        if unsafe { rec(p.done, p.stream) } != CUDA_SUCCESS {
            bail!("gdn ba par: done を打てない");
        }
        Ok(())
    })
}


/// null ストリームに `done` を待たせる (scan を積む前)。成立した層を数える。
fn gdn_ba_par_join() -> Result<()> {
    let l = lib()?;
    let Some(wait_ev) = l.stream_wait_event else {
        bail!("gdn ba par: stream_wait_event が無い");
    };
    GDN_BA_PAR_STREAM.with(|c| {
        let b = c.borrow();
        let Some(p) = b.as_ref() else {
            bail!("gdn ba par: 専用ストリームが無い");
        };
        if unsafe { wait_ev(std::ptr::null_mut(), p.done, 0) } != CUDA_SUCCESS {
            bail!("gdn ba par: join できない");
        }
        Ok(())
    })?;
    note_gdn_ba_par_layer();
    Ok(())
}


/// 控え (`mixed` のデバイスポインタと要素数)。
pub(crate) static GDN_X_PTR: AtomicU64 = AtomicU64::new(0);

pub(crate) static GDN_X_LEN: AtomicU64 = AtomicU64::new(0);


/// `gdn_decode_submit` のリクエスト。
pub struct GdnDecodeReq<'a> {
    /// SsmLayer インスタンスの nonce (ストアの鍵)。
    pub nonce: u64,
    /// 層番号 (診断ログ用)。
    pub li: usize,
    /// decode 窓の通し番号 (`gdn_gpu_win::token_index()`。診断ログ用)。
    pub tok: u64,
    pub n_embd: usize,
    /// qkv_gate の全出力行数 (qkv 行 + z 行の合計)。
    pub qkv_total: usize,
    /// scan へ渡す qkv 領域の行起点 (`FusedLinear::range(0).start`。実モデルは 0)。
    pub qkv_row_start: usize,
    /// scan へ渡す z 領域の行起点 (`FusedLinear::range(1).start`。実モデルは qkv 行数)。
    pub z_row_start: usize,
    /// qkv_gate の GEMV 単位 (連結済みなら 1 本) と出力行オフセットの組。
    pub qkv_units: &'a [(GdnGgufUnit<'a>, usize)],
    /// attn_out 投影 (y [vd] → out [n_embd])。
    pub out_unit: GdnGgufUnit<'a>,
    /// 層入力 (ホスト)。
    pub cur: &'a [f32],
    /// beta / alpha (ホスト f32。CPU の F32 GEMV で計算済み)。
    pub beta: &'a [f32],
    pub alpha: &'a [f32],
    /// task 269: beta/alpha をデバイスで計算する場合の単位と出力行オフセット。
    /// 空でなければホストの `beta` / `alpha` は読まず、`xba` の beta/alpha 節域へ
    /// f32 matvec を 1 発ずつ積む (`DENNO_GDN_BA_GPU`)。
    pub ba_units: &'a [(GdnGgufUnit<'a>, usize)],
    /// task 900: scan が読む beta / alpha の先頭行 (ba 出力ブロックの先頭から。
    /// 単位は行 = f32)。連結済みの ba をデバイスで計算するときは、デバイス出力が
    /// 連結バッファの座標 (= 写しなし連結では GGUF のファイル順) に並ぶので
    /// `range(0).start` / `range(1).start`。ホスト計算 (`ba_units` 空) は staging に
    /// 論理順で詰めるので (0, nv)。`DENNO_GDN_BA_ORDER_FIX` が off のときは常に
    /// (0, nv) (従来の固定の読み方 = ファイル順が逆のモデルでは入れ替わる)。
    pub ba_beta_row: usize,
    pub ba_alpha_row: usize,
    /// ホスト正本の状態 (デバイスが古いときのシードに使う)。
    pub conv_host: &'a [f32],
    pub ssm_host: &'a [f32],
    /// 小さい f32 重み (初回だけデバイスへ上げる)。
    pub wconv: &'a [f32],
    pub wa: &'a [f32],
    pub wdt: &'a [f32],
    pub wnorm: &'a [f32],
    /// GDN の寸法 ([`crate::gdn::GdnParams`] から)。
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub conv_dim: usize,
    pub d_conv: usize,
    pub eps: f32,
}


/// `gdn_decode_submit` の結果。
pub struct GdnDecodeRes {
    /// attn_out 投影まで済んだ層出力。`None` = GPU で完走できなかった
    /// (呼び出し側は従来の CPU 経路に落ちること。fail-closed)。
    pub out: Option<Vec<f32>>,
    /// scan 発行後に落ちたときの状態回復 (conv, ssm)。`Some` なら呼び出し側は
    /// この値をホスト state へ写してから CPU 経路に落ちること。
    pub state_back: Option<(Vec<f32>, Vec<f32>)>,
    /// scan を発行した後に回復にも失敗した (GPU が壊れた)。ホスト state は
    /// 1 トークン以上古い — 呼び出し側は要求を落とすこと。
    pub poisoned: bool,
    /// `DENNO_GDN_GPU_CHECK=1` のときだけ: このトークンを終えたデバイス状態
    /// (conv[書き込み面], ssm)。CPU 参照との突き合わせに使う (診断専用)。
    pub state_dev: Option<(Vec<f32>, Vec<f32>)>,
}


impl GdnDecodeRes {
    pub(crate) fn cpu() -> Self {
        GdnDecodeRes {
            out: None,
            state_back: None,
            poisoned: false,
            state_dev: None,
        }
    }
}


/// GPU 経路が 1 回だけ出す注記 (同じ理由の反復ログを防ぐ)。
fn gdn_note_once(msg: &str) {
    static DONE: OnceLock<()> = OnceLock::new();
    if DONE.set(()).is_ok() {
        tracing::info!("[gdn-gpu-scan] {msg}");
    }
}


/// task 114c: decode scan 経路の process-local 不使用ラッチ。
///
/// 実機 (8096) では fused submission の 1 回目の失敗のあともトークン毎に再発行が
/// 続き、失敗した launch の**遅延報告** (CUDA error 700 =
/// LAUNCH_OUT_OF_RESOURCES は同期点で後から返ることがある) をコンテキストに
/// 貯め続けた。保留エラーは scan 経路の外の最初の同期点 (gguf gemv intercept の
/// H2D など) からも返るので、結果として intercept 経路まで CPU フォールバックに
/// なっていた (21.7s → 66.8s)。そこで GPU 側の失敗を 1 回でも見たら、以後この
/// プロセスでは decode scan を**発行しない** (launch を積まない = 新しい保留
/// エラーを作らない)。intercept 側は自分の同期で保留エラーを 1 度返せば回復する
/// (700 は文脈破壊エラーではない)。
///
/// ラッチは **GPU_BROKEN とは独立** — scan の失敗で他の GPU 経路 (intercept /
/// router / prefill chain) を止めない。
static GDN_SCAN_UNAVAILABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// GPU 側の失敗を記録する。戻り値は「この呼び出しで立った」(初回だけ true)。
fn gdn_scan_mark_unavailable() -> bool {
    !GDN_SCAN_UNAVAILABLE.swap(true, std::sync::atomic::Ordering::Relaxed)
}


/// ラッチの状態 (テスト用)。
#[cfg(test)]
fn gdn_scan_unavailable_for_test() -> bool {
    GDN_SCAN_UNAVAILABLE.load(std::sync::atomic::Ordering::Relaxed)
}


/// テストがラッチを戻す (guard の Drop から呼んで panic でも復帰させる)。
#[cfg(test)]
fn gdn_scan_unavailable_reset_for_test() {
    GDN_SCAN_UNAVAILABLE.store(false, std::sync::atomic::Ordering::Relaxed);
}


/// GPU 側の失敗を記録してから応答を返す (ラッチが初めて立ったときだけ案内を
/// 1 行出す)。`res` は呼び出し側が組み立てたものをそのまま返す。
fn gdn_scan_record_failure(res: GdnDecodeRes) -> GdnDecodeRes {
    if gdn_scan_mark_unavailable() {
        tracing::info!(
            "[gdn-gpu-scan] GPU 側の失敗 → このプロセスでは decode scan 経路を不使用にする (intercept 経路には影響しない)"
        );
    }
    res
}


/// ストアに鍵のエントリを用意する (無ければ確保して入れる。上限到達で先頭を捨てる)。
///
/// 呼び出し側が `GDN_DECODE_STORE` のロックを保持している間にだけ呼ぶこと。
fn gdn_decode_entry_ensure_locked(
    store: &mut Vec<(GdnDecodeKey, GdnDecodeLayerDev)>,
    req: &GdnDecodeReq,
) -> Result<usize> {
    let key = GdnDecodeKey {
        nonce: req.nonce,
        li: req.li as u32,
    };
    if let Some(i) = store.iter().position(|(k, _)| *k == key) {
        return Ok(i);
    }
    // VRAM の目安 (空きが明らかに足りないなら確保前に落ちる)。
    let cs = req.d_conv - 1;
    let bytes = (req.conv_dim * cs * 2
        + req.conv_dim * req.d_conv
        + req.nv * 2
        + req.dv
        + req.n_embd * 2
        + req.qkv_total
        + req.nv * 2
        + req.nv * req.dv
        + req.nv * req.dk * req.dv)
        * 4;
    if let Some((free, _total)) = dev_mem_mb() {
        if (free << 20) < bytes + (16 << 20) {
            bail!(
                "gdn decode state: 空き VRAM 不足 (必要 {:.1} MiB / 空き {free} MiB)",
                bytes as f64 / (1 << 20) as f64
            );
        }
    }
    let e = GdnDecodeLayerDev {
        conv: [
            DevBuffer::alloc_tagged(req.conv_dim * cs * 4, crate::mem_audit::TAG_GDN_MIRROR)?,
            DevBuffer::alloc_tagged(req.conv_dim * cs * 4, crate::mem_audit::TAG_GDN_MIRROR)?,
        ],
        ssm: DevBuffer::alloc_tagged(
            req.nv * req.dk * req.dv * 4,
            crate::mem_audit::TAG_GDN_MIRROR,
        )?,
        cur: 0,
        wconv: DevBuffer::from_slice_tagged(req.wconv, crate::mem_audit::TAG_GDN_W)?,
        wa: DevBuffer::from_slice_tagged(req.wa, crate::mem_audit::TAG_GDN_W)?,
        wdt: DevBuffer::from_slice_tagged(req.wdt, crate::mem_audit::TAG_GDN_W)?,
        wnorm: DevBuffer::from_slice_tagged(req.wnorm, crate::mem_audit::TAG_GDN_W)?,
        xba: DevBuffer::alloc_tagged(
            (req.n_embd + req.nv * 2) * 4,
            crate::mem_audit::TAG_GDN_SC,
        )?,
        qkv: DevBuffer::alloc_tagged(req.qkv_total * 4, crate::mem_audit::TAG_GDN_SC)?,
        y: DevBuffer::alloc_tagged(req.nv * req.dv * 4, crate::mem_audit::TAG_GDN_SC)?,
        out: DevBuffer::alloc_tagged(req.n_embd * 4, crate::mem_audit::TAG_GDN_SC)?,
        n_embd: req.n_embd,
        qkv_total: req.qkv_total,
        vd: req.nv * req.dv,
        conv_len: req.conv_dim * cs,
        ssm_len: req.nv * req.dk * req.dv,
        nv: req.nv,
        dk: req.dk,
        dv: req.dv,
        nk: req.nk,
        conv_dim: req.conv_dim,
        d_conv: req.d_conv,
        device_authoritative: false,
    };
    if store.len() >= GDN_DECODE_STORE_MAX {
        store.remove(0);
    }
    store.push((key, e));
    Ok(store.len() - 1)
}


/// ストアのエントリ数 (テスト・診断用)。env off / GPU 不在では増えない
/// (確保は submit の中でだけ起きる = lazy)。
pub fn gdn_decode_state_entry_count() -> usize {
    GDN_DECODE_STORE.lock().map(|s| s.len()).unwrap_or(0)
}


/// デバイスが正のとき、層の状態を同期 D2H でホストへ回復する。
///
/// CPU 経路 (prefill / spec verify のバッチ・spec snapshot・CPU フォールバック)
/// が state を読む**前**に呼ぶこと。`HostCurrent` = デバイスは正でない
/// (何もしないでよい)。`Recovered` = 呼び出し側はホスト state へ写すこと。
/// `Lost` = D2H に失敗した (呼び出し側は要求を落とすこと)。
pub enum GdnDecodeFlush {
    HostCurrent,
    Recovered(Vec<f32>, Vec<f32>),
    Lost,
}


pub fn gdn_decode_state_take(nonce: u64, li: usize) -> GdnDecodeFlush {
    let mut store = match GDN_DECODE_STORE.lock() {
        Ok(s) => s,
        Err(_) => return GdnDecodeFlush::Lost,
    };
    let key = GdnDecodeKey {
        nonce,
        li: li as u32,
    };
    let Some((_, e)) = store.iter_mut().find(|(k, _)| *k == key) else {
        return GdnDecodeFlush::HostCurrent;
    };
    if !e.device_authoritative {
        return GdnDecodeFlush::HostCurrent;
    }
    match (
        e.conv[e.cur].to_vec::<f32>(e.conv_len),
        e.ssm.to_vec::<f32>(e.ssm_len),
    ) {
        (Ok(conv), Ok(ssm)) => {
            e.device_authoritative = false;
            GdnDecodeFlush::Recovered(conv, ssm)
        }
        (Err(err), _) | (_, Err(err)) => {
            tracing::error!("[gdn-gpu-scan] 状態回復 (D2H) に失敗 (li={li}): {err:#}");
            GdnDecodeFlush::Lost
        }
    }
}


/// デバイスが正のとき、層の状態を**読み取り専用**で同期 D2H する
/// ([`gdn_decode_state_take`] と違い正本フラグを動かさない)。
pub enum GdnDecodeRead {
    HostCurrent,
    Device(Vec<f32>, Vec<f32>),
    Lost,
}


pub fn gdn_decode_state_read(nonce: u64, li: usize) -> GdnDecodeRead {
    let store = match GDN_DECODE_STORE.lock() {
        Ok(s) => s,
        Err(_) => return GdnDecodeRead::Lost,
    };
    let key = GdnDecodeKey {
        nonce,
        li: li as u32,
    };
    let Some((_, e)) = store.iter().find(|(k, _)| *k == key) else {
        return GdnDecodeRead::HostCurrent;
    };
    if !e.device_authoritative {
        return GdnDecodeRead::HostCurrent;
    }
    match (
        e.conv[e.cur].to_vec::<f32>(e.conv_len),
        e.ssm.to_vec::<f32>(e.ssm_len),
    ) {
        (Ok(conv), Ok(ssm)) => GdnDecodeRead::Device(conv, ssm),
        (Err(err), _) | (_, Err(err)) => {
            tracing::error!("[gdn-gpu-scan] 状態読み取り (D2H) に失敗 (li={li}): {err:#}");
            GdnDecodeRead::Lost
        }
    }
}


/// デバイスのコピーを捨ててホストを正にする (reset / spec_restore が state を
/// 上書きしたあと)。D2H を行わない (デバイスの内容はもう要らない)。
pub fn gdn_decode_state_invalidate(nonce: u64, li: usize) {
    if let Ok(mut store) = GDN_DECODE_STORE.lock() {
        let key = GdnDecodeKey {
            nonce,
            li: li as u32,
        };
        if let Some((_, e)) = store.iter_mut().find(|(k, _)| *k == key) {
            e.device_authoritative = false;
        }
    }
}


/// task 128 の診断: `reset` の無効化が実際に走った回数 (テストが分岐の発火を
/// 数えるためだけのもの。数値経路からは読まない)。
#[cfg(test)]
static GDN_INVALIDATE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);


#[cfg(test)]
pub fn gdn_invalidate_calls_for_test() -> u64 {
    GDN_INVALIDATE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}


#[cfg(test)]
pub fn gdn_invalidate_calls_reset_for_test() {
    GDN_INVALIDATE_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// [`gdn_decode_state_invalidate`] の記帳付き版 (呼び出し側はこちらを使う)。
pub fn gdn_decode_state_invalidate_counted(nonce: u64, li: usize) {
    #[cfg(test)]
    GDN_INVALIDATE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    gdn_decode_state_invalidate(nonce, li);
}


/// GEMV を「デバイス入力 → デバイス出力」で 1 発積む (同期しない)。
///
/// 引数の組み立ては [`CudaCtx::moe_gemv_dev_with`] と同一 (同じ計画・同じ ABI)。
pub(crate) fn gdn_launch_gemv_dev(
    ctx: &CudaCtx,
    plan: &MoeGemvPlan,
    dev_w: &DevBuffer,
    x_dev: CUdeviceptr,
    out_dev: CUdeviceptr,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    let (f, rows_per_block, block_dim, shmem, lanes, rb, grid) = (
        plan.f,
        plan.rows_per_block,
        plan.block_dim,
        plan.shmem,
        plan.lanes,
        plan.rb,
        plan.grid,
    );
    if dev_w.bytes() != n_out * rb {
        bail!(
            "gdn gemv: 重みのデバイスバイト数 {} が n_out * row_bytes = {} と違う",
            dev_w.bytes(),
            n_out * rb
        );
    }
    let (mut pw, mut px, mut po) = (dev_w.ptr(), x_dev, out_dev);
    let (mut n_in_c, mut n_out_c, mut rb_c) = (n_in as c_int, n_out as c_int, rb as c_int);
    // v3 / fp16 は rows_per_block と lanes を余分に渡す (v2 は 6 本のまま)。
    let (mut rows_c, mut lanes_c) = (rows_per_block as c_int, lanes as c_int);
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut n_in_c as *mut _ as *mut c_void,
        &mut n_out_c as *mut _ as *mut c_void,
        &mut rb_c as *mut _ as *mut c_void,
        &mut rows_c as *mut _ as *mut c_void,
        &mut lanes_c as *mut _ as *mut c_void,
    ];
    let nargs = if lanes > 1 { 8 } else { 6 };
    unsafe {
        ctx.launch_async(
            f,
            (grid, 1, 1),
            (block_dim, 1, 1),
            shmem,
            &mut args[..nargs],
        )
    }
}


/// task 184: GDN decode のスキャンを第 2 版 (`kernels_gdn_scan2.cu` の
/// `dno_gdn_scan_sig2`、4 レーン + 列レジスタ) に差し替えるか (`DENNO_GDN_SCAN2`)。
/// lab t184a で採否 yes → 既定 on。
///
/// 第 2 版は dk == dv == 128 / d_conv == 4 のときだけ使う (それ以外は従来の
/// `dno_gdn_scan_sig` へ fail closed)。丸めは skv / o の縮約順が変わる。
pub(crate) fn gdn_scan2_from_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}


/// lab t184a (depth8k) で採否 yes: scan 1 発が 25.0 -> 12.7 us
/// (128 トークンの生 decode 秒で -1.8..-2.3%)。戻すのは `DENNO_GDN_SCAN2=0`。
pub(crate) const GDN_SCAN2_DEFAULT: bool = true;


pub(crate) fn gdn_scan2_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        gdn_scan2_from_value(
            std::env::var("DENNO_GDN_SCAN2").ok().as_deref(),
            GDN_SCAN2_DEFAULT,
        )
    })
}


static GDN_SCAN2_KERNEL: OnceLock<Option<Scan2Kernel>> = OnceLock::new();


/// 第 2 版のカーネル (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn gdn_scan2_kernel() -> Option<CUfunction> {
    GDN_SCAN2_KERNEL
        .get_or_init(|| {
            let path = scan2_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, "dno_gdn_scan_sig2") != Some(23) {
                eprintln!("[gdn-scan2] PTX の引数が 23 本でない → 従来 scan へ");
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(
                    &mut f,
                    module,
                    b"dno_gdn_scan_sig2\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
                {
                    return None;
                }
                Some(Scan2Kernel { module, f })
            }
        })
        .as_ref()
        .map(|k| k.f)
}


/// task 592: GDN decode のスキャンを第 3 版 (`kernels_gdn_scan3.cu` の
/// `dno_gdn_scan_sig3`、**1 列 8 レーン** + 列レジスタ) に差し替えるか
/// (`DENNO_GDN_SCAN3`、既定 off)。
///
/// lab t592b で **1 列 8 レーン (1024 スレッド) は ssm 段 +3.5% で負けた**
/// (2.82 -> 2.92 ms/tok、2 ラウンド一致、md5 SAME)。48 ブロックしか無いので
/// ブロックは 48 SM にしか載らず、warp を 2 倍にしても SM は増えない。増えるのは
/// `__syncthreads` の参加 warp だけ (5 回/token) で、損になる。このカーネルは
/// メモリのレイテンシ隠蔽では決まっていない (P588 §4 の見立ては外れ)。
///
/// 第 3 版はその反省で、スレッド構成は第 2 版と同じ (1 列 4 レーン / 512 スレッド)
/// のまま、**単一スレッドの逐次和 2 本の反復数をコンパイル時定数にして展開**する
/// (`dk` / `dv` が引数だと展開されず、shared のロード latency が 128 反復ぶん
/// 露出していた)。和の順序は変えないので**値は第 2 版とビット一致**。
///
/// 既定 on (lab t592c)。on のときは第 2 版より優先する (形と PTX が揃えば)。
pub(crate) fn gdn_scan3_from_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}


/// lab t592c で採否 yes: ssm 段 2.81-2.82 -> 2.71-2.73 ms/tok (-2.8%、2 ラウンド
/// 一致、md5 全 COND SAME = 値は scan2 とビット一致、errors 0)。既定 on。
/// 戻すのは `DENNO_GDN_SCAN3=0` (第 2 版 scan2 へ)。
pub(crate) const GDN_SCAN3_DEFAULT: bool = true;


pub(crate) fn gdn_scan3_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        gdn_scan3_from_value(
            std::env::var("DENNO_GDN_SCAN3").ok().as_deref(),
            GDN_SCAN3_DEFAULT,
        )
    })
}


static GDN_SCAN3_KERNEL: OnceLock<Option<Scan3Kernel>> = OnceLock::new();


/// 第 3 版のカーネル (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn gdn_scan3_kernel() -> Option<CUfunction> {
    GDN_SCAN3_KERNEL
        .get_or_init(|| {
            let path = scan3_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, "dno_gdn_scan_sig3") != Some(23) {
                eprintln!("[gdn-scan3] PTX の引数が 23 本でない → 従来 scan へ");
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(
                    &mut f,
                    module,
                    b"dno_gdn_scan_sig3\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
                {
                    return None;
                }
                // 1024 スレッドのブロックを積めるか (lab t592a は launch_bounds 無しで
                // error 701 = LAUNCH_OUT_OF_RESOURCES になった)。積めない実体なら
                // 起動せずに第 2 版へ落とす (ラッチで decode 経路ごと止めないため)。
                if let Some(g) = lib.func_get_attribute {
                    let mut maxthr: c_int = 0;
                    if g(&mut maxthr, 0, f) == CUDA_SUCCESS && (maxthr as u32) < 4 * 128 {
                        eprintln!(
                            "[gdn-scan3] max_threads={maxthr} < {} → 第 2 版 scan へ",
                            4 * 128
                        );
                        return None;
                    }
                }
                log_kernel_attr(lib, f, "dno_gdn_scan_sig3");
                Some(Scan3Kernel { module, f })
            }
        })
        .as_ref()
        .map(|k| k.f)
}


/// `gdn_decode_submit` が使う scan カーネルと blockDim.x。
///
/// 第 3 版は第 2 版と同じく 1 列 4 レーンなので block は dk の 4 倍
/// (`DENNO_GDN_SCAN3`。差は逐次和の定数展開だけ)。
/// 形が合わない / PTX が無いときは第 2 版 → 従来カーネルの順に落ちる。
pub(crate) fn gdn_scan_pick(base: CUfunction, dk: usize, dv: usize) -> (CUfunction, u32) {
    if dk == 128 && dv == 128 {
        if gdn_scan3_on() {
            if let Some(f) = gdn_scan3_kernel() {
                return (f, (4 * dk) as u32);
            }
        }
        if gdn_scan2_on() {
            if let Some(f) = gdn_scan2_kernel() {
                return (f, (4 * dk) as u32);
            }
        }
    }
    (base, dk as u32)
}


/// デコードの ssm 層 1 層ぶんを 1 つの GPU submission にまとめる。
///
/// 成功なら `out: Some(層出力)` を返す。失敗なら `out: None` で従来の CPU
/// 経路に落ちる (fail-closed)。scan を発行したあとの失敗では状態を回復して
/// `state_back` / `poisoned` で通知する。
///
/// sync 発数: 成功時は最後の D2H の 1 回だけ (check モードでは状態の D2H が
/// 増える)。途中の H2D はすべて同期コピーだが、発行時点で stream は空なので
/// 待ちにならない (前層の終点で同期済み)。
pub fn gdn_decode_submit(req: &GdnDecodeReq) -> GdnDecodeRes {
    if GDN_SCAN_UNAVAILABLE.load(std::sync::atomic::Ordering::Relaxed) {
        // 1 回でも GPU 側で失敗した後は CUDA に触れない (発行しない)。
        return GdnDecodeRes::cpu();
    }
    if !gdn_gpu_scan_on() || !cuda_enabled() {
        return GdnDecodeRes::cpu();
    }
    // dno_gdn_scan_sig の前提 (gdn.cu 冒頭コメント・ホスト側 assert と同一)。
    if req.dk != req.dv || req.d_conv != 4 || req.dk > 256 {
        gdn_note_once("前提外の寸法 (dk != dv / d_conv != 4 / dk > 256) → CPU 経路");
        return GdnDecodeRes::cpu();
    }
    let Some(ctx) = shared_ctx() else {
        return GdnDecodeRes::cpu();
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed (gdn decode): {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return GdnDecodeRes::cpu();
    }
    let Some(f_scan_base) = ctx.f_gdn_scan_sig else {
        // コミット済み PTX に dno_gdn_scan_sig が無い (再生成待ち)。
        gdn_note_once(
            "dno_gdn_scan_sig が使う PTX に無い (旧 kernels_gdn.ptx) → CPU 経路",
        );
        return GdnDecodeRes::cpu();
    };
    // task 184: scan 第 2 版 (`DENNO_GDN_SCAN2`)。形が合わない / PTX が無い
    // ときは従来カーネルへ落ちる (数値は従来どおり)。
    let (f_scan, scan_thr) = gdn_scan_pick(f_scan_base, req.dk, req.dv);
    // GEMV のカーネル選択は intercept 経路と同じ規約 (fp16 → v3 → v2)。
    // task 177: この経路だけ `DENNO_GDN_DEC_GEMV` で差し替えられる (既定は従来)。
    // v4 を選んでいても、その単位で形が合わなければ v3 へ落ちる (CPU へは落とさない)。
    let mode = gdn_dec_gemv_mode_eff();
    let v4 = gdn_dec_gemv_v4_on().then(gdn_dec_v4_lanes_cap);
    let v5 = gdn_dec_gemv_v5_on();
    let plan = |ty: crate::kquant::GgmlType, k: usize, n: usize| {
        v5.then(|| ctx.plan_moe_gemv_v5(ty, k, n))
            .flatten()
            .or_else(|| v4.and_then(|cap| ctx.plan_moe_gemv_v4(ty, k, n, cap)))
            .or_else(|| ctx.plan_moe_gemv(ty, k, n, mode).ok().flatten())
    };
    let mut qcfg = Vec::with_capacity(req.qkv_units.len());
    for (u, _) in req.qkv_units {
        match plan(u.ty, u.k, u.n) {
            Some(c) => qcfg.push(c),
            None => return GdnDecodeRes::cpu(),
        }
    }
    let Some(ocfg) = plan(req.out_unit.ty, req.out_unit.k, req.out_unit.n) else {
        return GdnDecodeRes::cpu();
    };
    // 重みは常駐登録簿から (intercept 経路と同じ uid → 同じ VRAM を共有)。
    let mut qw = Vec::with_capacity(req.qkv_units.len());
    for (u, _) in req.qkv_units {
        match weights_for_gguf(u.uid, u.ty, u.blocks, u.k, u.n) {
            Ok(w) => qw.push(w),
            Err(e) => {
                gdn_note_once(&format!("qkv_gate の重み登録に失敗 → CPU 経路: {e:#}"));
                return gdn_scan_record_failure(GdnDecodeRes::cpu());
            }
        }
    }
    let ow = match weights_for_gguf(
        req.out_unit.uid,
        req.out_unit.ty,
        req.out_unit.blocks,
        req.out_unit.k,
        req.out_unit.n,
    ) {
        Ok(w) => w,
        Err(e) => {
            gdn_note_once(&format!("attn_out の重み登録に失敗 → CPU 経路: {e:#}"));
            return gdn_scan_record_failure(GdnDecodeRes::cpu());
        }
    };
    let mut store_lock = match GDN_DECODE_STORE.lock() {
        Ok(s) => s,
        Err(_) => return GdnDecodeRes::cpu(),
    };
    let idx = match gdn_decode_entry_ensure_locked(&mut store_lock, req) {
        Ok(i) => i,
        Err(e) => {
            gdn_note_once(&format!("状態バッファの確保に失敗 → CPU 経路: {e:#}"));
            return gdn_scan_record_failure(GdnDecodeRes::cpu());
        }
    };
    {
        let (_, e) = &store_lock[idx];
        if (
            e.n_embd,
            e.qkv_total,
            e.vd,
            e.conv_len,
            e.ssm_len,
            e.nv,
            e.dk,
            e.dv,
            e.nk,
            e.conv_dim,
            e.d_conv,
        ) != (
            req.n_embd,
            req.qkv_total,
            req.nv * req.dv,
            req.conv_dim * (req.d_conv - 1),
            req.nv * req.dk * req.dv,
            req.nv,
            req.dk,
            req.dv,
            req.nk,
            req.conv_dim,
            req.d_conv,
        ) {
            gdn_note_once("寸法が確保済みエントリと違う (バグの疑い) → CPU 経路");
            return GdnDecodeRes::cpu();
        }
        // task 464: `DENNO_MIX_NOWAIT` の defer 回は `cur` が空で来る (入力は mix の
        // 鎖がデバイスに残した `dm`)。空は控えが armed のときだけ許し、控えが無ければ
        // **CPU 経路へ落とさず**に要求を落とす (空の入力で黙って計算しない)。
        let cur_empty = req.cur.is_empty();
        if (req.cur.len() != req.n_embd && !cur_empty)
            || req.conv_host.len() != e.conv_len
            || req.ssm_host.len() != e.ssm_len
            || (req.ba_units.is_empty()
                && (req.beta.len() != req.nv || req.alpha.len() != req.nv))
        {
            gdn_note_once("入力長が寸法と違う (バグの疑い) → CPU 経路");
            return GdnDecodeRes::cpu();
        }
        if cur_empty && !gdn_x_in_mix_armed(req.n_embd) {
            // 呼び出し側の前提 (mix の鎖が `dm` を残す) が崩れている。CPU 経路へ
            // 落とすと空の入力で計算してしまうので、poisoned で要求を落とす。
            gdn_note_once("入力が空で mix の控えも無い → 要求を落とす");
            return GdnDecodeRes {
                out: None,
                state_back: None,
                poisoned: true,
                state_dev: None,
            };
        }
        // task 140: qkv GEMV の出力は `e.qkv` の行オフセットへ書く。合計が
        // qkv_total を越えるとデバイスバッファの外へ書いて CUDA error 700 になる
        // (t140a で実際に踏んだ)。ここで fail-closed に落とす。
        if req
            .qkv_units
            .iter()
            .any(|(u, row)| row.saturating_add(u.n) > req.qkv_total)
        {
            gdn_note_once("qkv GEMV の出力行範囲が qkv_total を越える (バグの疑い) → CPU 経路");
            return GdnDecodeRes::cpu();
        }
    }

    // ---- シード (ホストが正のときだけ。sync H2D 1 回 = トークン初回のみ) ----
    let seeded_needed = !store_lock[idx].1.device_authoritative;
    if seeded_needed {
        let (_, e) = &store_lock[idx];
        if let Err(err) = (|| -> Result<()> {
            e.conv[e.cur].copy_from(req.conv_host)?;
            e.ssm.copy_from(req.ssm_host)?;
            Ok(())
        })() {
            gdn_note_once(&format!("状態のシードに失敗 → CPU 経路: {err:#}"));
            return gdn_scan_record_failure(GdnDecodeRes::cpu());
        }
    }

    // ---- 診断ダンプ (DENNO_GDN_SUBMIT_DUMP=1・既定 off) ----------------------
    // 1 トークン目だけ層ごとに 1 行。off のときは分岐 1 回で何も出さない。
    if gdn_submit_dump_on() && req.tok <= 1 {
        let (_, e) = &store_lock[idx];
        let mut s = format!(
            "[gdn-dump] li={} tok={} mode={:?} nv={} nk={} dk={} dv={} conv_dim={} qkv_total={} \
             qkv_row_start={} z_row_start={} \
             buf(xba={} qkv={} y={} out={} conv={} ssm={}) seeded={} cur={}",
            req.li,
            req.tok,
            mode,
            req.nv,
            req.nk,
            req.dk,
            req.dv,
            req.conv_dim,
            req.qkv_total,
            req.qkv_row_start,
            req.z_row_start,
            e.xba.bytes(),
            e.qkv.bytes(),
            e.y.bytes(),
            e.out.bytes(),
            e.conv[e.cur].bytes(),
            e.ssm.bytes(),
            seeded_needed,
            e.cur
        );
        for (i, ((u, row), cfg)) in req.qkv_units.iter().zip(qcfg.iter()).enumerate() {
            s.push_str(&format!(
                " | u{i} ty={:?} k={} n={} row={} rb={} w={}B off={}B f={:?} rows={} blk={} grid={} shmem={} lanes={}",
                u.ty,
                u.k,
                u.n,
                row,
                cfg.rb,
                qw[i].buf.bytes(),
                *row * 4,
                cfg.f,
                cfg.rows_per_block,
                cfg.block_dim,
                cfg.grid,
                cfg.shmem,
                cfg.lanes
            ));
        }
        s.push_str(&format!(
            " | out ty={:?} k={} n={} rb={} w={}B rows={} blk={} grid={} shmem={} lanes={}",
            req.out_unit.ty,
            req.out_unit.k,
            req.out_unit.n,
            ocfg.rb,
            ow.buf.bytes(),
            ocfg.rows_per_block,
            ocfg.block_dim,
            ocfg.grid,
            ocfg.shmem,
            ocfg.lanes
        ));
        tracing::info!("{s}");
    }

    // ---- submission (発行を積んで最後の D2H で 1 回だけ待つ) ----------------
    let check = gdn_gpu_check_on();
    let (cur_face, out_face) = {
        let (_, e) = &store_lock[idx];
        (e.cur, 1 - e.cur)
    };
    let scan_launched = std::cell::Cell::new(false);
    let (dk, dv) = (req.dk, req.dv);
    let sync_dbg = gdn_submit_sync_on();
    // 診断 (`DENNO_GDN_SUBMIT_SYNC=1`): 段ごとに同期し、失敗した段を名前に残す。
    // 700 は「壊れた launch の後、次にホストと同期する API」が返すので、これが
    // 無いと段を特定できない (off のときは 1 バイトも挙動が変わらない)。
    macro_rules! stage_sync {
        ($name:expr) => {
            if sync_dbg {
                if let Err(e) = ctx.sync() {
                    return Err(anyhow!("段 {} で失敗: {e:#}", $name));
                }
            }
        };
    }
    let run = || -> Result<(Vec<f32>, Option<(Vec<f32>, Vec<f32>)>)> {
        // task 177: この submission の操作別内訳 (`DENNO_GDN_DEC_KPROF`・既定 off)。
        // guard の drop (この closure の出口) で表を出す。
        let _kp_win = dec_kprof::window(dec_kprof::GDN, req.li, dec_kprof::cur_pos(), 0);
        {
            // 重みの実バイト (qkv の各単位 + attn_out)。実効帯域の分母に使う。
            let q: usize = req
                .qkv_units
                .iter()
                .zip(qcfg.iter())
                .map(|((u, _), c)| u.n.saturating_mul(c.rb))
                .sum();
            dec_kprof::note_wbytes(q.saturating_add(req.out_unit.n.saturating_mul(ocfg.rb)));
        }
        let (_, e) = &store_lock[idx];
        // task 392: 直前の mix 鎖が作った `mixed` (デバイス) をそのまま入力に
        // 使えるなら、cur の H2D を丸ごと省く (同じバイト列 = ビット一致)。
        // ba をホストで計算するモデル (`ba_units` が空) は staging に
        // beta/alpha を詰める必要があるので従来どおり。
        let x_alias = if gdn_x_mix_on() && !req.ba_units.is_empty() {
            gdn_x_in_mix_take(req.n_embd)
        } else {
            None
        };
        // task 464: mix の鎖を defer した回は `req.cur` が空で来る。控えが無いまま
        // 進むと 0 バイトを staging に積んで黙って壊れた値を出すので、ここで落とす。
        if x_alias.is_none() && req.cur.len() < req.n_embd {
            bail!(
                "gdn decode: 入力 cur が {} 要素 (n_embd={}) で mix の控えも無い (DENNO_MIX_NOWAIT の前提が崩れた)",
                req.cur.len(),
                req.n_embd
            );
        }
        // task 140: cur / beta / alpha を 1 本の staging に詰めて **H2D 1 回**で
        // 上げる (以前は同期 H2D 3 回。値・並びは同一で、カーネルへ渡す
        // ポインタだけが xba の先頭/中間を指す)。
        if x_alias.is_none() {
            GDN_STAGE.with(|cell| -> Result<()> {
                let mut st = cell.borrow_mut();
                st.clear();
                st.extend_from_slice(req.cur);
                // task 269: beta/alpha をデバイスで計算するときは詰めない
                // (節域は同じ位置にあり、matvec の出力がそのまま入る)。
                if req.ba_units.is_empty() {
                    st.extend_from_slice(req.beta);
                    st.extend_from_slice(req.alpha);
                }
                e.xba.copy_from(st.as_slice())
            })?;
            stage_sync!("xba_h2d");
            note_gdn_x_mix_miss();
        } else {
            note_gdn_x_mix_hit();
        }
        // x は xba の先頭 (従来) か mix の `mixed` (H2D を省いた回)。
        // beta/alpha は常に xba の側に置く。
        //
        // task 900: ba 出力ブロックの先頭 (`pba`) と、scan が読む beta / alpha の
        // 位置は別物。連結済みの ba をデバイスで計算するときは単位の出力が
        // 連結バッファ座標 (= ファイル順) に並ぶので、読み出し位置は req が持つ
        // (`range(i).start`。補正 off では従来どおり (0, nv))。
        let pba = match x_alias {
            Some(_) => e.xba.ptr(),
            None => e.xba.ptr() + (req.n_embd * 4) as CUdeviceptr,
        };
        let (pb, pa) = (
            pba + (req.ba_beta_row * 4) as CUdeviceptr,
            pba + (req.ba_alpha_row * 4) as CUdeviceptr,
        );
        // qkv GEMV の入力 (mix の控えがあればそれ)。
        let px = match x_alias {
            Some(p) => p,
            None => e.xba.ptr(),
        };
        // task 269: beta/alpha の f32 matvec を qkv GEMV より前に積む
        // (scan は最後なので順序は同一ストリームで保証される)。
        //
        // task 392: `DENNO_GDN_BA_PAR` のときは ba を専用ストリームへ出し、
        // qkv GEMV と同じ時間帯に走らせる (ba は x だけで決まる = 独立)。
        // 診断の段同期 (`DENNO_GDN_SUBMIT_SYNC`) が on のときは従来の直列
        // (段ごとの同期の意味を保つ)。
        let ba_stream = if !sync_dbg && !req.ba_units.is_empty() && gdn_ba_par_on() {
            gdn_ba_par_arm()
        } else {
            None
        };
        let ba_overlapped = ba_stream.is_some();
        for (u, off) in req.ba_units {
            gdn_launch_ba_on(ctx, ba_stream, u, px, pba + (*off * 4) as CUdeviceptr)?;
            stage_sync!("ba_gemv");
        }
        if ba_overlapped {
            // ba を積み終わった地点を専用ストリームに印す (join は scan の前)。
            gdn_ba_par_mark_done()?;
        }
        for (i, ((u, row), cfg)) in req.qkv_units.iter().zip(qcfg.iter()).enumerate() {
            // task 496: 先に dn8 (レーン詰め版)。np が 32 の倍数の形は None で dn6 へ。
            let dn8 = gdn_dn8_plan(u.ty, u.k, u.n, gdn_dn8_rows());
            if let (Some(dp), Some(rb)) = (dn8, gguf_row_bytes(u.ty, u.k)) {
                launch_mix_dn8(
                    ctx,
                    &dp,
                    qw[i].buf.ptr(),
                    px,
                    e.qkv.ptr() + (*row * 4) as CUdeviceptr,
                    u.k,
                    u.n,
                    rb,
                )?;
                MIX_DN8_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                stage_sync!(format!("qkv_gemv[{i}]"));
                continue;
            }
            // task 422: Q8_0 で形が合えば dn6 (1 ブロック rows 行 / 1 レーン 1 対)。
            match gdn_dn6_plan(u.ty, u.k, u.n) {
                Some((warps, rows)) => {
                    launch_mix_dn6(
                        ctx,
                        qw[i].buf.ptr(),
                        px,
                        e.qkv.ptr() + (*row * 4) as CUdeviceptr,
                        u.k,
                        u.n,
                        cfg.rb,
                        warps,
                        rows,
                    )?;
                    MIX_DN6_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                None => gdn_launch_gemv_dev(
                    ctx,
                    cfg,
                    &qw[i].buf,
                    px,
                    e.qkv.ptr() + (*row * 4) as CUdeviceptr,
                    u.k,
                    u.n,
                )?,
            }
            stage_sync!(format!("qkv_gemv[{i}]"));
        }
        let (mut pq, mut pz) = (
            e.qkv.ptr() + (req.qkv_row_start * 4) as CUdeviceptr,
            e.qkv.ptr() + (req.z_row_start * 4) as CUdeviceptr,
        );
        let (mut pb, mut pa) = (pb, pa);
        let (mut pci, mut pco) = (e.conv[cur_face].ptr(), e.conv[out_face].ptr());
        let (mut psi, mut pso) = (e.ssm.ptr(), e.ssm.ptr());
        let (mut pwc, mut pwa) = (e.wconv.ptr(), e.wa.ptr());
        let (mut pwd, mut pwn) = (e.wdt.ptr(), e.wnorm.ptr());
        let mut py = e.y.ptr();
        let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
            1 as c_int,
            req.nk as c_int,
            req.nv as c_int,
            req.dk as c_int,
            req.dv as c_int,
        );
        let (mut cdi, mut dci, mut csi) = (
            req.conv_dim as c_int,
            req.d_conv as c_int,
            (req.d_conv - 1) as c_int,
        );
        let (mut epsi, mut qsi) = (req.eps, 1.0f32 / (req.dk as f32).sqrt());
        let mut args = [
            &mut pq as *mut _ as *mut c_void,
            &mut pz as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut pci as *mut _ as *mut c_void,
            &mut psi as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void,
            &mut pwa as *mut _ as *mut c_void,
            &mut pwd as *mut _ as *mut c_void,
            &mut pwn as *mut _ as *mut c_void,
            &mut py as *mut _ as *mut c_void,
            &mut pco as *mut _ as *mut c_void,
            &mut pso as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut nki as *mut _ as *mut c_void,
            &mut nvi as *mut _ as *mut c_void,
            &mut dki as *mut _ as *mut c_void,
            &mut dvi as *mut _ as *mut c_void,
            &mut cdi as *mut _ as *mut c_void,
            &mut dci as *mut _ as *mut c_void,
            &mut csi as *mut _ as *mut c_void,
            &mut epsi as *mut _ as *mut c_void,
            &mut qsi as *mut _ as *mut c_void,
        ];
        if ba_overlapped {
            // task 392: 専用ストリームの ba を待ってから scan を積む。
            gdn_ba_par_join()?;
        }
        let shmem = ((3 * dk + dv + 5) * 4) as u32;
        unsafe {
            ctx.launch_async(
                f_scan,
                (req.nv as u32, 1, 1),
                (scan_thr, 1, 1),
                shmem,
                &mut args,
            )?;
        }
        scan_launched.set(true);
        stage_sync!("scan");
        // task 496: out 投影も `dn8a` のときは dn8 (np が 32 の倍数でも x を
        // shared へ 1 回だけ置く形が効くかを見る)。np 96 の形は dn8a でのみ。
        let dn8_out = gdn_dn8_plan(
            req.out_unit.ty,
            req.out_unit.k,
            req.out_unit.n,
            GDN_DN8_OUT_ROWS,
        );
        // task 504: `DENNO_GDN_DEC_GEMV=dn8ao7` のときは out だけ dn7
        // (1 対 = 4 レーン) で撃つ。形が合わなければ下の dn8 / dn6 へ落ちる。
        let dn7_out = if gdn_out_dn7_on() {
            dn7_dec_plan(req.out_unit.ty, req.out_unit.k, req.out_unit.n)
        } else {
            None
        };
        if let (Some((rows7, thr7)), Some(rb)) = (
            dn7_out,
            gguf_row_bytes(req.out_unit.ty, req.out_unit.k),
        ) {
            launch_mix_dn7(
                ctx,
                ow.buf.ptr(),
                e.y.ptr(),
                e.out.ptr(),
                req.out_unit.k,
                req.out_unit.n,
                rb,
                rows7,
                thr7,
            )?;
            note_dn7_out_hit();
            stage_sync!("out_gemv");
        } else if let (Some(dp), Some(rb)) = (dn8_out, gguf_row_bytes(req.out_unit.ty, req.out_unit.k))
        {
            launch_mix_dn8(
                ctx,
                &dp,
                ow.buf.ptr(),
                e.y.ptr(),
                e.out.ptr(),
                req.out_unit.k,
                req.out_unit.n,
                rb,
            )?;
            MIX_DN8_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            stage_sync!("out_gemv");
        } else {
        // task 422: out 投影も Q8_0 で形が合えば dn6。
        match gdn_dn6_plan(req.out_unit.ty, req.out_unit.k, req.out_unit.n) {
            Some((warps, rows)) => {
                launch_mix_dn6(
                    ctx,
                    ow.buf.ptr(),
                    e.y.ptr(),
                    e.out.ptr(),
                    req.out_unit.k,
                    req.out_unit.n,
                    ocfg.rb,
                    warps,
                    rows,
                )?;
                MIX_DN6_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            None => gdn_launch_gemv_dev(
                ctx,
                &ocfg,
                &ow.buf,
                e.y.ptr(),
                e.out.ptr(),
                req.nv * req.dv,
                req.n_embd,
            )?,
        }
        }
        stage_sync!("out_gemv");
        let out = e.out.to_vec::<f32>(req.n_embd)?;
        // 診断 (DENNO_GDN_GPU_CHECK): scan 後のデバイス状態も読む。
        let state_dev = if check {
            let conv = e.conv[out_face].to_vec::<f32>(e.conv_len)?;
            let ssm = e.ssm.to_vec::<f32>(e.ssm_len)?;
            Some((conv, ssm))
        } else {
            None
        };
        Ok((out, state_dev))
    };
    let r = run();
    match r {
        Ok((out, state_dev)) => {
            let (_, e) = &mut store_lock[idx];
            e.cur = out_face;
            e.device_authoritative = true;
            GdnDecodeRes {
                out: Some(out),
                state_back: None,
                poisoned: false,
                state_dev,
            }
        }
        Err(err) => {
            if !scan_launched.get() {
                // scan 発行前の失敗: デバイス状態はホストと等価 (シード済みなら
                // ホストの写しだけ)。ホスト state は無傷なのでそのまま CPU へ。
                gdn_note_once(&format!(
                    "submission の発行に失敗 (li={} tok={}) → CPU 経路: {err:#}",
                    req.li, req.tok
                ));
                return gdn_scan_record_failure(GdnDecodeRes::cpu());
            }
            // scan 発行後の失敗: 新状態はデバイスの書き込み面にある。回復を
            // 試みて、成功ならホストへ写させる。失敗したら poisoned。
            let (_, e) = &mut store_lock[idx];
            match (
                e.conv[out_face].to_vec::<f32>(e.conv_len),
                e.ssm.to_vec::<f32>(e.ssm_len),
            ) {
                (Ok(conv), Ok(ssm)) => {
                    e.device_authoritative = false;
                    gdn_note_once(&format!(
                        "submission の後段で失敗 (状態は回復済み) → CPU 経路: {err:#}"
                    ));
                    gdn_scan_record_failure(GdnDecodeRes {
                        out: None,
                        state_back: Some((conv, ssm)),
                        poisoned: false,
                        state_dev: None,
                    })
                }
                _ => {
                    tracing::error!(
                        "[gdn-gpu-scan] scan 発行後の失敗かつ状態回復も失敗 (li={}) → 要求を落とすこと: {err:#}",
                        req.li
                    );
                    gdn_scan_record_failure(GdnDecodeRes {
                        out: None,
                        state_back: None,
                        poisoned: true,
                        state_dev: None,
                    })
                }
            }
        }
    }
}


// ---- task 128: GPU 不要の CPU テスト (ゲート off / ラッチ / Req 組み立て) ----

#[cfg(test)]
mod gdn_decode_scan_tests {
    use super::*;

    /// テストが立てたフラグ (ラッチ / scan ゲート / GPU 強制) を panic でも戻す。
    struct LatchGuard;
    impl Drop for LatchGuard {
        fn drop(&mut self) {
            gdn_scan_unavailable_reset_for_test();
            gdn_gpu_scan_set_for_test(false);
        }
    }

    fn empty_req<'a>(units: &'a [(GdnGgufUnit<'a>, usize)]) -> GdnDecodeReq<'a> {
        GdnDecodeReq {
            nonce: 0xBEEF,
            li: 3,
            tok: 1,
            n_embd: 0,
            qkv_total: 0,
            qkv_row_start: 0,
            z_row_start: 0,
            qkv_units: units,
            out_unit: GdnGgufUnit {
                uid: 0,
                ty: crate::kquant::GgmlType::Q8_0,
                blocks: &[],
                k: 0,
                n: 0,
            },
            cur: &[],
            beta: &[],
            alpha: &[],
            ba_units: &[],
            ba_beta_row: 0,
            ba_alpha_row: 0,
            conv_host: &[],
            ssm_host: &[],
            wconv: &[],
            wa: &[],
            wdt: &[],
            wnorm: &[],
            nk: 0,
            nv: 0,
            dk: 0,
            dv: 0,
            conv_dim: 0,
            d_conv: 4,
            eps: 1e-6,
        }
    }

    /// task 140: `DENNO_GDN_SUBMIT_SYNC` / `DENNO_GDN_SUBMIT_DUMP` の真理値表
    /// (純関数)。既定はどちらも off。
    #[test]
    fn gdn_submit_sync_truth_table() {
        for (raw, want) in [
            (None, false),
            (Some("0"), false),
            (Some(""), false),
            (Some("off"), false),
            (Some("yes"), false),
            (Some("1"), true),
            (Some("on"), true),
            (Some("true"), true),
        ] {
            assert_eq!(
                parse_gdn_submit_sync(raw),
                want,
                "DENNO_GDN_SUBMIT_SYNC={raw:?} の解釈が違う"
            );
            assert_eq!(
                parse_gdn_submit_dump(raw),
                want,
                "DENNO_GDN_SUBMIT_DUMP={raw:?} の解釈が違う"
            );
        }
    }

    /// task 142: `DENNO_GDN_GPU_SCAN` の真理値表。**未設定は on** (task 142 で採用)、
    /// 明示的な `0|off|false|no` (空白・大小文字は無視) だけ off。
    #[test]
    fn gdn_gpu_scan_truth_table_default_on() {
        for (raw, want) in [
            (None, true),
            (Some(""), true),
            (Some(" "), true),
            (Some("1"), true),
            (Some("on"), true),
            (Some("true"), true),
            (Some("YES"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(" OFF "), false),
            (Some("no"), false),
        ] {
            assert_eq!(
                parse_gdn_gpu_scan(raw),
                want,
                "DENNO_GDN_GPU_SCAN={raw:?} の解釈が違う"
            );
        }
    }

    /// task 142: env を立てなくてもゲートは on に解決し、強制 off で落ちること
    /// (既定の解決と両方向の強制を数える)。
    #[test]
    fn gdn_gpu_scan_default_on_and_forced_off() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        gdn_gpu_scan_set_for_test(false);
        assert!(!gdn_gpu_scan_on(), "強制 off が効いていない");
        gdn_gpu_scan_set_for_test(true);
        assert!(gdn_gpu_scan_on(), "強制 on が効いていない");
        gdn_gpu_scan_reset_force_for_test();
        if std::env::var_os("DENNO_GDN_GPU_SCAN").is_none() {
            assert!(gdn_gpu_scan_on(), "env 未設定の既定が on でない");
        }
        // 後続テストのために強制 off に戻す (LatchGuard と同じ状態)。
        gdn_gpu_scan_set_for_test(false);
    }

    /// GPU 側の失敗ラッチ: (1) 初回だけ「立った」が true で以後冪等、(2) ラッチは
    /// GPU_BROKEN を立てない = intercept 経路に影響しない、(3) ラッチがあると
    /// submit は CUDA / デバイス状態ストアに触れずに即 CPU 経路を返す。
    #[test]
    fn gdn_scan_latch_blocks_submit_without_cuda_and_spares_gpu_broken() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let _lg = LatchGuard;
        gdn_scan_unavailable_reset_for_test();

        assert!(!gdn_scan_unavailable_for_test(), "初期状態は立っていない");
        assert!(gdn_scan_mark_unavailable(), "初回は立った");
        assert!(!gdn_scan_mark_unavailable(), "2 度目は既に立っている (冪等)");
        assert!(!gpu_broken_flag(), "ラッチは GPU_BROKEN を立てない");

        gdn_gpu_scan_set_for_test(true);
        let units: [(GdnGgufUnit<'_>, usize); 0] = [];
        let req = empty_req(&units);
        let res = gdn_decode_submit(&req);
        assert!(res.out.is_none(), "ラッチで即 CPU 経路");
        assert!(!res.poisoned, "ラッチは要求を落とさない");
        assert!(res.state_back.is_none());
        assert!(res.state_dev.is_none());
        assert_eq!(
            gdn_decode_state_entry_count(),
            0,
            "ラッチ経路はストアに触れない"
        );

        // ラッチを戻した同一 req でも、GPU 無しのコンテナでは同じく cpu() で
        // ストアは増えない (decline と latch が外から同じ結果 = 契約不変)。
        gdn_scan_unavailable_reset_for_test();
        let res2 = gdn_decode_submit(&req);
        assert!(res2.out.is_none());
        assert_eq!(gdn_decode_state_entry_count(), 0);
    }

    /// ゲート強制 off では submit が CUDA に触れず、ストアも増えない。
    #[test]
    fn gdn_scan_gate_off_never_touches_store() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let _lg = LatchGuard;
        gdn_scan_unavailable_reset_for_test();
        assert!(!gdn_gpu_scan_on(), "強制 off が効いていない");
        let units: [(GdnGgufUnit<'_>, usize); 0] = [];
        let req = empty_req(&units);
        let res = gdn_decode_submit(&req);
        assert!(res.out.is_none());
        assert!(!res.poisoned);
        assert_eq!(gdn_decode_state_entry_count(), 0, "ゲート off でストアが増えた");
    }

    /// task 128: reset の無効化が分岐として発火すること (数える)。
    /// `gdn_decode_state_invalidate_counted` は reset 経路から呼ばれる。
    #[test]
    fn gdn_invalidate_counted_increments() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let _lg = LatchGuard;
        gdn_invalidate_calls_reset_for_test();
        assert_eq!(gdn_invalidate_calls_for_test(), 0);
        gdn_decode_state_invalidate_counted(0xABC, 5);
        assert_eq!(gdn_invalidate_calls_for_test(), 1, "記帳されていない");
        // ストアにエントリが無くても panic しない (no-op)。
        assert_eq!(gdn_decode_state_entry_count(), 0);
    }
}


// ---- task 1577: 連結 GDN 状態プールの CPU テスト (GPU 不要・純関数だけ) ----

#[cfg(test)]
mod gdn_pool_layout_tests {
    use super::*;

    /// env の真偽表: 1/on/true/yes (大小・前後空白を許す) だけ有効。
    #[test]
    fn gdn_pool_env_truth_table() {
        for on in ["1", "on", "ON", " true ", "yes", "Yes", "TRUE"] {
            assert!(parse_gdn_pool(Some(on)), "{on:?} は有効のはず");
        }
        for off in ["", "0", "off", "OFF", "false", "no", "2", "enable", "1x"] {
            assert!(!parse_gdn_pool(Some(off)), "{off:?} は無効のはず");
        }
        assert!(!parse_gdn_pool(None));
    }

    /// 256 B 境界への切り上げ (conv と ssm のスロットが混ざらない前提)。
    #[test]
    fn gdn_pool_slot_alignment() {
        assert_eq!(GDN_POOL_ALIGN, 256);
        assert_eq!(gdn_pool_align_up(0), 0);
        assert_eq!(gdn_pool_align_up(1), 256);
        assert_eq!(gdn_pool_align_up(256), 256);
        assert_eq!(gdn_pool_align_up(257), 512);
        for n in [0usize, 1, 255, 256, 257, 172_032, 2_621_440] {
            assert_eq!(gdn_pool_align_up(n) % GDN_POOL_ALIGN, 0);
            assert!(gdn_pool_align_up(n) >= n);
        }
    }

    /// 1 層ぶんのレイアウト。0 要素の形は None (従来の層別確保へ落ちる)。
    #[test]
    fn gdn_pool_layer_layout() {
        // 27B の GDN: conv 43,008 要素 (172,032 B)、ssm [nv,dk,dv] = 2,621,440 要素。
        let (cb, sb, stride) = gdn_pool_layout(43_008, 1280, 2048).expect("載る形");
        assert_eq!(cb, 172_032); // 256 の倍数なのでそのまま
        assert_eq!(sb, gdn_pool_align_up(1280 * 2048 * 4));
        assert_eq!(stride, cb + sb);
        assert_eq!(stride % GDN_POOL_ALIGN, 0);

        // 0 要素は載せない (view の長さ 0 を作らない)。
        assert_eq!(gdn_pool_layout(0, 8, 8), None);
        assert_eq!(gdn_pool_layout(4, 0, 8), None);
        assert_eq!(gdn_pool_layout(4, 8, 0), None);
    }

    /// 総バイト数 = 層数 x ストライド。桁あふれは None。
    #[test]
    fn gdn_pool_total_bytes() {
        let (_, _, stride) = gdn_pool_layout(43_008, 1280, 2048).unwrap();
        assert_eq!(gdn_pool_bytes(8, 43_008, 1280, 2048), Some(8 * stride));
        assert_eq!(gdn_pool_bytes(0, 43_008, 1280, 2048), None);
        assert_eq!(gdn_pool_bytes(usize::MAX, 43_008, 1280, 2048), None);
    }

    /// 容量は 8 層から倍々。要求を満たす最小の 2 の冪 (8 以上)。
    #[test]
    fn gdn_pool_capacity_growth() {
        assert_eq!(gdn_pool_grow_cap(0, 1), 8);
        assert_eq!(gdn_pool_grow_cap(0, 8), 8);
        assert_eq!(gdn_pool_grow_cap(8, 9), 16);
        assert_eq!(gdn_pool_grow_cap(8, 16), 16);
        assert_eq!(gdn_pool_grow_cap(16, 17), 32);
        assert_eq!(gdn_pool_grow_cap(8, 48), 64);
        // 既に足りていれば伸ばさない (再確保しない = 窓を奪い直さない)。
        assert_eq!(gdn_pool_grow_cap(64, 48), 64);
    }

    /// task 1577 の主張: 全層が **1 本の連続領域** に収まり、P1372 が固定できた
    /// 1 層 (約 3 MiB) より多くの層を persisting 予算に載せられる。
    #[test]
    fn gdn_pool_single_range_covers_more_than_one_layer() {
        const LAYERS: usize = 48;
        let (cb, sb, stride) = gdn_pool_layout(43_008, 1280, 2048).unwrap();
        let total = gdn_pool_bytes(LAYERS, 43_008, 1280, 2048).unwrap();
        // 連続: 層 li の conv/ssm は [base, base+total) の中にあり、重ならない。
        for li in 0..LAYERS {
            let c = li * stride;
            let s = c + cb;
            assert!(c + cb <= total);
            assert!(s + sb <= total);
            assert!(c + cb <= s, "conv と ssm が重なる (layer {li})");
            if li + 1 < LAYERS {
                assert!(s + sb <= (li + 1) * stride, "層が重なる (layer {li})");
            }
        }
        // persisting 予算 (既定 48 MiB) に入る層数が 1 より大きいこと。
        // P1372 は層別確保のため 1 層 (3 MiB) しか固定できなかった。
        let budget = 48 * 1024 * 1024;
        assert!(
            budget / stride >= 4,
            "48 MiB で {} 層しか固定できない (stride {stride} B)",
            budget / stride
        );
        // 128 MiB の窓上限でも全層は入らない (窓は予算で clamp する前提)。
        assert!(total > 128 * 1024 * 1024);
    }
}
