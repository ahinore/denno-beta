//! GPU の計器の読み出し口。`cuda` feature の有無にかかわらず同じ名前で呼べる。
//!
//! `denno-cli` は `cuda` feature を持たない (feature は denno-engine 側にある) ので、
//! `crate::cuda` を直接呼ぶと CPU だけのビルドが通らない。ここを経由すれば、
//! CUDA 無しのビルドでは「何も積んでいない」(0 / `None`) が返る。

/// task 382: `DENNO_MIX_UPQ8` の融合版が発火した回数。
pub fn mix_upq8_hits() -> u64 {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_upq8_hits();
    #[cfg(not(feature = "cuda"))]
    0
}

/// task 382: [`mix_upq8_hits`] を 0 に戻す。
pub fn mix_upq8_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::mix_upq8_reset();
}

/// task 437: `DENNO_MIX_UPQ8=u4` 系 (1 対 = 4 レーンの up) が発火した回数。
pub fn mix_up4_hits() -> u64 {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_up4_hits();
    #[cfg(not(feature = "cuda"))]
    0
}

/// task 437: [`mix_up4_hits`] を 0 に戻す。
pub fn mix_up4_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::mix_up4_reset();
}

/// task 382: `DENNO_MIX_UPQ8=check` の比較結果 (積んでいるときだけ)。
pub fn mix_upq8_check_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_upq8_check_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// task 382: mix の鎖から H2D / D2H を消した回数 `(no_h2d, no_d2h)`。
pub fn mix_zc_hits() -> (u64, u64) {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_zc_hits();
    #[cfg(not(feature = "cuda"))]
    (0, 0)
}

/// task 392: GDN の ba matvec を qkv GEMV と重ねた層数の行
/// (`DENNO_GDN_BA_PAR=1` のときだけ 1 行 = 分岐が発火した証拠)。
pub fn gdn_ba_par_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::gdn_ba_par_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// task 392: [`gdn_ba_par_line`] の積算を 0 に戻す。
pub fn gdn_ba_par_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::gdn_ba_par_reset();
}

/// task 392: GDN の入力 `cur` の H2D を mix の device 出力で置き換えた層数の行
/// (`DENNO_GDN_X_MIX=1` のときだけ 1 行 = 分岐が発火した証拠)。
pub fn gdn_x_mix_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::gdn_x_mix_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// task 392: [`gdn_x_mix_line`] の積算を 0 に戻す。
pub fn gdn_x_mix_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::gdn_x_mix_reset();
}

/// task 464: mix の鎖の待ちを後ろ倒しにした層数の行
/// (`DENNO_MIX_NOWAIT=1` のときだけ 1 行 = 分岐が発火した証拠)。
pub fn mix_nowait_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_nowait_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// task 464: [`mix_nowait_line`] の積算を 0 に戻す。
pub fn mix_nowait_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::mix_nowait_reset();
}

/// task 468: full 層の attn mix も defer した層数の行
/// (`DENNO_MIX_NOWAIT=full` のときだけ 1 行 = 分岐が発火した証拠)。
pub fn mix_nowait_full_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::mix_nowait_full_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// task 468: [`mix_nowait_full_line`] の積算を 0 に戻す。
pub fn mix_nowait_full_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::mix_nowait_full_reset();
}

/// task 382: [`mix_zc_hits`] を 0 に戻す。
pub fn mix_zc_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::mix_zc_reset();
}

/// decode の同期ストリームの集計行 (積んでいるときだけ)。
pub fn dec_sync_stream_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::dec_sync_stream_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// [`dec_sync_stream_line`] の集計を 0 に戻す。
pub fn dec_sync_stream_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::dec_sync_stream_reset();
}

/// prefill の pinned 転送の集計行 (積んでいるときだけ)。
pub fn pf_pin_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return crate::cuda::pf_pin_line();
    #[cfg(not(feature = "cuda"))]
    None
}

/// [`pf_pin_line`] の集計を 0 に戻す。
pub fn pf_pin_reset() {
    #[cfg(feature = "cuda")]
    crate::cuda::pf_pin_reset();
}

/// task 448: decode の flash decoding の実装選択 (起動時に 1 行)。
/// lab の COND で env が届いたか、測ったバイナリが新しいかをログで判別する。
pub fn attn_fdec_boot_line() -> Option<String> {
    #[cfg(feature = "cuda")]
    return Some(crate::cuda::attn_fdec_boot_line());
    #[cfg(not(feature = "cuda"))]
    None
}
