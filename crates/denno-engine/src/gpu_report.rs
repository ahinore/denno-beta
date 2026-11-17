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
