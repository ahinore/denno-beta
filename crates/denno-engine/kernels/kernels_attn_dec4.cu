// task 460: decode の full attention で、combine (部分和の縮約) と
// 出力ゲート (sigmoid 乗算) を 1 起動にまとめたカーネル。
//
// 動機 (task 456 の `[dec-kprof-avg]`、14k の 1 層あたり):
//   `dno_attn_flashdec_combine` 16.2 us + `dno_dec_gate_mul` 3.8 us が
//   別々に走っていた。どちらも [heads * dh] の要素ごとの処理で、間に
//   同期もデータ依存も無い (combine の出力を gate が読むだけ)。
//   1 起動にすると 3.8 us の本体と、起動 1 本ぶんの隙間 (約 1.5 us) が消える。
//
// 値: combine の縮約順 (split 昇順・fmaf) も、ゲートの演算列
//   (dno_expf(-g) -> 1/(1+e) -> 掛ける) も変えないので、
//   `dno_attn_flashdec_combine` + `dno_dec_gate_mul` と **ビット一致**する。
//   gate に nullptr を渡すと縮約だけ (combine 単体と同じ値)。
//
// -fmad=false で作る (combine / dec_dev と同じ丸め。build_ptx.sh の attn_dec4)。

#include <cstdint>

#define DNO_CBG_LOG2E 1.4426950408889634f

// ---- dno_expf (kernels_dec_dev.cu の写し。定数も同じ) ----------------------
__device__ uint64_t DNO_CBG_EXP2_TAB[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
    0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
    0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
    0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
    0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
    0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
    0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
    0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
    0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
__device__ double DNO_CBG_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_CBG_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_CBG_EXP_SHIFT = 0x1.8p+52;

__device__ __forceinline__ uint64_t dno_cbg_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_cbg_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_cbg_asuint(float f) {
    return __float_as_uint(f);
}

__device__ __forceinline__ float dno_cbg_expf(float x) {
    uint32_t abstop = (dno_cbg_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (dno_cbg_asuint(x) == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    double xd = (double)x;
    double kd = __fma_rn(DNO_CBG_EXP_INVLN2N, xd, DNO_CBG_EXP_SHIFT);
    uint64_t ki = dno_cbg_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_CBG_EXP_SHIFT);
    double r = __fma_rn(DNO_CBG_EXP_INVLN2N, xd, -kd);
    uint64_t t = DNO_CBG_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_cbg_asdouble(t);
    double z2 = __fma_rn(DNO_CBG_EXP_POLY[0], r, DNO_CBG_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(DNO_CBG_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// part を split 昇順に畳み、その場で出力ゲート (sigmoid) を掛ける。
//   part [ (n_kv, n_split, group), dh + 2 ]  先頭 2 要素が m, l
//   att  [ heads * dh ]                      h = kv head * group + q head
//   gate [ heads * dh ]                      nullptr なら掛けない
// grid = (group, n_kv) / block = (dh)。
// 縮約 (fmaxf / exp2f / fmaf の順) は `dno_attn_flashdec_combine` と同一。
extern "C" __global__ void dno_attn_flashdec_cbg(
    const float *__restrict__ part,  // [(n_kv, n_split, group), dh + 2]
    float *__restrict__ att,         // [heads * dh]
    const float *__restrict__ gate,  // [heads * dh] or nullptr
    int group, int dh, int n_split) {
    const int kh = blockIdx.y;
    const int hh = blockIdx.x;
    const int d = threadIdx.x;
    const int h = kh * group + hh;
    const size_t stride = (size_t)(dh + 2);
    const size_t step = (size_t)group * stride;
    const float *base = part + ((size_t)(kh * n_split) * group + hh) * stride;

    float mg = -3.402823466e+38f;
    for (int sp = 0; sp < n_split; ++sp) mg = fmaxf(mg, base[(size_t)sp * step]);
    float denom = 0.0f;
    float num = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        const float *pp = base + (size_t)sp * step;
        // 空 split は m = -FLT_MAX -> e = 0 で寄与しない。
        const float e = exp2f((pp[0] - mg) * DNO_CBG_LOG2E);
        denom = fmaf(pp[1], e, denom);
        num = fmaf(pp[2 + d], e, num);
    }
    float val = (denom > 0.0f) ? __fdiv_rn(num, denom) : 0.0f;
    if (gate != nullptr) {
        const float g = gate[(size_t)h * dh + d];
        const float eg = dno_cbg_expf(-g);
        const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, eg));
        val = __fmul_rn(val, sg);
    }
    att[(size_t)h * dh + d] = val;
}
