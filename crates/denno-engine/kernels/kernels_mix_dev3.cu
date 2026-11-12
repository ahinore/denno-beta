// task 275: decode の hc mix で **inject を down の epilogue へ移す**。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh mix_dev3
//
// 生成物は `kernels_mix_dev3.ptx`。
//
// # なぜか (REPORT_P269 の mix 表)
//
// mix 1 回の GPU は down 9.3 + `dno_mix_norm_inj` 10.3 + up 7.8 + hc_mixed 5.8
// + D2H 3.5 = 36.7 us。そのうち `dno_mix_norm_inj` は **grid = 1** で
// 「RMS 正規化 → inject の内積」を 1 SM に載せている (10.3 us)。
// 正規化だけなら `dno_mix_norm` (grid = hc = 4 ブロック) で回せるが、inject は
// **正規化後の全ストリーム (k = hc*e)** を読むので 1 ブロックに閉じていた。
//
// 順序を見ると inject は正規化の直後である必要が無い: 正規化した x は
// 以降の down / hc_mixed から**読むだけ**で、書き換えない。したがって inject は
// down の epilogue へ移せる (down も同じ x を読む)。
//
//   いま: norm_inj (grid=1、正規化 + inject) → down → lo_apply → up → hc_mixed
//   あと: dno_mix_norm (grid=hc、正規化) → down+inject → lo_apply → up → hc_mixed
//
// 起動回数は増えない (`dno_mix_norm` は kernels_mix_dev.ptx に既にある)。
//
// # 数値の契約
//
// **ビット一致**: 正規化は `dno_mix_norm` (fold 版と同一の 256 スレッド木)、down は
// `dno_mix_down_bf16` (kernels_mix_down.cu) の写し、inject は `dno_mix_inj`
// (= `dno_mix_reduce256` と同じ 256 スレッド木・同じ刻み) の写し。
// ゲート `DENNO_DEC_MIX_NORM_PAR` で fold 版へ戻せる (既定 off)。
//
// 前提 (ホストが門番): hc <= 4、k が 8 の倍数、blockDim >= 256。
// inject は blockIdx.x < hc のブロックが行 blockIdx.x を担当する。

#include <cstdint>

#define N_TABLE 32  // EXP2F_TABLE_BITS = 5

// glibc sysdeps/ieee754/flt-32/e_exp2f_data.c (__exp2f_data)
// (kernels_mix_down.cu の写し。モジュールが別なので共有できない)
__device__ uint64_t DNO_EXP2_TAB[32] = {
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
// poly_scaled[i] = poly[i] / N^(3-i), N = 32
__device__ double DNO_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_EXP_SHIFT = 0x1.8p+52;  // 1.5 * 2^52

__device__ __forceinline__ uint64_t dno_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_asuint(float f) {
    return __float_as_uint(f);
}

/// glibc expf bit-exact replica (kernels_mix_down.cu と同一)。
__device__ __forceinline__ float dno_expf(float x) {
    uint32_t abstop = (dno_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (dno_asuint(x) == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    double xd = (double)x;
    double kd = __fma_rn(DNO_EXP_INVLN2N, xd, DNO_EXP_SHIFT);
    uint64_t ki = dno_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_EXP_SHIFT);
    double r = __fma_rn(DNO_EXP_INVLN2N, xd, -kd);
    uint64_t t = DNO_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_asdouble(t);
    double z2 = __fma_rn(DNO_EXP_POLY[0], r, DNO_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(DNO_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// BF16 の生 2 バイトを f32 へ (上位 16 bit の複製 = 丸めなし)。
__device__ __forceinline__ float dno_mixdn_bf16(unsigned int h)
{
    return __uint_as_float(h << 16);
}

// 256 スレッドの縮約 (kernels_mix_dev.cu の dno_mix_reduce256 と同一の木)。
//
// **ブロックの全スレッドが呼ぶ** (barrier を含むため)。有効な値を持つのは
// tid < 256 だけで、それ以外は木に寄与しない (warp 8 以降は red を書かない)。
__device__ __forceinline__ float dno_mix3_reduce256(float v, bool active, float* red)
{
    const int t = threadIdx.x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (active && (t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    if (t < 8) {
        float s = red[t];
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffu, s, off);
        }
        if (t == 0) red[8] = s;
    }
    __syncthreads();
    return red[8];
}

// ---------------------------------------------------------------------------
// down (BF16 重み + hc_lo_apply) + inject の epilogue。
//
// `dno_mix_down_bf16` (kernels_mix_down.cu) と同一の本体に、blockIdx.x < hc の
// ブロックだけ「inject の行 blockIdx.x」を足す:
//
//   inj[g] = Σ_i w_i[g][i] * x[i]      (i < ki、256 スレッドの木)
//
// x は正規化後の hc*e ベクトル (down が読むのと同じ)。wi が NULL のときは撃たない。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_down_bf16_inj(
    const float* __restrict__ x,          // [k]
    const unsigned char* __restrict__ w,  // [n][k] BF16
    float* __restrict__ out,              // [n]
    int k, int n, float inv_hc,
    const unsigned char* __restrict__ wi, // [hc][ki] BF16 (NULL 可)
    float* __restrict__ inj,              // [hc] (NULL 可)
    int ki, int hc)
{
    __shared__ float sh[32];
    __shared__ float ired[16];
    const int j = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    // ブロックで一様な条件 (barrier を分岐で割らないため)。
    const bool do_inj = (wi != 0) && (inj != 0) && (j < hc);

    if (j < n) {
        const unsigned short* rowp =
            (const unsigned short*)(w + (size_t)j * (size_t)k * 2u);
        float acc = 0.0f;
        // 1 スレッド = 8 要素 (uint4)。行頭もスレッドの開始位置も 16 B 整列。
        for (int c = tid * 8; c < k; c += (int)blockDim.x * 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(rowp + c);
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.x & 0xffffu), x[c + 0]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.x >> 16), x[c + 1]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.y & 0xffffu), x[c + 2]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.y >> 16), x[c + 3]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.z & 0xffffu), x[c + 4]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.z >> 16), x[c + 5]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.w & 0xffffu), x[c + 6]));
            acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.w >> 16), x[c + 7]));
        }
        // warp 内の固定木 (offset 16, 8, 4, 2, 1)。
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, off));
        }
        if (lane == 0) {
            sh[warp] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            float t = sh[0];
            for (int i = 1; i < nwarp; ++i) {
                t = __fadd_rn(t, sh[i]);
            }
            const float v = __fmul_rn(t, inv_hc);
            out[j] = __fdiv_rn(v, __fadd_rn(1.0f, dno_expf(-v)));
        }
    }

    // inject の行 j (blockIdx.x < hc のブロックだけ)。x は読み取りのみ。
    if (do_inj) {
        __syncthreads();   // down 側の sh[] の読みが終わるのを待ってから木に入る
        const bool act = (tid < 256);
        const unsigned short* rowp =
            (const unsigned short*)(wi + (size_t)j * (size_t)ki * 2u);
        float acc = 0.0f;
        if (act) {
            for (int i = tid; i < ki; i += 256) {
                const float wv = __uint_as_float((unsigned int)rowp[i] << 16);
                acc = __fadd_rn(acc, __fmul_rn(wv, x[i]));
            }
        }
        const float s = dno_mix3_reduce256(acc, act, ired);
        if (tid == 0) inj[j] = s;
    }
}
