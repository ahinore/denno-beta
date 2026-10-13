// denno-beta P17d: elementwise kernels (bit-exact vs CPU int8 path).
//
// Design constraints (exactness contract):
// - All floating point ops replicate the CPU (Rust/glibc) op order exactly.
//   - compiled with -fmad=false and explicit __*_rn intrinsics so only the
//     intended sites are fused.
//   - expf_glibc is a faithful re-implementation of glibc's e_expf.c as
//     compiled for x86-64 with FMA (gcc -ffp-contract=fast contracts
//     kd/r/poly mul+add pairs into fma — the runtime IFUNC selects this
//     variant on FMA-capable CPUs, and Rust f32::exp calls it). Verified
//     bit-identical to CPU f32::exp on the target host (see dno_expf).
//   - rmsnorm sum is computed strictly sequentially (i=0..cols-1) by one thread,
//     matching `iter().map(|v| v*v).sum::<f32>()`.

#include <cstdint>

#define N_TABLE 32  // EXP2F_TABLE_BITS = 5

// glibc sysdeps/ieee754/flt-32/e_exp2f_data.c (__exp2f_data)
// (constant memory ではなく通常の device メモリ: warp 内でインデックスが
//  分岐すると constant cache が直列化されるため。256B なので L1 で十分)
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

/// glibc expf bit-exact replica (round-to-nearest, WANT_ERRNO semantics —
/// overflow -> +inf, underflow -> 0.0f; errno side effects are irrelevant here).
__device__ __forceinline__ float dno_expf(float x) {
    uint32_t abstop = (dno_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {  // top12(88.0f): |x| >= 88 or inf/nan
        // glibc はここで早期 return せず、以下のいずれにも該当しない場合
        // ([88.0, 88.72] と [-103.97, -88]) はそのまま通常経路で計算する
        // (exp(88) は f32 最大値未満、exp(-88) は非正規化数として表現可能)。
        if (dno_asuint(x) == 0xff800000u) return 0.0f;       // -inf
        if (abstop >= 0x7f8u) return x + x;                  // inf / nan
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);  // > log(0x1p128) -> +inf
        if (x < -0x1.9fe368p6f) return 0.0f;                 // < log(0x1p-150) -> uflow 0
        // fall through to normal path
    }
    double xd = (double)x;
    // glibc の FMA ビルド (gcc -ffp-contract=fast) は以下を fma に縮約する:
    //   kd = fma(InvLn2N, xd, SHIFT)  (= z + SHIFT の縮約)
    //   r  = fma(InvLn2N, xd, -kd)    (= z - kd の縮約)
    // 多項式 3 ヶ所も同様。実機 (glibc 2.35 + RTX 5090 ホスト) で glibc expf /
    // Rust f32::exp の返すビットと完全一致することを確認済み
    // (x=32.564632f: 非縮約版は 1ULP 差 = 0x56fc9f1b vs 正 0x56fc9f1c)。
    double kd = __fma_rn(DNO_EXP_INVLN2N, xd, DNO_EXP_SHIFT);
    uint64_t ki = dno_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_EXP_SHIFT);
    double r = __fma_rn(DNO_EXP_INVLN2N, xd, -kd);
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
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

/// SiLU + up multiply (swiglu): out[i] = (g / (1 + exp(-g))) * u
/// CPU: silu_inplace (v = v / (1.0 + exp(-v))) then gate[i] *= up[i].
extern "C" __global__ void dno_swiglu(
    const float *gate, const float *up, float *out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float g = gate[i];
        float e = dno_expf(-g);
        float r = __fdiv_rn(g, __fadd_rn(1.0f, e));
        out[i] = __fmul_rn(r, up[i]);
    }
}

/// residual add + RMSNorm (proj == NULL で単純 rmsnorm)。
/// grid = rows, block = 256, dynamic shmem = cols * 4 bytes.
/// 平方和は CPU と同一の逐次順 (i = 0..cols-1) で thread 0 が計算する。
extern "C" __global__ void dno_add_rmsnorm(
    float *x, const float *proj, const float *w, float *out,
    int cols, float eps) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    float *xr = x + (size_t)row * cols;
    // 1) residual add (要素独立なので並列でよい)
    if (proj != nullptr) {
        const float *pr = proj + (size_t)row * cols;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float v = __fadd_rn(xr[i], pr[i]);
            xr[i] = v;
            sh[i] = v;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            sh[i] = xr[i];
        }
    }
    __syncthreads();
    // 2) ss = sum(x_i^2) / cols — CPU と同じ逐次順 (thread 0 のみ)
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < cols; ++i) {
            ss = __fadd_rn(ss, __fmul_rn(sh[i], sh[i]));
        }
        ss = __fdiv_rn(ss, (float)cols);
        float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
        s_inv = inv;
    }
    __syncthreads();
    // 3) out[i] = x[i] * inv * w[i] (左結合の乗算列 — CPU と同一)
    float inv = s_inv;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[(size_t)row * cols + i] = __fmul_rn(__fmul_rn(xr[i], inv), w[i]);
    }
}

/// テスト用: dno_expf を要素ごとに適用する
extern "C" __global__ void dno_expf_probe(
    const float *in, float *out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = dno_expf(in[i]);
}
