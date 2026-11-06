// task 154: prefill の活性を GPU に置いたまま回すための要素カーネル。
//
// 動機 (lab t151c の 14,018 トークン prefill): h2d 111 GB / d2h 168 GB / sync
// 36,208 回。hc mix 1 回ごとに「res を H2D → down GEMM → lo を D2H → lo を
// H2D → up GEMM → gate を D2H → ホストで norm/silu/gating」を往復していた。
// ここでは mix の要素演算 (grouped RMS norm / lo apply / mixed / inject) と
// hc combine をカーネルにし、down/up/inject の GEMM (kernels_elem の
// dno_gemm_bf16_mt / dno_gemm_f32_mt) と**同じデバイスバッファ**を繋ぐ。
//
// 数値: CPU 参照 (hyper.rs の *_batch_into / hc_combine_batch) と**同じ丸め順**を
// 使う。二乗和は 1 スレッドが i 昇順に逐次加算し (CPU と同じ)、各要素は
// __fmul_rn / __fadd_rn / __fdiv_rn / __fsqrt_rn で CPU の 1 演算 1 丸めに合わせる。
// exp は kernels_elem.cu の dno_expf (glibc e_expf のビット一致再現) を
// dno_pf_expf として写した (中身は 1 文字も変えていない)。
//
// 起動形状は cuda.rs の pf_* が決める (block 256 / grid はカーネルごとのコメント)。

#include <cstdint>

__device__ uint64_t DNO_PF_EXP2_TAB[32] = {
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
__device__ double DNO_PF_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_PF_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_PF_EXP_SHIFT = 0x1.8p+52;  // 1.5 * 2^52

__device__ __forceinline__ uint64_t dno_pf_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_pf_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_pf_asuint(float f) {
    return __float_as_uint(f);
}

/// glibc expf bit-exact replica (round-to-nearest, WANT_ERRNO semantics —
/// overflow -> +inf, underflow -> 0.0f; errno side effects are irrelevant here).
__device__ __forceinline__ float dno_pf_expf(float x) {
    uint32_t abstop = (dno_pf_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {  // top12(88.0f): |x| >= 88 or inf/nan
        // glibc はここで早期 return せず、以下のいずれにも該当しない場合
        // ([88.0, 88.72] と [-103.97, -88]) はそのまま通常経路で計算する
        // (exp(88) は f32 最大値未満、exp(-88) は非正規化数として表現可能)。
        if (dno_pf_asuint(x) == 0xff800000u) return 0.0f;       // -inf
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
    double kd = __fma_rn(DNO_PF_EXP_INVLN2N, xd, DNO_PF_EXP_SHIFT);
    uint64_t ki = dno_pf_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_PF_EXP_SHIFT);
    double r = __fma_rn(DNO_PF_EXP_INVLN2N, xd, -kd);
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    uint64_t t = DNO_PF_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_pf_asdouble(t);
    double z2 = __fma_rn(DNO_PF_EXP_POLY[0], r, DNO_PF_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(DNO_PF_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}


/// grouped RMS norm (batch)。1 ブロック = 1 行 (t, s)。
///
/// CPU (`hyper::grouped_rms_norm_batch_into`) と同じ:
///   ss = Σ_{i<e} v*v  (i 昇順・逐次) / e
///   inv = 1 / sqrt(ss + eps)
///   out = v * inv * gamma[s]
/// 起動: grid = (nt*hc, 1, 1) / block = (128, 1, 1) / static smem 4 B。
extern "C" __global__ void dno_pf_rmsgroup(
    const float *__restrict__ x,      // [rows][e]
    const float *__restrict__ gamma,  // [hc][e]
    float *__restrict__ out,          // [rows][e]
    int rows,
    int e,
    int hc,
    float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float *src = x + (size_t)row * (size_t)e;
    float *dst = out + (size_t)row * (size_t)e;
    const float *g = gamma + (size_t)(row % hc) * (size_t)e;
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < e; ++i) {
            const float v = src[i];
            ss = __fadd_rn(ss, __fmul_rn(v, v));
        }
        ss = __fdiv_rn(ss, (float)e);
        s_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    }
    __syncthreads();
    const float inv = s_inv;
    for (int i = threadIdx.x; i < e; i += blockDim.x) {
        dst[i] = __fmul_rn(__fmul_rn(src[i], inv), g[i]);
    }
}

/// hc mix の低ランク側: lo = silu(lo * (1/hc))。
/// CPU: `hc_lo_apply_batch`。silu(x) = x * sigmoid(x) (hyper::silu と同じ)。
/// 起動: grid = (ceil(n/256), 1, 1) / block = (256, 1, 1)。
extern "C" __global__ void dno_pf_lo(float *__restrict__ v, long long n, float inv_hc)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const float t = __fmul_rn(v[i], inv_hc);
    const float ex = dno_pf_expf(-t);
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, ex));
    v[i] = __fmul_rn(t, sg);
}

/// hc mix の結合: out[t][d] = (Σ_s xn[s][d] * sigmoid(gate[s][d])) * (1/hc)。
/// CPU: `hc_mixed_batch_into` (s の昇順に 1 要素ずつ足す)。
/// 起動: grid = (ceil(e/256), nt, 1) / block = (256, 1, 1)。
extern "C" __global__ void dno_pf_mixed(
    const float *__restrict__ xn,    // [nt][hc][e]
    const float *__restrict__ gate,  // [nt][hc][e]
    float *__restrict__ out,         // [nt][e]
    int e,
    int hc,
    float inv_hc)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= e) return;
    const size_t t = (size_t)blockIdx.y;
    const size_t base = t * (size_t)hc * (size_t)e + (size_t)d;
    float acc = __fmul_rn(xn[base], __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_pf_expf(-gate[base]))));
    for (int s = 1; s < hc; ++s) {
        const size_t i = base + (size_t)s * (size_t)e;
        acc = __fadd_rn(
            acc,
            __fmul_rn(xn[i], __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_pf_expf(-gate[i])))));
    }
    out[t * (size_t)e + (size_t)d] = __fmul_rn(acc, inv_hc);
}

/// hc mix の inject 列: v = v * (1/hc)。CPU: `hc_inject_apply_batch`。
/// 起動: grid = (1, 1, 1) / block = (256, 1, 1) (要素数は高々 nt*hc = 2048)。
extern "C" __global__ void dno_pf_inj(float *__restrict__ v, int n, float inv_hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = __fmul_rn(v[i], inv_hc);
}

/// hc combine: res[t][s][d] += block_out[t][d] * (2 * sigmoid(inject[t][s]))。
/// CPU: `hc_combine_batch`。1 ブロック = 1 行 (t, s)。
/// 起動: grid = (ceil(e/256), nt*hc, 1) / block = (256, 1, 1)。
extern "C" __global__ void dno_pf_combine(
    float *__restrict__ res,         // [nt][hc][e]
    const float *__restrict__ bo,    // [nt][e]
    const float *__restrict__ inj,   // [nt][hc]
    int e,
    int hc)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= e) return;
    const int row = blockIdx.y;  // t * hc + s
    const int t = row / hc;
    const float w = __fmul_rn(2.0f, __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_pf_expf(-inj[row]))));
    const size_t i = (size_t)row * (size_t)e + (size_t)d;
    res[i] = __fadd_rn(res[i], __fmul_rn(bo[(size_t)t * (size_t)e + (size_t)d], w));
}
