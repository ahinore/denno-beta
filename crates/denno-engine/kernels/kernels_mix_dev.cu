// task 184: decode の hc mix でホストが回していた 2 つの要素演算をデバイスへ移す。
//
// 背景 (lab t179a の平均表 `DENNO_GDN_DEC_KPROF`): mix 1 回のホスト仕事は
// mix_norm 5.1 us + mix_inj 3.8 us = 8.9 us で、decode は 1 トークンに
// 2*n_layer+1 = 97 回 mix を呼ぶ (0.86 ms/token)。どちらも
//   - 二乗和 / 内積を **1 スレッドの逐次和** で回していた (e=2,560 要素の依存鎖)
//   - 結果をホストの GEMV 経路へ渡すためにホストで計算する必要があった
// ので、デバイスへ移して mix の submission に畳む。
//
// 数値: 縮約を **ツリー** にする (CPU の逐次和とは丸め順が変わる = md5 は
// DIFF しうる)。1 要素あたりの演算列 (`x*inv*gamma` / `w*x` の乗算 1 回 +
// 加算 1 回) は CPU と同じで、-fmad=false + __*_rn により 1 演算 1 丸めにする。
// 決定論: スレッド割り当てと縮約の順序は固定 (実行ごとに変わらない)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dev.cu -o kernels_mix_dev.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dev`。

// 32 スレッドの warp 内縮約 → 8 warp の縮約 (block 256 固定) → 全スレッドへ配る。
// red は 16 float 以上の作業域 (呼び出し側の static shared)。
// 加算順は固定 (offset 16,8,4,2,1 と warp 番号の昇順)。
__device__ __forceinline__ float dno_mix_reduce256(float v, float* red)
{
    const int t = threadIdx.x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if ((t & 31) == 0) red[t >> 5] = v;
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
// grouped RMS 正規化 (1 ブロック = 1 ストリーム)
//
// out[s*e + i] = x[s*e + i] * inv * gamma[s*e + i]
//   ss  = Σ_i x^2 / e     (ツリー縮約)
//   inv = 1 / sqrt(ss + eps)
//
// out は x と同一ポインタでよい (縮約が終わるまで書き込まない)。
// grid = hc、block = 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_norm(
    const float* __restrict__ x,      // [hc][e]
    const float* __restrict__ gamma,  // [hc][e]
    float* __restrict__ out,          // [hc][e] (x と同一可)
    int e, float eps)
{
    __shared__ float red[16];
    const int s = blockIdx.x;
    const int t = threadIdx.x;
    const float* src = x + (size_t)s * (size_t)e;
    float ss = 0.0f;
    for (int i = t; i < e; i += 256) {
        const float v = src[i];
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    ss = dno_mix_reduce256(ss, red);
    const float mean = __fdiv_rn(ss, (float)e);
    const float den = __fsqrt_rn(__fadd_rn(mean, eps));
    const float inv = __fdiv_rn(1.0f, den);
    float* dst = out + (size_t)s * (size_t)e;
    const float* g = gamma + (size_t)s * (size_t)e;
    for (int i = t; i < e; i += 256) {
        dst[i] = __fmul_rn(__fmul_rn(src[i], inv), g[i]);
    }
}

// ---------------------------------------------------------------------------
// inject の内積 (1 ブロック = 1 行)
//
// w は BF16 の生バイト [n][k] (gguf の行優先)。BF16 → f32 は 16 bit 左シフトで
// 厳密 (丸めなし)。out[r] = Σ_i w[r][i] * x[i] (ツリー縮約)。
// 1/hc は呼び出し側 (ホストの hc_inject_apply) が掛ける。
// grid = n、block = 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_inj(
    const unsigned char* __restrict__ w,  // [n][k] BF16
    const float* __restrict__ x,          // [k]
    float* __restrict__ out,              // [n]
    int k, int n)
{
    __shared__ float red[16];
    const int r = blockIdx.x;
    if (r >= n) return;
    const int t = threadIdx.x;
    const unsigned short* rowp =
        (const unsigned short*)(w + (size_t)r * (size_t)k * 2u);
    float s = 0.0f;
    for (int i = t; i < k; i += 256) {
        const float wv = __uint_as_float((unsigned int)rowp[i] << 16);
        s = __fadd_rn(s, __fmul_rn(wv, x[i]));
    }
    s = dno_mix_reduce256(s, red);
    if (t == 0) out[r] = s;
}

// ---------------------------------------------------------------------------
// task 266: lo_apply (silu(v * inv_hc)) をデバイスで回す。
//
// 丸めはホストの `hyper::hc_lo_apply` と同一にする:
//   x = v * inv_hc
//   silu(x) = x * (1 / (1 + expf(-x)))     (hyper::silu = x * sigmoid(x))
// expf は glibc の逐語再現 (`kernels_elem.cu` の dno_expf をここへ写した。
// PTX はファイル単位で独立したモジュールになるため共有できない)。
// したがって CPU とビット一致する (下の `_rn` 3 本で 1 演算 1 丸め)。
// ---------------------------------------------------------------------------
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

extern "C" __global__ void dno_mix_lo_silu(
    float* __restrict__ v,   // [n] (in-place)
    int n,
    float inv_hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __fmul_rn(v[i], inv_hc);
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_expf(-x)));
    v[i] = __fmul_rn(x, sg);
}
