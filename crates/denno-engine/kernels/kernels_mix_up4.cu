// task 437: mix の silu + up (Q8_0) + hc_mixed を **1 対 = 4 レーン**で回す版。
//
// なぜ要るか (lab t437a の `[dec-kprof-avg]` mix band、1 呼び出しあたり):
//   H2D@a 9.8 us / down (dn7) 12.1 / up (upmix_q8) 11.3 / router 7.1 / D2H@a 5.2
// task 431 で down を 1 対 = 4 レーンにして 14.0 -> 12.1 us (mix 段 -4.6〜-7.1%)。
// 残る up (`dno_mix_upmix_q8_0`、3.48 MB を 322 GB/s) も
//   grid 427 x block 128 = 54,656 スレッド = **10 warp/SM**
// しか出ておらず、down と同じく同時 warp 数で頭打ちになっている。
//
// そこで 1 対 (64 要素 = 68 B) を 4 レーンで分ける (down の dn7 と同じ割り):
//   - x 側は pair*64 + 16*sub で 64 バイト境界
//   - 重み側は 16 バイトずつ。スケールはレーン 0,1 が d0、2,3 が d1
//   - 16 バイトの開始位置は 2 / 18 / 36 / 52 で **2 バイト境界**なので u16 x 8 本
// 1 行 = 5 対 x 4 レーン = 20 スレッド。1 ブロック = (JC * 4) 行 =
// **JC * 80 スレッド**、grid = ceil(e / JC)。
//   JC=6 -> 480 スレッド / 427 ブロック = 218,624 スレッド = 40 warp/SM
//   JC=12 -> 960 スレッド / 214 ブロック (1 SM 1 ブロック、1.26 ウェーブ)
//
// 縮約: 各レーン 16 要素 -> 4 アキュムレータ -> 4 レーンを xor 1,2 ->
// 共有メモリ part[行][5] -> スレッド 0..JC-1 が 5 対を固定順に足して hc_mixed。
//
// 数値: 丸め順が `dno_mix_upmix_q8_0` (1 レーン = 1 対の 4 アキュムレータ +
// レーン 0..4 の逐次加算) と違う。決定論と行独立性は保たれる。
//
// 3 本出す (ホストのゲートで選ぶ):
//   dno_mix_upmix_q8_up4   JC=6、exp は glibc ビット一致 (既定候補)
//   dno_mix_upmix_q8_up4f  JC=6、exp だけ `__expf` (SFU)。silu/sigmoid の
//                          丸めが変わるので比較用 (既定にはしない)
//   dno_mix_upmix_q8_up4w  JC=12 (ブロックを大きくしてブロック数を半分に)
//
// 前提 (ホストが門番): Q8_0、n_in = 320 (= 64 * 5)、hc = 4、
// row_bytes = n_in / 32 * 34。満たさなければ従来の融合版 / 3 本経路へ落ちる。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_up4.cu -o kernels_mix_up4.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_up4`。

#include <cstdint>

// ---- Q8_0 / f16 の読み方 (kernels_mix_dn7.cu と同じ実装) -------------------

__device__ __forceinline__ float dno_up4_f16_bits(unsigned int h)
{
    const unsigned int sign = (h & 0x8000u) << 16;
    const unsigned int exp = (h >> 10) & 0x1fu;
    const unsigned int man = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            unsigned int m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

// このレーンの 16 要素 (u16 x 8 = 16 バイト) と x の 16 要素の内積。
// 演算列は dn7 と同じ「1 積和 1 丸め」で、アキュムレータだけ 4 本。
__device__ __forceinline__ float dno_up4_dot16(
    const unsigned short* __restrict__ q, const float* xv, float d)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const unsigned int w0 = (unsigned int)q[i];
        const unsigned int w1 = (unsigned int)q[i + 1];
        const float x0 = xv[2 * i + 0];
        const float x1 = xv[2 * i + 1];
        const float x2 = xv[2 * i + 2];
        const float x3 = xv[2 * i + 3];
        a0 = __fadd_rn(a0, __fmul_rn(__fmul_rn((float)(signed char)(w0 & 0xffu), d), x0));
        a1 = __fadd_rn(a1, __fmul_rn(__fmul_rn((float)(signed char)((w0 >> 8) & 0xffu), d), x1));
        a2 = __fadd_rn(a2, __fmul_rn(__fmul_rn((float)(signed char)(w1 & 0xffu), d), x2));
        a3 = __fadd_rn(a3, __fmul_rn(__fmul_rn((float)(signed char)((w1 >> 8) & 0xffu), d), x3));
    }
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

// ---- glibc expf のビット一致実装 (kernels_mix_upmix_q8.cu と同じ) -----------

__device__ uint64_t UP4_EXP2_TAB[32] = {
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
__device__ double UP4_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double UP4_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double UP4_EXP_SHIFT = 0x1.8p+52;  // 1.5 * 2^52

__device__ __forceinline__ uint64_t dno_up4_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_up4_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_up4_asuint(float f) {
    return __float_as_uint(f);
}

/// glibc expf のビット一致複製 (kernels_mix_upmix_q8.cu の dno_expf と同じ)。
__device__ __forceinline__ float dno_up4_expf(float x) {
    uint32_t abstop = (dno_up4_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (dno_up4_asuint(x) == 0xff800000u) return 0.0f;       // -inf
        if (abstop >= 0x7f8u) return x + x;                      // inf / nan
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    double xd = (double)x;
    double kd = __fma_rn(UP4_EXP_INVLN2N, xd, UP4_EXP_SHIFT);
    uint64_t ki = dno_up4_asuint64(kd);
    kd = __dadd_rn(kd, -UP4_EXP_SHIFT);
    double r = __fma_rn(UP4_EXP_INVLN2N, xd, -kd);
    uint64_t t = UP4_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_up4_asdouble(t);
    double z2 = __fma_rn(UP4_EXP_POLY[0], r, UP4_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(UP4_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// FAST = false: glibc 一致 / true: __expf (SFU)
template <bool FAST>
__device__ __forceinline__ float dno_up4_exp(float x)
{
    if (FAST) return __expf(x);
    return dno_up4_expf(x);
}

// ---- up + hc_mixed の融合カーネル (1 対 = 4 レーン) -------------------------

#define UP4_PAIRS 5       // 1 行 = 5 対 (n_in = 320)
#define UP4_HC 4          // hc (ホストの門番と一致)
#define UP4_LPR (UP4_PAIRS * 4)     // 1 行 = 20 スレッド
#define UP4_XSTRIDE 68    // 1 対分の x の窓 64 + 4 (レーンごとに銀行を散らす)

// lo:   [n_in]         down の生出力 (silu はこの中で掛ける)
// w:    [hc*e][n_in]   Q8_0 の up の重み (行 = h*e + d)
// xn:   [hc*e]         grouped RMS 正規化の出力
// out:  [e]            mixed
template <bool FAST, int JC>
__device__ __forceinline__ void dno_up4_body(
    const float* __restrict__ lo,
    const unsigned char* __restrict__ w,
    const float* __restrict__ xn,
    float* __restrict__ out,
    int n_in, int e, float inv_hc, int row_bytes)
{
    const int rows = JC * UP4_HC;
    const int nthr = rows * UP4_LPR;
    __shared__ float xs[UP4_PAIRS * UP4_XSTRIDE];
    __shared__ float part[JC * UP4_HC][UP4_PAIRS];

    // 1) silu(lo * inv_hc) を共有メモリへ (レーンの窓ごとに 68 要素で区切る)。
    for (int i = (int)threadIdx.x; i < n_in; i += nthr) {
        const float v = __fmul_rn(lo[i], inv_hc);
        const float s = __fdiv_rn(v, __fadd_rn(1.0f, dno_up4_exp<FAST>(-v)));
        const int win = i >> 6;        // 64 要素 = 1 対 = 1 レーン群
        const int off = i & 63;
        if (win < UP4_PAIRS) xs[win * UP4_XSTRIDE + off] = s;
    }
    __syncthreads();

    const int j0 = (int)blockIdx.x * JC;
    const int tid = (int)threadIdx.x;
    const int r = tid / UP4_LPR;              // 0..rows-1
    const int within = tid - r * UP4_LPR;     // 0..19
    const int pair = within >> 2;
    const int sub = within & 3;
    const int h = r / JC;
    const int j = j0 + (r - h * JC);

    float v = 0.0f;
    if (j < e) {
        // 重み: 行頭は 4 バイト整列。対の 68 バイトのうち 16*sub 要素目から 16 個。
        const unsigned char* p =
            w + (size_t)(h * e + j) * (size_t)row_bytes + (size_t)pair * 68u;
        // スケール: レーン 0,1 は d0 (byte 0)、2,3 は d1 (byte 34)。
        const int soff = (sub < 2) ? 0 : 34;
        const unsigned int sb = (unsigned int)(*(const unsigned short*)(p + soff));
        const float d = dno_up4_f16_bits(sb);
        // 16 要素の qs の開始バイト: 0 -> 2 / 1 -> 18 / 2 -> 36 / 3 -> 52。
        const int qoff = 2 + 16 * sub + ((sub >= 2) ? 2 : 0);
        const unsigned short* q = (const unsigned short*)(p + qoff);
        const float* xb = xs + pair * UP4_XSTRIDE + 16 * sub;
        v = dno_up4_dot16(q, xb, d);
    }
    // 1 対 = 4 レーンの縮約 (xor 1 -> 2 で 4 本が同じ値になる)。
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    if (sub == 0) part[r][pair] = v;
    __syncthreads();

    // 2) hc_mixed: mixed[j] = (1/hc) * sum_{s 昇順} xn[s*e+j] * sigmoid(dot)
    if (tid < JC) {
        const int jj = j0 + tid;
        if (jj < e) {
            float acc = 0.0f;
            for (int h2 = 0; h2 < UP4_HC; ++h2) {
                const int rr = h2 * JC + tid;
                float dot = part[rr][0];
                for (int p2 = 1; p2 < UP4_PAIRS; ++p2) {
                    dot = __fadd_rn(dot, part[rr][p2]);
                }
                const float sig =
                    __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_up4_exp<FAST>(-dot)));
                const float t = __fmul_rn(xn[(size_t)h2 * (size_t)e + (size_t)jj], sig);
                acc = (h2 == 0) ? t : __fadd_rn(acc, t);
            }
            out[jj] = __fmul_rn(acc, inv_hc);
        }
    }
}

// grid = ceil(e / 6)、block = 480。
extern "C" __global__ void dno_mix_upmix_q8_up4(
    const float* __restrict__ lo,
    const unsigned char* __restrict__ w,
    const float* __restrict__ xn,
    float* __restrict__ out,
    int n_in, int e, float inv_hc, int row_bytes)
{
    dno_up4_body<false, 6>(lo, w, xn, out, n_in, e, inv_hc, row_bytes);
}

// 比較用 (exp だけ SFU)。既定にはしない。
extern "C" __global__ void dno_mix_upmix_q8_up4f(
    const float* __restrict__ lo,
    const unsigned char* __restrict__ w,
    const float* __restrict__ xn,
    float* __restrict__ out,
    int n_in, int e, float inv_hc, int row_bytes)
{
    dno_up4_body<true, 6>(lo, w, xn, out, n_in, e, inv_hc, row_bytes);
}

// JC=12 (block = 960、grid は半分)。
extern "C" __global__ void dno_mix_upmix_q8_up4w(
    const float* __restrict__ lo,
    const unsigned char* __restrict__ w,
    const float* __restrict__ xn,
    float* __restrict__ out,
    int n_in, int e, float inv_hc, int row_bytes)
{
    dno_up4_body<false, 12>(lo, w, xn, out, n_in, e, inv_hc, row_bytes);
}
