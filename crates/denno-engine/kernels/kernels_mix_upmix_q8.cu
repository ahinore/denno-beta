// task 382: decode の hc mix の **up (Q8_0) と hc_mixed を 1 本に畳む**カーネル。
//
// 従来 (task 266/286 の qchain) は 1 呼び出しで
//   down (dno_gemv_q8_0_v5) -> lo_silu -> up (dno_gemv_q8_0_v5) -> hc_mixed
// の 4 本を撃ち、up は 3.48 MB を 266 GB/s で読む (1 回 13.1 us)、lo_silu は
// 320 要素に 3.3 us、hc_mixed は 40 KB の gate を書いて読み直して 5.8 us。
// 実形 (n_in=320 = 10 ブロック = 5 対、n_out=10240、e=2560、hc=4) では
// **up の 1 行 = 320 要素しかない**ので、v5 の `lanes=1` (1 スレッド 1 行) では
// スレッド数が n_out = 10240 で頭打ちになり、80 ブロック x 128 スレッド =
// 4 warp/SM しか出ない (レイテンシを隠せない)。
//
// ここではレーン数を **5** にする (1 レーン = 1 対 = 64 要素、n_in/64 = 5 で
// 割り切れる)。5 は 2 の冪ではないので shuffle の木は使わず、レーンの部分和を
// 共有メモリに置いて固定順 (lane 0..4 の逐次加算) で畳む。ついでに
//   - silu(lo * inv_hc) を共有メモリで 1 回だけ作る (lo_silu の起動を消す)
//   - hc_mixed を同じカーネルの中で作る (gate 40 KB の往復と起動を消す)
//
// 1 ブロック = 128 スレッド = 25 グループ (5 レーン) のうち 24 グループを使い、
// 24 行 = **6 個の e (JC=6) x hc(4) 行**。残り 8 スレッドは遊ぶ。
// grid = ceil(e / JC) = 427。共有メモリは xs[5 * 65] (銀行衝突を避けるため
// 1 レーン 65 要素で区切る) + 部分和 [24][5]。
//
// 数値: 丸め順が v5 (lanes=1・4 アキュムレータ) と変わる (1 レーン 1 対の
// 4 アキュムレータ -> レーン 0..4 の逐次加算)。mixed の式と順序
// (s 昇順・最後に inv_hc) は dno_hc_mixed_f32 と同じ。
// ゲート `DENNO_MIX_UPQ8` (既定 off) で切り戻せる。
//
// 前提 (ホストが門番): Q8_0 の up、n_in % 64 == 0、n_in / 64 == 5 (n_in = 320)、
// hc == 4、row_bytes = n_in / 32 * 34。満たさなければ従来の 4 本経路。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_upmix_q8.cu -o kernels_mix_upmix_q8.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_upmix_q8`。

#include <cstdint>

// ---- Q8_0 の読み方 (kernels_moe_v5.cu と同じ実装。v5 の .cu は触らない) ------

// ---- f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------
__device__ __forceinline__ float dno_v5_f16_bits(unsigned int h)
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

__device__ __forceinline__ float dno_v5_f16_to_f32(const unsigned char* b)
{
    return dno_v5_f16_bits((unsigned int)b[0] | ((unsigned int)b[1] << 8));
}

// 4 バイトワード 1 本から 1 バイト (符号付き 8 bit) を取り出す。
__device__ __forceinline__ int dno_v5_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_v5_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// ---- glibc expf のビット一致実装 (kernels_mix_upmix.cu と同じ) --------------

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

// ---- up + hc_mixed の融合カーネル ------------------------------------------

#define UPQ8_LANES 5      // 1 レーン = 1 対 (64 要素)。n_in / 64 == 5 が前提
#define UPQ8_JC 6         // 1 ブロックが受け持つ e の数
#define UPQ8_HC 4         // hc (ホストの門番と一致)
#define UPQ8_ROWS (UPQ8_JC * UPQ8_HC)      // 24 行
#define UPQ8_XSTRIDE 65                    // 1 レーン分の x の窓 + 1 (銀行を散らす)
#define UPQ8_THREADS 128

// lo:   [n_in]         down の生出力 (silu はこの中で掛ける)
// w:    [hc*e][n_in]   Q8_0 の up の重み (行 = h*e + d)
// xn:   [hc*e]         grouped RMS 正規化の出力
// out:  [e]            mixed
extern "C" __global__ void dno_mix_upmix_q8_0(
    const float* __restrict__ lo,
    const unsigned char* __restrict__ w,
    const float* __restrict__ xn,
    float* __restrict__ out,
    int n_in, int e, float inv_hc, int row_bytes)
{
    __shared__ float xs[UPQ8_LANES * UPQ8_XSTRIDE];
    __shared__ float part[UPQ8_ROWS][UPQ8_LANES];

    // 1) silu(lo * inv_hc) を共有メモリへ (レーンの窓ごとに 65 要素で区切る)。
    for (int i = threadIdx.x; i < n_in; i += UPQ8_THREADS) {
        const float v = __fmul_rn(lo[i], inv_hc);
        const float s = __fdiv_rn(v, __fadd_rn(1.0f, dno_expf(-v)));
        const int win = i >> 6;        // 64 要素 = 1 対 = 1 レーン
        const int off = i & 63;
        if (win < UPQ8_LANES) xs[win * UPQ8_XSTRIDE + off] = s;
    }
    __syncthreads();

    const int j0 = blockIdx.x * UPQ8_JC;
    const int g = threadIdx.x / UPQ8_LANES;         // 0..25
    const int lane = threadIdx.x - g * UPQ8_LANES;  // 0..4
    if (g < UPQ8_ROWS) {
        const int h = g / UPQ8_JC;
        const int jl = g - h * UPQ8_JC;
        const int j = j0 + jl;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        if (j < e) {
            const unsigned char* rowp =
                w + (size_t)(h * e + j) * (size_t)row_bytes;
            // 1 レーン = 1 対 (Q8_0 の 2 ブロック = 68 B = 17 ワード)。
            unsigned int cur[17];
            dno_v5_load17(cur, rowp + (size_t)lane * 68u);
            const float d0 = dno_v5_f16_bits(cur[0] & 0xffffu);
            const float d1 = dno_v5_f16_bits((cur[8] >> 16) & 0xffffu);
            const float* xb = xs + lane * UPQ8_XSTRIDE;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int q = dno_v5_i8(cur[(2 + k) >> 2], (2 + k) & 3);
                acc[k & 3] = __fadd_rn(acc[k & 3], __fmul_rn(__fmul_rn((float)q, d0), xb[k]));
            }
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int q = dno_v5_i8(cur[9 + (k >> 2)], k & 3);
                acc[k & 3] = __fadd_rn(acc[k & 3], __fmul_rn(__fmul_rn((float)q, d1), xb[32 + k]));
            }
            acc0 = acc[0];
            acc1 = acc[1];
            acc2 = acc[2];
            acc3 = acc[3];
        }
        part[g][lane] = __fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3));
    }
    __syncthreads();

    // 2) hc_mixed: mixed[j] = (1/hc) * sum_{s 昇順} xn[s*e+j] * sigmoid(gate[s*e+j])
    if (threadIdx.x < UPQ8_JC) {
        const int j = j0 + threadIdx.x;
        if (j < e) {
            float acc = 0.0f;
            for (int h = 0; h < UPQ8_HC; ++h) {
                const int g2 = h * UPQ8_JC + threadIdx.x;
                float dot = part[g2][0];
                for (int l = 1; l < UPQ8_LANES; ++l) {
                    dot = __fadd_rn(dot, part[g2][l]);
                }
                const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, dno_expf(-dot)));
                const float t = __fmul_rn(xn[(size_t)h * (size_t)e + (size_t)j], sig);
                acc = (h == 0) ? t : __fadd_rn(acc, t);
            }
            out[j] = __fmul_rn(acc, inv_hc);
        }
    }
}
