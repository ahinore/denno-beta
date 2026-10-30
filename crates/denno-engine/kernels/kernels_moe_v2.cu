// denno CUDA MoE GEMV kernels (v2) — decode の量子 GEMV を速くする版。
//
// PTX へのコンパイル:
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_v2.cu -o kernels_moe_v2.ptx
//
// **数値契約は v1 と同一**: 1 行の累積は 1 スレッドが i の昇順に f32 で行い、
// 各要素は `y = (dequant) ; acc += y * x[i]` の 2 演算。`-fmad=false`。
// 変えたのは**データの取り回しだけ**:
//   - x を 32 要素ぶんレジスタへ先読みしてから加算する (v1 は 1 要素ごとに
//     グローバルから読んでいたため、依存チェーンに L1 レイテンシが乗っていた)。
//   - 内側ループを完全展開し、共有メモリのロードを加算より前にまとめて発行させる。
// したがって v1 と**ビット一致**する (同じ順序・同じ丸め)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

__device__ __forceinline__ float dno_f16_to_f32_v2(const unsigned char* b)
{
    unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp  = (h >> 10) & 0x1fu;
    unsigned int man  = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((man & 0x400u) == 0) { man <<= 1; --e; }
            man &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

__device__ __forceinline__ void dno_scale_min_k4_v2(int j, const unsigned char* q,
                                                    unsigned int* d, unsigned int* m)
{
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((unsigned int)(q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)     | ((unsigned int)(q[j]     >> 6) << 4);
    }
}

#define MOE_ROWS 4
#define MOE_LANES 64

__device__ __forceinline__ int moe_row_of_v2(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (MOE_LANES - 1);
    return blockIdx.x * MOE_ROWS + (t / MOE_LANES);
}

__device__ __forceinline__ void moe_stage_v2(const unsigned char* __restrict__ src,
                                             unsigned char* dst,
                                             int row_bytes, int lane)
{
    for (int i = lane; i < row_bytes; i += MOE_LANES) dst[i] = src[i];
}

// x[n]..x[n+31] をレジスタへ先読み (完全展開でレジスタに載る)
__device__ __forceinline__ void moe_prefetch32(const float* __restrict__ xb, float* xv)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(xb + 4 * i);
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q8_0_v2(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    int lane;
    const int row = moe_row_of_v2(&lane);
    const bool valid = (row < n_out);

    unsigned char* dst = smem + (size_t)(threadIdx.x / MOE_LANES) * (size_t)row_bytes;
    if (valid) moe_stage_v2(w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 34u;
        const float d = dno_f16_to_f32_v2(blk);
        const float* xb = x + (size_t)b * 32u;
        float xv[32];
        moe_prefetch32(xb, xv);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float y = (float)(int)(signed char)blk[2 + j] * d;
            acc += y * xv[j];
        }
    }
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5_1_v2(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    int lane;
    const int row = moe_row_of_v2(&lane);
    const bool valid = (row < n_out);

    unsigned char* dst = smem + (size_t)(threadIdx.x / MOE_LANES) * (size_t)row_bytes;
    if (valid) moe_stage_v2(w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 24u;
        const float d = dno_f16_to_f32_v2(blk);
        const float m = dno_f16_to_f32_v2(blk + 2);
        const unsigned int qh = (unsigned int)blk[4]
                              | ((unsigned int)blk[5] << 8)
                              | ((unsigned int)blk[6] << 16)
                              | ((unsigned int)blk[7] << 24);
        const float* xb = x + (size_t)b * 32u;
        float xv[32];
        moe_prefetch32(xb, xv);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int xh0 = (int)(((qh >> j) << 4) & 0x10u);
            const int q0 = (int)(blk[8 + j] & 0x0fu) | xh0;
            const float y = (float)q0 * d + m;
            acc += y * xv[j];
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int xh1 = (int)((qh >> (j + 12)) & 0x10u);
            const int q1 = (int)(blk[8 + j] >> 4) | xh1;
            const float y = (float)q1 * d + m;
            acc += y * xv[16 + j];
        }
    }
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q4k_v2(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    int lane;
    const int row = moe_row_of_v2(&lane);
    const bool valid = (row < n_out);

    unsigned char* dst = smem + (size_t)(threadIdx.x / MOE_LANES) * (size_t)row_bytes;
    if (valid) moe_stage_v2(w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 144u;
        const float d    = dno_f16_to_f32_v2(blk);
        const float dmin = dno_f16_to_f32_v2(blk + 2);
        const float* xb  = x + (size_t)b * 256u;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const unsigned char* q = blk + 16 + (j >> 1);
            unsigned int sc, mn;
            dno_scale_min_k4_v2(is, blk + 4, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)mn;
            dno_scale_min_k4_v2(is + 1, blk + 4, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)mn;
            float xv[64];
            moe_prefetch32(xb + j, xv);
            moe_prefetch32(xb + j + 32, xv + 32);
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const float y = d1 * (float)(int)(q[l] & 0x0fu) - m1;
                acc += y * xv[l];
            }
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const float y = d2 * (float)(int)(q[l] >> 4) - m2;
                acc += y * xv[32 + l];
            }
            is += 2;
        }
    }
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5k_v2(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    int lane;
    const int row = moe_row_of_v2(&lane);
    const bool valid = (row < n_out);

    unsigned char* dst = smem + (size_t)(threadIdx.x / MOE_LANES) * (size_t)row_bytes;
    if (valid) moe_stage_v2(w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 176u;
        const float d    = dno_f16_to_f32_v2(blk);
        const float dmin = dno_f16_to_f32_v2(blk + 2);
        const unsigned char* qh = blk + 16;
        const float* xb = x + (size_t)b * 256u;
        int is = 0;
        unsigned int u1 = 1u, u2 = 2u;
        for (int j = 0; j < 256; j += 64) {
            const unsigned char* ql = blk + 48 + (j >> 1);
            unsigned int sc, mn;
            dno_scale_min_k4_v2(is, blk + 4, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)mn;
            dno_scale_min_k4_v2(is + 1, blk + 4, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)mn;
            float xv[64];
            moe_prefetch32(xb + j, xv);
            moe_prefetch32(xb + j + 32, xv + 32);
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const int hi = (qh[l] & u1) ? 16 : 0;
                const float y = d1 * (float)((int)(ql[l] & 0x0fu) + hi) - m1;
                acc += y * xv[l];
            }
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const int hi = (qh[l] & u2) ? 16 : 0;
                const float y = d2 * (float)((int)(ql[l] >> 4) + hi) - m2;
                acc += y * xv[32 + l];
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out[row] = acc;
}
