// denno CUDA batched GEMM kernels (P-PF-GB) — Flash-Next (qwen4exp) の
// prefill 用。量子化ブロックを VRAM に常駐させたまま、nt トークン分を
// 1 回の起動でまとめて計算する。
//
// PTX へのコンパイル:
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_gemm.cu -o kernels_gemm.ptx
//
// 数値契約: **なし** (2026-10-28 のオペレーター指示で prefill は
// ビット一致契約の外に置いた)。ここは FMA 融合 (-fmad=true 既定) を使う。
// 27B の数値同一性契約はこのファイルの外 (kernels_moe.ptx / CPU 参照) で守る。
//
// レイアウトは `kernels_moe.cu` と同じ ggml ブロック仕様 (llama.cpp, MIT)。
// デコード規則は本リポジトリの `src/kquant.rs` と同一。
// ライセンス台帳は `docs/history/80_licenses.md`。
//
// 前提:
//   - w は [n_out][row_bytes] が連結されたもの (VRAM 常駐)。
//   - x は [nt][n_in] (トークン優先・連続)。
//   - out は [nt][n_out]。
//   - nt <= GEMM_NTG_MAX * 16 (= 128)。
//
// ブロック構成:
//   blockDim = (16, 16) = 256 スレッド。tx = 出力行 (16 行/ブロック)、
//   lane = トークン列 (16 本)。各スレッドは「1 行 × nt/16 トークン」を持ち、
//   重みのデコードは共有メモリ上で 1 回だけ行って 16 本で共有する。

#define GEMM_ROWS 16
#define GEMM_NTG_MAX 8

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_moe.cu と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_g_f16(const unsigned char* b)
{
    unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1fu;
    unsigned int man = h & 0x3ffu;
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

// Q4_K / Q5_K の 6bit scale/min 展開 (kquant.rs の get_scale_min_k4 と同一)
__device__ __forceinline__ void dno_g_scale_min_k4(int j, const unsigned char* q,
                                                   unsigned int* d, unsigned int* m)
{
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((unsigned int)(q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((unsigned int)(q[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemm_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    // 1 ブロックぶんのデコード済み f32 だけを共有メモリに置く ([16][32])。
    // 行の生バイトは global から直接読む (丸ごと載せると k=5120 で 62 KiB になり
    // 48 KiB の既定を超える)。
    float* sW = (float*)smem;

    const int tx = threadIdx.x;
    const int lane = threadIdx.y;
    const int tid = threadIdx.y * GEMM_ROWS + tx;
    const int row0 = blockIdx.x * GEMM_ROWS;
    const int nrow = min(GEMM_ROWS, n_out - row0);

    const int ntg = (nt + 15) >> 4;
    const int nb = n_in >> 5;
    float acc[GEMM_NTG_MAX];
#pragma unroll
    for (int i = 0; i < GEMM_NTG_MAX; ++i) acc[i] = 0.0f;
    const bool active = (tx < nrow);

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GEMM_ROWS * 32; i += 256) {
            const int r = i >> 5, j = i & 31;
            const unsigned char* blk = (row0 + r < n_out)
                ? w + (size_t)(row0 + r) * row_bytes + (size_t)b * 34u
                : (const unsigned char*)0;
            float d = 0.0f;
            int qv = 0;
            if (blk) {
                d = dno_g_f16(blk);
                qv = (int)(signed char)blk[2 + j];
            }
            sW[i] = (float)qv * d;
        }
        __syncthreads();
        if (active) {
            const float* wr = sW + tx * 32;
            const int koff = b * 32;
            for (int g = 0; g < ntg; ++g) {
                const int t = lane + g * 16;
                if (t < nt) {
                    const float* xb = x + (size_t)t * n_in + koff;
                    float a = 0.0f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) a = fmaf(wr[j], xb[j], a);
                    acc[g] += a;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        for (int g = 0; g < ntg; ++g) {
            const int t = lane + g * 16;
            if (t < nt) out[(size_t)t * n_out + row0 + tx] = acc[g];
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
//   qh のビット j が要素 j の 5bit 目。下位 16 要素は下ニブル、上位 16 要素は上ニブル。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemm_q5_1_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    float* sW = (float*)smem; // [16][32]、生バイトは global から直接読む

    const int tx = threadIdx.x;
    const int lane = threadIdx.y;
    const int tid = threadIdx.y * GEMM_ROWS + tx;
    const int row0 = blockIdx.x * GEMM_ROWS;
    const int nrow = min(GEMM_ROWS, n_out - row0);

    const int ntg = (nt + 15) >> 4;
    const int nb = n_in >> 5;
    float acc[GEMM_NTG_MAX];
#pragma unroll
    for (int i = 0; i < GEMM_NTG_MAX; ++i) acc[i] = 0.0f;
    const bool active = (tx < nrow);

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GEMM_ROWS * 32; i += 256) {
            const int r = i >> 5, j = i & 31;
            if (row0 + r >= n_out) { sW[i] = 0.0f; continue; }
            const unsigned char* blk = w + (size_t)(row0 + r) * row_bytes + (size_t)b * 24u;
            const float d = dno_g_f16(blk);
            const float m = dno_g_f16(blk + 2);
            const unsigned int qh = (unsigned int)blk[4] | ((unsigned int)blk[5] << 8)
                                  | ((unsigned int)blk[6] << 16) | ((unsigned int)blk[7] << 24);
            const int xh = (int)(((qh >> j) & 1u) << 4);
            const int qv = (j < 16) ? ((int)(blk[8 + j] & 0x0fu) | xh)
                                    : ((int)(blk[8 + j - 16] >> 4) | xh);
            sW[i] = (float)qv * d + m;
        }
        __syncthreads();
        if (active) {
            const float* wr = sW + tx * 32;
            const int koff = b * 32;
            for (int g = 0; g < ntg; ++g) {
                const int t = lane + g * 16;
                if (t < nt) {
                    const float* xb = x + (size_t)t * n_in + koff;
                    float a = 0.0f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) a = fmaf(wr[j], xb[j], a);
                    acc[g] += a;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        for (int g = 0; g < ntg; ++g) {
            const int t = lane + g * 16;
            if (t < nt) out[(size_t)t * n_out + row0 + tx] = acc[g];
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemm_q4k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    float* sW = (float*)smem; // [16][256]、生バイトは global から直接読む

    const int tx = threadIdx.x;
    const int lane = threadIdx.y;
    const int tid = threadIdx.y * GEMM_ROWS + tx;
    const int row0 = blockIdx.x * GEMM_ROWS;
    const int nrow = min(GEMM_ROWS, n_out - row0);

    const int ntg = (nt + 15) >> 4;
    const int nb = n_in >> 8;
    float acc[GEMM_NTG_MAX];
#pragma unroll
    for (int i = 0; i < GEMM_NTG_MAX; ++i) acc[i] = 0.0f;
    const bool active = (tx < nrow);

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GEMM_ROWS * 256; i += 256) {
            const int r = i >> 8, j = i & 255;
            if (row0 + r >= n_out) { sW[i] = 0.0f; continue; }
            const unsigned char* blk = w + (size_t)(row0 + r) * row_bytes + (size_t)b * 144u;
            const float d = dno_g_f16(blk);
            const float dmin = dno_g_f16(blk + 2);
            const int l = j & 63;
            const unsigned char* q = blk + 16 + ((j >> 6) << 5);
            const int qv = (l < 32) ? (int)(q[l] & 0x0fu) : (int)(q[l - 32] >> 4);
            unsigned int sc, mn;
            dno_g_scale_min_k4(j >> 5, blk + 4, &sc, &mn);
            sW[i] = (d * (float)sc) * (float)qv - (dmin * (float)mn);
        }
        __syncthreads();
        if (active) {
            const float* wr = sW + tx * 256;
            const int koff = b * 256;
            for (int g = 0; g < ntg; ++g) {
                const int t = lane + g * 16;
                if (t < nt) {
                    const float* xb = x + (size_t)t * n_in + koff;
                    float a = 0.0f;
#pragma unroll
                    for (int j = 0; j < 256; ++j) a = fmaf(wr[j], xb[j], a);
                    acc[g] += a;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        for (int g = 0; g < ntg; ++g) {
            const int t = lane + g * 16;
            if (t < nt) out[(size_t)t * n_out + row0 + tx] = acc[g];
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemm_q5k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    float* sW = (float*)smem; // [16][256]、生バイトは global から直接読む

    const int tx = threadIdx.x;
    const int lane = threadIdx.y;
    const int tid = threadIdx.y * GEMM_ROWS + tx;
    const int row0 = blockIdx.x * GEMM_ROWS;
    const int nrow = min(GEMM_ROWS, n_out - row0);

    const int ntg = (nt + 15) >> 4;
    const int nb = n_in >> 8;
    float acc[GEMM_NTG_MAX];
#pragma unroll
    for (int i = 0; i < GEMM_NTG_MAX; ++i) acc[i] = 0.0f;
    const bool active = (tx < nrow);

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GEMM_ROWS * 256; i += 256) {
            const int r = i >> 8, j = i & 255;
            if (row0 + r >= n_out) { sW[i] = 0.0f; continue; }
            const unsigned char* blk = w + (size_t)(row0 + r) * row_bytes + (size_t)b * 176u;
            const float d = dno_g_f16(blk);
            const float dmin = dno_g_f16(blk + 2);
            const unsigned char* qh = blk + 16;
            const int l = j & 63;
            const unsigned char* q = blk + 48 + ((j >> 6) << 5);
            const int hi = (int)(((unsigned int)qh[j & 31] >> (j >> 5)) & 1u) << 4;
            const int qv = ((l < 32) ? (int)(q[l] & 0x0fu) : (int)(q[l - 32] >> 4)) + hi;
            unsigned int sc, mn;
            dno_g_scale_min_k4(j >> 5, blk + 4, &sc, &mn);
            sW[i] = (d * (float)sc) * (float)qv - (dmin * (float)mn);
        }
        __syncthreads();
        if (active) {
            const float* wr = sW + tx * 256;
            const int koff = b * 256;
            for (int g = 0; g < ntg; ++g) {
                const int t = lane + g * 16;
                if (t < nt) {
                    const float* xb = x + (size_t)t * n_in + koff;
                    float a = 0.0f;
#pragma unroll
                    for (int j = 0; j < 256; ++j) a = fmaf(wr[j], xb[j], a);
                    acc[g] += a;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        for (int g = 0; g < ntg; ++g) {
            const int t = lane + g * 16;
            if (t < nt) out[(size_t)t * n_out + row0 + tx] = acc[g];
        }
    }
}
