// denno CUDA prefill の m 行 expert GEMV (task 148) — 512 トークンのチャンクで
// 1 expert あたり平均 20 行を 1 回の重み読みで回す。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボの担当):
//
//   bash build_ptx.sh exp_iq_pf
//
// 生成物は `kernels_exp_iq_pf.ptx` (アーキは既存と同じ compute_75)。表は
// `kernels_exp_iq_tables.h` (task 144 と同じ。値を 1 bit も変えない)。
//
// # decode のカーネル (`kernels_exp_iq.cu`) との違い
//
// decode は 1 起動 = 1 行 (m=1)。prefill は 1 起動で **16 行 (PF_MR) ずつ**受け、
// 重みを 1 回デコードして 16 行へ配る。1 warp は 2 出力行 (PF_OUTS) を持つ。
// 1 行ぶんの累積順は decode と同一 (b 昇順 → j 昇順 → __shfl_down_sync の木) なので、
// **同じ行を m=1 と m 行で回すとビット一致する**。
//
// # segment 表 (`const unsigned int* seg`、1 segment = 4 語)
//
//   seg[s*4 + 0] = m        (この expert の行数)
//   seg[s*4 + 1] = row_off  (rows の先頭添字。出力も同じ添字 × n_out)
//   seg[s*4 + 2] = 0        (予約)
//   seg[s*4 + 3] = 0        (予約)
//
// `rows[i]` は行 i のトークン番号で、入力は `xbase + rows[i] * n_in`。
// down 行列は入力が `act` (行が詰まっている) なので `rows[i] = i` を渡す。
//
// # 起動形
//
//   256 要素ブロックの型 (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS):
//     blockDim = 128 = 4 warp、1 block = 8 出力行、grid = (n_out/8, nseg, 1)
//   IQ4_NL / Q2_0 (1 レーン = 1 ブロック):
//     blockDim = 128 = 4 warp、1 block = 4 出力行、grid = (n_out/4, nseg, 1)
//   どちらも n_out はタイルで割り切れること (ホストの launch_plan が保証する)。
//   共有メモリは使わない。

#include "kernels_exp_iq_tables.h"

#define PF_MR 16     // 1 回にレジスタへ積む行数
#define PF_OUTS 2    // 1 warp が受ける出力行数 (256 要素ブロックの型)
#define PF_WARPS 4   // 1 block の warp 数
#define PF_TILE (PF_WARPS * PF_OUTS)  // 8 出力行 / block

// f16 -> f32 (kernels_exp_iq.cu と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float pf_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int pf_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

struct PfSeg {
    const unsigned char* w;
    unsigned int m;
    unsigned int ro;
};

// segment の前置き。blockIdx.y は全スレッドで同じなので早期 return してよい
// (barrier は無い)。
__device__ __forceinline__ bool pf_seg(const unsigned char* const* wptrs,
                                       const unsigned int* seg, int nseg, PfSeg* s)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    s->m = sg[0];
    if (s->m == 0u) return false;
    s->ro = sg[1];
    s->w = wptrs[si];
    return true;
}

// __shfl_down_sync と同じ木 (decode の ei_reduce と同一)。
__device__ __forceinline__ float pf_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

// ---------------------------------------------------------------------------
// 型ごとの「1 レーンぶん 8 重み」(256 要素ブロック、kernels_exp_iq.cu と 1:1)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pf_w8_iq2s(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 82u;
    const float d = pf_f16(blk);
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const unsigned int qh = (unsigned int)blk[66 + ib32];
    const unsigned int idx = (unsigned int)blk[2 + lane] | ((qh << (8 - 2 * l4)) & 0x300u);
    const unsigned int sg = (unsigned int)blk[34 + lane];
    const unsigned int sc = (unsigned int)blk[74 + ib32];
    const unsigned int nib = (l4 & 2) ? (sc >> 4) : (sc & 0x0fu);
    const float db = d * (0.5f + (float)nib) * 0.25f;
    const unsigned long long grid = __ldg(&dno_iq2s_grid[idx]);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float gv = (float)((grid >> (8 * j)) & 0xffull);
        const float sgn = (sg & (1u << j)) ? -1.0f : 1.0f;
        w[j] = db * gv * sgn;
    }
}

__device__ __forceinline__ void pf_w8_iq3xxs(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 98u;
    const float d = pf_f16(blk);
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const unsigned int aux = pf_u32(blk + 66 + ib32 * 4);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const unsigned int signs = (unsigned int)__ldg(&dno_ksigns_iq2xs[(aux >> (7 * l4)) & 127u]);
    const unsigned int g1 = __ldg(&dno_iq3xxs_grid[blk[2 + ib32 * 8 + 2 * l4]]);
    const unsigned int g2 = __ldg(&dno_iq3xxs_grid[blk[2 + ib32 * 8 + 2 * l4 + 1]]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gv = (float)((g1 >> (8 * j)) & 0xffu);
        const float sgn = (signs & (1u << j)) ? -1.0f : 1.0f;
        w[j] = db * gv * sgn;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gv = (float)((g2 >> (8 * j)) & 0xffu);
        const float sgn = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
        w[4 + j] = db * gv * sgn;
    }
}

__device__ __forceinline__ void pf_w8_iq3s(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 110u;
    const float d = pf_f16(blk);
    const int g = lane >> 2;
    const int l4 = lane & 3;
    const unsigned int sc = (unsigned int)blk[106 + (g >> 1)];
    const unsigned int nib = (g & 1) ? (sc >> 4) : (sc & 0x0fu);
    const float db = d * (1.0f + 2.0f * (float)nib);
    const unsigned int qh = (unsigned int)blk[66 + g];
    const unsigned int idx1 = (unsigned int)blk[2 + g * 8 + 2 * l4]
                              | ((qh << (8 - 2 * l4)) & 256u);
    const unsigned int idx2 = (unsigned int)blk[2 + g * 8 + 2 * l4 + 1]
                              | ((qh << (7 - 2 * l4)) & 256u);
    const unsigned int signs = (unsigned int)blk[74 + g * 4 + l4];
    const unsigned long long g1 = __ldg(&dno_iq3s_grid[idx1]);
    const unsigned long long g2 = __ldg(&dno_iq3s_grid[idx2]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gv = (float)((g1 >> (8 * j)) & 0xffull);
        const float sgn = (signs & (1u << j)) ? -1.0f : 1.0f;
        w[j] = db * gv * sgn;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gv = (float)((g2 >> (8 * j)) & 0xffull);
        const float sgn = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
        w[4 + j] = db * gv * sgn;
    }
}

__device__ __forceinline__ void pf_w8_iq4xs(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 136u;
    const float d = pf_f16(blk);
    const int g = lane >> 2;
    const int h4 = lane & 3;
    const int half = h4 >> 1;
    const int jbase = (h4 & 1) * 8;
    const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
    const unsigned int sl = (unsigned int)blk[4 + (g >> 1)];
    const unsigned int ls = ((sl >> (4 * (g & 1))) & 0x0fu) | (((sh >> (2 * g)) & 3u) << 4);
    const float dl = d * ((float)ls - 32.0f);
    const unsigned char* qs = blk + 8 + g * 16 + jbase;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned int q = half ? ((unsigned int)qs[j] >> 4) : ((unsigned int)qs[j] & 0x0fu);
        w[j] = dl * (float)dno_kvalues_iq4nl[q];
    }
}

// ---------------------------------------------------------------------------
// 256 要素ブロックの型: 1 warp = PF_OUTS 行、行は PF_MR ずつ。
// 本体は 4 型で同じなので、重みのデコードだけ差し替える。
// ---------------------------------------------------------------------------
#define PF_KERNEL_256(NAME, W8)                                                    \
extern "C" __global__ void NAME(                                                   \
    const unsigned char* const* __restrict__ wptrs,                                \
    const float* __restrict__ xbase,                                               \
    float* __restrict__ obase,                                                     \
    const unsigned int* __restrict__ seg,                                          \
    const unsigned int* __restrict__ rows,                                         \
    int nseg, int n_in, int n_out, int row_bytes)                                  \
{                                                                                  \
    PfSeg s;                                                                       \
    if (!pf_seg(wptrs, seg, nseg, &s)) return;                                     \
    const int lane = (int)(threadIdx.x & 31u);                                     \
    const int warp = (int)(threadIdx.x >> 5u);                                     \
    const int row0 = (int)blockIdx.x * PF_TILE + warp * PF_OUTS;                   \
    if (row0 >= n_out) return;                                                     \
    const int nb = n_in >> 8;                                                      \
    for (unsigned int r0 = 0; r0 < s.m; r0 += PF_MR) {                             \
        const unsigned int left = s.m - r0;                                        \
        const int nr = (int)(left < (unsigned int)PF_MR ? left : (unsigned int)PF_MR); \
        float acc0[PF_MR];                                                         \
        float acc1[PF_MR];                                                         \
        unsigned int tk[PF_MR];                                                    \
        _Pragma("unroll")                                                          \
        for (int r = 0; r < PF_MR; ++r) {                                          \
            acc0[r] = 0.0f;                                                        \
            acc1[r] = 0.0f;                                                        \
            tk[r] = (r < nr) ? rows[s.ro + r0 + (unsigned int)r] : 0u;             \
        }                                                                          \
        for (int b = 0; b < nb; ++b) {                                             \
            float w0[8];                                                           \
            float w1[8];                                                           \
            W8(s.w + (size_t)row0 * (size_t)row_bytes, b, lane, w0);                \
            W8(s.w + (size_t)(row0 + 1) * (size_t)row_bytes, b, lane, w1);          \
            const int xoff = b * 256 + lane * 8;                                   \
            _Pragma("unroll")                                                      \
            for (int r = 0; r < PF_MR; ++r) {                                      \
                if (r < nr) {                                                      \
                    const float4* xq = reinterpret_cast<const float4*>(xbase        \
                        + (size_t)tk[r] * (size_t)n_in + xoff);                     \
                    const float4 xa = xq[0];                                        \
                    const float4 xb4 = xq[1];                                       \
                    float xv[8];                                                    \
                    xv[0] = xa.x; xv[1] = xa.y; xv[2] = xa.z; xv[3] = xa.w;         \
                    xv[4] = xb4.x; xv[5] = xb4.y; xv[6] = xb4.z; xv[7] = xb4.w;     \
                    _Pragma("unroll")                                               \
                    for (int j = 0; j < 8; ++j) {                                   \
                        acc0[r] += w0[j] * xv[j];                                   \
                        acc1[r] += w1[j] * xv[j];                                   \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
        _Pragma("unroll")                                                          \
        for (int r = 0; r < PF_MR; ++r) {                                          \
            if (r < nr) {                                                          \
                const float a0 = pf_reduce(acc0[r]);                               \
                const float a1 = pf_reduce(acc1[r]);                               \
                if (lane == 0) {                                                   \
                    float* op = obase + (size_t)(s.ro + r0 + (unsigned int)r)      \
                                            * (size_t)n_out;                       \
                    op[row0] = a0;                                                 \
                    op[row0 + 1] = a1;                                             \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }                                                                              \
}

PF_KERNEL_256(dno_exp_pf_iq2s, pf_w8_iq2s)
PF_KERNEL_256(dno_exp_pf_iq3xxs, pf_w8_iq3xxs)
PF_KERNEL_256(dno_exp_pf_iq3s, pf_w8_iq3s)
PF_KERNEL_256(dno_exp_pf_iq4xs, pf_w8_iq4xs)

// ---------------------------------------------------------------------------
// IQ4_NL: d(2B) + qs[16] = 18 B / 32 要素。レーン 1 本 = 1 ブロック。
//   1 warp = 1 出力行 (PF_OUTS_L は 1)、1 block = 4 行。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_pf_iq4nl(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    PfSeg s;
    if (!pf_seg(wptrs, seg, nseg, &s)) return;
    const int lane = (int)(threadIdx.x & 31u);
    const int warp = (int)(threadIdx.x >> 5u);
    const int row0 = (int)blockIdx.x * PF_WARPS + warp;
    if (row0 >= n_out) return;
    const unsigned char* rowp = s.w + (size_t)row0 * (size_t)row_bytes;
    const int nb = n_in >> 5;
    for (unsigned int r0 = 0; r0 < s.m; r0 += PF_MR) {
        const unsigned int left = s.m - r0;
        const int nr = (int)(left < (unsigned int)PF_MR ? left : (unsigned int)PF_MR);
        float acc[PF_MR];
        unsigned int tk[PF_MR];
        _Pragma("unroll")
        for (int r = 0; r < PF_MR; ++r) {
            acc[r] = 0.0f;
            tk[r] = (r < nr) ? rows[s.ro + r0 + (unsigned int)r] : 0u;
        }
        for (int b = lane; b < nb; b += 32) {
            const unsigned char* blk = rowp + (size_t)b * 18u;
            const float d = pf_f16(blk);
            float w[32];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int lo = (unsigned int)blk[2 + j] & 0x0fu;
                const unsigned int hi = (unsigned int)blk[2 + j] >> 4;
                w[j] = d * (float)dno_kvalues_iq4nl[lo];
                w[16 + j] = d * (float)dno_kvalues_iq4nl[hi];
            }
            const int xoff = b * 32;
            _Pragma("unroll")
            for (int r = 0; r < PF_MR; ++r) {
                if (r < nr) {
                    const float4* xq = reinterpret_cast<const float4*>(
                        xbase + (size_t)tk[r] * (size_t)n_in + xoff);
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const float4 x4 = xq[v];
                        acc[r] += w[4 * v + 0] * x4.x;
                        acc[r] += w[4 * v + 1] * x4.y;
                        acc[r] += w[4 * v + 2] * x4.z;
                        acc[r] += w[4 * v + 3] * x4.w;
                    }
                }
            }
        }
        _Pragma("unroll")
        for (int r = 0; r < PF_MR; ++r) {
            if (r < nr) {
                const float a = pf_reduce(acc[r]);
                if (lane == 0) {
                    obase[(size_t)(s.ro + r0 + (unsigned int)r) * (size_t)n_out + row0] = a;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q2_0: d(2B) + qs[16] = 18 B / 64 要素 (2 bit、値は (q-1)*d)。レーン 1 本 = 1 ブロック。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_pf_q2_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    PfSeg s;
    if (!pf_seg(wptrs, seg, nseg, &s)) return;
    const int lane = (int)(threadIdx.x & 31u);
    const int warp = (int)(threadIdx.x >> 5u);
    const int row0 = (int)blockIdx.x * PF_WARPS + warp;
    if (row0 >= n_out) return;
    const unsigned char* rowp = s.w + (size_t)row0 * (size_t)row_bytes;
    const int nb = n_in >> 6;
    for (unsigned int r0 = 0; r0 < s.m; r0 += PF_MR) {
        const unsigned int left = s.m - r0;
        const int nr = (int)(left < (unsigned int)PF_MR ? left : (unsigned int)PF_MR);
        float acc[PF_MR];
        unsigned int tk[PF_MR];
        _Pragma("unroll")
        for (int r = 0; r < PF_MR; ++r) {
            acc[r] = 0.0f;
            tk[r] = (r < nr) ? rows[s.ro + r0 + (unsigned int)r] : 0u;
        }
        for (int b = lane; b < nb; b += 32) {
            const unsigned char* blk = rowp + (size_t)b * 18u;
            const float d = pf_f16(blk);
            float w[64];
            #pragma unroll
            for (int j = 0; j < 64; ++j) {
                const unsigned int q = ((unsigned int)blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 3u;
                w[j] = (float)((int)q - 1) * d;
            }
            const int xoff = b * 64;
            _Pragma("unroll")
            for (int r = 0; r < PF_MR; ++r) {
                if (r < nr) {
                    const float4* xq = reinterpret_cast<const float4*>(
                        xbase + (size_t)tk[r] * (size_t)n_in + xoff);
                    #pragma unroll
                    for (int v = 0; v < 16; ++v) {
                        const float4 x4 = xq[v];
                        acc[r] += w[4 * v + 0] * x4.x;
                        acc[r] += w[4 * v + 1] * x4.y;
                        acc[r] += w[4 * v + 2] * x4.z;
                        acc[r] += w[4 * v + 3] * x4.w;
                    }
                }
            }
        }
        _Pragma("unroll")
        for (int r = 0; r < PF_MR; ++r) {
            if (r < nr) {
                const float a = pf_reduce(acc[r]);
                if (lane == 0) {
                    obase[(size_t)(s.ro + r0 + (unsigned int)r) * (size_t)n_out + row0] = a;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ヒット行の加重和: od[t*e + i] = sum_j w[t*k + j] * h[prow[t*k + j] * e + i]
//
// h (全行 × e) をホストへ戻すと 1 層 42 MB になり、pageable D2H (実測 ~2 GB/s)
// が prefill の 1/4 を食う。トークンごとに j 昇順で足して nt x e だけ返す。
// ミスの対は呼び出し側が w = 0 にしてある (0 を足すので値は変わらない)。
// 1 block = 1 トークン、i は blockDim で割る。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_pf_accum(
    const float* __restrict__ h,
    const unsigned int* __restrict__ prow,
    const float* __restrict__ w,
    float* __restrict__ od,
    int k, int e, int ntok)
{
    const int t = (int)blockIdx.x;
    if (t >= ntok) return;
    for (int i = (int)threadIdx.x; i < e; i += (int)blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) {
            const int idx = t * k + j;
            acc += w[idx] * h[(size_t)prow[idx] * (size_t)e + (size_t)i];
        }
        od[(size_t)t * (size_t)e + (size_t)i] = acc;
    }
}

// ---------------------------------------------------------------------------
// act = silu(g) * u (要素ごと)。kernels_exp_iq.cu の dno_exp_silu_mul と同じ式。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_pf_silu_mul(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ act,
    int n)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float v = g[i];
    act[i] = (v / (1.0f + expf(-v))) * u[i];
}
