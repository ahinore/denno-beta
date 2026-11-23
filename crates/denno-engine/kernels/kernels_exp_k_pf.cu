// denno CUDA prefill の m 行 expert GEMV — **K 系 (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0)**
// (task 309)。
//
// `kernels_exp_iq_pf.cu` (IQ 系) と同じ形。1 起動で 1 expert の m 行を受け、
// 重みを 1 回だけデコードして m 行へ配る (batched GEMV)。1 行ぶんの累積順は
// decode の `kernels_seg_k.cu` とは違う (こちらは 32 レーン分割の木) ので
// ビット一致契約は無い。決定的ではある (木の形が固定)。prefill の丸め契約外。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボの担当):
//
//   bash build_ptx.sh exp_k_pf
//
// 生成物は `kernels_exp_k_pf.ptx` (アーキは既存と同じ compute_75)。
//
// # 型ごとのブロック
//
//   Q4_K  … 256 要素 / 144 B (d, dmin, scales[12], qs[128])
//   Q5_K  … 256 要素 / 176 B (d, dmin, scales[12], qh[32], qs[128])
//   Q6_K  … 256 要素 / 210 B (ql[128], qh[64], scales i8[16], d)
//   Q5_1  …  32 要素 /  24 B (d, m, qh[4], qs[16])
//   Q8_0  …  32 要素 /  34 B (d, qs[32] i8)
//
// 256 要素ブロックの型は **1 レーン = そのブロックの 8 要素** (lane*8 .. +7)、
// 1 warp = 2 出力行 (`PF_OUTS`)、1 block = 8 出力行。
// 32 要素ブロックの型は **1 レーン = 1 ブロック**、1 warp = 1 出力行、1 block = 4 行
// (IQ4_NL / Q2_0 と同じ形。down の k=640 は 256 で割れないのでこの形が要る)。
//
// # segment 表と rows は `kernels_exp_iq_pf.cu` と同一 (1 segment = 4 語)。
//
// # 起動形 (ホストの `exp_pf::launch_plan` / `tile_of` と一致させること)
//
//   Q4_K / Q5_K / Q6_K: blockDim = 128、1 block = 8 行、grid = (n_out/8, nseg, 1)
//   Q5_1 / Q8_0:        blockDim = 128、1 block = 4 行、grid = (n_out/4, nseg, 1)
//
//   共有メモリは使わない。n_out はタイルで割り切れること (launch_plan が保証する)。
#include <cuda_fp16.h>

#define PF_MR 16      // 1 回にレジスタへ積む行数
#define PF_OUTS 2     // 1 warp が受ける出力行数 (256 要素ブロックの型)
#define PF_WARPS 4    // 1 block の warp 数
#define PF_TILE (PF_WARPS * PF_OUTS)  // 8 出力行 / block

// f16 -> f32 (kernels_exp_iq_pf.cu と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float kpf_f16(const unsigned char* b)
{
    // task 640: 手書きのビット展開をハードウェアの cvt.f32.f16 (`__half2float`) に
    // 置き換えた。f16 -> f32 の拡張は IEEE 754 で一意なので結果は同じ (0 / 非正規化数
    // を含む全 65536 パターンで旧実装と一致することを cuda.rs の
    // f16_cvt_exhaustive で確認済み)。NaN と inf だけは旧実装の式をそのまま残す
    // (ハードウェアは NaN を quiet 化するので、ペイロードの再現性のため)。
    const unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    if ((h & 0x7c00u) == 0x7c00u) {
        const unsigned int sign = (h & 0x8000u) << 16;
        return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
    }
    return __half2float(__ushort_as_half((unsigned short)h));
}

__device__ __forceinline__ unsigned int kpf_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

struct KpfSeg {
    const unsigned char* w;
    unsigned int m;
    unsigned int ro;
};

// segment の前置き。blockIdx.y は全スレッドで同じなので早期 return してよい
// (barrier は無い)。
__device__ __forceinline__ bool kpf_seg(const unsigned char* const* wptrs,
                                        const unsigned int* seg, int nseg, KpfSeg* s)
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

// __shfl_down_sync と同じ木 (IQ 版の pf_reduce と同一)。
__device__ __forceinline__ float kpf_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

// Q4_K / Q5_K の 6 bit スケールと最小値 (ggml の get_scale_min_k4 と同じ式)。
__device__ __forceinline__ void kpf_scale_min_k4(int j, const unsigned char* q,
                                                 unsigned int* sc, unsigned int* mn)
{
    if (j < 4) {
        *sc = (unsigned int)q[j] & 63u;
        *mn = (unsigned int)q[j + 4] & 63u;
    } else {
        *sc = ((unsigned int)q[j + 4] & 0x0fu) | (((unsigned int)q[j - 4] >> 6) << 4);
        *mn = ((unsigned int)q[j + 4] >> 4) | (((unsigned int)q[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// 256 要素ブロックの型: 1 レーン = 8 重み (lane*8 .. +7)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void kpf_w8_q4k(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 144u;
    const float d = kpf_f16(blk);
    const float dmin = kpf_f16(blk + 2);
    const int sb = lane >> 2;         // 32 要素サブブロック
    const int p = (lane & 3) << 3;    // サブブロック内の先頭要素
    unsigned int sc, mn;
    kpf_scale_min_k4(sb, blk + 4, &sc, &mn);
    const float dv = d * (float)sc;
    const float mv = dmin * (float)mn;
    const unsigned char* qs = blk + 16u + (size_t)(sb >> 1) * 32u + p;
    const int sh = (sb & 1) ? 4 : 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int q = ((unsigned int)qs[i] >> sh) & 0x0fu;
        w[i] = dv * (float)q - mv;
    }
}

__device__ __forceinline__ void kpf_w8_q5k(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 176u;
    const float d = kpf_f16(blk);
    const float dmin = kpf_f16(blk + 2);
    const int sb = lane >> 2;
    const int p = (lane & 3) << 3;
    unsigned int sc, mn;
    kpf_scale_min_k4(sb, blk + 4, &sc, &mn);
    const float dv = d * (float)sc;
    const float mv = dmin * (float)mn;
    const unsigned char* qh = blk + 16u;                          // 32 B (8 サブブロック × 1 bit)
    const unsigned char* qs = blk + 48u + (size_t)(sb >> 1) * 32u + p;
    const int sh = (sb & 1) ? 4 : 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int q = ((unsigned int)qs[i] >> sh) & 0x0fu;
        const unsigned int hb = ((unsigned int)qh[p + i] >> sb) & 1u;
        w[i] = dv * (float)(q + (hb << 4)) - mv;
    }
}

__device__ __forceinline__ void kpf_w8_q6k(const unsigned char* rowp, int b, int lane, float* w)
{
    const unsigned char* blk = rowp + (size_t)b * 210u;
    const float d = kpf_f16(blk + 208);
    const int half = lane >> 4;             // 128 要素の半分
    const int o = (lane & 15) << 3;         // 半分の中の先頭要素
    const int ch = o >> 5;                  // 面 (32 要素)
    const int l2 = o & 31;                  // 面の中の先頭要素
    const unsigned char* ql = blk + (size_t)half * 64u + ((ch & 1) ? 32u : 0u) + l2;
    const unsigned char* qh = blk + 128u + (size_t)half * 32u + l2;
    const signed char* sc = (const signed char*)(blk + 192u) + (size_t)half * 8u;
    const int s = (int)sc[2 * ch + (l2 >= 16 ? 1 : 0)];
    const int sh = (ch >= 2) ? 4 : 0;
    const float dv = d * (float)s;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int nib = ((unsigned int)ql[i] >> sh) & 0x0fu;
        const unsigned int h2 = ((unsigned int)qh[i] >> (2 * ch)) & 3u;
        const int q6 = (int)(nib | (h2 << 4)) - 32;
        w[i] = dv * (float)q6;
    }
}

// 256 要素ブロックの型: 1 warp = PF_OUTS 行、行は PF_MR ずつ。
// 本体は 3 型で同じなので、重みのデコードだけ差し替える (IQ 版と同じマクロ)。
#define KPF_KERNEL_256(NAME, W8)                                                   \
extern "C" __global__ void NAME(                                                   \
    const unsigned char* const* __restrict__ wptrs,                                \
    const float* __restrict__ xbase,                                               \
    float* __restrict__ obase,                                                     \
    const unsigned int* __restrict__ seg,                                          \
    const unsigned int* __restrict__ rows,                                         \
    int nseg, int n_in, int n_out, int row_bytes)                                  \
{                                                                                  \
    KpfSeg s;                                                                      \
    if (!kpf_seg(wptrs, seg, nseg, &s)) return;                                    \
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
                const float a0 = kpf_reduce(acc0[r]);                              \
                const float a1 = kpf_reduce(acc1[r]);                              \
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

KPF_KERNEL_256(dno_exp_k_pf_q4k, kpf_w8_q4k)
KPF_KERNEL_256(dno_exp_k_pf_q5k, kpf_w8_q5k)
KPF_KERNEL_256(dno_exp_k_pf_q6k, kpf_w8_q6k)

// ---------------------------------------------------------------------------
// Q5_1: 32 要素 / 24 B。レーン 1 本 = 1 ブロック。1 warp = 1 出力行、1 block = 4 行。
//   y = d * (q + 16 * hb) + m
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_k_pf_q5_1(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    KpfSeg s;
    if (!kpf_seg(wptrs, seg, nseg, &s)) return;
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
            const unsigned char* blk = rowp + (size_t)b * 24u;
            const float d = kpf_f16(blk);
            const float m = kpf_f16(blk + 2);
            const unsigned int qh = kpf_u32(blk + 4);
            float w[32];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int lo = (unsigned int)blk[8 + j] & 0x0fu;
                const unsigned int hi = (unsigned int)blk[8 + j] >> 4;
                w[j] = d * (float)(lo + (((qh >> j) & 1u) << 4)) + m;
                w[16 + j] = d * (float)(hi + (((qh >> (j + 16)) & 1u) << 4)) + m;
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
                const float a = kpf_reduce(acc[r]);
                if (lane == 0) {
                    obase[(size_t)(s.ro + r0 + (unsigned int)r) * (size_t)n_out + row0] = a;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0: 32 要素 / 34 B (d + qs[32] i8)。レーン 1 本 = 1 ブロック。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_k_pf_q8_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    KpfSeg s;
    if (!kpf_seg(wptrs, seg, nseg, &s)) return;
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
            const unsigned char* blk = rowp + (size_t)b * 34u;
            const float d = kpf_f16(blk);
            float w[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                w[j] = d * (float)(int)(signed char)blk[2 + j];
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
                const float a = kpf_reduce(acc[r]);
                if (lane == 0) {
                    obase[(size_t)(s.ro + r0 + (unsigned int)r) * (size_t)n_out + row0] = a;
                }
            }
        }
    }
}
