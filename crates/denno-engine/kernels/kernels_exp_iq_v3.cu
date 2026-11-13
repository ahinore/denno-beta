// denno CUDA 層一括 expert GEMV (v3, task 275) — IQ2_S / IQ3_S の表引きを細くする。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq_v3
//
// 生成物は `kernels_exp_iq_v3.ptx`。表は `kernels_exp_iq_tables.h`。
//
// # なぜ v2 を触るか (REPORT_P190 の型ごとの帯域)
//
// v2 (`kernels_exp_iq_v2.cu`) で IQ3_XXS / IQ4_XS は 452 / 560 GB/s まで出たが、
// IQ2_S / IQ3_S は 231 / 272 GB/s しか出ていない。差は **グリッド表の 1 本の幅**:
//
//   IQ3_XXS 256 x u32 = 1 KB     IQ4_XS 表なし
//   IQ3_S   512 x u64 = 4 KB     IQ2_S 1024 x u64 = 8 KB
//
// 32 レーンがばらばらの 64 bit を shared から読むとバンク再送が乗り、ブロック先頭の
// 8 KB の写しも 1 ブロック 4 行 (3.3 KB) に対して重すぎる。表そのものを小さくできる:
//
//   IQ3_S: 512 本の u64 の **上位 4 バイトは全 0** で、どの経路も読まない
//          (kquant.rs の IQ3S 分岐も v1/v2 も `g[j]` (j < 4) だけ)。
//          → `dno_iq3s_grid32` (512 x u32 = 2 KB) で同じ値。
//   IQ2_S: 8 バイトすべてを使うが、現れる値は {8, 25, 43} の 3 種類だけ。
//          → `dno_iq2s_grid2` (1024 x u16、1 値 2 bit = 2 KB)。
//          復元は code -> 8/25/43。`db * 8` は 2 の冪なので、3 本の積
//          `p0/p1/p2 = db * {8,25,43}` を作って選ぶと **v2 と同じ値**になる。
//
// # 数値の契約
//
// **v2 とビット一致**(表の値・積の順・アキュムレータ 4 本・warp の木はすべて同じ)。
// 変わるのは shared に置く表の幅と、積を 3 本で持つことだけ。ゲート
// `DENNO_EXP_IQ_V3` で v2 へ戻せる (既定 off)。
//
// # segment 表と起動形は v1/v2 と同一
//
//   seg[s*4 + 0] = expert_id (未使用) / +1 = m (decode では 1)
//   seg[s*4 + 2] = x_off / +3 = out_off
//
//   blockDim = 128 = 4 warp (E3_ROWS = 4 行、1 行 = 1 warp)
//   grid.x   = ceil(n_out / 4)、grid.y = segment 数、grid.z = 1
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include "kernels_exp_iq_tables.h"

#define E3_ROWS 4

// f16 -> f32 (kernels_exp_iq_v2.cu と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float e3_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int e3_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

// 8 本の f32 を float4 x 2 本で読む (x の先頭は 32 B 境界)。
__device__ __forceinline__ void e3_load8(const float* __restrict__ p, float* v)
{
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ float e3_fold4(float a0, float a1, float a2, float a3)
{
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

struct E3Seg {
    const unsigned char* w;
    const float* x;
    float* out;
};

__device__ __forceinline__ int e3_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & 31;
    return blockIdx.x * E3_ROWS + (t >> 5);
}

__device__ __forceinline__ float e3_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

__device__ __forceinline__ float e3_sgn(unsigned int mask, int j)
{
    return (mask & (1u << j)) ? -1.0f : 1.0f;
}

__device__ __forceinline__ bool e3_prologue(const unsigned char* const* wptrs,
                                            const float* xbase, float* obase,
                                            const unsigned int* seg, int nseg,
                                            E3Seg* s, int* lane, int* row)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return false;
    s->w = wptrs[si];
    s->x = xbase + (size_t)sg[2];
    s->out = obase + (size_t)sg[3];
    *row = e3_row_of(lane);
    return true;
}

// ---------------------------------------------------------------------------
// IQ2_S: d(2B) + qs[32] + signs[32] + qh[8] + scales[8] = 82 B / 256 要素
//   レーン l = 8 要素。表は 2 bit x 8 に詰めた 1024 本 (2 KB) を shared へ。
//   code 0/1/2 = 8/25/43。積は db * {8,25,43} を 1 度だけ作って選ぶ。
// ---------------------------------------------------------------------------
struct E3Iq2s {
    float p0, p1, p2;
    unsigned short pack;
    unsigned int sg;
};

__device__ __forceinline__ E3Iq2s e3_prep_iq2s(const unsigned char* blk,
                                               const unsigned short* tbl,
                                               int lane, int ib32, int l4)
{
    E3Iq2s r;
    const float d = e3_f16(blk);
    const unsigned int qh = (unsigned int)blk[66 + ib32];
    const unsigned int idx = (unsigned int)blk[2 + lane]
                             | ((qh << (8 - 2 * l4)) & 0x300u);
    const unsigned int sc = (unsigned int)blk[74 + ib32];
    const unsigned int nib = (l4 & 2) ? (sc >> 4) : (sc & 0x0fu);
    const float db = d * (0.5f + (float)nib) * 0.25f;
    r.p0 = db * 8.0f;
    r.p1 = db * 25.0f;
    r.p2 = db * 43.0f;
    r.pack = tbl[idx];
    r.sg = (unsigned int)blk[34 + lane];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq2s_v3(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned short sh[1024];
    for (int i = (int)threadIdx.x; i < 1024; i += (int)blockDim.x) {
        sh[i] = dno_iq2s_grid2[i];
    }
    __syncthreads();

    E3Seg s;
    int lane, row;
    if (!e3_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E3Iq2s cur = e3_prep_iq2s(rowp, sh, lane, ib32, l4);
        for (int b = 0; b < nb; ++b) {
            E3Iq2s nxt;
            if (b + 1 < nb) {
                nxt = e3_prep_iq2s(rowp + (size_t)(b + 1) * 82u, sh, lane, ib32, l4);
            }
            float xv[8];
            e3_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const unsigned short pk = cur.pack;
            const unsigned int sg = cur.sg;
            const float p0 = cur.p0, p1 = cur.p1, p2 = cur.p2;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int k = (unsigned int)(pk >> (2 * j)) & 3u;
                const float p = (k == 0u) ? p0 : ((k == 1u) ? p1 : p2);
                const float q = p * e3_sgn(sg, j);
                if ((j & 3) == 0) a0 += q * xv[j];
                else if ((j & 3) == 1) a1 += q * xv[j];
                else if ((j & 3) == 2) a2 += q * xv[j];
                else a3 += q * xv[j];
            }
            cur = nxt;
        }
    }
    float acc = e3_reduce(e3_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
//   レーン l = 8 要素。表は下位 32 bit だけの 512 本 (2 KB) を shared へ。
//   各レーンは g1 / g2 の下位 4 バイトしか使わない (v1/v2 と同じ)。
// ---------------------------------------------------------------------------
struct E3Iq3s {
    float db;
    unsigned int g1;
    unsigned int g2;
    unsigned int signs;
};

__device__ __forceinline__ E3Iq3s e3_prep_iq3s(const unsigned char* blk,
                                               const unsigned int* tbl,
                                               int g, int l4)
{
    E3Iq3s r;
    const float d = e3_f16(blk);
    const unsigned int sc = (unsigned int)blk[106 + (g >> 1)];
    const unsigned int nib = (g & 1) ? (sc >> 4) : (sc & 0x0fu);
    r.db = d * (1.0f + 2.0f * (float)nib);
    const unsigned int qh = (unsigned int)blk[66 + g];
    const unsigned int idx1 = (unsigned int)blk[2 + g * 8 + 2 * l4]
                              | ((qh << (8 - 2 * l4)) & 256u);
    const unsigned int idx2 = (unsigned int)blk[2 + g * 8 + 2 * l4 + 1]
                              | ((qh << (7 - 2 * l4)) & 256u);
    r.signs = (unsigned int)blk[74 + g * 4 + l4];
    r.g1 = tbl[idx1];
    r.g2 = tbl[idx2];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq3s_v3(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned int sh[512];
    for (int i = (int)threadIdx.x; i < 512; i += (int)blockDim.x) {
        sh[i] = dno_iq3s_grid32[i];
    }
    __syncthreads();

    E3Seg s;
    int lane, row;
    if (!e3_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E3Iq3s cur = e3_prep_iq3s(rowp, sh, g, l4);
        for (int b = 0; b < nb; ++b) {
            E3Iq3s nxt;
            if (b + 1 < nb) {
                nxt = e3_prep_iq3s(rowp + (size_t)(b + 1) * 110u, sh, g, l4);
            }
            float xv[8];
            e3_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float db = cur.db;
            const unsigned int sg = cur.signs;
            const unsigned int g1 = cur.g1;
            const unsigned int g2 = cur.g2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g1 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e3_sgn(sg, j);
                if ((j & 1) == 0) a0 += p * xv[j];
                else a1 += p * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g2 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e3_sgn(sg, j + 4);
                if ((j & 1) == 0) a2 += p * xv[4 + j];
                else a3 += p * xv[4 + j];
            }
            cur = nxt;
        }
    }
    float acc = e3_reduce(e3_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}
