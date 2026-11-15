// denno CUDA 層一括 expert GEMV (v4, task 342) — IQ2_S / IQ3_S の**先読みを深くする**。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq_v4
//
// 生成物は `kernels_exp_iq_v4.ptx`。表は `kernels_exp_iq_tables.h` (v3 と共有)。
//
// # なぜ v3 を触るか (lab t342a / P338 の実測)
//
// IQ3_S の gate+up (10 expert x 512 行 x k=2560 = 11.2 MB) が 25.4 us = **441 GB/s**。
// 同じ run の `dno_gemv_q5k_v5` (28.8 MB, 30.7 us) は **940 GB/s** で走っている。
// 差は 1 warp = 1 行の内側の**ロードの細さ**:
//
//   v5 は 1 レーンが 1 ブロック 144 B を uint4 x 9 本で読む (同時に 9 本飛ぶ)。
//   v3 は 1 レーンが 1 ブロックあたり **6 バイト** (d 2 / sc 1 / qh 1 / qs 2 /
//      signs 1) しか読まず、しかも `cur` / `nxt` の 1 ブロック先読みなので
//     **1 ブロックぶんのレイテンシしか隠せない**。k=2560 は 10 ブロック = 10 回の
//     直列レイテンシになり、これが 441 GB/s の主因と読める。
//
// ここでは **nb (n_in/256) ブロックぶんのメタデータを先に全部読む**
// (`nb <= E4_MAXB` のとき)。10 本の独立ロードが同時に飛ぶので、レイテンシは
// 1 回で済む。x は全行で同じ (L1 に載る) のでそのまま 1 ブロックずつ読む。
//
// # 数値の契約
//
// **v3 とビット一致**。変わるのはロードの発行順だけ:
//   - メタデータ (d / sc / qh / qs / signs) の値は同じバイトから作る。
//   - 積と加算の順は 1 要素も変えない (a0..a3 の 4 アキュムレータ・
//     j 昇順・e3_fold4 と warp の木も v3 と同じ)。
//   - 縮約の形が同じなので、同じ入力なら同じビット列になる。
//
// # segment 表と起動形は v1/v2/v3 と同一
//
//   seg[s*4 + 0] = expert_id (未使用) / +1 = m (decode では 1)
//   seg[s*4 + 2] = x_off / +3 = out_off
//
//   blockDim = 128 = 4 warp (E4_ROWS = 4 行、1 行 = 1 warp)
//   grid.x   = ceil(n_out / 4)、grid.y = segment 数、grid.z = 1
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include "kernels_exp_iq_tables.h"

#define E4_ROWS 4
// 先読みでまとめて読むブロック数の上限 (n_in <= 256 * 12 = 3072)。
// これを超える形は v3 と同じ 1 ブロック先読みのループへ落ちる。
#define E4_MAXB 12

// f16 -> f32 (kernels_exp_iq_v3.cu の e3_f16 と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float e4_f16_bits(unsigned int h)
{
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

__device__ __forceinline__ float e4_f16(const unsigned char* b)
{
    return e4_f16_bits((unsigned int)b[0] | ((unsigned int)b[1] << 8));
}

// 8 本の f32 を float4 x 2 本で読む (x の先頭は 32 B 境界)。
__device__ __forceinline__ void e4_load8(const float* __restrict__ p, float* v)
{
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ float e4_fold4(float a0, float a1, float a2, float a3)
{
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

struct E4Seg {
    const unsigned char* w;
    const float* x;
    float* out;
};

__device__ __forceinline__ int e4_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & 31;
    return blockIdx.x * E4_ROWS + (t >> 5);
}

__device__ __forceinline__ float e4_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

__device__ __forceinline__ float e4_sgn(unsigned int mask, int j)
{
    return (mask & (1u << j)) ? -1.0f : 1.0f;
}

__device__ __forceinline__ bool e4_prologue(const unsigned char* const* wptrs,
                                            const float* xbase, float* obase,
                                            const unsigned int* seg, int nseg,
                                            E4Seg* s, int* lane, int* row)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return false;
    s->w = wptrs[si];
    s->x = xbase + (size_t)sg[2];
    s->out = obase + (size_t)sg[3];
    *row = e4_row_of(lane);
    return true;
}

// ---------------------------------------------------------------------------
// IQ2_S: d(2B) + qs[32] + signs[32] + qh[8] + scales[8] = 82 B / 256 要素
//   レーン l = 8 要素。表は 2 bit x 8 に詰めた 1024 本 (2 KB) を shared へ。
//   code 0/1/2 = 8/25/43。積は db * {8,25,43} を 1 度だけ作って選ぶ。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq2s_v4(
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

    E4Seg s;
    int lane, row;
    if (!e4_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    // ブロック内の固定オフセット (レーンごとに違う)
    const int qh_off = 66 + ib32;
    const int sc_off = 74 + ib32;
    const int sg_off = 34 + lane;
    const int qs_off = 2 + lane;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0 && nb <= E4_MAXB) {
        // メタデータを全ブロックぶん先に読む (独立ロードを同時に飛ばす)。
        // ma = d_raw(16) | qh(8) << 16 | sc(8) << 24
        // mb = qs(8) | signs(8) << 8
        unsigned int ma[E4_MAXB];
        unsigned int mb[E4_MAXB];
#pragma unroll
        for (int b = 0; b < E4_MAXB; ++b) {
            if (b < nb) {
                const unsigned char* blk = rowp + (size_t)b * 82u;
                ma[b] = ((unsigned int)blk[0] | ((unsigned int)blk[1] << 8))
                        | ((unsigned int)blk[qh_off] << 16)
                        | ((unsigned int)blk[sc_off] << 24);
                mb[b] = (unsigned int)blk[qs_off] | ((unsigned int)blk[sg_off] << 8);
            }
        }
#pragma unroll
        for (int b = 0; b < E4_MAXB; ++b) {
            if (b < nb) {
                const float d = e4_f16_bits(ma[b] & 0xffffu);
                const unsigned int qh = (ma[b] >> 16) & 0xffu;
                const unsigned int scv = (ma[b] >> 24) & 0xffu;
                const unsigned int nib = (l4 & 2) ? (scv >> 4) : (scv & 0x0fu);
                const float db = d * (0.5f + (float)nib) * 0.25f;
                const float p0 = db * 8.0f;
                const float p1 = db * 25.0f;
                const float p2 = db * 43.0f;
                const unsigned int idx =
                    (mb[b] & 0xffu) | ((qh << (8 - 2 * l4)) & 0x300u);
                const unsigned short pk = sh[idx];
                const unsigned int sg = (mb[b] >> 8) & 0xffu;
                float xv[8];
                e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned int k = (unsigned int)(pk >> (2 * j)) & 3u;
                    const float p = (k == 0u) ? p0 : ((k == 1u) ? p1 : p2);
                    const float q = p * e4_sgn(sg, j);
                    if ((j & 3) == 0) a0 += q * xv[j];
                    else if ((j & 3) == 1) a1 += q * xv[j];
                    else if ((j & 3) == 2) a2 += q * xv[j];
                    else a3 += q * xv[j];
                }
            }
        }
    } else if (nb > 0) {
        // 長すぎる形は v3 と同じ 1 ブロック先読み (値は同じ)。
        float d0 = e4_f16(rowp);
        unsigned int nq0 = (unsigned int)rowp[qh_off];
        unsigned int ns0 = (unsigned int)rowp[sc_off];
        unsigned int nq1 = (unsigned int)rowp[qs_off];
        unsigned int ns1 = (unsigned int)rowp[sg_off];
        for (int b = 0; b < nb; ++b) {
            const float d = d0;
            const unsigned int qh = nq0;
            const unsigned int scv = ns0;
            const unsigned int qsv = nq1;
            const unsigned int sgv = ns1;
            if (b + 1 < nb) {
                const unsigned char* nblk = rowp + (size_t)(b + 1) * 82u;
                d0 = e4_f16(nblk);
                nq0 = (unsigned int)nblk[qh_off];
                ns0 = (unsigned int)nblk[sc_off];
                nq1 = (unsigned int)nblk[qs_off];
                ns1 = (unsigned int)nblk[sg_off];
            }
            const unsigned int nib = (l4 & 2) ? (scv >> 4) : (scv & 0x0fu);
            const float db = d * (0.5f + (float)nib) * 0.25f;
            const float p0 = db * 8.0f;
            const float p1 = db * 25.0f;
            const float p2 = db * 43.0f;
            const unsigned int idx = qsv | ((qh << (8 - 2 * l4)) & 0x300u);
            const unsigned short pk = sh[idx];
            float xv[8];
            e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int k = (unsigned int)(pk >> (2 * j)) & 3u;
                const float p = (k == 0u) ? p0 : ((k == 1u) ? p1 : p2);
                const float q = p * e4_sgn(sgv, j);
                if ((j & 3) == 0) a0 += q * xv[j];
                else if ((j & 3) == 1) a1 += q * xv[j];
                else if ((j & 3) == 2) a2 += q * xv[j];
                else a3 += q * xv[j];
            }
        }
    }
    float acc = e4_reduce(e4_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
//   レーン l = 8 要素。表は下位 32 bit だけの 512 本 (2 KB) を shared へ。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq3s_v4(
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

    E4Seg s;
    int lane, row;
    if (!e4_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int l4 = lane & 3;
    const int sc_off = 106 + (g >> 1);
    const int qh_off = 66 + g;
    const int qs_off = 2 + g * 8 + 2 * l4;
    const int sg_off = 74 + g * 4 + l4;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0 && nb <= E4_MAXB) {
        // ma = d_raw(16) | sc(8) << 16 | qh(8) << 24
        // mb = qs0(8) | qs1(8) << 8 | signs(8) << 16
        unsigned int ma[E4_MAXB];
        unsigned int mb[E4_MAXB];
#pragma unroll
        for (int b = 0; b < E4_MAXB; ++b) {
            if (b < nb) {
                const unsigned char* blk = rowp + (size_t)b * 110u;
                ma[b] = ((unsigned int)blk[0] | ((unsigned int)blk[1] << 8))
                        | ((unsigned int)blk[sc_off] << 16)
                        | ((unsigned int)blk[qh_off] << 24);
                mb[b] = (unsigned int)blk[qs_off] | ((unsigned int)blk[qs_off + 1] << 8)
                        | ((unsigned int)blk[sg_off] << 16);
            }
        }
#pragma unroll
        for (int b = 0; b < E4_MAXB; ++b) {
            if (b < nb) {
                const float d = e4_f16_bits(ma[b] & 0xffffu);
                const unsigned int scv = (ma[b] >> 16) & 0xffu;
                const unsigned int qh = (ma[b] >> 24) & 0xffu;
                const unsigned int nib = (g & 1) ? (scv >> 4) : (scv & 0x0fu);
                const float db = d * (1.0f + 2.0f * (float)nib);
                const unsigned int idx1 =
                    ((mb[b] >> 0) & 0xffu) | ((qh << (8 - 2 * l4)) & 256u);
                const unsigned int idx2 =
                    ((mb[b] >> 8) & 0xffu) | ((qh << (7 - 2 * l4)) & 256u);
                const unsigned int g1 = sh[idx1];
                const unsigned int g2 = sh[idx2];
                const unsigned int sg = (mb[b] >> 16) & 0xffu;
                float xv[8];
                e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float gv = (float)((g1 >> (8 * j)) & 0xffu);
                    const float p = (db * gv) * e4_sgn(sg, j);
                    if ((j & 1) == 0) a0 += p * xv[j];
                    else a1 += p * xv[j];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float gv = (float)((g2 >> (8 * j)) & 0xffu);
                    const float p = (db * gv) * e4_sgn(sg, j + 4);
                    if ((j & 1) == 0) a2 += p * xv[4 + j];
                    else a3 += p * xv[4 + j];
                }
            }
        }
    } else if (nb > 0) {
        // 長すぎる形は v3 と同じ 1 ブロック先読み (値は同じ)。
        float d0 = e4_f16(rowp);
        unsigned int nsc = (unsigned int)rowp[sc_off];
        unsigned int nqh = (unsigned int)rowp[qh_off];
        unsigned int nq0 = (unsigned int)rowp[qs_off];
        unsigned int nq1 = (unsigned int)rowp[qs_off + 1];
        unsigned int nsg = (unsigned int)rowp[sg_off];
        for (int b = 0; b < nb; ++b) {
            const float d = d0;
            const unsigned int scv = nsc;
            const unsigned int qh = nqh;
            const unsigned int q0 = nq0;
            const unsigned int q1 = nq1;
            const unsigned int sgv = nsg;
            if (b + 1 < nb) {
                const unsigned char* nblk = rowp + (size_t)(b + 1) * 110u;
                d0 = e4_f16(nblk);
                nsc = (unsigned int)nblk[sc_off];
                nqh = (unsigned int)nblk[qh_off];
                nq0 = (unsigned int)nblk[qs_off];
                nq1 = (unsigned int)nblk[qs_off + 1];
                nsg = (unsigned int)nblk[sg_off];
            }
            const unsigned int nib = (g & 1) ? (scv >> 4) : (scv & 0x0fu);
            const float db = d * (1.0f + 2.0f * (float)nib);
            const unsigned int idx1 = q0 | ((qh << (8 - 2 * l4)) & 256u);
            const unsigned int idx2 = q1 | ((qh << (7 - 2 * l4)) & 256u);
            const unsigned int g1 = sh[idx1];
            const unsigned int g2 = sh[idx2];
            float xv[8];
            e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g1 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e4_sgn(sgv, j);
                if ((j & 1) == 0) a0 += p * xv[j];
                else a1 += p * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g2 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e4_sgn(sgv, j + 4);
                if ((j & 1) == 0) a2 += p * xv[4 + j];
                else a3 += p * xv[4 + j];
            }
        }
    }
    float acc = e4_reduce(e4_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}
