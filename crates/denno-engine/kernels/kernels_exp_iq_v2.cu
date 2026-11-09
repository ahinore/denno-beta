// denno CUDA 層一括 expert GEMV (v2, task 190) — IQ 系の量子型で回す。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq_v2
//
// 生成物は `kernels_exp_iq_v2.ptx`。表は task 144 と同じ
// `kernels_exp_iq_tables.h` (python3 gen_exp_iq_tables.py で kquant_tables.rs から生成)。
//
// # v1 (`kernels_exp_iq.cu`) との違い (task 184 の moe v5 と同じ手法)
//
//   1. **ベクタロード**: x は `float4` 2 本で読む (v1 は 8 本のスカラ
//      ロード)。IQ4_XS の qs は 8 バイトを uint32 x 2 で読む。
//   2. **次ブロックの先読み**: ブロックのヘッダ (d / 索引 / 表の引き) を
//      **今のブロックの計算より前に** 発行し、レジスタに置いてから畳む。
//      v1 は 1 ブロックごとに「読む → 待つ → 計算」の直列だった。
//   3. **アキュムレータ 4 本**: `acc +=` の依存鎖を 1/4 にする。最後に
//      (0+1)+(2+3) で畳んでから warp 縮約する (v1 は 1 本の鎖 + warp 縮約)。
//   4. **表を shared に置く**: IQ2_S (8 KB) / IQ3_S (4 KB) / IQ3_XXS (1 KB) の
//      グリッド表と ksigns をブロック先頭で共有メモリへ写す。32 レーンが
//      ばらばらの表引きをすると L1 のセクタ再送が 32 回まで膨らむが、
//      shared ならバンク衝突だけで済む。
//   5. **レーンの使い切り (IQ4_NL / Q2_0)**: v1 は 1 レーン = 1 ブロックで、
//      k=480 の down では 32 レーン中 15 本しか働いていなかった。v2 は
//      IQ4_NL を 2 レーン / ブロック (16 要素ずつ)、Q2_0 を 4 レーン /
//      ブロック (16 要素ずつ) に割る。
//
// # 数値の契約
//
// v1 と同じく CPU とはビット一致しない (1 行を 32 レーンで分割して足す)。
// v2 はさらに **アキュムレータ 4 本の合計順** が v1 と違うので、v1 とも
// ビット一致しない。起動形と入力が同じなら出力は決定的 (warp 内の木と
// 4 本の畳み方が固定)。ゲート `DENNO_EXP_IQ_V2` で v1 へ戻せる。
//
// # segment 表 (`const unsigned int* seg`、1 segment = 4 語) と起動形は v1 と同一
//
//   seg[s*4 + 0] = expert_id (未使用)
//   seg[s*4 + 1] = m         (decode では 1。1 以外は計算しない)
//   seg[s*4 + 2] = x_off     (入力 base からの f32 要素オフセット)
//   seg[s*4 + 3] = out_off   (出力 base からの f32 要素オフセット)
//
//   blockDim = 128 = 4 warp (EI_ROWS = 4 行、1 行 = 1 warp)
//   grid.x   = ceil(n_out / 4)、grid.y = segment 数、grid.z = 1
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include "kernels_exp_iq_tables.h"

#define E2_ROWS 4

// f16 -> f32 (kernels_exp_iq.cu と同一の式)
__device__ __forceinline__ float e2_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int e2_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

// 8 本の f32 を float4 x 2 本で読む (x の先頭は 32 B 境界。呼び出し側が保証)。
__device__ __forceinline__ void e2_load8(const float* __restrict__ p, float* v)
{
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// 16 本の f32 を float4 x 4 本で読む。
__device__ __forceinline__ void e2_load16(const float* __restrict__ p, float* v)
{
    e2_load8(p, v);
    e2_load8(p + 8, v + 8);
}

// 4 本のアキュムレータを (0+1)+(2+3) で畳む。
__device__ __forceinline__ float e2_fold4(float a0, float a1, float a2, float a3)
{
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

struct E2Seg {
    const unsigned char* w;
    const float* x;
    float* out;
};

// 行番号とレーン。行は warp 単位 (同じ warp の全レーンが同じ値を得る)。
__device__ __forceinline__ int e2_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & 31;
    return blockIdx.x * E2_ROWS + (t >> 5);
}

// warp 内の合計 (v1 と同じ 16,8,4,2,1 の木)。
__device__ __forceinline__ float e2_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

// 符号ビット 1 本を符号係数にする (分岐なし)。
__device__ __forceinline__ float e2_sgn(unsigned int mask, int j)
{
    return (mask & (1u << j)) ? -1.0f : 1.0f;
}

// ---------------------------------------------------------------------------
// 前置き (共通): segment を引いて、行とレーンを決める。
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool e2_prologue(const unsigned char* const* wptrs,
                                            const float* xbase, float* obase,
                                            const unsigned int* seg, int nseg,
                                            E2Seg* s, int* lane, int* row)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return false;
    s->w = wptrs[si];
    s->x = xbase + (size_t)sg[2];
    s->out = obase + (size_t)sg[3];
    *row = e2_row_of(lane);
    return true;
}

// ---------------------------------------------------------------------------
// IQ2_S: d(2B) + qs[32] + signs[32] + qh[8] + scales[8] = 82 B / 256 要素
//   レーン l = 8 要素 (256 / 32)。表は 1024 本 (10 bit) を shared に写す。
// ---------------------------------------------------------------------------
struct E2Iq2s {
    float db;
    unsigned long long grid;
    unsigned int sg;
};

__device__ __forceinline__ E2Iq2s e2_prep_iq2s(const unsigned char* blk,
                                               const unsigned long long* tbl,
                                               int lane, int ib32, int l4)
{
    E2Iq2s r;
    const float d = e2_f16(blk);
    const unsigned int qh = (unsigned int)blk[66 + ib32];
    const unsigned int idx = (unsigned int)blk[2 + lane]
                             | ((qh << (8 - 2 * l4)) & 0x300u);
    const unsigned int sc = (unsigned int)blk[74 + ib32];
    const unsigned int nib = (l4 & 2) ? (sc >> 4) : (sc & 0x0fu);
    r.db = d * (0.5f + (float)nib) * 0.25f;
    r.grid = tbl[idx];
    r.sg = (unsigned int)blk[34 + lane];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq2s(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned long long sh[1024];
    for (int i = (int)threadIdx.x; i < 1024; i += (int)blockDim.x) {
        sh[i] = dno_iq2s_grid[i];
    }
    __syncthreads();

    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E2Iq2s cur = e2_prep_iq2s(rowp, sh, lane, ib32, l4);
        for (int b = 0; b < nb; ++b) {
            E2Iq2s nxt;
            if (b + 1 < nb) {
                nxt = e2_prep_iq2s(rowp + (size_t)(b + 1) * 82u, sh, lane, ib32, l4);
            }
            float xv[8];
            e2_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const unsigned long long gr = cur.grid;
            const unsigned int sg = cur.sg;
            const float db = cur.db;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float gv = (float)((gr >> (8 * j)) & 0xffull);
                const float p = (db * gv) * e2_sgn(sg, j);
                if ((j & 3) == 0) a0 += p * xv[j];
                else if ((j & 3) == 1) a1 += p * xv[j];
                else if ((j & 3) == 2) a2 += p * xv[j];
                else a3 += p * xv[j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_XXS: d(2B) + qs[64] + ss[32] = 98 B / 256 要素 (ss の上位 4 bit がスケール、
//   7 bit x 4 が符号)。レーン l = 8 要素。グリッド 256 本 + ksigns を shared へ。
// ---------------------------------------------------------------------------
struct E2Iq3xxs {
    float db;
    unsigned int g1;
    unsigned int g2;
    unsigned int signs;
};

__device__ __forceinline__ E2Iq3xxs e2_prep_iq3xxs(const unsigned char* blk,
                                                   const unsigned int* grid,
                                                   const unsigned char* ksigns,
                                                   int ib32, int l4)
{
    E2Iq3xxs r;
    const float d = e2_f16(blk);
    const unsigned int aux = e2_u32(blk + 66 + ib32 * 4);
    r.db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    r.signs = (unsigned int)ksigns[(aux >> (7 * l4)) & 127u];
    r.g1 = grid[blk[2 + ib32 * 8 + 2 * l4]];
    r.g2 = grid[blk[2 + ib32 * 8 + 2 * l4 + 1]];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq3xxs(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned int sh_grid[256];
    __shared__ unsigned char sh_sign[128];
    for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
        sh_grid[i] = dno_iq3xxs_grid[i];
    }
    for (int i = (int)threadIdx.x; i < 128; i += (int)blockDim.x) {
        sh_sign[i] = dno_ksigns_iq2xs[i];
    }
    __syncthreads();

    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E2Iq3xxs cur = e2_prep_iq3xxs(rowp, sh_grid, sh_sign, ib32, l4);
        for (int b = 0; b < nb; ++b) {
            E2Iq3xxs nxt;
            if (b + 1 < nb) {
                nxt = e2_prep_iq3xxs(rowp + (size_t)(b + 1) * 98u, sh_grid, sh_sign,
                                     ib32, l4);
            }
            float xv[8];
            e2_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float db = cur.db;
            const unsigned int sg = cur.signs;
            const unsigned int g1 = cur.g1;
            const unsigned int g2 = cur.g2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g1 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e2_sgn(sg, j);
                if ((j & 1) == 0) a0 += p * xv[j];
                else a1 += p * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g2 >> (8 * j)) & 0xffu);
                const float p = (db * gv) * e2_sgn(sg, j + 4);
                if ((j & 1) == 0) a2 += p * xv[4 + j];
                else a3 += p * xv[4 + j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
//   レーン l = 8 要素 (32 要素グループ g = l/4 の 1/4)。グリッド 512 本を shared へ。
// ---------------------------------------------------------------------------
struct E2Iq3s {
    float db;
    unsigned long long g1;
    unsigned long long g2;
    unsigned int signs;
};

__device__ __forceinline__ E2Iq3s e2_prep_iq3s(const unsigned char* blk,
                                               const unsigned long long* tbl,
                                               int g, int l4)
{
    E2Iq3s r;
    const float d = e2_f16(blk);
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

extern "C" __global__ void dno_exp_gemv_iq3s(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned long long sh[512];
    for (int i = (int)threadIdx.x; i < 512; i += (int)blockDim.x) {
        sh[i] = dno_iq3s_grid[i];
    }
    __syncthreads();

    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E2Iq3s cur = e2_prep_iq3s(rowp, sh, g, l4);
        for (int b = 0; b < nb; ++b) {
            E2Iq3s nxt;
            if (b + 1 < nb) {
                nxt = e2_prep_iq3s(rowp + (size_t)(b + 1) * 110u, sh, g, l4);
            }
            float xv[8];
            e2_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float db = cur.db;
            const unsigned int sg = cur.signs;
            const unsigned long long g1 = cur.g1;
            const unsigned long long g2 = cur.g2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g1 >> (8 * j)) & 0xffull);
                const float p = (db * gv) * e2_sgn(sg, j);
                if ((j & 1) == 0) a0 += p * xv[j];
                else a1 += p * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g2 >> (8 * j)) & 0xffull);
                const float p = (db * gv) * e2_sgn(sg, j + 4);
                if ((j & 1) == 0) a2 += p * xv[4 + j];
                else a3 += p * xv[4 + j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ4_XS: d(2B) + scales_h(u16) + scales_l[4] + qs[128] = 136 B / 256 要素
//   レーン l = 8 要素。qs の 8 バイトは 8 B 境界なので uint32 x 2 で読む。
// ---------------------------------------------------------------------------
struct E2Iq4xs {
    float dl;
    unsigned int q0;
    unsigned int q1;
};

__device__ __forceinline__ E2Iq4xs e2_prep_iq4xs(const unsigned char* blk,
                                                 int g, int h4)
{
    E2Iq4xs r;
    const float d = e2_f16(blk);
    const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
    const unsigned int sl = (unsigned int)blk[4 + (g >> 1)];
    const unsigned int ls = ((sl >> (4 * (g & 1))) & 0x0fu) | (((sh >> (2 * g)) & 3u) << 4);
    r.dl = d * ((float)ls - 32.0f);
    const int jbase = (h4 & 1) * 8;
    const unsigned char* qs = blk + 8 + g * 16 + jbase;
    // 8+16g+8*(h4&1) は 8 の倍数で、ブロック先頭は 136 B 刻み (8 の倍数) なので
    // uint32 ロードは 4 B 境界を満たす。
    r.q0 = *reinterpret_cast<const unsigned int*>(qs);
    r.q1 = *reinterpret_cast<const unsigned int*>(qs + 4);
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq4xs(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ signed char sh_kv[16];
    for (int i = (int)threadIdx.x; i < 16; i += (int)blockDim.x) {
        sh_kv[i] = dno_kvalues_iq4nl[i];
    }
    __syncthreads();

    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int h4 = lane & 3;
    const int half = h4 >> 1;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E2Iq4xs cur = e2_prep_iq4xs(rowp, g, h4);
        for (int b = 0; b < nb; ++b) {
            E2Iq4xs nxt;
            if (b + 1 < nb) {
                nxt = e2_prep_iq4xs(rowp + (size_t)(b + 1) * 136u, g, h4);
            }
            float xv[8];
            e2_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float dl = cur.dl;
            const unsigned int w0 = cur.q0;
            const unsigned int w1 = cur.q1;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int word = (j < 4) ? w0 : w1;
                const unsigned int byte = (word >> (((j & 3) * 8))) & 0xffu;
                const unsigned int q = half ? (byte >> 4) : (byte & 0x0fu);
                const float p = dl * (float)sh_kv[q];
                if ((j & 3) == 0) a0 += p * xv[j];
                else if ((j & 3) == 1) a1 += p * xv[j];
                else if ((j & 3) == 2) a2 += p * xv[j];
                else a3 += p * xv[j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ4_NL: d(2B) + qs[16] = 18 B / 32 要素。
//   v2 は **2 レーン / ブロック** (16 要素ずつ)。v1 は 1 レーン = 1 ブロックで、
//   k=480 では 32 レーン中 15 本しか働いていなかった。
//   レーンは (half, block) = (lane & 1, lane >> 1)、ブロックは 16 個ずつ進む。
// ---------------------------------------------------------------------------
struct E2Iq4nl {
    float d;
    unsigned char q[16];
};

__device__ __forceinline__ E2Iq4nl e2_prep_iq4nl(const unsigned char* blk)
{
    E2Iq4nl r;
    r.d = e2_f16(blk);
#pragma unroll
    for (int i = 0; i < 16; ++i) r.q[i] = blk[2 + i];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq4nl(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ signed char sh_kv[16];
    for (int i = (int)threadIdx.x; i < 16; i += (int)blockDim.x) {
        sh_kv[i] = dno_kvalues_iq4nl[i];
    }
    __syncthreads();

    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int half = lane & 1;
    const int blk0 = lane >> 1;
    const int nb = n_in >> 5;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (blk0 < nb) {
        E2Iq4nl cur = e2_prep_iq4nl(rowp + (size_t)blk0 * 18u);
        for (int b = blk0; b < nb; b += 16) {
            E2Iq4nl nxt;
            if (b + 16 < nb) {
                nxt = e2_prep_iq4nl(rowp + (size_t)(b + 16) * 18u);
            }
            float xv[16];
            e2_load16(s.x + (size_t)b * 32u + (size_t)half * 16u, xv);
            const float d = cur.d;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int q = half ? ((unsigned int)cur.q[j] >> 4)
                                            : ((unsigned int)cur.q[j] & 0x0fu);
                const float p = d * (float)sh_kv[q];
                if ((j & 3) == 0) a0 += p * xv[j];
                else if ((j & 3) == 1) a1 += p * xv[j];
                else if ((j & 3) == 2) a2 += p * xv[j];
                else a3 += p * xv[j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q2_0: d(2B) + qs[16] = 18 B / 64 要素 (2 bit、値は (q-1)*d)。
//   v2 は **4 レーン / ブロック** (16 要素ずつ)。レーンは
//   (quarter, block) = (lane & 3, lane >> 2)、ブロックは 8 個ずつ進む。
// ---------------------------------------------------------------------------
struct E2Q20 {
    float d;
    unsigned char q[4];
};

__device__ __forceinline__ E2Q20 e2_prep_q20(const unsigned char* blk, int quarter)
{
    E2Q20 r;
    r.d = e2_f16(blk);
    const unsigned char* qb = blk + 2 + quarter * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) r.q[i] = qb[i];
    return r;
}

extern "C" __global__ void dno_exp_gemv_q2_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    E2Seg s;
    int lane, row;
    if (!e2_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int quarter = lane & 3;
    const int blk0 = lane >> 2;
    const int nb = n_in >> 6;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (blk0 < nb) {
        E2Q20 cur = e2_prep_q20(rowp + (size_t)blk0 * 18u, quarter);
        for (int b = blk0; b < nb; b += 8) {
            E2Q20 nxt;
            if (b + 8 < nb) {
                nxt = e2_prep_q20(rowp + (size_t)(b + 8) * 18u, quarter);
            }
            float xv[16];
            e2_load16(s.x + (size_t)b * 64u + (size_t)quarter * 16u, xv);
            const float d = cur.d;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int q =
                    ((unsigned int)cur.q[j >> 2] >> ((j & 3) * 2)) & 3u;
                const float p = (float)((int)q - 1) * d;
                if ((j & 3) == 0) a0 += p * xv[j];
                else if ((j & 3) == 1) a1 += p * xv[j];
                else if ((j & 3) == 2) a2 += p * xv[j];
                else a3 += p * xv[j];
            }
            cur = nxt;
        }
    }
    float acc = e2_reduce(e2_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// act = silu(g) * u (要素ごと)。v1 と同じ式・同じ起動形 (v2 では使わないが、
// PTX に同じ名前を残して Rust 側の読み込みを 1 本で済ませる)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_silu_mul(
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
