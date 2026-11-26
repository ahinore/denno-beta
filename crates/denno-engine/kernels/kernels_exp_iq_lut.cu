// denno CUDA 層一括 expert GEMV (lut, task 940) — IQ3_XXS / IQ3_S の decode 用。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq_lut
//
// 生成物は `kernels_exp_iq_lut.ptx`。表は `kernels_exp_iq_tables.h`。
// ゲートは `DENNO_CUDA_IQ3_LUT_OPT` (**既定 off**)。off のときは v2 / v3 のまま。
//
// # 何を変えるか (task 940 の監査結果)
//
// 1. **IQ3_XXS の ksigns を shared のバイト gather から warp shuffle へ。**
//    32 レーンが 128 バイトの表からばらばらの 1 バイトを読むと、bank は
//    (idx >> 2) % 32 で決まるので、32 本のランダム読みは平均 20.4 個の異なる語に
//    散らばる = 実質 20 回ぶんの LDS 再送になる (同じ語に当たったレーンだけ放送で
//    まとまる)。表は 128 バイト = ちょうど 32 語なので、**1 レーン = 1 語**を
//    レジスタに持ち、`__shfl_sync` で引ける (shuffle に bank 衝突は無い)。
//    レジスタ 1 本 + shuffle 1 本で、再送の無い 1 サイクルになる。
//
// 2. **IQ3_S の格子を u32 から u16 へ詰める (2 KB -> 1 KB)。**
//    IQ3_S の格子の値は 4 bit ({0,1,3,5,7,9,11,13,15}) に収まるので、4 値を
//    1 本の u16 に詰められる (`dno_iq3s_grid16`)。ブロック先頭の shared への
//    写しが半分になり、32 レーンのランダム読みが触る語の数も半分になる
//    (512 語 -> 256 語)。
//
// 表の値・積の順・アキュムレータ 4 本・warp の木・起動形 (4 行 / ブロック、
// blockDim 128、grid = ceil(n_out/4) x nseg) は v2 / v3 と同一なので、
// **出力は v2 / v3 とビット一致**する (数値契約は変えない。性能だけのゲート)。
// 索引の算術と詰め表の一致は `src/model/exp_iq.rs` の CPU ミラーで固定する。
//
// # 監査の結論 (何が効かないか)
//
// IQ3_XXS の格子 (256 語 / 1 KB) は、1 レーンが 256 要素ブロックあたり 2 語
// しか引かないため、衝突しても実測の再送は 32 語のランダム読みで平均 ~2 回。
// shared から __ldg へ移す案は、gather が触る L1 セクタ数 (約 20) が
// そのまま再送になるので不利 (ブロック先頭の写しは 1,152 バイトで、カーネル
// 全体の 1% 未満)。触るのは IQ3_XXS では ksigns だけにした。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include "kernels_exp_iq_tables.h"
#include <cuda_fp16.h>

#define E4_ROWS 4

// f16 -> f32 (kernels_exp_iq_v2.cu / _v3.cu と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float e4_f16(const unsigned char* b)
{
    const unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    if ((h & 0x7c00u) == 0x7c00u) {
        const unsigned int sign = (h & 0x8000u) << 16;
        return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
    }
    return __half2float(__ushort_as_half((unsigned short)h));
}

__device__ __forceinline__ unsigned int e4_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
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
// IQ3_XXS: d(2B) + qs[64] + ss[32] = 98 B / 256 要素 (ss の上位 4 bit がスケール、
//   7 bit x 4 が符号)。レーン l = 8 要素。
//
//   格子は v2 と同じ 1 KB を shared へ (256 語)。符号だけレジスタ + shuffle:
//   `ksw` はレーン番号の語 (ksigns[4*lane .. 4*lane+3]) で、符号バイト idx は
//   語 (idx >> 2) のバイト (idx & 3)。全レーンが同じ位置で shuffle に参加する
//   (呼び出しは nb と blockIdx だけで決まる一様な分岐の中)。
// ---------------------------------------------------------------------------
struct E4Iq3xxs {
    float db;
    unsigned int g1;
    unsigned int g2;
    unsigned int signs;
};

// ksigns の 1 バイトを shuffle で引く (`my` は自分のレーンが持つ語)。
__device__ __forceinline__ unsigned int e4_sign_byte(unsigned int my, unsigned int idx)
{
    const unsigned int w = __shfl_sync(0xffffffffu, my, (int)(idx >> 2));
    return (w >> (8u * (idx & 3u))) & 0xffu;
}

__device__ __forceinline__ E4Iq3xxs e4_prep_iq3xxs(const unsigned char* blk,
                                                   const unsigned int* grid,
                                                   unsigned int ksw,
                                                   int ib32, int l4)
{
    E4Iq3xxs r;
    const float d = e4_f16(blk);
    const unsigned int aux = e4_u32(blk + 66 + ib32 * 4);
    r.db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    r.signs = e4_sign_byte(ksw, (aux >> (7 * l4)) & 127u);
    r.g1 = grid[blk[2 + ib32 * 8 + 2 * l4]];
    r.g2 = grid[blk[2 + ib32 * 8 + 2 * l4 + 1]];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq3xxs_lut(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned int sh_grid[256];
    for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
        sh_grid[i] = dno_iq3xxs_grid[i];
    }
    __syncthreads();

    E4Seg s;
    int lane, row;
    if (!e4_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    // 符号表 (128 バイト = 32 語) は 1 レーン 1 語で持つ。行が無い warp は
    // ここで帰るが、行は warp で一様なので shuffle のマスクは揃う。
    if (row >= n_out) return;
    const unsigned int ksw = dno_ksigns_iq2xs32[lane];

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E4Iq3xxs cur = e4_prep_iq3xxs(rowp, sh_grid, ksw, ib32, l4);
        for (int b = 0; b < nb; ++b) {
            E4Iq3xxs nxt;
            if (b + 1 < nb) {
                nxt = e4_prep_iq3xxs(rowp + (size_t)(b + 1) * 98u, sh_grid, ksw,
                                     ib32, l4);
            }
            float xv[8];
            e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float db = cur.db;
            const unsigned int sg = cur.signs;
            const unsigned int g1 = cur.g1;
            const unsigned int g2 = cur.g2;
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
            cur = nxt;
        }
    }
    float acc = e4_reduce(e4_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
//   レーン l = 8 要素 (32 要素グループ g = l/4 の 1/4)。
//
//   格子は 4 bit を 4 本詰めた u16 (512 本 = 1 KB) を shared へ。v2 の u64 表
//   4 KB / v3 の u32 表 2 KB に対して、写しも読みも半分になる。値は同じ。
// ---------------------------------------------------------------------------
struct E4Iq3s {
    float db;
    unsigned int g1;
    unsigned int g2;
    unsigned int signs;
};

__device__ __forceinline__ E4Iq3s e4_prep_iq3s(const unsigned char* blk,
                                               const unsigned short* tbl,
                                               int g, int l4)
{
    E4Iq3s r;
    const float d = e4_f16(blk);
    const unsigned int sc = (unsigned int)blk[106 + (g >> 1)];
    const unsigned int nib = (g & 1) ? (sc >> 4) : (sc & 0x0fu);
    r.db = d * (1.0f + 2.0f * (float)nib);
    const unsigned int qh = (unsigned int)blk[66 + g];
    const unsigned int idx1 = (unsigned int)blk[2 + g * 8 + 2 * l4]
                              | ((qh << (8 - 2 * l4)) & 256u);
    const unsigned int idx2 = (unsigned int)blk[2 + g * 8 + 2 * l4 + 1]
                              | ((qh << (7 - 2 * l4)) & 256u);
    r.signs = (unsigned int)blk[74 + g * 4 + l4];
    r.g1 = (unsigned int)tbl[idx1];
    r.g2 = (unsigned int)tbl[idx2];
    return r;
}

extern "C" __global__ void dno_exp_gemv_iq3s_lut(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ unsigned short sh[512];
    for (int i = (int)threadIdx.x; i < 512; i += (int)blockDim.x) {
        sh[i] = dno_iq3s_grid16[i];
    }
    __syncthreads();

    E4Seg s;
    int lane, row;
    if (!e4_prologue(wptrs, xbase, obase, seg, nseg, &s, &lane, &row)) return;
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (nb > 0) {
        E4Iq3s cur = e4_prep_iq3s(rowp, sh, g, l4);
        for (int b = 0; b < nb; ++b) {
            E4Iq3s nxt;
            if (b + 1 < nb) {
                nxt = e4_prep_iq3s(rowp + (size_t)(b + 1) * 110u, sh, g, l4);
            }
            float xv[8];
            e4_load8(s.x + (size_t)b * 256u + (size_t)lane * 8u, xv);
            const float db = cur.db;
            const unsigned int sg = cur.signs;
            const unsigned int g1 = cur.g1;
            const unsigned int g2 = cur.g2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g1 >> (4 * j)) & 0x0fu);
                const float p = (db * gv) * e4_sgn(sg, j);
                if ((j & 1) == 0) a0 += p * xv[j];
                else a1 += p * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = (float)((g2 >> (4 * j)) & 0x0fu);
                const float p = (db * gv) * e4_sgn(sg, j + 4);
                if ((j & 1) == 0) a2 += p * xv[4 + j];
                else a3 += p * xv[4 + j];
            }
            cur = nxt;
        }
    }
    float acc = e4_reduce(e4_fold4(a0, a1, a2, a3));
    if (lane == 0) s.out[row] = acc;
}
