// denno CUDA decode: expert の down GEMV と重み付き和を 1 起動に畳む (task 480)
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq_ds
//
// 生成物は `kernels_exp_iq_ds.ptx`。表は `kernels_exp_iq_tables.h`。
//
// # なぜ要るか (REPORT_P476 の §7-4)
//
// 既定モデル (flash-next-UD-IQ3_XXS) の decode は 1 層あたり
// gate+up 21.8 us / silu 3.3 us / **down 16.6 us** / **sum 4.2 us** (us/層、
// [dec-exp-kprof] t476a) で、`sum` は 10 ブロック 256 スレッド = 起動律速
// (2,560 要素の和に 4.2 us)。down は 1 行 = 1 warp の短い GEMV なので、
// 1 warp が同時に持つロードが 1 本しかない。
//
// ここでは **1 出力行 (e 要素) を 1 warp が nseg 個ぶんまとめて**計算する:
//
//   out[j] = sum_t w[t] * dot(down_t の行 j, act_t)      (t 昇順・1 アキュムレータ)
//
// - 和の起動 (4.2 us/層) と、寄与 h の書き戻し/読み直し (nseg*e = 25,600 要素
//   = 100 KB/層) が丸ごと消える。
// - warp 数は 1/nseg になるが、1 warp が **nseg 本の独立したロード列**を持つので
//   未完了ロードの総数は保たれる (REPORT_P476 §7-1 の「warp 数 x warp あたりの
//   未完了ロード」を数える規約)。blockDim = 32*rows で rows を 4/8 から選べる
//   (rows を増やすと 1 ブロックが同じ act を覆う行数が増え、act の再読みが減る)。
//
// # 数値の契約
//
// **`kernels_exp_iq_v2.cu` の down + `kernels_dec_epi.cu` の `dno_dec_exp_sum` と
// ビット一致**。理由:
//   - 1 行の内積は v2 と同じレーン割り・同じブロック順・同じアキュムレータ 4 本
//     (`a[0..3]` へ j 昇順)・同じ `ds_fold4` + 同じ warp の木。
//   - 和は `acc += w[t] * acc_t` を t 昇順に 1 本のアキュムレータで回す。
//     この .cu は -fmad=true なので、`dno_dec_exp_sum` (-fmad=false) と同じ
//     「積 1 回 + 和 1 回」にするため `__fmul_rn` / `__fadd_rn` を明示する。
//
// # segment 表と起動形
//
//   seg[s*4 + 0] = expert_id (未使用) / +1 = m (decode では 1)
//   seg[s*4 + 2] = x_off / +3 = out_off (和を畳むので使わない)
//
//   blockDim = 32 * rows (rows = 4 or 8。ホストが決める)
//   grid.x   = ceil(n_out / rows)、grid.y = 1 (segment はカーネル内で回す)
//   nseg は **DS_NS (= 10) 固定**。違う層はホストが従来の 2 起動へ落とす。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include "kernels_exp_iq_tables.h"

// 1 度に畳む segment 数 (ホストの `crate::cuda::EXP_DS_NS` と一致させる)。
#define DS_NS 10

// f16 -> f32 (kernels_exp_iq_v2.cu と同一の式)
__device__ __forceinline__ float ds_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int ds_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

// 8 本の f32 を float4 x 2 本で読む (x の先頭は 32 B 境界)。
__device__ __forceinline__ void ds_load8(const float* __restrict__ p, float* v)
{
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ void ds_load16(const float* __restrict__ p, float* v)
{
    ds_load8(p, v);
    ds_load8(p + 8, v + 8);
}

// 要素数で読み方を選ぶ (XE = 8 / 16)。クラスの特殊化で選ぶ (関数テンプレートの
// 明示特殊化は nvcc の属性の扱いが版によって変わるため避ける)。
template <int XE>
struct DsLoad;

template <>
struct DsLoad<8> {
    __device__ __forceinline__ static void run(const float* __restrict__ p, float* v)
    {
        ds_load8(p, v);
    }
};

template <>
struct DsLoad<16> {
    __device__ __forceinline__ static void run(const float* __restrict__ p, float* v)
    {
        ds_load16(p, v);
    }
};

// 4 本のアキュムレータを (0+1)+(2+3) で畳む (v2 と同一)。
__device__ __forceinline__ float ds_fold4(float a0, float a1, float a2, float a3)
{
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

// warp 内の合計 (v1/v2 と同じ 16,8,4,2,1 の木)。
__device__ __forceinline__ float ds_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

__device__ __forceinline__ float ds_sgn(unsigned int mask, int j)
{
    return (mask & (1u << j)) ? -1.0f : 1.0f;
}

// ---------------------------------------------------------------------------
// 型ごとの traits
//
//   CH    : 1 ブロックが覆う x の要素数
//   BYTES : 1 ブロックの重みのバイト数
//   LPB   : 1 ブロックを分担するレーン数
//           (1 なら 32 レーンが同じブロックの自分の 8 要素を持ち、
//            2/4 ならブロックを分割する)
//   BL    : warp が同時に舐めるブロック数 = (LPB == 1) ? 32 : LPB
//   XE    : 1 レーンが 1 ブロックで受ける x の要素数
//   Sh    : shared に置く表
//   load(): ブロックのヘッダ (d / 索引 / 表引き) を読む
//   acc() : xv との積和を a[0..3] へ (v2 と同じ順)
// ---------------------------------------------------------------------------

// IQ2_S: d(2B) + qs[32] + signs[32] + qh[8] + scales[8] = 82 B / 256 要素
struct DsIq2sSh {
    unsigned long long grid[1024];
};

struct DsIq2s {
    float db;
    unsigned long long grid;
    unsigned int sg;
    enum { CH = 256, BYTES = 82, LPB = 1, BL = 32, XE = 8 };
    typedef DsIq2sSh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        for (int i = tid; i < 1024; i += nth) sh->grid[i] = dno_iq2s_grid[i];
    }
    __device__ __forceinline__ static DsIq2s load(const unsigned char* blk, int lane,
                                                  const Sh* sh) {
        DsIq2s r;
        const int ib32 = lane >> 2;
        const int l4 = lane & 3;
        const float d = ds_f16(blk);
        const unsigned int qh = (unsigned int)blk[66 + ib32];
        const unsigned int idx = (unsigned int)blk[2 + lane]
                                 | ((qh << (8 - 2 * l4)) & 0x300u);
        const unsigned int sc = (unsigned int)blk[74 + ib32];
        const unsigned int nib = (l4 & 2) ? (sc >> 4) : (sc & 0x0fu);
        r.db = d * (0.5f + (float)nib) * 0.25f;
        r.grid = sh->grid[idx];
        r.sg = (unsigned int)blk[34 + lane];
        return r;
    }
    __device__ __forceinline__ static void acc(const DsIq2s& c, const float* xv,
                                               float* a, const Sh*) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float)((c.grid >> (8 * j)) & 0xffull);
            const float p = (c.db * gv) * ds_sgn(c.sg, j);
            a[j & 3] += p * xv[j];
        }
    }
};

// IQ3_XXS: d(2B) + qs[64] + ss[32] = 98 B / 256 要素
struct DsIq3xxsSh {
    unsigned int grid[256];
    unsigned char sign[128];
};

struct DsIq3xxs {
    float db;
    unsigned int g1;
    unsigned int g2;
    unsigned int signs;
    enum { CH = 256, BYTES = 98, LPB = 1, BL = 32, XE = 8 };
    typedef DsIq3xxsSh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        for (int i = tid; i < 256; i += nth) sh->grid[i] = dno_iq3xxs_grid[i];
        for (int i = tid; i < 128; i += nth) sh->sign[i] = dno_ksigns_iq2xs[i];
    }
    __device__ __forceinline__ static DsIq3xxs load(const unsigned char* blk, int lane,
                                                    const Sh* sh) {
        DsIq3xxs r;
        const int ib32 = lane >> 2;
        const int l4 = lane & 3;
        const float d = ds_f16(blk);
        const unsigned int aux = ds_u32(blk + 66 + ib32 * 4);
        r.db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        r.signs = (unsigned int)sh->sign[(aux >> (7 * l4)) & 127u];
        r.g1 = sh->grid[blk[2 + ib32 * 8 + 2 * l4]];
        r.g2 = sh->grid[blk[2 + ib32 * 8 + 2 * l4 + 1]];
        return r;
    }
    __device__ __forceinline__ static void acc(const DsIq3xxs& c, const float* xv,
                                               float* a, const Sh*) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((c.g1 >> (8 * j)) & 0xffu);
            const float p = (c.db * gv) * ds_sgn(c.signs, j);
            a[j & 1] += p * xv[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((c.g2 >> (8 * j)) & 0xffu);
            const float p = (c.db * gv) * ds_sgn(c.signs, j + 4);
            a[2 + (j & 1)] += p * xv[4 + j];
        }
    }
};

// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
struct DsIq3sSh {
    unsigned long long grid[512];
};

struct DsIq3s {
    float db;
    unsigned long long g1;
    unsigned long long g2;
    unsigned int signs;
    enum { CH = 256, BYTES = 110, LPB = 1, BL = 32, XE = 8 };
    typedef DsIq3sSh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        for (int i = tid; i < 512; i += nth) sh->grid[i] = dno_iq3s_grid[i];
    }
    __device__ __forceinline__ static DsIq3s load(const unsigned char* blk, int lane,
                                                  const Sh* sh) {
        DsIq3s r;
        const int g = lane >> 2;
        const int l4 = lane & 3;
        const float d = ds_f16(blk);
        const unsigned int sc = (unsigned int)blk[106 + (g >> 1)];
        const unsigned int nib = (g & 1) ? (sc >> 4) : (sc & 0x0fu);
        r.db = d * (1.0f + 2.0f * (float)nib);
        const unsigned int qh = (unsigned int)blk[66 + g];
        const unsigned int idx1 = (unsigned int)blk[2 + g * 8 + 2 * l4]
                                  | ((qh << (8 - 2 * l4)) & 256u);
        const unsigned int idx2 = (unsigned int)blk[2 + g * 8 + 2 * l4 + 1]
                                  | ((qh << (7 - 2 * l4)) & 256u);
        r.signs = (unsigned int)blk[74 + g * 4 + l4];
        r.g1 = sh->grid[idx1];
        r.g2 = sh->grid[idx2];
        return r;
    }
    __device__ __forceinline__ static void acc(const DsIq3s& c, const float* xv,
                                               float* a, const Sh*) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((c.g1 >> (8 * j)) & 0xffull);
            const float p = (c.db * gv) * ds_sgn(c.signs, j);
            a[j & 1] += p * xv[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((c.g2 >> (8 * j)) & 0xffull);
            const float p = (c.db * gv) * ds_sgn(c.signs, j + 4);
            a[2 + (j & 1)] += p * xv[4 + j];
        }
    }
};

// IQ4_XS: d(2B) + scales_h(u16) + scales_l[4] + qs[128] = 136 B / 256 要素
struct DsIq4xsSh {
    signed char kv[16];
};

struct DsIq4xs {
    float dl;
    unsigned int q0;
    unsigned int q1;
    enum { CH = 256, BYTES = 136, LPB = 1, BL = 32, XE = 8 };
    typedef DsIq4xsSh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        for (int i = tid; i < 16; i += nth) sh->kv[i] = dno_kvalues_iq4nl[i];
    }
    __device__ __forceinline__ static DsIq4xs load(const unsigned char* blk, int lane,
                                                   const Sh*) {
        DsIq4xs r;
        const int g = lane >> 2;
        const int h4 = lane & 3;
        const float d = ds_f16(blk);
        const unsigned int shv = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned int sl = (unsigned int)blk[4 + (g >> 1)];
        const unsigned int ls = ((sl >> (4 * (g & 1))) & 0x0fu)
                                | (((shv >> (2 * g)) & 3u) << 4);
        r.dl = d * ((float)ls - 32.0f);
        const int jbase = (h4 & 1) * 8;
        const unsigned char* qs = blk + 8 + g * 16 + jbase;
        r.q0 = *reinterpret_cast<const unsigned int*>(qs);
        r.q1 = *reinterpret_cast<const unsigned int*>(qs + 4);
        return r;
    }
    __device__ __forceinline__ static void acc(const DsIq4xs& c, const float* xv,
                                               float* a, const Sh* sh) {
        const int half = ((int)threadIdx.x & 3) >> 1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned int word = (j < 4) ? c.q0 : c.q1;
            const unsigned int byte = (word >> (((j & 3) * 8))) & 0xffu;
            const unsigned int q = half ? (byte >> 4) : (byte & 0x0fu);
            const float p = c.dl * (float)sh->kv[q & 15u];
            a[j & 3] += p * xv[j];
        }
    }
};

// IQ4_NL: d(2B) + qs[16] = 18 B / 32 要素 (2 レーン / ブロック)
struct DsIq4nlSh {
    signed char kv[16];
};

struct DsIq4nl {
    float d;
    unsigned char q[16];
    enum { CH = 32, BYTES = 18, LPB = 2, BL = 2, XE = 16 };
    typedef DsIq4nlSh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        for (int i = tid; i < 16; i += nth) sh->kv[i] = dno_kvalues_iq4nl[i];
    }
    __device__ __forceinline__ static DsIq4nl load(const unsigned char* blk, int,
                                                   const Sh*) {
        DsIq4nl r;
        r.d = ds_f16(blk);
#pragma unroll
        for (int i = 0; i < 16; ++i) r.q[i] = blk[2 + i];
        return r;
    }
    __device__ __forceinline__ static void acc(const DsIq4nl& c, const float* xv,
                                               float* a, const Sh* sh) {
        const int half = (int)threadIdx.x & 1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned int q = half ? ((unsigned int)c.q[j] >> 4)
                                        : ((unsigned int)c.q[j] & 0x0fu);
            const float p = c.d * (float)sh->kv[q & 15u];
            a[j & 3] += p * xv[j];
        }
    }
};

// Q2_0: d(2B) + qs[16] = 18 B / 64 要素 (4 レーン / ブロック、値は (q-1)*d)
struct DsQ20Sh {
    signed char unused;
};

struct DsQ20 {
    float d;
    unsigned char q[4];
    enum { CH = 64, BYTES = 18, LPB = 4, BL = 4, XE = 16 };
    typedef DsQ20Sh Sh;
    __device__ __forceinline__ static void init_sh(Sh* sh, int tid, int nth) {
        (void)sh; (void)tid; (void)nth;
    }
    __device__ __forceinline__ static DsQ20 load(const unsigned char* blk, int lane,
                                                 const Sh*) {
        DsQ20 r;
        const int quarter = lane & 3;
        r.d = ds_f16(blk);
        const unsigned char* qb = blk + 2 + quarter * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.q[i] = qb[i];
        return r;
    }
    __device__ __forceinline__ static void acc(const DsQ20& c, const float* xv,
                                               float* a, const Sh*) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned int q = ((unsigned int)c.q[j >> 2] >> ((j & 3) * 2)) & 3u;
            const float p = (float)((int)q - 1) * c.d;
            a[j & 3] += p * xv[j];
        }
    }
};

// ---------------------------------------------------------------------------
// 本体: 1 行 (出力要素 j) を全 segment ぶん計算し、重み付き和を lane 0 が書く。
// ---------------------------------------------------------------------------
template <class TP>
__device__ __forceinline__ void ds_body(const unsigned char* const* __restrict__ wptrs,
                                        const float* __restrict__ xbase,
                                        const float* __restrict__ w,
                                        float* __restrict__ obase,
                                        const unsigned int* __restrict__ seg,
                                        const int nseg, const int n_in,
                                        const int n_out, const int row_bytes,
                                        const typename TP::Sh* sh)
{
    if (nseg != DS_NS) return;
    const int rows = (int)(blockDim.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int row = (int)blockIdx.x * rows + (int)(threadIdx.x >> 5);
    if (row >= n_out) return;

    const int b0 = lane / (int)TP::BL;
    const int step = 32 / (int)TP::BL;
    const int nblk = n_in / (int)TP::CH;
    const int xoff_lane = (lane & ((int)TP::BL - 1)) * (int)TP::XE;
    const size_t rowoff = (size_t)row * (size_t)row_bytes;

    const float* xp[DS_NS];
    float a[DS_NS][4];
#pragma unroll
    for (int t = 0; t < DS_NS; ++t) {
        xp[t] = xbase + (size_t)seg[t * 4 + 2];
        a[t][0] = 0.0f;
        a[t][1] = 0.0f;
        a[t][2] = 0.0f;
        a[t][3] = 0.0f;
    }

    for (int b = b0; b < nblk; b += step) {
        const size_t woff = rowoff + (size_t)b * (size_t)TP::BYTES;
        // 全 segment のヘッダをまとめて読む (1 warp が nseg 本のロード列を持つ)。
        TP cur[DS_NS];
#pragma unroll
        for (int t = 0; t < DS_NS; ++t) {
            cur[t] = TP::load(wptrs[t] + woff, lane, sh);
        }
        // x は 1 segment ずつ (xv を nseg 本ぶんは持たない)。
#pragma unroll
        for (int t = 0; t < DS_NS; ++t) {
            float xv[TP::XE];
            DsLoad<TP::XE>::run(
                xp[t] + (size_t)b * (size_t)TP::CH + (size_t)xoff_lane, xv);
            TP::acc(cur[t], xv, a[t], sh);
        }
    }

    // 和: `dno_dec_exp_sum` と同じく t 昇順に 1 本のアキュムレータ
    // (積 1 回 + 和 1 回。この .cu は -fmad=true なので明示する)。
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < DS_NS; ++t) {
        const float one = ds_reduce(ds_fold4(a[t][0], a[t][1], a[t][2], a[t][3]));
        acc = __fadd_rn(acc, __fmul_rn(w[t], one));
    }
    if (lane == 0) obase[row] = acc;
}

#define DS_DEFINE(NAME, TP)                                                                  \
    extern "C" __global__ void NAME(const unsigned char* const* __restrict__ wptrs,          \
                                    const float* __restrict__ xbase,                         \
                                    const float* __restrict__ w,                             \
                                    float* __restrict__ obase,                               \
                                    const unsigned int* __restrict__ seg, int nseg,          \
                                    int n_in, int n_out, int row_bytes)                      \
    {                                                                                        \
        __shared__ TP::Sh sh;                                                                \
        TP::init_sh(&sh, (int)threadIdx.x, (int)blockDim.x);                                 \
        __syncthreads();                                                                     \
        ds_body<TP>(wptrs, xbase, w, obase, seg, nseg, n_in, n_out, row_bytes, &sh);          \
    }

DS_DEFINE(dno_exp_gemv_ds_iq2s, DsIq2s)
DS_DEFINE(dno_exp_gemv_ds_iq3xxs, DsIq3xxs)
DS_DEFINE(dno_exp_gemv_ds_iq3s, DsIq3s)
DS_DEFINE(dno_exp_gemv_ds_iq4xs, DsIq4xs)
DS_DEFINE(dno_exp_gemv_ds_iq4nl, DsIq4nl)
DS_DEFINE(dno_exp_gemv_ds_q2_0, DsQ20)

// ---------------------------------------------------------------------------
// task 480: 1 行を 4 warp で分担する版 (`DENNO_DEC_EXPW4=1` / `DENNO_DEC_EXPGU=w4`)
// ---------------------------------------------------------------------------
//
// v2 の層一括 GEMV は **1 行 = 1 warp** で、1 warp が同時に持つロードが 1-2 本
// しかない。lab t480a の [dec-exp-kprof] は gate+up 22.0 us/層 / down 16.7 us/層で、
// どちらも 490-523 GB/s (ピークの 3 割) に留まる。同じモデルの mix / ssm の GEMV は
// 1 ブロック 1 行 (128-256 スレッド) で 700-900 GB/s 出ている = **行あたりの
// 並列度**が違う。ここでは 1 行を 4 warp (WPR) が分担し、行ごとの未完了ロードを
// 4 倍にする。1 ブロック = 4 行 x 4 warp = 512 スレッド。
//
// 数値の契約: **v2 とはビット一致しない** (行の内積を 4 本に割って足す順が変わる)。
// 丸め方の変更だけなので、値の差は f32 の和の順による最下位ビット程度。
// ゲート off (既定) では従来のカーネルを撃つので md5 は変わらない。

#define W4_WPR 4

template <class TP>
__device__ __forceinline__ void w4_body(const unsigned char* const* __restrict__ wptrs,
                                        const float* __restrict__ xbase,
                                        float* __restrict__ obase,
                                        const unsigned int* __restrict__ seg,
                                        const int nseg, const int n_in,
                                        const int n_out, const int row_bytes,
                                        const typename TP::Sh* sh)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int rows = (int)(blockDim.x >> 5) / W4_WPR;
    const int sw = warp % W4_WPR;      // 行の中で何本目の warp か
    const int row = (int)blockIdx.x * rows + warp / W4_WPR;
    const bool act = row < n_out;

    const unsigned int* sg = seg + (size_t)si * 4u;
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (act && (int)sg[1] == 1) {
        const unsigned char* rowp = wptrs[si] + (size_t)row * (size_t)row_bytes;
        const float* xp = xbase + (size_t)sg[2];
        const int nblk = n_in / (int)TP::CH;
        const int xoff_lane = lane * (int)TP::XE;
        for (int b = sw; b < nblk; b += W4_WPR) {
            const TP cur = TP::load(rowp + (size_t)b * (size_t)TP::BYTES, lane, sh);
            float xv[TP::XE];
            DsLoad<TP::XE>::run(xp + (size_t)b * (size_t)TP::CH + (size_t)xoff_lane, xv);
            TP::acc(cur, xv, a, sh);
        }
    }
    // 縮約: warp 内 (v2 と同じ木) -> block 内 (warp 番号の昇順に 1 本の和)。
    const float part = ds_reduce(ds_fold4(a[0], a[1], a[2], a[3]));
    __shared__ float sp[32];
    if (lane == 0) sp[warp] = part;
    __syncthreads();
    if (act && sw == 0 && lane == 0) {
        float s = sp[warp];
#pragma unroll
        for (int k = 1; k < W4_WPR; ++k) s = __fadd_rn(s, sp[warp + k]);
        obase[row] = s;
    }
}

#define W4_DEFINE(NAME, TP)                                                                  \
    extern "C" __global__ void NAME(const unsigned char* const* __restrict__ wptrs,          \
                                    const float* __restrict__ xbase,                         \
                                    float* __restrict__ obase,                               \
                                    const unsigned int* __restrict__ seg, int nseg,          \
                                    int n_in, int n_out, int row_bytes)                      \
    {                                                                                        \
        __shared__ TP::Sh sh;                                                                \
        TP::init_sh(&sh, (int)threadIdx.x, (int)blockDim.x);                                 \
        __syncthreads();                                                                     \
        w4_body<TP>(wptrs, xbase, obase, seg, nseg, n_in, n_out, row_bytes, &sh);             \
    }

W4_DEFINE(dno_exp_gemv_w4_iq2s, DsIq2s)
W4_DEFINE(dno_exp_gemv_w4_iq3xxs, DsIq3xxs)
W4_DEFINE(dno_exp_gemv_w4_iq3s, DsIq3s)
W4_DEFINE(dno_exp_gemv_w4_iq4xs, DsIq4xs)
W4_DEFINE(dno_exp_gemv_w4_iq4nl, DsIq4nl)
W4_DEFINE(dno_exp_gemv_w4_q2_0, DsQ20)
