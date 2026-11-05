// denno CUDA 層一括 expert GEMV (task 144) — VRAM 常駐 expert を IQ 系の量子型で回す。
//
// PTX の生成 (nvcc の無いコンテナでは実行できない。GPU ラボ / denno-gpu の担当):
//
//   bash build_ptx.sh exp_iq
//
// 生成物は `kernels_exp_iq.ptx` (アーキは既存と同じ compute_75)。表は
// `kernels_exp_iq_tables.h` (python3 gen_exp_iq_tables.py で kquant_tables.rs から生成)。
//
// # 何をするか
//
// `kernels_moe_batch.cu` (P43) と同じ segment 表を使う層一括 GEMV だが、対象の型が
// IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL / Q2_0 で、**累積の並列度が違う**:
//
//   既存: 1 行 = 64 レーンで共有メモリへ写し、lane 0 が 1 本の累積を回す
//         (CPU とビット一致させるため。1 行あたり 1 スレッドしか計算しない)
//   ここ: 1 行 = 1 warp。32 レーンが 8 要素ずつ (IQ4_NL / Q2_0 は 1 ブロックずつ)
//         計算し、最後に __shfl_down_sync の固定順で足す。
//
// task 144 は CPU とのビット一致を要求しない (丸め順の変更は許容)。**決定性**は
// 保つ: 同じ起動形・同じデータなら常に同じ値になる (warp 内の木の形が固定)。
//
// # 数値の出どころ
//
// デコード規則は `crates/denno-engine/src/kquant.rs` の `dequant_block` と 1:1。
// 表 (IQ2S_GRID 等) は `kquant_tables.rs` の写し (値を 1 bit も変えない)。
// 検算は `src/model/exp_iq.rs` のミラー実装 (`dequant_mirror` / `dot_mirror`) が
// GPU 無しで行う (kquant::dequant_row とビット一致すること)。
//
// # segment 表 (`const unsigned int* seg`、1 segment = 4 語)
//
//   seg[s*4 + 0] = expert_id (未使用)
//   seg[s*4 + 1] = m         (decode では 1。1 以外は計算しない)
//   seg[s*4 + 2] = x_off     (入力 base からの f32 要素オフセット)
//   seg[s*4 + 3] = out_off   (出力 base からの f32 要素オフセット)
//
// # 起動形
//
//   blockDim = 128 = 4 warp (EI_ROWS = 4 行、1 行 = 1 warp)
//   grid.x   = ceil(n_out / 4)、grid.y = segment 数、grid.z = 1
//   共有メモリは使わない (行の生バイトをステージしない)

#include "kernels_exp_iq_tables.h"

#define EI_ROWS 4

// f16 -> f32 (kernels_moe.cu と同一の式。モジュールが別なので写しを持つ)
__device__ __forceinline__ float ei_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int ei_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

struct EiSeg {
    const unsigned char* w;
    const float* x;
    float* out;
};

// segment の前置き。範囲外・m != 1 は false (barrier を跨がないので早期 return 可)。
__device__ __forceinline__ bool ei_seg(const unsigned char* const* wptrs,
                                       const float* xbase, float* obase,
                                       const unsigned int* seg, int nseg,
                                       int n_in, int n_out, EiSeg* s)
{
    (void)n_in;
    (void)n_out;
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return false;
    s->w = wptrs[si];
    s->x = xbase + (size_t)sg[2];
    s->out = obase + (size_t)sg[3];
    return true;
}

// 行番号とレーン。行は warp 単位 (同じ warp の全レーンが同じ値を得る)。
__device__ __forceinline__ int ei_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & 31;
    return blockIdx.x * EI_ROWS + (t >> 5);
}

__device__ __forceinline__ float ei_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

// ---------------------------------------------------------------------------
// IQ2_S: d(2B) + qs[32] + signs[32] + qh[8] + scales[8] = 82 B / 256 要素
//   レーン l = 8 要素 (256 / 32)。kquant.rs の IQ2S 分岐と 1:1。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq2s(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 82u;
        const float d = ei_f16(blk);
        const unsigned int qh = (unsigned int)blk[66 + ib32];
        const unsigned int idx = (unsigned int)blk[2 + lane]
                                 | ((qh << (8 - 2 * l4)) & 0x300u);
        const unsigned int sg = (unsigned int)blk[34 + lane];
        const unsigned int sc = (unsigned int)blk[74 + ib32];
        const unsigned int nib = (l4 & 2) ? (sc >> 4) : (sc & 0x0fu);
        const float db = d * (0.5f + (float)nib) * 0.25f;
        const unsigned long long grid = __ldg(&dno_iq2s_grid[idx]);
        const float* xp = s.x + (size_t)b * 256u + (size_t)lane * 8u;
        for (int j = 0; j < 8; ++j) {
            const float gv = (float)((grid >> (8 * j)) & 0xffull);
            const float sgn = (sg & (1u << j)) ? -1.0f : 1.0f;
            acc += (db * gv * sgn) * xp[j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_XXS: d(2B) + qs[64] + ss[32] = 98 B / 256 要素 (ss の上位 4 bit がスケール、
//   7 bit x 4 が符号)。レーン l = 8 要素。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq3xxs(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int ib32 = lane >> 2;
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 98u;
        const float d = ei_f16(blk);
        const unsigned int aux = ei_u32(blk + 66 + ib32 * 4);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        const unsigned int signs = (unsigned int)__ldg(&dno_ksigns_iq2xs[(aux >> (7 * l4)) & 127u]);
        const unsigned int g1 = __ldg(&dno_iq3xxs_grid[blk[2 + ib32 * 8 + 2 * l4]]);
        const unsigned int g2 = __ldg(&dno_iq3xxs_grid[blk[2 + ib32 * 8 + 2 * l4 + 1]]);
        const float* xp = s.x + (size_t)b * 256u + (size_t)lane * 8u;
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((g1 >> (8 * j)) & 0xffu);
            const float sgn = (signs & (1u << j)) ? -1.0f : 1.0f;
            acc += (db * gv * sgn) * xp[j];
        }
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((g2 >> (8 * j)) & 0xffu);
            const float sgn = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
            acc += (db * gv * sgn) * xp[4 + j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110 B / 256 要素
//   レーン l = 8 要素 (32 要素グループ g = l/4 の 1/4)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq3s(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;   // 0..7 (256 要素中の 32 要素グループ)
    const int l4 = lane & 3;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 110u;
        const float d = ei_f16(blk);
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
        const float* xp = s.x + (size_t)b * 256u + (size_t)lane * 8u;
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((g1 >> (8 * j)) & 0xffull);
            const float sgn = (signs & (1u << j)) ? -1.0f : 1.0f;
            acc += (db * gv * sgn) * xp[j];
        }
        for (int j = 0; j < 4; ++j) {
            const float gv = (float)((g2 >> (8 * j)) & 0xffull);
            const float sgn = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
            acc += (db * gv * sgn) * xp[4 + j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ4_XS: d(2B) + scales_h(u16) + scales_l[4] + qs[128] = 136 B / 256 要素
//   レーン l = 8 要素 (32 要素グループ g = l/4 の 4 分の 1、下位/上位ニブル)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq4xs(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int g = lane >> 2;
    const int h4 = lane & 3;              // 32 要素グループ内の 8 要素区画
    const int half = h4 >> 1;             // 0 = 下位ニブル、1 = 上位ニブル
    const int jbase = (h4 & 1) * 8;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 136u;
        const float d = ei_f16(blk);
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned int sl = (unsigned int)blk[4 + (g >> 1)];
        const unsigned int ls = ((sl >> (4 * (g & 1))) & 0x0fu) | (((sh >> (2 * g)) & 3u) << 4);
        const float dl = d * ((float)ls - 32.0f);
        const float* xp = s.x + (size_t)b * 256u + (size_t)lane * 8u;
        const unsigned char* qs = blk + 8 + g * 16 + jbase;
        for (int j = 0; j < 8; ++j) {
            const unsigned int q = half ? ((unsigned int)qs[j] >> 4) : ((unsigned int)qs[j] & 0x0fu);
            const float gv = (float)dno_kvalues_iq4nl[q];
            acc += (dl * gv) * xp[j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// IQ4_NL: d(2B) + qs[16] = 18 B / 32 要素。レーン 1 本 = 1 ブロック
//   (warp で 32 ブロック = 1,024 要素を 1 周で読む)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_iq4nl(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char* blk = rowp + (size_t)b * 18u;
        const float d = ei_f16(blk);
        const float* xp = s.x + (size_t)b * 32u;
        for (int j = 0; j < 16; ++j) {
            const unsigned int lo = (unsigned int)blk[2 + j] & 0x0fu;
            const unsigned int hi = (unsigned int)blk[2 + j] >> 4;
            acc += (d * (float)dno_kvalues_iq4nl[lo]) * xp[j];
            acc += (d * (float)dno_kvalues_iq4nl[hi]) * xp[16 + j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q2_0: d(2B) + qs[16] = 18 B / 64 要素 (2 bit、値は (q-1)*d)。レーン 1 本 = 1 ブロック。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_exp_gemv_q2_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    EiSeg s;
    if (!ei_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = ei_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 6;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char* blk = rowp + (size_t)b * 18u;
        const float d = ei_f16(blk);
        const float* xp = s.x + (size_t)b * 64u;
        for (int j = 0; j < 64; ++j) {
            const unsigned int q = ((unsigned int)blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 3u;
            acc += ((float)((int)q - 1) * d) * xp[j];
        }
    }
    acc = ei_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// act = silu(g) * u (要素ごと)。tensor::silu_inplace と同じ式 (v / (1 + exp(-v)))。
// ホストへ g / u を戻さずに済ませるための小さなカーネル。
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
