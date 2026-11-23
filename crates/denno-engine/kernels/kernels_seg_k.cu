// denno CUDA 層一括 expert GEMV の高速版 (task 278) — K 系 4 型を warp 単位で回す。
//
// PTX の生成 (このコンテナに nvcc は無い。lab / denno-gpu 側で実行する):
//
//   bash build_ptx.sh seg_k        -> kernels_seg_k.ptx
//
// # なぜ要るか
//
// `kernels_moe_batch.cu` (P43) は **CPU とビット一致**させるため、1 行を 64 レーンで
// 共有メモリへ写し、lane 0 が i の昇順に 1 本の累積を回す。この形は 1 行あたり
// 1 スレッドしか計算しないので、UD-Q4_K_XL の decode で実測 **gate 53 GB/s /
// up 54 GB/s / down 92 GB/s** しか出ない (lab t278a の `[seg-prof]`。5090 の
// VRAM 帯域は約 1.8 TB/s)。1 層 3 起動で 0.44 ms かかり、これが新しい
// 「1 待ち」経路の残りの主因になった。
//
// # 何をするか
//
// `kernels_exp_iq.cu` (task 144) と同じ形にする: **1 行 = 1 warp**。32 レーンが
// k 方向を分担して足し、最後に `__shfl_down_sync` の固定順で 1 本にまとめる。
// 数値契約は CPU とビット一致**しない** (丸め順が変わる) が、起動形が同じなら
// 値は決定的 (木の形が固定)。task 278 は速度優先なので許容する。
// 検算は `DENNO_EXPERT_GPU_CHECK=1` が層ごとに CPU との max_abs / max_rel を出す。
//
// # segment 表 (`const unsigned int* seg`、1 segment = 4 語)
//
//   seg[s*4 + 0] = expert_id (未使用)
//   seg[s*4 + 1] = m         (decode では 1。1 以外は計算しない)
//   seg[s*4 + 2] = x_off     (入力 base からの f32 要素オフセット)
//   seg[s*4 + 3] = out_off   (出力 base からの f32 要素オフセット)
//
// # 起動形 (`kernels_exp_iq.cu` と同じ)
//
//   blockDim = 128 = 4 warp (SK_ROWS = 4 行、1 行 = 1 warp)
//   grid.x   = ceil(n_out / 4)、grid.y = segment 数、grid.z = 1
//   共有メモリは使わない。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。デコード規則は
// `kernels_moe_batch.cu` / `src/kquant.rs` と同一 (Q4_K / Q5_K / Q5_1 / Q8_0)。
#include <cuda_fp16.h>

#define SK_ROWS 4

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_moe_batch.cu と同一。モジュールが別なので写しを持つ)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sk_f16(const unsigned char* b)
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

// Q4_K の 6bit scale/min 展開 (kquant.rs の get_scale_min_k4 と同一)
__device__ __forceinline__ void sk_scale_min_k4(int j, const unsigned char* q,
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

// 1 segment ぶんの前置き。m != 1 と範囲外は false (barrier を跨がないので早期 return 可)。
struct SkSeg {
    const unsigned char* w;
    const float* x;
    float* out;
};

__device__ __forceinline__ bool sk_seg(const unsigned char* const* wptrs,
                                       const float* xbase, float* obase,
                                       const unsigned int* seg, int nseg,
                                       int n_in, int n_out, SkSeg* s)
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

// 行番号とレーン (同じ warp の全レーンが同じ行を得る)。
__device__ __forceinline__ int sk_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & 31;
    return blockIdx.x * SK_ROWS + (t >> 5);
}

// warp 内の木の和 (形は固定 = 決定的)。
__device__ __forceinline__ float sk_reduce(float acc)
{
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    return acc;
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
//   1 レーン = 32 要素のうち 1 要素 (ブロックごとに lane 番号の要素)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_k_q8_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 34u;
        const float d = sk_f16(blk);
        const float y = (float)(int)(signed char)blk[2 + lane] * d;
        acc += y * s.x[(size_t)b * 32u + (size_t)lane];
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
//   要素 e (0..15) は qs[e] の下位ニブル、e (16..31) は qs[e-16] の上位ニブル、
//   6 ビット目は qh の bit e。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_k_q5_1(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 24u;
        const float d = sk_f16(blk);
        const float m = sk_f16(blk + 2);
        const unsigned int qh = (unsigned int)blk[4]
                              | ((unsigned int)blk[5] << 8)
                              | ((unsigned int)blk[6] << 16)
                              | ((unsigned int)blk[7] << 24);
        const unsigned int hi = ((qh >> lane) & 1u) ? 16u : 0u;
        const unsigned int q = (lane < 16)
            ? ((unsigned int)blk[8 + lane] & 0x0fu)
            : ((unsigned int)blk[8 + (lane - 16)] >> 4);
        const float y = (float)(q + hi) * d + m;
        acc += y * s.x[(size_t)b * 32u + (size_t)lane];
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
//   1 ブロック = 8 サブブロック (32 要素ずつ)。レーン = サブブロック (lane/4) と
//   その 4 分の 1 (lane%4、8 要素)。サブブロック sb の要素は
//   64*(sb/2) + 32*(sb%2) から 32 個で、qs[(sb/2)*32 + j] の下位/上位ニブル。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_k_q4k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int sb = lane >> 2;      // 0..7 (サブブロック)
    const int part = lane & 3;     // 0..3 (サブブロック内の 8 要素)
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 144u;
        const float d = sk_f16(blk);
        const float dmin = sk_f16(blk + 2);
        unsigned int sc, mn;
        sk_scale_min_k4(sb, blk + 4, &sc, &mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const unsigned char* q = blk + 16 + (size_t)(sb >> 1) * 32u;
        const int qoff = part * 8;
        const size_t xoff = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                            + (size_t)(sb & 1) * 32u + (size_t)qoff;
        for (int t = 0; t < 8; ++t) {
            const unsigned int qb = q[qoff + t];
            const unsigned int v = high ? (qb >> 4) : (qb & 0x0fu);
            const float y = d1 * (float)v - m1;
            acc += y * s.x[xoff + (size_t)t];
        }
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
//   Q4_K と同じ配置で、6 ビット目が qh[j] の bit (サブブロック番号)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_k_q5k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 176u;
        const float d = sk_f16(blk);
        const float dmin = sk_f16(blk + 2);
        unsigned int sc, mn;
        sk_scale_min_k4(sb, blk + 4, &sc, &mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const unsigned char* qh = blk + 16;
        const unsigned char* q = blk + 48 + (size_t)(sb >> 1) * 32u;
        const int qoff = part * 8;
        const size_t xoff = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                            + (size_t)(sb & 1) * 32u + (size_t)qoff;
        for (int t = 0; t < 8; ++t) {
            const unsigned int qb = q[qoff + t];
            const unsigned int base = high ? (qb >> 4) : (qb & 0x0fu);
            const unsigned int hi = (qh[qoff + t] & (1u << sb)) ? 16u : 0u;
            const float y = d1 * (float)(base + hi) - m1;
            acc += y * s.x[xoff + (size_t)t];
        }
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// task 282: 行数が多く 1 行が短い形 (UD-Q4_K_XL の mix up = 10,240 行 x 320 要素)
// 用に、**1 warp = 8 行**をまとめて縮約する版。
//
// 1 行 1 warp (`dno_seg_gemv_k_q8_0`) では 1 行 320 要素 = 10 反復しかしないので、
// 行アドレス / segment の読み / warp 縮約 / store の固定費が 340 B に対して重い
// (lab t282a: mix の up を seg_k にしたら旧 GEMV より遅くなった)。8 行を同じ
// warp で回すと固定費が 8 分の 1 になり、8 行 = 2,720 B が連続なので読みも繋がる。
//
// 値: 1 行 1 warp 版と同一 (行は独立。1 行の k 昇順アキュムレータと warp 縮約の
// 木は同じで、8 本のアキュムレータは互いに独立)。
//
// 起動形: blockDim = 128 (4 warp)、1 ブロック = 32 行、grid.x = ceil(n_out / 32)。
// ---------------------------------------------------------------------------
#define SK_R8 8

extern "C" __global__ void dno_mix_gemv_q8_0_r8(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_R8;
    const int nb = n_in >> 5;
    float acc[SK_R8];
    #pragma unroll
    for (int r = 0; r < SK_R8; ++r) acc[r] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float xv = s.x[(size_t)b * 32u + (size_t)lane];
        #pragma unroll
        for (int r = 0; r < SK_R8; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const unsigned char* blk =
                s.w + (size_t)row * (size_t)row_bytes + (size_t)b * 34u;
            const float d = sk_f16(blk);
            const float y = (float)(int)(signed char)blk[2 + lane] * d;
            acc[r] += y * xv;
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_R8; ++r) {
        const float a = sk_reduce(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) s.out[row] = a;
    }
}

// ---------------------------------------------------------------------------
// task 282: 行数が少なく 1 行が長い形 (UD-Q4_K_XL の mix down = 320 行 x 10,240
// 要素) 用。**1 行 = 1 ブロック (256 スレッド)** で k を warp 単位に分け、
// 共有メモリの固定木で縮約する。
//
// lab t282b の `[dec-kprof-avg]` 実測: 旧 v3 形 (lanes=2, rows=8, blockDim=16) は
// down に GPU 全体で 640 スレッドしか使わず、1 行列 3.5 MB に **54 us**
// (65 GB/s) かかっていた (up も同じ 54 us)。
//
// 縮約は「warp 内は sk_reduce の固定木 → 8 warp の部分和を thread 0 が固定順で
// 加算」で決定的。1 行 1 スレッドの逐次和とは丸め順が違う。
// 起動形: blockDim = 256 (8 warp)、grid.x = n_out、shared = 8 * 4 B。
// ---------------------------------------------------------------------------
#define SKROW_THREADS 256
#define SKROW_WARPS 8

extern "C" __global__ void dno_mix_gemv_q8_0_rowblk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    const int row = (int)blockIdx.x;
    if (row >= n_out) return;
    const int t = (int)threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = warp; b < nb; b += SKROW_WARPS) {
        const unsigned char* blk = rowp + (size_t)b * 34u;
        const float d = sk_f16(blk);
        const float y = (float)(int)(signed char)blk[2 + lane] * d;
        acc += y * x[(size_t)b * 32u + (size_t)lane];
    }
    acc = sk_reduce(acc);
    __shared__ float red[SKROW_WARPS];
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < SKROW_WARPS; ++w) s += red[w];
        out[row] = s;
    }
}

// ===========================================================================
// task 412 / 526: seg 経路の gate + up + silu を 1 起動に畳む
//
//   `dno_seg_gemv_gu_*`: gate と up を**同じ warp** で回し、`silu(g) * u` まで
//   書く (起動 3 本 = gate / up / silu -> 1 本)。gate と up は入力 x が同じで
//   行数も同じなので、1 ブロックあたりの x の読みが 1 回で済み、行ごとに
//   独立なアキュムレータが SK_GU_ROWS 本 x 2 (gate/up) になる。
//
// 数値: 行ごとの計算は既存カーネル (`dno_seg_gemv_k_*`) と**同じ順序**
// (ブロック昇順のアキュムレータ 1 本 + `sk_reduce` の固定木) なので、1 行の値は
// ビット一致する。行を跨いだ足し合わせはしない。silu も `dno_dec_silu_mul` と
// 同じ式 `x / (1 + __expf(-x))` を使う。
//
// 起動形: blockDim = 128 (4 warp)。
//   1 ブロック = 4 * SK_GU_ROWS 行。
//   grid = (ceil(n_out / (4 * SK_GU_ROWS)), nseg, 1)。ホスト側 (`cuda.rs` の
//   `SEG_GU_ROWS`) と一致させること。
// ===========================================================================

#define SK_GU_ROWS 2

// Q4_K の 6bit scale/min 展開 (kquant.rs の get_scale_min_k4 と同一)。
// 既存カーネルと同じ規則だが、この節だけ独立に読めるよう写しを持つ。
__device__ __forceinline__ void sk_k4_scalemin(int j, const unsigned char* q,
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

// Q4_K: 1 ブロック (256 要素) のうち lane の 8 要素を acc に足す。
__device__ __forceinline__ void sk_acc_q4k(float& acc, const unsigned char* blk,
                                           const float* xv, int sb, int part, bool high)
{
    const float d = sk_f16(blk);
    const float dmin = sk_f16(blk + 2);
    unsigned int sc, mn;
    sk_k4_scalemin(sb, blk + 4, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const unsigned char* q = blk + 16 + (size_t)(sb >> 1) * 32u;
    const int qoff = part * 8;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = q[qoff + t];
        const unsigned int v = high ? (qb >> 4) : (qb & 0x0fu);
        const float y = d1 * (float)v - m1;
        acc += y * xv[t];
    }
}

// Q5_K: 同上 (6 ビット目が qh の bit)。
__device__ __forceinline__ void sk_acc_q5k(float& acc, const unsigned char* blk,
                                           const float* xv, int sb, int part, bool high)
{
    const float d = sk_f16(blk);
    const float dmin = sk_f16(blk + 2);
    unsigned int sc, mn;
    sk_k4_scalemin(sb, blk + 4, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const unsigned char* qh = blk + 16;
    const unsigned char* q = blk + 48 + (size_t)(sb >> 1) * 32u;
    const int qoff = part * 8;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = q[qoff + t];
        const unsigned int base = high ? (qb >> 4) : (qb & 0x0fu);
        const unsigned int hi = (qh[qoff + t] & (1u << sb)) ? 16u : 0u;
        const float y = d1 * (float)(base + hi) - m1;
        acc += y * xv[t];
    }
}

// Q8_0: 1 ブロック (32 要素) のうち lane の 1 要素を acc に足す。
__device__ __forceinline__ void sk_acc_q8(float& acc, const unsigned char* blk,
                                          float xv, int lane)
{
    const float d = sk_f16(blk);
    const float y = (float)(int)(signed char)blk[2 + lane] * d;
    acc += y * xv;
}

// Q5_1: 同上。
__device__ __forceinline__ void sk_acc_q51(float& acc, const unsigned char* blk,
                                           float xv, int lane)
{
    const float d = sk_f16(blk);
    const float m = sk_f16(blk + 2);
    const unsigned int qh = (unsigned int)blk[4]
                          | ((unsigned int)blk[5] << 8)
                          | ((unsigned int)blk[6] << 16)
                          | ((unsigned int)blk[7] << 24);
    const unsigned int hi = ((qh >> lane) & 1u) ? 16u : 0u;
    const unsigned int q = (lane < 16)
        ? ((unsigned int)blk[8 + lane] & 0x0fu)
        : ((unsigned int)blk[8 + (lane - 16)] >> 4);
    const float y = (float)(q + hi) * d + m;
    acc += y * xv;
}

// ---------------------------------------------------------------------------
// gate + up + silu を 1 起動で (K 型 2 種)
//
//   gwptrs[s] / uwptrs[s] = segment s の gate / up の重み先頭
//   abase = act の先頭 (segment ごとに out_off を足す)
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_gu_q4k(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        float xv[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) xv[t] = x[xo + (size_t)t];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 144u;
            sk_acc_q4k(ag[r], gw + bo, xv, sb, part, high);
            sk_acc_q4k(au[r], uw + bo, xv, sb, part, high);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

extern "C" __global__ void dno_seg_gemv_gu_q5k(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        float xv[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) xv[t] = x[xo + (size_t)t];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 176u;
            sk_acc_q5k(ag[r], gw + bo, xv, sb, part, high);
            sk_acc_q5k(au[r], uw + bo, xv, sb, part, high);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

// ---------------------------------------------------------------------------
// gate + up + silu を 1 起動で (32 要素ブロックの 2 種 = 1 ブロック 1 要素/lane)
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_seg_gemv_gu_q8_0(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int nb = n_in >> 5;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const float xv = x[(size_t)b * 32u + (size_t)lane];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 34u;
            sk_acc_q8(ag[r], gw + bo, xv, lane);
            sk_acc_q8(au[r], uw + bo, xv, lane);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

extern "C" __global__ void dno_seg_gemv_gu_q5_1(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int nb = n_in >> 5;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const float xv = x[(size_t)b * 32u + (size_t)lane];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 24u;
            sk_acc_q51(ag[r], gw + bo, xv, lane);
            sk_acc_q51(au[r], uw + bo, xv, lane);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}


// ===========================================================================
// task 534: K 型の 8 バイトロード版 (gu8 = gate+up+silu 融合版、k8 = 1 行版)
//
// なぜ: `sk_acc_q4k` / `sk_acc_q5k` は `q[qoff + t]` を 8 回読むので
// **1 命令あたり 1 バイト**しか要求しない。Q4_K / Q5_K は lane の 8 要素が
// 連続 8 バイト (`q[qoff..qoff+8]`、`qoff = part * 8`、`row_bytes` は
// 144 / 176 の倍数) に並ぶので、64 ビットロード 1 回 + シフトで同じバイト列を
// 取れる (task 412 (3/n) と同じ手法。lab t412c で gate 463 -> 490 GB/s)。
//
// 値: 読むバイト列と足す順序 (t 昇順) が同じなので、1 行の値は
// `dno_seg_gemv_k_*` / `dno_seg_gemv_gu_*` とビット一致する。
//
// 起動形は融合版 (`dno_seg_gemv_gu_*`) / 1 行版 (`dno_seg_gemv_k_*`) と同一。
// Q8_0 / Q5_1 は 1 要素 1 バイトが最小なのでこの版は作らない (= 従来のまま)。
// ===========================================================================

// Q4_K の 6bit scale/min 展開 (kquant.rs の get_scale_min_k4 と同一)。
// 既存 2 つの写しと同じ規則。この節だけ独立に読めるよう 3 つ目の写しを持つ。
__device__ __forceinline__ void sk8_scale_min(int j, const unsigned char* q,
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

// Q4_K: 1 ブロック (256 要素) のうち lane の 8 要素を acc に足す (8 バイトロード)。
__device__ __forceinline__ void sk_acc8_q4k(float& acc, const unsigned char* blk,
                                            const float* xv, int sb, int part, bool high)
{
    const float d = sk_f16(blk);
    const float dmin = sk_f16(blk + 2);
    unsigned int sc, mn;
    sk8_scale_min(sb, blk + 4, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const unsigned char* q = blk + 16 + (size_t)(sb >> 1) * 32u;
    const int qoff = part * 8;
    const unsigned long long w8 = *(const unsigned long long*)(q + qoff);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = (unsigned int)((w8 >> (unsigned)(8 * t)) & 0xffull);
        const unsigned int v = high ? (qb >> 4) : (qb & 0x0fu);
        const float y = d1 * (float)v - m1;
        acc += y * xv[t];
    }
}

// Q5_K: 同上 (6 ビット目が qh の bit。qh も 8 バイトロード)。
__device__ __forceinline__ void sk_acc8_q5k(float& acc, const unsigned char* blk,
                                            const float* xv, int sb, int part, bool high)
{
    const float d = sk_f16(blk);
    const float dmin = sk_f16(blk + 2);
    unsigned int sc, mn;
    sk8_scale_min(sb, blk + 4, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const unsigned char* qh = blk + 16;
    const unsigned char* q = blk + 48 + (size_t)(sb >> 1) * 32u;
    const int qoff = part * 8;
    const unsigned long long h8 = *(const unsigned long long*)(qh + qoff);
    const unsigned long long w8 = *(const unsigned long long*)(q + qoff);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = (unsigned int)((w8 >> (unsigned)(8 * t)) & 0xffull);
        const unsigned int hb = (unsigned int)((h8 >> (unsigned)(8 * t)) & 0xffull);
        const unsigned int base = high ? (qb >> 4) : (qb & 0x0fu);
        const unsigned int hi = (hb & (1u << sb)) ? 16u : 0u;
        const float y = d1 * (float)(base + hi) - m1;
        acc += y * xv[t];
    }
}

// gate + up + silu を 1 起動 + 8 バイトロードで (Q4_K / Q5_K)。
// 形は `dno_seg_gemv_gu_q4k` / `_q5k` と同じ (1 warp = SK_GU_ROWS 行 x gate/up)。
extern "C" __global__ void dno_seg_gemv_gu8_q4k(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        float xv[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) xv[t] = x[xo + (size_t)t];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 144u;
            sk_acc8_q4k(ag[r], gw + bo, xv, sb, part, high);
            sk_acc8_q4k(au[r], uw + bo, xv, sb, part, high);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

extern "C" __global__ void dno_seg_gemv_gu8_q5k(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b = 0; b < nb; ++b) {
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        float xv[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) xv[t] = x[xo + (size_t)t];
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 176u;
            sk_acc8_q5k(ag[r], gw + bo, xv, sb, part, high);
            sk_acc8_q5k(au[r], uw + bo, xv, sb, part, high);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

// 1 行 = 1 warp の 8 バイトロード版 (down 用)。引数と起動形は
// `dno_seg_gemv_k_*` と完全に同一 (ホストは関数を差し替えるだけ)。
extern "C" __global__ void dno_seg_gemv_k8_q4k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 144u;
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        const float* xv = s.x + xo;
        sk_acc8_q4k(acc, blk, xv, sb, part, high);
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

extern "C" __global__ void dno_seg_gemv_k8_q5k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int sb = lane >> 2;
    const int part = lane & 3;
    const bool high = (sb & 1) != 0;
    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = rowp + (size_t)b * 176u;
        const size_t xo = (size_t)b * 256u + (size_t)(sb >> 1) * 64u
                          + (size_t)(sb & 1) * 32u + (size_t)(part * 8);
        const float* xv = s.x + xo;
        sk_acc8_q5k(acc, blk, xv, sb, part, high);
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ===========================================================================
// task 534: **Q5_1 の 4 バイトロード版** (down 用)
//
// なぜ: UD-Q4_K_XL の expert は gate/up が Q4_K、**down が Q5_1** (lab t534b の
// `[pf-ktc] gate/up=Q4K down=Q5_1 k=2560/640`)。Q5_1 の 1 ブロックは 24 B
// (d 2 + m 2 + qh 4 + qs 16) で 32 要素。既存 `dno_seg_gemv_k_q5_1` は
// **1 要素 = 1 バイトロード**で、しかも lane 0-15 と 16-31 が同じ 16 バイトを
// 読む (要求バイトの半分が無駄)。lab t534b の down は 348 GB/s で
// gate (Q4_K, 476 GB/s) より遅い。
//
// 何をするか: **1 warp = 8 ブロック = 256 要素**にし、lane ごとに qs の
// 連続 4 バイト (= 8 要素) を 1 回の 32 ビットロードで取る。
//   lane = 4 * (ブロック内の 4 バイト目 j) + ... ではなく
//   j = lane & 3 (qs の 4 バイト目)、bg = lane >> 2 (反復内のブロック番号 0..7)。
//   lane は qs[j*4 .. j*4+4] を読み、要素 j*4+t (下位ニブル) と
//   要素 j*4+16+t (上位ニブル) を足す。5 ビット目は qh の同じ要素番号の bit。
//   x の添字は `b * 32 + 要素番号` (既存カーネルと同じ並び)。
//
// 値: 1 行の和の順序が変わる (lane 内で 4 + 4 要素、縮約は同じ `sk_reduce`) ので
// **ビット一致しない**。起動形は同じで、形が固定なので値は決定的
// (丸め順の変更はタスクの規則で許容。`DENNO_SEG_DN8=0` で戻せる)。
// ===========================================================================

#define SK_V4_BLKS 8

extern "C" __global__ void dno_seg_gemv_v4_q5_1(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    SkSeg s;
    if (!sk_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;
    int lane;
    const int row = sk_row_of(&lane);
    if (row >= n_out) return;

    const unsigned char* rowp = s.w + (size_t)row * (size_t)row_bytes;
    const int nb = n_in >> 5;          // 32 要素ブロックの数
    const int j = lane & 3;            // qs の 4 バイト目 (要素 j*4 .. j*4+3)
    const int bg = lane >> 2;          // 反復の中のブロック番号 (0..SK_V4_BLKS-1)
    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += SK_V4_BLKS) {
        const int b = b0 + bg;
        if (b >= nb) continue;
        const unsigned char* blk = rowp + (size_t)b * 24u;
        const float d = sk_f16(blk);
        const float m = sk_f16(blk + 2);
        const unsigned int qh = (unsigned int)blk[4]
                              | ((unsigned int)blk[5] << 8)
                              | ((unsigned int)blk[6] << 16)
                              | ((unsigned int)blk[7] << 24);
        const unsigned int w4 = *(const unsigned int*)(blk + 8 + (size_t)j * 4u);
        const size_t xo = (size_t)b * 32u + (size_t)j * 4u;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const unsigned int qb = (w4 >> (unsigned)(8 * t)) & 0xffu;
            const unsigned int e_lo = (unsigned int)(j * 4 + t);
            const unsigned int e_hi = e_lo + 16u;
            const unsigned int lo = qb & 0x0fu;
            const unsigned int hi = qb >> 4;
            const float y0 = (float)(lo + (((qh >> e_lo) & 1u) ? 16u : 0u)) * d + m;
            const float y1 = (float)(hi + (((qh >> e_hi) & 1u) ? 16u : 0u)) * d + m;
            acc += y0 * s.x[xo + (size_t)t];
            acc += y1 * s.x[xo + 16u + (size_t)t];
        }
    }
    acc = sk_reduce(acc);
    if (lane == 0) s.out[row] = acc;
}

// ===========================================================================
// task 562: Q4_K の **16 バイトロード版** (gate + up + silu 融合、K 型の gate/up 用)
//
// なぜ: `sk_acc8_q4k` (task 534 の gu8) は lane が `q[qoff..qoff+8]` の 8 バイトを
// 読むが、**同じ 8 バイトを隣の lane (上位ニブル側) がもう一度読む** ので、
// 1 ブロックの要求が 256 B / 有効 128 B = 2 倍になる。lab t534d の gate+up は
// 476 GB/s で、同じ段の down (Q5_1 の v4) の 555-615 GB/s より低い。
//
// 何をするか: **1 warp = 4 ブロック、1 ブロック = 8 lane**。lane は qs の連続
// 16 バイト (= 32 要素) を 8 バイトロード 2 回で取り、下位ニブル 16 要素と
// 上位ニブル 16 要素を足す。要求バイト = 有効バイト (1 ブロック 128 B)。
//   lb = lane AND 7 (ブロック内の 16 バイト位置)、bi = lane >> 3 (反復のブロック 0..3)
//   j  = lb >> 1 (qs の 32 バイト塊 = サブブロック対 0..3)、h = lb AND 1 (前後半)
//   要素: 下位 = 64j + 16h + t、上位 = 64j + 32 + 16h + t (t = 0..15)
//   スケール: 下位 = サブブロック 2j、上位 = 2j+1
//   反復は `b0 += 4` で、ブロック数 10 (n_in=2560) なら 3 反復。
//
// 値: 1 lane が 32 要素を持つので lane 内の和の順序が既存カーネルと変わる
// (丸め順の変更はタスクの規則で許容)。起動形は融合版 (gu / gu8) と同一で、
// 形が固定なので値は決定的。`DENNO_SEG_GU16=0` で従来へ戻せる。
// ===========================================================================
#define SK_G16_BLKS 4
#define SK_G16_LANES 8

// Q4_K: 1 ブロックのうち lane の 32 要素 (下位 16 + 上位 16) を acc に足す。
// xv は「下位の先頭 16 要素」を指し、上位は xv[32..48] (要素 +32)。
__device__ __forceinline__ void sk_acc16_q4k(float& acc, const unsigned char* blk,
                                             const float* xv, int j, int h)
{
    const float d = sk_f16(blk);
    const float dmin = sk_f16(blk + 2);
    unsigned int sc, mn;
    sk8_scale_min(2 * j, blk + 4, &sc, &mn);
    const float d0 = d * (float)sc;
    const float m0 = dmin * (float)mn;
    sk8_scale_min(2 * j + 1, blk + 4, &sc, &mn);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)mn;
    const unsigned char* q = blk + 16 + (size_t)j * 32u + (size_t)h * 16u;
    const unsigned long long w0 = *(const unsigned long long*)q;
    const unsigned long long w1 = *(const unsigned long long*)(q + 8);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = (unsigned int)((w0 >> (unsigned)(8 * t)) & 0xffull);
        acc += (d0 * (float)(qb & 0x0fu) - m0) * xv[t];
        acc += (d1 * (float)(qb >> 4) - m1) * xv[32 + t];
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned int qb = (unsigned int)((w1 >> (unsigned)(8 * t)) & 0xffull);
        acc += (d0 * (float)(qb & 0x0fu) - m0) * xv[8 + t];
        acc += (d1 * (float)(qb >> 4) - m1) * xv[40 + t];
    }
}

// gate + up + silu を 1 起動 + 16 バイトロードで (Q4_K)。
// 引数と起動形は `dno_seg_gemv_gu_q4k` / `dno_seg_gemv_gu8_q4k` と同一。
extern "C" __global__ void dno_seg_gemv_gu16_q4k(
    const unsigned char* const* __restrict__ gwptrs,
    const unsigned char* const* __restrict__ uwptrs,
    const float* __restrict__ xbase,
    float* __restrict__ abase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return;
    const unsigned int* sg = seg + (size_t)si * 4u;
    if ((int)sg[1] != 1) return;
    const unsigned char* gw = gwptrs[si];
    const unsigned char* uw = uwptrs[si];
    const float* x = xbase + (size_t)sg[2];
    float* out = abase + (size_t)sg[3];
    const int lane = threadIdx.x & 31u;
    const int warps = (int)(blockDim.x >> 5);
    const int row0 = (blockIdx.x * warps + (int)(threadIdx.x >> 5)) * SK_GU_ROWS;
    if (row0 >= n_out) return;

    const int lb = lane & (SK_G16_LANES - 1);
    const int bi = lane >> 3;
    const int j = lb >> 1;
    const int h = lb & 1;
    const int nb = n_in >> 8;
    float ag[SK_GU_ROWS];
    float au[SK_GU_ROWS];
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int b0 = 0; b0 < nb; b0 += SK_G16_BLKS) {
        const int b = b0 + bi;
        if (b >= nb) continue;
        const size_t xo = (size_t)b * 256u + (size_t)j * 64u + (size_t)h * 16u;
        float xv[48];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            xv[t] = x[xo + (size_t)t];
            xv[32 + t] = x[xo + 32u + (size_t)t];
        }
        #pragma unroll
        for (int r = 0; r < SK_GU_ROWS; ++r) {
            const int row = row0 + r;
            if (row >= n_out) continue;
            const size_t bo = (size_t)row * (size_t)row_bytes + (size_t)b * 144u;
            sk_acc16_q4k(ag[r], gw + bo, xv, j, h);
            sk_acc16_q4k(au[r], uw + bo, xv, j, h);
        }
    }
    #pragma unroll
    for (int r = 0; r < SK_GU_ROWS; ++r) {
        const float g = sk_reduce(ag[r]);
        const float u = sk_reduce(au[r]);
        const int row = row0 + r;
        if (lane == 0 && row < n_out) out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}
