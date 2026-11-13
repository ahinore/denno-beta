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

#define SK_ROWS 4

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_moe_batch.cu と同一。モジュールが別なので写しを持つ)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sk_f16(const unsigned char* b)
{
    unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp  = (h >> 10) & 0x1fu;
    unsigned int man  = h & 0x3ffu;
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
