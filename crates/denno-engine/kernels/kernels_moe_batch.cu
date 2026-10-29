// denno CUDA 層一括 expert GEMV (P43) — decode の expert FFN を 1 層 3 起動にまとめる。
//
// PTX へのコンパイル (メインセッション / オペレーターの担当。このコンテナに nvcc は無い):
//
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_batch.cu -o kernels_moe_batch.ptx
//
// **-fmad=false は必須** (kernels_moe.cu と同じ)。数値契約は per-expert 版
// (`kernels_moe.cu` の dno_gemv_*_w) との**ビット一致**で、比較は
// `DENNO_EXPERT_GPU_CHECK=1` が層ごとに argmax / max abs diff を出す。
//
// # 何が違うか
//
// 既存カーネルは「1 本の重み `w` の n_out 行」を読む。こちらは segment 表を引き、
// **segment ごとに別の重みの先頭** (`wptrs[s]`) と入力/出力のオフセットを選ぶ。
// 1 行の累積は既存と同じく「1 スレッドが i の昇順に f32 で加算」なので、
// 同じ expert・同じ入力に対して 10 回起動版と 1 回起動版の出力はビット一致する。
//
// 10 expert x 3 行列を 1 本ずつ起動すると 1 トークン 1,440 起動 + 1,440 同期に
// なる (実測 382 ms/token の主因)。segment をまとめると
// 「gate 1 + up 1 + down 1 = 3 起動 / 層あたり 2 同期」になる。
//
// # segment 表 (`const unsigned int* seg`、1 segment = 4 語)
//
//   seg[s*4 + 0] = expert_id  (カーネルでは未使用。ログ・突き合わせ用)
//   seg[s*4 + 1] = m          (この segment のトークン数。decode では 1)
//   seg[s*4 + 2] = x_off      (入力 base からの f32 要素オフセット)
//   seg[s*4 + 3] = out_off    (出力 base からの f32 要素オフセット)
//
// gate / up は全 expert が同じ入力ベクトルを読むので x_off = 0、out_off = s * n_out。
// down は expert ごとに違う act を読むので x_off = s * n_in、out_off = s * n_out。
//
// # ブロック割当 (kernels_moe.cu と同じ)
//
//   blockDim = 256 = 4 行 x 64 レーン。1 行を 64 レーンで共有メモリへコアレスに
//   読み、lane 0 が i の昇順に累積する。
//   grid.x = 行タイル (ceil(n_out / 4))、grid.y = segment、grid.z = トークン。
//   動的共有メモリは 4 * row_bytes (Q4_K k=2560 で 5,760 B、Q5_1 k=640 で 1,920 B)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。デコード規則は
// 本リポジトリの `src/kquant.rs` / `kernels_moe.cu` と同一 (Q8_0 / Q5_1 / Q4_K / Q5_K)。

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_moe.cu と同一。PTX はファイル単位で独立したモジュールに
// なるため、別 .cu のヘルパは使えず写しが必要)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_mb_f16(const unsigned char* b)
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
__device__ __forceinline__ void dno_mb_scale_min_k4(int j, const unsigned char* q,
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

#define MB_ROWS 4
#define MB_LANES 64

// 1 segment ぶんの前置きを読む (segment 番号・トークン・重み/入出力の先頭)。
// 範囲外は false を返し、呼び出し側は **barrier を跨がずに** return する。
struct MbSeg {
    const unsigned char* w;
    const float* x;
    float* out;
    int m;
};

__device__ __forceinline__ bool mb_seg(const unsigned char* const* wptrs,
                                       const float* xbase, float* obase,
                                       const unsigned int* seg, int nseg,
                                       int n_in, int n_out, MbSeg* s)
{
    const int si = (int)blockIdx.y;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    const int m = (int)sg[1];
    const int t = (int)blockIdx.z;
    if (t >= m) return false;
    s->w = wptrs[si];
    s->x = xbase + (size_t)sg[2] + (size_t)t * (size_t)n_in;
    s->out = obase + (size_t)sg[3] + (size_t)t * (size_t)n_out;
    s->m = m;
    return true;
}

// 行番号とレーン (kernels_moe.cu の moe_row_of と同一)。
__device__ __forceinline__ int mb_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (MB_LANES - 1);
    return blockIdx.x * MB_ROWS + (t / MB_LANES);
}

__device__ __forceinline__ void mb_stage(const unsigned char* __restrict__ src,
                                         unsigned char* dst, int row_bytes, int lane)
{
    for (int i = lane; i < row_bytes; i += MB_LANES) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_gemv_seg_q8_0(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    MbSeg s;
    if (!mb_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;

    int lane;
    const int row = mb_row_of(&lane);
    const bool valid = (row < n_out);
    unsigned char* dst = smem + (size_t)(threadIdx.x / MB_LANES) * (size_t)row_bytes;
    if (valid) mb_stage(s.w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 34u;
        const float d = dno_mb_f16(blk);
        const float* xb = s.x + (size_t)b * 32u;
        for (int j = 0; j < 32; ++j) {
            const float y = (float)(int)(signed char)blk[2 + j] * d;
            acc += y * xb[j];
        }
    }
    s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_gemv_seg_q5_1(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    MbSeg s;
    if (!mb_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;

    int lane;
    const int row = mb_row_of(&lane);
    const bool valid = (row < n_out);
    unsigned char* dst = smem + (size_t)(threadIdx.x / MB_LANES) * (size_t)row_bytes;
    if (valid) mb_stage(s.w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 24u;
        const float d = dno_mb_f16(blk);
        const float m = dno_mb_f16(blk + 2);
        const unsigned int qh = (unsigned int)blk[4]
                              | ((unsigned int)blk[5] << 8)
                              | ((unsigned int)blk[6] << 16)
                              | ((unsigned int)blk[7] << 24);
        const float* xb = s.x + (size_t)b * 32u;
        // 下位ニブル (要素 0..15) を先に、上位ニブル (要素 16..31) を後に足す。
        for (int j = 0; j < 16; ++j) {
            const int xh0 = (int)(((qh >> j) << 4) & 0x10u);
            const int q0 = (int)(blk[8 + j] & 0x0fu) | xh0;
            const float y = (float)q0 * d + m;
            acc += y * xb[j];
        }
        for (int j = 0; j < 16; ++j) {
            const int xh1 = (int)((qh >> (j + 12)) & 0x10u);
            const int q1 = (int)(blk[8 + j] >> 4) | xh1;
            const float y = (float)q1 * d + m;
            acc += y * xb[16 + j];
        }
    }
    s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_gemv_seg_q4k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    MbSeg s;
    if (!mb_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;

    int lane;
    const int row = mb_row_of(&lane);
    const bool valid = (row < n_out);
    unsigned char* dst = smem + (size_t)(threadIdx.x / MB_LANES) * (size_t)row_bytes;
    if (valid) mb_stage(s.w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 8;   // 256 要素 / ブロック
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 144u;
        const float d    = dno_mb_f16(blk);
        const float dmin = dno_mb_f16(blk + 2);
        const float* xb  = s.x + (size_t)b * 256u;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const unsigned char* q = blk + 16 + (j >> 1);
            unsigned int sc, mn;
            dno_mb_scale_min_k4(is, blk + 4, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)mn;
            dno_mb_scale_min_k4(is + 1, blk + 4, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)mn;
            for (int l = 0; l < 32; ++l) {
                const float y = d1 * (float)(int)(q[l] & 0x0fu) - m1;
                acc += y * xb[j + l];
            }
            for (int l = 0; l < 32; ++l) {
                const float y = d2 * (float)(int)(q[l] >> 4) - m2;
                acc += y * xb[j + 32 + l];
            }
            is += 2;
        }
    }
    s.out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_gemv_seg_q5k(
    const unsigned char* const* __restrict__ wptrs,
    const float* __restrict__ xbase,
    float* __restrict__ obase,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes)
{
    extern __shared__ unsigned char smem[];
    MbSeg s;
    if (!mb_seg(wptrs, xbase, obase, seg, nseg, n_in, n_out, &s)) return;

    int lane;
    const int row = mb_row_of(&lane);
    const bool valid = (row < n_out);
    unsigned char* dst = smem + (size_t)(threadIdx.x / MB_LANES) * (size_t)row_bytes;
    if (valid) mb_stage(s.w + (size_t)row * (size_t)row_bytes, dst, row_bytes, lane);
    __syncthreads();
    if (!valid || lane != 0) return;

    const int nb = n_in >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = dst + (size_t)b * 176u;
        const float d    = dno_mb_f16(blk);
        const float dmin = dno_mb_f16(blk + 2);
        const unsigned char* qh = blk + 16;
        const float* xb = s.x + (size_t)b * 256u;
        int is = 0;
        unsigned int u1 = 1u, u2 = 2u;
        for (int j = 0; j < 256; j += 64) {
            const unsigned char* ql = blk + 48 + (j >> 1);
            unsigned int sc, mn;
            dno_mb_scale_min_k4(is, blk + 4, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)mn;
            dno_mb_scale_min_k4(is + 1, blk + 4, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)mn;
            for (int l = 0; l < 32; ++l) {
                const int hi = (qh[l] & u1) ? 16 : 0;
                const float y = d1 * (float)((int)(ql[l] & 0x0fu) + hi) - m1;
                acc += y * xb[j + l];
            }
            for (int l = 0; l < 32; ++l) {
                const int hi = (qh[l] & u2) ? 16 : 0;
                const float y = d2 * (float)((int)(ql[l] >> 4) + hi) - m2;
                acc += y * xb[j + 32 + l];
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    s.out[row] = acc;
}
