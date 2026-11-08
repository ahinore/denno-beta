// denno CUDA MoE GEMV kernels (v4) — decode の量子 GEMV を **レーン数可変** で分担する版。
//
// task 177。v3 (`kernels_moe_v3.cu`) との違いは 2 点だけ:
//   1. 1 行を担うレーン数 `lanes` を 2 / 4 / 8 / 16 / 32 のどれでも渡せる
//      (v3 は 2 と 4 だけ。行番号は `threadIdx.x / lanes`、レーンは
//      `threadIdx.x & (lanes-1)`)。
//   2. レーンの合計を `for (off = 1; off < lanes; off <<= 1)` の
//      `__shfl_down_sync` で行う (v3 は 4 レーン固定で 2 段)。
//      **lanes=4 では v3 と同じ順 (offset 1 → 2) になる**。
// データの読み方は v3 と同一 (レーン l が k を lanes 等分した連続区間を持つ)。
// dequant の演算列も v3 と同一のものを使う (`-fmad=false`)。
//
// 動機 (lab t177a の実測): v3 は lanes=4 固定なのでスレッド数が n_out*4 で
// 頭打ちになり、out 投影 (n=2,560 / k=6,144) は 320 warp = 5090 の 170 SM に
// 1.9 warp/SM しか立たず 328 GB/s しか出ていなかった。lanes を上げると
// スレッド数が n_out*lanes になるので、同じ形でも同時に走る読み出しが増える。
//
// PTX へのコンパイル (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_v4.cu -o kernels_moe_v4.ptx
// まとめて作るなら `kernels/build_ptx.sh moe_v4`。
//
// 数値契約: v3 と同じく **v2 とはビット一致しない** (k の分割と合計順が違う)。
// ホストは、形が合わない k・lanes のとき v3 へ落ちる (fail closed)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

__device__ __forceinline__ float dno_f16_to_f32_v4(const unsigned char* b)
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

__device__ __forceinline__ void dno_scale_min_k4_v4(int j, const unsigned char* q,
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

#define MOE_V4_ROWS_DEFAULT 8  // main の実測で rows 8 が最良 (decode 122.6 ms/token)

// 行番号とレーン番号。lanes レーンが同一行を担当し、スレッド番号が連続する
// (lane = threadIdx.x & (lanes-1)、行内 = threadIdx.x / lanes)。
// lanes は 2/4/8/16/32 のどれでもよい (2 の冪)。割り算はここで 1 回だけ。
// lanes は 32 を割り切るので、1 warp の中に行境界が来ても行は揃う。
__device__ __forceinline__ int moe_v4_row_of(int rows_per_block, int lanes, int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (lanes - 1);
    return blockIdx.x * rows_per_block + (t / lanes);
}

// シャッフルのマスク。ブロックが 1 warp 未満 (rows=8 の 2 レーン版 = 16 スレッド)
// のときは実在しないスレッドをマスクに含めない。
__device__ __forceinline__ unsigned int moe_v4_mask()
{
    const unsigned int n = blockDim.x;
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

// lanes レーンの合計 (offset 1, 2, 4, ... の log2(lanes) 段)。
// 加算順は lanes=4 のとき v3 と同じ (offset 1 → 2)。lane 0 の値だけが使われる。
// マスクは warp 全体 (または実在する全スレッド) なので、呼び出し時に揃っていること。
// lanes < 32 のとき、行の先頭でないレーンの値は隣の行の値を拾うが、
// 書き出すのは lane 0 だけなので結果には影響しない。
__device__ __forceinline__ float moe_v4_reduce(float v, int lanes, unsigned int mask)
{
    for (int off = 1; off < lanes; off <<= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

// x[n]..x[n+31] をレジスタへ先読み (完全展開でレジスタに載る)
__device__ __forceinline__ void moe_v4_prefetch32(const float* __restrict__ xb, float* xv)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(xb + 4 * i);
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q8_0_v4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_v4_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_v4_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb / lanes; // 1 レーンぶんのブロック数 (nb は lanes の倍数)
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 34u;
            const float d = dno_f16_to_f32_v4(blk);
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_v4_prefetch32(xb, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float y = (float)(int)(signed char)blk[2 + j] * d;
                acc += y * xv[j];
            }
        }
    }
    acc = moe_v4_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5_1_v4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_v4_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_v4_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb / lanes;
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 24u;
            const float d = dno_f16_to_f32_v4(blk);
            const float m = dno_f16_to_f32_v4(blk + 2);
            const unsigned int qh = (unsigned int)blk[4]
                                  | ((unsigned int)blk[5] << 8)
                                  | ((unsigned int)blk[6] << 16)
                                  | ((unsigned int)blk[7] << 24);
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_v4_prefetch32(xb, xv);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int xh0 = (int)(((qh >> j) << 4) & 0x10u);
                const int q0 = (int)(blk[8 + j] & 0x0fu) | xh0;
                const float y = (float)q0 * d + m;
                acc += y * xv[j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int xh1 = (int)((qh >> (j + 12)) & 0x10u);
                const int q1 = (int)(blk[8 + j] >> 4) | xh1;
                const float y = (float)q1 * d + m;
                acc += y * xv[16 + j];
            }
        }
    }
    acc = moe_v4_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
//
// レーン境界 k/lanes は 32 要素 (サブブロック) の倍数だが 256 要素 (スーパー
// ブロック) の倍数とは限らない (k=2560 / lanes=4 では 640 = 2.5 ブロック) ので、
// レーンの担当区間に重なるスーパーブロックを回り、その中でレーン内のサブブロック
// だけを計算する。レーン外のサブブロックは読まない (x も q も触らない)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q4k_v4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_v4_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_v4_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int seg = n_in / lanes; // 1 レーンぶんの要素数 (32 の倍数)
        const int lo = lane * seg;
        const int hi = lo + seg;
        const int b0 = lo >> 8;
        const int b1 = (hi - 1) >> 8;
        for (int b = b0; b <= b1; ++b) {
            const int bstart = b << 8;
            const unsigned char* blk = rowp + (size_t)b * 144u;
            const float d    = dno_f16_to_f32_v4(blk);
            const float dmin = dno_f16_to_f32_v4(blk + 2);
            // このスーパーブロックで自分が担当するサブブロック [s0, s1)
            const int s0 = (lo > bstart) ? ((lo - bstart) >> 5) : 0;
            const int s1 = (hi < bstart + 256) ? ((hi - bstart + 31) >> 5) : 8;
            for (int s = s0; s < s1; ++s) {
                const int e = s << 5; // スーパーブロック内の要素オフセット
                // qs は 64 要素あたり 32 B。e<32 の 64 要素区画では前半 32 要素が
                // 下位ニブル、後半 32 要素が上位ニブル。
                const unsigned char* q = blk + 16 + ((e >> 6) << 5);
                const unsigned int sh = (e & 32) ? 4u : 0u;
                unsigned int sc, mn;
                dno_scale_min_k4_v4(s, blk + 4, &sc, &mn);
                const float ds = d * (float)sc;
                const float ms = dmin * (float)mn;
                const float* xb = x + (size_t)bstart + (size_t)e;
                float xv[32];
                moe_v4_prefetch32(xb, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const float y = ds * (float)(int)((q[l] >> sh) & 0x0fu) - ms;
                    acc += y * xv[l];
                }
            }
        }
    }
    acc = moe_v4_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
//
// 5 ビット目の qh は「サブブロック番号 s のビット」を要素 l ごとに見る
// (v2 の u1/u2 の 2 ビットずらしと同じもの)。上位 4 ビットは `+ 16` で足す。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5k_v4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_v4_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_v4_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int seg = n_in / lanes;
        const int lo = lane * seg;
        const int hi = lo + seg;
        const int b0 = lo >> 8;
        const int b1 = (hi - 1) >> 8;
        for (int b = b0; b <= b1; ++b) {
            const int bstart = b << 8;
            const unsigned char* blk = rowp + (size_t)b * 176u;
            const float d    = dno_f16_to_f32_v4(blk);
            const float dmin = dno_f16_to_f32_v4(blk + 2);
            const unsigned char* qh = blk + 16;
            const int s0 = (lo > bstart) ? ((lo - bstart) >> 5) : 0;
            const int s1 = (hi < bstart + 256) ? ((hi - bstart + 31) >> 5) : 8;
            for (int s = s0; s < s1; ++s) {
                const int e = s << 5;
                const unsigned char* ql = blk + 48 + ((e >> 6) << 5);
                const unsigned int sh = (e & 32) ? 4u : 0u;
                unsigned int sc, mn;
                dno_scale_min_k4_v4(s, blk + 4, &sc, &mn);
                const float ds = d * (float)sc;
                const float ms = dmin * (float)mn;
                const float* xb = x + (size_t)bstart + (size_t)e;
                float xv[32];
                moe_v4_prefetch32(xb, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int hi_bit = (int)((qh[l] >> s) & 1u) << 4;
                    const float y = ds * (float)((int)((ql[l] >> sh) & 0x0fu) + hi_bit) - ms;
                    acc += y * xv[l];
                }
            }
        }
    }
    acc = moe_v4_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}
