// denno CUDA MoE GEMV kernels (v3) — decode の量子 GEMV を **4 レーンで分担** する版。
//
// PTX へのコンパイル (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_v3.cu -o kernels_moe_v3.ptx
// まとめて作るなら `kernels/build_ptx.sh moe_v3`。
//
// **数値契約**: v2 (`kernels_moe_v2.cu`) とは **ビット一致しない**。
// v2 は 1 行の累積を 1 スレッドが i 昇順に f32 で行うが、v3 は k を 4 等分した
// 連続区間を 4 レーンが各自 f32 で累積し、最後に 2 段のシャッフルで足す。
// 加算順が変わるので**意図的に**ビットが変わる (prefill の GPU バッチ GEMM と
// 同じ扱い = 契約の外)。dequant の演算列は v2 と同一 (`-fmad=false`) なので、
// CPU の逐次実装との差は丸め順だけ (相対 1e-5 以内。テスト
// `cuda_gemv_v3_matches_cpu`)。
//
// レイアウト: warp (32 lane) = 8 行 × 4 レーン。block 256 スレッド = 64 行。
//   1 行を 4 レーンが担当し、レーン l は k を 4 等分した連続区間
//   [l*k/4, (l+1)*k/4) を持つ。区間長 k/4 は 32 の倍数 (k が 128 の倍数) で
//   あることを前提にする。満たさない k はホスト側 (`moe_gemv_v3_k_ok`) が
//   v2 へ落とすので、このカーネルには来ない。
//
// v2 との違い (数値以外):
//   - v2 は行の生バイトを 64 レーンで共有メモリへステージしてから 1 レーンが
//     消費していた。v3 は 4 レーンが自分の区間を global から直接読む
//     (1 行あたりの読み出しバイト数は同じ。64 行/ブロックでは行の生バイトを
//     共有メモリに置くと k=2560 の Q4_K で 90 KiB になり 48 KiB の既定を超える
//     ため、ステージしない)。
//   - 無効行 (row >= n_out) のスレッドもシャッフルには参加する (マスクが
//     warp 全体なので、早期 return すると __shfl_down_sync が揃わない)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

__device__ __forceinline__ float dno_f16_to_f32_v3(const unsigned char* b)
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

__device__ __forceinline__ void dno_scale_min_k4_v3(int j, const unsigned char* q,
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

#define MOE_V3_ROWS_PER_BLOCK 64
#define MOE_V3_LANES 4

// 行番号とレーン番号。4 レーンは同一行を担当し、スレッド番号が連続する
// (warp 内の 0..3, 4..7, ... がそれぞれ 1 行) ので、シャッフルは 4 レーンの
// 中で閉じる。
__device__ __forceinline__ int moe_v3_row_of(int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (MOE_V3_LANES - 1);
    return blockIdx.x * MOE_V3_ROWS_PER_BLOCK + (t / MOE_V3_LANES);
}

// 4 レーンの合計 (offset 1, 2 の 2 段)。lane 0 の値だけが使われる。
// マスクは warp 全体なので、呼び出し時に warp の全スレッドが揃っていること。
__device__ __forceinline__ float moe_v3_reduce4(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 1);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    return v;
}

// x[n]..x[n+31] をレジスタへ先読み (完全展開でレジスタに載る)
__device__ __forceinline__ void moe_v3_prefetch32(const float* __restrict__ xb, float* xv)
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
extern "C" __global__ void dno_gemv_q8_0_v3(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    int lane;
    const int row = moe_v3_row_of(&lane);
    const bool valid = (row < n_out);

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb >> 2; // 1 レーンぶんのブロック数 (nb は 4 の倍数)
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 34u;
            const float d = dno_f16_to_f32_v3(blk);
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_v3_prefetch32(xb, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float y = (float)(int)(signed char)blk[2 + j] * d;
                acc += y * xv[j];
            }
        }
    }
    acc = moe_v3_reduce4(acc);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5_1_v3(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    int lane;
    const int row = moe_v3_row_of(&lane);
    const bool valid = (row < n_out);

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb >> 2;
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 24u;
            const float d = dno_f16_to_f32_v3(blk);
            const float m = dno_f16_to_f32_v3(blk + 2);
            const unsigned int qh = (unsigned int)blk[4]
                                  | ((unsigned int)blk[5] << 8)
                                  | ((unsigned int)blk[6] << 16)
                                  | ((unsigned int)blk[7] << 24);
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_v3_prefetch32(xb, xv);
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
    acc = moe_v3_reduce4(acc);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
//
// レーン境界 k/4 は 32 要素 (サブブロック) の倍数だが 256 要素 (スーパーブロック)
// の倍数とは限らない (k=2560 では 640 = 2.5 ブロック) ので、レーンの担当区間に
// 重なるスーパーブロックを回り、その中でレーン内のサブブロックだけを計算する。
// レーン外のサブブロックは読まない (x も q も触らない)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q4k_v3(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    int lane;
    const int row = moe_v3_row_of(&lane);
    const bool valid = (row < n_out);

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int seg = n_in >> 2; // 1 レーンぶんの要素数 (32 の倍数)
        const int lo = lane * seg;
        const int hi = lo + seg;
        const int b0 = lo >> 8;
        const int b1 = (hi - 1) >> 8;
        for (int b = b0; b <= b1; ++b) {
            const int bstart = b << 8;
            const unsigned char* blk = rowp + (size_t)b * 144u;
            const float d    = dno_f16_to_f32_v3(blk);
            const float dmin = dno_f16_to_f32_v3(blk + 2);
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
                dno_scale_min_k4_v3(s, blk + 4, &sc, &mn);
                const float ds = d * (float)sc;
                const float ms = dmin * (float)mn;
                const float* xb = x + (size_t)bstart + (size_t)e;
                float xv[32];
                moe_v3_prefetch32(xb, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const float y = ds * (float)(int)((q[l] >> sh) & 0x0fu) - ms;
                    acc += y * xv[l];
                }
            }
        }
    }
    acc = moe_v3_reduce4(acc);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素
//
// 5 ビット目の qh は「サブブロック番号 s のビット」を要素 l ごとに見る
// (v2 の u1/u2 の 2 ビットずらしと同じもの)。上位 4 ビットは `+ 16` で足す。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5k_v3(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    int lane;
    const int row = moe_v3_row_of(&lane);
    const bool valid = (row < n_out);

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int seg = n_in >> 2;
        const int lo = lane * seg;
        const int hi = lo + seg;
        const int b0 = lo >> 8;
        const int b1 = (hi - 1) >> 8;
        for (int b = b0; b <= b1; ++b) {
            const int bstart = b << 8;
            const unsigned char* blk = rowp + (size_t)b * 176u;
            const float d    = dno_f16_to_f32_v3(blk);
            const float dmin = dno_f16_to_f32_v3(blk + 2);
            const unsigned char* qh = blk + 16;
            const int s0 = (lo > bstart) ? ((lo - bstart) >> 5) : 0;
            const int s1 = (hi < bstart + 256) ? ((hi - bstart + 31) >> 5) : 8;
            for (int s = s0; s < s1; ++s) {
                const int e = s << 5;
                const unsigned char* ql = blk + 48 + ((e >> 6) << 5);
                const unsigned int sh = (e & 32) ? 4u : 0u;
                unsigned int sc, mn;
                dno_scale_min_k4_v3(s, blk + 4, &sc, &mn);
                const float ds = d * (float)sc;
                const float ms = dmin * (float)mn;
                const float* xb = x + (size_t)bstart + (size_t)e;
                float xv[32];
                moe_v3_prefetch32(xb, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int hi_bit = (int)((qh[l] >> s) & 1u) << 4;
                    const float y = ds * (float)((int)((ql[l] >> sh) & 0x0fu) + hi_bit) - ms;
                    acc += y * xv[l];
                }
            }
        }
    }
    acc = moe_v3_reduce4(acc);
    if (valid && lane == 0) out[row] = acc;
}
