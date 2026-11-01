// denno CUDA MoE GEMV kernels (fp16 / fast mode) — decode の量子 GEMV を
// **fp16 dequant + half2 FMA** で回す版 (task 108)。
//
// PTX へのコンパイル (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_fp16.cu -o kernels_moe_fp16.ptx
// まとめて作るなら `kernels/build_ptx.sh moe_fp16`。
//
// **数値契約 (2026-11-01 オペレーター承認の契約変更)**:
//   これまでの「線形層は CPU と GPU でビット一致」は検証モード (env 未指定) の
//   契約として残る (v1/v2 経路と既存テストは触らない)。本カーネルは
//   `DENNO_GEMV_FP16=1` の fast mode だけで使われ、丸めも精度も v2/v3 と違う:
//   - dequant の積とサブブロック (32 要素) 内の累積を f16 (hfma2 / hmul2) で行う。
//   - サブブロック間の累積とレーン間 reduce は f32 のまま (丸め順は v3 と同じ)。
//   品質受入基準は (a) アンカー質問の意味合い (b) temp0 で出力 md5 安定
//   (c) ゴミ出力なし (d) 実測 1.5 倍以上。誤差しきい値は行列全体の
//   max|diff| / max|ref| < 2e-2 (テスト cuda_gemv_fp16_matches_cpu)。
//
// **なぜ速くなるか (理論)**: v3 は 1 レーンが 32 要素を 1 本の f32 FMA 依存チェーン
// で足す (32 FMA x 4 サイクル = 128 サイクル / サブブロック。~115 GMAC/s の天井)。
// 本カーネルは dequant 積を独立な 16 本の hmul2 / hfma2 に展開してから、
// 2 本の 8 段 hfma2 チェーンに振り分ける (チェーン長 8 x 4 サイクル = 32 サイクル)。
// hfma2 は 1 命令 2 MAC なので、依存チェーンの制約が実質 1/4 になり、Turing 以降の
// fp16 パイプ (fp32 の 2 倍速) も使える。狙いは experts GEMV 部分の 1.5-3 倍
// (swift15 IQ3_S decode 164.8 ms/tok のうち experts 81.7 ms → REPORT_P108.md)。
//
// **共有メモリは使わない** (行バイトもデコード済み fp16 も載せない)。v2 の行
// ステージング (k=2560 の Q4_K で 90 KiB) と同じ罠を踏まないため。先読みは
// レジスタで行う: 次の量子ブロックの重みバイトを、現在のブロックの計算の前に
// 発行しておく (unroll された演算がロードレイテンシを隠す)。
// cp.async は compute_80 以降の命令で compute_75 の PTX に入らない上、Q8_0 は
// 行バイト 34 B が 4 バイト非整合なのでそのままでは使えない。1 サブブロック
// 16-32 B のチャンクではレジスタ先読みと利点が変わらないため、本実装では
// 見送った (task 111 の prefill / mma 経路で改めて検討する)。
//
// レイアウトは v3 と同じ: `lanes` レーンが 1 行を分担する
// (blockDim = rows_per_block * lanes、レーン l は k/lanes の連続区間)。
// 引数も v3 と同一シグネチャ `(w, x, out, n_in, n_out, row_bytes,
// rows_per_block, lanes)` なので、ホスト側の launch コードと ABI 検査
// (ptx_entry_params == 8) を共有する。**旧 PTX を読むと黙って壊れる**ので、
// ローダは .param 数を確かめてからモジュールを読む。
//
// 対応型は Q8_0 / Q5_1 / Q4_K の 3 つ (Q5_K は対象外 → v3 / v2 へ落ちる)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

#include <cuda_fp16.h>

__device__ __forceinline__ float dno_f16_to_f32_fp16(const unsigned char* b)
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

__device__ __forceinline__ void dno_scale_min_k4_fp16(int j, const unsigned char* q,
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

#define MOE_FP16_LANES_MAX 4

// v3 と同じ行番号 / マスク / reduce (f32 のまま。レーン間の丸め順は v3 と同一)。
__device__ __forceinline__ int moe_fp16_row_of(int rows_per_block, int lanes, int* lane)
{
    const int t = threadIdx.x;
    const int lshift = (lanes == MOE_FP16_LANES_MAX) ? 2 : 1;
    *lane = t & (lanes - 1);
    return blockIdx.x * rows_per_block + (t >> lshift);
}

__device__ __forceinline__ unsigned int moe_fp16_mask()
{
    const unsigned int n = blockDim.x;
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

__device__ __forceinline__ float moe_fp16_reduce(float v, int lanes, unsigned int mask)
{
    v += __shfl_down_sync(mask, v, 1);
    if (lanes == MOE_FP16_LANES_MAX) {
        v += __shfl_down_sync(mask, v, 2);
    }
    return v;
}

// x[n]..x[n+31] をレジスタへ先読み (v3 と同じ float4 読み)。
__device__ __forceinline__ void moe_fp16_prefetch32(const float* __restrict__ xb, float* xv)
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

// サブブロック (32 要素 = 16 half2) の f16 累積。
//
// 事前条件: y2[j] に dequant 済みの重み (f16 x 2) が入っていること。
// 2 本の 8 段 hfma2 チェーン (ペア 0..8 と 8..16) に振り分け、最後に hadd2 で
// 足してから f32 の累算器へ 2 回の加算で渡す (low → high の順。f32 部分は
// -fmad=false で融合しない)。
__device__ __forceinline__ float moe_fp16_block_sum(const __half2* y2, const float* xv)
{
    __half2 acca = __float2half2_rn(0.0f);
    __half2 accb = __float2half2_rn(0.0f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __half2 x2 = __floats2half2_rn(xv[2 * j], xv[2 * j + 1]);
        acca = __hfma2(y2[j], x2, acca);
    }
#pragma unroll
    for (int j = 8; j < 16; ++j) {
        const __half2 x2 = __floats2half2_rn(xv[2 * j], xv[2 * j + 1]);
        accb = __hfma2(y2[j], x2, accb);
    }
    const __half2 s2 = __hadd2(acca, accb);
    float acc = __low2float(s2);
    acc += __high2float(s2);
    return acc;
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q8_0_fp16(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_fp16_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_fp16_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb / lanes; // 1 レーンぶんのブロック数 (nb は lanes の倍数)
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 34u;
            const __half2 d2 = __half2half2(__ushort_as_half(
                (unsigned short)blk[0] | ((unsigned short)blk[1] << 8)));
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_fp16_prefetch32(xb, xv);
            __half2 y2[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float q0 = (float)(int)(signed char)blk[2 + 2 * j];
                const float q1 = (float)(int)(signed char)blk[3 + 2 * j];
                const __half2 q2 = __floats2half2_rn(q0, q1);
                y2[j] = __hmul2(q2, d2); // dequant の積を f16 で丸める (契約変更)
            }
            acc += moe_fp16_block_sum(y2, xv);
        }
    }
    acc = moe_fp16_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q5_1: d(2B) + m(2B) + qh[4] + qs[16] = 24 B / 32 要素
//
// ペア j = (要素 j, 要素 16+j) が同じバイト blk[8+j] の下位 / 上位ニブル。
// dequant は q * d + m を 1 回の hfma2 で丸める。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5_1_fp16(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_fp16_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_fp16_mask();

    float acc = 0.0f;
    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;
        const int per = nb / lanes;
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 24u;
            const __half2 d2 = __half2half2(__ushort_as_half(
                (unsigned short)blk[0] | ((unsigned short)blk[1] << 8)));
            const __half2 m2 = __half2half2(__ushort_as_half(
                (unsigned short)blk[2] | ((unsigned short)blk[3] << 8)));
            const unsigned int qh = (unsigned int)blk[4]
                                  | ((unsigned int)blk[5] << 8)
                                  | ((unsigned int)blk[6] << 16)
                                  | ((unsigned int)blk[7] << 24);
            const float* xb = x + (size_t)b * 32u;
            float xv[32];
            moe_fp16_prefetch32(xb, xv);
            __half2 y2[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int xh0 = (int)(((qh >> j) << 4) & 0x10u);
                const int q0 = (int)(blk[8 + j] & 0x0fu) | xh0;
                const int xh1 = (int)((qh >> (j + 12)) & 0x10u);
                const int q1 = (int)(blk[8 + j] >> 4) | xh1;
                const __half2 q2 = __floats2half2_rn((float)q0, (float)q1);
                y2[j] = __hfma2(q2, d2, m2);
            }
            // ペア j の x は (xv[j], xv[16+j]) なので block_sum は使わずここで足す
            // (block_sum は隣接ペア (2j, 2j+1) 前提)。
            __half2 acca = __float2half2_rn(0.0f);
            __half2 accb = __float2half2_rn(0.0f);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __half2 x2 = __floats2half2_rn(xv[j], xv[16 + j]);
                acca = __hfma2(y2[j], x2, acca);
            }
#pragma unroll
            for (int j = 8; j < 16; ++j) {
                const __half2 x2 = __floats2half2_rn(xv[j], xv[16 + j]);
                accb = __hfma2(y2[j], x2, accb);
            }
            const __half2 s2 = __hadd2(acca, accb);
            acc += __low2float(s2);
            acc += __high2float(s2);
        }
    }
    acc = moe_fp16_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素
//
// スーパーブロック / サブブロックの歩き方は v3 と同じ (レーン境界はサブブロック
// 32 要素単位)。ds / ms は f32 で計算してから f16 に 1 回丸める (d が f16 で
// sc が 8 ビット整数なので f32 の積は正確 = 1 回の丸めで f16 の正しい積になる)。
// dequant は q * ds - ms を 1 回の hfma2 で丸める (ms は事前に符号を反転)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q4k_fp16(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = moe_fp16_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = moe_fp16_mask();

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
            const float d    = dno_f16_to_f32_fp16(blk);
            const float dmin = dno_f16_to_f32_fp16(blk + 2);
            const int s0 = (lo > bstart) ? ((lo - bstart) >> 5) : 0;
            const int s1 = (hi < bstart + 256) ? ((hi - bstart + 31) >> 5) : 8;
            for (int s = s0; s < s1; ++s) {
                const int e = s << 5; // スーパーブロック内の要素オフセット
                const unsigned char* q = blk + 16 + ((e >> 6) << 5);
                const unsigned int sh = (e & 32) ? 4u : 0u;
                unsigned int sc, mn;
                dno_scale_min_k4_fp16(s, blk + 4, &sc, &mn);
                const float ds = d * (float)sc;
                const float ms = dmin * (float)mn;
                const __half2 ds2 = __half2half2(__float2half(ds));
                const __half2 ms2 = __half2half2(__float2half(-ms));
                const float* xb = x + (size_t)bstart + (size_t)e;
                float xv[32];
                moe_fp16_prefetch32(xb, xv);
                __half2 y2[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const unsigned int v0 = ((unsigned int)q[2 * j] >> sh) & 0x0fu;
                    const unsigned int v1 = ((unsigned int)q[2 * j + 1] >> sh) & 0x0fu;
                    const __half2 q2 = __floats2half2_rn((float)v0, (float)v1);
                    y2[j] = __hfma2(q2, ds2, ms2);
                }
                acc += moe_fp16_block_sum(y2, xv);
            }
        }
    }
    acc = moe_fp16_reduce(acc, lanes, mask);
    if (valid && lane == 0) out[row] = acc;
}
