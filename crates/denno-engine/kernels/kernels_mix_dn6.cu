// task 422: decode の GEMV (Q8_0) を **1 ブロック = rows 行 / 1 レーン = 1 対**で回す
// 汎用カーネル。task 418 の `dno_mix_gemv_q8_0_dn5` (1 ブロック 1 行) を
// 「1 ブロックが複数行を持つ」形に一般化したもの。
//
// なぜ要るか (lab t422a の `[dec-kprof-avg]`、band 0、flash-next-UD-IQ3_XXS):
//   - mix の down (`dno_mix_gemv_q8_0_dn5`): grid 320 x block 160、**15.4 us**。
//     重み 3.48 MB + x 13.1 MB (320 ブロックが同じ 41 KB を読む) = 16.6 MB を
//     1.08 TB/s で読んでいる。DRAM だけなら 3.48 MB / 1.8 TB/s = 1.9 us。
//   - GDN の qkv/out (`dno_gemv_q8_0_v5`): grid 320 x block 256 (v5 は 8 行 1
//     ブロック / 1 レーンが 5 対を逐次ループ)、**35.4 us x 2 本/層**。n_in=10,240 /
//     n_out=2,560 で重み 27.9 MB を 787 GB/s。
// どちらも「1 レーンが複数の対を逐次ループする」形が残っている (v5) か、
// 「行ごとに同じ x を読み直す」形 (dn5) が残っている。
//
// このカーネルは
//   - 1 ブロック = `rows` 行。block = rows * 32 * warps (warps = ceil(np / 32))
//   - 1 レーン = ちょうど 1 対 (Q8_0 の 2 ブロック = 68 B)。ループも先読みも無い
//   - 同じレーン (wid, lane) が rows 行ぶんの重みを読むので、x の 64 要素は
//     **レジスタに 1 回だけ載せて rows 行で使い回す** (x のトラフィックが 1/rows)
//   - 縮約は warp 内の木 (offset 16..1) -> warp ごとの部分和を共有メモリ ->
//     スレッド 0..rows-1 が **固定順 (warp 0..warps-1 の逐次加算)** で畳む
//
// 数値: `rows == 1` で撃つと dn5 と **1 ビットも変わらない** (同じ 68 B の読み方、
// 同じ 4 アキュムレータ、同じ木の段数、同じ逐次加算)。rows > 1 でも行ごとの
// 積和は同一で、変わるのは「x を 1 回だけ読む」ことだけ = 値は dn5 と同一。
// v5 から切り替えると合計順が変わる (丸めだけ) ので、ゲートで切り戻せる。
//
// 前提 (ホストが門番): Q8_0、n_in % 64 == 0、np = n_in / 64 が 1 以上、
// warps = ceil(np / 32) が 8 以下、rows * 32 * warps <= 1024、
// row_bytes = n_in / 32 * 34。満たさなければ v5 / dn5 へ落ちる。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dn6.cu -o kernels_mix_dn6.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dn6`。

#include <cstdint>
#include <cuda_fp16.h>

// ---- Q8_0 の読み方 (kernels_mix_dn5.cu と同じ実装。dn5 の .cu は触らない) ----

// f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------------
__device__ __forceinline__ float dno_dn6_f16_bits(unsigned int h)
{
    // task 640: 手書きのビット展開をハードウェアの cvt.f32.f16 (`__half2float`) に
    // 置き換えた。f16 -> f32 の拡張は IEEE 754 で一意なので結果は同じ (0 / 非正規化数
    // を含む全 65536 パターンで旧実装と一致することを cuda.rs の
    // f16_cvt_exhaustive で確認済み)。NaN と inf だけは旧実装の式をそのまま残す
    // (ハードウェアは NaN を quiet 化するので、ペイロードの再現性のため)。
    if ((h & 0x7c00u) == 0x7c00u) {
        const unsigned int sign = (h & 0x8000u) << 16;
        return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
    }
    return __half2float(__ushort_as_half((unsigned short)h));
}

// 4 バイトワード 1 本から 1 バイト (符号付き 8 bit) を取り出す。
__device__ __forceinline__ int dno_dn6_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック = 64 要素) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_dn6_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// x[0]..x[63] をレジスタへ先読み (float4 x 16)。
__device__ __forceinline__ void dno_dn6_load64(const float* __restrict__ xb, float* xv)
{
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(xb + 4 * i);
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// レジスタに載っている x の 64 要素と、Q8_0 の 2 ブロック (68 B = 64 要素) の内積。
// 演算列と丸めは dn5 の `dno_dn5_dot64` と同一 (acc[j&3] に 1 演算 1 丸め)。
__device__ __forceinline__ float dno_dn6_dot64(const unsigned int* cur, const float* xv)
{
    const float d0 = dno_dn6_f16_bits(cur[0] & 0xffffu);
    const float d1 = dno_dn6_f16_bits((cur[8] >> 16) & 0xffffu);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn6_i8(cur[(2 + j) >> 2], (2 + j) & 3);
        const float y = __fmul_rn((float)q, d0);
        const float t = __fmul_rn(y, xv[j]);
        if ((j & 3) == 0) acc0 = __fadd_rn(acc0, t);
        else if ((j & 3) == 1) acc1 = __fadd_rn(acc1, t);
        else if ((j & 3) == 2) acc2 = __fadd_rn(acc2, t);
        else acc3 = __fadd_rn(acc3, t);
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn6_i8(cur[9 + (j >> 2)], j & 3);
        const float y = __fmul_rn((float)q, d1);
        const float t = __fmul_rn(y, xv[32 + j]);
        if ((j & 3) == 0) acc0 = __fadd_rn(acc0, t);
        else if ((j & 3) == 1) acc1 = __fadd_rn(acc1, t);
        else if ((j & 3) == 2) acc2 = __fadd_rn(acc2, t);
        else acc3 = __fadd_rn(acc3, t);
    }
    return __fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3));
}

// grid = ceil(n_out / rows)、block = rows * 32 * warps。
// (r, wid, lane) のスレッドは pair = wid * 32 + lane を担当し、行 row0 + r を読む。
extern "C" __global__ void dno_mix_gemv_q8_0_dn6(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows)
{
    if (rows < 1) return;
    const int warps = (int)(blockDim.x / (unsigned int)(32 * rows));
    const int per_row = 32 * warps;
    const int tid = (int)threadIdx.x;
    const int r = tid / per_row;
    const int within = tid - r * per_row;
    const int wid = within >> 5;
    const int lane = within & 31;
    const int pair = wid * 32 + lane;
    const int np = n_in >> 6;  // 対 (= 64 要素) の数
    const int row = (int)blockIdx.x * rows + r;
    __shared__ float part[8][8];

    // x は 1 レーン 1 回だけ読む (rows 行で使い回す)。
    float xv[64];
    if (pair < np) dno_dn6_load64(x + (size_t)pair * 64u, xv);

    float v = 0.0f;
    if (row < n_out && pair < np) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        unsigned int cur[17];
        dno_dn6_load17(cur, rowp + (size_t)pair * 68u);
        v = dno_dn6_dot64(cur, xv);
    }
    // warp 内の縮約 (32 レーン)。dn5 と同じ段数・同じ順。
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (lane == 0) part[r][wid] = v;
    __syncthreads();
    if (tid < rows) {
        const int orow = (int)blockIdx.x * rows + tid;
        if (orow < n_out) {
            float s = part[tid][0];
            for (int i = 1; i < warps; ++i) s = __fadd_rn(s, part[tid][i]);
            out[orow] = s;
        }
    }
}
