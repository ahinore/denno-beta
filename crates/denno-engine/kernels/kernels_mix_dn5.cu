// task 418: decode の hc mix の **down (Q8_0) を 1 ブロック 1 行・1 レーン 1 対**で
// 回すカーネル。
//
// 従来 (task 286 の `dno_gemv_q8_0_v5`、mix の既定) は down を
// `lanes=32 / rows=2` で撃つ = **160 ブロック x 64 スレッド = 10,240 スレッド**。
// 実形 (n_in = 10,240 = 160 対、n_out = 320) では 1 レーンが 5 対 (= 340 B) を
// 逐次ループで読み、GPU 全体で 320 warp しか出ない (128 SM で 2.5 warp/SM)。
// lab t418a の `[dec-kprof-avg]` (band 0) で **1 回 19.4 us / 3.48 MB = 180 GB/s**。
// 同じ 3.48 MB を 512 ブロック x 256 スレッドで読む inject (`dno_matvec_f32_r`) は
// 7.1 us で、並列度がそのまま効いている。
//
// ここでは **1 ブロック = 1 行**、ブロックのスレッド数 = 対の数 (160 = 5 warp) に
// して、1 レーンがちょうど 1 対 (68 B) を 1 回だけ読む。ループも先読みも要らず、
// 全レーンが同時に 1 回のロードを出す = 3.48 MB が一斉に in-flight になる。
// 縮約は warp 内 (32 レーン、offset 16..1) → 共有メモリに warp ごとの部分和 →
// スレッド 0 が **固定順 (warp 0..warps-1) の逐次加算**で畳む (決定的)。
//
// 数値: Q8_0 の積和そのもの (d0/d1 の f16 展開、`(int8 -> float) * d * x` を
// `__fmul_rn` / `__fadd_rn` で 1 演算 1 丸め) は v5 と同じ。**違うのは合計順だけ**
// (v5 はレーンごとに 5 対を積んでから 32 レーンの木、こちらは 1 対ずつ 32 レーン
// の木 → 5 warp の逐次)。相対 1e-7 程度の丸め差で、ゲートで切り戻せる。
//
// 前提 (ホストが門番): Q8_0、n_in % 64 == 0、対の数 np = n_in / 64 が 32 の倍数で
// np / 32 <= 8、row_bytes = n_in / 32 * 34 (= 68 * np)。満たさなければ v5 へ。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dn5.cu -o kernels_mix_dn5.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dn5`。

#include <cstdint>
#include <cuda_fp16.h>

// ---- Q8_0 の読み方 (kernels_moe_v5.cu / kernels_mix_upmix_q8.cu と同じ実装) ---

// f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------------
__device__ __forceinline__ float dno_dn5_f16_bits(unsigned int h)
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
__device__ __forceinline__ int dno_dn5_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック = 64 要素) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_dn5_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// x[n]..x[n+31] をレジスタへ先読み (v5 と同じ形)。
__device__ __forceinline__ void dno_dn5_prefetch32(const float* __restrict__ xb, float* xv)
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

// Q8_0 の 2 ブロック (68 B = 64 要素) と x の 64 要素の内積。
// 68 B バッファの位置は v5 と同じ: u[0] = d0 | q0[0..1]、u[1..7] = q0[2..29]、
// u[8] = q0[30..31] | d1、u[9..16] = q1[0..31]。
__device__ __forceinline__ float dno_dn5_dot64(const unsigned int* cur, const float* xb)
{
    const float d0 = dno_dn5_f16_bits(cur[0] & 0xffffu);
    const float d1 = dno_dn5_f16_bits((cur[8] >> 16) & 0xffffu);
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
    float xv[32];
    dno_dn5_prefetch32(xb, xv);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn5_i8(cur[(2 + j) >> 2], (2 + j) & 3);
        const float y = __fmul_rn((float)q, d0);
        acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(y, xv[j]));
    }
    dno_dn5_prefetch32(xb + 32, xv);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn5_i8(cur[9 + (j >> 2)], j & 3);
        const float y = __fmul_rn((float)q, d1);
        acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(y, xv[j]));
    }
    return __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
}

// grid = n_out (1 ブロック = 1 行)、block = 32 * warps (warps = n_in / 64 / 32)。
extern "C" __global__ void dno_mix_gemv_q8_0_dn5(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes)
{
    const int warps = (int)(blockDim.x >> 5);
    const int wid = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int row = (int)blockIdx.x;
    const int pair = wid * 32 + lane;
    const int np = n_in >> 6;  // 対 (= 64 要素) の数
    __shared__ float part[8];

    float v = 0.0f;
    if (row < n_out && pair < np) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        unsigned int cur[17];
        dno_dn5_load17(cur, rowp + (size_t)pair * 68u);
        v = dno_dn5_dot64(cur, x + (size_t)pair * 64u);
    }
    // warp 内の縮約 (32 レーン)。
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if (lane == 0) part[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0 && row < n_out) {
        // warp 0..warps-1 の固定順で畳む (決定的)。
        float s = part[0];
        for (int i = 1; i < warps; ++i) s = __fadd_rn(s, part[i]);
        out[row] = s;
    }
}
