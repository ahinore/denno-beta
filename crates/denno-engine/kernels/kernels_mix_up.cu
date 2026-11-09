// task 192: decode の hc mix の up 投影 (BF16) を 1 warp = 1 行で回す。
//
// 背景 (lab t192a の `[dec-kprof-avg]` mix 表): `dno_matvec_bf16_r` (up、
// grid=10,240・block=256) が 14.0 us。1 行は k=320 要素 = 640 B しかないのに
// 1 ブロック (256 スレッド) を立てるので、**1 スレッドの仕事が 1.25 要素**で
// 起動と縮約の固定費が支配的だった (実効 468 GB/s)。
//
// ここでは 1 warp = 1 行、1 ブロック = 8 行 (8 warp = 256 スレッド) にする。
// 1 レーンは 8 要素 (16 B = uint4) ずつ読み、warp 内の固定木で縮約する。
// 10,240 行なら 1,280 ブロック。値は行内の合計順が変わる (丸め順の変更。
// 決定論は保つ)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_up.cu -o kernels_mix_up.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_up`。

#define DNO_MIXUP_WARPS 8
#define DNO_MIXUP_THREADS (DNO_MIXUP_WARPS * 32)

// BF16 の生 2 バイトを f32 へ (上位 16 bit の複製 = 丸めなし)。
__device__ __forceinline__ float dno_mixup_bf16(unsigned int h)
{
    return __uint_as_float(h << 16);
}

// ---------------------------------------------------------------------------
// out[r] = Σ_i w[r][i] * x[i]   (w は BF16 の生バイト [n][k]、行優先)
//
// grid = ceil(n / 8)、block = 256。k は 8 の倍数であること (ホストが門番)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_up_bf16(
    const unsigned char* __restrict__ w,  // [n][k] BF16
    const float* __restrict__ x,          // [k]
    float* __restrict__ out,              // [n]
    int k, int n)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * DNO_MIXUP_WARPS + warp;
    if (row >= n) {
        return;
    }
    const unsigned short* rowp =
        (const unsigned short*)(w + (size_t)row * (size_t)k * 2u);

    float acc = 0.0f;
    // 1 レーン = 8 要素 (uint4 = 8 BF16)。行頭は 16 B 整列。
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(rowp + c);
        // 8 要素を 1 本の鎖で足す (要素ごとに 1 演算 1 丸め)。
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.x & 0xffffu), x[c + 0]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.x >> 16), x[c + 1]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.y & 0xffffu), x[c + 2]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.y >> 16), x[c + 3]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.z & 0xffffu), x[c + 4]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.z >> 16), x[c + 5]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.w & 0xffffu), x[c + 6]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixup_bf16(v.w >> 16), x[c + 7]));
    }
    // warp 内の固定木 (offset 16, 8, 4, 2, 1)。
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, off));
    }
    if (lane == 0) {
        out[row] = acc;
    }
}
