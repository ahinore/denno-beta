// task 496: decode の Q8_0 GEMV (`kernels_mix_dn6.cu` の dn6) の**レーン詰め**版。
//
// なぜ要るか (t492c / t488a の `[dec-kprof-avg]`、既定モデル flash-next-UD-IQ3_XXS):
//   ssm 段 3.03 ms/token のうち GDN の qkv GEMV が最大の 1 本で、形は
//   n_in = 2,560 / n_out = 16,384 (q 2,560 + k 2,560 + v 6,144 + gate 6,144 の
//   連結 1 発、重み 44.6 MB)。dn6 は **1 レーン = 1 対 (64 要素)** なので
//   np = n_in / 64 = 40 対を 32 * ceil(40/32) = 64 レーンへ割り当てる。
//   このため **24/64 = 37.5% のレーンが遊ぶ** (帯域 1.17 TB/s はピーク 1.79 の 65%)。
//   同じ形は full 層の qkv (n_in = 2,560) にも出る。
//
// このカーネルは dn6 の
//   - 行の割り当てを「1 ブロック rows 行」から **1 ブロック = rows * np 個の (行, 対)**
//     に変えて、レーンを (行, 対) で埋める (items = rows * np、block = 32 * ceil(items/32))
//   - x を shared に 1 回だけ置き、同じ対を使う rows 本の行で使い回す
//   の 2 点だけを変える。**1 対の積和 (64 要素・4 アキュムレータ) は dn6 と同一**で、
//   変わるのは「対の部分和をどう足すか」だけ (dn6 = warp 木 + warp の逐次、
//   dn8 = 対の昇順 1 本)。行ごとに独立で決定的。
//
// 前提 (ホストが門番): Q8_0 / n_in % 64 == 0 / np = n_in / 64 >= 1 /
//   np % 32 != 0 (np が 32 の倍数なら dn6 が既に 100% 詰まるので使わない) /
//   items = rows * np <= 1024 / shared = (np * 65 + items) * 4 <= 48 KiB。
// 満たさなければ dn6 / v5 へ落ちる。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dn8.cu -o kernels_mix_dn8.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dn8`。

#include <cstdint>

// ---- Q8_0 の読み方 (kernels_mix_dn6.cu と同じ実装。dn6 の .cu は触らない) ----

// f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------------
__device__ __forceinline__ float dno_dn8_f16_bits(unsigned int h)
{
    const unsigned int sign = (h & 0x8000u) << 16;
    const unsigned int exp = (h >> 10) & 0x1fu;
    const unsigned int man = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            unsigned int m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

// 4 バイトワード 1 本から 1 バイト (符号付き 8 bit) を取り出す。
__device__ __forceinline__ int dno_dn8_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック = 64 要素) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_dn8_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// x[0]..x[63] とレジスタの 17 ワードの内積 (dn6 の `dno_dn6_dot64` と同一の演算列)。
__device__ __forceinline__ float dno_dn8_dot64(const unsigned int* cur, const float* xv)
{
    const float d0 = dno_dn8_f16_bits(cur[0] & 0xffffu);
    const float d1 = dno_dn8_f16_bits((cur[8] >> 16) & 0xffffu);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn8_i8(cur[(2 + j) >> 2], (2 + j) & 3);
        const float y = __fmul_rn((float)q, d0);
        const float t = __fmul_rn(y, xv[j]);
        if ((j & 3) == 0) acc0 = __fadd_rn(acc0, t);
        else if ((j & 3) == 1) acc1 = __fadd_rn(acc1, t);
        else if ((j & 3) == 2) acc2 = __fadd_rn(acc2, t);
        else acc3 = __fadd_rn(acc3, t);
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = dno_dn8_i8(cur[9 + (j >> 2)], j & 3);
        const float y = __fmul_rn((float)q, d1);
        const float t = __fmul_rn(y, xv[32 + j]);
        if ((j & 3) == 0) acc0 = __fadd_rn(acc0, t);
        else if ((j & 3) == 1) acc1 = __fadd_rn(acc1, t);
        else if ((j & 3) == 2) acc2 = __fadd_rn(acc2, t);
        else acc3 = __fadd_rn(acc3, t);
    }
    return __fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3));
}

// grid = ceil(n_out / rows)、block = 32 * ceil(rows * np / 32)、shmem = (np * 65 + items) * 4。
// スレッド tid < items は item = tid を担当し、行 = item / np、対 = item % np。
extern "C" __global__ void dno_mix_gemv_q8_0_dn8(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows)
{
    extern __shared__ float smem[];
    if (rows < 1) return;
    const int np = n_in >> 6;
    const int items = np * rows;
    const int tid = (int)threadIdx.x;
    float* xs = smem;              // np * 65 (対ごとに 1 float の詰め物 = bank を散らす)
    float* part = smem + np * 65;  // items 本の部分和
    // 1) x を shared へ 1 回だけ (対 p は xs[p * 65 .. p * 65 + 63])。
    for (int i = tid; i < n_in; i += (int)blockDim.x) {
        xs[(i >> 6) * 65 + (i & 63)] = x[i];
    }
    __syncthreads();
    // 2) (行, 対) を 1 レーンで埋める (items <= blockDim が前提)。
    if (tid < items) {
        const int r = tid / np;
        const int pair = tid - r * np;
        const int row = (int)blockIdx.x * rows + r;
        float xv[64];
        const float* xb = xs + pair * 65;
#pragma unroll
        for (int i = 0; i < 64; ++i) xv[i] = xb[i];
        float v = 0.0f;
        if (row < n_out) {
            const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
            unsigned int cur[17];
            dno_dn8_load17(cur, rowp + (size_t)pair * 68u);
            v = dno_dn8_dot64(cur, xv);
        }
        part[tid] = v;
    }
    __syncthreads();
    // 3) 行ごとに対の部分和を昇順 1 本で足す (dn6 の warp 木 + warp 逐次とは順が違う)。
    if (tid < rows) {
        const int orow = (int)blockIdx.x * rows + tid;
        if (orow < n_out) {
            const float* p = part + tid * np;
            float s = p[0];
            for (int i = 1; i < np; ++i) s = __fadd_rn(s, p[i]);
            out[orow] = s;
        }
    }
}
