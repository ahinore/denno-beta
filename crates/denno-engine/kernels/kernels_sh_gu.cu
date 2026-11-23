// task 576: shared expert (Q8_0) の **gate + up + silu*mul を 1 発**に畳むカーネル。
//
// なぜ要るか (lab t572b / t564a の `[q4ledger] shared`、warm 短 decode):
//   shared 段 0.59 ms/token のうち launch=0.24 (192 本 = 48 層 x 4) + cpu=0.21。
//   sh_wait は 0.05 しか無い = **段はホストの投入で決まっている** (GPU は待たせていない)。
//   1 層 4 発 (gate / up / silu / down) のうち、gate と up は同じ x を読み、
//   silu は 640 要素の要素演算で grid は 3 ブロック。起動の固定費だけを払っている。
//   そこで gate と up を **同じスレッドで同じ x を使って** 1 発にし、silu*gate*up まで
//   その epilogue で書く (= 3 発 -> 1 発、x の再読みも半分)。
//
// 数値: 1 行の積和の演算列は `kernels_moe_v5.cu` の `dno_gemv_q8_0_v5` と同一
// (1 積和 1 丸め・アキュムレータ 4 本 `j & 3`・ブロック 0 -> 1 の順・
//  lanes の shfl_down 1,2,4... の順)。silu も `kernels_dec_epi.cu` の
// `dno_dec_silu_mul` と同じ式 `x / (1 + __expf(-x))` を 1 要素ずつ書く。
// したがって **ビット一致** (ゲート `DENNO_DEC_SH_GU=1` で切り替える)。
//
// 前提 (ホストが門番): Q8_0、n_in % 64 == 0、n_in / 32 が偶数、
// 1 行あたり lanes スレッドで np = n_in/64 が lanes で割り切れる、rows_per_block * lanes <= 1024。
// 満たさなければ v5 (gate / up の 2 発) へ落ちる。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_sh_gu.cu -o kernels_sh_gu.ptx
// まとめて作るなら `bash build_ptx.sh sh_gu`。
#include <cuda_fp16.h>

// ---- v5 と同じ実装 (v5 の .cu は触らない。名前だけ _shgu にする) ------------

__device__ __forceinline__ float dno_shgu_f16_bits(unsigned int h)
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
__device__ __forceinline__ int dno_shgu_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_shgu_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// x[n]..x[n+31] をレジスタへ先読み。
__device__ __forceinline__ void dno_shgu_prefetch32(const float* __restrict__ xb, float* xv)
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

// 行番号とレーン番号 (v5 と同じ規約: lane = threadIdx.x & (lanes-1))。
__device__ __forceinline__ int dno_shgu_row_of(int rows_per_block, int lanes, int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (lanes - 1);
    return blockIdx.x * rows_per_block + (t / lanes);
}

// シャッフルのマスク。1 ブロックが 1 warp 未満のときは実在するスレッドだけ。
__device__ __forceinline__ unsigned int dno_shgu_mask()
{
    const unsigned int n = blockDim.x;
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

// grid = ceil(n_out / rows)、block = rows * lanes。
// (row, lane) のスレッドが gate 行 row と up 行 row の内積を同じ x で取る。
extern "C" __global__ void dno_dec_sh_gateup_silu(
    const unsigned char* __restrict__ wg,
    const unsigned char* __restrict__ wu,
    const float* __restrict__ x,
    float* __restrict__ act,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = dno_shgu_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = dno_shgu_mask();

    float ag[4];
    float au[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { ag[i] = 0.0f; au[i] = 0.0f; }

    if (valid) {
        const unsigned char* rg = wg + (size_t)row * (size_t)row_bytes;
        const unsigned char* ru = wu + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;      // 32 要素ブロック数 (偶数)
        const int np = nb >> 1;        // ブロック対の数
        const int per = np / lanes;    // 1 レーンの対数
        const int p0 = lane * per;
        for (int p = p0; p < p0 + per; ++p) {
            unsigned int cg[17];
            unsigned int cu[17];
            dno_shgu_load17(cg, rg + (size_t)p * 68u);
            dno_shgu_load17(cu, ru + (size_t)p * 68u);
            const float g0 = dno_shgu_f16_bits(cg[0] & 0xffffu);
            const float g1 = dno_shgu_f16_bits((cg[8] >> 16) & 0xffffu);
            const float u0 = dno_shgu_f16_bits(cu[0] & 0xffffu);
            const float u1 = dno_shgu_f16_bits((cu[8] >> 16) & 0xffffu);
            const float* xb = x + (size_t)p * 64u;
            float xv[32];
            // 前半 32 要素 (ブロック 0)。
            dno_shgu_prefetch32(xb, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float xf = xv[j];
                const int qg = dno_shgu_i8(cg[(2 + j) >> 2], (2 + j) & 3);
                const float yg = __fmul_rn((float)qg, g0);
                ag[j & 3] = __fadd_rn(ag[j & 3], __fmul_rn(yg, xf));
                const int qu = dno_shgu_i8(cu[(2 + j) >> 2], (2 + j) & 3);
                const float yu = __fmul_rn((float)qu, u0);
                au[j & 3] = __fadd_rn(au[j & 3], __fmul_rn(yu, xf));
            }
            // 後半 32 要素 (ブロック 1)。
            dno_shgu_prefetch32(xb + 32, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float xf = xv[j];
                const int qg = dno_shgu_i8(cg[9 + (j >> 2)], j & 3);
                const float yg = __fmul_rn((float)qg, g1);
                ag[j & 3] = __fadd_rn(ag[j & 3], __fmul_rn(yg, xf));
                const int qu = dno_shgu_i8(cu[9 + (j >> 2)], j & 3);
                const float yu = __fmul_rn((float)qu, u1);
                au[j & 3] = __fadd_rn(au[j & 3], __fmul_rn(yu, xf));
            }
        }
    }
    float vg = __fadd_rn(__fadd_rn(ag[0], ag[1]), __fadd_rn(ag[2], ag[3]));
    float vu = __fadd_rn(__fadd_rn(au[0], au[1]), __fadd_rn(au[2], au[3]));
    // v5 と同じ順で lanes 本を畳む (2 本を同じ段で回す)。
    for (int off = 1; off < lanes; off <<= 1) {
        vg += __shfl_down_sync(mask, vg, off);
        vu += __shfl_down_sync(mask, vu, off);
    }
    if (valid && lane == 0) {
        // kernels_dec_epi.cu の dno_dec_silu_mul と同じ式。
        const float s = vg / (1.0f + __expf(-vg));
        act[row] = s * vu;
    }
}
