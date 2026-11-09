// denno CUDA: prefill の hc mix の inject 射影 (k = hc*n_embd, n = hc <= 8) を
// 1 スレッド 1 出力で回す GEMV (task 189)。
//
// PTX へのコンパイル (build_ptx.sh の `mix_inj`。arch は既定の compute_75):
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_mix_inj.cu -o kernels_mix_inj.ptx
//
// なぜ別カーネルか (lab t189a の実測):
//   inject は [nt=512, k=10240] x [10240, 4] の 21 M MAC しかないのに、既存の
//   `dno_gemm_bf16_mt` (1 ブロック = 16 行 x 32 トークン) では grid.x が
//   ceil(4/16) = 1 になり、16 行のうち 4 行しか使わないうえブロック数が
//   512/32 = 16 本しか出ない。実測 329 us/呼び出し (2,688 呼び出しで 884 ms、
//   mix 全体の 20%)。同じ形を 1 スレッド 1 出力で回すと、スレッド数は
//   nt*n = 2,048 で、k を 1 回なめるだけの帯域律速 (~20 us) になる。
//
// 数値: **既存 `dno_gemm_bf16_mt` とビット一致**。
//   出力 1 要素につきアキュムレータは 1 個で、k は昇順に
//   `acc = __fmaf_rn(bf16(w), x, acc)` で積む。既存カーネルも同じ順序
//   (k0 を 32 ずつ、内側の kk は昇順) で `__fmaf_rn` を 1 個の acc に積む。
//   k が 32 の倍数でない端数は既存側が 0 を積む (fma(0,0,acc) = acc) ので
//   こちらも同じ値になる。
//
// 前提 (ホスト側 `mix_inj_ok` が同じ条件を検査する):
//   - w は [n][k] の BF16 row-major (バイト数 n*k*2)
//   - x は [nt][k] の f32、y は [nt][n] の f32
//   - 1 <= n <= MIXINJ_MAX_N、nt >= 1、k >= 1
//
// 起動: grid = (ceil(nt*n / MIXINJ_THREADS), 1, 1) / block = (MIXINJ_THREADS, 1, 1)。
// スレッドの並びは idx = t*n + j (x の同じ行を n 本のスレッドが共有する)。

#define MIXINJ_THREADS 128
// inject の n は hc (実形 4)。8 本までは 1 スレッド 1 出力で受ける。
#define MIXINJ_MAX_N 8

// BF16 -> f32 (上位 16 bit の複製。kernels_elem.cu の dno_bf16_to_f32 と同一)。
__device__ __forceinline__ float mixinj_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

extern "C" __global__ void __launch_bounds__(MIXINJ_THREADS, 8)
dno_mix_inj_bf16(
    const unsigned short *__restrict__ w, // [n, k] BF16 row-major
    const float *__restrict__ x,          // [nt, k]
    float *__restrict__ y,                // [nt, n]
    const int nt,
    const int k,
    const int n)
{
    const int idx = blockIdx.x * MIXINJ_THREADS + threadIdx.x;
    const int t = idx / n;
    const int j = idx - t * n;
    if (t >= nt) {
        return;
    }
    const unsigned short *__restrict__ row = w + (size_t)j * (size_t)k;
    const float *__restrict__ xr = x + (size_t)t * (size_t)k;
    // 1 出力 1 アキュムレータ・k 昇順 (既存 GEMM と同じ丸め順)。
    float acc = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < k; ++d) {
        acc = __fmaf_rn(mixinj_bf16_to_f32(row[d]), xr[d], acc);
    }
    y[(size_t)t * (size_t)n + j] = acc;
}
