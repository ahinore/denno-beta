// denno CUDA: prefill の hc mix の inject 射影 (k = hc*n_embd, n = hc <= 8) を
// **k 方向にレーン分割**して回す GEMV (task 195)。
//
// PTX へのコンパイル (build_ptx.sh の `mix_inj2`。arch は既定の compute_75):
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_mix_inj2.cu -o kernels_mix_inj2.ptx
//
// なぜ別カーネルか (lab t189a / t189b / t189c の実測):
//   inject は [nt=512, k=10,240] x [10,240, 4] = 21 M MAC しかないのに、既存の
//   `dno_gemm_bf16_mt` では 329 us/呼び出し (2,688 呼び出しで 884 ms)。task 189 の
//   `dno_mix_inj_bf16` (1 スレッド 1 出力・k 昇順 1 アキュムレータ) はビット一致を
//   保つ代わりに k の依存鎖がレイテンシ律速になり 444 us とさらに遅かった。
//
//   このカーネルは 1 ブロック = 1 トークン、MXI2_THREADS スレッドで k を
//   インターリーブ (d = tid, tid + T, ...) に分割し、出力ごとの部分和を
//   共有メモリの**固定木**で縮約する (atomic 無し = 実行ごとに同一結果)。
//   ブロック数は nt (実形 512) で SM を埋め、k は 256 スレッドに分散するので
//   1 スレッドあたり 40 要素 x n 出力だけになる。
//
// 数値: **ビット一致ではない** (k の和を分割するので丸め順が変わる)。
//   ただし縮約の順序は固定なので決定論的で、実行ごと・並列度ごとに同じ値になる。
//   速度優先の判断 (task 195 の「Speed has priority over bit identity here」)。
//
// 前提 (ホスト側 `mix_inj2_ok` が同じ条件を検査する):
//   - w は [n][k] の BF16 row-major (バイト数 n*k*2)
//   - x は [nt][k] の f32、y は [nt][n] の f32
//   - 1 <= n <= MXI2_MAX_N、nt >= 1、k >= 1
//
// 起動: grid = (nt, 1, 1) / block = (MXI2_THREADS, 1, 1)、smem は静的
// (MXI2_MAX_N * MXI2_THREADS floats = 8 KB)。

#define MXI2_THREADS 256
// inject の n は hc (実形 4)。8 本までは 1 ブロックで受ける。
#define MXI2_MAX_N 8

// BF16 -> f32 (上位 16 bit の複製。kernels_elem.cu の dno_bf16_to_f32 と同一)。
__device__ __forceinline__ float mixi2_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

extern "C" __global__ void __launch_bounds__(MXI2_THREADS, 2)
dno_mix_inj2_bf16(
    const unsigned short *__restrict__ w, // [n, k] BF16 row-major
    const float *__restrict__ x,          // [nt, k]
    float *__restrict__ y,                // [nt, n]
    const int nt,
    const int k,
    const int n)
{
    __shared__ float red[MXI2_MAX_N][MXI2_THREADS];

    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= nt) {
        return;
    }
    const float *__restrict__ xr = x + (size_t)t * (size_t)k;
    const unsigned short *__restrict__ w0 = w;

    // 部分和 (k をスレッドでインターリーブ分割)。
    float acc[MXI2_MAX_N];
#pragma unroll
    for (int j = 0; j < MXI2_MAX_N; ++j) {
        acc[j] = 0.0f;
    }
    for (int d = tid; d < k; d += MXI2_THREADS) {
        const float xv = xr[d];
        #pragma unroll
        for (int j = 0; j < MXI2_MAX_N; ++j) {
            if (j < n) {
                acc[j] = __fmaf_rn(mixi2_bf16_to_f32(w0[(size_t)j * (size_t)k + d]), xv, acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MXI2_MAX_N; ++j) {
        if (j < n) {
            red[j][tid] = acc[j];
        }
    }
    __syncthreads();
    // 固定木の縮約 (順序が固定なので決定論的)。
#pragma unroll
    for (int stride = MXI2_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int j = 0; j < MXI2_MAX_N; ++j) {
                if (j < n) {
                    red[j][tid] = red[j][tid] + red[j][tid + stride];
                }
            }
        }
        __syncthreads();
    }
    if (tid < n) {
        y[(size_t)t * (size_t)n + tid] = red[tid][0];
    }
}
