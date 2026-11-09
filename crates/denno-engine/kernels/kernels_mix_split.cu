// denno CUDA: prefill の hc mix (down / up) を **n 方向に細かく割る** GEMM (task 195)。
//
// PTX へのコンパイル (build_ptx.sh の `mix_split`。arch は既定の compute_75):
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_mix_split.cu -o kernels_mix_split.ptx
//
// なぜ別カーネルか (lab t189a / t189c の実測):
//   mix の down は 1 呼び出し [nt=512, k=10,240] x [10,240, n=320] の 1.68 G MAC で
//   490 us (3.4 TMAC/s = fp32 山の数 %)。up は同じ MAC で 406 us。既存の
//   `dno_gemm_bf16_mt` は 1 ブロック = 16 行 x 32 トークンで、down では
//   grid = 20 x 16 = 320 ブロック、1 スレッド 2 アキュムレータ (共有メモリ読み
//   1.5 回/FMA) なので ILP が足りない。mix_pf (task 186) は 16 アキュムレータに
//   増やしたが 1 ブロック = 64 行 x 32 トークンなので down では 5 x 16 = 80
//   ブロックにしかならず、170 SM の半分が遊ぶ。
//
//   このカーネルは 1 ブロック = MXS_BN 行 x MXS_BM トークンで、
//     - 128 スレッド = 16 (行グループ) x 8 (トークン群)
//     - 1 スレッド = 2 行 x 4 トークン = 8 アキュムレータ (共有メモリ読み 0.75 回/FMA)
//     - k タイル 32 要素、共有メモリは行ストライド 33 (bank 散らし)
//   で回す。down は 320/32 = 10 行タイル x 16 トークンタイル = 160 ブロック、
//   up は 10,240/32 = 320 行タイル x 16 = 5,120 ブロックになり、どちらも SM を埋める。
//
// 数値: **既存 `dno_gemm_bf16_mt` / `dno_mix_pf_gemm` とビット一致**。
//   BF16 -> f32 は上位 16 bit の複製 (`__uint_as_float((uint32_t)w << 16)`) で、
//   出力 1 要素につきアキュムレータは 1 個、k は昇順に `__fmaf_rn` で積む
//   (タイルの切り方だけが違う)。k が MXS_BK の倍数でない端数は 0 を積む
//   (fma(0,0,acc) = acc) ので既存側と同じ値になる。
//
// 前提 (ホスト側 `mix_split_ok` が同じ条件を検査する):
//   - w は [n][k] の BF16 row-major (バイト数 n*k*2)
//   - x は [nt][k] の f32、y は [nt][n] の f32
//   - n >= MXS_BN、nt >= MXS_BM、k >= 1
//
// 起動: grid = (ceil(n/MXS_BN), ceil(nt/MXS_BM), 1) / block = (128, 1, 1) /
// smem なし (共有メモリは静的 32*33 + 32*33 floats = 8.4 KB)。

// タイル (1 ブロック = MXS_BM トークン x MXS_BN 出力行、k は MXS_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define MXS_BM 32
#define MXS_BN 32
#define MXS_BK 32
#define MXS_TX 16
#define MXS_TY 8
#define MXS_RPT 2
#define MXS_TPT 4
#define MXS_THREADS 128
#define MXS_PAD 33

// BF16 -> f32 (上位 16 bit の複製。kernels_elem.cu の dno_bf16_to_f32 と同一)。
__device__ __forceinline__ float mixs_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

extern "C" __global__ void __launch_bounds__(MXS_THREADS, 8)
dno_mix_split_gemm(
    const unsigned short *__restrict__ w, // [n, k] BF16 row-major
    const float *__restrict__ x,          // [nt, k]
    float *__restrict__ y,                // [nt, n]
    const int nt,
    const int k,
    const int n)
{
    __shared__ float xs[MXS_BM][MXS_PAD];
    __shared__ float ws[MXS_BN][MXS_PAD];

    const int tid = threadIdx.x;
    const int tx = tid % MXS_TX; // 行グループ (2 行ずつ)
    const int ty = tid / MXS_TX; // トークン群 (4 トークンずつ)
    const int row0 = blockIdx.x * MXS_BN;
    const int tok0 = blockIdx.y * MXS_BM;

    // 2 行 x 4 トークン。
    float acc[MXS_RPT][MXS_TPT];
#pragma unroll
    for (int r = 0; r < MXS_RPT; ++r) {
#pragma unroll
        for (int t = 0; t < MXS_TPT; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += MXS_BK) {
        // x タイル: MXS_BM トークン x MXS_BK (1 スレッド 8 要素)。
#pragma unroll
        for (int j = 0; j < (MXS_BM * MXS_BK) / MXS_THREADS; ++j) {
            const int idx = tid + j * MXS_THREADS;
            const int t = idx / MXS_BK;
            const int kk = idx % MXS_BK;
            const int src = k0 + kk;
            xs[t][kk] = (tok0 + t < nt && src < k) ? x[(size_t)(tok0 + t) * k + src] : 0.0f;
        }
        // w タイル: MXS_BN 行 x MXS_BK (1 スレッド 8 要素)。
#pragma unroll
        for (int j = 0; j < (MXS_BN * MXS_BK) / MXS_THREADS; ++j) {
            const int idx = tid + j * MXS_THREADS;
            const int r = idx / MXS_BK;
            const int kk = idx % MXS_BK;
            const int src = k0 + kk;
            ws[r][kk] = (row0 + r < n && src < k)
                            ? mixs_bf16_to_f32(w[(size_t)(row0 + r) * k + src])
                            : 0.0f;
        }
        __syncthreads();
        // k 昇順のアキュムレート (出力 1 要素にアキュムレータ 1 個)。
#pragma unroll
        for (int kk = 0; kk < MXS_BK; ++kk) {
            float wv[MXS_RPT];
            float xv[MXS_TPT];
#pragma unroll
            for (int r = 0; r < MXS_RPT; ++r) {
                wv[r] = ws[tx * MXS_RPT + r][kk];
            }
#pragma unroll
            for (int t = 0; t < MXS_TPT; ++t) {
                xv[t] = xs[ty * MXS_TPT + t][kk];
            }
#pragma unroll
            for (int r = 0; r < MXS_RPT; ++r) {
#pragma unroll
                for (int t = 0; t < MXS_TPT; ++t) {
                    acc[r][t] = __fmaf_rn(wv[r], xv[t], acc[r][t]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MXS_RPT; ++r) {
        const int row = row0 + tx * MXS_RPT + r;
        if (row >= n) {
            continue;
        }
#pragma unroll
        for (int t = 0; t < MXS_TPT; ++t) {
            const int tok = tok0 + ty * MXS_TPT + t;
            if (tok < nt) {
                y[(size_t)tok * n + row] = acc[r][t];
            }
        }
    }
}
