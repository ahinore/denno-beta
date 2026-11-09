// denno CUDA: prefill の hc mix (down / up) を BF16 重みのまま回す GEMM (task 186)。
//
// PTX へのコンパイル (build_ptx.sh の `mix_pf`。arch は既定の compute_75):
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_mix_pf.cu -o kernels_mix_pf.ptx
//
// なぜ別カーネルか:
//   既存の `dno_gemm_bf16_mt` (kernels_elem.cu) は 1 ブロック = 16 行 x 32 トークンで、
//   k タイル 32 要素ごとに `__syncthreads` を 2 回打ち、1 スレッドあたり 64 FMA に
//   対して共有メモリ読みを 3 回/2 FMA 払う。lab t183a の 14,018 トークン prefill で
//   mix の down + up は 4.2 GFLOP を 1.27 ms (3.1 TFLOP/s = fp32 山の 3%) で回して
//   おり、同じホストの隣の GEMM (qkv_pf) は 74.8 TFLOP/s なので 20 倍以上遅い。
//
//   このカーネルは同じ行列積を
//     - 1 ブロック = 32 トークン x 64 行 (128 スレッド = 16 行グループ x 8 トークン群)
//     - 1 スレッド = 4 行 x 4 トークン = 16 アキュムレータ
//     - k タイル 16 要素、共有メモリは行ストライド 17 (bank 散らし)
//     - 共有メモリ読みは x が 16 スレッドで同一アドレス (ブロードキャスト)、
//       w が 4tx + 17r + kk で bank が分かれる
//   で回す。barrier あたりの FMA は 256 (旧 32) で 8 倍、共有メモリ読みは
//   1 FMA あたり 0.5 回 (旧 1.5 回)。
//
// 数値: **既存 `dno_gemm_bf16_mt` とビット一致**。BF16 -> f32 は上位 16 bit の複製
// (`__uint_as_float((uint32_t)w << 16)`、CPU の dequant と同一) で、出力 1 要素に
// つきアキュムレータは 1 個、k は昇順に `__fmaf_rn` で積む (旧カーネルと同じ順序)。
// したがって md5 は変わらない (値は丸め順まで同一)。
//
// 前提 (ホスト側 `mix_pf_ok` が同じ条件を検査する):
//   - w は [n][k] の BF16 row-major (バイト数 n*k*2)
//   - x は [nt][k] の f32、y は [nt][n] の f32
//   - n >= MIXP_BN、nt >= MIXP_BM (小さい形は既存カーネルのほうが速い)
//
// 起動: grid = (ceil(n/64), ceil(nt/32), 1) / block = (128, 1, 1) / smem なし
// (共有メモリは静的 32*17 + 64*17 floats = 6.5 KB)。

// タイル (1 ブロック = MIXP_BM トークン x MIXP_BN 出力行、k は MIXP_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define MIXP_BM 32
#define MIXP_BN 64
#define MIXP_BK 16
#define MIXP_TX 16
#define MIXP_TY 8
#define MIXP_TPT 4
#define MIXP_RPT 4
#define MIXP_THREADS 128
#define MIXP_PAD 17

// BF16 -> f32 (上位 16 bit の複製。kernels_elem.cu の dno_bf16_to_f32 と同一)。
__device__ __forceinline__ float mixp_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

extern "C" __global__ void __launch_bounds__(MIXP_THREADS, 8)
dno_mix_pf_gemm(
    const unsigned short *__restrict__ w, // [n, k] BF16 row-major
    const float *__restrict__ x,          // [nt, k]
    float *__restrict__ y,                // [nt, n]
    const int nt,
    const int k,
    const int n)
{
    __shared__ float xs[MIXP_BM][MIXP_PAD];
    __shared__ float ws[MIXP_BN][MIXP_PAD];

    const int tid = threadIdx.x;
    const int tx = tid % MIXP_TX; // 行グループ (4 行ずつ)
    const int ty = tid / MIXP_TX; // トークン群 (4 トークンずつ)
    const int row0 = blockIdx.x * MIXP_BN;
    const int tok0 = blockIdx.y * MIXP_BM;

    // 4 行 x 4 トークン。
    float acc[MIXP_RPT][MIXP_TPT];
#pragma unroll
    for (int r = 0; r < MIXP_RPT; ++r) {
#pragma unroll
        for (int t = 0; t < MIXP_TPT; ++t) {
            acc[r][t] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += MIXP_BK) {
        // x タイル: MIXP_BM トークン x MIXP_BK (1 スレッド 4 要素)。
#pragma unroll
        for (int j = 0; j < (MIXP_BM * MIXP_BK) / MIXP_THREADS; ++j) {
            const int idx = tid + j * MIXP_THREADS;
            const int t = idx / MIXP_BK;  // 0..MIXP_BM-1
            const int kk = idx % MIXP_BK;
            const int src = k0 + kk;
            xs[t][kk] = (tok0 + t < nt && src < k) ? x[(size_t)(tok0 + t) * k + src] : 0.0f;
        }
        // w タイル: MIXP_BN 行 x MIXP_BK (1 スレッド 8 要素)。
#pragma unroll
        for (int j = 0; j < (MIXP_BN * MIXP_BK) / MIXP_THREADS; ++j) {
            const int idx = tid + j * MIXP_THREADS;
            const int r = idx / MIXP_BK;  // 0..MIXP_BN-1
            const int kk = idx % MIXP_BK;
            const int src = k0 + kk;
            ws[r][kk] = (row0 + r < n && src < k)
                            ? mixp_bf16_to_f32(w[(size_t)(row0 + r) * k + src])
                            : 0.0f;
        }
        __syncthreads();
        // k 昇順のアキュムレート (出力 1 要素にアキュムレータ 1 個)。
#pragma unroll
        for (int kk = 0; kk < MIXP_BK; ++kk) {
            float wv[MIXP_RPT];
            float xv[MIXP_TPT];
#pragma unroll
            for (int r = 0; r < MIXP_RPT; ++r) {
                wv[r] = ws[tx * MIXP_RPT + r][kk];
            }
#pragma unroll
            for (int t = 0; t < MIXP_TPT; ++t) {
                xv[t] = xs[ty * MIXP_TPT + t][kk];
            }
#pragma unroll
            for (int r = 0; r < MIXP_RPT; ++r) {
#pragma unroll
                for (int t = 0; t < MIXP_TPT; ++t) {
                    acc[r][t] = __fmaf_rn(wv[r], xv[t], acc[r][t]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MIXP_RPT; ++r) {
        const int row = row0 + tx * MIXP_RPT + r;
        if (row >= n) {
            continue;
        }
#pragma unroll
        for (int t = 0; t < MIXP_TPT; ++t) {
            const int tok = tok0 + ty * MIXP_TPT + t;
            if (tok < nt) {
                y[(size_t)tok * n + row] = acc[r][t];
            }
        }
    }
}
