// task 225: prefill の ssm beta/alpha 射影 (F32 重み) をデバイスで回す GEMM。
//
// PTX へのコンパイル (build_ptx.sh の `gemm_ba225`):
//
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_gemm_ba225.cu \
//        -o kernels_gemm_ba225.ptx
//
// なぜ別カーネルか (lab t222c の [q4ledger-pf]、14,018 トークン prefill):
//   `sub ssm beta_pf` は 612.7 ms。中身はホストの `gemv_batch_split` (F32 重み
//   2,560 x 96 を 16 スレッドで回す) で、実効 0.4 TFLOP/s。同じ層の qkv GEMM は
//   93.6 TFLOP/s なので 200 倍の差。既存のデバイス版 (`dno_gemm_f32_mt` を
//   `DENNO_PF_SSM_GEMM_GPU` で使う経路) は lab t183a で 712 ms と**ホストより
//   遅い** (1 ブロック 16 行 x 32 トークンしかなく、n = 96 では 6 x 16 = 96
//   ブロックで 2 アキュムレータ/スレッド。レイテンシ律速)。
//
//   このカーネルは同じ形 (m トークン x k=2,560 x n=96) を
//     - 1 ブロック = 48 行 x 64 トークン (2 ブロックで 96 行)
//     - 1 スレッド = 3 行 x 4 トークン = 12 アキュムレータ
//     - k は 32 要素ずつ共有メモリへ積む (重み 48x33 / 入力 64x33 の転置置き)
//     - 出力 1 要素 1 アキュムレータ・k 昇順の `__fmaf_rn` (決定論)
//   で回す。n が小さいので N 方向に分割できないが、48 ブロック (14k の 1 チャンク
//   1,024 トークンで 32 ブロック) が立つ。
//
// 数値: F32 の積和を k 昇順に `__fmaf_rn` で積む。ホストの `gemv_batch_split` は
//   mul + add の逐次和なので、丸めだけが違う (FMA 融合とタイル順)。演算の意味は
//   同じで、決定論 (固定順) は保つ。
//
// 起動: grid = (ceil(nt/64), ceil(n/48), 1) / block = (256, 1, 1) / smem 21.1 KB。
#define BA225_ROWS 48          // 1 ブロックの行数
#define BA225_TOKS 64          // 1 ブロックのトークン数
#define BA225_KTILE 32         // k のタイル
#define BA225_THREADS 256
#define BA225_RPT 3            // 1 スレッドの行数 (48 = 16 スレッド x 3)
#define BA225_TPT 4            // 1 スレッドのトークン数 (64 = 16 スレッド x 4)
#define BA225_WTILE (BA225_ROWS * BA225_KTILE)   // 1536 (256 x 6)
#define BA225_XTILE (BA225_TOKS * BA225_KTILE)   // 2048 (256 x 8)

extern "C" __global__ void __launch_bounds__(BA225_THREADS, 2)
dno_gemm_f32_ba(
    const float* __restrict__ w,   // [n, k] 行優先 (beta 行 + alpha 行)
    const float* __restrict__ x,   // [nt, k] 行優先
    float* __restrict__ y,         // [nt, n] 行優先
    int nt,
    int k,
    int n)
{
    __shared__ float ws[BA225_KTILE][BA225_ROWS + 1];  // [kk][row]
    __shared__ float xs[BA225_KTILE][BA225_TOKS + 1];  // [kk][tok]

    const int tx = threadIdx.x & 15;   // トークン群 (4 トークン)
    const int ty = threadIdx.x >> 4;   // 行群 (3 行)
    const int row0 = blockIdx.y * BA225_ROWS + ty * BA225_RPT;
    const int tok0 = blockIdx.x * BA225_TOKS + tx * BA225_TPT;

    float acc[BA225_RPT][BA225_TPT];
#pragma unroll
    for (int i = 0; i < BA225_RPT; ++i) {
#pragma unroll
        for (int j = 0; j < BA225_TPT; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += BA225_KTILE) {
        // 重みタイル: 48 行 x 32 列。読みは行優先で coalesced、置きは転置。
#pragma unroll
        for (int j = 0; j < BA225_WTILE / BA225_THREADS; ++j) {
            const int id = threadIdx.x + j * BA225_THREADS;
            const int rr = id >> 5;
            const int kk = id & 31;
            const int gr = blockIdx.y * BA225_ROWS + rr;
            const int gk = k0 + kk;
            ws[kk][rr] = (gr < n && gk < k) ? w[(size_t)gr * k + gk] : 0.0f;
        }
        // 入力タイル: 64 トークン x 32 列。
#pragma unroll
        for (int j = 0; j < BA225_XTILE / BA225_THREADS; ++j) {
            const int id = threadIdx.x + j * BA225_THREADS;
            const int tt = id >> 5;
            const int kk = id & 31;
            const int gt = blockIdx.x * BA225_TOKS + tt;
            const int gk = k0 + kk;
            xs[kk][tt] = (gt < nt && gk < k) ? x[(size_t)gt * k + gk] : 0.0f;
        }
        __syncthreads();
        // 12 アキュムレータ x 32 kk。重みは同じ ty のスレッドで共通 (broadcast)、
        // 入力は同じ tx のスレッドで共通。
#pragma unroll
        for (int kk = 0; kk < BA225_KTILE; ++kk) {
            float wv[BA225_RPT];
            float xv[BA225_TPT];
#pragma unroll
            for (int i = 0; i < BA225_RPT; ++i) {
                wv[i] = ws[kk][ty * BA225_RPT + i];
            }
#pragma unroll
            for (int j = 0; j < BA225_TPT; ++j) {
                xv[j] = xs[kk][tx * BA225_TPT + j];
            }
#pragma unroll
            for (int i = 0; i < BA225_RPT; ++i) {
#pragma unroll
                for (int j = 0; j < BA225_TPT; ++j) {
                    acc[i][j] = __fmaf_rn(wv[i], xv[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < BA225_RPT; ++i) {
        const int r = row0 + i;
        if (r >= n) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < BA225_TPT; ++j) {
            const int t = tok0 + j;
            if (t < nt) {
                y[(size_t)t * n + r] = acc[i][j];
            }
        }
    }
}
