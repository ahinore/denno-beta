// denno CUDA int8 kernels (P17) — CPU int8 パス (tensor.rs) とビット一致する
// カーネル群。PTX へのコンパイル:
//
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_int8.cu -o kernels_int8.ptx
//
// compute_75 PTX はドライバ JIT で以降の全アーキ (sm_86/sm_89/sm_120 を含む)
// で動く。使用命令は基本の整数/浮動小数のみ (dp4a/Tensor Core 不使用)。
//
// ビット一致の設計 (docs/HISTORY P17 参照):
// - 量子化: scale = maxabs/127 (max は順序不変で厳密)、q = round(v*inv) を
//   ±127 にクランプ。CUDA roundf は half-away-from-zero で Rust f32::round と同一。
// - 積和: i8×i8 → i32 累積。整数は結合則が成立するためレーン分割/リダクション
//   順序に出力は依存しない (CPU の SIMD カーネルと同一の厳密性)。
// - スケール: (float)dot * ascale[i] * scale[j] を左から順に f32 乗算
//   (CPU: `dot as f32 * ascale * scale[j]` と同一の結合順)。

#include <math.h>

// ---------------------------------------------------------------------------
// per-row 動的量子化: aq[i*k+d] = quantize_row_i8(a[i*k..k])、ascale[i] = scale。
// CPU: tensor.rs quantize_row_i8 と同一演算列 (maxabs → scale → inv → round → clamp)。
// 1 block per row、block 内で maxabs を共有メモリリダクション (fmaxf は順序不変)。
extern "C" __global__ void dno_quantize_rows(
    const float* __restrict__ a,
    signed char* __restrict__ aq,
    float* __restrict__ ascale,
    int m, int k)
{
    const int i = blockIdx.x;
    if (i >= m) return;
    const float* row = a + (size_t)i * (size_t)k;

    // 1) maxabs: ストライド走査 + 共有メモリリダクション
    //    (fabsf / fmaxf はいずれも丸めを伴わない厳密演算 = 走査順序に依存しない)
    extern __shared__ float smax[];
    float local = 0.0f;
    for (int d = threadIdx.x; d < k; d += blockDim.x) {
        local = fmaxf(local, fabsf(row[d]));
    }
    smax[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float maxabs = smax[0];

    // 2) scale / inv (CPU と同一の f32 除算)
    const float scale = (maxabs > 0.0f) ? maxabs / 127.0f : 1.0f;
    const float inv = 1.0f / scale;

    // 3) 量子化: round (half away from zero) → clamp ±127 → i8
    signed char* out = aq + (size_t)i * (size_t)k;
    for (int d = threadIdx.x; d < k; d += blockDim.x) {
        float t = roundf(row[d] * inv);
        t = fminf(fmaxf(t, -127.0f), 127.0f);
        out[d] = (signed char)t;
    }
    if (threadIdx.x == 0) {
        ascale[i] = scale;
    }
}

// ---------------------------------------------------------------------------
// int8 GEMV/GEMM (nn.Linear 形式の重み q: [n,k] row-major):
//   out[i*n+j] = (float)(Σ_d aq[i*k+d] * q[j*k+d]) * ascale[i] * scale[j]
// 1 warp per 出力要素 (i,j)。レーンが k 方向をストライド走査 (coalesced) し、
// i32 を warp リダクション (整数なので順序不変 = CPU とビット一致)。
extern "C" __global__ void dno_matvec_rows_bt_i8(
    const signed char* __restrict__ aq,
    const float* __restrict__ ascale,
    const signed char* __restrict__ q,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int m, int k, int n)
{
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = (int)(threadIdx.x >> 5);
    const long long total = (long long)m * (long long)n;
    const long long stride = (long long)gridDim.x * warps_per_block;

    for (long long idx = (long long)blockIdx.x * warps_per_block + warp_in_block;
         idx < total;
         idx += stride)
    {
        const int i = (int)(idx / n);
        const int j = (int)(idx % n);
        const signed char* arow = aq + (size_t)i * (size_t)k;
        const signed char* wrow = q + (size_t)j * (size_t)k;

        // i32 累積: |部分和| <= 16129*k (k <= 133,169 で i32 内。実モデルの最大 k は
        // Qwen2.5-1.5B の中間層 8960 で十分下)。
        int acc = 0;
        for (int d = lane; d < k; d += 32) {
            acc += (int)arow[d] * (int)wrow[d];
        }
        // warp リダクション (整数加算は結合則が成立するので順序は結果に影響しない)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            // CPU と同一の f32 乗算順: ((dot as f32) * ascale) * scale[j]
            const float o = (float)acc * ascale[i];
            out[idx] = o * scale[j];
        }
    }
}

// ---------------------------------------------------------------------------
// P17i: tiled int8 GEMM (prefill 用の大きい m)。64x64 タイル・1 スレッド 4x4
// 出力・dp4a。重みはタイルごとに共有メモリへ一度読めばよいので、出力要素ごとに
// 重み行を再読する GEMV 型カーネル (上記) と異なり大 m で効く。
// ビット一致: 整数累積は結合則が成立するためタイル化/順序に依存しない。f32 側は
// GEMV カーネルと同一の ((dot as f32) * ascale) * scale[j] の順。k が 4 の倍数で
// ない場合もタイル境界の 0 パディングで正しく処理される (0 は積に寄与しない)。
#define GEMM_TILE 64
#define GEMM_TK 32
extern "C" __global__ void dno_gemm_i8(
    const signed char* __restrict__ aq,   // [m,k]
    const float* __restrict__ ascale,     // [m]
    const signed char* __restrict__ w,    // [n,k] row-major
    const float* __restrict__ scale,      // [n]
    float* __restrict__ out,              // [m,n] row-major
    int m, int k, int n)
{
    const int tx = threadIdx.x; // 0..15 → j 方向 4 列ずつ
    const int ty = threadIdx.y; // 0..15 → i 方向 4 行ずつ
    const int tid = ty * 16 + tx;
    const int i0 = blockIdx.x * GEMM_TILE;
    const int j0 = blockIdx.y * GEMM_TILE;

    int acc[4][4];
    #pragma unroll
    for (int u = 0; u < 4; u++) {
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            acc[u][v] = 0;
        }
    }

    __shared__ signed char sA[GEMM_TILE][GEMM_TK];
    __shared__ signed char sB[GEMM_TILE][GEMM_TK];

    for (int kb = 0; kb < k; kb += GEMM_TK) {
        // A/B タイル (各 64x32 = 2048B) を 256 thread で共有メモリへ
        for (int t = tid; t < GEMM_TILE * GEMM_TK; t += 256) {
            const int r = t / GEMM_TK;
            const int c = t - r * GEMM_TK;
            const int gi = i0 + r;
            const int gk = kb + c;
            sA[r][c] = (gi < m && gk < k) ? aq[(size_t)gi * (size_t)k + gk] : (signed char)0;
        }
        for (int t = tid; t < GEMM_TILE * GEMM_TK; t += 256) {
            const int r = t / GEMM_TK;
            const int c = t - r * GEMM_TK;
            const int gj = j0 + r;
            const int gk = kb + c;
            sB[r][c] = (gj < n && gk < k) ? w[(size_t)gj * (size_t)k + gk] : (signed char)0;
        }
        __syncthreads();
        #pragma unroll
        for (int c0 = 0; c0 < GEMM_TK; c0 += 4) {
            int a4[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const signed char* p = &sA[ty * 4 + u][c0];
                a4[u] = ((int)(unsigned char)p[0]) | (((int)(unsigned char)p[1]) << 8)
                      | (((int)(unsigned char)p[2]) << 16) | (((int)(unsigned char)p[3]) << 24);
            }
            #pragma unroll
            for (int v = 0; v < 4; v++) {
                const signed char* p = &sB[tx * 4 + v][c0];
                const int b4 = ((int)(unsigned char)p[0]) | (((int)(unsigned char)p[1]) << 8)
                             | (((int)(unsigned char)p[2]) << 16) | (((int)(unsigned char)p[3]) << 24);
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    acc[u][v] = __dp4a(a4[u], b4, acc[u][v]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < 4; u++) {
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const int gi = i0 + ty * 4 + u;
            const int gj = j0 + tx * 4 + v;
            if (gi < m && gj < n) {
                const float o = (float)acc[u][v] * ascale[gi];
                out[(size_t)gi * (size_t)n + gj] = o * scale[gj];
            }
        }
    }
}
