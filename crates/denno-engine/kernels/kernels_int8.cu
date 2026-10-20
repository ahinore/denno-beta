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
// P17i: tiled int8 GEMM (prefill 用の大きい m)。dp4a。重みはタイルごとに共有メモリ
// へ一度読めばよいので、出力要素ごとに重み行を再読する GEMV 型カーネル (上記) と
// 異なり大 m で効く。
// P17k ステップ 6: 128x128 タイル (1 thread 8x8 出力) + smem ストライドパッド。
// 旧版 (64x64・ストライド 32) は sB 行読み (隣接 thread が 8 行差) が
// ストライド 32B = 32 の倍数で同一 bank に衝突し 16-way serialized していた。
// ストライドを GEMM_TK+1 (33B) にすると隣接 thread の 8 行差 = 264B = 66 words で
// bank が 2 ずつ離れ conflict 無し。タイル 128 で A 再読みも半減。
// ビット一致: 整数累積は結合則が成立するためタイル化/順序に依存しない。f32 側は
// GEMV カーネルと同一の ((dot as f32) * ascale) * scale[j] の順。k が 4 の倍数で
// ない場合もタイル境界の 0 パディングで正しく処理される (0 は積に寄与しない)。
//
// s74 (T6 候補 1 / 数値緩和クラス R2): 1 thread の出力を 8x8 → **8x16** に拡大し、
// 1 ブロック = 128x256 出力・64 thread (block = 16x4・tile を 2 枚並べる)。
// 旧 128x128・256 thread では 1 回の内積 (dp4a) に対して smem からの B ロードが
// 1/8 回分しか使い回せていなかった (dp4a 144 = b64 ロード 16 + a32 4・実効
// 11 TOPS 前後)。B を 16 個使うようにして dp4a/B ロード比を 8 → 16 に上げる。
// - k 方向 4 要素は i32 にパックしたまま (s32 アキュムレータを直接用いる)。
// - j 方向の 2 枚目 (sB[128..256]) は**重み側だけ**を pad する: ホストが n を 256 の
//   倍数に切り上げたラッパ行を確保して 0 埋めする (gemm_pad_apply。i8 の 0 は寄与
//   しないので dot は厳密に同じ)。a4 を 2 タイルで共有するため、A 側の pad 値
//   (行方向のゼロ) が偽の寄与を生まない。
// - アキュムレータが 128 個になるので、旧版で spilling していた他のレジスタ
//   (a/b パック・アドレス計算・unroll 展開) をすべてロールバックした。この
//   kernel は次の点で**旧版と命令列が異なる**: 端 (i0+16) の境界を tag 方式で
//   判定・端行 (if (i0 + 16 < m)) での二重バッファ確保なし・epilogue の書き込みは
//   1 thread あたり 16 列連続。いずれも f32 の演算列 ((float)dot * ascale * scale[j])
//   と整数累積を変えないので、**CPU の同じ整数 dot からの結果と同じビット**になる
//   (gemm_pad_apply を純粋関数にして Rust 側でテストする)。
// s75 (不具合修正): 共有ステージの行ストライドは **4 バイト境界に揃える**。
// s74 の `GEMM_SP = GEMM_TK + 1` (= 33) は奇数で、下記のロード/参照が
//   *(int*)&sA[tid][c] / *(const int*)&sA[..][c0] / *(int*)&sB[r][c]
// と 4 バイトのポインタキャストをしているため、行 r が進むごとに実アドレスが
// 33 の倍数だけずれて未アラインになり、`Invalid __shared__ write` →
// sticky な CUDA error 716 (恒久 CPU フォールバック) を起こした。
// 32 のまま (パディング無し) でもアラインはするが、隣接 thread が 8 行差
// (8*32B = 256B = 64 words、64 mod 32 = 0) で全行が同一 bank に衝突する。
// 36B (= 9 words) なら 8 行差 = 288B = 72 words、72 mod 32 = 8 で bank が
// 8 個ずつずれて 8 行の並列読み出しで衝突しない。したがって
// **4 の倍数かつ 32 の倍数でない** 値として GEMM_TK + 4 を採る。
#define GEMM_TILE 128
#define GEMM_TILE_N 256
#define GEMM_TK 32
// s75: block 形状と 1 thread の出力タイル。block = (GEMM_BX, GEMM_BY) =
// (16,16) = 256 thread。1 thread 8 行 x 16 列なので
//   GEMM_BY * 8  = 128 = GEMM_TILE   (i 方向を過不足なく覆う)
//   GEMM_BX * 16 = 256 = GEMM_TILE_N (j 方向を過不足なく覆う)
// s74 は block を (16,4) = 64 thread に落としたが、これは 4*8 = 32 行しか
// 覆わず GEMM_TILE (128) と食い違っていた (gridDim.x = m.div_ceil(128) で
// 1 ブロック 128 行を進めるため、タイル内 32..127 行が誰にも書かれない)。
#define GEMM_BX 16
#define GEMM_BY 16
#define GEMM_SP (GEMM_TK + 4)
// 共有配列への i32 アクセスはすべて 4 バイト境界を前提にする (ホスト側の
// `*(int*)&sA[..]` と等価な PTX を保つ)。
static_assert(GEMM_SP % 4 == 0, "shared stride must be 4-byte aligned");
static_assert(GEMM_TK % 4 == 0, "k tile must be a multiple of 4 (dp4a pack)");
// bank conflict 回避の根拠 (8 行差が 32 words の倍数にならないこと) も固定する。
static_assert((8 * GEMM_SP / 4) % 32 != 0, "8-row stride must not alias one bank");
// s75: block 形状とタイル寸法の整合 (上記の行カバレッジ不具合の再発防止)。
static_assert(GEMM_BX * 16 == GEMM_TILE_N, "block.x * per-thread cols != j tile");
static_assert(GEMM_BY * 8 == GEMM_TILE, "block.y * per-thread rows != i tile");
static_assert(GEMM_BX * GEMM_BY <= 1024, "block too large");
extern "C" __global__ void dno_gemm_i8(
    const signed char* __restrict__ aq,   // [m,k]
    const float* __restrict__ ascale,     // [m]
    const signed char* __restrict__ w,    // [n,k] row-major (n は 256 の倍数に pad 済み)
    const float* __restrict__ scale,      // [n]
    float* __restrict__ out,              // [m,n] row-major
    int m, int k, int n)
{
    const int tx = threadIdx.x; // 0..15 → j 方向 16 列ずつ (16*16 = 256 = GEMM_TILE_N)
    const int ty = threadIdx.y; // 0..15 → i 方向 8 行ずつ (16*8  = 128 = GEMM_TILE)
    const int tid = ty * GEMM_BX + tx;
    const int i0 = blockIdx.x * GEMM_TILE;
    const int j0 = blockIdx.y * GEMM_TILE_N;

    // 1 thread = 8 行 x 16 列。行方向は 8 行連続の各行で独立した i32 アキュムレータ、
    // 列方向は 2 タイル (8 列ずつ) を各々独立に持つ (整数なので分割は厳密)。
    int acc[8][16];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        #pragma unroll
        for (int v = 0; v < 16; v++) {
            acc[u][v] = 0;
        }
    }

    __shared__ signed char sA[GEMM_TILE][GEMM_SP];
    __shared__ signed char sB[GEMM_TILE_N][GEMM_SP];

    for (int kb = 0; kb < k; kb += GEMM_TK) {
        // A (128x32 = 4608B) と B (256x32 = 9216B) を 256 thread で協調ロード。
        // A は tid < GEMM_TILE (128) の thread が 1 行ずつ、B は全 thread が
        // 1 行ずつ (256 行 / 256 thread)。端 (gi>=m / gk>=k) は 0 埋め
        // (= 積に寄与しない)。行ストライド GEMM_SP は 4 の倍数なので u32 の
        // 境界アクセスは常にアラインされる (static_assert で強制)。
        if (tid < GEMM_TILE) {
            const int gi = i0 + tid;
            if (gi < m) {
                const signed char* src = aq + (size_t)gi * (size_t)k + kb;
                const int* src4 = (const int*)src;
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    const int c = x * 4;
                    int val = 0;
                    if (kb + c + 3 < k) {
                        val = __ldg(&src4[x]);
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            if (kb + c + e < k) val |= (int)(unsigned char)src[c + e] << (8 * e);
                        }
                    }
                    *(int*)&sA[tid][c] = val;
                }
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++) *(int*)&sA[tid][x * 4] = 0;
            }
        }
        {
            const int r = tid;                        // 0..255
            const int gj = j0 + r;
            if (gj < n) {
                const signed char* src = w + (size_t)gj * (size_t)k + kb;
                const int* src4 = (const int*)src;
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    const int c = x * 4;
                    int val = 0;
                    if (kb + c + 3 < k) {
                        val = __ldg(&src4[x]);
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            if (kb + c + e < k) val |= (int)(unsigned char)src[c + e] << (8 * e);
                        }
                    }
                    *(int*)&sB[r][c] = val;
                }
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++) *(int*)&sB[r][x * 4] = 0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int c0 = 0; c0 < GEMM_TK; c0 += 4) {
            int a4[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                a4[u] = *(const int*)&sA[ty * 8 + u][c0];
            }
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    const int b4 = *(const int*)&sB[t * GEMM_TILE + tx * 8 + v][c0];
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        acc[u][t * 8 + v] = __dp4a(a4[u], b4, acc[u][t * 8 + v]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // 列 0..7 は tx*8 + v、列 8..15 は GEMM_TILE + tx*8 + v。
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        const int gi = i0 + ty * 8 + u;
        if (gi < m) {
            const float as = ascale[gi];
            float* row = out + (size_t)gi * (size_t)n;
            #pragma unroll
            for (int v = 0; v < 16; v++) {
                const int gj = j0 + (v < 8 ? 0 : GEMM_TILE) + tx * 8 + (v & 7);
                if (gj < n) {
                    const float o = (float)acc[u][v] * as;
                    row[gj] = o * scale[gj];
                }
            }
        }
    }
}
// ---------------------------------------------------------------------------
// s123: Q4_0 ネイティブ経路。
//
// 目的: GGUF の Q4_0 ブロック (f16 scale 2B + 16B ニブル = 32 要素/18B) を
// **再量子化せずに** VRAM へ置く。int8 モードは fp32 へ展開してから 8 bit に
// 詰め直すため、元が 4.5 bit でも常に 1 要素 1 バイト (27B で 25.7 GB) を
// 積んでいた。ネイティブ経路は 13.7 GB で済む。
//
// レイアウト (アップロード時にホストで変換する SOA):
//   nib[行 j][ブロック b] = uint4 (16B・16B アライン) … 32 要素ぶんのニブル
//   sc [行 j][ブロック b] = float (元の f16 スケールを f32 へ)
// ブロック b の要素 c (0..15) は下位ニブル、c+16 は上位ニブル (ggml Q4_0)。
// デ量子化は y = d * (q - 8) (d は負になり得る)。
//
// 数値契約: 累積は f32 の逐次 (上のカーネルは -fmad=false でビルドする)。
// CPU 経路と同じ順序で足すので決定論的。CPU との差は R2 (相対 1e-4) 以内。
// ---------------------------------------------------------------------------

// decode / 小さい m 用の GEMV。1 warp = 1 出力行、1 block = 8 行。
// lane はブロックを 32 個飛ばしで担当し、最後に warp 内で縮約する。
extern "C" __global__ void dno_gemv_q4_0(
    const float* __restrict__ a,    // [m, k]
    const uint4* __restrict__ nib,  // [n, nb]
    const float* __restrict__ sc,   // [n, nb]
    float* __restrict__ out,        // [m, n]
    int m, int k, int n)
{
    const int nb = k >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;   // 0..7 (blockDim.x = 256)
    const int gj = blockIdx.x * 8 + warp;
    const int i = blockIdx.y;
    if (gj >= n || i >= m || nb <= 0) return;

    const float* ai = a + (size_t)i * (size_t)k;
    const uint4* wn = nib + (size_t)gj * (size_t)nb;
    const float* ws = sc + (size_t)gj * (size_t)nb;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float d = ws[b];
        const uint4 v = wn[b];
        const unsigned int words[4] = {v.x, v.y, v.z, v.w};
        const int base = b << 5;
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const unsigned int w32 = words[wi];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int qlo = (int)((w32 >> (8 * e)) & 0xFu) - 8;
                const int qhi = (int)((w32 >> (8 * e + 4)) & 0xFu) - 8;
                acc += ai[base + wi * 4 + e] * (d * (float)qlo);
                acc += ai[base + 16 + wi * 4 + e] * (d * (float)qhi);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[(size_t)i * (size_t)n + gj] = acc;
    }
}

// prefill / バッチ用のタイル GEMM。block = (16,16) = 256 thread、
// 1 ブロック = m 方向 64 行 x n 方向 128 列。1 thread は 4 行 x 8 列 (32 個の acc) を
// 担当する。共有メモリは A タイル (64x32 f32) と B タイル (128x33 f32・展開済み) で
// 25 KB。B の行長を 33 にしているのは bank 衝突の回避 (下のコメント参照)。
#define Q4_GM 64
#define Q4_GN 128
static_assert(Q4_GM * 32 * 4 + Q4_GN * 33 * 4 <= 48 * 1024,
              "q4 gemm shared memory must stay under 48KB (no opt-in needed)");

extern "C" __global__ void dno_gemm_q4_0(
    const float* __restrict__ a,
    const uint4* __restrict__ nib,
    const float* __restrict__ sc,
    float* __restrict__ out,
    int m, int k, int n)
{
    __shared__ float sA[Q4_GM][32];
    // +1 padding: 行長 32 だと address % 32 が常に同じ値になり、列方向に
    // 読むときに warp 全員が同じ bank を叩く (32-way conflict)。
    __shared__ float sB[Q4_GN][33];
    const int nb = k >> 5;
    const int i0 = blockIdx.x * Q4_GM;
    const int j0 = blockIdx.y * Q4_GN;
    const int tx = threadIdx.x;   // 0..15 → 列 8 本 (tx + 16*j)
    const int ty = threadIdx.y;   // 0..15 → 行 4 本 (ty*4 + r)
    const int tid = ty * 16 + tx; // 0..255 → 展開する B の 1 列 x 16 要素

    float acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int v = 0; v < 8; v++) acc[r][v] = 0.0f;

    for (int b = 0; b < nb; b++) {
        // A タイル (64 行 x 32 要素)。thread は 4 行 x 2 要素を読む。
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const int gi = i0 + ty * 4 + r;
            if (gi < m) {
                const float* src = a + (size_t)gi * (size_t)k + (b << 5);
                sA[ty * 4 + r][tx] = src[tx];
                sA[ty * 4 + r][tx + 16] = src[tx + 16];
            } else {
                sA[ty * 4 + r][tx] = 0.0f;
                sA[ty * 4 + r][tx + 16] = 0.0f;
            }
        }
        // B タイル (128 列 x 32 要素) をニブルから展開する。thread は 1 列の
        // 16 要素を担当 (tid の下位 7 bit が列、最上位 bit が下位/上位ニブルの別)。
        {
            const int col = tid & 127;
            const int half = tid >> 7;   // 0: 要素 0..15, 1: 要素 16..31
            const int gj = j0 + col;
            const bool in = gj < n;
            const float d = in ? sc[(size_t)gj * (size_t)nb + b] : 0.0f;
            const uint4 v = in ? nib[(size_t)gj * (size_t)nb + b] : make_uint4(0, 0, 0, 0);
            const unsigned int words[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const unsigned int w32 = words[wi];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int sh = 8 * e + 4 * half;
                    const int q = (int)((w32 >> sh) & 0xFu) - 8;
                    sB[col][half * 16 + wi * 4 + e] = d * (float)q;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 32; c++) {
            float av[4];
            float bv[8];
            #pragma unroll
            for (int r = 0; r < 4; r++) av[r] = sA[ty * 4 + r][c];
            #pragma unroll
            for (int j = 0; j < 8; j++) bv[j] = sB[tx + 16 * j][c];
            #pragma unroll
            for (int r = 0; r < 4; r++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[r][j] += av[r] * bv[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gi = i0 + ty * 4 + r;
        if (gi >= m) continue;
        float* row = out + (size_t)gi * (size_t)n;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int gj = j0 + tx + 16 * j;
            if (gj < n) row[gj] = acc[r][j];
        }
    }
}