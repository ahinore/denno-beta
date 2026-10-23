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

// s132 (Task B1): 複数行を 1 block で受ける GEMV。`dno_gemv_q4_0` の grid は
// (n/8, m) なので、m 行のとき同じ 100 MB の重みを m 回読み直していた (実測
// m=2 が m=1 のちょうど 2 倍・m=8 がちょうど 8 倍で、いずれも 240 GB/s)。
// ここでは 1 block が n 方向 8 列 (warp 1 本 = 1 列) **と m 方向 mr 行**を受け、
// 重みブロックの読み (ws[b] / wn[b]) を 1 回で済ませて mr 行に使い回す。
// 起動側は grid = (n/8, ceil(m/mr))・mr = min(m, 8)。
//
// 数値: 1 block 内の重み読みが 1 回になるだけで、行 i の累積列は
//   レーンが受持つブロックを昇順に 1 本の鎖で足す → warp 内 shfl 縮約
// の両方が `dno_gemv_q4_0` と同一なので、出力は行ごとにビット一致する
// (mr = 1 で比較すると全要素が同じビットになる。CPU 参照の
// `matvec_rows_bt_q4_into` とも同じ R2 以内のまま)。
// 積と加算は `__fmul_rn` / `__fadd_rn` を明示し、コンパイル時の FMA 縮約の
// 有無で足し方が変わらないようにする (`dno_gemv_q4_0` は素の `acc += a * w`)。
extern "C" __global__ void dno_gemv_q4_0mr(
    const float* __restrict__ a,    // [m, k]
    const uint4* __restrict__ nib,  // [n, nb]
    const float* __restrict__ sc,   // [n, nb]
    float* __restrict__ out,        // [m, n]
    int m, int k, int n, int mr)
{
    const int nb = k >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;   // 0..7 (blockDim.x = 256)
    const int gj = blockIdx.x * 8 + warp;
    const int i0 = blockIdx.y * mr;      // この block の先頭行
    if (gj >= n || i0 >= m || nb <= 0) return;
    // mr は起動側の min(m, 8) 由来で 8 以下だが、引数なので明示的に落とす。
    const int rows = min(mr, m - i0);
    if (rows <= 0) return;

    const uint4* wn = nib + (size_t)gj * (size_t)nb;
    const float* ws = sc + (size_t)gj * (size_t)nb;

    // 行ごとの独立アキュムレータ (行 r は out の行 i0 + r)。
    float acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) acc[r] = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        // 重みは 1 回だけ読む (s132 の要点)。
        const float d = ws[b];
        const uint4 v = wn[b];
        const unsigned int words[4] = {v.x, v.y, v.z, v.w};
        const int base = b << 5;
        // 実行する行数でループ境界を切る (tail block で余分な A 読みをしない)。
        // 行ごとの積算は `dno_gemv_q4_0` の `acc` と同じ「レーンが受け持つブロック
        // を順に足していく 1 本の鎖」にする。ブロックごとの部分和を別に作って
        // 後から足す形にすると最終和が変わらずとも中間の丸めが変わり、レーンが
        // 2 ブロック以上を持つ形状 (k >= 2048) でビット一致しなくなるので不可。
        for (int r = 0; r < rows; r++) {
            const float* ai = a + (size_t)(i0 + r) * (size_t)k;
            float s = acc[r];
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const unsigned int w32 = words[wi];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int qlo = (int)((w32 >> (8 * e)) & 0xFu) - 8;
                    const int qhi = (int)((w32 >> (8 * e + 4)) & 0xFu) - 8;
                    s = __fadd_rn(s, __fmul_rn(ai[base + wi * 4 + e], __fmul_rn(d, (float)qlo)));
                    s = __fadd_rn(s, __fmul_rn(ai[base + 16 + wi * 4 + e], __fmul_rn(d, (float)qhi)));
                }
            }
            acc[r] = s;
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] = __fadd_rn(acc[r], __shfl_down_sync(0xffffffffu, acc[r], off));
        }
        if (lane == 0 && r < rows) {
            out[(size_t)(i0 + r) * (size_t)n + gj] = acc[r];
        }
    }
}

// ---------------------------------------------------------------------------
// s133 (Task B1, attempt 2): 列ブロック版 GEMV。
//
// s132 の `dno_gemv_q4_0mr` は重みの読みを行で使い回したが、実測では何も速く
// ならなかった (m=2 も m=8 も 1 行あたり約 0.40 ms のまま)。理由は帯域ではなく
// 命令発行量で、1 warp = 1 出力列の形では 1 MAC あたり
//   A のロード 1 + FMA (mul+add) 1
// の 2 命令が必ず要る (m=1 は 5120 本の依存する加算鎖を 34816 warp が持つ形)。
// ここでは warp が A の 8 値をレジスタに 1 回だけ読み、その 8 値を C 列ぶんの
// 重みに使い回す。命令数は 1 MAC あたり 1 + 1/C になる。
//
// 形: grid = (ceil(n / (8*C)), m)、block = 256 thread (8 warp)。
//   warp w は列 8 本 (gj = blockIdx.x * 8C + w*C + c, c = 0..C-1) を受け持つ。
//   列の持ち方は 1 warp 内で C 本なので、warp 数 (8) は変えない。
//   1 block の列数は 8*C に広がる (C=4 で 32 列/block)。
//
// 数値: 1 列ぶんの累積順は `dno_gemv_q4_0` / `dno_gemv_q4_0mr` と同一。
//   lane がブロック b (b = lane, lane+32, ... 昇順) を受け持ち、
//   各ブロック内の 32 要素を「wi 昇順 -> e 昇順 -> 低ニブル, 高ニブル」の順に
//   1 本の鎖で足し、最後に warp 内 shfl_down (16,8,4,2,1) で縮約する。
//   積と加算は `__fmul_rn` / `__fadd_rn` を明示する (FMA 縮約をさせない)。
//   縮約は c ごとに 1 本ずつ回す (同時に C 本回すと順序が c 間で混ざり、
//   `__shfl_down_sync` の full mask で他の列の値を読むことになるため)。
//
// A の読みは `ld.global.nc` (__ldg) で 1 ブロック 8 値 1 回。列方向には
// 隣の warp と同じ A を読むので L1/L2 で当たる。
//
// テンプレートにするのは C ごとに `acc` がレジスタへ載るか (local memory へ
// spill しないか) が変わるため。s132 の `dno_gemv_q4_0mr` は acc[8] が
// local memory へ落ちていた (コミット済み PTX に `ld.local` 9 / `st.local` 11
// があり、行ごとの鎖が毎回メモリ経由になっていた)。ここで生かすのは C 本の
// アキュムレータ、A の 32 値 (av)、重み 1 ブロックぶんの words[4] だけ。
template <int C>
__device__ __forceinline__ void dno_gemv_q4_0_cols(
    const float* __restrict__ a,    // [m, k]
    const uint4* __restrict__ nib,  // [n, nb]
    const float* __restrict__ sc,   // [n, nb]
    float* __restrict__ out,        // [m, n]
    int m, int k, int n)
{
    const int nb = k >> 5;                  // 1 ブロック = 32 要素
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;      // 0..7 (blockDim.x = 256)
    const int j0 = blockIdx.x * (8 * C) + warp * C;
    const int i = blockIdx.y;
    if (j0 >= n || i >= m || nb <= 0) return;

    const float* ai = a + (size_t)i * (size_t)k;

    // 列ごとの独立アキュムレータ。C 列とも「lane が受持つブロックを昇順に
    // 1 本の鎖で足す」形を崩さない。
    float acc[C];
    #pragma unroll
    for (int c = 0; c < C; c++) acc[c] = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const int base = b << 5;
        // A の 32 要素は C 列で共通。レジスタへ 1 回だけ読む (これが s133 の
        // 要点: 従来は列ごとに同じ A を読み直していた)。
        // lane が持つのはブロック b の先頭 32 要素なので、16 B 読み x8 で
        // ちょうど 32 値。各 e の反復で使う 4 値が 1 回の ld.global.v4.f32 に
        // なる並び (av[4t..4t+3]) で置く。
        float av[32];
        {
            const uint4* a4 = (const uint4*)(ai + base);
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                const uint4 v4 = __ldg(&a4[t]);
                av[t * 4 + 0] = __uint_as_float(v4.x);
                av[t * 4 + 1] = __uint_as_float(v4.y);
                av[t * 4 + 2] = __uint_as_float(v4.z);
                av[t * 4 + 3] = __uint_as_float(v4.w);
            }
        }

        // 重みは列ごとに違うので c ループの内側で読む。1 列 = 18 B (16 B
        // ニブル + f32 スケール) で、C 列でも 1 行あたり nb ブロック 1 回。
        #pragma unroll
        for (int c = 0; c < C; c++) {
            const int gj = j0 + c;
            if (gj >= n) break;
            const size_t row = (size_t)gj * (size_t)nb;
            const float d = sc[row + b];
            const uint4 v = nib[row + b];
            const unsigned int words[4] = {v.x, v.y, v.z, v.w};
            float s = acc[c];
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                const unsigned int w32 = words[wi];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int qlo = (int)((w32 >> (8 * e)) & 0xFu) - 8;
                    const int qhi = (int)((w32 >> (8 * e + 4)) & 0xFu) - 8;
                    s = __fadd_rn(s, __fmul_rn(av[wi * 4 + e], __fmul_rn(d, (float)qlo)));
                    s = __fadd_rn(s, __fmul_rn(av[16 + wi * 4 + e], __fmul_rn(d, (float)qhi)));
                }
            }
            acc[c] = s;
        }
    }

    // warp 内縮約は列ごとに 1 本ずつ。C 列を同時に shfl すると、off 段の
    // `__shfl_down_sync` が full mask で他の列の値まで読む形になり、
    // 「列ごとに独立な 16,8,4,2,1 の木」という契約が読みにくくなるため。
    #pragma unroll
    for (int c = 0; c < C; c++) {
        const int gj = j0 + c;
        if (gj >= n) break;
        float s = acc[c];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
        }
        if (lane == 0) {
            out[(size_t)i * (size_t)n + gj] = s;
        }
    }
}

#define Q4_GEMV_C 4
// 1 warp が A の 32 値 (18 ブロック分のうち 1 ブロック) をレジスタで持ち、
// C 列ぶんの重みを回す。acc は C 本 + av 32 本 + words 4 本 = C 付近で
// 40 本強。ローカルメモリへ落ちると (s132 の acc[8] は ld.local 9 回・
// st.local 11 回だった) 命令数が増えて狙いが消えるので、ブロック数を
// 抑えてレジスタを確保する。
#define Q4_GEMV_LAUNCH_BOUNDS __launch_bounds__(256, 4)
// C の候補 (2 / 4 / 8) を同じファイルで比較するための入口。
// 既定は C = Q4_GEMV_C、2 と 8 は DENNO 側の定数を変えて差し替える。
extern "C" __global__ Q4_GEMV_LAUNCH_BOUNDS void dno_gemv_q4_0c2(
    const float* __restrict__ a, const uint4* __restrict__ nib,
    const float* __restrict__ sc, float* __restrict__ out,
    int m, int k, int n) {
    dno_gemv_q4_0_cols<2>(a, nib, sc, out, m, k, n);
}
extern "C" __global__ Q4_GEMV_LAUNCH_BOUNDS void dno_gemv_q4_0c4(
    const float* __restrict__ a, const uint4* __restrict__ nib,
    const float* __restrict__ sc, float* __restrict__ out,
    int m, int k, int n) {
    dno_gemv_q4_0_cols<4>(a, nib, sc, out, m, k, n);
}
extern "C" __global__ Q4_GEMV_LAUNCH_BOUNDS void dno_gemv_q4_0c8(
    const float* __restrict__ a, const uint4* __restrict__ nib,
    const float* __restrict__ sc, float* __restrict__ out,
    int m, int k, int n) {
    dno_gemv_q4_0_cols<8>(a, nib, sc, out, m, k, n);
}

// prefill / バッチ用のタイル GEMM。block = (16,8) = 128 thread、
// 1 ブロック = m 方向 64 行 x n 方向 128 列。1 thread は 8 行 x 8 列 (64 個の acc) を
// 担当する。共有メモリは A タイル (64x32 f32) と B タイル (128x33 f32・展開済み) で
// 25 KB。B の行長を 33 にしているのは bank 衝突の回避 (下のコメント参照)。
//
// s249 (P52): 1 thread のタイルを 4x8 -> 8x8 に広げる。1 要素あたりの共有メモリ
// ロード数が 32 FFMA / 12 load から 64 FFMA / 16 load に下がる (2.7 -> 4.0)。
// **各出力要素の累積順 (b 昇順 -> c 昇順) は不変**なのでビット一致は保たれる。
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
    const int ty = threadIdx.y;   // 0..7  → 行 8 本 (ty*8 + r)
    const int tid = ty * 16 + tx; // 0..127 → 展開する B の 1 列 (32 要素すべて)

    float acc[8][8];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int v = 0; v < 8; v++) acc[r][v] = 0.0f;

    for (int b = 0; b < nb; b++) {
        // A タイル (64 行 x 32 要素)。thread は 8 行 x 2 要素を読む。
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            const int gi = i0 + ty * 8 + r;
            if (gi < m) {
                const float* src = a + (size_t)gi * (size_t)k + (b << 5);
                sA[ty * 8 + r][tx] = src[tx];
                sA[ty * 8 + r][tx + 16] = src[tx + 16];
            } else {
                sA[ty * 8 + r][tx] = 0.0f;
                sA[ty * 8 + r][tx + 16] = 0.0f;
            }
        }
        // B タイル (128 列 x 32 要素) をニブルから展開する。thread は 1 列の
        // 32 要素を担当 (tid がそのまま列)。1 語 = 下位 4 bit 群が要素
        // wi*4+e、上位 4 bit 群が要素 16+wi*4+e (旧 2 分割版と同じ対応)。
        {
            const int col = tid;
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
                    sB[col][wi * 4 + e] = d * (float)((int)((w32 >> (8 * e)) & 0xFu) - 8);
                    sB[col][16 + wi * 4 + e] =
                        d * (float)((int)((w32 >> (8 * e + 4)) & 0xFu) - 8);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 32; c++) {
            float av[8];
            float bv[8];
            #pragma unroll
            for (int r = 0; r < 8; r++) av[r] = sA[ty * 8 + r][c];
            #pragma unroll
            for (int j = 0; j < 8; j++) bv[j] = sB[tx + 16 * j][c];
            #pragma unroll
            for (int r = 0; r < 8; r++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[r][j] += av[r] * bv[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        const int gi = i0 + ty * 8 + r;
        if (gi >= m) continue;
        float* row = out + (size_t)gi * (size_t)n;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int gj = j0 + tx + 16 * j;
            if (gj < n) row[gj] = acc[r][j];
        }
    }
}