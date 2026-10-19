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
// s114 (P1-4 候補 1 の作り直し): gate/up の int8 GEMM + swiglu (epilogue 融合)。
//
// 置換対象 (chain_mlp_prefill の 2 ディスパッチ):
//   1) dno_gemm_i8 で gate/up を CH_GA [m, 2ff] (行内 [gate | up]) に書き出す
//   2) dno_swiglu_int で CH_GA を読み直して CH_ACT [m, ff] を作る
// を 1 本にする。CH_GA (m*2ff*4B) のグローバル書き出しと読み直しが丸ごと消える。
//
// 設計 (s113 の失敗 = gate と up の組が別 block に落ちた、の作り直し):
//   出力タイルは act [m, ff] の [128 行, 256 列] 1 枚を 1 ブロックが担当する。
//   - 1 ブロックが up 重み 256 行 (sB) と gate 重み 256 行 (sG) の両方を共有
//     メモリへ積み、accU[8][16] と accG[8][16] を並べて持つ。
//   - epilogue は同じ thread の同じ (u,v) で swiglu する: accU[u][v] と
//     accG[u][v] は同じ (行 u, 列 f) の up / gate なので、パート境界 (ff) と
//     タイル境界の対応を一切考えなくてよい -> 任意の ff で成立する
//     (ff % 256 == 0 という制約は無い)。
//   - sG は動的共有メモリに置く (静的 3 枚 = sA + sB + sG は 48KB 既定枠を
//     超えるため)。必要量 = GEMM_TILE_N * GEMM_SP = 9,216B。
//
// 数値 (2 段経路とのビット一致の根拠 - すべて既存カーネルと同一の演算列):
// - 整数累積: dno_gemm_i8 と同じタイル/内側ループ/__dp4a。整数は結合則が成立する
//   ので、accU / accG を分けても dot は厳密に同じ。
// - スケール復元: dno_gemm_i8 と同じく (float)acc * ascale[gi] の後に
//   * scale[gj] (中間値も演算順も同一)。
// - swiglu: dno_swiglu_int と同一
//     e = dno_expf(-g); s = __fdiv_rn(g, __fadd_rn(1.0f, e)); out = __fmul_rn(s, u)
//   -fmad=false なので独立命令のまま (縮約されない)。FMA 化・近似はしない。
// - (float)acc を経由しても値は変わらない: i32 の |acc| <= 127*127*k <= 2^24 は
//   f32 で厳密 (k <= 約 133k)。2 段経路も一度 f32 を経てから swiglu している。
// - 添字: dno_swiglu_int の gateup[r*width + f] / gateup[r*width + ff + f]
//   (width = 2ff) は、act[r*ff + f] と対で同じ行・同じ列 f を指す。
// - dno_expf は kernels_elem.cu の定義を逐語コピーしている (あちらはモジュール
//   内で完結し PTX に外部可視シンボルが無いため別 TU から共有できない)。コピーの
//   ドリフトは Rust 側のテスト fused_swiglu_expf_copy_matches_elem_source が
//   ソーステキストの突き合わせで検出する。
//
// 重み/スケール (取り違え防止):
//   w / scale は gate_up の連結 (行順 [gate 全行 | up 全行]) をそのまま渡す。
//   列 f の gate は w[f] / scale[f]、up は w[ff+f] / scale[ff+f]。
//   n = 2*ff より後ろのパディング行 (gemm_pad_apply) は 0 で、gj < n の判定で弾く。
//
// dno_gemm_i8 は一切変更しない (純追加)。decode / 単一トークン経路 / down_proj /
// chain_swiglu_rows は不変。

// --- dno_expf 逐語コピー (出典: kernels_elem.cu の N_TABLE 〜 dno_expf。
//     定数表とヘルパを含む。この範囲は sources とバイト一致させる) ---
#define N_TABLE 32  // EXP2F_TABLE_BITS = 5

// glibc sysdeps/ieee754/flt-32/e_exp2f_data.c (__exp2f_data)
// (constant memory ではなく通常の device メモリ: warp 内でインデックスが
//  分岐すると constant cache が直列化されるため。256B なので L1 で十分)
__device__ uint64_t DNO_EXP2_TAB[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
    0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
    0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
    0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
    0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
    0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
    0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
    0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
    0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
// poly_scaled[i] = poly[i] / N^(3-i), N = 32
__device__ double DNO_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_EXP_SHIFT = 0x1.8p+52;  // 1.5 * 2^52

__device__ __forceinline__ uint64_t dno_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_asuint(float f) {
    return __float_as_uint(f);
}

/// glibc expf bit-exact replica (round-to-nearest, WANT_ERRNO semantics —
/// overflow -> +inf, underflow -> 0.0f; errno side effects are irrelevant here).
__device__ __forceinline__ float dno_expf(float x) {
    uint32_t abstop = (dno_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {  // top12(88.0f): |x| >= 88 or inf/nan
        // glibc はここで早期 return せず、以下のいずれにも該当しない場合
        // ([88.0, 88.72] と [-103.97, -88]) はそのまま通常経路で計算する
        // (exp(88) は f32 最大値未満、exp(-88) は非正規化数として表現可能)。
        if (dno_asuint(x) == 0xff800000u) return 0.0f;       // -inf
        if (abstop >= 0x7f8u) return x + x;                  // inf / nan
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);  // > log(0x1p128) -> +inf
        if (x < -0x1.9fe368p6f) return 0.0f;                 // < log(0x1p-150) -> uflow 0
        // fall through to normal path
    }
    double xd = (double)x;
    // glibc の FMA ビルド (gcc -ffp-contract=fast) は以下を fma に縮約する:
    //   kd = fma(InvLn2N, xd, SHIFT)  (= z + SHIFT の縮約)
    //   r  = fma(InvLn2N, xd, -kd)    (= z - kd の縮約)
    // 多項式 3 ヶ所も同様。実機 (glibc 2.35 + RTX 5090 ホスト) で glibc expf /
    // Rust f32::exp の返すビットと完全一致することを確認済み
    // (x=32.564632f: 非縮約版は 1ULP 差 = 0x56fc9f1b vs 正 0x56fc9f1c)。
    double kd = __fma_rn(DNO_EXP_INVLN2N, xd, DNO_EXP_SHIFT);
    uint64_t ki = dno_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_EXP_SHIFT);
    double r = __fma_rn(DNO_EXP_INVLN2N, xd, -kd);
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    uint64_t t = DNO_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_asdouble(t);
    double z2 = __fma_rn(DNO_EXP_POLY[0], r, DNO_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(DNO_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// --- ここまで逐語コピー ---

// s114: 融合カーネルの動的共有メモリ量 (bytes) = gate 重み 1 枚。
// 起動側 (cuda.rs の gemm_swiglu_dyn_smem) と一致させること。
// (signed char (*)[GEMM_SP] にキャストして GEMM_TILE_N 行として使う)
#define GEMM_SWIGLU_DYN_SMEM (GEMM_TILE_N * GEMM_SP)
static_assert((GEMM_TILE + 2 * GEMM_TILE_N) * GEMM_SP <= 48 * 1024,
              "fused: static(sA+sB) + dynamic(sG) smem must fit the 48KB default");

/// gate/up (n = 2*ff・重み行順 [gate 全行 | up 全行]) の int8 GEMM + swiglu。
///   act[i*ff + f] = swiglu(gate[i,f], up[i,f])
///     gate[i,f] = ((float)dot(aq[i], w[f]))    * ascale[i] * scale[f]
///     up  [i,f] = ((float)dot(aq[i], w[ff+f])) * ascale[i] * scale[ff+f]
/// grid = (ceil(m/128), ceil(ff/256)) / block = (16,16) / dyn smem = sG 1 枚。
extern "C" __global__ void dno_gemm_i8_swiglu(
    const signed char* __restrict__ aq,   // [m,k]
    const float* __restrict__ ascale,     // [m]
    const signed char* __restrict__ w,    // [2ff,k] row-major (n は 256 の倍数に pad 済み)
    const float* __restrict__ scale,      // [2ff]
    float* __restrict__ act,              // [m,ff] row-major
    int m, int k, int n, int ff)
{
    const int tx = threadIdx.x; // 0..15 → 列方向 8 列ずつ
    const int ty = threadIdx.y; // 0..15 → 行方向 8 行ずつ
    const int tid = ty * GEMM_BX + tx;
    const int i0 = blockIdx.x * GEMM_TILE;
    const int j0 = blockIdx.y * GEMM_TILE_N;   // act の列 (= f) タイル

    // 1 thread = 8 行 x 16 列。up / gate を別アキュムレータで持つ
    // (整数なので 2 本に分けても dot は厳密に同じ)。
    int accU[8][16];
    int accG[8][16];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        #pragma unroll
        for (int v = 0; v < 16; v++) {
            accU[u][v] = 0;
            accG[u][v] = 0;
        }
    }

    __shared__ signed char sA[GEMM_TILE][GEMM_SP];
    __shared__ signed char sB[GEMM_TILE_N][GEMM_SP];   // up   重み (実列 ff + f)
    // gate 重み (実列 f) 1 枚分は動的共有メモリ (起動側で opt-in)。要素の型と
    // 添字 (行あたり GEMM_SP バイト) は静的版 sB と同じ。
    extern __shared__ signed char sGflat[];
    signed char (*sG)[GEMM_SP] = (signed char (*)[GEMM_SP])sGflat;

    for (int kb = 0; kb < k; kb += GEMM_TK) {
        // A (128x32) は tid < 128 の thread が 1 行ずつ。端 (gi>=m / gk>=k) は
        // 0 埋め = 積に寄与しない (dno_gemm_i8 と同一)。
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
        // 重み: 全 256 thread が up 1 行 (実列 ff+j0+r) と gate 1 行 (実列 j0+r)
        // を受け持つ。タイル末端 (f >= ff / 実列 >= n) は 0 埋め。
        {
            const int r = tid;                        // 0..255
            const int gu = ff + j0 + r;               // up   の実列 (= ff + f)
            const int gf = j0 + r;                    // gate の実列 (= act の列 f)
            if (gu < n) {
                const signed char* src = w + (size_t)gu * (size_t)k + kb;
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
            if (gf < ff) {
                const signed char* src = w + (size_t)gf * (size_t)k + kb;
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
                    *(int*)&sG[r][c] = val;
                }
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++) *(int*)&sG[r][x * 4] = 0;
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
                    const int bu = *(const int*)&sB[t * GEMM_TILE + tx * 8 + v][c0];
                    const int bg = *(const int*)&sG[t * GEMM_TILE + tx * 8 + v][c0];
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        accU[u][t * 8 + v] = __dp4a(a4[u], bu, accU[u][t * 8 + v]);
                        accG[u][t * 8 + v] = __dp4a(a4[u], bg, accG[u][t * 8 + v]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // epilogue: 担当列そのもので swiglu する (列 f の up と gate が同じ (u,v))。
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        const int gi = i0 + ty * 8 + u;
        if (gi < m) {
            const float as = ascale[gi];
            float* row = act + (size_t)gi * (size_t)ff;
            #pragma unroll
            for (int v = 0; v < 16; v++) {
                const int f = j0 + (v < 8 ? 0 : GEMM_TILE) + tx * 8 + (v & 7);
                if (f < ff) {
                    // gate / up とも既存 epilogue と同じ ((float)dot * as) * scale[j]
                    const float g = (float)accG[u][v] * as * scale[f];
                    const float uv = (float)accU[u][v] * as * scale[ff + f];
                    // dno_swiglu_int と同一の演算列 (順序・丸めとも不変)
                    const float e = dno_expf(-g);
                    const float s = __fdiv_rn(g, __fadd_rn(1.0f, e));
                    row[f] = __fmul_rn(s, uv);
                }
            }
        }
    }
}
