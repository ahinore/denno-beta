// denno CUDA tensor core GEMM v2 (task 176) — prefill の射影 (ssm qkv / z /
// full attention / shared expert / out) を BF16 の mma (FP32 アキュムレート) で回す。
//
// PTX へのコンパイル (build_ptx.sh の `gemm_tc2`、この target だけ compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_gemm_tc2.cu -o kernels_gemm_tc2.ptx
//
// compute_75 では bf16 の mma が無いので arch を上げる (PTX は前方互換なので
// 実機 (sm_120) で JIT される)。
//
// 数値契約: **なし** (prefill の射影は 2026-10-28 のオペレーター指示で
// ビット一致契約の外)。x と重みを BF16 に丸めて mma するので、既存の
// kernels_gemm.cu (f32 FMA) とも kernels_gemm_tc.cu (br169) とも値が変わる。
// 丸めだけの差で、演算の意味は同じ。アキュムレートは FP32。
//
// ---- br169 版 (kernels_gemm_tc.cu) との違い -------------------------------
//  br169 は 1 ブロック = 128 トークン × 64 行 + k 32 ずつで、各 k ステップで
//  「global を読む → 共有メモリへ書く → __syncthreads → mma → __syncthreads」
//  を直列に回していた (プリフェッチ無し)。lab t169b の台帳では ssm qkv が
//  8.0 TFLOP/s (K 2560 の形) で、同じカーネルの K 6144 の形 (out) が
//  24.3 TFLOP/s。**1 ステップあたりの露出レイテンシ**が支配している。
//
//  ここでは:
//   1. 出力行を 64 → 128 に広げる (1 ブロックの仕事が 2 倍。重みタイル 1 回の
//      読み込みで 2 倍の mma をするので、固定費が半分になる)。
//   2. 2 段のレジスタプリフェッチ: 次の k ステップの生データを**先に**レジスタへ
//      読み、その mma をしている間に smem へ書く (global の待ちを mma と重ねる)。
//   3. smem の行ストライドを 32 → 40 に詰め物する (ldmatrix/ld.shared の
//      バンク衝突を避ける。ldm は 16 バイトの倍数が要るので 8 の倍数から選ぶ)。
//   4. Q8_0 に加えて Q4_K / Q5_K を同じカーネルで扱う (実モデルの ssm qkv は
//      4 層だけ Q5_K、z は Q4_K。br169 はこの 4 層を f32 カーネルへ落として
//      いた = 14k の qkv_pf 6.27 s のうち約 1.8 s を占めていた)。
//
// ---- 前提 (ホスト側 `pf_tc2_eligible` が同じ条件を検査する) --------------
//   - w は [n_out][row_bytes] の Q8_0 / Q4_K / Q5_K
//       Q8_0: 34 B / 32 要素、  Q4_K: 144 B / 256 要素、  Q5_K: 176 B / 256 要素
//   - x は [nt][n_in] の f32 (行 = トークン)
//   - out は [nt][n_out] の f32 (行 = トークン)
//   - n_in % 32 == 0 (Q8_0) / n_in % 256 == 0 (Q4_K / Q5_K)
//   - n_out % TC2_BN == 0
//
// ブロック構成: 256 スレッド = 8 warp。1 ブロック = 128 トークン × 128 行。
// warp は 4 (トークン 32 本) × 2 (行 64 本) に並べ、各 warp が 2 × 4 個の
// 16x16 アキュムレータを持つ。
//
// 共有メモリ (静的、2 段ぶん): sW 2x128x40x2 + sX 2x128x40x2 = 40,960 B
// (端数タイルの書き出しは sX を作業域に再利用する = 追加の smem を持たない)。

#include <mma.h>
#include <cuda_bf16.h>

// タイル (1 ブロック = TC2_BM トークン × TC2_BN 出力行、k は TC2_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define TC2_BM 128
#define TC2_BN 128
#define TC2_BK 32
#define TC2_LD 40
#define TC2_THREADS 256
#define TC2_SMEM_BYTES 40960
// 2 段バッファぶんの bf16 要素数 (ホストの smem 見積りはこの 3 つから出す)。
#define TC2_SW_ELEMS 10240
#define TC2_SX_ELEMS 10240

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_gemm_tc.cu の dno_tc_f16 と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_tc2_f16(const unsigned char* b)
{
    unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1fu;
    unsigned int man = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((man & 0x400u) == 0) { man <<= 1; --e; }
            man &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

// Q4_K / Q5_K の 6bit scale/min 展開 (kernels_gemm.cu の dno_g_scale_min_k4 と
// 同一。kquant.rs の get_scale_min_k4 と同値)。
__device__ __forceinline__ void dno_tc2_scale_min_k4(int j, const unsigned char* q,
                                                    unsigned int* d, unsigned int* m)
{
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((unsigned int)(q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((unsigned int)(q[j] >> 6) << 4);
    }
}

// 8 個の f32 を bf16 に丸めて 16 B で書く (dst は 16 B 整列)。
__device__ __forceinline__ void dno_tc2_store8(__nv_bfloat16* dst, const float* v)
{
    unsigned int t[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        __nv_bfloat162 h = __floats2bfloat162_rn(v[2 * u], v[2 * u + 1]);
        t[u] = *reinterpret_cast<unsigned int*>(&h);
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(t[0], t[1], t[2], t[3]);
}

// ---------------------------------------------------------------------------
// ステージングの担当 (256 スレッドで 128 行 × 32 要素 = 4096 要素を覆う)。
// スレッドは 8 要素を 2 組 (= 16 要素) 持ち、どちらも 1 個の 32 要素サブブロック
// の内側に収まる。行 = (tid>>2) + 64*hf、列 j = (tid&3)*8。
// ---------------------------------------------------------------------------

// 1 行の k ステップ (32 要素) のうち j..j+7 を f32 へ展開する。
//   TY = 0: Q8_0   TY = 1: Q4_K   TY = 2: Q5_K
template <int TY>
__device__ __forceinline__ void dno_tc2_w8(const unsigned char* row, int k0, int j, float* v)
{
    if (TY == 0) {
        // Q8_0: d(2B) + qs[32]。k0 は 32 の倍数。
        const unsigned char* blk = row + (size_t)(k0 >> 5) * 34u;
        const float d = dno_tc2_f16(blk);
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2 + j);
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = (float)q[u] * d;
    } else {
        // Q4_K / Q5_K: 256 要素のスーパーブロックを 8 個の 32 要素サブブロックで見る。
        const int s = k0 >> 5;      // 32 要素サブブロックの通し番号
        const int sb = s >> 3;      // スーパーブロック (256 要素)
        const int sub = s & 7;      // スーパーブロック内のサブブロック 0..7
        const unsigned char* blk = row + (size_t)sb * (TY == 1 ? 144u : 176u);
        const float d = dno_tc2_f16(blk);
        const float dmin = dno_tc2_f16(blk + 2);
        unsigned int sc, mn;
        dno_tc2_scale_min_k4(sub, blk + 4, &sc, &mn);
        const float ds = d * (float)sc;
        const float dm = dmin * (float)mn;
        // qs は 4 個の 32 バイト群 (sub>>1)。バイト位置 = サブブロック内の要素番号
        // (0..31) そのもの。偶数 sub が下ニブル、奇数 sub が上ニブル。
        // (ggml の layout: 1 バイトが 4bit を 1 個だけ持ち、要素 0..31 が下ニブル、
        //  要素 32..63 が同じ 32 バイトの上ニブルに入る。interleave ではない。)
        const unsigned char* q = blk + (TY == 1 ? 16u : 48u)
                               + (unsigned int)((sub >> 1) << 5) + (unsigned int)j;
        const int shift = (sub & 1) ? 4 : 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int qv = (int)((q[u] >> shift) & 0x0fu);
            if (TY == 2) {
                // Q5_K の 5bit 目は qh[j+u] の bit sub。
                qv += (int)(((unsigned int)blk[16 + j + u] >> sub) & 1u) << 4;
            }
            v[u] = ds * (float)qv - dm;
        }
    }
}

// 重みタイル [TC2_BN][TC2_BK] を 1 回ぶん読む (スレッドは 8 要素 × 2)。
template <int TY>
__device__ __forceinline__ void dno_tc2_load_w(const unsigned char* __restrict__ w,
                                               int n0, int row_bytes, int k0, int tid,
                                               float v[2][8])
{
    const int j = (tid & 3) * 8;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int row = (tid >> 2) + hf * 64;
        dno_tc2_w8<TY>(w + (size_t)(n0 + row) * row_bytes, k0, j, v[hf]);
    }
}

// 入力タイル [TC2_BM][TC2_BK] を 1 回ぶん読む (スレッドは 8 要素 × 2)。
// nt を超えるトークンは 0 で埋める (端数タイルの mma を無害にする)。
__device__ __forceinline__ void dno_tc2_load_x(const float* __restrict__ x,
                                               int m0, int nt, int n_in, int k0, int tid,
                                               float v[2][8])
{
    const int j = (tid & 3) * 8;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int tok = (tid >> 2) + hf * 64;
        const int mt = m0 + tok;
        if (mt < nt) {
            const float* xp = x + (size_t)mt * n_in + k0 + j;
            const float4* xv = reinterpret_cast<const float4*>(xp);
            float4 a = xv[0];
            float4 b = xv[1];
            v[hf][0] = a.x; v[hf][1] = a.y; v[hf][2] = a.z; v[hf][3] = a.w;
            v[hf][4] = b.x; v[hf][5] = b.y; v[hf][6] = b.z; v[hf][7] = b.w;
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) v[hf][u] = 0.0f;
        }
    }
}

// GEMM 本体。TY は dno_tc2_w8 と同じ。
template <int TY>
__device__ __forceinline__ void dno_tc2_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    // 2 段バッファ。静的 smem (ホストは動的 smem を渡さない)。
    __shared__ __align__(32) __nv_bfloat16 sW[2 * TC2_BN * TC2_LD];
    __shared__ __align__(32) __nv_bfloat16 sX[2 * TC2_BM * TC2_LD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 3;   // トークン方向 (32 本ずつ)
    const int wn = warp >> 2;  // 行方向 (64 本ずつ)

    const int m0 = blockIdx.y * TC2_BM;
    const int n0 = blockIdx.x * TC2_BN;

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int kb = n_in >> 5;  // k ステップ数 (32 要素ずつ)
    const int jcol = (tid & 3) * 8;
    const int jrow = tid >> 2;

    // ---- 先頭ステップを流し込む ----
    {
        float wv[2][8], xv[2][8];
        dno_tc2_load_w<TY>(w, n0, row_bytes, 0, tid, wv);
        dno_tc2_load_x(x, m0, nt, n_in, 0, tid, xv);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int r = jrow + hf * 64;
            dno_tc2_store8(sW + r * TC2_LD + jcol, wv[hf]);
            dno_tc2_store8(sX + r * TC2_LD + jcol, xv[hf]);
        }
    }
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const int cur = (k & 1) * (TC2_BN * TC2_LD);
        const int nxt = ((k + 1) & 1) * (TC2_BN * TC2_LD);
        const int curx = (k & 1) * (TC2_BM * TC2_LD);
        const int nxtx = ((k + 1) & 1) * (TC2_BM * TC2_LD);
        const bool more = (k + 1) < kb;

        // ---- 次のステップの生データをレジスタへ (この mma と重ねる) ----
        float wv[2][8], xv[2][8];
        if (more) {
            dno_tc2_load_w<TY>(w, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_tc2_load_x(x, m0, nt, n_in, (k + 1) << 5, tid, xv);
        }

        // ---- 現ステップの mma (BK=32 = 16 の 2 段) ----
        const __nv_bfloat16* pw = sW + cur;
        const __nv_bfloat16* px = sX + curx;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = px + (wm * 32 + i * 16) * TC2_LD + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC2_LD);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat16* p = pw + (wn * 64 + j * 16) * TC2_LD + kk * 16;
                wmma::load_matrix_sync(bf[j], p, TC2_LD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }

        // ---- 次のステップを共有メモリへ (global の到着をここで待つ) ----
        if (more) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int r = jrow + hf * 64;
                dno_tc2_store8(sW + nxt + r * TC2_LD + jcol, wv[hf]);
                dno_tc2_store8(sX + nxtx + r * TC2_LD + jcol, xv[hf]);
            }
        }
        __syncthreads();
    }

    // ---- 書き出し ----
    const int mwarp = m0 + wm * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int mrow = mwarp + i * 16;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ncol = n0 + wn * 64 + j * 16;
            if (mrow + 16 <= nt) {
                float* dst = out + (size_t)mrow * n_out + ncol;
                wmma::store_matrix_sync(dst, acc[i][j], n_out, wmma::mem_row_major);
            } else if (mrow < nt) {
                // 端数: 一旦共有メモリへ落として、有効な行だけ写す (sX を作業域に
                // 使う。k ループは終わっているので読み手はいない)。
                float* st = reinterpret_cast<float*>(sX) + warp * 256;
                wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (int r = lane; r < nt - mrow; r += 32) {
                    const float* src = st + r * 16;
                    float* dst = out + (size_t)(mrow + r) * n_out + ncol;
#pragma unroll
                    for (int c = 0; c < 16; ++c) dst[c] = src[c];
                }
                __syncwarp();
            }
        }
    }
}

extern "C" __global__ void dno_gemm_tc2_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_tc2_body<0>(w, x, out, nt, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_gemm_tc2_q4k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_tc2_body<1>(w, x, out, nt, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_gemm_tc2_q5k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_tc2_body<2>(w, x, out, nt, n_in, n_out, row_bytes);
}
