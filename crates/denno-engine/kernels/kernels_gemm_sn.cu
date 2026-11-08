// denno CUDA tensor core GEMM (task 181) — prefill の **N が小さい** 射影
// (shared expert の gate/up/down N = 640 / 2560、ssm out_proj N = 2560) を
// BF16 の mma (FP32 アキュムレート) で回す。kernels_gemm_tc2.cu の BN=64 版。
//
// PTX へのコンパイル (build_ptx.sh の `gemm_sn`、この target だけ compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_gemm_sn.cu -o kernels_gemm_sn.ptx
//
// compute_75 では bf16 の mma が無いので arch を上げる (PTX は前方互換なので
// 実機 (sm_120) で JIT される)。
//
// 数値契約: **なし** (prefill の射影は 2026-10-28 のオペレーター指示で
// ビット一致契約の外)。x と重みを BF16 に丸めて mma するので、既存の
// kernels_gemm.cu (f32 FMA) とも kernels_gemm_tc.cu / kernels_gemm_tc2.cu とも
// 値が変わる。丸めだけの差で、演算の意味は同じ。アキュムレートは FP32。
//
// ---- なぜ BN=64 の別カーネルが要るか (lab t175b の台帳) --------------------
//  tc2 は BN=128 なので、N=640 の行列では 640/128 = 5 ブロックしか出ない。
//  M=512 のチャンクでも 5 x 4 = 20 ブロック = RTX 5090 の 170 SM の 12%。
//  shared expert は 14,018 トークンで 3.54 s (= 1.9 TFLOP/s) かかっていた
//  (gate は Q4_K なので br169 (Q8_0 専用) にも入らず f32 カーネル)。
//  BN=64 にすると N=640 で 10 ブロック x 4 = 40、N=2560 で 40 x 4 = 160
//  ブロックになり、out_proj の形 (N=2560, K=6144) は 1 波で埋まる。
//
// ---- 前提 (ホスト側 `pf_tc_smalln_eligible` が同じ条件を検査する) ----------
//   - w は [n_out][row_bytes] の Q8_0 / Q4_K
//       Q8_0: 34 B / 32 要素、  Q4_K: 144 B / 256 要素
//   - x は [nt][n_in] の f32 (行 = トークン)
//   - out は [nt][n_out] の f32 (行 = トークン)
//   - n_in % 32 == 0 (Q8_0) / n_in % 256 == 0 (Q4_K)
//   - n_out % TC_SN_BN == 0
//
// ブロック構成: 256 スレッド = 8 warp。1 ブロック = 128 トークン × 64 行。
// warp は 4 (トークン 32 本) × 2 (行 32 本) に並べ、各 warp が 2 × 2 個の
// 16x16 アキュムレータを持つ。
//
// 共有メモリ (静的、2 段ぶん): sW 2x64x40x2 + sX 2x128x40x2 = 30,720 B
// (端数タイルの書き出しは sX を作業域に再利用する = 追加の smem を持たない)。

#include <mma.h>
#include <cuda_bf16.h>

// タイル (1 ブロック = TC_SN_BM トークン × TC_SN_BN 出力行、k は TC_SN_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define TC_SN_BM 128
#define TC_SN_BN 64
#define TC_SN_BK 32
#define TC_SN_LD 40
#define TC_SN_THREADS 256
#define TC_SN_SMEM_BYTES 30720
// 2 段バッファぶんの bf16 要素数 (ホストの smem 見積りはこの 2 つから出す)。
#define TC_SN_SW_ELEMS 5120
#define TC_SN_SX_ELEMS 10240

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_gemm_tc2.cu の dno_tc2_f16 と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_sn_f16(const unsigned char* b)
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

// Q4_K の 6bit scale/min 展開 (kernels_gemm.cu の dno_g_scale_min_k4 と同一。
// kquant.rs の get_scale_min_k4 と同値)。
__device__ __forceinline__ void dno_sn_scale_min_k4(int j, const unsigned char* q,
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
__device__ __forceinline__ void dno_sn_store8(__nv_bfloat16* dst, const float* v)
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
// ステージングの担当
//   重み: 256 スレッドで 64 行 × 32 要素 = 2048 要素。1 スレッド 8 要素
//         (行 = tid>>2、列 j = (tid&3)*8)。1 個の 32 要素サブブロックの内側。
//   入力: 256 スレッドで 128 行 × 32 要素 = 4096 要素。1 スレッド 8 要素 x 2 組。
// ---------------------------------------------------------------------------

// 1 行の k ステップ (32 要素) のうち j..j+7 を f32 へ展開する。
//   TY = 0: Q8_0   TY = 1: Q4_K
template <int TY>
__device__ __forceinline__ void dno_sn_w8(const unsigned char* row, int k0, int j, float* v)
{
    if (TY == 0) {
        // Q8_0: d(2B) + qs[32]。k0 は 32 の倍数。
        const unsigned char* blk = row + (size_t)(k0 >> 5) * 34u;
        const float d = dno_sn_f16(blk);
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2 + j);
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = (float)q[u] * d;
    } else {
        // Q4_K: 256 要素のスーパーブロックを 8 個の 32 要素サブブロックで見る。
        const int s = k0 >> 5;      // 32 要素サブブロックの通し番号
        const int sb = s >> 3;      // スーパーブロック (256 要素)
        const int sub = s & 7;      // スーパーブロック内のサブブロック 0..7
        const unsigned char* blk = row + (size_t)sb * 144u;
        const float d = dno_sn_f16(blk);
        const float dmin = dno_sn_f16(blk + 2);
        unsigned int sc, mn;
        dno_sn_scale_min_k4(sub, blk + 4, &sc, &mn);
        const float ds = d * (float)sc;
        const float dm = dmin * (float)mn;
        // qs は 4 個の 32 バイト群 (sub>>1)。偶数 sub が下ニブル、奇数 sub が
        // 上ニブル (ggml の layout: 1 バイトが 4bit を 1 個だけ持つ)。
        const unsigned char* q = blk + 16u + (unsigned int)((sub >> 1) << 5) + (unsigned int)j;
        const int shift = (sub & 1) ? 4 : 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int qv = (int)((q[u] >> shift) & 0x0fu);
            v[u] = ds * (float)qv - dm;
        }
    }
}

// 重みタイル [TC_SN_BN][TC_SN_BK] を 1 回ぶん読む (スレッドは 8 要素 x 1)。
template <int TY>
__device__ __forceinline__ void dno_sn_load_w(const unsigned char* __restrict__ w,
                                              int n0, int row_bytes, int k0, int tid,
                                              float v[8])
{
    const int j = (tid & 3) * 8;
    const int row = tid >> 2;   // 0..63
    dno_sn_w8<TY>(w + (size_t)(n0 + row) * row_bytes, k0, j, v);
}

// 入力タイル [TC_SN_BM][TC_SN_BK] を 1 回ぶん読む (スレッドは 8 要素 x 2)。
// nt を超えるトークンは 0 で埋める (端数タイルの mma を無害にする)。
__device__ __forceinline__ void dno_sn_load_x(const float* __restrict__ x,
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

// GEMM 本体。TY は dno_sn_w8 と同じ。
template <int TY>
__device__ __forceinline__ void dno_sn_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    // 2 段バッファ。静的 smem (ホストは動的 smem を渡さない)。
    __shared__ __align__(32) __nv_bfloat16 sW[2 * TC_SN_BN * TC_SN_LD];
    __shared__ __align__(32) __nv_bfloat16 sX[2 * TC_SN_BM * TC_SN_LD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 3;   // トークン方向 (32 本ずつ)
    const int wn = warp >> 2;  // 行方向 (32 本ずつ)

    const int m0 = blockIdx.y * TC_SN_BM;
    const int n0 = blockIdx.x * TC_SN_BN;

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int kb = n_in >> 5;  // k ステップ数 (32 要素ずつ)
    const int jcol = (tid & 3) * 8;
    const int jrow = tid >> 2;

    // ---- 先頭ステップを流し込む ----
    {
        float wv[8], xv[2][8];
        dno_sn_load_w<TY>(w, n0, row_bytes, 0, tid, wv);
        dno_sn_load_x(x, m0, nt, n_in, 0, tid, xv);
        dno_sn_store8(sW + jrow * TC_SN_LD + jcol, wv);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int r = jrow + hf * 64;
            dno_sn_store8(sX + r * TC_SN_LD + jcol, xv[hf]);
        }
    }
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const int cur = (k & 1) * (TC_SN_BN * TC_SN_LD);
        const int nxt = ((k + 1) & 1) * (TC_SN_BN * TC_SN_LD);
        const int curx = (k & 1) * (TC_SN_BM * TC_SN_LD);
        const int nxtx = ((k + 1) & 1) * (TC_SN_BM * TC_SN_LD);
        const bool more = (k + 1) < kb;

        // ---- 次のステップの生データをレジスタへ (この mma と重ねる) ----
        float wv[8], xv[2][8];
        if (more) {
            dno_sn_load_w<TY>(w, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_sn_load_x(x, m0, nt, n_in, (k + 1) << 5, tid, xv);
        }

        // ---- 現ステップの mma (BK=32 = 16 の 2 段) ----
        const __nv_bfloat16* pw = sW + cur;
        const __nv_bfloat16* px = sX + curx;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = px + (wm * 32 + i * 16) * TC_SN_LD + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC_SN_LD);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const __nv_bfloat16* p = pw + (wn * 32 + j * 16) * TC_SN_LD + kk * 16;
                wmma::load_matrix_sync(bf[j], p, TC_SN_LD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }

        // ---- 次のステップを共有メモリへ (global の到着をここで待つ) ----
        if (more) {
            dno_sn_store8(sW + nxt + jrow * TC_SN_LD + jcol, wv);
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int r = jrow + hf * 64;
                dno_sn_store8(sX + nxtx + r * TC_SN_LD + jcol, xv[hf]);
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
        for (int j = 0; j < 2; ++j) {
            const int ncol = n0 + wn * 32 + j * 16;
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

extern "C" __global__ void dno_gemm_sn_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_sn_body<0>(w, x, out, nt, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_gemm_sn_q4k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_sn_body<1>(w, x, out, nt, n_in, n_out, row_bytes);
}
