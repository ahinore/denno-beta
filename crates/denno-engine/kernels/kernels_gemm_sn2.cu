// denno CUDA tensor core GEMM (task 183) — prefill の **N が小さい** 射影
// (shared expert の gate/up/down N = 640 / 2560) を、1 ブロック 64 トークン x
// 64 行のタイルで回す。kernels_gemm_sn.cu (task 181、128 トークン x 64 行) の
// BM を半分にした版で、カーネルの中身 (量子の展開・2 段プリフェッチ・
// 書き出し) は sn と同じ。
//
// PTX へのコンパイル (build_ptx.sh の `gemm_sn2`、compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_gemm_sn2.cu -o kernels_gemm_sn2.ptx
//
// ---- なぜ BM=64 の別カーネルが要るか (lab t181c の台帳) --------------------
//  sn (BM=128) のあと shared は gpu 1,220 ms (5.4 TFLOP/s) まで下がったが、
//  まだレイテンシ律速だった。N=640 の gate/up は M=512 のチャンクで
//  640/64 = 10 行タイル x 512/128 = 4 = **40 ブロック** しか出ず、
//  RTX 5090 の 170 SM の 24% しか埋めない。BM=64 にすると 10 x 8 = 80 ブロック
//  (47%) になり、1 ブロックの仕事は半分になる。
//
// ---- 前提 (ホスト側 `pf_tc_smalln2_eligible` が同じ条件を検査する) --------
//   - w は [n_out][row_bytes] の Q8_0 / Q4_K
//       Q8_0: 34 B / 32 要素、  Q4_K: 144 B / 256 要素
//   - x は [nt][n_in] の f32 (行 = トークン)
//   - out は [nt][n_out] の f32 (行 = トークン)
//   - n_in % 32 == 0 (Q8_0) / n_in % 256 == 0 (Q4_K)
//   - n_out % TC_SN2_BN == 0
//
// 数値契約: **なし** (prefill の射影はビット一致契約の外)。x と重みを BF16 に
// 丸めて mma する。アキュムレートは FP32。加算順は sn と同一 (k 昇順) なので
// sn とビット一致する (タイルの切り方だけが違う)。
//
// ブロック構成: 256 スレッド = 8 warp。1 ブロック = 64 トークン × 64 行。
// warp は 2 (トークン 32 本) × 4 (行 16 本) に並べ、各 warp が 2 x 1 個の
// 16x16 アキュムレータを持つ。
//
// 共有メモリ (静的、2 段ぶん): sW 2x64x40x2 + sX 2x64x40x2 = 20,480 B
// (端数タイルの書き出しは sX を作業域に再利用する = 追加の smem を持たない)。

#include <mma.h>
#include <cuda_bf16.h>

// タイル (1 ブロック = TC_SN2_BM トークン × TC_SN2_BN 出力行、k は TC_SN2_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define TC_SN2_BM 64
#define TC_SN2_BN 64
#define TC_SN2_BK 32
#define TC_SN2_LD 40
#define TC_SN2_THREADS 256
#define TC_SN2_SMEM_BYTES 20480
// 2 段バッファぶんの bf16 要素数 (ホストの smem 見積りはこの 2 つから出す)。
#define TC_SN2_SW_ELEMS 5120
#define TC_SN2_SX_ELEMS 5120

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_gemm_tc2.cu / kernels_gemm_sn.cu の同名関数と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_sn2_f16(const unsigned char* b)
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
__device__ __forceinline__ void dno_sn2_scale_min_k4(int j, const unsigned char* q,
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
__device__ __forceinline__ void dno_sn2_store8(__nv_bfloat16* dst, const float* v)
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
//         (行 = tid>>2、列 j = (tid&3)*8)。
//   入力: 256 スレッドで 64 行 × 32 要素 = 2048 要素。1 スレッド 8 要素。
// ---------------------------------------------------------------------------

// 1 行の k ステップ (32 要素) のうち j..j+7 を f32 へ展開する。
//   TY = 0: Q8_0   TY = 1: Q4_K
template <int TY>
__device__ __forceinline__ void dno_sn2_w8(const unsigned char* row, int k0, int j, float* v)
{
    if (TY == 0) {
        // Q8_0: d(2B) + qs[32]。k0 は 32 の倍数。
        const unsigned char* blk = row + (size_t)(k0 >> 5) * 34u;
        const float d = dno_sn2_f16(blk);
        const signed char* q = reinterpret_cast<const signed char*>(blk + 2 + j);
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = (float)q[u] * d;
    } else {
        // Q4_K: 256 要素のスーパーブロックを 8 個の 32 要素サブブロックで見る。
        const int s = k0 >> 5;      // 32 要素サブブロックの通し番号
        const int sb = s >> 3;      // スーパーブロック (256 要素)
        const int sub = s & 7;      // スーパーブロック内のサブブロック 0..7
        const unsigned char* blk = row + (size_t)sb * 144u;
        const float d = dno_sn2_f16(blk);
        const float dmin = dno_sn2_f16(blk + 2);
        unsigned int sc, mn;
        dno_sn2_scale_min_k4(sub, blk + 4, &sc, &mn);
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

// 重みタイル [TC_SN2_BN][TC_SN2_BK] を 1 回ぶん読む (スレッドは 8 要素 x 1)。
template <int TY>
__device__ __forceinline__ void dno_sn2_load_w(const unsigned char* __restrict__ w,
                                               int n0, int row_bytes, int k0, int tid,
                                               float v[8])
{
    const int j = (tid & 3) * 8;
    const int row = tid >> 2;   // 0..63
    dno_sn2_w8<TY>(w + (size_t)(n0 + row) * row_bytes, k0, j, v);
}

// 入力タイル [TC_SN2_BM][TC_SN2_BK] を 1 回ぶん読む (スレッドは 8 要素 x 1)。
// nt を超えるトークンは 0 で埋める (端数タイルの mma を無害にする)。
__device__ __forceinline__ void dno_sn2_load_x(const float* __restrict__ x,
                                               int m0, int nt, int n_in, int k0, int tid,
                                               float v[8])
{
    const int j = (tid & 3) * 8;
    const int tok = tid >> 2;   // 0..63
    const int mt = m0 + tok;
    if (mt < nt) {
        const float* xp = x + (size_t)mt * n_in + k0 + j;
        const float4* xv = reinterpret_cast<const float4*>(xp);
        float4 a = xv[0];
        float4 b = xv[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = 0.0f;
    }
}

// GEMM 本体。TY は dno_sn2_w8 と同じ。
template <int TY>
__device__ __forceinline__ void dno_sn2_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    // 2 段バッファ。静的 smem (ホストは動的 smem を渡さない)。
    __shared__ __align__(32) __nv_bfloat16 sW[2 * TC_SN2_BN * TC_SN2_LD];
    __shared__ __align__(32) __nv_bfloat16 sX[2 * TC_SN2_BM * TC_SN2_LD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 1;   // トークン方向 (32 本ずつ)
    const int wn = warp >> 1;  // 行方向 (16 本ずつ)

    const int m0 = blockIdx.y * TC_SN2_BM;
    const int n0 = blockIdx.x * TC_SN2_BN;

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
        wmma::fill_fragment(acc[i], 0.0f);

    const int kb = n_in >> 5;  // k ステップ数 (32 要素ずつ)
    const int jcol = (tid & 3) * 8;
    const int jrow = tid >> 2;

    // ---- 先頭ステップを流し込む ----
    {
        float wv[8], xv[8];
        dno_sn2_load_w<TY>(w, n0, row_bytes, 0, tid, wv);
        dno_sn2_load_x(x, m0, nt, n_in, 0, tid, xv);
        dno_sn2_store8(sW + jrow * TC_SN2_LD + jcol, wv);
        dno_sn2_store8(sX + jrow * TC_SN2_LD + jcol, xv);
    }
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const int cur = (k & 1) * (TC_SN2_BN * TC_SN2_LD);
        const int nxt = ((k + 1) & 1) * (TC_SN2_BN * TC_SN2_LD);
        const int curx = (k & 1) * (TC_SN2_BM * TC_SN2_LD);
        const int nxtx = ((k + 1) & 1) * (TC_SN2_BM * TC_SN2_LD);
        const bool more = (k + 1) < kb;

        // ---- 次のステップの生データをレジスタへ (この mma と重ねる) ----
        float wv[8], xv[8];
        if (more) {
            dno_sn2_load_w<TY>(w, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_sn2_load_x(x, m0, nt, n_in, (k + 1) << 5, tid, xv);
        }

        // ---- 現ステップの mma (BK=32 = 16 の 2 段) ----
        const __nv_bfloat16* pw = sW + cur;
        const __nv_bfloat16* px = sX + curx;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = px + (wm * 32 + i * 16) * TC_SN2_LD + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC_SN2_LD);
            }
            {
                const __nv_bfloat16* p = pw + (wn * 16) * TC_SN2_LD + kk * 16;
                wmma::load_matrix_sync(bf, p, TC_SN2_LD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::mma_sync(acc[i], af[i], bf, acc[i]);
        }

        // ---- 次のステップを共有メモリへ (global の到着をここで待つ) ----
        if (more) {
            dno_sn2_store8(sW + nxt + jrow * TC_SN2_LD + jcol, wv);
            dno_sn2_store8(sX + nxtx + jrow * TC_SN2_LD + jcol, xv);
        }
        __syncthreads();
    }

    // ---- 書き出し ----
    const int mwarp = m0 + wm * 32;
    const int ncol = n0 + wn * 16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int mrow = mwarp + i * 16;
        if (mrow + 16 <= nt) {
            float* dst = out + (size_t)mrow * n_out + ncol;
            wmma::store_matrix_sync(dst, acc[i], n_out, wmma::mem_row_major);
        } else if (mrow < nt) {
            // 端数: 一旦共有メモリへ落として、有効な行だけ写す (sX を作業域に
            // 使う。k ループは終わっているので読み手はいない)。
            float* st = reinterpret_cast<float*>(sX) + warp * 256;
            wmma::store_matrix_sync(st, acc[i], 16, wmma::mem_row_major);
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

extern "C" __global__ void dno_gemm_sn2_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_sn2_body<0>(w, x, out, nt, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_gemm_sn2_q4k_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    dno_sn2_body<1>(w, x, out, nt, n_in, n_out, row_bytes);
}
