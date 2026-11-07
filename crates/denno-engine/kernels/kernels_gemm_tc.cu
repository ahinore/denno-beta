// denno CUDA tensor core GEMM (task 169) — prefill の射影 (ssm qkv / shared
// expert / full attention) を BF16 の mma (FP32 アキュムレート) で回す。
//
// PTX へのコンパイル (build_ptx.sh の `gemm_tc`、ARCH は compute_90 固定):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_gemm_tc.cu -o kernels_gemm_tc.ptx
//
// compute_75 では bf16 の mma が無いので、このファイルだけ arch を上げる
// (PTX は前方互換なので実機 (sm_120) で JIT される)。
//
// 数値契約: **なし** (prefill の射影は 2026-10-28 のオペレーター指示で
// ビット一致契約の外)。x と重みを BF16 に丸めて mma するので、既存の
// kernels_gemm.cu (f32 FMA) とは値が変わる。丸めだけの差で、演算の意味は同じ。
//
// 前提 (ホスト側 `pf_tc_gemm_eligible` が同じ条件を検査する):
//   - w は [n_out][row_bytes] の Q8_0 (row_bytes = n_in/32*34)
//   - x は [nt][n_in] の f32 (行 = トークン)
//   - out は [nt][n_out] の f32 (行 = トークン)
//   - n_in % 32 == 0, n_out % 64 == 0
//
// ブロック構成: 256 スレッド = 8 warp。1 ブロック = 128 トークン × 64 行。
// warp は 4 (トークン 32 本ずつ) × 2 (行 32 本ずつ) に並べ、各 warp が
// 2 × 2 個の 16x16 アキュムレータを持つ。
//
// 重みタイル [64 行][32 k] (Q8_0 の 1 量子ブロック = 32 要素/行) を共有
// メモリへ **1 ブロック 1 回** 展開し、128 トークンの mma で使い回す
// (既存カーネルは 16 行 × 16 トークンで重みを読み直していた)。
//
// 出力の端数: n_out は 64 の倍数を前提にする。トークン数 nt が 128 の倍数で
// ないときは、はみ出す 16 トークンのタイルだけ共有メモリに落として有効行を写す。

#include <mma.h>
#include <cuda_bf16.h>

// タイル (1 ブロック = TC_BM トークン × TC_BN 出力行、k は TC_BK ずつ)。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define TC_BM 128
#define TC_BN 64
#define TC_BK 32

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_gemm.cu の dno_g_f16 と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_tc_f16(const unsigned char* b)
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

// 8 個の f32 を bf16 に丸めて 16 B で書く (dst は 16 B 整列)。
__device__ __forceinline__ void dno_tc_store8(__nv_bfloat16* dst, const float* v)
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
// Q8_0 重みの tensor core GEMM
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemm_tc_q8_0_w(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int nt, int n_in, int n_out, int row_bytes)
{
    // 共有メモリは静的 (1 ブロック = 20 KiB)。動的 smem を渡さないので、
    // ホストとサイズを取り違える余地が無い。uint4 で書くので 16 B 整列。
    __shared__ __align__(16) __nv_bfloat16 sW[TC_BN * TC_BK];   // 重み [行][k]
    __shared__ __align__(16) __nv_bfloat16 sX[TC_BM * TC_BK];   // 入力 [トークン][k]
    __shared__ __align__(16) float sStage[8 * 16 * 16];         // 端数 [warp][16][16]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 3;   // トークン方向 (32 本ずつ)
    const int wn = warp >> 2;  // 行方向 (32 本ずつ)

    const int m0 = blockIdx.y * TC_BM;
    const int n0 = blockIdx.x * TC_BN;

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < n_in; k0 += TC_BK) {
        // ---- 重みタイル (64 行 × 32 k) を共有メモリへ展開 ----
        // 2048 要素 / 256 スレッド = 8 要素ずつ。j は 0/8/16/24 のいずれかで、
        // 8 要素は必ず 1 個の量子ブロック (32 要素) の内側に収まる。
        {
            const int e0 = tid * 8;
            const int row = e0 >> 5;
            const int j = e0 & 31;
            const unsigned char* blk =
                w + (size_t)(n0 + row) * row_bytes + (size_t)(k0 >> 5) * 34u;
            const float d = dno_tc_f16(blk);
            const signed char* q = reinterpret_cast<const signed char*>(blk + 2 + j);
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = (float)q[u] * d;
            dno_tc_store8(sW + row * TC_BK + j, v);
        }
        // ---- 入力タイル (128 トークン × 32 k) ----
        // 4096 要素 / 256 スレッド = 16 要素ずつ (8 要素 × 2 回)。
        // nt を超えるトークンは 0 で埋める (端数タイルの mma を無害にする)。
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int e0 = (tid + half * 256) * 8;
            const int tok = e0 >> 5;
            const int j = e0 & 31;
            const int mt = m0 + tok;
            float v[8];
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
            dno_tc_store8(sX + tok * TC_BK + j, v);
        }
        __syncthreads();

        // ---- mma (BK=32 = 16 の 2 段) ----
#pragma unroll
        for (int kk = 0; kk < TC_BK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = sX + (wm * 32 + i * 16) * TC_BK + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC_BK);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const __nv_bfloat16* p = sW + (wn * 32 + j * 16) * TC_BK + kk * 16;
                wmma::load_matrix_sync(bf[j], p, TC_BK);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
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
                // 端数: 一旦共有メモリへ落として、有効な行だけ写す。
                float* st = sStage + warp * 256;
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
