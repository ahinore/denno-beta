// denno CUDA tensor core grouped expert GEMM (task 250) — prefill で 1 expert
// あたり数百行を受けるときに、重みを 1 回だけデコードし、活性をタイルで
// 1 回だけ読む形に直す。
//
// PTX へのコンパイル (build_ptx.sh の `exp_iq_tc`、この target だけ compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_exp_iq_tc.cu -o kernels_exp_iq_tc.ptx
//
// bf16 の mma は compute_80 以降にしか無いので arch を上げる (PTX は前方互換
// なので実機 (sm_120) で JIT される)。
//
// # 何を置き換えるか
//
// 現行の prefill は `kernels_exp_iq_pf.cu` の m 行 GEMV で、1 block = 8 出力行。
// 出力行ごとに活性の行 (n_in 要素) を読み直すので、1 expert あたりの活性読みが
// `n_out/8 × m × n_in × 4 B` になる (m=300 / e=2560 / ff=1024 で 393 MB)。
// layer major (REPORT_P242) ではこれが重みの 2 MB を上回り、活性律速になる。
//
// ここでは 64 トークン × 64 出力行のタイルで受けるので、活性の読みは
// `n_out/64 × m × n_in × 4 B` で済む (同じ形で 49 MB、8 分の 1)。
// 重みは smem に 1 回デコードし、bf16 の mma (FP32 アキュムレート) で積む。
//
// # 数値契約
//
// **なし** (prefill の射影は 2026-10-28 のオペレーター指示でビット一致契約の外)。
// 重みは IQ のデコード結果を bf16 に丸め、活性も bf16 に丸めて mma する。
// アキュムレートは FP32 で、行ごとの和は k 昇順のまま (タイル幅 32 の逐次)。
// 既存の `kernels_exp_iq_pf.cu` (f32 FMA・32 レーン分割) とは値が変わる。
// 丸めだけの差で、演算の意味は同じ。
//
// # segment 表 (const unsigned int* seg、1 segment = 4 語)
//
//   seg[s*4 + 0] = m        (この expert の行数)
//   seg[s*4 + 1] = row_off  (rows の先頭添字。出力も同じ添字 × n_out)
//   seg[s*4 + 2] = 0        (予約)
//   seg[s*4 + 3] = 0        (予約)
//
// `rows[i]` は行 i のトークン番号で、入力は `x + rows[i] * n_in`。
// down 行列は入力が `act` (行が詰まっている) なので `rows[i] = i` を渡す。
//
// # 起動形 (ホスト側 `pf_launch_group_tc` が同じ条件を検査する)
//
//   grid  = (n_out / TC_BN, ceil(max_m / TC_BM), nseg)
//   block = TC_THREADS = 256
//   n_out % TC_BN == 0、n_in % 256 == 0 (IQ の 256 要素ブロック)
//
// 共有メモリ (静的): gate/up は 2 枚、down は 1 枚。
//   gate/up: 2 × TC_BN × TC_LD × 2 B × 2 段 × 2 枚 = 40,960 B
//   down   : 2 × TC_BN × TC_LD × 2 B × 2 段     = 20,480 B
//   入力   : 2 × TC_BM × TC_LD × 2 B × 2 段     = 20,480 B
// 最大 40,960 B (48 KiB 既定の内側。opt-in は要らない)。

#include <mma.h>
#include <cuda_bf16.h>
#include "kernels_exp_iq_tables.h"
#include <cuda_fp16.h>

// タイル。値の行にコメントを付けないこと (kernel_invariants が `#define` の
// 行末までを値として読む)。
#define TC_BM 64
#define TC_BN 64
#define TC_BK 32
#define TC_LD 40
#define TC_THREADS 256

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_exp_iq_pf.cu の pf_f16 と同一)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_tcq_f16(const unsigned char* b)
{
    // task 640: 手書きのビット展開をハードウェアの cvt.f32.f16 (`__half2float`) に
    // 置き換えた。f16 -> f32 の拡張は IEEE 754 で一意なので結果は同じ (0 / 非正規化数
    // を含む全 65536 パターンで旧実装と一致することを cuda.rs の
    // f16_cvt_exhaustive で確認済み)。NaN と inf だけは旧実装の式をそのまま残す
    // (ハードウェアは NaN を quiet 化するので、ペイロードの再現性のため)。
    const unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    if ((h & 0x7c00u) == 0x7c00u) {
        const unsigned int sign = (h & 0x8000u) << 16;
        return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
    }
    return __half2float(__ushort_as_half((unsigned short)h));
}

__device__ __forceinline__ unsigned int dno_tcq_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

// ---------------------------------------------------------------------------
// 32 要素ぶんの重み 8 個 (要素 j*8 .. j*8+7、j = 0..3) を f32 へ。
// 各式は kernels_exp_iq_pf.cu の pf_w8_* と 1:1 (lane を「32 ブロック内の
// 8 要素グループ」= j に、ib32 を「32 ブロックの通し番号」= sub に置き換えた)。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dno_tc_w8_iq2s(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;          // 32 要素ブロックの通し番号
    const int sb = s >> 3;          // 256 要素ブロック
    const int sub = s & 7;          // 256 ブロック内の 32 ブロック
    const unsigned char* blk = rowp + (size_t)sb * 82u;
    const float d = dno_tcq_f16(blk);
    const unsigned int qh = (unsigned int)blk[66 + sub];
    const unsigned int idx = (unsigned int)blk[2 + sub * 4 + j] | ((qh << (8 - 2 * j)) & 0x300u);
    const unsigned int sg = (unsigned int)blk[34 + sub * 4 + j];
    const unsigned int sc = (unsigned int)blk[74 + sub];
    const unsigned int nib = (j & 2) ? (sc >> 4) : (sc & 0x0fu);
    const float db = d * (0.5f + (float)nib) * 0.25f;
    const unsigned long long grid = __ldg(&dno_iq2s_grid[idx]);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const float gv = (float)((grid >> (8 * u)) & 0xffull);
        w[u] = db * gv * ((sg & (1u << u)) ? -1.0f : 1.0f);
    }
}

__device__ __forceinline__ void dno_tc_w8_iq3xxs(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;
    const int sb = s >> 3;
    const int sub = s & 7;
    const unsigned char* blk = rowp + (size_t)sb * 98u;
    const float d = dno_tcq_f16(blk);
    const unsigned int aux = dno_tcq_u32(blk + 66 + sub * 4);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const unsigned int signs = (unsigned int)__ldg(&dno_ksigns_iq2xs[(aux >> (7 * j)) & 127u]);
    const unsigned int g1 = __ldg(&dno_iq3xxs_grid[blk[2 + sub * 8 + 2 * j]]);
    const unsigned int g2 = __ldg(&dno_iq3xxs_grid[blk[2 + sub * 8 + 2 * j + 1]]);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float gv = (float)((g1 >> (8 * u)) & 0xffu);
        w[u] = db * gv * ((signs & (1u << u)) ? -1.0f : 1.0f);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float gv = (float)((g2 >> (8 * u)) & 0xffu);
        w[4 + u] = db * gv * ((signs & (1u << (u + 4))) ? -1.0f : 1.0f);
    }
}

__device__ __forceinline__ void dno_tc_w8_iq3s(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;
    const int sb = s >> 3;
    const int sub = s & 7;
    const unsigned char* blk = rowp + (size_t)sb * 110u;
    const float d = dno_tcq_f16(blk);
    const unsigned int sc = (unsigned int)blk[106 + (sub >> 1)];
    const unsigned int nib = (sub & 1) ? (sc >> 4) : (sc & 0x0fu);
    const float db = d * (1.0f + 2.0f * (float)nib);
    const unsigned int qh = (unsigned int)blk[66 + sub];
    const unsigned int idx1 = (unsigned int)blk[2 + sub * 8 + 2 * j]
                              | ((qh << (8 - 2 * j)) & 256u);
    const unsigned int idx2 = (unsigned int)blk[2 + sub * 8 + 2 * j + 1]
                              | ((qh << (7 - 2 * j)) & 256u);
    const unsigned int signs = (unsigned int)blk[74 + sub * 4 + j];
    const unsigned long long g1 = __ldg(&dno_iq3s_grid[idx1]);
    const unsigned long long g2 = __ldg(&dno_iq3s_grid[idx2]);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float gv = (float)((g1 >> (8 * u)) & 0xffull);
        w[u] = db * gv * ((signs & (1u << u)) ? -1.0f : 1.0f);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float gv = (float)((g2 >> (8 * u)) & 0xffull);
        w[4 + u] = db * gv * ((signs & (1u << (u + 4))) ? -1.0f : 1.0f);
    }
}

__device__ __forceinline__ void dno_tc_w8_iq4xs(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;
    const int sb = s >> 3;
    const int sub = s & 7;
    const unsigned char* blk = rowp + (size_t)sb * 136u;
    const float d = dno_tcq_f16(blk);
    const int half = j >> 1;
    const int jbase = (j & 1) * 8;
    const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
    const unsigned int sl = (unsigned int)blk[4 + (sub >> 1)];
    const unsigned int ls = ((sl >> (4 * (sub & 1))) & 0x0fu) | (((sh >> (2 * sub)) & 3u) << 4);
    const float dl = d * ((float)ls - 32.0f);
    const unsigned char* qs = blk + 8 + sub * 16 + jbase;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const unsigned int q = half ? ((unsigned int)qs[u] >> 4) : ((unsigned int)qs[u] & 0x0fu);
        w[u] = dl * (float)dno_kvalues_iq4nl[q];
    }
}

// IQ4_NL: d(2B) + qs[16] = 18 B / 32 要素 (kernels_exp_iq_pf.cu の
// dno_exp_pf_iq4nl と同じ読み方。要素 0..15 が下ニブル、16..31 が上ニブル)。
__device__ __forceinline__ void dno_tc_w8_iq4nl(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;
    const unsigned char* blk = rowp + (size_t)s * 18u;
    const float d = dno_tcq_f16(blk);
    const int half = j >> 1;      // 0 = 下ニブル (要素 0..15) / 1 = 上ニブル (16..31)
    const int b0 = (j & 1) * 8;   // 32 ブロック内のバイト位置
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const unsigned int byte = (unsigned int)blk[2 + b0 + u];
        const unsigned int q = half ? (byte >> 4) : (byte & 0x0fu);
        w[u] = d * (float)dno_kvalues_iq4nl[q];
    }
}

// Q2_0: d(2B) + qs[16] = 18 B / 64 要素 (値は (q - 1) * d)。
__device__ __forceinline__ void dno_tc_w8_q2_0(const unsigned char* rowp, int k0, int j, float* w)
{
    const int b = k0 >> 6;   // 64 要素ブロック
    const unsigned char* blk = rowp + (size_t)b * 18u;
    const float d = dno_tcq_f16(blk);
    const int ofs = (k0 & 63) + j * 8;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int p = ofs + u;
        const unsigned int q = ((unsigned int)blk[2 + (p >> 2)] >> ((p & 3) * 2)) & 3u;
        w[u] = (float)((int)q - 1) * d;
    }
}

// 型番号: 0 = IQ2_S / 1 = IQ3_XXS / 2 = IQ3_S / 3 = IQ4_XS / 4 = IQ4_NL / 5 = Q2_0
template <int TY>
__device__ __forceinline__ void dno_tc_w8(const unsigned char* rowp, int k0, int j, float* w)
{
    if (TY == 0) dno_tc_w8_iq2s(rowp, k0, j, w);
    else if (TY == 1) dno_tc_w8_iq3xxs(rowp, k0, j, w);
    else if (TY == 2) dno_tc_w8_iq3s(rowp, k0, j, w);
    else if (TY == 3) dno_tc_w8_iq4xs(rowp, k0, j, w);
    else if (TY == 4) dno_tc_w8_iq4nl(rowp, k0, j, w);
    else dno_tc_w8_q2_0(rowp, k0, j, w);
}

// 8 個の f32 を bf16 に丸めて 16 B で書く (dst は 16 B 整列)。
__device__ __forceinline__ void dno_tcq_store8(__nv_bfloat16* dst, const float* v)
{
    unsigned int t[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        __nv_bfloat162 h = __floats2bfloat162_rn(v[2 * u], v[2 * u + 1]);
        t[u] = *reinterpret_cast<unsigned int*>(&h);
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(t[0], t[1], t[2], t[3]);
}

// 重みタイル 1 枚 (TC_BN 行 × 32 要素) を読む。スレッドは 1 行 8 要素。
template <int TY>
__device__ __forceinline__ void dno_tc_load_w(const unsigned char* __restrict__ w,
                                              int n0, int row_bytes, int k0, int tid, float* v)
{
    const int row = tid >> 2;
    const int j = tid & 3;
    dno_tc_w8<TY>(w + (size_t)(n0 + row) * (size_t)row_bytes, k0, j, v);
}

// 入力タイル (TC_BM 行 × 32 要素) を読む。行は rows[ro + m0 + row] のトークン。
__device__ __forceinline__ void dno_tc_load_x(const float* __restrict__ x,
                                              const unsigned int* __restrict__ rows,
                                              int ro, int m0, int m, int n_in, int k0, int tid,
                                              float* v)
{
    const int row = tid >> 2;
    const int j = tid & 3;
    const int t = m0 + row;
    if (t < m) {
        const unsigned int tk = rows[(size_t)ro + (size_t)t];
        const float* xp = x + (size_t)tk * (size_t)n_in + (size_t)k0 + (size_t)j * 8u;
        const float4 a = *reinterpret_cast<const float4*>(xp);
        const float4 b = *reinterpret_cast<const float4*>(xp + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = 0.0f;
    }
}

// segment の前置き (blockIdx.z)。barrier は無いので早期 return してよい。
struct TcSeg {
    int m;
    int ro;
};

__device__ __forceinline__ bool dno_tc_seg(const unsigned int* __restrict__ seg, int nseg,
                                           int m0, TcSeg* s)
{
    const int si = (int)blockIdx.z;
    if (si >= nseg) return false;
    const unsigned int* sg = seg + (size_t)si * 4u;
    s->m = (int)sg[0];
    s->ro = (int)sg[1];
    return s->m > 0 && m0 < s->m;
}

// ---------------------------------------------------------------------------
// gate + up + silu 積。act[(ro+t)*n_out + ncol] = silu(gate) * up
// ---------------------------------------------------------------------------
template <int TY>
__device__ __forceinline__ void dno_tc_body_gu(
    const unsigned long long* __restrict__ wgtab,
    const unsigned long long* __restrict__ wutab,
    const float* __restrict__ x,
    float* __restrict__ act,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ __align__(32) __nv_bfloat16 sWG[2 * TC_BN * TC_LD];
    __shared__ __align__(32) __nv_bfloat16 sWU[2 * TC_BN * TC_LD];
    __shared__ __align__(32) __nv_bfloat16 sX[2 * TC_BM * TC_LD];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 1;   // トークン方向 (32 本ずつ)
    const int wn = warp >> 1;  // 行方向 (16 本ずつ)

    const int m0 = (int)blockIdx.y * TC_BM;
    TcSeg s;
    if (!dno_tc_seg(seg, nseg, m0, &s)) return;
    const int n0 = (int)blockIdx.x * TC_BN;
    const unsigned char* wg = reinterpret_cast<const unsigned char*>(wgtab[blockIdx.z]);
    const unsigned char* wu = reinterpret_cast<const unsigned char*>(wutab[blockIdx.z]);

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accg[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accu[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        wmma::fill_fragment(accg[i], 0.0f);
        wmma::fill_fragment(accu[i], 0.0f);
    }

    const int kb = n_in >> 5;
    const int jcol = (tid & 3) * 8;
    const int jrow = tid >> 2;

    {
        float wv[8], uv[8], xv[8];
        dno_tc_load_w<TY>(wg, n0, row_bytes, 0, tid, wv);
        dno_tc_load_w<TY>(wu, n0, row_bytes, 0, tid, uv);
        dno_tc_load_x(x, rows, s.ro, m0, s.m, n_in, 0, tid, xv);
        dno_tcq_store8(sWG + jrow * TC_LD + jcol, wv);
        dno_tcq_store8(sWU + jrow * TC_LD + jcol, uv);
        dno_tcq_store8(sX + jrow * TC_LD + jcol, xv);
    }
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const int cur = (k & 1) * (TC_BN * TC_LD);
        const int curx = (k & 1) * (TC_BM * TC_LD);
        const int nxt = ((k + 1) & 1) * (TC_BN * TC_LD);
        const int nxtx = ((k + 1) & 1) * (TC_BM * TC_LD);
        const bool more = (k + 1) < kb;

        float wv[8], uv[8], xv[8];
        if (more) {
            dno_tc_load_w<TY>(wg, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_tc_load_w<TY>(wu, n0, row_bytes, (k + 1) << 5, tid, uv);
            dno_tc_load_x(x, rows, s.ro, m0, s.m, n_in, (k + 1) << 5, tid, xv);
        }

#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bg, bu;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = sX + curx + (wm * 32 + i * 16) * TC_LD + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC_LD);
            }
            wmma::load_matrix_sync(bg, sWG + cur + (wn * 16) * TC_LD + kk * 16, TC_LD);
            wmma::load_matrix_sync(bu, sWU + cur + (wn * 16) * TC_LD + kk * 16, TC_LD);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::mma_sync(accg[i], af[i], bg, accg[i]);
                wmma::mma_sync(accu[i], af[i], bu, accu[i]);
            }
        }

        if (more) {
            dno_tcq_store8(sWG + nxt + jrow * TC_LD + jcol, wv);
            dno_tcq_store8(sWU + nxt + jrow * TC_LD + jcol, uv);
            dno_tcq_store8(sX + nxtx + jrow * TC_LD + jcol, xv);
        }
        __syncthreads();
    }

    // 書き出し (行が m を超えるタイルは有効行だけ写す)。
    const int ncol = n0 + wn * 16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int trow = m0 + wm * 32 + i * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> tmp;
#pragma unroll
        for (int u = 0; u < tmp.num_elements; ++u) {
            const float g = accg[i].x[u];
            tmp.x[u] = (g / (1.0f + expf(-g))) * accu[i].x[u];
        }
        if (trow + 16 <= s.m) {
            wmma::store_matrix_sync(act + (size_t)(s.ro + trow) * (size_t)n_out + (size_t)ncol,
                                    tmp, n_out, wmma::mem_row_major);
        } else if (trow < s.m) {
            // 端数: 一旦 smem へ落として有効な行だけ写す (sX を作業域に使う)。
            float* st = reinterpret_cast<float*>(sX) + warp * 256;
            wmma::store_matrix_sync(st, tmp, 16, wmma::mem_row_major);
            __syncwarp();
            for (int r = lane; r < s.m - trow; r += 32) {
                const float* src = st + r * 16;
                float* dst = act + (size_t)(s.ro + trow + r) * (size_t)n_out + (size_t)ncol;
#pragma unroll
                for (int c = 0; c < 16; ++c) dst[c] = src[c];
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------
// down: out[(ro+t)*n_out + ncol] = W_down(ncol..ncol+15) · act[(ro+t)]
// ---------------------------------------------------------------------------
template <int TY>
__device__ __forceinline__ void dno_tc_body_dn(
    const unsigned long long* __restrict__ wdtab,
    const float* __restrict__ x,
    float* __restrict__ out,
    const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows,
    int nseg, int n_in, int n_out, int row_bytes)
{
    __shared__ __align__(32) __nv_bfloat16 sW[2 * TC_BN * TC_LD];
    __shared__ __align__(32) __nv_bfloat16 sX[2 * TC_BM * TC_LD];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp & 1;
    const int wn = warp >> 1;

    const int m0 = (int)blockIdx.y * TC_BM;
    TcSeg s;
    if (!dno_tc_seg(seg, nseg, m0, &s)) return;
    const int n0 = (int)blockIdx.x * TC_BN;
    const unsigned char* wd = reinterpret_cast<const unsigned char*>(wdtab[blockIdx.z]);

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) wmma::fill_fragment(acc[i], 0.0f);

    const int kb = n_in >> 5;
    const int jcol = (tid & 3) * 8;
    const int jrow = tid >> 2;

    {
        float wv[8], xv[8];
        dno_tc_load_w<TY>(wd, n0, row_bytes, 0, tid, wv);
        dno_tc_load_x(x, rows, s.ro, m0, s.m, n_in, 0, tid, xv);
        dno_tcq_store8(sW + jrow * TC_LD + jcol, wv);
        dno_tcq_store8(sX + jrow * TC_LD + jcol, xv);
    }
    __syncthreads();

    for (int k = 0; k < kb; ++k) {
        const int cur = (k & 1) * (TC_BN * TC_LD);
        const int curx = (k & 1) * (TC_BM * TC_LD);
        const int nxt = ((k + 1) & 1) * (TC_BN * TC_LD);
        const int nxtx = ((k + 1) & 1) * (TC_BM * TC_LD);
        const bool more = (k + 1) < kb;

        float wv[8], xv[8];
        if (more) {
            dno_tc_load_w<TY>(wd, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_tc_load_x(x, rows, s.ro, m0, s.m, n_in, (k + 1) << 5, tid, xv);
        }

#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __nv_bfloat16* p = sX + curx + (wm * 32 + i * 16) * TC_LD + kk * 16;
                wmma::load_matrix_sync(af[i], p, TC_LD);
            }
            wmma::load_matrix_sync(bf, sW + cur + (wn * 16) * TC_LD + kk * 16, TC_LD);
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::mma_sync(acc[i], af[i], bf, acc[i]);
        }

        if (more) {
            dno_tcq_store8(sW + nxt + jrow * TC_LD + jcol, wv);
            dno_tcq_store8(sX + nxtx + jrow * TC_LD + jcol, xv);
        }
        __syncthreads();
    }

    const int ncol = n0 + wn * 16;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int trow = m0 + wm * 32 + i * 16;
        if (trow + 16 <= s.m) {
            wmma::store_matrix_sync(out + (size_t)(s.ro + trow) * (size_t)n_out + (size_t)ncol,
                                    acc[i], n_out, wmma::mem_row_major);
        } else if (trow < s.m) {
            float* st = reinterpret_cast<float*>(sX) + warp * 256;
            wmma::store_matrix_sync(st, acc[i], 16, wmma::mem_row_major);
            __syncwarp();
            for (int r = lane; r < s.m - trow; r += 32) {
                const float* src = st + r * 16;
                float* dst = out + (size_t)(s.ro + trow + r) * (size_t)n_out + (size_t)ncol;
#pragma unroll
                for (int c = 0; c < 16; ++c) dst[c] = src[c];
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void dno_exp_tc_gu_iq2s(
    const unsigned long long* __restrict__ wgtab, const unsigned long long* __restrict__ wutab,
    const float* __restrict__ x, float* __restrict__ act, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_gu<0>(wgtab, wutab, x, act, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_gu_iq3xxs(
    const unsigned long long* __restrict__ wgtab, const unsigned long long* __restrict__ wutab,
    const float* __restrict__ x, float* __restrict__ act, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_gu<1>(wgtab, wutab, x, act, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_gu_iq3s(
    const unsigned long long* __restrict__ wgtab, const unsigned long long* __restrict__ wutab,
    const float* __restrict__ x, float* __restrict__ act, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_gu<2>(wgtab, wutab, x, act, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_gu_iq4xs(
    const unsigned long long* __restrict__ wgtab, const unsigned long long* __restrict__ wutab,
    const float* __restrict__ x, float* __restrict__ act, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_gu<3>(wgtab, wutab, x, act, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_iq2s(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<0>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_iq3xxs(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<1>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_iq3s(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<2>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_iq4xs(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<3>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_iq4nl(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<4>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}

extern "C" __global__ void dno_exp_tc_dn_q2_0(
    const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,
    float* __restrict__ out, const unsigned int* __restrict__ seg,
    const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes)
{
    dno_tc_body_dn<5>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);
}
