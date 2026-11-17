// denno CUDA tensor core grouped expert GEMM — **K 系** (Q4_K / Q5_K / Q6_K /
// Q5_1 / Q8_0) (task 416)。
//
// `kernels_exp_iq_tc.cu` (task 250) と同じ形で、重みのデコードだけを K 系に
// 差し替えたもの。IQ 版は既定 on で IQ の prefill を 2.5 倍にしており、K 系
// (UD-Q4_K_XL) だけが m 行 GEMV (`kernels_exp_k_pf.cu`) のまま残っていた。
//
// PTX へのコンパイル (build_ptx.sh の `exp_k_tc`、この target は compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 kernels_exp_k_tc.cu -o kernels_exp_k_tc.ptx
//
// bf16 の mma は compute_80 以降にしか無いので arch を上げる (PTX は前方互換
// なので実機 (sm_120) で JIT される)。
//
// # 何を置き換えるか
//
// 現行の prefill は `kernels_exp_k_pf.cu` の m 行 GEMV で、1 block = 8 出力行。
// 出力行ごとに活性の行 (n_in 要素) を読み直すので、1 expert あたりの活性読みが
// `n_out/8 × m × n_in × 4 B` になる (m=438 / e=2560 / ff=1536 で 861 MB)。
// 1 expert の重みは約 2 MB なので、層優先 (行数が多い) では**活性律速**になる。
//
// ここでは 64 トークン × 64 出力行のタイルで受けるので、活性の読みは
// `n_out/64 × m × n_in × 4 B` で済む (同じ形で 108 MB、8 分の 1)。
// 重みは smem に 1 回デコードし、bf16 の mma (FP32 アキュムレート) で積む。
//
// # 数値契約
//
// **なし** (prefill の射影は 2026-10-28 のオペレーター指示でビット一致契約の外。
// `kernels_exp_iq_tc.cu` と同じ)。重みは K 系のデコード結果を bf16 に丸め、
// 活性も bf16 に丸めて mma する。アキュムレートは FP32 で、行ごとの和は k 昇順
// のまま (タイル幅 32 の逐次)。既存の `kernels_exp_k_pf.cu` (f32 FMA・32 レーン
// 分割) とも `kernels_seg_k.cu` とも値が変わる。丸めだけの差で、演算の意味は同じ。
//
// # segment 表 (const unsigned int* seg、1 segment = 4 語)
//
//   seg[s*4 + 0] = m        (この expert の行数)
//   seg[s*4 + 1] = row_off  (rows の先頭添字。出力も同じ添字 × n_out)
//   seg[s*4 + 2] = 0        (予約)
//   seg[s*4 + 3] = 0        (予約)
//
// `rows[i]` は行 i のトークン番号で、入力は `x + rows[i] * n_in`。down 行列は
// 入力が `act` (行が詰まっている) なので `rows[i] = i` を渡す。
//
// # 型番号 (ホストの `cuda::EXPKTC_*` と一致させること)
//
//   0 = Q4_K / 1 = Q5_K / 2 = Q6_K / 3 = Q5_1 / 4 = Q8_0
//
// # 起動形 (ホスト側 `pf_launch_group_tc` が同じ条件を検査する)
//
//   grid  = (n_out / TC_BN, ceil(max_m / TC_BM), nseg)
//   block = TC_THREADS = 256
//   n_out % TC_BN == 0、n_in % 32 == 0 (タイル幅は 32 要素)
//   256 要素ブロックの型 (Q4_K / Q5_K / Q6_K) は n_in % 256 == 0 も要る。
//
// 共有メモリ (静的): gate/up は 2 枚、down は 1 枚。
//   gate/up: 2 × TC_BN × TC_LD × 2 B × 2 段 × 2 枚 = 40,960 B
//   down   : 2 × TC_BN × TC_LD × 2 B × 2 段     = 20,480 B
//   入力   : 2 × TC_BM × TC_LD × 2 B × 2 段     = 20,480 B
// 最大 40,960 B (48 KiB 既定の内側。opt-in は要らない)。

#include <mma.h>
#include <cuda_bf16.h>

// タイル。値の行にコメントを付けないこと (kernel_invariants が `#define` の
// 行末までを値として読む)。
#define TC_BM 64
#define TC_BN 64
#define TC_BK 32
#define TC_LD 40
#define TC_THREADS 256

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_exp_k_pf.cu の kpf_f16 と同一)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_ktcq_f16(const unsigned char* b)
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

__device__ __forceinline__ unsigned int dno_ktcq_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16)
           | ((unsigned int)p[3] << 24);
}

// Q4_K / Q5_K の 6 bit スケールと最小値 (ggml の get_scale_min_k4 と同じ式)。
__device__ __forceinline__ void dno_ktcq_scale_min_k4(int j, const unsigned char* q,
                                                      unsigned int* sc, unsigned int* mn)
{
    if (j < 4) {
        *sc = (unsigned int)q[j] & 63u;
        *mn = (unsigned int)q[j + 4] & 63u;
    } else {
        *sc = ((unsigned int)q[j + 4] & 0x0fu) | (((unsigned int)q[j - 4] >> 6) << 4);
        *mn = ((unsigned int)q[j + 4] >> 4) | (((unsigned int)q[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// 8 要素ぶんの重み (絶対位置 k0 + j*8) を f32 へ。k0 は 32 の倍数
// (タイル幅 TC_BK = 32) なので、j = 0..3 が 32 要素タイルの中の 8 要素を指す。
// 各式は kernels_exp_k_pf.cu の kpf_w8_* と 1:1 (lane を j に置き換えただけ)。
// ---------------------------------------------------------------------------

// Q4_K: 256 要素 / 144 B (d, dmin, scales[12], qs[128])
__device__ __forceinline__ void dno_ktc_w8_q4k(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;                 // 256 ブロック内の 32 要素サブブロック
    const unsigned char* blk = rowp + (size_t)(k0 >> 8) * 144u;
    const float d = dno_ktcq_f16(blk);
    const float dmin = dno_ktcq_f16(blk + 2);
    const int sb = s & 7;                  // サブブロック
    const int p = j << 3;                  // サブブロック内の先頭要素
    unsigned int sc, mn;
    dno_ktcq_scale_min_k4(sb, blk + 4, &sc, &mn);
    const float dv = d * (float)sc;
    const float mv = dmin * (float)mn;
    const unsigned char* qs = blk + 16u + (size_t)(sb >> 1) * 32u + p;
    const int sh = (sb & 1) ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int q = ((unsigned int)qs[i] >> sh) & 0x0fu;
        w[i] = dv * (float)q - mv;
    }
}

// Q5_K: 256 要素 / 176 B (d, dmin, scales[12], qh[32], qs[128])
__device__ __forceinline__ void dno_ktc_w8_q5k(const unsigned char* rowp, int k0, int j, float* w)
{
    const int s = k0 >> 5;
    const unsigned char* blk = rowp + (size_t)(k0 >> 8) * 176u;
    const float d = dno_ktcq_f16(blk);
    const float dmin = dno_ktcq_f16(blk + 2);
    const int sb = s & 7;
    const int p = j << 3;
    unsigned int sc, mn;
    dno_ktcq_scale_min_k4(sb, blk + 4, &sc, &mn);
    const float dv = d * (float)sc;
    const float mv = dmin * (float)mn;
    const unsigned char* qh = blk + 16u;   // 32 B (8 サブブロック × 1 bit)
    const unsigned char* qs = blk + 48u + (size_t)(sb >> 1) * 32u + p;
    const int sh = (sb & 1) ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int q = ((unsigned int)qs[i] >> sh) & 0x0fu;
        const unsigned int hb = ((unsigned int)qh[p + i] >> sb) & 1u;
        w[i] = dv * (float)(q + (hb << 4)) - mv;
    }
}

// Q6_K: 256 要素 / 210 B (ql[128], qh[64], scales i8[16], d)
__device__ __forceinline__ void dno_ktc_w8_q6k(const unsigned char* rowp, int k0, int j, float* w)
{
    const unsigned char* blk = rowp + (size_t)(k0 >> 8) * 210u;
    const float d = dno_ktcq_f16(blk + 208);
    const int half = (k0 >> 7) & 1;        // 128 要素の半分
    const int ch = (k0 >> 5) & 3;          // 半分の中の 32 要素の面
    const int l2 = j << 3;                 // 面の中の先頭要素 (0..24)
    const unsigned char* ql = blk + (size_t)half * 64u + ((ch & 1) ? 32u : 0u) + l2;
    const unsigned char* qh = blk + 128u + (size_t)half * 32u + l2;
    const signed char* sc = (const signed char*)(blk + 192u) + (size_t)half * 8u;
    const int s = (int)sc[2 * ch + (l2 >= 16 ? 1 : 0)];
    const int sh = (ch >= 2) ? 4 : 0;
    const float dv = d * (float)s;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int nib = ((unsigned int)ql[i] >> sh) & 0x0fu;
        const unsigned int h2 = ((unsigned int)qh[i] >> (2 * ch)) & 3u;
        const int q6 = (int)(nib | (h2 << 4)) - 32;
        w[i] = dv * (float)q6;
    }
}

// Q5_1: 32 要素 / 24 B (d, m, qh[4], qs[16])。y = d * (q + 16 * hb) + m
__device__ __forceinline__ void dno_ktc_w8_q5_1(const unsigned char* rowp, int k0, int j, float* w)
{
    const unsigned char* blk = rowp + (size_t)(k0 >> 5) * 24u;
    const float d = dno_ktcq_f16(blk);
    const float m = dno_ktcq_f16(blk + 2);
    const unsigned int qh = dno_ktcq_u32(blk + 4);
    const int p = j << 3;                          // 32 要素の中の先頭 (0..24)
    const int hi = (p >= 16) ? 1 : 0;              // 要素 16..31 は上ニブル
    const unsigned char* qs = blk + 8 + (p & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int byte = (unsigned int)qs[i];
        const unsigned int q = hi ? (byte >> 4) : (byte & 0x0fu);
        const unsigned int hb = (qh >> (p + i)) & 1u;
        w[i] = d * (float)(q + (hb << 4)) + m;
    }
}

// Q8_0: 32 要素 / 34 B (d + qs[32] i8)
__device__ __forceinline__ void dno_ktc_w8_q8_0(const unsigned char* rowp, int k0, int j, float* w)
{
    const unsigned char* blk = rowp + (size_t)(k0 >> 5) * 34u;
    const float d = dno_ktcq_f16(blk);
    const int p = j << 3;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        w[i] = d * (float)(int)(signed char)blk[2 + p + i];
    }
}

// 型番号: 0 = Q4_K / 1 = Q5_K / 2 = Q6_K / 3 = Q5_1 / 4 = Q8_0
template <int TY>
__device__ __forceinline__ void dno_ktc_w8(const unsigned char* rowp, int k0, int j, float* w)
{
    if (TY == 0) dno_ktc_w8_q4k(rowp, k0, j, w);
    else if (TY == 1) dno_ktc_w8_q5k(rowp, k0, j, w);
    else if (TY == 2) dno_ktc_w8_q6k(rowp, k0, j, w);
    else if (TY == 3) dno_ktc_w8_q5_1(rowp, k0, j, w);
    else dno_ktc_w8_q8_0(rowp, k0, j, w);
}

// 8 個の f32 を bf16 に丸めて 16 B で書く (dst は 16 B 整列)。
__device__ __forceinline__ void dno_ktcq_store8(__nv_bfloat16* dst, const float* v)
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
__device__ __forceinline__ void dno_ktc_load_w(const unsigned char* __restrict__ w,
                                               int n0, int row_bytes, int k0, int tid, float* v)
{
    const int row = tid >> 2;
    const int j = tid & 3;
    dno_ktc_w8<TY>(w + (size_t)(n0 + row) * (size_t)row_bytes, k0, j, v);
}

// 入力タイル (TC_BM 行 × 32 要素) を読む。行は rows[ro + m0 + row] のトークン。
__device__ __forceinline__ void dno_ktc_load_x(const float* __restrict__ x,
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
struct KtcSeg {
    int m;
    int ro;
};

__device__ __forceinline__ bool dno_ktc_seg(const unsigned int* __restrict__ seg, int nseg,
                                            int m0, KtcSeg* s)
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
__device__ __forceinline__ void dno_ktc_body_gu(
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
    KtcSeg s;
    if (!dno_ktc_seg(seg, nseg, m0, &s)) return;
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
        dno_ktc_load_w<TY>(wg, n0, row_bytes, 0, tid, wv);
        dno_ktc_load_w<TY>(wu, n0, row_bytes, 0, tid, uv);
        dno_ktc_load_x(x, rows, s.ro, m0, s.m, n_in, 0, tid, xv);
        dno_ktcq_store8(sWG + jrow * TC_LD + jcol, wv);
        dno_ktcq_store8(sWU + jrow * TC_LD + jcol, uv);
        dno_ktcq_store8(sX + jrow * TC_LD + jcol, xv);
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
            dno_ktc_load_w<TY>(wg, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_ktc_load_w<TY>(wu, n0, row_bytes, (k + 1) << 5, tid, uv);
            dno_ktc_load_x(x, rows, s.ro, m0, s.m, n_in, (k + 1) << 5, tid, xv);
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
            dno_ktcq_store8(sWG + nxt + jrow * TC_LD + jcol, wv);
            dno_ktcq_store8(sWU + nxt + jrow * TC_LD + jcol, uv);
            dno_ktcq_store8(sX + nxtx + jrow * TC_LD + jcol, xv);
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
__device__ __forceinline__ void dno_ktc_body_dn(
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
    KtcSeg s;
    if (!dno_ktc_seg(seg, nseg, m0, &s)) return;
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
        dno_ktc_load_w<TY>(wd, n0, row_bytes, 0, tid, wv);
        dno_ktc_load_x(x, rows, s.ro, m0, s.m, n_in, 0, tid, xv);
        dno_ktcq_store8(sW + jrow * TC_LD + jcol, wv);
        dno_ktcq_store8(sX + jrow * TC_LD + jcol, xv);
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
            dno_ktc_load_w<TY>(wd, n0, row_bytes, (k + 1) << 5, tid, wv);
            dno_ktc_load_x(x, rows, s.ro, m0, s.m, n_in, (k + 1) << 5, tid, xv);
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
            dno_ktcq_store8(sW + nxt + jrow * TC_LD + jcol, wv);
            dno_ktcq_store8(sX + nxtx + jrow * TC_LD + jcol, xv);
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

#define KTC_GU_ENTRY(NAME, TY)                                                              \
    extern "C" __global__ void NAME(                                                        \
        const unsigned long long* __restrict__ wgtab, const unsigned long long* __restrict__ wutab, \
        const float* __restrict__ x, float* __restrict__ act,                               \
        const unsigned int* __restrict__ seg, const unsigned int* __restrict__ rows,        \
        int nseg, int n_in, int n_out, int row_bytes)                                       \
    {                                                                                       \
        dno_ktc_body_gu<TY>(wgtab, wutab, x, act, seg, rows, nseg, n_in, n_out, row_bytes);  \
    }

#define KTC_DN_ENTRY(NAME, TY)                                                              \
    extern "C" __global__ void NAME(                                                        \
        const unsigned long long* __restrict__ wdtab, const float* __restrict__ x,           \
        float* __restrict__ out, const unsigned int* __restrict__ seg,                       \
        const unsigned int* __restrict__ rows, int nseg, int n_in, int n_out, int row_bytes) \
    {                                                                                       \
        dno_ktc_body_dn<TY>(wdtab, x, out, seg, rows, nseg, n_in, n_out, row_bytes);         \
    }

KTC_GU_ENTRY(dno_exp_ktc_gu_q4k, 0)
KTC_GU_ENTRY(dno_exp_ktc_gu_q5k, 1)
KTC_GU_ENTRY(dno_exp_ktc_gu_q6k, 2)
KTC_GU_ENTRY(dno_exp_ktc_gu_q5_1, 3)
KTC_GU_ENTRY(dno_exp_ktc_gu_q8_0, 4)

KTC_DN_ENTRY(dno_exp_ktc_dn_q4k, 0)
KTC_DN_ENTRY(dno_exp_ktc_dn_q5k, 1)
KTC_DN_ENTRY(dno_exp_ktc_dn_q6k, 2)
KTC_DN_ENTRY(dno_exp_ktc_dn_q5_1, 3)
KTC_DN_ENTRY(dno_exp_ktc_dn_q8_0, 4)
