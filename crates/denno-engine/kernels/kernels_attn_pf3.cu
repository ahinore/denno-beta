// denno CUDA prefill attention v3 (task 230) — PV も bf16 の mma にし、
// グリッドを head 優先にして K/V の L2 再利用を狙う版。
//
// PTX へのコンパイル (build_ptx.sh の `attn_pf3`、compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 -fmad=false kernels_attn_pf3.cu -o kernels_attn_pf3.ptx
//
// ---- なぜ v3 が要るか (lab t230a の台帳、14,018 トークン) ------------------
//  full の wall は 1,976.7 ms / gpu 1,631.7 ms で、qkv + out + gate + 転送を
//  引くと attention 本体が約 1,450 ms。v2 の無駄は 3 つ:
//    - PV が f32 のスカラー FMA。1 キーあたり 1 スレッドが smem から V を 16 個
//      読むが、同じ V を 16 個の行グループが重複して読むので smem は 512 KB/タイル。
//    - 1 タイル 1 ブロック 64 クエリなので、K/V のグローバル読みが 64 KB/タイル
//      (全体で 883 GB)。full は 12 層 = 24 head が同じ kv head を共有するのに、
//      グリッドが (クエリブロック, head) なので同時に走るブロックが同じ K/V を
//      読まない (L2 に乗らない)。
//    - マスク / 行最大 / exp が別パスで、s_sm を 3 回読み書きする。
//
//  ここでは:
//    1. PV を bf16 の wmma にする (V は smem で bf16)。smem は 72 KB/タイル。
//    2. 行最大は最初のタイルで 1 回だけ取り (m0)、全タイルで
//       p = exp2((s - m0) * log2e) とする。オンライン再スケールが消えるので
//       アキュムレータを mma のフラグメントのまま使える (行ごとの掛け直しが不要)。
//       スコアは scale 済みで |s| は数十に収まるので f32 の範囲は問題ない。
//    3. マスクを exp の直前の 1 パスに畳む (v2 は 3 パス)。
//    4. グリッドを (head, クエリブロック) に入れ替え、クエリブロックは重い方
//       (大きい i0) から走らせる。同じクエリ範囲の 24 head が同時に同じ K/V を
//       読むので L2 に乗る。
//
// ---- 数値 (attention の緩和枠。P17g で CPU とのビット一致は対象外) --------
//  スコアは q と k を BF16 に丸めて mma (v2 と同じ)。p は bf16 に丸めて PV の
//  mma に渡し、l もその bf16 の p を足す (acc と l が同じ p で閉じる)。
//  アキュムレートは f32・決定論 (固定タイル・固定順・原子操作なし)。
//
// ---- 前提 ----------------------------------------------------------------
//   - q は [n][heads*dh] の f32 (per-head RMSNorm + rope 済み)
//   - kc / vc は [t_total][kv_dim] の f32
//   - att は [n][heads*dh] の f32 (このカーネルが最後に 1/l を掛ける)
//   - dh == 256 固定。ホストが dh == 256 を検査する。
//   - 1 ブロック = 64 クエリ行 x 1 head、256 スレッド。
//
// 共有メモリ (動的、ホストが ATTN_PF3_SMEM_BYTES を渡す):
//   q_sm 64x264 bf16 = 33,792 B / k_sm 32x264 bf16 = 16,896 B
//   v_sm 32x264 bf16 = 16,896 B / p_sm 64x40 bf16  =  5,120 B
//   s_sm 64x32 f32  =  8,192 B / m0 64 f32 + lst 64 f32 = 512 B
//   stg 8 ワープ x 16x16 f32 = 8,192 B            合計 89,600 B

#include <mma.h>
#include <cuda_bf16.h>

// タイル (値の行にコメントを付けないこと。kernel_invariants が読む)。
#define A3_BQ 64
#define A3_BK 32
#define A3_QD 264
#define A3_PD 40
#define A3_THREADS 256
#define A3_SMEM_BYTES 89600

// exp2f 用の定数 (kernels_elem.cu の LOG2E_F と同じ値)。
#define A3_LOG2E 1.4426950408889634f

// 4 個の f32 を bf16 4 個 (8 B) にして書く。
__device__ __forceinline__ void dno_a3_store4(__nv_bfloat16 *dst, const float *v)
{
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
    unsigned int t0 = *reinterpret_cast<unsigned int *>(&a);
    unsigned int t1 = *reinterpret_cast<unsigned int *>(&b);
    *reinterpret_cast<uint2 *>(dst) = make_uint2(t0, t1);
}

extern "C" __global__ void dno_attention_prefill3(
    const float *__restrict__ q,   // [n, heads*dh]
    const float *__restrict__ kc,  // [t_total, kv_dim]
    const float *__restrict__ vc,  // [t_total, kv_dim]
    float *__restrict__ att,       // [n, heads*dh]
    int heads, int n, int pos0, int kv_dim, int t_total, int group, int dh,
    float scale)
{
    extern __shared__ __align__(32) unsigned char sh_raw[];
    __nv_bfloat16 *q_sm = reinterpret_cast<__nv_bfloat16 *>(sh_raw);
    __nv_bfloat16 *k_sm = q_sm + A3_BQ * A3_QD;
    __nv_bfloat16 *v_sm = k_sm + A3_BK * A3_QD;
    __nv_bfloat16 *p_sm = v_sm + A3_BK * A3_QD;
    float *s_sm = reinterpret_cast<float *>(p_sm + A3_BQ * A3_PD);
    float *m0_sm = s_sm + A3_BQ * A3_BK;
    float *lst_sm = m0_sm + A3_BQ;
    float *stg = lst_sm + A3_BQ;   // 8 warp x 256 f32

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // グリッドは (head, クエリブロック)。クエリブロックは重い方から。
    const int h = blockIdx.x;
    const int nb = gridDim.y;
    const int i0 = (nb - 1 - (int)blockIdx.y) * A3_BQ;
    const int qd = heads * dh;
    const int kh_off = (h / group) * dh;
    // このブロックが見る必要のある最大キー (行ごとの causal 界の最大)
    const int jend = min(t_total, pos0 + i0 + A3_BQ);

    // ---- Q の bf16 化 (1 回だけ) ------------------------------------------
    {
        const int r = tid >> 2;             // 0..63
        const int qd0 = (tid & 3) * 64;     // 0, 64, 128, 192
        const int i = i0 + r;
        __nv_bfloat16 *dst = q_sm + r * A3_QD + qd0;
        if (i < n) {
            const float *src = q + (size_t)i * heads * dh + (size_t)h * dh + qd0;
#pragma unroll
            for (int blk = 0; blk < 16; ++blk) {
                const float4 x = *reinterpret_cast<const float4 *>(src + 4 * blk);
                const float v[4] = {x.x, x.y, x.z, x.w};
                dno_a3_store4(dst + 4 * blk, v);
            }
        } else {
#pragma unroll
            for (int u = 0; u < 32; ++u) {
                *reinterpret_cast<uint2 *>(dst + 2 * u) = make_uint2(0u, 0u);
            }
        }
    }
    if (tid < A3_BQ) {
        lst_sm[tid] = 0.0f;
        m0_sm[tid] = 0.0f;
    }
    __syncthreads();

    using namespace nvcuda;

    // PV のアキュムレータ: ワープ w が行 (w>>1)*16 の 16 行 x 次元 (w&1)*128 の
    // 8 タイル (16x16) を持つ。行ごとの再スケールが無いので mma のまま使える。
    const int mt = warp >> 1;
    const int nd = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) wmma::fill_fragment(acc[t], 0.0f);

    for (int j0 = 0; j0 < jend; j0 += A3_BK) {
        // ---- K (bf16) と V (bf16) のタイルを読む --------------------------
        {
            const int jl = tid >> 3;             // 0..31 (タイル内のキー)
            const int dd = (tid & 7) * 32;       // 0, 32, ..., 224
            const int j = j0 + jl;
            __nv_bfloat16 *kd = k_sm + jl * A3_QD + dd;
            __nv_bfloat16 *vd = v_sm + jl * A3_QD + dd;
            if (j < t_total) {
                const float *ksrc = kc + (size_t)j * kv_dim + kh_off + dd;
                const float *vsrc = vc + (size_t)j * kv_dim + kh_off + dd;
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    const float4 kv = *reinterpret_cast<const float4 *>(ksrc + 4 * blk);
                    const float kx[4] = {kv.x, kv.y, kv.z, kv.w};
                    dno_a3_store4(kd + 4 * blk, kx);
                    const float4 vv4 = *reinterpret_cast<const float4 *>(vsrc + 4 * blk);
                    const float vx[4] = {vv4.x, vv4.y, vv4.z, vv4.w};
                    dno_a3_store4(vd + 4 * blk, vx);
                }
            } else {
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    *reinterpret_cast<uint2 *>(kd + 4 * blk) = make_uint2(0u, 0u);
                    *reinterpret_cast<uint2 *>(vd + 4 * blk) = make_uint2(0u, 0u);
                }
            }
        }
        __syncthreads();

        // ---- スコア: S = Q @ K^T (bf16 mma, FP32 アキュムレート) ------------
        {
            const int nt = warp & 1;    // 0..1  (16 キータイル)
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sp;
            wmma::fill_fragment(sp, 0.0f);
            for (int kk = 0; kk < (dh >> 4); ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, q_sm + (mt * 16) * A3_QD + kk * 16, A3_QD);
                wmma::load_matrix_sync(bf, k_sm + (nt * 16) * A3_QD + kk * 16, A3_QD);
                wmma::mma_sync(sp, af, bf, sp);
            }
            wmma::store_matrix_sync(
                s_sm + (mt * 16) * A3_BK + nt * 16, sp, A3_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- マスク + scale (無効 pair は -inf) -----------------------------
        for (int idx = tid; idx < A3_BQ * A3_BK; idx += A3_THREADS) {
            const int r = idx >> 5;
            const int key = idx & 31;
            const int i = i0 + r;
            const int j = j0 + key;
            float s;
            if (i < n && j < t_total && j <= pos0 + i) {
                s = s_sm[idx] * scale;
            } else {
                s = -INFINITY;
            }
            s_sm[idx] = s;
        }
        __syncthreads();

        // ---- 行最大は最初のタイルで 1 回だけ (以降は m0 を基準にする) -------
        if (j0 == 0) {
            if (tid < A3_BQ) {
                float mx = -INFINITY;
                for (int kk = 0; kk < A3_BK; ++kk) mx = fmaxf(mx, s_sm[tid * A3_BK + kk]);
                // 全滅 (行が範囲外) のときは 0 を基準にする (-inf - 0 → exp 0)。
                m0_sm[tid] = (mx > -INFINITY) ? mx : 0.0f;
            }
            __syncthreads();
        }

        // ---- p = exp2((s - m0) * log2e) を bf16 で書く ----------------------
        for (int idx = tid; idx < A3_BQ * A3_BK; idx += A3_THREADS) {
            const int r = idx >> 5;
            const float p = exp2f((s_sm[idx] - m0_sm[r]) * A3_LOG2E);
            p_sm[r * A3_PD + (idx & 31)] = __float2bfloat16_rn(p);
        }
        __syncthreads();

        // ---- l の更新 (owner = thread r) と PV (bf16 mma) を並べて回す ------
        {
            if (tid < A3_BQ) {
                float l = lst_sm[tid];
                for (int kk = 0; kk < A3_BK; ++kk) {
                    l += __bfloat162float(p_sm[tid * A3_PD + kk]);
                }
                lst_sm[tid] = l;
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
#pragma unroll
            for (int ks = 0; ks < (A3_BK >> 4); ++ks) {
                wmma::load_matrix_sync(af, p_sm + (mt * 16) * A3_PD + ks * 16, A3_PD);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int d0 = nd * 128 + t * 16;
                    wmma::load_matrix_sync(bf, v_sm + (ks * 16) * A3_QD + d0, A3_QD);
                    wmma::mma_sync(acc[t], af, bf, acc[t]);
                }
            }
        }
        __syncthreads();
    }

    // ---- 正規化前の acc を書き出す ---------------------------------------
    // 1/l を掛けるのは、書いた後にもう 1 パス (行ごとの係数は smem にしか無い)。
    {
        const int r0 = i0 + mt * 16;
        const bool full = (r0 + 16 <= n);
        for (int t = 0; t < 8; ++t) {
            const int d0 = h * dh + nd * 128 + t * 16;
            if (full) {
                wmma::store_matrix_sync(
                    att + (size_t)r0 * qd + d0, acc[t], qd, wmma::mem_row_major);
            } else {
                float *st = stg + warp * 256;
                wmma::store_matrix_sync(st, acc[t], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32) {
                    const int i = r0 + (e >> 4);
                    if (i < n) {
                        att[(size_t)i * qd + d0 + (e & 15)] = st[e];
                    }
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // ---- 1/l を掛ける (1 スレッドが 1 次元を 64 行ぶん) --------------------
    // 行ごとの係数を配列に持つと 64 レジスタを常時生かすことになるので、
    // 1 行ずつ読んで掛ける (acc のフラグメントと同居させる)。
    for (int r = 0; r < A3_BQ; ++r) {
        const int i = i0 + r;
        if (i < n) {
            const float l = lst_sm[r];
            const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            att[(size_t)i * qd + h * dh + tid] *= inv;
        }
    }
}
