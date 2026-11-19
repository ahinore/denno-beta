// denno CUDA prefill attention v4 (task 465) — K/V を **bf16 で読む** v3。
//
// PTX へのコンパイル (build_ptx.sh の `attn_pf4`、compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 -fmad=false kernels_attn_pf4.cu -o kernels_attn_pf4.ptx
//
// ---- なぜ v4 が要るか (lab t461d の台帳、14,018 トークン) --------------------
//  full 段の GPU は 637.7 ms。うち約 100 ms が qkv / out の GEMM なので、
//  attention 本体は約 540 ms。v3 の 1 ブロックは 64 クエリ行で、K/V のタイルを
//  32 キーずつ読む。1 クエリ行あたりの K/V 読みはプレフィックス全体なので
//  14k の 1 層で 219 ブロック x 24 head x 平均 7,010 キー x 512 次元 x 4 B x 2 本
//  = 75 GB、12 層で 約 900 GB。540 ms は **1.67 TB/s で HBM の帯域
//  (1.79 TB/s) に張り付いている** (24 head の共有は L2 では吸いきれない)。
//
//  クエリ行を増やす手は無い: sm_120 の動的 smem の上限が 99 KB で、
//  v3 の 89.6 KB のうち q_sm だけで 33.8 KB、k/v のタイルが 33.8 KB を占める。
//  128 行にすると q_sm だけで 67.6 KB になり p/s/stg が入らない (実測の
//  `attn_pf dyn smem 99KB` が上限)。
//
//  そこで**読む側の型**を変える。v3 は K/V を f32 で読み、smem へ入れるときに
//  `__floats2bfloat162_rn` で bf16 に丸めている。つまり **カーネルが使う値は
//  元から bf16** なので、bf16 のミラーを読んでも値は 1 ビットも変わらない。
//  読みはちょうど半分 (900 -> 450 GB) になり、変換もしなくなる。
//
// ---- v3 との違い (これだけ) ------------------------------------------------
//  - 引数 `kc` / `vc` が `const __nv_bfloat16 *`。
//  - K/V タイルの読みが 8 バイトのコピー (uint2) x 8。f32 の float4 ロードと
//    丸めが消える。smem のレイアウト・オフセット・タイル形状は v3 と同一。
//  - それ以外 (スコア / マスク / exp2 / PV / 書き出し / 1/l) は v3 と同一。
//
// 数値: v3 と**同一**。bf16 のミラーは v3 が smem に入れていたのと同じ値
//  (`__floats2bfloat162_rn` の丸め) を持つ。決定論も同じ (固定タイル・固定順)。
//
// ---- 前提 ----------------------------------------------------------------
//   - q は [n][heads*dh] の f32 (per-head RMSNorm + rope 済み)
//   - kc / vc は [t_total][kv_dim] の **bf16** (ホストが f32 のミラーから作る)
//   - att は [n][heads*dh] の f32 (このカーネルが最後に 1/l を掛ける)
//   - dh == 256 固定。ホストが dh == 256 を検査する。
//   - 1 ブロック = 64 クエリ行 x 1 head、256 スレッド。
//
// 共有メモリ (動的、ホストが A4_SMEM_BYTES を渡す):
//   q_sm 64x264 bf16 = 33,792 B / k_sm 32x264 bf16 = 16,896 B
//   v_sm 32x264 bf16 = 16,896 B / p_sm 64x40 bf16  =  5,120 B
//   s_sm 64x32 f32  =  8,192 B / m0 64 f32 + lst 64 f32 = 512 B
//   stg 8 ワープ x 16x16 f32 = 8,192 B            合計 89,600 B
//
// ---- 同じファイルの補助カーネル --------------------------------------------
//   `dno_kv_f32_to_bf16`: f32 の K/V (行 [r0, r0+n)) を bf16 へ落とす。
//   ホストは層の最初のチャンクで行 0 から、以降は前回の続きから呼ぶ。

#include <mma.h>
#include <cuda_bf16.h>

// タイル (値の行にコメントを付けないこと。kernel_invariants が読む)。
#define A4_BQ 64
#define A4_BK 32
#define A4_QD 264
#define A4_PD 40
#define A4_THREADS 256
#define A4_SMEM_BYTES 89600
#define A4_CONV_THREADS 256

// exp2f 用の定数 (kernels_elem.cu の LOG2E_F と同じ値)。
#define A4_LOG2E 1.4426950408889634f

// 4 個の f32 を bf16 4 個 (8 B) にして書く。
__device__ __forceinline__ void dno_a4_store4(__nv_bfloat16 *dst, const float *v)
{
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
    unsigned int t0 = *reinterpret_cast<unsigned int *>(&a);
    unsigned int t1 = *reinterpret_cast<unsigned int *>(&b);
    *reinterpret_cast<uint2 *>(dst) = make_uint2(t0, t1);
}

// f32 の K/V の行 [r0, r0+n) を bf16 へ落とす (丸めは __floats2bfloat162_rn =
// v3 が smem へ書いていたのと同じ)。1 スレッド = 4 要素。
extern "C" __global__ void dno_kv_f32_to_bf16(
    const float *__restrict__ k,   // [t_total, kv_dim]
    const float *__restrict__ v,   // [t_total, kv_dim]
    __nv_bfloat16 *__restrict__ kb,
    __nv_bfloat16 *__restrict__ vb,
    int r0, int n, int kv_dim)
{
    const long long base = (long long)r0 * kv_dim;
    const long long total = (long long)n * kv_dim;
    const long long step = (long long)gridDim.x * blockDim.x * 4;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
         i < total;
         i += step) {
        const float4 kx = *reinterpret_cast<const float4 *>(k + base + i);
        const float4 vx = *reinterpret_cast<const float4 *>(v + base + i);
        const float kf[4] = {kx.x, kx.y, kx.z, kx.w};
        const float vf[4] = {vx.x, vx.y, vx.z, vx.w};
        dno_a4_store4(kb + base + i, kf);
        dno_a4_store4(vb + base + i, vf);
    }
}

extern "C" __global__ void dno_attention_prefill4(
    const float *__restrict__ q,          // [n, heads*dh]
    const __nv_bfloat16 *__restrict__ kc, // [t_total, kv_dim] bf16
    const __nv_bfloat16 *__restrict__ vc, // [t_total, kv_dim] bf16
    float *__restrict__ att,              // [n, heads*dh]
    int heads, int n, int pos0, int kv_dim, int t_total, int group, int dh,
    float scale)
{
    extern __shared__ __align__(32) unsigned char sh_raw[];
    __nv_bfloat16 *q_sm = reinterpret_cast<__nv_bfloat16 *>(sh_raw);
    __nv_bfloat16 *k_sm = q_sm + A4_BQ * A4_QD;
    __nv_bfloat16 *v_sm = k_sm + A4_BK * A4_QD;
    __nv_bfloat16 *p_sm = v_sm + A4_BK * A4_QD;
    float *s_sm = reinterpret_cast<float *>(p_sm + A4_BQ * A4_PD);
    float *m0_sm = s_sm + A4_BQ * A4_BK;
    float *lst_sm = m0_sm + A4_BQ;
    float *stg = lst_sm + A4_BQ;   // 8 warp x 256 f32

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // グリッドは (head, クエリブロック)。クエリブロックは重い方から。
    const int h = blockIdx.x;
    const int nb = gridDim.y;
    const int i0 = (nb - 1 - (int)blockIdx.y) * A4_BQ;
    const int qd = heads * dh;
    const int kh_off = (h / group) * dh;
    // このブロックが見る必要のある最大キー (行ごとの causal 界の最大)
    const int jend = min(t_total, pos0 + i0 + A4_BQ);

    // ---- Q の bf16 化 (1 回だけ) ------------------------------------------
    {
        const int r = tid >> 2;             // 0..63
        const int qd0 = (tid & 3) * 64;     // 0, 64, 128, 192
        const int i = i0 + r;
        __nv_bfloat16 *dst = q_sm + r * A4_QD + qd0;
        if (i < n) {
            const float *src = q + (size_t)i * heads * dh + (size_t)h * dh + qd0;
#pragma unroll
            for (int blk = 0; blk < 16; ++blk) {
                const float4 x = *reinterpret_cast<const float4 *>(src + 4 * blk);
                const float v[4] = {x.x, x.y, x.z, x.w};
                dno_a4_store4(dst + 4 * blk, v);
            }
        } else {
#pragma unroll
            for (int u = 0; u < 32; ++u) {
                *reinterpret_cast<uint2 *>(dst + 2 * u) = make_uint2(0u, 0u);
            }
        }
    }
    if (tid < A4_BQ) {
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

    for (int j0 = 0; j0 < jend; j0 += A4_BK) {
        // ---- K (bf16) と V (bf16) のタイルを読む --------------------------
        // v4: 入力が bf16 なので 8 バイトのコピー x 8 (丸めも変換もしない)。
        {
            const int jl = tid >> 3;             // 0..31 (タイル内のキー)
            const int dd = (tid & 7) * 32;       // 0, 32, ..., 224
            const int j = j0 + jl;
            __nv_bfloat16 *kd = k_sm + jl * A4_QD + dd;
            __nv_bfloat16 *vd = v_sm + jl * A4_QD + dd;
            if (j < t_total) {
                const __nv_bfloat16 *ksrc = kc + (size_t)j * kv_dim + kh_off + dd;
                const __nv_bfloat16 *vsrc = vc + (size_t)j * kv_dim + kh_off + dd;
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    *reinterpret_cast<uint2 *>(kd + 4 * blk) =
                        *reinterpret_cast<const uint2 *>(ksrc + 4 * blk);
                    *reinterpret_cast<uint2 *>(vd + 4 * blk) =
                        *reinterpret_cast<const uint2 *>(vsrc + 4 * blk);
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
                wmma::load_matrix_sync(af, q_sm + (mt * 16) * A4_QD + kk * 16, A4_QD);
                wmma::load_matrix_sync(bf, k_sm + (nt * 16) * A4_QD + kk * 16, A4_QD);
                wmma::mma_sync(sp, af, bf, sp);
            }
            wmma::store_matrix_sync(
                s_sm + (mt * 16) * A4_BK + nt * 16, sp, A4_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- マスク + scale (無効 pair は -inf) -----------------------------
        for (int idx = tid; idx < A4_BQ * A4_BK; idx += A4_THREADS) {
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
            if (tid < A4_BQ) {
                float mx = -INFINITY;
                for (int kk = 0; kk < A4_BK; ++kk) mx = fmaxf(mx, s_sm[tid * A4_BK + kk]);
                // 全滅 (行が範囲外) のときは 0 を基準にする (-inf - 0 → exp 0)。
                m0_sm[tid] = (mx > -INFINITY) ? mx : 0.0f;
            }
            __syncthreads();
        }

        // ---- p = exp2((s - m0) * log2e) を bf16 で書く ----------------------
        for (int idx = tid; idx < A4_BQ * A4_BK; idx += A4_THREADS) {
            const int r = idx >> 5;
            const float p = exp2f((s_sm[idx] - m0_sm[r]) * A4_LOG2E);
            p_sm[r * A4_PD + (idx & 31)] = __float2bfloat16_rn(p);
        }
        __syncthreads();

        // ---- l の更新 (owner = thread r) と PV (bf16 mma) を並べて回す ------
        {
            if (tid < A4_BQ) {
                float l = lst_sm[tid];
                for (int kk = 0; kk < A4_BK; ++kk) {
                    l += __bfloat162float(p_sm[tid * A4_PD + kk]);
                }
                lst_sm[tid] = l;
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
#pragma unroll
            for (int ks = 0; ks < (A4_BK >> 4); ++ks) {
                wmma::load_matrix_sync(af, p_sm + (mt * 16) * A4_PD + ks * 16, A4_PD);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int d0 = nd * 128 + t * 16;
                    wmma::load_matrix_sync(bf, v_sm + (ks * 16) * A4_QD + d0, A4_QD);
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
    for (int r = 0; r < A4_BQ; ++r) {
        const int i = i0 + r;
        if (i < n) {
            const float l = lst_sm[r];
            const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            att[(size_t)i * qd + h * dh + tid] *= inv;
        }
    }
}
