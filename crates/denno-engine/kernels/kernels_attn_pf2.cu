// denno CUDA prefill attention v2 (task 222) — スコアを BF16 の mma で、PV を
// 4 行 x 16 次元のレジスタタイルで回す版。
//
// PTX へのコンパイル (build_ptx.sh の `attn_pf2`、compute_90):
//
//   nvcc -ptx -arch=compute_90 -O3 -fmad=false kernels_attn_pf2.cu -o kernels_attn_pf2.ptx
//
// ---- なぜ別カーネルが要るか (lab t222a の台帳) --------------------------
//  14,018 トークン / 12 層の full は gpu 2,184 ms。射影 (tc2 / v1) が約 0.31 s、
//  attention 本体が約 1.87 s = 15 TFLOP/s。既存の `dno_attention_prefill`
//  (kernels_elem.cu, PF_BQ=64 / PF_BK=8) の無駄は 3 つ:
//    - スコア (QK^T) を f32 のスカラー FMA で、1 スレッド 1 クアッド。
//      64x8 で 128 クアッドを 256 スレッドに配るので半分が遊ぶ。
//    - PV は 1 スレッド 1 次元。1 FMA あたり smem を 1 回読み、しかも同じ
//      p の行を 256 スレッドが重複して読むので smem 帯域が上限になる。
//    - 1 タイル 8 キーなので同期が 1,752 回/層。
//
//  ここでは:
//    1. スコアを BF16 の wmma (16x16x16, FP32 アキュムレート) で作る。
//       Q は [64][dh] を 1 回だけ bf16 で smem に置き、K タイル [32][dh] と
//       mma する (8 warp = 64 行 4 タイル x 32 キー 2 タイル)。
//    2. PV は 1 スレッドが 4 行 x 16 次元 (= 64 出力) を持ち、1 キーあたり
//       smem 読み 20 回で 64 FMA する (従来は 64 回/64 FMA)。
//    3. 1 タイル 32 キーにして同期を 1/4 に減らす。
//
// ---- 数値 (attention の緩和枠。P17g で CPU とのビット一致は対象外) --------
//  スコアは q と k を BF16 に丸めて mma する (丸めだけの差)。softmax は
//  exp2f のまま、PV は f32 の fmaf で kk 昇順 (従来カーネルと同じ順序)。
//  決定論は保つ (固定タイル・固定順・原子操作なし)。
//
// ---- 前提 ----------------------------------------------------------------
//   - q は [n][heads*dh] の f32 (per-head RMSNorm + rope 済み)
//   - kc / vc は [t_total][kv_dim] の f32 (kv_dim = (heads/group)*dh)
//   - att は [n][heads*dh] の f32
//   - dh == 256 固定 (スレッド配置が 16 次元群 x 16 行群のため)。ホストが
//     dh == 256 と nt >= 64 を検査する。
//   - 1 ブロック = 64 クエリ行 x 1 ヘッド、256 スレッド。
//
// 共有メモリ (動的、ホストが ATTN_PF2_SMEM_BYTES を渡す):
//   q_sm 64x264 bf16 = 33,792 B / k_sm 32x264 bf16 = 16,896 B
//   v_sm 32x260 f32  = 33,280 B / s_sm 64x32 f32   = 8,192 B
//   corr 64 f32 + mst 64 f32 + lst 64 f32 = 768 B  合計 92,928 B

#include <mma.h>
#include <cuda_bf16.h>

// タイル (値の行にコメントを付けないこと。kernel_invariants が読む)。
#define A2_BQ 64
#define A2_BK 32
#define A2_QD 264
#define A2_VD 260
#define A2_THREADS 256
#define A2_SMEM_BYTES 92928

// exp2f 用の定数 (kernels_elem.cu の LOG2E_F と同じ値)。
#define A2_LOG2E 1.4426950408889634f

// 4 個の f32 を bf16 4 個 (8 B) にして書く。
__device__ __forceinline__ void dno_a2_store4(__nv_bfloat16* dst, const float* v)
{
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
    unsigned int t0 = *reinterpret_cast<unsigned int*>(&a);
    unsigned int t1 = *reinterpret_cast<unsigned int*>(&b);
    *reinterpret_cast<uint2*>(dst) = make_uint2(t0, t1);
}

extern "C" __global__ void dno_attention_prefill2(
    const float *__restrict__ q,     // [n, heads*dh]
    const float *__restrict__ kc,    // [t_total, kv_dim]
    const float *__restrict__ vc,    // [t_total, kv_dim]
    float *__restrict__ att,         // [n, heads*dh]
    int heads, int n, int pos0, int kv_dim, int t_total, int group, int dh,
    float scale)
{
    extern __shared__ __align__(32) unsigned char sh_raw[];
    __nv_bfloat16 *q_sm = reinterpret_cast<__nv_bfloat16 *>(sh_raw);
    __nv_bfloat16 *k_sm = q_sm + A2_BQ * A2_QD;
    float *v_sm = reinterpret_cast<float *>(k_sm + A2_BK * A2_QD);
    float *s_sm = v_sm + A2_BK * A2_VD;
    float *corr_sm = s_sm + A2_BQ * A2_BK;
    float *mst_sm = corr_sm + A2_BQ;
    float *lst_sm = mst_sm + A2_BQ;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int h = blockIdx.y;
    const int i0 = blockIdx.x * A2_BQ;
    const int kh_off = (h / group) * dh;
    // このブロックが見る必要のある最大キー (行ごとの causal 界の最大)
    const int jend = min(t_total, pos0 + i0 + A2_BQ);

    // 出力の担当: スレッド (a, b) が行 4a..4a+3 と次元 16b..16b+15 を持つ。
    const int a = tid >> 4;   // 0..15 (行群)
    const int b = tid & 15;   // 0..15 (次元群)
    const int d0 = b * 16;

    // ---- 初期化 (Q の bf16 化と状態) --------------------------------------
    {
        const int r = tid >> 2;             // 0..63
        const int qd0 = (tid & 3) * 64;     // 0, 64, 128, 192
        const int i = i0 + r;
        __nv_bfloat16 *dst = q_sm + r * A2_QD + qd0;
        if (i < n) {
            const float *src = q + (size_t)i * heads * dh + (size_t)h * dh + qd0;
#pragma unroll
            for (int blk = 0; blk < 16; ++blk) {
                const float4 x = *reinterpret_cast<const float4 *>(src + 4 * blk);
                const float v[4] = {x.x, x.y, x.z, x.w};
                dno_a2_store4(dst + 4 * blk, v);
            }
        } else {
#pragma unroll
            for (int u = 0; u < 32; ++u) {
                *reinterpret_cast<uint2 *>(dst + 2 * u) = make_uint2(0u, 0u);
            }
        }
    }
    if (tid < A2_BQ) {
        mst_sm[tid] = -INFINITY;
        lst_sm[tid] = 0.0f;
    }
    float acc[4][16];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[r][d] = 0.0f;
    __syncthreads();

    using namespace nvcuda;

    for (int j0 = 0; j0 < jend; j0 += A2_BK) {
        // ---- K (bf16) と V (f32) のタイルを読む ----------------------------
        {
            const int jl = tid >> 3;             // 0..31 (タイル内のキー)
            const int dd = (tid & 7) * 32;       // 0, 32, ..., 224
            const int j = j0 + jl;
            __nv_bfloat16 *kd = k_sm + jl * A2_QD + dd;
            float *vd = v_sm + jl * A2_VD + dd;
            if (j < t_total) {
                const float *ksrc = kc + (size_t)j * kv_dim + kh_off + dd;
                const float *vsrc = vc + (size_t)j * kv_dim + kh_off + dd;
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    const float4 kv = *reinterpret_cast<const float4 *>(ksrc + 4 * blk);
                    const float kx[4] = {kv.x, kv.y, kv.z, kv.w};
                    dno_a2_store4(kd + 4 * blk, kx);
                    *reinterpret_cast<float4 *>(vd + 4 * blk) =
                        *reinterpret_cast<const float4 *>(vsrc + 4 * blk);
                }
            } else {
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    *reinterpret_cast<uint2 *>(kd + 4 * blk) = make_uint2(0u, 0u);
                    *reinterpret_cast<float4 *>(vd + 4 * blk) =
                        make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        // ---- スコア: S = Q @ K^T (bf16 mma, FP32 アキュムレート) ------------
        {
            const int mt = warp >> 1;   // 0..3 (16 行タイル)
            const int nt = warp & 1;    // 0..1  (16 キータイル)
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accs;
            wmma::fill_fragment(accs, 0.0f);
            for (int kk = 0; kk < (dh >> 4); ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, q_sm + (mt * 16) * A2_QD + kk * 16, A2_QD);
                wmma::load_matrix_sync(bf, k_sm + (nt * 16) * A2_QD + kk * 16, A2_QD);
                wmma::mma_sync(accs, af, bf, accs);
            }
            wmma::store_matrix_sync(
                s_sm + (mt * 16) * A2_BK + nt * 16, accs, A2_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- scale と causal マスク (従来カーネルと同じ規則) ----------------
        for (int idx = tid; idx < A2_BQ * A2_BK; idx += A2_THREADS) {
            const int r = idx >> 5;
            const int key = idx & 31;
            const int i = i0 + r;
            const int j = j0 + key;
            float s;
            if (i >= n) {
                s = 0.0f;  // 無効行: exp が 1 になり l > 0 (書き出さない)
            } else if (j < t_total && j <= pos0 + i) {
                s = s_sm[idx] * scale;
            } else {
                s = -INFINITY;
            }
            s_sm[idx] = s;
        }
        __syncthreads();

        // ---- タイル内 row max と online 補正 (owner = thread r) -------------
        if (tid < A2_BQ) {
            float mx = mst_sm[tid];
            for (int kk = 0; kk < A2_BK; ++kk) mx = fmaxf(mx, s_sm[tid * A2_BK + kk]);
            // mst=-inf → exp2f(-inf) = 0
            corr_sm[tid] = exp2f((mst_sm[tid] - mx) * A2_LOG2E);
            mst_sm[tid] = mx;
        }
        __syncthreads();
        // p = exp2((s - m_new) * log2e) を s_sm に上書き (無効 pair は 0)
        for (int pid = tid; pid < A2_BQ * A2_BK; pid += A2_THREADS) {
            const int r = pid >> 5;
            s_sm[pid] = exp2f((s_sm[pid] - mst_sm[r]) * A2_LOG2E);
        }
        __syncthreads();
        // ---- l の更新 (owner = thread r)。acc の再スケールより先に済ませる --
        if (tid < A2_BQ) {
            float l = lst_sm[tid] * corr_sm[tid];
            for (int kk = 0; kk < A2_BK; ++kk) l += s_sm[tid * A2_BK + kk];
            lst_sm[tid] = l;
        }

        // ---- PV: p @ V を 4 行 x 16 次元で累積 ------------------------------
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float c = corr_sm[a * 4 + r];
#pragma unroll
            for (int d = 0; d < 16; ++d) acc[r][d] *= c;
        }
        for (int kk = 0; kk < A2_BK; ++kk) {
            float vv[16];
            const float *vp = v_sm + kk * A2_VD + d0;
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                const float4 x = *reinterpret_cast<const float4 *>(vp + 4 * q4);
                vv[4 * q4] = x.x;
                vv[4 * q4 + 1] = x.y;
                vv[4 * q4 + 2] = x.z;
                vv[4 * q4 + 3] = x.w;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float p = s_sm[(a * 4 + r) * A2_BK + kk];
#pragma unroll
                for (int d = 0; d < 16; ++d) acc[r][d] = fmaf(p, vv[d], acc[r][d]);
            }
        }
        __syncthreads();
    }

    // ---- 正規化 + 書き出し -------------------------------------------------
    // 有効行は必ず j=0 が causal 内なので l >= 1 > 0。
    {
        const int r0 = a * 4;
        float inv[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = i0 + r0 + r;
            inv[r] = (i < n) ? 1.0f / lst_sm[r0 + r] : 0.0f;
            if (i < n) {
                float *dst = att + (size_t)i * heads * dh + (size_t)h * dh + d0;
#pragma unroll
                for (int d = 0; d < 16; ++d) dst[d] = acc[r][d] * inv[r];
            }
        }
    }
}
