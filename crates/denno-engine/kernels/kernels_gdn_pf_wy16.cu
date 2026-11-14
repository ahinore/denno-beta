// task 311: prefill の GDN スキャン (WY 形式) の v 列 4 分割 + チャンク長 16 版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy16`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy16.cu \
//        -o kernels_gdn_pf_wy16.ptx
//
// なぜ別カーネルか: t311b で v 列 4 分割 (`dno_gdn_pf_wy4`, C = 8, 192 ブロック)
// が scan_pf を 3,826 で 359 -> 257 ms、14k で 1316 -> 938 ms にした
// (-28.6%)。8 分割 (wy8) は 413 ms で悪化した (A/G の重複が増えるため)。
// 残る時間はチャンクごとの 7 バリアなので、C を 8 -> 16 にしてバリアを
// 半分にする版を測る (A/G の仕事は +8% 増える)。
//
// 1 ブロック = (1 ヘッド, v 列 32 本)。q/k の conv と L2 正規化、A/G は
// 4 ブロックが同じ値を計算する (wy4 と同じ。値も同じ)。
//
// 数値: wy4 / wy32 と **要素ごとに同じ順序** (C を変えると Rdec の積の順は
// 同じで、チャンクの切れ目だけが変わる)。よって C = 8 とは丸め順が変わりうる。
//
// 前提: dk = dv = 128 / d_conv = 4 / dv == WY16_DV * 4。
//   形が違う / PTX が無いときは wy4 → wy32 → v3 へ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY16_DK 128
#define WY16_DV 32
#define WY16_C 16
#define WY16_THREADS 256
#define WY16_WARPS 8
#define WY16_DCONV 4
#define WY16_CS 3

// 動的共有メモリの並び (float 単位)
#define WY16_S 0
#define WY16_KC 4096
#define WY16_QC 6144
#define WY16_VC 8192
#define WY16_DC 8704
#define WY16_PC 9216
#define WY16_QPC 9728
#define WY16_RDEC 10240
#define WY16_AC 10496
#define WY16_GC 10752
#define WY16_LAM 11008
#define WY16_RL 11024
#define WY16_EG 11040
#define WY16_BC 11056
#define WY16_LL 11072
#define WY16_TOTAL 11073

extern "C" __global__ void __launch_bounds__(WY16_THREADS, 1)
dno_gdn_pf_wy16(
    const float* __restrict__ qkv,        // [n, conv_dim]
    const float* __restrict__ beta,       // [n, nv]
    const float* __restrict__ alpha,      // [n, nv]
    const float* __restrict__ conv_st_in, // [conv_dim, cs]
    const float* __restrict__ ssm_in,     // [nv, dk, dv]
    const float* __restrict__ wconv,      // [conv_dim, d_conv]
    const float* __restrict__ wa,         // [nv] (負値)
    const float* __restrict__ wdt,        // [nv]
    float* __restrict__ o_out,            // [n, nv*dv] (RMS 前の o)
    float* __restrict__ conv_st_out,      // [conv_dim, cs]
    float* __restrict__ ssm_out,          // [nv, dk, dv]
    int n, int nk, int nv, int conv_dim, float eps, float q_scale)
{
    extern __shared__ float sh[];
    float* S = sh + WY16_S;
    float* Kc = sh + WY16_KC;
    float* Qc = sh + WY16_QC;
    float* Vc = sh + WY16_VC;
    float* Dc = sh + WY16_DC;
    float* Pc = sh + WY16_PC;
    float* QPc = sh + WY16_QPC;
    float* Rdec = sh + WY16_RDEC;
    float* Ac = sh + WY16_AC;
    float* Gc = sh + WY16_GC;
    float* lam = sh + WY16_LAM;
    float* rl = sh + WY16_RL;
    float* egc = sh + WY16_EG;
    float* bc = sh + WY16_BC;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/4 (0..3)
    const int voff = vb * WY16_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY16_DK;
    const int vd = nv * WY16_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  32         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY16_DK);
    const bool is_k = (tid >= WY16_DK && tid < 2 * WY16_DK);
    const bool is_v = (tid < WY16_DV);
    const int chq = kh * WY16_DK + tid;
    const int chk = kd_total + kh * WY16_DK + (tid - WY16_DK);
    const int chv = 2 * kd_total + h * WY16_DK + voff + tid;

    float w4q[WY16_DCONV];
    float wndq[WY16_CS];
    float w4k[WY16_DCONV];
    float wndk[WY16_CS];
    float w4v[WY16_DCONV];
    float wndv[WY16_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY16_DCONV; ++j) w4q[j] = wconv[chq * WY16_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) wndq[j] = conv_st_in[chq * WY16_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY16_DCONV; ++j) w4k[j] = wconv[chk * WY16_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) wndk[j] = conv_st_in[chk * WY16_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY16_DCONV; ++j) w4v[j] = wconv[chv * WY16_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) wndv[j] = conv_st_in[chv * WY16_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY16_DK * WY16_DV; idx += WY16_THREADS) {
        const int r = idx / WY16_DV;
        const int c = idx - r * WY16_DV;
        S[idx] = ssm_in[(long long)h * WY16_DK * WY16_DK + (long long)r * WY16_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY16_C) {
        const int len = (n - t0 < WY16_C) ? (n - t0) : WY16_C;

        // ---- (1) conv + silu (このチャンクの全トークン) と gate -----------
        if (is_q) {
            const int r = tid;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float x = qkv[row + chq];
                const float acc =
                    w4q[0] * wndq[0] + w4q[1] * wndq[1] + w4q[2] * wndq[2] + w4q[3] * x;
                wndq[0] = wndq[1];
                wndq[1] = wndq[2];
                wndq[2] = x;
                Qc[tt * WY16_DK + r] = acc / (1.0f + expf(-acc));
            }
        }
        if (is_v) {
            const int r = tid;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float xv = qkv[row + chv];
                const float accv =
                    w4v[0] * wndv[0] + w4v[1] * wndv[1] + w4v[2] * wndv[2] + w4v[3] * xv;
                wndv[0] = wndv[1];
                wndv[1] = wndv[2];
                wndv[2] = xv;
                Vc[tt * WY16_DV + r] = accv / (1.0f + expf(-accv));
            }
        }
        if (is_k) {
            const int r = tid - WY16_DK;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float x = qkv[row + chk];
                const float acc =
                    w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                wndk[0] = wndk[1];
                wndk[1] = wndk[2];
                wndk[2] = x;
                Kc[tt * WY16_DK + r] = acc / (1.0f + expf(-acc));
            }
        }
        if (tid < len) {
            const long long row = (long long)(t0 + tid) * nv + h;
            const float a = alpha[row] + wdt[h];
            const float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            egc[tid] = expf(wa[h] * sp);
            bc[tid] = 1.0f / (1.0f + expf(-beta[row]));
        }
        __syncthreads();

        // ---- (2) q/k の L2 正規化 (1 ワープ = 1 トークン) ------------------
        for (int tt = warp; tt < len; tt += WY16_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY16_DK; r += 32) {
                const float qv = Qc[tt * WY16_DK + r];
                const float kv = Kc[tt * WY16_DK + r];
                sq += qv * qv;
                sk2 += kv * kv;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sq += __shfl_xor_sync(0xffffffffu, sq, off);
                sk2 += __shfl_xor_sync(0xffffffffu, sk2, off);
            }
            const float q_inv = (1.0f / sqrtf(sq + eps)) * q_scale;
            const float k_inv = 1.0f / sqrtf(sk2 + eps);
            for (int r = lane; r < WY16_DK; r += 32) {
                Qc[tt * WY16_DK + r] *= q_inv;
                Kc[tt * WY16_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY16_C + j] = r;
                r *= egc[j];
            }
            lam[tid] = r;
        }
        if (tid == 0) {
            float r = 1.0f;
            for (int t = len - 1; t >= 0; --t) {
                rl[t] = r;
                r *= egc[t];
            }
            sh[WY16_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        for (int idx = tid; idx < len * WY16_C; idx += WY16_THREADS) {
            const int i = idx / WY16_C;
            const int j = idx - i * WY16_C;
            if (j > i) continue;
            float kg = 0.0f;
            float qg = 0.0f;
            for (int r = 0; r < WY16_DK; ++r) {
                const float kv = Kc[j * WY16_DK + r];
                kg += Kc[i * WY16_DK + r] * kv;
                qg += Qc[i * WY16_DK + r] * kv;
            }
            if (j < i) Ac[i * WY16_C + j] = kg * Rdec[i * WY16_C + j];
            Gc[i * WY16_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY16_C + j]);
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        for (int idx = tid; idx < len * WY16_DV; idx += WY16_THREADS) {
            const int i = idx / WY16_DV;
            const int c = idx - i * WY16_DV;
            const float* ki = Kc + i * WY16_DK;
            const float* qi = Qc + i * WY16_DK;
            float sk = 0.0f;
            float sq = 0.0f;
            for (int r = 0; r < WY16_DK; ++r) {
                const float sv = S[r * WY16_DV + c];
                sk += ki[r] * sv;
                sq += qi[r] * sv;
            }
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY16_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY16_DV + c] - Pc[i * WY16_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY16_C + j] * Dc[j * WY16_DV + c];
                Dc[i * WY16_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        if (tid < WY16_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = QPc[i * WY16_DV + c];
                for (int j = 0; j <= i; ++j) acc += Gc[i * WY16_C + j] * Dc[j * WY16_DV + c];
                o_out[(long long)(t0 + i) * vd + h * WY16_DK + voff + c] = acc;
            }
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY16_LL];
            for (int idx = tid; idx < WY16_DK * WY16_DV; idx += WY16_THREADS) {
                const int r = idx / WY16_DV;
                const int c = idx - r * WY16_DV;
                float acc = S[idx] * lam_last;
                for (int t = 0; t < len; ++t) {
                    acc += rl[t] * Kc[t * WY16_DK + r] * Dc[t * WY16_DV + c];
                }
                S[idx] = acc;
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY16_DK * WY16_DV; idx += WY16_THREADS) {
        const int r = idx / WY16_DV;
        const int c = idx - r * WY16_DV;
        ssm_out[(long long)h * WY16_DK * WY16_DK + (long long)r * WY16_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) conv_st_out[chq * WY16_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) conv_st_out[chk * WY16_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY16_CS; ++j) conv_st_out[chv * WY16_CS + j] = wndv[j];
    }
}
