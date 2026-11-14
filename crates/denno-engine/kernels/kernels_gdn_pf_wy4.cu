// task 311: prefill の GDN スキャン (WY 形式) を **v 列で 4 分割**した版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy4`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy4.cu \
//        -o kernels_gdn_pf_wy4.ptx
//
// なぜ別カーネルか (lab t231a / REPORT_P231 / REPORT_P296):
//   `dno_gdn_pf_wy32` は 1 ブロック = 1 ヘッド (状態 dk x dv = 128 x 128 を
//   共有メモリへ) なので **48 ブロックしか立たない** (170 SM)。C = 8 では
//   scan_pf が 392.8 ms で v3 (192 ブロック) の 357.0 ms より 11% 遅かった。
//   遅さは SM を使い切れていないこと (48/170 = 28%) で、WY の演算そのものでは
//   ない — チャンク内の仕事は列ごとに独立なので、**v 列を 4 分割**すれば
//   192 ブロックになり、1 ブロックあたりの状態仕事が 1/4 になる。
//
//   このカーネルは 1 ブロック = (1 ヘッド, v 列の 1/4 = 32 列)。q/k の conv と
//   L2 正規化、A/G (k_i·k_j, q_i·k_j) は 4 ブロックが同じ値を計算する
//   (v3 の conv と同じ重複。値は同じ)。
//
// 数値: `dno_gdn_pf_wy32` と **要素ごとに同じ順序**で書く (状態の添字だけ
//   128 列 → 32 列 + voff に変える)。よって同一入力なら wy32 とビット一致する
//   はず (lab で md5 が一致するかを見る)。v3 (トークン逐次) とは丸め順が違う。
//
// 前提 (ホスト側 `gdn_pf_wy4_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == WY4_DK (128) / d_conv == 4 (cs == 3)
//   - dv == 128 (WY4_DV * 4)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
//   形が違う / PTX が無いときは wy32 → v3 → v2 → v1 → CPU へ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY4_DK 128
#define WY4_DV 32
#define WY4_C 8
#define WY4_THREADS 256
#define WY4_WARPS 8
#define WY4_DCONV 4
#define WY4_CS 3

// 動的共有メモリの並び (float 単位)
#define WY4_S 0
#define WY4_KC 4096
#define WY4_QC 5120
#define WY4_VC 6144
#define WY4_DC 6400
#define WY4_PC 6656
#define WY4_QPC 6912
#define WY4_RDEC 7168
#define WY4_AC 7232
#define WY4_GC 7296
#define WY4_LAM 7360
#define WY4_RL 7368
#define WY4_EG 7376
#define WY4_BC 7384
#define WY4_LL 7392
#define WY4_TOTAL 7393

extern "C" __global__ void __launch_bounds__(WY4_THREADS, 1)
dno_gdn_pf_wy4(
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
    float* S = sh + WY4_S;
    float* Kc = sh + WY4_KC;
    float* Qc = sh + WY4_QC;
    float* Vc = sh + WY4_VC;
    float* Dc = sh + WY4_DC;
    float* Pc = sh + WY4_PC;
    float* QPc = sh + WY4_QPC;
    float* Rdec = sh + WY4_RDEC;
    float* Ac = sh + WY4_AC;
    float* Gc = sh + WY4_GC;
    float* lam = sh + WY4_LAM;
    float* rl = sh + WY4_RL;
    float* egc = sh + WY4_EG;
    float* bc = sh + WY4_BC;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/4 (0..3)
    const int voff = vb * WY4_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY4_DK;
    const int vd = nv * WY4_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  32         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY4_DK);
    const bool is_k = (tid >= WY4_DK && tid < 2 * WY4_DK);
    const bool is_v = (tid < WY4_DV);
    const int chq = kh * WY4_DK + tid;
    const int chk = kd_total + kh * WY4_DK + (tid - WY4_DK);
    const int chv = 2 * kd_total + h * WY4_DK + voff + tid;

    float w4q[WY4_DCONV];
    float wndq[WY4_CS];
    float w4k[WY4_DCONV];
    float wndk[WY4_CS];
    float w4v[WY4_DCONV];
    float wndv[WY4_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY4_DCONV; ++j) w4q[j] = wconv[chq * WY4_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) wndq[j] = conv_st_in[chq * WY4_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY4_DCONV; ++j) w4k[j] = wconv[chk * WY4_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) wndk[j] = conv_st_in[chk * WY4_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY4_DCONV; ++j) w4v[j] = wconv[chv * WY4_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) wndv[j] = conv_st_in[chv * WY4_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY4_DK * WY4_DV; idx += WY4_THREADS) {
        const int r = idx / WY4_DV;
        const int c = idx - r * WY4_DV;
        S[idx] = ssm_in[(long long)h * WY4_DK * WY4_DK + (long long)r * WY4_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY4_C) {
        const int len = (n - t0 < WY4_C) ? (n - t0) : WY4_C;

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
                Qc[tt * WY4_DK + r] = acc / (1.0f + expf(-acc));
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
                Vc[tt * WY4_DV + r] = accv / (1.0f + expf(-accv));
            }
        }
        if (is_k) {
            const int r = tid - WY4_DK;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float x = qkv[row + chk];
                const float acc =
                    w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                wndk[0] = wndk[1];
                wndk[1] = wndk[2];
                wndk[2] = x;
                Kc[tt * WY4_DK + r] = acc / (1.0f + expf(-acc));
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
        for (int tt = warp; tt < len; tt += WY4_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY4_DK; r += 32) {
                const float qv = Qc[tt * WY4_DK + r];
                const float kv = Kc[tt * WY4_DK + r];
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
            for (int r = lane; r < WY4_DK; r += 32) {
                Qc[tt * WY4_DK + r] *= q_inv;
                Kc[tt * WY4_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY4_C + j] = r;
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
            sh[WY4_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        for (int idx = tid; idx < len * WY4_C; idx += WY4_THREADS) {
            const int i = idx / WY4_C;
            const int j = idx - i * WY4_C;
            if (j > i) continue;
            float kg = 0.0f;
            float qg = 0.0f;
            for (int r = 0; r < WY4_DK; ++r) {
                const float kv = Kc[j * WY4_DK + r];
                kg += Kc[i * WY4_DK + r] * kv;
                qg += Qc[i * WY4_DK + r] * kv;
            }
            if (j < i) Ac[i * WY4_C + j] = kg * Rdec[i * WY4_C + j];
            Gc[i * WY4_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY4_C + j]);
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        for (int idx = tid; idx < len * WY4_DV; idx += WY4_THREADS) {
            const int i = idx / WY4_DV;
            const int c = idx - i * WY4_DV;
            const float* ki = Kc + i * WY4_DK;
            const float* qi = Qc + i * WY4_DK;
            float sk = 0.0f;
            float sq = 0.0f;
            for (int r = 0; r < WY4_DK; ++r) {
                const float sv = S[r * WY4_DV + c];
                sk += ki[r] * sv;
                sq += qi[r] * sv;
            }
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY4_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY4_DV + c] - Pc[i * WY4_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY4_C + j] * Dc[j * WY4_DV + c];
                Dc[i * WY4_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        if (tid < WY4_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = QPc[i * WY4_DV + c];
                for (int j = 0; j <= i; ++j) acc += Gc[i * WY4_C + j] * Dc[j * WY4_DV + c];
                o_out[(long long)(t0 + i) * vd + h * WY4_DK + voff + c] = acc;
            }
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY4_LL];
            for (int idx = tid; idx < WY4_DK * WY4_DV; idx += WY4_THREADS) {
                const int r = idx / WY4_DV;
                const int c = idx - r * WY4_DV;
                float acc = S[idx] * lam_last;
                for (int t = 0; t < len; ++t) {
                    acc += rl[t] * Kc[t * WY4_DK + r] * Dc[t * WY4_DV + c];
                }
                S[idx] = acc;
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY4_DK * WY4_DV; idx += WY4_THREADS) {
        const int r = idx / WY4_DV;
        const int c = idx - r * WY4_DV;
        ssm_out[(long long)h * WY4_DK * WY4_DK + (long long)r * WY4_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) conv_st_out[chq * WY4_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) conv_st_out[chk * WY4_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY4_CS; ++j) conv_st_out[chv * WY4_CS + j] = wndv[j];
    }
}
