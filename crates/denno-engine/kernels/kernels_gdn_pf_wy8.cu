// task 311: prefill の GDN スキャン (WY 形式) を **v 列で 8 分割**した版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy8`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy8.cu \
//        -o kernels_gdn_pf_wy8.ptx
//
// なぜ別カーネルか (lab t231a / t311a):
//   `dno_gdn_pf_wy32` は 1 ブロック = 1 ヘッド (状態 dk x dv = 128 x 128 を
//   共有メモリへ) なので **48 ブロックしか立たない** (170 SM)。C = 8 では
//   scan_pf が 392.8 ms で v3 (192 ブロック) の 357.0 ms より 11% 遅かった。
//   t311a で v 列 4 分割 (`dno_gdn_pf_wy4`, 192 ブロック, 32 列) が
//   3,826 で 400.4 -> 256.5 ms / 14k で 1465.7 -> 938.7 ms と効いたので、
//   さらに 8 分割 (384 ブロック, 16 列) を測るための版。
//
//   1 ブロック = (1 ヘッド, v 列の 1/8 = 16 列)。q/k の conv と L2 正規化、
//   A/G (k_i·k_j, q_i·k_j) は 8 ブロックが同じ値を計算する (wy4 の 4 ブロック
//   と同じ重複。値は同じ)。
//
// 数値: `dno_gdn_pf_wy32` / `dno_gdn_pf_wy4` と **要素ごとに同じ順序**で書く
//   (状態の添字だけ列数 + voff に変える)。同一入力なら wy32 / wy4 とビット
//   一致する (lab t311a で wy4 と wy32 の md5 が一致したことを確認済み)。
//
// 前提 (ホスト側 `gdn_pf_wy8_ok` / `gdn_pf_wy4_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == WY8_DK (128) / d_conv == 4 (cs == 3)
//   - dv == 128 (WY8_DV * 8)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
//   形が違う / PTX が無いときは wy4 → wy32 → v3 → v2 → v1 → CPU へ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY8_DK 128
#define WY8_DV 16
#define WY8_C 8
#define WY8_THREADS 256
#define WY8_WARPS 8
#define WY8_DCONV 4
#define WY8_CS 3

// 動的共有メモリの並び (float 単位)
#define WY8_S 0
#define WY8_KC 2048
#define WY8_QC 3072
#define WY8_VC 4096
#define WY8_DC 4224
#define WY8_PC 4352
#define WY8_QPC 4480
#define WY8_RDEC 4608
#define WY8_AC 4672
#define WY8_GC 4736
#define WY8_LAM 4800
#define WY8_RL 4808
#define WY8_EG 4816
#define WY8_BC 4824
#define WY8_LL 4832
#define WY8_TOTAL 4833

extern "C" __global__ void __launch_bounds__(WY8_THREADS, 1)
dno_gdn_pf_wy8(
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
    float* S = sh + WY8_S;
    float* Kc = sh + WY8_KC;
    float* Qc = sh + WY8_QC;
    float* Vc = sh + WY8_VC;
    float* Dc = sh + WY8_DC;
    float* Pc = sh + WY8_PC;
    float* QPc = sh + WY8_QPC;
    float* Rdec = sh + WY8_RDEC;
    float* Ac = sh + WY8_AC;
    float* Gc = sh + WY8_GC;
    float* lam = sh + WY8_LAM;
    float* rl = sh + WY8_RL;
    float* egc = sh + WY8_EG;
    float* bc = sh + WY8_BC;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/8 (0..7)
    const int voff = vb * WY8_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY8_DK;
    const int vd = nv * WY8_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  16         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY8_DK);
    const bool is_k = (tid >= WY8_DK && tid < 2 * WY8_DK);
    const bool is_v = (tid < WY8_DV);
    const int chq = kh * WY8_DK + tid;
    const int chk = kd_total + kh * WY8_DK + (tid - WY8_DK);
    const int chv = 2 * kd_total + h * WY8_DK + voff + tid;

    float w4q[WY8_DCONV];
    float wndq[WY8_CS];
    float w4k[WY8_DCONV];
    float wndk[WY8_CS];
    float w4v[WY8_DCONV];
    float wndv[WY8_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY8_DCONV; ++j) w4q[j] = wconv[chq * WY8_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) wndq[j] = conv_st_in[chq * WY8_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY8_DCONV; ++j) w4k[j] = wconv[chk * WY8_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) wndk[j] = conv_st_in[chk * WY8_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY8_DCONV; ++j) w4v[j] = wconv[chv * WY8_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) wndv[j] = conv_st_in[chv * WY8_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY8_DK * WY8_DV; idx += WY8_THREADS) {
        const int r = idx / WY8_DV;
        const int c = idx - r * WY8_DV;
        S[idx] = ssm_in[(long long)h * WY8_DK * WY8_DK + (long long)r * WY8_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY8_C) {
        const int len = (n - t0 < WY8_C) ? (n - t0) : WY8_C;

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
                Qc[tt * WY8_DK + r] = acc / (1.0f + expf(-acc));
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
                Vc[tt * WY8_DV + r] = accv / (1.0f + expf(-accv));
            }
        }
        if (is_k) {
            const int r = tid - WY8_DK;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float x = qkv[row + chk];
                const float acc =
                    w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                wndk[0] = wndk[1];
                wndk[1] = wndk[2];
                wndk[2] = x;
                Kc[tt * WY8_DK + r] = acc / (1.0f + expf(-acc));
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
        for (int tt = warp; tt < len; tt += WY8_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY8_DK; r += 32) {
                const float qv = Qc[tt * WY8_DK + r];
                const float kv = Kc[tt * WY8_DK + r];
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
            for (int r = lane; r < WY8_DK; r += 32) {
                Qc[tt * WY8_DK + r] *= q_inv;
                Kc[tt * WY8_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY8_C + j] = r;
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
            sh[WY8_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        for (int idx = tid; idx < len * WY8_C; idx += WY8_THREADS) {
            const int i = idx / WY8_C;
            const int j = idx - i * WY8_C;
            if (j > i) continue;
            float kg = 0.0f;
            float qg = 0.0f;
            for (int r = 0; r < WY8_DK; ++r) {
                const float kv = Kc[j * WY8_DK + r];
                kg += Kc[i * WY8_DK + r] * kv;
                qg += Qc[i * WY8_DK + r] * kv;
            }
            if (j < i) Ac[i * WY8_C + j] = kg * Rdec[i * WY8_C + j];
            Gc[i * WY8_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY8_C + j]);
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        for (int idx = tid; idx < len * WY8_DV; idx += WY8_THREADS) {
            const int i = idx / WY8_DV;
            const int c = idx - i * WY8_DV;
            const float* ki = Kc + i * WY8_DK;
            const float* qi = Qc + i * WY8_DK;
            float sk = 0.0f;
            float sq = 0.0f;
            for (int r = 0; r < WY8_DK; ++r) {
                const float sv = S[r * WY8_DV + c];
                sk += ki[r] * sv;
                sq += qi[r] * sv;
            }
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY8_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY8_DV + c] - Pc[i * WY8_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY8_C + j] * Dc[j * WY8_DV + c];
                Dc[i * WY8_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        if (tid < WY8_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = QPc[i * WY8_DV + c];
                for (int j = 0; j <= i; ++j) acc += Gc[i * WY8_C + j] * Dc[j * WY8_DV + c];
                o_out[(long long)(t0 + i) * vd + h * WY8_DK + voff + c] = acc;
            }
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY8_LL];
            for (int idx = tid; idx < WY8_DK * WY8_DV; idx += WY8_THREADS) {
                const int r = idx / WY8_DV;
                const int c = idx - r * WY8_DV;
                float acc = S[idx] * lam_last;
                for (int t = 0; t < len; ++t) {
                    acc += rl[t] * Kc[t * WY8_DK + r] * Dc[t * WY8_DV + c];
                }
                S[idx] = acc;
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY8_DK * WY8_DV; idx += WY8_THREADS) {
        const int r = idx / WY8_DV;
        const int c = idx - r * WY8_DV;
        ssm_out[(long long)h * WY8_DK * WY8_DK + (long long)r * WY8_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) conv_st_out[chq * WY8_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) conv_st_out[chk * WY8_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY8_CS; ++j) conv_st_out[chv * WY8_CS + j] = wndv[j];
    }
}
