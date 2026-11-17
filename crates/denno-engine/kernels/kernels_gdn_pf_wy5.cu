// task 397: prefill の GDN スキャン (WY 形式、v 列 4 分割) の従属連鎖を短くした版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy5`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy5.cu \
//        -o kernels_gdn_pf_wy5.ptx
//
// なぜ別カーネルか (lab の実測、REPORT_P397):
//   `dno_gdn_pf_wy4` は 192 ブロックで SM を埋めたが、演算は 1.5〜3 TFLOPS 級
//   (14k の scan 936 ms) で、ピークの 2〜3% しか出ていない。ボトルネックは
//   共有メモリを経由する**長い従属連鎖**で、次の 3 つを直した:
//
//   (1) A/G (旧: 1 スレッド = 1 組で 128 要素の直列ドット積 x 2 本)
//       → 1 ワープ = 1 組。レーンは 4 要素ずつ積んでから `__shfl_down_sync` で
//         5 段の木にする (従属連鎖 128 -> 4+5)。和の木が変わるので**丸めは
//         wy4 と違う** (値の差は 1e-7 級、md5 は変わる)。
//   (2) P/QP (旧: 2 アキュムレータで 128 段)
//       → 4 アキュムレータ (r を 4 本ずつ) で従属連鎖を 32 段にし、最後に
//         1 回だけ足す。ここも**丸めが wy4 と違う**。
//   (3) conv の入力 (旧: 1 トークンずつ global を読んでから 4 タップの直列 FMA)
//       → チャンクの 8 トークンぶんを先に読む (コンパイル時に固定した 8 回の
//         独立ロード)。conv 窓の更新は `tt < len` の内側だけ (窓の前進量は wy4 と
//         同じ)。値は wy4 と同一の演算列 (-fmad=false)。
//
// 数値: (1) と (2) は同じ式の別の括り方なので、**要素ごとに数 ulp 変わりうる**。
//   ゲート `DENNO_PF_SCAN_WY5` (**lab t397b で採ったので既定 on**) で wy4 へ戻せる。
//
// 実測 (lab t397b、既定モデル、depth8k、PROFILE=1、`[q4ledger-pf]` の ssm busy):
//   3,826 cold 433.3/433.1 (b1/b2) -> 370.9 = -14.4%
//   14k cold   1,554.5/1,557.9      -> 1,330.9 = -14.5%
//   decode 15.5-15.9 ms/tok で不変、errors 0。md5 は turn2 の 2 本だけ DIFF。
//   初回 (t397a、上の 3 点のうち (1)(2) だけ) は -5.7% だったので、
//   (7) o の (i,c) 並列と (8) 状態更新の読み直し削減が残りの大半を占めた。
//
// 前提 (ホスト側 `gdn_pf_wy5_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == WY5_DK (128) / d_conv == 4 (cs == 3)
//   - dv == 128 (WY5_DV * 4)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
//   形が違う / PTX が無いときは wy4 -> v3 -> v2 -> v1 -> CPU へ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY5_DK 128
#define WY5_DV 32
#define WY5_C 8
#define WY5_THREADS 256
#define WY5_WARPS 8
#define WY5_DCONV 4
#define WY5_CS 3

// 動的共有メモリの並び (float 単位)。wy4 と同じ (作業域を増やしていない)。
#define WY5_S 0
#define WY5_KC 4096
#define WY5_QC 5120
#define WY5_VC 6144
#define WY5_DC 6400
#define WY5_PC 6656
#define WY5_QPC 6912
#define WY5_RDEC 7168
#define WY5_AC 7232
#define WY5_GC 7296
#define WY5_LAM 7360
#define WY5_RL 7368
#define WY5_EG 7376
#define WY5_BC 7384
#define WY5_LL 7392
#define WY5_TOTAL 7393

extern "C" __global__ void __launch_bounds__(WY5_THREADS, 1)
dno_gdn_pf_wy5(
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
    float* S = sh + WY5_S;
    float* Kc = sh + WY5_KC;
    float* Qc = sh + WY5_QC;
    float* Vc = sh + WY5_VC;
    float* Dc = sh + WY5_DC;
    float* Pc = sh + WY5_PC;
    float* QPc = sh + WY5_QPC;
    float* Rdec = sh + WY5_RDEC;
    float* Ac = sh + WY5_AC;
    float* Gc = sh + WY5_GC;
    float* lam = sh + WY5_LAM;
    float* rl = sh + WY5_RL;
    float* egc = sh + WY5_EG;
    float* bc = sh + WY5_BC;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/4 (0..3)
    const int voff = vb * WY5_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY5_DK;
    const int vd = nv * WY5_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  32         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY5_DK);
    const bool is_k = (tid >= WY5_DK && tid < 2 * WY5_DK);
    const bool is_v = (tid < WY5_DV);
    const int chq = kh * WY5_DK + tid;
    const int chk = kd_total + kh * WY5_DK + (tid - WY5_DK);
    const int chv = 2 * kd_total + h * WY5_DK + voff + tid;

    float w4q[WY5_DCONV];
    float wndq[WY5_CS];
    float w4k[WY5_DCONV];
    float wndk[WY5_CS];
    float w4v[WY5_DCONV];
    float wndv[WY5_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY5_DCONV; ++j) w4q[j] = wconv[chq * WY5_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) wndq[j] = conv_st_in[chq * WY5_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY5_DCONV; ++j) w4k[j] = wconv[chk * WY5_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) wndk[j] = conv_st_in[chk * WY5_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY5_DCONV; ++j) w4v[j] = wconv[chv * WY5_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) wndv[j] = conv_st_in[chv * WY5_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY5_DK * WY5_DV; idx += WY5_THREADS) {
        const int r = idx / WY5_DV;
        const int c = idx - r * WY5_DV;
        S[idx] = ssm_in[(long long)h * WY5_DK * WY5_DK + (long long)r * WY5_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY5_C) {
        const int len = (n - t0 < WY5_C) ? (n - t0) : WY5_C;

        // ---- (1) conv + silu (このチャンクの全トークン) と gate -----------
        // task 397: 入力 8 本を先に読む (独立なロードをまとめて出す)。
        // 窓の前進 (`wnd?` の更新) は `tt < len` の内側だけ = wy4 と同じ。
        if (is_q) {
            float xq[WY5_C];
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                xq[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chq]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                if (tt < len) {
                    const float x = xq[tt];
                    const float acc =
                        w4q[0] * wndq[0] + w4q[1] * wndq[1] + w4q[2] * wndq[2] + w4q[3] * x;
                    wndq[0] = wndq[1];
                    wndq[1] = wndq[2];
                    wndq[2] = x;
                    Qc[tt * WY5_DK + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_v) {
            float xv[WY5_C];
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                xv[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chv]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                if (tt < len) {
                    const float x = xv[tt];
                    const float acc =
                        w4v[0] * wndv[0] + w4v[1] * wndv[1] + w4v[2] * wndv[2] + w4v[3] * x;
                    wndv[0] = wndv[1];
                    wndv[1] = wndv[2];
                    wndv[2] = x;
                    Vc[tt * WY5_DV + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_k) {
            float xk[WY5_C];
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                xk[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chk]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY5_C; ++tt) {
                if (tt < len) {
                    const float x = xk[tt];
                    const float acc =
                        w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                    wndk[0] = wndk[1];
                    wndk[1] = wndk[2];
                    wndk[2] = x;
                    Kc[tt * WY5_DK + (tid - WY5_DK)] = acc / (1.0f + expf(-acc));
                }
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
        for (int tt = warp; tt < len; tt += WY5_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY5_DK; r += 32) {
                const float qv = Qc[tt * WY5_DK + r];
                const float kv = Kc[tt * WY5_DK + r];
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
            for (int r = lane; r < WY5_DK; r += 32) {
                Qc[tt * WY5_DK + r] *= q_inv;
                Kc[tt * WY5_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY5_C + j] = r;
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
            sh[WY5_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        // task 397: 1 ワープ = 1 組。レーン 4 要素 -> 5 段の木 (従属連鎖 128 -> 9)。
        {
            const int npair = len * (len + 1) / 2;
            for (int p = warp; p < npair; p += WY5_WARPS) {
                int i = 0;
                int rem = p;
                while (rem > i) {
                    rem -= (i + 1);
                    ++i;
                }
                const int j = rem;
                const float* ki = Kc + i * WY5_DK;
                const float* kj = Kc + j * WY5_DK;
                const float* qi = Qc + i * WY5_DK;
                float kg = 0.0f;
                float qg = 0.0f;
#pragma unroll
                for (int r = lane; r < WY5_DK; r += 32) {
                    const float kv = kj[r];
                    kg += ki[r] * kv;
                    qg += qi[r] * kv;
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    kg += __shfl_down_sync(0xffffffffu, kg, off);
                    qg += __shfl_down_sync(0xffffffffu, qg, off);
                }
                if (lane == 0) {
                    if (j < i) Ac[i * WY5_C + j] = kg * Rdec[i * WY5_C + j];
                    Gc[i * WY5_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY5_C + j]);
                }
            }
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        // task 397: 4 アキュムレータ (従属連鎖 128 -> 32)。
        for (int idx = tid; idx < len * WY5_DV; idx += WY5_THREADS) {
            const int i = idx / WY5_DV;
            const int c = idx - i * WY5_DV;
            const float* ki = Kc + i * WY5_DK;
            const float* qi = Qc + i * WY5_DK;
            float sk0 = 0.0f;
            float sk1 = 0.0f;
            float sk2 = 0.0f;
            float sk3 = 0.0f;
            float sq0 = 0.0f;
            float sq1 = 0.0f;
            float sq2 = 0.0f;
            float sq3 = 0.0f;
#pragma unroll 4
            for (int r = 0; r < WY5_DK; r += 4) {
                const float s0 = S[r * WY5_DV + c];
                const float s1 = S[(r + 1) * WY5_DV + c];
                const float s2 = S[(r + 2) * WY5_DV + c];
                const float s3 = S[(r + 3) * WY5_DV + c];
                sk0 += ki[r] * s0;
                sk1 += ki[r + 1] * s1;
                sk2 += ki[r + 2] * s2;
                sk3 += ki[r + 3] * s3;
                sq0 += qi[r] * s0;
                sq1 += qi[r + 1] * s1;
                sq2 += qi[r + 2] * s2;
                sq3 += qi[r + 3] * s3;
            }
            const float sk = (sk0 + sk1) + (sk2 + sk3);
            const float sq = (sq0 + sq1) + (sq2 + sq3);
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY5_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY5_DV + c] - Pc[i * WY5_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY5_C + j] * Dc[j * WY5_DV + c];
                Dc[i * WY5_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        // task 397: D は確定済みなので行は独立。wy4 は 32 スレッドで行を直列に
        // 回していたが、(i, c) の全要素を 1 スレッドずつに割る (和の順は同じ)。
        for (int idx = tid; idx < len * WY5_DV; idx += WY5_THREADS) {
            const int i = idx / WY5_DV;
            const int c = idx - i * WY5_DV;
            float acc = QPc[idx];
            for (int j = 0; j <= i; ++j) acc += Gc[i * WY5_C + j] * Dc[j * WY5_DV + c];
            o_out[(long long)(t0 + i) * vd + h * WY5_DK + voff + c] = acc;
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY5_LL];
            // task 397: 1 スレッドは列 c を固定して r 方向に 16 本を持つ
            // (wy4 の idx += THREADS は r が 8 ずつ進むので、同じ 16 本)。
            // t を外に回して rl[t] と Dc[t][c] を 1 回ずつ読み、積の順は wy4 と
            // 同じ (rl * Kc * Dc、左から) に保つ。
            const int per = WY5_DK * WY5_DV / WY5_THREADS;
            const int step = WY5_THREADS / WY5_DV;
            const int c = tid % WY5_DV;
            const int r0 = tid / WY5_DV;
            float acc[16];
#pragma unroll
            for (int k = 0; k < per; ++k) {
                acc[k] = S[(r0 + k * step) * WY5_DV + c] * lam_last;
            }
            for (int t = 0; t < len; ++t) {
                const float rlt = rl[t];
                const float dt = Dc[t * WY5_DV + c];
#pragma unroll
                for (int k = 0; k < per; ++k) {
                    acc[k] += rlt * Kc[t * WY5_DK + r0 + k * step] * dt;
                }
            }
#pragma unroll
            for (int k = 0; k < per; ++k) {
                S[(r0 + k * step) * WY5_DV + c] = acc[k];
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY5_DK * WY5_DV; idx += WY5_THREADS) {
        const int r = idx / WY5_DV;
        const int c = idx - r * WY5_DV;
        ssm_out[(long long)h * WY5_DK * WY5_DK + (long long)r * WY5_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) conv_st_out[chq * WY5_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) conv_st_out[chk * WY5_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY5_CS; ++j) conv_st_out[chv * WY5_CS + j] = wndv[j];
    }
}
