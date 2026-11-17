// task 407: prefill の GDN スキャン (WY 形式、v 列 4 分割) の従属連鎖をさらに短くした版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy6p`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy6p.cu \
//        -o kernels_gdn_pf_wy6p.ptx
//
// wy6 (同じファイル群の 512 スレッド版) との違いは **(5) P/QP の r 分割だけ**:
//   128 要素のドット積を 1 スレッドが持つのをやめ、隣り合う 2 レーンで
//   前半 64 / 後半 64 を持って `__shfl_xor_sync(...,1)` で足す。1 スレッドの
//   従属連鎖は 32 -> 16 段になり、(5) が 512 スレッド全部を使う。
//
// 数値: 和の木が wy6 (= wy5) と違うので要素ごとに数 ulp 変わりうる
//   (**意図した丸め差**。決定論的)。ゲート `DENNO_PF_SCAN_WY6P` (既定 off)。
//
//   ペア (2e, 2e+1) は必ず同じワープに居る: 反復回数は 2*len*dv で dv=32 なので
//   常に 64 の倍数 = ワープ境界をまたがない (len < 8 の端数チャンクでも
//   ペアが片側だけ inactive にならない)。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY6P_DK 128
#define WY6P_DV 32
#define WY6P_C 8
#define WY6P_THREADS 512
#define WY6P_WARPS 16
#define WY6P_DCONV 4
#define WY6P_CS 3

// 動的共有メモリの並び (float 単位)。wy4 と同じ (作業域を増やしていない)。
#define WY6P_S 0
#define WY6P_KC 4096
#define WY6P_QC 5120
#define WY6P_VC 6144
#define WY6P_DC 6400
#define WY6P_PC 6656
#define WY6P_QPC 6912
#define WY6P_RDEC 7168
#define WY6P_AC 7232
#define WY6P_GC 7296
#define WY6P_LAM 7360
#define WY6P_RL 7368
#define WY6P_EG 7376
#define WY6P_BC 7384
#define WY6P_LL 7392
#define WY6P_TOTAL 7393

extern "C" __global__ void __launch_bounds__(WY6P_THREADS, 1)
dno_gdn_pf_wy6p(
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
    float* S = sh + WY6P_S;
    float* Kc = sh + WY6P_KC;
    float* Qc = sh + WY6P_QC;
    float* Vc = sh + WY6P_VC;
    float* Dc = sh + WY6P_DC;
    float* Pc = sh + WY6P_PC;
    float* QPc = sh + WY6P_QPC;
    float* Rdec = sh + WY6P_RDEC;
    float* Ac = sh + WY6P_AC;
    float* Gc = sh + WY6P_GC;
    float* lam = sh + WY6P_LAM;
    float* rl = sh + WY6P_RL;
    float* egc = sh + WY6P_EG;
    float* bc = sh + WY6P_BC;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/4 (0..3)
    const int voff = vb * WY6P_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY6P_DK;
    const int vd = nv * WY6P_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  32         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY6P_DK);
    const bool is_k = (tid >= WY6P_DK && tid < 2 * WY6P_DK);
    const bool is_v = (tid < WY6P_DV);
    const int chq = kh * WY6P_DK + tid;
    const int chk = kd_total + kh * WY6P_DK + (tid - WY6P_DK);
    const int chv = 2 * kd_total + h * WY6P_DK + voff + tid;

    float w4q[WY6P_DCONV];
    float wndq[WY6P_CS];
    float w4k[WY6P_DCONV];
    float wndk[WY6P_CS];
    float w4v[WY6P_DCONV];
    float wndv[WY6P_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY6P_DCONV; ++j) w4q[j] = wconv[chq * WY6P_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) wndq[j] = conv_st_in[chq * WY6P_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY6P_DCONV; ++j) w4k[j] = wconv[chk * WY6P_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) wndk[j] = conv_st_in[chk * WY6P_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY6P_DCONV; ++j) w4v[j] = wconv[chv * WY6P_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) wndv[j] = conv_st_in[chv * WY6P_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY6P_DK * WY6P_DV; idx += WY6P_THREADS) {
        const int r = idx / WY6P_DV;
        const int c = idx - r * WY6P_DV;
        S[idx] = ssm_in[(long long)h * WY6P_DK * WY6P_DK + (long long)r * WY6P_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY6P_C) {
        const int len = (n - t0 < WY6P_C) ? (n - t0) : WY6P_C;

        // ---- (1) conv + silu (このチャンクの全トークン) と gate -----------
        // task 397: 入力 8 本を先に読む (独立なロードをまとめて出す)。
        // 窓の前進 (`wnd?` の更新) は `tt < len` の内側だけ = wy4 と同じ。
        if (is_q) {
            float xq[WY6P_C];
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                xq[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chq]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                if (tt < len) {
                    const float x = xq[tt];
                    const float acc =
                        w4q[0] * wndq[0] + w4q[1] * wndq[1] + w4q[2] * wndq[2] + w4q[3] * x;
                    wndq[0] = wndq[1];
                    wndq[1] = wndq[2];
                    wndq[2] = x;
                    Qc[tt * WY6P_DK + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_v) {
            float xv[WY6P_C];
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                xv[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chv]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                if (tt < len) {
                    const float x = xv[tt];
                    const float acc =
                        w4v[0] * wndv[0] + w4v[1] * wndv[1] + w4v[2] * wndv[2] + w4v[3] * x;
                    wndv[0] = wndv[1];
                    wndv[1] = wndv[2];
                    wndv[2] = x;
                    Vc[tt * WY6P_DV + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_k) {
            float xk[WY6P_C];
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                xk[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chk]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY6P_C; ++tt) {
                if (tt < len) {
                    const float x = xk[tt];
                    const float acc =
                        w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                    wndk[0] = wndk[1];
                    wndk[1] = wndk[2];
                    wndk[2] = x;
                    Kc[tt * WY6P_DK + (tid - WY6P_DK)] = acc / (1.0f + expf(-acc));
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
        for (int tt = warp; tt < len; tt += WY6P_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY6P_DK; r += 32) {
                const float qv = Qc[tt * WY6P_DK + r];
                const float kv = Kc[tt * WY6P_DK + r];
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
            for (int r = lane; r < WY6P_DK; r += 32) {
                Qc[tt * WY6P_DK + r] *= q_inv;
                Kc[tt * WY6P_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY6P_C + j] = r;
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
            sh[WY6P_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        // task 397: 1 ワープ = 1 組。レーン 4 要素 -> 5 段の木 (従属連鎖 128 -> 9)。
        {
            const int npair = len * (len + 1) / 2;
            for (int p = warp; p < npair; p += WY6P_WARPS) {
                int i = 0;
                int rem = p;
                while (rem > i) {
                    rem -= (i + 1);
                    ++i;
                }
                const int j = rem;
                const float* ki = Kc + i * WY6P_DK;
                const float* kj = Kc + j * WY6P_DK;
                const float* qi = Qc + i * WY6P_DK;
                float kg = 0.0f;
                float qg = 0.0f;
#pragma unroll
                for (int r = lane; r < WY6P_DK; r += 32) {
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
                    if (j < i) Ac[i * WY6P_C + j] = kg * Rdec[i * WY6P_C + j];
                    Gc[i * WY6P_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY6P_C + j]);
                }
            }
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        // task 407: 1 要素 (i,c) の r 方向 (dk=128) を**2 本に割る**。隣り合う
        // レーン (2e, 2e+1) が同じ (i,c) の前半 64 / 後半 64 を持ち、最後に
        // `__shfl_xor_sync(...,1)` で足す。1 スレッドの従属連鎖は 32 -> 16 段。
        // 和の木が wy6 (= wy5) と違うので**丸めが変わる** (決定論的)。
        for (int j = tid; j < 2 * len * WY6P_DV; j += WY6P_THREADS) {
            const int idx = j >> 1;
            const int half = j & 1;
            const int i = idx / WY6P_DV;
            const int c = idx - i * WY6P_DV;
            const float* ki = Kc + i * WY6P_DK;
            const float* qi = Qc + i * WY6P_DK;
            const int rh = WY6P_DK / 2;
            const int r0 = half * rh;
            float sk0 = 0.0f;
            float sk1 = 0.0f;
            float sk2 = 0.0f;
            float sk3 = 0.0f;
            float sq0 = 0.0f;
            float sq1 = 0.0f;
            float sq2 = 0.0f;
            float sq3 = 0.0f;
#pragma unroll 4
            for (int r = r0; r < r0 + rh; r += 4) {
                const float s0 = S[r * WY6P_DV + c];
                const float s1 = S[(r + 1) * WY6P_DV + c];
                const float s2 = S[(r + 2) * WY6P_DV + c];
                const float s3 = S[(r + 3) * WY6P_DV + c];
                sk0 += ki[r] * s0;
                sk1 += ki[r + 1] * s1;
                sk2 += ki[r + 2] * s2;
                sk3 += ki[r + 3] * s3;
                sq0 += qi[r] * s0;
                sq1 += qi[r + 1] * s1;
                sq2 += qi[r + 2] * s2;
                sq3 += qi[r + 3] * s3;
            }
            float sk = (sk0 + sk1) + (sk2 + sk3);
            float sq = (sq0 + sq1) + (sq2 + sq3);
            sk += __shfl_xor_sync(0xffffffffu, sk, 1);
            sq += __shfl_xor_sync(0xffffffffu, sq, 1);
            if (half == 0) {
                Pc[idx] = sk * lam[i];
                QPc[idx] = sq * lam[i];
            }
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY6P_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY6P_DV + c] - Pc[i * WY6P_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY6P_C + j] * Dc[j * WY6P_DV + c];
                Dc[i * WY6P_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        // task 397: D は確定済みなので行は独立。wy4 は 32 スレッドで行を直列に
        // 回していたが、(i, c) の全要素を 1 スレッドずつに割る (和の順は同じ)。
        for (int idx = tid; idx < len * WY6P_DV; idx += WY6P_THREADS) {
            const int i = idx / WY6P_DV;
            const int c = idx - i * WY6P_DV;
            float acc = QPc[idx];
            for (int j = 0; j <= i; ++j) acc += Gc[i * WY6P_C + j] * Dc[j * WY6P_DV + c];
            o_out[(long long)(t0 + i) * vd + h * WY6P_DK + voff + c] = acc;
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY6P_LL];
            // task 397: 1 スレッドは列 c を固定して r 方向に 16 本を持つ
            // (wy4 の idx += THREADS は r が 8 ずつ進むので、同じ 16 本)。
            // t を外に回して rl[t] と Dc[t][c] を 1 回ずつ読み、積の順は wy4 と
            // 同じ (rl * Kc * Dc、左から) に保つ。
            const int per = WY6P_DK * WY6P_DV / WY6P_THREADS;
            const int step = WY6P_THREADS / WY6P_DV;
            const int c = tid % WY6P_DV;
            const int r0 = tid / WY6P_DV;
            float acc[16];
#pragma unroll
            for (int k = 0; k < per; ++k) {
                acc[k] = S[(r0 + k * step) * WY6P_DV + c] * lam_last;
            }
            for (int t = 0; t < len; ++t) {
                const float rlt = rl[t];
                const float dt = Dc[t * WY6P_DV + c];
#pragma unroll
                for (int k = 0; k < per; ++k) {
                    acc[k] += rlt * Kc[t * WY6P_DK + r0 + k * step] * dt;
                }
            }
#pragma unroll
            for (int k = 0; k < per; ++k) {
                S[(r0 + k * step) * WY6P_DV + c] = acc[k];
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY6P_DK * WY6P_DV; idx += WY6P_THREADS) {
        const int r = idx / WY6P_DV;
        const int c = idx - r * WY6P_DV;
        ssm_out[(long long)h * WY6P_DK * WY6P_DK + (long long)r * WY6P_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) conv_st_out[chq * WY6P_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) conv_st_out[chk * WY6P_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY6P_CS; ++j) conv_st_out[chv * WY6P_CS + j] = wndv[j];
    }
}
