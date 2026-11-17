// task 407: GDN スキャンの (5) P/QP が S を 8 回読み直すのをやめた版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy7q`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy7q.cu \
//        -o kernels_gdn_pf_wy7q.ptx
//
// wy7 (バリア 8 -> 6 本) との違いは (5) P/QP の割り方だけ:
//   wy5/wy7 は warp i が P[i][*] を 128 段のドット積で作るので、8 ワープが
//   **同じ S の列**を 8 回読む (1 チャンク 128 KB)。wy7q は r (dk=128) を
//   ワープで 16 本ずつに分け (レーン = 列 c)、8 行ぶんの部分和をレジスタに
//   持ってから共有メモリ (Pp / Qp、各 8 x 8 x 32 = 2,048 float) で 8 本を
//   足す。1 レーンの共有メモリロードは 256 -> 152、従属連鎖は 32 -> 16 段。
//   和の木が変わるので**丸めが wy5 / wy7 と違う** (決定論的)。
//
// 共有メモリ: 7,393 + 2 x 2,048 = 11,489 float (45,956 B。48 KB の内側)。
// ゲート `DENNO_PF_SCAN_WY7Q` (既定 off)。落ちたら wy7 -> wy5 へ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY7Q_DK 128
#define WY7Q_DV 32
#define WY7Q_C 8
#define WY7Q_THREADS 256
#define WY7Q_WARPS 8
#define WY7Q_DCONV 4
#define WY7Q_CS 3

// 動的共有メモリの並び (float 単位)。wy4 と同じ (作業域を増やしていない)。
#define WY7Q_S 0
#define WY7Q_KC 4096
#define WY7Q_QC 5120
#define WY7Q_VC 6144
#define WY7Q_DC 6400
#define WY7Q_PC 6656
#define WY7Q_QPC 6912
#define WY7Q_RDEC 7168
#define WY7Q_AC 7232
#define WY7Q_GC 7296
#define WY7Q_LAM 7360
#define WY7Q_RL 7368
#define WY7Q_EG 7376
#define WY7Q_BC 7384
#define WY7Q_LL 7392
#define WY7Q_PP 7393
#define WY7Q_QP 9441
#define WY7Q_TOTAL 11489

extern "C" __global__ void __launch_bounds__(WY7Q_THREADS, 1)
dno_gdn_pf_wy7q(
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
    float* S = sh + WY7Q_S;
    float* Kc = sh + WY7Q_KC;
    float* Qc = sh + WY7Q_QC;
    float* Vc = sh + WY7Q_VC;
    float* Dc = sh + WY7Q_DC;
    float* Pc = sh + WY7Q_PC;
    float* QPc = sh + WY7Q_QPC;
    float* Rdec = sh + WY7Q_RDEC;
    float* Ac = sh + WY7Q_AC;
    float* Gc = sh + WY7Q_GC;
    float* lam = sh + WY7Q_LAM;
    float* rl = sh + WY7Q_RL;
    float* egc = sh + WY7Q_EG;
    float* bc = sh + WY7Q_BC;
    float* Pp = sh + WY7Q_PP;
    float* Qp = sh + WY7Q_QP;

    const int h = blockIdx.x;
    const int vb = blockIdx.y;              // v 列の 1/4 (0..3)
    const int voff = vb * WY7Q_DV;
    const int kh = h % nk;
    const int kd_total = nk * WY7Q_DK;
    const int vd = nv * WY7Q_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    //   tid <  32         : さらに v の列 r = voff + tid
    const bool is_q = (tid < WY7Q_DK);
    const bool is_k = (tid >= WY7Q_DK && tid < 2 * WY7Q_DK);
    const bool is_v = (tid < WY7Q_DV);
    const int chq = kh * WY7Q_DK + tid;
    const int chk = kd_total + kh * WY7Q_DK + (tid - WY7Q_DK);
    const int chv = 2 * kd_total + h * WY7Q_DK + voff + tid;

    float w4q[WY7Q_DCONV];
    float wndq[WY7Q_CS];
    float w4k[WY7Q_DCONV];
    float wndk[WY7Q_CS];
    float w4v[WY7Q_DCONV];
    float wndv[WY7Q_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY7Q_DCONV; ++j) w4q[j] = wconv[chq * WY7Q_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) wndq[j] = conv_st_in[chq * WY7Q_CS + j];
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY7Q_DCONV; ++j) w4k[j] = wconv[chk * WY7Q_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) wndk[j] = conv_st_in[chk * WY7Q_CS + j];
    }
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY7Q_DCONV; ++j) w4v[j] = wconv[chv * WY7Q_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) wndv[j] = conv_st_in[chv * WY7Q_CS + j];
    }

    // ---- 状態 (このヘッドの dk x このブロックの dv 列) を共有メモリへ -------
    for (int idx = tid; idx < WY7Q_DK * WY7Q_DV; idx += WY7Q_THREADS) {
        const int r = idx / WY7Q_DV;
        const int c = idx - r * WY7Q_DV;
        S[idx] = ssm_in[(long long)h * WY7Q_DK * WY7Q_DK + (long long)r * WY7Q_DK + voff + c];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY7Q_C) {
        const int len = (n - t0 < WY7Q_C) ? (n - t0) : WY7Q_C;

        // ---- (1) conv + silu (このチャンクの全トークン) と gate -----------
        // task 397: 入力 8 本を先に読む (独立なロードをまとめて出す)。
        // 窓の前進 (`wnd?` の更新) は `tt < len` の内側だけ = wy4 と同じ。
        if (is_q) {
            float xq[WY7Q_C];
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                xq[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chq]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                if (tt < len) {
                    const float x = xq[tt];
                    const float acc =
                        w4q[0] * wndq[0] + w4q[1] * wndq[1] + w4q[2] * wndq[2] + w4q[3] * x;
                    wndq[0] = wndq[1];
                    wndq[1] = wndq[2];
                    wndq[2] = x;
                    Qc[tt * WY7Q_DK + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_v) {
            float xv[WY7Q_C];
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                xv[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chv]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                if (tt < len) {
                    const float x = xv[tt];
                    const float acc =
                        w4v[0] * wndv[0] + w4v[1] * wndv[1] + w4v[2] * wndv[2] + w4v[3] * x;
                    wndv[0] = wndv[1];
                    wndv[1] = wndv[2];
                    wndv[2] = x;
                    Vc[tt * WY7Q_DV + tid] = acc / (1.0f + expf(-acc));
                }
            }
        }
        if (is_k) {
            float xk[WY7Q_C];
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                xk[tt] = (tt < len)
                    ? qkv[(long long)(t0 + tt) * conv_dim + chk]
                    : 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < WY7Q_C; ++tt) {
                if (tt < len) {
                    const float x = xk[tt];
                    const float acc =
                        w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                    wndk[0] = wndk[1];
                    wndk[1] = wndk[2];
                    wndk[2] = x;
                    Kc[tt * WY7Q_DK + (tid - WY7Q_DK)] = acc / (1.0f + expf(-acc));
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
        for (int tt = warp; tt < len; tt += WY7Q_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY7Q_DK; r += 32) {
                const float qv = Qc[tt * WY7Q_DK + r];
                const float kv = Kc[tt * WY7Q_DK + r];
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
            for (int r = lane; r < WY7Q_DK; r += 32) {
                Qc[tt * WY7Q_DK + r] *= q_inv;
                Kc[tt * WY7Q_DK + r] *= k_inv;
            }
        }
        // task 407: (2) と (3) は別の配列しか触らない (Qc/Kc と Rdec/lam/rl) ので
        // 間のバリアを落とす (両方とも (1) のバリアの後ろで、 (4) のバリアで締める)。

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY7Q_C + j] = r;
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
            sh[WY7Q_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        // task 397: 1 ワープ = 1 組。レーン 4 要素 -> 5 段の木 (従属連鎖 128 -> 9)。
        {
            const int npair = len * (len + 1) / 2;
            for (int p = warp; p < npair; p += WY7Q_WARPS) {
                int i = 0;
                int rem = p;
                while (rem > i) {
                    rem -= (i + 1);
                    ++i;
                }
                const int j = rem;
                const float* ki = Kc + i * WY7Q_DK;
                const float* kj = Kc + j * WY7Q_DK;
                const float* qi = Qc + i * WY7Q_DK;
                float kg = 0.0f;
                float qg = 0.0f;
#pragma unroll
                for (int r = lane; r < WY7Q_DK; r += 32) {
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
                    if (j < i) Ac[i * WY7Q_C + j] = kg * Rdec[i * WY7Q_C + j];
                    Gc[i * WY7Q_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY7Q_C + j]);
                }
            }
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        // task 407: wy5 は 8 ワープが**同じ S の列**を 8 回読み直していた
        // (1 チャンク 128 KB)。r (dk=128) をワープで 16 本ずつに分け、
        // レーン = 列 c として 8 行ぶんの部分和をレジスタに持ち、最後に
        // ワープ分 (8 本) を共有メモリで足す。1 レーンのロードは 256 -> 152。
        // 和の木が wy5 と違うので**丸めが変わる** (決定論的)。
        {
            const int c = lane;
            const int rn = WY7Q_DK / WY7Q_WARPS;
            const int r0 = warp * rn;
            float sk[WY7Q_C];
            float sq[WY7Q_C];
#pragma unroll
            for (int i = 0; i < WY7Q_C; ++i) {
                sk[i] = 0.0f;
                sq[i] = 0.0f;
            }
            for (int r = r0; r < r0 + rn; ++r) {
                const float s = S[r * WY7Q_DV + c];
                for (int i = 0; i < len; ++i) {
                    sk[i] += Kc[i * WY7Q_DK + r] * s;
                    sq[i] += Qc[i * WY7Q_DK + r] * s;
                }
            }
            for (int i = 0; i < len; ++i) {
                Pp[warp * WY7Q_C * WY7Q_DV + i * WY7Q_DV + c] = sk[i];
                Qp[warp * WY7Q_C * WY7Q_DV + i * WY7Q_DV + c] = sq[i];
            }
        }
        __syncthreads();
        // ワープ 8 本ぶんの部分和を w 昇順で足す (順序は固定 = 決定論的)。
        for (int idx = tid; idx < len * WY7Q_DV; idx += WY7Q_THREADS) {
            const int i = idx / WY7Q_DV;
            float sk = 0.0f;
            float sq = 0.0f;
#pragma unroll
            for (int w = 0; w < WY7Q_WARPS; ++w) {
                sk += Pp[w * WY7Q_C * WY7Q_DV + idx];
                sq += Qp[w * WY7Q_C * WY7Q_DV + idx];
            }
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY7Q_DV) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY7Q_DV + c] - Pc[i * WY7Q_DV + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY7Q_C + j] * Dc[j * WY7Q_DV + c];
                Dc[i * WY7Q_DV + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        // task 397: D は確定済みなので行は独立。wy4 は 32 スレッドで行を直列に
        // 回していたが、(i, c) の全要素を 1 スレッドずつに割る (和の順は同じ)。
        for (int idx = tid; idx < len * WY7Q_DV; idx += WY7Q_THREADS) {
            const int i = idx / WY7Q_DV;
            const int c = idx - i * WY7Q_DV;
            float acc = QPc[idx];
            for (int j = 0; j <= i; ++j) acc += Gc[i * WY7Q_C + j] * Dc[j * WY7Q_DV + c];
            o_out[(long long)(t0 + i) * vd + h * WY7Q_DK + voff + c] = acc;
        }
        // task 407: (7) は o_out (global) だけを書き、(8) は S だけを書く。
        // どちらも Dc を読むだけなので間のバリアを落とす。

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY7Q_LL];
            // task 397: 1 スレッドは列 c を固定して r 方向に 16 本を持つ
            // (wy4 の idx += THREADS は r が 8 ずつ進むので、同じ 16 本)。
            // t を外に回して rl[t] と Dc[t][c] を 1 回ずつ読み、積の順は wy4 と
            // 同じ (rl * Kc * Dc、左から) に保つ。
            const int per = WY7Q_DK * WY7Q_DV / WY7Q_THREADS;
            const int step = WY7Q_THREADS / WY7Q_DV;
            const int c = tid % WY7Q_DV;
            const int r0 = tid / WY7Q_DV;
            float acc[16];
#pragma unroll
            for (int k = 0; k < per; ++k) {
                acc[k] = S[(r0 + k * step) * WY7Q_DV + c] * lam_last;
            }
            for (int t = 0; t < len; ++t) {
                const float rlt = rl[t];
                const float dt = Dc[t * WY7Q_DV + c];
#pragma unroll
                for (int k = 0; k < per; ++k) {
                    acc[k] += rlt * Kc[t * WY7Q_DK + r0 + k * step] * dt;
                }
            }
#pragma unroll
            for (int k = 0; k < per; ++k) {
                S[(r0 + k * step) * WY7Q_DV + c] = acc[k];
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY7Q_DK * WY7Q_DV; idx += WY7Q_THREADS) {
        const int r = idx / WY7Q_DV;
        const int c = idx - r * WY7Q_DV;
        ssm_out[(long long)h * WY7Q_DK * WY7Q_DK + (long long)r * WY7Q_DK + voff + c] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh かつ vb == 0) だけが行う。
    if (is_q && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) conv_st_out[chq * WY7Q_CS + j] = wndq[j];
    }
    if (is_k && h == kh && vb == 0) {
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) conv_st_out[chk * WY7Q_CS + j] = wndk[j];
    }
    // v の窓は列の持ち主 (このブロックの 32 列) が書く。
    if (is_v) {
#pragma unroll
        for (int j = 0; j < WY7Q_CS; ++j) conv_st_out[chv * WY7Q_CS + j] = wndv[j];
    }
}
