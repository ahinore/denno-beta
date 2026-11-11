// task 228: prefill の GDN スキャンを WY (UT) 形式でチャンク化した版。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf_wy`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf_wy.cu \
//        -o kernels_gdn_pf_wy.ptx
//
// なぜ別カーネルか (lab t225c の [q4ledger-pf]、14,018 トークン prefill):
//   `ssm scan_pf` は v3 (`dno_gdn_pf_scan32`) でも 1,310 ms。v3 はレーン・
//   ブロック・書き出しを全部試して -12% で頭打ちで、効いていないのは
//   **1 トークンあたりの依存鎖 (状態の読み → 縮約 → 更新 → 次のトークン)**。
//   ここでは 32 トークンを 1 チャンクにまとめ、チャンク内を
//     A[t][j] = (k_t·k_j) * Π_{m=j+1..t} eg_m       (下三角・減衰込み)
//     D_t = b_t (v_t - Λ_t (k_tᵀ S_0) - Σ_{j<t} A[t][j] D_j)
//     o_t = Λ_t (q_tᵀ S_0) + Σ_{j≤t} (q_t·k_j) Π_{m=j+1..t} eg_m D_j
//     S_new = Λ_last S_0 + Σ_j (Λ_last/Λ_j) k_j ⊗ D_j
//   の行列積 + 前進代入に書き換える。列 (v 方向 128) は互いに独立なので、
//   代入にも出力にもバリアが要らない。**直列なのはチャンク間だけ**。
//
// 数値: 演算の並べ替えなので、v3 とは丸め順が違う (md5 は DIFF しうる)。
//   要素ごとの式 (conv / silu / softplus / sigmoid / L2 / RMS 前の o) は v3 と
//   同じ。減衰の比 Π_{m=j+1..t} eg_m は**割り算ではなく積**で作る (Λ が f32 で
//   0 に落ちても比は落ちない)。CPU 参照実装は
//   crates/denno-engine/src/gdn/wy.rs (`wy_head_scan`) で、テスト
//   `wy_matches_token_scan_tolerance` がトークン逐次との差 6e-8 を固定している。
//   このカーネルはその写しで、ループの順序も同じ。
//
// 前提 (ホスト側 `gdn_pf_wy_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == WY_DK (128) / d_conv == 4 (cs == 3)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
//   形が違う / PTX が無いときは v3 → v2 → v1 → CPU へ fail closed。
//
// 1 ブロック = 1 ヘッド。状態 64 KB を共有メモリに置き、系列をチャンク順に
// 走査する (48 ブロック = 48 SM。SM あたり 1 ブロック)。
//
// 共有メモリは 91,012 B (状態 64 KB + 作業域) で、デバイスの opt-in 上限に収める。
// lab t228a の実機 (RTX 5090, sm_120) は上限が 99 KB で、C = 32 のときの
// 176,644 B は opt-in に失敗して v3 へ落ちた (測定が v3 になっていた)。
// 作業域は C に比例するので C = 8 にすると 91,012 B になる
// (`pf228_scan_wy_smem_matches_source` が上限との比を固定している)。
// チャンク長を変えるときは WY_C と下のオフセット、Rust 側の
// GDN_PF_WY_SMEM_FLOATS を一緒に直すこと。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define WY_DK 128
#define WY_C 8
#define WY_THREADS 512
#define WY_WARPS 16
#define WY_DCONV 4
#define WY_CS 3

// 動的共有メモリの並び (float 単位)
#define WY_S 0
#define WY_KC 16384
#define WY_QC 17408
#define WY_VC 18432
#define WY_DC 19456
#define WY_PC 20480
#define WY_QPC 21504
#define WY_RDEC 22528
#define WY_AC 22592
#define WY_GC 22656
#define WY_LAM 22720
#define WY_RL 22728
#define WY_EG 22736
#define WY_BC 22744
#define WY_LL 22752
#define WY_TOTAL 22753

extern "C" __global__ void __launch_bounds__(WY_THREADS, 1)
dno_gdn_pf_wy32(
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
    float* S = sh + WY_S;
    float* Kc = sh + WY_KC;
    float* Qc = sh + WY_QC;
    float* Vc = sh + WY_VC;
    float* Dc = sh + WY_DC;
    float* Pc = sh + WY_PC;
    float* QPc = sh + WY_QPC;
    float* Rdec = sh + WY_RDEC;
    float* Ac = sh + WY_AC;
    float* Gc = sh + WY_GC;
    float* lam = sh + WY_LAM;
    float* rl = sh + WY_RL;
    float* egc = sh + WY_EG;
    float* bc = sh + WY_BC;

    const int h = blockIdx.x;
    const int kh = h % nk;
    const int kd_total = nk * WY_DK;
    const int vd = nv * WY_DK;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // このスレッドが conv を持つチャネル。
    //   tid <  128        : q の列 r = tid と v の列 r = tid
    //   128 <= tid < 256  : k の列 r = tid - 128
    const bool is_q = (tid < WY_DK);
    const bool is_k = (tid >= WY_DK && tid < 2 * WY_DK);
    const int chq = kh * WY_DK + tid;
    const int chk = kd_total + kh * WY_DK + (tid - WY_DK);
    const int chv = 2 * kd_total + h * WY_DK + tid;

    float w4q[WY_DCONV];
    float wndq[WY_CS];
    float w4k[WY_DCONV];
    float wndk[WY_CS];
    float w4v[WY_DCONV];
    float wndv[WY_CS];
    if (is_q) {
#pragma unroll
        for (int j = 0; j < WY_DCONV; ++j) {
            w4q[j] = wconv[chq * WY_DCONV + j];
            w4v[j] = wconv[chv * WY_DCONV + j];
        }
#pragma unroll
        for (int j = 0; j < WY_CS; ++j) {
            wndq[j] = conv_st_in[chq * WY_CS + j];
            wndv[j] = conv_st_in[chv * WY_CS + j];
        }
    }
    if (is_k) {
#pragma unroll
        for (int j = 0; j < WY_DCONV; ++j) w4k[j] = wconv[chk * WY_DCONV + j];
#pragma unroll
        for (int j = 0; j < WY_CS; ++j) wndk[j] = conv_st_in[chk * WY_CS + j];
    }

    // ---- 状態 (このヘッドの dk x dv) を共有メモリへ -------------------------
    for (int idx = tid; idx < WY_DK * WY_DK; idx += WY_THREADS) {
        S[idx] = ssm_in[(long long)h * WY_DK * WY_DK + idx];
    }
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += WY_C) {
        const int len = (n - t0 < WY_C) ? (n - t0) : WY_C;

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
                Qc[tt * WY_DK + r] = acc / (1.0f + expf(-acc));
                const float xv = qkv[row + chv];
                const float accv =
                    w4v[0] * wndv[0] + w4v[1] * wndv[1] + w4v[2] * wndv[2] + w4v[3] * xv;
                wndv[0] = wndv[1];
                wndv[1] = wndv[2];
                wndv[2] = xv;
                Vc[tt * WY_DK + r] = accv / (1.0f + expf(-accv));
            }
        }
        if (is_k) {
            const int r = tid - WY_DK;
            for (int tt = 0; tt < len; ++tt) {
                const long long row = (long long)(t0 + tt) * conv_dim;
                const float x = qkv[row + chk];
                const float acc =
                    w4k[0] * wndk[0] + w4k[1] * wndk[1] + w4k[2] * wndk[2] + w4k[3] * x;
                wndk[0] = wndk[1];
                wndk[1] = wndk[2];
                wndk[2] = x;
                Kc[tt * WY_DK + r] = acc / (1.0f + expf(-acc));
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
        for (int tt = warp; tt < len; tt += WY_WARPS) {
            float sq = 0.0f;
            float sk2 = 0.0f;
            for (int r = lane; r < WY_DK; r += 32) {
                const float qv = Qc[tt * WY_DK + r];
                const float kv = Kc[tt * WY_DK + r];
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
            for (int r = lane; r < WY_DK; r += 32) {
                Qc[tt * WY_DK + r] *= q_inv;
                Kc[tt * WY_DK + r] *= k_inv;
            }
        }
        __syncthreads();

        // ---- (3) 減衰の比 Π_{m=j+1..t} eg_m と Λ_t / Λ_last/Λ_t ------------
        // CPU 参照 (gdn/wy.rs) と同じ順で作る: Rdec は t ごとに j を降順、
        // rl と Λ_last は t を降順に 1 本の鎖で。
        if (tid < len) {
            float r = egc[tid];
            for (int j = tid - 1; j >= 0; --j) {
                Rdec[tid * WY_C + j] = r;
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
            sh[WY_LL] = r;
        }
        __syncthreads();

        // ---- (4) A[t][j] (j < t) と G[t][j] (j <= t) ------------------------
        for (int idx = tid; idx < len * WY_C; idx += WY_THREADS) {
            const int i = idx / WY_C;
            const int j = idx - i * WY_C;
            if (j > i) continue;
            float kg = 0.0f;
            float qg = 0.0f;
            for (int r = 0; r < WY_DK; ++r) {
                const float kv = Kc[j * WY_DK + r];
                kg += Kc[i * WY_DK + r] * kv;
                qg += Qc[i * WY_DK + r] * kv;
            }
            if (j < i) Ac[i * WY_C + j] = kg * Rdec[i * WY_C + j];
            Gc[i * WY_C + j] = qg * ((j == i) ? 1.0f : Rdec[i * WY_C + j]);
        }
        __syncthreads();

        // ---- (5) P = Λ_t (k_tᵀ S_0), QP = Λ_t (q_tᵀ S_0) --------------------
        for (int idx = tid; idx < len * WY_DK; idx += WY_THREADS) {
            const int i = idx / WY_DK;
            const int c = idx - i * WY_DK;
            const float* ki = Kc + i * WY_DK;
            const float* qi = Qc + i * WY_DK;
            float sk = 0.0f;
            float sq = 0.0f;
            for (int r = 0; r < WY_DK; ++r) {
                const float sv = S[r * WY_DK + c];
                sk += ki[r] * sv;
                sq += qi[r] * sv;
            }
            Pc[idx] = sk * lam[i];
            QPc[idx] = sq * lam[i];
        }
        __syncthreads();

        // ---- (6) 前進代入 (列ごとに独立。バリア不要) ------------------------
        if (tid < WY_DK) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = Vc[i * WY_DK + c] - Pc[i * WY_DK + c];
                for (int j = 0; j < i; ++j) acc -= Ac[i * WY_C + j] * Dc[j * WY_DK + c];
                Dc[i * WY_DK + c] = bc[i] * acc;
            }
        }
        __syncthreads();

        // ---- (7) o = QP + (G の下三角) D ----------------------------------
        if (tid < WY_DK) {
            const int c = tid;
            for (int i = 0; i < len; ++i) {
                float acc = QPc[i * WY_DK + c];
                for (int j = 0; j <= i; ++j) acc += Gc[i * WY_C + j] * Dc[j * WY_DK + c];
                o_out[(long long)(t0 + i) * vd + h * WY_DK + c] = acc;
            }
        }
        __syncthreads();

        // ---- (8) 状態の更新 (チャンクに 1 回) -------------------------------
        {
            const float lam_last = sh[WY_LL];
            for (int idx = tid; idx < WY_DK * WY_DK; idx += WY_THREADS) {
                const int r = idx / WY_DK;
                const int c = idx - r * WY_DK;
                float acc = S[idx] * lam_last;
                for (int t = 0; t < len; ++t) {
                    acc += rl[t] * Kc[t * WY_DK + r] * Dc[t * WY_DK + c];
                }
                S[idx] = acc;
            }
        }
        __syncthreads();
    }

    // ---- 最終状態と conv 窓の書き戻し ---------------------------------------
    for (int idx = tid; idx < WY_DK * WY_DK; idx += WY_THREADS) {
        ssm_out[(long long)h * WY_DK * WY_DK + idx] = S[idx];
    }
    // q/k は同じ k-head を持つ複数ブロックが同じ値を計算するので、書き込みは
    // k-head の先頭ブロック (h == kh) だけが行う (write-write 競合を作らない)。
    if (is_q) {
        if (h == kh) {
#pragma unroll
            for (int j = 0; j < WY_CS; ++j) conv_st_out[chq * WY_CS + j] = wndq[j];
        }
#pragma unroll
        for (int j = 0; j < WY_CS; ++j) conv_st_out[chv * WY_CS + j] = wndv[j];
    }
    if (is_k && h == kh) {
#pragma unroll
        for (int j = 0; j < WY_CS; ++j) conv_st_out[chk * WY_CS + j] = wndk[j];
    }
}
