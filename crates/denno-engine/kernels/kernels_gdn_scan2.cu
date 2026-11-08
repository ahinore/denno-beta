// task 184: decode の GDN (ssm) スキャンの第 2 版。
//
// 背景 (lab t177a の `DENNO_GDN_DEC_KPROF`): `dno_gdn_scan_sig` は 48 ブロック x
// 128 スレッド = 192 warp しか立たず (5090 の 170 SM に 1.1 warp/SM)、状態
// 3.1 MB を 26.2 us で舐める (240 GB/s)。1 スレッドが列 (dk=128) を **ローカル
// メモリ** に持ち、4 本の 128 段の依存鎖 (decay / skv / update / o) を直列に
// 回していたため、レイテンシを隠す warp が無い。
//
// ここでは 1 列を **4 レーン** で分け、列を **レジスタ** に置く (1 レーン =
// dk/4 = 32 要素)。ブロックは 128 列 x 4 レーン = 512 スレッド = 16 warp になり、
// 依存鎖は 32 段に縮む。縮約 (skv / o) は同じ列の 4 レーンを `__shfl_*_sync` の
// 幅 4 で畳む (固定順 = 決定論)。
//
// 数値: skv と o の加算順が「1 スレッドの逐次和」から「4 分割 + 固定木」に
// 変わる = 丸めが変わる (md5 は DIFF しうる)。それ以外の演算列 (decay の乗算、
// update の k*dlt 加算、conv の左結合、q/k の L2 正規化の逐次和、出力の
// rmsnorm * wnorm * sigmoid(z)) は `dno_gdn_scan_sig` と 1 演算 1 丸めで揃える。
//
// 前提 (ホスト側で保証する): dk == dv == 128 / d_conv == 4 / cs == 3。
// それ以外は従来カーネル (`kernels_gdn.ptx` の dno_gdn_scan_sig) を使う。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_gdn_scan2.cu -o kernels_gdn_scan2.ptx
// まとめて作るなら `kernels/build_ptx.sh gdn_scan2`。

#define DNO_S2_DK 128
#define DNO_S2_NS 32   // 1 レーンが持つ列の要素数 (dk / 4)

// 同じ列の 4 レーン (幅 4 の quad) の合計。lane 0 の値だけが使われる。
// 加算順は固定 (offset 1 → 2)。
__device__ __forceinline__ float dno_s2_quad_sum(float v)
{
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 1, 4));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 2, 4));
    return v;
}

extern "C" __global__ void dno_gdn_scan_sig2(
    const float* __restrict__ qkv,   // [n, conv_dim] (conv 入力 = 投影済み qkv 行)
    const float* __restrict__ z,     // [n, value_dim]
    const float* __restrict__ beta,  // [n, nv]
    const float* __restrict__ alpha, // [n, nv]
    const float* __restrict__ conv_st_in,  // [conv_dim, cs] 初期 conv 状態
    const float* __restrict__ ssm_in,      // [nv, dk, dv] 初期 SSM 状態
    const float* __restrict__ wconv, // [conv_dim, d_conv]
    const float* __restrict__ wa,    // [nv] (負値)
    const float* __restrict__ wdt,   // [nv]
    const float* __restrict__ wnorm, // [dv]
    float* __restrict__ y_out,       // [n, value_dim]
    float* __restrict__ conv_st_out, // [conv_dim, cs] 最終 conv 状態
    float* __restrict__ ssm_out,     // [nv, dk, dv] 最終 SSM 状態
    int n, int nk, int nv, int dk, int dv, int conv_dim, int d_conv, int cs,
    float eps, float q_scale)
{
    const int h = blockIdx.x;       // v-head
    const int kh = h % nk;
    const int kd_total = nk * dk;
    const int vd = nv * dv;
    const int t = threadIdx.x;      // 0 .. dv*4-1
    const int c = t >> 2;           // 列 (0..dv-1)
    const int lane = t & 3;         // 0..3 (同じ列の 4 レーン)

    extern __shared__ float sh[];
    // layout: q_sh[dk] | k_sh[dk] | v_sh[dk] | o_sh[dv] | scalars[5]
    float* q_sh = sh;
    float* k_sh = sh + dk;
    float* v_sh = sh + 2 * dk;
    float* o_sh = sh + 3 * dk;
    float* sc = sh + 3 * dk + dv;

    // ---- S の列 (レジスタ) ------------------------------------------------
    // in == out (同一ポインタ) を許す: 各レーンは自分の列の担当行を先頭で全読み
    // してから末尾で書く。他スレッドはその行を触らない。
    float S[DNO_S2_NS];
#pragma unroll
    for (int i = 0; i < DNO_S2_NS; ++i) {
        S[i] = ssm_in[((long long)h * dk + lane * DNO_S2_NS + i) * dv + c];
    }

    // ---- conv 重み + 初期ウィンドウ (lane 0 が担当 3 チャネル) -------------
    const int qch = kh * dk + c;
    const int kch = kd_total + kh * dk + c;
    const int vch = 2 * kd_total + h * dv + c;
    float wq[4], wk[4], wv[4];
    float inq[3], ink[3], inv[3];
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wq[j] = wconv[qch * d_conv + j];
            wk[j] = wconv[kch * d_conv + j];
            wv[j] = wconv[vch * d_conv + j];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            inq[j] = conv_st_in[qch * cs + j];
            ink[j] = conv_st_in[kch * cs + j];
            inv[j] = conv_st_in[vch * cs + j];
        }
    }

    for (int tt = 0; tt < n; ++tt) {
        // 前トークンの共有メモリ読み取りが残っていないか保護
        __syncthreads();

        // ---- 1. causal depthwise conv + SiLU (lane 0 が 3 チャネル) --------
        if (lane == 0) {
            const long long row = (long long)tt * conv_dim;
            const float xq = qkv[row + qch];
            const float xk = qkv[row + kch];
            const float xv = qkv[row + vch];
            const float accq =
                wq[0] * inq[0] + wq[1] * inq[1] + wq[2] * inq[2] + wq[3] * xq;
            const float acck =
                wk[0] * ink[0] + wk[1] * ink[1] + wk[2] * ink[2] + wk[3] * xk;
            const float accv =
                wv[0] * inv[0] + wv[1] * inv[1] + wv[2] * inv[2] + wv[3] * xv;
            inq[0] = inq[1]; inq[1] = inq[2]; inq[2] = xq;
            ink[0] = ink[1]; ink[1] = ink[2]; ink[2] = xk;
            inv[0] = inv[1]; inv[1] = inv[2]; inv[2] = xv;
            q_sh[c] = accq / (1.0f + expf(-accq));
            k_sh[c] = acck / (1.0f + expf(-acck));
            v_sh[c] = accv / (1.0f + expf(-accv));
        }
        __syncthreads();

        // ---- 2. q/k L2 正規化スケール + decay/beta (単一スレッド逐次和) -----
        if (t == 0) {
            float sq = 0.0f, sk2 = 0.0f;
            for (int i = 0; i < dk; ++i) {
                sq += q_sh[i] * q_sh[i];
                sk2 += k_sh[i] * k_sh[i];
            }
            sc[0] = 1.0f / sqrtf(sq + eps);
            sc[1] = 1.0f / sqrtf(sk2 + eps);
            const float a = alpha[(long long)tt * nv + h] + wdt[h];
            const float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            sc[2] = expf(wa[h] * sp);
            const float bt = beta[(long long)tt * nv + h];
            sc[3] = 1.0f / (1.0f + expf(-bt));
        }
        __syncthreads();

        // ---- 3. q スケール + k 正規化 (自列) -------------------------------
        if (lane == 0) {
            q_sh[c] *= sc[0] * q_scale;
            k_sh[c] *= sc[1];
        }
        __syncthreads();

        // ---- 4. デルタルール (列ごとに完結・4 レーンで分担) ----------------
        const float eg = sc[2];
        const float b = sc[3];
        float part = 0.0f; // sk[c] = Σ_r k[r] * S[r][c] のレーン分担
#pragma unroll
        for (int i = 0; i < DNO_S2_NS; ++i) {
            const int r = lane * DNO_S2_NS + i;
            S[i] = __fmul_rn(S[i], eg); // S ← S * exp(g)
            part = __fadd_rn(part, __fmul_rn(k_sh[r], S[i]));
        }
        part = dno_s2_quad_sum(part);
        float dlt = 0.0f;
        if (lane == 0) {
            dlt = __fmul_rn(b, __fsub_rn(v_sh[c], part)); // d[c] = beta * (v[c] - sk[c])
        }
        dlt = __shfl_sync(0xffffffffu, dlt, 0, 4);
        float o = 0.0f; // o[c] = Σ_r q[r] * S[r][c] のレーン分担
#pragma unroll
        for (int i = 0; i < DNO_S2_NS; ++i) {
            const int r = lane * DNO_S2_NS + i;
            S[i] = __fadd_rn(S[i], __fmul_rn(k_sh[r], dlt));
            o = __fadd_rn(o, __fmul_rn(q_sh[r], S[i]));
        }
        o = dno_s2_quad_sum(o);
        if (lane == 0) {
            o_sh[c] = o;
        }
        __syncthreads();

        // ---- 5. gated RMS norm: rmsnorm(o) * w * sigmoid(z) ----------------
        if (t == 0) {
            float s = 0.0f;
            for (int i = 0; i < dv; ++i) {
                s += o_sh[i] * o_sh[i];
            }
            sc[4] = 1.0f / sqrtf(s / (float)dv + eps);
        }
        __syncthreads();
        if (lane == 0) {
            const long long zo = (long long)tt * vd + h * dv + c;
            const float zv = z[zo];
            y_out[zo] = o_sh[c] * sc[4] * wnorm[c] * (1.0f / (1.0f + expf(-zv)));
        }
    }

    // ---- 最終状態の書き戻し -----------------------------------------------
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            conv_st_out[qch * cs + j] = inq[j];
            conv_st_out[kch * cs + j] = ink[j];
            conv_st_out[vch * cs + j] = inv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < DNO_S2_NS; ++i) {
        const int r = lane * DNO_S2_NS + i;
        ssm_out[((long long)h * dk + r) * dv + c] = S[i];
    }
}
