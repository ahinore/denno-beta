// denno CUDA: prefill の GDN (ssm) スキャンを「状態レジスタ常駐 + 4 レーン分割」で
// 回すカーネル (task 173)。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf`、この target だけ arch を上げる):
//
//   nvcc -ptx -arch=compute_90 -O3 -fmad=false kernels_gdn_pf.cu -o kernels_gdn_pf.ptx
//
// なぜ別カーネルか:
//   既存の `dno_gdn_scan_sig` (gdn.cu) はスレッド 1 本が「状態の列 1 本
//   (dk = 128 float = 1 KiB)」をローカルメモリの配列に持ち、トークンごとに
//   4 パスで読み書きする。1 ブロック = dk スレッド = 128 スレッドしかなく、
//   作業集合 128 KiB が L1 に収まらないため毎トークンの往復が L2 へ落ちる
//   (lab t169b: ssm 1 層 1 チャンク 512 トークンで 10.7 us/token)。
//
//   このカーネルは同じ演算を、状態を **レジスタ** に置いて回す:
//     - 1 ブロック = 1 v-head、512 スレッド = 128 列 × 4 レーン
//     - レーンは dk を 4 分割し、自分の 32 行ぶんの状態をレジスタに持つ
//     - 列方向の縮約 (q/k の L2 ノルム・skv・o・出力 RMS) は 4 レーンの
//       部分和 + __shfl_xor_sync で作る
//     - レジスタ配列はローカルメモリへ落ちない (全ループを #pragma unroll で
//       静的に展開する。添字が定数になるため)
//
// 数値: 要素ごとの更新 (S *= eg / S += k*dlt) は既存カーネルと同一の丸め。
// 変わるのは **縮約の順序だけ** (1 本の昇順逐次和 → 4 本の部分和の結合)。
// よって値はビット一致しない (丸め順の差のみで、演算の意味は同じ)。
//
// 前提 (ホスト側 `gdn_pf_fast_eligible` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == PF_DK (128) / d_conv == 4 (cs == 3)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
// 形が違うときは既存カーネル (`dno_gdn_scan_sig`) へ fail closed する。

// 1 ヘッドの次元 (k/v 共通)・レーン数・1 レーンが持つ行数・スレッド数。
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define PF_DK 128
#define PF_LANES 4
#define PF_NR 32
#define PF_THREADS 512
#define PF_DCONV 4
#define PF_CS 3

extern "C" __global__ void __launch_bounds__(PF_THREADS, 1)
dno_gdn_pf_scan128(
    const float* __restrict__ qkv,        // [n, conv_dim] (conv 入力 = 投影済み qkv 行)
    const float* __restrict__ z,          // [n, vd]
    const float* __restrict__ beta,       // [n, nv]
    const float* __restrict__ alpha,      // [n, nv]
    const float* __restrict__ conv_st_in, // [conv_dim, cs] 初期 conv 状態
    const float* __restrict__ ssm_in,     // [nv, dk, dv] 初期 SSM 状態
    const float* __restrict__ wconv,      // [conv_dim, d_conv]
    const float* __restrict__ wa,         // [nv] (負値)
    const float* __restrict__ wdt,        // [nv]
    const float* __restrict__ wnorm,      // [dv]
    float* __restrict__ y_out,            // [n, vd]
    float* __restrict__ conv_st_out,      // [conv_dim, cs] 最終 conv 状態
    float* __restrict__ ssm_out,          // [nv, dk, dv] 最終 SSM 状態
    int n, int nk, int nv, int conv_dim, float eps, float q_scale)
{
    // q/k/v はトークン内で 1 本ずつ共有する (列ごとに 1 スレッドが書く)。
    // 末尾の 2 語はゲート (eg, beta) の受け渡し。
    __shared__ float sh[4 * PF_DK + 2];
    float* q_sh = sh;
    float* k_sh = sh + PF_DK;
    float* v_sh = sh + 2 * PF_DK;
    float* o_sh = sh + 3 * PF_DK;
    float* g_sh = sh + 4 * PF_DK; // g_sh[0]=eg g_sh[1]=beta

    const int h = blockIdx.x;   // v-head
    const int kh = h % nk;      // q/k はタイル repeat 規約 (dno_gdn_scan と同一)
    const int kd_total = nk * PF_DK;
    const int vd = nv * PF_DK;
    const int lane = threadIdx.x % PF_LANES;
    const int c = threadIdx.x / PF_LANES; // 0..PF_DK-1 (状態の列)
    const int r0 = lane * PF_NR;          // このレーンが持つ行の先頭

    // ---- 状態 (自列の担当 32 行) をレジスタへ ------------------------------
    float S[PF_NR];
#pragma unroll
    for (int i = 0; i < PF_NR; ++i) {
        S[i] = ssm_in[((long long)h * PF_DK + r0 + i) * PF_DK + c];
    }

    // ---- conv 重み + 初期ウィンドウ (担当 3 チャネル) ---------------------
    // レーンごとに同じ値を読む (同一アドレスなのでブロードキャストされる)。
    // ウィンドウの更新は lane 0 だけが行う (他のレーンは読まない)。
    const int qch = kh * PF_DK + c;
    const int kch = kd_total + kh * PF_DK + c;
    const int vch = 2 * kd_total + h * PF_DK + c;
    float wq[PF_DCONV], wk[PF_DCONV], wv[PF_DCONV];
    float inq[PF_CS], ink[PF_CS], inv[PF_CS];
#pragma unroll
    for (int j = 0; j < PF_DCONV; ++j) {
        wq[j] = wconv[qch * PF_DCONV + j];
        wk[j] = wconv[kch * PF_DCONV + j];
        wv[j] = wconv[vch * PF_DCONV + j];
    }
#pragma unroll
    for (int j = 0; j < PF_CS; ++j) {
        inq[j] = conv_st_in[qch * PF_CS + j];
        ink[j] = conv_st_in[kch * PF_CS + j];
        inv[j] = conv_st_in[vch * PF_CS + j];
    }

    for (int t = 0; t < n; ++t) {
        // 前トークンの共有メモリ読み取りが残っていないか保護
        __syncthreads();

        // ---- 1. causal depthwise conv + SiLU (lane 0 が担当 3 チャネル) ----
        if (lane == 0) {
            const long long row = (long long)t * conv_dim;
            const float xq = qkv[row + qch];
            const float xk = qkv[row + kch];
            const float xv = qkv[row + vch];
            const float accq = wq[0] * inq[0] + wq[1] * inq[1] + wq[2] * inq[2] + wq[3] * xq;
            const float acck = wk[0] * ink[0] + wk[1] * ink[1] + wk[2] * ink[2] + wk[3] * xk;
            const float accv = wv[0] * inv[0] + wv[1] * inv[1] + wv[2] * inv[2] + wv[3] * xv;
            inq[0] = inq[1]; inq[1] = inq[2]; inq[2] = xq;
            ink[0] = ink[1]; ink[1] = ink[2]; ink[2] = xk;
            inv[0] = inv[1]; inv[1] = inv[2]; inv[2] = xv;
            q_sh[c] = accq / (1.0f + expf(-accq)); // silu
            k_sh[c] = acck / (1.0f + expf(-acck));
            v_sh[c] = accv / (1.0f + expf(-accv));
        }
        // ---- 2. ゲート (eg / beta) はブロックで 1 回だけ計算して共有 ------
        if (threadIdx.x == 0) {
            const float a = alpha[(long long)t * nv + h] + wdt[h];
            const float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            g_sh[0] = expf(wa[h] * sp);
            const float bt = beta[(long long)t * nv + h];
            g_sh[1] = 1.0f / (1.0f + expf(-bt));
        }
        __syncthreads();

        // ---- 3. q/k の L2 ノルム (4 レーンの部分和) -----------------------
        float sq = 0.0f;
        float sk2 = 0.0f;
#pragma unroll
        for (int i = 0; i < PF_NR; ++i) {
            const float qv = q_sh[r0 + i];
            sq += qv * qv;
            const float kv = k_sh[r0 + i];
            sk2 += kv * kv;
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 1);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 2);
        const float q_inv = 1.0f / sqrtf(sq + eps);
        const float k_inv = 1.0f / sqrtf(sk2 + eps);
        const float qs = q_inv * q_scale;

        // ---- 4. デルタルール (要素は列内で完結・縮約だけレーン結合) -------
        const float eg = g_sh[0];
        const float b = g_sh[1];
        float skv = 0.0f; // sk[c] = Σ_r k[r] * S[r][c]
#pragma unroll
        for (int i = 0; i < PF_NR; ++i) {
            S[i] *= eg; // S ← S * exp(g)
            skv += (k_sh[r0 + i] * k_inv) * S[i];
        }
        skv += __shfl_xor_sync(0xffffffffu, skv, 1);
        skv += __shfl_xor_sync(0xffffffffu, skv, 2);
        const float dlt = b * (v_sh[c] - skv); // d[c] = beta * (v[c] - sk[c])
        float o = 0.0f;                        // o[c] = Σ_r q[r] * S[r][c]
#pragma unroll
        for (int i = 0; i < PF_NR; ++i) {
            S[i] += (k_sh[r0 + i] * k_inv) * dlt; // S[r][c] += k[r] * d[c]
            o += (q_sh[r0 + i] * qs) * S[i];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (lane == 0) {
            o_sh[c] = o;
        }
        __syncthreads();

        // ---- 5. gated RMS norm: rmsnorm(o) * w * sigmoid(z) ----------------
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < PF_NR; ++i) {
            const float ov = o_sh[r0 + i];
            ss += ov * ov;
        }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        if (lane == 0) {
            const float rms = 1.0f / sqrtf(ss / (float)PF_DK + eps);
            const long long zo = (long long)t * vd + h * PF_DK + c;
            const float zv = z[zo];
            y_out[zo] = o_sh[c] * rms * wnorm[c] * (1.0f / (1.0f + expf(-zv)));
        }
    }

    // ---- 最終状態の書き戻し -----------------------------------------------
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < PF_CS; ++j) {
            conv_st_out[qch * PF_CS + j] = inq[j];
            conv_st_out[kch * PF_CS + j] = ink[j];
            conv_st_out[vch * PF_CS + j] = inv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < PF_NR; ++i) {
        ssm_out[((long long)h * PF_DK + r0 + i) * PF_DK + c] = S[i];
    }
}
