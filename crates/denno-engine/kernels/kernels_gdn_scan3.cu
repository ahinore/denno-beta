// task 592: decode の GDN (ssm) スキャンの第 3 版。第 2 版
// (`kernels_gdn_scan2.cu` / `dno_gdn_scan_sig2`) のレーン数を 4 -> 8 にしただけ。
//
// 背景 (REPORT_P588 §4): decode の scan は 1 ヘッド = 1 ブロック (grid = nv = 48)
// で、第 2 版は 512 スレッド = 16 warp/ブロック。5090 は 170 SM あるので
// **48 ブロックでは 28% の SM しか埋まらず、1 SM あたり 4.5 warp しか立たない**
// (状態 3.1 MB の読み書きが 590 GB/s でしか流れない)。ヘッド数 (= ブロック数) は
// モデルで固定なので、ブロックを増やさずに warp を増やすには 1 列を分担する
// レーン数を増やすしかない。列 (dv) を割ると step 2/5 の全和がブロックを跨ぐので
// 2 パスが要る (P588 §4)。**レーンを 8 にすると 1024 スレッド = 32 warp/ブロック、
// 48 ブロックで 1536 warp = 9 warp/SM (第 2 版の 2 倍) になり、パスを増やさずに
// メモリレベルの並列度を稼げる。**
//
// 数値: レーンを増やすと skv / o の縮約の木が変わる (第 2 版は 32 行ずつ 4 レーン、
// こちらは 16 行ずつ 8 レーン)。**要素ごとの演算 (decay の乗算、update の k*dlt
// 加算、conv の左結合、q/k の L2 正規化の逐次和、出力の rmsnorm * wnorm * sigmoid(z))
// は第 2 版と 1 演算 1 丸めで同一**で、変わるのは和の結合順だけ (固定木 = 決定的)。
// よって md5 は DIFF しうる (意図した丸めの変更)。
//
// 前提 (ホスト側で保証する): dk == dv == 128 / d_conv == 4 / cs == 3。
// それ以外は従来カーネル (`kernels_gdn.ptx` の dno_gdn_scan_sig) を使う。
//
// `__launch_bounds__(1024, 1)` は必須 (lab t592a で外したら cuLaunchKernel が
// error 701 = LAUNCH_OUT_OF_RESOURCES で落ちた): 1024 スレッドのブロックは
// 1 SM あたり 1 ブロックなので **1 スレッド 64 レジスタが上限**で、これを
// 宣言しないと ptxas は 64 を超える割り当てをして起動できない。S (16) は
// ホットなのでレジスタに残り、溢れるのは lane 0 だけが使う conv の窓と
// ポインタ引数 (1 トークン 1 回) になる。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_gdn_scan3.cu -o kernels_gdn_scan3.ptx
// まとめて作るなら `kernels/build_ptx.sh gdn_scan3`。

#define DNO_S3_DK 128
#define DNO_S3_NS 16   // 1 レーンが持つ列の要素数 (dk / 8)

// 同じ列の 8 レーン (幅 8) の合計。lane 0 の値だけが使われる。
// 加算順は固定 (offset 1 -> 2 -> 4)。
__device__ __forceinline__ float dno_s3_octet_sum(float v)
{
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 1, 8));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 2, 8));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 4, 8));
    return v;
}

extern "C" __global__ void __launch_bounds__(1024, 1) dno_gdn_scan_sig3(
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
    const int t = threadIdx.x;      // 0 .. dv*8-1
    const int c = t >> 3;           // 列 (0..dv-1)
    const int lane = t & 7;         // 0..7 (同じ列の 8 レーン)

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
    float S[DNO_S3_NS];
#pragma unroll
    for (int i = 0; i < DNO_S3_NS; ++i) {
        S[i] = ssm_in[((long long)h * dk + lane * DNO_S3_NS + i) * dv + c];
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

        // ---- 4. デルタルール (列ごとに完結・8 レーンで分担) ----------------
        const float eg = sc[2];
        const float b = sc[3];
        float part = 0.0f; // sk[c] = Σ_r k[r] * S[r][c] のレーン分担
#pragma unroll
        for (int i = 0; i < DNO_S3_NS; ++i) {
            const int r = lane * DNO_S3_NS + i;
            S[i] = __fmul_rn(S[i], eg); // S ← S * exp(g)
            part = __fadd_rn(part, __fmul_rn(k_sh[r], S[i]));
        }
        part = dno_s3_octet_sum(part);
        float dlt = 0.0f;
        if (lane == 0) {
            dlt = __fmul_rn(b, __fsub_rn(v_sh[c], part)); // d[c] = beta * (v[c] - sk[c])
        }
        dlt = __shfl_sync(0xffffffffu, dlt, 0, 8);
        float o = 0.0f; // o[c] = Σ_r q[r] * S[r][c] のレーン分担
#pragma unroll
        for (int i = 0; i < DNO_S3_NS; ++i) {
            const int r = lane * DNO_S3_NS + i;
            S[i] = __fadd_rn(S[i], __fmul_rn(k_sh[r], dlt));
            o = __fadd_rn(o, __fmul_rn(q_sh[r], S[i]));
        }
        o = dno_s3_octet_sum(o);
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
    for (int i = 0; i < DNO_S3_NS; ++i) {
        const int r = lane * DNO_S3_NS + i;
        ssm_out[((long long)h * dk + r) * dv + c] = S[i];
    }
}
