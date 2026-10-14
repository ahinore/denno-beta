// denno-beta P19-A1: Gated DeltaNet 逐次 AR スキャンカーネル (prefill バッチ用)。
//
// 参照: llama.cpp delta-net-base.cpp の数学 (MIT License —
// Copyright (c) 2023-2026 The ggml authors)。denno の GDN 経路は bit 一致不要で、
// 契約は「同一入力→同一出力 + 決定論」のみ。以下により決定論を保証する:
// - launch 構成固定: grid = n_v_heads (v-head ごと 1 ブロック)、block = head_k_dim
// - すべての逐次和は固定順序 (dk/dv の昇順)・単一スレッドまたはスレッド内完結
// - スレッド間の非決定要素 (atomic / ワープ縮約) は不使用
//
// 設計: スレッド c が S[h] の c 列をローカルメモリに保持し、デeltaルールの全更新
// (decay / sk / 外積更新 / 出力) を列ごとに完結させる → スレッド間 reduction 不要。
// conv は depthwise なので各スレッドが担当チャネル (q/k/v の 3 チャネル) の
// ウィンドウをレジスタでロールする。q/k の L2 正規化と出力 RMS のみ
// block 内単一スレッドの逐次和 (CPU と同一順序) を用いる。
//
// 前提 (ホスト側で assert): head_k_dim == head_v_dim、d_conv == 4、head_k_dim <= 256
// (Qwen3.5 0.8B/9B/27B はいずれも dk=dv=128, d_conv=4)。

extern "C" __global__ void dno_gdn_scan(
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
    const int kd_total = nk * dk;   // q/k チャネル総数
    const int vd = nv * dv;
    const int c = threadIdx.x;      // 0..dk (== dv)

    extern __shared__ float sh[];
    // layout: q_sh[dk] | k_sh[dk] | v_sh[dk] | o_sh[dv] | scalars[5]
    float* q_sh = sh;
    float* k_sh = sh + dk;
    float* v_sh = sh + 2 * dk;
    float* o_sh = sh + 3 * dk;
    float* sc = sh + 3 * dk + dv; // sc[0]=q_inv sc[1]=k_inv sc[2]=eg sc[3]=beta sc[4]=o_inv

    // ---- S 列 (ローカルメモリ) ------------------------------------------
    float Scol[256];
    for (int i = 0; i < dk; ++i) {
        Scol[i] = ssm_in[((long long)h * dk + i) * dv + c];
    }

    // ---- conv 重み + 初期ウィンドウ (担当 3 チャネル) ---------------------
    const int qch = kh * dk + c;
    const int kch = kd_total + kh * dk + c;
    const int vch = 2 * kd_total + h * dv + c;
    float wq[4], wk[4], wv[4];
    for (int j = 0; j < d_conv; ++j) {
        wq[j] = wconv[qch * d_conv + j];
        wk[j] = wconv[kch * d_conv + j];
        wv[j] = wconv[vch * d_conv + j];
    }
    float inq[3], ink[3], inv[3]; // 直前 cs 入力のウィンドウ
    for (int j = 0; j < cs; ++j) {
        inq[j] = conv_st_in[qch * cs + j];
        ink[j] = conv_st_in[kch * cs + j];
        inv[j] = conv_st_in[vch * cs + j];
    }

    for (int t = 0; t < n; ++t) {
        // 前トークンの共有メモリ読み取りが残っていないか保護
        __syncthreads();

        // ---- 1. causal depthwise conv + SiLU (担当 3 チャネル) ------------
        const long long row = (long long)t * conv_dim;
        const float xq = qkv[row + qch];
        const float xk = qkv[row + kch];
        const float xv = qkv[row + vch];
        float accq = wq[0] * inq[0] + wq[1] * inq[1] + wq[2] * inq[2] + wq[3] * xq;
        float acck = wk[0] * ink[0] + wk[1] * ink[1] + wk[2] * ink[2] + wk[3] * xk;
        float accv = wv[0] * inv[0] + wv[1] * inv[1] + wv[2] * inv[2] + wv[3] * xv;
        // ウィンドウをロール (CPU と同一: 新 state = 直近 d_conv-1 入力)
        inq[0] = inq[1]; inq[1] = inq[2]; inq[2] = xq;
        ink[0] = ink[1]; ink[1] = ink[2]; ink[2] = xk;
        inv[0] = inv[1]; inv[1] = inv[2]; inv[2] = xv;
        q_sh[c] = accq / (1.0f + expf(-accq)); // silu
        k_sh[c] = acck / (1.0f + expf(-acck));
        v_sh[c] = accv / (1.0f + expf(-accv));
        __syncthreads();

        // ---- 2. q/k L2 正規化スケール + decay/beta (単一スレッド逐次和) ----
        if (c == 0) {
            float sq = 0.0f, sk2 = 0.0f;
            for (int i = 0; i < dk; ++i) {
                sq += q_sh[i] * q_sh[i];
                sk2 += k_sh[i] * k_sh[i];
            }
            sc[0] = 1.0f / sqrtf(sq + eps);
            sc[1] = 1.0f / sqrtf(sk2 + eps);
            // g = ssm_a * softplus(alpha + ssm_dt)  (softplus: ggml 規約)
            const float a = alpha[(long long)t * nv + h] + wdt[h];
            const float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            sc[2] = expf(wa[h] * sp);
            const float bt = beta[(long long)t * nv + h];
            sc[3] = 1.0f / (1.0f + expf(-bt));
        }
        __syncthreads();

        // ---- 3. q スケール + k 正規化 (自要素) -----------------------------
        q_sh[c] *= sc[0] * q_scale;
        k_sh[c] *= sc[1];
        __syncthreads();

        // ---- 4. デルタルール (列ごとに完結・reduction 無し) ----------------
        const float eg = sc[2];
        const float b = sc[3];
        for (int i = 0; i < dk; ++i) {
            Scol[i] *= eg; // S ← S * exp(g)
        }
        float skv = 0.0f; // sk[c] = Σ_r k[r] * S[r][c]
        for (int i = 0; i < dk; ++i) {
            skv += k_sh[i] * Scol[i];
        }
        const float dlt = b * (v_sh[c] - skv); // d[c] = beta * (v[c] - sk[c])
        for (int i = 0; i < dk; ++i) {
            Scol[i] += k_sh[i] * dlt; // S[r][c] += k[r] * d[c]
        }
        float o = 0.0f; // o[c] = Σ_r q[r] * S[r][c]
        for (int i = 0; i < dk; ++i) {
            o += q_sh[i] * Scol[i];
        }

        // ---- 5. gated RMS norm: rmsnorm(o) * w * silu(z) -------------------
        o_sh[c] = o;
        __syncthreads();
        if (c == 0) {
            float s = 0.0f;
            for (int i = 0; i < dv; ++i) {
                s += o_sh[i] * o_sh[i];
            }
            sc[4] = 1.0f / sqrtf(s / (float)dv + eps);
        }
        __syncthreads();
        const long long zo = (long long)t * vd + h * dv + c;
        const float zv = z[zo];
        y_out[zo] = o_sh[c] * sc[4] * wnorm[c] * (zv / (1.0f + expf(-zv)));
    }

    // ---- 最終状態の書き戻し ------------------------------------------------
    for (int j = 0; j < cs; ++j) {
        conv_st_out[qch * cs + j] = inq[j];
        conv_st_out[kch * cs + j] = ink[j];
        conv_st_out[vch * cs + j] = inv[j];
    }
    for (int i = 0; i < dk; ++i) {
        ssm_out[((long long)h * dk + i) * dv + c] = Scol[i];
    }
}
