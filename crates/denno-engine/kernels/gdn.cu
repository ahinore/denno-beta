// denno-beta P19-A1: Gated DeltaNet 逐次 AR スキャンカーネル (prefill バッチ用)。
//
// 参照: llama.cpp delta-net-base.cpp の数学 (MIT License —
// Copyright (c) 2023-2026 The ggml authors)。denno の GDN 経路は bit 一致不要で、
// 契約は「同一入力→同一出力 + 決定論」のみ。以下により決定論を保証する:
// - launch 構成固定: grid = n_v_heads (v-head ごと 1 ブロック)、block = head_k_dim
// - すべての逐次和は固定順序 (dk/dv の昇順)・単一スレッドまたはスレッド内完結
// - スレッド間の非決定要素 (atomic / ワープ縮約) は不使用
//
// s89 (P1-1): スキャン状態をホストへ往復させない経路を追加した (2 launch)。
//   ・dno_gdn_chunk_scan_hoist: 状態をグラフメモリに常駐させて chunk を跨ぐ。
//     2 変種は dno_gdn_step を共有する (演算列 1 本)。
//   ・(パス削減) pass 2 の Sinit 直列伝播を chunk 0 ブロックへ折りたたむ。
//   いずれも S の列を「i 昇順の逐次和」で扱う点は不変。
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
    // q/k はタイル repeat 規約 (GGUF / llama.cpp ggml_repeat / fused カーネル
    // `iv1 % neq1` = strided): v-head h → k/q head h % nk。0.8B は nk==nv で同一・
    // 9B は nk=16/nv=32・27B は nk=16/nv=48。HF transformers の repeat_interleave
    // (blocked) とは異なる規約 (第15/16セッション lchk サブステージ比較で確定)。
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

// ---------------------------------------------------------------------------
// A1-r2: GDN chunked scan 3 カーネル (長 prefill 用・3-pass 構成)
// ---------------------------------------------------------------------------
//
// CPU 参照実装: crates/denno-engine/src/gdn.rs の scan_chunked_core。
//   pass 1 (chunk 並列): dno_gdn_chunk_pre — 各 chunk のローカル遷移
//     W[ci] = S=0 起点ローカル AR スキャンの終状態・
//     M[ci] = 恒等行列から列ごとに伝播した遷移行列
//   pass 2 (chunk 直列): dno_gdn_chunk_state — chunk 先頭状態 Sinit の直列伝播
//     Sinit[0] = S_in・Sinit[ci+1] = M[ci]·Sinit[ci] + W[ci]
//   pass 3 (chunk 並列): dno_gdn_chunk_scan — Sinit[ci] から chunk 内を逐次 AR
//     スキャン (y 出力・最終 chunk ブロックが最終状態を書き戻す)
//
// 決定論 (既存 dno_gdn_scan と同一方針):
// - launch 構成固定: pass 1/3 は grid=(n_chunks, nv)・block=dk、pass 2 は
//   grid=nv・block=dk。スレッド c が S/W/M の c 列をローカルメモリに保持し
//   列ごとに完結する
// - すべての逐次和は固定順序 (昇順)・単一スレッドまたはスレッド内完結
// - スレッド間の非決定要素 (atomic / ワープ縮約) は不使用
//
// 前提 (ホスト側で assert): head_k_dim == head_v_dim、d_conv == 4 (cs == 3)、
// head_k_dim <= 256。chunk は chunk_len で均等分割 (最終 chunk は端数切り詰め・
// 全 chunk 非空)。n_chunks == 1 のとき pass 3 は dno_gdn_scan と同一演算列
// (Sinit[0] = S_in) になる。

// conv 入力 (時刻 τ = t0 - cs + back)。τ < 0 は初期 conv 状態 conv_st。
// CPU の conv_input_at (gdn.rs) と同一規約。
__device__ __forceinline__ float dno_gdn_conv_input(
    const float* __restrict__ qkv, const float* __restrict__ conv_st,
    int ch, int t0, int back, int conv_dim, int cs)
{
    const int idx = t0 - cs + back;
    if (idx >= 0) {
        return qkv[(long long)idx * conv_dim + ch];
    }
    return conv_st[ch * cs + (idx + cs)];
}

// pass 1: 各 chunk のローカル遷移 W[ci] ([dk][dv]) と M[ci] ([dk][dk]) を
// chunk 並列で計算する。スレッド c は列 c を担当。フロントエンド (k/v の
// conv+SiLU・k の L2 正規化・ゲート) をブロック内で先出し計算する (q は不要)。
// 演算順序は gdn.rs の w_step_col / m_step_col と同一。
extern "C" __global__ void dno_gdn_chunk_pre(
    const float* __restrict__ qkv,        // [n, conv_dim]
    const float* __restrict__ beta,       // [n, nv]
    const float* __restrict__ alpha,      // [n, nv]
    const float* __restrict__ conv_st_in, // [conv_dim, cs] 初期 conv 状態
    const float* __restrict__ wconv,      // [conv_dim, d_conv]
    const float* __restrict__ wa,         // [nv] (負値)
    const float* __restrict__ wdt,        // [nv]
    float* __restrict__ w_out,            // [n_chunks, nv, dk, dv]
    float* __restrict__ m_out,            // [n_chunks, nv, dk, dk]
    int n, int chunk_len, int nk, int nv, int dk, int dv, int conv_dim, int cs,
    float eps)
{
    const int ci = blockIdx.x; // chunk
    const int h = blockIdx.y;  // v-head
    const int kh = h % nk;     // タイル repeat 規約 (gdn.rs と同一・llama.cpp 相当)
    const int kd_total = nk * dk;
    const int c = threadIdx.x; // 0..dk (== dv)
    const int t0 = ci * chunk_len;
    const int t1 = min(n, t0 + chunk_len);

    extern __shared__ float sh[];
    float* k_sh = sh;        // [dk]
    float* v_sh = sh + dk;   // [dk]
    float* sc = sh + 2 * dk; // sc[0]=k_inv sc[1]=eg sc[2]=beta

    // ---- 列ローカル: W 列 (S=0 起点スキャン) と M 列 (恒等行列) -------------
    float Wcol[256];
    float Mcol[256];
    for (int i = 0; i < dk; ++i) {
        Wcol[i] = 0.0f;
        Mcol[i] = (i == c) ? 1.0f : 0.0f;
    }

    // ---- conv 重み + chunk 開始時点のウィンドウ (k/v の担当チャネル) --------
    const int kch = kd_total + kh * dk + c;
    const int vch = 2 * kd_total + h * dv + c;
    float wk[4], wv[4];
    for (int j = 0; j < 4; ++j) {
        wk[j] = wconv[kch * 4 + j];
        wv[j] = wconv[vch * 4 + j];
    }
    float ink[3], inv[3];
    for (int j = 0; j < cs; ++j) {
        ink[j] = dno_gdn_conv_input(qkv, conv_st_in, kch, t0, j, conv_dim, cs);
        inv[j] = dno_gdn_conv_input(qkv, conv_st_in, vch, t0, j, conv_dim, cs);
    }

    for (int t = t0; t < t1; ++t) {
        __syncthreads();
        const long long row = (long long)t * conv_dim;
        const float xk = qkv[row + kch];
        const float xv = qkv[row + vch];
        const float acck = wk[0] * ink[0] + wk[1] * ink[1] + wk[2] * ink[2] + wk[3] * xk;
        const float accv = wv[0] * inv[0] + wv[1] * inv[1] + wv[2] * inv[2] + wv[3] * xv;
        ink[0] = ink[1]; ink[1] = ink[2]; ink[2] = xk;
        inv[0] = inv[1]; inv[1] = inv[2]; inv[2] = xv;
        k_sh[c] = acck / (1.0f + expf(-acck)); // silu
        v_sh[c] = accv / (1.0f + expf(-accv));
        __syncthreads();

        // ---- k L2 正規化スケール + ゲート (単一スレッド逐次和) --------------
        if (c == 0) {
            float s = 0.0f;
            for (int i = 0; i < dk; ++i) {
                s += k_sh[i] * k_sh[i];
            }
            sc[0] = 1.0f / sqrtf(s + eps);
            const float a = alpha[(long long)t * nv + h] + wdt[h];
            const float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            sc[1] = expf(wa[h] * sp);
            const float bt = beta[(long long)t * nv + h];
            sc[2] = 1.0f / (1.0f + expf(-bt));
        }
        __syncthreads();
        k_sh[c] *= sc[0];
        __syncthreads();

        // ---- W ステップ (列 c): decay → 内積 → 更新 (w_step_col と同一順序) --
        const float eg = sc[1];
        const float b = sc[2];
        for (int i = 0; i < dk; ++i) {
            Wcol[i] *= eg;
        }
        float skv = 0.0f;
        for (int i = 0; i < dk; ++i) {
            skv += k_sh[i] * Wcol[i];
        }
        const float dlt = b * (v_sh[c] - skv);
        for (int i = 0; i < dk; ++i) {
            Wcol[i] += k_sh[i] * dlt;
        }
        // ---- M ステップ (列 c): decay → 内積 → 更新 (m_step_col と同一順序) --
        for (int i = 0; i < dk; ++i) {
            Mcol[i] *= eg;
        }
        float mk = 0.0f;
        for (int i = 0; i < dk; ++i) {
            mk += k_sh[i] * Mcol[i];
        }
        for (int i = 0; i < dk; ++i) {
            Mcol[i] -= b * k_sh[i] * mk;
        }
    }

    // ---- 書き戻し ([ci][h][r][c] 行優先・スレッド c が列 c) -----------------
    const long long base = ((long long)(ci * nv + h)) * dk;
    for (int r = 0; r < dk; ++r) {
        w_out[(base + r) * dv + c] = Wcol[r];
        m_out[(base + r) * dk + c] = Mcol[r];
    }
}

// pass 2: chunk 先頭状態 Sinit の直列伝播 (grid=nv・block=dk、chunk ループは
// ブロック内直列)。スレッド c が S の c 列を保持: Scol ← M[ci]·Scol + W[ci]
// (j 昇順の内積 — gdn.rs pass 2 と同一順序)。Sinit[0] = S_in。
extern "C" __global__ void dno_gdn_chunk_state(
    const float* __restrict__ ssm_in, // [nv, dk, dv]
    const float* __restrict__ m_in,   // [n_chunks, nv, dk, dk]
    const float* __restrict__ w_in,   // [n_chunks, nv, dk, dv]
    float* __restrict__ sinit,        // [n_chunks, nv, dk, dv]
    int n_chunks, int nv, int dk, int dv)
{
    const int h = blockIdx.x;
    const int c = threadIdx.x; // 列 c を担当

    float Scol[256];
    for (int r = 0; r < dk; ++r) {
        Scol[r] = ssm_in[((long long)h * dk + r) * dv + c];
    }
    for (int ci = 0; ci < n_chunks; ++ci) {
        const long long base = ((long long)(ci * nv + h)) * dk;
        // Sinit[ci] を書き出し (ci==0 は S_in そのもの)
        for (int r = 0; r < dk; ++r) {
            sinit[(base + r) * dv + c] = Scol[r];
        }
        if (ci + 1 >= n_chunks) {
            break;
        }
        // Scol ← M[ci]·Scol + W[ci] (j 昇順)
        float next[256];
        for (int r = 0; r < dk; ++r) {
            float acc = 0.0f;
            for (int j = 0; j < dk; ++j) {
                acc += m_in[(base + r) * dk + j] * Scol[j];
            }
            next[r] = acc + w_in[(base + r) * dv + c];
        }
        for (int r = 0; r < dk; ++r) {
            Scol[r] = next[r];
        }
    }
}

// pass 3: Sinit[ci] から chunk 内を逐次 AR スキャン (chunk 並列)。トークン
// ループ本体は dno_gdn_scan と同一演算列 (conv ウィンドウ初期化のみ chunk
// 開始時点への巻き戻しに変更 — 値は連続 AR のロール結果と一致)。y は
// [n, value_dim] の担当範囲を各ブロックが排他的に書く。最終状態は最終 chunk
// ブロック (ci == n_chunks-1) のみ書き戻す。
//
// s89 (P1-1): 本体の後段 (デルタルール → gated RMS norm) は dno_gdn_step に
// 取り出し、`dno_gdn_chunk_scan` (レジスタ列) と `dno_gdn_chunk_scan_hoist`
// (グラフ常駐列) が共有する。このカーネルは従来と同一の演算列・同一の
// レジスタ構成 (Scol[256]) を保つ。
extern "C" __global__ void dno_gdn_chunk_scan(
    const float* __restrict__ qkv,   // [n, conv_dim]
    const float* __restrict__ z,     // [n, value_dim]
    const float* __restrict__ beta,  // [n, nv]
    const float* __restrict__ alpha, // [n, nv]
    const float* __restrict__ conv_st_in, // [conv_dim, cs] 初期 conv 状態
    const float* __restrict__ sinit, // [n_chunks, nv, dk, dv]
    const float* __restrict__ wconv, // [conv_dim, d_conv]
    const float* __restrict__ wa,    // [nv] (負値)
    const float* __restrict__ wdt,   // [nv]
    const float* __restrict__ wnorm, // [dv]
    float* __restrict__ y_out,       // [n, value_dim]
    float* __restrict__ conv_st_out, // [conv_dim, cs] 最終 conv 状態
    float* __restrict__ ssm_out,     // [nv, dk, dv] 最終 SSM 状態
    int n, int chunk_len, int nk, int nv, int dk, int dv, int conv_dim, int cs,
    float eps, float q_scale)
{
    const int ci = blockIdx.x; // chunk
    const int h = blockIdx.y;  // v-head
    const int kh = h % nk;     // タイル repeat 規約 (gdn.rs と同一・llama.cpp 相当)
    const int kd_total = nk * dk;
    const int vd = nv * dv;
    const int c = threadIdx.x; // 0..dk (== dv)
    const int t0 = ci * chunk_len;
    const int t1 = min(n, t0 + chunk_len);
    const int is_last = (ci == gridDim.x - 1);

    extern __shared__ float sh[];
    // layout: q_sh[dk] | k_sh[dk] | v_sh[dk] | o_sh[dv] | scalars[5]
    float* q_sh = sh;
    float* k_sh = sh + dk;
    float* v_sh = sh + 2 * dk;
    float* o_sh = sh + 3 * dk;
    float* sc = sh + 3 * dk + dv; // sc[0]=q_inv sc[1]=k_inv sc[2]=eg sc[3]=beta sc[4]=o_inv

    // ---- S 列 = Sinit[ci] の c 列 ------------------------------------------
    float Scol[256];
    const long long sbase = ((long long)(ci * nv + h)) * dk;
    for (int i = 0; i < dk; ++i) {
        Scol[i] = sinit[(sbase + i) * dv + c];
    }

    // ---- conv 重み + chunk 開始時点のウィンドウ (担当 3 チャネル) -----------
    const int qch = kh * dk + c;
    const int kch = kd_total + kh * dk + c;
    const int vch = 2 * kd_total + h * dv + c;
    float wq[4], wk[4], wv[4];
    for (int j = 0; j < 4; ++j) {
        wq[j] = wconv[qch * 4 + j];
        wk[j] = wconv[kch * 4 + j];
        wv[j] = wconv[vch * 4 + j];
    }
    float inq[3], ink[3], inv[3];
    for (int j = 0; j < cs; ++j) {
        inq[j] = dno_gdn_conv_input(qkv, conv_st_in, qch, t0, j, conv_dim, cs);
        ink[j] = dno_gdn_conv_input(qkv, conv_st_in, kch, t0, j, conv_dim, cs);
        inv[j] = dno_gdn_conv_input(qkv, conv_st_in, vch, t0, j, conv_dim, cs);
    }

    for (int t = t0; t < t1; ++t) {
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

        // ---- 4/5. デルタルール + gated RMS norm (dno_gdn_step と共有) -------
        // s89 (P1-1): 後段は 2 変種で共有する (演算列を 1 本に保つ)。
        dno_gdn_step(q_sh, k_sh, v_sh, o_sh, sc, nullptr, Scol, dk, dv, eps);
        const long long zo = (long long)t * vd + h * dv + c;
        const float zv = z[zo];
        y_out[zo] = o_sh[c] * sc[4] * wnorm[c] * (zv / (1.0f + expf(-zv)));
    }

    // ---- 最終状態の書き戻し (最終 chunk ブロックのみ) -----------------------
    if (is_last) {
        for (int j = 0; j < cs; ++j) {
            conv_st_out[qch * cs + j] = inq[j];
            conv_st_out[kch * cs + j] = ink[j];
            conv_st_out[vch * cs + j] = inv[j];
        }
        for (int i = 0; i < dk; ++i) {
            ssm_out[((long long)h * dk + i) * dv + c] = Scol[i];
        }
    }
}

// ---------------------------------------------------------------------------
// s89 (P1-1): デルタルール 1 ステップ + gated RMS norm (両変種で共有)
//
// chunked の 2 変種 (dno_gdn_chunk_scan = レジスタ列 / dno_gdn_chunk_scan_hoist =
// グラフ常駐列) が同じ演算列を通るように 1 本にまとめた関数。S 列の置き場所だけ
// が違う:
//   - Sreg != nullptr: スレッドのレジスタ配列 (i 昇順)。
//   - Sreg == nullptr:  `s_col` を起点とする [dk] の列。r 行目は s_col[r*dv]。
//     ここでは i 昇順の列走査のまま、減衰 (S *= eg) と外積 (S += k[i]*dlt) を
//     i ループの外へ出す — 各 i に掛かる係数は同じなので積と和の値は不変。
//
// skv / dlt / o の和はどれも i 昇順の逐次和 (決定論)。スレッド間 reduction 無し。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dno_gdn_step(
    const float* __restrict__ q_sh, const float* __restrict__ k_sh,
    const float* __restrict__ v_sh, float* __restrict__ o_sh,
    float* __restrict__ sc, // sc[2]=eg sc[3]=beta (読み) sc[4]=o_inv (書き)
    float* __restrict__ s_col, // 列の先頭 (Sreg==nullptr のとき)・r 行目は s_col[r*dv]
    float* Sreg,               // レジスタ配列 (nullptr なら s_col を使う)
    int dk, int dv, float eps)
{
    const int c = threadIdx.x;
    const float eg = sc[2];
    const float b = sc[3];
    float skv = 0.0f;
    float dlt;
    float o = 0.0f;
    if (Sreg != nullptr) {
        // ---- レジスタ版 (dno_gdn_chunk_scan・従来と同一演算列) ----------------
        for (int i = 0; i < dk; ++i) {
            Sreg[i] *= eg; // S ← S * exp(g)
        }
        for (int i = 0; i < dk; ++i) {
            skv += k_sh[i] * Sreg[i]; // sk[c] = Σ_r k[r] * S[r][c]
        }
        dlt = b * (v_sh[c] - skv); // d[c] = beta * (v[c] - sk[c])
        for (int i = 0; i < dk; ++i) {
            Sreg[i] += k_sh[i] * dlt; // S[r][c] += k[r] * d[c]
        }
        for (int i = 0; i < dk; ++i) {
            o += q_sh[i] * Sreg[i]; // o[c] = Σ_r q[r] * S[r][c]
        }
    } else {
        // ---- グラフ常駐列版 (dno_gdn_chunk_scan_hoist) ------------------------
        float* p = s_col;
        for (int i = 0; i < dk; ++i, p += dv) {
            *p *= eg;
        }
        p = s_col;
        for (int i = 0; i < dk; ++i, p += dv) {
            skv += k_sh[i] * (*p);
        }
        dlt = b * (v_sh[c] - skv);
        p = s_col;
        for (int i = 0; i < dk; ++i, p += dv) {
            *p += k_sh[i] * dlt;
        }
        p = s_col;
        for (int i = 0; i < dk; ++i, p += dv) {
            o += q_sh[i] * (*p);
        }
    }

    // ---- gated RMS norm: 正規化係数だけ単一スレッドの逐次和 (CPU と同一順序) --
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
}

// ---------------------------------------------------------------------------
// s89 (P1-1): スキャン状態のグラフ常駐化 (パス削減 + D2H 削減)
// ---------------------------------------------------------------------------
//
// 動機 (計測): chunked 経路は pass 1 (W/M) → pass 2 (Sinit の直列伝播) →
// pass 3 (逐次 AR) の 3 launch + 3 sync を回し、なおかつ S を往復していた
// (H2D: ssm_in・D2H: ssm_out)。9B の 5,686 tok では n_chunks = 6 なので
// pass 2 は 6 段の逐次伝播で、占有も 32 ブロック×128 スレッドに過ぎない。
//
// ここでは pass 2 を「pass 3 の chunk 0 ブロック」へ折りたたむ:
//   - pass 1 のまま W / M を作る。
//   - 続く 1 launch (dno_gdn_chunk_scan_hoist) で、chunk 0 のブロックだけが
//     ssm_in を S0 にして自己のトークン列をスキャンし、各トークン後の S を
//     グラフメモリ s_rows[t] = [nv, dk, dv] へ書く (層全域で保持)。
//   - chunk ci (>=1) のブロックは、chunk 境界 (t = ci*chunk_len) に相当する
//     行 s_rows[ci*chunk_len - 1] を Sinit として自分の列をスキャンする。
//   → 2 pass・2 sync。S は S0 と最終行 (と最終 conv 状態) 以外ホストへ戻さない。
//
// 決定論 / 数値:
//   - launch 構成 (grid = (n_chunks, nv)・block = dk) と pass 数は固定。
//   - s_rows[ci*chunk_len-1] の値は、pass 3 が Sinit[ci] を経て到達する値と
//     **同一の演算列**で作られる (chunk 0 は S0 = Sinit[0]・推移は同じ式)。
//     したがって chunk 境界の状態は 3-pass 版と 1 対 1。
//   - 列アクセスを i 昇順のまま保ちつつ、減衰 (S *= eg) と外積
//     (S += k[i]*dlt) を i ループの外へ出す — 各 i で掛ける係数は同じなので
//     積と和の値は変わらない (dno_gdn_step の hoist 分岐と同じ演算)。
//
// 書き込みの不変条件 (重要・s90 で訂正):
//   s89 の版は「chunk ci は t >= t0 + chunk_len の行だけを書く」としていたが、
//   これは**境界行を 1 行も書かない** (chunk ci の t1 = t0 + chunk_len なので
//   [t0+chunk_len, t1) は空)。すると chunk ci が読む s_rows[ci*chunk_len - 1] は
//   誰も書かず、全 chunk が古い値 (= 前の launch の行) から走ってしまう。
//   s90 の正しい規則:
//     1. 最終でない chunk ci は **自分の最終行 t1-1 を 1 行だけ**書く。
//        次の chunk ちょうど 1 つがそれを読む。
//     2. 最終 chunk は [t0, n) の全行を書く (s_rows を次に使う呼び出しの
//        ため・行 n-1 の書き手はこれ 1 つだけ)。
//   中間行は誰も読まないので書かない (転送量も増えない)。
//
// 前提 (s90 の結論・重要):
//   この折り畳みは「読む行を別 launch が書いてくれること」を要求する。
//   chunk ci が読む行は自分の最終行より前にあるが、2 launch 経路では
//   その書き手は**同一 launch の兄弟ブロックにならない** (書き手 launch は
//   この launch の直前 = 前の層 or 前の区間)。同一 launch 内で
//   「別ブロックの書いた行を読む」ことは無いので、ブロック間の順序保証は不要。
//   その「別 launch の書き手」が今の forward に存在しないため、このカーネルは
//   ホスト側 (`gdn/scan_schedule.rs` の route_hoist) で現状まったく選ばれない。
//   選択条件が整うまでは **呼ばないこと**。
extern "C" __global__ void dno_gdn_chunk_scan_hoist(
    const float* __restrict__ qkv,   // [n, conv_dim]
    const float* __restrict__ z,     // [n, value_dim]
    const float* __restrict__ beta,  // [n, nv]
    const float* __restrict__ alpha, // [n, nv]
    const float* __restrict__ conv_st_in, // [conv_dim, cs] 初期 conv 状態 (conv のみ)
    const float* __restrict__ ssm_in, // [nv, dk, dv] 初期 SSM 状態 (chunk 0 のみが読む)
    const float* __restrict__ wconv, // [conv_dim, d_conv]
    const float* __restrict__ wa,    // [nv] (負値)
    const float* __restrict__ wdt,   // [nv]
    const float* __restrict__ wnorm, // [dv]
    float* __restrict__ s_rows,      // [n, nv, dk, dv] 入口域: 各終端行と
                                     //   入力域: 前層が書いた境界行
    float* __restrict__ y_out,       // [n, value_dim]
    float* __restrict__ conv_st_out, // [conv_dim, cs] 最終 conv 状態
    float* __restrict__ ssm_out,     // [nv, dk, dv] 最終 SSM 状態
    int n, int chunk_len, int nk, int nv, int dk, int dv, int conv_dim, int cs,
    float eps, float q_scale)
{
    const int ci = blockIdx.x; // chunk
    const int h = blockIdx.y;  // v-head
    const int kh = h % nk;     // タイル repeat 規約 (gdn.rs と同一)
    const int kd_total = nk * dk;
    const int vd = nv * dv;
    const int c = threadIdx.x; // 0..dk (== dv)
    const int t0 = ci * chunk_len;
    const int t1 = min(n, t0 + chunk_len);
    const int is_last = (ci == gridDim.x - 1);

    extern __shared__ float sh[];
    // layout: q_sh[dk] | k_sh[dk] | v_sh[dk] | o_sh[dv] | scalars[5]
    float* q_sh = sh;
    float* k_sh = sh + dk;
    float* v_sh = sh + 2 * dk;
    float* o_sh = sh + 3 * dk;
    float* sc = sh + 3 * dk + dv; // sc[0]=q_inv sc[1]=k_inv sc[2]=eg sc[3]=beta sc[4]=o_inv

    // ---- 初期状態: chunk 0 は ssm_in・それ以外は前層の境界行 -----------------
    // 層内の行 [h][r][c] はスレッド c が書く列なので、読み出しも同じアドレス。
    // 列方向の歩幅は dv、head の歩幅は dk*dv (テンソル [nv, dk, dv] と同一)。
    const long long hbase = (long long)h * dk; // 行 r は (hbase + r) * dv
    float* s_col;
    if (ci == 0) {
        s_col = (float*)ssm_in + hbase * dv + c;
    } else {
        // 直前トークン (= 当該 chunk の 1 つ前) の行を、この head の担当域から
        // 「自分の head の [dk, dv] の c 列」として読む。
        const long long row_t = (long long)(ci * chunk_len - 1);
        s_col = s_rows + (row_t * nv + h) * (long long)dk * dv + (long long)c;
    }

    // ---- conv 重み + chunk 開始時点のウィンドウ (担当 3 チャネル) -----------
    const int qch = kh * dk + c;
    const int kch = kd_total + kh * dk + c;
    const int vch = 2 * kd_total + h * dv + c;
    float wq[4], wk[4], wv[4];
    for (int j = 0; j < 4; ++j) {
        wq[j] = wconv[qch * 4 + j];
        wk[j] = wconv[kch * 4 + j];
        wv[j] = wconv[vch * 4 + j];
    }
    float inq[3], ink[3], inv[3];
    for (int j = 0; j < cs; ++j) {
        inq[j] = dno_gdn_conv_input(qkv, conv_st_in, qch, t0, j, conv_dim, cs);
        ink[j] = dno_gdn_conv_input(qkv, conv_st_in, kch, t0, j, conv_dim, cs);
        inv[j] = dno_gdn_conv_input(qkv, conv_st_in, vch, t0, j, conv_dim, cs);
    }

    for (int t = t0; t < t1; ++t) {
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

        // ---- 4/5. デルタルール + gated RMS norm (hoist 版) -----------------
        dno_gdn_step(q_sh, k_sh, v_sh, o_sh, sc, s_col, nullptr, dk, dv, eps);

        // ---- このトークン後の S 行を書く -----------------------------------
        // 境界行の不変条件: chunk ci が読む行は ci*chunk_len - 1。これは
        // chunk ci-1 の**最終トークン**の状態なので、行 (ci-1)*chunk_len 以降を
        // 書くのは chunk ci-1 自身に限る、というだけでは足りない (chunk ci-1 は
        // 自分の最終行で終わるため)。逆に「自分の最終トークンまで書く」と、
        // chunk ci が t0-1 行 (自分が読み終えた行) を上書きすることになるが、
        // 2 launch 経路の前提 (別 launch の書き手) では読む行の書き手は前の
        // launch なので、この launch 内の上書きは読む前に済んでいる。
        //
        // したがって書き手を 2 つに限る (どちらも 1 行だけ・最終 chunk だけ複数行):
        //   1. chunk ci (最後以外) は **自分の最終行 t1-1 を 1 行だけ**書く。
        //      次の chunk ちょうど 1 つがそれを読む。
        //   2. 最終 chunk は残りの全行 [t0, n) を書く。
        // 中間行は誰も読まないので書かない (転送量も増えない)。
        if (is_last || t == t1 - 1) {
            float* p = s_rows + (((long long)t * nv + h) * dk) * dv + c;
            const float* q = s_col;
            for (int i = 0; i < dk; ++i) {
                p[(long long)i * dv] = q[(long long)i * dv];
            }
        }
        const long long zo = (long long)t * vd + h * dv + c;
        const float zv = z[zo];
        y_out[zo] = o_sh[c] * sc[4] * wnorm[c] * (zv / (1.0f + expf(-zv)));
    }

    // ---- 最終状態の書き戻し (最終 chunk ブロックのみ) -----------------------
    if (is_last) {
        for (int j = 0; j < cs; ++j) {
            conv_st_out[qch * cs + j] = inq[j];
            conv_st_out[kch * cs + j] = ink[j];
            conv_st_out[vch * cs + j] = inv[j];
        }
        for (int i = 0; i < dk; ++i) {
            ssm_out[(hbase + (long long)i) * dv + c] = s_col[(long long)i * dv];
        }
    }
}
