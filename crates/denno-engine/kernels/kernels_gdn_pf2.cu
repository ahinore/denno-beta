// denno CUDA: prefill の GDN (ssm) スキャンを「ヘッドあたり 2 ブロック + 1 トークン
// 1 barrier」で回す版 (task 186)。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf2`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf2.cu -o kernels_gdn_pf2.ptx
//
// なぜ別カーネルか (lab t183a の [q4ledger-pf]、14,018 トークン prefill):
//   `dno_gdn_pf_scan128` (kernels_gdn_pf.cu) は 1 ブロック = 1 v-head (128 列) で、
//   48 ブロックしかなく SM の 28% しか使わない。しかも 1 トークンあたり
//   `__syncthreads` を 3 回、conv の `div.rn` 3 本と gate の logf/expf を
//   スレッド 1 本が直列に払い、その待ちがそのまま積まれる。
//   実測 2.832 ms/呼び出し = 5.53 us/トークン、実効 16.3 GB/s・0.85 TFLOP/s で、
//   同じ層の qkv GEMM (74.8 TFLOP/s) の 88 倍非効率。帯域でも演算でもなく
//   **latency (少ブロック + barrier + 直列の超越関数)** が律速 ([REPORT_P186.md] 0.2)。
//
//   このカーネルは同じ演算を
//     - 1 ブロック = 1 v-head の **64 列** (2 ブロックで 1 ヘッド) → 96 ブロック
//     - conv の窓と重みを **レーンごとに 1 チャネルずつ**分散 (128 q + 128 k を
//       64 列 x 4 レーンで分担)。v は自分の 64 列ぶんだけ
//     - 共有メモリは 3 段のリング (t+2 の conv を t の計算中に先に走らせる)
//     - barrier は **1 トークン 1 回** (読み終わったリングを 2 トークン後に
//       上書きするので、末尾の 1 回で足りる)
//     - 出力の gated RMS norm は列方向 128 個の縮約なので別カーネル
//       (`dno_gdn_pf_norm128`) へ出す。ここは o を書くだけ
//   で回す。
//
// 数値: **`dno_gdn_pf_scan128` とビット一致**を狙って書いてある。
//   - 状態の持ち方 (列 c を 4 レーンが 32 行ずつ)・縮約の木 (xor 1 -> xor 2) は同じ
//   - conv の 4 タップの和・silu・gate の式は同じ順序で書く (-fmad=false なので
//     mul と add が分離される。式を変えないこと)
//   - 出力 RMS は 4 本の部分和 s_l (列 l*32..l*32+31 の昇順和) を
//     ((s0+s1)+(s2+s3)) で結合する (scan128 の shfl_xor 1 -> 2 と同じ木)
//   よって md5 は base と一致するはず (違えばどこかの順序が変わっている)。
//
// 前提 (ホスト側 `gdn_pf2_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == PF2_DK (128) / d_conv == 4 (cs == 3)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
// 形が違うときは既存カーネル (`dno_gdn_pf_scan128` / `dno_gdn_scan_sig`) へ fail closed。

// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define PF2_DK 128
#define PF2_LANES 4
#define PF2_NR 32
#define PF2_COLS 64
#define PF2_THREADS 256
#define PF2_DCONV 4
#define PF2_CS 3
#define PF2_RING 3
#define PF2_NORM_THREADS 128

// リング 1 段の並び: q[128] / k[128] / v[64] / eg / beta
#define PF2_OFF_K 128
#define PF2_OFF_V 256
#define PF2_OFF_G 320

extern "C" __global__ void __launch_bounds__(PF2_THREADS, 1)
dno_gdn_pf_scan64(
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
    __shared__ float ring[PF2_RING][PF2_OFF_G + 2];

    const int h = blockIdx.x;             // v-head
    const int half = blockIdx.y;          // 列の半分 (0 / 1)
    const int kh = h % nk;                // q/k はタイル repeat 規約
    const int kd_total = nk * PF2_DK;
    const int vd = nv * PF2_DK;
    const int lane = threadIdx.x % PF2_LANES;
    const int cl = threadIdx.x / PF2_LANES; // 0..PF2_COLS-1 (ブロック内の列)
    const int c = half * PF2_COLS + cl;     // ヘッド内の列 (0..127)
    const int r0 = lane * PF2_NR;           // このレーンが持つ行の先頭

    // このスレッドが conv を担当するチャネル (1 本) と、出力先のリング位置。
    // lane 0: q[cl] (+ v[cl]) / lane 1: q[cl+64] / lane 2: k[cl] / lane 3: k[cl+64]
    int ch;
    int slot;
    if (lane == 0) {
        ch = kh * PF2_DK + cl;
        slot = cl;
    } else if (lane == 1) {
        ch = kh * PF2_DK + PF2_COLS + cl;
        slot = PF2_COLS + cl;
    } else if (lane == 2) {
        ch = kd_total + kh * PF2_DK + cl;
        slot = PF2_OFF_K + cl;
    } else {
        ch = kd_total + kh * PF2_DK + PF2_COLS + cl;
        slot = PF2_OFF_K + PF2_COLS + cl;
    }
    const int vch = 2 * kd_total + h * PF2_DK + c;

    // ---- 状態 (自列の担当 32 行) をレジスタへ ------------------------------
    float S[PF2_NR];
#pragma unroll
    for (int i = 0; i < PF2_NR; ++i) {
        S[i] = ssm_in[((long long)h * PF2_DK + r0 + i) * PF2_DK + c];
    }

    // ---- conv の重みと初期ウィンドウ --------------------------------------
    float w4[PF2_DCONV];
    float wnd[PF2_CS];
#pragma unroll
    for (int j = 0; j < PF2_DCONV; ++j) {
        w4[j] = wconv[ch * PF2_DCONV + j];
    }
#pragma unroll
    for (int j = 0; j < PF2_CS; ++j) {
        wnd[j] = conv_st_in[ch * PF2_CS + j];
    }
    // lane 0 は自分の列の v チャネルも受け持つ。
    float w4v[PF2_DCONV];
    float wndv[PF2_CS];
#pragma unroll
    for (int j = 0; j < PF2_DCONV; ++j) {
        w4v[j] = wconv[vch * PF2_DCONV + j];
    }
#pragma unroll
    for (int j = 0; j < PF2_CS; ++j) {
        wndv[j] = conv_st_in[vch * PF2_CS + j];
    }

    // ---- 1 トークンぶんの conv + silu + gate をリングへ -------------------
    // 現在の式 (kernels_gdn_pf.cu) と同じ順序・同じ丸めで書く。
#define PF2_FILL(T, B)                                                             \
    {                                                                              \
        const long long _row = (long long)(T) * conv_dim;                          \
        const float _x = qkv[_row + ch];                                           \
        const float _acc =                                                         \
            w4[0] * wnd[0] + w4[1] * wnd[1] + w4[2] * wnd[2] + w4[3] * _x;         \
        wnd[0] = wnd[1];                                                           \
        wnd[1] = wnd[2];                                                           \
        wnd[2] = _x;                                                               \
        ring[(B)][slot] = _acc / (1.0f + expf(-_acc));                             \
        if (lane == 0) {                                                           \
            const float _xv = qkv[_row + vch];                                     \
            const float _accv = w4v[0] * wndv[0] + w4v[1] * wndv[1] +               \
                                w4v[2] * wndv[2] + w4v[3] * _xv;                    \
            wndv[0] = wndv[1];                                                     \
            wndv[1] = wndv[2];                                                     \
            wndv[2] = _xv;                                                         \
            ring[(B)][PF2_OFF_V + cl] = _accv / (1.0f + expf(-_accv));             \
            if (cl == 0) {                                                         \
                const float _a = alpha[(long long)(T) * nv + h] + wdt[h];          \
                const float _sp = (_a > 20.0f) ? _a : logf(1.0f + expf(_a));       \
                ring[(B)][PF2_OFF_G] = expf(wa[h] * _sp);                          \
                const float _bt = beta[(long long)(T) * nv + h];                   \
                ring[(B)][PF2_OFF_G + 1] = 1.0f / (1.0f + expf(-_bt));             \
            }                                                                      \
        }                                                                          \
    }

    // 2 トークン先まで先に埋める (t の計算中に t+2 の conv を走らせる)。
    if (n > 0) {
        PF2_FILL(0, 0);
    }
    if (n > 1) {
        PF2_FILL(1, 1);
    }
    __syncthreads();

    for (int t = 0; t < n; ++t) {
        // 次の次をここで仕込む (リングは 3 段なので t%3 の読みと衝突しない)。
        if (t + 2 < n) {
            PF2_FILL(t + 2, (t + 2) % PF2_RING);
        }
        const int b = t % PF2_RING;
        const float eg = ring[b][PF2_OFF_G];
        const float bta = ring[b][PF2_OFF_G + 1];

        // ---- q/k の L2 ノルム (4 レーンの部分和) -------------------------
        float sq = 0.0f;
        float sk2 = 0.0f;
#pragma unroll
        for (int i = 0; i < PF2_NR; ++i) {
            const float qv = ring[b][r0 + i];
            sq += qv * qv;
            const float kv = ring[b][PF2_OFF_K + r0 + i];
            sk2 += kv * kv;
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 1);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 2);
        const float q_inv = 1.0f / sqrtf(sq + eps);
        const float k_inv = 1.0f / sqrtf(sk2 + eps);
        const float qs = q_inv * q_scale;

        // ---- デルタ則 ---------------------------------------------------
        float skv = 0.0f;
#pragma unroll
        for (int i = 0; i < PF2_NR; ++i) {
            S[i] *= eg;
            skv += (ring[b][PF2_OFF_K + r0 + i] * k_inv) * S[i];
        }
        skv += __shfl_xor_sync(0xffffffffu, skv, 1);
        skv += __shfl_xor_sync(0xffffffffu, skv, 2);
        const float dlt = bta * (ring[b][PF2_OFF_V + cl] - skv);
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < PF2_NR; ++i) {
            S[i] += (ring[b][PF2_OFF_K + r0 + i] * k_inv) * dlt;
            o += (ring[b][r0 + i] * qs) * S[i];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (lane == 0) {
            o_out[(long long)t * vd + h * PF2_DK + c] = o;
        }
        // 次のトークンは別のリング段を使う。読み終わった段を 2 トークン後に
        // 上書きするので、ここは 1 回で足りる。
        __syncthreads();
    }
#undef PF2_FILL

    // ---- 最終状態の書き戻し -----------------------------------------------
    // 窓は**チャネルの持ち主**が書く (q/k はレーンごとに別チャネル)。q/k は
    // 両ブロックが同じ値を計算するので、書き込みは half==0 のブロックだけが行う
    // (同じ値の write-write 競合を作らない)。v は各ブロックが自分の列ぶんだけ。
    if (half == 0) {
#pragma unroll
        for (int j = 0; j < PF2_CS; ++j) {
            conv_st_out[ch * PF2_CS + j] = wnd[j];
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < PF2_CS; ++j) {
            conv_st_out[vch * PF2_CS + j] = wndv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < PF2_NR; ++i) {
        ssm_out[((long long)h * PF2_DK + r0 + i) * PF2_DK + c] = S[i];
    }
}

// ---------------------------------------------------------------------------
// 出力の gated RMS norm + z ゲート (o -> y、in-place 可)
// ---------------------------------------------------------------------------
//
// 1 トークン 1 ヘッドを 4 スレッドで受け持つ (1 ブロック = 32 組)。
// s_l = Σ_{i=0..31} o[l*32+i]^2 を昇順に積み、shfl_xor 1 -> 2 で
// ((s0+s1)+(s2+s3)) を作る (scan128 の木と同じ)。128 列の書き込みは
// レーンごとに 32 列ずつ分担する。
extern "C" __global__ void __launch_bounds__(PF2_NORM_THREADS, 1)
dno_gdn_pf_norm128(
    float* __restrict__ oy,           // [n, nv*dv] (o を読み、y を書く)
    const float* __restrict__ z,      // [n, nv*dv]
    const float* __restrict__ wnorm,  // [dv]
    int n, int nv, float eps)
{
    const int dv = PF2_DK;
    const int vd = nv * PF2_DK;
    const int lane = threadIdx.x % 4;
    const int slot = threadIdx.x / 4;
    const long long rows = (long long)n * nv;
    const long long row = (long long)blockIdx.x * (PF2_NORM_THREADS / 4) + slot;
    // 端数ブロックでも shfl の参加者を揃えるため、範囲外は row 0 を計算して
    // 書き込みだけ落とす (分岐で早期 return しない)。
    const bool live = row < rows;
    const long long use = live ? row : 0;
    const int t = (int)(use / nv);
    const int h = (int)(use % nv);
    const long long base = (long long)t * vd + h * dv;

    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < PF2_NR; ++i) {
        const float ov = oy[base + lane * PF2_NR + i];
        ss += ov * ov;
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rms = 1.0f / sqrtf(ss / (float)dv + eps);
    if (!live) {
        return;
    }
#pragma unroll
    for (int i = 0; i < PF2_NR; ++i) {
        const int c = lane * PF2_NR + i;
        const long long k = base + c;
        const float zv = z[k];
        oy[k] = oy[k] * rms * wnorm[c] * (1.0f / (1.0f + expf(-zv)));
    }
}
