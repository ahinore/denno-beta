// task 225: prefill の GDN (ssm) スキャン第 3 版。1 ヘッドを 4 ブロックに分け、
// レーン数を 8 に増やす (状態はレジスタ常駐のまま)。
//
// PTX へのコンパイル (build_ptx.sh の `gdn_pf3`):
//
//   nvcc -ptx -arch=compute_90 -fmad=false -O3 kernels_gdn_pf3.cu \
//        -o kernels_gdn_pf3.ptx
//
// なぜ別カーネルか (lab t222c の [q4ledger-pf]、14,018 トークン prefill):
//   `ssm scan_pf` は 1,470.3 ms。カーネルは `dno_gdn_pf_scan64` (task 186) で、
//   1 ブロック = 1 ヘッドの 64 列 → **96 ブロックしかない**。170 SM に対して
//   0.56 ブロック/SM、8 ワープ/ブロックなので 4.5 ワープ/SM しか立たず、
//   1 トークンあたりの依存鎖 (32 行の skv 和 → 32 行の更新 + o 和) が
//   そのまま latency になる。実測 2.9 us/トークン。
//
//   このカーネルは同じ演算を
//     - 1 ブロック = 1 ヘッドの **32 列** (4 ブロックで 1 ヘッド) → 192 ブロック
//     - 1 ブロック = 32 列 x **8 レーン** = 256 スレッド (1 レーン 16 行)
//     - 依存鎖は 16 行 (v2 の半分)、縮約は 8 レーンの固定木 (xor 1 -> 2 -> 4)
//     - リングを 4 段にして、t+3 のグローバル読みを t の計算の**前に**発行し
//       silu とリング書きを計算の後に回す (読みの latency を計算で隠す)
//     - conv は 8 レーンが q/k の 1 チャネルずつを分担 (v は lane 0)
//   で回す。
//
// 数値: 要素ごとの更新 (S *= eg / S += k*dlt)・conv の左結合・silu・gate の式は
//   v2 (`dno_gdn_pf_scan64`) と同じ順序で書く。変わるのは **縮約の木**だけ
//   (4 レーン -> 8 レーン。skv / q・k の L2 / o の和)。よって md5 は DIFF しうる
//   (丸め順の差のみ)。
//
// 前提 (ホスト側 `gdn_pf3_ok` が同じ条件を検査する):
//   - head_k_dim == head_v_dim == PF3_DK (128) / d_conv == 4 (cs == 3)
//   - n >= 1 / nk >= 1 / nv >= 1 / conv_dim == 2*nk*dk + nv*dv
// 形が違うときは v2 / 従来カーネルへ fail closed。
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。
#define PF3_DK 128
#define PF3_LANES 8
#define PF3_NR 16
#define PF3_COLS 32
#define PF3_THREADS 256
#define PF3_DCONV 4
#define PF3_CS 3
#define PF3_RING 4
#define PF3_NORM_THREADS 128

// リング 1 段の並び: q[128] / k[128] / v[32] / eg / beta
#define PF3_OFF_K 128
#define PF3_OFF_V 256
#define PF3_OFF_G 288

extern "C" __global__ void __launch_bounds__(PF3_THREADS, 1)
dno_gdn_pf_scan32(
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
    __shared__ float ring[PF3_RING][PF3_OFF_G + 2];
    // task 225: o を 4 B ずつ散らして書くと 1 トークンあたり 32 個の部分書きに
    // なる。列 32 本をここに溜め、バリア後に warp 0 が 128 B で 1 回書く
    // (値は同じ・決定論。書き方だけ変える)。
    __shared__ float so[2][PF3_COLS];

    const int h = blockIdx.x;              // v-head
    const int half = blockIdx.y;           // 列の 1/4 (0..3)
    const int kh = h % nk;                 // q/k はタイル repeat 規約
    const int kd_total = nk * PF3_DK;
    const int vd = nv * PF3_DK;
    const int lane = threadIdx.x % PF3_LANES;
    const int cl = threadIdx.x / PF3_LANES; // 0..PF3_COLS-1 (ブロック内の列)
    const int c = half * PF3_COLS + cl;     // ヘッド内の列 (0..127)
    const int r0 = lane * PF3_NR;           // このレーンが持つ行の先頭

    // このスレッドが conv を担当するチャネル (1 本) と、出力先のリング位置。
    // lane 0..3: q[cl + lane*32] / lane 4..7: k[cl + (lane-4)*32]
    int ch;
    int slot;
    if (lane < 4) {
        ch = kh * PF3_DK + lane * PF3_COLS + cl;
        slot = lane * PF3_COLS + cl;
    } else {
        ch = kd_total + kh * PF3_DK + (lane - 4) * PF3_COLS + cl;
        slot = PF3_OFF_K + (lane - 4) * PF3_COLS + cl;
    }
    const int vch = 2 * kd_total + h * PF3_DK + c;

    // ---- 状態 (自列の担当 16 行) をレジスタへ ------------------------------
    float S[PF3_NR];
#pragma unroll
    for (int i = 0; i < PF3_NR; ++i) {
        S[i] = ssm_in[((long long)h * PF3_DK + r0 + i) * PF3_DK + c];
    }

    // ---- conv の重みと初期ウィンドウ --------------------------------------
    float w4[PF3_DCONV];
    float wnd[PF3_CS];
#pragma unroll
    for (int j = 0; j < PF3_DCONV; ++j) {
        w4[j] = wconv[ch * PF3_DCONV + j];
    }
#pragma unroll
    for (int j = 0; j < PF3_CS; ++j) {
        wnd[j] = conv_st_in[ch * PF3_CS + j];
    }
    // lane 0 は自分の列の v チャネルも受け持つ。
    float w4v[PF3_DCONV];
    float wndv[PF3_CS];
#pragma unroll
    for (int j = 0; j < PF3_DCONV; ++j) {
        w4v[j] = wconv[vch * PF3_DCONV + j];
    }
#pragma unroll
    for (int j = 0; j < PF3_CS; ++j) {
        wndv[j] = conv_st_in[vch * PF3_CS + j];
    }

    // ---- 1 トークンぶんの conv + silu + gate をリングへ -------------------
    // 式・丸めは v2 (`kernels_gdn_pf2.cu`) と同一。読みは呼び出し側で先に済ませ、
    // ここでは受け取った値だけを使う (グローバル読みの latency を計算で隠す)。
#define PF3_STORE(T, B, X, XV)                                                     \
    {                                                                              \
        const float _x = (X);                                                      \
        const float _acc =                                                         \
            w4[0] * wnd[0] + w4[1] * wnd[1] + w4[2] * wnd[2] + w4[3] * _x;         \
        wnd[0] = wnd[1];                                                           \
        wnd[1] = wnd[2];                                                           \
        wnd[2] = _x;                                                               \
        ring[(B)][slot] = _acc / (1.0f + expf(-_acc));                             \
        if (lane == 0) {                                                           \
            const float _xv = (XV);                                                \
            const float _accv = w4v[0] * wndv[0] + w4v[1] * wndv[1] +               \
                                w4v[2] * wndv[2] + w4v[3] * _xv;                    \
            wndv[0] = wndv[1];                                                     \
            wndv[1] = wndv[2];                                                     \
            wndv[2] = _xv;                                                         \
            ring[(B)][PF3_OFF_V + cl] = _accv / (1.0f + expf(-_accv));             \
            if (cl == 0) {                                                         \
                const float _a = alpha[(long long)(T) * nv + h] + wdt[h];          \
                const float _sp = (_a > 20.0f) ? _a : logf(1.0f + expf(_a));       \
                ring[(B)][PF3_OFF_G] = expf(wa[h] * _sp);                          \
                const float _bt = beta[(long long)(T) * nv + h];                   \
                ring[(B)][PF3_OFF_G + 1] = 1.0f / (1.0f + expf(-_bt));             \
            }                                                                      \
        }                                                                          \
    }

    // 3 トークン先まで先に埋める (ループ内で t+3 を仕込む)。
    if (n > 0) {
        PF3_STORE(0, 0, qkv[(long long)0 * conv_dim + ch],
                  (lane == 0) ? qkv[(long long)0 * conv_dim + vch] : 0.0f);
    }
    if (n > 1) {
        PF3_STORE(1, 1, qkv[(long long)1 * conv_dim + ch],
                  (lane == 0) ? qkv[(long long)1 * conv_dim + vch] : 0.0f);
    }
    if (n > 2) {
        PF3_STORE(2, 2, qkv[(long long)2 * conv_dim + ch],
                  (lane == 0) ? qkv[(long long)2 * conv_dim + vch] : 0.0f);
    }
    __syncthreads();

    for (int t = 0; t < n; ++t) {
        // t+3 の読みをここで発行する (使うのは計算の後 = latency を隠す)。
        const bool ld = (t + 3 < n);
        float x3 = 0.0f;
        float xv3 = 0.0f;
        if (ld) {
            x3 = qkv[(long long)(t + 3) * conv_dim + ch];
            if (lane == 0) {
                xv3 = qkv[(long long)(t + 3) * conv_dim + vch];
            }
        }
        const int b = t % PF3_RING;
        const float eg = ring[b][PF3_OFF_G];
        const float bta = ring[b][PF3_OFF_G + 1];

        // ---- q/k の L2 ノルム (8 レーンの部分和) -------------------------
        float sq = 0.0f;
        float sk2 = 0.0f;
#pragma unroll
        for (int i = 0; i < PF3_NR; ++i) {
            const float qv = ring[b][r0 + i];
            sq += qv * qv;
            const float kv = ring[b][PF3_OFF_K + r0 + i];
            sk2 += kv * kv;
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 1);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 2);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 4);
        const float q_inv = 1.0f / sqrtf(sq + eps);
        const float k_inv = 1.0f / sqrtf(sk2 + eps);
        const float qs = q_inv * q_scale;

        // ---- デルタ則 ---------------------------------------------------
        float skv = 0.0f;
#pragma unroll
        for (int i = 0; i < PF3_NR; ++i) {
            S[i] *= eg;
            skv += (ring[b][PF3_OFF_K + r0 + i] * k_inv) * S[i];
        }
        skv += __shfl_xor_sync(0xffffffffu, skv, 1);
        skv += __shfl_xor_sync(0xffffffffu, skv, 2);
        skv += __shfl_xor_sync(0xffffffffu, skv, 4);
        const float dlt = bta * (ring[b][PF3_OFF_V + cl] - skv);
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < PF3_NR; ++i) {
            S[i] += (ring[b][PF3_OFF_K + r0 + i] * k_inv) * dlt;
            o += (ring[b][r0 + i] * qs) * S[i];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        o += __shfl_xor_sync(0xffffffffu, o, 4);
        if (lane == 0) {
            so[t & 1][cl] = o;
        }
        // t+3 のリング書き (読みは t の計算と重なる)。
        if (ld) {
            PF3_STORE(t + 3, (t + 3) % PF3_RING, x3, xv3);
        }
        // 次のトークンは別のリング段を使う。読み終わった段を 3 トークン後に
        // 上書きするので、ここは 1 回で足りる。
        __syncthreads();
        // 直前のトークンの o を warp 0 が 32 列まとめて書く (128 B 連続)。
        if (t > 0 && threadIdx.x < PF3_COLS) {
            o_out[(long long)(t - 1) * vd + h * PF3_DK + half * PF3_COLS + threadIdx.x] =
                so[(t - 1) & 1][threadIdx.x];
        }
    }
#undef PF3_STORE
    // 最後のトークンの o (ループ末尾のバリアの後なので so は確定している)。
    if (n > 0 && threadIdx.x < PF3_COLS) {
        o_out[(long long)(n - 1) * vd + h * PF3_DK + half * PF3_COLS + threadIdx.x] =
            so[(n - 1) & 1][threadIdx.x];
    }

    // ---- 最終状態の書き戻し -----------------------------------------------
    // 窓は**チャネルの持ち主**が書く。q/k は 4 ブロックが同じ値を計算するので、
    // 書き込みは half==0 のブロックだけが行う (同じ値の write-write 競合を作らない)。
    if (half == 0) {
#pragma unroll
        for (int j = 0; j < PF3_CS; ++j) {
            conv_st_out[ch * PF3_CS + j] = wnd[j];
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < PF3_CS; ++j) {
            conv_st_out[vch * PF3_CS + j] = wndv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < PF3_NR; ++i) {
        ssm_out[((long long)h * PF3_DK + r0 + i) * PF3_DK + c] = S[i];
    }
}
