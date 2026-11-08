// task 184: decode の hc mix でホストが回していた 2 つの要素演算をデバイスへ移す。
//
// 背景 (lab t179a の平均表 `DENNO_GDN_DEC_KPROF`): mix 1 回のホスト仕事は
// mix_norm 5.1 us + mix_inj 3.8 us = 8.9 us で、decode は 1 トークンに
// 2*n_layer+1 = 97 回 mix を呼ぶ (0.86 ms/token)。どちらも
//   - 二乗和 / 内積を **1 スレッドの逐次和** で回していた (e=2,560 要素の依存鎖)
//   - 結果をホストの GEMV 経路へ渡すためにホストで計算する必要があった
// ので、デバイスへ移して mix の submission に畳む。
//
// 数値: 縮約を **ツリー** にする (CPU の逐次和とは丸め順が変わる = md5 は
// DIFF しうる)。1 要素あたりの演算列 (`x*inv*gamma` / `w*x` の乗算 1 回 +
// 加算 1 回) は CPU と同じで、-fmad=false + __*_rn により 1 演算 1 丸めにする。
// 決定論: スレッド割り当てと縮約の順序は固定 (実行ごとに変わらない)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dev.cu -o kernels_mix_dev.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dev`。

// 32 スレッドの warp 内縮約 → 8 warp の縮約 (block 256 固定) → 全スレッドへ配る。
// red は 16 float 以上の作業域 (呼び出し側の static shared)。
// 加算順は固定 (offset 16,8,4,2,1 と warp 番号の昇順)。
__device__ __forceinline__ float dno_mix_reduce256(float v, float* red)
{
    const int t = threadIdx.x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if ((t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    if (t < 8) {
        float s = red[t];
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffu, s, off);
        }
        if (t == 0) red[8] = s;
    }
    __syncthreads();
    return red[8];
}

// ---------------------------------------------------------------------------
// grouped RMS 正規化 (1 ブロック = 1 ストリーム)
//
// out[s*e + i] = x[s*e + i] * inv * gamma[s*e + i]
//   ss  = Σ_i x^2 / e     (ツリー縮約)
//   inv = 1 / sqrt(ss + eps)
//
// out は x と同一ポインタでよい (縮約が終わるまで書き込まない)。
// grid = hc、block = 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_norm(
    const float* __restrict__ x,      // [hc][e]
    const float* __restrict__ gamma,  // [hc][e]
    float* __restrict__ out,          // [hc][e] (x と同一可)
    int e, float eps)
{
    __shared__ float red[16];
    const int s = blockIdx.x;
    const int t = threadIdx.x;
    const float* src = x + (size_t)s * (size_t)e;
    float ss = 0.0f;
    for (int i = t; i < e; i += 256) {
        const float v = src[i];
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    ss = dno_mix_reduce256(ss, red);
    const float mean = __fdiv_rn(ss, (float)e);
    const float den = __fsqrt_rn(__fadd_rn(mean, eps));
    const float inv = __fdiv_rn(1.0f, den);
    float* dst = out + (size_t)s * (size_t)e;
    const float* g = gamma + (size_t)s * (size_t)e;
    for (int i = t; i < e; i += 256) {
        dst[i] = __fmul_rn(__fmul_rn(src[i], inv), g[i]);
    }
}

// ---------------------------------------------------------------------------
// inject の内積 (1 ブロック = 1 行)
//
// w は BF16 の生バイト [n][k] (gguf の行優先)。BF16 → f32 は 16 bit 左シフトで
// 厳密 (丸めなし)。out[r] = Σ_i w[r][i] * x[i] (ツリー縮約)。
// 1/hc は呼び出し側 (ホストの hc_inject_apply) が掛ける。
// grid = n、block = 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_inj(
    const unsigned char* __restrict__ w,  // [n][k] BF16
    const float* __restrict__ x,          // [k]
    float* __restrict__ out,              // [n]
    int k, int n)
{
    __shared__ float red[16];
    const int r = blockIdx.x;
    if (r >= n) return;
    const int t = threadIdx.x;
    const unsigned short* rowp =
        (const unsigned short*)(w + (size_t)r * (size_t)k * 2u);
    float s = 0.0f;
    for (int i = t; i < k; i += 256) {
        const float wv = __uint_as_float((unsigned int)rowp[i] << 16);
        s = __fadd_rn(s, __fmul_rn(wv, x[i]));
    }
    s = dno_mix_reduce256(s, red);
    if (t == 0) out[r] = s;
}
