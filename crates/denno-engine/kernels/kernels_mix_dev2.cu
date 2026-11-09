// task 192: decode の hc mix で 2 本に分かれていた norm / inject を 1 本に畳む。
//
// 背景 (lab t184a / t192a の `[dec-kprof-avg]` mix 表): `dno_mix_norm` 6.2 us +
// `dno_mix_inj` 5.4 us。hc=4 / e=640 / k=2,560 の小さな仕事で、**起動の固定費と
// 待ち**が支配的なので 2 本を 1 本にする。
//
// **hc 方向を 1 ブロックに畳む**が、並列度は 2 本版と同じに保つ:
//   - block = 256 * hc。256 スレッドのグループ g がストリーム g を持つ
//     (= 旧版の 1 ブロック = 1 ストリームと同じ割り当て)。
//   - 二乗和・inject の縮約木もグループ内で旧版と同一 (warp 5 段 -> 8 warp を
//     幅 8 の shuffle)。**値は 2 本版とビット一致**する。
//   - 正規化を書き終えてから __syncthreads() を 1 回置き、inject の hc 行を
//     グループごとに回す。
// 1 ブロック版 (grid=1, block=256 で hc 本を直列) は lab t192a で 34.7 us と
// 2 本版 (11.6 us) より遅かった。並列度を落とさないのが要点。
//
// out は x と同一ポインタでよい (各ストリームの二乗和を全部取り終えてから書く)。
// w == 0 / inj == 0 のときは inject を撃たない (head の mix = norm だけ)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dev2.cu -o kernels_mix_dev2.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dev2`。

#define DNO_MIX2_HC_MAX 4
#define DNO_MIX2_THREADS 256

// グループ (256 スレッド) 内の縮約: 32 スレッドの warp 内 5 段 -> 8 warp を
// 幅 8 の shuffle で 3 段 -> 全スレッドへ配る。kernels_mix_dev.cu の
// dno_mix_reduce256 と同一の加算順 (g = グループ番号, red は [hc][16])。
__device__ __forceinline__ float dno_mix2_reduce(float v, int g, float* red)
{
    const int t = threadIdx.x & (DNO_MIX2_THREADS - 1);   // グループ内の通し番号
    float* r = red + g * 16;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if ((t & 31) == 0) r[t >> 5] = v;
    __syncthreads();
    if (t < 8) {
        float s = r[t];
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffu, s, off);
        }
        if (t == 0) r[8] = s;
    }
    __syncthreads();
    return r[8];
}

// ---------------------------------------------------------------------------
// grouped RMS 正規化 + inject の内積 (1 ブロック = hc ストリーム)
//
// out[s*e + i] = x[s*e + i] * inv[s] * gamma[s*e + i]
//   ss[s]  = Σ_i x[s*e+i]^2 / e        (旧版と同じ 256 スレッドの縮約木)
//   inv[s] = 1 / sqrt(ss[s] + eps)
// inj[r] = Σ_i w[r][i] * out[i]        (w は BF16 の生バイト [hc][hc*e])
//
// grid = 1、block = 256 * hc (hc <= 4)。グループ g (スレッド 256g..256g+255) が
// ストリーム g と inject 行 g を受け持つ。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_norm_inj(
    const float* __restrict__ x,             // [hc][e]
    const float* __restrict__ gamma,         // [hc][e]
    float* __restrict__ out,                 // [hc][e] (x と同一可)
    const unsigned char* __restrict__ w,     // [hc][k] BF16 (NULL 可)
    float* __restrict__ inj,                 // [hc] (NULL 可)
    int e, int hc, int k, float eps)
{
    __shared__ float red[DNO_MIX2_HC_MAX * 16];
    __shared__ float invs[DNO_MIX2_HC_MAX];
    const int g = threadIdx.x / DNO_MIX2_THREADS;      // グループ = ストリーム
    const int t = threadIdx.x & (DNO_MIX2_THREADS - 1);
    if (hc <= 0 || hc > DNO_MIX2_HC_MAX || g >= hc) {
        return;
    }

    // 1. グループ g のストリーム g の二乗和 (旧版の 1 ブロック版と同じ刻み)。
    const float* src = x + (size_t)g * (size_t)e;
    float ss = 0.0f;
    for (int i = t; i < e; i += DNO_MIX2_THREADS) {
        const float v = src[i];
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    const float sum = dno_mix2_reduce(ss, g, red);
    if (t == 0) {
        const float mean = __fdiv_rn(sum, (float)e);
        const float den = __fsqrt_rn(__fadd_rn(mean, eps));
        invs[g] = __fdiv_rn(1.0f, den);
    }
    __syncthreads();

    // 2. 正規化して書く (in-place 可。各添字は 1 スレッドだけが書く)。
    {
        const float iv = invs[g];
        float* dst = out + (size_t)g * (size_t)e;
        const float* gm = gamma + (size_t)g * (size_t)e;
        for (int i = t; i < e; i += DNO_MIX2_THREADS) {
            dst[i] = __fmul_rn(__fmul_rn(src[i], iv), gm[i]);
        }
    }
    // inject は正規化後の out を全部読むので、書き終わるまで待つ。
    __syncthreads();

    // 3. inject の内積 (グループ g が行 g。旧版の 1 ブロック版と同じ刻み)。
    if (w != 0 && inj != 0) {
        const unsigned short* rowp =
            (const unsigned short*)(w + (size_t)g * (size_t)k * 2u);
        float acc = 0.0f;
        for (int i = t; i < k; i += DNO_MIX2_THREADS) {
            const float wv = __uint_as_float((unsigned int)rowp[i] << 16);
            acc = __fadd_rn(acc, __fmul_rn(wv, out[i]));
        }
        const float d = dno_mix2_reduce(acc, g, red);
        if (t == 0) inj[g] = d;
    }
}
