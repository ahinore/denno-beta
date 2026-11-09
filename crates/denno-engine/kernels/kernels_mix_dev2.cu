// task 192: decode の hc mix で 2 本に分かれていた norm / inject を 1 本に畳む。
//
// 背景 (lab t184a の `[dec-kprof-avg]` mix 表): `dno_mix_norm` 6.2 us +
// `dno_mix_inj` 5.6 us で、hc=4 の形では **どちらも起動の固定費が支配的**
// (計算は 40 KB の読み書きだけ)。2 本を 1 本にすると起動が 1 回減る。
//
// hc 方向を 1 ブロックに畳む:
//   - norm は hc 本の二乗和を **同じ 256 スレッドで 1 本ずつ** 回す
//     (現行の「1 ストリーム = 1 ブロック」と同じ i の刻み・同じ縮約木)。
//   - 正規化を書き終えてから __syncthreads() を 1 回置き、inject の
//     hc 行を同じ 256 スレッドで 1 行ずつ回す (現行の「1 行 = 1 ブロック」と
//     同じ i の刻み・同じ縮約木)。
// したがって **値は 2 本版とビット一致**する (丸め順を変えない)。
// 決定論も同じ (スレッド割り当てと縮約順は固定)。
//
// out は x と同一ポインタでよい (各ストリームの二乗和を全部取り終えてから書く)。
// w == 0 / inj == 0 のときは inject を撃たない (head の mix = norm だけ)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dev2.cu -o kernels_mix_dev2.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dev2`。

#define DNO_MIX2_HC_MAX 8

// 32 スレッドの warp 内縮約 → 8 warp の縮約 (block 256 固定) → 全スレッドへ配る。
// kernels_mix_dev.cu の dno_mix_reduce256 と同一 (同じ加算順)。
__device__ __forceinline__ float dno_mix2_reduce256(float v, float* red)
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
// grouped RMS 正規化 + inject の内積 (1 ブロック = hc ストリーム)
//
// out[s*e + i] = x[s*e + i] * inv[s] * gamma[s*e + i]
//   ss[s]  = Σ_i x[s*e+i]^2 / e        (現行と同じ 256 スレッドの縮約木)
//   inv[s] = 1 / sqrt(ss[s] + eps)
// inj[r] = Σ_i w[r][i] * out[i]        (w は BF16 の生バイト [hc][hc*e])
//
// grid = 1、block = 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_mix_norm_inj(
    const float* __restrict__ x,             // [hc][e]
    const float* __restrict__ gamma,         // [hc][e]
    float* __restrict__ out,                 // [hc][e] (x と同一可)
    const unsigned char* __restrict__ w,     // [hc][k] BF16 (NULL 可)
    float* __restrict__ inj,                 // [hc] (NULL 可)
    int e, int hc, int k, float eps)
{
    __shared__ float red[16];
    __shared__ float invs[DNO_MIX2_HC_MAX];
    const int t = threadIdx.x;
    if (hc <= 0 || hc > DNO_MIX2_HC_MAX) {
        return;
    }

    // 1. hc ストリームの二乗和 (1 本ずつ、現行の 1 ブロック版と同じ刻み)。
    //    ss をレジスタに置くため、ストリームのループは定数上限で展開する。
    float ss[DNO_MIX2_HC_MAX];
#pragma unroll
    for (int s = 0; s < DNO_MIX2_HC_MAX; ++s) ss[s] = 0.0f;
    for (int i = t; i < e; i += 256) {
#pragma unroll
        for (int s = 0; s < DNO_MIX2_HC_MAX; ++s) {
            if (s < hc) {
                const float v = x[(size_t)s * (size_t)e + i];
                ss[s] = __fadd_rn(ss[s], __fmul_rn(v, v));
            }
        }
    }
    for (int s = 0; s < hc; ++s) {
        const float sum = dno_mix2_reduce256(ss[s], red);
        if (t == 0) {
            const float mean = __fdiv_rn(sum, (float)e);
            const float den = __fsqrt_rn(__fadd_rn(mean, eps));
            invs[s] = __fdiv_rn(1.0f, den);
        }
    }
    __syncthreads();

    // 2. 正規化して書く (in-place 可。各添字は 1 スレッドだけが書く)。
    for (int s = 0; s < hc; ++s) {
        const float iv = invs[s];
        for (int i = t; i < e; i += 256) {
            const size_t j = (size_t)s * (size_t)e + i;
            out[j] = __fmul_rn(__fmul_rn(x[j], iv), gamma[j]);
        }
    }
    // inject は正規化後の out を全部読むので、書き終わるまで待つ。
    __syncthreads();

    // 3. inject の内積 (1 行ずつ、現行の 1 ブロック版と同じ刻み)。
    if (w != 0 && inj != 0) {
        for (int r = 0; r < hc; ++r) {
            const unsigned short* rowp =
                (const unsigned short*)(w + (size_t)r * (size_t)k * 2u);
            float acc = 0.0f;
            for (int i = t; i < k; i += 256) {
                const float wv = __uint_as_float((unsigned int)rowp[i] << 16);
                acc = __fadd_rn(acc, __fmul_rn(wv, out[i]));
            }
            const float sum = dno_mix2_reduce256(acc, red);
            if (t == 0) inj[r] = sum;
        }
    }
}
