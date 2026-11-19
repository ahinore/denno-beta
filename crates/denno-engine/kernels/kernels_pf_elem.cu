// task 457: prefill hc mix の要素カーネル第 2 版 (norm の二乗和を block で分担する
// 版と、mixed の sigmoid を f32 の速い経路にした版)。
//
// なぜ (lab t451c の `[pf-mix-kprof]`、3,826 / 192 呼び出しの平均):
//   norm 278.5 us / down 421.5 / lo 11.2、残り (up+mixed+inj) が約 552 us。
//   down は 588 MB を 421.5 us = 1.4 TB/s で、もう帯域の天井に近い (触らない)。
//   残る 2 つは**バイトではなく並列度・演算**で決まっている:
//     - norm は 1 行 e=2,560 の二乗和を **thread 0 が逐次**で舐め、128 スレッドは
//       書き出しにしか使っていない (157 MB を 278.5 us = 564 GB/s)。
//     - mixed は 1 要素あたり hc 回の `dno_pf_expf` を呼ぶ。あれは glibc expf の
//       **double 精度**のビット一致再現で、fp64 はこの GPU では 1/64 速。
//       19.6M 回/呼び出し x 10 演算 = 200M fp64 op が mixed の主コスト (推定)。
//   t451 は「mix の DRAM を 27% 減らして 0%」だったので、バイトを減らす方向は
//   閉じている。ここでは (a) 二乗和の分担と (b) expf の f32 化だけを変える。
//
// 数値: どちらも丸め順が変わる (決定論的)。
//   - norm: 二乗和を「スレッド内は添字昇順、そのあと 32 本の木 → 8 warp の木」で
//     足す。1 演算 1 丸め (__fadd_rn / __fmul_rn) は保つ。
//   - mixed: sigmoid を `__expf` + `__fdividef` にする (exp は f32 で 2 ULP 程度)。
//     重みの量子化誤差 (相対 1e-2 台) より 5 桁小さい。
//   ゲートは cuda.rs 側 (`DENNO_PF_MIX_NORMP` / `DENNO_PF_MIX_FASTSIG`)。どちらも
//   PTX / 形が合わなければ従来カーネルへ fail closed (値も既定と同じ)。
//
// 起動形 (cuda.rs の pf_mix が決める):
//   dno_pf_rmsgroup_p: grid = (nt*hc, 1, 1) / block = (256, 1, 1) / static smem 8*4+4 B
//   dno_pf_mixed_fs  : grid = (ceil(e/256), nt, 1) / block = (256, 1, 1) / smem 0

#include <cstdint>

// 1 ブロックのスレッド数 (cuda.rs の PF_ELEM_NORM_THREADS と一致させる)。
#define PFE_THREADS 256
// 1 スレッドがレジスタに持てる要素数の上限 (= e <= PFE_THREADS * PFE_MAX_PER まで)。
#define PFE_MAX_PER 16
// mixed のスレッド数 (dno_pf_mixed と同じ 256)。
#define PFE_MIXED_THREADS 256

/// grouped RMS norm (batch) の第 2 版。1 ブロック = 1 行 (t, s)。
///
/// 従来 (kernels_pf.cu の dno_pf_rmsgroup) との違いは二乗和の分担だけ:
///   - 各スレッドが自分の列 (tid + k*blockDim) を添字昇順に足す
///   - warp 内は __shfl_down_sync の木、warp 間は shared の 8 本を順に足す
///   - 要素はレジスタに持ったまま書き出しへ回す (2 回目のグローバル読みが無い)
/// 書き出しの式は従来と同一 (`out = (v * inv) * gamma`)。
extern "C" __global__ void dno_pf_rmsgroup_p(
    const float *__restrict__ x,      // [rows][e]
    const float *__restrict__ gamma,  // [hc][e]
    float *__restrict__ out,          // [rows][e]
    int rows,
    int e,
    int hc,
    float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int nt = (int)blockDim.x;
    const int tid = (int)threadIdx.x;
    const float *src = x + (size_t)row * (size_t)e;
    float *dst = out + (size_t)row * (size_t)e;
    // gamma の行はスレッド内で固定 (row % hc)。
    const float *g = gamma + (size_t)(row % hc) * (size_t)e;

    const int nper = (e + nt - 1) / nt;  // 1 スレッドあたりの要素数 (<= PFE_MAX_PER)
    float v[PFE_MAX_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PFE_MAX_PER; ++k) {
        const int i = tid + k * nt;
        v[k] = 0.0f;
        if (k < nper && i < e) {
            const float x0 = src[i];
            v[k] = x0;
            ss = __fadd_rn(ss, __fmul_rn(x0, x0));
        }
    }
    // warp 内の木 (レーン 0 が部分和を持つ)。
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss = __fadd_rn(ss, __shfl_down_sync(0xffffffffu, ss, off));
    }
    // warp 間 (8 本を番号順に足す)。
    __shared__ float s_part[32];
    const int nwarp = (nt + 31) >> 5;
    if ((tid & 31) == 0) s_part[tid >> 5] = ss;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float t = s_part[0];
        for (int w = 1; w < nwarp; ++w) t = __fadd_rn(t, s_part[w]);
        t = __fdiv_rn(t, (float)e);
        s_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(t, eps)));
    }
    __syncthreads();
    const float inv = s_inv;
#pragma unroll
    for (int k = 0; k < PFE_MAX_PER; ++k) {
        const int i = tid + k * nt;
        if (k < nper && i < e) {
            dst[i] = __fmul_rn(__fmul_rn(v[k], inv), g[i]);
        }
    }
}

/// hc mix の結合 (fast sigmoid 版)。1 スレッド = 1 出力列。
///
/// 式は kernels_pf.cu の dno_pf_mixed と同じ:
///   out = (Σ_s xn[s] * sigmoid(gate[s])) * (1/hc)
/// 違いは sigmoid の exp を `__expf` (SFU) にしたことだけ (2 ULP 程度)。
extern "C" __global__ void dno_pf_mixed_fs(
    const float *__restrict__ xn,    // [nt][hc][e]
    const float *__restrict__ gate,  // [nt][hc][e]
    float *__restrict__ out,         // [nt][e]
    int e,
    int hc,
    float inv_hc)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= e) return;
    const size_t t = (size_t)blockIdx.y;
    const size_t base = t * (size_t)hc * (size_t)e + (size_t)d;
    float acc = __fmul_rn(xn[base], __fdividef(1.0f, __fadd_rn(1.0f, __expf(-gate[base]))));
    for (int s = 1; s < hc; ++s) {
        const size_t i = base + (size_t)s * (size_t)e;
        acc = __fadd_rn(
            acc,
            __fmul_rn(xn[i], __fdividef(1.0f, __fadd_rn(1.0f, __expf(-gate[i])))));
    }
    out[t * (size_t)e + (size_t)d] = __fmul_rn(acc, inv_hc);
}
