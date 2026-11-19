// task 461: prefill の PLE 注入 (gate / conv+silu) をデバイスで回す 2 本。
//
// なぜ要るか (lab t457c の `[q4ledger-pf]`、既定モデル、PROFILE=1):
//   PLE 層 (48 層中 2 層) は 1 チャンク層あたり
//     res の D2H 82 MB → ホストのトークンループ → res の H2D 82 MB
//   を往復し、14k cold では `sub other` の `ple` 212.9 ms + `res_up` 245.8 ms
//   = prefill の 8% を占めていた。データ (res / key / value) はどれも**元から
//   GPU にある**ので、往復をやめて要素演算をデバイスへ置く。
//
// 2 本に分ける理由:
//   - gate は 1 トークンの全列 (e) の縮約で決まる → 1 ブロック = 1 トークン。
//   - conv は列ごとにトークン方向へ因果 (history 付き) → 1 ブロック = 列タイル。
//   並列の軸が違うので 1 本にはできない。
//
// 数値: 1 演算 1 丸め (明示 __*_rn) で CPU 参照 (`ple_inject_into`) と同じ式。
//   違うのは (a) 縮約の木 (block の分担 + warp の木)、(b) sigmoid の exp を
//   `__expf` にしたこと (2 ULP 程度)。conv の積和は k 昇順の逐次で CPU と同一。
//
// 前提 (ホスト側 `pf_ple_dev_ok` が同じ条件を検査する):
//   - 1 <= hc <= PLE_MAX_HC / e <= PLE_MAX_E / kern <= PLE_MAX_KERN
//   - hist = (kern - 1) * dil <= PLE_MAX_HIST
//   - gate の grid は (nt, 1, 1)、conv の grid は (ceil(e / PLE_THREADS), hc, 1)
//
// 値の行にコメントを付けないこと (kernel_invariants が `#define` の行末までを
// 値として読む)。

#include <cstddef>
#include <cstdint>

// 1 ブロックのスレッド数 (cuda.rs の PF_PLE_THREADS と一致させる)。
#define PLE_THREADS 256
// hc の上限 (実形は 4)。
#define PLE_MAX_HC 8
// e の上限 (実形は 2,560)。
#define PLE_MAX_E 8192
// conv のカーネル幅の上限 (実形は 2..4)。
#define PLE_MAX_KERN 8
// 履歴の長さの上限 (hist = (kern-1) * dil。実形は 2..8)。
#define PLE_MAX_HIST 32

/// `1 / (1 + exp(-x))` (CPU の `hyper::sigmoid` と同じ式。exp だけ `__expf`)。
__device__ __forceinline__ float ple_sigmoid(float x)
{
    return __fdividef(1.0f, __fadd_rn(1.0f, __expf(-x)));
}

/// `x * (1 / (1 + exp(-x)))` (CPU の `hyper::silu` と同じ式)。
__device__ __forceinline__ float ple_silu(float x)
{
    return __fmul_rn(x, __fdividef(1.0f, __fadd_rn(1.0f, __expf(-x))));
}

/// 1 ブロックぶんの和 (warp 内は off=16..1 の木、warp 間は番号順)。
/// **全スレッドが呼ぶこと** (戻る前に共有メモリを解放する)。
__device__ __forceinline__ float ple_block_sum(float v, float *sp)
{
    const int nwarp = PLE_THREADS >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    if ((threadIdx.x & 31) == 0) {
        sp[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = sp[0];
        for (int w = 1; w < nwarp; ++w) {
            t = __fadd_rn(t, sp[w]);
        }
        sp[0] = t;
    }
    __syncthreads();
    const float r = sp[0];
    __syncthreads();
    return r;
}

/// PLE の gate と conv 用の逆 RMS を作る。1 ブロック = 1 トークン。
///
/// gate[s] = sigmoid(sign(sv) * sqrt(max(|sv|, 1e-6)))、sv = dot(key_n, query_n) / sqrt(e)
///   key_n  = rms(key[t][s]) * gk[s] / query_n = rms(res[t][s]) * gq[s]
/// invc[s] = 1 / sqrt(mean((value[t] * gate[s])^2) + eps)
extern "C" __global__ void dno_pf_ple_gate(
    const float *__restrict__ key,    // [nt][hc][e]
    const float *__restrict__ res,    // [nt][hc][e]
    const float *__restrict__ value,  // [nt][e]
    const float *__restrict__ gk,     // [hc][e]
    const float *__restrict__ gq,     // [hc][e]
    float *__restrict__ gate,         // [nt][hc]
    float *__restrict__ invc,         // [nt][hc]
    int e,
    int hc,
    float eps,
    float inv_sqrt_e)
{
    __shared__ float sp[32];
    const int tid = (int)threadIdx.x;
    const size_t t = (size_t)blockIdx.x;
    const float fe = (float)e;
    for (int s = 0; s < hc; ++s) {
        const float *krow = key + (t * (size_t)hc + (size_t)s) * (size_t)e;
        const float *rrow = res + (t * (size_t)hc + (size_t)s) * (size_t)e;
        float a = 0.0f;
        float b = 0.0f;
        for (int i = tid; i < e; i += PLE_THREADS) {
            const float kv = krow[i];
            const float rv = rrow[i];
            a = __fmaf_rn(kv, kv, a);
            b = __fmaf_rn(rv, rv, b);
        }
        const float s1 = ple_block_sum(a, sp);
        const float s2 = ple_block_sum(b, sp);
        const float invk = __fdividef(1.0f, __fsqrt_rn(__fadd_rn(__fdividef(s1, fe), eps)));
        const float invq = __fdividef(1.0f, __fsqrt_rn(__fadd_rn(__fdividef(s2, fe), eps)));
        const float *gkp = gk + (size_t)s * (size_t)e;
        const float *gqp = gq + (size_t)s * (size_t)e;
        float d = 0.0f;
        for (int i = tid; i < e; i += PLE_THREADS) {
            const float kn = __fmul_rn(__fmul_rn(krow[i], invk), gkp[i]);
            const float qn = __fmul_rn(__fmul_rn(rrow[i], invq), gqp[i]);
            d = __fmaf_rn(kn, qn, d);
        }
        const float dot = ple_block_sum(d, sp);
        if (tid == 0) {
            const float sv = __fmul_rn(dot, inv_sqrt_e);
            const float av = sv < 0.0f ? -sv : sv;
            const float mag = __fsqrt_rn(av < 1e-6f ? 1e-6f : av);
            const float sgn = sv > 0.0f ? 1.0f : (sv < 0.0f ? -1.0f : 0.0f);
            gate[t * (size_t)hc + (size_t)s] = ple_sigmoid(__fmul_rn(sgn, mag));
        }
        __syncthreads();
    }
    for (int s = 0; s < hc; ++s) {
        const float g = gate[t * (size_t)hc + (size_t)s];
        const float *vrow = value + t * (size_t)e;
        float a = 0.0f;
        for (int i = tid; i < e; i += PLE_THREADS) {
            const float gv = __fmul_rn(vrow[i], g);
            a = __fmaf_rn(gv, gv, a);
        }
        const float s3 = ple_block_sum(a, sp);
        if (tid == 0) {
            invc[t * (size_t)hc + (size_t)s] =
                __fdividef(1.0f, __fsqrt_rn(__fadd_rn(__fdividef(s3, fe), eps)));
        }
        __syncthreads();
    }
}

/// 列ごとにトークン方向へ conv (causal, history 付き) → silu → res へ加算。
///
///   gated[t][s][d] = value[t][d] * gate[t][s]
///   x[t][s][d]     = gated * invc[t][s] * gc[s][d]        (grouped RMS の出力)
///   conv[t][s][d]  = silu(sum_k wt[k][s][d] * x[t - (kern-1-k)*dil])
///   res[t][s][d]  += gated + conv
///
/// 履歴の**読み**は CPU の ring と同じ式 (slot = (head + hist + t - back) % hist、
/// head は 0 = 線形)。**書き**は線形 (先頭が最古) で返す: CPU の
/// `ple_inject_batch_*` も終端で `ple_hist_to_linear` を通すため。
/// `want` のときは最終トークンの (gated + conv)^2 を列ごとに足し、ブロックの
/// 和を `part[s * ntiles + blockIdx.x]` へ書く (ホストが s → tile の順に足す)。
extern "C" __global__ void dno_pf_ple_conv(
    const float *__restrict__ value,    // [nt][e]
    const float *__restrict__ gate,     // [nt][hc]
    const float *__restrict__ invc,     // [nt][hc]
    const float *__restrict__ gc,       // [hc][e]
    const float *__restrict__ wt,       // [kern][hc][e]
    const float *__restrict__ hist_in,  // [hist][hc][e]
    float *__restrict__ hist_out,       // [hist][hc][e]
    float *__restrict__ res,            // [nt][hc][e]
    float *__restrict__ part,           // [hc][ntiles] (want == 0 なら触らない)
    int nt,
    int e,
    int hc,
    int kern,
    int dil,
    int hist,
    int head,
    int want,
    int ntiles)
{
    const int s = (int)blockIdx.y;
    const int d = (int)blockIdx.x * PLE_THREADS + (int)threadIdx.x;
    const bool act = d < e;
    const size_t hc_dim = (size_t)hc * (size_t)e;
    const size_t stride = (size_t)hc * (size_t)e;
    const size_t off = (size_t)s * (size_t)e + (size_t)d;
    float wreg[PLE_MAX_KERN];
    for (int k = 0; k < kern && k < PLE_MAX_KERN; ++k) {
        wreg[k] = act ? wt[(size_t)k * hc_dim + off] : 0.0f;
    }
    const float gcol = act ? gc[off] : 0.0f;
    float xr[PLE_MAX_HIST];
    for (int i = 0; i < PLE_MAX_HIST; ++i) {
        xr[i] = 0.0f;
    }
    float contrib = 0.0f;
    for (int t = 0; t < nt; ++t) {
        if (!act) {
            continue;
        }
        const float g = gate[(size_t)t * (size_t)hc + (size_t)s];
        const float iv = invc[(size_t)t * (size_t)hc + (size_t)s];
        const float v = value[(size_t)t * (size_t)e + (size_t)d];
        const float gated = __fmul_rn(v, g);
        const float x = __fmul_rn(__fmul_rn(gated, iv), gcol);
        float acc = 0.0f;
        for (int k = 0; k < kern; ++k) {
            const int back = (kern - 1 - k) * dil;
            float src;
            if (back == 0) {
                src = x;
            } else if (t >= back) {
                src = xr[(t - back) % hist];
            } else {
                int slot = (head + hist + ((t - back) % hist)) % hist;
                src = hist_in[(size_t)slot * hc_dim + off];
            }
            acc = __fmaf_rn(wreg[k], src, acc);
        }
        const float conv = ple_silu(acc);
        if (hist > 0) {
            xr[t % hist] = x;
        }
        const size_t o = (size_t)t * stride + off;
        const float add = __fadd_rn(gated, conv);
        res[o] = __fadd_rn(res[o], add);
        if (want && t + 1 == nt) {
            contrib = __fmaf_rn(add, add, contrib);
        }
    }
    // 履歴は **線形** (先頭が最古) で書き戻す。CPU の `ple_inject_batch_*` も
    // チャンク終端で `ple_hist_to_linear` を通すので、ここを ring のままにすると
    // 次のチャンク / decode が履歴を 1 段ずらして読む (lab 前の自己点検で発見)。
    if (hist > 0 && act) {
        for (int j = 0; j < hist; ++j) {
            const int back_j = hist - 1 - j;
            const int tj = nt - 1 - back_j;
            float v;
            if (tj >= 0) {
                v = xr[tj % hist];
            } else {
                const int slot = (head + hist + ((tj % hist) + hist) % hist) % hist;
                v = hist_in[(size_t)slot * hc_dim + off];
            }
            hist_out[(size_t)j * hc_dim + off] = v;
        }
    }
    if (want) {
        __shared__ float sp[32];
        const float sum = ple_block_sum(contrib, sp);
        if (threadIdx.x == 0) {
            part[(size_t)s * (size_t)ntiles + (size_t)blockIdx.x] = sum;
        }
    }
}
