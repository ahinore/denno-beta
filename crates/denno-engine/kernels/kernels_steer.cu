// task 218: 残差ストリームの平均の記録とステアリング (prefill 常駐連鎖用)。
//
// 常駐 prefill (`DENNO_PF_RESIDENT`) では残差 `res` がデバイスに載ったまま
// 層を回る。層ごとに D2H すると 1 層 1 チャンクあたり 84 MB の往復になるので、
// 加算と適用もデバイスで閉じる。
//
// レイアウト: `res` は `[nt][hc][e]` (1 トークンの hc ストリームが連続)、
// `acc` / `v` / `vhat` は `[n_layer][hc][e]` (層ごとに hc ストリームが連続)。
// 1 層ぶんの要素数 `stride = hc * e`。
//
// 数値:
// - 加算は要素ごとに t 昇順の逐次 (ホストの `steer::acc_host` と同じ順)。
//   チャンクを跨いでも順序は変わらないので、同じトークン列なら同じ値になる。
// - 適用は `h - alpha * v` と `h - alpha * (h . vhat) * vhat`。ホストの
//   `Apply::apply_one` と同じ式・同じ演算 (明示 `__*_rn`)。ただし project の
//   内積はブロック縮約 (ツリー) なので、ホストの逐次加算とは最下位ビットが
//   ずれ得る (GPU とホストを跨いだ 1 ビット一致は契約にしない)。
//
// 起動形状は cuda.rs の pf_steer_* が決める (block は 256 固定)。

#include <cstdint>

/// `acc[layer*stride + i] += Σ_t res[t*stride + i]`。
/// grid = ceil(stride / 256), block = 256。1 要素を 1 スレッドが t 昇順に足す。
extern "C" __global__ void dno_steer_acc(const float* __restrict__ res,
                                         float* __restrict__ acc, int nt, int hc,
                                         int e, int layer) {
    const int stride = hc * e;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stride) return;
    float s = 0.0f;
    for (int t = 0; t < nt; ++t) {
        s = __fadd_rn(s, res[(long long)t * stride + idx]);
    }
    acc[(long long)layer * stride + idx] =
        __fadd_rn(acc[(long long)layer * stride + idx], s);
}

/// `res[t*stride + i] -= alpha * v[layer*stride + i]` (subtract)。
/// grid = ceil(stride / 256), block = 256。
extern "C" __global__ void dno_steer_sub(float* __restrict__ res,
                                         const float* __restrict__ v, int nt,
                                         int hc, int e, int layer, float alpha) {
    const int stride = hc * e;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stride) return;
    const float d = __fmul_rn(v[(long long)layer * stride + idx], alpha);
    for (int t = 0; t < nt; ++t) {
        float* p = &res[(long long)t * stride + idx];
        *p = __fsub_rn(*p, d);
    }
}

/// project: (トークン, ストリーム) ごとに `d = h . vhat` を取り、
/// `h -= alpha * d * vhat`。1 ブロック = 1 (t, s)。
/// grid = nt * hc, block = 256 (e が 256 の倍数でなくてもよい)。
extern "C" __global__ void dno_steer_proj(float* __restrict__ res,
                                          const float* __restrict__ vhat, int nt,
                                          int hc, int e, int layer, float alpha) {
    __shared__ float sh[32];
    const int stride = hc * e;
    const int row = blockIdx.x;  // t * hc + s
    const int t = row / hc;
    const int s = row - t * hc;
    const float* vp = &vhat[(long long)layer * stride + s * e];
    float* hp = &res[(long long)t * stride + s * e];
    float acc = 0.0f;
    for (int d = threadIdx.x; d < e; d += blockDim.x) {
        acc = __fadd_rn(acc, __fmul_rn(hp[d], vp[d]));
    }
    const unsigned int mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, off));
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float s2 = sh[0];
        for (int i = 1; i < nw; ++i) s2 = __fadd_rn(s2, sh[i]);
        sh[0] = s2;
    }
    __syncthreads();
    const float c = __fmul_rn(alpha, sh[0]);
    for (int d = threadIdx.x; d < e; d += blockDim.x) {
        hp[d] = __fsub_rn(hp[d], __fmul_rn(c, vp[d]));
    }
}
