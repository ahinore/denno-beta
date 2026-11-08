// denno CUDA decode epilogue kernels (task 187)
//
// PTX:
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_dec_epi.cu -o kernels_dec_epi.ptx
// まとめて作るなら `kernels/build_ptx.sh dec_epi`。
//
// 数値契約: **ホストの足し込みと同じ順序・同じ丸め**。
//   ホストは選択 expert を index 昇順に
//     out[j] += w_t * h_t[j]
//   と積み、最後に shared の寄与 `out[j] += sh[j] * sgw` を足す。ここでは
//   `acc += w[t] * h[t*e+j]` を t の昇順に 1 本のアキュムレータで回し、
//   最後に `acc += sgw * sh[j]` を足す。-fmad=false なので積と和は別の丸め
//   (ホストの `pe * hh[j]` + `+=` と同じ 2 演算)。
//   したがって **ミスした expert が無い層では従来のホスト足し込みとビット一致**
//   する (ミスがある層だけ「ヒットを先に全部足してからミスを足す」順序になる)。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

// ---------------------------------------------------------------------------
// 1. expert の寄与 + shared の足し込み (1 スレッド = 1 出力要素)
//
//   h   : [nseg][e] ヒットした expert の寄与 (選択順)
//   w   : [nseg]    再正規化後の重み (選択順)
//   sh  : [e]       shared expert の出力 (has_sh = 0 なら読まない)
//   out : [e]
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_dec_exp_sum(
    const float* __restrict__ h,
    const float* __restrict__ w,
    const float* __restrict__ sh,
    float sgw,
    float* __restrict__ out,
    int e, int nseg, int has_sh)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= e) return;
    float acc = 0.0f;
    for (int t = 0; t < nseg; ++t) {
        acc += w[t] * h[(size_t)t * (size_t)e + (size_t)j];
    }
    if (has_sh != 0) {
        acc += sgw * sh[j];
    }
    out[j] = acc;
}

// ---------------------------------------------------------------------------
// 2. shared expert の silu(gate) * up (デバイス側で閉じるため)
//
//   act[i] = silu(g[i]) * u[i],  silu(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_dec_silu_mul(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ act,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    const float s = x / (1.0f + __expf(-x));
    act[i] = s * u[i];
}
