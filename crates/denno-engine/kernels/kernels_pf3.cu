// task 183: prefill の full attention で、融合済みの qkv 1 本 (q|gate|k|v) を
// **1 回の GEMM** で撃ったあとの後処理。kernels_pf2.cu の 2 本と式は同じで、
// 「融合バッファの行ストライド (ld)」を受ける点だけが違う。
//
// なぜ要るか: 実モデルの 12 本の full 層のうち 8 本は attn_q がファイル上で
// q|gate|k|v の 1 本 (k=2560 n=13312) に融合しており、`pf_full_qkv` の
// 「行列ごとに重みポインタが取れる」前提を満たさない。1 回の GEMM の出力は
// 1 トークン = 13,312 行の連続なので、q / k / v は**行オフセット**で読む。
//
// PTX: `bash build_ptx.sh pf3` (pf2 と同じ -fmad=false)。
//
// 数値: CPU 参照 (qwen4exp_attn::AttnLayer::gate_mul / 行のコピー) と同じ式。
// 積和は __fmul_rn / __fadd_rn で分離する (pf2 と同じ)。

// 出力ゲート: att[t][h][i] *= sigmoid(qg[t][ld の h*2*dh + dh + i])
//
// grid = (nt, heads), block = (dh, 1, 1)。ld は qg の 1 トークンあたりの行数
// (融合 1 本なら 13,312、[q|gate] だけのバッファなら 2*heads*dh)。
extern "C" __global__ void dno_pf3_gate_mul(
    float *__restrict__ att,
    const float *__restrict__ qg,
    int heads,
    int dh,
    int ld) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= dh) {
        return;
    }
    float *av = att + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)dh + (size_t)i;
    const float g =
        qg[(size_t)t * (size_t)ld + ((size_t)h * 2 + 1) * (size_t)dh + (size_t)i];
    const float e = expf(-g);
    *av = __fmul_rn(*av, __fdiv_rn(1.0f, __fadd_rn(1.0f, e)));
}

// 融合バッファから列範囲 [off, off + n) を詰めて写す (行の gather)。
//   dst[t][i] = src[t][ld + off + i]        t = 0..rows-1, i = 0..n-1
//
// 1 ブロック = 1 行 (n <= 1024)、block = (n, 1, 1)。値の変換はしない。
extern "C" __global__ void dno_pf3_gather(
    const float *__restrict__ src,
    float *__restrict__ dst,
    int rows,
    int ld,
    int off,
    int n) {
    const int t = blockIdx.x;
    const int i = threadIdx.x;
    if (t >= rows || i >= n) {
        return;
    }
    dst[(size_t)t * (size_t)n + (size_t)i] =
        src[(size_t)t * (size_t)ld + (size_t)off + (size_t)i];
}
