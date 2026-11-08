// task 172: prefill の full attention 後処理 (per-head RMSNorm + partial NeoX rope /
// 出力ゲート) をデバイスで回すためのカーネル。
//
// どちらも qwen4exp (Flash-Next) の実形 (heads 24 / kv_heads 2 / dh 256 / n_rot 64)
// を前提にしない汎用形で書く (dh は blockDim、heads は grid.y)。
//
// 数値: CPU 参照 (tensor::qknorm_rows_inplace + qwen4exp_attn::rope_neox_partial /
// AttnLayer::gate_mul) と同じ式・同じ順。積和は __fmul_rn / __fadd_rn / __fsub_rn
// で分離し -fmad=false で作る (build_ptx.sh の pf2 target)。
// expf だけは CUDA の libm を使うので、CPU の glibc expf と 1 ULP だけずれうる
// (prefill の射影はビット一致契約の外。decode はこのカーネルを通らない)。

// 1 head ぶんの RMSNorm + partial NeoX rope。
//
// grid  = (nt, heads)   : blockIdx.x = トークン, blockIdx.y = head
// block = (cols, 1, 1)  : cols = head_dim (<= 1024)
// shmem = cols * 4 B
//
// x は行-major だが行ストライドと head のストライドが違う (q は [q|gate] 融合で
// x_stride = 2*heads*dh / hstride = 2*dh、k は x_stride = kv_heads*dh /
// hstride = dh) ので、両方を引数で受ける。
// cs は pos ごとの表 [nt][n_rot] で、前半 half が cos、後半 half が sin
// (ホストが CPU と同一式で作る)。
extern "C" __global__ void dno_pf2_qknorm_rope(
    const float *__restrict__ x,
    int x_stride,
    int hstride,
    const float *__restrict__ w,
    const float *__restrict__ cs,
    float *__restrict__ out,
    int out_stride,
    int cols,
    int n_rot,
    float eps) {
    extern __shared__ float sh[];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= cols) {
        return;
    }
    const float *row = x + (size_t)t * (size_t)x_stride + (size_t)h * (size_t)hstride;
    // 1) 2 乗和は thread 0 が i 昇順の逐次で取る (CPU の sum と同一の列)
    __shared__ float s_inv;
    if (d == 0) {
        float ss = 0.0f;
        for (int i = 0; i < cols; ++i) {
            ss = __fadd_rn(ss, __fmul_rn(row[i], row[i]));
        }
        ss = __fdiv_rn(ss, (float)cols);
        s_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    }
    __syncthreads();
    // 2) normed = (x * inv) * w  (左結合 = CPU の *v * inv * w[i])
    sh[d] = __fmul_rn(__fmul_rn(row[d], s_inv), w[d]);
    __syncthreads();
    // 3) partial NeoX rope: 対 (j, j + half) を 1 スレッドが回す
    const int half = n_rot / 2;
    if (d < half) {
        const float *csp = cs + (size_t)t * (size_t)n_rot;
        const float c = csp[d];
        const float s = csp[half + d];
        const float x0 = sh[d];
        const float x1 = sh[d + half];
        sh[d] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
        sh[d + half] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
    }
    __syncthreads();
    // 4) 出力 (head は out_stride の中で cols 間隔)
    out[(size_t)t * (size_t)out_stride + (size_t)h * (size_t)cols + d] = sh[d];
}

// 出力ゲート: att[t][h][i] *= sigmoid(qg[t][h][dh + i])
// qg は [q|gate] 融合 (head ごとに 2*dh) の生の投影出力。
//
// grid = (nt, heads), block = (dh, 1, 1)
extern "C" __global__ void dno_pf2_gate_mul(
    float *__restrict__ att,
    const float *__restrict__ qg,
    int heads,
    int dh) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= dh) {
        return;
    }
    float *av = att + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)dh + (size_t)i;
    const float g = qg[((size_t)t * (size_t)heads + (size_t)h) * (size_t)(2 * dh) + (size_t)dh +
                       (size_t)i];
    const float e = expf(-g);
    *av = __fmul_rn(*av, __fdiv_rn(1.0f, __fadd_rn(1.0f, e)));
}
