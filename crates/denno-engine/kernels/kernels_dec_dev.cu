// task 179: decode の full attention 1 層をデバイスで閉じるためのカーネル。
//
// 背景: lab t179a の平均表 (`DENNO_GDN_DEC_KPROF`) で、full 層 1 回の
// wall ~264 us のうち GPU カーネル ~176 us・コピー ~43 us・gap ~28 us・
// ホスト仕事 ~31 us だった。ホスト仕事の内訳は gate_mul 10.4 us / rope_q 3.8 /
// kv_split 1.1 / rope_k 0.6 / mirror_book 0.5 / cache_push 0.4 で、
// その間 GPU が遊ぶ (gap)。qkv GEMV の出力をホストへ戻さず、正規化 + rope +
// ミラー追記 + attention + 出力ゲート + 出力投影までデバイスで繋ぐと、
// コピー 43 us・ホスト 31 us・gap 28 us の大半が消える。
//
// 数値: 既存の CPU 経路と**ビット一致**させる。
// - RMS の二乗和は thread 0 が `i = 0..dh-1` の逐次順で足す
//   (`qknorm_rows_inplace` の `iter().map(|v| v*v).sum::<f32>()` と同じ順)。
// - rope の cos/sin は**ホストで作った表**をそのまま受け取る (f32 の powf/cos/sin
//   の結果をデバイスで作り直さない)。回転は `x0*c - x1*s` / `x0*s + x1*c` の
//   左結合 (CPU と同一)。
// - 正規化は `(x*inv)*w` の左結合、`inv = 1/sqrt(ss/dh + eps)`。
// - 出力ゲートは dno_expf (kernels_elem.cu の glibc expf 再現) + `1/(1+e)` + 乗算。
//   dno_expf は kernels_elem.cu からの**写し** (既存ファイルは触らない規約)。
// - -fmad=false で作る (build_ptx.sh の dec_dev)。

#include <cstdint>

// ---- dno_expf (kernels_elem.cu の写し。定数も同じ) ------------------------
__device__ uint64_t DNO_EXP2_TAB[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
    0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
    0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
    0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
    0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
    0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
    0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
    0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
    0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
__device__ double DNO_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_EXP_SHIFT = 0x1.8p+52;

__device__ __forceinline__ uint64_t dno_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_asuint(float f) {
    return __float_as_uint(f);
}

__device__ __forceinline__ float dno_expf(float x) {
    uint32_t abstop = (dno_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (dno_asuint(x) == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    double xd = (double)x;
    double kd = __fma_rn(DNO_EXP_INVLN2N, xd, DNO_EXP_SHIFT);
    uint64_t ki = dno_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_EXP_SHIFT);
    double r = __fma_rn(DNO_EXP_INVLN2N, xd, -kd);
    uint64_t t = DNO_EXP2_TAB[ki % 32];
    t += ki << (52 - 5);
    double s = dno_asdouble(t);
    double z2 = __fma_rn(DNO_EXP_POLY[0], r, DNO_EXP_POLY[1]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(DNO_EXP_POLY[2], r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// ---- q/k の head 単位 RMS 正規化 + 部分 rope -------------------------------
//
// 1 block = 1 head。blockIdx.x < heads は q (qkv の [h*2*dh, h*2*dh+dh))、
// それ以降は k (qkv の [qg_dim + kh*dh, ...))。
// q の gate (qkv の [h*2*dh+dh, ...)) は正規化せず gate_out へ素通しする。
// k の v (qkv の [qg_dim + kv_dim + kh*dh, ...)) は kv_out の後半へ写す
// (ホスト KV キャッシュへの push 用。1 回の D2H にまとめるため)。
//
// cs はホストが作った cos/sin の表 (`cs.v[2*j] = cos`, `cs.v[2*j+1] = sin`、
// j = 0..n_rot/2-1)。pos 依存の超越関数をデバイスで計算し直さない
// (デバイスの powf/cosf/sinf は glibc とビット一致しないため)。
// 値渡しの struct にしてある: ポインタ引数だとホストの表を指せない。
struct DnoRopeCs {
    float v[128];
};

extern "C" __global__ void dno_dec_qknorm_rope(
    const float* __restrict__ qkv,
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    float* __restrict__ k_out,
    float* __restrict__ kv_out,
    const float* __restrict__ wq,
    const float* __restrict__ wk,
    DnoRopeCs cs,
    int heads,
    int kv_heads,
    int dh,
    int n_rot,
    int qg_dim,
    int kv_dim,
    float eps) {
    extern __shared__ float sh[];
    float* sn = sh + dh;
    const int bx = blockIdx.x;
    const bool is_q = bx < heads;
    const int kh = bx - heads;
    const float* seg;
    const float* w;
    float* out;
    if (is_q) {
        seg = qkv + (size_t)bx * (size_t)(2 * dh);
        w = wq;
        out = q_out + (size_t)bx * (size_t)dh;
    } else {
        seg = qkv + (size_t)qg_dim + (size_t)kh * (size_t)dh;
        w = wk;
        out = k_out + (size_t)kh * (size_t)dh;
    }
    for (int i = threadIdx.x; i < dh; i += blockDim.x) sh[i] = seg[i];
    __syncthreads();
    // 二乗和は thread 0 が逐次 (CPU と同じ順)。
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < dh; ++i) ss = __fadd_rn(ss, __fmul_rn(sh[i], sh[i]));
        ss = __fdiv_rn(ss, (float)dh);
        s_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    }
    __syncthreads();
    const float inv = s_inv;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) {
        sn[i] = __fmul_rn(__fmul_rn(sh[i], inv), w[i]);
    }
    __syncthreads();
    const int half = n_rot >> 1;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) {
        float v = sn[i];
        if (i < half) {
            const float c = cs.v[2 * i];
            const float s = cs.v[2 * i + 1];
            v = __fsub_rn(__fmul_rn(sn[i], c), __fmul_rn(sn[i + half], s));
        } else if (i < n_rot) {
            const int j = i - half;
            const float c = cs.v[2 * j];
            const float s = cs.v[2 * j + 1];
            v = __fadd_rn(__fmul_rn(sn[j], s), __fmul_rn(sn[i], c));
        }
        out[i] = v;
        if (!is_q) kv_out[(size_t)kh * (size_t)dh + i] = v;
    }
    if (is_q) {
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            gate_out[(size_t)bx * (size_t)dh + i] = seg[dh + i];
        }
    } else {
        const float* vsrc =
            qkv + (size_t)qg_dim + (size_t)kv_dim + (size_t)kh * (size_t)dh;
        float* vdst = kv_out + (size_t)kv_dim + (size_t)kh * (size_t)dh;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) vdst[i] = vsrc[i];
    }
}

// ---- 出力ゲート: out[i] *= sigmoid(gate[i]) (CPU の gate_mul と同一) --------
extern "C" __global__ void dno_dec_gate_mul(
    float* __restrict__ out,
    const float* __restrict__ gate,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float e = dno_expf(-g);
        const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
        out[i] = __fmul_rn(out[i], sg);
    }
}
