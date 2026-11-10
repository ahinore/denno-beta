// task 216: decode の hc mix の down 投影 (BF16 重み + hc_lo_apply) を
// 1 ブロック = 1 行・uint4 (8 BF16 = 16 B) 読みで回す版。
//
// 背景 (lab t192a / t204b の実測): 従来の `dno_matvec_bf16_r_silu` は
// 1 ブロック 256 スレッドで 1 行 (k = 10,240 = 20 KB) を読み、レーンは 2 B ずつ
// (bf16 を 1 要素) 読む。up 側は同じ形で 14.0 us (実効 468 GB/s) だったので
// `kernels_mix_up.cu` で uint4 読みに変えた (task 192)。down はこの形のまま
// 残っていたので、同じ手当てをする。
//
// 値は **行内の合計順が変わる** (レーンの分割と縮約木が 256 スレッド版と違う)。
// 決定論は保つ (固定の木・固定のスレッド数)。silu の epilogue は
// `dno_matvec_bf16_r_silu` と同一 (dno_expf = glibc expf のビット一致複製)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_down.cu -o kernels_mix_down.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_down`。

#include <cstdint>

#define N_TABLE 32  // EXP2F_TABLE_BITS = 5

// glibc sysdeps/ieee754/flt-32/e_exp2f_data.c (__exp2f_data)
// (constant memory ではなく通常の device メモリ: warp 内でインデックスが
//  分岐すると constant cache が直列化されるため。256B なので L1 で十分)
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
// poly_scaled[i] = poly[i] / N^(3-i), N = 32
__device__ double DNO_EXP_POLY[3] = {
    0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0,
    0x1.ebfce50fac4f3p-3 / 32.0 / 32.0,
    0x1.62e42ff0c52d6p-1 / 32.0,
};
__device__ __constant__ double DNO_EXP_INVLN2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double DNO_EXP_SHIFT = 0x1.8p+52;  // 1.5 * 2^52

__device__ __forceinline__ uint64_t dno_asuint64(double f) {
    return __double_as_longlong(f);
}
__device__ __forceinline__ double dno_asdouble(uint64_t i) {
    return __longlong_as_double((long long)i);
}
__device__ __forceinline__ uint32_t dno_asuint(float f) {
    return __float_as_uint(f);
}

/// glibc expf bit-exact replica (round-to-nearest, WANT_ERRNO semantics —
/// overflow -> +inf, underflow -> 0.0f; errno side effects are irrelevant here).
__device__ __forceinline__ float dno_expf(float x) {
    uint32_t abstop = (dno_asuint(x) >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {  // top12(88.0f): |x| >= 88 or inf/nan
        // glibc はここで早期 return せず、以下のいずれにも該当しない場合
        // ([88.0, 88.72] と [-103.97, -88]) はそのまま通常経路で計算する
        // (exp(88) は f32 最大値未満、exp(-88) は非正規化数として表現可能)。
        if (dno_asuint(x) == 0xff800000u) return 0.0f;       // -inf
        if (abstop >= 0x7f8u) return x + x;                  // inf / nan
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);  // > log(0x1p128) -> +inf
        if (x < -0x1.9fe368p6f) return 0.0f;                 // < log(0x1p-150) -> uflow 0
        // fall through to normal path
    }
    double xd = (double)x;
    // glibc の FMA ビルド (gcc -ffp-contract=fast) は以下を fma に縮約する:
    //   kd = fma(InvLn2N, xd, SHIFT)  (= z + SHIFT の縮約)
    //   r  = fma(InvLn2N, xd, -kd)    (= z - kd の縮約)
    // 多項式 3 ヶ所も同様。実機 (glibc 2.35 + RTX 5090 ホスト) で glibc expf /
    // Rust f32::exp の返すビットと完全一致することを確認済み
    // (x=32.564632f: 非縮約版は 1ULP 差 = 0x56fc9f1b vs 正 0x56fc9f1c)。
    double kd = __fma_rn(DNO_EXP_INVLN2N, xd, DNO_EXP_SHIFT);
    uint64_t ki = dno_asuint64(kd);
    kd = __dadd_rn(kd, -DNO_EXP_SHIFT);
    double r = __fma_rn(DNO_EXP_INVLN2N, xd, -kd);
    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
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

// BF16 の生 2 バイトを f32 へ (上位 16 bit の複製 = 丸めなし)。
__device__ __forceinline__ float dno_mixdn_bf16(unsigned int h)
{
    return __uint_as_float(h << 16);
}

// 1 ブロック = 1 行。block = 512 (16 warp)。k は 8 の倍数であること (ホストが門番)。
// 引数の並びは `dno_matvec_bf16_r_silu(h, w, out, k, n, inv_hc)` と同じ。
extern "C" __global__ void dno_mix_down_bf16(
    const float* __restrict__ x,          // [k]
    const unsigned char* __restrict__ w,  // [n][k] BF16
    float* __restrict__ out,              // [n]
    int k, int n, float inv_hc)
{
    const int j = blockIdx.x;
    if (j >= n) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const unsigned short* rowp =
        (const unsigned short*)(w + (size_t)j * (size_t)k * 2u);

    float acc = 0.0f;
    // 1 スレッド = 8 要素 (uint4)。行頭もスレッドの開始位置も 16 B 整列。
    for (int c = tid * 8; c < k; c += (int)blockDim.x * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(rowp + c);
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.x & 0xffffu), x[c + 0]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.x >> 16), x[c + 1]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.y & 0xffffu), x[c + 2]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.y >> 16), x[c + 3]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.z & 0xffffu), x[c + 4]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.z >> 16), x[c + 5]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.w & 0xffffu), x[c + 6]));
        acc = __fadd_rn(acc, __fmul_rn(dno_mixdn_bf16(v.w >> 16), x[c + 7]));
    }
    // warp 内の固定木 (offset 16, 8, 4, 2, 1)。
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, off));
    }
    __shared__ float sh[32];
    if (lane == 0) {
        sh[warp] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float t = sh[0];
        for (int i = 1; i < nwarp; ++i) {
            t = __fadd_rn(t, sh[i]);
        }
        const float v = __fmul_rn(t, inv_hc);
        out[j] = __fdiv_rn(v, __fadd_rn(1.0f, dno_expf(-v)));
    }
}
