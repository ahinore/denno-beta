// task 431: mix の down (Q8_0) を **1 対 = 4 レーン**に割るカーネル。
//
// なぜ要るか (lab t431a の `[dec-kprof-avg]` mix band、1 呼び出しあたり):
//   H2D@a 10.3 us / down 14.0 / up 10.8 / router 6.9 / D2H@a 5.4
// down (`dno_mix_gemv_q8_0_dn6` grid 160 x block 320) は 3.48 MB を 245 GB/s。
// 同じ PTX の dn6 を GDN の qkv (grid 1,280 x 192) で撃つと **962 GB/s** 出る。
// 差は形ではなく **同時に走る warp 数** (mix down は 51,200 スレッド =
// 1,600 warp = 9.4 warp/SM、GDN は 45 warp/SM)。1 レーン = 1 対のままだと
// スレッド数は n_out * n_pairs で固定され、増やせない。
//
// そこで 1 対 (64 要素 = 68 B) を 4 レーンで分ける。16 要素ずつに割ると
//   - x 側は pair*64 + 16*sub で 64 バイト境界 = float4 で読める
//   - 重み側は 16 バイトずつ (レーン 0/1 は 2 バイト境界、2/3 は 4 バイト境界)
//     → 全部 u16 x 8 本で読む (2 バイト境界なら常に安全)
// スレッド数は 320 行 x 160 対 x 4 = **204,800** (6,400 warp = 37.6 warp/SM)。
//
// 縮約: 各レーンが 16 要素を 4 アキュムレータ (要素番号 & 3) に積む ->
// 4 レーンを xor 1, 2 で畳む -> warp 内の 8 グループ (32 レーン) を
// shfl_down 16, 8, 4 で畳む (4 レーンが同じ値を持つのでこの 3 段で 8 本が揃う)
// -> 共有メモリで warp ごとの部分和 -> スレッド 0..rows-1 が固定順に加算。
//
// 数値: 1 行の要素を 16 個ずつ 4 本に割って足すので **丸め順が dn5/dn6 と違う**
// (決定論と行独立性は保たれる)。ゲート `DENNO_MIX_QK=dn7` で切り戻せる。
//
// 前提 (ホストが門番): Q8_0、n_in % 512 == 0 (np = n_in/64 が 8 の倍数)、
// np*4 <= 1024、row_bytes = n_in/32*34。満たさなければ dn6 / dn5 / v5 へ落ちる。
//
// PTX (このコンテナには nvcc が無いので lab / denno-gpu 側で作る):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_mix_dn7.cu -o kernels_mix_dn7.ptx
// まとめて作るなら `kernels/build_ptx.sh mix_dn7`。

#include <cstdint>
#include <cuda_fp16.h>

// ---- Q8_0 の読み方 (kernels_mix_dn6.cu と同じ実装。dn6 の .cu は触らない) ----

// f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------------
__device__ __forceinline__ float dno_dn7_f16_bits(unsigned int h)
{
    // task 640: 手書きのビット展開をハードウェアの cvt.f32.f16 (`__half2float`) に
    // 置き換えた。f16 -> f32 の拡張は IEEE 754 で一意なので結果は同じ (0 / 非正規化数
    // を含む全 65536 パターンで旧実装と一致することを cuda.rs の
    // f16_cvt_exhaustive で確認済み)。NaN と inf だけは旧実装の式をそのまま残す
    // (ハードウェアは NaN を quiet 化するので、ペイロードの再現性のため)。
    if ((h & 0x7c00u) == 0x7c00u) {
        const unsigned int sign = (h & 0x8000u) << 16;
        return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
    }
    return __half2float(__ushort_as_half((unsigned short)h));
}

// このレーンの 16 要素 (u16 x 8 = 32 バイト) と x の 16 要素の内積。
// 演算列は dn6 と同じ「1 積和 1 丸め」で、アキュムレータだけ 4 本。
__device__ __forceinline__ float dno_dn7_dot16(
    const unsigned short* __restrict__ q, const float* xv, float d)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const unsigned int w0 = (unsigned int)q[i];
        const unsigned int w1 = (unsigned int)q[i + 1];
        const float x0 = xv[2 * i + 0];
        const float x1 = xv[2 * i + 1];
        const float x2 = xv[2 * i + 2];
        const float x3 = xv[2 * i + 3];
        a0 = __fadd_rn(a0, __fmul_rn(__fmul_rn((float)(signed char)(w0 & 0xffu), d), x0));
        a1 = __fadd_rn(a1, __fmul_rn(__fmul_rn((float)(signed char)((w0 >> 8) & 0xffu), d), x1));
        a2 = __fadd_rn(a2, __fmul_rn(__fmul_rn((float)(signed char)(w1 & 0xffu), d), x2));
        a3 = __fadd_rn(a3, __fmul_rn(__fmul_rn((float)(signed char)((w1 >> 8) & 0xffu), d), x3));
    }
    return __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
}

// grid = ceil(n_out / rows)、block = rows * (np * 4)。
// (r, pair, sub) のスレッドは対 pair の 16*sub 要素目から 16 要素を担当する。
extern "C" __global__ void dno_mix_gemv_q8_0_dn7(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows)
{
    if (rows < 1) return;
    const int np = n_in >> 6;      // 対 (= 64 要素) の数
    const int per_row = np * 4;    // 1 行あたりのスレッド数 (32 の倍数が前提)
    const int warps = per_row >> 5;
    const int tid = (int)threadIdx.x;
    const int r = tid / per_row;
    const int within = tid - r * per_row;
    const int pair = within >> 2;
    const int sub = within & 3;
    const int row = (int)blockIdx.x * rows + r;
    __shared__ float part[8][32];  // rows <= 8 / warps <= 32

    // x は 16 要素 (float4 x 4)。pair*64 + 16*sub は 64 バイト境界なので整列する。
    float xv[16];
    if (pair < np) {
        const float* xb = x + (size_t)pair * 64u + (size_t)(16 * sub);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = *reinterpret_cast<const float4*>(xb + 4 * i);
            xv[4 * i + 0] = v.x;
            xv[4 * i + 1] = v.y;
            xv[4 * i + 2] = v.z;
            xv[4 * i + 3] = v.w;
        }
    }

    float v = 0.0f;
    if (row < n_out && pair < np) {
        const unsigned char* p = w + (size_t)row * (size_t)row_bytes + (size_t)pair * 68u;
        // スケール: レーン 0,1 は d0 (byte 0)、レーン 2,3 は d1 (byte 34)。
        const int soff = (sub < 2) ? 0 : 34;
        const unsigned int sb = (unsigned int)(*(const unsigned short*)(p + soff));
        const float d = dno_dn7_f16_bits(sb);
        // 16 要素の qs の開始バイト: 0 -> 2 / 1 -> 18 / 2 -> 36 / 3 -> 52。
        const int qoff = 2 + 16 * sub + ((sub >= 2) ? 2 : 0);
        const unsigned short* q = (const unsigned short*)(p + qoff);
        v = dno_dn7_dot16(q, xv, d);
    }
    // 1 対 = 4 レーンの縮約 (xor 1 -> 2 で 4 本が同じ値になる)。
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    // warp 内の 8 グループを 16 -> 8 -> 4 で畳む (レーン 0 が 8 本の和を持つ)。
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 16));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 8));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 4));
    const int wid = within >> 5;
    if ((tid & 31) == 0) part[r][wid] = v;
    __syncthreads();
    if (tid < rows) {
        const int orow = (int)blockIdx.x * rows + tid;
        if (orow < n_out) {
            float s = part[tid][0];
            for (int i = 1; i < warps; ++i) s = __fadd_rn(s, part[tid][i]);
            out[orow] = s;
        }
    }
}
