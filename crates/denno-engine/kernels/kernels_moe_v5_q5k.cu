// denno CUDA MoE GEMV kernel (v5 family, Q5_K) — task 192。
//
// Q5_K の 1 スーパーブロック (256 要素) は 176 B で、16 B の倍数 (11 x uint4)。
//   bytes  0..16  = d(2) + dmin(2) + scales[12]      -> uint4 1 本 (hdr)
//   bytes 16..48  = qh[32]                            -> uint4 2 本
//   bytes 48..176 = qs[128]                           -> uint4 8 本
// 行頭は row_bytes = 176 * ブロック数 なので 16 B 整列 = uint4 で読める。
//
// v5 (`kernels_moe_v5.cu`) の Q4_K と同じ組み立て:
//   1. 1 スーパーブロックを 11 本の uint4 でいっぺんに読む (176 バイトロード -> 11 本)。
//   2. サブブロック s (32 要素) ごとに x[32] をレジスタへ先読みしてから畳む。
//   3. アキュムレータを 4 本に分ける (1 本の acc += の依存鎖を 1/4 に)。
//   4. レーンは **スーパーブロック単位** で分ける (nb / lanes 個ずつ)。
//      割り切れない形はホストが v4 へ落とす (fail closed)。
//
// Q4_K との違いは 5 ビット目 (qh) の取り出しだけ:
//   サブブロック s の要素 l (0..31) の上位ビットは qh[l] の bit s。
//   値は ds * (ql のニブル + 16*bit) - ms (CPU / v4 と同じ演算列)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_v5_q5k.cu -o kernels_moe_v5_q5k.ptx
// まとめて作るなら `kernels/build_ptx.sh moe_v5_q5k`。
//
// 数値契約: v5 の Q8_0 / Q4_K と同じく **v4 とはビット一致しない**
// (レーンの区間と 4 アキュムレータの合計順が違う)。dequant の演算列
// (1 要素あたり `ds*q - ms` の 2 演算と `y*x` の乗算 + 加算) は v4 と同じで、
// -fmad=false + 明示 __*_rn により 1 演算 1 丸め。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

// ---- f16 → f32 (kernels_moe_v5.cu の dno_v5_f16_bits と同じ実装) -----------
__device__ __forceinline__ float dno_v5q_f16_bits(unsigned int h)
{
    const unsigned int sign = (h & 0x8000u) << 16;
    const unsigned int exp = (h >> 10) & 0x1fu;
    const unsigned int man = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            unsigned int m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

// x[n]..x[n+31] をレジスタへ先読み (kernels_moe_v5.cu と同じ形)。
__device__ __forceinline__ void dno_v5q_prefetch32(const float* __restrict__ xb, float* xv)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(xb + 4 * i);
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// uint4 の成分を添字で選ぶ (添字はコンパイル時定数で呼ぶ)。
__device__ __forceinline__ unsigned int dno_v5q_u4(uint4 v, int i)
{
    return (i == 0) ? v.x : (i == 1) ? v.y : (i == 2) ? v.z : v.w;
}

// scales[12] (blk+4 から 12 バイト) の k バイト目。k はコンパイル時定数で呼ぶ。
// hdr = (d|dmin, scales[0..3], scales[4..7], scales[8..11])。
__device__ __forceinline__ unsigned int dno_v5q_scale_byte(uint4 hdr, int k)
{
    return (dno_v5q_u4(hdr, 1 + (k >> 2)) >> ((k & 3) * 8)) & 0xffu;
}

// 行番号とレーン番号 (v5 と同じ規約: lane = threadIdx.x & (lanes-1))。
__device__ __forceinline__ int dno_v5q_row_of(int rows_per_block, int lanes, int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (lanes - 1);
    return blockIdx.x * rows_per_block + (t / lanes);
}

// シャッフルのマスク。1 ブロックが 1 warp 未満のときは実在するスレッドだけ。
__device__ __forceinline__ unsigned int dno_v5q_mask()
{
    const unsigned int n = blockDim.x;
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

// lanes レーンの合計 (offset 1, 2, 4, ... の log2(lanes) 段)。
__device__ __forceinline__ float dno_v5q_reduce(float v, int lanes, unsigned int mask)
{
    for (int off = 1; off < lanes; off <<= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Q5_K: d(2B) + dmin(2B) + scales[12] + qh[32] + qs[128] = 176 B / 256 要素。
//
// サブブロック s (32 要素) のスケールは get_scale_min_k4 の 6 bit 対:
//   s < 4:  sc = scales[s] & 63,        mn = scales[s+4] & 63
//   s >= 4: sc = (scales[s+4] & 15) | ((scales[s-4] >> 6) << 4)
//           mn = (scales[s+4] >> 4) | ((scales[s]   >> 6) << 4)
// ニブルは qs の (s>>1)*32 バイト目から 32 B (= uint4 2 本) で、
// 前半 32 要素が下位ニブル、後半が上位ニブル (sh = (s&1) ? 4 : 0)。
// 5 ビット目は qh[l] の bit s (l = サブブロック内の位置)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q5k_v5(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = dno_v5q_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = dno_v5q_mask();

    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.0f;

    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 8;      // 256 要素スーパーブロック数
        const int per = nb / lanes;    // 1 レーンのスーパーブロック数
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 176u;
            const uint4* src = reinterpret_cast<const uint4*>(blk);
            uint4 u[11];
#pragma unroll
            for (int i = 0; i < 11; ++i) u[i] = src[i];
            const float d = dno_v5q_f16_bits(u[0].x & 0xffffu);
            const float dmin = dno_v5q_f16_bits(u[0].x >> 16);
            const float* xb = x + (size_t)b * 256u;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                // scales[12] は u[0].y (0..3) / u[0].z (4..7) / u[0].w (8..11)。
                unsigned int sc, mn;
                if (s < 4) {
                    sc = dno_v5q_scale_byte(u[0], s) & 63u;
                    mn = dno_v5q_scale_byte(u[0], s + 4) & 63u;
                } else {
                    sc = (dno_v5q_scale_byte(u[0], s + 4) & 0x0fu)
                       | ((dno_v5q_scale_byte(u[0], s - 4) >> 6) << 4);
                    mn = (dno_v5q_scale_byte(u[0], s + 4) >> 4)
                       | ((dno_v5q_scale_byte(u[0], s) >> 6) << 4);
                }
                const float ds = __fmul_rn(d, (float)sc);
                const float ms = __fmul_rn(dmin, (float)mn);
                // qs は u[3..10] の 8 本。サブブロック対 (s>>1) が 2 本。
                const uint4 qa = u[3 + (s >> 1) * 2];
                const uint4 qb = u[3 + (s >> 1) * 2 + 1];
                // qh は u[1] (l 0..15) と u[2] (l 16..31)。
                const uint4 h0 = u[1];
                const uint4 h1 = u[2];
                const unsigned int sh = (s & 1) ? 4u : 0u;
                const float* xs = xb + (size_t)s * 32u;
                float xv[32];
                dno_v5q_prefetch32(xs, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const uint4 qq = (l < 16) ? qa : qb;
                    const int li = (l < 16) ? l : (l - 16);
                    const unsigned int word = dno_v5q_u4(qq, li >> 2);
                    const int nib = (int)((word >> (((li & 3) * 8) + sh)) & 0x0fu);
                    const uint4 hh = (l < 16) ? h0 : h1;
                    const unsigned int hw = dno_v5q_u4(hh, (l & 15) >> 2);
                    const int hb = (int)((hw >> (((l & 3) * 8) + s)) & 1u) << 4;
                    const float y = __fsub_rn(__fmul_rn(ds, (float)(nib + hb)), ms);
                    acc[l & 3] = __fadd_rn(acc[l & 3], __fmul_rn(y, xv[l]));
                }
            }
        }
    }
    float v = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
    v = dno_v5q_reduce(v, lanes, mask);
    if (valid && lane == 0) out[row] = v;
}
