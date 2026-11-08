// denno CUDA MoE GEMV kernels (v5) — decode の量子 GEMV を **ベクタロード +
// 複数アキュムレータ + 次ブロックの先読み** で回す版。
//
// task 184。v4 (`kernels_moe_v4.cu`) との違い:
//   1. 重みブロックを 1 バイトずつ読まない。Q8_0 は **2 ブロック (68 B) を
//      17 x uint32** で、Q4_K は **1 スーパーブロック (144 B) を 9 x uint4** で
//      読む (バイトロード 32〜128 本 → 17 本 / 9 本)。Q8_0 のブロックは 34 B で
//      2 バイト整列なので 32 B 幅のベクタロードは置けない。行頭は
//      row_bytes = 68 * 対数 (Q8_0) / 144 * ブロック数 (Q4_K) で 4 B / 16 B 整列
//      なので、対 / スーパーブロック単位なら整列ロードが使える。
//   2. 次のブロックのロードを **今のブロックの計算より前に発行** する
//      (Q8_0 は 17 ワードをレジスタへ先読みしてから前の対を畳む)。
//   3. アキュムレータを 4 本に分ける (1 本の acc += の依存鎖を 1/4 に)。
//   4. レーン数は「レーンが担当するブロック数が割り切れる」最大の 2 の冪
//      (Q8_0 はブロック対、Q4_K はスーパーブロック単位)。割り切れない形は
//      ホストが v4 へ落とす (fail closed)。
//
// PTX (このコンテナには nvcc が無いので denno-gpu 側で行う):
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_moe_v5.cu -o kernels_moe_v5.ptx
// まとめて作るなら `kernels/build_ptx.sh moe_v5`。
//
// 数値契約: v4 と同じく **v2 とはビット一致しない** (k の分割と合計順が違う)。
// さらにアキュムレータ 4 本の合計順も v4 と違う。
//
// ブロックレイアウトは ggml 仕様 (llama.cpp, MIT) に従う。ライセンス台帳は
// `docs/history/80_licenses.md`。

// ---- f16 → f32 (kernels_moe_v4.cu の dno_f16_to_f32_v4 と同じ実装) ---------
__device__ __forceinline__ float dno_v5_f16_bits(unsigned int h)
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

__device__ __forceinline__ float dno_v5_f16_to_f32(const unsigned char* b)
{
    return dno_v5_f16_bits((unsigned int)b[0] | ((unsigned int)b[1] << 8));
}

// 4 バイトワード 1 本から 1 バイト (符号付き 8 bit) を取り出す。
__device__ __forceinline__ int dno_v5_i8(unsigned int word, int byte)
{
    return (int)(signed char)((word >> (byte * 8)) & 0xffu);
}

// 68 バイト (Q8_0 の 2 ブロック) を 17 ワードで読む。行頭は 4 バイト整列。
__device__ __forceinline__ void dno_v5_load17(unsigned int* d, const unsigned char* p)
{
    const unsigned int* q = (const unsigned int*)p;
#pragma unroll
    for (int i = 0; i < 17; ++i) d[i] = q[i];
}

// x[n]..x[n+31] をレジスタへ先読み (kernels_moe_v4.cu と同じ形)。
__device__ __forceinline__ void dno_v5_prefetch32(const float* __restrict__ xb, float* xv)
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
__device__ __forceinline__ unsigned int dno_v5_u4(uint4 v, int i)
{
    return (i == 0) ? v.x : (i == 1) ? v.y : (i == 2) ? v.z : v.w;
}

// scales[12] (blk+4 から 12 バイト) の k バイト目。k はコンパイル時定数で呼ぶ。
__device__ __forceinline__ unsigned int dno_v5_scale_byte(uint4 hdr, int k)
{
    return (dno_v5_u4(hdr, 1 + (k >> 2)) >> ((k & 3) * 8)) & 0xffu;
}

// 行番号とレーン番号 (v4 と同じ規約: lane = threadIdx.x & (lanes-1))。
__device__ __forceinline__ int dno_v5_row_of(int rows_per_block, int lanes, int* lane)
{
    const int t = threadIdx.x;
    *lane = t & (lanes - 1);
    return blockIdx.x * rows_per_block + (t / lanes);
}

// シャッフルのマスク。1 ブロックが 1 warp 未満のときは実在するスレッドだけ。
__device__ __forceinline__ unsigned int dno_v5_mask()
{
    const unsigned int n = blockDim.x;
    return (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
}

// lanes レーンの合計 (offset 1, 2, 4, ... の log2(lanes) 段)。
__device__ __forceinline__ float dno_v5_reduce(float v, int lanes, unsigned int mask)
{
    for (int off = 1; off < lanes; off <<= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Q8_0: d(2B, f16) + qs[32] = 34 B / 32 要素。
//
// 2 ブロック = 68 B を 17 ワードで読む (u[17])。中身の位置:
//   u[0]  = d0(2B) | q0[0..1]        u[1..7]  = q0[2..29]
//   u[8]  = q0[30..31] | d1(2B)      u[9..16] = q1[0..31]
// したがって q0[j] は 68 B バッファのバイト (2+j)、q1[j] はバイト (36+j)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q8_0_v5(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = dno_v5_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = dno_v5_mask();

    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.0f;

    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 5;      // 32 要素ブロック数 (偶数)
        const int np = nb >> 1;        // ブロック対の数
        const int per = np / lanes;    // 1 レーンの対数
        const int p0 = lane * per;
        unsigned int nxt[17];
        dno_v5_load17(nxt, rowp + (size_t)p0 * 68u);
        for (int p = p0; p < p0 + per; ++p) {
            unsigned int cur[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) cur[i] = nxt[i];
            // 次の対のロードを今の対の計算より先に発行する (ソフトウェア先読み)。
            if (p + 1 < p0 + per) {
                dno_v5_load17(nxt, rowp + (size_t)(p + 1) * 68u);
            }
            const float d0 = dno_v5_f16_bits(cur[0] & 0xffffu);
            const float d1 = dno_v5_f16_bits((cur[8] >> 16) & 0xffffu);
            const float* xb = x + (size_t)p * 64u;
            float xv[32];
            dno_v5_prefetch32(xb, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const int q = dno_v5_i8(cur[(2 + j) >> 2], (2 + j) & 3);
                const float y = __fmul_rn((float)q, d0);
                acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(y, xv[j]));
            }
            dno_v5_prefetch32(xb + 32, xv);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const int q = dno_v5_i8(cur[9 + (j >> 2)], j & 3);
                const float y = __fmul_rn((float)q, d1);
                acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(y, xv[j]));
            }
        }
    }
    float v = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
    v = dno_v5_reduce(v, lanes, mask);
    if (valid && lane == 0) out[row] = v;
}

// ---------------------------------------------------------------------------
// Q4_K: d(2B) + dmin(2B) + scales[12] + qs[128] = 144 B / 256 要素。
//
// 144 B は 16 B の倍数なので、ヘッダ (d/dmin/scales = 16 B) と qs (128 B) を
// uint4 で読める。qs は 64 要素あたり 32 B で、サブブロック s (32 要素) は
// qs の (s>>1)*32 バイト目から 32 B = uint4 2 本。前半 32 要素が下位ニブル、
// 後半が上位ニブル (sh = (s&1) ? 4 : 0)。
//
// レーンは **スーパーブロック単位** で分ける (nb / lanes 個ずつ)。端数の
// スーパーブロックを読まないので、v4 の「レーン区間に重なるブロックを全部
// 読む」形より読み出しが少ない。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_gemv_q4k_v5(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_in, int n_out, int row_bytes, int rows_per_block, int lanes)
{
    int lane;
    const int row = dno_v5_row_of(rows_per_block, lanes, &lane);
    const bool valid = (row < n_out);
    const unsigned int mask = dno_v5_mask();

    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.0f;

    if (valid) {
        const unsigned char* rowp = w + (size_t)row * (size_t)row_bytes;
        const int nb = n_in >> 8;      // 256 要素スーパーブロック数
        const int per = nb / lanes;    // 1 レーンのスーパーブロック数
        for (int b = lane * per; b < (lane + 1) * per; ++b) {
            const unsigned char* blk = rowp + (size_t)b * 144u;
            const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            const uint4* q4 = reinterpret_cast<const uint4*>(blk + 16);
            uint4 qv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[i] = q4[i];
            const float d = dno_v5_f16_bits(hdr.x & 0xffffu);
            const float dmin = dno_v5_f16_bits(hdr.x >> 16);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                // scales[12] は hdr.y (0..3) / hdr.z (4..7) / hdr.w (8..11)。
                // k<4: sc = q[k] & 63, mn = q[k+4] & 63
                // k>=4: sc = (q[k+4] & 0x0f) | ((q[k-4] >> 6) << 4)
                //       mn = (q[k+4] >> 4)  | ((q[k]   >> 6) << 4)
                unsigned int sc, mn;
                if (s < 4) {
                    sc = dno_v5_scale_byte(hdr, s) & 63u;
                    mn = dno_v5_scale_byte(hdr, s + 4) & 63u;
                } else {
                    sc = (dno_v5_scale_byte(hdr, s + 4) & 0x0fu)
                       | ((dno_v5_scale_byte(hdr, s - 4) >> 6) << 4);
                    mn = (dno_v5_scale_byte(hdr, s + 4) >> 4)
                       | ((dno_v5_scale_byte(hdr, s) >> 6) << 4);
                }
                const float ds = __fmul_rn(d, (float)sc);
                const float ms = __fmul_rn(dmin, (float)mn);
                const uint4 qa = qv[(s >> 1) * 2];
                const uint4 qb = qv[(s >> 1) * 2 + 1];
                const unsigned int sh = (s & 1) ? 4u : 0u;
                const float* xb = x + (size_t)b * 256u + (size_t)s * 32u;
                float xv[32];
                dno_v5_prefetch32(xb, xv);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const uint4 qq = (l < 16) ? qa : qb;
                    const int li = (l < 16) ? l : (l - 16);
                    const unsigned int word = dno_v5_u4(qq, li >> 2);
                    const int nib = (int)((word >> (((li & 3) * 8) + sh)) & 0x0fu);
                    const float y = __fsub_rn(__fmul_rn(ds, (float)nib), ms);
                    acc[l & 3] = __fadd_rn(acc[l & 3], __fmul_rn(y, xv[l]));
                }
            }
        }
    }
    float v = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
    v = dno_v5_reduce(v, lanes, mask);
    if (valid && lane == 0) out[row] = v;
}
