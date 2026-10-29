// denno CUDA grouped expert GEMM (P40) — Flash-Next (qwen4exp) の prefill で
// 「重みが expert ごとに違い、トークン数 m も expert ごとに違う」GEMM を
// 1 起動にまとめる。
//
// PTX へのコンパイル (メインセッションの担当。このコンテナには nvcc を置かない):
//
//   nvcc -ptx -arch=compute_75 -O3 kernels_gemm_grouped.cu -o kernels_gemm_grouped.ptx
//
// 数値契約: **なし** (prefill は 2026-10-28 のオペレーター指示でビット一致契約の
// 外)。ここは FMA 融合 (-fmad=true 既定) を使う。比較は argmax / max abs diff で
// 行う (ホスト側 `DENNO_MOE_GROUPED_VERIFY=1` が CPU 参照と並べて出す)。
//
// レイアウトは `kernels_moe.cu` と同じ ggml ブロック仕様 (llama.cpp, MIT)。
// デコード規則は本リポジトリの `src/kquant.rs` と同一。f16 / 6bit scale-min の
// ヘルパは `kernels_gemm.cu` から写した (PTX はファイル単位で独立したモジュールに
// なるため、別 .cu からは共有できない)。
//
// 前提:
//   - 重みは **パック済み** (`model::moe_grouped::pack_expert_weights`)。
//     レイアウトは [gate(e0..eN)][up(e0..eN)][down(e0..eN)] の 3 ブロックで、
//     `seg[i].w_off` は「そのブロック内の expert 先頭バイト」。
//   - x / act / h は segment の並びに詰め直した行優先 (m_total × 幅)。
//   - ブロック単位の量子型だけを扱う (Q8_0 32 / Q5_1 32 / Q4_K 256 / Q5_K 256)。
//
// segment 表 (`const unsigned int* seg`、1 segment = 5 個):
//
//   seg[s*5 + 0] = expert_id   (カーネルでは未使用。ログ・突き合わせ用)
//   seg[s*5 + 1] = m            (この expert のトークン数)
//   seg[s*5 + 2] = w_off        (パック済み重みブロック内の先頭バイト)
//   seg[s*5 + 3] = e_off        (Σ m * n_embd。gate/up の x 先頭)
//   seg[s*5 + 4] = ff_off       (Σ m * n_ff。gate/up の出力先頭 = down の入力先頭)
//
// gate_up カーネルは (x = e_off, out = ff_off)、down カーネルは (x = ff_off,
// out = e_off) を使う。**1 枚の表で 2 パスを回す**ため、行数が同じという性質
// (gate/up の n_out == down の n_in == n_ff) を前提にしている。
//
// ブロック割当:
//   blockDim = (16, 16) = 256 スレッド。tx = 出力行 (16 行/タイル)、
//   lane = トークン (16 本/タイル)。
//   grid.x = segment 番号 (nseg)、grid.y = 行タイル (ceil(n_out/16))、
//   grid.z = トークンタイル ((blockIdx.z*16) が segment 内の先頭トークン)。
//   m < 16 の expert (実形では平均 10) では z 方向の余分なブロックが即 return する。
//
// 共有メモリ:
//   デコード済み f32 を「1 ブロックぶん」だけ置く。生バイトは global から直接読む。
//     gate_up: 2 × 16 × blkj × 4 B = Q4_K/Q5_K で 32 KiB、Q8_0/Q5_1 で 4 KiB
//     down   : 1 × 16 × blkj × 4 B = Q4_K/Q5_K で 16 KiB、Q8_0/Q5_1 で 2 KiB
//   (過去に 62 KiB を要求して 48 KiB 既定を超え、実モデルで 1 度も発火しなかった
//    経緯があるので、2 パスに割って 32 KiB に収めた)。

#define GR_ROWS 16
#define GR_TG 16

// ---------------------------------------------------------------------------
// f16 -> f32 (kernels_gemm.cu と同一。IEEE 準拠)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dno_g_f16(const unsigned char* b)
{
    unsigned int h = (unsigned int)b[0] | ((unsigned int)b[1] << 8);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1fu;
    unsigned int man = h & 0x3ffu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((man & 0x400u) == 0) { man <<= 1; --e; }
            man &= 0x3ffu;
            bits = sign | ((unsigned int)(e + 127) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    return __int_as_float(bits);
}

// Q4_K / Q5_K の 6bit scale/min 展開 (kquant.rs の get_scale_min_k4 と同一)
__device__ __forceinline__ void dno_g_scale_min_k4(int j, const unsigned char* q,
                                                   unsigned int* d, unsigned int* m)
{
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((unsigned int)(q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((unsigned int)(q[j] >> 6) << 4);
    }
}

// 1 ブロックの j 番目 (j < blkj) を f32 へ。
//   wtype: 0 = Q8_0 (32 要素/34 B)、1 = Q5_1 (32/24)、2 = Q4_K (256/144)、
//          3 = Q5_K (256/176)
__device__ __forceinline__ float dno_gr_deq(const unsigned char* q, int wtype, int j)
{
    if (wtype == 0) {
        // Q8_0: d(2B, f16) + qs[32]
        const float d = dno_g_f16(q);
        return (float)(int)(signed char)q[2 + j] * d;
    }
    if (wtype == 1) {
        // Q5_1: d(2B) + m(2B) + qh[4] + qs[16]
        const float d = dno_g_f16(q);
        const float m = dno_g_f16(q + 2);
        const unsigned int qh = (unsigned int)q[4] | ((unsigned int)q[5] << 8)
                              | ((unsigned int)q[6] << 16) | ((unsigned int)q[7] << 24);
        const int xh = (int)(((qh >> j) & 1u) << 4);
        const int qv = (j < 16) ? ((int)(q[8 + j] & 0x0fu) | xh)
                                : ((int)(q[8 + j - 16] >> 4) | xh);
        return (float)qv * d + m;
    }
    // Q4_K / Q5_K: d(2B) + dmin(2B) + scales[12] + (Q5_K のみ qh[32]) + qs[128]
    const float d = dno_g_f16(q);
    const float dmin = dno_g_f16(q + 2);
    const int l = j & 63;
    const unsigned char* qs = q + ((wtype == 2) ? 16 : 48) + ((j >> 6) << 5);
    int qv = (l < 32) ? (int)(qs[l] & 0x0fu) : (int)(qs[l - 32] >> 4);
    if (wtype == 3) {
        qv += (int)(((unsigned int)q[16 + (j & 31)] >> (j >> 5)) & 1u) << 4;
    }
    unsigned int sc, mn;
    dno_g_scale_min_k4(j >> 5, q + 4, &sc, &mn);
    return (d * (float)sc) * (float)qv - (dmin * (float)mn);
}

// ---------------------------------------------------------------------------
// gate + up + silu 積: act[o_off + t*n_out + row] = silu(gate) * up
//   2 本の重みブロック (gate, up) を同時にデコードして 1 回の x 読みで 2 本の
//   内積を取る (smem は 2 倍になるが、x の読み直しとバリアが半分で済む)。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_grouped_gate_up(
    const unsigned char* __restrict__ wg,
    const unsigned char* __restrict__ wu,
    const float* __restrict__ x,
    float* __restrict__ act,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes,
    int wtype, int blkj, int blkb)
{
    extern __shared__ unsigned char smem[];
    float* sG = (float*)smem;                  // [GR_ROWS][blkj]
    float* sU = sG + GR_ROWS * blkj;           // [GR_ROWS][blkj]

    const int s = (int)blockIdx.x;
    if (s >= nseg) return;
    const unsigned int* sg = seg + (size_t)s * 5u;
    const int m = (int)sg[1];
    const unsigned int w_off = sg[2];
    const unsigned int x_off = sg[3];   // gate/up: 入力は e_off
    const unsigned int o_off = sg[4];   // gate/up: 出力は ff_off

    const int row0 = (int)blockIdx.y * GR_ROWS;
    if (row0 >= n_out) return;
    const int t0 = (int)blockIdx.z * GR_TG;
    if (t0 >= m) return;
    const int nt = min(GR_TG, m - t0);

    const int tx = (int)threadIdx.x;    // 出力行 (タイル内)
    const int lane = (int)threadIdx.y;  // トークン (タイル内)
    const int tid = lane * GR_ROWS + tx;
    const int nrow = min(GR_ROWS, n_out - row0);
    const int nb = n_in / blkj;

    float accg = 0.0f;
    float accu = 0.0f;
    const bool active = (tx < nrow) && (lane < nt);
    const size_t xbase = (size_t)x_off + (size_t)(t0 + lane) * (size_t)n_in;

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GR_ROWS * blkj; i += GR_ROWS * GR_TG) {
            const int r = i / blkj;
            const int j = i - r * blkj;
            if (row0 + r >= n_out) {
                sG[i] = 0.0f;
                sU[i] = 0.0f;
                continue;
            }
            const size_t wo = (size_t)w_off
                            + (size_t)(row0 + r) * (size_t)row_bytes
                            + (size_t)b * (size_t)blkb;
            sG[i] = dno_gr_deq(wg + wo, wtype, j);
            sU[i] = dno_gr_deq(wu + wo, wtype, j);
        }
        __syncthreads();
        if (active) {
            const float* g = sG + tx * blkj;
            const float* u = sU + tx * blkj;
            const float* xb = x + xbase + (size_t)b * (size_t)blkj;
            for (int j = 0; j < blkj; ++j) {
                const float xv = xb[j];
                accg = fmaf(g[j], xv, accg);
                accu = fmaf(u[j], xv, accu);
            }
        }
        __syncthreads();
    }
    if (active) {
        const float silu = accg / (1.0f + expf(-accg));
        act[(size_t)o_off + (size_t)(t0 + lane) * (size_t)n_out + (size_t)row0 + (size_t)tx]
            = silu * accu;
    }
}

// ---------------------------------------------------------------------------
// down: out[o_off + t*n_out + row] = W_down(row) · act[t]
//   gate_up の出力 (act = silu(g)*u) を segment の並びのまま受け取る。
//   表の使い方が逆 (x = ff_off, out = e_off) なだけで、割当は gate_up と同じ。
// ---------------------------------------------------------------------------
extern "C" __global__ void dno_moe_grouped_down(
    const unsigned char* __restrict__ wd,
    const float* __restrict__ x,
    float* __restrict__ out,
    const unsigned int* __restrict__ seg,
    int nseg, int n_in, int n_out, int row_bytes,
    int wtype, int blkj, int blkb)
{
    extern __shared__ unsigned char smem[];
    float* sW = (float*)smem;                  // [GR_ROWS][blkj]

    const int s = (int)blockIdx.x;
    if (s >= nseg) return;
    const unsigned int* sg = seg + (size_t)s * 5u;
    const int m = (int)sg[1];
    const unsigned int w_off = sg[2];
    const unsigned int x_off = sg[4];   // down: 入力は ff_off (gate_up の出力)
    const unsigned int o_off = sg[3];   // down: 出力は e_off

    const int row0 = (int)blockIdx.y * GR_ROWS;
    if (row0 >= n_out) return;
    const int t0 = (int)blockIdx.z * GR_TG;
    if (t0 >= m) return;
    const int nt = min(GR_TG, m - t0);

    const int tx = (int)threadIdx.x;
    const int lane = (int)threadIdx.y;
    const int tid = lane * GR_ROWS + tx;
    const int nrow = min(GR_ROWS, n_out - row0);
    const int nb = n_in / blkj;

    float acc = 0.0f;
    const bool active = (tx < nrow) && (lane < nt);
    const size_t xbase = (size_t)x_off + (size_t)(t0 + lane) * (size_t)n_in;

    for (int b = 0; b < nb; ++b) {
        for (int i = tid; i < GR_ROWS * blkj; i += GR_ROWS * GR_TG) {
            const int r = i / blkj;
            const int j = i - r * blkj;
            if (row0 + r >= n_out) {
                sW[i] = 0.0f;
                continue;
            }
            const size_t wo = (size_t)w_off
                            + (size_t)(row0 + r) * (size_t)row_bytes
                            + (size_t)b * (size_t)blkb;
            sW[i] = dno_gr_deq(wd + wo, wtype, j);
        }
        __syncthreads();
        if (active) {
            const float* w = sW + tx * blkj;
            const float* xb = x + xbase + (size_t)b * (size_t)blkj;
            for (int j = 0; j < blkj; ++j) {
                acc = fmaf(w[j], xb[j], acc);
            }
        }
        __syncthreads();
    }
    if (active) {
        out[(size_t)o_off + (size_t)(t0 + lane) * (size_t)n_out + (size_t)row0 + (size_t)tx] = acc;
    }
}
