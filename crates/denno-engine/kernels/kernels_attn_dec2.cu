// task 448: decode の full attention (flash decoding) 第 2 版。
// 分割・縮約順・1 key あたりの演算は `kernels_attn_dec.cu` の
// `dno_attn_flashdec` と同一で、**K/V の読み方だけ**を変える。
//
// 動機 (task 448 の前提 / REPORT_P410 §0-7):
//   t441a の `[dec-kprof-avg]` で 14k の 1 層は flashdec 86.3 us + combine 16.2 us。
//   KV は 1 層 55 MB なので **637 GB/s** しか出ておらず、帯域の天井 (1.7 TB/s) の
//   37% で止まっている。v1 は 1 block = (split, kv head)・1 warp = 1 q head で、
//   block 内の 12 warp が**同じ K/V 行を 12 回**読む (L1 任せ)。1 warp が持つ
//   未完了ロードも 2 key ぶんしかないので、レイテンシも隠れない。
//
// 設計 (v1 との差分だけ):
//   - K/V を **block の全スレッドで協調ロードして shared メモリのタイルへ**置く。
//     tile 行ずつ (行 = dh 個の float4)。読みは 1 行 1 回になり、ロードは
//     float4 x 連続スレッドで coalesced、1 スレッドが複数本を同時に出す。
//   - タイルごとに __syncthreads() を 2 回 (読み込み後 / 使い切った後)。
//   - 計算は shared から読む。**1 key の演算列は v1 と 1 命令も変えない**
//     (fmaf の順、shfl_xor 16..1、online softmax の分岐、acc の更新式)。
//     よって分割境界が同じなら part は v1 とビット一致する。
//   - 動的 shared は 2 * tile * dh * float。ホストが tile を渡す
//     (`DENNO_ATTN_FLASHDEC_TILE`、既定 16)。48 KB を超えない範囲で選ぶ。
//
// 数値: 決定論 (固定分割・固定順・原子操作なし) は v1 と同じ。
// ホスト側の `ATTN_FDEC_MAXB` と下の `DNO_FDT_MAXB` は一致させること
// (kernel_invariants が見ている)。

#include <cstdint>

#ifndef DNO_FDT_MAXB
// 1 lane が 4 要素ずつ担当するので、1 巡で 128 要素。dh <= DNO_FDT_MAXB * 128。
#define DNO_FDT_MAXB 4
#endif

#define DNO_FDT_LOG2E 1.4426950408889634f

// decode の 1 トークンぶん attention の部分和 (shared タイル版)。
//   q     [heads * dh]      (q head 順、rope 済み)
//   kc/vc [t_total, kv_dim] (kv_dim = n_kv * dh)
//   part  [(n_kv, n_split, group), dh + 2] (先頭 2 要素が m, l)
// grid = (n_split, n_kv) / block = (32 * group) / 動的 smem = 2 * tile * dh * 4 B。
extern "C" __global__ void dno_attn_flashdec_t(
    const float *__restrict__ q,   // [heads * dh]
    const float *__restrict__ kc,  // [t_total, kv_dim]
    const float *__restrict__ vc,  // [t_total, kv_dim]
    float *__restrict__ part,      // [(n_kv, n_split, group), dh + 2]
    int kv_dim, int t_total, int group, int dh, float scale, int n_split,
    int tile) {
    // [tile * dh] が K、その次の [tile * dh] が V。
    extern __shared__ float sh[];
    float *sk = sh;
    float *sv = sh + (size_t)tile * (size_t)dh;

    const int kh = blockIdx.y;        // kv head
    const int sp = blockIdx.x;        // t の分割
    const int hh = threadIdx.x >> 5;  // この kv head を共有する q head (warp)
    const int lane = threadIdx.x & 31;
    const int kh_off = kh * dh;
    const int nthreads = (int)blockDim.x;
    const int dh4 = dh >> 2;  // 1 行の float4 数 (dh は 4 の倍数)

    // 固定分割: ホストの `q4_fdec_split` と同じ式 (先頭から chunk 個ずつ)。
    const int chunk = (t_total + n_split - 1) / n_split;
    int jlo = sp * chunk;
    int jhi = jlo + chunk;
    if (jhi > t_total) jhi = t_total;
    if (jlo > t_total) jlo = t_total;

    const int h = kh * group + hh;
    const float *qp = q + (size_t)h * dh;

    // q はレジスタへ 1 回だけ。dh < 128 のときは余った lane を 0 で埋める
    // (0 を掛けるだけなので dot の値は変わらない)。
    float4 qv[DNO_FDT_MAXB];
    float4 acc[DNO_FDT_MAXB];
#pragma unroll
    for (int b = 0; b < DNO_FDT_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            qv[b] = *reinterpret_cast<const float4 *>(qp + i);
        } else {
            qv[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        acc[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // online softmax の状態 (warp 内で全 lane が同じ値を持つ)。
    float m_run = -3.402823466e+38f;
    float l_run = 0.0f;

    for (int base = jlo; base < jhi; base += tile) {
        int n = jhi - base;
        if (n > tile) n = tile;
        // 協調ロード: このタイルの n 行 x dh を float4 で詰める。
        // f は「タイル内の float4 番号」で、row = f / dh4, col4 = f % dh4。
        const int nf4 = n * dh4;
        for (int f = threadIdx.x; f < nf4; f += nthreads) {
            const int row = f / dh4;
            const int c4 = f - row * dh4;
            const size_t src = (size_t)(base + row) * (size_t)kv_dim + (size_t)kh_off +
                               (size_t)c4 * 4;
            *(reinterpret_cast<float4 *>(sk) + (size_t)f) =
                *reinterpret_cast<const float4 *>(kc + src);
            *(reinterpret_cast<float4 *>(sv) + (size_t)f) =
                *reinterpret_cast<const float4 *>(vc + src);
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            const float *krow = sk + (size_t)j * (size_t)dh;
            const float *vrow = sv + (size_t)j * (size_t)dh;
            float s_ = 0.0f;
#pragma unroll
            for (int b = 0; b < DNO_FDT_MAXB; ++b) {
                const int i = b * 128 + lane * 4;
                if (i < dh) {
                    const float4 kb = *reinterpret_cast<const float4 *>(krow + i);
                    s_ = fmaf(qv[b].x, kb.x, s_);
                    s_ = fmaf(qv[b].y, kb.y, s_);
                    s_ = fmaf(qv[b].z, kb.z, s_);
                    s_ = fmaf(qv[b].w, kb.w, s_);
                }
            }
            // lane 内和を xor 縮約 (全 lane がビット同一の値を受け取る)。
            s_ += __shfl_xor_sync(0xffffffffu, s_, 16);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 8);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 4);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 2);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 1);
            const float sc_ = s_ * scale;
            // warp 内で一様な分岐 (sc も m_run も warp で同じ値)。
            float corr_;
            float p_;
            if (sc_ > m_run) {
                corr_ = exp2f((m_run - sc_) * DNO_FDT_LOG2E);
                m_run = sc_;
                p_ = 1.0f;  // exp2f(0) と厳密に同じ
            } else {
                corr_ = 1.0f;
                p_ = exp2f((sc_ - m_run) * DNO_FDT_LOG2E);
            }
            l_run = l_run * corr_ + p_;
#pragma unroll
            for (int b = 0; b < DNO_FDT_MAXB; ++b) {
                const int i = b * 128 + lane * 4;
                if (i < dh) {
                    const float4 vb = *reinterpret_cast<const float4 *>(vrow + i);
                    acc[b].x = acc[b].x * corr_ + p_ * vb.x;
                    acc[b].y = acc[b].y * corr_ + p_ * vb.y;
                    acc[b].z = acc[b].z * corr_ + p_ * vb.z;
                    acc[b].w = acc[b].w * corr_ + p_ * vb.w;
                }
            }
        }
        __syncthreads();
    }

    // 部分和 (acc は m_run を基準にした未正規化の和)。空 split は m = -FLT_MAX。
    float *pp = part + ((size_t)(kh * n_split + sp) * group + hh) * (size_t)(dh + 2);
    if (lane == 0) {
        pp[0] = m_run;
        pp[1] = l_run;
    }
#pragma unroll
    for (int b = 0; b < DNO_FDT_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            pp[2 + i + 0] = acc[b].x;
            pp[2 + i + 1] = acc[b].y;
            pp[2 + i + 2] = acc[b].z;
            pp[2 + i + 3] = acc[b].w;
        }
    }
}
