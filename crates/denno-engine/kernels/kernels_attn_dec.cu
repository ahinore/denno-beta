// task 160: decode の full attention を flash decoding (split-K) で回すカーネル。
//
// 動機 (task 158 の lab t158a / task 160 の前提):
//   full attention 12 層は 14,139 文脈で 1 層 631 us、実効 101 GB/s。カードは
//   1.8 TB/s なので、K/V を 1 回だけ読む形にすれば 1 層 80-160 us が射程に入る。
//   既存 `dno_attention_decode` (kernels_elem.cu) は 1 block = (split, q head) で
//   GQA の group 本の q head が同じ KV 行を group 回読み、block も 4 warp しか
//   無いのでレイテンシ律速になっている。
//
// 設計:
//   - grid = (n_split, n_kv)。1 block = (kv head, t の分割)。
//   - block の 1 warp = その kv head を共有する q head 1 本 (blockDim = 32*group)。
//     q はレジスタに 1 回だけ読み、online softmax の状態 (m, l, acc) も warp 内で
//     閉じる = **block 内の同期が 0 回**。warp 同士は同じ行を L1 で共有する
//     (DRAM からは 1 行 1 回)。
//   - K/V の行は 16 バイト (float4) で読む。1 warp の 1 命令が 128 要素を覆う。
//   - **2 段のソフトウェアパイプライン**: key j を計算している間に key j+1 の
//     K と V を読む。実測 (lab t160a) では素直な実装だと 1 key = K の待ち +
//     V の待ちの 2 回の DRAM レイテンシになり、14k で 258 us (212 GB/s) だった。
//     K と V を同じ反復で出し、さらに 1 key 先を読むことで待ちを計算の裏へ隠す。
//   - 分割は (t_total, n_split) だけで決まる固定分割、縮約は split 昇順なので
//     **決定論** (原子操作なし)。CPU とのビット一致はしない (attention の緩和枠)。
//   - 途中結果は part[(kv head, split, q head)] に [m, l, acc(dh)] で書く。
//     2 発目の `dno_attn_flashdec_combine` が split 昇順に畳んで att を書く。
//
// 数値: exp は exp2f (MUFU.EX2、prefill / 既存 split と同じ)。丸め順は
// `dno_attention_decode` と違ってよいが、同じ入力で常に同じ値になること
// (固定分割・固定順) を守る。ホスト側の `ATTN_FDEC_MAXB` と下の `DNO_FD_MAXB`
// は一致させること (kernel_invariants が見ている)。

#include <cstdint>

#ifndef DNO_FD_MAXB
// 1 lane が 4 要素ずつ担当するので、1 巡で 128 要素。dh <= DNO_FD_MAXB * 128。
#define DNO_FD_MAXB 4
#endif

#define DNO_FD_LOG2E 1.4426950408889634f

// key 1 本ぶんの K と V をまとめてレジスタへ読む (2 つの独立ロードを同時に出す)。
#define DNO_FD_LOAD(JJ, KBUF, VBUF)                                              \
    do {                                                                         \
        const float *krow_ = kc + (size_t)(JJ) * kv_dim + kh_off;                \
        const float *vrow_ = vc + (size_t)(JJ) * kv_dim + kh_off;                \
        _Pragma("unroll") for (int b_ = 0; b_ < DNO_FD_MAXB; ++b_) {             \
            const int i_ = b_ * 128 + lane * 4;                                  \
            if (i_ < dh) {                                                       \
                KBUF[b_] = *reinterpret_cast<const float4 *>(krow_ + i_);        \
                VBUF[b_] = *reinterpret_cast<const float4 *>(vrow_ + i_);        \
            }                                                                    \
        }                                                                        \
    } while (0)

// 読み込み済みの K/V で 1 key ぶんの online softmax と acc 更新を行う。
#define DNO_FD_KEY(KBUF, VBUF)                                                   \
    do {                                                                         \
        float s_ = 0.0f;                                                         \
        _Pragma("unroll") for (int b_ = 0; b_ < DNO_FD_MAXB; ++b_) {              \
            const int i_ = b_ * 128 + lane * 4;                                  \
            if (i_ < dh) {                                                       \
                s_ = fmaf(qv[b_].x, KBUF[b_].x, s_);                              \
                s_ = fmaf(qv[b_].y, KBUF[b_].y, s_);                              \
                s_ = fmaf(qv[b_].z, KBUF[b_].z, s_);                              \
                s_ = fmaf(qv[b_].w, KBUF[b_].w, s_);                              \
            }                                                                    \
        }                                                                        \
        /* lane 内和を xor 縮約 (全 lane がビット同一の値を受け取る) */            \
        s_ += __shfl_xor_sync(0xffffffffu, s_, 16);                              \
        s_ += __shfl_xor_sync(0xffffffffu, s_, 8);                               \
        s_ += __shfl_xor_sync(0xffffffffu, s_, 4);                               \
        s_ += __shfl_xor_sync(0xffffffffu, s_, 2);                               \
        s_ += __shfl_xor_sync(0xffffffffu, s_, 1);                               \
        const float sc_ = s_ * scale;                                            \
        /* warp 内で一様な分岐 (sc も m_run も warp で同じ値) */                  \
        float corr_;                                                             \
        float p_;                                                                \
        if (sc_ > m_run) {                                                       \
            corr_ = exp2f((m_run - sc_) * DNO_FD_LOG2E);                         \
            m_run = sc_;                                                         \
            p_ = 1.0f; /* exp2f(0) と厳密に同じ */                               \
        } else {                                                                 \
            corr_ = 1.0f;                                                        \
            p_ = exp2f((sc_ - m_run) * DNO_FD_LOG2E);                            \
        }                                                                        \
        l_run = l_run * corr_ + p_;                                               \
        _Pragma("unroll") for (int b_ = 0; b_ < DNO_FD_MAXB; ++b_) {              \
            const int i_ = b_ * 128 + lane * 4;                                  \
            if (i_ < dh) {                                                       \
                acc[b_].x = acc[b_].x * corr_ + p_ * VBUF[b_].x;                 \
                acc[b_].y = acc[b_].y * corr_ + p_ * VBUF[b_].y;                 \
                acc[b_].z = acc[b_].z * corr_ + p_ * VBUF[b_].z;                 \
                acc[b_].w = acc[b_].w * corr_ + p_ * VBUF[b_].w;                 \
            }                                                                    \
        }                                                                        \
    } while (0)

// decode の 1 トークンぶん attention の部分和。
//   q     [heads * dh]   (q head 順、rope 済み)
//   kc/vc [t_total, kv_dim]  (kv_dim = n_kv * dh)
//   part  [(n_kv * n_split * group), dh + 2]  (先頭 2 要素が m, l)
// grid = (n_split, n_kv) / block = (32 * group) / smem なし。
extern "C" __global__ void dno_attn_flashdec(
    const float *__restrict__ q,   // [heads * dh]
    const float *__restrict__ kc,  // [t_total, kv_dim]
    const float *__restrict__ vc,  // [t_total, kv_dim]
    float *__restrict__ part,      // [(n_kv, n_split, group), dh + 2]
    int kv_dim, int t_total, int group, int dh, float scale, int n_split) {
    const int kh = blockIdx.y;        // kv head
    const int sp = blockIdx.x;        // t の分割
    const int hh = threadIdx.x >> 5;  // この kv head を共有する q head (warp)
    const int lane = threadIdx.x & 31;
    const int kh_off = kh * dh;

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
    float4 qv[DNO_FD_MAXB];
    float4 acc[DNO_FD_MAXB];
#pragma unroll
    for (int b = 0; b < DNO_FD_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            qv[b] = *reinterpret_cast<const float4 *>(qp + i);
        } else {
            qv[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        acc[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // online softmax の状態 (warp 内で全 lane が同じ値を持つ)。
    // -FLT_MAX 始まり: 最初の key で corr = exp2f(-inf) = 0 になり、NaN にならない。
    float m_run = -3.402823466e+38f;
    float l_run = 0.0f;

    // 2 段パイプライン: j を A、j+1 を B に持ち、j の計算中に j+1 の K/V が飛ぶ。
    float4 kA[DNO_FD_MAXB], vA[DNO_FD_MAXB];
    float4 kB[DNO_FD_MAXB], vB[DNO_FD_MAXB];
    int j = jlo;
    if (j < jhi) DNO_FD_LOAD(j, kA, vA);
    if (j + 1 < jhi) DNO_FD_LOAD(j + 1, kB, vB);
    while (j < jhi) {
        DNO_FD_KEY(kA, vA);
        ++j;
        if (j < jhi) DNO_FD_LOAD(j + 1, kA, vA);  // 次の次を A の枠へ (j は B を使う)
        if (j >= jhi) break;
        DNO_FD_KEY(kB, vB);
        ++j;
        if (j < jhi) DNO_FD_LOAD(j + 1, kB, vB);
    }
    // 上のループは A が j, B が j+1 の順に使うので、j+1 を A の枠へ読むのは
    // 「j+2 を A に入れる」と同義 (A は j で使い終わっている)。末尾の半端は
    // ループの j < jhi 判定で落ちる。

    // 部分和 (acc は m_run を基準にした未正規化の和)。空 split は m = -FLT_MAX。
    float *pp = part + ((size_t)(kh * n_split + sp) * group + hh) * (size_t)(dh + 2);
    if (lane == 0) {
        pp[0] = m_run;
        pp[1] = l_run;
    }
#pragma unroll
    for (int b = 0; b < DNO_FD_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            pp[2 + i + 0] = acc[b].x;
            pp[2 + i + 1] = acc[b].y;
            pp[2 + i + 2] = acc[b].z;
            pp[2 + i + 3] = acc[b].w;
        }
    }
}

// part を split 昇順に畳む。grid = (group, n_kv) / block = (dh)。
// 分割数・head 数はホストが決める (どちらも固定なので順序は決定論的)。
extern "C" __global__ void dno_attn_flashdec_combine(
    const float *__restrict__ part,  // [(n_kv, n_split, group), dh + 2]
    float *__restrict__ att,         // [heads * dh]
    int group, int dh, int n_split) {
    const int kh = blockIdx.y;
    const int hh = blockIdx.x;
    const int d = threadIdx.x;
    const int h = kh * group + hh;
    const size_t stride = (size_t)(dh + 2);
    const size_t step = (size_t)group * stride;
    const float *base = part + ((size_t)(kh * n_split) * group + hh) * stride;

    float mg = -3.402823466e+38f;
    for (int sp = 0; sp < n_split; ++sp) mg = fmaxf(mg, base[(size_t)sp * step]);
    float denom = 0.0f;
    float num = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        const float *pp = base + (size_t)sp * step;
        // 空 split は m = -FLT_MAX → e = 0 で寄与しない。
        const float e = exp2f((pp[0] - mg) * DNO_FD_LOG2E);
        denom = fmaf(pp[1], e, denom);
        num = fmaf(pp[2 + d], e, num);
    }
    att[(size_t)h * dh + d] = (denom > 0.0f) ? __fdiv_rn(num, denom) : 0.0f;
}
