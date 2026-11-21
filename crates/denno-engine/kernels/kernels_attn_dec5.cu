// task 520: decode の full attention (flash decoding) 第 3 版。
// 第 2 版 (`kernels_attn_dec2.cu` の `dno_attn_flashdec_t`) の **key ループの
// 依存鎖と命令数だけ**を変える。分割・部分和の置き場・縮約順・1 key あたりの
// 演算列は変えないので **値は第 2 版とビット一致**する。
//
// 動機 (task 520 の段プロファイル):
//   lab の `[dec-kprof-avg]` で `dno_attn_flashdec_t` は 14k で 66.4 us/層、
//   2k で 29.3 us/層。key 数は 7 倍違うのに時間は 2.3 倍しか違わない
//   = **1 key あたりの単価がほぼ一定 (458 -> 519 ns)**。つまり帯域律速ではなく
//   **命令発行と依存鎖**で決まっている (block 内は 12 warp = 12 q head が同じ
//   K/V を shared から読み、1 key あたり 4 本の ld.shared.v4 と 32 個の
//   float 演算 + 5 段の shfl 縮約を回す)。バイトを減らす (f16 KV) より、
//   1 key あたりの命令を減らすほうが効く。
//
// 変えたこと (どれも算術の列を 1 命令も変えない):
//   1. **2 key を同時に処理する**。j と j+1 の内積 (fmaf 鎖 x 2) と
//      shfl 縮約 (5 段 x 2) を交互に出す。online softmax の更新は
//      従来どおり j -> j+1 の順で行う (m_run / l_run / acc の更新列は同一)。
//   2. **max が更新されない key では acc の乗算を省く**。従来は
//      `acc = acc*corr + p*v` を corr = 1.0 でも計算していた (3 演算/要素)。
//      このカーネルは `-fmad=false` でビルドされるので、乗算を省いた
//      `acc = acc + p*v` は `fl(acc + fl(p*v))` と厳密に同じ値になる
//      (元の式も `fl(fl(acc*1.0) + fl(p*v))`)。1 key あたり 16 演算減る。
//   3. 実際に使う float4 ブロック数 `nb = (dh+127)/128` を先に出す。dh = 256 の
//      実形では 4 本のうち 2 本が丸ごと述語落ちだった。
//
// 数値: 決定論 (固定分割・固定順・原子操作なし) は第 2 版と同じ。
// ホスト側の `ATTN_FDEC_MAXB` と下の `DNO_FDT_MAXB` は一致させること
// (kernel_invariants が見ている)。引数は第 2 版と同じ 11 本。

#include <cstdint>

#ifndef DNO_FDT_MAXB
// 1 lane が 4 要素ずつ担当するので、1 巡で 128 要素。dh <= DNO_FDT_MAXB * 128。
#define DNO_FDT_MAXB 4
#endif

#define DNO_FDT_LOG2E 1.4426950408889634f

// acc を corr 倍してから p * v を足す (新しい max が出たとき)。
// 演算は `acc*corr + p*v` を要素ごとに 3 演算 (-fmad=false なので
// 乗算 2 本 + 加算 1 本。融合しない)。
__device__ __forceinline__ void dno_fdt2_acc_scale(float4 *acc, const float *vrow,
                                                   int lane, int dh, float corr,
                                                   float p) {
#pragma unroll
    for (int b = 0; b < DNO_FDT_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            const float4 y = *reinterpret_cast<const float4 *>(vrow + i);
            acc[b].x = acc[b].x * corr + p * y.x;
            acc[b].y = acc[b].y * corr + p * y.y;
            acc[b].z = acc[b].z * corr + p * y.z;
            acc[b].w = acc[b].w * corr + p * y.w;
        }
    }
}

// corr = 1 のときの acc 更新。`acc + p*v` は `acc*1.0 + p*v` と同値。
__device__ __forceinline__ void dno_fdt2_acc_add(float4 *acc, const float *vrow,
                                                 int lane, int dh, float p) {
#pragma unroll
    for (int b = 0; b < DNO_FDT_MAXB; ++b) {
        const int i = b * 128 + lane * 4;
        if (i < dh) {
            const float4 y = *reinterpret_cast<const float4 *>(vrow + i);
            acc[b].x = acc[b].x + p * y.x;
            acc[b].y = acc[b].y + p * y.y;
            acc[b].z = acc[b].z + p * y.z;
            acc[b].w = acc[b].w + p * y.w;
        }
    }
}

// decode の 1 トークンぶん attention の部分和 (2 key 同時版)。
//   q     [heads * dh]      (q head 順、rope 済み)
//   kc/vc [t_total, kv_dim] (kv_dim = n_kv * dh)
//   part  [(n_kv, n_split, group), dh + 2] (先頭 2 要素が m, l)
// grid = (n_split, n_kv) / block = (32 * group) / 動的 smem = 2 * tile * dh * 4 B。
extern "C" __global__ void dno_attn_flashdec_t2(
    const float *__restrict__ q,   // [heads * dh]
    const float *__restrict__ kc,  // [t_total, kv_dim]
    const float *__restrict__ vc,  // [t_total, kv_dim]
    float *__restrict__ part,      // [(n_kv, n_split, group), dh + 2]
    int kv_dim, int t_total, int group, int dh, float scale, int n_split,
    int tile,
    // task 520 (2): grid の入れ替え。0 = grid (n_split, n_kv) で
    // kh = blockIdx.y / sp = blockIdx.x (第 2 版と同じ)。1 = grid (n_kv, n_split)
    // で kh = blockIdx.x / sp = blockIdx.y。**計算は 1 ビットも変わらない**
    // (どの block がどの (kh, sp) を持つかだけ)。1 にすると同じ row の 2 つの
    // 1 KB 半分 (kv head 0 と 1) を隣り合う block が同時に読むので、DRAM から見た
    // アクセスの粒度が 1 KB から 2 KB になる。
    int khx) {
    // [tile * dh] が K、その次の [tile * dh] が V。
    extern __shared__ float sh[];
    float *sk = sh;
    float *sv = sh + (size_t)tile * (size_t)dh;

    const int kh = khx ? (int)blockIdx.x : (int)blockIdx.y;  // kv head
    const int sp = khx ? (int)blockIdx.y : (int)blockIdx.x;  // t の分割
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
        int j = 0;
        // 2 key ずつ: 内積と縮約を 2 本ずつ交互に出して依存鎖を埋める。
        for (; j + 1 < n; j += 2) {
            const float *ka = sk + (size_t)j * (size_t)dh;
            const float *kb = sk + (size_t)(j + 1) * (size_t)dh;
            const float *va = sv + (size_t)j * (size_t)dh;
            const float *vb = sv + (size_t)(j + 1) * (size_t)dh;
            float s0 = 0.0f;
            float s1 = 0.0f;
#pragma unroll
            for (int b = 0; b < DNO_FDT_MAXB; ++b) {
                const int i = b * 128 + lane * 4;
                if (i < dh) {
                    const float4 x0 = *reinterpret_cast<const float4 *>(ka + i);
                    const float4 x1 = *reinterpret_cast<const float4 *>(kb + i);
                    // 積和の順は第 2 版と同一 (b 昇順、x,y,z,w)。
                    s0 = fmaf(qv[b].x, x0.x, s0);
                    s1 = fmaf(qv[b].x, x1.x, s1);
                    s0 = fmaf(qv[b].y, x0.y, s0);
                    s1 = fmaf(qv[b].y, x1.y, s1);
                    s0 = fmaf(qv[b].z, x0.z, s0);
                    s1 = fmaf(qv[b].z, x1.z, s1);
                    s0 = fmaf(qv[b].w, x0.w, s0);
                    s1 = fmaf(qv[b].w, x1.w, s1);
                }
            }
            // lane 内和を xor 縮約 (全 lane がビット同一の値を受け取る)。
            s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 4);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
            s1 += __shfl_xor_sync(0xffffffffu, s1, 1);

            // --- key j (第 2 版と同じ列) ---
            const float sc0 = s0 * scale;
            // warp 内で一様な分岐 (sc も m_run も warp で同じ値)。
            if (sc0 > m_run) {
                const float corr = exp2f((m_run - sc0) * DNO_FDT_LOG2E);
                m_run = sc0;
                l_run = l_run * corr + 1.0f;  // exp2f(0) と厳密に同じ
                dno_fdt2_acc_scale(acc, va, lane, dh, corr, 1.0f);
            } else {
                const float p_ = exp2f((sc0 - m_run) * DNO_FDT_LOG2E);
                l_run = l_run + p_;  // 乗算 1.0 は厳密 (従来と同値)
                dno_fdt2_acc_add(acc, va, lane, dh, p_);
            }
            // --- key j+1 ---
            const float sc1 = s1 * scale;
            if (sc1 > m_run) {
                const float corr = exp2f((m_run - sc1) * DNO_FDT_LOG2E);
                m_run = sc1;
                l_run = l_run * corr + 1.0f;
                dno_fdt2_acc_scale(acc, vb, lane, dh, corr, 1.0f);
            } else {
                const float p_ = exp2f((sc1 - m_run) * DNO_FDT_LOG2E);
                l_run = l_run + p_;
                dno_fdt2_acc_add(acc, vb, lane, dh, p_);
            }
        }
        // 端数 (n が奇数) は第 2 版と同一の 1 key 経路。
        for (; j < n; ++j) {
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
            s_ += __shfl_xor_sync(0xffffffffu, s_, 16);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 8);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 4);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 2);
            s_ += __shfl_xor_sync(0xffffffffu, s_, 1);
            const float sc_ = s_ * scale;
            if (sc_ > m_run) {
                const float corr = exp2f((m_run - sc_) * DNO_FDT_LOG2E);
                m_run = sc_;
                l_run = l_run * corr + 1.0f;
                dno_fdt2_acc_scale(acc, vrow, lane, dh, corr, 1.0f);
            } else {
                const float p_ = exp2f((sc_ - m_run) * DNO_FDT_LOG2E);
                l_run = l_run + p_;
                dno_fdt2_acc_add(acc, vrow, lane, dh, p_);
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
