// denno-beta P17d: elementwise kernels (bit-exact vs CPU int8 path).
//
// Design constraints (exactness contract):
// - All floating point ops replicate the CPU (Rust/glibc) op order exactly.
//   - compiled with -fmad=false and explicit __*_rn intrinsics so only the
//     intended sites are fused.
//   - expf_glibc is a faithful re-implementation of glibc's e_expf.c as
//     compiled for x86-64 with FMA (gcc -ffp-contract=fast contracts
//     kd/r/poly mul+add pairs into fma — the runtime IFUNC selects this
//     variant on FMA-capable CPUs, and Rust f32::exp calls it). Verified
//     bit-identical to CPU f32::exp on the target host (see dno_expf).
//   - rmsnorm sum is computed strictly sequentially (i=0..cols-1) by one thread,
//     matching `iter().map(|v| v*v).sum::<f32>()`.

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

/// SiLU + up multiply (swiglu): out[i] = (g / (1 + exp(-g))) * u
/// CPU: silu_inplace (v = v / (1.0 + exp(-v))) then gate[i] *= up[i].
extern "C" __global__ void dno_swiglu(
    const float *gate, const float *up, float *out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float g = gate[i];
        float e = dno_expf(-g);
        float r = __fdiv_rn(g, __fadd_rn(1.0f, e));
        out[i] = __fmul_rn(r, up[i]);
    }
}

/// P17k: [m, 2ff] の interleaved (行ごとに [gate|up]) レイアウトから swiglu。
/// gate_up grouped GEMM の出力を D2H せず直接読むための変種。
/// 演算自体は dno_swiglu と同一 (dno_expf + __fdiv_rn + __fadd_rn + __fmul_rn)
/// なので CPU 経路 (silu_inplace + gate*=up) とビット一致する。
extern "C" __global__ void dno_swiglu_int(
    const float *gateup, float *out, int ff, int width, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int r = (int)(i / ff);
        int f = (int)(i - (long long)r * ff);
        float g = gateup[(size_t)r * width + f];
        float u = gateup[(size_t)r * width + ff + f];
        float e = dno_expf(-g);
        float s = __fdiv_rn(g, __fadd_rn(1.0f, e));
        out[i] = __fmul_rn(s, u);
    }
}

/// residual add + RMSNorm (proj == NULL で単純 rmsnorm)。
/// grid = rows, block = 256, dynamic shmem = cols * 4 bytes.
/// 平方和は CPU と同一の逐次順 (i = 0..cols-1) で thread 0 が計算する。
extern "C" __global__ void dno_add_rmsnorm(
    float *x, const float *proj, const float *w, float *out,
    int cols, float eps) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    float *xr = x + (size_t)row * cols;
    // 1) residual add (要素独立なので並列でよい)
    if (proj != nullptr) {
        const float *pr = proj + (size_t)row * cols;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float v = __fadd_rn(xr[i], pr[i]);
            xr[i] = v;
            sh[i] = v;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            sh[i] = xr[i];
        }
    }
    __syncthreads();
    // 2) ss = sum(x_i^2) / cols — CPU と同じ逐次順 (thread 0 のみ)
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < cols; ++i) {
            ss = __fadd_rn(ss, __fmul_rn(sh[i], sh[i]));
        }
        ss = __fdiv_rn(ss, (float)cols);
        float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
        s_inv = inv;
    }
    __syncthreads();
    // 3) out[i] = x[i] * inv * w[i] (左結合の乗算列 — CPU と同一)
    float inv = s_inv;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[(size_t)row * cols + i] = __fmul_rn(__fmul_rn(xr[i], inv), w[i]);
    }
}

/// テスト用: dno_expf を要素ごとに適用する
extern "C" __global__ void dno_expf_probe(
    const float *in, float *out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = dno_expf(in[i]);
}

// ---------------------------------------------------------------------------
// P17g: decode attention (GQA) — exactness 契約は attention のみ緩和 (ユーザー承認)。
// 線形層 (int8) / expf / swiglu / rmsnorm のビット一致契約は変更しない。
//
// grid = (heads, 1, 1), block = (dh, 1, 1) (dh <= 1024), shared = dh * 4 bytes。
// block h が q ヘッド h の scores/softmax/probs@V を担当する。
//   kvh = h / group, kh_off = kvh * dh
//   scores[j] = q[h] · K[j, kh_off..] * scale   (j = 0..t_total)
//   att[h,i]  = Σ_j softmax(scores)[j] * V[j, kh_off + i]
// K/V は GPU 側 KV ミラー ([t_total, kv_dim] 行-major、ホスト KvCache と同一内容)。
// softmax の exp は dno_expf (glibc 再現) を使い CPU との差を最小化する
// (ビット一致は要求しない — 累積順が異なるため)。
extern "C" __global__ void dno_attention(
    const float *__restrict__ q,     // [heads * dh]
    const float *__restrict__ kc,    // [t_total, kv_dim]
    const float *__restrict__ vc,    // [t_total, kv_dim]
    float *__restrict__ att,         // [heads * dh]
    float *__restrict__ scores,      // scratch [heads * t_total]
    int kv_dim, int t_total, int group, int dh, float scale) {
    extern __shared__ float sh[];
    int h = blockIdx.x;
    int kvh = h / group;
    int kh_off = kvh * dh;
    const float *qg = q + (size_t)h * dh;
    float *sg = scores + (size_t)h * t_total;

    // phase 1: scores + max (スレッド間で t をストライド分担)
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < t_total; j += blockDim.x) {
        const float *krow = kc + (size_t)j * kv_dim + kh_off;
        float s = 0.0f;
        for (int i = 0; i < dh; ++i) s = fmaf(qg[i], krow[i], s);
        s *= scale;
        sg[j] = s;
        mx = fmaxf(mx, s);
    }
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + off]);
        __syncthreads();
    }
    mx = sh[0];

    // phase 2: exp + sum (block 足し合わせ)
    float sum = 0.0f;
    for (int j = threadIdx.x; j < t_total; j += blockDim.x) {
        float e = dno_expf(sg[j] - mx);
        sg[j] = e;
        sum += e;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    float inv = 1.0f / sh[0];

    // phase 3: att[h, threadIdx.x] = inv * Σ_j p[j] * V[j, kh_off + threadIdx.x]
    // 各反復でスレッド列が V の 1 行 (dh 連続要素) を coalesced 読み。
    // p[j] (sg[j]) は全スレッドが同じアドレスを読むので broadcast。
    float acc = 0.0f;
    for (int j = 0; j < t_total; ++j) {
        acc = fmaf(sg[j], vc[(size_t)j * kv_dim + kh_off + threadIdx.x], acc);
    }
    att[(size_t)h * dh + threadIdx.x] = acc * inv;
}

// ---------------------------------------------------------------------------
// P17j: prefill attention (n > 1, causal) — flash 風 online softmax。
// exactness 契約は decode attention (P17g) と同じく attention のみ緩和
// (ユーザー承認)。数値は CPU (sgemm 経路) とビット不一致だが、**決定論は
// 維持** (タイルサイズ・累積順固定・原子操作なし: 同一入力 → 同一出力)。
//
// grid = (heads, ceil(n / PF_BQ)), block = (dh), dynamic shmem =
//   (PF_BQ + 2*PF_BK) * dh + PF_BQ*PF_BK + 2*PF_BQ floats (<= 48KB)。
// block (h, qt) が q ヘッド h のクエリ行 [qt*PF_BQ, qt*PF_BQ+PF_BQ) を担当。
//   kvh = h / group, kh_off = kvh * dh
//   scores[i,j] = q[i,h] · K[j,kh_off..] * scale   (j = 0..pos0+i — causal)
//   att[i,h]    = Σ_j softmax_j * V[j, kh_off..]
// K/V は GPU 側 KV ミラー ([t_total, kv_dim] 行-major、ホスト KvCache と同一)。
// exp は exp2f ベース (P17k): attention は数値緩和枠 (ユーザー承認済み) のため
// glibc 完全再現の dno_expf (f64 演算主体・デスクトップ GPU では 1/64 レートで
// 増悪) をやめ、ハード命令 MUFU.EX2 の exp2f に置換する。expf(x) =
// exp2f(x * log2(e)) で置き換えるため p に ~1e-6 相対差が出るが、**決定論は
// 維持** (exp2f は固定のハード演算・同一入力に対し常に同一結果)。decode 用の
// dno_attention はビット一致枠のため dno_expf のまま変更しない。
// online softmax の累積順は「タイル昇順・タイル内 kk 昇順・出力次元 d はスレッド
// tid = d 専属」で固定。
// 制約 (ホスト側でも検証): dh >= 32 (block が PF_BQ 行 owner を持つため) &&
//   shmem <= 48KB (dh <= 152 @ BQ=16/BK=32)。
#define PF_BQ 16
#define PF_BK 32
// exp2f ベース softmax 用 (expf(x) = exp2f(x * log2(e)))
#define LOG2E_F 1.44269504088896340736f
extern "C" __global__ void dno_attention_prefill(
    const float *__restrict__ q,     // [n, heads*dh]
    const float *__restrict__ kc,    // [t_total, kv_dim]
    const float *__restrict__ vc,    // [t_total, kv_dim]
    float *__restrict__ att,         // [n, heads*dh]
    int heads, int n, int pos0, int kv_dim, int t_total, int group, int dh,
    float scale) {
    extern __shared__ float sh[];
    float *q_sm = sh;                     // [PF_BQ][dh]
    float *k_sm = q_sm + PF_BQ * dh;      // [PF_BK][dh]
    float *v_sm = k_sm + PF_BK * dh;      // [PF_BK][dh]
    float *s_sm = v_sm + PF_BK * dh;      // [PF_BQ][PF_BK] scores → probs
    float *corr_sm = s_sm + PF_BQ * PF_BK;  // [PF_BQ] (最終フェーズで 1/l に再利用)
    float *mnew_sm = corr_sm + PF_BQ;       // [PF_BQ]

    const int h = blockIdx.x;
    const int i0 = blockIdx.y * PF_BQ;
    const int kh_off = (h / group) * dh;
    const int tid = threadIdx.x;
    // この block が見る必要のある最大 key (行ごとの causal 界の最大値)
    const int jend = min(t_total, pos0 + i0 + PF_BQ);

    // 出力次元 d=tid の全 row 累積 / row tid (= owner スレッド) の online 状態
    float acc[PF_BQ];
    for (int r = 0; r < PF_BQ; ++r) acc[r] = 0.0f;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // q を shared へ (範囲外行は 0 埋め → スコア 0・l > 0 で NaN なし・出力は書かない)
    for (int r = 0; r < PF_BQ; ++r) {
        const int i = i0 + r;
        q_sm[r * dh + tid] =
            (i < n) ? q[(size_t)i * heads * dh + (size_t)h * dh + tid] : 0.0f;
    }

    for (int j0 = 0; j0 < jend; j0 += PF_BK) {
        // K/V タイルを shared へ (範囲外は 0 — masked pair は p=0 で参照されない)
        for (int kk = 0; kk < PF_BK; ++kk) {
            const int j = j0 + kk;
            if (j < t_total) {
                k_sm[kk * dh + tid] = kc[(size_t)j * kv_dim + kh_off + tid];
                v_sm[kk * dh + tid] = vc[(size_t)j * kv_dim + kh_off + tid];
            } else {
                k_sm[kk * dh + tid] = 0.0f;
                v_sm[kk * dh + tid] = 0.0f;
            }
        }
        __syncthreads();
        // scores: (r, kk) ペアをスレッド分散。ドットは dh 逐次 fmaf
        // (decode kernel dno_attention と同一の演算順)。
        for (int pid = tid; pid < PF_BQ * PF_BK; pid += dh) {
            const int r = pid / PF_BK;
            const int kk = pid % PF_BK;
            const int i = i0 + r;
            const int j = j0 + kk;
            float s;
            if (i >= n) {
                // 無効行: m_new を 0 に保ち NaN を避ける (acc はゴミだが書かない)
                s = 0.0f;
            } else if (j < t_total && j <= pos0 + i) {
                const float *qr = &q_sm[r * dh];
                const float *kr = &k_sm[kk * dh];
                float d = 0.0f;
                for (int t2 = 0; t2 < dh; ++t2) d = fmaf(qr[t2], kr[t2], d);
                s = d * scale;
            } else {
                s = -INFINITY;  // causal mask (exp 後 0)
            }
            s_sm[r * PF_BK + kk] = s;
        }
        __syncthreads();
        // タイル内 row max + online 更新 (owner = thread r)
        if (tid < PF_BQ) {
            float mx = m_run;
            for (int kk = 0; kk < PF_BK; ++kk) mx = fmaxf(mx, s_sm[tid * PF_BK + kk]);
            // m_run=-inf → exp2f(-inf * log2e)=0 (ハード命令の -inf 処理)
            corr_sm[tid] = exp2f((m_run - mx) * LOG2E_F);
            mnew_sm[tid] = mx;
        }
        __syncthreads();
        // p = exp(s - m_new) を s_sm に上書き (無効 pair は 0)
        for (int pid = tid; pid < PF_BQ * PF_BK; pid += dh) {
            const int r = pid / PF_BK;
            const int kk = pid % PF_BK;
            s_sm[r * PF_BK + kk] = exp2f((s_sm[r * PF_BK + kk] - mnew_sm[r]) * LOG2E_F);
        }
        __syncthreads();
        if (tid < PF_BQ) {
            float l = l_run * corr_sm[tid];
            for (int kk = 0; kk < PF_BK; ++kk) l += s_sm[tid * PF_BK + kk];
            l_run = l;
            m_run = mnew_sm[tid];
        }
        // acc 再スケール + probs@V 累積 (出力次元 d=tid 専属)
        for (int r = 0; r < PF_BQ; ++r) acc[r] *= corr_sm[r];
        for (int r = 0; r < PF_BQ; ++r) {
            for (int kk = 0; kk < PF_BK; ++kk) {
                acc[r] = fmaf(s_sm[r * PF_BK + kk], v_sm[kk * dh + tid], acc[r]);
            }
        }
        __syncthreads();  // 次タイルの k_sm/v_sm/s_sm 上書き前の壁
    }

    // 正規化 + 書き出し (corr_sm を 1/l に再利用)。有効 row は必ず j=0 が
    // causal 内なので l >= 1 > 0。無効行 (i >= n) は書かない。
    if (tid < PF_BQ) corr_sm[tid] = 1.0f / l_run;
    __syncthreads();
    for (int r = 0; r < PF_BQ; ++r) {
        const int i = i0 + r;
        if (i < n) {
            att[(size_t)i * heads * dh + (size_t)h * dh + tid] = acc[r] * corr_sm[r];
        }
    }
}

// P17h: decode 用 qkv rope (qkv bias 対応)。CPU tensor::rope_row とビット一致
// させるため cos/sin はホスト (同一プロセスの glibc sinf/cosf) で計算した
// テーブルを渡し、演算は -fmad=false の mul.rn/add.rn/sub.rn (Rust のスカラー
// 演算と同一の丸め) を使う。bias 加算は rope の前 (CPU 経路と同一順)。
// grid = (heads + 2*kv_heads), block = (dh), shared なし。
//   block b < heads             : q ヘッド b — bias 加算 + rope
//   heads <= b < heads+kv_heads : k ヘッド    — bias 加算 + rope
//   heads+kv_heads <= b         : v ヘッド    — bias 加算のみ
// bias は [qb(heads*dh) | kb(kv_dim) | vb(kv_dim)] 連結の 1 バッファ (無ければ NULL)。
// tab は [cos(half) | sin(half)] (half = dh/2)。
extern "C" __global__ void dno_rope_qkv(float *qkv, const float *tab, const float *bias,
                                        int heads, int kv_heads, int dh) {
    const int half = dh / 2;
    const int b = blockIdx.x;
    if (b < heads + kv_heads) {
        float *dst;
        const float *bl;
        if (b < heads) {
            dst = qkv + (size_t)b * dh;
            bl = bias ? bias + (size_t)b * dh : nullptr;
        } else {
            const int kh = b - heads;
            dst = qkv + (size_t)heads * dh + (size_t)kh * dh;
            bl = bias ? bias + (size_t)heads * dh + (size_t)kh * dh : nullptr;
        }
        if (threadIdx.x < half) {
            const int d = threadIdx.x;
            const float c = tab[d];
            const float s = tab[half + d];
            float x1 = dst[d];
            float x2 = dst[half + d];
            if (bl) {
                x1 += bl[d];
                x2 += bl[half + d];
            }
            dst[d] = x1 * c - x2 * s;
            dst[half + d] = x1 * s + x2 * c;
        }
    } else {
        const int vh = b - heads - kv_heads;
        float *dst = qkv + (size_t)(heads + kv_heads) * dh + (size_t)vh * dh;
        const float *bl = bias ? bias + (size_t)(heads + kv_heads) * dh + (size_t)vh * dh
                               : nullptr;
        const int d = threadIdx.x;
        float x = dst[d];
        if (bl) x += bl[d];
        dst[d] = x;
    }
}
