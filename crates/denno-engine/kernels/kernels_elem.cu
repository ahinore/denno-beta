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
// grid = (ceil(n / PF_BQ), heads), block = (dh), dynamic shmem =
//   (PF_BQ + 2*PF_BK) * (dh+1) + PF_BQ*PF_BK + 2*PF_BQ floats (<= 48KB)。
// block (qt, h) が q ヘッド h のクエリ行 [qt*PF_BQ, qt*PF_BQ+PF_BQ) を担当。
//   x=行ブロック (最速次元) → 同時実行ブロックが同一 head の K/V を共有 (L2)。
//   kvh = h / group, kh_off = kvh * dh
//   scores[i,j] = q[i,h] · K[j,kh_off..] * scale   (j = 0..pos0+i — causal)
//   att[i,h]    = Σ_j softmax_j * V[j, kh_off..]
// K/V は GPU 側 KV ミラー ([t_total, kv_dim] 行-major、ホスト KvCache と同一)。
// P17k 性能修正 ③: q/k/v smem の行ストライドを dh+1 にパディング。dh=128 は
// そのままだとストライドが 32 の倍数でスコア dot の smem 読み (k_sm[kk*dh+t2])
// が 32-way bank conflict になり SM 帯域が ~1/32 に落ちる。+1 パッドで bank を
// 行ごとにずらし conflict 解消。メモリ配置のみの変更で**数値は完全不変**。
// exp は exp2f ベース (P17k): attention は数値緩和枠 (ユーザー承認済み) のため
// glibc 完全再現の dno_expf (f64 演算主体・デスクトップ GPU では 1/64 レートで
// 増悪) をやめ、ハード命令 MUFU.EX2 の exp2f に置換する。expf(x) =
// exp2f(x * log2(e)) で置き換えるため p に ~1e-6 相対差が出るが、**決定論は
// 維持** (exp2f は固定のハード演算・同一入力に対し常に同一結果)。decode 用の
// dno_attention はビット一致枠のため dno_expf のまま変更しない。
// online softmax の累積順は「タイル昇順・タイル内 kk 昇順・出力次元 d はスレッド
// tid = d 専属」で固定。
// 制約 (ホスト側でも検証): dh >= PF_BQ (block が PF_BQ 行 owner を持つため) &&
//   shmem <= 48KB (dh <= 144 @ BQ=64/BK=8)。BQ=64/BK=8 (43.8KB) は実測安定
//   (かつて「~43KB で error 700」とされたのはホスト sh_f がカーネル必要量を
//   下回る領域外書きが原因 — ステップ 7 で定数同期により解消、上記参照)。
// P17k 性能修正 (2026-10-14): ①PF_BQ 16→48 (K/V の再読みトラフィックは BQ に
// 反比例 — 総トラフィック ~314GB→104GB)・PF_BK 32→8 (smem を実効安全域に収める)。
// ②grid を (行ブロック, heads) に反転 (x=行ブロックが最速次元 → 同時実行
// ブロックが同一 head の K/V 27MB を共有し L2 に乗る。旧 layout は 14 head 分
// ~380MB のワーキングセットで L2 に乗らず DRAM 律速だった)。③q/k/v smem の
// 行ストライドを dh+1 にパディング (dh=128 は 32 の倍数でスコア dot の smem 読み
// が 32-way bank conflict → SM 帯域 ~1/32。+1 パッドで解消)。③はメモリ配置のみ
// で**数値完全不変**。①はタイル境界が変わり online softmax の中間丸めが変わるが
// attention 緩和枠内・決定論は維持 (固定タイル・固定累積順)。②は数値に影響しない。
// 実測 (n=13252・t=13252・14 heads・RTX 5090): 368ms → **112ms/call** (3.3 倍)。
//   内訳: ③パッドで 284ms→149ms、①+②で→112ms。
// P17k ステップ 7 (2026-10-14): レイテンシ律速対策の 2 点。
// ④ **K/V タイルのソフトウェアパイプライン**: 次タイルをレジスタにプリフェッチ
//   してからスコア/softmax/V 累積を回すことで、グローバル読み (~500-800 cycle)
//   を演算フェーズで隠蔽 (旧構造は syncthreads 壁で毎タイル待ち)。読み条件
//   (j < t_total・範囲外 0 埋め) は旧版と同一 → **数値は完全不変**。
// ⑤ **スコア dot の 4-way アンロール**: 依存 FMA チェーン (128 段) を 4 本の
//   独立チェーンに分割 (d0..d3 → ((d0+d1)+(d2+d3))・結合順固定) して ILP を向上。
//   累積順が変わるため CPU との差は僅かに変わりうるが attention 緩和枠内・
//   **決定論は維持** (結合順は dh とタイルサイズのみで決まる固定順)。
//   dh%4 の余りはスカラーで処理 (実運用 dh=128 では発生しない)。
// ⑥ **2r×2k クアッド dot**: 1 thread が行 {r0,r0+1} × key {k0,k0+1} の 4 ペアを
//   担当し q/k の smem 読みを 4 FMA で共有 → スコア phase の smem 読みが
//   FMA あたり 2 回 → 1 回に半減 (V 累積と合わせるとカーネルは smem 帯域律速で、
//   ④⑤単体では ~1% しか出なかったが ⑥ で **90→69ms (1.3 倍)**)。
//   クアッド数 = PF_BQ*PF_BK/4 (= dh @ 64×8)・nth > dh でも動作 (余剰 thread は idle)。
//   累積順は ⑥ でも更に変わるが緩和枠内・決定論維持 (テスト: maxabs 1.97e-7 @ n=2056)。
// 実測 (n=13252・t=13252・14 heads・RTX 5090): 368ms (ステップ 4 前) → 112ms
//   (ステップ 6 = ③+過大 grid) → **69.3ms/call** (ステップ 7 = ④⑤⑥+grid 修正)。
// P17k ステップ 7 の重要修正 (cuda.rs 側): ATTN_PF_BQ/BK が 16/32 のままだった
//   ため grid が約 4 倍過剰 (i0 >= n の無効ブロックがフル time ループを回す)。
//   64/8 に統一で **attn_core 4.5s → 3.5s**。「~43KB で error 700」の正体も
//   ホスト sh_f (16/32 基準 43.4KB) がカーネル必要量を下回る smem 領域外書き。
// B2 (2026-10-15): probs@V 累積の行チャンク化。v_sm[kk] を行ループの外 (kk ごと
//   1 回) に hoist し 2 行インターリーブで再利用 + s_sm/corr_sm を float2 (8B)
//   荷載にする。スレッドあたりタイル当たりの LDS 命令: s 512 + v 512 + corr 64
//   → s 256 + v 8 + corr 32 (v_sm の再読みが行数 PF_BQ 回 → 1 回に減る)。
//   acc[r] への FMA 列は「r 固定・kk 昇順」で旧実装と完全同一
//   (fmaf(p.y, v1, fmaf(p.x, v0, acc)) の内側が kk・外側が kk+1) のため
//   **数値はビット一致** (attention 緩和枠だが変化なし・決定論も固定順のまま・
//   原子操作なし)。ホスト側は無変更 (sh_f / grid / block は同一)。
//   実測は実機で cuda_attn_pf_bench を再実行すること。s_sm の行先頭は
//   PF_BK=8 floats (32B) 刻み・動的 smem ベースは 16B 整列のため float2 (8B)
//   アクセスの整列は常に成立する (kk は偶数ステップ)。
#define PF_BQ 64
#define PF_BK 8
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
    float *q_sm = sh;                     // [PF_BQ][dh+1] (bank conflict 回避の pad)
    float *k_sm = sh + PF_BQ * (dh + 1);  // [PF_BK][dh+1]
    float *v_sm = k_sm + PF_BK * (dh + 1);  // [PF_BK][dh+1]
    float *s_sm = v_sm + PF_BK * (dh + 1);  // [PF_BQ][PF_BK] scores → probs
    float *corr_sm = s_sm + PF_BQ * PF_BK;  // [PF_BQ] (最終フェーズで 1/l に再利用)
    float *mnew_sm = corr_sm + PF_BQ;       // [PF_BQ]

    const int h = blockIdx.y;
    const int i0 = blockIdx.x * PF_BQ;
    const int kh_off = (h / group) * dh;
    const int tid = threadIdx.x;
    // P17k ステップ 7: block は dh または 2*dh スレッド。tid >= dh のスレッドは
    // スコア/exp フェーズのみ参加 (出力次元を持たない)。blockDim を 2 倍にして
    // スコア phase のワープ数を増やしレイテンシを隠蔽する。
    const int nth = blockDim.x;
    // この block が見る必要のある最大 key (行ごとの causal 界の最大値)
    const int jend = min(t_total, pos0 + i0 + PF_BQ);

    // 出力次元 d=tid の全 row 累積 / row tid (= owner スレッド) の online 状態
    float acc[PF_BQ];
    for (int r = 0; r < PF_BQ; ++r) acc[r] = 0.0f;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // ④ プリフェッチバッファ (スレッド tid の K/V 列要素 × BK)
    float k_next[PF_BK], v_next[PF_BK];

    // q を shared へ (範囲外行は 0 埋め → スコア 0・l > 0 で NaN なし・出力は書かない)
    if (tid < dh) {
        for (int r = 0; r < PF_BQ; ++r) {
            const int i = i0 + r;
            q_sm[r * (dh + 1) + tid] =
                (i < n) ? q[(size_t)i * heads * dh + (size_t)h * dh + tid] : 0.0f;
        }
    }

    // ④ プロローグ: 最初のタイル (j0=0) をレジスタへ
    if (tid < dh) {
        for (int kk = 0; kk < PF_BK; ++kk) {
            const int j = kk;
            if (j < t_total) {
                k_next[kk] = kc[(size_t)j * kv_dim + kh_off + tid];
                v_next[kk] = vc[(size_t)j * kv_dim + kh_off + tid];
            } else {
                k_next[kk] = 0.0f;
                v_next[kk] = 0.0f;
            }
        }
    }

    for (int j0 = 0; j0 < jend; j0 += PF_BK) {
        // ④ レジスタ → shared (前回プリフェッチ分)
        if (tid < dh) {
            for (int kk = 0; kk < PF_BK; ++kk) {
                k_sm[kk * (dh + 1) + tid] = k_next[kk];
                v_sm[kk * (dh + 1) + tid] = v_next[kk];
            }
        }
        __syncthreads();
        // ④ 次タイルを先に読み始める (グローバル読みを演算フェーズと重ねる)。
        //    読み条件は旧版と同一 (j < t_total・範囲外 0)。
        if (tid < dh) {
            const int jn = j0 + PF_BK;
            for (int kk = 0; kk < PF_BK; ++kk) {
                const int j = jn + kk;
                if (j < t_total) {
                    k_next[kk] = kc[(size_t)j * kv_dim + kh_off + tid];
                    v_next[kk] = vc[(size_t)j * kv_dim + kh_off + tid];
                } else {
                    k_next[kk] = 0.0f;
                    v_next[kk] = 0.0f;
                }
            }
        }
        // scores: (r, kk) ペアをスレッド分散。⑥ 2r×2k クアッド + 4-way アンロール:
        // thread が行 {r0,r0+1} × key {k0,k0+1} の 4 ペアを担当し q/k の smem 読みを
        // 共有 (読み量を FMA あたり 2 回 → 1 回に半減)。t2 も 4 本の独立チェーンに
        // 分割 ((a+b)+(c+d) の固定順で統合 — 決定論維持・attention 緩和枠内)。
        // クアッド数 = PF_BQ*PF_BK/4 (= dh @ 64×8)。
        for (int q4 = tid; q4 < PF_BQ * PF_BK / 4; q4 += nth) {
            const int r0 = (q4 / (PF_BK / 2)) * 2;
            const int k0 = (q4 % (PF_BK / 2)) * 2;
            const int ia = i0 + r0;
            const int ib = i0 + r0 + 1;
            const int ja = j0 + k0;
            const int jb = j0 + k0 + 1;
            const float *qa = &q_sm[r0 * (dh + 1)];
            const float *qb = &q_sm[(r0 + 1) * (dh + 1)];
            const float *ka = &k_sm[k0 * (dh + 1)];
            const float *kb = &k_sm[(k0 + 1) * (dh + 1)];
            float d00a = 0.0f, d00b = 0.0f, d00c = 0.0f, d00d = 0.0f;
            float d01a = 0.0f, d01b = 0.0f, d01c = 0.0f, d01d = 0.0f;
            float d10a = 0.0f, d10b = 0.0f, d10c = 0.0f, d10d = 0.0f;
            float d11a = 0.0f, d11b = 0.0f, d11c = 0.0f, d11d = 0.0f;
            int t2 = 0;
            for (; t2 + 4 <= dh; t2 += 4) {
                const float qa0 = qa[t2], qa1 = qa[t2 + 1], qa2 = qa[t2 + 2], qa3 = qa[t2 + 3];
                const float qb0 = qb[t2], qb1 = qb[t2 + 1], qb2 = qb[t2 + 2], qb3 = qb[t2 + 3];
                const float ka0 = ka[t2], ka1 = ka[t2 + 1], ka2 = ka[t2 + 2], ka3 = ka[t2 + 3];
                const float kb0 = kb[t2], kb1 = kb[t2 + 1], kb2 = kb[t2 + 2], kb3 = kb[t2 + 3];
                d00a = fmaf(qa0, ka0, d00a); d01a = fmaf(qa0, kb0, d01a);
                d10a = fmaf(qb0, ka0, d10a); d11a = fmaf(qb0, kb0, d11a);
                d00b = fmaf(qa1, ka1, d00b); d01b = fmaf(qa1, kb1, d01b);
                d10b = fmaf(qb1, ka1, d10b); d11b = fmaf(qb1, kb1, d11b);
                d00c = fmaf(qa2, ka2, d00c); d01c = fmaf(qa2, kb2, d01c);
                d10c = fmaf(qb2, ka2, d10c); d11c = fmaf(qb2, kb2, d11c);
                d00d = fmaf(qa3, ka3, d00d); d01d = fmaf(qa3, kb3, d01d);
                d10d = fmaf(qb3, ka3, d10d); d11d = fmaf(qb3, kb3, d11d);
            }
            float d00 = ((d00a + d00b) + (d00c + d00d));
            float d01 = ((d01a + d01b) + (d01c + d01d));
            float d10 = ((d10a + d10b) + (d10c + d10d));
            float d11 = ((d11a + d11b) + (d11c + d11d));
            for (; t2 < dh; ++t2) {
                const float qa0 = qa[t2], qb0 = qb[t2], ka0 = ka[t2], kb0 = kb[t2];
                // 余り分は d00 (先頭チェーン) に逐次加算 (実運用 dh=128 では不使用)
                d00 += qa0 * ka0; d01 += qa0 * kb0; d10 += qb0 * ka0; d11 += qb0 * kb0;
            }
            float s00, s01, s10, s11;
            if (ia >= n) { s00 = 0.0f; s01 = 0.0f; } else {
                s00 = (ja < t_total && ja <= pos0 + ia) ? d00 * scale : -INFINITY;
                s01 = (jb < t_total && jb <= pos0 + ia) ? d01 * scale : -INFINITY;
            }
            if (ib >= n) { s10 = 0.0f; s11 = 0.0f; } else {
                s10 = (ja < t_total && ja <= pos0 + ib) ? d10 * scale : -INFINITY;
                s11 = (jb < t_total && jb <= pos0 + ib) ? d11 * scale : -INFINITY;
            }
            s_sm[r0 * PF_BK + k0] = s00;
            s_sm[r0 * PF_BK + k0 + 1] = s01;
            s_sm[(r0 + 1) * PF_BK + k0] = s10;
            s_sm[(r0 + 1) * PF_BK + k0 + 1] = s11;
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
        for (int pid = tid; pid < PF_BQ * PF_BK; pid += nth) {
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
        // acc 再スケール + probs@V 累積 (出力次元 d=tid 専属 — tid < dh のみ)。
        // B2: 行チャンク化 — v_sm[kk] を hoist (行数回の再読み → kk ごと 1 回)・
        // 2 行インターリーブ・s/corr を float2 荷載。acc[r] への FMA 列は
        // 「r 固定・kk 昇順」で旧実装と完全同一 → ビット一致 (決定論も不変)。
        if (tid < dh) {
            for (int rc = 0; rc < PF_BQ / 2; ++rc) {
                const float2 c = *reinterpret_cast<const float2 *>(&corr_sm[rc * 2]);
                acc[rc * 2] *= c.x;
                acc[rc * 2 + 1] *= c.y;
            }
            for (int kk = 0; kk < PF_BK; kk += 2) {
                const float v0 = v_sm[kk * (dh + 1) + tid];
                const float v1 = v_sm[(kk + 1) * (dh + 1) + tid];
                #pragma unroll
                for (int rp = 0; rp < PF_BQ / 2; ++rp) {
                    const int r0 = rp * 2;
                    const float2 pa =
                        *reinterpret_cast<const float2 *>(&s_sm[r0 * PF_BK + kk]);
                    const float2 pb =
                        *reinterpret_cast<const float2 *>(&s_sm[(r0 + 1) * PF_BK + kk]);
                    acc[r0] = fmaf(pa.y, v1, fmaf(pa.x, v0, acc[r0]));
                    acc[r0 + 1] = fmaf(pb.y, v1, fmaf(pb.x, v0, acc[r0 + 1]));
                }
            }
        }
        __syncthreads();  // 次タイルの k_sm/v_sm/s_sm 上書き前の壁
    }

    // 正規化 + 書き出し (corr_sm を 1/l に再利用)。有効 row は必ず j=0 が
    // causal 内なので l >= 1 > 0。無効行 (i >= n) は書かない。
    if (tid < PF_BQ) corr_sm[tid] = 1.0f / l_run;
    __syncthreads();
    if (tid < dh) {
        for (int r = 0; r < PF_BQ; ++r) {
            const int i = i0 + r;
            if (i < n) {
                att[(size_t)i * heads * dh + (size_t)h * dh + tid] = acc[r] * corr_sm[r];
            }
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

// P17k ステップ 5: prefill 用行並列 rope (qkv→rope→attention デバイス連続化)。
// grid = (heads + 2*kv_heads, n), block = (dh), shared なし。
//   block b < heads             : q ヘッド b — rope して qout [n, heads*dh] (連続) へ
//   heads <= b < heads+kv_heads : k ヘッド    — rope して km の (pos0+i) 行へ
//   heads+kv_heads <= b         : v ヘッド    — vm の (pos0+i) 行へコピー
// qkv は [n, width] 行-major (width = (heads+2*kv_heads)*dh、行内 [qn|kn|vn])。
// km/vm は GPU KV ミラー ([cap, kv_dim] 行-major・ホスト KvCache と同一)。
// tab は [n, dh] — 行 i が pos=pos0+i の [cos(half) | sin(half)] (CPU が
// rope_row と同一式で生成 → mul/sub のみの適用でビット一致)。
// bias は [qb | kb | vb] 連結 1 バッファ (無ければ NULL)。
extern "C" __global__ void dno_rope_qkv_pf(float *qkv, float *qout, float *km, float *vm,
                                           const float *tab, const float *bias, int pos0,
                                           int heads, int kv_heads, int dh, int kv_dim) {
    const int half = dh / 2;
    const int b = blockIdx.x;
    const int i = blockIdx.y;
    const int width = (heads + 2 * kv_heads) * dh;
    const float *src = qkv + (size_t)i * width;
    const float *trow = tab + (size_t)i * dh;  // 2*half == dh
    if (b < heads) {
        // q: rope → qout[i, b*dh..]
        const int d = threadIdx.x;
        if (d < half) {
            const float *bl = bias ? bias + (size_t)b * dh : nullptr;
            const float c = trow[d];
            const float s = trow[half + d];
            float x1 = src[(size_t)b * dh + d];
            float x2 = src[(size_t)b * dh + half + d];
            if (bl) {
                x1 += bl[d];
                x2 += bl[half + d];
            }
            float *dst = qout + (size_t)i * heads * dh + (size_t)b * dh;
            dst[d] = x1 * c - x2 * s;
            dst[half + d] = x1 * s + x2 * c;
        }
    } else if (b < heads + kv_heads) {
        // k: rope → km の (pos0+i) 行
        const int kh = b - heads;
        const int d = threadIdx.x;
        if (d < half) {
            const float *bl = bias ? bias + (size_t)heads * dh + (size_t)kh * dh : nullptr;
            const float c = trow[d];
            const float s = trow[half + d];
            float x1 = src[(size_t)heads * dh + (size_t)kh * dh + d];
            float x2 = src[(size_t)heads * dh + (size_t)kh * dh + half + d];
            if (bl) {
                x1 += bl[d];
                x2 += bl[half + d];
            }
            float *dst = km + (size_t)(pos0 + i) * kv_dim + (size_t)kh * dh;
            dst[d] = x1 * c - x2 * s;
            dst[half + d] = x1 * s + x2 * c;
        }
    } else {
        // v: bias 加算して vm の (pos0+i) 行へ
        const int vh = b - heads - kv_heads;
        const int d = threadIdx.x;
        const float *bl =
            bias ? bias + (size_t)(heads + kv_heads) * dh + (size_t)vh * dh : nullptr;
        float x = src[(size_t)(heads + kv_heads) * dh + (size_t)vh * dh + d];
        if (bl) x += bl[d];
        vm[(size_t)(pos0 + i) * kv_dim + (size_t)vh * dh + d] = x;
    }
}

// B3: b>1 バッチ decode 用 rope (q/k in-place・セッションごとに独立した位置)。
// P17h/P17k の rope カーネルと同一の演算列 (mul/sub のみ・-fmad=false) で、
// cos/sin はホストが tensor::rope_row と同一式で生成したテーブルを渡す →
// CPU の session ごと rope_row 適用 (forward_batch) とビット一致
// (exactness 契約: rope は緩和対象外・bias/QK-Norm は既に呼び出し側が適用済み)。
// grid = (heads + kv_heads, b), block = (dh), shared なし。
//   block b < heads : q ヘッド b — q[i, b*dh ..] を rope
//   heads <= b      : k ヘッド b-heads — k[i, kh*dh ..] を rope
// q は [b, heads*dh] 行-major・k は [b, kv_dim] 行-major (ともに in-place)。
// tab は [b, dh] — 行 i が pos=poss[i] の [cos(half) | sin(half)] (half = dh/2)。
extern "C" __global__ void dno_rope_qk_batch(float *q, float *k, const float *tab,
                                             int heads, int kv_heads, int dh) {
    const int half = dh / 2;
    const int b = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= half) return;
    const float *trow = tab + (size_t)i * dh;
    const float c = trow[d];
    const float s = trow[half + d];
    if (b < heads) {
        float *dst = q + (size_t)i * heads * dh + (size_t)b * dh;
        const float x1 = dst[d];
        const float x2 = dst[half + d];
        dst[d] = x1 * c - x2 * s;
        dst[half + d] = x1 * s + x2 * c;
    } else {
        const int kh = b - heads;
        float *dst = k + (size_t)i * kv_heads * dh + (size_t)kh * dh;
        const float x1 = dst[d];
        const float x2 = dst[half + d];
        dst[d] = x1 * c - x2 * s;
        dst[half + d] = x1 * s + x2 * c;
    }
}

// ---------------------------------------------------------------------------
// C4: qwen35 デバイス常駐 decode チェーン用カーネル
// (QK-Norm+partial interleaved rope / sigmoid 出力ゲート / fp32 matvec)
// ---------------------------------------------------------------------------

// QK-Norm + partial interleaved RoPE (Qwen3.5 full-attention 層の decode 用)。
// grid = (heads|kv_heads, 1, 1), block = (cols, 1, 1) (cols = head_dim <= 1024),
// shared = (cols + 1) * 4 bytes (末尾 1 要素は inv の受け渡し)。
//
// 演算列 (CPU とビット一致・exactness 契約: QK-Norm は rmsnorm 枠・rope は緩和対象外):
//   1) thread 0 が CPU tensor::rmsnorm と同一の逐次 2 乗和
//      (ss += x_i*x_i を i = 0..cols-1、ss/cols、1/sqrt(ss/cols + eps))
//   2) 全スレッド: normed[i] = (x[i] * inv) * w[i] (左結合 — dno_add_rmsnorm と同一)
//   3) d < n_rot/2 のスレッド: interleaved ペア (2d, 2d+1) を回転
//      (x1*c - x2*s / x1*s + x2*c — mul/sub のみ・-fmad=false。tab はホストが
//      rope_partial_interleaved と同一式 (ang = pos * theta^(-2d/n_rot)) で生成)
//   4) out[b*out_stride + i] へ書き戻し
// x は行-major だが行ストライドが cols と異なるレイアウト (q: [q|gate] 融合で
// stride 2*cols / k: kv 行で stride kv_dim) を x_stride で受ける。
// rope しない次元 (d >= n_rot) は normed 値をそのまま出力する。
extern "C" __global__ void dno_qknorm_rope_pi(
    const float *__restrict__ x,  // [rows, x_stride] — block b が head b の行を読む
    int x_stride,
    const float *__restrict__ w,  // [cols] (q_norm/k_norm・ヘッド間共有)
    const float *__restrict__ tab, // [cos(n_rot/2) | sin(n_rot/2)]
    float *__restrict__ out,      // [rows, out_stride]
    int out_stride,
    int cols, int n_rot, float eps) {
    extern __shared__ float sh[];
    const int b = blockIdx.x;
    const int half = n_rot / 2;
    const float *row = x + (size_t)b * x_stride;
    // 1) 逐次 2 乗和 (thread 0 のみ — dno_add_rmsnorm と同一演算列)
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < cols; ++i) {
            ss = __fadd_rn(ss, __fmul_rn(row[i], row[i]));
        }
        ss = __fdiv_rn(ss, (float)cols);
        s_inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    }
    __syncthreads();
    // 2) normed[i] = (x[i] * inv) * w[i]
    const int d = threadIdx.x;
    const float inv = s_inv;
    sh[d] = __fmul_rn(__fmul_rn(row[d], inv), w[d]);
    __syncthreads();
    // 3) partial interleaved rope (ペアは 1 スレッドが担当 → 書き込み競合なし)
    if (d < half) {
        const float c = tab[d];
        const float s = tab[half + d];
        const float x1 = sh[2 * d];
        const float x2 = sh[2 * d + 1];
        sh[2 * d] = x1 * c - x2 * s;
        sh[2 * d + 1] = x1 * s + x2 * c;
    }
    __syncthreads();
    // 4) 出力
    out[(size_t)b * out_stride + d] = sh[d];
}

// sigmoid 出力ゲート (Qwen3.5 full-attention: att[h,i] *= sigmoid(gate[h,i]))。
// gate は q 投影出力の後半 ([q|gate] 融合・head ごと [dh|dh]) なので
// qg + h*2*dh + dh から読む。CPU (full_attn_core): `*av /= 1.0 + (-gate[d]).exp()`
// — dno_expf (glibc expf 再現) + __fdiv_rn/__fadd_rn でビット一致。
// grid = (heads, 1, 1), block = (dh, 1, 1)。
extern "C" __global__ void dno_attn_gate_mul(
    float *__restrict__ att,        // [heads * dh] (in-place 更新)
    const float *__restrict__ qg,   // [heads * 2*dh] — head ごと [q | gate]
    int heads, int dh) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    float *av = att + (size_t)h * dh + i;
    const float g = qg[(size_t)h * 2 * dh + dh + i];
    const float e = dno_expf(-g);
    *av = __fdiv_rn(*av, __fadd_rn(1.0f, e));
}

// fp32 matvec (beta/alpha 制御投影専用 — これらは int8 では delta-rule ゲートの
// 精度が崩れるため Fp32 保持が仕様)。1 スレッド 1 出力・k 逐次
// (`s += h[d] * w[d]` と同一順序・積と加算の丸めを分離 = -fmad=false 相当) →
// tensor::matvec_bt とビット一致。n (nv) は最大数十のため逐次ループでも十分軽い。
// grid = (n, 1, 1), block = (128, 1, 1)。
extern "C" __global__ void dno_matvec_f32(
    const float *__restrict__ h,   // [k]
    const float *__restrict__ w,   // [n, k] row-major
    float *__restrict__ out,       // [n]
    int k, int n) {
    const int j = blockIdx.x;
    if (j >= n) return;
    const float *row = w + (size_t)j * k;
    float s = 0.0f;
    for (int d = 0; d < k; ++d) {
        s = __fadd_rn(s, __fmul_rn(h[d], row[d]));
    }
    out[j] = s;
}


// s78 (T7): prefill の beta/alpha 投影 = [m,k] x [n,k]^T -> [m,n] (1 トークン = 1 行)。
// CPU の `tensor::matvec_rows_bt_into` (n < JB=64 の小 n 経路
// `matvec_rows_bt_small_n_impl`) と **ビット一致** する。
//
// CPU 側の累積順は「k を TILE=256 で区切り、タイルごとに acc=0 から d 昇順の
// 逐次和 -> タイル末尾で打ち切り和 -> タイル和を d 昇順で逐次加算」。
// 単純な「k 全体 1 本の逐次和」とは丸めが異なる (誤差は小さいが bit は違う) ので、
// 本カーネルも同じ分割を写す。**区切り点は CPU の `MVROWS_TILE` と一致させる**
// (片方だけ変えるとビット一致が崩れる)。
//
// 累積を逆順 (末尾から) に走らせる理由: カーネルは 1 ブロック = 1 出力 (i,j) とし、
// ブロック内の全スレッドが同じ計算を冗長に行う (決定論 = スレッド数に依存しない)。
// 打ち切り和は末尾連鎖 (x[a] + (x[a+1] + ...) の形) なので、値を先頭から出そうと
// するとループ変数をデータ依存にせざるを得ない。逆順に走らせると各項は独立に
// 積めばよく (ILP)、依存は加算のみになる。
//
// 逆順和と「0 から昇順の和」は IEEE の丸めでは一般には一致しない。差が出るのは
//   (a) 項/部分和が非正規 (denormal) になる場合 — 昇順では丸め時の
//        sticky bit が非正規を保つのに対し、逆順では項が独立に非正規化される
//   (b) 開始値が -0.0 の場合 (CPU は 0.0f = +0.0)
// の 2 つだけである (有限値の加算の丸め誤差は、丸め済みの値を足す形なら向きに
// よらず同じ)。したがって
//   ・開始値は常に +0.0
//   ・部分和が非正規になったら flush-to-zero (符号は保存)
//   ・アンダーフローしかけた正規項も、CPU の昇順和が同じ値を通る限り同じ結果に
//      なる (テスト cuda_gdn_proj_f32_bit_exact_vs_cpu で 0/非正規混在入力も照合)
// で CPU のビット列に一致する。非正規が実際に現れるかはホスト側の事前検査
// (`gdn_proj_f32_into` の sqsum 判定) が見て、疑わしければ CPU へ落とす。
//
// 決定論: (i,j) ごとに 1 ブロック。共有メモリ・atomic・warp 縮約を使わない。
// grid = (n, m, 1), block = (32, 1, 1) — grid.x = nv (32) / grid.y = トークン数。
// 既存 `dno_matvec_f32` (1 出力・grid=(n,1,1)・block=(128,1,1)) は変更しない。
#define DNO_MVROWS_TILE 256

// [lo, hi) の 1 区間を **逆順** に足す。返る bit は
//   float t = 0.0f; for (d = lo; d < hi; ++d) t = t + x[d];
// の末尾連鎖 (gcc -O2 が生成する `x[126] + (x[127] + (x[128] + x[129]))`) と一致する。
__device__ __forceinline__ float dno_mvrows_acc_rev(
    const float *__restrict__ av, const float *__restrict__ row, int n) {
    // 末尾 2 項は mul.rn だけで足し込み不要 (0.0f + x == x)
    float t = __fadd_rn(__fmul_rn(av[n - 2], row[n - 2]), __fmul_rn(av[n - 1], row[n - 1]));
    for (int d = n - 3; d >= 0; --d) {
        t = __fadd_rn(__fmul_rn(av[d], row[d]), t);
        // 部分和の非正規を flush (IEEE の flush-to-zero と同じ・誤差 < 1 ulp)。
        // 符号は残すので -0.0 は -0.0 のまま (CPU 側も同じ値になる)。
        if ((__float_as_uint(t) & 0x7fffffffu) < 0x00800000u) {
            t = __int_as_float(__float_as_uint(t) & 0x80000000u);
        }
    }
    return t;
}

extern "C" __global__ void dno_matvec_rows_f32(
    const float *__restrict__ a,   // [m, k] row-major
    const float *__restrict__ w,   // [n, k] row-major
    float *__restrict__ out,       // [m, n] row-major
    int k, int n, int m) {
    const int j = blockIdx.x; // 出力行 (beta/alpha は nv=32)
    const int i = blockIdx.y; // トークン
    if (j >= n || i >= m) return;
    const float *row = w + (size_t)j * k;
    const float *av = a + (size_t)i * k;
    float total = 0.0f;
    for (int lo = 0; lo < k; lo += DNO_MVROWS_TILE) {
        const int hi = (lo + DNO_MVROWS_TILE < k) ? lo + DNO_MVROWS_TILE : k;
        float t = dno_mvrows_acc_rev(av + lo, row + lo, hi - lo);
        if (lo == 0 && (__float_as_uint(t) & 0x7fffffffu) < 0x00800000u) {
            t = 0.0f; // CPU の `acc = 0.0f` と同じ +0.0 に揃える
        }
        total = __fadd_rn(total, t);
    }
    out[(size_t)i * n + j] = total;
}