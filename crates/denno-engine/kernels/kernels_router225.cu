// task 225: prefill の router softmax と top-k をデバイスで回すカーネル。
//
// PTX へのコンパイル (build_ptx.sh の `router225`):
//
//   nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_router225.cu \
//        -o kernels_router225.ptx
//
// なぜ別カーネルか (lab t222c の [q4ledger-pf]、14,018 トークン prefill):
//   `sub router` は 239.9 ms がホストの `softmax` + top-k。logits は既に GPU の
//   f32 GEMM (`dno_gemm_f32_mt`) で作っているが、いったん全行 (14k で 1,378 MB) を
//   ホストへ降ろして 512 列の softmax と `select_nth_unstable_by` を回していた。
//   ここでは logits をデバイスに残したまま softmax → top-k → 再正規化まで行い、
//   ホストへは **選んだ k 個の index と重みだけ** を返す。
//
// 1 ワープ = 1 トークン。レーン l が index l, l+32, l+64, ... を持つ (昇順)。
//   - softmax: 行 max を warp 縮約、和はレーン内の昇順部分和 → 固定木で結合
//   - top-k: 各レーンが自分の担当の上位 k を挿入法で作り (同値は小さい index)、
//     そこから k 回の warp 最大 (値の降順 → index 昇順) で全体の上位 k を選ぶ
//   - 再正規化: 選んだ k 個を index 昇順に並べ、その順で和を取って割る
//     (ホストの `router_renorm` と同じ順序)
//
// 数値: 和の結合順がホストの「行の昇順逐次和」と違うので、softmax の分母が
//   1 ulp 級に動く (丸めだけの差)。top-k の選択規則 (値の降順・同値は小さい
//   index・出力は index 昇順) と再正規化の加算順はホストと同一。決定論。
//
// 前提 (ホスト側 `router_dev_ok` が同じ条件を検査する):
//   - 1 <= k <= RT_MAX_K (32) / ne <= 32 * RT_MAX_PER (1024) / nt >= 1
// それ以外は従来のホスト経路へ fail closed。
//
// 起動: grid = (ceil(nt / 4), 1, 1) / block = (128, 1, 1) / smem なし。
#define RT_MAX_PER 32          // 1 レーンが持つ最大要素数 (ne <= 32 * 32)
#define RT_MAX_K 32            // top-k の上限
#define RT_WARPS 4             // 1 ブロック = 4 ワープ = 4 トークン
#define RT_THREADS (RT_WARPS * 32)

extern "C" __global__ void __launch_bounds__(RT_THREADS, 4)
dno_router_softmax_topk(
    const float* __restrict__ logits,  // [nt, ne]
    unsigned int* __restrict__ sels,   // [nt, k] (index 昇順)
    float* __restrict__ wts,           // [nt, k] (再正規化済み)
    float* __restrict__ rowsum,        // [nt] (再正規化前の選択和。P54 の role RMS 用)
    int nt,
    int ne,
    int k)
{
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int t = blockIdx.x * RT_WARPS + wid;
    if (t >= nt) {
        return;
    }
    const long long base = (long long)t * ne;

    // ---- 行 max (レーン内の昇順 → warp 固定木) ------------------------------
    float m = -INFINITY;
#pragma unroll
    for (int j = 0; j < RT_MAX_PER; ++j) {
        const int i = lane + j * 32;
        if (i < ne) {
            const float v = logits[base + i];
            m = (v > m) ? v : m;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, m, off);
        m = (o > m) ? o : m;
    }

    // ---- exp と和 ---------------------------------------------------------
    float e[RT_MAX_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < RT_MAX_PER; ++j) {
        const int i = lane + j * 32;
        const float v = (i < ne) ? logits[base + i] : m;
        const float ev = (i < ne) ? expf(v - m) : 0.0f;
        e[j] = ev;
        s += ev;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, off);
    }
    // ホストの softmax は sum > 0 のときだけ割る (0 なら exp の値のまま)。
    const bool divide = (s > 0.0f) && (s == s);
    const float inv = divide ? (1.0f / s) : 1.0f;

    // ---- レーンごとの上位 k (挿入法。i は j について昇順なので同値は小さい index)
    float lv[RT_MAX_K];
    int li[RT_MAX_K];
    int cnt = 0;
#pragma unroll
    for (int j = 0; j < RT_MAX_PER; ++j) {
        const int i = lane + j * 32;
        if (i >= ne) {
            continue;
        }
        const float p = divide ? (e[j] / s) : e[j];
        if (cnt == k && !(p > lv[k - 1])) {
            continue;
        }
        int pos = (cnt < k) ? cnt : (k - 1);
        while (pos > 0 && (p > lv[pos - 1] || (p == lv[pos - 1] && i < li[pos - 1]))) {
            lv[pos] = lv[pos - 1];
            li[pos] = li[pos - 1];
            pos -= 1;
        }
        lv[pos] = p;
        li[pos] = i;
        if (cnt < k) {
            cnt += 1;
        }
    }

    // ---- k 回の warp 最大で全体の上位 k を選ぶ ----------------------------
    float sv[RT_MAX_K];
    int si[RT_MAX_K];
    int cur = 0;
    for (int r = 0; r < k; ++r) {
        float pv = -INFINITY;
        int pi = 2147483647;
        if (cur < cnt) {
            pv = lv[cur];
            pi = li[cur];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, pv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, pi, off);
            if (ov > pv || (ov == pv && oi < pi)) {
                pv = ov;
                pi = oi;
            }
        }
        sv[r] = pv;
        si[r] = pi;
        if (cur < cnt && lv[cur] == pv && li[cur] == pi) {
            cur += 1;
        }
    }

    // ---- index 昇順に並べ替え (k <= 32 の挿入法。全レーンが同じ配列を持つ) ----
    for (int a = 1; a < k; ++a) {
        const float v = sv[a];
        const int i = si[a];
        int b = a;
        while (b > 0 && si[b - 1] > i) {
            sv[b] = sv[b - 1];
            si[b] = si[b - 1];
            b -= 1;
        }
        sv[b] = v;
        si[b] = i;
    }

    // ---- 再正規化 (index 昇順の逐次和) と書き出し -------------------------
    if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            sum += sv[j];
        }
        const bool ok = (sum == sum) && (sum > 0.0f) && (sum < INFINITY);
        const float uni = 1.0f / (float)k;
        rowsum[t] = sum;
        for (int j = 0; j < k; ++j) {
            sels[(long long)t * k + j] = (unsigned int)si[j];
            wts[(long long)t * k + j] = ok ? (sv[j] / sum) : uni;
        }
    }
}
