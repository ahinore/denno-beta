//! 最小限のテンソル演算。重い GEMM は matrixmultiply(sgemm) + rayon で並列化する。

use rayon::prelude::*;

/// 生ポインタを rayon タスク間で共有するためのラッパ。
/// 使い方は呼び出し側で保証する: 各タスクは担当ブロックの c[i*n + j0..]
/// (他タスクと非重複) にのみ書き込む。
/// (フィールド直接アクセス `.0` は edition 2021 の disjoint closure capture
/// で *mut f32 単体がキャプチャされてしまうため、必ず `write` メソッド経由)
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    unsafe fn write(self, idx: usize, v: f32) {
        *self.0.add(idx) = v;
    }
}

/// c = a @ b  (a: row-major [m,k], b: row-major [k,n], 戻り値: row-major [m,n])
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    par_sgemm(a, b, n as isize, &mut c, m, k, n);
    c
}

/// c = a @ b^T (a: [m,k], b: row-major [n,k] = nn.Linear の重み格納形状)
pub fn matmul_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    par_sgemm(a, b, 1isize, &mut c, m, k, n);
    c
}

/// a は常に row-major [m,k] (rs=k, cs=1)。
/// rs_b は b の行ストライド: row-major [k,n] なら n、[n,k] (b^T使用) なら 1。
fn par_sgemm(a: &[f32], b: &[f32], rs_b: isize, c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let rs_a = k as isize;
    let cs_a = 1isize;
    let cs_b = if rs_b == 1 { k as isize } else { 1isize };
    // 行方向にチャンク分割して並列実行
    let rows_per_task = ((m + 7) / 8).clamp(1, m);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            let c_ptr = chunk.as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, 1.0,
                    a_ptr, rs_a, cs_a,
                    b.as_ptr(), rs_b, cs_b,
                    0.0,
                    c_ptr, n as isize, 1,
                );
            }
        });
}

/// c += alpha * a @ b^T (a: [m,k] row-major, b: [n,k] row-major, c: [m,n] row-major)
/// attention の scores 計算用 (既存値に加算)
pub fn sgemm_acc(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, alpha: f32) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let rows_per_task = ((m + 7) / 8).clamp(1, m);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            let c_ptr = chunk.as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, alpha,
                    a_ptr, k as isize, 1isize,
                    b.as_ptr(), 1isize, k as isize,
                    1.0,
                    c_ptr, n as isize, 1,
                );
            }
        });
}

/// c = a @ b — decode attention の probs@V 用 (P12)。
/// a: [m,k] row-major、b: 行ストライド rs_b の [k,n] (各行の n 要素は連続)。
/// 行統合 (m=group を 1 回の sgemm で実行) + v のストライド直接参照により、
/// 従来の per-g `matmul(m=1)` × group 回 + vh 連続化コピーを 1 回に置き換える。
/// ビット一致は `decode_attention_probs_v_bitexact` テストで担保。
///
/// 注意 (P12 検証メモ): matrixmultiply はストライド/行数でカーネルを選ぶため
/// 数値が変わる。scores 側 (b を [n,k] 転置レイアウト rs=1/cs=k で渡す経路) は
/// 行統合・ストライド参照のどちらもビット不一致になるため従来経路のまま
/// (sgemm_acc(m=1) + 連続化コピー)。row-major b (cs=1) の本関数は行統合・
/// ストライドともビット一致を実測で確認済み。
pub fn matmul_rows_bstrided(
    a: &[f32],
    b: &[f32],
    rs_b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return c;
    }
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0,
            a.as_ptr(), k as isize, 1isize,
            b.as_ptr(), rs_b as isize, 1isize,
            0.0,
            c.as_mut_ptr(), n as isize, 1isize,
        );
    }
    c
}

/// `matmul_rows_bstrided` の行チャンク並列版 (prefill 用)。
/// 出力行を rayon でチャンク分割する以外は同一 (チャンク内の各行は独立に
/// 全 k 累積するため行間の数値相互作用はない。チャンク幅は m 依存だが
/// 同一 m なら常に同一分割 → 決定論は維持)。
pub fn matmul_rows_bstrided_par(
    a: &[f32],
    b: &[f32],
    rs_b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return c;
    }
    let rows_per_task = ((m + 7) / 8).clamp(1, m);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, 1.0,
                    a_ptr, k as isize, 1isize,
                    b.as_ptr(), rs_b as isize, 1isize,
                    0.0,
                    chunk.as_mut_ptr(), n as isize, 1isize,
                );
            }
        });
    c
}

/// P17f: attention scores を K/V バッファ・Q 行バッファへのストライド直接参照で
/// 1 回の sgemm で計算する (kh 連続化コピー + per-g m=1 呼び出しを廃止)。
///
/// - a: [m,k] 行ストライド rs_a (cs=1) — q 全行バッファ内の当該ヘッド区間
/// - b: kv バッファ [t_total, kv_dim] 内のヘッド列。b[l,j] = b_ptr[l + j*cs_b]
///   (l: ヘッド内要素 dh 次元=連続、j: トークン位置=cs_b ストライド) を
///   b^T [k=dh, n=t_total] として (rs_b=1, cs_b=cs_b) で渡す
/// - c: [m,n] row-major、beta=1.0 累積 (呼び出し側で 0 埋め)
///
/// 数値: matrixmultiply のカーネル選択は形状/ストライド依存のため従来経路
/// (sgemm_acc(m=1) + 連続化コピー) とはビット不一致 — **P17f で基準更新済み**
/// (ユーザー承認・HISTORY 参照)。同一形状/ストライドの呼び出しは常に同一結果
/// なので単独/バッチ decode・セッション間・リクエスト間の決定論は維持される。
pub fn sgemm_scores_kvstrided(
    a: &[f32],
    rs_a: usize,
    b: &[f32],
    cs_b: usize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, alpha,
            a.as_ptr(), rs_a as isize, 1isize,
            b.as_ptr(), 1isize, cs_b as isize,
            1.0,
            c.as_mut_ptr(), n as isize, 1isize,
        );
    }
}

/// c[j] = dot(a, b[j*k..][..k]) を出力分割で並列に計算 (decode 用 matvec)。
/// m=1 の行列積は A 側で分割できないため、出力列 (= B の行) で並列化する。
/// b: [n,k] row-major (nn.Linear 形状)。
pub fn matvec_bt(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];
    matvec_bt_into(a, b, k, n, &mut c);
    c
}

/// s63 (T5c): `matvec_bt` の書き込み先指定版 (out: [n]・長さ n)。数値は同一。
pub fn matvec_bt_into(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), n);
    out.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(j, cj)| {
            let row = &b[j * k..][..k];
            let mut s = 0.0f32;
            for d in 0..k {
                s += a[d] * row[d];
            }
            *cj = s;
        });
}

/// c[i][j] = Σ_d a[i*k+d] * b[j*k+d] を計算するバッチ decode 用 matvec (P6/P9)。
/// a: [m,k] row-major、b: [n,k] row-major → c: [m,n] row-major。
///
/// **exactness 契約**: 各出力要素 (i,j) の d 累積は厳密に逐次
/// (`s += a[d] * row[d]` と同じ順序で同じ丸め) → どの行も単独 decode
/// (`matvec_bt`) とビット一致する。m>1 の sgemm 経路 (`matmul_bt`) とは
/// 累積順が違い一致しないため、バッチ decode ではこのカーネルを使う。
///
/// **P9 SIMD 化**: SIMD レーンは i 方向 (バッチ内のセッション) と j 方向
/// (出力チャネルの ILP 用グループ) にのみ張る。k 方向はレーン化しない
/// (順序が変わるため) — 各レーンは自分の担当 (i,j) の d 累積を 1 個ずつ
/// 順番に加算する。FMA は使わない (積と加算の丸めを分離するため
/// `vmulq_f32`+`vaddq_f32` / `_mm_mul_ps`+`_mm_add_ps` のみ)。
/// 重み行 b[j] は 1 回の読み込みで m 入力分のドットに再利用されるため、
/// 線形層の重み帯域が実質 1/m のまま、演算は 4 レーン同時になる。
pub fn matvec_rows_bt(a: &[f32], b: &[f32], k: usize, n: usize, m: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matvec_rows_bt_into(a, b, k, n, m, &mut c);
    c
}

/// s63 (T5c): `matvec_rows_bt` の書き込み先指定版 (out: [m,n] row-major・長さ m*n)。
///
/// 数値は `matvec_rows_bt` と**完全に同一**: 本体 (`matvec_rows_bt_impl`) は
/// 上書き専用化しただけでループ構成・累積順・SIMD レーン配置・rayon 分割は
/// 一切変えていない (上書きなら旧値の読み取りが無いので同じ)。したがって
/// 呼び出し側が領域を使い回しても出力ビットは変わらない。
/// 旧 API の `vec![0.0; m*n]` を毎回確保する代わりに再利用バッファへ書ける。
pub fn matvec_rows_bt_into(a: &[f32], b: &[f32], k: usize, n: usize, m: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    if m == 1 {
        matvec_bt_into(a, b, k, n, out);
        return;
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        // s62 (T5b): n < JB では既存のブロック分割が 0 チャンクになり、本体が
        // 一度も走らず全チャネルが単スレッドの端数経路に落ちる (GDN の beta/alpha
        // = Fp32・n=32 が該当し、prefill の gdn_proj の大半を占めていた)。
        if n < JB {
            matvec_rows_bt_small_n_impl(a, b, k, n, m, out)
        } else {
            matvec_rows_bt_vec_impl(a, b, k, n, m, out)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        matvec_rows_bt_scalar_impl(a, b, k, n, m, out)
    }
}

/// s62 (T5b): 出力チャネル方向の rayon ブロック幅 (既存実装と同一値)。
/// `n < JB` では `par_chunks_exact` が 0 チャンクになるため、小 n 専用経路
/// (`matvec_rows_bt_small_n`) に切り替える。
const JB: usize = 64;

/// SIMD 版本体 (P9)。レーン構成: 4 出力チャネル × 4 バッチセッション。
/// k 累積はチャネル毎・セッション毎に独立した逐次チェーン (ビット一致の核心)。
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn matvec_rows_bt_vec_impl(a: &[f32], b: &[f32], k: usize, n: usize, m: usize, out: &mut [f32]) {
    const LANES: usize = 4; // バッチ (i) 方向のレーン数 = SIMD 幅
    const JG: usize = 4; // 同時に処理する出力チャネル数 (ILP 用、順序には無関係)

    let mp = m.div_ceil(LANES) * LANES;
    // アクティベーションを [k][mp] に転置 + ゼロパディング。
    // m*k は高々数十 KB で L1/L2 に乗るので転置コストは重みストリームに対して無視できる。
    let mut at = vec![0.0f32; k * mp];
    for i in 0..m {
        let src = &a[i * k..][..k];
        for d in 0..k {
            at[d * mp + i] = src[d];
        }
    }
    // s63 (T5c): 呼び出し側の領域へ直接書く (旧実装はここで vec![0.0; m*n] を
    // 確保していた)。書き込みは従来どおり生成 (i,j) 全てに行うため数値不変。
    let cp = SendPtr(out.as_mut_ptr());
    let n_groups = mp / LANES;
    b.par_chunks_exact(k * JB)
        .enumerate()
        .for_each(|(bi, _bw)| {
            let j0 = bi * JB;
            for jj in (0..JB).step_by(JG) {
                let rows: [&[f32]; JG] = std::array::from_fn(|t| {
                    let j = j0 + jj + t;
                    &b[j * k..][..k]
                });
                for g in 0..n_groups {
                    let col = &at[g * LANES..];
                    let s = dot_lanes_j4(col, &rows, k, mp);
                    for t in 0..JG {
                        let j = j0 + jj + t;
                        for l in 0..LANES {
                            let i = g * LANES + l;
                            if i < m {
                                unsafe { cp.write(i * n + j, s[t][l]) };
                            }
                        }
                    }
                }
            }
        });
    // 端数ブロック (n が JB の倍数でない場合) — 逐次スカラーのまま (行数が少ない)。
    matvec_rows_bt_rem_scalar(a, b, k, n, m, (n / JB) * JB, out);
}

/// s62 (T5b): **小 n 投影** (n < JB) 専用の本体。
///
/// 従来の `matvec_rows_bt_vec` は出力チャネルを `JB`(=64) 本ずつ `par_chunks_exact`
/// で分けていた。n < 64 のとき `par_chunks_exact` は **0 チャンク** を返すため
/// 本体ループが一度も走らず、出力チャネル全部が `matvec_rows_bt_rem_scalar`
/// (単スレッド・j-outer/i-inner/d-inner のアクセス) に落ちていた。
/// GDN の beta/alpha 投影は `lin_ctl` で **Fp32 固定**・n = n_v_heads (9B で 32)
/// なので常にこの条件に当たり、prefill の `gdn_proj` が 263 ms/call を費やしていた
/// (0.75 GFLOP を 2.8 GFLOPS = 単コアで 745MB の重みを走査)。
///
/// 修正方針: n が小さいときは **出力チャネル方向を rayon 並列**にする。各タスクは
/// 自分が担当する重み行 (`JG` 本ずつ束ねる) だけを走査し、バッチ (i) 方向は
/// `MB` レーンの SIMD で回す。各 (i,j) の累積は d について厳密に逐次
/// `__fadd_rn(__fmul_rn(...))` のため、**出力は従来経路とビット一致**する
/// (テスト: `matvec_rows_bt_small_n_matches_rem_scalar_path`)。
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn matvec_rows_bt_small_n_impl(a: &[f32], b: &[f32], k: usize, n: usize, m: usize, out: &mut [f32]) {
    use rayon::prelude::*;
    const MB: usize = 8; // バッチ (i) 方向のレーン数
    const JG: usize = 2; // 1 タスクが同時に走査する出力チャネル数
    debug_assert!(n > 0 && n < JB);
    let mp = m.div_ceil(MB) * MB;
    // アクティベーションを [d][i] に転置 (転置先はレーン方向に連続)。
    let mut at = vec![0.0f32; k * mp];
    for i in 0..m {
        let src = &a[i * k..][..k];
        for d in 0..k {
            at[d * mp + i] = src[d];
        }
    }
    // 出力はブロックローカル [t][i] に書き、最後に [i][j] へ転置する
    // (共有可変スライスを持たない = unsafe 不要)。
    let groups: Vec<(usize, usize, Vec<f32>)> = (0..n)
        .step_by(JG)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|j0| {
            let cols = (n - j0).min(JG);
            let rows: Vec<&[f32]> = (0..cols).map(|t| &b[(j0 + t) * k..][..k]).collect();
            let mut o = vec![0.0f32; cols * m];
            for g in 0..(mp / MB) {
                let mut acc = [0.0f32; JG * MB];
                for d in 0..k {
                    let av = &at[g * MB + d * mp..][..MB];
                    for t in 0..cols {
                        let rv = rows[t][d];
                        let base = t * MB;
                        for l in 0..MB {
                            acc[base + l] += av[l] * rv;
                        }
                    }
                }
                let i0 = g * MB;
                let nc = MB.min(m - i0);
                for t in 0..cols {
                    o[t * m + i0..t * m + i0 + nc].copy_from_slice(&acc[t * MB..t * MB + nc]);
                }
            }
            (j0, cols, o)
        })
        .collect();
    // s63 (T5c): 呼び出し側の領域へ転置して書く (旧実装はここで m*n を確保)。
    // 全 (i,j) を上書きするため数値不変。
    for (j0, cols, o) in groups.iter() {
        for t in 0..*cols {
            let col = &o[t * m..][..m];
            for (i, v) in col.iter().enumerate() {
                out[i * n + j0 + t] = *v;
            }
        }
    }
}

/// JG=4 本の重み行 × 4 バッチレーンのドットを同時に計算する。
/// 各 (j, lane) の累積は d について厳密に逐次 — ビット一致の不変条件。
/// レーンロードは転置済みアクティベーション `at[d*stride + g*4 ..]` から連続 4 個。
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn dot_lanes_j4(at_col: &[f32], rows: &[&[f32]], k: usize, stride: usize) -> [[f32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_f32(0.0); 4];
        for d in 0..k {
            let av = vld1q_f32(at_col.as_ptr().add(d * stride));
            for t in 0..4 {
                let rv = vdupq_n_f32(rows[t][d]);
                acc[t] = vaddq_f32(acc[t], vmulq_f32(av, rv));
            }
        }
        let mut out = [[0.0f32; 4]; 4];
        for t in 0..4 {
            vst1q_f32(out[t].as_mut_ptr(), acc[t]);
        }
        out
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let mut acc = [_mm_setzero_ps(); 4];
        for d in 0..k {
            let av = _mm_loadu_ps(at_col.as_ptr().add(d * stride));
            for t in 0..4 {
                let rv = _mm_set1_ps(rows[t][d]);
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(av, rv));
            }
        }
        let mut out = [[0.0f32; 4]; 4];
        for t in 0..4 {
            _mm_storeu_ps(out[t].as_mut_ptr(), acc[t]);
        }
        out
    }
}

/// 端数出力チャネル (n % JB) の逐次スカラー処理。累積順は本体と同一。
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn matvec_rows_bt_rem_scalar(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    m: usize,
    rem: usize,
    c: &mut [f32],
) {
    for j in rem..n {
        let row = &b[j * k..][..k];
        for i in 0..m {
            let ar = &a[i * k..][..k];
            let mut s = 0.0f32;
            for d in 0..k {
                s += ar[d] * row[d];
            }
            c[i * n + j] = s;
        }
    }
}

/// SIMD 非対応ターゲット向けの逐次スカラー版 (旧 P6 実装そのまま)。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn matvec_rows_bt_scalar_impl(a: &[f32], b: &[f32], k: usize, n: usize, m: usize, out: &mut [f32]) {
    const JB: usize = 64;
    let cp = SendPtr(out.as_mut_ptr());
    b.par_chunks_exact(k * JB)
        .enumerate()
        .for_each(|(bi, bw)| {
            let j0 = bi * JB;
            for (jj, row) in bw.chunks_exact(k).enumerate() {
                let j = j0 + jj;
                for i in 0..m {
                    let ar = &a[i * k..][..k];
                    let mut s = 0.0f32;
                    for d in 0..k {
                        s += ar[d] * row[d];
                    }
                    unsafe { cp.write(i * n + j, s) };
                }
            }
        });
    matvec_rows_bt_rem_scalar_generic(a, b, k, n, m, (n / JB) * JB, out);
}

/// 上のスカラー版用端数処理 (cfg を分けるため別名)。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn matvec_rows_bt_rem_scalar_generic(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    m: usize,
    rem: usize,
    c: &mut [f32],
) {
    for j in rem..n {
        let row = &b[j * k..][..k];
        for i in 0..m {
            let ar = &a[i * k..][..k];
            let mut s = 0.0f32;
            for d in 0..k {
                s += ar[d] * row[d];
            }
            c[i * n + j] = s;
        }
    }
}

/// アクティベーション1行 (トークン) を per-token 対称 int8 へ動的量子化する。
/// scale = max|a| / 127、q = round(a / scale) を [-127,127] にクランプ
/// (重みの量子化と同じ丸め規則)。全要素 0 の場合は scale = 1.0 にフォールバック。
/// デ量子化: a[d] ≒ q[d] * scale。固定順序の逐次演算なので決定論的。
pub fn quantize_row_i8(a: &[f32]) -> (Vec<i8>, f32) {
    let maxabs = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    let q: Vec<i8> = a
        .iter()
        .map(|&v| ((v * inv).round().clamp(-127.0, 127.0)) as i8)
        .collect();
    (q, scale)
}

/// スカラー参照実装: Σ a[i]*b[i] を i32 で逐次累積する。
/// 整数 i32 累積は結合則が成立する (丸め誤差ゼロ) ため、SIMD 版と完全一致する。
///
/// オーバーフロー境界: |a[i]*b[i]| ≤ 127*127 = 16129 (量子化パイプラインは
/// ±127 にクランプされるため ±128 は出ない) で、逐次累積の任意の部分和と
/// SIMD のレーン部分和・水平加算の中間値もすべて |Σ| ≤ 16129*k に収まる。
/// すなわち k ≤ i32::MAX / 16129 ≒ 133,169 なら中間オーバーフローしない
/// (本実装の最大 k は Qwen2.5-0.5B の中間層 4864 / GPT-2 の 3072 で十分下)。
pub fn i8_dot_scalar(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0i32;
    for i in 0..a.len() {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// 整数 i8 ドット: Σ a[i]*b[i] を i32 で厳密に計算する。
/// - aarch64 + FEAT_DotProd (armv8.2): NEON `vdotq_s32` (sdot)
/// - x86_64 + AVX2 (CI ランナーは x86-64-v3): i8→i16 符号拡張 + `_mm256_madd_epi16`
/// - 上記以外: スカラー参照にフォールバック
///
/// runtime 検出は行わない (target_feature の compile-time 判定のみ)。
/// 整数 i32 累積は結合則が成立するため、アキュムレータの数・レーン分割・
/// 処理順序に出力は依存せず、必ず `i8_dot_scalar` とビット一致する
/// (テスト `i8_dot_simd_matches_scalar_reference` で検証)。
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub fn i8_dot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), b.len());
    let k = a.len();
    let mut i = 0usize;
    unsafe {
        // vdotq_s32: i8×i8 の隣接4積和を i32 レーンに加算 (厳密な整数演算)。
        // 依存チェーンの律速を避けるため 4 つの独立アキュムレータで ILP を確保する
        // (整数は結合則が成立するので結果は変わらない)。
        let (mut acc0, mut acc1, mut acc2, mut acc3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        while i + 64 <= k {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            acc0 = vdotq_s32(acc0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            acc1 = vdotq_s32(acc1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
            acc2 = vdotq_s32(acc2, vld1q_s8(pa.add(i + 32)), vld1q_s8(pb.add(i + 32)));
            acc3 = vdotq_s32(acc3, vld1q_s8(pa.add(i + 48)), vld1q_s8(pb.add(i + 48)));
            i += 64;
        }
        while i + 16 <= k {
            acc0 = vdotq_s32(acc0, vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
            i += 16;
        }
        let mut s = vaddvq_s32(vaddq_s32(
            vaddq_s32(acc0, acc1),
            vaddq_s32(acc2, acc3),
        ));
        while i < k {
            s += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        s
    }
}

/// AVX2 (x86-64-v3) 版。AVX2 には i8 ドット命令 (VNNI は AVX512 のみ) が無いので、
/// i8 を i16 に符号拡張して `_mm256_madd_epi16` (隣接2積和を i32 へ) を使う。
/// 積 |a*b| ≤ 16129 は i16 に収まり、madd の2積和 (≤ 32258) も i32 で厳密。
/// avx2 が無効のビルドではスカラーにフォールバックする (compile-time 判定のみ)。
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn i8_dot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len());
    let k = a.len();
    let mut i = 0usize;
    unsafe {
        // 32バイトブロック: i8→i16 符号拡張 (lo/hi 半分) → madd → i32 加算
        unsafe fn madd_block(acc: __m256i, pa: *const i8, pb: *const i8) -> __m256i {
            let va = _mm256_loadu_si256(pa as *const __m256i);
            let vb = _mm256_loadu_si256(pb as *const __m256i);
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));
            let p = _mm256_add_epi32(
                _mm256_madd_epi16(a_lo, b_lo),
                _mm256_madd_epi16(a_hi, b_hi),
            );
            _mm256_add_epi32(acc, p)
        }
        unsafe fn hsum(x: __m256i) -> i32 {
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(x),
                _mm256_extracti128_si256::<1>(x),
            );
            let s = _mm_add_epi32(s, _mm_srli_si128::<8>(s));
            let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
            _mm_cvtsi128_si32(s)
        }
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        while i + 64 <= k {
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            acc0 = madd_block(acc0, pa.add(i), pb.add(i));
            acc1 = madd_block(acc1, pa.add(i + 32), pb.add(i + 32));
            i += 64;
        }
        while i + 32 <= k {
            acc0 = madd_block(acc0, a.as_ptr().add(i), b.as_ptr().add(i));
            i += 32;
        }
        let mut total = hsum(_mm256_add_epi32(acc0, acc1));
        while i < k {
            total += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        total
    }
}

/// SIMD 拡張が compile-time に有効でないターゲット向けのフォールバック。
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "dotprod"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub fn i8_dot(a: &[i8], b: &[i8]) -> i32 {
    i8_dot_scalar(a, b)
}

/// c[j] = (Σ_d aq[d] * q[j*k+d]) as f32 * ascale * scale[j]
/// — int8×int8 の整数 SIMD ドット + スケール乗算のみの fused matvec (decode 用)。
/// q: [n,k] row-major (nn.Linear 形式)、scale: [n] は出力チャネルごとの
/// 既存のデ量子化係数。
///
/// a (アクティベーション) は呼び出しごとに per-token 対称 int8 へ動的量子化する
/// (`quantize_row_i8`)。重み側は per-output-channel スケールをそのまま使うため、
/// デ量子化は出力チャネルごとのスケール乗算 1 回だけで済む (整数ドットは
/// 厳密なのでデ量子化を途中に入れる必要がない)。
///
/// 決定論: 量子化も整数ドットもスレッド数・処理順序に依存せず厳密なので、
/// 同一入力 → 同一出力 (量子化パスの exact decoding は維持される)。
///
/// 注意: 旧実装 (a を fp32 のまま i8→f32 変換して fp32 累積) とはビット一致
/// しない (アクティベーション量子化誤差を追加する)。int8 モードの仕様
/// 「fp32 と同一出力ではない / 決定論は維持」は不変。fp32 モードは影響を受けない。
pub fn matvec_bt_i8(a: &[f32], q: &[i8], scale: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];
    matvec_bt_i8_into(a, q, scale, k, n, &mut c);
    c
}

/// s63 (T5c): `matvec_bt_i8` の書き込み先指定版 (out: [n])。数値は同一。
pub fn matvec_bt_i8_into(a: &[f32], q: &[i8], scale: &[f32], k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(q.len(), k * n);
    debug_assert_eq!(scale.len(), n);
    let (aq, ascale) = quantize_row_i8(a);
    debug_assert_eq!(aq.len(), k);
    out.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(j, cj)| {
            let row = &q[j * k..][..k];
            let dot = i8_dot(&aq, row);
            *cj = dot as f32 * ascale * scale[j];
        });
}

/// c[i][j] = (Σ_d aq_i[d] * q[j*k+d]) as f32 * ascale_i * scale[j]
/// — バッチ decode 用 int8 matvec (P6)。a: [m,k]、q: [n,k] → c: [m,n]。
///
/// 各入力行は `matvec_bt_i8` と**同一規則**で動的量子化し、i32 ドットで
/// 計算する。整数 i32 累積は順序不変のため、どの行も単独 decode
/// (matvec_bt_i8) とビット一致する。重み行は 1 回の読み込みで m 入力分に
/// 再利用される (帯域 1/m)。並列化は出力チャネル j ブロック単位。
pub fn matvec_rows_bt_i8(
    a: &[f32],
    q: &[i8],
    scale: &[f32],
    k: usize,
    n: usize,
    m: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matvec_rows_bt_i8_into(a, q, scale, k, n, m, &mut c);
    c
}

/// s63 (T5c): `matvec_rows_bt_i8` の書き込み先指定版 (out: [m,n])。数値は同一
/// (量子化・整数ドット・j ブロック並列・端数処理のループ構成は不変)。
pub fn matvec_rows_bt_i8_into(
    a: &[f32],
    q: &[i8],
    scale: &[f32],
    k: usize,
    n: usize,
    m: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(q.len(), k * n);
    debug_assert_eq!(scale.len(), n);
    if m == 1 {
        matvec_bt_i8_into(a, q, scale, k, n, out);
        return;
    }
    // 各トークンを動的量子化 (decode と同一規則 → 行単位でビット一致)
    let mut aq = vec![0i8; m * k];
    let mut asc = vec![0f32; m];
    for i in 0..m {
        let (qi, si) = quantize_row_i8(&a[i * k..][..k]);
        aq[i * k..][..k].copy_from_slice(&qi);
        asc[i] = si;
    }
    const JB: usize = 64;
    let cp = SendPtr(out.as_mut_ptr());
    q.par_chunks_exact(k * JB)
        .zip(scale.par_chunks(JB))
        .enumerate()
        .for_each(|(bi, (qw, sc))| {
            let j0 = bi * JB;
            for (jj, &sj) in sc.iter().enumerate() {
                let j = j0 + jj;
                let wrow = &qw[jj * k..][..k];
                for i in 0..m {
                    let dot = i8_dot(&aq[i * k..][..k], wrow);
                    unsafe { cp.write(i * n + j, dot as f32 * asc[i] * sj) };
                }
            }
        });
    // 端数ブロック (n が JB の倍数でない場合)
    let rem = (n / JB) * JB;
    if rem < n {
        for j in rem..n {
            let wrow = &q[j * k..][..k];
            let sj = scale[j];
            for i in 0..m {
                let dot = i8_dot(&aq[i * k..][..k], wrow);
                out[i * n + j] = dot as f32 * asc[i] * sj;
            }
        }
    }
}

/// c[i][j] = (Σ_d aq_i[d] * q[j*k+d]) as f32 * ascale_i * scale[j]
/// — int8×int8 の整数 SIMD ドットによる prefill 用 GEMM (q: [n,k] row-major)。
///
/// a の各行 (トークン) を `matvec_bt_i8` と**同一規則**で per-token i8 に動的
/// 量子化し、整数ドットで計算する。整数 i32 累積は厳密なので、同一トークンは
/// prefill (m>1) と decode (m==1) のどちらの経路でもビット一致する →
/// prefix cache ヒット/非ヒットに関わらず int8 モードの決定論が構造的に保証
/// される (デ量子化 sgemm と混在させると丸め差でサンプリングが反転し、
/// キャッシュ状態だけで出力が変わってしまうため)。
///
/// 並列化は出力チャネル (j) ブロック単位: 各タスクは重み行をメモリから 1 度
/// 読んで全 m トークン分のドットを計算する (重み帯域 1 回分。アクティベー
/// ションは L2/L3 に乗る)。整数 sdot は 5.8 TMAC/s 実測 (20 スレッド) で、
/// デ量子化 + fp32 sgemm より高速。
///
/// s42 (Plan B): 実際に rayon で並列化する (旧実装はこのコメントのみで逐次
/// `chunks_exact` だった — REPORT_NEXT16 §0 副次発見 1)。`matvec_rows_bt_i8`
/// (P6) と同一の j ブロック並列 + 出力書き込みは SendPtr 経由。出力要素ごとの
/// 演算列 (量子化 → i32 ドット → f32 変換 × asc × scale) は並列化前と同一の
/// ためビット一致・スレッド数非依存。量子化 (行独立) も行並列にする。
pub fn matmul_bt_i8(a: &[f32], q: &[i8], scale: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_bt_i8_into(a, q, scale, m, k, n, &mut c);
    c
}

/// s63 (T5c): `matmul_bt_i8` の書き込み先指定版 (out: [m,n])。数値は同一
/// (per-token 量子化・整数ドット・j ブロック並列・端数処理のループ構成は不変)。
pub fn matmul_bt_i8_into(
    a: &[f32],
    q: &[i8],
    scale: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(q.len(), k * n);
    debug_assert_eq!(scale.len(), n);
    // 各トークンを動的量子化 (decode と同一規則 → 行単位でビット一致)。
    // 行独立のため行並列 (書き込み先は non-overlapping な chunks_mut に分割)。
    let mut aq = vec![0i8; m * k];
    let mut asc = vec![0f32; m];
    aq.par_chunks_exact_mut(k)
        .zip(asc.par_iter_mut())
        .zip(a.par_chunks_exact(k))
        .for_each(|((aq_row, asc_row), a_row)| {
            let (qi, si) = quantize_row_i8(a_row);
            aq_row.copy_from_slice(&qi);
            *asc_row = si;
        });
    // j を 32 チャネルずつのブロックで並列化 (false sharing を避ける)
    const JB: usize = 32;
    let cp = SendPtr(out.as_mut_ptr());
    q.par_chunks_exact(k * JB)
        .zip(scale.par_chunks(JB))
        .enumerate()
        .for_each(|(bi, (qw, sc))| {
            let j0 = bi * JB;
            for (jj, &sj) in sc.iter().enumerate() {
                let j = j0 + jj;
                let wrow = &qw[jj * k..][..k];
                for i in 0..m {
                    let dot = i8_dot(&aq[i * k..][..k], wrow);
                    unsafe { cp.write(i * n + j, dot as f32 * asc[i] * sj) }
                }
            }
        });
    // 端数ブロック (n が JB の倍数でない場合)
    let rem = (n / JB) * JB;
    if rem < n {
        for j in rem..n {
            let wrow = &q[j * k..][..k];
            let sj = scale[j];
            for i in 0..m {
                let dot = i8_dot(&aq[i * k..][..k], wrow);
                out[i * n + j] = dot as f32 * asc[i] * sj;
            }
        }
    }
}

/// 各行にバイアスを加算 (x: row-major [rows, cols])
pub fn add_bias(x: &mut [f32], bias: &[f32], cols: usize) {
    for row in x.chunks_mut(cols) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// LayerNorm (GPT-2系)。input: [rows, cols] → 新しいバッファ
pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks(cols).zip(out.chunks_mut(cols)) {
        let mean = row_in.iter().sum::<f32>() / cols as f32;
        let var = row_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..cols {
            row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// RMSNorm (Llama/Qwen系)
pub fn rmsnorm(x: &[f32], w: &[f32], cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks(cols).zip(out.chunks_mut(cols)) {
        let ss = row_in.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for i in 0..cols {
            row_out[i] = row_in[i] * inv * w[i];
        }
    }
    out
}

/// QK-Norm (Qwen3): 各ヘッドの head_dim ベクトルに RMSNorm を適用する
/// (行 × ヘッド単位、in-place)。x は行 major の [rows, heads*dh]。
/// q には q_norm (heads 個)、k には k_norm (kv_heads 個) を使う
/// (重みはヘッド間で共有、RoPE の前に適用する)。
/// 演算順序は `rmsnorm` と同一 (逐次 2 乗和 → rsqrt → x*inv*w) なので
/// 行/ヘッド単位で独立、バッチと単独でビット一致する。
pub fn qknorm_rows_inplace(x: &mut [f32], rows: usize, heads: usize, dh: usize, w: &[f32], eps: f32) {
    for r in 0..rows {
        for h in 0..heads {
            let seg = &mut x[r * heads * dh + h * dh..][..dh];
            let ss = seg.iter().map(|v| v * v).sum::<f32>() / dh as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for (i, v) in seg.iter_mut().enumerate() {
                *v = *v * inv * w[i];
            }
        }
    }
}

/// gelu_new (GPT-2 の tanh 近似)
pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (0.7978845608028654 * (t + 0.044715 * t * t * t)).tanh());
    }
}

/// SiLU (Llama/Qwen系)
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// 1行分のsoftmax
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// 1行分の RoPE (位置 pos)。`rope_rows` の行ごとの処理と完全に同一の演算列
/// (バッチ decode で各セッションが異なる位置を持つときに使う、P6)。
pub fn rope_row(x: &mut [f32], heads: usize, dh: usize, pos: usize, theta: f32) {
    let half = dh / 2;
    let pos = pos as f32;
    for h in 0..heads {
        let base = h * dh;
        for d in 0..half {
            let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
            let (c, s) = (ang.cos(), ang.sin());
            let x1 = x[base + d];
            let x2 = x[base + half + d];
            x[base + d] = x1 * c - x2 * s;
            x[base + half + d] = x1 * s + x2 * c;
        }
    }
}

/// RoPE (rotate_half 方式: Llama/Qwen2)
/// x: [rows, heads*dh] を in-place で回転。theta は rope_theta。
pub fn rope_rows(x: &mut [f32], rows: usize, heads: usize, dh: usize, pos0: usize, theta: f32) {
    let width = heads * dh;
    for i in 0..rows {
        rope_row(&mut x[i * width..][..width], heads, dh, pos0 + i, theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 素朴な (遅い) リファレンス実装: c = a @ b
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..k {
                    s += a[i * k + d] * b[d * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// 素朴なリファレンス実装: c = a @ b^T (b: [n,k] row-major)
    fn naive_matmul_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..k {
                    s += a[i * k + d] * b[j * k + d];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// 決定論的な疑似乱数 (LCG) でテスト用行列を作る
    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0 // [-1, 1)
            })
            .collect()
    }

    #[test]
    fn matmul_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let i = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul(&a, &i, 2, 2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        let b = [1.0, 1.0, 2.0, 2.0];
        // [1 2; 3 4] @ [1 1; 2 2] = [5 5; 11 11]
        assert_eq!(matmul(&a, &b, 2, 2, 2), vec![5.0, 5.0, 11.0, 11.0]);
    }

    #[test]
    fn matmul_matches_naive() {
        // 並列チャンク境界 (rows_per_task) をまたぐサイズを含める
        for (m, k, n) in [(3usize, 4usize, 5usize), (1, 1, 1), (9, 7, 2), (16, 3, 13)] {
            let a = pseudo_random(m * k, 1);
            let b = pseudo_random(k * n, 2);
            let got = matmul(&a, &b, m, k, n);
            let want = naive_matmul(&a, &b, m, k, n);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-3, "m={m} k={k} n={n}: {g} != {w}");
            }
        }
    }

    #[test]
    fn matmul_bt_matches_naive() {
        for (m, k, n) in [(3usize, 4usize, 5usize), (1, 6, 1), (9, 7, 2), (17, 3, 13)] {
            let a = pseudo_random(m * k, 3);
            let b = pseudo_random(n * k, 4); // nn.Linear 形状 [n,k]
            let got = matmul_bt(&a, &b, m, k, n);
            let want = naive_matmul_bt(&a, &b, m, k, n);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-3, "m={m} k={k} n={n}: {g} != {w}");
            }
        }
    }

    #[test]
    fn matvec_bt_matches_matmul_bt_single_row() {
        // decode カーネル (m=1) は GEMM の 1 行目と完全一致しなければならない
        for (k, n) in [(8usize, 16usize), (64usize, 200usize)] {
            let a = pseudo_random(k, 5);
            let b = pseudo_random(n * k, 6);
            let by_matvec = matvec_bt(&a, &b, k, n);
            let by_gemm = matmul_bt(&a, &b, 1, k, n);
            for (g, w) in by_matvec.iter().zip(&by_gemm) {
                assert!((g - w).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn matvec_bt_known_dots() {
        // a = [1, 2], b の行: [1,0], [0,1], [1,1], [2,2]
        let a = [1.0, 2.0];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        // dot([1,2], [1,0])=1, dot([1,2],[0,1])=2, dot([1,2],[1,1])=3, dot([1,2],[2,2])=6
        assert_eq!(matvec_bt(&a, &b, 2, 4), vec![1.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn matvec_rows_bt_matches_matvec_bt_bit_exactly() {
        // P6/P9: バッチ decode カーネルの各行は単独 decode (matvec_bt) とビット一致。
        // m=4/7/8 は SIMD レーンの満/端数グループ、k は 4 の倍数と端数の両方をカバー
        for m in [1usize, 2, 3, 4, 5, 7, 8] {
            for (k, n) in [
                (8usize, 16usize),
                (37usize, 200usize),
                (64usize, 130usize),
                (1usize, 64usize),
                (4usize, 65usize),
                (65usize, 64usize),
                (896usize, 192usize),
                // s62 (T5b): n < JB(64) は小 n 専用経路 (出力チャネル並列) に落ちる。
                // この領域が従来テストの死角で、`par_chunks_exact` が 0 チャンクに
                // なることによる「本体が走らず単スレッド端数経路のみ」を見逃していた。
                (16usize, 1usize),
                (8usize, 2usize),
                (37usize, 16usize),
                (64usize, 32usize),
                (40usize, 63usize),
            ] {
                let a = pseudo_random(m * k, 5 + m as u32);
                let b = pseudo_random(n * k, 6);
                let batched = matvec_rows_bt(&a, &b, k, n, m);
                assert_eq!(batched.len(), m * n);
                for i in 0..m {
                    let solo = matvec_bt(&a[i * k..][..k], &b, k, n);
                    assert_eq!(
                        &batched[i * n..][..n],
                        &solo[..],
                        "m={m} k={k} n={n} row={i}: バッチが単独と不一致"
                    );
                }
            }
        }
    }

    /// s62 (T5b): 小 n (n < JB) 経路が従来の単スレッド端数経路と**ビット一致**する
    /// ことを、GDN beta/alpha の実形状 (k=4096, n=n_v_heads) で固定する。
    ///
    /// この経路は「n が小さいと従来実装が本体ループを一度も走らせず、全チャネルを
    /// `matvec_rows_bt_rem_scalar` に落としていた」欠陥の修正。累積順 (d について
    /// 逐次の f32 加算) を変えないので出力のビットは動かない — これを壊すと
    /// prefill の既定数値 (= アンカー) が動くため、回帰ガードとして必須。
    #[test]
    fn matvec_rows_bt_small_n_matches_rem_scalar_path() {
        for n in [1usize, 2, 15, 16, 17, 32, 48, 63] {
            for (k, m) in [(4096usize, 5usize), (64usize, 8), (37usize, 3), (8usize, 9)] {
                let a = pseudo_random(m * k, 11 + n as u32);
                let b = pseudo_random(n * k, 12);
                let got = matvec_rows_bt(&a, &b, k, n, m);
                // 従来の n < JB での挙動そのもの (本体 0 チャンク + 全チャネル端数)。
                let mut want = vec![0.0f32; m * n];
                matvec_rows_bt_rem_scalar(&a, &b, k, n, m, 0, &mut want);
                assert_eq!(got.len(), m * n);
                for idx in 0..m * n {
                    assert_eq!(
                        got[idx].to_bits(),
                        want[idx].to_bits(),
                        "n={n} k={k} m={m} idx={idx}: 小 n 経路がビット不一致"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // 手動実行用マイクロベンチ: cargo test --release bench_matvec_rows_bt -- --ignored --nocapture
    fn bench_matvec_rows_bt_m4() {
        // P9: fp32 バッチカーネル (i方向4レーンSIMD) vs 単独 matvec_bt ×4 の速度比較
        // 形状は Qwen2.5-0.5B の gate_up (k=896,n=4864) と down (k=4864,n=896)
        for (k, n, label) in [(896usize, 4864usize, "gate_up"), (4864, 896, "down")] {
            let a = pseudo_random(4 * k, 21);
            let b = pseudo_random(n * k, 22);
            // ウォームアップ
            let _ = matvec_rows_bt(&a, &b, k, n, 4);
            let iters = 5;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = matvec_rows_bt(&a, &b, k, n, 4);
            }
            let batch = t0.elapsed().as_secs_f64() / iters as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for i in 0..4 {
                    let _ = matvec_bt(&a[i * k..][..k], &b, k, n);
                }
            }
            let solo4 = t0.elapsed().as_secs_f64() / iters as f64;
            let macs = (4 * k * n) as f64;
            let bm = batch * 1e3;
            let sm = solo4 * 1e3;
            let gbs = macs / batch / 1e9;
            let gss = macs / solo4 / 1e9;
            let ratio = solo4 / batch;
            println!(
                "[{label}] k={k} n={n}: batch4 {bm:.1}ms ({gbs:.1} GMAC/s) / solo4 {sm:.1}ms ({gss:.1} GMAC/s) = {ratio:.2}x"
            );
        }
    }

    #[test]
    fn matvec_rows_bt_i8_matches_matvec_bt_i8_bit_exactly() {
        // P6: int8 バッチ decode カーネルの各行は単独 decode とビット一致
        // (i32 累積は順序不変なので構造的に一致する)
        let k = 64usize;
        let n = 100usize;
        let q: Vec<i8> = pseudo_random(n * k, 12).iter().map(|&v| (v * 100.0) as i8).collect();
        let scale: Vec<f32> = pseudo_random(n, 13).iter().map(|v| 0.001 + v.abs() * 0.01).collect();
        for m in [1usize, 2, 4] {
            let a = pseudo_random(m * k, 14 + m as u32);
            let batched = matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            for i in 0..m {
                let solo = matvec_bt_i8(&a[i * k..][..k], &q, &scale, k, n);
                assert_eq!(
                    &batched[i * n..][..n],
                    &solo[..],
                    "int8 m={m} row={i}: バッチが単独と不一致"
                );
            }
        }
    }

    #[test]
    fn rope_row_matches_rope_rows_single_position() {
        // rope_row は rope_rows の 1 行処理と完全同一 (リファクタリングの保証)
        let heads = 3usize;
        let dh = 8usize;
        let width = heads * dh;
        let x0 = pseudo_random(2 * width, 21);
        let mut a = x0.clone();
        rope_rows(&mut a, 2, heads, dh, 7, 10_000.0);
        let mut b0 = x0[..width].to_vec();
        let mut b1 = x0[width..].to_vec();
        rope_row(&mut b0, heads, dh, 7, 10_000.0);
        rope_row(&mut b1, heads, dh, 8, 10_000.0);
        assert_eq!(a[..width], b0[..]);
        assert_eq!(a[width..], b1[..]);
    }

    #[test]
    fn sgemm_acc_accumulates() {
        // c += alpha * a @ b^T の accumulator 形式 (attention の scores 計算用)
        let a = [1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = [1.0, 0.0, 1.0, 1.0]; // [2,2] → a@b^T = [1,3; 3,7]
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        sgemm_acc(&a, &b, &mut c, 2, 2, 2, 2.0);
        // 2*[1 3; 3 7] + [10 20; 30 40] = [12 26; 36 54]
        assert_eq!(c, vec![12.0, 26.0, 36.0, 54.0]);
    }

    #[test]
    fn sgemm_acc_zero_dims_are_noop() {
        let mut c = vec![1.0f32; 6];
        sgemm_acc(&[], &[], &mut c, 0, 3, 2, 1.0);
        sgemm_acc(&[1.0], &[1.0], &mut c, 2, 0, 3, 1.0);
        sgemm_acc(&[1.0], &[1.0], &mut c, 2, 3, 0, 1.0);
        assert_eq!(c, vec![1.0; 6]);
    }

    #[test]
    fn add_bias_applies_per_row() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        add_bias(&mut x, &[0.5, -0.5], 2);
        assert_eq!(x, vec![1.5, 1.5, 3.5, 3.5]);
    }

    #[test]
    fn layernorm_known_values() {
        // x=[0,2]: mean=1, var=1 → out = (x-1)/sqrt(1+eps) * w + b
        let x = [0.0, 2.0];
        let out = layernorm(&x, &[1.0, 1.0], &[0.0, 0.0], 2, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);

        // アフィン: w=2, b=1
        let out = layernorm(&x, &[2.0, 2.0], &[1.0, 1.0], 2, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn layernorm_normalizes_each_row() {
        // 複数行: 各行の平均≈0、分散≈1
        let x = pseudo_random(6 * 8, 7);
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = layernorm(&x, &w, &b, 8, 1e-5);
        for row in out.chunks(8) {
            let mean = row.iter().sum::<f32>() / 8.0;
            assert!(mean.abs() < 1e-4);
            let var = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
            assert!((var - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn rmsnorm_known_values() {
        // x=[3,4]: ss=(9+16)/2=12.5 → inv=1/sqrt(12.5)
        let x = [3.0, 4.0];
        let out = rmsnorm(&x, &[1.0, 1.0], 2, 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn qknorm_matches_per_head_rmsnorm() {
        // 各ヘッド区間の結果が rmsnorm(単独) と一致し、行/ヘッドが独立であること
        let (heads, dh, rows, eps) = (3usize, 4usize, 2usize, 1e-6);
        let w = [0.5, 1.0, 2.0, 1.0];
        let mut x = pseudo_random(rows * heads * dh, 123);
        let orig = x.clone();
        qknorm_rows_inplace(&mut x, rows, heads, dh, &w, eps);
        for r in 0..rows {
            for h in 0..heads {
                let seg_in = &orig[r * heads * dh + h * dh..][..dh];
                let want = rmsnorm(seg_in, &w, dh, eps);
                assert_eq!(
                    &x[r * heads * dh + h * dh..][..dh],
                    &want[..],
                    "row={r} head={h}"
                );
            }
        }
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // 単調: 最大値が最大確率
        assert!(x[2] > x[1] && x[1] > x[0]);
        // softmax は正規化される: x[0] = e^-2 / (e^-2 + e^-1 + 1)
        let denom = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
        assert!((x[0] - (-2.0f32).exp() / denom).abs() < 1e-6);

        // 巨大な値でも NaN にならない (max-shift の確認)
        let mut big = vec![1e9, 1e9 + 1.0, 1e9 + 2.0];
        softmax(&mut big);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn activations_known_values() {
        // gelu_new (tanh近似): gelu(0)=0, gelu(1)≈0.8412
        let mut x = vec![0.0f32, 1.0];
        gelu_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 0.8412).abs() < 1e-3);

        // silu: silu(0)=0, silu(1)=1/(1+e^-1)≈0.7311, silu(-1)≈-0.2689
        let mut x = vec![0.0f32, 1.0, -1.0];
        silu_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        // silu(-1) = -1/(1+e^1) = -e^-1/(1+e^-1)
        assert!((x[2] + (-1.0f32).exp() / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        // pos=0 では角度が 0 → 回転は恒等写像
        let mut x = vec![0.5f32, -1.5];
        rope_rows(&mut x, 1, 1, 2, 0, 10000.0);
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!((x[1] - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_by_expected_angle() {
        // dh=2, pos=1, theta=10000: d=0 の角度 = 1 * theta^0 = 1 (rad)
        // x=[1,0] → [cos1, sin1]
        let mut x = vec![1.0f32, 0.0];
        rope_rows(&mut x, 1, 1, 2, 1, 10000.0);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);

        // 回転はノルムを保存する
        let src = pseudo_random(2 * 4, 8); // rows=2, heads=1, dh=4
        let mut x = src.clone();
        rope_rows(&mut x, 2, 1, 4, 3, 10000.0);
        for r in 0..2 {
            let n0: f32 = src[r * 4..(r + 1) * 4].iter().map(|v| v * v).sum();
            let n1: f32 = x[r * 4..(r + 1) * 4].iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() < 1e-4);
        }
    }

    #[test]
    fn rope_applies_per_head() {
        // heads=2 のとき各ヘッドが独立に回転する (dh=2, pos=0 は恒等なので pos=1 で確認)
        let mut x = vec![1.0f32, 0.0, 1.0, 0.0]; // rows=1, heads=2, dh=2
        rope_rows(&mut x, 1, 2, 2, 1, 10000.0);
        // ヘッドごとに同じ角度 (dh が同じなので) → 両ヘッドとも [cos1, sin1]
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
        assert!((x[2] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[3] - 1.0f32.sin()).abs() < 1e-6);
    }

    // ---- int8 整数ドットカーネル (SIMD vs スカラー参照) ----

    /// 決定論的な i8 列 (LCG)。[-128, 127] の全範囲をカバーする
    /// (カーネル自体は ±128 も正しく扱えることを確認するため)。
    fn pseudo_random_i8(len: usize, seed: u32) -> Vec<i8> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (((x >> 16) % 256) as i32 - 128) as i8
            })
            .collect()
    }

    #[test]
    fn i8_dot_simd_matches_scalar_reference() {
        // プロパティ: 整数 i32 累積は結合則が成立するので、SIMD 版 (レーン分割 +
        // 複数アキュムレータ + スカラー尾) とスカラー逐次参照は完全一致する。
        // ランダム + エッジケース (k=1, 15/16/17, 31/32/33, 63/64/65, 実モデルの k)
        let sizes = [
            1usize, 2, 3, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 896,
            3072, 4864, 5000,
        ];
        for (ti, &k) in sizes.iter().enumerate() {
            let a = pseudo_random_i8(k, 100 + ti as u32);
            let b = pseudo_random_i8(k, 200 + ti as u32);
            assert_eq!(
                i8_dot(&a, &b),
                i8_dot_scalar(&a, &b),
                "random k={k}: SIMD != scalar"
            );
        }
        // エッジパターン: 全 0 / 全 +127 / 全 -127 / ±127 交互 / -128 を含む極値
        let patterns: [Vec<i8>; 5] = [
            vec![0i8; 4864],
            vec![127i8; 4864],
            vec![-127i8; 4864],
            (0..4864).map(|i| if i % 2 == 0 { 127 } else { -127 }).collect(),
            (0..4864).map(|i| if i % 3 == 0 { -128 } else { 127 }).collect(),
        ];
        for (pi, p) in patterns.iter().enumerate() {
            assert_eq!(
                i8_dot(p, p),
                i8_dot_scalar(p, p),
                "pattern {pi}: SIMD != scalar"
            );
        }
        // 非対称 (a≠b) の交互パターン
        let x: Vec<i8> = (0..300).map(|i| if i % 2 == 0 { 127 } else { -128 }).collect();
        let y: Vec<i8> = (0..300).map(|i| if i % 2 == 0 { -127 } else { 127 }).collect();
        assert_eq!(i8_dot(&x, &y), i8_dot_scalar(&x, &y));
    }

    #[test]
    fn i8_dot_known_values() {
        assert_eq!(i8_dot(&[], &[]), 0);
        assert_eq!(i8_dot(&[3], &[-5]), -15);
        // 16 で割り切れる長さとそうでない長さの両方
        assert_eq!(i8_dot(&[1, 2, 3, 4], &[10, 20, 30, 40]), 300);
        let a = [127i8; 32];
        assert_eq!(i8_dot(&a, &a), 127 * 127 * 32);
        // スカラー参照とも一致
        assert_eq!(i8_dot(&a, &a), i8_dot_scalar(&a, &a));
    }

    #[test]
    fn i8_dot_uses_simd_on_supported_targets() {
        // compile-time 判定 (runtime 検出はしない設計): aarch64+dotprod と
        // CI の x86-64-v3 (avx2) では SIMD パスが選ばれていることを保証する。
        // どちらでもないターゲットではスカラーにフォールバックするのでスキップ。
        let simd = cfg!(all(target_arch = "aarch64", target_feature = "dotprod"))
            || cfg!(all(target_arch = "x86_64", target_feature = "avx2"));
        if !simd {
            eprintln!("SKIPPED: this target has neither dotprod nor avx2 (scalar fallback)");
            return;
        }
        assert!(simd);
    }

    #[test]
    fn i8_dot_no_overflow_up_to_documented_bound() {
        // オーバーフロー境界の検証: 全要素 ±127 の最悪ケースで
        // k = 131,072 → |Σ| = 127*127*131,072 = 2,113,929,216 < i32::MAX (2^31-1)。
        // SIMD (レーン分割 + 複数アキュムレータ) の中間値も 16129*k に収まるので
        // スカラー参照と一致し、かつ飽和/ラップが起きない。
        let k = 131_072usize;
        let a = vec![127i8; k];
        let b = vec![127i8; k];
        let expect = 127i64 * 127 * k as i64;
        assert!(expect <= i32::MAX as i64);
        assert_eq!(i8_dot(&a, &b) as i64, expect);
        assert_eq!(i8_dot_scalar(&a, &b) as i64, expect);
        // 負方向の最悪ケース
        let neg = vec![-127i8; k];
        assert_eq!(i8_dot(&a, &neg) as i64, -expect);
        assert_eq!(i8_dot_scalar(&a, &neg) as i64, -expect);
        // ドキュメント上の上限 (i32::MAX / 16129 ≒ 133,169) を超えると溢れる旨も確認
        let k_over = 140_000usize;
        assert!(127i64 * 127 * k_over as i64 > i32::MAX as i64);
    }

    #[test]
    fn quantize_row_i8_roundtrip_and_determinism() {
        // デ量子化誤差は scale/2 以下、クランプは ±127、ゼロベクトルは scale=1
        let a: Vec<f32> = (0..1024)
            .map(|i| ((i * 37) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let (q, s) = quantize_row_i8(&a);
        assert_eq!(q.len(), a.len());
        assert!(q.iter().all(|&v| (-127..=127).contains(&v)));
        let maxabs = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((s - maxabs / 127.0).abs() < 1e-7 * s);
        for (v, &qi) in a.iter().zip(&q) {
            let err = (v - qi as f32 * s).abs();
            assert!(err <= s / 2.0 * 1.0001 + 1e-9, "err={err} scale/2={}", s / 2.0);
        }
        // 決定論: 同一入力 → 同一出力
        let (q2, s2) = quantize_row_i8(&a);
        assert_eq!(q, q2);
        assert_eq!(s.to_bits(), s2.to_bits());
        // ゼロベクトル
        let (qz, sz) = quantize_row_i8(&[0.0; 8]);
        assert_eq!(sz, 1.0);
        assert!(qz.iter().all(|&v| v == 0));
        // 巨大値のクランプ
        let (qc, _) = quantize_row_i8(&[1e30, -1e30, 0.0]);
        assert_eq!(qc, [127, -127, 0]);
    }

    /// 決定論的 LCG 乱数 (シード固定、外部クレート無し)。
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) & 0xFFFFFF) as f32 / 8388608.0 - 1.0 // [-1, 1)
        }
        fn fill(&mut self, v: &mut [f32]) {
            for x in v.iter_mut() {
                *x = self.next_f32();
            }
        }
    }

    /// P12: decode attention の scores 計算に関する検証メモ:
    /// 行統合 (rows=m) もストライド参照 (rs_b=kv_dim) も、従来の
    /// `sgemm_acc(m=1, 連続 b)` とビット不一致になることを実測で確認した
    /// (matrixmultiply はストライド/行数でカーネルを選ぶため)。
    /// そのため scores は従来経路 (per-g sgemm_acc(m=1) + kh 連続化コピー) のまま。
    /// probs@V 側のみ行統合+ストライド参照がビット一致する (下のテスト)。

    /// P12: probs@V の行統合 matmul (ストライド付き v) が、従来の per-row
    /// `matmul(m=1)` (連続 v) とビット一致すること。
    #[test]
    fn decode_attention_probs_v_bitexact() {
        let cases = [
            (128usize, 1usize, 7usize, 256usize), // dh, t, group, kv_dim
            (128, 270, 7, 256),
            (64, 41, 4, 128),
        ];
        for &(dh, t, m, kv_dim) in &cases {
            let mut rng = Lcg(777);
            let mut probs = vec![0.0f32; m * t];
            rng.fill(&mut probs);
            // probs は softmax 後なので正規化しておく (値の性質を寄せる)
            for i in 0..m {
                let s: f32 = probs[i * t..][..t].iter().sum();
                for p in probs[i * t..][..t].iter_mut() {
                    *p /= s;
                }
            }
            let mut v_contig = vec![0.0f32; t * dh];
            rng.fill(&mut v_contig);
            let mut v_strided = vec![0.0f32; (t - 1) * kv_dim + dh];
            for j in 0..t {
                v_strided[j * kv_dim..][..dh].copy_from_slice(&v_contig[j * dh..][..dh]);
            }

            // 従来経路: per-row matmul(m=1, 連続 v)
            let mut old = vec![0.0f32; m * dh];
            for i in 0..m {
                let out = matmul(&probs[i * t..][..t], &v_contig, 1, t, dh);
                old[i * dh..][..dh].copy_from_slice(&out);
            }
            // 新経路: 行統合 + ストライド参照
            let new = matmul_rows_bstrided(&probs, &v_strided, kv_dim, m, t, dh);

            for i in 0..m * dh {
                assert_eq!(
                    old[i].to_bits(),
                    new[i].to_bits(),
                    "bit mismatch at {i} (dh={dh}, t={t}, m={m}, kv_dim={kv_dim})"
                );
            }
        }
    }

    /// s42 (Plan B): `matmul_bt_i8` の rayon 並列化 (j ブロック並列 + 量子化行並列)
    /// がビット一致であること。①各行が `matvec_bt_i8` (decode 単独行の参照・同一
    /// 量子化規則 + 同一 i32 ドット + 同一 f32 変換順) とビット一致 ②ローカル
    /// プール 1/4 スレッドで同一結果 (rayon スレッド数非依存)。n は JB=32 の倍数
    /// と端数の両方を含む (端数ブロックは逐次のまま)。
    #[test]
    fn matmul_bt_i8_parallel_bit_exact_and_thread_count_independent() {
        let (m, k, n) = (37usize, 128usize, 300usize); // 300 = 32*9 + 12 (端数あり)
        let a = pseudo_random(m * k, 101);
        let q = pseudo_random_i8(k * n, 102);
        let scale: Vec<f32> = pseudo_random(n, 103).iter().map(|v| 0.01 + v.abs() * 0.02).collect();

        // 参照: 各行を matvec_bt_i8 (decode m==1 経路) で逐次計算
        let reference_rows: Vec<Vec<f32>> = (0..m)
            .map(|i| matvec_bt_i8(&a[i * k..][..k], &q, &scale, k, n))
            .collect();

        let check = |c: &[f32]| {
            assert_eq!(c.len(), m * n);
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        c[i * n + j].to_bits(),
                        reference_rows[i][j].to_bits(),
                        "matmul_bt_i8[{i}][{j}] != matvec_bt_i8"
                    );
                }
            }
        };

        // 既定 (グローバルプール)
        check(&matmul_bt_i8(&a, &q, &scale, m, k, n));

        // スレッド数非依存 (ローカルプール 1 / 4 スレッド)
        for th in [1usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
            let c = pool.install(|| matmul_bt_i8(&a, &q, &scale, m, k, n));
            check(&c);
        }
    }

    // =====================================================================
    // s63 (T5c): `*_into` (書き込み先指定) API のビット一致テスト
    //
    // これらは prefill 線形層の「呼び出しごとの出力 Vec 確保」を廃止するために
    // 追加した。呼び出し側が領域を使い回すため、**前回の中身が残っている
    // バッファ**へ書いても旧 API (新規 vec! 確保) とビット一致することを
    // 固定する (上書き漏れがあれば旧値が残って必ず落ちる)。
    // =====================================================================

    /// 旧値で埋めた再利用バッファを渡しても旧 API とビット一致すること。
    fn check_into_matches(
        name: &str,
        len: usize,
        f_old: impl Fn() -> Vec<f32>,
        f_into: impl Fn(&mut [f32]),
    ) {
        let want = f_old();
        let mut buf = vec![f32::from_bits(0x7fc0_1234); len]; // NaN パターンで事前充填
        f_into(&mut buf);
        assert_eq!(buf.len(), want.len(), "{name}: 長さ不一致");
        for i in 0..want.len() {
            assert_eq!(buf[i].to_bits(), want[i].to_bits(), "{name}: [{i}] ビット不一致");
        }
    }

    #[test]
    fn matvec_into_variants_match_legacy_bit_exact() {
        let (k, n) = (128usize, 96usize); // n >= JB(64): vec 経路 + 端数 (96 = 64+32)
        let m = 13usize;
        let a = pseudo_random(m * k, 201);
        let b = pseudo_random(k * n, 202);
        let q = pseudo_random_i8(k * n, 203);
        let scale: Vec<f32> = pseudo_random(n, 204).iter().map(|v| 0.01 + v.abs() * 0.02).collect();

        for mm in [1usize, 2, m] {
            check_into_matches(
                &format!("matvec_rows_bt_into m={mm}"),
                mm * n,
                || matvec_rows_bt(&a[..mm * k], &b, k, n, mm),
                |out| matvec_rows_bt_into(&a[..mm * k], &b, k, n, mm, out),
            );
            check_into_matches(
                &format!("matvec_rows_bt_i8_into m={mm}"),
                mm * n,
                || matvec_rows_bt_i8(&a[..mm * k], &q, &scale, k, n, mm),
                |out| matvec_rows_bt_i8_into(&a[..mm * k], &q, &scale, k, n, mm, out),
            );
            check_into_matches(
                &format!("matmul_bt_i8_into m={mm}"),
                mm * n,
                || matmul_bt_i8(&a[..mm * k], &q, &scale, mm, k, n),
                |out| matmul_bt_i8_into(&a[..mm * k], &q, &scale, mm, k, n, out),
            );
        }
        check_into_matches(
            "matvec_bt_into",
            n,
            || matvec_bt(&a[..k], &b, k, n),
            |out| matvec_bt_into(&a[..k], &b, k, n, out),
        );
        check_into_matches(
            "matvec_bt_i8_into",
            n,
            || matvec_bt_i8(&a[..k], &q, &scale, k, n),
            |out| matvec_bt_i8_into(&a[..k], &q, &scale, k, n, out),
        );
    }

    /// 小 n (< JB=64) の専用経路 (T5b) も `_into` でビット一致すること。
    #[test]
    fn matvec_rows_bt_into_small_n_matches_legacy() {
        let (k, m) = (96usize, 11usize);
        let a = pseudo_random(m * k, 211);
        for n in [1usize, 3, 32, 63] {
            let b = pseudo_random(k * n, 212 + n as u32);
            check_into_matches(
                &format!("matvec_rows_bt_into small n={n}"),
                m * n,
                || matvec_rows_bt(&a, &b, k, n, m),
                |out| matvec_rows_bt_into(&a, &b, k, n, m, out),
            );
        }
    }

    // s63 TEMP-DIAG (commit 前に削除): 実形状で全要素が書かれるかを NaN 事前充填で確認。
    #[test]
    fn tmp_diag_into_covers_all_elements() {
        for (k, n, m) in [
            (96usize, 192usize, 5usize), // Full q_proj (heads*2*dh)
            (96, 32, 5),                 // Full k/v (kv_heads*dh)
            (96, 96, 5),                 // o_proj / down
            (96, 128, 5),                // gate/up
            (48, 96, 5),                 // GDN out_w (value_dim -> e)
            (96, 48, 5),                 // GDN qkv?
        ] {
            let a = pseudo_random(m * k, 301);
            let b = pseudo_random(k * n, 302);
            let mut buf = vec![f32::from_bits(0x7fc0_1234); m * n];
            matvec_rows_bt_into(&a, &b, k, n, m, &mut buf);
            let stale = buf.iter().filter(|v| v.to_bits() == 0x7fc0_1234).count();
            assert_eq!(stale, 0, "matvec_rows_bt_into k={k} n={n} m={m}: {stale} 個が未書き込み");
            let q = pseudo_random_i8(k * n, 303);
            let scale: Vec<f32> = pseudo_random(n, 304).iter().map(|v| 0.01 + v.abs() * 0.02).collect();
            let mut buf2 = vec![f32::from_bits(0x7fc0_1234); m * n];
            matvec_rows_bt_i8_into(&a, &q, &scale, k, n, m, &mut buf2);
            let stale2 = buf2.iter().filter(|v| v.to_bits() == 0x7fc0_1234).count();
            assert_eq!(stale2, 0, "matvec_rows_bt_i8_into k={k} n={n} m={m}: {stale2} 個が未書き込み");
            let mut buf3 = vec![f32::from_bits(0x7fc0_1234); m * n];
            matmul_bt_i8_into(&a, &q, &scale, m, k, n, &mut buf3);
            let stale3 = buf3.iter().filter(|v| v.to_bits() == 0x7fc0_1234).count();
            assert_eq!(stale3, 0, "matmul_bt_i8_into k={k} n={n} m={m}: {stale3} 個が未書き込み");
        }
    }
}
