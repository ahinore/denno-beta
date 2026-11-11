//! task 228: GDN スキャンの WY (UT) 形式チャンク版 — **CPU 参照実装**。
//!
//! デバイスカーネル `dno_gdn_pf_wy32` (`kernels_gdn_pf_wy.cu`) と**同じ演算を同じ
//! 順**で書いた写し。このコンテナには GPU も nvcc も無いので、数値の妥当性
//! (現行のトークン逐次スキャンとの一致) はここで確かめる。
//!
//! # なぜチャンク化するか
//!
//! 現行の prefill スキャン (`gdn_head_step` / `dno_gdn_pf_scan32`) は 1 トークン
//! あたり「S *= eg → sk → d → S += k⊗d → o」の 2 つの 128 行縮約を、状態の
//! 読み書きを挟みながら**直列**に回す。lab t225c の実測は 14,018 トークンで
//! `ssm scan_pf` 1,310 ms (v3)。レーンやブロックを増やしても -12% で頭打ちで、
//! 効いていないのは 1 トークンあたりの依存鎖 (バリア) そのもの。
//!
//! # 数学 (1 ヘッド、チャンク c トークン)
//!
//! 漸化式 (現行と同一):
//!   S_t = eg_t * S_{t-1} + k_t ⊗ d_t,  d_t = b_t (v_t - (eg_t S_{t-1})ᵀ k_t),
//!   o_t = q_tᵀ S_t
//! ここで Λ_t = Π_{m≤t} eg_m (チャンク先頭からの累積減衰) と置くと
//!   S_t = Λ_t S_0 + Σ_{j≤t} (Λ_t/Λ_j) k_j ⊗ d_j
//!   (eg_t S_{t-1})ᵀ k_t = Λ_t (k_tᵀ S_0) + Σ_{j<t} (Λ_t/Λ_j)(k_t·k_j) d_j
//! なので、チャンク内は
//!   D_t = b_t (v_t - Λ_t (k_tᵀ S_0) - Σ_{j<t} A[t][j] D_j),
//!   A[t][j] = (k_t·k_j) * Π_{m=j+1..t} eg_m,
//!   o_t = Λ_t (q_tᵀ S_0) + Σ_{j≤t} (q_t·k_j) Π_{m=j+1..t} eg_m D_j,
//!   S_new = Λ_last S_0 + Σ_j (Λ_last/Λ_j) k_j ⊗ D_j
//! の 3 つの行列積 + 「列ごとに独立な」前進代入で書ける。列 (dv 方向) は互いに
//! 独立なので、代入にバリアは要らない。チャンク間だけが直列。
//!
//! 減衰の比 Π_{m=j+1..t} eg_m は**割り算ではなく積**で作る (Λ が f32 で 0 に
//! 落ちても比は落ちない)。
//!
//! # 数値
//!
//! 演算の並べ替えなので、現行のトークン逐次とは丸め順が違う (許容差は
//! [`crate::gdn::wy`] のテスト `wy_matches_token_scan_tolerance` を見る)。
//! 決定論 (固定順・ヘッド独立・列独立) は保つ。

/// カーネルと共有するチャンク長 (C)。1 ブロック = 1 ヘッドで状態 64 KB を
/// 共有メモリに置くため、C は 32 にして作業域を 132 KB に収める。
pub const WY_CHUNK: usize = 32;

/// チャンク版スキャン 1 ヘッドぶん。`k` / `q` は L2 正規化済み (`q` は
/// q_scale 乗済み)、`v` は conv+silu 済み、`eg` は exp(g)、`b` は sigmoid(beta)。
///
/// `s` は [dk, dv] 行優先 (入出力)、`o` は [n, dv] (出力、gated RMS norm の前)。
/// `c` はチャンク長 (カーネルと同じ値。テストで 8/32 を回す)。
#[allow(clippy::too_many_arguments)]
pub fn wy_head_scan(
    dk: usize,
    dv: usize,
    c: usize,
    k: &[f32],
    q: &[f32],
    v: &[f32],
    eg: &[f32],
    b: &[f32],
    s: &mut [f32],
    o: &mut [f32],
) {
    let n = o.len() / dv.max(1);
    assert!(c >= 1 && dk >= 1 && dv >= 1);
    assert_eq!(k.len(), n * dk);
    assert_eq!(q.len(), n * dk);
    assert_eq!(v.len(), n * dv);
    assert_eq!(eg.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(s.len(), dk * dv);
    // チャンク内の作業域 (カーネルの共有メモリと同じ並び)。
    let mut r_dec = vec![0f32; c * c]; // Π_{m=j+1..t} eg_m (j < t)。対角は 1。
    let mut a = vec![0f32; c * c]; // 減衰込みグラム (j < t)
    let mut g = vec![0f32; c * c]; // q·k の減衰込み (j <= t)
    let mut lam = vec![0f32; c]; // Λ_t
    let mut rl = vec![0f32; c]; // Λ_last / Λ_t
    let mut d = vec![0f32; c * dv]; // D_t
    let mut p = vec![0f32; c * dv]; // Λ_t (k_tᵀ S_0)
    let mut qp = vec![0f32; c * dv]; // Λ_t (q_tᵀ S_0)

    let mut t0 = 0usize;
    while t0 < n {
        let len = (n - t0).min(c);
        // ---- (1) 減衰の比と Λ ------------------------------------------------
        for i in 0..len {
            let mut r = eg[t0 + i];
            for j in (0..i).rev() {
                r_dec[i * c + j] = r;
                r *= eg[t0 + j];
            }
            lam[i] = r;
        }
        // ---- (2) 減衰込みグラム A[t][j] (j < t) と G[t][j] (j <= t) -----------
        for i in 0..len {
            for j in 0..i {
                let mut acc = 0f32;
                for r in 0..dk {
                    acc += k[(t0 + i) * dk + r] * k[(t0 + j) * dk + r];
                }
                a[i * c + j] = acc * r_dec[i * c + j];
            }
            for j in 0..=i {
                let mut acc = 0f32;
                for r in 0..dk {
                    acc += q[(t0 + i) * dk + r] * k[(t0 + j) * dk + r];
                }
                let dec = if j == i { 1.0 } else { r_dec[i * c + j] };
                g[i * c + j] = acc * dec;
            }
        }
        // ---- (3) P = Λ_t (K S_0), QP = Λ_t (Q S_0) ---------------------------
        for i in 0..len {
            for cc in 0..dv {
                let (mut sk, mut sq) = (0f32, 0f32);
                for r in 0..dk {
                    let sv = s[r * dv + cc];
                    sk += k[(t0 + i) * dk + r] * sv;
                    sq += q[(t0 + i) * dk + r] * sv;
                }
                p[i * dv + cc] = sk * lam[i];
                qp[i * dv + cc] = sq * lam[i];
            }
        }
        // ---- (4) 列ごとに独立な前進代入 (バリア不要) -------------------------
        for cc in 0..dv {
            for i in 0..len {
                let mut acc = v[(t0 + i) * dv + cc] - p[i * dv + cc];
                for j in 0..i {
                    acc -= a[i * c + j] * d[j * dv + cc];
                }
                d[i * dv + cc] = b[t0 + i] * acc;
            }
        }
        // ---- (5) o = QP + (G の下三角) D ------------------------------------
        for cc in 0..dv {
            for i in 0..len {
                let mut acc = qp[i * dv + cc];
                for j in 0..=i {
                    acc += g[i * c + j] * d[j * dv + cc];
                }
                o[(t0 + i) * dv + cc] = acc;
            }
        }
        // ---- (6) 状態の更新 (チャンクに 1 回) --------------------------------
        // rl[t] = Λ_last / Λ_t = Π_{m=t+1..len-1} eg_m
        let mut r = 1.0f32;
        for t in (0..len).rev() {
            rl[t] = r;
            r *= eg[t0 + t];
        }
        let lam_last = r; // = Λ_{len-1}
        for rr in 0..dk {
            for cc in 0..dv {
                let mut acc = s[rr * dv + cc] * lam_last;
                for t in 0..len {
                    acc += rl[t] * k[(t0 + t) * dk + rr] * d[t * dv + cc];
                }
                s[rr * dv + cc] = acc;
            }
        }
        t0 += len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::{gdn_head_step};

    fn rnd(seed: &mut u64) -> f32 {
        // xorshift。テスト専用 (再現性だけ要る)。
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32) / 16_777_216.0 - 0.5
    }

    /// 現行のトークン逐次 (`gdn_head_step`) と比べる。k/q は L2 正規化相当の
    /// 大きさ (1/sqrt(dk))、v は O(1)、eg は実際の減衰域 (0.5〜1.0)、b は 0〜1。
    #[test]
    fn wy_matches_token_scan_tolerance() {
        let (dk, dv) = (128usize, 128usize);
        let n = 200usize; // 32 の倍数でない端数も通す
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut k = vec![0f32; n * dk];
        let mut q = vec![0f32; n * dk];
        let mut v = vec![0f32; n * dv];
        for t in 0..n {
            for r in 0..dk {
                k[t * dk + r] = rnd(&mut seed) * 0.25;
                q[t * dk + r] = rnd(&mut seed) * 0.25;
            }
            for cc in 0..dv {
                v[t * dv + cc] = rnd(&mut seed) * 2.0;
            }
        }
        let eg: Vec<f32> = (0..n).map(|_| 0.5 + 0.5 * (rnd(&mut seed) + 0.5)).collect();
        let b: Vec<f32> = (0..n).map(|_| 0.5 + 0.5 * rnd(&mut seed)).collect();
        assert!(eg.iter().all(|&e| (0.0..=1.0).contains(&e)));
        let s_init: Vec<f32> = (0..dk * dv)
            .map(|i| rnd(&mut seed) * 0.1 + (i % 7) as f32 * 0.01)
            .collect();

        // 参照: トークン逐次 (gdn_head_step)。状態は同じ初期値から。
        let mut s_ref = s_init.clone();
        let mut o_ref = vec![0f32; n * dv];
        let (mut sk, mut dl) = (vec![0f32; dv], vec![0f32; dv]);
        for t in 0..n {
            gdn_head_step(
                dk,
                dv,
                &q[t * dk..(t + 1) * dk],
                &k[t * dk..(t + 1) * dk],
                &v[t * dv..(t + 1) * dv],
                eg[t],
                b[t],
                &mut s_ref,
                &mut o_ref[t * dv..(t + 1) * dv],
                &mut sk,
                &mut dl,
            );
        }

        // WY 版
        let mut s_wy = s_init.clone();
        let mut o_wy = vec![0f32; n * dv];
        wy_head_scan(dk, dv, WY_CHUNK, &k, &q, &v, &eg, &b, &mut s_wy, &mut o_wy);

        // 比較: 絶対 1e-4 + 相対 1e-5 (AGENTS.md の標準)。値のスケールは
        // Σ 200 トークンぶんの状態なので O(1)〜O(10)。
        let mut worst_o = 0f32;
        for (a, b2) in o_ref.iter().zip(o_wy.iter()) {
            worst_o = worst_o.max((a - b2).abs() / (1.0 + a.abs()));
        }
        let mut worst_s = 0f32;
        for (a, b2) in s_ref.iter().zip(s_wy.iter()) {
            worst_s = worst_s.max((a - b2).abs() / (1.0 + a.abs()));
        }
        println!("worst o={worst_o:.3e} s={worst_s:.3e}");
        assert!(
            worst_o < 1e-4,
            "出力の最大相対差 {worst_o:.3e} が許容 (1e-4) を超えた"
        );
        assert!(
            worst_s < 1e-4,
            "状態の最大相対差 {worst_s:.3e} が許容 (1e-4) を超えた"
        );
    }

    /// チャンク長を変えても (8 / 32 / 64) 同じ値に収束する (端数チャンクも通す)。
    #[test]
    fn wy_chunk_len_invariant_tolerance() {
        let (dk, dv) = (16usize, 16usize);
        let n = 53usize;
        let mut seed = 0x0bad_c0de_dead_beefu64;
        let k: Vec<f32> = (0..n * dk).map(|_| rnd(&mut seed) * 0.3).collect();
        let q: Vec<f32> = (0..n * dk).map(|_| rnd(&mut seed) * 0.3).collect();
        let v: Vec<f32> = (0..n * dv).map(|_| rnd(&mut seed)).collect();
        let eg: Vec<f32> = (0..n).map(|_| 0.6 + 0.4 * (rnd(&mut seed) + 0.5)).collect();
        let b: Vec<f32> = (0..n).map(|_| 0.5 + 0.5 * rnd(&mut seed)).collect();
        let s0: Vec<f32> = (0..dk * dv).map(|_| rnd(&mut seed) * 0.2).collect();
        let mut outs = Vec::new();
        for c in [8usize, 32, 64] {
            let mut s = s0.clone();
            let mut o = vec![0f32; n * dv];
            wy_head_scan(dk, dv, c, &k, &q, &v, &eg, &b, &mut s, &mut o);
            outs.push((s, o));
        }
        for (s, o) in &outs[1..] {
            for (a, b2) in outs[0].1.iter().zip(o.iter()) {
                assert!(
                    (a - b2).abs() <= 1e-5 * (1.0 + a.abs()),
                    "チャンク長で出力が動いた: {a} vs {b2}"
                );
            }
            for (a, b2) in outs[0].0.iter().zip(s.iter()) {
                assert!(
                    (a - b2).abs() <= 1e-5 * (1.0 + a.abs()),
                    "チャンク長で状態が動いた: {a} vs {b2}"
                );
            }
        }
    }
}
