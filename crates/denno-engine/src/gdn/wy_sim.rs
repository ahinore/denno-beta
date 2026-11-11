//! task 228: カーネル `dno_gdn_pf_wy32` の**構造の写し** (テスト専用)。
//!
//! `wy_head_scan` はチャンク内の数学だけを写したもの。こちらは
//! `kernels_gdn_pf_wy.cu` の
//!   - スレッド → チャネル / (token, col) の割り当て
//!   - 共有メモリのオフセット (`#define WY_*` と同じ値)
//!   - conv 窓をスレッドのレジスタで持ち回る形
//!   - L2 正規化の 32 レーン xor 木
//! まで含めて 1 スレッドずつ順に実行する。バリアは順序を保証するだけなので
//! 逐次実行で等価。
//!
//! これで「チャンクの数学」ではなく「カーネルの配線 (チャネル・歩幅・窓の
//! 持ち回り)」の間違いを、GPU 無しで見つけられる。

pub const WY_DK: usize = 128;
/// カーネルと同じチャンク長。lab t228a で C = 32 の共有メモリ 176,644 B が
/// デバイスの opt-in 上限 99 KB を超えて v3 へ落ちたため C = 8 にした
/// (作業域が C に比例する。91,012 B)。
pub const WY_C: usize = 8;
pub const WY_THREADS: usize = 512;
pub const WY_WARPS: usize = 16;
pub const WY_DCONV: usize = 4;
pub const WY_CS: usize = 3;

pub const OFF_S: usize = 0;
pub const OFF_KC: usize = 16384;
pub const OFF_QC: usize = 17408;
pub const OFF_VC: usize = 18432;
pub const OFF_DC: usize = 19456;
pub const OFF_PC: usize = 20480;
pub const OFF_QPC: usize = 21504;
pub const OFF_RDEC: usize = 22528;
pub const OFF_AC: usize = 22592;
pub const OFF_GC: usize = 22656;
pub const OFF_LAM: usize = 22720;
pub const OFF_RL: usize = 22728;
pub const OFF_EG: usize = 22736;
pub const OFF_BC: usize = 22744;
pub const OFF_LL: usize = 22752;
pub const TOTAL: usize = 22753;

/// カーネル 1 回ぶんの入力 (引数の並びと同じ)。
pub struct SimInput<'a> {
    pub n: usize,
    pub nk: usize,
    pub nv: usize,
    pub conv_dim: usize,
    pub eps: f32,
    pub q_scale: f32,
    pub qkv: &'a [f32],
    pub beta: &'a [f32],
    pub alpha: &'a [f32],
    pub conv_st_in: &'a [f32],
    pub ssm_in: &'a [f32],
    pub wconv: &'a [f32],
    pub wa: &'a [f32],
    pub wdt: &'a [f32],
}

/// カーネル 1 回ぶんの出力。
pub struct SimOutput {
    pub o: Vec<f32>,
    pub conv_st_out: Vec<f32>,
    pub ssm_out: Vec<f32>,
}

/// `dno_gdn_pf_wy32` を 1 スレッドずつ順に実行する。
pub fn kernel_sim(s: &SimInput<'_>) -> SimOutput {
    let n = s.n;
    let nk = s.nk;
    let nv = s.nv;
    let conv_dim = s.conv_dim;
    let kd_total = nk * WY_DK;
    let vd = nv * WY_DK;
    let mut sh = vec![0f32; TOTAL];
    let mut o = vec![0f32; n * vd];
    let mut conv_st_out = s.conv_st_in.to_vec();
    let mut ssm_out = vec![0f32; nv * WY_DK * WY_DK];

    // スレッドごとの conv 窓 (レジスタ)。
    let mut wndq = vec![[0f32; WY_CS]; WY_THREADS];
    let mut wndk = vec![[0f32; WY_CS]; WY_THREADS];
    let mut wndv = vec![[0f32; WY_CS]; WY_THREADS];
    let w4 = |ch: usize, j: usize| s.wconv[ch * WY_DCONV + j];

    for h in 0..nv {
        let kh = h % nk;
        // ---- 状態を共有メモリへ ------------------------------------------
        for idx in 0..WY_DK * WY_DK {
            sh[OFF_S + idx] = s.ssm_in[h * WY_DK * WY_DK + idx];
        }
        // ---- conv 窓の初期化 (スレッドごと) ------------------------------
        for tid in 0..WY_THREADS {
            if tid < WY_DK {
                let chq = kh * WY_DK + tid;
                let chv = 2 * kd_total + h * WY_DK + tid;
                for j in 0..WY_CS {
                    wndq[tid][j] = s.conv_st_in[chq * WY_CS + j];
                    wndv[tid][j] = s.conv_st_in[chv * WY_CS + j];
                }
            } else if tid < 2 * WY_DK {
                let chk = kd_total + kh * WY_DK + (tid - WY_DK);
                for j in 0..WY_CS {
                    wndk[tid][j] = s.conv_st_in[chk * WY_CS + j];
                }
            }
        }

        let mut t0 = 0usize;
        while t0 < n {
            let len = (n - t0).min(WY_C);
            // ---- (1) conv + silu + gate ----------------------------------
            for tid in 0..WY_THREADS {
                if tid < WY_DK {
                    let chq = kh * WY_DK + tid;
                    let chv = 2 * kd_total + h * WY_DK + tid;
                    for tt in 0..len {
                        let row = (t0 + tt) * conv_dim;
                        let x = s.qkv[row + chq];
                        let acc = w4(chq, 0) * wndq[tid][0]
                            + w4(chq, 1) * wndq[tid][1]
                            + w4(chq, 2) * wndq[tid][2]
                            + w4(chq, 3) * x;
                        wndq[tid][0] = wndq[tid][1];
                        wndq[tid][1] = wndq[tid][2];
                        wndq[tid][2] = x;
                        sh[OFF_QC + tt * WY_DK + tid] = acc / (1.0 + (-acc).exp());
                        let xv = s.qkv[row + chv];
                        let accv = w4(chv, 0) * wndv[tid][0]
                            + w4(chv, 1) * wndv[tid][1]
                            + w4(chv, 2) * wndv[tid][2]
                            + w4(chv, 3) * xv;
                        wndv[tid][0] = wndv[tid][1];
                        wndv[tid][1] = wndv[tid][2];
                        wndv[tid][2] = xv;
                        sh[OFF_VC + tt * WY_DK + tid] = accv / (1.0 + (-accv).exp());
                    }
                } else if tid < 2 * WY_DK {
                    let r = tid - WY_DK;
                    let chk = kd_total + kh * WY_DK + r;
                    for tt in 0..len {
                        let row = (t0 + tt) * conv_dim;
                        let x = s.qkv[row + chk];
                        let acc = w4(chk, 0) * wndk[tid][0]
                            + w4(chk, 1) * wndk[tid][1]
                            + w4(chk, 2) * wndk[tid][2]
                            + w4(chk, 3) * x;
                        wndk[tid][0] = wndk[tid][1];
                        wndk[tid][1] = wndk[tid][2];
                        wndk[tid][2] = x;
                        sh[OFF_KC + tt * WY_DK + r] = acc / (1.0 + (-acc).exp());
                    }
                }
                if tid < len {
                    let row = (t0 + tid) * nv + h;
                    let a = s.alpha[row] + s.wdt[h];
                    let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
                    sh[OFF_EG + tid] = (s.wa[h] * sp).exp();
                    let bt = s.beta[row];
                    sh[OFF_BC + tid] = 1.0 / (1.0 + (-bt).exp());
                }
            }
            // barrier
            // ---- (2) L2 正規化 (warp ごと) --------------------------------
            for warp in 0..WY_WARPS {
                let mut tt = warp;
                while tt < len {
                    let mut p_q = [0f32; 32];
                    let mut p_k = [0f32; 32];
                    for lane in 0..32 {
                        let mut sq = 0f32;
                        let mut k2 = 0f32;
                        let mut r = lane;
                        while r < WY_DK {
                            let qv = sh[OFF_QC + tt * WY_DK + r];
                            let kv = sh[OFF_KC + tt * WY_DK + r];
                            sq += qv * qv;
                            k2 += kv * kv;
                            r += 32;
                        }
                        p_q[lane] = sq;
                        p_k[lane] = k2;
                    }
                    for off in [16usize, 8, 4, 2, 1] {
                        let old_q = p_q;
                        let old_k = p_k;
                        for l in 0..32 {
                            p_q[l] = old_q[l] + old_q[l ^ off];
                            p_k[l] = old_k[l] + old_k[l ^ off];
                        }
                    }
                    let q_inv = (1.0 / (p_q[0] + s.eps).sqrt()) * s.q_scale;
                    let k_inv = 1.0 / (p_k[0] + s.eps).sqrt();
                    let mut r = 0usize;
                    while r < WY_DK {
                        sh[OFF_QC + tt * WY_DK + r] *= q_inv;
                        sh[OFF_KC + tt * WY_DK + r] *= k_inv;
                        r += 1;
                    }
                    tt += WY_WARPS;
                }
            }
            // barrier
            // ---- (3) 減衰の比 / Λ / rl ------------------------------------
            for tid in 0..WY_THREADS {
                if tid < len {
                    let mut r = sh[OFF_EG + tid];
                    let mut j = tid as isize - 1;
                    while j >= 0 {
                        sh[OFF_RDEC + tid * WY_C + j as usize] = r;
                        r *= sh[OFF_EG + j as usize];
                        j -= 1;
                    }
                    sh[OFF_LAM + tid] = r;
                }
                if tid == 0 {
                    let mut r = 1.0f32;
                    let mut t = len as isize - 1;
                    while t >= 0 {
                        sh[OFF_RL + t as usize] = r;
                        r *= sh[OFF_EG + t as usize];
                        t -= 1;
                    }
                    sh[OFF_LL] = r;
                }
            }
            // barrier
            // ---- (4) A / G -------------------------------------------------
            for tid in 0..WY_THREADS {
                let mut idx = tid;
                while idx < len * WY_C {
                    let i = idx / WY_C;
                    let j = idx - i * WY_C;
                    idx += WY_THREADS;
                    if j > i {
                        continue;
                    }
                    let mut kg = 0f32;
                    let mut qg = 0f32;
                    for r in 0..WY_DK {
                        let kv = sh[OFF_KC + j * WY_DK + r];
                        kg += sh[OFF_KC + i * WY_DK + r] * kv;
                        qg += sh[OFF_QC + i * WY_DK + r] * kv;
                    }
                    if j < i {
                        sh[OFF_AC + i * WY_C + j] = kg * sh[OFF_RDEC + i * WY_C + j];
                    }
                    let dec = if j == i { 1.0 } else { sh[OFF_RDEC + i * WY_C + j] };
                    sh[OFF_GC + i * WY_C + j] = qg * dec;
                }
            }
            // barrier
            // ---- (5) P / QP ------------------------------------------------
            for tid in 0..WY_THREADS {
                let mut idx = tid;
                while idx < len * WY_DK {
                    let i = idx / WY_DK;
                    let c = idx - i * WY_DK;
                    let mut sk = 0f32;
                    let mut sq = 0f32;
                    for r in 0..WY_DK {
                        let sv = sh[OFF_S + r * WY_DK + c];
                        sk += sh[OFF_KC + i * WY_DK + r] * sv;
                        sq += sh[OFF_QC + i * WY_DK + r] * sv;
                    }
                    sh[OFF_PC + idx] = sk * sh[OFF_LAM + i];
                    sh[OFF_QPC + idx] = sq * sh[OFF_LAM + i];
                    idx += WY_THREADS;
                }
            }
            // barrier
            // ---- (6) 前進代入 ----------------------------------------------
            for tid in 0..WY_DK {
                let c = tid;
                for i in 0..len {
                    let mut acc = sh[OFF_VC + i * WY_DK + c] - sh[OFF_PC + i * WY_DK + c];
                    for j in 0..i {
                        acc -= sh[OFF_AC + i * WY_C + j] * sh[OFF_DC + j * WY_DK + c];
                    }
                    sh[OFF_DC + i * WY_DK + c] = sh[OFF_BC + i] * acc;
                }
            }
            // barrier
            // ---- (7) o ------------------------------------------------------
            for tid in 0..WY_DK {
                let c = tid;
                for i in 0..len {
                    let mut acc = sh[OFF_QPC + i * WY_DK + c];
                    for j in 0..=i {
                        acc += sh[OFF_GC + i * WY_C + j] * sh[OFF_DC + j * WY_DK + c];
                    }
                    o[(t0 + i) * vd + h * WY_DK + c] = acc;
                }
            }
            // barrier
            // ---- (8) 状態の更新 --------------------------------------------
            let lam_last = sh[OFF_LL];
            for tid in 0..WY_THREADS {
                let mut idx = tid;
                while idx < WY_DK * WY_DK {
                    let r = idx / WY_DK;
                    let c = idx - r * WY_DK;
                    let mut acc = sh[OFF_S + idx] * lam_last;
                    for t in 0..len {
                        acc += sh[OFF_RL + t]
                            * sh[OFF_KC + t * WY_DK + r]
                            * sh[OFF_DC + t * WY_DK + c];
                    }
                    sh[OFF_S + idx] = acc;
                    idx += WY_THREADS;
                }
            }
            // barrier
            t0 += len;
        }

        // ---- 書き戻し -----------------------------------------------------
        ssm_out[h * WY_DK * WY_DK..(h + 1) * WY_DK * WY_DK]
            .copy_from_slice(&sh[OFF_S..OFF_S + WY_DK * WY_DK]);
        // conv 窓の書き戻し (q/k は k-head の先頭ブロック h == kh だけが書く)
        for tid in 0..WY_THREADS {
            if tid < WY_DK {
                let chq = kh * WY_DK + tid;
                let chv = 2 * kd_total + h * WY_DK + tid;
                if h == kh {
                    for j in 0..WY_CS {
                        conv_st_out[chq * WY_CS + j] = wndq[tid][j];
                    }
                }
                for j in 0..WY_CS {
                    conv_st_out[chv * WY_CS + j] = wndv[tid][j];
                }
            } else if tid < 2 * WY_DK {
                let chk = kd_total + kh * WY_DK + (tid - WY_DK);
                if h == kh {
                    for j in 0..WY_CS {
                        conv_st_out[chk * WY_CS + j] = wndk[tid][j];
                    }
                }
            }
        }
    }
    SimOutput {
        o,
        conv_st_out,
        ssm_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::{gdn_head_step, l2_norm};

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32) / 16_777_216.0 - 0.5
    }

    /// 現行経路 (conv+silu → head ごとの L2 → トークン逐次のデルタ則) の参照。
    fn reference(s: &SimInput<'_>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (n, nk, nv, conv_dim) = (s.n, s.nk, s.nv, s.conv_dim);
        let kd_total = nk * WY_DK;
        let vd = nv * WY_DK;
        // conv + silu (全チャネル)
        let mut cv = vec![0f32; n * conv_dim];
        let mut win = s.conv_st_in.to_vec();
        for ch in 0..conv_dim {
            let (w0, w1, w2, w3) = (
                s.wconv[ch * WY_DCONV],
                s.wconv[ch * WY_DCONV + 1],
                s.wconv[ch * WY_DCONV + 2],
                s.wconv[ch * WY_DCONV + 3],
            );
            let (mut a, mut b, mut c) = (
                win[ch * WY_CS],
                win[ch * WY_CS + 1],
                win[ch * WY_CS + 2],
            );
            for t in 0..n {
                let x = s.qkv[t * conv_dim + ch];
                let acc = w0 * a + w1 * b + w2 * c + w3 * x;
                a = b;
                b = c;
                c = x;
                cv[t * conv_dim + ch] = acc / (1.0 + (-acc).exp());
            }
            win[ch * WY_CS] = a;
            win[ch * WY_CS + 1] = b;
            win[ch * WY_CS + 2] = c;
        }
        let mut o = vec![0f32; n * vd];
        let mut ssm = s.ssm_in.to_vec();
        let (mut sk, mut dl) = (vec![0f32; WY_DK], vec![0f32; WY_DK]);
        for h in 0..nv {
            let kh = h % nk;
            let mut q = vec![0f32; WY_DK];
            let mut k = vec![0f32; WY_DK];
            let sh = &mut ssm[h * WY_DK * WY_DK..(h + 1) * WY_DK * WY_DK];
            for t in 0..n {
                let row = t * conv_dim;
                q.copy_from_slice(&cv[row + kh * WY_DK..row + kh * WY_DK + WY_DK]);
                k.copy_from_slice(&cv[row + kd_total + kh * WY_DK..row + kd_total + kh * WY_DK + WY_DK]);
                l2_norm(&mut q, s.eps);
                l2_norm(&mut k, s.eps);
                for v in q.iter_mut() {
                    *v *= s.q_scale;
                }
                let a = s.alpha[t * nv + h] + s.wdt[h];
                let sp = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
                let eg = (s.wa[h] * sp).exp();
                let bt = 1.0 / (1.0 + (-s.beta[t * nv + h]).exp());
                let v = &cv[row + 2 * kd_total + h * WY_DK..row + 2 * kd_total + h * WY_DK + WY_DK];
                let (lo, hi) = (t * vd + h * WY_DK, (t + 1) * vd);
                gdn_head_step(
                    WY_DK,
                    WY_DK,
                    &q,
                    &k,
                    v,
                    eg,
                    bt,
                    sh,
                    &mut o[lo..lo + WY_DK],
                    &mut sk,
                    &mut dl,
                );
                let _ = hi;
            }
        }
        (o, win, ssm)
    }

    /// カーネルの構造の写し (スレッド割り当て・共有メモリの歩幅・conv 窓の
    /// 持ち回り・warp 木) が現行経路と一致すること。
    ///
    /// 差は L2 の縮約順 (32 レーン木 vs 昇順) だけなので、許容は相対 1e-5。
    #[test]
    fn wy_kernel_sim_matches_reference() {
        let (nk, nv) = (2usize, 4usize);
        let conv_dim = 2 * nk * WY_DK + nv * WY_DK;
        let n = 70usize; // 32 + 32 + 6 (端数チャンク)
        let mut seed = 0x5eed_1234_5678_9abcu64;
        let qkv: Vec<f32> = (0..n * conv_dim).map(|_| rnd(&mut seed) * 1.5).collect();
        let beta: Vec<f32> = (0..n * nv).map(|_| rnd(&mut seed) * 3.0).collect();
        let alpha: Vec<f32> = (0..n * nv).map(|_| rnd(&mut seed) * 3.0).collect();
        let conv_st: Vec<f32> = (0..conv_dim * WY_CS).map(|_| rnd(&mut seed)).collect();
        let ssm: Vec<f32> = (0..nv * WY_DK * WY_DK)
            .map(|_| rnd(&mut seed) * 0.1)
            .collect();
        let wconv: Vec<f32> = (0..conv_dim * WY_DCONV)
            .map(|_| 0.2 + 0.3 * (rnd(&mut seed) + 0.5))
            .collect();
        let wa: Vec<f32> = (0..nv).map(|_| -0.05 - 0.2 * (rnd(&mut seed) + 0.5)).collect();
        let wdt: Vec<f32> = (0..nv).map(|_| rnd(&mut seed)).collect();
        let q_scale = 1.0f32 / (WY_DK as f32).sqrt();
        let eps = 1e-6f32;
        let input = SimInput {
            n,
            nk,
            nv,
            conv_dim,
            eps,
            q_scale,
            qkv: &qkv,
            beta: &beta,
            alpha: &alpha,
            conv_st_in: &conv_st,
            ssm_in: &ssm,
            wconv: &wconv,
            wa: &wa,
            wdt: &wdt,
        };
        let sim = kernel_sim(&input);
        let (o_ref, conv_ref, ssm_ref) = reference(&input);
        let vd = nv * WY_DK;
        let mut worst_o = 0f32;
        for (a, b) in o_ref.iter().zip(sim.o.iter()) {
            worst_o = worst_o.max((a - b).abs() / (1.0 + a.abs()));
        }
        let mut worst_s = 0f32;
        for (a, b) in ssm_ref.iter().zip(sim.ssm_out.iter()) {
            worst_s = worst_s.max((a - b).abs() / (1.0 + a.abs()));
        }
        let mut worst_c = 0f32;
        for (a, b) in conv_ref.iter().zip(sim.conv_st_out.iter()) {
            worst_c = worst_c.max((a - b).abs());
        }
        println!("sim: worst o={worst_o:.3e} s={worst_s:.3e} conv={worst_c:.3e}");
        assert_eq!(sim.o.len(), n * vd);
        assert!(worst_o < 1e-5, "o の最大相対差 {worst_o:.3e} が 1e-5 を超えた");
        assert!(worst_s < 1e-5, "状態の最大相対差 {worst_s:.3e} が 1e-5 を超えた");
        assert!(worst_c < 1e-6, "conv 窓の差 {worst_c:.3e} が 1e-6 を超えた");
    }
}
