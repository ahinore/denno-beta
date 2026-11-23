//! task 668: 合成データだけで回る単体テストのうち、分割の都合で独立させた分。

use super::*;

/// task 225: `dno_router_softmax_topk` (kernels_router225.cu) の**アルゴリズムの
/// 写し**。CUDA をこのコンテナで走らせられないので、選択規則 (値の降順・同値は
/// 小さい index・出力は index 昇順) と再正規化の加算順がホストの
/// [`router_softmax_rows`] + [`router_topk_rows`] と同じ結果になることを、
/// 丸めの木だけを写して確かめる。
///
/// - レーン l は index l, l+32, l+64, ... を持つ (CUDA と同じ昇順)
/// - 和は「レーン内の昇順部分和 → xor 1,2,4,8,16 の木」(CUDA と同じ結合順)
/// - レーンごとの上位 k は挿入法、全体の上位 k は k 回の最大
#[cfg(test)]
mod task225_router_dev_mirror {
    use super::{router_softmax_rows, router_topk_rows};

    fn mirror(logits: &[f32], ne: usize, k: usize) -> (Vec<usize>, Vec<f32>) {
        let nt = logits.len() / ne;
        let mut sels = vec![0usize; nt * k];
        let mut wts = vec![0f32; nt * k];
        for t in 0..nt {
            let row = &logits[t * ne..(t + 1) * ne];
            // --- max (全レーンの max の max) ---
            let mut m = f32::NEG_INFINITY;
            for l in 0..32 {
                let mut j = 0;
                while l + j * 32 < ne {
                    let v = row[l + j * 32];
                    if v > m {
                        m = v;
                    }
                    j += 1;
                }
            }
            // --- exp とレーン内の昇順部分和 ---
            let mut e = vec![0f32; ne];
            let mut lane = [0f32; 32];
            for l in 0..32 {
                let mut s = 0.0f32;
                let mut j = 0;
                while l + j * 32 < ne {
                    let i = l + j * 32;
                    let ev = (row[i] - m).exp();
                    e[i] = ev;
                    s += ev;
                    j += 1;
                }
                lane[l] = s;
            }
            // --- xor 木 (CUDA の s += shfl_xor(s, off) と同じ) ---
            for off in [1usize, 2, 4, 8, 16] {
                let prev = lane;
                for l in 0..32 {
                    lane[l] = prev[l] + prev[l ^ off];
                }
            }
            let s_total = lane[0];
            let divide = s_total > 0.0 && s_total == s_total;
            let p = |i: usize| if divide { e[i] / s_total } else { e[i] };
            // --- レーンごとの上位 k (挿入法。i は昇順) ---
            let mut lv = vec![vec![0f32; k]; 32];
            let mut li = vec![vec![0usize; k]; 32];
            let mut cnt = vec![0usize; 32];
            for l in 0..32 {
                let mut j = 0;
                while l + j * 32 < ne {
                    let i = l + j * 32;
                    let pv = p(i);
                    let full = cnt[l] == k;
                    if full && !(pv > lv[l][k - 1]) {
                        j += 1;
                        continue;
                    }
                    let mut pos = if cnt[l] < k { cnt[l] } else { k - 1 };
                    while pos > 0
                        && (pv > lv[l][pos - 1] || (pv == lv[l][pos - 1] && i < li[l][pos - 1]))
                    {
                        lv[l][pos] = lv[l][pos - 1];
                        li[l][pos] = li[l][pos - 1];
                        pos -= 1;
                    }
                    lv[l][pos] = pv;
                    li[l][pos] = i;
                    if cnt[l] < k {
                        cnt[l] += 1;
                    }
                    j += 1;
                }
            }
            // --- k 回の最大 (値の降順 → index 昇順) ---
            let mut cur = vec![0usize; 32];
            let mut sv = Vec::with_capacity(k);
            let mut si = Vec::with_capacity(k);
            for _ in 0..k {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = usize::MAX;
                for l in 0..32 {
                    if cur[l] < cnt[l] {
                        let (v, i) = (lv[l][cur[l]], li[l][cur[l]]);
                        if v > bv || (v == bv && i < bi) {
                            bv = v;
                            bi = i;
                        }
                    }
                }
                for l in 0..32 {
                    if cur[l] < cnt[l] && li[l][cur[l]] == bi {
                        cur[l] += 1;
                    }
                }
                sv.push(bv);
                si.push(bi);
            }
            // --- index 昇順へ (挿入法) ---
            for a in 1..k {
                let (v, i) = (sv[a], si[a]);
                let mut b = a;
                while b > 0 && si[b - 1] > i {
                    sv[b] = sv[b - 1];
                    si[b] = si[b - 1];
                    b -= 1;
                }
                sv[b] = v;
                si[b] = i;
            }
            // --- 再正規化 (index 昇順の逐次和) ---
            let mut sum = 0.0f32;
            for j in 0..k {
                sum += sv[j];
            }
            let ok = sum == sum && sum > 0.0 && sum < f32::INFINITY;
            for j in 0..k {
                sels[t * k + j] = si[j];
                wts[t * k + j] = if ok { sv[j] / sum } else { 1.0 / k as f32 };
            }
        }
        (sels, wts)
    }

    /// 選択 (index) はホストと完全一致し、重みは丸めの差だけ。
    #[test]
    fn mirror_matches_host_selection() {
        let (ne, k, nt) = (512usize, 8usize, 37usize);
        // 同値を含む入力を混ぜる (index の刻みを粗くして衝突を作る)。
        let mut xs = vec![0f32; nt * ne];
        for t in 0..nt {
            for i in 0..ne {
                let v = (((t * 7 + i * 13) % 97) as f32) * 0.01 - 0.4;
                xs[t * ne + i] = if i % 5 == 0 { (v * 4.0).round() * 0.25 } else { v };
            }
        }
        let (ds, dw) = mirror(&xs, ne, k);
        let mut probs = xs.clone();
        router_softmax_rows(&mut probs, ne, None);
        let mut hs = vec![0usize; nt * k];
        let mut hw = vec![0f32; nt * k];
        router_topk_rows(&probs, ne, k, &mut hs, &mut hw, None, true, None);
        assert_eq!(ds, hs, "選択 index がホストと違う");
        for i in 0..nt * k {
            let a = dw[i];
            let b = hw[i];
            assert!(
                (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                "重み {i}: device {a} vs host {b}"
            );
        }
        // 各行の重みの和は 1。
        for t in 0..nt {
            let s: f32 = dw[t * k..(t + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "行 {t} の和 {s}");
        }
    }
}

// task 261: `DENNO_TENSOR_TYPES` の役割分類 (診断の集計キー)。実 GGUF 無しで
// 名前 → 役割の対応だけを固定する。
#[test]
fn tensor_role_groups_by_suffix() {
    assert_eq!(tensor_role("blk.7.ffn_gate_exps.weight"), "expert_gate");
    assert_eq!(tensor_role("blk.7.ffn_up_exps.weight"), "expert_up");
    assert_eq!(tensor_role("blk.7.ffn_down_exps.weight"), "expert_down");
    assert_eq!(tensor_role("blk.7.ffn_gate_shexp.weight"), "shared");
    assert_eq!(tensor_role("blk.7.ffn_up_shexp.weight"), "shared");
    assert_eq!(tensor_role("blk.7.ffn_gate_inp.weight"), "router");
    assert_eq!(tensor_role("blk.7.ffn_gate_inp_shexp.weight"), "router");
    assert_eq!(tensor_role("blk.7.attn_qkv.weight"), "attn_gdn");
    assert_eq!(tensor_role("blk.7.ssm_out.weight"), "attn_gdn");
    assert_eq!(tensor_role("blk.7.attn_q.weight"), "attn_gdn");
    assert_eq!(tensor_role("blk.3.indexer.q_proj.weight"), "indexer");
    assert_eq!(tensor_role("blk.7.hc_attn_up.weight"), "mix");
    assert_eq!(tensor_role("blk.1.ple_key.weight"), "ple_layer");
    assert_eq!(tensor_role("per_layer_token_embd.weight"), "ple_table");
    assert_eq!(tensor_role("token_embd.weight"), "embed");
    assert_eq!(tensor_role("output.weight"), "head");
    assert_eq!(tensor_role("output_hc_norm.weight"), "head");
    assert_eq!(tensor_role("blk.0.ssm_a"), "attn_gdn");
}
