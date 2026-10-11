//! サンプリング: greedy / temperature / top-k / top-p
//! seed を固定すれば同一設定で同一出力(TensorFold の exact decoding 思想)。

use rand::{rngs::StdRng, SeedableRng, Rng};

pub struct Sampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: Option<f32>, top_k: Option<usize>, top_p: Option<f32>, seed: Option<u64>) -> Self {
        Self {
            temperature: temperature.unwrap_or(1.0),
            top_k,
            top_p,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            },
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        // greedy
        if self.temperature <= 0.0 {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best;
        }

        // temperature 適用して降順ソート
        let mut v: Vec<(f32, usize)> = logits
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x / self.temperature, i))
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // top-k
        if let Some(k) = self.top_k {
            v.truncate(k.max(1));
        }

        // softmax
        let max = v[0].0;
        let mut probs: Vec<f32> = v.iter().map(|(x, _)| (x - max).exp()).collect();

        // top-p (nucleus)
        if let Some(p) = self.top_p {
            let total: f32 = probs.iter().sum();
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for i in 0..probs.len() {
                cum += probs[i] / total;
                if cum >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            v.truncate(cutoff);
        }

        let total: f32 = probs.iter().sum();
        let mut u = self.rng.gen::<f32>() * total;
        for (i, pr) in probs.iter().enumerate() {
            u -= pr;
            if u <= 0.0 {
                return v[i].1;
            }
        }
        v[v.len() - 1].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temp: Option<f32>, top_k: Option<usize>, top_p: Option<f32>, seed: Option<u64>) -> Sampler {
        Sampler::new(temp, top_k, top_p, seed)
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = sampler(Some(0.0), None, None, None);
        assert_eq!(s.sample(&[0.1, 3.0, 2.0]), 1);
        assert_eq!(s.sample(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(s.sample(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn greedy_tie_breaks_to_first_index() {
        // 実装は `v > best_v` の厳密比較なので同点は先頭が勝つ
        let mut s = sampler(Some(0.0), None, None, None);
        assert_eq!(s.sample(&[1.0, 2.0, 2.0]), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 0.5, 3.0],
            vec![3.0, 1.0, 2.0, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let mut a = sampler(Some(0.8), None, None, Some(42));
        let mut b = sampler(Some(0.8), None, None, Some(42));
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..30 {
            for lg in &logits {
                seq_a.push(a.sample(lg));
                seq_b.push(b.sample(lg));
            }
        }
        assert_eq!(seq_a, seq_b, "同一seed+同一logits列なら同一サンプル列");
    }

    #[test]
    fn top_k_one_always_picks_argmax() {
        let mut s = sampler(Some(1.5), Some(1), None, Some(7));
        for _ in 0..50 {
            assert_eq!(s.sample(&[0.1, 4.0, 0.2]), 1);
        }
    }

    #[test]
    fn top_k_respects_k_choices() {
        // k=2 なら argmax(1) と準高峰(3) の2択だけ
        let mut s = sampler(Some(1.0), Some(2), None, Some(7));
        for _ in 0..100 {
            let i = s.sample(&[0.1, 4.0, 0.0, 3.0]);
            assert!(i == 1 || i == 3, "top_k=2 で id {i} が選ばれた");
        }
    }

    #[test]
    fn top_p_dominant_logit_collapses_to_argmax() {
        // logit 差が大きいと p=0.5 でも top-1 だけで累積 0.5 を超える
        let mut s = sampler(Some(1.0), None, Some(0.5), Some(7));
        for _ in 0..50 {
            assert_eq!(s.sample(&[10.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn unseeded_temp_one_can_pick_non_argmax() {
        // サニティ: 確率的サンプリングが argmax 固定になっていないことの逆方向確認
        let mut s = sampler(Some(1.0), None, None, Some(123));
        let picked: std::collections::HashSet<usize> =
            (0..100).map(|_| s.sample(&[2.0, 1.5])).collect();
        assert!(picked.contains(&0));
        assert!(picked.contains(&1), "temp=1.0 で非argmaxが一度も選ばれない");
    }
}
