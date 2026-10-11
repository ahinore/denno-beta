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
