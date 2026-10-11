//! KVキャッシュ。層ごとに [max_len, kv_dim] の連続バッファを保持する。

pub struct KvCache {
    n_layers: usize,
    kv_dim: usize,
    max_len: usize,
    len: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_len: usize) -> Self {
        Self {
            n_layers,
            kv_dim,
            max_len,
            len: 0,
            k: vec![vec![0.0; max_len * kv_dim]; n_layers],
            v: vec![vec![0.0; max_len * kv_dim]; n_layers],
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// 指定層の K/V バッファ(分割借用)
    pub fn kv_bufs(&mut self, layer: usize) -> (&mut [f32], &mut [f32]) {
        (&mut self.k[layer], &mut self.v[layer])
    }

    /// 指定層の K バッファ全体 (使用済みは先頭 len 行)
    pub fn k_buf(&mut self, layer: usize) -> &mut [f32] {
        &mut self.k[layer]
    }

    /// 指定層の V バッファ全体
    pub fn v_buf(&mut self, layer: usize) -> &mut [f32] {
        &mut self.v[layer]
    }

    /// n トークン書き終わったら進める
    pub fn advance(&mut self, n: usize) {
        assert!(
            self.len + n <= self.max_len,
            "KV cache overflow: {} + {} > {}",
            self.len,
            n,
            self.max_len
        );
        self.len += n;
    }
}
