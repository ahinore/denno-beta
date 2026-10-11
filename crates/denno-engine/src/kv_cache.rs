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

    /// バッファの最大トークン数 (コンテキスト長と一致する)
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// トークンあたりの KV メモリ (K + V、f32)
    pub fn bytes_per_token(&self) -> usize {
        self.k.len() * self.kv_dim * 2 * std::mem::size_of::<f32>()
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

    /// キャッシュを len トークンまで縮める (prefix cache 用)。
    /// 縮められた領域の行は stale になるが、attention は 0..len+n しか
    /// 参照しないため読まれることはない。次の forward で上書きされる。
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len, "KV cache truncate: {len} > {}", self.len);
        self.len = len;
    }

    /// 先頭 n 行 (トークン) のスナップショットを抽出する (マルチエントリ prefix cache 用)。
    /// 層ごとに [n * kv_dim] のベクタを返す。
    pub fn snapshot(&self, n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        assert!(n <= self.len, "KV cache snapshot: {n} > {}", self.len);
        let take = n * self.kv_dim;
        let k = self.k.iter().map(|buf| buf[..take].to_vec()).collect();
        let v = self.v.iter().map(|buf| buf[..take].to_vec()).collect();
        (k, v)
    }

    /// スナップショットの先頭 n 行を復元し、len を n に設定する。
    /// prefix cache のエントリから共通接頭辞分だけ KV を戻すのに使う。
    pub fn restore(&mut self, k: &[Vec<f32>], v: &[Vec<f32>], n: usize) {
        assert!(n <= self.max_len, "KV cache restore: {n} > {}", self.max_len);
        assert_eq!(k.len(), self.k.len(), "KV cache restore: layer count mismatch");
        assert_eq!(v.len(), self.v.len(), "KV cache restore: layer count mismatch");
        let take = n * self.kv_dim;
        for (dst, src) in self.k.iter_mut().zip(k.iter()) {
            assert!(src.len() >= take, "KV cache restore: snapshot too short");
            dst[..take].copy_from_slice(&src[..take]);
        }
        for (dst, src) in self.v.iter_mut().zip(v.iter()) {
            assert!(src.len() >= take, "KV cache restore: snapshot too short");
            dst[..take].copy_from_slice(&src[..take]);
        }
        self.len = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_is_empty() {
        let c = KvCache::new(2, 8, 32);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.kv_dim(), 8);
    }

    #[test]
    fn advance_increments_len() {
        let mut c = KvCache::new(1, 4, 16);
        c.advance(3);
        assert_eq!(c.len(), 3);
        c.advance(2);
        assert_eq!(c.len(), 5);
        assert!(!c.is_empty());
    }

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn advance_overflow_panics() {
        let mut c = KvCache::new(1, 4, 4);
        c.advance(3);
        c.advance(2); // 3 + 2 > 4
    }

    #[test]
    fn reset_clears_len() {
        let mut c = KvCache::new(1, 4, 16);
        c.advance(5);
        c.reset();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        // reset 後も書き込みできる
        c.advance(2);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn truncate_shrinks_len() {
        let mut c = KvCache::new(1, 4, 16);
        c.advance(10);
        c.truncate(4);
        assert_eq!(c.len(), 4);
        // 縮めた後も続きから書ける (prefix cache の再計算フロー)
        c.advance(3);
        assert_eq!(c.len(), 7);
        // 全体 truncate で空になる
        c.truncate(0);
        assert!(c.is_empty());
    }

    #[test]
    #[should_panic(expected = "truncate")]
    fn truncate_beyond_len_panics() {
        let mut c = KvCache::new(1, 4, 16);
        c.advance(5);
        c.truncate(6);
    }

    #[test]
    fn kv_bufs_are_per_layer_and_persist() {
        let mut c = KvCache::new(2, 4, 8);
        // レイヤ0に書き込み
        let (k, v) = c.kv_bufs(0);
        k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        v[0..4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        // レイヤ1には影響しない
        let (k1, v1) = c.kv_bufs(1);
        assert_eq!(k1[0..4], [0.0; 4]);
        assert_eq!(v1[0..4], [0.0; 4]);
        // レイヤ0の値は保持されている
        let (k0, v0) = c.kv_bufs(0);
        assert_eq!(k0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v0[0..4], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn k_buf_v_buf_have_full_capacity() {
        let layers = 3;
        let kv_dim = 4;
        let max_len = 10;
        let mut c = KvCache::new(layers, kv_dim, max_len);
        for l in 0..layers {
            assert_eq!(c.k_buf(l).len(), max_len * kv_dim);
            assert_eq!(c.v_buf(l).len(), max_len * kv_dim);
        }
    }

    #[test]
    fn max_len_reports_capacity() {
        let c = KvCache::new(2, 8, 32);
        assert_eq!(c.max_len(), 32);
    }

    #[test]
    fn snapshot_takes_first_n_rows() {
        let mut c = KvCache::new(2, 2, 8);
        let (k, v) = c.kv_bufs(0);
        k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]); // 行0,1
        v[0..4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        c.advance(3);
        let (sk, sv) = c.snapshot(2);
        assert_eq!(sk.len(), 2); // 層数
        assert_eq!(sk[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sv[0], vec![5.0, 6.0, 7.0, 8.0]);
        // 層1は未書き込み (先頭4要素のみ検証、長さは n*kv_dim)
        assert_eq!(sk[1].len(), 4);
        assert_eq!(sk[1], vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "snapshot")]
    fn snapshot_beyond_len_panics() {
        let c = KvCache::new(1, 4, 16);
        let _ = c.snapshot(1); // len = 0
    }

    #[test]
    fn restore_sets_len_and_copies_rows() {
        let mut src = KvCache::new(1, 2, 8);
        {
            let (k, _) = src.kv_bufs(0);
            k[0..4].copy_from_slice(&[9.0, 8.0, 7.0, 6.0]);
        }
        src.advance(2);
        let (sk, sv) = src.snapshot(2);

        let mut dst = KvCache::new(1, 2, 8);
        dst.advance(5); // 復元で len は上書きされる
        dst.restore(&sk, &sv, 2);
        assert_eq!(dst.len(), 2);
        let (k, _) = dst.kv_bufs(0);
        assert_eq!(k[0..4], [9.0, 8.0, 7.0, 6.0]);
        // 復元後も続きから書ける (prefix 再計算フロー)
        dst.advance(1);
        assert_eq!(dst.len(), 3);
    }

    #[test]
    fn restore_only_copies_requested_rows() {
        let mut src = KvCache::new(1, 2, 8);
        {
            let (k, _) = src.kv_bufs(0);
            k[0..8].copy_from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        }
        src.advance(4);
        let (sk, sv) = src.snapshot(4);

        let mut dst = KvCache::new(1, 2, 8);
        // 先頭 1 行だけ復元 (LCP が 1 のとき)
        dst.restore(&sk, &sv, 1);
        assert_eq!(dst.len(), 1);
        let (k, _) = dst.kv_bufs(0);
        assert_eq!(k[0..2], [1.0, 1.0]);
        assert_eq!(k[2..4], [0.0, 0.0]); // 2行目以降は触られない
    }

    #[test]
    #[should_panic(expected = "layer count mismatch")]
    fn restore_layer_mismatch_panics() {
        let mut dst = KvCache::new(2, 2, 8);
        let k = vec![vec![0.0; 2]; 1];
        let v = vec![vec![0.0; 2]; 1];
        dst.restore(&k, &v, 1);
    }
}
