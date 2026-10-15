//! KVキャッシュ。層ごとに連続バッファを保持する (P17e: 需要ベース確保)。
//!
//! 旧実装は `[max_len, kv_dim]` をスロット作成時に全量確保していた
//! (1 スロット = ctx 全幅 × bytes_per_token。Qwen2.5-1.5B で ~1.9GB)。
//! P17e では**初期確保 0** で、書き込み量に応じて全層のバッファを
//! 倍々で成長させる (上限 max_len)。メモリレイアウトは「先頭 len 行が
//! 使用済みの連続バッファ」のままだので、attention を含む数値経路は
//! 旧実装と完全に同一 (ビット一致)。成長時の realloc コピーは既に
//! 書き済みの行を保存する (倍々なので償却 O(1))。
//!
//! A3 (Qwen3.8-27B スケール): GDN 状態スロットを持つ層 (Qwen3.5 の
//! linear_attention 層) は KV を書かないため、バッファ確保・
//! bytes_per_token・snapshot 対象から除外する。27B (64 層 = 48 GDN +
//! 16 full) では KV 実量が 1/4 になる (512KB→128KB B/token)。GDN スロット
//! の無いモデル (llama / gpt2) は従来動作のまま。

pub struct KvCache {
    n_layers: usize,
    kv_dim: usize,
    max_len: usize,
    len: usize,
    /// 確保済みトークン数 (全層共通・倍々成長)
    cap: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// GPU KV ミラー (P17g) のキー。プロセス内で一意 (Clone しないので衝突しない)
    id: u64,
    /// GDN (Qwen3.5 linear_attention) 層の状態スロット。layer index → (conv, ssm)。
    /// attach_gdn_slot で作られ、reset でゼロに戻る。通常の full attention 層は None。
    gdn: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

/// KvCache インスタンス ID 採番 (P17g: GPU KV ミラーのキー)
static KV_ID_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_len: usize) -> Self {
        Self {
            n_layers,
            kv_dim,
            max_len,
            len: 0,
            cap: 0,
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
            id: KV_ID_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            gdn: vec![None; n_layers],
        }
    }

    /// GDN 層の状態スロットを用意する (Qwen3.5 用)。conv_len = conv_dim*(d_conv-1)、
    /// ssm_len = n_v_heads*head_k_dim*head_v_dim。
    pub fn attach_gdn_slot(&mut self, layer: usize, conv_len: usize, ssm_len: usize) {
        assert!(layer < self.n_layers);
        self.gdn[layer] = Some((vec![0.0; conv_len], vec![0.0; ssm_len]));
    }

    /// GDN 層の状態 (conv バッファ, SSM 行列) への可変参照。
    pub fn gdn_state_mut(&mut self, layer: usize) -> Option<(&mut [f32], &mut [f32])> {
        match &mut self.gdn[layer] {
            Some((c, s)) => Some((c.as_mut_slice(), s.as_mut_slice())),
            None => None,
        }
    }

    /// GPU KV ミラーのキー (P17g)
    pub fn kv_id(&self) -> u64 {
        self.id
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for g in self.gdn.iter_mut().flatten() {
            g.0.fill(0.0);
            g.1.fill(0.0);
        }
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

    /// 層数 (P14: ビューアの層パネル用)。
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// バッファの最大トークン数 (コンテキスト長と一致する)
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// 現在確保済みのトークン数 (P17e: フットプリント計算用)
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// トークンあたりの KV メモリ (K + V、f32)。静的な値
    /// (レイアウトは先頭 len 行使用の連続バッファで不変)。
    /// GDN スロット層は KV を使わないため対象外 (A3)。
    pub fn bytes_per_token(&self) -> usize {
        let kv_layers = self
            .k
            .iter()
            .zip(self.gdn.iter())
            .filter(|(_, g)| g.is_none())
            .count();
        kv_layers * self.kv_dim * 2 * std::mem::size_of::<f32>()
    }

    /// 全層のバッファを need トークン分まで成長させる (倍々・上限 max_len)。
    /// 既に書き込んだ行は保存される。
    fn ensure(&mut self, need: usize) {
        assert!(
            need <= self.max_len,
            "KV cache overflow: {need} > {}",
            self.max_len
        );
        if need <= self.cap {
            return;
        }
        let new_cap = self.max_len.min(need.max(self.cap.saturating_mul(2)));
        let elems = new_cap * self.kv_dim;
        // GDN スロット層は KV を書かないので確保しない (A3)。空 Vec のまま →
        // kv_bufs/kv_bufs_ref が GDN 層で呼ばれないのは呼び出し側契約
        // (qwen35.rs は full_attention 層にのみ書く)。
        for (buf, g) in self.k.iter_mut().zip(self.gdn.iter()) {
            if g.is_some() {
                continue;
            }
            buf.resize(elems, 0.0);
        }
        for (buf, g) in self.v.iter_mut().zip(self.gdn.iter()) {
            if g.is_some() {
                continue;
            }
            buf.resize(elems, 0.0);
        }
        self.cap = new_cap;
    }

    /// 層 `layer` の K/V バッファ (分割借用)。続けて n トークン分を
    /// `len` 行目から書き込めるよう、事前に全層の容量を確保する (P17e)。
    pub fn kv_bufs(&mut self, layer: usize, n: usize) -> (&mut [f32], &mut [f32]) {
        self.ensure(self.len + n);
        (&mut self.k[layer], &mut self.v[layer])
    }

    /// 指定層の K/V バッファ (読み取り専用)。attention など並列タスクから
    /// 各セッションのキャッシュを共有読み込みする際に使う (P10)。
    pub fn kv_bufs_ref(&self, layer: usize) -> (&[f32], &[f32]) {
        (&self.k[layer], &self.v[layer])
    }

    /// 指定層の K バッファ全体 (使用済みは先頭 len 行)。n トークン追記分を確保。
    pub fn k_buf(&mut self, layer: usize, n: usize) -> &mut [f32] {
        self.ensure(self.len + n);
        &mut self.k[layer]
    }

    /// 指定層の V バッファ全体。n トークン追記分を確保。
    pub fn v_buf(&mut self, layer: usize, n: usize) -> &mut [f32] {
        self.ensure(self.len + n);
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
        // GDN スロット層は空ベクタ (KV 非保持)。restore 側も空なら skip する。
        let k = self
            .k
            .iter()
            .zip(self.gdn.iter())
            .map(|(buf, g)| {
                if g.is_some() {
                    Vec::new()
                } else {
                    buf[..take].to_vec()
                }
            })
            .collect();
        let v = self
            .v
            .iter()
            .zip(self.gdn.iter())
            .map(|(buf, g)| {
                if g.is_some() {
                    Vec::new()
                } else {
                    buf[..take].to_vec()
                }
            })
            .collect();
        (k, v)
    }

    /// スナップショットの先頭 n 行を復元し、len を n に設定する。
    /// prefix cache のエントリから共通接頭辞分だけ KV を戻すのに使う。
    /// 復元分の容量が無ければ成長させる (P17e)。
    pub fn restore(&mut self, k: &[Vec<f32>], v: &[Vec<f32>], n: usize) {
        assert!(n <= self.max_len, "KV cache restore: {n} > {}", self.max_len);
        assert_eq!(k.len(), self.k.len(), "KV cache restore: layer count mismatch");
        assert_eq!(v.len(), self.v.len(), "KV cache restore: layer count mismatch");
        self.ensure(n);
        let take = n * self.kv_dim;
        for (i, (dst, src)) in self.k.iter_mut().zip(k.iter()).enumerate() {
            if src.is_empty() {
                assert!(
                    self.gdn[i].is_some(),
                    "KV cache restore: 空スナップショット層は GDN 層のみ (layer {i})"
                );
                continue;
            }
            assert!(src.len() >= take, "KV cache restore: snapshot too short");
            dst[..take].copy_from_slice(&src[..take]);
        }
        for (i, (dst, src)) in self.v.iter_mut().zip(v.iter()).enumerate() {
            if src.is_empty() {
                assert!(
                    self.gdn[i].is_some(),
                    "KV cache restore: 空スナップショット層は GDN 層のみ (layer {i})"
                );
                continue;
            }
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
        // P17e: 初期確保は 0
        assert_eq!(c.capacity(), 0);
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
        // レイヤ0に書き込み (kv_bufs が成長を担う)
        let (k, v) = c.kv_bufs(0, 1);
        k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        v[0..4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        // レイヤ1には影響しない
        let (k1, v1) = c.kv_bufs(1, 0);
        assert_eq!(k1[0..4], [0.0; 4]);
        assert_eq!(v1[0..4], [0.0; 4]);
        // レイヤ0の値は保持されている
        let (k0, v0) = c.kv_bufs(0, 0);
        assert_eq!(k0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v0[0..4], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn kv_bufs_grows_on_demand_and_preserves_rows() {
        let mut c = KvCache::new(2, 2, 64);
        // 1 トークン書き込み → cap は最低 1
        {
            let (k, _) = c.kv_bufs(0, 1);
            k[0..2].copy_from_slice(&[7.0, 7.0]);
        }
        c.advance(1);
        assert_eq!(c.capacity(), 1);
        // 追記で倍々成長・既存行は保存される
        {
            let (k, _) = c.kv_bufs(0, 1);
            k[2..4].copy_from_slice(&[8.0, 8.0]);
        }
        c.advance(1);
        assert!(c.capacity() >= 2);
        let (k, _) = c.kv_bufs(0, 0);
        assert_eq!(k[0..4], [7.0, 7.0, 8.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn kv_bufs_overflow_panics() {
        let mut c = KvCache::new(1, 4, 4);
        let _ = c.kv_bufs(0, 5);
    }

    #[test]
    fn k_buf_v_buf_grow_and_hold_capacity() {
        let layers = 3;
        let kv_dim = 4;
        let mut c = KvCache::new(layers, kv_dim, 10);
        for l in 0..layers {
            let kb = c.k_buf(l, 10);
            assert_eq!(kb.len(), 10 * kv_dim);
            let vb = c.v_buf(l, 10);
            assert_eq!(vb.len(), 10 * kv_dim);
        }
        assert_eq!(c.capacity(), 10);
    }

    #[test]
    fn max_len_reports_capacity() {
        let c = KvCache::new(2, 8, 32);
        assert_eq!(c.max_len(), 32);
    }

    #[test]
    fn bytes_per_token_is_static() {
        // P17e: 確保状態に依らず n_layers * kv_dim * 2 * 4
        let mut c = KvCache::new(3, 2, 16);
        assert_eq!(c.bytes_per_token(), 3 * 2 * 2 * 4);
        let _ = c.kv_bufs(0, 4);
        c.advance(4);
        assert_eq!(c.bytes_per_token(), 3 * 2 * 2 * 4);
    }

    #[test]
    fn snapshot_takes_first_n_rows() {
        let mut c = KvCache::new(2, 2, 8);
        {
            let (k, v) = c.kv_bufs(0, 3);
            k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]); // 行0,1
            v[0..4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        }
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
            let (k, _) = src.kv_bufs(0, 2);
            k[0..4].copy_from_slice(&[9.0, 8.0, 7.0, 6.0]);
        }
        src.advance(2);
        let (sk, sv) = src.snapshot(2);

        let mut dst = KvCache::new(1, 2, 8);
        dst.advance(5); // 復元で len は上書きされる
        dst.restore(&sk, &sv, 2);
        assert_eq!(dst.len(), 2);
        let (k, _) = dst.kv_bufs(0, 0);
        assert_eq!(k[0..4], [9.0, 8.0, 7.0, 6.0]);
        // 復元後も続きから書ける (prefix 再計算フロー)
        dst.advance(1);
        assert_eq!(dst.len(), 3);
    }

    #[test]
    fn restore_grows_capacity_when_needed() {
        // P17e: 復元先が空 (未確保) でも成長して復元できる
        let mut src = KvCache::new(1, 2, 8);
        {
            let (k, _) = src.kv_bufs(0, 2);
            k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        src.advance(2);
        let (sk, sv) = src.snapshot(2);

        let mut dst = KvCache::new(1, 2, 8);
        assert_eq!(dst.capacity(), 0);
        dst.restore(&sk, &sv, 2);
        assert_eq!(dst.capacity(), 2);
        assert_eq!(dst.len(), 2);
        let (k, _) = dst.kv_bufs(0, 0);
        assert_eq!(k[0..4], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn restore_only_copies_requested_rows() {
        let mut src = KvCache::new(1, 2, 8);
        {
            let (k, _) = src.kv_bufs(0, 4);
            k[0..8].copy_from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        }
        src.advance(4);
        let (sk, sv) = src.snapshot(4);

        let mut dst = KvCache::new(1, 2, 8);
        // 先頭 1 行だけ復元 (LCP が 1 のとき)
        dst.restore(&sk, &sv, 1);
        assert_eq!(dst.len(), 1);
        let (k, _) = dst.kv_bufs(0, 1); // 2 行目まで成長させて stale 行を確認
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

    /// A3: GDN スロット層は KV バッファ確保・bytes_per_token・snapshot/restore
    /// の対象外になる (Qwen3.5 の 48 GDN + 16 full で KV 実量 1/4)。
    #[test]
    fn gdn_slot_layers_are_excluded_from_kv() {
        let mut c = KvCache::new(4, 2, 16);
        c.attach_gdn_slot(0, 3, 5);
        c.attach_gdn_slot(2, 3, 5);
        // bytes_per_token: KV 層 2 のみ (GDN 層 2 つは除外)
        assert_eq!(c.bytes_per_token(), 2 * 2 * 2 * 4);
        // 書き込みは GDN でない層経由 (kv_bufs(0,..) は成長トリガとしてだけ使う)
        {
            let (k, _) = c.kv_bufs(1, 2);
            k[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        c.advance(2);
        assert_eq!(c.capacity(), 2);
        // GDN 層のバッファは空のまま (未確保)
        assert!(c.kv_bufs_ref(0).0.is_empty());
        assert!(c.kv_bufs_ref(2).1.is_empty());
        // KV 層は確保されている
        assert_eq!(c.kv_bufs_ref(1).0.len(), 2 * 2);
        // snapshot: GDN 層は空・KV 層は実データ
        let (sk, sv) = c.snapshot(2);
        assert!(sk[0].is_empty() && sk[2].is_empty());
        assert!(sv[0].is_empty() && sv[2].is_empty());
        assert_eq!(sk[1], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sk[3].len(), 4);
        // restore: GDN 層は skip・KV 層は復元される
        let mut d = KvCache::new(4, 2, 16);
        d.attach_gdn_slot(0, 3, 5);
        d.attach_gdn_slot(2, 3, 5);
        d.restore(&sk, &sv, 2);
        assert_eq!(d.len(), 2);
        assert_eq!(d.kv_bufs_ref(1).0[0..4], [1.0, 2.0, 3.0, 4.0]);
        assert!(d.kv_bufs_ref(0).0.is_empty());
        // GDN スロットの無いキャッシュは従来動作 (全層が対象)
        let mut plain = KvCache::new(2, 2, 8);
        assert_eq!(plain.bytes_per_token(), 2 * 2 * 2 * 4);
    }
}
