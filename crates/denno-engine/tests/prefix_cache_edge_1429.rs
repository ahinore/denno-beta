//! PrefixCache の LRU 順序・容量境界・重複キー置換・空/接頭辞一致・
//! ゼロ容量周辺の境界テスト (task 1429)。
//!
//! `src/prefix_cache.rs` の単体テスト (P917) を補完し、公開 API のみで
//! 観測できる振る舞いを固定する。CPU-only・決定的・高速。
//!
//! `last_used` は非公開なので、LRU 順序は eviction 後に「どのエントリが
//! 残っているか」(`contains_prompt` / `best_match` / `total_weight`) で観測する。

use denno_engine::kv_cache::KvCache;
use denno_engine::prefix_cache::PrefixCache;

/// 1 層 × kv_dim 2 の KvCache。bytes_per_token = 1*2*2*4 = 16。
fn cache_with(len: usize) -> KvCache {
    let mut c = KvCache::new(1, 2, 64);
    let _ = c.kv_bufs(0, len);
    c.advance(len);
    c
}

/// cache_with の KvCache に合わせた bytes_per_token = 16 の PrefixCache。
fn pc(max_tokens: usize) -> PrefixCache {
    PrefixCache::new(max_tokens, 16)
}

/// 値の入った KV を持つ KvCache (復元内容の検証用)。
fn filled_cache(len: usize, value: f32) -> KvCache {
    let mut c = KvCache::new(1, 2, 64);
    {
        let (k, v) = c.kv_bufs(0, len);
        for x in k.iter_mut().chain(v.iter_mut()) {
            *x = value;
        }
    }
    c.advance(len);
    c
}

// ---------------------------------------------------------------- 空ストア

#[test]
fn empty_store_has_no_entries_and_no_match() {
    let store = pc(16);
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
    assert_eq!(store.total_weight(), 0);
    assert!(store.best_match(&[1, 2]).is_none());
    assert!(store.best_full_match(&[1, 2]).is_none());
    assert!(!store.contains_prompt(&[1, 2]));

    // 空ストアへの復元要求は false で、キャッシュも無変更
    let mut dst = KvCache::new(1, 2, 64);
    assert!(!store.restore_into(0, 1, &mut dst));
    assert_eq!(dst.len(), 0);
}

#[test]
fn empty_or_zero_length_insert_is_ignored() {
    let mut store = pc(16);

    // 空 prompt は格納しない
    store.insert(Vec::new(), &cache_with(4), 4);
    assert!(store.is_empty());

    // kv_len = 0 は格納しない (prompt があっても)
    store.insert(vec![1, 2], &cache_with(4), 0);
    assert!(store.is_empty());

    // insert_snapshot 直呼びでも同じ
    store.insert_snapshot(Vec::new(), vec![vec![]], vec![vec![]], 3, None);
    assert!(store.is_empty());
    store.insert_snapshot(vec![1], vec![vec![]], vec![vec![]], 0, None);
    assert!(store.is_empty());
    assert_eq!(store.total_weight(), 0);
}

// ------------------------------------------------------------ 容量 1 相当

#[test]
fn capacity_of_one_entry_evicts_previous() {
    // 重み 4 のエントリ 1 つだけが収まる容量
    let mut store = pc(4);
    store.insert(vec![1], &cache_with(4), 4);
    assert_eq!(store.len(), 1);
    assert_eq!(store.total_weight(), 4);

    // 2 つ目は同時に保持できず、古い方が落ちる
    store.insert(vec![2], &cache_with(4), 4);
    assert_eq!(store.len(), 1, "容量 1 エントリ分では常に 1 件");
    assert_eq!(store.total_weight(), 4);
    assert!(store.contains_prompt(&[2]));
    assert!(!store.contains_prompt(&[1]), "LRU で旧エントリが退避される");

    // 3 つ目も同様に直前のものだけが残る
    store.insert(vec![3], &cache_with(4), 4);
    assert_eq!(store.len(), 1);
    assert!(store.contains_prompt(&[3]));
    assert!(!store.contains_prompt(&[2]));
}

#[test]
fn entry_exactly_at_limit_is_accepted_but_one_over_is_rejected() {
    // 上限ちょうど (4) は格納
    let mut store = pc(4);
    store.insert(vec![1, 2, 3, 4], &cache_with(4), 4);
    assert_eq!(store.len(), 1);
    assert_eq!(store.total_weight(), 4);

    // 1 トークンでも超えるエントリは単体で拒否 (ストアは無変更)
    let mut store2 = pc(4);
    store2.insert(vec![1, 2, 3, 4, 5], &cache_with(5), 5);
    assert!(store2.is_empty());
    assert_eq!(store2.total_weight(), 0);
}

// -------------------------------------------------------------- LRU touch

#[test]
fn touch_protects_entry_from_eviction() {
    // 各重み 3、上限 9 → 3 件ちょうど
    let mut store = pc(9);
    store.insert(vec![1], &cache_with(3), 3); // A (最古)
    store.insert(vec![2], &cache_with(3), 3); // B
    store.insert(vec![3], &cache_with(3), 3); // C (最新)

    // A を touch して最新化 → 最古は B になる
    store.touch(0);

    store.insert(vec![4], &cache_with(3), 3); // 計 12 > 9 → B が退避
    assert_eq!(store.len(), 3);
    assert_eq!(store.total_weight(), 9);
    assert!(store.contains_prompt(&[1]), "touch 済みの A は残る");
    assert!(!store.contains_prompt(&[2]), "最古の B が退避される");
    assert!(store.contains_prompt(&[3]));
    assert!(store.contains_prompt(&[4]));
}

#[test]
fn touch_order_decides_consecutive_evictions() {
    // 上限 6、重み 3 → 常に 2 件。touch の順序だけでどちらが落ちるかが決まる
    let mut store = pc(6);
    store.insert(vec![1], &cache_with(3), 3); // A
    store.insert(vec![2], &cache_with(3), 3); // B
    store.touch(0); // A を最新化

    store.insert(vec![3], &cache_with(3), 3); // 計 9 → B 退避
    assert!(!store.contains_prompt(&[2]) && store.contains_prompt(&[1]));

    // ここで B の位置に居るのは A(index 0) / C(index 1)。C を最新化
    assert!(store.contains_prompt(&[1]));
    let c_idx = store.best_match(&[3]).unwrap().0;
    store.touch(c_idx); // C を最新化 → 最古は A
    store.insert(vec![4], &cache_with(3), 3); // A 退避
    assert!(store.contains_prompt(&[3]));
    assert!(store.contains_prompt(&[4]));
    assert!(!store.contains_prompt(&[1]));
}

#[test]
fn touch_out_of_range_is_a_noop_for_contents() {
    let mut store = pc(9);
    store.insert(vec![1], &cache_with(3), 3);
    store.insert(vec![2], &cache_with(3), 3);

    // 範囲外 index の touch は panic せず、件数・重み・内容を変えない
    store.touch(99);
    assert_eq!(store.len(), 2);
    assert_eq!(store.total_weight(), 6);
    assert!(store.contains_prompt(&[1]));
    assert!(store.contains_prompt(&[2]));

    // 不正 touch 後も通常の LRU 退避は動く
    store.insert(vec![3], &cache_with(3), 3); // 計 9 = 上限、退避なし
    assert_eq!(store.len(), 3);
    store.insert(vec![4], &cache_with(3), 3); // 計 12 > 9 → 最古 (A) 退避
    assert!(!store.contains_prompt(&[1]));
}

// --------------------------------------------------- 重複キー (同一 prompt)

#[test]
fn duplicate_key_overwrites_payload_and_weight() {
    let mut store = pc(16);
    store.insert(vec![1, 2], &filled_cache(2, 1.0), 2);
    assert_eq!(store.len(), 1);
    assert_eq!(store.total_weight(), 2);

    // 同一 prompt を長い kv_len で再挿入 → 件数は増えず重みが更新される
    store.insert_snapshot(
        vec![1, 2],
        filled_cache(5, 7.0).snapshot(5).0,
        filled_cache(5, 7.0).snapshot(5).1,
        5,
        None,
    );
    assert_eq!(store.len(), 1, "同一 prompt は置き換えで件数不変");
    assert_eq!(store.total_weight(), 5, "kv_len の更新が重みに反映される");

    // 復元内容も新しいスナップショットのもの
    let mut dst = KvCache::new(1, 2, 64);
    assert!(store.restore_into(0, 2, &mut dst));
    assert_eq!(dst.len(), 2);
    let (k, _) = dst.kv_bufs_ref(0);
    assert_eq!(k[0..4], [7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn duplicate_key_touch_moves_it_out_of_the_eviction_victim_slot() {
    // A(4) → B(4) の順で挿入し、A を再挿入 (= touch 相当) して最新化する
    let mut store = pc(10);
    store.insert(vec![1], &cache_with(4), 4); // A 古い
    store.insert(vec![2], &cache_with(4), 4); // B 新しい

    // 同一 prompt A の再挿入: 件数は 2 のまま、A が最新になる
    store.insert(vec![1], &cache_with(4), 4);
    assert_eq!(store.len(), 2);

    store.insert(vec![3], &cache_with(4), 4); // 計 12 > 10 → 最古は B
    assert_eq!(store.len(), 2);
    assert!(store.contains_prompt(&[1]), "再挿入で最新化された A は残る");
    assert!(!store.contains_prompt(&[2]), "B が退避される");
    assert!(store.contains_prompt(&[3]));
}

#[test]
fn duplicate_key_with_smaller_payload_frees_weight() {
    // 置き換えで重みが減り、その分だけ他エントリを保持できる
    let mut store = pc(8);
    store.insert(vec![1], &cache_with(5), 5); // A: 5
    store.insert(vec![2], &cache_with(3), 3); // B: 3 (計 8)
    assert_eq!(store.len(), 2);

    store.insert(vec![1], &cache_with(2), 2); // A を 2 に縮小 → 計 5
    assert_eq!(store.total_weight(), 5);
    store.insert(vec![3], &cache_with(3), 3); // 計 8、退避なし
    assert_eq!(store.len(), 3);
    assert!(store.contains_prompt(&[1]));
    assert!(store.contains_prompt(&[2]));
    assert!(store.contains_prompt(&[3]));
}

// -------------------------------------------------- 空/接頭辞一致の境界

#[test]
fn best_match_boundaries_on_shared_prefix() {
    let mut store = pc(16);
    store.insert(vec![1, 2, 3], &cache_with(3), 3);

    // 完全一致: lcp == prompt 長
    let (i, lcp) = store.best_match(&[1, 2, 3]).unwrap();
    assert_eq!((i, lcp), (0, 3));

    // クエリが長い: エントリ長で頭打ち
    let (i, lcp) = store.best_match(&[1, 2, 3, 4, 5]).unwrap();
    assert_eq!((i, lcp), (0, 3));

    // クエリが短い接頭辞: クエリ長で頭打ち
    assert_eq!(store.best_match(&[1, 2]).unwrap(), (0, 2));
    assert_eq!(store.best_match(&[1]).unwrap(), (0, 1));

    // 最後で食い違う: 1 トークン手前まで
    assert_eq!(store.best_match(&[1, 2, 9]).unwrap(), (0, 2));

    // 先頭で食い違う: LCP 0 は候補にしない
    assert!(store.best_match(&[9, 1, 2]).is_none());
    assert!(store.best_match(&[]).is_none(), "空クエリは一致なし");
}

#[test]
fn best_full_match_boundaries_on_shared_prefix() {
    let mut store = pc(16);
    store.insert(vec![1, 2, 3], &cache_with(3), 3);

    // エントリ prompt を完全に含むクエリのみ一致、lcp はエントリ長
    assert_eq!(store.best_full_match(&[1, 2, 3]).unwrap(), (0, 3));
    assert_eq!(store.best_full_match(&[1, 2, 3, 4]).unwrap(), (0, 3));

    // クエリが短い (エントリを完全に含まない) と不一致
    assert!(store.best_full_match(&[1, 2]).is_none());
    assert!(store.best_full_match(&[1, 2, 9]).is_none());
    assert!(store.best_full_match(&[]).is_none());

    // 空ストアでは何も一致しない
    let empty = pc(16);
    assert!(empty.best_full_match(&[1, 2, 3, 4]).is_none());
}

#[test]
fn restore_into_rejects_boundary_n_values() {
    // prompt 長 3 / kv_len 5 のエントリ
    let mut store = pc(16);
    store.insert_snapshot(
        vec![1, 2, 3],
        vec![vec![1.0; 10]],
        vec![vec![1.0; 10]],
        5,
        None,
    );

    let mut dst = KvCache::new(1, 2, 64);
    assert!(!store.restore_into(0, 0, &mut dst), "n = 0 は拒否");
    assert!(!store.restore_into(0, 4, &mut dst), "n > prompt 長は拒否");
    assert!(!store.restore_into(5, 3, &mut dst), "存在しない index は拒否");
    assert_eq!(dst.len(), 0, "拒否時にキャッシュは無変更");

    // n == prompt 長は許可 (kv_len 未満)
    assert!(store.restore_into(0, 3, &mut dst));
    assert_eq!(dst.len(), 3);
}

// ---------------------------------------------------------- ゼロ容量

#[test]
fn zero_capacity_stores_nothing() {
    let mut store = pc(0);
    assert_eq!(store.max_tokens, 0);

    store.insert(vec![1], &cache_with(1), 1);
    store.insert(vec![1, 2], &cache_with(2), 2);
    assert!(store.is_empty(), "容量 0 では如何なるエントリも保持しない");
    assert_eq!(store.total_weight(), 0);
    assert!(store.best_match(&[1]).is_none());
    assert!(!store.contains_prompt(&[1]));

    // 状態付き (GDN) でも同様に拒否される
    let mut c = KvCache::new(1, 2, 64);
    c.attach_gdn_slot(0, 2, 4);
    c.advance(4);
    let (k, v) = c.snapshot(4);
    store.insert_snapshot(vec![1], k, v, 4, Some(c.gdn_snapshot()));
    assert!(store.is_empty());
}

#[test]
fn set_max_tokens_zero_evicts_everything() {
    let mut store = pc(16);
    store.insert(vec![1], &cache_with(4), 4);
    store.insert(vec![2], &cache_with(4), 4);
    assert_eq!(store.len(), 2);

    store.set_max_tokens(0);
    assert_eq!(store.max_tokens, 0);
    assert!(store.is_empty(), "上限 0 への引き下げで全エントリ退避");
    assert_eq!(store.total_weight(), 0);
    assert!(store.best_match(&[1]).is_none());

    // 0 から引き上げれば再び格納できる
    store.set_max_tokens(4);
    store.insert(vec![3], &cache_with(4), 4);
    assert_eq!(store.len(), 1);
    assert!(store.contains_prompt(&[3]));
}

#[test]
fn set_max_tokens_shrinks_in_lru_order() {
    let mut store = pc(9);
    store.insert(vec![1], &cache_with(3), 3); // A 最古
    store.insert(vec![2], &cache_with(3), 3); // B
    store.insert(vec![3], &cache_with(3), 3); // C 最新
    store.touch(0); // A を最新化 → 最古は B

    store.set_max_tokens(6); // 9 → 6: 1 件退避、最古の B
    assert_eq!(store.len(), 2);
    assert_eq!(store.total_weight(), 6);
    assert!(store.contains_prompt(&[1]));
    assert!(!store.contains_prompt(&[2]), "touch 済みでない最古が退避");
    assert!(store.contains_prompt(&[3]));
}

// ------------------------------------------- bytes_per_token 0 のガード

#[test]
fn zero_bytes_per_token_is_clamped_not_divided_by_zero() {
    // bpt = 0 は内部で 1 にクランプされ、0 除算 panic を起こさない
    let mut store = PrefixCache::new(64, 0);
    store.insert(vec![1], &cache_with(4), 4);
    assert_eq!(store.len(), 1);
    // KV 4 トークン + GDN 状態なし → 重み 4
    assert_eq!(store.total_weight(), 4);
    assert!(store.restore_into(0, 1, &mut KvCache::new(1, 2, 64)));

    // GDN 状態 (conv2 + ssm4 = 6 f32 = 24B) は bpt 1 換算で 24 トークン分
    let mut c = KvCache::new(1, 2, 64);
    c.attach_gdn_slot(0, 2, 4);
    c.advance(4);
    let (k, v) = c.snapshot(4);
    store.insert_snapshot(vec![2], k, v, 4, Some(c.gdn_snapshot()));
    assert_eq!(store.total_weight(), 4 + 4 + 24, "既存 4 + 新規 KV 4 + 状態 24");
}
