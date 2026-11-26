//! task 1359: kv_snapshot / kv_cache の公開 API 境界・破損入力テスト。
//!
//! 対象は公開 API のみ (`encode` / `save` / `load` / 定数 / `KvCache`)。
//! 合成した小さな次元で、往復のビット一致・切り詰め・magic/version 不一致・
//! checksum/長さ不一致・ゼロトークン・容量上限を検証する (CPU-only, 高速)。

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use denno_engine::kv_cache::KvCache;
use denno_engine::kv_snapshot::{
    encode, load, prompt_hash, save, SnapshotError, HEADER_LEN, MAGIC, VERSION,
};
use denno_engine::model::exp_stats::{FNV_OFFSET, FNV_PRIME};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// テスト用の一時ディレクトリ (外部クレートを足さない)。
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "denno_kv_edge_1359_{tag}_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("一時ディレクトリを作れない");
        TempDir(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_bytes(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, bytes).expect("テスト用バイト列を書けない");
    p
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn checksum(bytes: &[u8]) -> u64 {
    fnv(FNV_OFFSET, bytes)
}

fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn get_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// 3 層・kv_dim 2。層 1 だけ GDN スロットを持つ小さなキャッシュ。
fn tiny_cache() -> KvCache {
    let mut c = KvCache::new(3, 2, 8);
    c.attach_gdn_slot(1, 2, 3);
    c
}

/// ビット列をそのまま f32 として入れる (NaN / -0.0 / 非正規化も往復対象)。
fn fill_bits(c: &mut KvCache, n: usize) {
    let dim = c.kv_dim();
    for layer in 0..c.n_layers() {
        if c.has_gdn_slot(layer) {
            continue;
        }
        let (k, v) = c.kv_bufs(layer, n);
        for i in 0..n * dim {
            let base = 0x3f80_0000u32
                .wrapping_add((layer as u32) << 12)
                .wrapping_add((i as u32).wrapping_mul(0x1f));
            k[i] = f32::from_bits(base);
            let vb = match i % 5 {
                0 => 0x8000_0000,        // -0.0
                1 => 0x7fc0_0001,        // NaN (payload を保つ)
                2 => 0x0000_0001,        // 非正規化数
                3 => 0xbf80_0000,        // -1.0
                _ => 0x3f00_0000u32.wrapping_add((i as u32) << 3),
            };
            v[i] = f32::from_bits(vb);
        }
    }
    c.advance(n);
    if let Some((conv, ssm)) = c.gdn_state_mut(1) {
        conv.copy_from_slice(&[f32::NAN, -0.0]);
        ssm[0] = -1.5;
        ssm[1] = 3.25;
        ssm[2] = f32::from_bits(0x7fc0_1234);
    }
}

fn cache_bits(c: &KvCache, n: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let (k, v) = c.snapshot(n);
    (
        k.iter()
            .map(|l| l.iter().map(|x| x.to_bits()).collect())
            .collect(),
        v.iter()
            .map(|l| l.iter().map(|x| x.to_bits()).collect())
            .collect(),
    )
}

fn gdn_bits(c: &KvCache) -> Vec<Option<(Vec<u32>, Vec<u32>)>> {
    c.gdn_snapshot()
        .iter()
        .map(|o| {
            o.as_ref().map(|(a, b)| {
                (
                    a.iter().map(|x| x.to_bits()).collect(),
                    b.iter().map(|x| x.to_bits()).collect(),
                )
            })
        })
        .collect()
}

#[test]
fn roundtrip_preserves_exact_bits_and_metadata() {
    let dir = TempDir::new("roundtrip");
    let prompt: Vec<u32> = vec![7, 11, 13, 17];
    let fp = 0x0123_4567_89ab_cdefu64;

    let mut src = tiny_cache();
    fill_bits(&mut src, prompt.len());

    let bytes = encode(&src, prompt.len(), fp, &prompt).expect("encode が失敗した");
    // ヘッダのメタデータが正しい
    assert_eq!(&bytes[0..8], &MAGIC);
    assert_eq!(get_u32(&bytes, 8), VERSION);
    assert_eq!(get_u32(&bytes, 12) as usize, HEADER_LEN);
    assert_eq!(get_u64(&bytes, 16), fp);
    assert_eq!(get_u64(&bytes, 24), prompt_hash(&prompt));
    assert_eq!(get_u64(&bytes, 32) as usize, prompt.len());
    assert_eq!(get_u32(&bytes, 40) as usize, src.n_layers());
    assert_eq!(get_u32(&bytes, 44) as usize, src.kv_dim());
    assert_eq!(get_u32(&bytes, 48), 1, "GDN スロット層の数");
    assert_eq!(get_u32(&bytes, 52), 0, "reserved は 0");
    assert_eq!(
        get_u64(&bytes, 56) as usize,
        bytes.len() - HEADER_LEN,
        "payload_len がヘッダと一致しない"
    );
    assert_eq!(
        get_u64(&bytes, 64),
        checksum(&bytes[HEADER_LEN..]),
        "payload checksum がヘッダと一致しない"
    );
    // 同じ入力からは同じバイト列 (決定的)
    assert_eq!(bytes, encode(&src, prompt.len(), fp, &prompt).unwrap());

    let path = save(&src, prompt.len(), dir.path(), fp, &prompt, 0).expect("save が失敗した");
    let mut dst = tiny_cache();
    load(&mut dst, &path, fp, &prompt).expect("load が失敗した");

    assert_eq!(dst.len(), src.len());
    assert_eq!(cache_bits(&src, prompt.len()), cache_bits(&dst, prompt.len()));
    assert_eq!(gdn_bits(&src), gdn_bits(&dst), "GDN 状態がビット一致しない");
    // 復元後も続きを書ける
    dst.advance(1);
    assert_eq!(dst.len(), prompt.len() + 1);
}

#[test]
fn truncated_inputs_are_rejected_without_panic() {
    let dir = TempDir::new("truncated");
    let prompt = vec![1u32, 2, 3];
    let mut src = tiny_cache();
    fill_bits(&mut src, prompt.len());
    let good = encode(&src, prompt.len(), 9, &prompt).unwrap();

    let payload_len = good.len() - HEADER_LEN;
    let cases: Vec<(&str, usize)> = vec![
        ("empty", 0),
        ("magic_only", 8),
        ("header_minus_one", HEADER_LEN - 1),
        ("header_only", HEADER_LEN),
        ("mid_payload", HEADER_LEN + payload_len / 2),
        ("last_byte_missing", good.len() - 1),
    ];
    for (name, cut) in cases {
        let path = write_bytes(dir.path(), &format!("{name}.kvsnap"), &good[..cut]);
        let mut dst = tiny_cache();
        let err = load(&mut dst, &path, 9, &prompt).expect_err("切り詰めを受け入れてしまった");
        match name {
            "empty" | "magic_only" | "header_minus_one" => assert!(
                matches!(err, SnapshotError::Truncated { .. }),
                "{name}: Truncated を期待: {err}"
            ),
            _ => assert!(
                matches!(
                    err,
                    SnapshotError::LengthMismatch { .. } | SnapshotError::Truncated { .. }
                ),
                "{name}: 長さ系エラーを期待: {err}"
            ),
        }
        // 失敗時に復元先を触らない
        assert_eq!(dst.len(), 0, "{name}: 失敗時にキャッシュが書き換わった");
        assert_eq!(dst.capacity(), 0, "{name}: 失敗時にバッファを確保した");
    }
}

#[test]
fn bad_magic_and_unsupported_version_are_rejected() {
    let dir = TempDir::new("magic_version");
    let prompt = vec![4u32, 5];
    let mut src = tiny_cache();
    fill_bits(&mut src, prompt.len());
    let good = encode(&src, prompt.len(), 3, &prompt).unwrap();

    for (i, name) in [(0usize, "first"), (7, "last")] {
        let mut b = good.clone();
        b[i] ^= 0xff;
        let path = write_bytes(dir.path(), &format!("magic_{name}.kvsnap"), &b);
        let mut dst = tiny_cache();
        let err = load(&mut dst, &path, 3, &prompt).expect_err("壊れた magic を受け入れた");
        assert!(matches!(err, SnapshotError::BadMagic(_)), "{name}: {err}");
        assert_eq!(dst.len(), 0);
    }

    for (name, ver) in [("future", VERSION + 1), ("zero", 0), ("max", u32::MAX)] {
        let mut b = good.clone();
        put_u32(&mut b, 8, ver);
        let path = write_bytes(dir.path(), &format!("version_{name}.kvsnap"), &b);
        let mut dst = tiny_cache();
        let err = load(&mut dst, &path, 3, &prompt).expect_err("未知 version を受け入れた");
        assert!(
            matches!(err, SnapshotError::UnsupportedVersion(v) if v == ver),
            "{name}: UnsupportedVersion({ver}) を期待: {err}"
        );
        assert_eq!(dst.len(), 0);
    }
}

#[test]
fn corrupted_checksum_and_length_are_rejected_without_huge_allocation() {
    let dir = TempDir::new("corrupt");
    let prompt = vec![21u32, 22, 23, 24];
    let mut src = tiny_cache();
    fill_bits(&mut src, prompt.len());
    let good = encode(&src, prompt.len(), 5, &prompt).unwrap();

    // 1) ペイロードの 1 バイトだけ反転 → checksum 不一致 (長さは正しいまま)
    let mut b = good.clone();
    let last = b.len() - 1;
    b[last] ^= 0x01;
    let path = write_bytes(dir.path(), "payload_flip.kvsnap", &b);
    let mut dst = tiny_cache();
    let err = load(&mut dst, &path, 5, &prompt).expect_err("破損ペイロードを受け入れた");
    match err {
        SnapshotError::BadChecksum { expected, actual } => assert_ne!(expected, actual),
        other => panic!("BadChecksum を期待: {other}"),
    }
    assert_eq!(dst.len(), 0);

    // 2) checksum フィールドだけ書き換え → BadChecksum
    let mut b = good.clone();
    let bad_sum = get_u64(&b, 64) ^ 0xdead_beef;
    put_u64(&mut b, 64, bad_sum);
    let path = write_bytes(dir.path(), "checksum_field.kvsnap", &b);
    let mut dst = tiny_cache();
    let err = load(&mut dst, &path, 5, &prompt).expect_err("checksum 改竄を受け入れた");
    assert!(matches!(err, SnapshotError::BadChecksum { .. }), "{err}");

    // 3) payload_len が実体と違う (過大・過小) → 長さ検証で落ちる。
    //    u64::MAX を宣言しても確保は起きない (実体長との比較が先)。
    for (name, declared) in [
        ("len_u64_max", u64::MAX),
        ("len_plus_one", (good.len() - HEADER_LEN) as u64 + 1),
        ("len_zero", 0u64),
    ] {
        let mut b = good.clone();
        put_u64(&mut b, 56, declared);
        let path = write_bytes(dir.path(), &format!("{name}.kvsnap"), &b);
        let mut dst = tiny_cache();
        let err = load(&mut dst, &path, 5, &prompt).expect_err("不正な payload_len を受け入れた");
        assert!(
            matches!(err, SnapshotError::LengthMismatch { declared: d, .. } if d == declared),
            "{name}: LengthMismatch({declared}) を期待: {err}"
        );
        assert_eq!(dst.len(), 0);
        assert_eq!(dst.capacity(), 0, "{name}: 長さ不一致でバッファを確保した");
    }

    // 4) n_tokens を巨大に詐称 (checksum は実ペイロードに合わせて再計算) →
    //    読み出しで Truncated。巨大 Vec の確保より前に弾かれる。
    let mut b = good.clone();
    put_u64(&mut b, 32, 1u64 << 40);
    let sum = checksum(&b[HEADER_LEN..]);
    put_u64(&mut b, 64, sum);
    let path = write_bytes(dir.path(), "huge_n_tokens.kvsnap", &b);
    let mut dst = tiny_cache();
    let err = load(&mut dst, &path, 5, &prompt).expect_err("巨大 n_tokens を受け入れた");
    assert!(
        matches!(
            err,
            SnapshotError::Truncated { .. } | SnapshotError::LengthMismatch { .. }
        ),
        "huge n_tokens: 想定外のエラー: {err}"
    );
    assert_eq!(dst.len(), 0);
    assert_eq!(dst.capacity(), 0);
}

#[test]
fn zero_token_and_boundary_capacity_cache_edges() {
    // max_len 0 のキャッシュ: 空のまま snapshot / restore / advance(0) が通る
    let mut z = KvCache::new(2, 3, 0);
    assert_eq!(z.len(), 0);
    assert_eq!(z.capacity(), 0);
    assert_eq!(z.max_len(), 0);
    {
        let (k, v) = z.snapshot(0);
        assert_eq!(k.len(), 2);
        assert_eq!(v.len(), 2);
        assert!(k.iter().all(|l| l.is_empty()));
        assert!(v.iter().all(|l| l.is_empty()));
    }
    z.advance(0);
    assert_eq!(z.len(), 0);
    assert!(z.is_empty());

    // トークン 0 のスナップショットは encode が拒否する (空ファイルを作らない)
    assert!(matches!(
        encode(&z, 0, 1, &[]).unwrap_err(),
        SnapshotError::ShapeMismatch(_)
    ));

    // 容量上限ちょうどまで書いて round-trip する
    let max = 4usize;
    let dim = 3usize;
    let mut full = KvCache::new(2, dim, max);
    let (k0, v0) = full.kv_bufs(0, max);
    assert_eq!(k0.len(), max * dim, "capacity は上限ちょうどまで伸びる");
    assert_eq!(v0.len(), max * dim);
    for i in 0..max * dim {
        k0[i] = i as f32 + 0.5;
        v0[i] = -(i as f32);
    }
    full.advance(max);
    assert_eq!(full.len(), max);
    assert_eq!(full.capacity(), max);

    let prompt: Vec<u32> = vec![100, 101, 102, 103];
    let bytes = encode(&full, max, 0x42, &prompt).expect("上限ちょうどの encode が失敗した");
    assert_eq!(
        bytes.len() - HEADER_LEN,
        2 * 4 + 2 * dim * max * 4 * 2,
        "payload = 層タグ + 各層 K/V の全量"
    );

    let dir = TempDir::new("boundary");
    let path = save(&full, max, dir.path(), 0x42, &prompt, 0).unwrap();
    let mut dst = KvCache::new(2, dim, max);
    load(&mut dst, &path, 0x42, &prompt).unwrap();
    assert_eq!(dst.len(), max);
    assert_eq!(cache_bits(&full, max), cache_bits(&dst, max));

    // 縮めてから同じ上限まで復元できる
    dst.truncate(0);
    dst.restore(
        &full.snapshot(max).0,
        &full.snapshot(max).1,
        max,
    );
    assert_eq!(dst.len(), max);
    assert_eq!(cache_bits(&full, max), cache_bits(&dst, max));

    // len を超える n_tokens は encode が拒否する
    let mut short = KvCache::new(1, dim, 8);
    let (k, _) = short.kv_bufs(0, 2);
    k[0..2 * dim].fill(1.0);
    short.advance(2);
    assert!(matches!(
        encode(&short, 3, 1, &[1, 2, 3]).unwrap_err(),
        SnapshotError::ShapeMismatch(_)
    ));
}

/// 契約の明文化: `restore` の空レイヤは GDN スロット層専用。n=0 の
/// スナップショット (全層が空) を KV 層へ復元しようとすると panic する。
/// ゼロに戻す用途は `reset` を使う。
#[test]
#[should_panic(expected = "空スナップショット層は GDN 層のみ")]
fn restore_zero_tokens_panics_for_kv_only_cache() {
    let mut c = KvCache::new(2, 3, 4);
    let (k, v) = c.snapshot(0);
    c.restore(&k, &v, 0);
}
