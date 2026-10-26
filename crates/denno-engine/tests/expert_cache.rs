//! P-MOE-B2: expert 重みの SSD pread とスロットキャッシュのテスト。
//!
//! 実 GGUF は開かない。合成バイト列とテスト用 [`PreadAt`] 実装
//! (`Vec<u8>` + 読み取り回数) で検証し、実ファイル経路は
//! テストが作った一時ファイル (CARGO_TARGET_TMPDIR) だけで確かめる。
//! GPU / CUDA は通さない。

use std::path::{Path, PathBuf};

use denno_engine::kquant::GgmlType;
use denno_engine::model::expert_cache::{
    CacheStats, ExpertCache, ExpertKey, FileReader, PreadAt, DEFAULT_SLOT_BYTES, DEFAULT_SLOTS,
};
use denno_engine::model::moe_expert::ExpertSlice;

/// 決定的なバイト列 (位置ごとに値が違うので、オフセットのずれが即座に分かる)。
fn pat(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

/// `Vec<u8>` をファイルに見立てるテスト用の口。
///
/// 読み取り回数と読めたバイト数を数える。ファイル末尾を超えるオフセットでは
/// `Ok(0)` (EOF) を返し、エラー化はキャッシュ側の責務に任せる。
struct Mem {
    data: Vec<u8>,
    reads: usize,
    bytes: u64,
    /// 1 回の pread で返す上限 (短読みの再現用)。既定は無制限。
    max_chunk: usize,
}

impl Mem {
    fn new(data: Vec<u8>) -> Self {
        Mem { data, reads: 0, bytes: 0, max_chunk: usize::MAX }
    }

    fn short(data: Vec<u8>, max_chunk: usize) -> Self {
        Mem { data, reads: 0, bytes: 0, max_chunk }
    }
}

impl PreadAt for Mem {
    fn pread_at(&mut self, off: u64, buf: &mut [u8]) -> anyhow::Result<usize> {
        self.reads += 1;
        let off = off as usize;
        if off >= self.data.len() {
            return Ok(0); // EOF
        }
        let n = buf.len().min(self.data.len() - off).min(self.max_chunk);
        buf[..n].copy_from_slice(&self.data[off..off + n]);
        self.bytes += n as u64;
        Ok(n)
    }
}

fn key(layer: u16, expert: u16) -> ExpertKey {
    ExpertKey { layer, expert }
}

// ---- 1. 未命中 → ヒット -------------------------------------------------

#[test]
fn miss_then_hit_returns_same_bytes_and_reads_once() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(2, 1024);
    let k = key(0, 1);

    let a = c.get(k, 100, 512, &mut src).unwrap();
    assert_eq!(a, &data[100..612], "未命中の戻り値がファイル内容と違う");
    assert_eq!(src.reads, 1);
    assert_eq!(src.bytes, 512);

    let b = c.get(k, 100, 512, &mut src).unwrap();
    assert_eq!(b, &data[100..612], "ヒットの戻り値が未命中時と違う");
    assert_eq!(src.reads, 1, "ヒットなのに読み直している");
    assert_eq!(src.bytes, 512);

    let s = c.stats();
    assert_eq!((s.hits, s.misses, s.bytes_read, s.evictions), (1, 1, 512, 0));
}

#[test]
fn same_key_with_other_len_is_a_miss() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(1, 1024);
    let k = key(3, 7);
    assert_eq!(c.get(k, 0, 256, &mut src).unwrap(), &data[0..256]);
    assert_eq!(c.get(k, 0, 512, &mut src).unwrap(), &data[0..512]);
    assert_eq!(src.reads, 2, "長さ違いをヒット扱いした");
}

// ---- 2. LRU の追い出し ---------------------------------------------------

#[test]
fn lru_evicts_the_oldest_of_three_slots() {
    let data = pat(8192);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(3, 512);
    for e in 0..3u16 {
        c.get(key(0, e), (e as u64) * 512, 512, &mut src).unwrap();
    }
    assert_eq!(src.reads, 3);
    assert_eq!(c.stats().evictions, 0);

    // 4 種類目 → 最初の 1 つ (expert 0) が追い出される
    c.get(key(0, 3), 3 * 512, 512, &mut src).unwrap();
    assert_eq!(src.reads, 4);
    assert_eq!(c.stats().evictions, 1);

    // 残っている expert 1 は読み直さない
    assert_eq!(c.get(key(0, 1), 512, 512, &mut src).unwrap(), &data[512..1024]);
    assert_eq!(src.reads, 4, "残っているはずの expert を読み直した");

    // 追い出された expert 0 を再要求すると読み取り回数が増える
    assert_eq!(c.get(key(0, 0), 0, 512, &mut src).unwrap(), &data[0..512]);
    assert_eq!(src.reads, 5, "追い出された expert がヒットした");
    assert_eq!(c.stats().evictions, 2);
}

#[test]
fn lru_order_follows_last_use_not_insertion() {
    // 0,1,2 を入れて 0 を触り直す → 追い出されるのは 1。
    let data = pat(8192);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(3, 512);
    for e in 0..3u16 {
        c.get(key(0, e), (e as u64) * 512, 512, &mut src).unwrap();
    }
    c.get(key(0, 0), 0, 512, &mut src).unwrap();
    c.get(key(0, 3), 3 * 512, 512, &mut src).unwrap();
    assert_eq!(c.stats().evictions, 1);

    let reads = src.reads;
    c.get(key(0, 0), 0, 512, &mut src).unwrap();
    assert_eq!(src.reads, reads, "触り直した expert 0 が追い出された");
    c.get(key(0, 1), 512, 512, &mut src).unwrap();
    assert_eq!(src.reads, reads + 1, "最終使用が最も古い expert 1 が残っていた");
}

// ---- 3. エラー系 ---------------------------------------------------------

#[test]
fn len_over_slot_bytes_is_an_error() {
    let mut src = Mem::new(pat(4096));
    let mut c = ExpertCache::new(1, 1024);
    assert!(c.get(key(0, 0), 0, 1025, &mut src).is_err(), "スロット超過が通った");
    assert_eq!(src.reads, 0, "エラーなのに読んでいる");
    // 0 バイトも弾く
    assert!(c.get(key(0, 0), 0, 0, &mut src).is_err());
    // ちょうど slot_bytes は通る (境界)
    assert_eq!(c.get(key(0, 0), 0, 1024, &mut src).unwrap().len(), 1024);
}

#[test]
fn out_of_range_offset_is_an_error() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(2, 1024);
    // 末尾をまたぐ (off + len > ファイル長)
    assert!(c.get(key(0, 0), 4000, 512, &mut src).is_err(), "末尾超えが通った");
    // 完全に外
    assert!(c.get(key(0, 1), 99_999, 512, &mut src).is_err(), "範囲外が通った");
    // off + len が u64 を溢れる
    assert!(c.get(key(0, 0), u64::MAX - 8, 512, &mut src).is_err());
}

#[test]
fn failed_read_does_not_leave_a_valid_slot() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(1, 1024);
    let k = key(1, 2);
    assert!(c.get(k, 4000, 512, &mut src).is_err());
    // 同じ鍵・同じ長さで正しい位置を要求 → ヒットしてはいけない
    assert_eq!(c.get(k, 0, 512, &mut src).unwrap(), &data[0..512]);
    let s = c.stats();
    assert_eq!(s.hits, 0, "失敗した読みがヒットになった");
    assert_eq!(s.misses, 2);
    assert_eq!(s.bytes_read, 512, "失敗した読みを bytes_read に計上した");
}

#[test]
fn short_read_is_an_error() {
    // 要求より短く返す口 (trait 契約違反だが、キャッシュ側の門番を確かめる)。
    let mut src = Mem::short(pat(4096), 256);
    let mut c = ExpertCache::new(1, 1024);
    assert!(c.get(key(0, 0), 0, 512, &mut src).is_err(), "短読みが通った");
}

// ---- 4. slots == 0 -------------------------------------------------------

#[test]
fn zero_slots_reads_every_time_but_values_are_right() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(0, 1024);
    assert_eq!(c.slots(), 0);
    let k = key(4, 9);
    for _ in 0..3 {
        assert_eq!(c.get(k, 128, 512, &mut src).unwrap(), &data[128..640]);
    }
    assert_eq!(src.reads, 3, "slots=0 なのに読み回数が 3 でない");
    let s = c.stats();
    assert_eq!((s.hits, s.misses, s.evictions), (0, 3, 0));
    assert_eq!(s.bytes_read, 3 * 512);
    // スロット超過は slots=0 でもエラー (scratch を無制限に伸ばさない)
    assert!(c.get(k, 0, 1025, &mut src).is_err());
}

// ---- 5. 統計 -------------------------------------------------------------

#[test]
fn stats_are_exact_over_a_mixed_sequence() {
    let data = pat(8192);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(2, 512);
    // 0:miss, 1:miss, 0:hit, 2:miss(evict 1), 0:hit, 1:miss(evict 2), 1:hit
    for (e, off) in [(0u16, 0u64), (1, 512), (0, 0), (2, 1024), (0, 0), (1, 512), (1, 512)] {
        c.get(key(0, e), off, 512, &mut src).unwrap();
    }
    let s = c.stats();
    assert_eq!(s.hits, 3, "hits");
    assert_eq!(s.misses, 4, "misses");
    assert_eq!(s.evictions, 2, "evictions");
    assert_eq!(s.bytes_read, 4 * 512, "bytes_read");
    assert_eq!(src.reads, 4, "実際の読み取り回数");
}

#[test]
fn reset_stats_keeps_slots_warm() {
    let data = pat(4096);
    let mut src = Mem::new(data.clone());
    let mut c = ExpertCache::new(2, 1024);
    let k = key(0, 0);
    c.get(k, 0, 512, &mut src).unwrap();
    c.get(k, 0, 512, &mut src).unwrap();
    assert_eq!(c.stats().hits, 1);

    c.reset_stats();
    assert_eq!(c.stats(), CacheStats::default());

    c.get(k, 0, 512, &mut src).unwrap();
    assert_eq!(c.stats().hits, 1, "reset でスロットまで捨てた");
    assert_eq!(src.reads, 1, "reset 後に読み直している");
}

// ---- 6. 定数 (前提の実測値) ---------------------------------------------

#[test]
fn defaults_cover_the_largest_expert() {
    // 前提表: Q5_1 の ffn_down_exps が 1 expert = 1,228,800 B で最大。
    let down = ExpertSlice::new(GgmlType::Q5_1, 640, 2560, 512, 0).unwrap();
    assert_eq!(down.len, 1_228_800);
    assert!(
        DEFAULT_SLOT_BYTES >= down.len,
        "既定のスロット {DEFAULT_SLOT_BYTES} B が最大 expert {} B に足りない",
        down.len
    );
    // gate/up も収まる
    let gate = ExpertSlice::new(GgmlType::Q4K, 2560, 640, 512, 0).unwrap();
    assert_eq!(gate.len, 921_600);
    assert!(DEFAULT_SLOT_BYTES >= gate.len);
    // 常駐量 (64 × 1.25 MiB = 83.9 MB) が RAM に載る範囲であること
    assert_eq!(DEFAULT_SLOTS, 64);
    assert_eq!(DEFAULT_SLOTS * DEFAULT_SLOT_BYTES, 83_886_080);
    // new() がその組み合わせで作れる
    let c = ExpertCache::new(DEFAULT_SLOTS, DEFAULT_SLOT_BYTES);
    assert_eq!(c.slots(), DEFAULT_SLOTS);
    assert_eq!(c.slot_bytes(), DEFAULT_SLOT_BYTES);
    assert_eq!(c.stats(), CacheStats::default());
}

#[test]
fn cache_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ExpertCache>();
    assert_send::<FileReader>();
}

// ---- 7. FileReader (実ファイル) -----------------------------------------

/// テストが作った一時ファイル。`CARGO_TARGET_TMPDIR` 配下に置き、drop で消す。
struct TempFile(PathBuf);

impl TempFile {
    fn new(tag: &str, data: &[u8]) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(dir).expect("tmpdir");
        let p = dir.join(format!("denno_expert_cache_{tag}_{}_{n}.bin", std::process::id()));
        std::fs::write(&p, data).expect("write temp file");
        TempFile(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn file_reader_reads_exact_bytes_at_offset() {
    let data = pat(4096);
    let f = TempFile::new("read", &data);
    let mut r = FileReader::open(f.path()).unwrap();

    let mut buf = vec![0u8; 512];
    assert_eq!(r.pread_at(100, &mut buf).unwrap(), 512);
    assert_eq!(buf, &data[100..612]);

    // 0 オフセット
    assert_eq!(r.pread_at(0, &mut buf).unwrap(), 512);
    assert_eq!(buf, &data[0..512]);

    // 末尾ちょうど
    let mut tail = vec![0u8; 96];
    assert_eq!(r.pread_at(4000, &mut tail).unwrap(), 96);
    assert_eq!(tail, &data[4000..4096]);

    // 1 バイト
    let mut one = [0u8; 1];
    assert_eq!(r.pread_at(7, &mut one).unwrap(), 1);
    assert_eq!(one[0], data[7]);

    // 長さ 0 の要求は 0 バイト読んで成功
    assert_eq!(r.pread_at(1234, &mut []).unwrap(), 0);
}

#[test]
fn file_reader_past_eof_is_an_error() {
    let data = pat(1024);
    let f = TempFile::new("eof", &data);
    let mut r = FileReader::open(f.path()).unwrap();

    // 完全に外
    let mut buf = vec![0u8; 64];
    assert!(r.pread_at(1024, &mut buf).is_err(), "EOF ちょうどからの読みが通った");
    assert!(r.pread_at(99_999, &mut buf).is_err(), "範囲外の読みが通った");
    // 末尾をまたぐ (短読み → エラー)
    assert!(r.pread_at(1000, &mut buf).is_err(), "末尾をまたぐ読みが通った");
    // 正常位置は通る (上の失敗で壊れていないこと)
    assert_eq!(r.pread_at(0, &mut buf).unwrap(), 64);
    assert_eq!(buf, &data[0..64]);
}

#[test]
fn file_reader_open_missing_path_is_an_error() {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join("denno_expert_cache_missing.bin");
    let _ = std::fs::remove_file(&p);
    assert!(FileReader::open(&p).is_err());
}

#[test]
fn cache_over_file_reader_matches_file_bytes() {
    // 4 expert × 512 B を 2 スロットで回す (実ファイル経路の end-to-end)。
    let data = pat(2048);
    let f = TempFile::new("cache", &data);
    let mut src = FileReader::open(f.path()).unwrap();
    let mut c = ExpertCache::new(2, 512);

    for e in 0..4u16 {
        let off = e as u64 * 512;
        let got = c.get(key(5, e), off, 512, &mut src).unwrap();
        assert_eq!(got, &data[e as usize * 512..(e as usize + 1) * 512], "expert {e}");
    }
    let s = c.stats();
    assert_eq!((s.hits, s.misses, s.evictions), (0, 4, 2));
    assert_eq!(s.bytes_read, 4 * 512);

    // 直近 2 つは温まっている
    assert_eq!(c.get(key(5, 3), 3 * 512, 512, &mut src).unwrap(), &data[1536..2048]);
    assert_eq!(c.get(key(5, 2), 2 * 512, 512, &mut src).unwrap(), &data[1024..1536]);
    assert_eq!(c.stats().hits, 2);
    assert_eq!(c.stats().bytes_read, 4 * 512, "ヒットで読んでいる");

    // 範囲外はエラー
    assert!(c.get(key(5, 9), 4096, 512, &mut src).is_err());
}
