//! task 794: VRAM expert キャッシュの**頻度ゲート付き**ポリシー (TinyLFU 系)。
//!
//! # 何をするか
//!
//! [`crate::model::expert_vram`] の LRU は「最終使用が最も古い 1 本」を無条件に
//! 追い出して、ミスした expert を載せていた。decode の作業集合 (よく使う expert)
//! は偏っているので、prefill や長いプロンプトで 1 回だけ使う expert が入ると、
//! よく使う expert が押し出されて次のトークンでまたミスする (churn)。1 回の
//! 載せ替えは H2D 0.15〜0.25 ms + 追い出した expert の CPU 再計算なので、
//! 1 回だけ使う expert を通すのは往復で損になる。
//!
//! ここでは 2 つを分けて持つ:
//!
//! 1. [`FreqSketch`] — 4 bit カウンタの count-min sketch (depth 4)。**重み付き**で
//!    加算し、加算回数が `10 * width` に達するたびに全カウンタを半減する
//!    (TinyLFU の aging)。O(1) で、メモリは depth * width / 2 バイト。
//! 2. [`CachePolicy`] — `tinylfu` (既定。頻度ゲート: 候補の推定頻度が犠牲者の
//!    推定頻度より**低い**ときは載せない) と `lru` (切り戻し。従来と 1 バイトも
//!    変わらない)。
//!
//! # 何をしないか
//!
//! 本家 W-TinyLFU の「window (小さな LRU 領域)」は持たない。expert の鍵は
//! 層×expert で数千、1 本 3 MB なので、window 1% は 1〜2 本ぶんにしかならず、
//! 領域を分ける会計の複雑さに見合わない。代わりに aging が「最近使われていない
//! 高いカウンタ」を落とすので、頻度は自動的に古びる (LFU の固定化を避ける)。
//!
//! # 既定
//!
//! 既定は `tinylfu`。task 794 の lab (t794b Q4_K_XL / t794c IQ3_XXS、14 GB 予算、
//! depth8k) で 8k/14k が 3% 以上速く、errors=0、IQ3_XXS は md5 一致だった。
//! `DENNO_EXPERT_CACHE_POLICY=lru` で従来の LRU に戻せる。

use std::hash::{Hash, Hasher};

/// `DENNO_EXPERT_CACHE_POLICY`: VRAM expert キャッシュの載せ替え方。
///
/// 既定は `tinylfu` (task 794 の lab で Q4 / IQ3_XXS とも 8k/14k が 3% 以上速く、
/// errors=0、IQ3 は md5 一致だったため)。`lru` / `0` / `off` / `false` で
/// 従来の LRU に戻せる (切り戻し用)。
pub const EXPERT_CACHE_POLICY_ENV: &str = "DENNO_EXPERT_CACHE_POLICY";

/// sketch の 1 行あたりのカウンタ数 (2 の冪)。expert の鍵は数千なので 16,384 で
/// 衝突は十分少ない (4 bit × 16,384 × 4 行 = 32 KiB)。
pub const DEFAULT_SKETCH_WIDTH: usize = 16_384;

/// count-min の行数。
pub const DEFAULT_SKETCH_DEPTH: usize = 4;

/// 加算がこの倍数に達するたびに全カウンタを半減する (本家 TinyLFU と同じ 10)。
pub const SKETCH_SAMPLE_MULTIPLIER: u64 = 10;

/// 4 bit カウンタの上限。
pub const SKETCH_MAX_COUNT: u8 = 15;

/// 載せ替えの方針。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// 従来の LRU (切り戻し用)。頻度は見ない。
    #[default]
    Lru,
    /// 頻度ゲート付き (TinyLFU の admission filter)。既定。
    TinyLfu,
}

impl CachePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            CachePolicy::Lru => "lru",
            CachePolicy::TinyLfu => "tinylfu",
        }
    }
}

/// [`EXPERT_CACHE_POLICY_ENV`] の解釈 (純粋関数)。
///
/// - `lru` / `0` / `off` / `false` → [`CachePolicy::Lru`] (従来。切り戻し用)
/// - それ以外 (未設定・空・`tinylfu`・`w-tinylfu`・`1`・`on`・知らない値)
///   → [`CachePolicy::TinyLfu`] (**既定**)
///
/// task 794: lab (t794b Q4_K_XL / t794c IQ3_XXS) で 8k/14k が 3% 以上速く、
/// errors=0、IQ3_XXS は md5 一致だったので既定を TinyLfu にした。
pub fn policy_from_value(v: Option<&str>) -> CachePolicy {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("lru") | Some("0") | Some("off") | Some("false") => CachePolicy::Lru,
        _ => CachePolicy::TinyLfu,
    }
}

/// プロセスのポリシーを env から読む (未初期化なら `Lru`)。
pub fn policy_from_env() -> CachePolicy {
    policy_from_value(std::env::var(EXPERT_CACHE_POLICY_ENV).ok().as_deref())
}

/// 鍵のハッシュ。`DefaultHasher::new()` は固定鍵なので**プロセス内で再現する**
/// (sketch は同じ実行の中で頻度を比べるだけなので、プロセス間の一致は要らない)。
pub fn key_hash<K: Hash + ?Sized>(key: &K) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

/// splitmix64 (行ごとの種を混ぜて index を作る)。
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 4 bit カウンタの count-min sketch (TinyLFU の frequency sketch)。
///
/// 各行は 1 バイトに 2 カウンタを詰める (下半分が偶数 index、上半分が奇数 index)。
/// `record` は全行を +1 (15 で飽和)、`estimate` は行の最小値。
#[derive(Clone, Debug)]
pub struct FreqSketch {
    width: usize,
    depth: usize,
    rows: Vec<Vec<u8>>,
    additions: u64,
    sample_size: u64,
    halvings: u64,
}

impl Default for FreqSketch {
    fn default() -> Self {
        FreqSketch::new(DEFAULT_SKETCH_WIDTH, DEFAULT_SKETCH_DEPTH)
    }
}

impl FreqSketch {
    /// `width` は 2 の冪へ切り上げ (最低 64)、`depth` は 1..=4 に丸める。
    pub fn new(width: usize, depth: usize) -> Self {
        let width = width.next_power_of_two().max(64);
        let depth = depth.clamp(1, 4);
        FreqSketch {
            width,
            depth,
            rows: vec![vec![0u8; width / 2]; depth],
            additions: 0,
            sample_size: SKETCH_SAMPLE_MULTIPLIER * width as u64,
            halvings: 0,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// 半減した回数 (発火の証拠)。
    pub fn halvings(&self) -> u64 {
        self.halvings
    }

    /// 最後に半減してからの加算回数。
    pub fn additions(&self) -> u64 {
        self.additions
    }

    /// sketch が占めるバイト数。
    pub fn bytes(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    fn slot(&self, h: u64, row: usize) -> (usize, bool) {
        let seed = (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let idx = (splitmix64(h ^ seed) as usize) & (self.width - 1);
        (idx >> 1, idx & 1 == 1)
    }

    /// 1 回ぶん加算する (加算回数が `10 * width` に達したら全行を半減)。
    pub fn record(&mut self, h: u64) {
        for r in 0..self.depth {
            let (i, hi) = self.slot(h, r);
            let b = &mut self.rows[r][i];
            let cur = if hi { *b >> 4 } else { *b & 0x0f };
            if cur < SKETCH_MAX_COUNT {
                let next = cur + 1;
                *b = if hi {
                    (*b & 0x0f) | (next << 4)
                } else {
                    (*b & 0xf0) | next
                };
            }
        }
        self.additions += 1;
        if self.additions >= self.sample_size {
            self.halve();
        }
    }

    /// 推定頻度 (行の最小値。0..=15)。
    pub fn estimate(&self, h: u64) -> u8 {
        let mut best = SKETCH_MAX_COUNT;
        for r in 0..self.depth {
            let (i, hi) = self.slot(h, r);
            let b = self.rows[r][i];
            let cur = if hi { b >> 4 } else { b & 0x0f };
            if cur < best {
                best = cur;
            }
        }
        best
    }

    /// 全カウンタを半減する (aging)。加算回数も半分に戻す。
    fn halve(&mut self) {
        for row in self.rows.iter_mut() {
            for b in row.iter_mut() {
                // 2 つの 4 bit を同時に 1 ビット右へ (0x77 で桁上がりを止める)。
                *b = (*b >> 1) & 0x77;
            }
        }
        self.additions /= 2;
        self.halvings += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_env_values() {
        // task 794: 既定は tinylfu (未設定・空・知らない値)。切り戻しは lru。
        assert_eq!(policy_from_value(None), CachePolicy::TinyLfu);
        assert_eq!(policy_from_value(Some("")), CachePolicy::TinyLfu);
        assert_eq!(policy_from_value(Some("nope")), CachePolicy::TinyLfu);
        for v in ["lru", "LRU ", "0", "off", "false", "OFF"] {
            assert_eq!(policy_from_value(Some(v)), CachePolicy::Lru, "{v}");
        }
        for v in [
            "tinylfu",
            "TinyLFU",
            "w-tinylfu",
            "wtinylfu",
            "tlfu",
            "1",
            "on",
            "true",
        ] {
            assert_eq!(policy_from_value(Some(v)), CachePolicy::TinyLfu, "{v}");
        }
    }

    #[test]
    fn sketch_counts_and_saturates() {
        let mut s = FreqSketch::new(1024, 4);
        let h = key_hash(&(3u32, 17u32));
        assert_eq!(s.estimate(h), 0);
        s.record(h);
        assert_eq!(s.estimate(h), 1);
        for _ in 0..100 {
            s.record(h);
        }
        assert_eq!(s.estimate(h), SKETCH_MAX_COUNT, "15 で飽和する");
        // 別の鍵は影響を受けない (幅 1024 に対して 2 鍵なので衝突しない)。
        assert_eq!(s.estimate(key_hash(&(999u32, 888u32))), 0);
    }

    #[test]
    fn sketch_halves_at_sample_size() {
        let mut s = FreqSketch::new(64, 4); // sample_size = 640
        let h = key_hash(&7u8);
        for _ in 0..SKETCH_MAX_COUNT {
            s.record(h);
        }
        assert_eq!(s.estimate(h), SKETCH_MAX_COUNT);
        // 640 回に満たすため、別の鍵で加算を進める。
        for i in 0..(640 - SKETCH_MAX_COUNT as u64) {
            s.record(key_hash(&(1000u64 + i)));
        }
        // 加算が sample_size に達した直後に半減する。
        assert_eq!(s.halvings(), 1, "1 回半減する");
        assert!(s.estimate(h) <= SKETCH_MAX_COUNT / 2 + 1, "{}", s.estimate(h));
        assert!(s.additions() < s.sample_size);
    }

    /// 半減は 4 bit の両方の桁を正しく落とす (上位 nibble が下位へ混ざらない)。
    #[test]
    fn sketch_halve_keeps_nibbles_separate() {
        let mut s = FreqSketch::new(64, 1);
        // 1 行目だけ見たいので depth 1。偶数 index と奇数 index を別の鍵で叩く。
        // index を直接作れないので、半減のビット演算そのものを確かめる。
        s.rows[0][0] = 0xfe; // 下位 14, 上位 15
        s.halve();
        assert_eq!(s.rows[0][0], 0x77, "14>>1=7, 15>>1=7");
        s.rows[0][1] = 0x8f; // 下位 15, 上位 8
        s.halve();
        assert_eq!(s.rows[0][1], 0x47, "下位 15>>1=7, 上位 8>>1=4");
    }

    #[test]
    fn sketch_is_o1_and_small() {
        let s = FreqSketch::default();
        assert_eq!(s.bytes(), DEFAULT_SKETCH_WIDTH / 2 * DEFAULT_SKETCH_DEPTH);
        assert_eq!(s.width(), DEFAULT_SKETCH_WIDTH);
        assert_eq!(s.depth(), DEFAULT_SKETCH_DEPTH);
    }
}
