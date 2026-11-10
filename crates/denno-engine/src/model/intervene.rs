//! task 217: 推論中の介入 (expert 無効化 / block skip / 残差ストリーム無効化)。
//!
//! ビューワから「モデルの一部を切って」チャットできるようにするための engine 側。
//! task 218 (残差の平均記録とステアリング) と task 219/220 (ビューワ) がこの
//! モジュールの上に載る。API 名はタスクの契約なので変えない。
//!
//! # 設計
//!
//! - 設定は [`Config`] 1 個。グローバルに [`RwLock`] 付きで置き、**forward step の
//!   先頭で 1 回だけ** [`snapshot`] して [`View`] にして層ループへ渡す。ホットパス
//!   (層 / expert / ルータ) ではロックを取らない。
//! - [`Config::is_empty`] のとき [`View::of`] は `None` を持つので、介入が無ければ
//!   ホットパスは「フラグ 1 個の分岐」だけで従来と 1 ビットも変わらない。
//! - 設定を変えるたびに epoch を 1 つ進める ([`current_epoch`])。serve の prefix
//!   cache は epoch が違う文脈を使わない (task 217 の 5.)。
//! - リクエスト中は [`pin_request`] で設定を固定する。生成の途中で POST が来ても
//!   そのリクエストには効かず、次のリクエストから効く。
//!
//! # env プリロード (`DENNO_INTERVENE`)
//!
//! lab の COND に書けるように、ラボが受け付ける文字 (A-Z a-z 0-9 _ . / -) だけで
//! 書けるコンパクトな spec を受ける。項目は `.` で区切り、`<種別><層>-<引数>`:
//!
//! | 種別 | 例 | 意味 |
//! |------|----|------|
//! | `e`  | `e3-17`    | 層 3 の expert 17 を無効化 |
//! | `b`  | `b5-moe`   | 層 5 の block (`attn` / `moe` / `shared` / `ple`) を skip |
//! | `s`  | `s7-2`     | 層 7 の残差ストリーム 2 への更新を落とす |
//!
//! 例: `e3-17.e3-20.b5-moe.s7-2`。空文字は「空だが有効な spec」。
//! 範囲外・未知の block 名・層あたりの残り expert が `n_expert_used` 未満になる
//! spec はエラー (HTTP は 400、env は警告して無視)。

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once, OnceLock, RwLock};

/// 設定が持ち得る層数の上限 (実モデルは 48)。範囲外は設定時にエラー。
pub const MAX_LAYERS: usize = 256;
/// expert 数の上限 (実モデルは 512)。
pub const MAX_EXPERTS: usize = 4096;
/// hyper-connection のストリーム数の上限 (実モデルは 4)。ビットマスクが u32。
pub const MAX_STREAMS: usize = 32;

/// 層の中で切る block の種別。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Block {
    /// full attention または GDN (その層が持つ方)。
    Attn = 0,
    /// routed expert (MoE)。shared expert は含まない。
    Moe = 1,
    /// shared expert。
    Shared = 2,
    /// PLE (per-layer embedding) の注入。
    Ple = 3,
}

impl Block {
    /// spec / HTTP で使う名前。
    pub fn name(self) -> &'static str {
        match self {
            Block::Attn => "attn",
            Block::Moe => "moe",
            Block::Shared => "shared",
            Block::Ple => "ple",
        }
    }

    /// 名前から引く (未知の名前は `None`)。
    pub fn parse(s: &str) -> Option<Block> {
        match s {
            "attn" => Some(Block::Attn),
            "moe" => Some(Block::Moe),
            "shared" => Some(Block::Shared),
            "ple" => Some(Block::Ple),
            _ => None,
        }
    }

    #[inline]
    fn bit(self) -> u8 {
        1u8 << (self as u8)
    }
}

/// モデルの形 (検証に使う)。serve は GGUF のメタデータから作る。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub n_layer: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_hc: usize,
}

/// 現在の介入設定。ホットパスは [`View`] 経由で読む。
#[derive(Clone, Debug)]
pub struct Config {
    epoch: u64,
    n_layer: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_hc: usize,
    ne_words: usize,
    /// `n_layer * ne_words` のビット列 (1 = 無効)。
    expert_off: Vec<u64>,
    /// 層ごとの block ビット (`Block::bit` の和)。
    block_bits: Vec<u8>,
    /// 層ごとの「更新を落とすストリーム」ビット。
    stream_bits: Vec<u32>,
    /// 指定そのもの (昇順・重複なし)。GET /v1/intervene とテストが読む。
    experts_off: Vec<(usize, usize)>,
    blocks_off: Vec<(usize, Block)>,
    streams_off: Vec<(usize, usize)>,
}

impl Config {
    /// 何も切っていない設定 (epoch 0)。グローバルの初期値。
    pub fn empty() -> Config {
        Config {
            epoch: 0,
            n_layer: 0,
            n_expert: 0,
            n_expert_used: 0,
            n_hc: 0,
            ne_words: 0,
            expert_off: Vec::new(),
            block_bits: Vec::new(),
            stream_bits: Vec::new(),
            experts_off: Vec::new(),
            blocks_off: Vec::new(),
            streams_off: Vec::new(),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn is_empty(&self) -> bool {
        self.experts_off.is_empty() && self.blocks_off.is_empty() && self.streams_off.is_empty()
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    pub fn n_expert_used(&self) -> usize {
        self.n_expert_used
    }

    pub fn n_hc(&self) -> usize {
        self.n_hc
    }

    pub fn experts_off(&self) -> &[(usize, usize)] {
        &self.experts_off
    }

    pub fn blocks_off(&self) -> &[(usize, Block)] {
        &self.blocks_off
    }

    pub fn streams_off(&self) -> &[(usize, usize)] {
        &self.streams_off
    }

    /// 層 `il` の expert 無効ビット列。その層に無効 expert が無ければ `None`
    /// (ホットパスはこの `None` で従来経路のまま走る)。
    #[inline]
    pub fn expert_mask(&self, il: usize) -> Option<&[u64]> {
        if self.ne_words == 0 || il >= self.n_layer {
            return None;
        }
        let s = &self.expert_off[il * self.ne_words..(il + 1) * self.ne_words];
        if s.iter().all(|&w| w == 0) {
            None
        } else {
            Some(s)
        }
    }

    /// 層 `il` の block `b` を skip するか。
    #[inline]
    pub fn block_off(&self, il: usize, b: Block) -> bool {
        match self.block_bits.get(il) {
            Some(&m) => m & b.bit() != 0,
            None => false,
        }
    }

    /// 層 `il` で更新を落とすストリームのビット (0 = 落とさない)。
    #[inline]
    pub fn stream_drop_bits(&self, il: usize) -> u32 {
        self.stream_bits.get(il).copied().unwrap_or(0)
    }

    /// prefill の常駐連鎖 (`DENNO_PF_RESIDENT`) と両立できるか。
    ///
    /// 常駐連鎖は attn / mix / combine / PLE をデバイス側で閉じるので、その 3 つを
    /// 切る設定では使えない (routed / shared expert はホスト側の MoE なので、
    /// 切っていてもそのまま連鎖に載せられる)。ストリームの無効化も combine が
    /// デバイス側なので不可。
    pub fn resident_safe(&self) -> bool {
        let ng = Block::Attn.bit() | Block::Ple.bit();
        self.block_bits.iter().all(|&b| b & ng == 0)
            && self.stream_bits.iter().all(|&s| s == 0)
    }
}

/// ホットパス用の参照。`None` は「介入なし」で、全メソッドが従来動作を返す。
#[derive(Clone, Copy, Default)]
pub struct View<'a> {
    cfg: Option<&'a Config>,
}

impl<'a> View<'a> {
    /// 空の設定なら `None` を持つ View にする (分岐 1 個で従来経路)。
    pub fn of(c: &'a Config) -> View<'a> {
        if c.is_empty() {
            View { cfg: None }
        } else {
            View { cfg: Some(c) }
        }
    }

    /// 介入なし。
    pub fn none() -> View<'a> {
        View { cfg: None }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.cfg.is_some()
    }

    #[inline]
    pub fn expert_mask(&self, il: usize) -> Option<&'a [u64]> {
        self.cfg.and_then(|c| c.expert_mask(il))
    }

    #[inline]
    pub fn block_off(&self, il: usize, b: Block) -> bool {
        self.cfg.map(|c| c.block_off(il, b)).unwrap_or(false)
    }

    /// prefill の常駐連鎖と両立できるか (介入なしも true)。
    #[inline]
    pub fn resident_safe(&self) -> bool {
        self.cfg.map(|c| c.resident_safe()).unwrap_or(true)
    }

    /// `hc` 本のストリームのうち更新を残すビット (bit s = 1 で残す)。
    #[inline]
    pub fn stream_keep(&self, il: usize, hc: usize) -> u32 {
        let all = if hc >= 32 { u32::MAX } else { (1u32 << hc) - 1 };
        match self.cfg {
            Some(c) => all & !c.stream_drop_bits(il),
            None => all,
        }
    }
}

// ---------------------------------------------------------------------------
// グローバル設定
// ---------------------------------------------------------------------------

static EPOCH: AtomicU64 = AtomicU64::new(0);

fn cell() -> &'static RwLock<Arc<Config>> {
    static CUR: OnceLock<RwLock<Arc<Config>>> = OnceLock::new();
    CUR.get_or_init(|| RwLock::new(Arc::new(Config::empty())))
}

/// 現在の設定 (ロックを 1 回取って Arc を clone する)。forward step の先頭で
/// 1 回だけ呼ぶ。層 / expert / ルータの中では呼ばない。
pub fn snapshot() -> Arc<Config> {
    // リクエスト中はピン留めした設定を返す (ロック無し)。
    if let Some(c) = pinned() {
        return c;
    }
    let g = cell().read().unwrap_or_else(|e| e.into_inner());
    Arc::clone(&g)
}

/// 現在の epoch。
pub fn current_epoch() -> u64 {
    snapshot().epoch()
}

fn pinned() -> Option<Arc<Config>> {
    PINS.with(|p| p.borrow().last().cloned())
}

thread_local! {
    /// リクエスト中に固定した設定のスタック ([`pin_request`])。
    static PINS: RefCell<Vec<Arc<Config>>> = const { RefCell::new(Vec::new()) };
}

/// リクエスト 1 本の間、いまの設定を固定する。
///
/// 生成の途中で `POST /v1/intervene` が来ても、このガードが生きている間は
/// [`snapshot`] が固定した設定を返すので、走っている生成の数値は変わらない。
/// ドロップで解除する (入れ子も可)。
pub struct Pin {
    _priv: (),
}

impl Drop for Pin {
    fn drop(&mut self) {
        PINS.with(|p| {
            p.borrow_mut().pop();
        });
    }
}

/// いまの設定を固定する。[`Pin`] をドロップするまで解除されない。
pub fn pin_request() -> Pin {
    let c = {
        let g = cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    PINS.with(|p| p.borrow_mut().push(c));
    Pin { _priv: () }
}

fn install(build: impl FnOnce(u64) -> Config) -> Arc<Config> {
    let mut g = cell().write().unwrap_or_else(|e| e.into_inner());
    let epoch = EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
    let cfg = Arc::new(build(epoch));
    *g = Arc::clone(&cfg);
    cfg
}

/// 設定を置き換える (epoch を 1 つ進める)。
pub fn set_config(
    shape: &Shape,
    experts: Vec<(usize, usize)>,
    blocks: Vec<(usize, Block)>,
    streams: Vec<(usize, usize)>,
) -> Result<Arc<Config>, String> {
    let (experts, blocks, streams) = validate(shape, experts, blocks, streams)?;
    Ok(install(|epoch| Config {
        epoch,
        ..build(shape, experts, blocks, streams)
    }))
}

/// 何も切っていない設定にする (epoch は進む)。
pub fn clear(shape: &Shape) -> Arc<Config> {
    install(|epoch| Config {
        epoch,
        ..build(shape, Vec::new(), Vec::new(), Vec::new())
    })
}

/// コンパクト spec ([モジュール doc](self)) から設定する。
pub fn set_from_spec(spec: &str, shape: &Shape) -> Result<Arc<Config>, String> {
    let (experts, blocks, streams) = parse_spec(spec, shape)?;
    set_config(shape, experts, blocks, streams)
}

/// 検証 (範囲・block 名・層あたりの残り expert 数) だけを行う。
#[allow(clippy::type_complexity)]
pub fn validate(
    shape: &Shape,
    experts: Vec<(usize, usize)>,
    blocks: Vec<(usize, Block)>,
    streams: Vec<(usize, usize)>,
) -> Result<
    (
        Vec<(usize, usize)>,
        Vec<(usize, Block)>,
        Vec<(usize, usize)>,
    ),
    String,
> {
    if shape.n_layer > MAX_LAYERS {
        return Err(format!(
            "層数 {} が上限 {MAX_LAYERS} を超える",
            shape.n_layer
        ));
    }
    if shape.n_expert > MAX_EXPERTS {
        return Err(format!(
            "expert 数 {} が上限 {MAX_EXPERTS} を超える",
            shape.n_expert
        ));
    }
    if shape.n_hc > MAX_STREAMS {
        return Err(format!(
            "ストリーム数 {} が上限 {MAX_STREAMS} を超える",
            shape.n_hc
        ));
    }
    let mut experts: Vec<(usize, usize)> = experts;
    let mut blocks: Vec<(usize, Block)> = blocks;
    let mut streams: Vec<(usize, usize)> = streams;
    experts.sort_unstable();
    experts.dedup();
    blocks.sort_unstable();
    blocks.dedup();
    streams.sort_unstable();
    streams.dedup();

    for &(l, e) in &experts {
        if l >= shape.n_layer {
            return Err(format!("層 {l} が範囲外 (0..{})", shape.n_layer));
        }
        if shape.n_expert == 0 {
            return Err(format!("層 {l}: このモデルに expert が無い"));
        }
        if e >= shape.n_expert {
            return Err(format!(
                "expert {e} が範囲外 (層 {l}, 0..{})",
                shape.n_expert
            ));
        }
    }
    for &(l, _) in &blocks {
        if l >= shape.n_layer {
            return Err(format!("層 {l} が範囲外 (0..{})", shape.n_layer));
        }
    }
    for &(l, s) in &streams {
        if l >= shape.n_layer {
            return Err(format!("層 {l} が範囲外 (0..{})", shape.n_layer));
        }
        if s >= shape.n_hc {
            return Err(format!(
                "ストリーム {s} が範囲外 (層 {l}, 0..{})",
                shape.n_hc
            ));
        }
    }
    // 層あたり「残る expert」が top-k 未満だとルータが k 個選べない。
    if !experts.is_empty() && shape.n_expert_used > 0 {
        let mut cnt = vec![0usize; shape.n_layer.max(1)];
        for &(l, _) in &experts {
            cnt[l] += 1;
        }
        let ne = shape.n_expert;
        let k = shape.n_expert_used;
        for (l, &c) in cnt.iter().enumerate() {
            if ne.saturating_sub(c) < k {
                return Err(format!(
                    "層 {l}: 残り expert {} < n_expert_used {k}",
                    ne.saturating_sub(c)
                ));
            }
        }
    }
    Ok((experts, blocks, streams))
}

fn build(
    shape: &Shape,
    experts: Vec<(usize, usize)>,
    blocks: Vec<(usize, Block)>,
    streams: Vec<(usize, usize)>,
) -> Config {
    let ne_words = shape.n_expert.div_ceil(64);
    let mut cfg = Config {
        epoch: 0,
        n_layer: shape.n_layer,
        n_expert: shape.n_expert,
        n_expert_used: shape.n_expert_used,
        n_hc: shape.n_hc,
        ne_words,
        expert_off: vec![0u64; shape.n_layer * ne_words],
        block_bits: vec![0u8; shape.n_layer],
        stream_bits: vec![0u32; shape.n_layer],
        experts_off: experts.clone(),
        blocks_off: blocks.clone(),
        streams_off: streams.clone(),
    };
    for &(l, e) in &experts {
        let w = l * ne_words + e / 64;
        cfg.expert_off[w] |= 1u64 << (e % 64);
    }
    for &(l, b) in &blocks {
        cfg.block_bits[l] |= b.bit();
    }
    for &(l, s) in &streams {
        cfg.stream_bits[l] |= 1u32 << s;
    }
    cfg
}

// ---------------------------------------------------------------------------
// spec パーサ
// ---------------------------------------------------------------------------

/// ラボが受け付ける文字だけで書かれた spec を分解する。
#[allow(clippy::type_complexity)]
pub fn parse_spec(
    spec: &str,
    shape: &Shape,
) -> Result<
    (
        Vec<(usize, usize)>,
        Vec<(usize, Block)>,
        Vec<(usize, usize)>,
    ),
    String,
> {
    let spec = spec.trim();
    let mut experts = Vec::new();
    let mut blocks = Vec::new();
    let mut streams = Vec::new();
    if spec.is_empty() {
        return Ok((experts, blocks, streams));
    }
    for item in spec.split('.') {
        if item.is_empty() {
            return Err(format!("空の項目がある: {spec:?}"));
        }
        for ch in item.chars() {
            let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '/' | '-');
            if !ok {
                return Err(format!(
                    "spec に使えない文字 {ch:?} (A-Z a-z 0-9 _ . / - のみ)"
                ));
            }
        }
        let (kind, rest) = item.split_at(1);
        let (layer_s, arg) = rest
            .split_once('-')
            .ok_or_else(|| format!("項目 {item:?} に '-' が無い (例 e3-17)"))?;
        let layer: usize = layer_s
            .parse()
            .map_err(|_| format!("項目 {item:?} の層番号が数値でない: {layer_s:?}"))?;
        match kind {
            "e" => {
                let e: usize = arg
                    .parse()
                    .map_err(|_| format!("項目 {item:?} の expert 番号が数値でない: {arg:?}"))?;
                experts.push((layer, e));
            }
            "b" => {
                let b = Block::parse(arg).ok_or_else(|| {
                    format!("項目 {item:?}: 未知の block {arg:?} (attn / moe / shared / ple)")
                })?;
                blocks.push((layer, b));
            }
            "s" => {
                let s: usize = arg
                    .parse()
                    .map_err(|_| format!("項目 {item:?} の stream 番号が数値でない: {arg:?}"))?;
                streams.push((layer, s));
            }
            other => {
                return Err(format!(
                    "項目 {item:?}: 未知の種別 {other:?} (e = expert / b = block / s = stream)"
                ))
            }
        }
    }
    validate(shape, experts, blocks, streams)
}

// ---------------------------------------------------------------------------
// env プリロード
// ---------------------------------------------------------------------------

/// `DENNO_INTERVENE` があれば 1 回だけ読んで設定する。
///
/// モデル構築のたびに呼ばれるが、プロセスで 1 回しか効かない (2 回目以降は
/// 何もしない)。不正な spec は警告を出して無視する = 起動は止めない。
pub fn init_from_env(shape: &Shape) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Ok(spec) = std::env::var("DENNO_INTERVENE") else {
            return;
        };
        match set_from_spec(&spec, shape) {
            Ok(c) => eprintln!(
                "[intervene] DENNO_INTERVENE={spec:?} epoch={} experts_off={} blocks_off={} streams_off={}",
                c.epoch(),
                c.experts_off().len(),
                c.blocks_off().len(),
                c.streams_off().len()
            ),
            Err(err) => eprintln!("[intervene] DENNO_INTERVENE={spec:?} を無視: {err}"),
        }
    });
}

/// テスト用: グローバルに触らずに [`Config`] を作る (epoch は 0)。
///
/// 本番の設定は [`set_config`] / [`clear`] / [`set_from_spec`] を通す。テストが
/// グローバルを書き換えると並列に走る他のテストの forward に漏れるため、
/// 層ループへ直接渡す用にこれを用意してある。
#[doc(hidden)]
pub fn config_for_test(
    shape: &Shape,
    experts: Vec<(usize, usize)>,
    blocks: Vec<(usize, Block)>,
    streams: Vec<(usize, usize)>,
) -> Result<Config, String> {
    let (experts, blocks, streams) = validate(shape, experts, blocks, streams)?;
    Ok(Config {
        epoch: 0,
        ..build(shape, experts, blocks, streams)
    })
}

/// テスト用: グローバルを空に戻す (epoch は進む)。テストは直列に走らせること。
#[doc(hidden)]
pub fn reset_for_test(shape: &Shape) {
    let _ = clear(shape);
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            n_layer: 4,
            n_expert: 8,
            n_expert_used: 2,
            n_hc: 4,
        }
    }

    /// グローバルの設定と epoch を触るテストの直列化 (cargo test は既定で並列)。
    static GLOBAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn parse_empty_spec_is_valid_and_empty() {
        let s = shape();
        let (e, b, st) = parse_spec("", &s).expect("空 spec は有効");
        assert!(e.is_empty() && b.is_empty() && st.is_empty());
        let (e, b, st) = parse_spec("   ", &s).expect("空白だけも空");
        assert!(e.is_empty() && b.is_empty() && st.is_empty());
    }

    #[test]
    fn parse_all_three_kinds() {
        let s = Shape {
            n_layer: 48,
            n_expert: 512,
            n_expert_used: 10,
            n_hc: 4,
        };
        let (e, b, st) = parse_spec("e3-17.e3-20.b5-moe.s7-2.b9-ple", &s).unwrap();
        assert_eq!(e, vec![(3, 17), (3, 20)]);
        assert_eq!(b, vec![(5, Block::Moe), (9, Block::Ple)]);
        assert_eq!(st, vec![(7, 2)]);
    }

    #[test]
    fn parse_rejects_bad_input() {
        let s = shape();
        // ラボが通さない文字
        assert!(parse_spec("e3-17:e3-20", &s).is_err());
        assert!(parse_spec("e3 17", &s).is_err());
        assert!(parse_spec("e3-17.", &s).is_err(), "空の項目");
        // 未知の種別 / block
        assert!(parse_spec("x3-1", &s).is_err());
        assert!(parse_spec("b3-head", &s).is_err());
        // 数値でない
        assert!(parse_spec("e3-x", &s).is_err());
        assert!(parse_spec("eA-1", &s).is_err());
        // 範囲外
        assert!(parse_spec("e4-1", &s).is_err(), "層 4 は範囲外");
        assert!(parse_spec("e3-8", &s).is_err(), "expert 8 は範囲外");
        assert!(parse_spec("s3-4", &s).is_err(), "stream 4 は範囲外");
    }

    #[test]
    fn validation_keeps_enough_experts_for_topk() {
        let s = Shape {
            n_layer: 2,
            n_expert: 8,
            n_expert_used: 3,
            n_hc: 2,
        };
        // 層 1 で 5 本まで切れる (残り 3 = k)。6 本はエラー。
        let five: Vec<(usize, usize)> = (0..5).map(|e| (1, e)).collect();
        assert!(validate(&s, five.clone(), vec![], vec![]).is_ok());
        let six: Vec<(usize, usize)> = (0..6).map(|e| (1, e)).collect();
        let err = validate(&s, six, vec![], vec![]).unwrap_err();
        assert!(err.contains("n_expert_used"), "メッセージ: {err}");
        // 別の層なら影響しない (層ごとに数える)。
        let mut ok = five;
        ok.push((0, 7));
        assert!(validate(&s, ok, vec![], vec![]).is_ok());
    }

    #[test]
    fn block_bits_and_expert_mask_layout() {
        let s = shape();
        let cfg = build(
            &s,
            vec![(1, 2), (1, 7), (3, 0)],
            vec![(2, Block::Moe), (2, Block::Ple)],
            vec![(0, 1)],
        );
        assert!(!cfg.is_empty());
        assert!(cfg.expert_mask(0).is_none());
        let m = cfg.expert_mask(1).expect("層 1 の mask");
        assert_eq!(m[0] & (1 << 2), 1 << 2);
        assert_eq!(m[0] & (1 << 7), 1 << 7);
        assert_eq!(m[0] & (1 << 3), 0);
        assert!(cfg.block_off(2, Block::Moe));
        assert!(cfg.block_off(2, Block::Ple));
        assert!(!cfg.block_off(2, Block::Attn));
        assert_eq!(cfg.stream_drop_bits(0), 1 << 1);
        let v = View::of(&cfg);
        assert_eq!(v.stream_keep(0, 4), 0b1101);
        assert_eq!(v.stream_keep(1, 4), 0b1111);
        // 空 config は View が none になり、全メソッドが従来動作。
        let empty = build(&s, vec![], vec![], vec![]);
        let v = View::of(&empty);
        assert!(!v.is_active());
        assert!(!v.block_off(2, Block::Moe));
        assert_eq!(v.stream_keep(0, 4), 0b1111);
        assert!(v.expert_mask(1).is_none());
    }

    /// task 217: prefill の常駐連鎖と両立できる設定か。
    /// expert 無効化と routed / shared の skip はホスト側の MoE なので両立する。
    /// attn / ple / ストリームはデバイス側で閉じるので不可。
    #[test]
    fn resident_safe_only_rejects_device_side_blocks() {
        let s = shape();
        let ok = |experts, blocks, streams| {
            build(&s, experts, blocks, streams).resident_safe()
        };
        assert!(ok(vec![], vec![], vec![]), "介入なしが不可になっている");
        assert!(ok(vec![(0, 1), (3, 2)], vec![], vec![]), "expert 無効化が不可");
        assert!(ok(vec![], vec![(1, Block::Moe)], vec![]), "moe skip が不可");
        assert!(ok(vec![], vec![(1, Block::Shared)], vec![]), "shared skip が不可");
        assert!(!ok(vec![], vec![(1, Block::Attn)], vec![]), "attn skip が可になっている");
        assert!(!ok(vec![], vec![(1, Block::Ple)], vec![]), "ple skip が可になっている");
        assert!(!ok(vec![], vec![], vec![(0, 1)]), "ストリーム無効化が可になっている");
        // View 経由も同じ。
        let cfg = build(&s, vec![(0, 1)], vec![], vec![]);
        assert!(View::of(&cfg).resident_safe());
        let none = View::<'_>::none();
        assert!(none.resident_safe());
    }

    #[test]
    fn set_and_clear_move_the_epoch() {
        let _g = GLOBAL_LOCK.lock().unwrap();
        let s = shape();
        // 他のテストと同じグローバルを触るので、値は相対で見る。
        let e0 = current_epoch();
        let c1 = set_config(&s, vec![(0, 1)], vec![], vec![]).unwrap();
        assert_eq!(c1.epoch(), e0 + 1);
        assert_eq!(current_epoch(), e0 + 1);
        let c2 = clear(&s);
        assert_eq!(c2.epoch(), e0 + 2);
        assert!(c2.is_empty());
        // 空に戻したので snapshot は「介入なし」の View になる。
        let snap = snapshot();
        let v = View::of(&snap);
        assert!(!v.is_active());
    }

    #[test]
    fn pin_freezes_the_epoch_until_drop() {
        let _g = GLOBAL_LOCK.lock().unwrap();
        let s = shape();
        let _ = clear(&s);
        let pin = pin_request();
        let e = current_epoch();
        let c = set_config(&s, vec![(0, 1)], vec![], vec![]).unwrap();
        assert_ne!(c.epoch(), e);
        // ピン中は古い設定が見える。
        assert_eq!(snapshot().epoch(), e);
        assert!(snapshot().is_empty());
        drop(pin);
        // 外したら新しい設定。
        assert_eq!(snapshot().epoch(), c.epoch());
        assert!(!snapshot().is_empty());
        let _ = clear(&s);
    }
}
