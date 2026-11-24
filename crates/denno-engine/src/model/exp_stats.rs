//! task 146: (層, expert) ごとの使用回数と、起動時の「よく使う expert の事前ロード」。
//!
//! # なぜ要るか
//!
//! 再起動直後の VRAM は空なので、最初のトークンは全 expert がミスになる。ログに残した
//! 選択をオフライン再生した見積り (task 146 の前提) では、**別トピックの記録から
//! 上位 expert を先に載せる**だけで最初のトークンのヒット率が 0 → 0.70 (14 GB) /
//! 0.82 (26 GB) になり、最初のリクエストの所要も 0.85 → 0.90〜0.96 倍になる。
//! 同じ再生で **使用回数による追い出しは現行 LRU より悪い** (0.78 対 0.91) ので、
//! 回数は「何を事前に載せるか」だけに使い、追い出しは変えない。
//!
//! # 置き場所
//!
//! - 数える: decode の router が決めた `sel` を層ごとに 1 回転写する
//!   ([`observe_layer`])。1 トークン = 48 層 x 10 本 = 480 回の `fetch_add`
//!   (Relaxed)。ロックなし・確保なし。
//! - 保存: [`save_now`] が **decode の外** (別スレッド) で書く。書くのは
//!   `一時ファイル → rename` の 1 手順 ([`save_atomic`])。
//! - 読む: 起動時に [`load_file`] で読み、形式・版・モデル同一性を確かめる。
//!   1 つでも違えば `Err` を返し、呼び出し側は事前ロードをしない (fail closed)。
//!
//! # 数の扱い (減衰と上限)
//!
//! 1 回のセッションが永久に履歴を支配しないように、2 段で減衰させる。
//!
//! 1. **起動のたびに過去ぶんを 1/2** にする ([`install_prior`]、[`DECAY_SHIFT`])。
//!    順位は変わらないが、今回のセッションの重みが相対的に上がる。再起動を
//!    重ねると 2^-N で古い履歴は消える。
//! 2. 保存の値が [`COUNT_CAP`] の半分に達したら **全体を 1/2** にする
//!    ([`merge`])。セル単位の飽和は上位を同値に潰して順位を壊すので使わない。
//!    比を保ったまま全体を割ることで、長いセッションでも直近の使用が順位に効く。
//!
//! 保存は**新しい回数があるときだけ**、リクエスト終端と 20 秒ごとに行う
//! (どちらも decode の外)。
//!
//! # ファイル形式 (版 1、リトルエンディアン)
//!
//! ```text
//! 0   magic[8] = "DNNOEXPS"
//! 8   version   u32 (= 1)
//! 12  n_layer   u32
//! 16  n_expert  u32
//! 20  dir_len   u32
//! 24  dir       dir_len バイト (モデルディレクトリ名。UTF-8)
//! +   expert_bytes u64 (全層の expert 3 行列の合計バイト数 = モデルの同一性)
//! +   cells     n_layer * n_expert 個の u32 (row-major: [layer * n_expert + expert])
//! ```
//!
//! `dir` と `expert_bytes` の組をモデルの同一性とする (同じ名前で量子型や分割が
//! 変わったファイルを取り違えない)。
//!
//! # ラボ実測 (task 146, swift15-abl IQ3_S, 14 T, 8098 停止, t146c)
//!
//! `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=20000 DENNO_EXPERT_GPU_MIN_HIT=0
//! DENNO_GPU_WEIGHT_MB=12000`、SCENARIO=depth8k、commit 3bc2baef8。
//! 記録は t146b (depth: a / b / c2000) が作ったもの。
//!
//! | リクエスト | base hit | pre hit | base ms/tok | pre ms/tok |
//! | --- | --- | --- | --- | --- |
//! | 起動後 1 本目 (28 tok) | 0.514 | **0.918** | 95 | **67** |
//! | a1 (28) | 0.915 | 0.982 | 62 | 49 |
//! | b (33) | 0.895 | 0.951 | 69 | 60 |
//! | c2000_turn2 | 0.922 | 0.924 | 58 | 58 |
//! | c8000_cold (記録に無い) | 0.966 | 0.966 | 59 | 59 |
//!
//! 事前ロードは 5,897 本 / 19.45 GB を wall 23.5 秒 (うち 20.2 秒はリクエスト中に
//! 譲った) で載せた。起動の `load ok` は 2.5 -> 2.8 秒 (+0.3 秒)。
//! 記録に無い 14k プロンプトには効かない (その prefill 自体が作業集合を積む)。
//!
//! # task 798: 明示の warm プロファイルと起動時の本数・バイト上限
//!
//! 既存の経路 (使用回数 → 起動プラン) はそのままに、運用側から 3 つだけ
//! 口を出す。どれも未設定なら従来と 1 バイトも変わらない (既定 off)。
//!
//! - `DENNO_EXPERT_WARM_FILE` ([`WARM_FILE_ENV`]): warm プロファイルの明示パス。
//!   `DENNO_EXPERT_IQ_STATS_FILE` / `DENNO_EXPERT_STATS_DIR` より優先する
//!   (IQ ゲートも要らない)。別ホストで作った履歴を配る用。
//! - `DENNO_START_WARM_EXPERTS` ([`WARM_EXPERTS_ENV`]): 起動時に **page cache** を
//!   温める expert の本数 (熱い側から N 本、`0`/未設定で何もしない。既存の
//!   起動プラン (VRAM → RAM) はどちらでもそのまま動く)。
//!   RAM tier / VRAM には積まない。UD-Q4_K_XL で 4,000 本 (11.98 GB) を RAM tier に
//!   写したら page cache を追い出して decode が 12.0 -> 19.5 ms/tok (+63%) になった
//!   (lab t798b)。pread で page を載せるだけなら RSS は増えない。
//! - `DENNO_START_WARM_MB` ([`WARM_MB_ENV`]): その warm で読むバイト上限 (MiB)。
//!   RAM 59 GiB の機械で expert が 100 GB 級のときの読み過ぎを止める。
//!
//! 本数・バイトの上限はどちらも**起動時の 1 パスだけ**に効く (走行中の
//! 再調整 `rebalance_once` は使用回数に追従したまま)。
//!
//! # task 798: 指紋 (モデルの同一性の 3 つ目)
//!
//! 版 1 の同一性は (ディレクトリ名, expert 合計バイト) で、量子型の取り違えは
//! 防げるが「同じサイズの別リビジョン」は防げない。版 2 はテンソル目録
//! (名前・量子型・次元) の FNV-1a 64 を `fingerprint` として持ち、読むときに
//! 一致を要求する。指紋 0 のファイル (版 1) は従来どおり (名前, バイト) だけで
//! 判定する (古い履歴を捨てない)。**一度も warm していないモデルで
//! プロファイルを読むことはない** (版・形・同一性のいずれかが違えば fail closed)。
//!
//! # 既定
//!
//! `DENNO_EXPERT_IQ_STATS_FILE` か `DENNO_EXPERT_STATS_DIR` が未設定なら、
//! この機能は丸ごと off ([`enabled`] が false = 数える分岐 1 個も通らない)。
//!
//! # task 263: prefill も数える / モデルごとのファイル / 起動プラン
//!
//! - **prefill の選択も数える** ([`observe_batch`])。重みは **1/8**
//!   ([`PREFILL_STRIDE`] = 8 トークンに 1 回)。理由: 1 本の長いプロンプト
//!   (14k トークン) を重み 1 で数えると 14,000 トークン分になり、何時間もの
//!   decode 履歴を一撃で塗り潰す。1/8 なら 1,750 トークン分 = 120 トークンの
//!   decode 約 15 本ぶんで、「長いプロンプトでよく使う expert」は残しつつ
//!   decode の履歴が主役のままになる。1/8 は「8 トークンに 1 回、その
//!   トークンの選択を数える」で実装する (整数カウンタのまま、割り算なし)。
//! - **モデルごとのファイル** (`DENNO_EXPERT_STATS_DIR`)。名前は
//!   (ディレクトリ名, expert 合計バイト) から作る ([`stats_file_name`])。
//!   モデルを切り替えても数え間違えないし、別モデルのファイルを読んで
//!   「別モデルの記録」で捨てることもない。`DENNO_EXPERT_IQ_STATS_FILE` の
//!   明示パスは従来どおり最優先 (既存の運用手順をそのまま使える)。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// 使用回数のファイルパス。未設定なら `DENNO_EXPERT_STATS_DIR` を見る。
pub const STATS_FILE_ENV: &str = "DENNO_EXPERT_IQ_STATS_FILE";

/// task 263: モデルごとのファイルを置くディレクトリ。
///
/// ファイル名はモデルの同一性 (ディレクトリ名 + expert 合計バイト) から
/// 決まる ([`stats_file_name`]) ので、モデルを切り替えても混ざらない。
/// 明示パス (`DENNO_EXPERT_IQ_STATS_FILE`) があればそちらが優先。
pub const STATS_DIR_ENV: &str = "DENNO_EXPERT_STATS_DIR";

/// task 263: prefill の選択を数える間隔 (トークン)。重み = 1/8。
///
/// 8 トークンに 1 回、そのトークンの top-k を 1 回ずつ数える。理由は
/// モジュールの doc。1 にすると長いプロンプトが decode の履歴を塗り潰し、
/// 大きくすると 1 リクエストのプロンプトでしか使わない expert が拾えない。
pub const PREFILL_STRIDE: usize = 8;

/// 事前ロードで予算から空けておく MiB (`DENNO_EXPERT_VRAM_MB` の内数)。
///
/// 事前ロードは起動直後に走るので、dense 重みの upload・KV・prefill の
/// staging が後から来る。LRU を上限まで埋めると 14k prefill で OOM した
/// (t147a の 24,000 MiB)。t147b が「20,000 で 2 GB の余裕」を推奨しているのに
/// 合わせ、既定は 2,000 MiB。
pub const PRELOAD_RESERVE_MB_ENV: &str = "DENNO_EXPERT_IQ_PRELOAD_RESERVE_MB";

/// task 798: warm プロファイルの明示パス (最優先)。
///
/// `DENNO_EXPERT_IQ_STATS_FILE` より先に見る。IQ ゲートも要らない
/// (RAM tier だけの構成や、別ホストで作った履歴を配る用途)。
pub const WARM_FILE_ENV: &str = "DENNO_EXPERT_WARM_FILE";

/// task 798: 起動時に page cache を温める expert の本数 (`DENNO_START_WARM_EXPERTS`)。
///
/// 使用回数の多い側から N 本。`0` と未設定は warm しない (既存の起動プラン
/// = VRAM tier → RAM tier はそのまま動く。この env は追加の 1 手だけ)。
/// RAM tier / VRAM には積まない (lab t798b で RAM tier 版は +63% だった)。
pub const WARM_EXPERTS_ENV: &str = "DENNO_START_WARM_EXPERTS";

/// task 798: 起動時の warm で読むバイト上限 (`DENNO_START_WARM_MB`、MiB)。
///
/// 未設定は無制限 (熱い側から N 本ぶん全部)。
pub const WARM_MB_ENV: &str = "DENNO_START_WARM_MB";

/// [`WARM_EXPERTS_ENV`] の上限 (これを超える値は丸める)。
///
/// UD-Q4_K_XL でも 48 層 x 512 本 = 24,576 本なので、1 M 本あれば実質無制限。
/// 桁の打ち間違い (10^9 など) で確保を試みないための柵。
pub const MAX_WARM_EXPERTS: usize = 1 << 20;

/// [`WARM_MB_ENV`] の上限 (MiB)。1 TiB。桁の打ち間違い用の柵。
pub const MAX_WARM_MB: u64 = 1 << 20;

/// [`PRELOAD_RESERVE_MB_ENV`] の既定値。
pub const DEFAULT_PRELOAD_RESERVE_MB: usize = 2000;

/// 形式の版。読む側は一致しなければ事前ロードしない。
///
/// 版 2 は `expert_bytes` の後ろに `fingerprint: u64` を持つ (task 798)。
pub const FORMAT_VERSION: u32 = 2;

/// 版 1 (指紋なし) のファイルも読む (古い履歴を捨てない)。
pub const LEGACY_FORMAT_VERSION: u32 = 1;

/// ファイル先頭の識別子 (8 バイト)。
pub const MAGIC: [u8; 8] = *b"DNNOEXPS";

/// 1 セルの上限 (飽和)。1 セッションで 1 本が来る回数より十分大きい。
pub const COUNT_CAP: u32 = 1 << 20;

/// 保存のたびに過去ぶんを右シフトする量 (1 = 1/2)。
pub const DECAY_SHIFT: u32 = 1;

/// 保存の既定間隔 (秒)。
pub const SAVE_PERIOD_SECS: u64 = 20;

/// `DENNO_EXPERT_IQ_STATS_FILE` の解釈 (純関数)。空白だけ・空は未設定と同じ。
pub fn stats_file_from_value(v: Option<&str>) -> Option<PathBuf> {
    let s = v.map(str::trim).filter(|s| !s.is_empty())?;
    Some(PathBuf::from(s))
}

/// [`PRELOAD_RESERVE_MB_ENV`] の解釈 (純関数)。未設定・空・数値でない値は既定。
pub fn reserve_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => match s.parse::<usize>() {
            Ok(n) => n,
            Err(_) => DEFAULT_PRELOAD_RESERVE_MB,
        },
        _ => DEFAULT_PRELOAD_RESERVE_MB,
    }
}

/// 事前ロードで予算から空けておくバイト数 (プロセスで 1 回だけ env を読む)。
pub fn preload_reserve_bytes() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        reserve_mb_from_value(std::env::var(PRELOAD_RESERVE_MB_ENV).ok().as_deref())
            .saturating_mul(1024 * 1024)
    })
}

/// task 798: [`WARM_EXPERTS_ENV`] の解釈 (純関数)。
///
/// 未設定・空・非数は `None` (上限なし = 従来)。`0` は `Some(0)` (起動時に
/// 載せない)。[`MAX_WARM_EXPERTS`] を超える値は丸める。
pub fn warm_experts_from_value(v: Option<&str>) -> Option<usize> {
    let s = v.map(str::trim).filter(|s| !s.is_empty())?;
    let n = s.parse::<u64>().ok()?;
    Some(n.min(MAX_WARM_EXPERTS as u64) as usize)
}

/// task 798: [`WARM_MB_ENV`] の解釈 (純関数)。
///
/// 未設定・空・非数は `None` (上限なし)。`0` は `Some(0)` (RAM tier へ
/// 1 本も入れない)。[`MAX_WARM_MB`] を超える値は丸める。
pub fn warm_mb_from_value(v: Option<&str>) -> Option<u64> {
    let s = v.map(str::trim).filter(|s| !s.is_empty())?;
    let n = s.parse::<u64>().ok()?;
    Some(n.min(MAX_WARM_MB))
}

/// 起動時に載せる本数の上限 (env を 1 回だけ読む)。
pub fn warm_experts_cap() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| warm_experts_from_value(std::env::var(WARM_EXPERTS_ENV).ok().as_deref()))
}

/// 起動時に RAM tier へ入れるバイト上限 (env を 1 回だけ読む)。
pub fn warm_mb_bytes() -> Option<u64> {
    static V: OnceLock<Option<u64>> = OnceLock::new();
    *V.get_or_init(|| {
        warm_mb_from_value(std::env::var(WARM_MB_ENV).ok().as_deref())
            .map(|mb| mb.saturating_mul(1024 * 1024))
    })
}

/// 有効条件 (純関数): IQ ゲートが on かつ記録先が設定されていること。
///
/// どちらか欠けたら `None` = この機能は丸ごと off (数える分岐も通らない)。
pub fn gate(iq_gpu: bool, file: Option<&str>) -> Option<PathBuf> {
    if !iq_gpu {
        return None;
    }
    stats_file_from_value(file)
}

/// task 263: モデルの同一性からファイル名を作る (`dir` は正規化する)。
///
/// `dir` はモデルディレクトリの**末尾の名前**を想定する (パスでも動くが、
/// 区切りは `_` になる)。英数字・`.`・`-`・`_` 以外は `_` にし、長さは
/// 80 バイトで切る (ext4 の 255 バイト制限に収める)。`expert_bytes` を
/// 付けるので、同じ名前で量子型が変わっても別ファイルになる。
pub fn stats_file_name(dir: &str, expert_bytes: u64) -> String {
    let mut key = String::with_capacity(dir.len() + 24);
    for ch in dir.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            key.push(ch);
        } else {
            key.push('_');
        }
    }
    let key = key.trim_matches('_');
    let key: String = key.chars().take(80).collect();
    if key.is_empty() {
        format!("dnnoexps-model-{expert_bytes}.bin")
    } else {
        format!("dnnoexps-{key}-{expert_bytes}.bin")
    }
}

/// task 263/798: 記録先を決める (純関数)。task 798 で warm の明示パスが最優先。
///
/// 1. `DENNO_EXPERT_WARM_FILE` (明示パス) があればそれ。IQ ゲートも要らない。
/// 2. `DENNO_EXPERT_IQ_STATS_FILE` (明示パス) があればそれ。ただし従来どおり
///    `DENNO_EXPERT_IQ_GPU=1` が前提。
/// 3. 無ければ `DENNO_EXPERT_STATS_DIR` + [`stats_file_name`]。こちらは
///    数える・保存するだけなので IQ ゲートを要らない (RAM tier だけ使う
///    構成でも履歴を貯められる)。
/// 4. どれも無ければ `None` (機能 off)。
pub fn resolve_stats_path(
    warm: Option<&str>,
    file: Option<&str>,
    dir: Option<&str>,
    model_dir: &str,
    expert_bytes: u64,
    iq_gpu: bool,
) -> Option<PathBuf> {
    if let Some(p) = stats_file_from_value(warm) {
        return Some(p);
    }
    if let Some(p) = gate(iq_gpu, file) {
        return Some(p);
    }
    let d = dir.map(str::trim).filter(|s| !s.is_empty())?;
    Some(Path::new(d).join(stats_file_name(model_dir, expert_bytes)))
}

/// 記録先 (未解決なら `None`)。起動時に [`set_identity`] が決める。
pub fn stats_path() -> Option<PathBuf> {
    PATH.get()
        .and_then(|l| l.read().unwrap_or_else(|e| e.into_inner()).clone())
}

static PATH: OnceLock<std::sync::RwLock<Option<PathBuf>>> = OnceLock::new();
/// 記録先が決まっているか (ホットパスはこれ 1 個の atomic load で返る)。
static ENABLED: AtomicBool = AtomicBool::new(false);

fn path_slot() -> &'static std::sync::RwLock<Option<PathBuf>> {
    PATH.get_or_init(|| std::sync::RwLock::new(None))
}

/// 記録先を決めて有効化する (env を 1 回だけ読む)。戻り値は決まったパス。
///
/// モデルの同一性が分かった時点 (起動時、モデル構築の前) で呼ぶ。
/// 何度呼んでも最後の値が勝つ (モデルの切り替えはプロセス再起動なので
/// 実際には 1 回)。
pub fn configure(model_dir: &str, expert_bytes: u64) -> Option<PathBuf> {
    let warm = std::env::var(WARM_FILE_ENV).ok();
    let file = std::env::var(STATS_FILE_ENV).ok();
    let dir = std::env::var(STATS_DIR_ENV).ok();
    let p = resolve_stats_path(
        warm.as_deref(),
        file.as_deref(),
        dir.as_deref(),
        model_dir,
        expert_bytes,
        crate::model::exp_iq::iq_gpu_enabled(),
    );
    *path_slot().write().unwrap_or_else(|e| e.into_inner()) = p.clone();
    ENABLED.store(p.is_some(), Ordering::Relaxed);
    // 置き場所が無ければ作る (失敗しても起動は止めない。保存だけが失敗する)。
    if let Some(dir) = p.as_ref().and_then(|p| p.parent()) {
        if !dir.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(dir);
        }
    }
    p
}

/// 数える・保存する・事前ロードするのいずれかが有効か。
pub fn enabled() -> bool {
    #[cfg(test)]
    if TEST_ON.load(Ordering::Relaxed) {
        return true;
    }
    ENABLED.load(Ordering::Relaxed)
}

/// テスト専用: ゲートを強制 on にする (戻り値のガードを落とすと戻る)。
///
/// 発火テスト用。env はプロセスで 1 回しか読まないので、これが無いと
/// 「decode 経路が本当に数えているか」を確かめられない。
#[cfg(test)]
static TEST_ON: AtomicBool = AtomicBool::new(false);

/// [`enable_for_test`] のガード。
#[cfg(test)]
pub struct TestEnable;

#[cfg(test)]
pub fn enable_for_test() -> TestEnable {
    TEST_ON.store(true, Ordering::Relaxed);
    TestEnable
}

#[cfg(test)]
impl Drop for TestEnable {
    fn drop(&mut self) {
        TEST_ON.store(false, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// 数える (decode のホットパス)
// ---------------------------------------------------------------------------

/// プロセス内のカウンタ。`n_layer * n_expert` 個の `AtomicU32` を 1 本で持つ。
///
/// ロックを持たない。decode は 1 スレッドなので競合しない
/// (prefill の層並列から呼ばれても、触るセルが違うので高々キャッシュラインの
/// 共有だけ)。
pub struct Cells {
    n_layer: usize,
    n_expert: usize,
    v: Vec<AtomicU32>,
}

impl Cells {
    /// 0 で埋めたカウンタ。
    pub fn new(n_layer: usize, n_expert: usize) -> Self {
        Self {
            n_layer,
            n_expert,
            v: (0..n_layer.saturating_mul(n_expert))
                .map(|_| AtomicU32::new(0))
                .collect(),
        }
    }

    /// 層 `layer` で選ばれた expert を 1 回ずつ数える。範囲外は黙って捨てる。
    pub fn observe(&self, layer: usize, sel: &[usize]) {
        if layer >= self.n_layer {
            return;
        }
        let base = layer * self.n_expert;
        for &e in sel {
            if e < self.n_expert {
                self.v[base + e].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// task 263: prefill の 1 層ぶん (`nt` トークン x `k` 本の平らな配列) を
    /// [`PREFILL_STRIDE`] 本おきに数える。
    pub fn observe_batch(&self, layer: usize, sels: &[usize], k: usize, nt: usize) {
        if k == 0 {
            return;
        }
        for t in sampled_tokens(nt) {
            let row = &sels[t * k..(t * k + k).min(sels.len())];
            self.observe(layer, row);
        }
    }

    /// セルの生値 (`[layer * n_expert + expert]` の順)。
    pub fn snapshot(&self) -> Vec<u32> {
        self.v.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// 0 に戻す。
    pub fn reset(&self) {
        for c in &self.v {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// `(n_layer, n_expert)`。
    pub fn dims(&self) -> (usize, usize) {
        (self.n_layer, self.n_expert)
    }
}

static SESSION: OnceLock<Cells> = OnceLock::new();

/// このセッションのカウンタ (まだ無ければ `None`)。
pub fn session() -> Option<&'static Cells> {
    SESSION.get()
}

/// 前回の保存から数えた回数があるか (書き直しを避けるため)。
static DIRTY: AtomicBool = AtomicBool::new(false);

/// まだ保存していない回数があるか。
pub fn dirty() -> bool {
    DIRTY.load(Ordering::Relaxed)
}

/// decode の router が決めた `sel` を層ぶん数える。
///
/// ゲートが off なら atomic load 1 回で返る。on のときの
/// コストは `sel.len()` 回の `fetch_add` (Relaxed) だけで、ロックも確保も無い。
pub fn observe_layer(n_layer: usize, n_expert: usize, layer: usize, sel: &[usize]) {
    if !enabled() {
        return;
    }
    if n_layer == 0 || n_expert == 0 {
        return;
    }
    let cells = SESSION.get_or_init(|| Cells::new(n_layer, n_expert));
    cells.observe(layer, sel);
    DIRTY.store(true, Ordering::Relaxed);
}

/// task 263: prefill の 1 層ぶんの選択 (`nt` トークン x `k` 本の平らな配列) を
/// [`PREFILL_STRIDE`] 本おきに数える (重み 1/8)。
///
/// ホットパスの制約は decode と同じ: ロックなし・確保なし。`enabled()` が
/// off なら atomic load 1 回で返る。on のときのコストは
/// `(nt / 8) * k` 回の `fetch_add` (Relaxed)。1 トークンの decode が `k`
/// 回なので、512 トークンのチャンク 1 個 = decode 64 トークンぶん。
///
/// 数えるトークンは `t % PREFILL_STRIDE == 0` のもの (チャンクの先頭から
/// 8 本おき)。チャンクの割り方で標本が変わるが、順位に効くのは
/// 「プロンプト全体で繰り返し選ばれる expert」なので実用上は同じ。
pub fn observe_batch(
    n_layer: usize,
    n_expert: usize,
    layer: usize,
    sels: &[usize],
    k: usize,
    nt: usize,
) {
    if !enabled() {
        return;
    }
    if n_layer == 0 || n_expert == 0 || k == 0 || nt == 0 {
        return;
    }
    let cells = SESSION.get_or_init(|| Cells::new(n_layer, n_expert));
    cells.observe_batch(layer, sels, k, nt);
    DIRTY.store(true, Ordering::Relaxed);
}

/// [`observe_batch`] が数えるトークンの位置 (`PREFILL_STRIDE` 本おき)。
fn sampled_tokens(nt: usize) -> impl Iterator<Item = usize> {
    (0..nt).step_by(PREFILL_STRIDE)
}

// ---------------------------------------------------------------------------
// ファイル形式
// ---------------------------------------------------------------------------

/// 使用回数ファイルの中身。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    /// モデルディレクトリ名 (同一性の 1 つ目)
    pub dir: String,
    /// 全層の expert 3 行列の合計バイト数 (同一性の 2 つ目)
    pub expert_bytes: u64,
    /// task 798: テンソル目録の指紋 (同一性の 3 つ目)。版 1 のファイルは 0。
    pub fingerprint: u64,
    pub n_layer: u32,
    pub n_expert: u32,
    /// `[layer * n_expert + expert]` の回数
    pub cells: Vec<u32>,
}

impl Counts {
    /// 0 で埋めたカウンタ (ファイルが無いときの代わり)。指紋は 0。
    pub fn zeros(dir: &str, expert_bytes: u64, n_layer: u32, n_expert: u32) -> Self {
        Self {
            dir: dir.to_string(),
            expert_bytes,
            fingerprint: 0,
            n_layer,
            n_expert,
            cells: vec![0; n_layer as usize * n_expert as usize],
        }
    }

    /// task 798: 指紋つきの 0 カウンタ (保存で指紋を落とさないため)。
    pub fn zeros_with_fingerprint(
        dir: &str,
        expert_bytes: u64,
        fingerprint: u64,
        n_layer: u32,
        n_expert: u32,
    ) -> Self {
        let mut c = Self::zeros(dir, expert_bytes, n_layer, n_expert);
        c.fingerprint = fingerprint;
        c
    }

    /// 回数 `> 0` のセルを `(層, expert)` で引く。範囲外は `None`。
    pub fn get(&self, layer: usize, expert: usize) -> Option<u32> {
        if layer >= self.n_layer as usize || expert >= self.n_expert as usize {
            return None;
        }
        self.cells
            .get(layer * self.n_expert as usize + expert)
            .copied()
    }

    /// 非 0 のセル数。
    pub fn nonzero(&self) -> usize {
        self.cells.iter().filter(|c| **c > 0).count()
    }
}

/// ファイルへ書くバイト列 (版 2、指紋つき)。
pub fn encode(c: &Counts) -> Vec<u8> {
    let dir = c.dir.as_bytes();
    let mut out = Vec::with_capacity(32 + dir.len() + 8 + c.cells.len() * 4);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&c.n_layer.to_le_bytes());
    out.extend_from_slice(&c.n_expert.to_le_bytes());
    out.extend_from_slice(&(dir.len() as u32).to_le_bytes());
    out.extend_from_slice(dir);
    out.extend_from_slice(&c.expert_bytes.to_le_bytes());
    out.extend_from_slice(&c.fingerprint.to_le_bytes());
    for v in &c.cells {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn take_u32(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn take_u64(b: &[u8], at: usize) -> Option<u64> {
    let s = b.get(at..at + 8)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// バイト列を読む。形式・版・長さのどれかが違えば 1 行の理由を返す。
///
/// 版 2 は `expert_bytes` の後ろに `fingerprint: u64` を持つ。版 1 は指紋が
/// 無いので 0 として読む (同一性の判定は従来どおり名前 + バイトだけで行う)。
pub fn decode(b: &[u8]) -> Result<Counts, String> {
    if b.len() < 24 {
        return Err(format!("短すぎる ({} バイト < ヘッダ 24)", b.len()));
    }
    if b[..8] != MAGIC {
        return Err("識別子が違う (このファイルは expert 使用回数ではない)".to_string());
    }
    let version = take_u32(b, 8).ok_or("版が読めない")?;
    if version != FORMAT_VERSION && version != LEGACY_FORMAT_VERSION {
        return Err(format!(
            "版が違う ({version} != {FORMAT_VERSION}、受理する旧版は {LEGACY_FORMAT_VERSION})"
        ));
    }
    let n_layer = take_u32(b, 12).ok_or("n_layer が読めない")?;
    let n_expert = take_u32(b, 16).ok_or("n_expert が読めない")?;
    let dir_len = take_u32(b, 20).ok_or("dir_len が読めない")? as usize;
    let dir_end = 24usize
        .checked_add(dir_len)
        .filter(|e| *e <= b.len())
        .ok_or("ディレクトリ名の長さがファイル長を超える")?;
    let dir = String::from_utf8_lossy(&b[24..dir_end]).to_string();
    let expert_bytes = take_u64(b, dir_end).ok_or("expert_bytes が読めない")?;
    let (fingerprint, body) = if version >= 2 {
        let fp = take_u64(b, dir_end + 8).ok_or("指紋が読めない")?;
        (fp, dir_end + 16)
    } else {
        (0u64, dir_end + 8)
    };
    let n_cells = (n_layer as usize)
        .checked_mul(n_expert as usize)
        .ok_or("セル数が溢れる")?;
    let want = body
        .checked_add(n_cells.saturating_mul(4))
        .ok_or("ファイル長が溢れる")?;
    if b.len() != want {
        return Err(format!(
            "長さが違う ({} バイト、期待 {want} = {n_layer}x{n_expert} セル)",
            b.len()
        ));
    }
    let mut cells = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        cells.push(take_u32(b, body + i * 4).ok_or("セルが読めない")?);
    }
    Ok(Counts {
        dir,
        expert_bytes,
        fingerprint,
        n_layer,
        n_expert,
        cells,
    })
}

/// 記録したモデルと目の前のモデルが同じか (名前と expert 合計バイト)。
pub fn identity_matches(c: &Counts, dir: &str, expert_bytes: u64) -> bool {
    c.dir == dir && c.expert_bytes == expert_bytes
}

/// task 798: 指紋が一致するか。版 1 のファイル (指紋 0) は判定しない。
pub fn fingerprint_matches(c: &Counts, fingerprint: u64) -> bool {
    c.fingerprint == 0 || c.fingerprint == fingerprint
}

/// task 798: 名前・バイト・指紋の 3 つで同一性を見る。
pub fn identity_ok(c: &Counts, dir: &str, expert_bytes: u64, fingerprint: u64) -> bool {
    identity_matches(c, dir, expert_bytes) && fingerprint_matches(c, fingerprint)
}

/// ファイルを読んで同一性を確かめる (task 798 で指紋も見る)。
///
/// 失敗はすべて `Err` (呼び出し側は事前ロードをしない)。理由は 1 行。
pub fn load_file(
    path: &Path,
    dir: &str,
    expert_bytes: u64,
    fingerprint: u64,
) -> Result<Counts, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{} が読めない: {e}", path.display()))?;
    let c = decode(&bytes)?;
    if !identity_matches(&c, dir, expert_bytes) {
        return Err(format!(
            "別のモデルの記録 (記録 {} / {:.1} GB, 今回 {dir} / {:.1} GB)",
            c.dir,
            c.expert_bytes as f64 / 1e9,
            expert_bytes as f64 / 1e9
        ));
    }
    if !fingerprint_matches(&c, fingerprint) {
        return Err(format!(
            "別のリビジョンの記録 (指紋 {:016x} / 今回 {:016x})",
            c.fingerprint, fingerprint
        ));
    }
    Ok(c)
}

/// 一時ファイルへ書いて rename する (途中の状態を読ませない)。
///
/// 失敗しても呼び出し側は落ちない (リクエストにも起動にも影響させない)。
pub fn save_atomic(path: &Path, c: &Counts) -> std::io::Result<()> {
    let bytes = encode(c);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        // 電源断で「長さ 0 のファイル」を残さない (rename の前後に落ちた場合の保険)。
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// 保存する値: `prior + session`。上限に近づいたら**全体を 1/2** にする。
///
/// セル単位で飽和させると上位が同値に潰れて順位が壊れるので、全体を同じ率で
/// 割って比を保つ (1 セルが [`COUNT_CAP`] の半分に達したら 1 回だけ半減)。
/// これが「長いセッションが永久に履歴を支配しない」ための減衰で、直近の使用が
/// 相対的に重くなる。加えて起動のたびに過去を 1/2 にする ([`install_prior`])。
pub fn merge(prior: &[u32], session: &[u32]) -> Vec<u32> {
    let n = prior.len().max(session.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = prior.get(i).copied().unwrap_or(0);
        let s = session.get(i).copied().unwrap_or(0);
        out.push(p.saturating_add(s).min(COUNT_CAP));
    }
    if out.iter().any(|v| *v >= COUNT_CAP / 2) {
        for v in out.iter_mut() {
            *v /= 2;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 事前ロードの順序 (純関数)
// ---------------------------------------------------------------------------

/// 事前ロードする順 (回数の少ない順 = 最後に載るものが一番よく使われる)。
///
/// 同数は `(層, expert)` 昇順 (決定的)。回数 0 のセルは含めない。
pub fn preload_order(c: &Counts) -> Vec<(u16, u16)> {
    let mut v: Vec<(u32, u16, u16)> = Vec::new();
    for layer in 0..c.n_layer as usize {
        for expert in 0..c.n_expert as usize {
            let n = c.cells[layer * c.n_expert as usize + expert];
            if n > 0 && layer <= u16::MAX as usize && expert <= u16::MAX as usize {
                v.push((n, layer as u16, expert as u16));
            }
        }
    }
    v.sort_unstable();
    v.into_iter().map(|(_, l, e)| (l, e)).collect()
}

/// 事前ロードを止めるか (`used + want` が目標を超える)。純関数。
///
/// 足し算が溢れるなら「確実に超えている」ので止める。
pub fn budget_full(used: usize, want: usize, target: usize) -> bool {
    used.checked_add(want).map_or(true, |s| s > target)
}

/// 1 本を載せてよいか ([`budget_full`] の裏返し)。
pub fn fits(used: u64, want: u64, target: u64) -> bool {
    !budget_full(used as usize, want as usize, target as usize)
}

// ---------------------------------------------------------------------------
// task 798: テンソル目録の指紋 (モデルの同一性の 3 つ目)
// ---------------------------------------------------------------------------

/// FNV-1a 64 の初期値。
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 の素数。
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_bytes(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn fnv_u64(mut h: u64, v: u64) -> u64 {
    for x in v.to_le_bytes() {
        h ^= x as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// テンソル目録 `(名前, 量子型名, 次元)` の指紋 (純関数)。
///
/// **順序に依存する** (GGUF のファイル内順 = [`crate::gguf::GgufFile::tensor_order`])。
/// 名前・型・次元のどれかが 1 つでも変われば別の値になる。バイト列 (= 重みの値)
/// は見ない: 同じ目録で重みだけ差し替えたリビジョンは区別できないが、量子型の
/// 取り違えとテンソルの増減・改名・次元変更は捕まえられる。
pub fn inventory_fingerprint<'a, I>(items: I) -> u64
where
    I: IntoIterator<Item = (&'a str, &'a str, Vec<u64>)>,
{
    let mut h = FNV_OFFSET;
    for (name, ty, ne) in items {
        h = fnv_bytes(h, name.as_bytes());
        h = fnv_bytes(h, ty.as_bytes());
        h = fnv_u64(h, ne.len() as u64);
        for d in ne {
            h = fnv_u64(h, d);
        }
    }
    h
}

/// GGUF のテンソル目録から指紋を作る (起動時に 1 回)。
pub fn fingerprint_tensors(f: &crate::gguf::GgufFile) -> u64 {
    let items: Vec<(&str, &str, Vec<u64>)> = f
        .tensor_order
        .iter()
        .map(|name| match f.tensors.get(name) {
            Some(t) => (name.as_str(), t.ty.name(), t.dims.to_vec()),
            None => (name.as_str(), "?", Vec::new()),
        })
        .collect();
    inventory_fingerprint(items)
}

/// task 263: plan (回数の昇順) を VRAM / RAM / mmap に割る (純関数)。
///
/// `sizes[i]` は `plan[i]` の 1 本ぶんのバイト数。回数の多い側 (末尾) から
/// 見て、VRAM 予算 (`vram_target`) に入るあいだ `vram`、続けて RAM 予算
/// (`ram_budget`) に入るあいだ `ram`、残りは mmap。
///
/// 戻り値は `(vram_start, ram_start)`:
/// `plan[vram_start..]` = VRAM、`plan[ram_start..vram_start]` = RAM、
/// `plan[..ram_start]` = mmap。1 本も入らないときは両方 `n`。
pub fn split_plan(sizes: &[u64], vram_target: u64, ram_budget: u64) -> (usize, usize) {
    let n = sizes.len();
    let mut used = 0u64;
    let mut i = n;
    while i > 0 {
        let want = sizes[i - 1];
        if !fits(used, want, vram_target) {
            break;
        }
        used = used.saturating_add(want);
        i -= 1;
    }
    let vram_start = i;
    let mut used_ram = 0u64;
    while i > 0 {
        let want = sizes[i - 1];
        if !fits(used_ram, want, ram_budget) {
            break;
        }
        used_ram = used_ram.saturating_add(want);
        i -= 1;
    }
    (vram_start, i)
}

// ---------------------------------------------------------------------------
// 状態 (同一性・履歴・セッション)
// ---------------------------------------------------------------------------

struct SaveState {
    dir: String,
    expert_bytes: u64,
    /// task 798: テンソル目録の指紋 (保存に載せる)
    fingerprint: u64,
    n_layer: u32,
    n_expert: u32,
    /// 前回までに保存した回数 (次回の保存で 1/2 に減衰させる)
    prior: Vec<u32>,
    /// 事前ロード用に読み終えたか (false の間は保存しない)
    ready: bool,
}

static STATE: OnceLock<Mutex<SaveState>> = OnceLock::new();

fn state() -> &'static Mutex<SaveState> {
    STATE.get_or_init(|| {
        Mutex::new(SaveState {
            dir: String::new(),
            expert_bytes: 0,
            fingerprint: 0,
            n_layer: 0,
            n_expert: 0,
            prior: Vec::new(),
            ready: false,
        })
    })
}

/// モデルの同一性と形を覚える (起動時、スレッドを起こす前)。
///
/// 事前ロードをしない条件 (ファイル無し・壊れている・別モデル) でも呼ぶ。
/// 保存だけは続けたいため。task 263: ここで記録先も決める ([`configure`])。
/// task 798: テンソル目録の指紋も覚える (保存ファイルに載る)。
pub fn set_identity(
    dir: &str,
    expert_bytes: u64,
    fingerprint: u64,
    n_layer: usize,
    n_expert: usize,
) -> Option<PathBuf> {
    let path = configure(dir, expert_bytes);
    if let Ok(mut s) = state().lock() {
        s.dir = dir.to_string();
        s.expert_bytes = expert_bytes;
        s.fingerprint = fingerprint;
        s.n_layer = n_layer as u32;
        s.n_expert = n_expert as u32;
    }
    path
}

/// 読み込んだ履歴を「前回まで」として据え、保存を許可する。
///
/// ここで過去ぶんを 1/2 にする (起動 1 回 = 減衰 1 回)。順位は変わらないが、
/// これから積む今回のセッションの重みが相対的に上がる。何度も再起動すると
/// 2^-N で古い履歴は消えていく。
///
/// task 798: 記録の指紋が 0 (版 1 か、空の代用カウンタ) なら
/// [`set_identity`] が入れた指紋を消さない。
pub fn install_prior(c: &Counts) {
    if let Ok(mut s) = state().lock() {
        s.dir = c.dir.clone();
        s.expert_bytes = c.expert_bytes;
        if c.fingerprint != 0 {
            s.fingerprint = c.fingerprint;
        }
        s.n_layer = c.n_layer;
        s.n_expert = c.n_expert;
        s.prior = c.cells.iter().map(|v| v >> DECAY_SHIFT).collect();
        s.ready = true;
    }
}

/// 事前ロードの材料 (据えた履歴) を返す。据えていなければ `None`。
pub fn prior_counts() -> Option<Counts> {
    let s = state().lock().ok()?;
    if !s.ready || s.prior.is_empty() {
        return None;
    }
    Some(Counts {
        dir: s.dir.clone(),
        expert_bytes: s.expert_bytes,
        fingerprint: s.fingerprint,
        n_layer: s.n_layer,
        n_expert: s.n_expert,
        cells: s.prior.clone(),
    })
}

/// いまのセッション回数を `prior/2 + session` にしてファイルへ書く。
///
/// 書けたら `prior` を新しい値に置き換え、セッションを 0 に戻す
/// (次の保存で二重に足さない)。戻り値は 1 行のログ (書かなかったら `None`)。
pub fn save_now() -> Option<String> {
    let path = stats_path()?;
    // 新しい回数が無ければ書き直さない (idle で 20 秒ごとに同じ内容を書かない)。
    if !dirty() {
        return None;
    }
    // まだ 1 度も decode していないなら書かない (履歴だけを半分に減衰させない)。
    let session_cells = session()?;
    let (n_layer, n_expert) = session_cells.dims();
    let mut s = state().lock().ok()?;
    if !s.ready {
        return None;
    }
    let snapshot = session_cells.snapshot();
    let merged = merge(&s.prior, &snapshot);
    let total: u64 = merged.iter().map(|v| *v as u64).sum();
    if total == 0 {
        return None;
    }
    let c = Counts {
        dir: s.dir.clone(),
        expert_bytes: s.expert_bytes,
        fingerprint: s.fingerprint,
        n_layer: n_layer as u32,
        n_expert: n_expert as u32,
        cells: merged.clone(),
    };
    match save_atomic(&path, &c) {
        Ok(()) => {
            session_cells.reset();
            DIRTY.store(false, Ordering::Relaxed);
            s.prior = merged;
            let hits = c.nonzero();
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            *LAST_SAVE.lock().unwrap_or_else(|e| e.into_inner()) = Some((
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                bytes,
            ));
            Some(format!(
                "[exp-stats] saved {} counted={} cells bytes={} (prior>>{} at load + session)",
                path.display(),
                hits,
                bytes,
                DECAY_SHIFT
            ))
        }
        Err(e) => Some(format!("[exp-stats] save failed {}: {e}", path.display())),
    }
}

/// 最後に保存できた時刻 (unix 秒) とバイト数。まだ保存していなければ `None`。
static LAST_SAVE: Mutex<Option<(u64, u64)>> = Mutex::new(None);

/// [`LAST_SAVE`] の読み口 (エンドポイント用)。
pub fn last_save() -> Option<(u64, u64)> {
    *LAST_SAVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// いまの回数 (`prior + session`、減衰なし) のスナップショット。
///
/// 起動プランと /v1/experts/stats が使う。保存 ([`save_now`]) と違って
/// 状態を進めない (prior を置き換えない・セッションを 0 に戻さない)。
pub fn current_counts() -> Option<Counts> {
    let s = state().lock().ok()?;
    if !s.ready {
        return None;
    }
    let (n_layer, n_expert) = (s.n_layer, s.n_expert);
    if n_layer == 0 || n_expert == 0 {
        return None;
    }
    let mut cells = s.prior.clone();
    if let Some(sess) = session() {
        let snap = sess.snapshot();
        if snap.len() == cells.len() {
            for (a, b) in cells.iter_mut().zip(snap.iter()) {
                *a = a.saturating_add(*b);
            }
        }
    }
    Some(Counts {
        dir: s.dir.clone(),
        expert_bytes: s.expert_bytes,
        fingerprint: s.fingerprint,
        n_layer,
        n_expert,
        cells,
    })
}

/// キーの集合が記録した使用回数のうち何割を覆うか (`(覆った回数, 全体)`)。
///
/// 重複キーは 1 回だけ数える。範囲外は無視する。
pub fn coverage(c: &Counts, keys: &[(u16, u16)]) -> (u64, u64) {
    let ne = c.n_expert as usize;
    let mut seen = std::collections::HashSet::with_capacity(keys.len());
    let mut cov = 0u64;
    for &(l, e) in keys {
        let (l, e) = (l as usize, e as usize);
        if l >= c.n_layer as usize || e >= ne {
            continue;
        }
        if !seen.insert((l, e)) {
            continue;
        }
        cov = cov.saturating_add(c.cells[l * ne + e] as u64);
    }
    let total = c.cells.iter().map(|v| *v as u64).sum();
    (cov, total)
}

// ---------------------------------------------------------------------------
// 起動プランの記録 (エンドポイント用)
// ---------------------------------------------------------------------------

/// 起動プラン 1 回ぶんの結果。`/v1/experts/stats` がそのまま返す。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanSnapshot {
    /// プランを立てた時刻 (unix 秒)。
    pub at: u64,
    /// 記録にあった expert の総数 (回数 > 0)。
    pub planned_experts: usize,
    /// VRAM tier に載せた本数 / バイト。
    pub vram_experts: usize,
    pub vram_bytes: u64,
    /// RAM tier に載せた本数 / バイト。
    pub ram_experts: usize,
    pub ram_bytes: u64,
    /// VRAM / RAM / 合計が覆う「記録された使用回数」の割合 (0..1)。
    pub cover_vram: f64,
    pub cover_ram: f64,
    pub cover_total: f64,
    /// プランにかかった秒。
    pub secs: f64,
    /// 再調整で移した本数 (累計)。
    pub promoted: usize,
    pub dropped: usize,
}

static PLAN: Mutex<Option<PlanSnapshot>> = Mutex::new(None);

/// 起動プランの結果を残す。エンドポイントが読む。
pub fn set_plan(p: PlanSnapshot) {
    *PLAN.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
}

/// 残した起動プラン (まだ無ければ `None`)。
pub fn plan() -> Option<PlanSnapshot> {
    PLAN.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// 再調整の累計だけを足す (プランが無ければ何もしない)。
pub fn add_rebalance(promoted: usize, dropped: usize) {
    if let Some(p) = PLAN.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        p.promoted = p.promoted.saturating_add(promoted);
        p.dropped = p.dropped.saturating_add(dropped);
    }
}

// ---------------------------------------------------------------------------
// decode 中かどうか (事前ロードが譲るため)
// ---------------------------------------------------------------------------

static BUSY: AtomicBool = AtomicBool::new(false);

/// リクエスト (prefill / decode) の実行中か。
pub fn busy() -> bool {
    BUSY.load(Ordering::Relaxed)
}

/// 実行中フラグを立てる。戻り値のガードを落とすと下ろす。
pub fn busy_guard() -> BusyGuard {
    BUSY.store(true, Ordering::Relaxed);
    BusyGuard
}

/// [`busy_guard`] のガード。
pub struct BusyGuard;

impl Drop for BusyGuard {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Counts {
        let mut c = Counts::zeros("swift15-abl", 12_345, 2, 3);
        c.cells[0] = 5; // (0,0)
        c.cells[4] = 9; // (1,1)
        c.cells[5] = 1; // (1,2)
        c
    }

    /// 書き出しと読み戻しが一致する (版・次元・同一性・指紋つき)。
    #[test]
    fn encode_decode_roundtrip() {
        let mut c = sample();
        c.fingerprint = 0x1122_3344_5566_7788;
        let b = encode(&c);
        assert_eq!(decode(&b).expect("読める"), c);
        assert_eq!(b.len(), 24 + "swift15-abl".len() + 8 + 8 + 6 * 4);
        // 版は 2
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), FORMAT_VERSION);
    }

    /// task 798: 版 1 のファイル (指紋なし) も読める。指紋は 0 になる。
    #[test]
    fn decode_accepts_legacy_version_one() {
        let c = sample();
        // 版 1 のバイト列を手で組む (指紋フィールドが無い)。
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&LEGACY_FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(&c.n_layer.to_le_bytes());
        b.extend_from_slice(&c.n_expert.to_le_bytes());
        b.extend_from_slice(&(c.dir.len() as u32).to_le_bytes());
        b.extend_from_slice(c.dir.as_bytes());
        b.extend_from_slice(&c.expert_bytes.to_le_bytes());
        for v in &c.cells {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let got = decode(&b).expect("版 1 が読めない");
        assert_eq!(got.dir, c.dir);
        assert_eq!(got.cells, c.cells);
        assert_eq!(got.fingerprint, 0);
        // 指紋 0 の記録はどの指紋でも通す (従来どおり名前 + バイトだけで判定)。
        assert!(fingerprint_matches(&got, 0x1234_5678));
        // 版 3 は拒否 (将来の形式を黙って読まない)。
        let mut bad = b.clone();
        bad[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(decode(&bad).unwrap_err().contains("版"));
    }

    /// task 798: 指紋が違えば別リビジョンとして落ちる。同じなら通る。
    #[test]
    fn fingerprint_rejects_other_revision() {
        let mut c = sample();
        c.fingerprint = 0xdead_beef_0000_0001;
        assert!(fingerprint_matches(&c, 0xdead_beef_0000_0001));
        assert!(!fingerprint_matches(&c, 0xdead_beef_0000_0002));
        assert!(identity_ok(&c, "swift15-abl", 12_345, 0xdead_beef_0000_0001));
        assert!(!identity_ok(&c, "swift15-abl", 12_345, 0xdead_beef_0000_0002));
        // 名前が違えば指紋が同じでも落ちる
        assert!(!identity_ok(&c, "swift15", 12_345, 0xdead_beef_0000_0001));
    }

    /// task 798: 指紋は目録の名前・型・次元に依存する (順序も見る)。
    #[test]
    fn inventory_fingerprint_is_deterministic_and_sensitive() {
        let a = vec![
            ("blk.0.ffn_gate_exps.weight", "Q4_K", vec![2560u64, 512]),
            ("blk.0.ffn_down_exps.weight", "Q5_1", vec![640, 512]),
        ];
        let base = inventory_fingerprint(a.clone());
        assert_eq!(base, inventory_fingerprint(a.clone()));
        assert_ne!(base, FNV_OFFSET);
        // 量子型が 1 つ違う (同じ次元・同じサイズ) → 別の指紋
        let mut b = a.clone();
        b[0].1 = "Q5_K";
        assert_ne!(base, inventory_fingerprint(b));
        // 次元が違う
        let mut c2 = a.clone();
        c2[1].2 = vec![640, 511];
        assert_ne!(base, inventory_fingerprint(c2));
        // 順序が違う
        let mut d = a.clone();
        d.reverse();
        assert_ne!(base, inventory_fingerprint(d));
        // 空の目録は初期値のまま (決定的)
        assert_eq!(inventory_fingerprint(Vec::new()), FNV_OFFSET);
    }

    /// 壊れたファイル・別版・別長はすべて理由つきで落ちる (fail closed)。
    #[test]
    fn decode_rejects_bad_files() {
        let c = sample();
        let b = encode(&c);
        assert!(decode(&[]).is_err());
        assert!(decode(&b[..10]).is_err());
        // 識別子
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(decode(&bad).unwrap_err().contains("識別子"));
        // 版 (受理しない番号)
        let mut bad = b.clone();
        bad[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(decode(&bad).unwrap_err().contains("版"));
        // 長さ (1 セル足りない)
        assert!(decode(&b[..b.len() - 4]).unwrap_err().contains("長さ"));
        // でたらめ
        assert!(decode(&vec![0xAB; 64]).is_err());
    }

    /// 同一性はディレクトリ名と expert 合計バイトの両方を見る。
    #[test]
    fn identity_requires_both_parts() {
        let c = sample();
        assert!(identity_matches(&c, "swift15-abl", 12_345));
        assert!(!identity_matches(&c, "swift15-abl", 12_346));
        assert!(!identity_matches(&c, "swift15", 12_345));
    }

    /// 減衰と上限。合計が上限の半分に達したら全体を 1/2 にする (比は保つ)。
    #[test]
    fn merge_halves_all_when_near_cap() {
        // 通常は足すだけ
        assert_eq!(merge(&[100, 0, 7], &[3, 5, 1]), vec![103, 5, 8]);
        // 長さが違っても短い方は 0 扱い
        assert_eq!(merge(&[10], &[1, 2]), vec![11, 2]);
        // 上限の半分に達したら全体を 1/2 (順位は変わらない)
        let m = merge(&[COUNT_CAP / 2, 10, 0], &[0, 0, 4]);
        assert_eq!(m, vec![COUNT_CAP / 4, 5, 2]);
        // 起動 1 回で過去は半分になる (install_prior と同じ規則)
        let d: Vec<u32> = [100u32, 7].iter().map(|v| v >> DECAY_SHIFT).collect();
        assert_eq!(d, vec![50, 3]);
    }

    /// 事前ロードの順は「回数の少ない順、同数は (層, expert) 昇順」。0 は含めない。
    #[test]
    fn preload_order_is_ascending_and_deterministic() {
        let c = sample();
        assert_eq!(preload_order(&c), vec![(1, 2), (0, 0), (1, 1)]);
        // 0 だけなら空
        assert!(preload_order(&Counts::zeros("m", 1, 2, 2)).is_empty());
        // 同数は層→expert の昇順
        let mut c = Counts::zeros("m", 1, 2, 2);
        c.cells = vec![4, 4, 4, 4];
        assert_eq!(preload_order(&c), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// 予算の判定は「足したら超える」で止める (ちょうどは載せる)。
    #[test]
    fn budget_full_only_stops_when_exceeding() {
        assert!(!budget_full(100, 50, 150));
        assert!(budget_full(100, 51, 150));
        assert!(!budget_full(usize::MAX - 1, 1, usize::MAX));
        assert!(budget_full(usize::MAX, 1, usize::MAX));
    }

    /// 数えるのは指定した層の指定した expert だけ。範囲外は捨てる。
    #[test]
    fn observe_counts_only_valid_cells() {
        let c = Cells::new(2, 3);
        c.observe(0, &[0, 2, 2]);
        c.observe(1, &[1]);
        c.observe(2, &[0]); // 層が範囲外
        c.observe(0, &[3]); // expert が範囲外
        assert_eq!(c.snapshot(), vec![1, 0, 2, 0, 1, 0]);
        assert_eq!(c.dims(), (2, 3));
        c.reset();
        assert_eq!(c.snapshot(), vec![0; 6]);
    }

    /// task 263: prefill は 8 トークンに 1 回だけ数える (重み 1/8)。
    ///
    /// 20 トークン x k=2 なら標本は t = 0, 8, 16 の 3 本 x 2 = 6 回。
    /// グローバルの SESSION は触らない (decode 経路のテストが次元を見るため)。
    #[test]
    fn prefill_batch_counts_every_eighth_token() {
        assert_eq!(sampled_tokens(20).collect::<Vec<_>>(), vec![0, 8, 16]);
        assert_eq!(sampled_tokens(0).count(), 0);
        // t ごとに違う expert を選ばせて、数えた t が分かるようにする。
        let (nt, k) = (20usize, 2usize);
        let mut sels = vec![0usize; nt * k];
        for t in 0..nt {
            sels[t * k] = t % 4;
            sels[t * k + 1] = 3;
        }
        let c = Cells::new(2, 4);
        c.observe_batch(1, &sels, k, nt);
        let snap = c.snapshot();
        // 層 1 の行 = [layer * 4 + expert]。標本の t は 0, 8, 16 で、
        // すべて t % 4 == 0 なので expert 0 が 3 回、expert 3 が 3 回。
        assert_eq!(&snap[4..8], &[3, 0, 0, 3], "標本の t が 0,8,16 でない: {snap:?}");
        // 層 0 は触っていない
        assert_eq!(&snap[0..4], &[0, 0, 0, 0]);
        // 範囲外の層・k=0 は何もしない
        c.observe_batch(9, &sels, k, nt);
        c.observe_batch(1, &sels, 0, nt);
        assert_eq!(c.snapshot(), snap);
    }

    /// task 263: ファイル名は (ディレクトリ名, expert バイト) から作る。
    #[test]
    fn per_model_file_name_is_identity_based() {
        let a = stats_file_name("swift15-abl-IQ3_S", 52_815_000_000);
        assert_eq!(a, "dnnoexps-swift15-abl-IQ3_S-52815000000.bin");
        // 同じ名前でも expert バイトが違えば別ファイル
        assert_ne!(a, stats_file_name("swift15-abl-IQ3_S", 52_815_000_001));
        // パス区切り・空白は `_` に、先頭末尾の `_` は落とす
        assert_eq!(
            stats_file_name("/models/a b/c", 7),
            "dnnoexps-models_a_b_c-7.bin"
        );
        // 空でも衝突しない名前になる
        assert_eq!(stats_file_name("", 7), "dnnoexps-model-7.bin");
        // 長すぎる名前は 80 文字で切る
        let long = "x".repeat(200);
        let n = stats_file_name(&long, 7);
        assert!(n.len() <= "dnnoexps-".len() + 80 + "-7.bin".len(), "{n}");
    }

    /// task 263/798: warm の明示ファイル > 明示ファイル > ディレクトリ > off。
    #[test]
    fn resolve_path_prefers_explicit_file() {
        // task 798: warm の明示パスが最優先 (IQ ゲートも要らない)
        assert_eq!(
            resolve_stats_path(Some("/w.bin"), Some("/tmp/s.bin"), Some("/d"), "m", 1, false),
            Some(PathBuf::from("/w.bin"))
        );
        // 明示ファイルは IQ ゲートが要る (従来どおり)
        assert_eq!(
            resolve_stats_path(None, Some("/tmp/s.bin"), Some("/d"), "m", 1, true),
            Some(PathBuf::from("/tmp/s.bin"))
        );
        assert_eq!(
            resolve_stats_path(None, Some("/tmp/s.bin"), Some("/d"), "m", 1, false),
            Some(PathBuf::from("/d").join(stats_file_name("m", 1)))
        );
        // ファイルが無ければディレクトリ + モデル名 (IQ ゲート不要)
        assert_eq!(
            resolve_stats_path(None, None, Some(" /d "), "m", 1, false),
            Some(PathBuf::from("/d").join("dnnoexps-m-1.bin"))
        );
        assert_eq!(resolve_stats_path(None, None, Some("  "), "m", 1, true), None);
        assert_eq!(resolve_stats_path(None, None, None, "m", 1, true), None);
    }

    /// task 798: 起動時の本数・バイト上限の解釈と丸め。
    #[test]
    fn warm_limits_parse_and_clamp() {
        // 本数: 未設定・空・非数は上限なし
        assert_eq!(warm_experts_from_value(None), None);
        assert_eq!(warm_experts_from_value(Some("")), None);
        assert_eq!(warm_experts_from_value(Some("  ")), None);
        assert_eq!(warm_experts_from_value(Some("x")), None);
        // 0 は「載せない」
        assert_eq!(warm_experts_from_value(Some("0")), Some(0));
        assert_eq!(warm_experts_from_value(Some(" 4000 ")), Some(4000));
        // 桁の打ち間違いは MAX に丸める
        assert_eq!(warm_experts_from_value(Some("99999999999")), Some(MAX_WARM_EXPERTS));
        // バイト: 未設定・空・非数は上限なし、0 は「1 本も入れない」
        assert_eq!(warm_mb_from_value(None), None);
        assert_eq!(warm_mb_from_value(Some("")), None);
        assert_eq!(warm_mb_from_value(Some("z")), None);
        assert_eq!(warm_mb_from_value(Some("0")), Some(0));
        assert_eq!(warm_mb_from_value(Some("14336")), Some(14336));
        assert_eq!(warm_mb_from_value(Some("99999999999")), Some(MAX_WARM_MB));
    }

    /// task 263: 覆う割合はキーの重複を 1 回だけ数える。
    #[test]
    fn coverage_counts_uses_once_per_key() {
        let mut c = Counts::zeros("m", 1, 2, 3);
        c.cells = vec![1, 2, 3, 4, 5, 6];
        // (0,0)=1 + (1,1)=5 = 6 / 全体 21
        assert_eq!(coverage(&c, &[(0, 0), (1, 1)]), (6, 21));
        // 重複は 1 回
        assert_eq!(coverage(&c, &[(0, 0), (0, 0)]), (1, 21));
        // 範囲外は無視
        assert_eq!(coverage(&c, &[(9, 0)]), (0, 21));
        assert_eq!(coverage(&c, &[]), (0, 21));
    }

    /// task 263: plan の割り方。回数の多い側から VRAM → RAM の順に埋め、
    /// どちらにも入らない残りは mmap。境界は「ちょうどは入る」。
    #[test]
    fn split_plan_fills_vram_then_ram_from_the_hottest() {
        // 5 本 x 100 バイト (昇順 = index 4 が一番熱い)。
        let sizes = vec![100u64; 5];
        // VRAM 300 = 3 本 (上から)、RAM 200 = 2 本 → mmap 0
        assert_eq!(split_plan(&sizes, 300, 200), (2, 0));
        // VRAM 100 ちょうど = 1 本、RAM 100 = 1 本、残り 3 本は mmap
        assert_eq!(split_plan(&sizes, 100, 100), (4, 3));
        // 予算 0 なら 1 本も載せない
        assert_eq!(split_plan(&sizes, 0, 0), (5, 5));
        // VRAM が 1 本未満 (99) でも RAM には入る
        assert_eq!(split_plan(&sizes, 99, 1000), (5, 0));
        // 大きさが違う: 末尾 (熱い) が 400 で VRAM 300 には入らない → RAM には入る
        let sizes = vec![100u64, 100, 100, 100, 400];
        assert_eq!(split_plan(&sizes, 300, 500), (5, 3));
        // 空
        assert_eq!(split_plan(&[], 100, 100), (0, 0));
    }

    /// 有効条件: IQ ゲートと記録先の両方が要る。
    #[test]
    fn gate_needs_both_switch_and_path() {
        assert_eq!(gate(false, Some("/tmp/a.bin")), None);
        assert_eq!(gate(true, None), None);
        assert_eq!(gate(true, Some("  ")), None);
        assert_eq!(gate(true, Some("/tmp/a.bin")), Some(PathBuf::from("/tmp/a.bin")));
    }

    /// env の解釈。空・空白は未設定と同じ、予備は既定 2000 MiB。
    #[test]
    fn env_parsing() {
        assert_eq!(stats_file_from_value(None), None);
        assert_eq!(stats_file_from_value(Some("")), None);
        assert_eq!(stats_file_from_value(Some("   ")), None);
        assert_eq!(
            stats_file_from_value(Some(" /tmp/a.bin ")),
            Some(PathBuf::from("/tmp/a.bin"))
        );
        assert_eq!(reserve_mb_from_value(None), DEFAULT_PRELOAD_RESERVE_MB);
        assert_eq!(reserve_mb_from_value(Some("")), DEFAULT_PRELOAD_RESERVE_MB);
        assert_eq!(reserve_mb_from_value(Some("0")), 0);
        assert_eq!(reserve_mb_from_value(Some("512")), 512);
        assert_eq!(reserve_mb_from_value(Some("x")), DEFAULT_PRELOAD_RESERVE_MB);
        // 既定は t147b の推奨 (20,000 で 2 GB 余裕) と同じ
        assert_eq!(DEFAULT_PRELOAD_RESERVE_MB, 2000);
    }

    /// 一時ファイル経由の保存が読める形で残る (rename 後は元の名前だけ)。
    #[test]
    fn save_atomic_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dnno-exp-stats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let p = dir.join("s.bin");
        let c = sample();
        save_atomic(&p, &c).expect("保存");
        let got = load_file(&p, "swift15-abl", 12_345, 0).expect("読める");
        assert_eq!(got, c);
        // 別モデルとしては読めない
        let e = load_file(&p, "swift15-abl", 999, 0).unwrap_err();
        assert!(e.contains("別のモデル"), "{e}");
        // task 798: 指紋つきで保存したファイルは、別の指紋では読めない
        let mut c = sample();
        c.fingerprint = 0xabcd_ef01_2345_6789;
        save_atomic(&p, &c).expect("保存");
        let e = load_file(&p, "swift15-abl", 12_345, 0xabcd_ef01_2345_678a).unwrap_err();
        assert!(e.contains("別のリビジョン"), "{e}");
        load_file(&p, "swift15-abl", 12_345, 0xabcd_ef01_2345_6789).expect("同じ指紋なら読める");
        // 一時ファイルは残っていない
        let left: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(left.is_empty(), "一時ファイルが残った: {left:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 数える関数の実測コスト (1 トークン = 48 層 x 10 本)。
    ///
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_observe`
    #[test]
    #[ignore]
    fn bench_observe_layer_cost() {
        let c = Cells::new(48, 512);
        let sel: Vec<usize> = (0..10).collect();
        let n = 20_000usize;
        let t = std::time::Instant::now();
        for i in 0..n {
            for layer in 0..48 {
                c.observe(layer, &sel);
            }
            std::hint::black_box(i);
        }
        let el = t.elapsed();
        let per_token = el.as_secs_f64() / n as f64;
        println!(
            "observe_layer: {:.1} ns/token (480 セル), {:.1} us/token",
            per_token * 1e9,
            per_token * 1e6
        );
        assert!(per_token < 0.3e-3, "0.3 ms/token を超えた: {per_token}");
    }

    /// task 263: prefill の計数の実測コスト (512 トークン x k=10 の 1 層ぶん)。
    ///
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_observe_batch`
    #[test]
    #[ignore]
    fn bench_observe_batch_cost() {
        let c = Cells::new(48, 512);
        let sel: Vec<usize> = (0..10).collect();
        let nt = 512usize;
        let n = 2_000usize; // チャンク数
        let t = std::time::Instant::now();
        for _ in 0..n {
            for layer in 0..48 {
                for _ in (0..nt).step_by(PREFILL_STRIDE) {
                    c.observe(layer, &sel);
                }
            }
        }
        let el = t.elapsed();
        let per_chunk = el.as_secs_f64() / n as f64;
        // 1 トークンの decode は 48 層 x 10 本 = 480 回。
        println!(
            "observe_batch: {:.1} us / 512-token chunk ({} adds) = {:.3} us/token",
            per_chunk * 1e6,
            nt.div_ceil(PREFILL_STRIDE) * 48 * sel.len(),
            per_chunk * 1e6 / nt as f64
        );
        assert!(per_chunk < 5e-3, "5 ms/chunk を超えた: {per_chunk}");
    }
}
