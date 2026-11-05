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
//! # 既定
//!
//! `DENNO_EXPERT_IQ_STATS_FILE` が未設定なら、この機能は丸ごと off
//! ([`enabled`] が false = 数える分岐 1 個も通らない)。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// 使用回数のファイルパス。未設定ならこの機能は off。
pub const STATS_FILE_ENV: &str = "DENNO_EXPERT_IQ_STATS_FILE";

/// 事前ロードで予算から空けておく MiB (`DENNO_EXPERT_VRAM_MB` の内数)。
///
/// 事前ロードは起動直後に走るので、dense 重みの upload・KV・prefill の
/// staging が後から来る。LRU を上限まで埋めると 14k prefill で OOM した
/// (t147a の 24,000 MiB)。t147b が「20,000 で 2 GB の余裕」を推奨しているのに
/// 合わせ、既定は 2,000 MiB。
pub const PRELOAD_RESERVE_MB_ENV: &str = "DENNO_EXPERT_IQ_PRELOAD_RESERVE_MB";

/// [`PRELOAD_RESERVE_MB_ENV`] の既定値。
pub const DEFAULT_PRELOAD_RESERVE_MB: usize = 2000;

/// 形式の版。読む側は一致しなければ事前ロードしない。
pub const FORMAT_VERSION: u32 = 1;

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

/// 有効条件 (純関数): IQ ゲートが on かつ記録先が設定されていること。
///
/// どちらか欠けたら `None` = この機能は丸ごと off (数える分岐も通らない)。
pub fn gate(iq_gpu: bool, file: Option<&str>) -> Option<PathBuf> {
    if !iq_gpu {
        return None;
    }
    stats_file_from_value(file)
}

/// 使用回数のファイル (ゲートが off なら `None`)。
///
/// 有効条件は「`DENNO_EXPERT_IQ_GPU=1`」かつ「`DENNO_EXPERT_IQ_STATS_FILE` が
/// 空でない」。どちらか欠けたらこの機能は丸ごと off。
pub fn stats_path() -> Option<&'static Path> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| {
        gate(
            crate::model::exp_iq::iq_gpu_enabled(),
            std::env::var(STATS_FILE_ENV).ok().as_deref(),
        )
    })
    .as_deref()
}

/// 数える・保存する・事前ロードするのいずれかが有効か。
pub fn enabled() -> bool {
    #[cfg(test)]
    if TEST_ON.load(Ordering::Relaxed) {
        return true;
    }
    stats_path().is_some()
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
    pub n_layer: u32,
    pub n_expert: u32,
    /// `[layer * n_expert + expert]` の回数
    pub cells: Vec<u32>,
}

impl Counts {
    /// 0 で埋めたカウンタ (ファイルが無いときの代わり)。
    pub fn zeros(dir: &str, expert_bytes: u64, n_layer: u32, n_expert: u32) -> Self {
        Self {
            dir: dir.to_string(),
            expert_bytes,
            n_layer,
            n_expert,
            cells: vec![0; n_layer as usize * n_expert as usize],
        }
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

/// ファイルへ書くバイト列 (版 1)。
pub fn encode(c: &Counts) -> Vec<u8> {
    let dir = c.dir.as_bytes();
    let mut out = Vec::with_capacity(24 + dir.len() + 8 + c.cells.len() * 4);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&c.n_layer.to_le_bytes());
    out.extend_from_slice(&c.n_expert.to_le_bytes());
    out.extend_from_slice(&(dir.len() as u32).to_le_bytes());
    out.extend_from_slice(dir);
    out.extend_from_slice(&c.expert_bytes.to_le_bytes());
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
pub fn decode(b: &[u8]) -> Result<Counts, String> {
    if b.len() < 24 {
        return Err(format!("短すぎる ({} バイト < ヘッダ 24)", b.len()));
    }
    if b[..8] != MAGIC {
        return Err("識別子が違う (このファイルは expert 使用回数ではない)".to_string());
    }
    let version = take_u32(b, 8).ok_or("版が読めない")?;
    if version != FORMAT_VERSION {
        return Err(format!(
            "版が違う ({version} != {FORMAT_VERSION})"
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
    let n_cells = (n_layer as usize)
        .checked_mul(n_expert as usize)
        .ok_or("セル数が溢れる")?;
    let body = dir_end + 8;
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
        n_layer,
        n_expert,
        cells,
    })
}

/// 記録したモデルと目の前のモデルが同じか。
pub fn identity_matches(c: &Counts, dir: &str, expert_bytes: u64) -> bool {
    c.dir == dir && c.expert_bytes == expert_bytes
}

/// ファイルを読んで同一性を確かめる。
///
/// 失敗はすべて `Err` (呼び出し側は事前ロードをしない)。理由は 1 行。
pub fn load_file(path: &Path, dir: &str, expert_bytes: u64) -> Result<Counts, String> {
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

// ---------------------------------------------------------------------------
// 状態 (同一性・履歴・セッション)
// ---------------------------------------------------------------------------

struct SaveState {
    dir: String,
    expert_bytes: u64,
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
/// 保存だけは続けたいため。
pub fn set_identity(dir: &str, expert_bytes: u64, n_layer: usize, n_expert: usize) {
    if let Ok(mut s) = state().lock() {
        s.dir = dir.to_string();
        s.expert_bytes = expert_bytes;
        s.n_layer = n_layer as u32;
        s.n_expert = n_expert as u32;
    }
}

/// 読み込んだ履歴を「前回まで」として据え、保存を許可する。
///
/// ここで過去ぶんを 1/2 にする (起動 1 回 = 減衰 1 回)。順位は変わらないが、
/// これから積む今回のセッションの重みが相対的に上がる。何度も再起動すると
/// 2^-N で古い履歴は消えていく。
pub fn install_prior(c: &Counts) {
    if let Ok(mut s) = state().lock() {
        s.dir = c.dir.clone();
        s.expert_bytes = c.expert_bytes;
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
        n_layer: n_layer as u32,
        n_expert: n_expert as u32,
        cells: merged.clone(),
    };
    match save_atomic(path, &c) {
        Ok(()) => {
            session_cells.reset();
            DIRTY.store(false, Ordering::Relaxed);
            s.prior = merged;
            let hits = c.nonzero();
            Some(format!(
                "[exp-stats] saved {} counted={} cells bytes={} (prior>>{} at load + session)",
                path.display(),
                hits,
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
                DECAY_SHIFT
            ))
        }
        Err(e) => Some(format!("[exp-stats] save failed {}: {e}", path.display())),
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

    /// 書き出しと読み戻しが一致する (版・次元・同一性つき)。
    #[test]
    fn encode_decode_roundtrip() {
        let c = sample();
        let b = encode(&c);
        assert_eq!(decode(&b).expect("読める"), c);
        assert_eq!(b.len(), 24 + "swift15-abl".len() + 8 + 6 * 4);
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
        // 版
        let mut bad = b.clone();
        bad[8..12].copy_from_slice(&2u32.to_le_bytes());
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
        let got = load_file(&p, "swift15-abl", 12_345).expect("読める");
        assert_eq!(got, c);
        // 別モデルとしては読めない
        let e = load_file(&p, "swift15-abl", 999).unwrap_err();
        assert!(e.contains("別のモデル"), "{e}");
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
}
