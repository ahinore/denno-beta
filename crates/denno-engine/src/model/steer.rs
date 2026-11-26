//! task 218: 残差ストリームの平均記録とステアリング (abliteration)。
//!
//! 「ある会話の残差ストリームの平均」を名前付きスナップショットとして貯め、
//! 新しい会話に組み合わせて引く (subtract) / 射影で落とす (project)。
//! task 217 の [`super::intervene`] と同じ epoch を共有し、設定が変わったら
//! serve の prefix cache を捨てさせる。
//!
//! # 記録する点 (重要)
//!
//! **層の出口、2 つの combine を両方終えた後の残差**を記録し、**同じ点**に
//! ステアリングを掛ける (= 層 `il` の出口 = 層 `il+1` の入口)。
//!
//! - 記録点が層の入口 (mix の前) だと、層 0 の入口は embedding の複製で、
//!   まだ何も層を通っていない。層の出口なら「その層までのブロック出力を
//!   足し込んだ後のストリーム」で、ablation の文献が使う residual stream と
//!   同じ意味になる。
//! - 適用量を「層 `il` の出口」に置くと、`il+1` の mix がそのまま読む値に
//!   なるので、層の範囲 `[from, to)` を「層 `from` の出力から層 `to-1` の
//!   出力まで」と読める。最後の層 (`il == n_layer-1`) の出口はヘッドの mix の
//!   入力なので、範囲に `n_layer-1` を含めれば logits にも効く。
//! - 実装は「層ループの本体を抜けた直後 (ffn combine の後)」1 箇所だけ。
//!   decode / 非常駐 prefill / 常駐 prefill (デバイス) の 3 経路で同じ点。
//!
//! # どこで計算するか
//!
//! 残差がホストにある経路 (decode と、`DENNO_PF_RESIDENT` を使わない prefill)
//! はホストで足す。**残差がデバイスにある常駐 prefill では GPU のカーネルで
//! 足す** (`crate::cuda::pf_steer_*`)。層ごとに残差を D2H しない。
//! 平均の合成 (`v = Σ coef_i * mean_i`) と `v_hat` の正規化はホストで 1 回だけ
//! やり、`v` をデバイスへ置く。
//!
//! # 数値
//!
//! - 記録は「トークン昇順の逐次加算」。要素ごとの加算順は
//!   ホストも GPU も同じ (`acc[d] += x_t[d]` を t 昇順)。チャンクを跨いでも
//!   順序は同じなので、同じトークン列なら同じ値になる。
//! - 平均は `sum / tokens` (1 回の除算)。
//! - `v` は項の順に加算 (項の順序は呼び出し側が決める)。
//! - `v_hat` は二乗和を f64 で取ってから `1/sqrt` を掛け、f32 に丸める。
//! - ステアリング後の値は `h - alpha * v` / `h - alpha * (h・v_hat) v_hat`。
//!
//! # 既定 (採否)
//!
//! lab t218c (`/work/gpu_lab/done/t218c/`, sha fb8040d47) で採否 OK。
//! 介入もステアリングもしない既定経路は md5 全項目 SAME で、記録のコストは
//! decode +1.0% / prefill +0.0% (target 5% 以内)。**ゲートは無い** — 何も
//! 設定しなければ層ループの分岐 1 個だけで不発なので、既定 on/off の概念が
//! 無い (HTTP か env で明示的に on にしたときだけ効く)。
//!
//! 実測で分かったこと: 同じプロンプトを記録して同じプロンプトに引くと、
//! 平均がそのプロンプトの残差そのものに近いため、`alpha 1.0` を全 48 層に
//! 掛けると自己消去してテキストが壊れる (t218b)。実用的な例は
//! `A/1.0/20-28/0.05/subtract` (層 20-28 の 9 層、alpha 0.05) で、
//! 内容を保ったまま言い回しが変わる (t218c の `sub`)。
//!
//! # フックの配線 (task 957)
//!
//! - 層ループの入口は [`apply_on`] (原子的読み 1 回) と [`active`] だけ。何も
//!   掛けていない既定経路は `RwLock` も取らず、規則も 1 本も掛けない
//!   (`steer::snapshot()` を毎トークン読まない)。設定したときだけ
//!   [`snapshot`] を 1 回読み、層ループの中では `Arc` を参照する。
//! - 重ね掛け ([`Apply::apply_one`]) は規則を順に掛ける (task 849 の数値のまま)。
//!   `DENNO_STEER_ORTHO=1` では、同じ層に掛かる project 規則の方向を
//!   (layer, stream) ごとに正規直交化してから掛けるので、
//!   `h -= Σ_i alpha_i (h・q_i) q_i` になり、規則の順序に依らず、
//!   alpha 1 の方向は残差から厳密に落ちる (既定 off)。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock, RwLock};

/// スナップショット名に使える最大長。
pub const MAX_NAME: usize = 64;
/// スナップショットファイルの先頭行。
pub const FILE_MAGIC: &str = "DNSTEER1";
/// スナップショットファイルの拡張子。
pub const FILE_EXT: &str = "dns";
/// 同時に保持するスナップショット数の上限 (メモリ保護)。
pub const MAX_SNAPSHOTS: usize = 64;

/// task 829: 記録する残差の取り方。
///
/// - [`CaptureMode::All`] は従来どおり「記録したトークン全部の平均」。
/// - [`CaptureMode::LastPromptToken`] は prefill の最後のトークン (`nt - 1`) の
///   残差だけを層ごとに記録する (生成中の decode トークンは載せない)。
///   abliteration の difference-in-means で、指示境界の残差だけを使うためのもの。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CaptureMode {
    /// 全トークンの平均 (既定、後方互換)。
    #[default]
    All,
    /// prefill の最後のトークンだけ。
    LastPromptToken,
}

impl CaptureMode {
    pub fn name(self) -> &'static str {
        match self {
            CaptureMode::All => "all",
            CaptureMode::LastPromptToken => "last_prompt_token",
        }
    }

    pub fn parse(s: &str) -> Option<CaptureMode> {
        match s {
            "all" => Some(CaptureMode::All),
            "last_prompt_token" => Some(CaptureMode::LastPromptToken),
            _ => None,
        }
    }

    /// ホットパス用の 1 バイト表現。
    fn code(self) -> u8 {
        match self {
            CaptureMode::All => 0,
            CaptureMode::LastPromptToken => 1,
        }
    }

    fn from_code(c: u8) -> CaptureMode {
        if c == 1 {
            CaptureMode::LastPromptToken
        } else {
            CaptureMode::All
        }
    }
}

/// 形 (記録・適用に要る最小限)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub n_layer: usize,
    pub n_hc: usize,
    pub n_embd: usize,
}

impl Shape {
    /// 1 層ぶんの要素数 (`hc * n_embd`)。
    #[inline]
    pub fn stride(&self) -> usize {
        self.n_hc * self.n_embd
    }

    /// 全層ぶんの要素数。
    #[inline]
    pub fn len(&self) -> usize {
        self.n_layer * self.stride()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 常識的な範囲か (壊れた GGUF で巨大確保しないため)。
    pub fn is_valid(&self) -> bool {
        self.n_layer > 0
            && self.n_layer <= 1024
            && self.n_hc > 0
            && self.n_hc <= 32
            && self.n_embd > 0
            && self.n_embd <= 65536
    }
}

/// 適用のしかた。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `h <- h - alpha * v`
    Subtract,
    /// `h <- h - alpha * (h . v_hat) * v_hat` (ストリームごと)
    Project,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Subtract => "subtract",
            Mode::Project => "project",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "subtract" => Some(Mode::Subtract),
            "project" => Some(Mode::Project),
            _ => None,
        }
    }
}

/// 名前付きの平均ベクトル。
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub name: String,
    pub tokens: u64,
    pub created: u64,
    pub model: String,
    pub layers: usize,
    pub n_hc: usize,
    pub n_embd: usize,
    /// `[layer][stream][n_embd]`
    pub mean: Vec<f32>,
}

impl Snapshot {
    pub fn shape(&self) -> Shape {
        Shape {
            n_layer: self.layers,
            n_hc: self.n_hc,
            n_embd: self.n_embd,
        }
    }
}

/// 適用項 1 個。
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub snapshot: String,
    pub coef: f32,
}

/// task 849: 重ね掛けの 1 規則 (概念 1 個ぶん)。
///
/// 単一規則の [`Apply`] と違い、層範囲・alpha・mode を規則ごとに持つ。
/// サーバーの `POST /v1/steer/apply` の `rules: [{terms, layers, alpha, mode}]`
/// がこれに落ちる。`v` / `vhat` は `from..to` の外が 0 (全層ぶんの長さ)。
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub terms: Vec<Term>,
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
    /// `[layer][stream][n_embd]` の合成ベクトル (範囲外は 0)。
    pub v: Vec<f32>,
    /// `v` を (layer, stream) ごとに正規化したもの (範囲外は 0)。
    pub vhat: Vec<f32>,
    /// task 957: `DENNO_STEER_ORTHO=1` のときだけ入る、先行する project 規則の
    /// 方向に対して正規直交化した方向 (範囲外と、依存で落ちた (layer, stream) は 0)。
    /// 空なら project は `vhat` を使う (従来 = 規則を順に掛けるだけ)。
    pub vortho: Vec<f32>,
}

/// task 957: ホットパスが実際に規則を掛けた回数 (テストで「無効時に 0 回」を数える)。
#[cfg(test)]
pub(crate) static APPLY_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl Rule {
    #[inline]
    pub fn covers(&self, layer: usize) -> bool {
        layer >= self.from && layer < self.to
    }

    /// project に使う方向 (`vortho` があればそちら、無ければ `vhat`)。
    #[inline]
    fn project_dir(&self) -> &[f32] {
        if self.vortho.is_empty() {
            &self.vhat
        } else {
            &self.vortho
        }
    }

    /// 1 トークンぶんの残差 (`[hc][n_embd]`) に、この規則だけを適用する。
    /// 演算の順序は [`Apply::apply_one`] の単一規則経路と同じ。
    pub fn apply_one(&self, layer: usize, data: &mut [f32], st: usize, e: usize, hc: usize) {
        if st == 0 || !self.covers(layer) || data.len() < st {
            return;
        }
        #[cfg(test)]
        APPLY_CALLS.fetch_add(1, Ordering::Relaxed);
        let base = layer * st;
        match self.mode {
            Mode::Subtract => {
                let v = &self.v[base..base + st];
                let a = self.alpha;
                for d in 0..st {
                    data[d] -= a * v[d];
                }
            }
            Mode::Project => {
                let dir = self.project_dir();
                for s in 0..hc {
                    let vp = &dir[base + s * e..base + (s + 1) * e];
                    let hp = &mut data[s * e..(s + 1) * e];
                    let mut dot = 0.0f32;
                    for d in 0..e {
                        dot += hp[d] * vp[d];
                    }
                    let c = self.alpha * dot;
                    for d in 0..e {
                        hp[d] -= c * vp[d];
                    }
                }
            }
        }
    }
}

/// task 849: 「規則 1 個ぶんの指定」。`set_rules` に渡す。
#[derive(Clone, Debug, PartialEq)]
pub struct RuleSpec {
    pub terms: Vec<Term>,
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
}

/// いま効いている適用。
#[derive(Clone, Debug)]
pub struct Apply {
    pub terms: Vec<Term>,
    /// `[from, to)`。`to` は含まない。重ね掛けでは全規則の外側の包み。
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
    /// `[layer][stream][n_embd]` の合成ベクトル。
    pub v: Vec<f32>,
    /// `v` を (layer, stream) ごとに正規化したもの。
    pub vhat: Vec<f32>,
    /// task 849: 重ね掛けの規則。空なら従来の単一規則 (`v` / `alpha` / `mode`)。
    pub rules: Vec<Rule>,
    shape: Shape,
}

impl Apply {
    #[inline]
    pub fn covers(&self, layer: usize) -> bool {
        if !self.rules.is_empty() {
            return self.rules.iter().any(|r| r.covers(layer));
        }
        layer >= self.from && layer < self.to
    }

    /// 単一規則 (`v` / `alpha` / `mode`) だけで表せるか。
    /// GPU の常駐 prefill は 1 本の合成ベクトルと 1 個の alpha しか置けないので、
    /// 重ね掛けのときはホスト経路へ落とす (task 849)。
    #[inline]
    pub fn is_single_rule(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// 1 トークンぶんの残差 (`[hc][n_embd]`) に適用する。
    ///
    /// 範囲外の層・形違いは何もしない (呼び出し側で `covers` を見るのが速い)。
    pub fn apply_one(&self, layer: usize, data: &mut [f32]) {
        let st = self.shape.stride();
        if st == 0 || !self.covers(layer) || data.len() < st {
            return;
        }
        #[cfg(test)]
        APPLY_CALLS.fetch_add(1, Ordering::Relaxed);
        // task 849: 重ね掛けは規則の順に 1 個ずつ掛ける (範囲も alpha も別)。
        if !self.rules.is_empty() {
            let e = self.shape.n_embd;
            let hc = self.shape.n_hc;
            for r in &self.rules {
                r.apply_one(layer, data, st, e, hc);
            }
            return;
        }
        let base = layer * st;
        match self.mode {
            Mode::Subtract => {
                let v = &self.v[base..base + st];
                let a = self.alpha;
                for d in 0..st {
                    data[d] -= a * v[d];
                }
            }
            Mode::Project => {
                let e = self.shape.n_embd;
                for s in 0..self.shape.n_hc {
                    let vp = &self.vhat[base + s * e..base + (s + 1) * e];
                    let hp = &mut data[s * e..(s + 1) * e];
                    let mut dot = 0.0f32;
                    for d in 0..e {
                        dot += hp[d] * vp[d];
                    }
                    let c = self.alpha * dot;
                    for d in 0..e {
                        hp[d] -= c * vp[d];
                    }
                }
            }
        }
    }
}

/// 1 リクエストぶんのスナップショット (不変)。ホットパスは Arc を clone するだけ。
#[derive(Clone, Debug)]
pub struct Steer {
    pub epoch: u64,
    pub apply: Option<Apply>,
}

/// 適用中の設定の要約 (GET /v1/steer 用。`v` は持たない)。
#[derive(Clone, Debug)]
pub struct ApplyInfo {
    pub terms: Vec<Term>,
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
    /// task 849: 重ね掛けの規則 (空なら従来の単一規則)。
    pub rules: Vec<RuleInfo>,
}

/// 重ね掛け 1 規則の要約 (GET /v1/steer 用)。
#[derive(Clone, Debug)]
pub struct RuleInfo {
    pub terms: Vec<Term>,
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
}

impl Steer {
    fn empty(epoch: u64) -> Steer {
        Steer {
            epoch,
            apply: None,
        }
    }
}

// ---------------------------------------------------------------------------
// グローバル
// ---------------------------------------------------------------------------

fn cur_cell() -> &'static RwLock<Arc<Steer>> {
    static CUR: OnceLock<RwLock<Arc<Steer>>> = OnceLock::new();
    CUR.get_or_init(|| RwLock::new(Arc::new(Steer::empty(0))))
}

fn snaps_cell() -> &'static RwLock<Arc<Vec<Arc<Snapshot>>>> {
    static S: OnceLock<RwLock<Arc<Vec<Arc<Snapshot>>>>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(Arc::new(Vec::new())))
}

fn meta_cell() -> &'static RwLock<Option<(Shape, String)>> {
    static M: OnceLock<RwLock<Option<(Shape, String)>>> = OnceLock::new();
    M.get_or_init(|| RwLock::new(None))
}

fn rec_cell() -> &'static Mutex<Option<Rec>> {
    static R: OnceLock<Mutex<Option<Rec>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

/// task 957: 適用 (`apply`) が設定されているか。ホットパスの入口で 1 回見るだけで、
/// 何も設定されていないときは `RwLock` も取らずに層ループを素通りするためのもの。
/// 値は [`install_apply`] / [`clear_apply`] / [`reset_for_test`] だけが書く。
static APPLY_ON: AtomicBool = AtomicBool::new(false);

/// task 957: 適用が設定されているか (ホットパスの分岐 1 個)。
#[inline]
pub fn apply_on() -> bool {
    APPLY_ON.load(Ordering::Relaxed)
}

/// task 957: ホットパス用の入口。適用が無ければ `None` を返し、ロックも取らない。
///
/// 何も掛けていない既定経路の decode / prefill は、これ 1 回の原子的読みだけで
/// ステアリングの分岐を抜ける (`steer::snapshot()` の `RwLock` + `Arc` clone をしない)。
/// 適用があるときだけ [`snapshot`] を 1 回読み、層ループの中では `Arc` を参照する。
#[inline]
pub fn active() -> Option<Arc<Steer>> {
    if !apply_on() {
        return None;
    }
    let s = snapshot();
    if s.apply.is_some() {
        Some(s)
    } else {
        None
    }
}

/// 記録中か (ホットパスの分岐 1 個)。
static REC_ON: AtomicBool = AtomicBool::new(false);

/// task 829: 記録中の [`CaptureMode`] (`REC_ON` が false のときは無意味)。
/// ホットパスは `rec_on() && last_only()` の 2 分岐だけ。
static REC_MODE: AtomicU8 = AtomicU8::new(0);

/// 直近に使った [`CaptureMode`] (記録していないときの `GET /v1/steer` 用)。
static LAST_MODE: AtomicU8 = AtomicU8::new(0);

/// `DENNO_STEER_RECORD` で始まった記録か (毎リクエスト後に flush する)。
static REC_ENV: AtomicBool = AtomicBool::new(false);

/// 記録を始めたのでデバイスの加算域を 0 にし直す必要がある、の印。
/// 加算域は thread_local なので、実際に 0 にするのは次の常駐 prefill の入口
/// (`pf_steer_acc_ensure`)。ここで立てないと、前の記録の残りを引き継ぐ。
static REC_DEV_RESET: AtomicBool = AtomicBool::new(false);

/// [`REC_DEV_RESET`] を 1 回だけ消費する。
pub fn take_dev_reset() -> bool {
    REC_DEV_RESET.swap(false, Ordering::SeqCst)
}

thread_local! {
    static PINS: std::cell::RefCell<Vec<Arc<Steer>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// いまの適用状態 (ロックを 1 回取って Arc を clone)。forward step の先頭で 1 回。
pub fn snapshot() -> Arc<Steer> {
    if let Some(c) = PINS.with(|p| p.borrow().last().cloned()) {
        return c;
    }
    let g = cur_cell().read().unwrap_or_else(|e| e.into_inner());
    Arc::clone(&g)
}

/// 現在の形とモデル id (未初期化なら `None`)。
pub fn meta() -> Option<(Shape, String)> {
    meta_cell().read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// 初期化 (モデルを組むときに 1 回)。スナップショットをディレクトリから読み直す。
///
/// 同じプロセスでモデルを組み直す経路 (テスト) があるので、何度呼んでもよい。
pub fn init(shape: Shape, model: &str) {
    if !shape.is_valid() {
        return;
    }
    *meta_cell().write().unwrap_or_else(|e| e.into_inner()) = Some((shape, model.to_string()));
    reload_snapshots();
    init_env_once();
}

/// `DENNO_STEER_DIR` (既定 `/work/serve8098/steer`)。
pub fn dir() -> PathBuf {
    match std::env::var("DENNO_STEER_DIR") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from("/work/serve8098/steer"),
    }
}

// ---------------------------------------------------------------------------
// ピン (リクエスト中の固定)
// ---------------------------------------------------------------------------

/// ドロップで解除する。入れ子も可。
pub struct SteerPin {
    _priv: (),
}

impl Drop for SteerPin {
    fn drop(&mut self) {
        PINS.with(|p| {
            p.borrow_mut().pop();
        });
    }
}

/// いまの適用状態を固定する ([`super::intervene::pin_request`] が呼ぶ)。
pub fn pin_request() -> SteerPin {
    let c = {
        let g = cur_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    PINS.with(|p| p.borrow_mut().push(c));
    SteerPin { _priv: () }
}

// ---------------------------------------------------------------------------
// 適用の組み立て
// ---------------------------------------------------------------------------

/// 項と範囲から `v` / `v_hat` を作る。
#[allow(clippy::type_complexity)]
fn build_apply(
    shape: &Shape,
    snaps: &[Arc<Snapshot>],
    terms: &[Term],
    from: usize,
    to: usize,
    alpha: f32,
    mode: Mode,
) -> Result<Apply, String> {
    if !shape.is_valid() {
        return Err("モデルの形が未初期化 (steer::init を先に呼ぶ)".into());
    }
    if terms.is_empty() {
        return Err("terms が空 (何も掛けないなら DELETE /v1/steer/apply)".into());
    }
    if from > to {
        return Err(format!("layers の from {from} が to {to} より大きい"));
    }
    if to > shape.n_layer {
        return Err(format!(
            "layers の to {to} が層数 {} を超える ({from}-{})",
            shape.n_layer, shape.n_layer
        ));
    }
    if !alpha.is_finite() {
        return Err("alpha が有限でない".into());
    }
    for t in terms {
        if !t.coef.is_finite() {
            return Err(format!("coef が有限でない ({})", t.snapshot));
        }
        if !snaps.iter().any(|s| s.name == t.snapshot) {
            return Err(format!("スナップショット {:?} が無い", t.snapshot));
        }
    }
    let st = shape.stride();
    let mut v = vec![0.0f32; shape.len()];
    for t in terms {
        let s = snaps
            .iter()
            .find(|s| s.name == t.snapshot)
            .expect("存在は上で確認済み");
        // 形はロード時に検証済みだが、念のためもう一度見る。
        if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
            return Err(format!(
                "スナップショット {:?} の形が違う ({}x{}x{} / モデル {}x{}x{})",
                t.snapshot, s.layers, s.n_hc, s.n_embd, shape.n_layer, shape.n_hc, shape.n_embd
            ));
        }
        let c = t.coef;
        for il in from..to {
            let base = il * st;
            let m = &s.mean[base..base + st];
            let dst = &mut v[base..base + st];
            for d in 0..st {
                dst[d] += c * m[d];
            }
        }
    }
    // `v_hat`: (layer, stream) ごとに正規化する。0 ベクトルはそのまま 0。
    let vhat = normalize_layers(shape, &v, from, to);
    Ok(Apply {
        terms: terms.to_vec(),
        from,
        to,
        alpha,
        mode,
        v,
        vhat,
        rules: Vec::new(),
        shape: *shape,
    })
}

/// `v` のうち `from..to` の層を (layer, stream) ごとに正規化する。
/// 範囲外と 0 ベクトルはそのまま 0。
fn normalize_layers(shape: &Shape, v: &[f32], from: usize, to: usize) -> Vec<f32> {
    let st = shape.stride();
    let e = shape.n_embd;
    let mut vhat = vec![0.0f32; v.len()];
    for il in from..to {
        for s in 0..shape.n_hc {
            let base = il * st + s * e;
            let src = &v[base..base + e];
            let mut ss = 0.0f64;
            for &x in src {
                ss += (x as f64) * (x as f64);
            }
            if ss <= 0.0 {
                continue;
            }
            let inv = 1.0 / ss.sqrt();
            let dst = &mut vhat[base..base + e];
            for d in 0..e {
                dst[d] = (src[d] as f64 * inv) as f32;
            }
        }
    }
    vhat
}

/// task 849: 重ね掛けする規則の上限 (壊れた入力でメモリを食い潰さないため)。
pub const MAX_RULES: usize = 16;

/// task 957: 重ね掛けの project を正規直交基底で掛ける (`DENNO_STEER_ORTHO=1`)。
///
/// 既定は off (従来 = 規則を順に掛ける)。on にすると、同じ層に掛かる project 規則の
/// 方向を (layer, stream) ごとに Gram-Schmidt で正規直交化してから掛けるので、
/// `h -= Σ_i alpha_i (h·q_i) q_i` の形になり、規則の順序に依存せず、
/// alpha 1 の規則の方向は残差から厳密に落ちる (2 回掛けても変わらない)。
pub fn ortho_on() -> bool {
    match std::env::var("DENNO_STEER_ORTHO") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// task 957: 重ね掛けの project 方向を (layer, stream) ごとに正規直交化する。
///
/// 各規則の `vortho` に正規化した基底を書く。先行する project 方向と一次従属な
/// (layer, stream) は 0 のまま (その規則はその層では何もしない = 射影の像が同じ)。
/// 正規化は `vhat` と同じく f64 の二乗和から `1/sqrt` を掛けて f32 に丸める。
fn orthogonalize_rules(shape: &Shape, rules: &mut [Rule]) {
    let e = shape.n_embd;
    let st = shape.stride();
    if e == 0 || st == 0 || rules.is_empty() {
        return;
    }
    let mut out = vec![vec![0.0f32; shape.len()]; rules.len()];
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(rules.len());
    let mut x = vec![0.0f32; e];
    for il in 0..shape.n_layer {
        let base = il * st;
        for s in 0..shape.n_hc {
            basis.clear();
            for (ri, r) in rules.iter().enumerate() {
                if r.mode != Mode::Project || !r.covers(il) {
                    continue;
                }
                let src = &r.vhat[base + s * e..base + (s + 1) * e];
                x.copy_from_slice(src);
                // 2 回まわす (f32 の直交性を上げる。値は `vhat` 由来なので安い)。
                for _ in 0..2 {
                    for q in &basis {
                        let mut d = 0.0f32;
                        for k in 0..e {
                            d += x[k] * q[k];
                        }
                        for k in 0..e {
                            x[k] -= d * q[k];
                        }
                    }
                }
                let mut ss = 0.0f64;
                for k in 0..e {
                    ss += (x[k] as f64) * (x[k] as f64);
                }
                // 依存 (残りが 0) ならその層では掛けない (0 ベクトルで何もしない)。
                if ss <= 1e-18 {
                    continue;
                }
                let inv = 1.0 / ss.sqrt();
                let dst = &mut out[ri][base + s * e..base + (s + 1) * e];
                for k in 0..e {
                    dst[k] = (x[k] as f64 * inv) as f32;
                }
                basis.push(dst.to_vec());
            }
        }
    }
    for (r, v) in rules.iter_mut().zip(out) {
        r.vortho = v;
    }
}

/// task 849: 規則ごとの層範囲・alpha・mode を持つ適用を組み立てる。
///
/// 単一規則の [`build_apply`] と同じ規則で各項を検証し、規則ごとに
/// `v` / `vhat` を作る。`Apply` の `terms` は全規則の項を連結したもの、
/// `from` / `to` は全規則の外側の包み、`alpha` / `mode` は先頭規則のもの
/// (実効値は `rules` を見る)。
fn build_rules_apply(
    shape: &Shape,
    snaps: &[Arc<Snapshot>],
    specs: &[RuleSpec],
) -> Result<Apply, String> {
    // task 957: 正規直交化は env ゲート (`DENNO_STEER_ORTHO`)。既定は従来どおり
    // 規則を順に掛けるだけ (task 849 の数値と 1 ビットも変えない)。
    build_rules_apply_opt(shape, snaps, specs, ortho_on())
}

/// task 849 / 957: `ortho` を明示できる版 (テストが env を触らずに両方を確かめる)。
fn build_rules_apply_opt(
    shape: &Shape,
    snaps: &[Arc<Snapshot>],
    specs: &[RuleSpec],
    ortho: bool,
) -> Result<Apply, String> {
    if !shape.is_valid() {
        return Err("モデルの形が未初期化 (steer::init を先に呼ぶ)".into());
    }
    if specs.is_empty() {
        return Err("rules が空 (何も掛けないなら DELETE /v1/steer/apply)".into());
    }
    if specs.len() > MAX_RULES {
        return Err(format!(
            "rules が多すぎる ({} 個 / 上限 {MAX_RULES})",
            specs.len()
        ));
    }
    let st = shape.stride();
    let mut rules = Vec::with_capacity(specs.len());
    let mut combined = vec![0.0f32; shape.len()];
    for (i, sp) in specs.iter().enumerate() {
        if sp.terms.is_empty() {
            return Err(format!("rules[{i}].terms が空"));
        }
        if sp.from > sp.to {
            return Err(format!(
                "rules[{i}].layers の from {} が to {} より大きい",
                sp.from, sp.to
            ));
        }
        if sp.to > shape.n_layer {
            return Err(format!(
                "rules[{i}].layers の to {} が層数 {} を超える ({}-{})",
                sp.to, shape.n_layer, sp.from, shape.n_layer
            ));
        }
        if !sp.alpha.is_finite() {
            return Err(format!("rules[{i}].alpha が有限でない"));
        }
        for t in &sp.terms {
            if !t.coef.is_finite() {
                return Err(format!("rules[{i}].coef が有限でない ({})", t.snapshot));
            }
            if !snaps.iter().any(|s| s.name == t.snapshot) {
                return Err(format!("スナップショット {:?} が無い", t.snapshot));
            }
        }
        let mut v = vec![0.0f32; shape.len()];
        for t in &sp.terms {
            let s = snaps
                .iter()
                .find(|s| s.name == t.snapshot)
                .expect("存在は上で確認済み");
            if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
                return Err(format!(
                    "スナップショット {:?} の形が違う ({}x{}x{} / モデル {}x{}x{})",
                    t.snapshot, s.layers, s.n_hc, s.n_embd, shape.n_layer, shape.n_hc, shape.n_embd
                ));
            }
            let c = t.coef;
            for il in sp.from..sp.to {
                let base = il * st;
                let m = &s.mean[base..base + st];
                let dst = &mut v[base..base + st];
                for d in 0..st {
                    dst[d] += c * m[d];
                }
            }
        }
        let vhat = normalize_layers(shape, &v, sp.from, sp.to);
        for (c, x) in combined.iter_mut().zip(v.iter()) {
            *c += *x;
        }
        rules.push(Rule {
            terms: sp.terms.clone(),
            from: sp.from,
            to: sp.to,
            alpha: sp.alpha,
            mode: sp.mode,
            v,
            vhat,
            vortho: Vec::new(),
        });
    }
    if ortho {
        orthogonalize_rules(shape, &mut rules);
    }
    let from = rules.iter().map(|r| r.from).min().unwrap_or(0);
    let to = rules.iter().map(|r| r.to).max().unwrap_or(0);
    let terms = rules.iter().flat_map(|r| r.terms.iter().cloned()).collect();
    let vhat = normalize_layers(shape, &combined, from, to);
    Ok(Apply {
        terms,
        from,
        to,
        alpha: rules[0].alpha,
        mode: rules[0].mode,
        v: combined,
        vhat,
        rules,
        shape: *shape,
    })
}

fn install_apply(a: Apply) -> Arc<Steer> {
    let epoch = super::intervene::bump_epoch();
    let mut g = cur_cell().write().unwrap_or_else(|e| e.into_inner());
    let s = Arc::new(Steer {
        epoch,
        apply: Some(a),
    });
    *g = Arc::clone(&s);
    // task 957: ホットパスの入口 (`apply_on`) を先に立ててから差し替える。
    APPLY_ON.store(true, Ordering::Release);
    s
}

/// 適用を設定する (epoch が進む)。
pub fn set_apply(
    terms: Vec<Term>,
    from: usize,
    to: usize,
    alpha: f32,
    mode: Mode,
) -> Result<Arc<Steer>, String> {
    let (shape, _) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let a = build_apply(&shape, &snaps, &terms, from, to, alpha, mode)?;
    Ok(install_apply(a))
}

/// task 849: 複数概念の重ね掛けを設定する (epoch が進む)。
///
/// 単一規則の [`set_apply`] はそのまま (この関数を通らない) なので、
/// 従来の API と数値は 1 ビットも変わらない。
pub fn set_rules(specs: Vec<RuleSpec>) -> Result<Arc<Steer>, String> {
    let (shape, _) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let a = build_rules_apply(&shape, &snaps, &specs)?;
    Ok(install_apply(a))
}

/// 適用を消す (epoch が進む)。
pub fn clear_apply() -> Arc<Steer> {
    let epoch = super::intervene::bump_epoch();
    let mut g = cur_cell().write().unwrap_or_else(|e| e.into_inner());
    let s = Arc::new(Steer::empty(epoch));
    *g = Arc::clone(&s);
    // task 957: ホットパスの入口を落とす (次からは `active()` がロックを取らない)。
    APPLY_ON.store(false, Ordering::Release);
    s
}

// ---------------------------------------------------------------------------
// 記録
// ---------------------------------------------------------------------------

struct Rec {
    name: String,
    model: String,
    shape: Shape,
    /// task 829: 残差の取り方 (`All` なら従来の平均)。
    mode: CaptureMode,
    /// `[layer][stream][n_embd]`
    sums: Vec<f32>,
    /// 層ごとのトークン数 (通常は全部同じ)。
    counts: Vec<u64>,
    /// 常駐 prefill の連鎖がデバイスへ足したトークン数。
    dev_tokens: u64,
    /// デバイスの加算域を使ったか。
    dev_used: bool,
}

impl Rec {
    fn new(name: &str, shape: Shape, model: &str, mode: CaptureMode) -> Rec {
        Rec {
            name: name.to_string(),
            model: model.to_string(),
            shape,
            mode,
            sums: vec![0.0f32; shape.len()],
            counts: vec![0; shape.n_layer],
            dev_tokens: 0,
            dev_used: false,
        }
    }

    fn tokens(&self) -> u64 {
        let host = self.counts.iter().copied().min().unwrap_or(0);
        host.saturating_add(self.dev_tokens)
    }

    /// `data` は `[rows][stream][n_embd]` の連続領域。
    fn acc(&mut self, layer: usize, data: &[f32]) {
        let st = self.shape.stride();
        if st == 0 || layer >= self.shape.n_layer {
            return;
        }
        let rows = data.len() / st;
        if rows == 0 {
            return;
        }
        let base = layer * st;
        let dst = &mut self.sums[base..base + st];
        for t in 0..rows {
            let src = &data[t * st..(t + 1) * st];
            for d in 0..st {
                dst[d] += src[d];
            }
        }
        self.counts[layer] += rows as u64;
    }
}

/// 記録中か。ホットパスはこれ 1 個だけ見る。
#[inline]
pub fn rec_on() -> bool {
    REC_ON.load(Ordering::Relaxed)
}

/// task 829: 記録中で、かつ「prefill の最後のトークンだけ」の取り方か。
/// ホットパスは `rec_on()` の内側でこれを見るだけ (AtomicU8 のロード 1 回)。
#[inline]
pub fn last_only() -> bool {
    REC_MODE.load(Ordering::Relaxed) == 1
}

/// task 829: いまの (記録中でなければ直近の) [`CaptureMode`]。
pub fn capture_mode() -> CaptureMode {
    if rec_on() {
        CaptureMode::from_code(REC_MODE.load(Ordering::Relaxed))
    } else {
        CaptureMode::from_code(LAST_MODE.load(Ordering::Relaxed))
    }
}

/// ホストの残差を 1 層ぶん記録する (`data` = 1 トークンの `[hc][n_embd]`)。
///
/// decode と非常駐 prefill が呼ぶ。[`rec_on`] が false なら即返る。
/// [`CaptureMode::LastPromptToken`] では decode のトークンを載せないので、
/// この入口 (位置を持たない = decode 扱い) では何もしない。
pub fn acc_host(layer: usize, data: &[f32]) {
    if !rec_on() {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.as_mut() {
        if r.mode == CaptureMode::All {
            r.acc(layer, data);
        }
    }
}

/// task 829: prefill の 1 トークン (`t` = その prefill 内の位置、`nt` = 全長)。
///
/// [`CaptureMode::All`] は [`acc_host`] と同じ (全トークンを足す)。
/// [`CaptureMode::LastPromptToken`] は `t + 1 == nt` のときだけ記録する。
pub fn acc_host_prefill(layer: usize, data: &[f32], t: usize, nt: usize) {
    if !rec_on() {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.as_mut() {
        match r.mode {
            CaptureMode::All => r.acc(layer, data),
            CaptureMode::LastPromptToken => {
                if nt > 0 && t + 1 == nt {
                    r.acc(layer, data);
                }
            }
        }
    }
}

/// ホストの残差をチャンクぶん (行の配列) 記録する。
///
/// [`CaptureMode::LastPromptToken`] では `final_chunk` が真のときだけ、
/// チャンクの最後の行 (= prefill の最終トークン) を記録する。
pub fn acc_host_rows(layer: usize, rows: &[&[f32]], final_chunk: bool) {
    if !rec_on() {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = g.as_mut() else { return };
    match r.mode {
        CaptureMode::All => {
            for row in rows {
                r.acc(layer, row);
            }
        }
        CaptureMode::LastPromptToken => {
            if final_chunk {
                if let Some(row) = rows.last() {
                    r.acc(layer, row);
                }
            }
        }
    }
}

/// 常駐 prefill の連鎖が `nt` トークンをデバイスへ足したことを記録する。
///
/// 合計は [`flush`] / [`stop_record`] でデバイスから降ろして足す。
pub fn note_dev_tokens(nt: u64) {
    if !rec_on() || nt == 0 {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.as_mut() {
        r.dev_tokens += nt;
        r.dev_used = true;
    }
}

/// 記録中の名前と現時点のトークン数。
pub fn rec_status() -> Option<(String, u64)> {
    if !rec_on() {
        return None;
    }
    let g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map(|r| (r.name.clone(), r.tokens()))
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("name が空".into());
    }
    if name.len() > MAX_NAME {
        return Err(format!("name が長すぎる (最大 {MAX_NAME} 文字)"));
    }
    for ch in name.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || ch == '-';
        if !ok {
            return Err(format!(
                "name に使えない文字 {ch:?} (A-Z a-z 0-9 _ - のみ)"
            ));
        }
    }
    if name.starts_with('.') {
        return Err("name を '.' で始めない".into());
    }
    Ok(())
}

/// 記録を始める。既に記録中ならエラー (取り違えを防ぐ)。
///
/// 従来どおり [`CaptureMode::All`] (全トークンの平均) で始める。
pub fn start_record(name: &str) -> Result<(String, u64), String> {
    start_record_mode(name, CaptureMode::All)
}

/// task 829: [`CaptureMode`] を指定して記録を始める。
pub fn start_record_mode(name: &str, mode: CaptureMode) -> Result<(String, u64), String> {
    validate_name(name)?;
    let (shape, model) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    if !shape.is_valid() {
        return Err("モデルの形が不正".into());
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.as_ref() {
        return Err(format!(
            "既に記録中 ({}) — 先に stop する",
            r.name
        ));
    }
    *g = Some(Rec::new(name, shape, &model, mode));
    REC_MODE.store(mode.code(), Ordering::SeqCst);
    LAST_MODE.store(mode.code(), Ordering::SeqCst);
    REC_ON.store(true, Ordering::SeqCst);
    // 次の常駐 prefill の入口でデバイスの加算域を 0 にし直す。
    REC_DEV_RESET.store(true, Ordering::SeqCst);
    if let Some((n, t)) = rec_status_in(&g) {
        Ok((n, t))
    } else {
        Ok((name.to_string(), 0))
    }
}

fn rec_status_in(g: &Option<Rec>) -> Option<(String, u64)> {
    g.as_ref().map(|r| (r.name.clone(), r.tokens()))
}

/// 記録を止めて平均を保存する。
pub fn stop_record() -> Result<Snapshot, String> {
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut r) = g.take() else {
        return Err("記録していない".into());
    };
    // 先にフラグを落とす (ここから先の acc は捨てる)。
    REC_ON.store(false, Ordering::SeqCst);
    REC_ENV.store(false, Ordering::SeqCst);
    merge_dev(&mut r);
    let snap = snapshot_of(&r)?;
    save_snapshot(&snap)?;
    push_snapshot(Arc::new(snap.clone()));
    Ok(snap)
}

/// 記録を続けたまま、いまの平均をディスクへ書く (env の「毎リクエスト保存」)。
pub fn flush() -> Result<Snapshot, String> {
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = g.as_mut() else {
        return Err("記録していない".into());
    };
    merge_dev(r);
    let snap = snapshot_of(r)?;
    save_snapshot(&snap)?;
    push_snapshot(Arc::new(snap.clone()));
    Ok(snap)
}

fn snapshot_of(r: &Rec) -> Result<Snapshot, String> {
    let tokens = r.tokens();
    if tokens == 0 {
        return Err("記録したトークンが 0".into());
    }
    if !r.counts.iter().all(|&c| c == r.counts[0]) {
        eprintln!(
            "[steer] 警告: 層ごとのトークン数が揃っていない (min {} / max {:?})",
            r.counts.iter().copied().min().unwrap_or(0),
            r.counts.iter().copied().max()
        );
    }
    let inv = 1.0f32 / tokens as f32;
    let mut mean = vec![0.0f32; r.shape.len()];
    for i in 0..mean.len() {
        mean[i] = r.sums[i] * inv;
    }
    Ok(Snapshot {
        name: r.name.clone(),
        tokens,
        created: now_secs(),
        model: r.model.clone(),
        layers: r.shape.n_layer,
        n_hc: r.shape.n_hc,
        n_embd: r.shape.n_embd,
        mean,
    })
}

/// デバイス側の加算域を降ろして `sums` に足し、0 に戻す。
///
/// 常駐 prefill の終端 (推論スレッド) と [`flush`] / [`stop_record`] の両方から
/// 呼べる。加算域は thread_local なので、**加算したスレッドで降ろす**のが原則
/// (常駐 prefill の終端で 1 リクエスト 1 回 = 2 MB の D2H)。
pub fn merge_device() {
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = g.as_mut() else { return };
    merge_dev(r);
}

fn merge_dev(r: &mut Rec) {
    if !r.dev_used || r.dev_tokens == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    {
        let mut buf = vec![0.0f32; r.shape.len()];
        if crate::cuda::pf_steer_acc_take(&mut buf) {
            for i in 0..buf.len() {
                r.sums[i] += buf[i];
            }
            r.dev_used = false;
        } else {
            eprintln!("[steer] 警告: デバイスの加算域を降ろせなかった (0 として扱う)");
            r.dev_used = false;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        r.dev_used = false;
    }
}

fn push_snapshot(s: Arc<Snapshot>) {
    let mut g = snaps_cell().write().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<Arc<Snapshot>> = g.iter().filter(|x| x.name != s.name).cloned().collect();
    v.push(s);
    v.sort_by(|a, b| a.name.cmp(&b.name));
    while v.len() > MAX_SNAPSHOTS {
        v.remove(0);
    }
    *g = Arc::new(v);
}

/// スナップショットの一覧 (名前順)。
pub fn snapshots() -> Arc<Vec<Arc<Snapshot>>> {
    let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
    Arc::clone(&g)
}

/// 適用中の設定。
pub fn apply_status() -> Option<ApplyInfo> {
    snapshot().apply.as_ref().map(|a| ApplyInfo {
        terms: a.terms.clone(),
        from: a.from,
        to: a.to,
        alpha: a.alpha,
        mode: a.mode,
        rules: a
            .rules
            .iter()
            .map(|r| RuleInfo {
                terms: r.terms.clone(),
                from: r.from,
                to: r.to,
                alpha: r.alpha,
                mode: r.mode,
            })
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// ディスク
// ---------------------------------------------------------------------------

fn snap_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{FILE_EXT}"))
}

fn save_snapshot(s: &Snapshot) -> Result<(), String> {
    let dir = dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let buf = encode_snapshot(s);
    let path = snap_path(&dir, &s.name);
    let tmp = dir.join(format!(".{}.{}.tmp", s.name, std::process::id()));
    std::fs::write(&tmp, &buf).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// 1 ファイル読む。壊れていれば `Err`。
pub fn load_file(path: &Path) -> Result<Snapshot, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_snapshot_impl(&path.display().to_string(), &raw)
}

/// DNSTEER1 の本体を復号する。`label` はエラー文に出す出所。
fn decode_snapshot_impl(label: &str, raw: &[u8]) -> Result<Snapshot, String> {
    let magic = FILE_MAGIC.as_bytes();
    if raw.len() < magic.len() + 1 || &raw[..magic.len()] != magic || raw[magic.len()] != b'\n' {
        return Err(format!("{label}: 先頭が {FILE_MAGIC} でない"));
    }
    // ヘッダは ASCII の行。`data\n` の直後から f32 が並ぶ。
    let mut off = magic.len() + 1;
    let mut name = String::new();
    let mut tokens = 0u64;
    let mut created = 0u64;
    let mut model = String::new();
    let mut layers = 0usize;
    let mut n_hc = 0usize;
    let mut n_embd = 0usize;
    loop {
        let end = raw[off..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| off + p)
            .ok_or_else(|| format!("{label}: ヘッダが終わらない"))?;
        let line = std::str::from_utf8(&raw[off..end])
            .map_err(|_| format!("{label}: ヘッダが ASCII でない"))?;
        off = end + 1;
        if line == "data" {
            break;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| format!("{label}: ヘッダ行 {line:?} に '=' が無い"))?;
        let num = |v: &str| -> Result<u64, String> {
            v.parse::<u64>()
                .map_err(|_| format!("{label}: {k} が数値でない: {v:?}"))
        };
        match k {
            "name" => name = v.to_string(),
            "tokens" => tokens = num(v)?,
            "created" => created = num(v)?,
            "model" => model = v.to_string(),
            "layers" => layers = num(v)? as usize,
            "n_hc" => n_hc = num(v)? as usize,
            "n_embd" => n_embd = num(v)? as usize,
            other => return Err(format!("{label}: 未知のヘッダ {other:?}")),
        }
    }
    let shape = Shape {
        n_layer: layers,
        n_hc,
        n_embd,
    };
    if !shape.is_valid() {
        return Err(format!("{label}: 形が不正"));
    }
    let want = shape.len() * 4;
    let have = raw.len() - off;
    if have != want {
        return Err(format!(
            "{label}: 本体の長さが違う ({have} バイト / 期待 {want})"
        ));
    }
    let mut mean = vec![0.0f32; shape.len()];
    for (i, c) in raw[off..].chunks_exact(4).enumerate() {
        mean[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(Snapshot {
        name,
        tokens,
        created,
        model,
        layers,
        n_hc,
        n_embd,
        mean,
    })
}

/// `DENNO_STEER_DIR` の `*.dns` を全部読む。
///
/// 形かモデル id が違うものは**読み込まない** (警告 1 行)。
pub fn reload_snapshots() {
    let Some((shape, model)) = meta() else {
        return;
    };
    let dir = dir();
    let mut out: Vec<Arc<Snapshot>> = Vec::new();
    let rd = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => {
            // ディレクトリが無いのは正常 (まだ 1 本も記録していない)。
            let mut g = snaps_cell().write().unwrap_or_else(|e| e.into_inner());
            *g = Arc::new(out);
            return;
        }
    };
    let mut paths: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == FILE_EXT).unwrap_or(false))
        .collect();
    paths.sort();
    for p in paths {
        match load_file(&p) {
            Ok(s) => {
                if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
                    eprintln!(
                        "[steer] {} を無視: 形が違う ({}x{}x{} / モデル {}x{}x{})",
                        p.display(),
                        s.layers,
                        s.n_hc,
                        s.n_embd,
                        shape.n_layer,
                        shape.n_hc,
                        shape.n_embd
                    );
                    continue;
                }
                if !model.is_empty() && !s.model.is_empty() && s.model != model {
                    eprintln!(
                        "[steer] {} を無視: モデル id が違う ({:?} / {:?})",
                        p.display(),
                        s.model,
                        model
                    );
                    continue;
                }
                out.push(Arc::new(s));
            }
            Err(e) => eprintln!("[steer] {e}"),
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    let mut g = snaps_cell().write().unwrap_or_else(|e| e.into_inner());
    *g = Arc::new(out);
}

/// スナップショットを消す (ディスクとメモリの両方)。
pub fn delete(name: &str) -> Result<(), String> {
    validate_name(name)?;
    let path = snap_path(&dir(), name);
    let existed = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        g.iter().any(|s| s.name == name)
    };
    let mut removed = false;
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        removed = true;
    }
    if existed {
        let mut g = snaps_cell().write().unwrap_or_else(|e| e.into_inner());
        let v: Vec<Arc<Snapshot>> = g.iter().filter(|s| s.name != name).cloned().collect();
        *g = Arc::new(v);
        removed = true;
    }
    if !removed {
        return Err(format!("スナップショット {name:?} が無い"));
    }
    // 消したスナップショットを使っている適用は落とす (次のリクエストから効く)。
    let using = {
        let g = cur_cell().read().unwrap_or_else(|e| e.into_inner());
        g.apply
            .as_ref()
            .map(|a| a.terms.iter().any(|t| t.snapshot == name))
            .unwrap_or(false)
    };
    if using {
        clear_apply();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// task 825: 対比の分析 (analyze) とエクスポート / インポート
// ---------------------------------------------------------------------------
//
// abliteration の作業台 (`/2d` の Abliterate タブ) が使う:
//
// - `analyze`: 2 つの平均 (Set A / Set B) の層ごとの差 `d_l = mu_l(A) - mu_l(B)`
//   と、その大きさ / 向き / 分離度を出す。分離度が高い層の連続区間を
//   `recommended_range` として返す (層 0 と最終層は避ける)。
// - `export` / `import`: 抽出したスナップショットをブラウザへ持ち出し、
//   再起動後や別のプロセスへ戻せるようにする。形式は既存の DNSTEER1
//   バイナリ (そのまま / base64) と、`mean` 配列を持つ JSON。
//
// 数値: ノルムと内積は f64 で足し、結果だけ f32 に丸める。`cos_sim` は
// どちらかのノルムが 0 なら 0 (未定義を NaN にしない)。`sep_score` は
// `||d|| / (||a|| + ||b|| + 1e-12)` で、0..1 に収まる (三角不等式)。

/// 層 1 枚ぶんの対比の指標。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerSep {
    pub layer: usize,
    /// `||mu_A - mu_B||`
    pub diff_norm: f32,
    /// `||mu_A||`
    pub norm_pos: f32,
    /// `||mu_B||`
    pub norm_neg: f32,
    /// `cos(mu_A, mu_B)`
    pub cos: f32,
    /// `||d|| / (||a|| + ||b|| + eps)` (0..1)
    pub sep: f32,
}

/// `analyze` の結果。
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub pos: String,
    pub neg: String,
    pub layers: Vec<LayerSep>,
    /// `sep` が最大の層 (推奨区間の中心)。
    pub peak_layer: usize,
    /// `[from, to)` (to は含まない)。層 0 と最終層は除く。
    pub recommended_range: (usize, usize),
}

/// 層ごとの指標を計算する。`a` と `b` は `[layer][stream][n_embd]` の連続領域。
pub fn layer_metrics(shape: &Shape, a: &[f32], b: &[f32]) -> Result<Vec<LayerSep>, String> {
    let st = shape.stride();
    if !shape.is_valid() || st == 0 {
        return Err("モデルの形が不正".into());
    }
    if a.len() != shape.len() || b.len() != shape.len() {
        return Err(format!(
            "ベクトルの長さが違う ({} / {} / 期待 {})",
            a.len(),
            b.len(),
            shape.len()
        ));
    }
    let mut out = Vec::with_capacity(shape.n_layer);
    for il in 0..shape.n_layer {
        let base = il * st;
        let (mut na2, mut nb2, mut dot, mut dn2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in 0..st {
            let x = a[base + d] as f64;
            let y = b[base + d] as f64;
            na2 += x * x;
            nb2 += y * y;
            dot += x * y;
            let e = x - y;
            dn2 += e * e;
        }
        let na = na2.sqrt();
        let nb = nb2.sqrt();
        let dn = dn2.sqrt();
        let cos = if na > 0.0 && nb > 0.0 {
            (dot / (na * nb)).clamp(-1.0, 1.0) as f32
        } else {
            0.0
        };
        let sep = (dn / (na + nb + 1e-12)) as f32;
        out.push(LayerSep {
            layer: il,
            diff_norm: dn as f32,
            norm_pos: na as f32,
            norm_neg: nb as f32,
            cos,
            sep,
        });
    }
    Ok(out)
}

/// 分離の高い層の連続区間 `[from, to)` を選ぶ。
///
/// 層 0 と最終層は対象から外す (層 0 の出口はまだ層を通っておらず、最終層は
/// ヘッドの直前で、ablation の文献でも避けるのが普通)。ピークの半分以上の
/// 分離が続く範囲を連続窓として広げる。分離が全部 0 なら内部全体を返す。
pub fn recommended_range(sep: &[f32]) -> (usize, usize) {
    let n = sep.len();
    if n == 0 {
        return (0, 0);
    }
    if n < 3 {
        return (0, n);
    }
    let lo = 1usize;
    let hi = n - 1; // exclusive
    let mut peak = lo;
    let mut best = sep[lo];
    for (i, &v) in sep.iter().enumerate().take(hi).skip(lo) {
        if v > best {
            best = v;
            peak = i;
        }
    }
    if !(best > 0.0) {
        return (lo, hi);
    }
    let thr = best * 0.5;
    let mut from = peak;
    while from > lo && sep[from - 1] >= thr {
        from -= 1;
    }
    let mut to = peak + 1;
    while to < hi && sep[to] >= thr {
        to += 1;
    }
    (from, to)
}

fn finish_analysis(pos: &str, neg: &str, layers: Vec<LayerSep>) -> Analysis {
    let seps: Vec<f32> = layers.iter().map(|x| x.sep).collect();
    let peak = layers
        .iter()
        .enumerate()
        .take(layers.len().saturating_sub(1))
        .skip(1)
        .max_by(|a, b| a.1.sep.partial_cmp(&b.1.sep).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Analysis {
        pos: pos.to_string(),
        neg: neg.to_string(),
        layers,
        peak_layer: peak,
        recommended_range: recommended_range(&seps),
    }
}

fn find_snapshot(snaps: &[Arc<Snapshot>], name: &str) -> Result<Arc<Snapshot>, String> {
    snaps
        .iter()
        .find(|s| s.name == name)
        .cloned()
        .ok_or_else(|| format!("スナップショット {name:?} が無い"))
}

/// 2 つの名前付きスナップショットの対比を分析する。
pub fn analyze(pos: &str, neg: &str) -> Result<Analysis, String> {
    let (shape, _) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let a = find_snapshot(&snaps, pos)?;
    let b = find_snapshot(&snaps, neg)?;
    if a.layers != shape.n_layer || a.n_hc != shape.n_hc || a.n_embd != shape.n_embd {
        return Err(format!("スナップショット {pos:?} の形がモデルと違う"));
    }
    if b.layers != shape.n_layer || b.n_hc != shape.n_hc || b.n_embd != shape.n_embd {
        return Err(format!("スナップショット {neg:?} の形がモデルと違う"));
    }
    let layers = layer_metrics(&shape, &a.mean, &b.mean)?;
    Ok(finish_analysis(pos, neg, layers))
}

/// 任意の項の組 (`coef` の正が A 側、負が B 側) の対比を分析する。
///
/// `terms: [{snapshot:"ablit_A",coef:1},{snapshot:"ablit_B",coef:-1}]` を
/// そのまま渡せる。片側が空ならエラー。
pub fn analyze_terms(terms: &[Term]) -> Result<Analysis, String> {
    let (shape, _) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let mut a = vec![0.0f32; shape.len()];
    let mut b = vec![0.0f32; shape.len()];
    let mut n_pos = 0;
    let mut n_neg = 0;
    for t in terms {
        if !t.coef.is_finite() {
            return Err(format!("coef が有限でない ({})", t.snapshot));
        }
        let s = find_snapshot(&snaps, &t.snapshot)?;
        if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
            return Err(format!("スナップショット {:?} の形がモデルと違う", t.snapshot));
        }
        if t.coef > 0.0 {
            n_pos += 1;
            for (d, x) in a.iter_mut().zip(s.mean.iter()) {
                *d += t.coef * *x;
            }
        } else if t.coef < 0.0 {
            n_neg += 1;
            let c = -t.coef;
            for (d, x) in b.iter_mut().zip(s.mean.iter()) {
                *d += c * *x;
            }
        }
    }
    if n_pos == 0 || n_neg == 0 {
        return Err("正の係数と負の係数の両方が要る (例: A +1, B -1)".into());
    }
    let layers = layer_metrics(&shape, &a, &b)?;
    let pos = terms
        .iter()
        .filter(|t| t.coef > 0.0)
        .map(|t| t.snapshot.clone())
        .collect::<Vec<_>>()
        .join("+");
    let neg = terms
        .iter()
        .filter(|t| t.coef < 0.0)
        .map(|t| t.snapshot.clone())
        .collect::<Vec<_>>()
        .join("+");
    Ok(finish_analysis(&pos, &neg, layers))
}

/// メモリ上の DNSTEER1 を復号する ([`load_file`] の中身)。
pub fn decode_snapshot(raw: &[u8]) -> Result<Snapshot, String> {
    decode_snapshot_impl("<memory>", raw)
}

/// DNSTEER1 バイナリを組み立てる ([`save_snapshot`] の中身)。
pub fn encode_snapshot(s: &Snapshot) -> Vec<u8> {
    let mut buf = Vec::with_capacity(s.mean.len() * 4 + 160);
    buf.extend_from_slice(FILE_MAGIC.as_bytes());
    buf.push(b'\n');
    buf.extend_from_slice(format!("name={}\n", s.name).as_bytes());
    buf.extend_from_slice(format!("tokens={}\n", s.tokens).as_bytes());
    buf.extend_from_slice(format!("created={}\n", s.created).as_bytes());
    buf.extend_from_slice(format!("model={}\n", s.model).as_bytes());
    buf.extend_from_slice(format!("layers={}\n", s.layers).as_bytes());
    buf.extend_from_slice(format!("n_hc={}\n", s.n_hc).as_bytes());
    buf.extend_from_slice(format!("n_embd={}\n", s.n_embd).as_bytes());
    buf.extend_from_slice(b"data\n");
    for v in &s.mean {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// 名前付きスナップショットの DNSTEER1 バイナリ (ダウンロード用)。
pub fn export_raw(name: &str) -> Result<Vec<u8>, String> {
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let s = find_snapshot(&snaps, name)?;
    Ok(encode_snapshot(&s))
}

/// 名前付きスナップショットの JSON (`mean` を持たず base64 で本体を載せる)。
pub fn export_json(name: &str) -> Result<serde_json::Value, String> {
    let snaps = {
        let g = snaps_cell().read().unwrap_or_else(|e| e.into_inner());
        Arc::clone(&g)
    };
    let s = find_snapshot(&snaps, name)?;
    let raw = encode_snapshot(&s);
    Ok(serde_json::json!({
        "name": s.name,
        "tokens": s.tokens,
        "created": s.created,
        "model": s.model,
        "layers": s.layers,
        "n_hc": s.n_hc,
        "n_embd": s.n_embd,
        "format": FILE_MAGIC,
        "data_base64": base64_encode(&raw),
    }))
}

/// 復号済みのスナップショットを検証して保存する。`rename` で名前を差し替える。
pub fn import_snapshot(mut s: Snapshot, rename: Option<&str>) -> Result<Snapshot, String> {
    if let Some(n) = rename {
        if !n.is_empty() {
            validate_name(n)?;
            s.name = n.to_string();
        }
    }
    validate_name(&s.name)?;
    let (shape, model) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
        return Err(format!(
            "形が違う ({}x{}x{} / モデル {}x{}x{})",
            s.layers, s.n_hc, s.n_embd, shape.n_layer, shape.n_hc, shape.n_embd
        ));
    }
    if s.mean.len() != shape.len() {
        return Err(format!("本体の長さが違う ({} / 期待 {})", s.mean.len(), shape.len()));
    }
    if !s.mean.iter().all(|x| x.is_finite()) {
        return Err("本体に非有限の値が入っている".into());
    }
    if !model.is_empty() && !s.model.is_empty() && s.model != model {
        return Err(format!(
            "モデル id が違う ({:?} / モデル {:?})",
            s.model, model
        ));
    }
    if s.model.is_empty() {
        s.model = model;
    }
    save_snapshot(&s)?;
    push_snapshot(Arc::new(s.clone()));
    Ok(s)
}

/// DNSTEER1 バイナリをインポートする。
pub fn import_bytes(raw: &[u8], rename: Option<&str>) -> Result<Snapshot, String> {
    let s = decode_snapshot(raw)?;
    import_snapshot(s, rename)
}

/// JSON のインポート。`data_base64` (DNSTEER1 の base64) か `mean` 配列を受ける。
pub fn import_json(v: &serde_json::Value) -> Result<Snapshot, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    let rename = obj.get("name").and_then(|x| x.as_str());
    if let Some(b64) = obj.get("data_base64") {
        let s = b64
            .as_str()
            .ok_or_else(|| "data_base64 は文字列".to_string())?;
        let raw = base64_decode(s)?;
        return import_bytes(&raw, rename);
    }
    let num = |k: &str| -> Result<u64, String> {
        obj.get(k)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| format!("{k} は 0 以上の整数"))
    };
    let layers = num("layers")? as usize;
    let n_hc = num("n_hc")? as usize;
    let n_embd = num("n_embd")? as usize;
    let shape = Shape {
        n_layer: layers,
        n_hc,
        n_embd,
    };
    if !shape.is_valid() {
        return Err("形が不正 (layers / n_hc / n_embd)".into());
    }
    let arr = obj
        .get("mean")
        .ok_or_else(|| "mean か data_base64 が要る".to_string())?
        .as_array()
        .ok_or_else(|| "mean は配列".to_string())?;
    if arr.len() != shape.len() {
        return Err(format!("mean の長さが違う ({} / 期待 {})", arr.len(), shape.len()));
    }
    let mut mean = Vec::with_capacity(arr.len());
    for (i, x) in arr.iter().enumerate() {
        let f = x
            .as_f64()
            .ok_or_else(|| format!("mean[{i}] が数値でない"))?;
        if !f.is_finite() || f.abs() > 1e30 {
            return Err(format!("mean[{i}] が有限の範囲に無い"));
        }
        mean.push(f as f32);
    }
    let name = rename.ok_or_else(|| "name が要る".to_string())?;
    let s = Snapshot {
        name: name.to_string(),
        tokens: obj.get("tokens").and_then(|x| x.as_u64()).unwrap_or(0),
        created: obj
            .get("created")
            .and_then(|x| x.as_u64())
            .filter(|&c| c > 0)
            .unwrap_or_else(now_secs),
        model: obj
            .get("model")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        layers,
        n_hc,
        n_embd,
        mean,
    };
    import_snapshot(s, None)
}

/// 標準の base64 (RFC 4648, `+/`, `=` 詰め)。外部 crate を足さないための実装。
pub fn base64_encode(raw: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((raw.len() + 2) / 3 * 4);
    for c in raw.chunks(3) {
        let b0 = c[0] as u32;
        let b1 = *c.get(1).unwrap_or(&0) as u32;
        let b2 = *c.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// base64 を復号する。空白は無視し、`=` で終わる。不正は `Err`。
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 3);
    for (i, ch) in s.bytes().enumerate() {
        let v = match ch {
            b'A'..=b'Z' => ch - b'A',
            b'a'..=b'z' => ch - b'a' + 26,
            b'0'..=b'9' => ch - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => {
                return Err(format!(
                    "base64 に使えない文字 {:?} (位置 {i})",
                    ch as char
                ))
            }
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    if bits > 4 {
        return Err("base64 の長さが不正".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// task 833: 持ち運びできるレシピ (SteerRecipe)
// ---------------------------------------------------------------------------
//
// `/2d` で見つけた (snapshot, layer range, alpha, mode) の組を 1 ファイルに
// まとめ、`denno serve-flash --steer-recipe <path>` で起動時に掛け直せるように
// する。スナップショットは `$DENNO_STEER_DIR/<name>.dns` を参照してもよいし、
// `snapshots` に base64 で埋め込んで 1 ファイルで完結してもよい。
//
// JSON (version 1):
//
// ```json
// {
//   "version": 1,
//   "name": "no_refusal",
//   "description": "任意",
//   "model_id": "flash-next-UD-IQ3_XXS",
//   "mode": "project",
//   "alpha": 0.5,
//   "layer_from": 8,
//   "layer_to": 32,
//   "terms": [{ "snapshot": "ablit_A", "coef": 1.0 },
//             { "snapshot": "ablit_B", "coef": -1.0 }],
//   "snapshots": { "ablit_A": "<base64 DNSTEER1>" }
// }
// ```
//
// 埋め込みは**メモリにだけ**載せる (`apply_recipe`)。ディスクは触らない。

/// レシピの JSON スキーマ版。
pub const RECIPE_VERSION: u64 = 1;
/// レシピ 1 本の項数の上限 (取り違え防止)。
pub const MAX_RECIPE_TERMS: usize = 16;

/// 適用の材料を 1 ファイルにまとめたもの。
#[derive(Clone, Debug, PartialEq)]
pub struct SteerRecipe {
    pub name: String,
    pub description: Option<String>,
    /// モデル id (任意。空でない同士が食い違ったら適用しない)。
    pub model_id: Option<String>,
    pub mode: Mode,
    pub alpha: f32,
    /// `[layer_from, layer_to)`。
    pub layer_from: usize,
    pub layer_to: usize,
    pub terms: Vec<Term>,
    /// 埋め込みスナップショット (名前順)。
    pub snapshots: Vec<Snapshot>,
}

fn recipe_usize(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<usize, String> {
    let v = obj
        .get(key)
        .ok_or_else(|| format!("{key} が無い"))?
        .as_u64()
        .ok_or_else(|| format!("{key} は 0 以上の整数"))?;
    if v > 1 << 20 {
        return Err(format!("{key} が大きすぎる ({v})"));
    }
    Ok(v as usize)
}

impl SteerRecipe {
    /// JSON を検証して読む。未知のフィールド・非有限・空 terms は `Err`。
    pub fn from_json(v: &serde_json::Value) -> Result<SteerRecipe, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| "レシピは JSON オブジェクト".to_string())?;
        for k in obj.keys() {
            if !matches!(
                k.as_str(),
                "version"
                    | "name"
                    | "description"
                    | "model_id"
                    | "mode"
                    | "alpha"
                    | "layer_from"
                    | "layer_to"
                    | "terms"
                    | "snapshots"
            ) {
                return Err(format!(
                    "未知のフィールド {k:?} (version / name / description / model_id / mode / alpha / layer_from / layer_to / terms / snapshots のみ)"
                ));
            }
        }
        let version = obj
            .get("version")
            .ok_or_else(|| "version が無い (1)".to_string())?
            .as_u64()
            .ok_or_else(|| "version は 1 (整数)".to_string())?;
        if version != RECIPE_VERSION {
            return Err(format!(
                "version {version} は未対応 ({RECIPE_VERSION} のみ)"
            ));
        }
        let name = obj
            .get("name")
            .and_then(|x| x.as_str())
            .ok_or_else(|| "name は文字列".to_string())?;
        validate_name(name)?;
        let description = match obj.get("description") {
            None | Some(serde_json::Value::Null) => None,
            Some(x) => Some(
                x.as_str()
                    .ok_or_else(|| "description は文字列".to_string())?
                    .to_string(),
            ),
        };
        let model_id = match obj.get("model_id") {
            None | Some(serde_json::Value::Null) => None,
            Some(x) => {
                let s = x
                    .as_str()
                    .ok_or_else(|| "model_id は文字列".to_string())?;
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            }
        };
        let mode_s = obj
            .get("mode")
            .and_then(|x| x.as_str())
            .ok_or_else(|| "mode は文字列 (project / subtract)".to_string())?;
        let mode = Mode::parse(mode_s)
            .ok_or_else(|| format!("未知の mode {mode_s:?} (project / subtract)"))?;
        let alpha = match obj.get("alpha") {
            Some(x) => {
                let f = x.as_f64().ok_or_else(|| "alpha は数値".to_string())?;
                if !f.is_finite() || f.abs() > 1e30 {
                    return Err("alpha が有限の範囲に無い".into());
                }
                f as f32
            }
            None => 1.0,
        };
        if !alpha.is_finite() {
            return Err("alpha が有限でない".into());
        }
        let layer_from = recipe_usize(obj, "layer_from")?;
        let layer_to = recipe_usize(obj, "layer_to")?;
        if layer_from > layer_to {
            return Err(format!(
                "layers の from {layer_from} が to {layer_to} より大きい"
            ));
        }
        let tv = obj
            .get("terms")
            .ok_or_else(|| "terms が無い".to_string())?
            .as_array()
            .ok_or_else(|| "terms は配列".to_string())?;
        if tv.is_empty() {
            return Err("terms が空 (掛けないレシピは作らない)".into());
        }
        if tv.len() > MAX_RECIPE_TERMS {
            return Err(format!(
                "terms が多すぎる ({} > {MAX_RECIPE_TERMS})",
                tv.len()
            ));
        }
        let mut terms = Vec::with_capacity(tv.len());
        for (i, t) in tv.iter().enumerate() {
            let o = t
                .as_object()
                .ok_or_else(|| format!("terms[{i}] はオブジェクト"))?;
            for k in o.keys() {
                if !matches!(k.as_str(), "snapshot" | "coef") {
                    return Err(format!("terms[{i}]: 未知のフィールド {k:?}"));
                }
            }
            let snap = o
                .get("snapshot")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("terms[{i}].snapshot は文字列"))?;
            validate_name(snap)?;
            let coef = o
                .get("coef")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| format!("terms[{i}].coef は数値"))?;
            if !coef.is_finite() || coef.abs() > 1e30 {
                return Err(format!("terms[{i}].coef が有限の範囲に無い"));
            }
            terms.push(Term {
                snapshot: snap.to_string(),
                coef: coef as f32,
            });
        }
        let mut snapshots: Vec<Snapshot> = Vec::new();
        if let Some(sv) = obj.get("snapshots") {
            let m = sv
                .as_object()
                .ok_or_else(|| "snapshots は { name: base64 } のオブジェクト".to_string())?;
            for (key, val) in m {
                validate_name(key)?;
                let b64 = val
                    .as_str()
                    .ok_or_else(|| format!("snapshots[{key:?}] は base64 文字列"))?;
                let raw = base64_decode(b64)
                    .map_err(|e| format!("snapshots[{key:?}]: {e}"))?;
                let s = decode_snapshot(&raw)
                    .map_err(|e| format!("snapshots[{key:?}]: {e}"))?;
                if s.name != *key {
                    return Err(format!(
                        "snapshots[{key:?}] の中の名前が違う ({:?})",
                        s.name
                    ));
                }
                if !s.mean.iter().all(|x| x.is_finite()) {
                    return Err(format!("snapshots[{key:?}] に非有限の値が入っている"));
                }
                snapshots.push(s);
            }
            snapshots.sort_by(|a, b| a.name.cmp(&b.name));
            if let Some(first) = snapshots.first() {
                for s in &snapshots[1..] {
                    if (s.layers, s.n_hc, s.n_embd)
                        != (first.layers, first.n_hc, first.n_embd)
                    {
                        return Err(format!(
                            "埋め込みスナップショットの形が揃っていない ({:?} / {:?})",
                            first.name, s.name
                        ));
                    }
                }
            }
        }
        Ok(SteerRecipe {
            name: name.to_string(),
            description,
            model_id,
            mode,
            alpha,
            layer_from,
            layer_to,
            terms,
            snapshots,
        })
    }

    /// レシピ → JSON (`to_json` → 文字列 → `from_json` で同じものに戻る)。
    pub fn to_json(&self) -> serde_json::Value {
        let mut o = serde_json::Map::new();
        o.insert("version".into(), serde_json::json!(RECIPE_VERSION));
        o.insert("name".into(), serde_json::json!(self.name));
        if let Some(d) = &self.description {
            o.insert("description".into(), serde_json::json!(d));
        }
        if let Some(m) = &self.model_id {
            o.insert("model_id".into(), serde_json::json!(m));
        }
        o.insert("mode".into(), serde_json::json!(self.mode.name()));
        o.insert("alpha".into(), serde_json::json!(self.alpha));
        o.insert("layer_from".into(), serde_json::json!(self.layer_from));
        o.insert("layer_to".into(), serde_json::json!(self.layer_to));
        o.insert(
            "terms".into(),
            serde_json::json!(self
                .terms
                .iter()
                .map(|t| serde_json::json!({"snapshot": t.snapshot, "coef": t.coef}))
                .collect::<Vec<_>>()),
        );
        if !self.snapshots.is_empty() {
            let mut m = serde_json::Map::new();
            for s in &self.snapshots {
                m.insert(
                    s.name.clone(),
                    serde_json::json!(base64_encode(&encode_snapshot(s))),
                );
            }
            o.insert("snapshots".into(), serde_json::Value::Object(m));
        }
        serde_json::Value::Object(o)
    }

    /// ファイルを読んで検証する。
    pub fn from_file(path: &Path) -> Result<SteerRecipe, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: JSON でない: {e}", path.display()))?;
        Self::from_json(&v)
    }

    /// 項が参照するスナップショットを埋め込む (メモリ上のものだけ)。
    ///
    /// 参照先がメモリにも埋め込みにも無ければ `Err` (黙って欠けたレシピを作らない)。
    pub fn embed_snapshots(&mut self) -> Result<(), String> {
        let have = snapshots();
        let mut add: Vec<Snapshot> = Vec::new();
        for t in &self.terms {
            if self.snapshots.iter().any(|s| s.name == t.snapshot) {
                continue;
            }
            let s = have
                .iter()
                .find(|s| s.name == t.snapshot)
                .ok_or_else(|| format!("スナップショット {:?} が無い", t.snapshot))?;
            add.push((**s).clone());
        }
        self.snapshots.extend(add);
        self.snapshots.sort_by(|a, b| a.name.cmp(&b.name));
        self.snapshots.dedup_by(|a, b| a.name == b.name);
        Ok(())
    }

    /// 項と埋め込みの食い違い (terms の参照先が無い) を検査する。
    fn check_terms_resolvable(&self) -> Result<(), String> {
        for t in &self.terms {
            let embedded = self.snapshots.iter().any(|s| s.name == t.snapshot);
            let on_disk = snapshots().iter().any(|s| s.name == t.snapshot);
            if !embedded && !on_disk {
                return Err(format!("スナップショット {:?} が無い", t.snapshot));
            }
        }
        Ok(())
    }
}

/// レシピの適用先。グローバルのステアリング状態を指す軽いハンドル。
#[derive(Clone, Copy, Debug, Default)]
pub struct SteerState;

impl SteerState {
    pub fn new() -> SteerState {
        SteerState
    }

    /// 埋め込みスナップショットを**メモリにだけ**載せる (ディスクは書かない)。
    /// 返り値は載せた本数。
    pub fn load_embedded(&self, r: &SteerRecipe) -> Result<usize, String> {
        let (shape, _) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
        for s in &r.snapshots {
            if s.layers != shape.n_layer || s.n_hc != shape.n_hc || s.n_embd != shape.n_embd {
                return Err(format!(
                    "埋め込みスナップショット {:?} の形が違う ({}x{}x{} / モデル {}x{}x{})",
                    s.name, s.layers, s.n_hc, s.n_embd, shape.n_layer, shape.n_hc, shape.n_embd
                ));
            }
            if s.mean.len() != shape.len() {
                return Err(format!(
                    "埋め込みスナップショット {:?} の本体長が違う",
                    s.name
                ));
            }
            push_snapshot(Arc::new(s.clone()));
        }
        Ok(r.snapshots.len())
    }

    /// レシピを検証して適用する (epoch が進む)。
    pub fn apply_recipe(&self, r: &SteerRecipe) -> Result<Arc<Steer>, String> {
        let (shape, model) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
        if let Some(mid) = &r.model_id {
            if !mid.is_empty() && !model.is_empty() && *mid != model {
                return Err(format!(
                    "レシピの model_id {mid:?} がモデル {model:?} と違う"
                ));
            }
        }
        if r.layer_to > shape.n_layer {
            return Err(format!(
                "layers の to {} が層数 {} を超える ({}-{})",
                r.layer_to, shape.n_layer, r.layer_from, r.layer_to
            ));
        }
        if r.terms.is_empty() {
            return Err("terms が空".into());
        }
        self.load_embedded(r)?;
        r.check_terms_resolvable()?;
        set_apply(
            r.terms.clone(),
            r.layer_from,
            r.layer_to,
            r.alpha,
            r.mode,
        )
    }
}

/// いまの適用状態からレシピを組み立てる (`embed` でスナップショットを埋め込む)。
pub fn current_recipe(name: &str, embed: bool) -> Result<SteerRecipe, String> {
    validate_name(name)?;
    let info = apply_status().ok_or_else(|| "適用中のステアリングが無い".to_string())?;
    // task 849: レシピ (version 1) は 1 規則ぶんの形しか持たない。
    // 重ね掛けを黙って 1 規則に潰さない (lossy な保存はしない)。
    if !info.rules.is_empty() {
        return Err(
            "重ね掛け (rules) の適用はレシピ 1 個に畳めない (単一規則に戻してから保存)".into(),
        );
    }
    let (_, model) = meta().ok_or_else(|| "モデルが未初期化".to_string())?;
    let mut r = SteerRecipe {
        name: name.to_string(),
        description: None,
        model_id: if model.is_empty() { None } else { Some(model) },
        mode: info.mode,
        alpha: info.alpha,
        layer_from: info.from,
        layer_to: info.to,
        terms: info.terms,
        snapshots: Vec::new(),
    };
    if embed {
        r.embed_snapshots()?;
    }
    Ok(r)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// env プリロード
// ---------------------------------------------------------------------------

/// `DENNO_STEER_APPLY` の書式:
///
/// ```text
/// <name>/<coef>[/<name>/<coef>...]/<from>-<to>/<alpha>/<mode>
/// ```
///
/// 例: `A/1.0/0-48/1.0/subtract`、`A/1.0/B/-1.0/8-40/0.5/project`。
///
/// 区切りは `/` を使う。ラボの COND 検証が `:` を弾いた (t218a が
/// `COND format` で REJECTED) ため、`:` は使わない。`:` 区切りも受け付ける
/// (古い書き方との互換。正規化してから分解する)。
/// `to` は含まない。
pub fn parse_apply_spec(spec: &str) -> Result<(Vec<Term>, usize, usize, f32, Mode), String> {
    for ch in spec.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '/' | '-' | ':');
        if !ok {
            return Err(format!(
                "spec に使えない文字 {ch:?} (A-Z a-z 0-9 _ . / - のみ)"
            ));
        }
    }
    // `:` は `/` と同じ区切りとして扱う。
    let norm = spec.replace(':', "/");
    let parts: Vec<&str> = norm.split('/').collect();
    if parts.len() < 5 {
        return Err(format!(
            "spec が短い (name/coef/from-to/alpha/mode): {spec:?}"
        ));
    }
    let n = parts.len();
    let mode = Mode::parse(parts[n - 1])
        .ok_or_else(|| format!("未知の mode {:?} (subtract / project)", parts[n - 1]))?;
    let alpha: f32 = parts[n - 2]
        .parse()
        .map_err(|_| format!("alpha が数値でない: {:?}", parts[n - 2]))?;
    if !alpha.is_finite() {
        return Err("alpha が有限でない".into());
    }
    let (from_s, to_s) = parts[n - 3]
        .split_once('-')
        .ok_or_else(|| format!("layers が from-to でない: {:?}", parts[n - 3]))?;
    let from: usize = from_s
        .parse()
        .map_err(|_| format!("layers の from が数値でない: {from_s:?}"))?;
    let to: usize = to_s
        .parse()
        .map_err(|_| format!("layers の to が数値でない: {to_s:?}"))?;
    let head = &parts[..n - 3];
    if head.len() < 2 || head.len() % 2 != 0 {
        return Err("項は name:coef の組で並べる".into());
    }
    let mut terms = Vec::new();
    for pair in head.chunks(2) {
        validate_name(pair[0])?;
        let coef: f32 = pair[1]
            .parse()
            .map_err(|_| format!("coef が数値でない: {:?}", pair[1]))?;
        if !coef.is_finite() {
            return Err(format!("coef が有限でない: {:?}", pair[1]));
        }
        terms.push(Term {
            snapshot: pair[0].to_string(),
            coef,
        });
    }
    Ok((terms, from, to, alpha, mode))
}

fn init_env_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // 適用 (先に読む: 記録の有無に関係なく効く)
        if let Ok(spec) = std::env::var("DENNO_STEER_APPLY") {
            if !spec.trim().is_empty() {
                match parse_apply_spec(&spec).and_then(|(t, f, to, a, m)| {
                    set_apply(t, f, to, a, m).map(|_| ())
                }) {
                    Ok(()) => eprintln!("[steer] DENNO_STEER_APPLY={spec:?} を設定した"),
                    Err(e) => eprintln!("[steer] DENNO_STEER_APPLY={spec:?} を無視: {e}"),
                }
            }
        }
        if let Ok(name) = std::env::var("DENNO_STEER_RECORD") {
            let name = name.trim();
            if !name.is_empty() {
                // 既に同じ名前があれば上書き (記録し直し)。ディスクは残す。
                match start_record(name) {
                    Ok(_) => {
                        REC_ENV.store(true, Ordering::SeqCst);
                        eprintln!("[steer] DENNO_STEER_RECORD={name:?} で記録を始めた");
                    }
                    Err(e) => eprintln!("[steer] DENNO_STEER_RECORD={name:?} を無視: {e}"),
                }
            }
        }
    });
}

/// `DENNO_STEER_RECORD` で始まった記録か (毎リクエスト後に flush する)。
pub fn env_recording() -> bool {
    REC_ENV.load(Ordering::Relaxed)
}

/// テスト用: グローバルを初期状態に戻す (スナップショットは読み直さない)。
#[doc(hidden)]
pub fn reset_for_test() {
    REC_ON.store(false, Ordering::SeqCst);
    REC_ENV.store(false, Ordering::SeqCst);
    REC_MODE.store(0, Ordering::SeqCst);
    LAST_MODE.store(0, Ordering::SeqCst);
    *rec_cell().lock().unwrap_or_else(|e| e.into_inner()) = None;
    let epoch = super::intervene::bump_epoch();
    *cur_cell().write().unwrap_or_else(|e| e.into_inner()) = Arc::new(Steer::empty(epoch));
    APPLY_ON.store(false, Ordering::SeqCst);
    *snaps_cell().write().unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// グローバル (env / スナップショット / epoch) を触るテストの直列化。
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn tmpdir(tag: &str) -> PathBuf {
        let n = {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            N.fetch_add(1, Ordering::Relaxed)
        };
        let d = std::env::temp_dir().join(format!("denno-steer-{}-{tag}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("一時ディレクトリ");
        d
    }

    fn shape() -> Shape {
        Shape {
            n_layer: 3,
            n_hc: 2,
            n_embd: 4,
        }
    }

    fn rows(s: &Shape, n: usize, seed: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|t| {
                (0..s.stride())
                    .map(|i| (t * s.stride() + i) as f32 * 0.125 + seed)
                    .collect()
            })
            .collect()
    }

    /// 記録した平均が「トークンの和 / トークン数」の素朴な参照と一致する。
    #[test]
    fn recording_mean_equals_cpu_reference() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("mean");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "ref-model");
        let toks = rows(&s, 4, -0.5);

        start_record("A").expect("記録開始");
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let snap = stop_record().expect("保存");
        assert_eq!(snap.tokens, 4, "トークン数");
        assert_eq!(snap.name, "A");
        assert_eq!(snap.model, "ref-model");
        assert_eq!((snap.layers, snap.n_hc, snap.n_embd), (3, 2, 4));
        let inv = 1.0f32 / 4.0f32;
        for il in 0..s.n_layer {
            for d in 0..s.stride() {
                let mut sum = 0.0f32;
                for row in &toks {
                    sum += row[d];
                }
                let want = sum * inv;
                let got = snap.mean[il * s.stride() + d];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "層 {il} 要素 {d}: {got} != {want}"
                );
                // 割り算で書いた参照とも 1e-6 以内 (丸めの違いだけ)。
                let want_div = sum / 4.0f32;
                assert!((got - want_div).abs() <= 1e-6, "{got} vs {want_div}");
            }
        }
        assert!(!rec_on(), "stop 後も記録中になっている");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// スナップショットの保存 → 読み直し → 削除。
    #[test]
    fn snapshot_round_trip_on_disk() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("rt");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "rt-model");
        let toks = rows(&s, 2, 1.0);
        start_record("A").expect("記録開始");
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let saved = stop_record().expect("保存");
        let path = dir.join("A.dns");
        assert!(path.exists(), "ファイルが無い: {}", path.display());

        // メモリを捨ててディスクから読み直す。
        *snaps_cell().write().unwrap() = Arc::new(Vec::new());
        reload_snapshots();
        let got = snapshots();
        assert_eq!(got.len(), 1, "読み直したスナップショット数");
        let g = &got[0];
        assert_eq!(g.name, "A");
        assert_eq!(g.tokens, saved.tokens);
        assert_eq!(g.created, saved.created);
        assert_eq!(g.mean.len(), saved.mean.len());
        for i in 0..saved.mean.len() {
            assert_eq!(g.mean[i].to_bits(), saved.mean[i].to_bits(), "要素 {i}");
        }

        // 削除 (ファイルもメモリも)。
        delete("A").expect("削除");
        assert!(!path.exists(), "ファイルが残っている");
        assert!(snapshots().is_empty(), "メモリに残っている");
        assert!(delete("A").is_err(), "2 回目の削除が成功した");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 形かモデル id が違うスナップショットは読み込まない。
    #[test]
    fn shape_or_model_mismatch_is_refused() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("mismatch");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        // 形の違うファイルを直に置く。
        let bad_shape = Snapshot {
            name: "bad".into(),
            tokens: 3,
            created: 1,
            model: "m".into(),
            layers: 6,
            n_hc: 2,
            n_embd: 4,
            mean: vec![0.0; 6 * 2 * 4],
        };
        save_snapshot(&bad_shape).expect("書き込み");
        // モデル id の違うファイル。
        let bad_model = Snapshot {
            name: "other".into(),
            tokens: 3,
            created: 1,
            model: "another-model".into(),
            layers: s.n_layer,
            n_hc: s.n_hc,
            n_embd: s.n_embd,
            mean: vec![0.0; s.len()],
        };
        save_snapshot(&bad_model).expect("書き込み");
        init(s, "this-model");
        assert!(
            snapshots().is_empty(),
            "形/モデルの違うスナップショットを読み込んだ: {:?}",
            snapshots().iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        // 壊れたファイルは Err (panic しない)。
        let p = dir.join("broken.dns");
        std::fs::write(&p, b"DNSTEER1\nname=x\n").unwrap();
        assert!(load_file(&p).is_err());
        std::fs::write(&p, b"nope").unwrap();
        assert!(load_file(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 係数 0 / alpha 0 は 1 ビットも変えない。
    #[test]
    fn zero_vector_or_alpha_zero_is_bit_identical() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("zero");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "z");
        let toks = rows(&s, 2, 0.25);
        start_record("A").unwrap();
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        stop_record().unwrap();
        let snaps = {
            let g = snaps_cell().read().unwrap();
            g.as_ref().clone()
        };
        let h0: Vec<f32> = rows(&s, 1, 0.75).pop().unwrap();

        // coef 0
        let a = build_apply(
            &s,
            &snaps,
            &[Term {
                snapshot: "A".into(),
                coef: 0.0,
            }],
            0,
            s.n_layer,
            1.0,
            Mode::Subtract,
        )
        .expect("build");
        let mut h = h0.clone();
        for il in 0..s.n_layer {
            a.apply_one(il, &mut h);
        }
        assert_eq!(h, h0, "coef 0 で値が変わった");

        // alpha 0 (project でも同じ)
        for mode in [Mode::Subtract, Mode::Project] {
            let a = build_apply(
                &s,
                &snaps,
                &[Term {
                    snapshot: "A".into(),
                    coef: 1.0,
                }],
                0,
                s.n_layer,
                0.0,
                mode,
            )
            .expect("build");
            let mut h = h0.clone();
            for il in 0..s.n_layer {
                a.apply_one(il, &mut h);
            }
            assert_eq!(h, h0, "alpha 0 ({}) で値が変わった", mode.name());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn apply_fixture(tag: &str) -> (Vec<f32>, Shape, Vec<Arc<Snapshot>>, PathBuf) {
        let dir = tmpdir(tag);
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "fixture");
        let toks = rows(&s, 3, -0.25);
        start_record("A").unwrap();
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        stop_record().unwrap();
        let snaps = {
            let g = snaps_cell().read().unwrap();
            g.as_ref().clone()
        };
        let h: Vec<f32> = (0..s.stride())
            .map(|i| ((i * 7) % 11) as f32 * 0.375 - 1.0)
            .collect();
        (h, s, snaps, dir)
    }

    /// subtract が CPU 参照と 1 ビット一致する。
    #[test]
    fn subtract_matches_cpu_reference() {
        let _g = LOCK.lock().unwrap();
        let (h0, s, snaps, dir) = apply_fixture("sub");
        let alpha = 1.25f32;
        let a = build_apply(
            &s,
            &snaps,
            &[Term {
                snapshot: "A".into(),
                coef: 1.0,
            }],
            0,
            s.n_layer,
            alpha,
            Mode::Subtract,
        )
        .expect("build");
        let st = s.stride();
        for il in 0..s.n_layer {
            let mut got = h0.clone();
            a.apply_one(il, &mut got);
            for d in 0..st {
                let want = h0[d] - alpha * a.v[il * st + d];
                assert_eq!(
                    got[d].to_bits(),
                    want.to_bits(),
                    "層 {il} 要素 {d}: {} != {want}",
                    got[d]
                );
            }
        }
        // 範囲外の層は触らない。
        let mut got = h0.clone();
        a.apply_one(s.n_layer + 1, &mut got);
        assert_eq!(got, h0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// project が CPU 参照と一致し、v 方向の成分が残らない。
    #[test]
    fn project_matches_cpu_reference_and_removes_the_component() {
        let _g = LOCK.lock().unwrap();
        let (h0, s, snaps, dir) = apply_fixture("proj");
        let alpha = 1.0f32;
        let a = build_apply(
            &s,
            &snaps,
            &[Term {
                snapshot: "A".into(),
                coef: 1.0,
            }],
            0,
            s.n_layer,
            alpha,
            Mode::Project,
        )
        .expect("build");
        let (e, hc) = (s.n_embd, s.n_hc);
        for il in 0..s.n_layer {
            let mut got = h0.clone();
            a.apply_one(il, &mut got);
            for st_i in 0..hc {
                let base = il * s.stride() + st_i * e;
                let vp = &a.vhat[base..base + e];
                // CPU 参照 (d 昇順の逐次内積)
                let mut dot = 0.0f32;
                for d in 0..e {
                    dot += h0[st_i * e + d] * vp[d];
                }
                let c = alpha * dot;
                for d in 0..e {
                    let want = h0[st_i * e + d] - c * vp[d];
                    let g = got[st_i * e + d];
                    assert!(
                        (g - want).abs() <= 1e-6,
                        "層 {il} ストリーム {st_i} 要素 {d}: {g} != {want}"
                    );
                }
                // 適用後は v 方向の成分が (ほぼ) 0。
                let mut after = 0.0f64;
                let mut norm = 0.0f64;
                for d in 0..e {
                    after += (got[st_i * e + d] as f64) * (vp[d] as f64);
                    norm += (vp[d] as f64) * (vp[d] as f64);
                }
                assert!(norm > 0.5, "vhat のノルムが 1 でない: {norm}");
                assert!(after.abs() <= 1e-5, "v 方向の成分が残った: {after}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 適用と解除が epoch を進める (= prefix cache が捨てられる)。
    #[test]
    fn apply_and_clear_move_the_epoch() {
        let _g = LOCK.lock().unwrap();
        let (_h, _s, _snaps, dir) = apply_fixture("epoch");
        let e0 = crate::model::intervene::epoch_now();
        let st = set_apply(
            vec![Term {
                snapshot: "A".into(),
                coef: 1.0,
            }],
            0,
            3,
            1.0,
            Mode::Subtract,
        )
        .expect("set");
        assert_eq!(st.epoch, e0 + 1, "適用で epoch が 1 つ進んでいない");
        assert!(crate::model::intervene::current_epoch() >= e0 + 1);
        let st2 = clear_apply();
        assert_eq!(st2.epoch, e0 + 2, "解除で epoch が進んでいない");
        assert!(st2.apply.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 適用の検証 (存在しない名前 / 範囲外 / 未知の mode)。
    #[test]
    fn apply_validation_rejects_bad_input() {
        let _g = LOCK.lock().unwrap();
        let (_h, s, snaps, dir) = apply_fixture("val");
        let t = |name: &str| Term {
            snapshot: name.to_string(),
            coef: 1.0,
        };
        assert!(build_apply(&s, &snaps, &[], 0, 3, 1.0, Mode::Subtract).is_err());
        assert!(build_apply(
            &s,
            &snaps,
            &[t("nope")],
            0,
            3,
            1.0,
            Mode::Subtract
        )
        .is_err());
        assert!(build_apply(&s, &snaps, &[t("A")], 0, 4, 1.0, Mode::Subtract).is_err());
        assert!(build_apply(&s, &snaps, &[t("A")], 2, 1, 1.0, Mode::Subtract).is_err());
        assert!(build_apply(&s, &snaps, &[t("A")], 0, 3, f32::NAN, Mode::Subtract).is_err());
        // 名前の検証
        assert!(validate_name("").is_err());
        assert!(validate_name("a.b").is_err());
        assert!(validate_name("a/b").is_err());
        assert!(validate_name("A-1_2").is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `DENNO_STEER_APPLY` の spec パーサ。
    #[test]
    fn parse_apply_spec_forms() {
        let (t, f, to, a, m) = parse_apply_spec("A/-1.0/0-48/1.0/subtract").expect("1 項");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].snapshot, "A");
        assert_eq!(t[0].coef, -1.0);
        assert_eq!((f, to), (0, 48));
        assert_eq!(a, 1.0);
        assert_eq!(m, Mode::Subtract);

        let (t, f, to, a, m) = parse_apply_spec("A/1.0/B/-1.0/8-40/0.5/project").expect("2 項");
        assert_eq!(t.len(), 2);
        assert_eq!(t[1].snapshot, "B");
        assert_eq!(t[1].coef, -1.0);
        assert_eq!((f, to), (8, 40));
        assert_eq!(a, 0.5);
        assert_eq!(m, Mode::Project);

        // `:` 区切りも同じに読める (古い書き方)。
        let (t, f, to, a, m) = parse_apply_spec("A:-1.0:0-48:1.0:subtract").expect("コロン区切り");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].snapshot, "A");
        assert_eq!((f, to), (0, 48));
        assert_eq!(a, 1.0);
        assert_eq!(m, Mode::Subtract);

        // 不正
        assert!(parse_apply_spec("A/1.0/0-48/1.0").is_err(), "項が無い");
        assert!(parse_apply_spec("A/1.0/0-48/1.0/sigmoid").is_err());
        assert!(parse_apply_spec("A/B/0-48/1.0/subtract").is_err(), "coef が数値でない");
        assert!(parse_apply_spec("A/1.0/0..48/1.0/subtract").is_err());
        assert!(parse_apply_spec("A/1.0/0-48/1.0/subtract,").is_err(), "使えない文字");
        // ラボの COND が通す文字だけであること (t218a の REJECTED の再発防止)。
        for s in ["A/1.0/0-48/1.0/subtract", "A/1.0/B/-1.0/8-40/0.5/project"] {
            for ch in s.chars() {
                assert!(
                    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '/' | '-'),
                    "spec {s:?} にラボが弾く文字 {ch:?} が入っている"
                );
            }
        }
    }

    // ---- task 825: 対比の分析 / エクスポート / インポート -----------------

    /// 層ごとのノルム・余弦・分離度が手計算と一致する。
    #[test]
    fn layer_metrics_match_hand_computation() {
        let s = Shape {
            n_layer: 2,
            n_hc: 1,
            n_embd: 3,
        };
        let a = [3.0f32, 0.0, 0.0, 1.0, 2.0, 2.0];
        let b = [0.0f32, 4.0, 0.0, 1.0, 2.0, 2.0];
        let got = layer_metrics(&s, &a, &b).expect("metrics");
        assert_eq!(got.len(), 2);
        let l0 = got[0];
        assert_eq!(l0.layer, 0);
        assert!((l0.norm_pos - 3.0).abs() < 1e-6, "{}", l0.norm_pos);
        assert!((l0.norm_neg - 4.0).abs() < 1e-6, "{}", l0.norm_neg);
        assert!((l0.diff_norm - 5.0).abs() < 1e-6, "{}", l0.diff_norm);
        assert!(l0.cos.abs() < 1e-6, "直交の余弦が 0 でない: {}", l0.cos);
        assert!((l0.sep - 5.0 / 7.0).abs() < 1e-6, "{}", l0.sep);
        let l1 = got[1];
        assert!(l1.diff_norm.abs() < 1e-6);
        assert!((l1.cos - 1.0).abs() < 1e-6, "{}", l1.cos);
        assert!(l1.sep.abs() < 1e-9);
        // 形が合わないベクトルは Err (panic しない)。
        assert!(layer_metrics(&s, &a[..5], &b).is_err());
        assert!(layer_metrics(&s, &a, &b[..5]).is_err());
    }

    /// 推奨区間はピークを中心に半値以上が続く連続窓で、層 0 と最終層を避ける。
    #[test]
    fn recommended_range_is_half_max_window_inside_the_model() {
        assert_eq!(recommended_range(&[]), (0, 0));
        assert_eq!(recommended_range(&[0.3]), (0, 1));
        assert_eq!(recommended_range(&[0.3, 0.4]), (0, 2));
        // 層 0 が最大でも選ばない。
        assert_eq!(recommended_range(&[0.9, 0.2, 0.5, 0.45, 0.1]), (2, 4));
        // 最終層が最大でも選ばない。
        assert_eq!(recommended_range(&[0.1, 0.3, 0.6, 0.99]), (1, 3));
        // 分離が全部 0 なら内部全体。
        assert_eq!(recommended_range(&[0.0, 0.0, 0.0, 0.0, 0.0]), (1, 4));
        // ピークの左右に半値以上が続く限り広げる。
        assert_eq!(recommended_range(&[0.0, 0.4, 0.8, 0.5, 0.2, 0.0, 0.0]), (1, 4));
    }

    /// 2 つのスナップショットの対比 (analyze) が層ごとの指標と推奨区間を返す。
    #[test]
    fn analyze_two_snapshots_reports_layers_and_range() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("analyze");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "an-model");
        // A は [1,1,..] 側、B は [-1,-1,..] 側 (差が層ごとに一定)。
        let a = rows(&s, 2, 0.5);
        // B は A の符号反転 (余弦 -1・差ノルム 2||mu|| が手計算で決まる)。
        let b: Vec<Vec<f32>> = a
            .iter()
            .map(|r| r.iter().map(|x| -x).collect())
            .collect();
        start_record("A").unwrap();
        for row in &a {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        stop_record().unwrap();
        start_record("B").unwrap();
        for row in &b {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        stop_record().unwrap();

        let an = analyze("A", "B").expect("analyze");
        assert_eq!(an.pos, "A");
        assert_eq!(an.neg, "B");
        assert_eq!(an.layers.len(), s.n_layer);
        for l in &an.layers {
            assert!(l.diff_norm > 0.0);
            assert!((l.cos + 1.0).abs() < 1e-5, "逆向きの余弦が -1 でない: {}", l.cos);
            assert!(l.sep > 0.0 && l.sep <= 1.0, "sep {}", l.sep);
        }
        let (f, t) = an.recommended_range;
        assert!(f >= 1 && t > f && t <= s.n_layer - 1, "range {f}-{t}");
        assert!(an.peak_layer >= 1 && an.peak_layer < s.n_layer - 1);

        // 同じスナップショット同士は差 0・余弦 1・分離 0。
        let same = analyze("A", "A").expect("analyze same");
        for l in &same.layers {
            assert!(l.diff_norm.abs() < 1e-9);
            assert!((l.cos - 1.0).abs() < 1e-6);
            assert!(l.sep.abs() < 1e-9);
        }
        assert_eq!(same.recommended_range, (1, s.n_layer - 1));

        // 無い名前は Err (panic しない)。
        assert!(analyze("A", "nope").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 0 ベクトルのスナップショットでも NaN を出さない (余弦 0 / 分離 0)。
    #[test]
    fn analyze_zero_snapshot_yields_zero_cos_and_sep() {
        let _g = LOCK.lock().unwrap();
        let s = shape();
        let zero = vec![0.0f32; s.len()];
        let nonzero: Vec<f32> = (0..s.len()).map(|i| (i % 5) as f32 + 0.5).collect();
        let m = layer_metrics(&s, &nonzero, &zero).expect("metrics");
        for l in &m {
            assert_eq!(l.cos, 0.0);
            assert!(l.sep > 0.0);
            assert!(l.sep.is_finite());
        }
        let m = layer_metrics(&s, &zero, &zero).expect("metrics");
        for l in &m {
            assert_eq!((l.cos, l.sep, l.diff_norm), (0.0, 0.0, 0.0));
        }
    }

    /// terms の対比 (A +1 / B -1) が名前付きの対比と同じ結果になる。
    #[test]
    fn analyze_terms_splits_by_sign() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("terms");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "terms-model");
        for (name, seed) in [("A", 0.5f32), ("B", -0.5)] {
            let toks = rows(&s, 2, seed);
            start_record(name).unwrap();
            for row in &toks {
                for il in 0..s.n_layer {
                    acc_host(il, row);
                }
            }
            stop_record().unwrap();
        }
        let by_name = analyze("A", "B").expect("by name");
        let by_terms = analyze_terms(&[
            Term {
                snapshot: "A".into(),
                coef: 1.0,
            },
            Term {
                snapshot: "B".into(),
                coef: -1.0,
            },
        ])
        .expect("by terms");
        assert_eq!(by_name.layers, by_terms.layers);
        assert_eq!(by_name.recommended_range, by_terms.recommended_range);
        assert_eq!(by_terms.pos, "A");
        assert_eq!(by_terms.neg, "B");
        // 片側だけ / coef 0 だけは Err。
        assert!(analyze_terms(&[Term {
            snapshot: "A".into(),
            coef: 1.0,
        }])
        .is_err());
        assert!(analyze_terms(&[
            Term {
                snapshot: "A".into(),
                coef: 0.0
            },
            Term {
                snapshot: "B".into(),
                coef: -1.0
            }
        ])
        .is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// base64 の往復と不正入力 (外部 crate 無しの実装)。
    #[test]
    fn base64_round_trip_and_rejects_garbage() {
        for raw in [
            b"".to_vec(),
            b"f".to_vec(),
            b"fo".to_vec(),
            b"foo".to_vec(),
            b"foobar".to_vec(),
            vec![0u8, 255, 128, 7, 9, 10, 13],
        ] {
            let enc = base64_encode(&raw);
            assert_eq!(base64_decode(&enc).unwrap(), raw, "enc {enc}");
        }
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        // 空白は無視。
        assert_eq!(base64_decode("Zm9v\n").unwrap(), b"foo");
        assert!(base64_decode("Zm9v!!!!").is_err());
        assert!(base64_decode("Z").is_err(), "端数のビットを黙って捨てた");
    }

    /// export → import の往復で値が 1 ビットも変わらない (binary / base64 / JSON)。
    #[test]
    fn export_import_round_trip_keeps_bits() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("expimp");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "ei-model");
        let toks = rows(&s, 3, 0.125);
        start_record("src").unwrap();
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let src = stop_record().unwrap();

        // バイナリ
        let raw = export_raw("src").expect("export raw");
        assert!(raw.starts_with(FILE_MAGIC.as_bytes()));
        let back = import_bytes(&raw, Some("copy")).expect("import raw");
        assert_eq!(back.name, "copy");
        assert_eq!(back.mean.len(), src.mean.len());
        for i in 0..src.mean.len() {
            assert_eq!(back.mean[i].to_bits(), src.mean[i].to_bits(), "要素 {i}");
        }
        assert!(dir.join("copy.dns").exists(), "ディスクに書いていない");
        assert!(snapshots().iter().any(|x| x.name == "copy"));

        // JSON (base64)
        let j = export_json("src").expect("export json");
        assert_eq!(j["name"], "src");
        assert!(j["data_base64"].as_str().unwrap().len() > 10);
        assert!(serde_json::to_string(&j).is_ok());
        let back = import_json(&j).expect("import json");
        for i in 0..src.mean.len() {
            assert_eq!(back.mean[i].to_bits(), src.mean[i].to_bits(), "JSON 要素 {i}");
        }

        // JSON (mean 配列)
        let j2 = json!({
            "name": "arr",
            "layers": s.n_layer,
            "n_hc": s.n_hc,
            "n_embd": s.n_embd,
            "mean": src.mean.iter().map(|x| json!(*x)).collect::<Vec<_>>(),
        });
        let back = import_json(&j2).expect("import mean array");
        assert_eq!(back.name, "arr");
        for i in 0..src.mean.len() {
            assert_eq!(back.mean[i].to_bits(), src.mean[i].to_bits());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 壊れた入力は Err (panic しない)。形・名前・モデル id の検証。
    #[test]
    fn import_rejects_bad_input() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("badimp");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "bad-model");
        let toks = rows(&s, 1, 0.0);
        start_record("A").unwrap();
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let good = stop_record().unwrap();

        // 壊れたバイナリ
        assert!(import_bytes(b"nope", None).is_err());
        let mut trunc = encode_snapshot(&good);
        trunc.truncate(trunc.len() - 3);
        assert!(import_bytes(&trunc, None).is_err());
        // 形違い
        let mut other = good.clone();
        other.layers = 5;
        other.mean = vec![0.0; 5 * s.n_hc * s.n_embd];
        assert!(import_snapshot(other, None).is_err());
        // モデル id 違い
        let mut other_model = good.clone();
        other_model.model = "some-other-model".into();
        assert!(import_snapshot(other_model, None).is_err());
        // 使えない名前
        let mut bad_name = good.clone();
        bad_name.name = "a.b".into();
        assert!(import_snapshot(bad_name, None).is_err());
        // 非有限の値
        let mut nan = good.clone();
        nan.mean[0] = f32::NAN;
        assert!(import_snapshot(nan, None).is_err());
        // JSON の形違い / 壊れた base64 / 未知の形
        assert!(import_json(&json!({"name": "x"})).is_err());
        assert!(import_json(&json!({"name": "x", "data_base64": "!!!"})).is_err());
        assert!(import_json(&json!({
            "name": "x",
            "layers": 0,
            "n_hc": 0,
            "n_embd": 0,
            "mean": [],
        }))
        .is_err());
        // 名前の上書きも検証される。
        let raw = export_raw("A").unwrap();
        assert!(import_bytes(&raw, Some("a/b")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 829: `LastPromptToken` は prefill の最終トークンだけを記録し、
    /// その後の decode トークンは載せない (全トークン平均ではない)。
    #[test]
    fn last_prompt_token_records_only_final_prefill_token() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("lasttok");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "last-model");
        // prefill 4 トークン + decode 2 トークン (合計 6)。
        let toks = rows(&s, 6, 0.25);
        let nt = 4usize;

        start_record_mode("L", CaptureMode::LastPromptToken).expect("記録開始");
        assert_eq!(capture_mode(), CaptureMode::LastPromptToken);
        for (t, row) in toks.iter().take(nt).enumerate() {
            for il in 0..s.n_layer {
                acc_host_prefill(il, row, t, nt);
            }
        }
        // decode は acc_host (位置なし) — LastPromptToken では捨てる。
        for row in toks.iter().skip(nt) {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let snap = stop_record().expect("保存");
        assert_eq!(snap.tokens, 1, "最終トークン 1 個だけが数えられる");
        // 平均 = 最終トークンそのもの (ビット一致)。
        for il in 0..s.n_layer {
            for d in 0..s.stride() {
                let want = toks[nt - 1][d];
                let got = snap.mean[il * s.stride() + d];
                assert_eq!(got.to_bits(), want.to_bits(), "層 {il} 要素 {d}");
            }
        }
        // 全トークン平均とは違う (取り違えの検出)。
        assert_ne!(snap.mean[0].to_bits(), {
            let sum: f32 = toks.iter().take(nt).map(|r| r[0]).sum();
            (sum / nt as f32).to_bits()
        });
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 829: `All` のホスト経路は prefill / decode のどちらの入口でも
    /// 従来の「全トークン平均」とビット一致する。
    #[test]
    fn all_mode_prefill_matches_previous_mean_bit_for_bit() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("allbit");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "all-model");
        let toks = rows(&s, 5, -0.75);
        let nt = 3usize;

        // 従来の入口 (acc_host のみ) で記録。
        start_record("A").expect("記録開始");
        for row in &toks {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let old = stop_record().expect("保存");

        // prefill は acc_host_prefill、decode は acc_host。
        start_record("B").expect("記録開始");
        for (t, row) in toks.iter().take(nt).enumerate() {
            for il in 0..s.n_layer {
                acc_host_prefill(il, row, t, nt);
            }
        }
        for row in toks.iter().skip(nt) {
            for il in 0..s.n_layer {
                acc_host(il, row);
            }
        }
        let new = stop_record().expect("保存");

        assert_eq!(old.tokens, 5);
        assert_eq!(new.tokens, 5);
        for i in 0..old.mean.len() {
            assert_eq!(old.mean[i].to_bits(), new.mean[i].to_bits(), "要素 {i}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 829: チャンク行の入口も `LastPromptToken` では最終チャンクの
    /// 最後の行だけ、`All` では全部を足す。
    #[test]
    fn last_prompt_token_chunk_rows_take_final_row_only() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("chunkrows");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "chunk-model");
        let toks = rows(&s, 6, 0.5);

        start_record_mode("L", CaptureMode::LastPromptToken).expect("記録開始");
        // チャンク 0 (t=0..3, 最終でない) は捨てる。
        let c0: Vec<&[f32]> = toks[0..3].iter().map(|r| r.as_slice()).collect();
        // チャンク 1 (t=3..6, 最終) の最後の行だけが載る。
        let c1: Vec<&[f32]> = toks[3..6].iter().map(|r| r.as_slice()).collect();
        for il in 0..s.n_layer {
            acc_host_rows(il, &c0, false);
            acc_host_rows(il, &c1, true);
        }
        let last = stop_record().expect("保存");
        assert_eq!(last.tokens, 1);
        assert_eq!(last.mean[0].to_bits(), toks[5][0].to_bits());

        start_record("A").expect("記録開始");
        for il in 0..s.n_layer {
            acc_host_rows(il, &c0, false);
            acc_host_rows(il, &c1, true);
        }
        let all = stop_record().expect("保存");
        assert_eq!(all.tokens, 6);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- task 833: レシピ ------------------------------------------------

    /// 記録済みの A / B を埋め込んだレシピを作る (テスト用の材料)。
    fn recipe_fixture(s: &Shape, dir: &Path) -> (SteerRecipe, Vec<Arc<Snapshot>>) {
        std::env::set_var("DENNO_STEER_DIR", dir);
        init(*s, "recipe-model");
        for (name, seed) in [("A", -0.25f32), ("B", 0.75f32)] {
            let toks = rows(s, 3, seed);
            start_record(name).expect("記録開始");
            for row in &toks {
                for il in 0..s.n_layer {
                    acc_host(il, row);
                }
            }
            stop_record().expect("保存");
        }
        let snaps: Vec<Arc<Snapshot>> = snaps_cell().read().unwrap().as_ref().clone();
        let mut r = SteerRecipe {
            name: "rc".into(),
            description: Some("test recipe".into()),
            model_id: Some("recipe-model".into()),
            mode: Mode::Project,
            alpha: 0.5,
            layer_from: 1,
            layer_to: 3,
            terms: vec![
                Term {
                    snapshot: "A".into(),
                    coef: 1.0,
                },
                Term {
                    snapshot: "B".into(),
                    coef: -1.0,
                },
            ],
            snapshots: Vec::new(),
        };
        r.embed_snapshots().expect("埋め込み");
        (r, snaps)
    }

    /// to_json → 文字列 → from_json が同じレシピに戻り、埋め込みも保たれる。
    #[test]
    fn recipe_json_round_trip_keeps_embedded_snapshots() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("recipe-rt");
        let s = shape();
        let (r, _) = recipe_fixture(&s, &dir);
        assert_eq!(r.snapshots.len(), 2, "埋め込みの本数");
        let text = serde_json::to_string(&r.to_json()).expect("JSON");
        let v: serde_json::Value = serde_json::from_str(&text).expect("parse");
        assert_eq!(v["version"], json!(RECIPE_VERSION));
        let back = SteerRecipe::from_json(&v).expect("from_json");
        assert_eq!(back, r, "往復で変わった");
        // 埋め込みは base64 の文字列で、名前は項と同じ。
        let snapobj = v["snapshots"].as_object().expect("snapshots");
        assert_eq!(snapobj.len(), 2);
        for (k, val) in snapobj {
            assert!(val.as_str().map(|x| !x.is_empty()).unwrap_or(false), "{k}");
        }
        // ファイル経由でも同じ。
        let path = dir.join("rc.json");
        std::fs::write(&path, serde_json::to_string_pretty(&r.to_json()).unwrap()).unwrap();
        assert_eq!(SteerRecipe::from_file(&path).expect("from_file"), r);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 埋め込みレシピの適用は、ディスクのスナップショットで組んだ適用と
    /// ビット一致し、**ディスクを書かない**。
    #[test]
    fn recipe_apply_matches_disk_and_writes_no_files() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("recipe-apply");
        let s = shape();
        let (r, _) = recipe_fixture(&s, &dir);
        SteerState::new().apply_recipe(&r).expect("適用 (ディスク)");
        let want = snapshot().apply.as_ref().expect("apply").clone();
        let want_info = apply_status().expect("status");
        assert_eq!((want_info.from, want_info.to), (1, 3));
        assert_eq!(want_info.alpha, 0.5);
        assert_eq!(want_info.mode, Mode::Project);

        // 新しいプロセスを模す: メモリを捨て、ディスクにも何も無い。
        *snaps_cell().write().unwrap() = Arc::new(Vec::new());
        let empty = tmpdir("recipe-apply-empty");
        std::env::set_var("DENNO_STEER_DIR", &empty);
        SteerState::new().apply_recipe(&r).expect("適用 (埋め込み)");
        assert_eq!(snapshots().len(), 2, "メモリへ載せた本数");
        let got = snapshot().apply.as_ref().expect("apply").clone();
        assert_eq!(got.v.len(), want.v.len());
        for i in 0..want.v.len() {
            assert_eq!(got.v[i].to_bits(), want.v[i].to_bits(), "v[{i}]");
            assert_eq!(got.vhat[i].to_bits(), want.vhat[i].to_bits(), "vhat[{i}]");
        }
        // 1 トークンぶんの適用結果も一致する。
        let mut h0 = rows(&s, 1, 0.125).pop().unwrap();
        let mut h1 = h0.clone();
        for il in 0..s.n_layer {
            want.apply_one(il, &mut h0);
            got.apply_one(il, &mut h1);
        }
        assert_eq!(h0, h1, "適用後の残差が違う");
        // ディスクは空のまま (埋め込みはメモリだけ)。
        let files: Vec<_> = std::fs::read_dir(&empty)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert!(files.is_empty(), "ディスクに書いた: {files:?}");
        std::fs::remove_dir_all(&empty).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 壊れたレシピは `from_json` が弾く。
    #[test]
    fn recipe_json_rejects_bad_input() {
        let _g = LOCK.lock().unwrap();
        let base = |patch: serde_json::Value| -> serde_json::Value {
            let mut v = json!({
                "version": 1,
                "name": "rc",
                "mode": "project",
                "alpha": 0.5,
                "layer_from": 0,
                "layer_to": 3,
                "terms": [{ "snapshot": "A", "coef": 1.0 }],
            });
            for (k, x) in patch.as_object().unwrap() {
                if x.is_null() {
                    v.as_object_mut().unwrap().remove(k);
                } else {
                    v.as_object_mut().unwrap().insert(k.clone(), x.clone());
                }
            }
            v
        };
        assert!(SteerRecipe::from_json(&base(json!({}))).is_ok());
        // 未知のフィールド / 版 / 必須キー
        assert!(SteerRecipe::from_json(&base(json!({"x": 1}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"version": 2}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"version": "1"}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"name": null}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"name": "a b"}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"mode": "bogus"}))).is_err());
        // alpha / 層範囲 / terms
        assert!(SteerRecipe::from_json(&base(json!({"alpha": "x"}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"alpha": 1e40}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"layer_from": -1}))).is_err());
        assert!(
            SteerRecipe::from_json(&base(json!({"layer_from": 4, "layer_to": 3}))).is_err()
        );
        assert!(SteerRecipe::from_json(&base(json!({"terms": []}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"terms": null}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"terms": [{"snapshot": "A"}]}))).is_err());
        assert!(SteerRecipe::from_json(
            &base(json!({"terms": [{"snapshot": "A", "coef": 1.0, "x": 1}]}))
        )
        .is_err());
        // 埋め込み
        assert!(SteerRecipe::from_json(&base(json!({"snapshots": "x"}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"snapshots": {"A": "!!!"}}))).is_err());
        assert!(SteerRecipe::from_json(&base(json!({"snapshots": {"A": 1}}))).is_err());
        // 名前が中身と違う / 中身が DNSTEER1 でない。
        let good = base(json!({}));
        let s = shape();
        let snap = Snapshot {
            name: "A".into(),
            tokens: 1,
            created: 1,
            model: "recipe-model".into(),
            layers: s.n_layer,
            n_hc: s.n_hc,
            n_embd: s.n_embd,
            mean: vec![1.0; s.len()],
        };
        let b64 = base64_encode(&encode_snapshot(&snap));
        let mut ok = good.clone();
        ok.as_object_mut().unwrap().insert(
            "snapshots".into(),
            json!({ "A": b64.clone() }),
        );
        assert_eq!(SteerRecipe::from_json(&ok).unwrap().snapshots.len(), 1);
        let mut misnamed = good.clone();
        misnamed
            .as_object_mut()
            .unwrap()
            .insert("snapshots".into(), json!({ "B": b64 }));
        assert!(SteerRecipe::from_json(&misnamed).is_err(), "名前の食い違い");
        let notdns = base64_encode(b"nope");
        let mut bad = good;
        bad.as_object_mut()
            .unwrap()
            .insert("snapshots".into(), json!({ "A": notdns }));
        assert!(SteerRecipe::from_json(&bad).is_err(), "DNSTEER1 でない");
    }

    /// model_id / 層数 / 参照先の検証。埋め込みが形違いなら適用しない。
    #[test]
    fn recipe_apply_checks_model_range_and_shape() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("recipe-checks");
        let s = shape();
        let (r, snaps) = recipe_fixture(&s, &dir);
        SteerState::new().apply_recipe(&r).expect("基準は通る");

        let mut other = r.clone();
        other.model_id = Some("another-model".into());
        let e = SteerState::new().apply_recipe(&other).unwrap_err();
        assert!(e.contains("model_id"), "{e}");
        // model_id 無しは通る。
        other.model_id = None;
        SteerState::new().apply_recipe(&other).expect("model_id 無し");

        let mut wide = r.clone();
        wide.layer_to = s.n_layer + 1;
        assert!(SteerState::new().apply_recipe(&wide).is_err(), "層数超過");

        let mut missing = r.clone();
        missing.terms.push(Term {
            snapshot: "ghost".into(),
            coef: 1.0,
        });
        assert!(SteerState::new().apply_recipe(&missing).is_err(), "無い参照");

        // 埋め込みだけが形違い (4 層) → 適用しない。
        let mut bad_shape = r.clone();
        let mut bad = bad_shape.snapshots[0].clone();
        bad.layers = 4;
        bad.mean = vec![0.0; 4 * s.n_hc * s.n_embd];
        bad_shape.snapshots[0] = bad;
        assert!(SteerState::new().apply_recipe(&bad_shape).is_err(), "形違い");

        // current_recipe は適用中の設定をそのまま写す。
        SteerState::new().apply_recipe(&r).expect("適用");
        let cur = current_recipe("saved", true).expect("current_recipe");
        assert_eq!(cur.model_id.as_deref(), Some("recipe-model"));
        assert_eq!((cur.layer_from, cur.layer_to), (1, 3));
        assert_eq!(cur.mode, Mode::Project);
        assert_eq!(cur.alpha, 0.5);
        assert_eq!(cur.terms, r.terms);
        assert_eq!(cur.snapshots.len(), 2, "項の参照を埋め込む");
        for s in &cur.snapshots {
            let want = snaps.iter().find(|x| x.name == s.name).expect("元");
            assert_eq!(&s.mean, &want.mean);
        }
        clear_apply();
        assert!(current_recipe("saved", false).is_err(), "適用が無い");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- task 849: 重ね掛け (rules) -------------------------------------

    /// テスト用: スナップショットをメモリに載せる。
    fn add_snap(s: &Shape, name: &str, seed: f32) {
        push_snapshot(Arc::new(Snapshot {
            name: name.into(),
            tokens: 4,
            created: 1,
            model: "stack-model".into(),
            layers: s.n_layer,
            n_hc: s.n_hc,
            n_embd: s.n_embd,
            // 全層ぶん (`[layer][stream][n_embd]`)。
            mean: (0..s.len())
                .map(|i| i as f32 * 0.125 + seed)
                .collect(),
        }));
    }

    fn tr(name: &str, coef: f32) -> Term {
        Term {
            snapshot: name.into(),
            coef,
        }
    }

    /// 重ね掛けは「規則を 1 本ずつ順に掛けた結果」と 1 ビット一致する
    /// (層範囲・alpha・mode が規則ごとに違っても)。
    #[test]
    fn stacked_rules_match_sequential_single_applies() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("stack");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "stack-model");
        add_snap(&s, "A", 0.25);
        add_snap(&s, "B", -0.5);
        add_snap(&s, "C", 0.75);
        let h0 = rows(&s, 1, 0.125).pop().unwrap();

        // 参照: 単一規則を規則の順に、層ごとに 1 回ずつ。
        let mut want = h0.clone();
        for il in 0..s.n_layer {
            let a = set_apply(
                vec![tr("A", 1.0), tr("B", -1.0)],
                0,
                2,
                0.5,
                Mode::Project,
            )
            .expect("単一規則 1");
            assert!(a.apply.as_ref().unwrap().rules.is_empty(), "単一規則に rules が入った");
            a.apply.as_ref().unwrap().apply_one(il, &mut want);
            let b = set_apply(vec![tr("C", 1.0)], 1, 3, 1.25, Mode::Subtract).expect("単一規則 2");
            b.apply.as_ref().unwrap().apply_one(il, &mut want);
        }

        let st = set_rules(vec![
            RuleSpec {
                terms: vec![tr("A", 1.0), tr("B", -1.0)],
                from: 0,
                to: 2,
                alpha: 0.5,
                mode: Mode::Project,
            },
            RuleSpec {
                terms: vec![tr("C", 1.0)],
                from: 1,
                to: 3,
                alpha: 1.25,
                mode: Mode::Subtract,
            },
        ])
        .expect("重ね掛け");
        let a = st.apply.as_ref().unwrap();
        assert_eq!(a.rules.len(), 2, "規則の本数");
        for il in 0..s.n_layer {
            assert!(a.covers(il), "層 {il} を包んでいない");
        }
        assert!(!a.covers(s.n_layer), "層数を超えて包んでいる");
        // 包みは全規則の外側。
        assert_eq!((a.from, a.to), (0, 3));
        // terms は全規則の連結。
        assert_eq!(a.terms.len(), 3);
        let mut got = h0.clone();
        for il in 0..s.n_layer {
            a.apply_one(il, &mut got);
        }
        assert_eq!(got, want, "重ね掛けが逐次適用と 1 ビット一致しない");

        // 範囲外の層は何もしない (単一規則と同じ)。
        let mut outside = h0.clone();
        a.apply_one(s.n_layer, &mut outside);
        a.apply_one(s.n_layer + 3, &mut outside);
        assert_eq!(outside, h0);

        // status は規則ごとの層範囲 / alpha / mode を返す。
        let info = apply_status().expect("status");
        assert_eq!(info.rules.len(), 2);
        assert_eq!((info.rules[0].from, info.rules[0].to), (0, 2));
        assert_eq!(info.rules[0].alpha, 0.5);
        assert_eq!(info.rules[0].mode, Mode::Project);
        assert_eq!((info.rules[1].from, info.rules[1].to), (1, 3));
        assert_eq!(info.rules[1].alpha, 1.25);
        assert_eq!(info.rules[1].mode, Mode::Subtract);
        // レシピ (version 1) へは黙って畳まない。
        assert!(current_recipe("stacked", false).is_err(), "重ね掛けがレシピに落ちた");

        clear_apply();
        assert!(apply_status().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 重ね掛けの入力検証 (壊れた rules は適用しない)。
    #[test]
    fn stacked_rules_reject_bad_input() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("stack-bad");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "stack-model");
        add_snap(&s, "A", 0.25);
        let ok = || RuleSpec {
            terms: vec![tr("A", 1.0)],
            from: 0,
            to: 2,
            alpha: 1.0,
            mode: Mode::Project,
        };
        assert!(set_rules(vec![]).is_err(), "空の rules");
        let mut bad = ok();
        bad.terms = vec![];
        assert!(set_rules(vec![bad]).is_err(), "空の terms");
        let mut bad = ok();
        bad.from = 3;
        assert!(set_rules(vec![bad]).is_err(), "from > to");
        let mut bad = ok();
        bad.to = s.n_layer + 1;
        assert!(set_rules(vec![bad]).is_err(), "to > 層数");
        let mut bad = ok();
        bad.alpha = f32::NAN;
        assert!(set_rules(vec![bad]).is_err(), "alpha が非有限");
        let mut bad = ok();
        bad.terms = vec![tr("ghost", 1.0)];
        assert!(set_rules(vec![bad]).is_err(), "無いスナップショット");
        assert!(
            set_rules(vec![ok(); MAX_RULES + 1]).is_err(),
            "規則が多すぎる"
        );
        // 失敗した適用は状態を変えない。
        assert!(apply_status().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 957: 何も掛けていない既定経路はロックを取らず (`active()` が `None`)、
    /// 規則を 1 本も掛けない。掛けたら層数ぶんだけ発火し、外したら元に戻る。
    #[test]
    fn disabled_dispatch_takes_no_lock_and_applies_nothing() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("dispatch");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "stack-model");
        reset_for_test();
        assert!(!apply_on(), "初期状態で適用が立っている");
        assert!(active().is_none(), "初期状態で active() が Some");

        let h0 = rows(&s, 1, 0.375).pop().unwrap();
        let mut h = h0.clone();
        APPLY_CALLS.store(0, Ordering::Relaxed);
        for il in 0..s.n_layer {
            // ホットパスと同じ形 (`active()` の分岐 1 個)。
            let state = active();
            if let Some(a) = state.as_ref().and_then(|x| x.apply.as_ref()) {
                a.apply_one(il, &mut h);
            }
        }
        assert_eq!(h, h0, "無効時に残差が変わった");
        assert_eq!(
            APPLY_CALLS.load(Ordering::Relaxed),
            0,
            "無効時に規則が掛かった"
        );

        // 掛けたら層数ぶんだけ発火する (分岐が本当に開く)。
        add_snap(&s, "A", 0.25);
        set_apply(vec![tr("A", 1.0)], 0, s.n_layer, 0.5, Mode::Subtract).expect("適用");
        assert!(apply_on(), "設定後に apply_on() が偽");
        assert!(active().is_some(), "設定後に active() が None");
        let mut h = h0.clone();
        APPLY_CALLS.store(0, Ordering::Relaxed);
        for il in 0..s.n_layer {
            let state = active();
            if let Some(a) = state.as_ref().and_then(|x| x.apply.as_ref()) {
                a.apply_one(il, &mut h);
            }
        }
        assert_eq!(
            APPLY_CALLS.load(Ordering::Relaxed),
            s.n_layer,
            "適用の回数が層数と違う"
        );
        assert_ne!(h, h0, "適用しても残差が変わらない");

        // 重ね掛けでも同じ入口で開く。
        set_rules(vec![
            RuleSpec {
                terms: vec![tr("A", 1.0)],
                from: 0,
                to: 2,
                alpha: 0.5,
                mode: Mode::Project,
            },
            RuleSpec {
                terms: vec![tr("A", -1.0)],
                from: 1,
                to: 3,
                alpha: 1.0,
                mode: Mode::Subtract,
            },
        ])
        .expect("重ね掛け");
        assert!(apply_on());
        let mut h = h0.clone();
        APPLY_CALLS.store(0, Ordering::Relaxed);
        for il in 0..s.n_layer {
            let state = active();
            if let Some(a) = state.as_ref().and_then(|x| x.apply.as_ref()) {
                a.apply_one(il, &mut h);
            }
        }
        // 層ごとに `Apply::apply_one` が 1 回 + 規則が層 0:1 本 / 層 1:2 本 / 層 2:1 本。
        assert_eq!(
            APPLY_CALLS.load(Ordering::Relaxed),
            s.n_layer + 1 + 2 + 1,
            "重ね掛けの適用回数"
        );

        // 外したら分岐が閉じる。
        clear_apply();
        assert!(!apply_on(), "clear 後も apply_on() が真");
        assert!(active().is_none(), "clear 後も active() が Some");
        let mut h = h0.clone();
        APPLY_CALLS.store(0, Ordering::Relaxed);
        for il in 0..s.n_layer {
            let state = active();
            if let Some(a) = state.as_ref().and_then(|x| x.apply.as_ref()) {
                a.apply_one(il, &mut h);
            }
        }
        assert_eq!(APPLY_CALLS.load(Ordering::Relaxed), 0, "clear 後に掛かった");
        assert_eq!(h, h0, "clear 後に残差が変わった");
        reset_for_test();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 重ね掛けの project を正規直交化したときの参照 (Gram-Schmidt を f64 で)。
    fn ortho_reference(vhats: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let e = vhats[0].len();
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut out = Vec::new();
        for v in vhats {
            let mut x: Vec<f64> = v.iter().map(|&z| z as f64).collect();
            for q in &basis {
                let d: f64 = x.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                for k in 0..e {
                    x[k] -= d * q[k];
                }
            }
            let n: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
            if n <= 1e-6 {
                out.push(vec![0.0f32; e]);
                continue;
            }
            let q: Vec<f64> = x.iter().map(|a| a / n).collect();
            out.push(q.iter().map(|&a| a as f32).collect());
            basis.push(q);
        }
        out
    }

    fn dotf(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// task 957: `DENNO_STEER_ORTHO` の on/off で、重ね掛け project の合成が
    /// 「逐次 (従来)」と「正規直交基底 (順序に依らず厳密)」に切り替わる。
    #[test]
    fn orthogonal_composition_is_order_independent_and_exact() {
        let _g = LOCK.lock().unwrap();
        let dir = tmpdir("ortho");
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = shape();
        init(s, "stack-model");
        // 2 本の方向が平行でないことを保証する (add_snap の ramps は平行になる)。
        let add_custom = |name: &str, f: &dyn Fn(usize) -> f32| {
            push_snapshot(Arc::new(Snapshot {
                name: name.into(),
                tokens: 4,
                created: 1,
                model: "stack-model".into(),
                layers: s.n_layer,
                n_hc: s.n_hc,
                n_embd: s.n_embd,
                mean: (0..s.len()).map(f).collect(),
            }));
        };
        add_custom("A", &|i| i as f32 * 0.125 + 0.25);
        add_custom("B", &|i| ((i * 37) % 17) as f32 * 0.5 - 3.0);
        // 方向が平行でないこと (vhat の内積が 1 から離れる) を確かめる。
        let snaps = {
            let g = snaps_cell().read().unwrap();
            g.as_ref().clone()
        };
        {
            let a = &snaps.iter().find(|z| z.name == "A").unwrap().mean;
            let b = &snaps.iter().find(|z| z.name == "B").unwrap().mean;
            let na: f32 = a[..s.n_embd].iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b[..s.n_embd].iter().map(|x| x * x).sum::<f32>().sqrt();
            let d: f32 = a[..s.n_embd]
                .iter()
                .zip(&b[..s.n_embd])
                .map(|(x, y)| x * y)
                .sum();
            assert!((d / (na * nb)).abs() < 0.99, "2 本の方向がほぼ平行");
        }
        let specs = vec![
            RuleSpec {
                terms: vec![tr("A", 1.0)],
                from: 0,
                to: s.n_layer,
                alpha: 1.0,
                mode: Mode::Project,
            },
            RuleSpec {
                terms: vec![tr("B", 1.0)],
                from: 0,
                to: s.n_layer,
                alpha: 1.0,
                mode: Mode::Project,
            },
        ];
        let h0 = rows(&s, 1, 0.375).pop().unwrap();

        // 既定 (ortho off) は vortho を作らず、逐次適用と 1 ビット一致。
        let seq = build_rules_apply_opt(&s, &snaps, &specs, false).expect("既定");
        assert!(
            seq.rules.iter().all(|r| r.vortho.is_empty()),
            "off で vortho を作った"
        );
        let mut want = h0.clone();
        for il in 0..s.n_layer {
            for r in &seq.rules {
                r.apply_one(il, &mut want, s.stride(), s.n_embd, s.n_hc);
            }
        }
        let mut got_seq = h0.clone();
        for il in 0..s.n_layer {
            seq.apply_one(il, &mut got_seq);
        }
        assert_eq!(got_seq, want, "既定が逐次適用と違う");

        // ortho on: 各層/ストリームで参照の正規直交基底と一致する。
        // (残差 1 行を層ごとに別々に掛ける。1 本のバッファに層を続けて掛けると
        //  最後の層の結果しか残らないため。)
        let orth = build_rules_apply_opt(&s, &snaps, &specs, true).expect("直交化");
        let per_layer = |a: &Apply| -> Vec<Vec<f32>> {
            (0..s.n_layer)
                .map(|il| {
                    let mut row = h0.clone();
                    a.apply_one(il, &mut row);
                    row
                })
                .collect()
        };
        let got = per_layer(&orth);
        let got_seq_l = per_layer(&seq);
        assert_ne!(got, got_seq_l, "直交化しても値が同じ (ゲートが不発)");
        for il in 0..s.n_layer {
            let row = &got[il];
            for st in 0..s.n_hc {
                let lo = st * s.n_embd;
                let hi = lo + s.n_embd;
                let vhats: Vec<Vec<f32>> = orth
                    .rules
                    .iter()
                    .map(|r| r.vhat[il * s.stride() + lo..il * s.stride() + hi].to_vec())
                    .collect();
                let want_q = ortho_reference(&vhats);
                for (ri, r) in orth.rules.iter().enumerate() {
                    let q = &r.vortho[il * s.stride() + lo..il * s.stride() + hi];
                    for k in 0..s.n_embd {
                        assert!(
                            (q[k] - want_q[ri][k]).abs() <= 1e-5,
                            "層 {il} ストリーム {st} 規則 {ri} 要素 {k}: {} != {}",
                            q[k],
                            want_q[ri][k]
                        );
                    }
                    // 残差の q 方向の成分は落ちている (alpha 1)。
                    assert!(
                        dotf(&row[lo..hi], q).abs() <= 1e-4,
                        "層 {il} ストリーム {st} 規則 {ri} の成分が残る"
                    );
                }
                // 2 本の方向に直交する成分は保存される (直交射影)。
                let mut hh = h0[lo..hi].to_vec();
                for r in &orth.rules {
                    let q = &r.vortho[il * s.stride() + lo..il * s.stride() + hi];
                    let d = dotf(&hh, q);
                    for k in 0..s.n_embd {
                        hh[k] -= d * q[k];
                    }
                }
                for k in 0..s.n_embd {
                    assert!(
                        (row[lo + k] - hh[k]).abs() <= 1e-5,
                        "層 {il} ストリーム {st} 要素 {k} が閉じた式と違う"
                    );
                }
            }
        }

        // 2 回目は不変 (冪等)。
        for il in 0..s.n_layer {
            let mut twice = got[il].clone();
            orth.apply_one(il, &mut twice);
            for k in 0..twice.len() {
                assert!(
                    (twice[k] - got[il][k]).abs() <= 1e-5,
                    "直交化した射影が冪等でない (層 {il} 要素 {k})"
                );
            }
        }

        // 規則の順を入れ替えても結果が変わらない (順序非依存)。
        let swapped = build_rules_apply_opt(&s, &snaps, &[specs[1].clone(), specs[0].clone()], true)
            .expect("入れ替え");
        let got_sw = per_layer(&swapped);
        for il in 0..s.n_layer {
            for k in 0..s.n_embd * s.n_hc {
                assert!(
                    (got_sw[il][k] - got[il][k]).abs() <= 1e-5,
                    "規則の順で結果が変わる (層 {il} 要素 {k})"
                );
            }
        }
        reset_for_test();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 957: `DENNO_STEER_ORTHO` は既定 off で、`=1` のときだけ効く。
    #[test]
    fn ortho_env_gate_defaults_off() {
        let _g = LOCK.lock().unwrap();
        std::env::remove_var("DENNO_STEER_ORTHO");
        assert!(!ortho_on(), "既定で on になっている");
        std::env::set_var("DENNO_STEER_ORTHO", "0");
        assert!(!ortho_on(), "=0 で on になっている");
        std::env::set_var("DENNO_STEER_ORTHO", "");
        assert!(!ortho_on(), "空文字で on になっている");
        std::env::set_var("DENNO_STEER_ORTHO", "1");
        assert!(ortho_on(), "=1 で on にならない");
        std::env::remove_var("DENNO_STEER_ORTHO");
    }
}
