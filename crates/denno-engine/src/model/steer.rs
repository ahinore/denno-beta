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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock, RwLock};

/// スナップショット名に使える最大長。
pub const MAX_NAME: usize = 64;
/// スナップショットファイルの先頭行。
pub const FILE_MAGIC: &str = "DNSTEER1";
/// スナップショットファイルの拡張子。
pub const FILE_EXT: &str = "dns";
/// 同時に保持するスナップショット数の上限 (メモリ保護)。
pub const MAX_SNAPSHOTS: usize = 64;

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
#[derive(Clone, Debug)]
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

/// いま効いている適用。
#[derive(Clone, Debug)]
pub struct Apply {
    pub terms: Vec<Term>,
    /// `[from, to)`。`to` は含まない。
    pub from: usize,
    pub to: usize,
    pub alpha: f32,
    pub mode: Mode,
    /// `[layer][stream][n_embd]` の合成ベクトル。
    pub v: Vec<f32>,
    /// `v` を (layer, stream) ごとに正規化したもの。
    pub vhat: Vec<f32>,
    shape: Shape,
}

impl Apply {
    #[inline]
    pub fn covers(&self, layer: usize) -> bool {
        layer >= self.from && layer < self.to
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

/// 記録中か (ホットパスの分岐 1 個)。
static REC_ON: AtomicBool = AtomicBool::new(false);

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
    let mut vhat = vec![0.0f32; v.len()];
    let e = shape.n_embd;
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
    Ok(Apply {
        terms: terms.to_vec(),
        from,
        to,
        alpha,
        mode,
        v,
        vhat,
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

/// 適用を消す (epoch が進む)。
pub fn clear_apply() -> Arc<Steer> {
    let epoch = super::intervene::bump_epoch();
    let mut g = cur_cell().write().unwrap_or_else(|e| e.into_inner());
    let s = Arc::new(Steer::empty(epoch));
    *g = Arc::clone(&s);
    s
}

// ---------------------------------------------------------------------------
// 記録
// ---------------------------------------------------------------------------

struct Rec {
    name: String,
    model: String,
    shape: Shape,
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
    fn new(name: &str, shape: Shape, model: &str) -> Rec {
        Rec {
            name: name.to_string(),
            model: model.to_string(),
            shape,
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

/// ホストの残差を 1 層ぶん記録する (`data` = 1 トークンの `[hc][n_embd]`)。
///
/// decode と非常駐 prefill が呼ぶ。[`rec_on`] が false なら即返る。
pub fn acc_host(layer: usize, data: &[f32]) {
    if !rec_on() {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.as_mut() {
        r.acc(layer, data);
    }
}

/// ホストの残差をチャンクぶん (行の配列) 記録する。
pub fn acc_host_rows(layer: usize, rows: &[&[f32]]) {
    if !rec_on() {
        return;
    }
    let mut g = rec_cell().lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = g.as_mut() else { return };
    for row in rows {
        r.acc(layer, row);
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
pub fn start_record(name: &str) -> Result<(String, u64), String> {
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
    *g = Some(Rec::new(name, shape, &model));
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
    let path = snap_path(&dir, &s.name);
    let tmp = dir.join(format!(".{}.{}.tmp", s.name, std::process::id()));
    std::fs::write(&tmp, &buf).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// 1 ファイル読む。壊れていれば `Err`。
pub fn load_file(path: &Path) -> Result<Snapshot, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let magic = FILE_MAGIC.as_bytes();
    if raw.len() < magic.len() + 1 || &raw[..magic.len()] != magic || raw[magic.len()] != b'\n' {
        return Err(format!("{}: 先頭が {FILE_MAGIC} でない", path.display()));
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
            .ok_or_else(|| format!("{}: ヘッダが終わらない", path.display()))?;
        let line = std::str::from_utf8(&raw[off..end])
            .map_err(|_| format!("{}: ヘッダが ASCII でない", path.display()))?;
        off = end + 1;
        if line == "data" {
            break;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| format!("{}: ヘッダ行 {line:?} に '=' が無い", path.display()))?;
        let num = |v: &str| -> Result<u64, String> {
            v.parse::<u64>()
                .map_err(|_| format!("{}: {k} が数値でない: {v:?}", path.display()))
        };
        match k {
            "name" => name = v.to_string(),
            "tokens" => tokens = num(v)?,
            "created" => created = num(v)?,
            "model" => model = v.to_string(),
            "layers" => layers = num(v)? as usize,
            "n_hc" => n_hc = num(v)? as usize,
            "n_embd" => n_embd = num(v)? as usize,
            other => {
                return Err(format!(
                    "{}: 未知のヘッダ {other:?}",
                    path.display()
                ))
            }
        }
    }
    let shape = Shape {
        n_layer: layers,
        n_hc,
        n_embd,
    };
    if !shape.is_valid() {
        return Err(format!("{}: 形が不正", path.display()));
    }
    let want = shape.len() * 4;
    let have = raw.len() - off;
    if have != want {
        return Err(format!(
            "{}: 本体の長さが違う ({have} バイト / 期待 {want})",
            path.display()
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
    *rec_cell().lock().unwrap_or_else(|e| e.into_inner()) = None;
    let epoch = super::intervene::bump_epoch();
    *cur_cell().write().unwrap_or_else(|e| e.into_inner()) = Arc::new(Steer::empty(epoch));
    *snaps_cell().write().unwrap_or_else(|e| e.into_inner()) = Arc::new(Vec::new());
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
