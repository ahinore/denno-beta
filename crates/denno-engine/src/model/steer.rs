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
}
