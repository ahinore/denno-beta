//! GPU を使わずに検証できる「VRAM / ホスト RAM の予算見積り」と
//! 「重みアップロード失敗の集計」(P-pf-arena)。
//!
//! # なぜここに置くか
//!
//! `--max-ctx` を上げたときに `gguf weight register failed` が増える、という
//! 実測 (メインセッション) の原因を、GPU 無しで切り分けるための材料を 2 つ置く。
//!
//! 1. [`flash_budget`] — モデルのテンソル表から「デバイスに載る重み」「mmap の
//!    ままの重み」「max_ctx に比例するホスト側 KV」を**式で**出す純関数。
//!    起動時に 1 行だけログへ出せば、どの項が VRAM を食うかが実測前に分かる。
//! 2. [`WeightUploadStats`] — `cuda.rs` の重み登録が何本成功し何本失敗したかの
//!    集計。40,554 行の ERROR を 1 行に畳むためのカウンタで、こちらは
//!    `cuda` feature の有無に関係なく存在する (feature 無しでは 0 のまま)。
//!
//! # 数値経路には触らない
//!
//! ここは「数える・見積もる・整形する」だけ。カーネル引数・発行順・丸め順は
//! 1 つも変えない。`cuda.rs` 側の変更は集計とログの抑制だけで、GPU に渡す値は
//! 同一 (ビット一致の議論が不要な種類の変更)。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::qwen4exp::Qwen4ExpParams;

/// `cuda.rs` のスクラッチ arena の成長単位 (1 GiB)。あちらは usize なので
/// ここでは u64 で持ち、`cuda.rs` 側の値と一致することをテストで縛る。
pub const SCRATCH_ARENA_CHUNK: u64 = 1 << 30;

/// デバイス常駐レジストリに入る量子型か。
///
/// `cuda.rs::gguf_row_bytes` が `Some` を返す集合と同じ (Q8_0 / Q5_1 / Q4_K / Q5_K)。
/// これ以外 (F32 / F16 / BF16) は `GgufLinear::gemv` の CPU 参照へ落ちるので
/// VRAM を消費しない。Q4_0 は別レジストリ (`weights_for_q4`) の担当で、
/// Flash-Next (`qwen4exp`) の線形層はこの型を使わない。
pub fn is_device_weight(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K
    )
}

/// MoE の expert 行列か (`ffn_gate_exps.weight` など)。
///
/// expert は `ExpertW` (mmap / SSD pread) のままで、`expert_gemv_any` が CPU で
/// 読む。`cuda.rs` の重みレジストリには入らない (＝ VRAM を食わない)。
pub fn is_expert_weight(name: &str) -> bool {
    name.ends_with("_exps.weight")
}

/// テンソル表を「デバイスに載る / mmap のまま / expert」に分けた合計。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeightTotals {
    /// デバイス常駐になる重みのバイト数 (量子型のみ)
    pub dev_bytes: u64,
    /// その本数
    pub dev_tensors: usize,
    /// mmap のまま使う重み (F32 の hc_* / ルータ / norm など)
    pub host_bytes: u64,
    /// その本数
    pub host_tensors: usize,
    /// expert 行列 (mmap / pread)
    pub expert_bytes: u64,
    /// その本数
    pub expert_tensors: usize,
    /// デバイスに載る重みの最大入力長 (1 回の GEMV の staging 見積り用)
    pub max_k: u64,
    /// デバイスに載る重みの最大出力長
    pub max_n: u64,
}

/// `(名前, 型, 次元, バイト数)` を分類する純関数。
///
/// `GgufFile::tensors` の 1 要素からそのまま作れる形にしてあるので、
/// テストでは偽のイテレータを渡せる (GGUF を書かなくてよい)。
pub fn classify_weights<'a, I>(tensors: I) -> WeightTotals
where
    I: IntoIterator<Item = (&'a str, GgmlType, [u64; 4], u64)>,
{
    let mut t = WeightTotals::default();
    for (name, ty, dims, nbytes) in tensors {
        if is_expert_weight(name) {
            t.expert_bytes += nbytes;
            t.expert_tensors += 1;
            continue;
        }
        if is_device_weight(ty) {
            t.dev_bytes += nbytes;
            t.dev_tensors += 1;
            t.max_k = t.max_k.max(dims[0]);
            t.max_n = t.max_n.max(dims[1]);
        } else {
            t.host_bytes += nbytes;
            t.host_tensors += 1;
        }
    }
    t
}

/// Flash-Next の VRAM / RAM の内訳見積り (すべてバイト)。
///
/// 「max_ctx に比例する項」は [`FlashBudget::kv_host_max`] **だけ**で、しかも
/// それはホスト RAM 側 (`AttnCache` の `Vec<f32>`) である。GPU 側の項
/// ([`FlashBudget::weights_dev`] / [`FlashBudget::staging_peak`] /
/// [`FlashBudget::arena_dev`]) に `max_ctx` は入らない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashBudget {
    /// 見積りに使った上限コンテキスト長
    pub max_ctx: usize,
    /// デバイス常駐の重み (全種類を 1 度ずつ触った場合の上限)
    pub weights_dev: u64,
    /// その本数
    pub weights_dev_tensors: usize,
    /// `moe_gemv_dev` の入出力スクラッチの上限 (x + out)。
    /// P-pf-moe-scratch 以降は thread_local の grow-only で**常駐**する
    /// (1 スレッドあたり最大 max_k + max_n 要素。呼び出しスレッド数だけ積み上がる)。
    pub staging_peak: u64,
    /// q4 経路のスクラッチ arena (1 GiB チャンク単位・grow-only)。
    /// Flash-Next は q4 の線形層を持たないので 0。
    pub arena_dev: u64,
    /// `max_ctx` を使い切った場合の KV (ホスト RAM)。
    /// 実際は積んだトークン数までしか伸びない (`AttnCache::push` が resize)。
    pub kv_host_max: u64,
    /// mmap のまま置く重み (F32 など)
    pub host_weights: u64,
    /// expert 行列 (mmap / pread)
    pub experts_ram: u64,
}

/// full attention 層の本数 (`full_attention_interval` ごとに 1 本)。
pub fn n_full_layers(n_layer: usize, full_attn_interval: usize) -> usize {
    if full_attn_interval == 0 {
        return 0;
    }
    n_layer.div_ceil(full_attn_interval)
}

/// [`FlashBudget`] を式から組む純関数。
///
/// - `kv_host_max = full 層数 × max_ctx × kv_dim × 2(K,V) × 4 B`
///   (`AttnCache` のレイアウト。層ごとに k/v を `Vec<f32>` で持つ)
/// - `staging_peak = (max_k + max_n) × 4 B` (`moe_gemv_dev` の x + out の
///   スクラッチ上限。P-pf-moe-scratch で per-call alloc/free から grow-only の
///   常駐に変わったが、**上限の式と値は同じ**なので項は増やさない)
/// - `arena_dev = q4 の重みがあるときだけ 1 GiB` (チャンク単位。無ければ 0)
pub fn flash_budget(
    totals: WeightTotals,
    full_layers: usize,
    kv_dim: usize,
    max_ctx: usize,
    q4_present: bool,
) -> FlashBudget {
    let kv_host_max = (full_layers as u64)
        .saturating_mul(max_ctx as u64)
        .saturating_mul(kv_dim as u64)
        .saturating_mul(2)
        .saturating_mul(4);
    let staging_peak = totals
        .max_k
        .saturating_add(totals.max_n)
        .saturating_mul(std::mem::size_of::<f32>() as u64);
    FlashBudget {
        max_ctx,
        weights_dev: totals.dev_bytes,
        weights_dev_tensors: totals.dev_tensors,
        staging_peak,
        arena_dev: if q4_present { SCRATCH_ARENA_CHUNK } else { 0 },
        kv_host_max,
        host_weights: totals.host_bytes,
        experts_ram: totals.expert_bytes,
    }
}

/// GGUF とメタデータから [`FlashBudget`] を作る (起動時に 1 回)。
pub fn flash_budget_from_gguf(f: &GgufFile, p: &Qwen4ExpParams, max_ctx: usize) -> FlashBudget {
    let totals = classify_weights(f.tensors.values().map(|t| {
        (
            t.name.as_str(),
            t.ty,
            t.dims,
            t.nbytes,
        )
    }));
    // Flash-Next の線形層は Q4_0 を使わない (Q4_0 があるなら q4 経路 = arena が動く)。
    let q4_present = f
        .tensors
        .values()
        .any(|t| t.ty == GgmlType::Q4_0 && !is_expert_weight(&t.name));
    let kv_dim = p.n_head_kv.saturating_mul(p.n_embd_head_k);
    flash_budget(
        totals,
        n_full_layers(p.n_layer, p.full_attn_interval),
        kv_dim,
        max_ctx,
        q4_present,
    )
}

/// MiB (四捨五入)。
fn mib(bytes: u64) -> u64 {
    bytes.saturating_add(1 << 19) >> 20
}

/// 起動時に 1 行だけ出す整形。
///
/// `weights` は「デバイスに載り得る重みの上限」で、実際に載った本数は
/// リクエスト後の [`summary_line`] が出す。
pub fn budget_line(b: &FlashBudget) -> String {
    format!(
        "[flash] vram budget: max_ctx={} weights={} MiB ({} 本) staging={} MiB arena={} MiB kv={} MiB (host, max_ctx 比例) host_weights={} MiB experts={} MiB (mmap)",
        b.max_ctx,
        mib(b.weights_dev),
        b.weights_dev_tensors,
        mib(b.staging_peak),
        mib(b.arena_dev),
        mib(b.kv_host_max),
        mib(b.host_weights),
        mib(b.experts_ram),
    )
}

// ---------------------------------------------------------------------------
// 重みアップロードの集計 (cuda.rs から呼ぶ)
// ---------------------------------------------------------------------------

/// 失敗ログを出す間隔 (これおきに 1 行)。1 回目は必ず出す。
pub const FAIL_LOG_EVERY: u64 = 1000;

/// 何回目の失敗かから「出す / 出さない」を決める純関数。
///
/// - 1 回目 → `Some(true)` = ERROR (原因が読める 1 行)
/// - `FAIL_LOG_EVERY` の倍数 → `Some(false)` = WARN (進捗)
/// - それ以外 → `None` (出さない。数えるだけ)
///
/// これが無いと 1 リクエストで 40,554 行の ERROR が出る (実測)。ログは
/// 「原因 1 行 + 1,000 回ごとの進捗 + リクエスト末尾の集計 1 行」に畳む。
pub fn register_log_for(fail_count: u64) -> Option<bool> {
    if fail_count == 1 {
        Some(true)
    } else if fail_count != 0 && fail_count % FAIL_LOG_EVERY == 0 {
        Some(false)
    } else {
        None
    }
}

static UPLOADED: AtomicU64 = AtomicU64::new(0);
static UPLOADED_BYTES: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static LAST_FAILED_UID: AtomicU64 = AtomicU64::new(u64::MAX);
/// レジストリ常駐 (cuda.rs が登録・退避のたびに絶対値を書く)
static RESIDENT: AtomicU64 = AtomicU64::new(0);
static RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
/// LRU の退避 (プロセス累積)
static EVICTED: AtomicU64 = AtomicU64::new(0);
static EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);
static GPU_BROKEN: AtomicBool = AtomicBool::new(false);
static FIRST_ERROR: Mutex<Option<String>> = Mutex::new(None);
static BROKEN_CAUSE: Mutex<Option<String>> = Mutex::new(None);

/// 重み登録の集計 (1 リクエスト分・[`reset_weight_upload_stats`] で 0 に戻る)。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightUploadStats {
    /// VRAM へ上がった本数
    pub uploaded: u64,
    /// 上がらなかった本数 (登録を試みた回数で数える)
    pub failed: u64,
    /// VRAM へ上がった合計バイト数 (登録 1 本ごとの `n * row_bytes` の和)
    pub uploaded_bytes: u64,
    /// 最後に失敗した uid
    pub last_failed_uid: Option<u64>,
    /// 最初の失敗のメッセージ (原因はこれ 1 本で足りる)
    pub first_error: Option<String>,
    /// 重みの失敗で GPU 経路全体を止めたか (`cuda::GPU_BROKEN`)
    pub gpu_broken: bool,
    /// 止めた原因 (最初の 1 件)
    pub broken_cause: Option<String>,
    /// 今レジストリに常駐している本数 (プロセス累積。`reset` しない)
    pub resident: usize,
    /// 常駐の合計バイト数 (同上)
    pub resident_bytes: u64,
    /// 上限 (バイト)。0 = 無制限 (既定)
    pub cap_bytes: u64,
    /// LRU が退避した本数 (プロセス累積。`reset` しない)
    pub evicted: u64,
    /// 退避した合計バイト数 (同上)
    pub evicted_bytes: u64,
}

impl WeightUploadStats {
    /// 登録を試みた総数
    pub fn attempted(&self) -> u64 {
        self.uploaded.saturating_add(self.failed)
    }
    /// 失敗率 (%)。試行 0 は 0。
    pub fn fail_percent(&self) -> f64 {
        let n = self.attempted();
        if n == 0 {
            0.0
        } else {
            self.failed as f64 * 100.0 / n as f64
        }
    }
}

/// VRAM へ 1 本上がった。
pub fn note_weight_upload() {
    UPLOADED.fetch_add(1, Ordering::Relaxed);
}

/// VRAM へ上がった 1 本ぶんのバイト数を足す (P46: 合計 bytes を 1 行で出すため)。
pub fn note_weight_upload_bytes(bytes: u64) {
    UPLOADED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// レジストリの常駐量 (本数 / バイト数) を**絶対値**で書く。
///
/// 登録・退避のたびに cuda.rs が現在値を渡す (`reset` では戻さない =
/// リクエストの切れ目とは無関係なプロセスの状態)。
pub fn note_weight_resident(tensors: usize, bytes: u64) {
    RESIDENT.store(tensors as u64, Ordering::Relaxed);
    RESIDENT_BYTES.store(bytes, Ordering::Relaxed);
}

/// LRU が 1 本以上退避した (プロセス累積)。
pub fn note_weight_evicted(tensors: u64, bytes: u64) {
    EVICTED.fetch_add(tensors, Ordering::Relaxed);
    EVICTED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// 1 本失敗した。出すべきログの種別を返す ([`register_log_for`])。
///
/// `err` は整形済みのメッセージ (`format!("{e:#}")`)。最初の 1 件だけ保持する
/// (VRAM が埋まった後の失敗は同じ理由が並ぶので、最後ではなく最初が診断に効く)。
pub fn note_weight_failure(uid: u64, err: &str) -> Option<bool> {
    LAST_FAILED_UID.store(uid, Ordering::Relaxed);
    if let Ok(mut g) = FIRST_ERROR.lock() {
        if g.is_none() {
            *g = Some(err.to_string());
        }
    }
    let n = FAILED.fetch_add(1, Ordering::Relaxed) + 1;
    register_log_for(n)
}

/// 重みの失敗で GPU 経路全体を止めた (`GPU_BROKEN`)。原因は最初の 1 件だけ残す。
pub fn note_gpu_broken(cause: &str) {
    GPU_BROKEN.store(true, Ordering::Relaxed);
    if let Ok(mut g) = BROKEN_CAUSE.lock() {
        if g.is_none() {
            *g = Some(cause.to_string());
        }
    }
}

/// 現在の集計。
pub fn weight_upload_stats() -> WeightUploadStats {
    let last = LAST_FAILED_UID.load(Ordering::Relaxed);
    WeightUploadStats {
        uploaded: UPLOADED.load(Ordering::Relaxed),
        failed: FAILED.load(Ordering::Relaxed),
        uploaded_bytes: UPLOADED_BYTES.load(Ordering::Relaxed),
        last_failed_uid: if last == u64::MAX { None } else { Some(last) },
        first_error: FIRST_ERROR.lock().ok().and_then(|g| g.clone()),
        gpu_broken: gpu_broken_now(),
        broken_cause: BROKEN_CAUSE.lock().ok().and_then(|g| g.clone()),
        resident: RESIDENT.load(Ordering::Relaxed) as usize,
        resident_bytes: RESIDENT_BYTES.load(Ordering::Relaxed),
        cap_bytes: weight_cap_bytes(),
        evicted: EVICTED.load(Ordering::Relaxed),
        evicted_bytes: EVICTED_BYTES.load(Ordering::Relaxed),
    }
}

/// GPU 経路が恒久停止しているか。
///
/// `cuda` feature があるときは `cuda::GPU_BROKEN` (重み以外の失敗でも立つ) を
/// そのまま読み、無いときはこのモジュールのカウンタだけを見る。
fn gpu_broken_now() -> bool {
    #[cfg(feature = "cuda")]
    {
        if crate::cuda::gpu_broken_flag() {
            return true;
        }
    }
    GPU_BROKEN.load(Ordering::Relaxed)
}

/// リクエスト境界で 0 に戻す (`GPU_BROKEN` は**戻さない** — プロセス寿命のラッチ)。
pub fn reset_weight_upload_stats() {
    UPLOADED.store(0, Ordering::Relaxed);
    UPLOADED_BYTES.store(0, Ordering::Relaxed);
    FAILED.store(0, Ordering::Relaxed);
    LAST_FAILED_UID.store(u64::MAX, Ordering::Relaxed);
    if let Ok(mut g) = FIRST_ERROR.lock() {
        *g = None;
    }
}

/// リクエスト末尾に出す 1 行。「何枚中何枚が失敗したか」が読める形にする。
pub fn summary_line(s: &WeightUploadStats) -> String {
    if s.attempted() == 0 {
        let mut line = "[flash] gpu weights: 0 本 ok / 0 本 失敗 (登録を試みていない)".to_string();
        line.push_str(&lru_suffix(s));
        return line;
    }
    let mut line = format!(
        "[flash] gpu weights: {} 本 ok / {} 本 失敗 (全 {} 本のうち {:.1}%)",
        s.uploaded,
        s.failed,
        s.attempted(),
        s.fail_percent()
    );
    if let Some(uid) = s.last_failed_uid {
        line.push_str(&format!(" last_uid={uid}"));
    }
    if let Some(e) = &s.first_error {
        line.push_str(&format!(" 最初の失敗: {e}"));
    }
    if s.gpu_broken {
        line.push_str(&format!(
            " GPU_BROKEN=true (以降 CPU 経路: {})",
            s.broken_cause.as_deref().unwrap_or("原因不明")
        ));
    }
    line.push_str(&lru_suffix(s));
    line
}

/// P46: レジストリの常駐 / 上限 / 退避の 1 段落 (リクエスト末尾の 1 行に足す)。
///
/// 「登録本数 / 合計 bytes / 失敗回数」は本体側に出ているので、ここは
/// 「今 VRAM に何本・何 MiB 残っているか」「上限は効いているか」を出す。
fn lru_suffix(s: &WeightUploadStats) -> String {
    let cap = if s.cap_bytes == 0 {
        "無制限".to_string()
    } else {
        format!("上限 {} MiB", mib(s.cap_bytes))
    };
    let mut out = format!(
        " [gguf 常駐 {} 本 {} MiB / {} / 転送計 {} MiB",
        s.resident,
        mib(s.resident_bytes),
        cap,
        mib(s.uploaded_bytes)
    );
    if s.evicted > 0 {
        out.push_str(&format!(" / 退避 計 {} 本 {} MiB", s.evicted, mib(s.evicted_bytes)));
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// P46: 重みレジストリのバイト上限つき LRU
// ---------------------------------------------------------------------------
//
// 8098 の VRAM がリクエストごとに約 4.5 GB 増える問題 (実測: 10,286 →
// 14,800 → 19,346 MiB) への対策。Flash-Next の重みレジストリ
// (`cuda.rs::weights_for_gguf`) は uid 単位でしか一致を見ない。モデルを
// 組み直すと uid が払い直されるので、同じ重みが別 uid としてもう一度載る。
// ここでは「レジストリ全体で何バイトまで」を env で縛り、超えたら最終使用が
// 最も古い uid から解放する。
//
// 既定 (`DENNO_GPU_WEIGHT_MB` 未設定 / 0) は上限なし = 退避 0 本 =
// 従来と同じ挙動・同じ数値。

/// `DENNO_GPU_WEIGHT_MB` の解釈 (純関数)。単位は MiB。
///
/// 未設定・空・0・読めない値は 0 = 無制限 (現状と同じ挙動)。
pub fn parse_weight_cap_mb(raw: Option<&str>) -> u64 {
    match raw {
        Some(s) => s.trim().parse::<u64>().unwrap_or(0),
        None => 0,
    }
}

/// 重みレジストリの上限 (バイト)。env はプロセスで 1 度だけ読む。
pub fn weight_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        parse_weight_cap_mb(std::env::var("DENNO_GPU_WEIGHT_MB").ok().as_deref())
            .saturating_mul(1 << 20)
    })
}

/// LRU の 1 本ぶんの記帳。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LruEntry {
    bytes: u64,
    tick: u64,
    pins: u32,
}

/// バイト上限つき LRU (デバイスには触らない純データ構造)。
///
/// 規約:
/// - `cap_bytes == 0` は無制限。`note_insert` は記帳だけして 1 本も返さない。
/// - 上限超過時は「最終使用が最も古い uid」から返す。`pins > 0` (使用中) と
///   登録した本人は飛ばす。
/// - 1 本だけで上限を超える登録は、その 1 本を落とさない (登録直後に使うため)。
///   このとき上限は一時的に超過したままになる。
#[derive(Debug, Default, Clone)]
pub struct WeightLru {
    cap_bytes: u64,
    used_bytes: u64,
    tick: u64,
    evicted: u64,
    evicted_bytes: u64,
    map: HashMap<u64, LruEntry>,
}

impl WeightLru {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            ..Default::default()
        }
    }

    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }
    /// 常駐分の合計バイト数 (上限判定に使う値そのもの)
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn contains(&self, uid: u64) -> bool {
        self.map.contains_key(&uid)
    }
    /// 使用中の本数 (`pins`)。テストと診断用。
    pub fn pins(&self, uid: u64) -> u32 {
        self.map.get(&uid).map(|e| e.pins).unwrap_or(0)
    }
    /// これまでに退避した本数 (プロセス累積)
    pub fn evicted_tensors(&self) -> u64 {
        self.evicted
    }
    /// これまでに退避したバイト数 (プロセス累積)
    pub fn evicted_bytes(&self) -> u64 {
        self.evicted_bytes
    }

    /// 新規登録を記帳し、上限に収めるために**解放してよい uid** を古い順に返す。
    ///
    /// 返した uid はこの構造体からは既に消えている (`used_bytes` も減っている)
    /// ので、呼び出し側は自分の表から消して実体を Drop するだけでよい。
    pub fn note_insert(&mut self, uid: u64, bytes: u64) -> Vec<u64> {
        if let Some(old) = self.map.remove(&uid) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        self.tick = self.tick.saturating_add(1);
        self.map.insert(
            uid,
            LruEntry {
                bytes,
                tick: self.tick,
                pins: 0,
            },
        );
        self.used_bytes = self.used_bytes.saturating_add(bytes);

        let mut out = Vec::new();
        if self.cap_bytes == 0 {
            return out;
        }
        while self.used_bytes > self.cap_bytes {
            let Some(victim) = self
                .map
                .iter()
                .filter(|(k, e)| **k != uid && e.pins == 0)
                .min_by_key(|(k, e)| (e.tick, **k))
                .map(|(k, _)| *k)
            else {
                break;
            };
            self.evict(victim);
            out.push(victim);
        }
        out
    }

    /// 退避する 1 本を表から外す (退避の集計も更新)。戻り値は解放したバイト数。
    pub fn evict(&mut self, uid: u64) -> u64 {
        let b = self.note_remove(uid);
        if b > 0 {
            self.evicted += 1;
            self.evicted_bytes = self.evicted_bytes.saturating_add(b);
        }
        b
    }

    /// 使った = 最終使用時刻を今にする。
    pub fn touch(&mut self, uid: u64) {
        self.tick = self.tick.saturating_add(1);
        if let Some(e) = self.map.get_mut(&uid) {
            e.tick = self.tick;
        }
    }

    /// 使用中にする (この uid は退避対象から外れる)。未登録なら `false`。
    pub fn pin(&mut self, uid: u64) -> bool {
        match self.map.get_mut(&uid) {
            Some(e) => {
                e.pins = e.pins.saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// 使用中をやめる。
    pub fn unpin(&mut self, uid: u64) {
        if let Some(e) = self.map.get_mut(&uid) {
            e.pins = e.pins.saturating_sub(1);
        }
    }

    /// 登録を消す (確保失敗の後始末・退避の実体解放と対で呼ぶ)。
    /// 戻り値は減らしたバイト数。
    pub fn note_remove(&mut self, uid: u64) -> u64 {
        match self.map.remove(&uid) {
            Some(e) => {
                self.used_bytes = self.used_bytes.saturating_sub(e.bytes);
                e.bytes
            }
            None => 0,
        }
    }

    /// 退避候補 (最終使用が最も古い・未使用中・`except` 以外) の uid。
    pub fn oldest_unpinned(&self, except: u64) -> Option<u64> {
        self.map
            .iter()
            .filter(|(k, e)| **k != except && e.pins == 0)
            .min_by_key(|(k, e)| (e.tick, **k))
            .map(|(k, _)| *k)
    }
}

/// 上限が有効なときに、確保後の空き VRAM へ残す余裕 (MiB)。
///
/// `cuMemAlloc` の失敗を通常経路にしないための遊び。上限 0 (既定) のときは
/// 使わない。
pub const GPU_W_RESERVE_MB: u64 = 256;

/// 確保の前に落とすべき uid を決める (cuda.rs の登録経路から呼ぶ純ロジック)。
///
/// 1. バイト上限 — [`WeightLru::note_insert`] が上限に収めるために返した分
/// 2. 空き VRAM — `cap_bytes != 0` かつ `free_mb` が分かるとき、確保後に
///    [`GPU_W_RESERVE_MB`] を残せるまで「最終使用が最も古い未使用中」を追加
///
/// 戻り値は解放してよい uid (古い順)。[`WeightLru`] 側の記帳 (使用バイト数と
/// 退避の集計) もこの関数の中で済む。`free_mb = None` は「空きが分からない」
/// なので 1 だけやる。
pub fn plan_weight_evictions(
    lru: &mut WeightLru,
    uid: u64,
    want_bytes: u64,
    cap_bytes: u64,
    free_mb: Option<u64>,
) -> Vec<u64> {
    let before = lru.evicted_bytes();
    let mut victims = lru.note_insert(uid, want_bytes);
    if cap_bytes == 0 {
        // 既定 (無制限) はここで終わり = 1 本も落とさない。
        return victims;
    }
    let Some(free_mb) = free_mb else {
        return victims;
    };
    let mut freed = lru.evicted_bytes().saturating_sub(before);
    let floor_mb = (want_bytes >> 20).saturating_add(1).saturating_add(GPU_W_RESERVE_MB);
    let mut free_now = free_mb.saturating_add(freed >> 20);
    while free_now < floor_mb {
        let Some(v) = lru.oldest_unpinned(uid) else {
            break;
        };
        let b = lru.evict(v);
        freed = freed.saturating_add(b);
        free_now = free_now.saturating_add(b >> 20);
        victims.push(v);
    }
    victims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, ty: GgmlType, k: u64, n: u64, bytes: u64) -> (String, GgmlType, [u64; 4], u64) {
        (name.to_string(), ty, [k, n, 1, 1], bytes)
    }

    #[test]
    fn device_set_matches_cuda_registry() {
        // cuda.rs::gguf_row_bytes が Some を返す 4 型だけがデバイス常駐になる。
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_1,
            GgmlType::Q4K,
            GgmlType::Q5K,
        ] {
            assert!(is_device_weight(ty), "{ty:?} は載る側のはず");
        }
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q4_0,
            GgmlType::Q6K,
        ] {
            assert!(!is_device_weight(ty), "{ty:?} は載らない側のはず");
        }
    }

    #[test]
    fn classify_splits_device_host_and_experts() {
        let rows = vec![
            t("blk.0.attn_q.weight", GgmlType::Q5_1, 2560, 12288, 1000),
            t("blk.0.ffn_gate_exps.weight", GgmlType::Q4K, 2560, 32768, 5000),
            t("blk.0.ffn_gate_shexp.weight", GgmlType::Q8_0, 2560, 640, 300),
            t("blk.0.ffn_gate_inp.weight", GgmlType::F32, 2560, 512, 200),
            t("output_norm.weight", GgmlType::F32, 2560, 1, 10),
        ];
        let w = classify_weights(rows.iter().map(|r| (r.0.as_str(), r.1, r.2, r.3)));
        assert_eq!(w.dev_bytes, 1300, "Q5_1 と Q8_0 の 2 本だけがデバイス側");
        assert_eq!(w.dev_tensors, 2);
        assert_eq!(w.expert_bytes, 5000, "expert は mmap のまま");
        assert_eq!(w.expert_tensors, 1);
        assert_eq!(w.host_bytes, 210, "F32 は CPU 経路");
        assert_eq!(w.host_tensors, 2);
        assert_eq!(w.max_k, 2560);
        assert_eq!(w.max_n, 12288, "最大出力長は attn_q の 12288");
    }

    #[test]
    fn kv_term_is_the_only_max_ctx_proportional_term() {
        let totals = WeightTotals {
            dev_bytes: 6_000_000_000,
            dev_tensors: 400,
            max_k: 2560,
            max_n: 248_320,
            ..Default::default()
        };
        let a = flash_budget(totals, 12, 512, 16_384, false);
        let b = flash_budget(totals, 12, 512, 32_768, false);
        // max_ctx が 2 倍になると KV (host) だけが 2 倍。
        assert_eq!(b.kv_host_max, a.kv_host_max * 2);
        assert_eq!(a.weights_dev, b.weights_dev);
        assert_eq!(a.staging_peak, b.staging_peak);
        assert_eq!(a.arena_dev, 0, "Flash-Next は q4 経路を通らない");
        // 12 層 × 16384 tok × 512 × 2 × 4 B = 768 MiB
        assert_eq!(a.kv_host_max, 12 * 16_384 * 512 * 2 * 4);
        assert_eq!(mib(a.kv_host_max), 768);
        assert_eq!(mib(b.kv_host_max), 1536);
        // staging = (2560 + 248320) * 4 B
        assert_eq!(a.staging_peak, (2560 + 248_320) * 4);
    }

    #[test]
    fn q4_model_reports_one_arena_chunk() {
        let totals = WeightTotals {
            dev_bytes: 100,
            dev_tensors: 1,
            ..Default::default()
        };
        let b = flash_budget(totals, 0, 0, 4096, true);
        assert_eq!(b.arena_dev, SCRATCH_ARENA_CHUNK);
        assert_eq!(b.kv_host_max, 0, "full 層が無ければ KV も無い");
    }

    #[test]
    fn n_full_layers_rounds_up() {
        assert_eq!(n_full_layers(48, 4), 12);
        assert_eq!(n_full_layers(48, 5), 10); // 48/5 = 9.6 -> 10
        assert_eq!(n_full_layers(7, 4), 2);
        assert_eq!(n_full_layers(48, 0), 0, "0 除算にしない");
    }

    #[test]
    fn budget_line_has_all_terms() {
        let b = FlashBudget {
            max_ctx: 32_768,
            weights_dev: 5_736 * (1 << 20),
            weights_dev_tensors: 497,
            staging_peak: 1024,
            arena_dev: 0,
            kv_host_max: 1536 * (1 << 20),
            host_weights: 2 * (1 << 20),
            experts_ram: 20_000 * (1 << 20),
        };
        let line = budget_line(&b);
        assert!(line.starts_with("[flash] vram budget: max_ctx=32768"), "{line}");
        assert!(line.contains("weights=5736 MiB (497 本)"), "{line}");
        assert!(line.contains("arena=0 MiB"), "{line}");
        assert!(line.contains("kv=1536 MiB (host, max_ctx 比例)"), "{line}");
        assert!(line.contains("experts=20000 MiB (mmap)"), "{line}");
    }

    #[test]
    fn register_log_is_first_then_every_n() {
        assert_eq!(register_log_for(0), None);
        assert_eq!(register_log_for(1), Some(true), "1 回目は ERROR");
        assert_eq!(register_log_for(2), None);
        assert_eq!(register_log_for(999), None);
        assert_eq!(register_log_for(1000), Some(false), "1,000 回ごとに 1 行");
        assert_eq!(register_log_for(1001), None);
        assert_eq!(register_log_for(5000), Some(false));
    }

    #[test]
    fn stats_are_global_and_resettable() {
        reset_weight_upload_stats();
        assert_eq!(weight_upload_stats().attempted(), 0);
        note_weight_upload();
        note_weight_upload();
        // `None` は「出さない」だけ。数えるのは別。
        assert_eq!(note_weight_failure(7, "CUDA cuMemAlloc が失敗"), Some(true));
        assert_eq!(note_weight_failure(8, "2 回目"), None);
        note_gpu_broken("q4 weight upload failed");
        let s = weight_upload_stats();
        assert_eq!(s.uploaded, 2);
        assert_eq!(s.failed, 2);
        assert_eq!(s.last_failed_uid, Some(8));
        assert_eq!(s.first_error.as_deref(), Some("CUDA cuMemAlloc が失敗"));
        assert!(s.gpu_broken);
        assert_eq!(s.broken_cause.as_deref(), Some("q4 weight upload failed"));
        // GPU_BROKEN は reset で戻さない (プロセス寿命のラッチ)。
        reset_weight_upload_stats();
        let s = weight_upload_stats();
        assert_eq!(s.attempted(), 0);
        assert!(s.gpu_broken, "reset でラッチを落とさない");
        assert_eq!(s.first_error, None);
    }

    #[test]
    fn summary_line_reads_as_counts() {
        let s = WeightUploadStats {
            uploaded: 497,
            failed: 2977,
            uploaded_bytes: 4914 << 20,
            last_failed_uid: Some(6328),
            first_error: Some("CUDA cuMemAlloc が失敗: error 2".into()),
            gpu_broken: false,
            broken_cause: None,
            resident: 498,
            resident_bytes: 4914 << 20,
            cap_bytes: 12_288 << 20,
            evicted: 3,
            evicted_bytes: 96 << 20,
        };
        let line = summary_line(&s);
        assert!(line.contains("497 本 ok / 2977 本 失敗"), "{line}");
        assert!(line.contains("全 3474 本のうち 85.7%"), "{line}");
        assert!(line.contains("last_uid=6328"), "{line}");
        assert!(line.contains("最初の失敗: CUDA cuMemAlloc"), "{line}");
        assert!(!line.contains("GPU_BROKEN"), "止めていないときは出さない");
        // P46: 常駐 / 上限 / 転送計 / 退避 が同じ 1 行から読める。
        assert!(line.contains("gguf 常駐 498 本 4914 MiB"), "{line}");
        assert!(line.contains("上限 12288 MiB"), "{line}");
        assert!(line.contains("転送計 4914 MiB"), "{line}");
        assert!(line.contains("退避 計 3 本 96 MiB"), "{line}");

        let clean = WeightUploadStats {
            uploaded: 497,
            ..Default::default()
        };
        let line = summary_line(&clean);
        assert!(line.contains("497 本 ok / 0 本 失敗"), "{line}");
        assert!(line.contains("0.0%"), "{line}");

        let empty = summary_line(&WeightUploadStats::default());
        assert!(empty.contains("登録を試みていない"), "{empty}");
    }

    // ---- P46: 重みレジストリの LRU ----------------------------------------

    const MB: u64 = 1 << 20;

    #[test]
    fn weight_cap_env_parsing() {
        assert_eq!(parse_weight_cap_mb(None), 0, "未設定 = 無制限");
        assert_eq!(parse_weight_cap_mb(Some("")), 0);
        assert_eq!(parse_weight_cap_mb(Some("0")), 0);
        assert_eq!(parse_weight_cap_mb(Some("12288")), 12_288);
        assert_eq!(parse_weight_cap_mb(Some(" 4096 ")), 4096, "前後の空白は許す");
        assert_eq!(parse_weight_cap_mb(Some("8G")), 0, "読めない値は無制限に倒す");
        assert_eq!(parse_weight_cap_mb(Some("-1")), 0, "負も無制限");
    }

    #[test]
    fn lru_evicts_the_oldest_over_the_cap() {
        // 上限 100 MB に 40 MB を 3 本 → 3 本目で最古 (uid=1) が落ちる。
        let mut lru = WeightLru::new(100 * MB);
        assert!(lru.note_insert(1, 40 * MB).is_empty(), "80 MB までは収まる");
        assert!(lru.note_insert(2, 40 * MB).is_empty());
        assert_eq!(lru.used_bytes(), 80 * MB);
        assert_eq!(lru.note_insert(3, 40 * MB), vec![1], "最古が落ちる");
        assert_eq!(lru.used_bytes(), 80 * MB);
        assert_eq!(lru.len(), 2);
        assert!(!lru.contains(1));
        assert!(lru.contains(2) && lru.contains(3));
        assert_eq!(lru.evicted_tensors(), 1);
        assert_eq!(lru.evicted_bytes(), 40 * MB);
    }

    #[test]
    fn lru_skips_the_one_in_use() {
        let mut lru = WeightLru::new(100 * MB);
        lru.note_insert(1, 40 * MB);
        lru.note_insert(2, 40 * MB);
        assert!(lru.pin(1), "使用中にする");
        // 最古は uid=1 だが使用中なので飛ばし、次に古い uid=2 を落とす。
        assert_eq!(lru.note_insert(3, 40 * MB), vec![2]);
        assert!(lru.contains(1) && lru.contains(3));
        assert_eq!(lru.used_bytes(), 80 * MB);
        lru.unpin(1);
        assert_eq!(lru.pins(1), 0);
        assert_eq!(lru.oldest_unpinned(3), Some(1), "使用をやめた 1 が次の候補");
    }

    #[test]
    fn lru_does_not_evict_the_tensor_it_just_inserted() {
        // 1 本だけで上限を超える登録は自分を落とさない (登録直後に使うため)。
        let mut lru = WeightLru::new(100 * MB);
        assert!(lru.note_insert(7, 200 * MB).is_empty());
        assert!(lru.contains(7));
        assert_eq!(lru.used_bytes(), 200 * MB, "上限は一時的に超過してよい");
        assert_eq!(lru.evicted_tensors(), 0);
    }

    #[test]
    fn lru_zero_cap_never_evicts() {
        // 既定 (DENNO_GPU_WEIGHT_MB 未設定 = 0) は従来と同じ = 1 本も落とさない。
        let mut lru = WeightLru::new(0);
        for uid in 0..1_000u64 {
            assert!(lru.note_insert(uid, 1 << 30).is_empty());
        }
        assert_eq!(lru.len(), 1_000);
        assert_eq!(lru.evicted_tensors(), 0);
        assert_eq!(lru.evicted_bytes(), 0);
    }

    #[test]
    fn touch_makes_an_entry_youngest() {
        let mut lru = WeightLru::new(100);
        lru.note_insert(1, 40);
        lru.note_insert(2, 40);
        lru.touch(1);
        assert_eq!(lru.note_insert(3, 40), vec![2], "触った 1 は残る");
    }

    /// 退避 → ホストからの再登録で、デバイス上のバイト列が 1 bit も変わらない。
    ///
    /// `weights_for_gguf` は退避された uid を次に使うとき、呼び出し側が渡す
    /// `blocks` (ホストの生ブロック) からもう一度 `DevBuffer::from_slice` する。
    /// 転送はコピーなので、同じ `blocks` からは同じバイト列が載る = GEMV の
    /// 入力が同一 = 出力がビット一致する。ここではレジストリの代わりに
    /// `Vec<u8>` を置いて、その不変だけを確かめる (GPU 無しで回る)。
    #[test]
    fn reregister_after_eviction_reproduces_the_same_bytes() {
        fn checksum(b: &[u8]) -> u64 {
            b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
                (h ^ x as u64).wrapping_mul(0x0100_0000_01b3)
            })
        }
        let host: Vec<Vec<u8>> = (0..3u64)
            .map(|i| (0..64u64).map(|j| ((i * 37 + j * 11 + 5) % 251) as u8).collect())
            .collect();
        // 1 本 64 B を 2 本まで (128 B)。3 本目で最古が落ちる。
        let mut lru = WeightLru::new(128);
        let mut dev: HashMap<u64, Vec<u8>> = HashMap::new();
        for uid in 0..3u64 {
            for v in lru.note_insert(uid, 64) {
                dev.remove(&v);
            }
            dev.insert(uid, host[uid as usize].clone()); // 転送 = コピー
            assert!(lru.pin(uid));
            lru.unpin(uid);
        }
        assert_eq!(dev.len(), 2, "上限 128 B に 2 本");
        assert!(!dev.contains_key(&0), "最古の uid=0 が落ちている");
        let kept = checksum(&dev[&2]);
        // 退避された uid=0 をホストの生ブロックから再登録する。
        for v in lru.note_insert(0, 64) {
            assert_eq!(v, 1u64, "次に古い uid=1 が落ちる");
            dev.remove(&v);
        }
        dev.insert(0, host[0].clone());
        assert_eq!(dev[&0], host[0], "再登録のバイト列 = ホストの生ブロック");
        assert_eq!(checksum(&dev[&0]), checksum(&host[0]), "出力はビット一致");
        assert_eq!(checksum(&dev[&2]), kept, "残った本は 1 bit も変わらない");
        assert!(!dev.contains_key(&1));
    }

    #[test]
    fn plan_evicts_oldest_to_fit_the_cap_only() {
        // 上限 100 MB / 空き 8 GiB (十分) → 上限の分だけ落とす。
        let mut lru = WeightLru::new(100 * MB);
        lru.note_insert(1, 40 * MB);
        lru.note_insert(2, 40 * MB);
        let v = plan_weight_evictions(&mut lru, 3, 40 * MB, 100 * MB, Some(8 * 1024));
        assert_eq!(v, vec![1], "上限超過の分だけ最古が落ちる");
        assert_eq!(lru.evicted_tensors(), 1);
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn plan_reserves_free_vram_before_allocating() {
        // 上限には余裕があるが、空きが 300 MiB しかない。200 MiB を確保して
        // 256 MiB を残すには空き 457 MiB が要るので、あと 157 MiB 落とす。
        let mut lru = WeightLru::new(8 * 1024 * MB);
        for uid in 1..=3u64 {
            lru.note_insert(uid, 100 * MB);
        }
        let v = plan_weight_evictions(&mut lru, 9, 200 * MB, 8 * 1024 * MB, Some(300));
        assert_eq!(v, vec![1, 2], "古い順に 2 本落として空きを 500 MiB にする");
        assert_eq!(lru.evicted_tensors(), 2);
        assert!(lru.contains(3) && lru.contains(9));
    }

    #[test]
    fn plan_skips_the_one_in_use_even_for_free_vram() {
        let mut lru = WeightLru::new(8 * 1024 * MB);
        lru.note_insert(1, 100 * MB);
        lru.note_insert(2, 100 * MB);
        lru.pin(1);
        let v = plan_weight_evictions(&mut lru, 9, 200 * MB, 8 * 1024 * MB, Some(300));
        assert_eq!(v, vec![2], "使用中の 1 は空き VRAM のためにも落とさない");
        assert!(lru.contains(1));
    }

    #[test]
    fn plan_does_nothing_when_cap_is_zero() {
        // 既定 (上限 0) は空きがいくら少なくても 1 本も落とさない = 挙動不変。
        let mut lru = WeightLru::new(0);
        lru.note_insert(1, 100 * MB);
        lru.note_insert(2, 100 * MB);
        let v = plan_weight_evictions(&mut lru, 3, 200 * MB, 0, Some(1));
        assert!(v.is_empty());
        assert_eq!(lru.len(), 3);
        assert_eq!(lru.evicted_tensors(), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn arena_chunk_matches_cuda() {
        // cuda.rs の ARENA_CHUNK と drift しないこと (推定と実装の式を縛る)。
        assert_eq!(SCRATCH_ARENA_CHUNK, crate::cuda::ARENA_CHUNK as u64);
    }
}
