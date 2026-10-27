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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

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
    /// 1 回の GEMV が作る一時バッファのピーク (`moe_gemv_dev` の x + out)。
    /// 呼び出しごとに alloc/free されるので**積み上がらない**。
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
/// - `staging_peak = (max_k + max_n) × 4 B` (`moe_gemv_dev` が 1 呼び出しで作る 2 本)
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
static FAILED: AtomicU64 = AtomicU64::new(0);
static LAST_FAILED_UID: AtomicU64 = AtomicU64::new(u64::MAX);
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
    /// 最後に失敗した uid
    pub last_failed_uid: Option<u64>,
    /// 最初の失敗のメッセージ (原因はこれ 1 本で足りる)
    pub first_error: Option<String>,
    /// 重みの失敗で GPU 経路全体を止めたか (`cuda::GPU_BROKEN`)
    pub gpu_broken: bool,
    /// 止めた原因 (最初の 1 件)
    pub broken_cause: Option<String>,
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
        last_failed_uid: if last == u64::MAX { None } else { Some(last) },
        first_error: FIRST_ERROR.lock().ok().and_then(|g| g.clone()),
        gpu_broken: gpu_broken_now(),
        broken_cause: BROKEN_CAUSE.lock().ok().and_then(|g| g.clone()),
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
    FAILED.store(0, Ordering::Relaxed);
    LAST_FAILED_UID.store(u64::MAX, Ordering::Relaxed);
    if let Ok(mut g) = FIRST_ERROR.lock() {
        *g = None;
    }
}

/// リクエスト末尾に出す 1 行。「何枚中何枚が失敗したか」が読める形にする。
pub fn summary_line(s: &WeightUploadStats) -> String {
    if s.attempted() == 0 {
        return "[flash] gpu weights: 0 本 ok / 0 本 失敗 (登録を試みていない)".to_string();
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
    line
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
            last_failed_uid: Some(6328),
            first_error: Some("CUDA cuMemAlloc が失敗: error 2".into()),
            gpu_broken: false,
            broken_cause: None,
        };
        let line = summary_line(&s);
        assert!(line.contains("497 本 ok / 2977 本 失敗"), "{line}");
        assert!(line.contains("全 3474 本のうち 85.7%"), "{line}");
        assert!(line.contains("last_uid=6328"), "{line}");
        assert!(line.contains("最初の失敗: CUDA cuMemAlloc"), "{line}");
        assert!(!line.contains("GPU_BROKEN"), "止めていないときは出さない");

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

    #[cfg(feature = "cuda")]
    #[test]
    fn arena_chunk_matches_cuda() {
        // cuda.rs の ARENA_CHUNK と drift しないこと (推定と実装の式を縛る)。
        assert_eq!(SCRATCH_ARENA_CHUNK, crate::cuda::ARENA_CHUNK as u64);
    }
}
