//! task 214: decode の expert 上げ直し (ミス → VRAM) を**背景スレッド**へ移す。
//!
//! # なぜ要るか
//!
//! 短い文脈の decode 1 トークン (23.4 ms) のうち experts が 9.72 ms で最大の項目。
//! その内訳は「上げ直し (mmap 読み + H2D) 4.19 ms」「ミスの CPU 計算 2.09 ms」
//! (REPORT_P201.md の台帳、t201b の a2)。上げ直しは 1 本 3.07 MB を
//! **約 0.16 ms (19.4 GB/s = pageable の実測)** で送る同期の仕事で、
//! メインスレッドの上の直列に載っている。t201b は「上げる本数を減らす」を
//! 測って**負けた** (8 本 +67% / 0 本 +165%) ので、バイト数は要る。
//! ならば**送る場所**をメインスレッドから外すしかない。
//!
//! # 何をするか
//!
//! `DENNO_DEC_UP_WORKER=1` のとき、decode のミス expert の上げ直しを
//! 専用スレッドの待ち行列へ渡す。ワーカーは CUDA の primary context を
//! 自分のスレッドへ張り ([`crate::cuda::make_current_on_this_thread`])、
//! `expert_vram::acquire_soft` で 1 本ずつ載せる (LRU・予算・追い出し・
//! slab は従来と**同じ関数**。値も同じバイト列・同じカーネル)。
//!
//! 変わるのは**タイミングだけ**:
//!
//! - 従来: 層の先頭で同期 H2D → その expert は**同じ層**の GPU バッチで回る。
//! - ワーカー: 層の先頭で待ち行列へ入れる → メインスレッドは待たない。
//!   その expert は**同じ層では CPU** に落ち、**次のトークン**で GPU ヒットになる。
//!
//! したがって代償は「1 トークンぶん遅れる」= その expert 1 本の CPU 計算
//! (実測 約 75 us/本) で、利得は H2D の 0.16 ms/本がメインスレッドから消えること。
//! 短い文脈 (26 本/token を上げる) では -4.1 ms + 26x0.075 ms = 約 -2.2 ms/token。
//!
//! # 落とし穴 (実装で踏んだ順)
//!
//! 1. **CUDA の current context はスレッドローカル**。ワーカーの入口で
//!    `make_current_on_this_thread()` を呼ばないと最初の `cuMemAlloc` が
//!    error 201 で落ち、`acquire` の系譜が `ALLOC_DEAD` を立てて以後ずっと
//!    CPU になる (task 146 で踏んだのと同じ穴)。ここでは `acquire_soft`
//!    (ラッチしない版) を使う。
//! 2. **mmap の寿命**。`&[u8]` をスレッドへ渡せないので、`Arc<Mmap>` の
//!    ハンドルとオフセットを持つ ([`Part`])。`WeightBlocks::mapping()` が
//!    ヒープ実体のときは `None` を返し、呼び出し側は従来の同期経路へ落ちる。
//! 3. **LRU のロック**。`acquire_soft` は転送の間ずっとロックを持つ。
//!    メインスレッドの参照が 1 層 10 回のロック取得だと待たされるので、
//!    呼び出し側は `lookup_many` (1 層 1 回) に寄せる。
//! 4. **二重 DMA**。同じ鍵が飛行中なら積まない ([`INFLIGHT`])。
//!    待ち行列が溢れたら**落とす** (積まない)。落としても従来経路と同じ結果。
//!
//! # 既定
//!
//! **既定 on** (lab t214b で短い点 -4.4〜-10.3%、中央値 24 -> 21 ms/tok、prefill 不変)。
//! `DENNO_DEC_UP_WORKER=0` / `off` / `false` で切れる。
//! 上限は `DENNO_DEC_UP_WORKER_MAX` (1 トークンあたりに積む本数)。
//! 未設定なら従来の規則 (`DENNO_EXPERT_IQ_UPLOAD` 48 / 埋まったら 16) をそのまま使う。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::Mmap;

/// `DENNO_DEC_UP_WORKER`: decode の expert 上げ直しを背景スレッドへ移す。
pub const ENV: &str = "DENNO_DEC_UP_WORKER";

/// `DENNO_DEC_UP_WORKER_MAX`: 1 トークンあたりに待ち行列へ積む本数の上限。
/// 未設定なら従来の規則 (48 / 埋まったら 16) をそのまま使う。
pub const MAX_ENV: &str = "DENNO_DEC_UP_WORKER_MAX";

/// 待ち行列の上限 (本)。溢れた分は積まない (従来経路と同じ結果になる)。
///
/// 1 本 約 0.16 ms なので 384 本 = 約 61 ms。prefill の前に溜まっても
/// 次の turn の prefill (0.4 s) を 2% 以上は食わない大きさにしている。
pub const MAX_QUEUE: usize = 384;

/// [`ENV`] の解釈 (純関数)。**既定 on**。
///
/// task 214 は lab t214b で採否 yes (短 a1 -10.3% / b -9.5% / a2 -4.4%、
/// 3,965 -3.7%、中央値 24 -> 21 ms/tok、prefill 不変、md5 は深い点だけ DIFF)。
/// 明示的な偽値 `0` / `off` / `false` / `no` だけ切る。
pub fn on_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on_from_value(std::env::var(ENV).ok().as_deref()))
}

/// [`MAX_ENV`] の解釈 (純関数)。未設定・0・数値でない値は `None` (従来の規則)。
pub fn max_from_value(v: Option<&str>) -> Option<usize> {
    let n: usize = v?.trim().parse().ok()?;
    (n > 0).then_some(n)
}

/// `DENNO_DEC_UP_STREAM`: 上げ直しの H2D を専用ストリームで出す。**既定 on**。
///
/// null ストリーム (メインスレッドのカーネル) と直列にならないので、上げ直しが
/// GPU の帯を塞がない。ドライバのシンボルが無い / 生成に失敗したら同期経路へ
/// 落ちる (fail closed)。明示的な偽値だけ切る。
pub const STREAM_ENV: &str = "DENNO_DEC_UP_STREAM";

/// [`STREAM_ENV`] の解釈 (純関数)。既定 on。
pub fn stream_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// 専用ストリームを使うか (プロセスで 1 回だけ env を読む)。
pub fn stream_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| stream_from_value(std::env::var(STREAM_ENV).ok().as_deref()))
}

/// 1 トークンあたりに積む本数の上限。`None` は「従来の規則を使う」。
pub fn max_per_token() -> Option<usize> {
    static M: OnceLock<Option<usize>> = OnceLock::new();
    *M.get_or_init(|| {
        if !enabled() {
            return None;
        }
        max_from_value(std::env::var(MAX_ENV).ok().as_deref())
    })
}

/// expert 1 本の 1 行列ぶんのバイト範囲 (mmap のハンドル + 絶対オフセット + 長さ)。
#[derive(Clone)]
pub struct Part {
    pub map: Arc<Mmap>,
    pub off: usize,
    pub len: usize,
}

impl Part {
    /// 範囲が写像の内側ならスライスを返す (外なら `None` = 積まない)。
    pub fn get(&self) -> Option<&[u8]> {
        let end = self.off.checked_add(self.len)?;
        self.map.get(self.off..end)
    }
}

// ---------------------------------------------------------------------------
// 統計 (1 リクエストぶん)
// ---------------------------------------------------------------------------

static SUBMITTED: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static ERRORS: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static JOB_NS: AtomicU64 = AtomicU64::new(0);
static QUEUE_MAX: AtomicU64 = AtomicU64::new(0);
/// 専用ストリームが張れたか (診断用)。
static STREAM_ON: AtomicBool = AtomicBool::new(false);

/// 1 リクエストぶんの 1 行。ゲート off / 実績 0 なら `None`。
pub fn stats_line() -> Option<String> {
    if !enabled() {
        return None;
    }
    let sub = SUBMITTED.load(Ordering::Relaxed);
    if sub == 0 {
        return None;
    }
    let done = DONE.load(Ordering::Relaxed);
    let ns = JOB_NS.load(Ordering::Relaxed);
    let us = if done > 0 {
        ns as f64 / 1e3 / done as f64
    } else {
        0.0
    };
    let inflight = INFLIGHT
        .lock()
        .map(|s| s.len() as u64)
        .unwrap_or(0);
    Some(format!(
        "[dec-up-w] submitted={sub} done={done} fail={} err={} dropped={} inflight={inflight} \
         queue_max={} stream={} dma={:.2}ms us_per_job={us:.1}",
        FAILED.load(Ordering::Relaxed),
        ERRORS.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        QUEUE_MAX.load(Ordering::Relaxed),
        u8::from(STREAM_ON.load(Ordering::Relaxed)),
        ns as f64 / 1e6
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
///
/// ワーカーは非同期なので、戻した直後に前のリクエストの job が 1〜2 本
/// 完了することがある (次の行に混ざる)。数字は「ほぼ 1 リクエストぶん」として読む。
pub fn stats_reset() {
    for c in [
        &SUBMITTED,
        &DONE,
        &FAILED,
        &ERRORS,
        &DROPPED,
        &JOB_NS,
        &QUEUE_MAX,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// 待ち行列とワーカー (cuda のみ)
// ---------------------------------------------------------------------------

/// 飛行中 (積んだがまだ完了していない) の鍵。二重 DMA を防ぐ。
static INFLIGHT: std::sync::LazyLock<Mutex<std::collections::HashSet<(u16, u16)>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::model::expert_vram::ExpertShape;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::Instant;

    /// ワーカーへ渡す 1 本ぶん。
    pub struct Job {
        pub layer: u16,
        pub expert: u16,
        pub shapes: [ExpertShape; 3],
        pub parts: [Part; 3],
    }

    static TX: OnceLock<Option<Sender<Job>>> = OnceLock::new();
    static DEAD: AtomicBool = AtomicBool::new(false);

    fn sender() -> Option<&'static Sender<Job>> {
        TX.get_or_init(|| spawn()).as_ref()
    }

    fn spawn() -> Option<Sender<Job>> {
        let (tx, rx) = mpsc::channel::<Job>();
        let h = std::thread::Builder::new()
            .name("dec-up-worker".to_string())
            .spawn(move || worker_loop(rx));
        match h {
            Ok(_) => Some(tx),
            Err(e) => {
                tracing::warn!("task214: expert 上げ直しのスレッドを作れない: {e}");
                None
            }
        }
    }

    /// 1 本を載せる。`Ok(None)` は「この形では載せられない」(fail closed)。
    fn run_job(job: &Job) -> anyhow::Result<Option<()>> {
        let Some(g) = job.parts[0].get() else {
            return Ok(None);
        };
        let Some(u) = job.parts[1].get() else {
            return Ok(None);
        };
        let Some(d) = job.parts[2].get() else {
            return Ok(None);
        };
        let key = crate::model::expert_cache::ExpertKey {
            layer: job.layer,
            expert: job.expert,
        };
        let got =
            crate::model::expert_vram::acquire_deferred(key, job.shapes, || Ok([g, u, d]))?;
        Ok(got.map(|_| ()))
    }

    fn worker_loop(rx: Receiver<Job>) {
        // CUDA の current context はスレッドローカル。これをしないと最初の
        // 確保が error 201 で落ちる (task 146 と同じ穴)。
        if !crate::cuda::make_current_on_this_thread() {
            DEAD.store(true, Ordering::Relaxed);
            tracing::warn!("task214: ワーカーに CUDA context を張れない → 従来の同期経路のみ");
            return;
        }
        // 上げ直しの H2D は専用ストリームへ出す (null ストリームのカーネルと
        // 直列にしない)。失敗しても同期経路で動く。
        let stream = stream_enabled() && crate::cuda::enable_thread_upload_stream();
        STREAM_ON.store(stream, Ordering::Relaxed);
        while let Ok(job) = rx.recv() {
            let key = (job.layer, job.expert);
            let t = Instant::now();
            match run_job(&job) {
                Ok(Some(())) => {
                    DONE.fetch_add(1, Ordering::Relaxed);
                }
                Ok(None) => {
                    FAILED.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    let n = ERRORS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 3 {
                        tracing::warn!("task214: expert ({}, {}) を上げられない: {e:#}", key.0, key.1);
                    }
                    if n >= 16 {
                        // 何度も失敗するなら以後積まない (メインスレッドは従来経路)。
                        DEAD.store(true, Ordering::Relaxed);
                    }
                }
            }
            JOB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Ok(mut s) = INFLIGHT.lock() {
                s.remove(&key);
            }
        }
    }

    /// 1 本を待ち行列へ積む。積めたら true。
    ///
    /// 積めない (ゲート off / スレッドが死んだ / 同じ鍵が飛行中 / 溢れた) ときは
    /// false を返す。呼び出し側は従来の同期経路へ落ちる。
    pub fn submit(job: Job) -> bool {
        if DEAD.load(Ordering::Relaxed) {
            return false;
        }
        let Some(tx) = sender() else {
            return false;
        };
        let key = (job.layer, job.expert);
        {
            let Ok(mut s) = INFLIGHT.lock() else {
                return false;
            };
            if s.contains(&key) {
                return false;
            }
            if s.len() >= MAX_QUEUE {
                DROPPED.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            s.insert(key);
            let n = s.len() as u64;
            QUEUE_MAX.fetch_max(n, Ordering::Relaxed);
        }
        if tx.send(job).is_err() {
            if let Ok(mut s) = INFLIGHT.lock() {
                s.remove(&key);
            }
            return false;
        }
        SUBMITTED.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// 積んだがまだ完了していない本数 (診断用)。
    pub fn inflight() -> usize {
        INFLIGHT.lock().map(|s| s.len()).unwrap_or(0)
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{inflight, submit, Job};

#[cfg(test)]
mod tests {
    use super::*;

    /// task 214: `DENNO_DEC_UP_WORKER` は既定 on、明示的な偽値だけ切る
    /// (lab t214b で採否 yes)。
    #[test]
    fn gate_defaults_on_and_only_false_values_turn_it_off() {
        assert!(on_from_value(None));
        assert!(on_from_value(Some("")));
        assert!(on_from_value(Some("1")));
        assert!(on_from_value(Some(" on ")));
        assert!(on_from_value(Some("TRUE")));
        assert!(!on_from_value(Some("0")));
        assert!(!on_from_value(Some("off")));
        assert!(!on_from_value(Some("false")));
        assert!(!on_from_value(Some("no")));
    }

    /// task 214: `DENNO_DEC_UP_STREAM` は既定 on、明示的な偽値だけ切る。
    #[test]
    fn upload_stream_gate_defaults_on() {
        assert!(stream_from_value(None));
        assert!(stream_from_value(Some("1")));
        assert!(stream_from_value(Some(" on ")));
        assert!(!stream_from_value(Some("0")));
        assert!(!stream_from_value(Some("off")));
        assert!(!stream_from_value(Some("false")));
        assert!(!stream_from_value(Some("no")));
    }

    #[test]
    fn max_parses_positive_numbers_only() {
        assert_eq!(max_from_value(None), None);
        assert_eq!(max_from_value(Some("")), None);
        assert_eq!(max_from_value(Some("0")), None);
        assert_eq!(max_from_value(Some("abc")), None);
        assert_eq!(max_from_value(Some(" 128 ")), Some(128));
        assert_eq!(max_from_value(Some("1")), Some(1));
    }

    #[test]
    fn part_get_rejects_ranges_outside_the_map() {
        // tempfile は dev-dependency に無いので、自前で 1 本作って消す。
        let path = std::env::temp_dir().join(format!(
            "denno-task214-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, vec![7u8; 4096]).expect("書ける");
        let file = std::fs::File::open(&path).expect("開ける");
        // SAFETY: テストの間このファイルは書き換えない。
        let map = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("mmap") });
        let ok = Part {
            map: Arc::clone(&map),
            off: 1024,
            len: 64,
        };
        assert_eq!(ok.get().map(|s| s.len()), Some(64));
        let out = Part {
            map: Arc::clone(&map),
            off: 4096,
            len: 8,
        };
        assert!(out.get().is_none(), "写像の外は None");
        let over = Part {
            map,
            off: usize::MAX - 3,
            len: 16,
        };
        assert!(over.get().is_none(), "オフセットの桁上がりも None");
        drop(over);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn stats_line_is_silent_when_nothing_happened() {
        // ゲートの on/off に依らず、実績 0 なら行を出さない。
        stats_reset();
        assert!(stats_line().is_none());
        assert_eq!(SUBMITTED.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stats_roundtrip() {
        stats_reset();
        SUBMITTED.fetch_add(3, Ordering::Relaxed);
        DONE.fetch_add(2, Ordering::Relaxed);
        JOB_NS.fetch_add(400_000, Ordering::Relaxed);
        let line = stats_line();
        if enabled() {
            let line = line.expect("ゲート on なら行が出る");
            assert!(line.contains("submitted=3"), "{line}");
            assert!(line.contains("done=2"), "{line}");
            assert!(line.contains("us_per_job=200.0"), "{line}");
        } else {
            assert!(line.is_none());
        }
        stats_reset();
        assert_eq!(SUBMITTED.load(Ordering::Relaxed), 0);
        assert_eq!(JOB_NS.load(Ordering::Relaxed), 0);
    }

    /// task 214: 上限の既定は「従来の規則」(None)。ゲート off でも None。
    #[test]
    fn max_per_token_defaults_to_the_old_rule() {
        // `DENNO_DEC_UP_WORKER_MAX` 未設定なら None (= 従来の規則)。
        if std::env::var(MAX_ENV).is_err() {
            assert_eq!(max_per_token(), None);
        }
    }
}
