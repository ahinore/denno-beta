//! device op のプロファイラ (op 別の時間・転送量・同期待ち)。
//!
//! decode は 1 トークンで 1 層あたり 10 本前後の GEMV を host から 1 本ずつ
//! 叩く。`linear_gguf::GgufLinear::gemv` → `cuda::maybe_gpu_gguf_gemv_into` →
//! `CudaCtx::moe_gemv_dev` の各呼び出しが「活性の H2D・カーネル・出力の D2H・
//! 同期」を 1 組やるので、計算そのもの (数十 µs) 以外の固定費が積み上がる。
//! `q4prof` は層の区切りでしか測らないため、この内訳は見えない。ここでは
//! **op 名 (役割)** ごとに 1 呼び出し単位で積む。
//!
//! 有効化は `DENNO_DEV_PROFILE=1` だけ (env はプロセスで 1 回読む)。
//! off のとき呼び出し側が払うのは [`on`] の atomic load 1 回で、`Instant` も
//! ロックも取らない。数値経路には触れない (積むのは診断カウンタだけで、
//! GEMV の引数・カーネル・丸め順は変えない)。

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[cfg(test)]
use std::sync::atomic::{AtomicU8, Ordering};

/// 同時に積める op 名の上限。名前は線形探索するので小さく保つ
/// (超えた分は捨てる。プロファイラが本番の邪魔をしない側に倒す)。
pub const MAX_SLOTS: usize = 24;

/// 1 op ぶんの集計行。[`take`] が返す。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Row {
    /// op 名 (役割。`qkv` / `gate_inp` など)
    pub name: &'static str,
    /// 呼び出し回数
    pub calls: u64,
    /// 呼び出しの所要時間の合計 (ns)
    pub ns: u64,
    /// H2D バイト数 (入力 + 初回の重み)
    pub up: u64,
    /// D2H バイト数 (出力)
    pub down: u64,
    /// カーネル完了待ち + 結果読み出しにかかった時間の合計 (ns)
    pub sync_ns: u64,
}

struct Slot {
    name: &'static str,
    calls: u64,
    ns: u64,
    up: u64,
    down: u64,
    sync_ns: u64,
}

impl Slot {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            calls: 0,
            ns: 0,
            up: 0,
            down: 0,
            sync_ns: 0,
        }
    }
}

/// 積み先。op 数は 20 未満なので線形探索で足りる。
static SLOTS: Mutex<Vec<Slot>> = Mutex::new(Vec::new());

/// `DENNO_DEV_PROFILE=1` か。env はプロセスで 1 回だけ読む。
pub fn on() -> bool {
    #[cfg(test)]
    {
        // テストは env を書き換えられない (OnceLock + 並列実行) ので、
        // プロセス内フラグで on/off を切り替える。
        match FORCE_ON.load(Ordering::Relaxed) {
            FORCE_OFF => return false,
            FORCE_ON_VAL => return true,
            _ => {}
        }
    }
    env_on()
}

fn env_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DENNO_DEV_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

#[cfg(test)]
const FORCE_OFF: u8 = 1;
#[cfg(test)]
const FORCE_ON_VAL: u8 = 2;
/// テスト用の強制フラグ (0 = env に従う)。
#[cfg(test)]
static FORCE_ON: AtomicU8 = AtomicU8::new(0);

/// テストが env を立てずに on/off を切り替えるための入口
/// (`None` で env の値に戻す)。
#[cfg(test)]
pub fn set_on_for_test(v: Option<bool>) {
    FORCE_ON.store(
        match v {
            None => 0,
            Some(false) => FORCE_OFF,
            Some(true) => FORCE_ON_VAL,
        },
        Ordering::Relaxed,
    );
}

/// 1 op ぶんを積む。off ならそのまま戻る (ロックを取らない)。
pub fn add(slot: &'static str, ns: u64, up: u64, down: u64, sync_ns: u64) {
    if !on() {
        return;
    }
    let Ok(mut slots) = SLOTS.lock() else {
        return;
    };
    let idx = match slots.iter().position(|s| s.name == slot) {
        Some(i) => i,
        None => {
            if slots.len() >= MAX_SLOTS {
                return;
            }
            slots.push(Slot::new(slot));
            slots.len() - 1
        }
    };
    let s = &mut slots[idx];
    s.calls += 1;
    s.ns += ns;
    s.up += up;
    s.down += down;
    s.sync_ns += sync_ns;
}

/// 積んだ行を取り出して 0 に戻す (報告の区切りで呼ぶ)。
pub fn take() -> Vec<Row> {
    let Ok(mut slots) = SLOTS.lock() else {
        return Vec::new();
    };
    slots
        .drain(..)
        .map(|s| Row {
            name: s.name,
            calls: s.calls,
            ns: s.ns,
            up: s.up,
            down: s.down,
            sync_ns: s.sync_ns,
        })
        .collect()
}

/// 積んだ行を捨てる。
pub fn reset() {
    if let Ok(mut slots) = SLOTS.lock() {
        slots.clear();
    }
}

/// 報告用の行を作る (`tok` トークンあたりに割る)。off・空なら空 vec。
///
/// 1 行目が `[devprof] <tag> tok=N`、2 行目が列見出し、以降 1 op 1 行。
/// 並びは `ns` の降順 (重い op が上)。
pub fn report(tag: &str, tok: usize) -> Vec<String> {
    if !on() {
        return Vec::new();
    }
    let mut rows = take();
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by(|a, b| b.ns.cmp(&a.ns).then_with(|| a.name.cmp(b.name)));
    let d = tok.max(1) as f64;
    let mut out = Vec::with_capacity(rows.len() + 2);
    out.push(format!("[devprof] {tag} tok={tok}"));
    out.push(format!(
        "{:<16}{:>10}{:>9}{:>8}{:>10}{:>10}",
        "name", "calls/tok", "ms/tok", "MB_up", "MB_down", "sync_ms"
    ));
    for r in rows {
        out.push(format!(
            "{:<16}{:>10.1}{:>9.1}{:>8.1}{:>10.1}{:>10.1}",
            r.name,
            r.calls as f64 / d,
            r.ns as f64 / d / 1e6,
            r.up as f64 / d / 1e6,
            r.down as f64 / d / 1e6,
            r.sync_ns as f64 / d / 1e6
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// 呼び出し 1 回ぶんの一時記録 (H2D / D2H / 同期)
// ---------------------------------------------------------------------------
//
// バイト数と同期時間は cuda 側でしか分からない (重みの初回アップロードは
// `gpu_budget::weight_upload_stats()` の差分、同期は `CudaCtx::sync` +
// 結果読み出しの実測)。そこで cuda 側は thread_local に足すだけにし、
// 呼び出しの外側 (`GgufLinear::gemv_label` / `maybe_gpu_gguf_gemm_into`) が
// 時間を測って [`add`] に渡す。同じスレッドで完結する。

#[derive(Clone, Copy, Default)]
struct Pending {
    up: u64,
    down: u64,
    sync: u64,
}

thread_local! {
    static PENDING: Cell<Pending> = const { Cell::new(Pending { up: 0, down: 0, sync: 0 }) };
}

/// 直前の呼び出しの記録を捨てる (呼び出しの直前に呼ぶ)。
pub fn clear() {
    PENDING.with(|c| c.set(Pending::default()));
}

/// H2D / D2H を足す。
pub fn note_bytes(up: u64, down: u64) {
    PENDING.with(|c| {
        let mut p = c.get();
        p.up += up;
        p.down += down;
        c.set(p);
    });
}

/// カーネル完了待ち + 結果読み出しの時間を足す。
pub fn note_sync(ns: u64) {
    PENDING.with(|c| {
        let mut p = c.get();
        p.sync += ns;
        c.set(p);
    });
}

/// (up, down, sync_ns) を取り出して 0 に戻す。
pub fn take_pending() -> (u64, u64, u64) {
    PENDING.with(|c| {
        let p = c.replace(Pending::default());
        (p.up, p.down, p.sync)
    })
}

/// 同期の計測を始める (`off` なら `None` = `Instant` を取らない)。
pub fn sync_start() -> Option<Instant> {
    if on() {
        Some(Instant::now())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquant::GgmlType;
    use crate::model::linear_gguf::GgufLinear;

    /// このモジュールのテストはプロセス全体のフラグを触るので直列化する。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// k 列 n 行の Q8_0 ブロック (スケール 0.125・値はすべて 0)。
    /// 値は 0 なので出力も 0。ここで見たいのは時間とカウンタだけ。
    fn q8_blocks(k: usize, n: usize) -> Vec<u8> {
        let nblk = k / 32;
        let mut v = Vec::with_capacity(n * nblk * 34);
        for _ in 0..n * nblk {
            v.extend_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
            v.extend_from_slice(&[0u8; 32]);
        }
        v
    }

    #[test]
    fn add_accumulates_and_take_resets() {
        let _g = lock();
        reset();
        set_on_for_test(Some(true));
        add("t_gemv", 100, 10, 20, 30);
        add("t_gemv", 50, 1, 2, 3);
        add("t_other", 7, 0, 0, 0);
        let rows = take();
        let a = rows.iter().find(|r| r.name == "t_gemv").expect("t_gemv");
        assert_eq!(
            (a.calls, a.ns, a.up, a.down, a.sync_ns),
            (2, 150, 11, 22, 33)
        );
        assert!(rows.iter().any(|r| r.name == "t_other"));
        // take はリセットも行う
        assert!(take().is_empty());
        set_on_for_test(None);
    }

    #[test]
    fn off_records_nothing() {
        let _g = lock();
        reset();
        set_on_for_test(Some(false));
        assert!(!on());
        add("t_off", 1, 1, 1, 1);
        assert!(take().is_empty());
        set_on_for_test(None);
    }

    /// CPU 経路 (GPU 無し) でも呼び出し回数と時間は積まれ、転送量は 0 になる。
    #[test]
    fn cpu_path_records_calls_without_transfers() {
        let _g = lock();
        reset();
        set_on_for_test(Some(true));
        let blocks = q8_blocks(32, 4);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, 32, 4, &blocks).expect("GgufLinear");
        let x = vec![0.5f32; 32];
        let mut out = vec![0.0f32; 4];
        for _ in 0..3 {
            lin.gemv_label("t_cpu_qkv", &x, &mut out).expect("gemv");
        }
        let rows = take();
        let r = rows
            .iter()
            .find(|r| r.name == "t_cpu_qkv")
            .expect("t_cpu_qkv が積まれていない");
        assert_eq!(r.calls, 3, "CPU 経路でも呼び出しは数える");
        assert_eq!((r.up, r.down, r.sync_ns), (0, 0, 0), "GPU 無しなら転送は 0");
        assert!(r.ns > 0, "CPU 時間も積む");
        set_on_for_test(None);
    }

    /// 既定の gemv (ラベル無し) は "gemv" のスロットに入る。
    #[test]
    fn plain_gemv_uses_default_label() {
        let _g = lock();
        reset();
        set_on_for_test(Some(true));
        let blocks = q8_blocks(32, 4);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, 32, 4, &blocks).expect("GgufLinear");
        let x = vec![0.25f32; 32];
        let mut out = vec![0.0f32; 4];
        lin.gemv(&x, &mut out).expect("gemv");
        let rows = take();
        assert!(rows.iter().any(|r| r.name == "gemv" && r.calls >= 1));
        set_on_for_test(None);
    }

    /// 報告の書式 (列見出し + 1 op 1 行)。
    ///
    /// 実形に近い大きさ (k=2560) の Q8_0 行列を CPU の GEMV 経路
    /// (`GgufLinear::gemv_label`) で 1 回走らせ、その行を出す。
    /// `DENNO_DEV_PROFILE=1` で走らせたときの出力を REPORT_task63.md に貼る。
    #[test]
    fn report_prints_header_and_one_line_per_op() {
        let _g = lock();
        // env で有効になっていればそれを優先する (env 経路の確認を兼ねる)。
        let from_env = on();
        if !from_env {
            set_on_for_test(Some(true));
        }
        reset();
        let k = 2560usize;
        // ルータ相当 (2560 → 512) を 48 回 + qkv 相当 (2560 → 4096) を 1 回。
        let router = GgufLinear::from_slice(GgmlType::Q8_0, k, 512, &q8_blocks(k, 512))
            .expect("GgufLinear(router)");
        let qkv = GgufLinear::from_slice(GgmlType::Q8_0, k, 4096, &q8_blocks(k, 4096))
            .expect("GgufLinear(qkv)");
        let x = vec![0.01f32; k];
        let mut out = vec![0.0f32; 512];
        for _ in 0..48 {
            router
                .gemv_label("demo_gate_inp", &x, &mut out)
                .expect("gemv(router)");
        }
        let mut big = vec![0.0f32; 4096];
        qkv.gemv_label("demo_qkv", &x, &mut big)
            .expect("gemv(qkv)");
        let lines = report("unit", 1);
        println!(
            "[devprof] enabled by {}",
            if from_env {
                "DENNO_DEV_PROFILE=1"
            } else {
                "test flag (env 未設定)"
            }
        );
        for l in &lines {
            println!("{l}");
        }
        assert!(lines[0].starts_with("[devprof] unit tok=1"), "{lines:?}");
        assert!(lines[1].starts_with("name"), "{lines:?}");
        // 他のテストが並行して "gemv" を積むことがあるので、行数は下限だけ見る。
        assert!(lines.len() >= 4, "{lines:?}");
        let g = lines
            .iter()
            .find(|l| l.starts_with("demo_gate_inp"))
            .expect("gate_inp 行");
        assert!(g.contains("      48.0"), "{g}");
        // CPU 経路なので転送量は 0 (列は MB_up / MB_down / sync_ms)。
        assert!(g.ends_with("0.0       0.0       0.0"), "{g}");
        assert!(lines.iter().any(|l| l.starts_with("demo_qkv")), "{lines:?}");
        if !from_env {
            set_on_for_test(None);
        }
    }

    #[test]
    fn report_is_empty_when_off() {
        let _g = lock();
        reset();
        set_on_for_test(Some(false));
        add("t_hidden", 9, 9, 9, 9);
        assert!(report("unit", 1).is_empty());
        set_on_for_test(None);
    }
}
