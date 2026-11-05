//! task 148: prefill の選択 expert を GPU で回すための入口 (GPU 非依存の部分)。
//!
//! # なぜ要るか
//!
//! prefill の選択 expert は丸ごと CPU で、cold prefill (3,826 tok) の 53% を占める
//! (lab t147a の台帳 `[q4ledger-pf]`: `experts` 47.4 s / `wall` 89.3 s。launch 0 /
//! sync 0 / H2D 0 = GPU を 1 度も通っていない)。
//!
//! prefill は 512 トークンのチャンクで回るので、1 層 1 チャンクで見ると
//! 「256 expert × 平均 20 行」の GEMM になる。decode の m=1 カーネル
//! (`kernels_exp_iq.cu`) を **行方向へ広げた** カーネルなら、重みの読みを
//! 1 回に畳んだまま 20 行ぶんの積和を回せる。
//!
//! # カーネルの契約 (`kernels_exp_iq_pf.cu` と共有)
//!
//! - 1 warp = 2 出力行 (PF_OUTS)。32 レーンが 8 要素ずつ (IQ4_NL / Q2_0 は
//!   1 レーン = 1 ブロック) 持ち、**行方向は 16 行ずつ** (PF_MR) まとめて
//!   レジスタに積む。重みは 1 回デコードして 16 行に配る。
//! - block = 4 warp = 8 出力行、grid = (ceil(n_out / 8), segment 数, 1)。
//! - 1 行ぶんの累積順は decode の m=1 カーネルと同じ (b 昇順 → j 昇順 →
//!   `__shfl_down_sync` の木)。**同じ行を m=1 と m 行で回すとビット一致する**
//!   (FMA の縮約は nvcc の既定でどちらも同じ)。
//! - segment 表 (1 segment = 4 語):
//!
//!   seg[s*4 + 0] = m        (この expert の行数)
//!   seg[s*4 + 1] = row_off  (`rows` の先頭添字。出力も同じ添字 × n_out)
//!   seg[s*4 + 2] = 0        (予約)
//!   seg[s*4 + 3] = 0        (予約)
//!
//! - `rows[i]` は行 i の**トークン番号**。入力は `xbase + rows[i] * n_in`。
//!   down 行列は入力が `act` (行が詰まっている) なので `rows[i] = i` を渡す。
//!
//! # ゲート
//!
//! `DENNO_EXPERT_IQ_GPU_PF=1` かつ `DENNO_EXPERT_IQ_GPU=1` のときだけ。
//! 片方でも欠けたら [`pf_enabled`] が false を返し、呼び出し側は従来の CPU 経路
//! (`moe_experts_flat`) をそのまま回す (fail closed)。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::kquant::GgmlType;

/// `DENNO_EXPERT_IQ_GPU_PF=1`: prefill の選択 expert を GPU (m 行) で回す。
///
/// **`DENNO_EXPERT_IQ_GPU=1` が前提** (VRAM 常駐の LRU を使うため)。
pub const EXPERT_IQ_PF_ENV: &str = "DENNO_EXPERT_IQ_GPU_PF";

/// `DENNO_EXPERT_IQ_PF_MIN_HITS`: 1 層を GPU で回すのに要るヒット expert 数 (既定 4)。
///
/// 1 層 1 同期は prefill の実測で 30 us 前後 (t147a の `sync_floor`)。ヒットが
/// 数本しか無い層は CPU のままの方が安い。
pub const EXPERT_IQ_PF_MIN_HITS_ENV: &str = "DENNO_EXPERT_IQ_PF_MIN_HITS";

/// `DENNO_EXPERT_IQ_PF_UPLOAD_MB`: 1 チャンクで VRAM へ上げ直す上限 (MiB、既定 512)。
///
/// **LRU に空きがある間だけ**使う。埋まっているときの上げ直しは「1 本上げるために
/// 1 本追い出す」= 次のチャンクで同じだけミスが増えるだけで、H2D の時間が純損に
/// なる (ミス 1 本の CPU 計算 0.5 ms に対し H2D 0.27 ms)。
pub const EXPERT_IQ_PF_UPLOAD_MB_ENV: &str = "DENNO_EXPERT_IQ_PF_UPLOAD_MB";

/// `DENNO_EXPERT_IQ_PF_UPLOAD_ROWS`: 上げ直すミスの最小行数 (既定 8)。
///
/// 19 トークンの prefill では 1 expert あたり平均 0.74 行なので、この床で
/// **短い prefill では 1 本も上げない** (0.97 s の next turn に 512 MB を
/// 載せるような真似をしない)。
pub const EXPERT_IQ_PF_UPLOAD_ROWS_ENV: &str = "DENNO_EXPERT_IQ_PF_UPLOAD_ROWS";

/// [`EXPERT_IQ_PF_MIN_HITS_ENV`] の既定値。
pub const DEFAULT_MIN_HITS: usize = 4;

/// [`EXPERT_IQ_PF_UPLOAD_MB_ENV`] の既定値。
pub const DEFAULT_UPLOAD_MB: usize = 512;

/// [`EXPERT_IQ_PF_UPLOAD_ROWS_ENV`] の既定値。
pub const DEFAULT_UPLOAD_ROWS: usize = 8;

/// 1 回の起動で受ける行数 (`kernels_exp_iq_pf.cu` の `PF_MR`)。
pub const ROWS_PER_LAUNCH: usize = 16;

/// 1 warp が受ける出力行数 (`PF_OUTS`)。256 要素ブロックの型だけ 2 行。
pub const OUT_PER_WARP: usize = 2;

/// 1 block の warp 数 (`PF_WARPS`)。
pub const WARPS: usize = 4;

/// このスロットの 1 block あたりの出力行数。
///
/// 256 要素ブロックの型 (スロット 0..=3) は 1 warp = 2 行で 8 行 / block、
/// IQ4_NL / Q2_0 (4..=5) は 1 行で 4 行 / block (重み 32/64 要素をレジスタに
/// 持つので 2 行ぶんは入らない)。
pub fn tile_of(slot: usize) -> usize {
    if slot <= 3 {
        WARPS * OUT_PER_WARP
    } else {
        WARPS
    }
}

/// カーネルのスロット数 (型の数)。
pub const SLOTS: usize = 6;

/// 既定で有効か (タスクの採否で決める)。
///
/// **false のまま**: `DENNO_EXPERT_IQ_GPU_PF` 未設定では従来の CPU 経路。
/// 採用時はここを true にする (gate `DENNO_EXPERT_IQ_GPU=1` の内側で既定 on)。
pub const DEFAULT_ON: bool = false;

/// `DENNO_EXPERT_IQ_GPU_PF` の解釈 (純関数)。`0` / `off` / `false` だけが偽で、
/// 未設定は [`DEFAULT_ON`]。**`iq_on` が false なら常に false**。
pub fn pf_from_value(iq_on: bool, v: Option<&str>) -> bool {
    if !iq_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_ON,
        Some(s) => crate::model::expert_vram::flag_on(Some(s)) || !matches!(s, "0" | "off" | "false"),
    }
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn pf_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_from_value(
            crate::model::exp_iq::iq_gpu_enabled(),
            std::env::var(EXPERT_IQ_PF_ENV).ok().as_deref(),
        )
    })
}

fn env_usize(key: &str, dflt: usize) -> usize {
    match std::env::var(key).ok() {
        Some(s) if !s.trim().is_empty() => s.trim().parse().unwrap_or(dflt),
        _ => dflt,
    }
}

/// 1 層を GPU で回すのに要るヒット本数。
pub fn min_hits_per_layer() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_MIN_HITS_ENV, DEFAULT_MIN_HITS))
}

/// 1 チャンクで上げ直す上限 (MiB)。
pub fn upload_mb() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_UPLOAD_MB_ENV, DEFAULT_UPLOAD_MB))
}

/// 上げ直すミスの最小行数。
pub fn upload_min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_UPLOAD_ROWS_ENV, DEFAULT_UPLOAD_ROWS))
}

/// 型 → カーネルのスロット (decode の `kernels_exp_iq` と同じ順)。
pub fn slot(ty: GgmlType) -> Option<usize> {
    crate::model::exp_iq::slot(ty)
}

/// この型を m 行カーネルで読めるか。
pub fn supported(ty: GgmlType) -> bool {
    slot(ty).is_some()
}

/// 1 行あたりのバイト数。ブロックサイズで割り切れない k は `None`。
///
/// decode の [`crate::model::exp_iq::row_bytes`] と同じ式 (型の許可制は持たない)。
pub fn row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if bs == 0 || ts == 0 || k == 0 || !k.is_multiple_of(bs) {
        return None;
    }
    Some((k / bs) * ts)
}

/// カーネル名 (`kernels_exp_iq_pf.cu` の `.visible .entry`)。
pub fn kernel_name(slot: usize) -> Option<&'static str> {
    Some(match slot {
        0 => "dno_exp_pf_iq2s",
        1 => "dno_exp_pf_iq3xxs",
        2 => "dno_exp_pf_iq3s",
        3 => "dno_exp_pf_iq4xs",
        4 => "dno_exp_pf_iq4nl",
        5 => "dno_exp_pf_q2_0",
        _ => return None,
    })
}

/// silu カーネルの名前 (このモジュール専用。decode の silu とは別モジュール)。
pub const SILU_KERNEL: &str = "dno_exp_pf_silu_mul";

/// ヒット行の加重和カーネルの名前 (`od[t*e+i] = sum_j w[t*k+j] * h[row_j*e+i]`)。
///
/// これを GPU で回すことで、1 層あたりホストへ戻す量が「全行 × e」から
/// 「nt × e」になる (42 MB → 5.2 MB)。
pub const ACCUM_KERNEL: &str = "dno_exp_pf_accum";

/// 起動形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PfLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

/// 層一括 m 行 GEMV の起動形 (純関数)。`None` = この形では起動しない。
///
/// block = 128 (4 warp)。1 block が受ける出力行数は型で違う ([`tile_of`])。
/// `n_out` がタイルで割り切れない形は起動しない (端数の行は CPU で計算する。
/// 実形は 640 と 2,560 でどちらも割り切れる)。grid.y = segment 数 (上限 65535)。
pub fn launch_plan(slot: usize, nseg: usize, n_out: usize) -> Option<PfLaunch> {
    if nseg == 0 || n_out == 0 || nseg > 65535 {
        return None;
    }
    let tile = tile_of(slot);
    if tile == 0 || !n_out.is_multiple_of(tile) {
        return None;
    }
    let tiles = n_out / tile;
    if tiles > 65535 {
        return None;
    }
    Some(PfLaunch {
        grid: (tiles as u32, nseg as u32, 1),
        block: ((32 * WARPS) as u32, 1, 1),
    })
}

// ---------------------------------------------------------------------------
// 層 1 回ぶんの segment 表 (GPU 無しで検証できる部分)
// ---------------------------------------------------------------------------

/// 1 層 1 チャンクぶんの segment 表。
///
/// `seg` は 4 語 × ヒット expert 数、`rows` は行数ぶんのトークン番号。
/// 行は expert index 昇順、expert の中はトークン昇順 (`by_expert` の順)。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PfPlan {
    /// segment 表 (4 語 × nseg)
    pub seg: Vec<u32>,
    /// 行 → トークン番号
    pub rows: Vec<u32>,
    /// ヒットした expert の番号 (昇順)
    pub experts: Vec<usize>,
    /// 行の総数
    pub total_rows: usize,
}

impl PfPlan {
    /// segment 数。
    pub fn nseg(&self) -> usize {
        self.experts.len()
    }

    /// 1 行ぶんのトークン番号。
    pub fn token_of(&self, row: usize) -> u32 {
        self.rows[row]
    }
}

/// ヒットした expert だけの segment 表を作る (`pick[ei]` が true の expert)。
///
/// 行の並びは `by_expert` のまま (トークン昇順)。出力も同じ順に詰めるので、
/// `rows` の添字がそのまま出力行になる。
pub fn build_plan(by_expert: &[Vec<(usize, usize)>], pick: &[bool]) -> PfPlan {
    let mut plan = PfPlan::default();
    let mut off = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() || !pick.get(ei).copied().unwrap_or(false) {
            continue;
        }
        plan.seg.push(picks.len() as u32);
        plan.seg.push(off as u32);
        plan.seg.push(0);
        plan.seg.push(0);
        plan.experts.push(ei);
        for &(t, _) in picks {
            plan.rows.push(t as u32);
        }
        off += picks.len();
    }
    plan.total_rows = off;
    plan
}

/// down 行列用の行表 (`act` は行が詰まっているので 0,1,2,...)。
pub fn identity_rows(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

// ---------------------------------------------------------------------------
// 1 リクエストぶんの観測 (REPORT 用)
// ---------------------------------------------------------------------------

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static LAYERS: AtomicU64 = AtomicU64::new(0);
static EXPERTS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static D2H_NS: AtomicU64 = AtomicU64::new(0);
static ADD_NS: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static SKIP_LAYERS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんの参照を積む (`lookups` = 選択 expert 数、`hits` = 載っていた数)。
pub fn note_lookup(lookups: u32, hits: u32) {
    LOOKUPS.fetch_add(lookups as u64, Ordering::Relaxed);
    HITS.fetch_add(hits as u64, Ordering::Relaxed);
}

/// 1 層ぶんの GPU 実行を積む (`rows` = GPU に載った行数、`experts` = 本数)。
pub fn note_layer(rows: u32, experts: u32, gpu_ns: u64) {
    LAYERS.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    EXPERTS.fetch_add(experts as u64, Ordering::Relaxed);
    GPU_NS.fetch_add(gpu_ns, Ordering::Relaxed);
}

/// 1 層ぶんの発行〜同期 (カーネル本体) の時間を積む。
pub fn note_kernel_ns(ns: u64) {
    KERNEL_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 層ぶんの D2H (加重和の nt×e) の時間を積む。
pub fn note_d2h_ns(ns: u64) {
    D2H_NS.fetch_add(ns, Ordering::Relaxed);
}

/// ヒット寄与の足し込み (ホスト側) の時間を積む。
pub fn note_add_ns(ns: u64) {
    ADD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// ヒットが少なくて GPU を呼ばなかった層を積む。
pub fn note_skip_layer() {
    SKIP_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// 上げ直し 1 本と、そのホスト時間。
pub fn note_upload_ns(ns: u64) {
    UPLOADS.fetch_add(1, Ordering::Relaxed);
    UPLOAD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行。ゲート off / GPU 層 0 なら `None`。
pub fn stats_line() -> Option<String> {
    if !pf_enabled() {
        return None;
    }
    let layers = LAYERS.load(Ordering::Relaxed);
    if layers == 0 && SKIP_LAYERS.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let vram = match crate::model::expert_vram::budget_usage() {
        Some((used, pinned, cap)) => format!(
            "vram={}MiB/{}MiB pin={}MiB",
            used >> 20,
            cap >> 20,
            pinned >> 20
        ),
        None => "vram=-".to_string(),
    };
    let lk = LOOKUPS.load(Ordering::Relaxed);
    let hr = if lk > 0 {
        HITS.load(Ordering::Relaxed) as f64 / lk as f64
    } else {
        0.0
    };
    Some(format!(
        "[exp-pf] lookups={lk} hit_rate={hr:.3} gpu_layers={layers} skip_layers={} experts={} \
         rows={} gpu_ms={:.1} kern_ms={:.1} d2h_ms={:.1} add_ms={:.1} uploads={} up_ms={:.1} {vram}",
        SKIP_LAYERS.load(Ordering::Relaxed),
        EXPERTS.load(Ordering::Relaxed),
        ROWS.load(Ordering::Relaxed),
        GPU_NS.load(Ordering::Relaxed) as f64 / 1e6,
        KERNEL_NS.load(Ordering::Relaxed) as f64 / 1e6,
        D2H_NS.load(Ordering::Relaxed) as f64 / 1e6,
        ADD_NS.load(Ordering::Relaxed) as f64 / 1e6,
        UPLOADS.load(Ordering::Relaxed),
        UPLOAD_NS.load(Ordering::Relaxed) as f64 / 1e6,
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [
        &LOOKUPS,
        &HITS,
        &LAYERS,
        &EXPERTS,
        &ROWS,
        &GPU_NS,
        &KERNEL_NS,
        &D2H_NS,
        &ADD_NS,
        &UPLOADS,
        &UPLOAD_NS,
        &SKIP_LAYERS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// 1 チャンクぶんの上げ直し予算 (本数)。層 0 で戻す。
///
/// `DENNO_EXPERT_IQ_PF_UPLOAD_MB` を 1 本 2.05 MB で割った本数を上限にする。
pub fn upload_budget_experts() -> usize {
    let mb = upload_mb();
    if mb == 0 {
        return 0;
    }
    (mb << 20) / (2_048_000usize.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_expert_fixture() -> Vec<Vec<(usize, usize)>> {
        // expert 0: 3 行、1: 0 行、2: 5 行、3: 1 行、4: 2 行
        let mut v: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 5];
        v[0] = vec![(0, 0), (2, 1), (9, 2)];
        v[2] = vec![(0, 1), (1, 0), (3, 3), (4, 1), (9, 0)];
        v[3] = vec![(7, 0)];
        v[4] = vec![(5, 2), (6, 1)];
        v
    }

    /// 選んだ expert の行だけが 1 回ずつ入り、`seg` の row_off が詰まっていること。
    #[test]
    fn plan_keeps_expert_order_and_offsets() {
        let be = by_expert_fixture();
        let pick = [true, true, false, true, true];
        let plan = build_plan(&be, &pick);
        assert_eq!(plan.nseg(), 3, "空の expert 1 と未選択の expert 2 は入らない");
        assert_eq!(plan.experts, vec![0, 3, 4]);
        assert_eq!(plan.total_rows, 3 + 1 + 2);
        assert_eq!(plan.seg.len(), 3 * 4);
        assert_eq!(&plan.seg[0..4], &[3, 0, 0, 0]);
        assert_eq!(&plan.seg[4..8], &[1, 3, 0, 0]);
        assert_eq!(&plan.seg[8..12], &[2, 4, 0, 0]);
        assert_eq!(plan.rows, vec![0, 2, 9, 7, 5, 6]);
        // 各行はちょうど 1 回、(expert, token) の対応が保たれている
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for (s, &ei) in plan.experts.iter().enumerate() {
            let m = plan.seg[s * 4] as usize;
            let off = plan.seg[s * 4 + 1] as usize;
            for r in 0..m {
                let t = plan.rows[off + r] as usize;
                seen.push((ei, t));
            }
        }
        let mut want: Vec<(usize, usize)> = be
            .iter()
            .enumerate()
            .filter(|(i, _)| pick[*i])
            .flat_map(|(i, v)| v.iter().map(move |&(t, _)| (i, t)))
            .collect();
        seen.sort_unstable();
        want.sort_unstable();
        assert_eq!(seen, want, "行の取りこぼし・重複がある");
    }

    /// 全部ヒットなら行数は選択数 (nt x k) と一致する。
    #[test]
    fn plan_all_hits_covers_every_pair() {
        let mut be: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 4];
        for t in 0..32usize {
            for j in 0..3usize {
                be[(t * 3 + j) % 4].push((t, j));
            }
        }
        let pick = [true; 4];
        let plan = build_plan(&be, &pick);
        assert_eq!(plan.total_rows, 96);
        assert_eq!(plan.nseg(), 4);
        assert_eq!(identity_rows(4), vec![0, 1, 2, 3]);
    }

    /// 起動形: 256 ブロック型は 8 行 / block、レーンブロック型は 4 行 / block。
    #[test]
    fn launch_plan_shapes() {
        let p = launch_plan(2, 200, 640).expect("plan");
        assert_eq!(p.grid, (80, 200, 1));
        assert_eq!(p.block, (128, 1, 1));
        let q = launch_plan(2, 200, 2560).expect("plan");
        assert_eq!(q.grid, (320, 200, 1));
        // down は IQ4_NL / Q2_0 (スロット 4/5) = 4 行 / block
        let r = launch_plan(4, 200, 2560).expect("plan");
        assert_eq!(r.grid, (640, 200, 1));
        assert_eq!(tile_of(0), 8);
        assert_eq!(tile_of(5), 4);
        assert!(launch_plan(0, 0, 640).is_none());
        assert!(launch_plan(0, 1, 0).is_none());
        assert!(launch_plan(0, 65536, 640).is_none());
        // タイルで割り切れない形は起動しない (端数は CPU で計算する)
        assert!(launch_plan(0, 1, 641).is_none());
        assert!(launch_plan(4, 1, 642).is_none());
    }

    /// ゲートは `DENNO_EXPERT_IQ_GPU=1` の内側だけ。既定は [`DEFAULT_ON`]。
    #[test]
    fn gate_needs_iq_and_parses_values() {
        assert!(!pf_from_value(false, Some("1")), "IQ ゲート off なら動かない");
        assert!(!pf_from_value(false, None));
        assert_eq!(pf_from_value(true, None), DEFAULT_ON);
        assert_eq!(pf_from_value(true, Some("")), DEFAULT_ON);
        assert!(pf_from_value(true, Some("1")));
        assert!(pf_from_value(true, Some("on")));
        assert!(!pf_from_value(true, Some("0")));
        assert!(!pf_from_value(true, Some("off")));
        assert!(!pf_from_value(true, Some("false")));
    }

    /// 上げ直しの予算は MB から本数へ (1 本 2.05 MB)。
    #[test]
    fn upload_budget_is_bytes_over_expert_size() {
        assert_eq!(upload_budget_experts(), (upload_mb() << 20) / 2_048_000);
    }

    /// ホストが読む名前が .cu のカーネル名と一致する (実機でしか分からない
    /// 名前の食い違いをソースで突き合わせる)。
    #[test]
    fn kernel_names_exist_in_cu_source() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let cu = std::fs::read_to_string(dir.join("kernels_exp_iq_pf.cu")).expect(".cu が読めない");
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            // 256 要素ブロックの 4 型はマクロ `PF_KERNEL_256(名前, デコード)` で
            // 生成するので、どちらの書き方でも受ける。
            assert!(
                cu.contains(&format!("__global__ void {name}("))
                    || cu.contains(&format!("PF_KERNEL_256({name},")),
                "{name} が kernels_exp_iq_pf.cu に無い"
            );
        }
        assert!(
            cu.contains(&format!("__global__ void {SILU_KERNEL}(")),
            "{SILU_KERNEL} が kernels_exp_iq_pf.cu に無い"
        );
        assert!(
            cu.contains(&format!("__global__ void {ACCUM_KERNEL}(")),
            "{ACCUM_KERNEL} が kernels_exp_iq_pf.cu に無い"
        );
        // 起動形の定数が .cu と揃っていること (片方だけ変えると行が落ちる)。
        assert!(cu.contains("#define PF_MR 16"), "PF_MR が 16 でない");
        assert!(cu.contains("#define PF_OUTS 2"), "PF_OUTS が 2 でない");
        assert!(cu.contains("#define PF_WARPS 4"), "PF_WARPS が 4 でない");
        assert_eq!(ROWS_PER_LAUNCH, 16);
        // 生成規則 (build_ptx.sh の exp_iq_pf) と cuda.rs が読む名前が同じこと。
        let sh = std::fs::read_to_string(dir.join("build_ptx.sh")).expect("build_ptx.sh");
        assert!(
            sh.contains("exp_iq_pf)") && sh.contains("kernels_exp_iq_pf.cu|fmad"),
            "build_ptx.sh に exp_iq_pf の規則が無い"
        );
        assert!(
            sh.contains("targets_all=(") && sh.contains(" exp_iq_pf"),
            "build_ptx.sh の targets_all に exp_iq_pf が無い"
        );
    }

    /// コミット済みの PTX が .cu と同じ引数の数を持つ (ABI の取り違えは実機で
    /// しか分からないので、PTX の `.param` 行を数えて固定する)。
    #[test]
    fn committed_ptx_has_expected_params() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq_pf.ptx")) else {
            // PTX 未生成のツリーでは何も言えない (カーネル無し = CPU へ落ちるだけ)。
            return;
        };
        let count = |name: &str| -> Option<usize> {
            let head = format!(".visible .entry {name}(");
            let i = ptx.find(&head)?;
            let rest = &ptx[i + head.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            Some(rest[..end].matches(".param").count())
        };
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert_eq!(count(name), Some(9), "{name} の引数の数が .cu と違う");
        }
        assert_eq!(count(SILU_KERNEL), Some(4), "{SILU_KERNEL} の引数の数");
        assert_eq!(count(ACCUM_KERNEL), Some(7), "{ACCUM_KERNEL} の引数の数");
    }
}
