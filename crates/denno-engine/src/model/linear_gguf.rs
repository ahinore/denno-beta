//! P-MOE-E4: GGUF の量子化ブロックを**そのまま保持する**線形層。
//!
//! 既存の `Quant8Linear` / `Q4Linear` はロード時に int8 / Q4_0 へ量子化し直す
//! (27B の重みはその経路で作られている)。Flash-Next の重みは GGUF の中で既に
//! Q8_0 / Q4_K / Q5_K / Q5_1 / BF16 / F32 なので、**作り直さず生バイトのまま
//! 計算する**。量子化し直すと数値が変わり、アンカー比較の意味が無くなる。
//!
//! 数値契約:
//! - `y[r] = dot(row_r, x)` を r の昇順、内積は i の昇順に f32 で加算する
//!   (本体は `model::moe_expert::expert_gemv` を共有する。CPU と GPU の
//!   ビット一致はこの順序で決まる)。
//! - 行のデコードは `kquant::dequant_row_into` と同一 (GGML のブロック仕様)。
//!
//! GPU 経路はまだ繋いでいない (重みの常駐キャッシュと一緒に入れる)。現状は
//! 常に CPU 参照で計算する。Flash-Next の 1 トークンあたりの実測はこの経路で
//! 取れるが、120B を CPU で回すので秒/token 級になる。

use anyhow::{bail, Result};

use crate::kquant::{dequant_row_into, GgmlType};
use crate::model::moe_expert::{
    bf16_fast_enabled, expert_gemv_bf16, expert_gemv_bf16_2row, expert_gemv_bf16_min,
    expert_gemv_bf16_min_2row, expert_gemv, expert_gemv_iq4xs, expert_gemv_iq4xs_2row,
    expert_gemv_iq4xs_min, expert_gemv_iq4xs_min_2row, expert_gemv_par_min, expert_gemv_q8_fast,
    expert_gemv_q8_fast_2row, expert_gemv_q8_fast_min, expert_gemv_q8_fast_min_2row,
    iq4xs_fast_enabled, PAR_MIN_ROWS,
};

use crate::quant::WeightBlocks;

/// テスト用: `two_row` の呼び出しが GPU 経路を素通りして CPU 側に届いた回数。
///
/// P86 のバイパス (`two_row` は `maybe_gpu_gguf_gemv_into` を試みない) が実際に
/// 通ったことの証拠。GPU が呼び出しを奪った (旧経路) 場合は増えない。
#[cfg(test)]
pub(crate) static TWO_ROW_CPU_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 1 行のバイト数。`moe_expert::row_bytes` は量子型だけを受けるので、
/// F32 / F16 / BF16 も含めて受ける版をここに持つ (ブロック仕様は同じ)。
fn row_bytes_any(ty: GgmlType, k: usize) -> Result<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if k == 0 || k % bs != 0 {
        bail!("GgufLinear: k={k} が {ty:?} のブロックサイズ {bs} の倍数でない");
    }
    Ok((k / bs) * ts)
}

/// `DENNO_CPU_GEMV_PAR` の値の解釈 (純粋関数)。**既定 on**、`0` のときだけ無効。
///
/// 2026-10-30 の実測 (118 出力 tok、3 ラン/構成) で 25.66 -> 25.14 s (-2.2 %)、
/// 出力 md5 は全ラン一致 (`3b77a3b996fd`)。行は互いに独立なので数値は変わらない。
/// env を読む部分から切り離してあるのは、テストが環境変数を触らずに両方の分岐を
/// 確かめられるようにするため (`moe_par_from_value` と同じ流儀)。
fn cpu_gemv_par_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
///
/// env はプロセスで 1 回しか読めず、テストは並列に走るので、`devprof` の
/// `set_on_for_test` と同じく明示的に切り替えられる入口を用意する。
static CPU_GEMV_PAR_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// 行方向の並列化が有効か (`DENNO_CPU_GEMV_PAR=1`、既定 off)。プロセスで 1 回だけ読む。
fn cpu_gemv_par_enabled() -> bool {
    match CPU_GEMV_PAR_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cpu_gemv_par_from_value(std::env::var("DENNO_CPU_GEMV_PAR").ok().as_deref()))
}

/// テスト専用: 行並列の on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_cpu_gemv_par_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    CPU_GEMV_PAR_OVERRIDE.store(match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }, Ordering::Relaxed);
}

/// task 69: `DENNO_ROUTER_GPU` の値の解釈 (純関数)。**既定 off**。
///
/// ルータ (`ffn_gate_inp`, f32 512x2560) だけを GPU (`dno_matvec_f32`) で回す。
/// このカーネルは 1 行 = 1 ブロックで `s = fadd_rn(s, fmul_rn(x[d], w[d]))` を
/// d の昇順に繰り返す (PTX に `fma.rn` も `.ftz` も無い) ので、CPU 側の
/// `acc += row[i] * x[i]` (i 昇順・積と加算を別々に丸める) と**ビット一致**する。
///
/// ただし main の実測 (2026-10-31 03:00 JST、v3 rows=8 の decode 118 tok、3 ラン平均) では
/// **遅くなる**: off 121.7 ms/token (17.82/18.15/18.04 s) -> on 124.5 (18.18/18.35/18.39)。
/// 1 token あたり 48 回の同期が支配的で、CPU 16.6 ms/token の削減分を食い潰す。
/// 既定を on に戻すときは、まず同期を減らす (層をまたいでまとめる) こと。
fn router_gpu_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => false,
    }
}

/// ルータ GPU 経路のテスト用上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static ROUTER_GPU_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// ルータを GPU で回してよいか (`DENNO_ROUTER_GPU`、既定 on)。プロセスで 1 回だけ読む。
///
/// GPU が無いビルド (`feature = "cuda"` なし) では呼び出し側が無いので dead に
/// なるが、判定関数そのものはテストから同じ形で叩けるよう残す。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn router_gpu_enabled() -> bool {
    match ROUTER_GPU_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| router_gpu_from_value(std::env::var("DENNO_ROUTER_GPU").ok().as_deref()))
}

/// テスト専用: ルータ GPU 経路の on/off を強制する (`None` で env に戻す)。
///
/// `CPU_GEMV_PAR_OVERRIDE` と同じくプロセス全体で 1 個なので、触るテストは
/// `par_test_lock()` を取って直列化する。
#[cfg(test)]
pub(crate) fn set_router_gpu_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    ROUTER_GPU_OVERRIDE.store(match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }, Ordering::Relaxed);
}

/// task 72: `DENNO_FUSED_GEMV` の値の解釈 (純関数)。**既定 on**、明示的な偽値だけ無効。
///
/// `FusedLinear::concat` を通る層 (ssm の qkv+gate / beta+alpha、full attention の
/// q/k/v) だけが対象。行は互いに独立で、1 行の内積は `i` の昇順に f32 で加算する
/// (連結しても 1 行の中の順序は変わらない) ので、個別に呼んだ場合と**ビット一致**
/// する。崩れたときは `DENNO_FUSED_GEMV=0` で従来経路へ戻せる。
fn fused_gemv_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// 融合 GEMV のテスト用上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static FUSED_GEMV_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// 融合 GEMV を有効にしてよいか (`DENNO_FUSED_GEMV`、既定 on)。プロセスで 1 回だけ読む。
fn fused_gemv_enabled() -> bool {
    match FUSED_GEMV_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fused_gemv_from_value(std::env::var("DENNO_FUSED_GEMV").ok().as_deref()))
}

/// テスト専用: 融合 GEMV の on/off を強制する (`None` で env に戻す)。
///
/// 上書きはプロセス全体で 1 個なので、触るテストは `par_test_lock()` を取る。
#[cfg(test)]
pub(crate) fn set_fused_gemv_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    FUSED_GEMV_OVERRIDE.store(match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }, Ordering::Relaxed);
}

/// 行並列を立てる条件 (実測で決めた)。
///
/// - **行数が 8 以上**、または
/// - **1 呼び出しの積和 (行数 × k) が 32768 以上** (行数 4 でも k が広ければ得)。
///
/// 根拠は `bench_cpu_gemv_par_threshold_sweep` (8 / 16 スレッド、reps=50):
/// 行数 4・k=2560 (10240 MAC) は 0.78〜0.92x で損、行数 2〜3 は k=10240 でも
/// 0.87〜1.08x で損益なし。行数 4・k=10240 (40960 MAC) は 1.22〜1.45x、
/// 行数 8 は k=2560 で 1.46〜1.99x、k=10240 で 1.89〜2.31x。
/// 実モデルの 4 形は router (512×2560)・ssm_beta/alpha (48×2560)・
/// hc_inject (4×10240) がすべて条件を満たす (`REPORT_task65.md`)。
const CPU_GEMV_PAR_MIN_ROWS: usize = 8;
/// 行数がこれ未満でも、積和がこの値以上なら並列化する。
const CPU_GEMV_PAR_MIN_MACS: usize = 32_768;

/// テスト専用: しきい値を下げて小さい形の並列損得を測るための上書き (0 = 既定)。
#[cfg(test)]
static PAR_MIN_ROWS_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 実際に使うしきい値 (テストでは上書きできる)。
fn cpu_gemv_par_min_rows() -> usize {
    #[cfg(test)]
    {
        let v = PAR_MIN_ROWS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if v > 0 {
            return v;
        }
    }
    CPU_GEMV_PAR_MIN_ROWS
}

#[cfg(test)]
pub(crate) fn set_par_min_rows_for_test(v: Option<usize>) {
    PAR_MIN_ROWS_OVERRIDE.store(v.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// この形を並列化するか (行数と積和のしきい値)。
fn cpu_gemv_par_applies(n: usize, k: usize) -> bool {
    n >= cpu_gemv_par_min_rows() || n.saturating_mul(k) >= CPU_GEMV_PAR_MIN_MACS
}

/// 行並列の起動回数 (テストが「1 呼び出し = 1 領域」を数えるため)。
#[cfg(test)]
static PAR_LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// 1 呼び出しで割った行チャンクの本数 (行ごとに並列領域を立てていないことの確認)。
#[cfg(test)]
static PAR_TASKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// task 106: `gemv_rows_par` の F32 direct 経路 (行バッファ無し分支) が処理した行数。
///
/// 「F32 の行並列が direct 分岐に実際に乗った」ことの証拠。F16 / BF16 (copy 経路)
/// や env off (旧経路) では増えない。
#[cfg(test)]
static F32_DIRECT_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// テスト専用: 行並列の状態 (env 上書き・カウンタ) を触るテストを直列化する。
///
/// 上書きはプロセス全体で 1 個なので、crate 内の他のテスト (`qwen4exp` の
/// 入口テストなど) も同じ mutex を取って、カウンタの取り合いを避ける。
#[cfg(test)]
pub(crate) fn par_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(crate) fn par_reset() {
    PAR_LAUNCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    PAR_TASKS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn par_launches() -> usize {
    PAR_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn par_tasks() -> usize {
    PAR_TASKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// テスト専用: F32 direct 経路が処理した行数 (`F32_DIRECT_ROWS`)。
#[cfg(test)]
pub(crate) fn f32_direct_rows() -> usize {
    F32_DIRECT_ROWS.load(std::sync::atomic::Ordering::Relaxed)
}

/// テスト専用: F32 direct カウンタを 0 に戻す。
#[cfg(test)]
pub(crate) fn f32_direct_reset() {
    F32_DIRECT_ROWS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_par_launch() {
    #[cfg(test)]
    PAR_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_par_task() {
    #[cfg(test)]
    PAR_TASKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 非量子型 (F32 / F16 / BF16) の `out[r] = dot(row_r, x)` を**行方向に**分ける。
///
/// - 行は互いに独立で、1 行の内積は `i` の昇順に f32 で逐次加算する (従来と同じ
///   `dequant_row_into` + 同じ順の `acc +=`)。どの行をどのスレッドが担当しても
///   各要素の丸め順序は変わらないので、**分割の有無でビット一致する**。
/// - 並列領域は 1 呼び出しにつき 1 回だけ立てる (行ループの内側では立てない)。
/// - task 106: F32 だけ行バッファを介さない。`dequant_row_into(F32)` は
///   `f32::from_le_bytes` の恒等変換なので、重みバイトから直接
///   `acc += from_le_bytes(w[i]) * x[i]` を `i` 昇順で回しても**値も順序も同じ**
///   (ビット一致。[`crate::kquant::dequant_row_into`] の F32 アームと同じ変換)。
///   行バッファの確保 (k×4 B) と 2 パス目の読み出しが消える。ルータ
///   (`ffn_gate_inp` 512x2560 = 5.24 MB) は decode の router bucket の主体で、
///   copy 版は重みを「読んで書いて、また読む」3 パスになるのが無駄。
///   値の契約は [`tests::cpu_gemv_par_bitwise_on_real_shapes`] (独立参照
///   `naive_gemv` との突き合わせ) が担う。
fn gemv_rows_par(ty: GgmlType, blocks: &[u8], rb: usize, k: usize, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    let threads = rayon::current_num_threads().max(1);
    // 1 スレッドあたり最低 1 行。行数がスレッド数で割り切れない端数は最後の
    // チャンクに入る (`par_chunks` の末尾は短くなる)。
    let rows = out.len().div_ceil(threads).max(1).min(out.len());
    let bytes = rows * rb;
    note_par_launch();
    blocks
        .par_chunks(bytes)
        .zip(out.par_chunks_mut(rows))
        .for_each(|(bb, oo)| {
            note_par_task();
            if ty == GgmlType::F32 {
                debug_assert_eq!(rb, k * 4, "F32 の行バイトは k*4");
                debug_assert_eq!(x.len(), k, "gemv_rows_par: x.len() は k");
                #[cfg(test)]
                F32_DIRECT_ROWS.fetch_add(oo.len(), std::sync::atomic::Ordering::Relaxed);
                for (r, slot) in oo.iter_mut().enumerate() {
                    let wb = &bb[r * rb..(r + 1) * rb];
                    let mut acc = 0f32;
                    for (c, &xv) in wb.chunks_exact(4).zip(x.iter()) {
                        acc += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xv;
                    }
                    *slot = acc;
                }
                return;
            }
            let mut row = vec![0f32; k];
            for (r, slot) in oo.iter_mut().enumerate() {
                dequant_row_into(ty, &bb[r * rb..(r + 1) * rb], &mut row);
                let mut acc = 0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                *slot = acc;
            }
        });
}

/// GGUF のブロックを保持したまま GEMV する線形層 (n 行 k 列、y = W x)。
pub struct GgufLinear {
    uid: u64,
    ty: GgmlType,
    k: usize,
    n: usize,
    blocks: WeightBlocks,
    /// Q8_0 のときだけ、AVX2 融合カーネルを使うか。
    ///
    /// 既定は false (27B の Q8_0 テンソル 1 本がビット一致契約の対象なので、
    /// 明示的に選んだモデルだけ速い経路に載せる)。
    fast_q8: bool,
    /// task 69: ルータ (`ffn_gate_inp`) だけ true。f32 重みを VRAM に常駐させ、
    /// `dno_matvec_f32` で計算する (ビット一致・`DENNO_ROUTER_GPU` で on/off)。
    router_gpu: bool,
}

/// 融合カーネルを持たない型を Q8_0 へ作り直す共通の入口。
///
/// `DENNO_REQUANT_Q8=0` で無効化 (`DENNO_REQUANT_Q8` 未設定は有効)。
pub fn maybe_requant(l: GgufLinear) -> Result<GgufLinear> {
    let off = std::env::var("DENNO_REQUANT_Q8").ok().as_deref() == Some("0");
    if off || l.has_fused_kernel() {
        return Ok(l);
    }
    // F32/F16/BF16 は作り直さない。ルータ (`ffn_gate_inp`)・norm・hc_inject など
    // 小さな行列で、F32 のまま行並列 GEMV / GPU ルータ経路を使うほうが速く、
    // 量子化すると数値も変わる (task 65 の経路が死ぬ)。
    if matches!(l.ty(), GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16) {
        return Ok(l);
    }
    l.requant_to_q8_0()
}

impl GgufLinear {
    /// 生バイトから作る。`blocks.len()` は `n * row_bytes(ty, k)` と一致すること。
    pub fn new(ty: GgmlType, k: usize, n: usize, blocks: WeightBlocks) -> Result<Self> {
        let rb = row_bytes_any(ty, k)?;
        if n == 0 || k == 0 {
            bail!("GgufLinear: 形状が 0 (k={k}, n={n})");
        }
        let want = n
            .checked_mul(rb)
            .ok_or_else(|| anyhow::anyhow!("GgufLinear: n * row_bytes が usize を溢れる"))?;
        if blocks.as_slice().len() != want {
            bail!(
                "GgufLinear: バイト数 {} が n * row_bytes = {} と違う (ty={ty:?}, k={k}, n={n})",
                blocks.as_slice().len(),
                want
            );
        }
        Ok(Self {
            uid: crate::quant::next_q8_uid(),
            ty,
            k,
            n,
            blocks,
            fast_q8: false,
            router_gpu: false,
        })
    }

    /// task 69: この線形層をルータ GPU 経路の対象にする (`ffn_gate_inp` 専用)。
    pub fn with_router_gpu(mut self, on: bool) -> Self {
        self.router_gpu = on;
        self
    }

    /// ルータ GPU 経路の対象か。
    pub fn router_gpu(&self) -> bool {
        self.router_gpu
    }

    /// Q8_0 の AVX2 融合カーネルを有効にする (ビット一致は保証しない)。
    pub fn with_fast_q8(mut self, on: bool) -> Self {
        self.fast_q8 = on;
        self
    }

    /// Q8_0 の融合カーネルが有効か (既定 false)。
    pub fn fast_q8(&self) -> bool {
        self.fast_q8
    }

    /// 密行列を Q8_0 へ**作り直す** (融合カーネル・GPU 常駐の対象型へ載せ替える)。
    ///
    /// 2026-10-31: Swift 1.5 の密行列は IQ4_XS / Q4_K / Q5_K / Q6_K が層ごとに
    /// 混在する。これらの型には融合 GEMV も GPU GEMM も無く、`gemv` は 1 行ずつ
    /// デコードするスカラー経路に落ちる (実測 decode 3960 ms/token、うち attn の
    /// qkv+out が 15 s / 9 token)。Q8_0 へ作り直すと既存の AVX2 融合カーネル・
    /// GPU 常駐・バッチ経路がそのまま使える。量子化誤差は元の 4 bit 量子化に対して
    /// 十分小さく、**元の値からの相対誤差は Q8_0 の丸め (≤0.4%) だけ**になる。
    ///
    /// 行は独立なので並列に作れる。`DENNO_REQUANT_Q8=0` で無効化できる。
    pub fn requant_to_q8_0(self) -> Result<Self> {
        if self.ty == GgmlType::Q8_0 {
            return Ok(self);
        }
        let (k, n) = (self.k, self.n);
        if k % 32 != 0 {
            bail!("requant_to_q8_0: k={k} が 32 の倍数でない");
        }
        let rb = row_bytes_any(self.ty, k)?;
        let src = self.blocks.as_slice();
        let nblk = k / 32;
        let out_rb = nblk * 34;
        let mut out = vec![0u8; n * out_rb];
        let ty = self.ty;
        let rows_per = n.div_ceil(std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4));
        std::thread::scope(|sc| {
            let mut rest: &mut [u8] = &mut out;
            let mut done = 0usize;
            while done < n {
                let take = rows_per.min(n - done);
                let (chunk, tail) = rest.split_at_mut(take * out_rb);
                let s = src;
                sc.spawn(move || {
                    let mut row = vec![0f32; k];
                    for r in 0..take {
                        let g = done + r;
                        dequant_row_into(ty, &s[g * rb..(g + 1) * rb], &mut row);
                        let dst = &mut chunk[r * out_rb..(r + 1) * out_rb];
                        for b in 0..nblk {
                            let x = &row[b * 32..(b + 1) * 32];
                            let mut amax = 0f32;
                            for v in x {
                                let a = v.abs();
                                if a > amax {
                                    amax = a;
                                }
                            }
                            let d = amax / 127.0;
                            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                            let o = &mut dst[b * 34..(b + 1) * 34];
                            let dbits = denno_telemetry::f32_to_f16_bits(d);
                            o[0..2].copy_from_slice(&dbits.to_le_bytes());
                            for j in 0..32 {
                                let q = (x[j] * id).round().clamp(-127.0, 127.0) as i8;
                                o[2 + j] = q as u8;
                            }
                        }
                    }
                });
                rest = tail;
                done += take;
            }
        });
        GgufLinear::new(GgmlType::Q8_0, k, n, WeightBlocks::from_vec(out))
            .map(|l| l.with_fast_q8(true))
    }

    /// 融合カーネルがある型か (Q8_0 / Q5_1 / Q4_K / Q5_K)。
    pub fn has_fused_kernel(&self) -> bool {
        matches!(
            self.ty,
            GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K
        )
    }

    /// スライスから作る (テスト・小さいテンソル用)。
    pub fn from_slice(ty: GgmlType, k: usize, n: usize, bytes: &[u8]) -> Result<Self> {
        Self::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
    }

    pub fn uid(&self) -> u64 {
        self.uid
    }
    pub fn ty(&self) -> GgmlType {
        self.ty
    }
    /// 入力長 (列数)
    pub fn k(&self) -> usize {
        self.k
    }
    /// 出力長 (行数)
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn blocks(&self) -> &WeightBlocks {
        &self.blocks
    }
    pub fn weight_bytes(&self) -> usize {
        self.blocks.as_slice().len()
    }

    /// `out[r] = dot(row_r, x)`。`x.len() == k`、`out.len() == n` が必要。
    ///
    /// プロファイラ上の行名は既定で `"gemv"` (op 名を付けたいときは
    /// [`Self::gemv_label`])。
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.gemv_label("gemv", x, out)
    }

    /// [`Self::gemv`] の op 名付き入口。
    ///
    /// `label` は device プロファイラ (`DENNO_DEV_PROFILE=1`) の行名にだけ使う。
    /// 数値経路には影響しない (文字列を渡すだけで、カーネル・引数・丸め順は
    /// 同一)。off のときは `devprof::on()` の atomic load 1 回で従来と同じ経路。
    pub fn gemv_label(&self, label: &'static str, x: &[f32], out: &mut [f32]) -> Result<()> {
        use crate::model::devprof;
        if !devprof::on() {
            return self.gemv_impl(x, out);
        }
        let t = std::time::Instant::now();
        devprof::clear();
        let r = self.gemv_impl(x, out);
        let ns = t.elapsed().as_nanos() as u64;
        let (up, down, sync_ns) = devprof::take_pending();
        devprof::add(label, ns, up, down, sync_ns);
        r
    }

    /// [`Self::gemv_label`] の行並列しきい値を呼び出し側が決める版。
    ///
    /// `row_par_min` 行以上のときだけ行を rayon で分ける (`0` は既定 = 512 行)。
    /// **行は独立**なので、しきい値を下げても出力はビット一致する (変わるのは
    /// 分割の粒度だけ)。対象は Q8_0 の融合カーネル (`fast_q8`) のみで、他の型は
    /// 既定の門番のまま。
    pub fn gemv_label_par(
        &self,
        label: &'static str,
        x: &[f32],
        out: &mut [f32],
        row_par_min: usize,
    ) -> Result<()> {
        use crate::model::devprof;
        if !devprof::on() {
            return self.gemv_impl_par(x, out, row_par_min, false);
        }
        let t = std::time::Instant::now();
        devprof::clear();
        let r = self.gemv_impl_par(x, out, row_par_min, false);
        let ns = t.elapsed().as_nanos() as u64;
        let (up, down, sync_ns) = devprof::take_pending();
        devprof::add(label, ns, up, down, sync_ns);
        r
    }

    /// [`Self::gemv_label_par`] の 2 行同時カーネル版 (P84、hc mix 専用)。
    ///
    /// Q8_0 融合カーネルを隣接 2 行を同時に処理する版に変えるだけで、出力は
    /// [`Self::gemv_label_par`] と**ビット一致**する (行ごとの累積順序は不変)。
    /// 対象・しきい値の意味は [`Self::gemv_label_par`] と同じ
    /// (`row_par_min = 0` は既定の門番)。
    ///
    /// task 88: `two_row` 明示の呼び出しでは Q8_0 以外の量子化型も行並列に乗る
    /// (融合カーネルが無い型は行チャンクで割るだけ。[`expert_gemv_par_min`] 参照)。
    pub fn gemv_label_par_2row(
        &self,
        label: &'static str,
        x: &[f32],
        out: &mut [f32],
        row_par_min: usize,
    ) -> Result<()> {
        use crate::model::devprof;
        if !devprof::on() {
            return self.gemv_impl_par(x, out, row_par_min, true);
        }
        let t = std::time::Instant::now();
        devprof::clear();
        let r = self.gemv_impl_par(x, out, row_par_min, true);
        let ns = t.elapsed().as_nanos() as u64;
        let (up, down, sync_ns) = devprof::take_pending();
        devprof::add(label, ns, up, down, sync_ns);
        r
    }

    /// 本体 ([`Self::gemv_label`] 用。行並列は既定の門番)。
    fn gemv_impl(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.gemv_impl_par(x, out, 0, false)
    }

    /// 本体。[`Self::gemv_label`] が時間と転送量を測るために外側から包む。
    ///
    /// `row_par_min != 0` のとき、Q8_0 融合カーネルの行並列の門番を差し替える。
    /// `two_row` は Q8_0 融合カーネルに 2 行同時版
    /// ([`expert_gemv_q8_fast_2row`]) を選ばせる (値は classic 版とビット一致)。
    fn gemv_impl_par(
        &self,
        x: &[f32],
        out: &mut [f32],
        row_par_min: usize,
        two_row: bool,
    ) -> Result<()> {
        if x.len() != self.k {
            bail!("GgufLinear::gemv: x の長さ {} が k={} と違う", x.len(), self.k);
        }
        if out.len() != self.n {
            bail!("GgufLinear::gemv: out の長さ {} が n={} と違う", out.len(), self.n);
        }
        // task 87: 実行先フラグの初期値 (s57 の `maybe_gpu_*` 入口と同じ規約)。
        // GPU 経路を**試みない**呼び出し (P86 の two_row バイパス / F32・F16・BF16
        // の CPU 参照) でも毎回 false から始めないと、直前の呼び出しの true が残り、
        // attn_kprof (`note_gate`) が「GPU を通った」ように誤って数える。計測のみ。
        #[cfg(feature = "cuda")]
        crate::cuda::set_last_linear_ran_on_gpu(false);
        match self.ty {
            // 量子型は expert GEMV と本体を共有する (1 行ずつデコードして逐次内積)。
            // 2026-10-31: 融合カーネルの有無ではなく「デコードできるか」で通す。
            // ここに来た型は expert GEMV (= 1 行デコード + 逐次内積) の
            // スカラー経路で計算する。Swift 1.5 は IQ4_XS / IQ3_S / IQ2_S /
            // Q2_0 などを密行列にも使うため、旧 whitelist では読めなかった。
            ty if !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16)
                && crate::kquant::dequant_supported(ty) => {
                // P-MOE-E5: 重みが VRAM に常駐していれば GPU で計算する。
                // 対象外・GPU 無効・不調では false が返り、下の CPU 参照へ落ちる。
                //
                // P86: `two_row` が明示された呼び出し (hc mix の decode 1 トークン経路
                // だけが使う) は GPU に奪わない。実機では hc down/up (Q8_0, 3.4 MB) の
                // GPU GEMV が 1 呼び出し 0.6-0.7 ms かかり、CPU 側の 2 行 + 行並列は
                // 一度も走っていなかった ([mix-kprof] down 48% / up 43.4%、
                // `rowpar regions=0` = `for_each_row_chunk` 未到達の証拠)。
                // two_row は「CPU の 2 行カーネルでビット一致計算したい」という呼び出し
                // 側の明示なので、ここを素通しさせない。旧経路 (GPU に落ちる) は
                // `DENNO_HC_MIX_FAST=0` で戻る。
                #[cfg(feature = "cuda")]
                {
                    if !two_row
                        && crate::cuda::maybe_gpu_gguf_gemv_into(
                            self.uid,
                            self.ty,
                            self.blocks.as_slice(),
                            self.k,
                            self.n,
                            x,
                            out,
                        )
                    {
                        return Ok(());
                    }
                }
                #[cfg(test)]
                if two_row {
                    TWO_ROW_CPU_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if self.fast_q8 && self.ty == GgmlType::Q8_0 {
                    return if row_par_min == 0 {
                        if two_row {
                            expert_gemv_q8_fast_2row(self.blocks.as_slice(), self.k, self.n, x, out)
                        } else {
                            expert_gemv_q8_fast(self.blocks.as_slice(), self.k, self.n, x, out)
                        }
                    } else if two_row {
                        expert_gemv_q8_fast_min_2row(
                            self.blocks.as_slice(),
                            self.k,
                            self.n,
                            x,
                            out,
                            row_par_min,
                        )
                    } else {
                        expert_gemv_q8_fast_min(
                            self.blocks.as_slice(),
                            self.k,
                            self.n,
                            x,
                            out,
                            row_par_min,
                        )
                    };
                }
                // P89: IQ4_XS の融合カーネル (本番モデルの hc テンソルの実型)。
                // fast_q8 分岐と同じ形で `two_row` / `row_par_min` を伝える。
                // `DENNO_IQ4XS_FAST=0` で下の classic ([`expert_gemv`] 経由のスカラー)
                // に戻る。値は classic とビット一致しない (内積の累積順が違う。相対
                // 1e-6 程度)。
                if self.ty == GgmlType::IQ4Xs && iq4xs_fast_enabled() {
                    return if row_par_min == 0 {
                        if two_row {
                            expert_gemv_iq4xs_2row(self.blocks.as_slice(), self.k, self.n, x, out)
                        } else {
                            expert_gemv_iq4xs(self.blocks.as_slice(), self.k, self.n, x, out)
                        }
                    } else if two_row {
                        expert_gemv_iq4xs_min_2row(
                            self.blocks.as_slice(),
                            self.k,
                            self.n,
                            x,
                            out,
                            row_par_min,
                        )
                    } else {
                        expert_gemv_iq4xs_min(
                            self.blocks.as_slice(),
                            self.k,
                            self.n,
                            x,
                            out,
                            row_par_min,
                        )
                    };
                }
                // task 88: Q8_0 以外の量子化型 (`two_row` 明示の呼び出し = hc mix の
                // decode 経路) は、融合カーネルが無くても行並列の門番 (`row_par_min`)
                // を効かせる。実機の hc down/up は Q8_0 でないことが GGUF ヘッダ実読
                // で判明した (P84/P86 の 2 行カーネルが型の門番で発火していなかった)。
                // 行は独立で 1 行の加算順は不変なので、[`expert_gemv`] とビット一致。
                // `row_par_min == 0` は既定の門番 ([`moe_expert::PAR_MIN_ROWS`]) に委譲。
                if two_row {
                    let min_rows = if row_par_min == 0 { PAR_MIN_ROWS } else { row_par_min };
                    return expert_gemv_par_min(self.ty, self.blocks.as_slice(), self.k, self.n, x, out, min_rows);
                }
                expert_gemv(self.ty, self.blocks.as_slice(), self.k, self.n, x, out)
            }
            // 無量子型 (F32 / F16 / BF16) は expert GEMV の対象外なので同じ規則で回す。
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
                let rb = row_bytes_any(self.ty, self.k)?;
                let b = self.blocks.as_slice();
                // task 69: ルータ (`ffn_gate_inp`, 512x2560 の F32) だけ GPU 経路。
                // カーネルは `dno_matvec_f32` (1 行 = 1 ブロック・k 昇順の逐次加算・
                // `mul.rn`+`add.rn` で FMA 無し) なので、下の CPU 参照
                // (`acc += row[i] * x[i]` を i 昇順) と丸め順が同じ = ビット一致。
                // 不発 (GPU 無効・予算外・失敗) はすべて CPU 参照へ落ちる。
                #[cfg(feature = "cuda")]
                {
                    if self.router_gpu
                        && router_gpu_enabled()
                        && crate::cuda::maybe_gpu_router_gemv_into(
                            self.uid,
                            b,
                            self.k,
                            self.n,
                            x,
                            out,
                        )
                    {
                        return Ok(());
                    }
                }
                // task 92: BF16 (hc mix の down/up 実型) は AVX2 融合カーネルへ。
                // BF16 -> f32 変換は正確で、カーネルは積を i 昇順で逐次加算するので
                // 下の classic (dequant + 逐次内積) と**ビット一致**する
                // (`DENNO_BF16_FAST=0` で下に戻る。戻っても値は変わらない)。
                // two_row / row_par_min は Q8_0 / IQ4_XS と同じ形で伝える (カーネルは
                // もともと複数行を交互に回すので 2 行指定の有無で経路は変わらない)。
                if self.ty == GgmlType::Bf16 && bf16_fast_enabled() {
                    return if row_par_min == 0 {
                        if two_row {
                            expert_gemv_bf16_2row(b, self.k, self.n, x, out)
                        } else {
                            expert_gemv_bf16(b, self.k, self.n, x, out)
                        }
                    } else if two_row {
                        expert_gemv_bf16_min_2row(b, self.k, self.n, x, out, row_par_min)
                    } else {
                        expert_gemv_bf16_min(b, self.k, self.n, x, out, row_par_min)
                    };
                }
                // task 91: `two_row` / `row_par_min` の明示呼び出し (hc mix の decode
                // 経路) は、無量子型 (実機 P90 の mix-kprof で判明した hc down/up の
                // BF16 を含む) でも行並列の行チャンクに乗せる ([`expert_gemv_par_min`]
                // = `for_each_row_chunk` 枠組み)。旧来の無量子型アームは two_row /
                // row_par_min を無視し、`mix_kprof::rowpar` の region 計上も素通りして
                // いた (実機 [mix-kprof] の `regions=0` の最終的な理由)。
                //
                // 1 行の計算は `gemv_scalar_body` = `dequant_row_into` (BF16 -> f32 は
                // 上位 16bit の複製で正確に定義される) + `dot_seq` (k 昇順の逐次加算)
                // で、下の逐次経路・`gemv_rows_par` と**ビット一致**する (行は独立、
                // 行内の加算順はどのチャンクに割っても不変)。region 計上は
                // `for_each_row_chunk` 側が面倒を見る (ラベルは hc mix 側が立てる)。
                // `row_par_min == 0` は既定の門番 ([`moe_expert::PAR_MIN_ROWS`]) に
                // 委譲する (量子型 two_row と同じ規約)。ルータ (`router_gpu`) は
                // `two_row` / `row_par_min` を付けない呼び出しなので、この分岐の
                // 影響を受けない。
                if two_row || row_par_min != 0 {
                    let min_rows = if row_par_min == 0 { PAR_MIN_ROWS } else { row_par_min };
                    return expert_gemv_par_min(self.ty, b, self.k, self.n, x, out, min_rows);
                }
                // task 65: `DENNO_CPU_GEMV_PAR=1` のときだけ、行方向を rayon で分ける。
                // 既定 off は従来経路 (下の feature="par" の分割も含めて 1 行も変えない)。
                // 行は独立・行内の加算順は不変なので、分割の有無で値はビット一致する。
                if cpu_gemv_par_enabled() && self.k >= 256 && cpu_gemv_par_applies(self.n, self.k) {
                    gemv_rows_par(self.ty, b, rb, self.k, x, out);
                    return Ok(());
                }
                // P-MOE-G11: 行を rayon で分割する。行は独立で、行内の逐次加算順は
                // 変えないので、分割の有無で結果はビット一致する (従来の逐次版と
                // 同じ `dequant_row_into` + 同じ順の内積を通る)。
                // MoE のルータ (`ffn_gate_inp`, F32, 2560x512 = 5.2 MB) がここを通り、
                // 1 層あたり 1.07 ms かかっていた (moe の 42%)。
                #[cfg(feature = "par")]
                if self.n >= 64 && self.k >= 256 {
                    use rayon::prelude::*;
                    let threads = rayon::current_num_threads().max(1);
                    let rows = self.n.div_ceil(threads).max(32).min(self.n);
                    let bytes = rows * rb;
                    b.par_chunks(bytes).zip(out.par_chunks_mut(rows)).for_each(
                        |(bb, oo)| {
                            let mut row = vec![0f32; self.k];
                            for (r, slot) in oo.iter_mut().enumerate() {
                                dequant_row_into(self.ty, &bb[r * rb..(r + 1) * rb], &mut row);
                                let mut acc = 0f32;
                                for i in 0..self.k {
                                    acc += row[i] * x[i];
                                }
                                *slot = acc;
                            }
                        },
                    );
                    return Ok(());
                }
                let mut row = vec![0f32; self.k];
                for r in 0..self.n {
                    dequant_row_into(self.ty, &b[r * rb..(r + 1) * rb], &mut row);
                    let mut acc = 0f32;
                    for i in 0..self.k {
                        acc += row[i] * x[i];
                    }
                    out[r] = acc;
                }
                Ok(())
            }
            other => bail!("GgufLinear::gemv: 未対応の量子型 {other:?}"),
        }
    }

    /// 行 j を f32 に展開する (検証・デバッグ用)。
    pub fn dequant_row(&self, j: usize) -> Result<Vec<f32>> {
        if j >= self.n {
            bail!("GgufLinear::dequant_row: 行 {j} が n={} の範囲外", self.n);
        }
        let rb = row_bytes_any(self.ty, self.k)?;
        let b = self.blocks.as_slice();
        let mut out = vec![0f32; self.k];
        dequant_row_into(self.ty, &b[j * rb..(j + 1) * rb], &mut out);
        Ok(out)
    }
}

/// task 72: 同じ入力 `x` を取る複数の投影を、行方向に連結して 1 本の GEMV にする。
///
/// 行は互いに独立で、1 行の内積は `i` の昇順に f32 で加算する (個別に呼んだ
/// 場合と同じ順序)。よって連結の有無・切り分けの位置で値は**ビット一致**する
/// (`fused_split_matches_separate_bits` が固定)。
///
/// 連結のしかたは 2 通り:
/// - **写しなし**: 元の層が mmap の隣接した範囲を指しているときは、まとめた 1 本の
///   共有写像 ([`WeightBlocks::mapped`]) を指すだけ。匿名 RSS は増えない。
///   ファイル上の並びが逆でも、行範囲の対応表 (`ranges`) で吸収する。
/// - **写しあり**: 隣接していない / ヒープ実体のときは 1 本の `Vec` に連結する。
///   元の層は捨てるので、常駐するバイト数は連結前と同じ。
///
/// 型・`k` が揃わないときは連結せず、`parts` を順に呼ぶ (従来と同じ経路)。
pub struct FusedLinear {
    /// 連結済みの 1 本 (`Some` のとき `parts` は空)。
    inner: Option<GgufLinear>,
    /// 連結しなかったときの元の層 (呼び出し順)。
    parts: Vec<GgufLinear>,
    /// 呼び出し順の行境界 (累積終端)。長さ = 層数。合計 = [`Self::n`]。
    splits: Vec<usize>,
    /// 呼び出し順 → 連結バッファ内の行範囲。連結していないときは `splits` と同じ。
    ranges: Vec<std::ops::Range<usize>>,
    /// 写しなしで連結できたか (診断・テスト用)。
    zero_copy: bool,
}

impl FusedLinear {
    /// 与えた順に並べた層を 1 本にまとめる。まとめられなければ個別のまま持つ。
    ///
    /// `DENNO_FUSED_GEMV=0` のときは何もせず個別のまま ([`Self::separate`])。
    pub fn concat(parts: Vec<GgufLinear>) -> Self {
        let mut me = Self::separate(parts);
        if !fused_gemv_enabled() || me.parts.len() < 2 {
            return me;
        }
        if let Some((inner, ranges)) = fuse_parts(&me.parts) {
            me.zero_copy = inner.blocks().is_mapped();
            me.inner = Some(inner);
            me.ranges = ranges;
            me.parts = Vec::new();
        }
        me
    }

    /// まとめずに持つ (従来の呼び出し)。`DENNO_FUSED_GEMV=0` と同じ経路。
    pub fn separate(parts: Vec<GgufLinear>) -> Self {
        let mut splits = Vec::with_capacity(parts.len());
        let mut ranges = Vec::with_capacity(parts.len());
        let mut at = 0usize;
        for p in &parts {
            let n = p.n();
            splits.push(at + n);
            ranges.push(at..at + n);
            at += n;
        }
        Self { inner: None, parts, splits, ranges, zero_copy: false }
    }

    /// 1 本に連結できているか (偽なら個別に呼ぶ)。
    pub fn is_fused(&self) -> bool {
        self.inner.is_some()
    }

    /// 写しなしで連結できたか。`is_fused()` が偽のときは false。
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy
    }

    /// 連結前の本数。
    pub fn n_parts(&self) -> usize {
        self.splits.len()
    }

    /// 出力の合計行数 (連結バッファの長さ)。
    pub fn n(&self) -> usize {
        self.splits.last().copied().unwrap_or(0)
    }

    /// 入力長 (列数)。層が空なら 0。
    pub fn k(&self) -> usize {
        match &self.inner {
            Some(i) => i.k(),
            None => self.parts.first().map(|p| p.k()).unwrap_or(0),
        }
    }

    /// 呼び出し順の行境界 (累積終端)。
    pub fn splits(&self) -> &[usize] {
        &self.splits
    }

    /// 呼び出し順 `i` 番目の層が、連結バッファのどの行範囲に入るか。
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.ranges[i].clone()
    }

    pub fn weight_bytes(&self) -> usize {
        match &self.inner {
            Some(i) => i.weight_bytes(),
            None => self.parts.iter().map(|p| p.weight_bytes()).sum(),
        }
    }

    /// 連結 1 本の GEMV を `buf` (`n()` 要素) に書く。
    ///
    /// 連結していないときは各層を順に `buf` の行範囲へ書く (値は同じ)。
    pub fn gemv_concat_into(
        &self,
        label: &'static str,
        x: &[f32],
        buf: &mut [f32],
    ) -> Result<()> {
        if buf.len() != self.n() {
            bail!(
                "FusedLinear::gemv_concat_into: buf の長さ {} が n={} と違う",
                buf.len(),
                self.n()
            );
        }
        match &self.inner {
            Some(inner) => inner.gemv_label(label, x, buf),
            None => {
                for (p, r) in self.parts.iter().zip(self.ranges.iter()) {
                    p.gemv_label(label, x, &mut buf[r.clone()])?;
                }
                Ok(())
            }
        }
    }

    /// 連結 1 本の GEMV を計算し、層ごとに切り分けて `outs` へ写す。
    ///
    /// `buf` は `n()` 要素の作業域。連結していないときは各層が直接 `outs` へ書く。
    pub fn gemv_split(
        &self,
        label: &'static str,
        x: &[f32],
        buf: &mut [f32],
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        self.check_outs(outs, 1)?;
        self.gemv_concat_into(label, x, buf)?;
        for (o, r) in outs.iter_mut().zip(self.ranges.iter()) {
            o.copy_from_slice(&buf[r.clone()]);
        }
        Ok(())
    }

    /// チャンクぶんのバッチ GEMV (`nt` トークン) を計算し、層ごとに切り分ける。
    ///
    /// `outs[i]` は `nt * (その層の行数)` 要素。連結しているときだけ、連結ぶんの
    /// 出力を置く作業域を使う (スレッドごとに grow-only で使い回す)。
    pub fn gemv_batch_split(
        &self,
        xs: &[f32],
        nt: usize,
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        self.check_outs(outs, nt)?;
        if nt == 0 {
            return Ok(());
        }
        let Some(inner) = &self.inner else {
            for (p, o) in self.parts.iter().zip(outs.iter_mut()) {
                crate::model::qwen4exp::gemv_batch(p, xs, nt, o)?;
            }
            return Ok(());
        };
        let n = self.n();
        FUSED_BATCH_BUF.with(|cell| -> Result<()> {
            let mut b = cell.borrow_mut();
            if b.len() < nt * n {
                b.resize(nt * n, 0.0);
            }
            let buf = &mut b[..nt * n];
            crate::model::qwen4exp::gemv_batch(inner, xs, nt, buf)?;
            for (o, r) in outs.iter_mut().zip(self.ranges.iter()) {
                let w = r.len();
                for t in 0..nt {
                    o[t * w..(t + 1) * w]
                        .copy_from_slice(&buf[t * n + r.start..t * n + r.end]);
                }
            }
            Ok(())
        })
    }

    /// `outs` の本数と長さが、この層の切り分けと一致するか。
    fn check_outs(&self, outs: &[&mut [f32]], nt: usize) -> Result<()> {
        // `gemv_split` は連結バッファに書いてから写すので、長さの検算だけ行う。
        if outs.len() != self.ranges.len() {
            bail!(
                "FusedLinear: outs の本数 {} が層数 {} と違う",
                outs.len(),
                self.ranges.len()
            );
        }
        for (i, (o, r)) in outs.iter().zip(self.ranges.iter()).enumerate() {
            if o.len() != nt * r.len() {
                bail!(
                    "FusedLinear: outs[{i}] の長さ {} が {} と違う",
                    o.len(),
                    nt * r.len()
                );
            }
        }
        Ok(())
    }
}

// 連結バッチの作業域 (スレッドごとに grow-only)。
//
// `vec![0.0; nt * n]` を呼び出しごとに確保すると、初回タッチのページフォルトで
// 数十 ms 落ちる (P13 と同じ理由)。prefill のチャンクは層ごとに同じ大きさなので、
// 1 本を使い回すのが一番安い。
thread_local! {
    static FUSED_BATCH_BUF: std::cell::RefCell<Vec<f32>> =
        std::cell::RefCell::new(Vec::new());
}

/// 層を 1 本に連結する。できないときは `None`。
///
/// 型と `k` が揃っていることが前提 (揃わなければ `None`)。並びはファイル上の
/// 隣接を優先し、隣接しなければ呼び出し順のまま写して連結する。
fn fuse_parts(parts: &[GgufLinear]) -> Option<(GgufLinear, Vec<std::ops::Range<usize>>)> {
    let first = parts.first()?;
    let (ty, k) = (first.ty(), first.k());
    if !parts.iter().all(|p| p.ty() == ty && p.k() == k) {
        return None;
    }
    let total: usize = parts.iter().map(|p| p.n()).sum();
    let rb = row_bytes_any(ty, k).ok()?;
    let fast_q8 = parts.iter().all(|p| p.fast_q8());

    // 1. 写しなし: 同じ写像の隣接範囲を、どの並びで連結できるか探す
    //    (層数は 2〜3 本なので全順列で足りる)。
    if let Some(order) = contiguous_order(parts) {
        let (map, off, _) = parts[order[0]].blocks().mapping()?;
        let blocks = WeightBlocks::mapped(std::sync::Arc::clone(map), off, total * rb);
        let inner = GgufLinear::new(ty, k, total, blocks).ok()?.with_fast_q8(fast_q8);
        let mut ranges = vec![0..0; parts.len()];
        let mut at = 0usize;
        for &i in &order {
            let n = parts[i].n();
            ranges[i] = at..at + n;
            at += n;
        }
        return Some((inner, ranges));
    }

    // 2. 写しあり: 呼び出し順のまま 1 本に連結する。
    let mut bytes = Vec::with_capacity(total * rb);
    for p in parts {
        bytes.extend_from_slice(p.blocks().as_slice());
    }
    if bytes.len() != total * rb {
        return None;
    }
    let inner = GgufLinear::new(ty, k, total, WeightBlocks::from_vec(bytes))
        .ok()?
        .with_fast_q8(fast_q8);
    let mut ranges = Vec::with_capacity(parts.len());
    let mut at = 0usize;
    for p in parts {
        let n = p.n();
        ranges.push(at..at + n);
        at += n;
    }
    Some((inner, ranges))
}

/// 全順列のうち、mmap 上で隣接して並ぶものを 1 つ返す (見つからなければ `None`)。
fn contiguous_order(parts: &[GgufLinear]) -> Option<Vec<usize>> {
    let mut idx: Vec<usize> = (0..parts.len()).collect();
    let mut found: Option<Vec<usize>> = None;
    permute_into_contiguous(&mut idx, 0, parts, &mut found);
    found
}

fn permute_into_contiguous(
    idx: &mut Vec<usize>,
    k: usize,
    parts: &[GgufLinear],
    found: &mut Option<Vec<usize>>,
) {
    if found.is_some() {
        return;
    }
    if k == idx.len() {
        if contiguous_in_order(parts, idx) {
            *found = Some(idx.clone());
        }
        return;
    }
    for i in k..idx.len() {
        idx.swap(k, i);
        permute_into_contiguous(idx, k + 1, parts, found);
        idx.swap(k, i);
    }
}

/// `order` の順に並べたとき、ブロックが同じ写像の連続した範囲になるか。
fn contiguous_in_order(parts: &[GgufLinear], order: &[usize]) -> bool {
    let mut prev: Option<(&std::sync::Arc<memmap2::Mmap>, usize)> = None;
    for &i in order {
        let Some((map, off, len)) = parts[i].blocks().mapping() else {
            return false;
        };
        match prev {
            None => prev = Some((map, off + len)),
            Some((pmap, end)) => {
                if !std::sync::Arc::ptr_eq(pmap, map) || off != end {
                    return false;
                }
                prev = Some((map, off + len));
            }
        }
    }
    true
}

/// 重みを捨てるときに VRAM の常駐も返す。
///
/// serve-flash は prefix が一致しないとモデルを組み直す (= 新しい uid を払い出す)。
/// この Drop が無いと古い重みが VRAM に残り、会話を重ねるたびに使用量が増えて
/// 最後は dense の確保に失敗して CPU に落ちる (decode が倍近く遅くなる症状)。
///
/// task 69: VRAM に上げた f32 重み (ルータ) も同じ uid で登録されるので、
/// ここで両方まとめて外す。レジストリは uid キーなので、モデルを差し替えて
/// 新しい uid の層が載ると古いエントリは `DENNO_GPU_WEIGHT_MB` の会計に残り続ける。
impl Drop for GgufLinear {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::gguf_unregister(self.uid);
            crate::cuda::unregister_f32_weights(self.uid);
            // task 110: hc mix BF16 展開重み (f32) の常駐も同じ uid で外す。
            crate::cuda::mix_gpu_unregister(self.uid);
        }
    }
}

#[cfg(test)]
mod tests {
    //! 実 GGUF はこのコンテナから見えないのでブロックはテスト内で組み立てる。
    //! 期待値は「全行列を f32 へ展開してから素直な逐次内積」を独立に書き下した
    //! 参照ループと `to_bits()` で比較する (融合実装のレイアウト検算)。

    use super::*;
    use crate::kquant::dequant_row;

    const K: usize = 256;
    const N: usize = 5;

    /// 決定的な擬似乱数バイト。
    fn pat(i: usize, seed: u32) -> u8 {
        let mut x = (i as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(seed.wrapping_mul(40_503));
        x ^= x >> 13;
        x = x.wrapping_mul(0x9E37_79B1);
        x ^= x >> 16;
        x as u8
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// 1 ブロックを組み立てる。スケール欄は有限値に固定する。
    fn make_block(ty: GgmlType, seed: u32) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b: Vec<u8> = (0..ts).map(|i| pat(i, seed)).collect();
        match ty {
            GgmlType::Q8_0 => b[0..2].copy_from_slice(&f16_bytes(0.125)),
            GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
                b[0..2].copy_from_slice(&f16_bytes(0.125));
                b[2..4].copy_from_slice(&f16_bytes(0.0625));
            }
            GgmlType::F32 => {
                for i in 0..(ts / 4) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            GgmlType::F16 | GgmlType::Bf16 => {
                for i in 0..(ts / 2) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    if ty == GgmlType::F16 {
                        b[i * 2..i * 2 + 2].copy_from_slice(&f16_bytes(v));
                    } else {
                        let bits = (v.to_bits() >> 16) as u16;
                        b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            _ => {}
        }
        b
    }

    fn make_matrix(ty: GgmlType, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0);
        let nb = k / bs;
        let mut out = Vec::with_capacity(n * nb * ts);
        for r in 0..n {
            for blk in 0..nb {
                out.extend_from_slice(&make_block(ty, seed.wrapping_add((r * 31 + blk) as u32)));
            }
        }
        out
    }

    fn make_x(k: usize, seed: u32) -> Vec<f32> {
        (0..k)
            .map(|i| ((pat(i, seed) as f32) / 128.0) - 1.0)
            .collect()
    }

    /// 独立参照: 全行列を f32 に展開してから素直な逐次内積。
    fn naive_gemv(ty: GgmlType, w: &[u8], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        let rb = (k / bs) * ts;
        let mut out = vec![0f32; n];
        for r in 0..n {
            let row = dequant_row(ty, &w[r * rb..(r + 1) * rb], k);
            let mut acc = 0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            out[r] = acc;
        }
        out
    }

    fn check(ty: GgmlType) {
        let w = make_matrix(ty, K, N, 7);
        let x = make_x(K, 11);
        let lin = GgufLinear::from_slice(ty, K, N, &w).expect("GgufLinear");
        assert_eq!(lin.k(), K);
        assert_eq!(lin.n(), N);
        assert_eq!(lin.ty(), ty);
        assert_eq!(lin.weight_bytes(), w.len());

        let mut got = vec![0f32; N];
        lin.gemv(&x, &mut got).expect("gemv");
        let want = naive_gemv(ty, &w, K, N, &x);
        for r in 0..N {
            assert_eq!(got[r].to_bits(), want[r].to_bits(), "{ty:?}: 行 {r}");
        }
    }

    #[test]
    fn q8_0_matches_naive() {
        check(GgmlType::Q8_0);
    }

    #[test]
    fn q5_1_matches_naive() {
        check(GgmlType::Q5_1);
    }

    #[test]
    fn q4_k_matches_naive() {
        check(GgmlType::Q4K);
    }

    #[test]
    fn q5_k_matches_naive() {
        check(GgmlType::Q5K);
    }

    #[test]
    fn f32_matches_naive() {
        check(GgmlType::F32);
    }

    #[test]
    fn f16_matches_naive() {
        check(GgmlType::F16);
    }

    #[test]
    fn bf16_matches_naive() {
        check(GgmlType::Bf16);
    }

    #[test]
    fn dequant_row_is_consistent_with_matrix_layout() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 3);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        for r in 0..N {
            let row = lin.dequant_row(r).unwrap();
            let bs = GgmlType::Q8_0.blck_size();
            let ts = GgmlType::Q8_0.type_size();
            let rb = (K / bs) * ts;
            let direct = dequant_row(GgmlType::Q8_0, &w[r * rb..(r + 1) * rb], K);
            assert_eq!(row, direct);
        }
        assert!(lin.dequant_row(N).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 5);
        // バイト数不足
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w[..w.len() - 1]).is_err());
        // 行数とバイト数の不一致
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N + 1, &w).is_err());
        // ブロック境界でない k
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K + 1, N, &w).is_err());

        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        let x = make_x(K, 1);
        let mut out = vec![0f32; N];
        assert!(lin.gemv(&x[..K - 1], &mut out).is_err());
        assert!(lin.gemv(&x, &mut out[..N - 1]).is_err());
    }

    #[test]
    fn zero_input_gives_zero() {
        let w = make_matrix(GgmlType::Q4K, K, N, 9);
        let lin = GgufLinear::from_slice(GgmlType::Q4K, K, N, &w).unwrap();
        let x = vec![0f32; K];
        let mut out = vec![1.0f32; N];
        lin.gemv(&x, &mut out).unwrap();
        for r in 0..N {
            assert_eq!(out[r], 0.0, "行 {r}");
        }
    }

    // ---- task 65: CPU GEMV の行並列 (`DENNO_CPU_GEMV_PAR=1`) ---------------
    //
    // 新経路は「行を rayon で分けるだけで、1 行の内積は i 昇順の逐次加算のまま」。
    // そのため旧経路 (逐次 / 既存の feature="par" の分割) と **f32 のビットで
    // 一致**する。ここでは実形状 4 本 (router / ssm_beta / ssm_alpha / hc_inject)
    // としきい値の境界を固定する。上書き・カウンタ・`par_test_lock` は
    // `super` (linear_gguf) 側にあり、crate 内の他のテストと共有する。

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct ParGuard;
    impl ParGuard {
        fn new() -> Self {
            set_cpu_gemv_par_for_test(None);
            set_par_min_rows_for_test(None);
            Self
        }
    }
    impl Drop for ParGuard {
        fn drop(&mut self) {
            set_cpu_gemv_par_for_test(None);
            set_par_min_rows_for_test(None);
        }
    }

    /// 実形状 (k, n) を作る。F32 は 1 要素 4 バイトなので k*4*n バイト。
    fn f32_lin(k: usize, n: usize, seed: u32) -> (GgufLinear, Vec<f32>) {
        let w = make_matrix(GgmlType::F32, k, n, seed);
        let lin = GgufLinear::from_slice(GgmlType::F32, k, n, &w).expect("GgufLinear");
        (lin, make_x(k, seed.wrapping_add(1)))
    }

    #[test]
    fn cpu_gemv_par_env_values() {
        // 既定 on。`0` / `off` / `false` のときだけ無効。
        for v in [None, Some("1"), Some("true"), Some("")] {
            assert!(cpu_gemv_par_from_value(v), "value={v:?}");
        }
        for v in [Some("0"), Some("off"), Some("false")] {
            assert!(!cpu_gemv_par_from_value(v), "value={v:?}");
        }
    }

    // ---- task 69: ルータ GPU 経路 (`DENNO_ROUTER_GPU`) ---------------------

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct RouterGuard;
    impl RouterGuard {
        fn new() -> Self {
            set_router_gpu_for_test(None);
            Self
        }
    }
    impl Drop for RouterGuard {
        fn drop(&mut self) {
            set_router_gpu_for_test(None);
        }
    }

    /// task 69: `DENNO_ROUTER_GPU` の真理値表。既定 on、明示的な偽値だけ off。
    #[test]
    fn router_gpu_gate_defaults() {
        for v in [
            Some("1"),
            Some("on"),
            Some("ON"),
            Some("true"),
            Some("TRUE"),
            Some(" on "),
        ] {
            assert!(router_gpu_from_value(v), "value={v:?} は on のはず");
        }
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("false"),
            Some("no"),
            Some("OFF"),
            Some("False"),
        ] {
            assert!(!router_gpu_from_value(v), "value={v:?} は off のはず");
        }
        // テスト用上書きが実際に効くこと (`None` で env 読みに戻る)。
        let _g = par_test_lock();
        let _p = RouterGuard::new();
        set_router_gpu_for_test(Some(false));
        assert!(!router_gpu_enabled(), "off の上書きが効いていない");
        set_router_gpu_for_test(Some(true));
        assert!(router_gpu_enabled(), "on の上書きが効いていない");
    }

    /// task 69: ルータ扱いは層ごとの opt-in。既定は false (他の F32 線形は CPU のまま)。
    #[test]
    fn router_gpu_flag_is_opt_in_per_linear() {
        let (lin, _x) = f32_lin(2560, 512, 3);
        assert!(!lin.router_gpu(), "既定でルータ扱いになっている");
        let lin = lin.with_router_gpu(true);
        assert!(lin.router_gpu());
    }

    /// 実形状 4 本で旧経路と新経路がビット一致し、期待どおりの側に落ちること。
    #[test]
    fn cpu_gemv_par_bitwise_on_real_shapes() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        // (k, n, seed, 名前, 並列化されるか)
        let cases = [
            (2560usize, 512usize, 11u32, "router (ffn_gate_inp)", true),
            (2560, 48, 13, "ssm_beta / ssm_alpha", true),
            (10240, 4, 17, "hc_inject", true),
            (10240, 32, 19, "hc_up/inject (k=10240, 32 行)", true),
            (2560, 4, 47, "しきい値未満 (2560->4)", false),
        ];
        for (k, n, seed, name, want_par) in cases {
            let (lin, x) = f32_lin(k, n, seed);

            // 旧: env off (既定)。既存の feature="par" の分割はそのまま通る。
            set_cpu_gemv_par_for_test(Some(false));
            par_reset();
            let mut old = vec![-1.0e30f32; n];
            lin.gemv(&x, &mut old).expect(name);
            assert_eq!(par_launches(), 0, "{name}: off で新経路が立った");

            // 新: env on。
            set_cpu_gemv_par_for_test(Some(true));
            par_reset();
            let mut new = vec![1.0e30f32; n];
            lin.gemv(&x, &mut new).expect(name);
            let launches = par_launches();
            if want_par {
                assert_eq!(launches, 1, "{name}: 並列領域が 1 回でない");
                // 行ごとに領域を立てていないこと (チャンク数 <= 行数)。
                let tasks = par_tasks();
                assert!(tasks >= 1 && tasks <= n, "{name}: tasks={tasks} n={n}");
                if n >= 32 {
                    assert!(tasks < n, "{name}: 行ごとにタスクを立てている tasks={tasks}");
                }
            } else {
                assert_eq!(launches, 0, "{name}: しきい値未満で並列化された");
            }

            // ビット一致 (f32)。
            for r in 0..n {
                assert_eq!(
                    old[r].to_bits(),
                    new[r].to_bits(),
                    "{name}: 行 {r} が違う (old={} new={})",
                    old[r],
                    new[r]
                );
            }
            // どちらも独立参照 (全行 f32 展開 + 素直な逐次内積) と一致する。
            let w = lin.blocks().as_slice();
            let want = naive_gemv(GgmlType::F32, w, k, n, &x);
            for r in 0..n {
                assert_eq!(new[r].to_bits(), want[r].to_bits(), "{name}: 参照と違う 行 {r}");
            }
        }
    }

    /// 1 行だけの形は並列化しない側に落ち、値は逐次と一致する。
    #[test]
    fn cpu_gemv_par_skips_single_row() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 1usize);
        let (lin, x) = f32_lin(k, n, 23);
        set_cpu_gemv_par_for_test(Some(true));
        par_reset();
        let mut got = vec![0f32; n];
        lin.gemv(&x, &mut got).expect("gemv");
        assert_eq!(par_launches(), 0, "1 行で並列領域を立てた");
        let want = naive_gemv(GgmlType::F32, lin.blocks().as_slice(), k, n, &x);
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }

    /// しきい値の境界。行数の下限 (8) と積和の下限 (32768) の両方を見る。
    #[test]
    fn cpu_gemv_par_threshold_boundary() {
        // 純粋関数の側 (起動カウンタを立てずに境界を固定する)。
        set_par_min_rows_for_test(None);
        // 行数 >= 8 は無条件。
        assert!(cpu_gemv_par_applies(8, 2560));
        assert!(!cpu_gemv_par_applies(7, 2560));
        // 行数 < 8 でも積和 >= 32768 なら並列 (hc_inject 4×10240 = 40960)。
        assert!(cpu_gemv_par_applies(4, 10240));
        assert!(!cpu_gemv_par_applies(3, 10240)); // 30720 < 32768
        assert!(!cpu_gemv_par_applies(4, 2560)); // 10240 < 32768
        assert!(!cpu_gemv_par_applies(2, 10240)); // 20480 < 32768 (実測でも損)

        let _g = par_test_lock();
        let _p = ParGuard::new();
        let k = 2560usize;
        set_cpu_gemv_par_for_test(Some(true));

        let n_below = CPU_GEMV_PAR_MIN_ROWS - 1;
        let (lin, x) = f32_lin(k, n_below, 29);
        par_reset();
        let mut out = vec![0f32; n_below];
        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 0, "{n_below} 行で並列化した");

        let (lin, x) = f32_lin(k, CPU_GEMV_PAR_MIN_ROWS, 31);
        par_reset();
        let mut out = vec![0f32; CPU_GEMV_PAR_MIN_ROWS];
        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 1, "しきい値ちょうどで並列化しない");
    }

    /// 1 呼び出し = 1 並列領域。行数ぶん起動しないこと (連続 2 回で 2)。
    #[test]
    fn cpu_gemv_par_one_region_per_call() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 512usize);
        let (lin, x) = f32_lin(k, n, 37);
        set_cpu_gemv_par_for_test(Some(true));
        let mut out = vec![0f32; n];

        par_reset();
        lin.gemv(&x, &mut out).expect("gemv");
        let tasks_one = par_tasks();
        assert_eq!(par_launches(), 1, "1 呼び出しで {tasks_one} タスク / 領域 1 のはず");
        assert!(tasks_one < n, "行ごとにタスクを立てている (tasks={tasks_one})");
        let threads = rayon::current_num_threads().max(1);
        let rows = n.div_ceil(threads).max(1);
        assert_eq!(tasks_one, n.div_ceil(rows), "分割数が想定と違う");

        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 2, "2 回目で領域が 2 にならない");
    }

    /// task 106: F32 の行並列 GEMV は direct 経路 (行バッファ無し) を通る。
    ///
    /// 「変更経路が発動した」ことの証拠 (カウンタ)。F16 (copy 経路) と
    /// env off (旧経路 = P-MOE-G11 の copy 分割) では増えないことも固定する。
    #[test]
    fn cpu_gemv_par_f32_takes_direct_branch() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (lin, x) = f32_lin(2560, 512, 11);
        set_cpu_gemv_par_for_test(Some(true));
        par_reset();
        f32_direct_reset();
        let mut out = vec![0f32; 512];
        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 1, "行並列が立っていない");
        assert!(
            f32_direct_rows() >= 512,
            "F32 direct 経路が発火していない (rows={})",
            f32_direct_rows()
        );

        // F16 は copy 経路のまま (direct は F32 だけ)。
        let w16 = make_matrix(GgmlType::F16, 2560, 512, 13);
        let lin16 = GgufLinear::from_slice(GgmlType::F16, 2560, 512, &w16).expect("f16");
        let x16 = make_x(2560, 14);
        f32_direct_reset();
        let mut out16 = vec![0f32; 512];
        lin16.gemv(&x16, &mut out16).expect("f16 gemv");
        assert_eq!(f32_direct_rows(), 0, "F16 で F32 direct が発火した");

        // env off では旧経路 (copy 分割) のまま。
        set_cpu_gemv_par_for_test(Some(false));
        f32_direct_reset();
        let mut out2 = vec![0f32; 512];
        lin.gemv(&x, &mut out2).expect("gemv off");
        assert_eq!(f32_direct_rows(), 0, "env off で F32 direct が発火した");
    }

    /// off (既定) では新経路が発動しないこと (従来経路そのまま)。
    #[test]
    fn cpu_gemv_par_off_keeps_old_path() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 512usize);
        let (lin, x) = f32_lin(k, n, 41);
        set_cpu_gemv_par_for_test(Some(false));
        par_reset();
        let mut got = vec![0f32; n];
        lin.gemv(&x, &mut got).expect("gemv");
        assert_eq!(par_launches(), 0);
        let want = naive_gemv(GgmlType::F32, lin.blocks().as_slice(), k, n, &x);
        for r in 0..n {
            assert_eq!(got[r].to_bits(), want[r].to_bits(), "行 {r}");
        }
    }

    /// 旧/新の最小時間を交互に取る (片方だけ先に回すとキャッシュ状態が揃わない)。
    /// 出力バッファはここで持つ (両方の closure が同じ `out` を借りるため)。
    fn best_pair(
        reps: usize,
        n: usize,
        mut old: impl FnMut(&mut [f32]),
        mut new: impl FnMut(&mut [f32]),
    ) -> (f64, f64) {
        let mut out = vec![0f32; n];
        let (mut bo, mut bn) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            old(&mut out);
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            new(&mut out);
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (bo, bn)
    }

    /// 実形状 4 本の 1 呼び出しを 100 回ずつ、旧 (off) / 新 (on) で測る。
    ///
    /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --lib -- \
    ///  --ignored --nocapture bench_cpu_gemv_par_real_shapes`
    #[test]
    #[ignore = "実形状の CPU マイクロベンチ (手動)"]
    fn bench_cpu_gemv_par_real_shapes() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 100usize;
        println!(
            "\n=== CPU GEMV 行並列: 実形状 1 呼び出し (reps={reps}, rayon threads={}) ===",
            rayon::current_num_threads()
        );
        println!(
            "{:<34} {:>9} {:>9} {:>8} {:>10} {:>6}",
            "shape", "old_ms", "new_ms", "gain", "MAC", "GMAC/s"
        );
        // (k, n, seed, 名前) — 実モデルの 4 形 (router / ssm_beta / ssm_alpha / hc_inject)
        for (k, n, seed, name) in [
            (2560usize, 512usize, 11u32, "router ffn_gate_inp 2560->512"),
            (2560, 48, 13, "ssm_beta/alpha 2560->48"),
            (10240, 4, 17, "hc_inject 10240->4"),
            (2560, 2048, 43, "参考: 2560->2048"),
        ] {
            let (lin, x) = f32_lin(k, n, seed);
            set_cpu_gemv_par_for_test(Some(false));
            let (old_ms, new_ms) = best_pair(
                reps,
                n,
                |out| {
                    lin.gemv(&x, out).expect("old");
                    std::hint::black_box(out);
                },
                |out| {
                    set_cpu_gemv_par_for_test(Some(true));
                    lin.gemv(&x, out).expect("new");
                    std::hint::black_box(out);
                    set_cpu_gemv_par_for_test(Some(false));
                },
            );
            let us = old_ms * 1e3;
            let mac = (k * n) as f64;
            println!(
                "{:<34} {:>9.3} {:>9.3} {:>7.2}x {:>10.0} {:>6.1}",
                name,
                us,
                new_ms * 1e3,
                if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                mac,
                mac / (new_ms * 1e6)
            );
        }
    }

    /// task 106: decode の router 周り (gate GEMV F32 + softmax + top-k) の内訳。
    ///
    /// gate GEMV は「task 65 の copy 本体」(`dequant_row_into` で行バッファに開いて
    /// から内積) と「task 106 の direct 本体」(重みバイトから `from_le_bytes` で
    /// 直接内積) を**同一実行内で**交互に測る。両者は `naive_gemv` (独立参照) と
    /// `to_bits()` で一致することもここで固定する。
    ///
    /// softmax / select_topk は実形 (ne=512, k=10) の 1 呼び出し。
    ///
    /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --lib --features cuda \
    ///   -- --ignored --nocapture --test-threads=1 bench_decode_router_parts`
    #[test]
    #[ignore = "実形状の CPU マイクロベンチ (手動)"]
    fn bench_decode_router_parts() {
        use crate::model::moe::select_topk;
        use crate::tensor::softmax;

        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 100usize;
        println!(
            "\n=== decode router 部品 (reps={reps}, rayon threads={}) ===",
            rayon::current_num_threads()
        );

        // 旧 copy 本体 (task 65 の gemv_rows_par から F32 部だけ取り出し)。
        fn rows_par_copy(blocks: &[u8], rb: usize, k: usize, x: &[f32], out: &mut [f32]) {
            use rayon::prelude::*;
            let threads = rayon::current_num_threads().max(1);
            let rows = out.len().div_ceil(threads).max(1).min(out.len());
            let bytes = rows * rb;
            blocks
                .par_chunks(bytes)
                .zip(out.par_chunks_mut(rows))
                .for_each(|(bb, oo)| {
                    let mut row = vec![0f32; k];
                    for (r, slot) in oo.iter_mut().enumerate() {
                        crate::kquant::dequant_row_into(GgmlType::F32, &bb[r * rb..(r + 1) * rb], &mut row);
                        let mut acc = 0f32;
                        for i in 0..k {
                            acc += row[i] * x[i];
                        }
                        *slot = acc;
                    }
                });
        }

        for (k, n, seed, name) in [
            (2560usize, 512usize, 11u32, "gate_inp (router) 2560->512"),
            (10240, 4, 17, "hc_inject 10240->4"),
            (2560, 48, 13, "ssm_beta/alpha 2560->48"),
        ] {
            let (lin, x) = f32_lin(k, n, seed);
            let blocks = lin.blocks().as_slice();
            let rb = k * 4;
            set_cpu_gemv_par_for_test(Some(true));
            let (old_ms, new_ms) = best_pair(
                reps,
                n,
                |out| {
                    rows_par_copy(blocks, rb, k, &x, out);
                    std::hint::black_box(out);
                },
                |out| {
                    lin.gemv(&x, out).expect("new (direct)");
                    std::hint::black_box(out);
                },
            );
            set_cpu_gemv_par_for_test(Some(false));
            // ビット一致 (copy 版 / direct 版 / 独立参照の 3 点)。
            let mut a = vec![0f32; n];
            rows_par_copy(blocks, rb, k, &x, &mut a);
            let mut b = vec![0f32; n];
            set_cpu_gemv_par_for_test(Some(true));
            lin.gemv(&x, &mut b).expect("gemv");
            set_cpu_gemv_par_for_test(Some(false));
            let w = naive_gemv(GgmlType::F32, blocks, k, n, &x);
            for r in 0..n {
                assert_eq!(a[r].to_bits(), w[r].to_bits(), "{name}: copy 版が参照と違う 行 {r}");
                assert_eq!(b[r].to_bits(), w[r].to_bits(), "{name}: direct 版が参照と違う 行 {r}");
            }
            let mb = blocks.len() as f64 / 1e6;
            println!(
                "{:<28} copy {:>8.3} us / direct {:>8.3} us  ({:.2}x, {:.1} MB)",
                name,
                old_ms * 1e3,
                new_ms * 1e3,
                if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                mb
            );
        }

        // softmax 1 行 (ne=512)。実形の logits は N(0, 3.4) 程度 (moe_router_microbench と同じ入力)。
        let ne = 512usize;
        let probs: Vec<f32> = (0..ne)
            .map(|i| {
                let r = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((r >> 33) % 2001) as f32 - 1000.0) * 6e-3
            })
            .collect();
        let mut p = probs.clone();
        let ms = best_pair(reps, 1, |_| {}, |out| {
            p.copy_from_slice(&probs);
            softmax(&mut p);
            std::hint::black_box(&p);
            let _ = out;
        })
        .1;
        println!("softmax ne=512              : {:>8.3} us / 呼び出し", ms * 1e3);

        // select_topk 1 回 (ne=512, k=10)。probs は softmax 済みの値を使う。
        softmax(&mut p);
        let k_used = 10usize;
        let (_, topk_ms) = best_pair(reps, 1, |_| {}, |out| {
            let sel = select_topk(&p, k_used);
            std::hint::black_box(&sel);
            let _ = out;
        });
        println!(
            "select_topk ne=512 k=10     : {:>8.3} us / 呼び出し (現行・確保あり)",
            topk_ms * 1e3
        );
        assert_eq!(select_topk(&p, k_used).len(), k_used);
    }

    /// しきい値の根拠: 行数を 1..64 で振り、旧 (逐次寄り) と新 (行並列) を測る。
    /// 新は**しきい値を 1 に強制**して、小さい形でも実際に並列を立てた値を取る
    /// (しきい値の損得はこの表で決める)。
    ///
    /// `RAYON_NUM_THREADS=8 cargo test --release -p denno-engine --lib -- \
    ///  --ignored --nocapture bench_cpu_gemv_par_threshold_sweep`
    #[test]
    #[ignore = "しきい値の根拠を取る CPU マイクロベンチ (手動)"]
    fn bench_cpu_gemv_par_threshold_sweep() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 50usize;
        println!(
            "\n=== 行並列のしきい値 (reps={reps}, rayon threads={}) ===",
            rayon::current_num_threads()
        );
        println!(
            "{:<6} {:>7} {:>10} {:>10} {:>8} {:>9} {:>9}",
            "k", "rows", "old_us", "new_us", "gain", "old_MAC/us", "new_MAC/us"
        );
        // 新はしきい値 1 (= 何行でも並列領域を立てる) で測る。
        set_par_min_rows_for_test(Some(1));
        for k in [2560usize, 10240] {
            for n in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] {
                let (lin, x) = f32_lin(k, n, 101 + n as u32);
                set_cpu_gemv_par_for_test(Some(false));
                let (old_ms, new_ms) = best_pair(
                    reps,
                    n,
                    |out| {
                        lin.gemv(&x, out).expect("old");
                        std::hint::black_box(out);
                    },
                    |out| {
                        set_cpu_gemv_par_for_test(Some(true));
                        lin.gemv(&x, out).expect("new");
                        std::hint::black_box(out);
                        set_cpu_gemv_par_for_test(Some(false));
                    },
                );
                let mac = (k * n) as f64;
                println!(
                    "{:<6} {:>7} {:>10.2} {:>10.2} {:>7.2}x {:>9.2} {:>9.2}",
                    k,
                    n,
                    old_ms * 1e3,
                    new_ms * 1e3,
                    if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                    mac / (old_ms * 1e6),
                    mac / (new_ms * 1e6)
                );
            }
        }
    }

    // ---- task 91: BF16 (hc down/up の実型) の行並列 GEMV ---------------------

    /// BF16 の実形 2 形 (hc down 10240->320 / hc up 320->10240) を 1/2/4/8 スレッドで、
    /// 旧経路 ([`GgufLinear::gemv`] = `cpu_gemv_par` の行チャンク) と
    /// 新経路 ([`GgufLinear::gemv_label_par_2row`] = `for_each_row_chunk` の行チャンク)
    /// で測る。実機の decode 1 呼び出しは down / up 各 1 発 (6.55 MB / 発)。
    ///
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_bf16_hc_real_shape`
    #[test]
    #[ignore = "実形状の CPU マイクロベンチ (手動)"]
    fn bench_bf16_hc_real_shape_rowpar() {
        use crate::model::qwen4exp::HC_DOWN_ROW_PAR_MIN;
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 30usize;
        println!("\n=== BF16 hc GEMV 実形 (reps={reps}) ===");
        for (k, n, min, name) in [
            (10240usize, 320usize, HC_DOWN_ROW_PAR_MIN, "hc_down 10240->320"),
            (320usize, 10240usize, 0usize, "hc_up 320->10240"),
        ] {
            let w = make_matrix(GgmlType::Bf16, k, n, 11);
            let lin = GgufLinear::from_slice(GgmlType::Bf16, k, n, &w).expect("GgufLinear");
            let x = make_x(k, 13);
            let bytes = (k * n * 2) as f64; // BF16 は 1 要素 2 B
            for th in [1usize, 2, 4, 8] {
                let pool =
                    rayon::ThreadPoolBuilder::new().num_threads(th).build().expect("thread pool");
                let (old_ms, new_ms) = pool.install(|| {
                    // warm-up (スレッドプールとページを立ち上げる)
                    let mut warm = vec![0f32; n];
                    for _ in 0..3 {
                        lin.gemv(&x, &mut warm).expect("old");
                        lin.gemv_label_par_2row("bench_bf16", &x, &mut warm, min).expect("new");
                    }
                    best_pair(
                        reps,
                        n,
                        |out| {
                            lin.gemv(&x, out).expect("old");
                            std::hint::black_box(out);
                        },
                        |out| {
                            lin.gemv_label_par_2row("bench_bf16", &x, out, min).expect("new");
                            std::hint::black_box(out);
                        },
                    )
                });
                println!(
                    "{:<22} threads={th}  old={old_ms:.3} ms ({:.1} GB/s)  \
                     new={new_ms:.3} ms ({:.1} GB/s)  gain {:.2}x",
                    name,
                    bytes / (old_ms * 1e6),
                    bytes / (new_ms * 1e6),
                    if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                );
            }
        }
    }

    // ---- task 93: DENNO_BF16_CHUNK (BF16 融合経路の目標チャンク数) -----------

    /// `DENNO_BF16_CHUNK` が値を変えないこと (実形 2 形)。
    ///
    /// 行は互いに独立で、1 行の累積順序は行の割り方に依らない。だから自動分割
    /// (rayon スレッド数から決まる) も、N=1 (1 チャンク) も、N=3/7 の割り切れない
    /// 端数も、N=64 の細かいチャンクも、classic (融合 off・scalar body + 行チャンク)
    /// とビット一致するはず。env 掃引の前にここで固定する。
    #[test]
    fn bf16_chunk_override_bit_matches_auto() {
        use crate::model::moe_expert::{set_bf16_chunk_for_test, set_bf16_fast_for_test};
        use crate::model::qwen4exp::HC_DOWN_ROW_PAR_MIN;
        let _g = par_test_lock();
        for (k, n, min, name) in [
            (10240usize, 320usize, HC_DOWN_ROW_PAR_MIN, "hc_down 10240->320"),
            (320usize, 10240usize, 0usize, "hc_up 320->10240"),
        ] {
            let w = make_matrix(GgmlType::Bf16, k, n, 7);
            let lin = GgufLinear::from_slice(GgmlType::Bf16, k, n, &w).expect("GgufLinear");
            let x = make_x(k, 9);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();

            // classic: 融合 off (scalar body を行チャンクで回る退避路)。
            set_bf16_fast_for_test(Some(false));
            let mut classic = vec![0f32; n];
            lin.gemv_label_par_2row("bit", &x, &mut classic, min)
                .unwrap_or_else(|e| panic!("{name}: classic: {e}"));
            set_bf16_fast_for_test(None);

            // 融合 + 自動分割。
            set_bf16_chunk_for_test(None);
            let mut auto = vec![0f32; n];
            lin.gemv_label_par_2row("bit", &x, &mut auto, min)
                .unwrap_or_else(|e| panic!("{name}: auto: {e}"));
            assert_eq!(
                bits(&classic),
                bits(&auto),
                "{name}: 融合 (自動分割) が classic とビット不一致"
            );

            // 融合 + 明示チャンク数 (端数が出る N も含む)。
            for chunk in [1usize, 3, 7, 64] {
                set_bf16_chunk_for_test(Some(chunk));
                let mut got = vec![0f32; n];
                lin.gemv_label_par_2row("bit", &x, &mut got, min)
                    .unwrap_or_else(|e| panic!("{name}: chunk={chunk}: {e}"));
                set_bf16_chunk_for_test(None);
                assert_eq!(
                    bits(&classic),
                    bits(&got),
                    "{name}: chunk={chunk} が classic とビット不一致"
                );
            }
        }
    }

    /// task 107: BF16 融合カーネルの行グループ幅 (`DENNO_BF16_GROUP`) と
    /// ソフトウェアプリフェッチ距離 (`DENNO_BF16_PF`) が値を変えないこと
    /// (実形 2 形)。classic (融合 off) と to_bits で突き合わせる。
    ///
    /// 根拠: 行は互いに独立で、1 行の累積は (カーネルの c 昇順・レーン順の
    /// 逐次加算) 不変。グループ化は行をまたぐ演算を作らず、prefetch は
    /// キャッシュに読むだけ (演算を足さない)。チャンク分割との組合せ
    /// (端数チャンク × グループ幅) は `bf16_chunk_override_bit_matches_auto`
    /// が既定幾何のままカバーする。
    ///
    /// このテストは `row_par_min = usize::MAX` で**逐次**に固定する。行並列を
    /// 発火させると [`crate::model::moe_expert::ROW_PAR_REGIONS`] (global) が
    /// 増え、同じカウンタの差分を見る並行テスト
    /// (`expert_loop_one_region_per_layer` 等) の邪魔になるため。
    /// classic 参照も global フラグ (`BF16_FAST_OVERRIDE`) を立てず、
    /// dequant + 逐次内積を直に計算する (並行テストの経路を変えない)。
    #[test]
    fn bf16_group_pf_bit_matches_classic() {
        use crate::model::moe_expert::{set_bf16_group_for_test, set_bf16_pf_for_test};
        let _g = par_test_lock();
        let n_max = usize::MAX;
        for (k, n, name) in [
            (10240usize, 320usize, "hc_down 10240->320"),
            (320usize, 10240usize, "hc_up 320->10240"),
            (64usize, 10usize, "small n=10 (グループ端数)"),
            (64usize, 7usize, "small n=7 (端数のみ)"),
        ] {
            let w = make_matrix(GgmlType::Bf16, k, n, 11);
            let lin = GgufLinear::from_slice(GgmlType::Bf16, k, n, &w).expect("GgufLinear");
            let x = make_x(k, 13);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();

            // classic 参照: `gemv_scalar_body` の 1 行の定義そのもの
            // (dequant_row_into -> i 昇順の逐次加算)。global フラグは触らない。
            let rb = k * 2;
            let mut row = vec![0f32; k];
            let mut classic = vec![0f32; n];
            for (r, slot) in classic.iter_mut().enumerate() {
                crate::kquant::dequant_row_into(GgmlType::Bf16, &w[r * rb..(r + 1) * rb], &mut row);
                let mut acc = 0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                *slot = acc;
            }

            for g in [1usize, 2, 4, 8] {
                for pf in [0usize, 8, 88] {
                    set_bf16_group_for_test(Some(g));
                    set_bf16_pf_for_test(Some(pf));
                    let mut got = vec![0f32; n];
                    lin.gemv_label_par_2row("bit", &x, &mut got, n_max)
                        .unwrap_or_else(|e| panic!("{name}: group={g} pf={pf}: {e}"));
                    set_bf16_group_for_test(None);
                    set_bf16_pf_for_test(None);
                    assert_eq!(
                        bits(&classic),
                        bits(&got),
                        "{name}: group={g} pf={pf} が classic とビット不一致"
                    );
                }
            }
        }
    }

    /// task 93: BF16 融合 GEMV のチャンク数掃引 (`DENNO_BF16_CHUNK` の A/B 用)。
    ///
    /// 実機の乖離 (コンテナ 1 スレッド 0.107 ms vs 実機 8 スレッドで 0.238 ms) の
    /// うち「チャンク粒度」と「スレッド取り合い」を切り分ける。実機は呼び出しごとに
    /// 別の expert (6.55 MB) を読むので L3 に載らない — `mats` 本の独立な行列を
    /// 轮回して DRAM 条件を作る (`mats=1` は同じ行列の再読み = L3 常駐条件で、
    /// コンテナの 0.107 ms の条件)。`chunk=auto` は従来動作
    /// (1 チャンクの行数 = n_out / rayon スレッド数)。
    ///
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_bf16_chunk_sweep`
    #[test]
    #[ignore = "実形状の CPU マイクロベンチ (手動)"]
    fn bench_bf16_chunk_sweep() {
        use crate::model::moe_expert::set_bf16_chunk_for_test;
        use crate::model::qwen4exp::mix_kprof;
        use crate::model::qwen4exp::HC_DOWN_ROW_PAR_MIN;
        let _g = par_test_lock();
        let reps = 24usize;
        let bytes = 6_553_600f64; // 320 * 10240 * 2 (down / up とも同サイズ)
        for (k, n, min, name, idx) in [
            (10240usize, 320usize, HC_DOWN_ROW_PAR_MIN, "hc_down 10240->320", 0usize),
            (320usize, 10240usize, 0usize, "hc_up 320->10240", 1usize),
        ] {
            let cpus = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0);
            println!("\n=== {name} (reps={reps}, cpus={cpus}) ===");
            for mats in [1usize, 32] {
                let lins: Vec<GgufLinear> = (0..mats)
                    .map(|m| {
                        let w = make_matrix(GgmlType::Bf16, k, n, (m as u32) * 17 + 3);
                        GgufLinear::from_slice(GgmlType::Bf16, k, n, &w).expect("GgufLinear")
                    })
                    .collect();
                let x = make_x(k, 13);
                for th in [1usize, 8, 16] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(th)
                        .build()
                        .expect("pool");
                    pool.install(|| {
                        for chunk in [0usize, 1, 2, 4, 8, 16, 32] {
                            set_bf16_chunk_for_test(if chunk == 0 { None } else { Some(chunk) });
                            let mut out = vec![0f32; n];
                            for w in 0..4 {
                                lins[w % mats]
                                    .gemv_label_par_2row("sweep", &x, &mut out, min)
                                    .expect("gemv");
                            }
                            // rowpar カウンタで chunks / wall / comp / par_eff を取る。
                            mix_kprof::set_for_test(true);
                            mix_kprof::reset();
                            mix_kprof::rowpar::set_label(if idx == 0 {
                                mix_kprof::rowpar::L_DOWN
                            } else {
                                mix_kprof::rowpar::L_UP
                            });
                            let mut best = f64::INFINITY;
                            for r in 0..reps {
                                let t = std::time::Instant::now();
                                lins[r % mats]
                                    .gemv_label_par_2row("sweep", &x, &mut out, min)
                                    .expect("gemv");
                                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                            }
                            std::hint::black_box(&out);
                            mix_kprof::rowpar::set_label(mix_kprof::rowpar::NONE);
                            let (regions, fired, chunks, wall_ns, comp_ns) =
                                mix_kprof::rowpar::stats(idx);
                            let regions = regions.max(1);
                            let wall_us = wall_ns as f64 / regions as f64 / 1e3;
                            let comp_us = comp_ns as f64 / regions as f64 / 1e3;
                            let par_eff = if wall_ns > 0 {
                                comp_ns as f64 / wall_ns as f64
                            } else {
                                1.0
                            };
                            mix_kprof::reset();
                            mix_kprof::set_for_test(false);
                            set_bf16_chunk_for_test(None);
                            println!(
                                "mats={mats:>2} th={th:>2} chunk={:>4}  best={best:.3} ms \
                                 ({:.1} GB/s)  rowpar regions={regions} fired={fired} \
                                 chunks={:.1} wall={wall_us:.0}us comp={comp_us:.0}us \
                                 par_eff={par_eff:.1}",
                                if chunk == 0 {
                                    "auto".to_string()
                                } else {
                                    chunk.to_string()
                                },
                                bytes / (best * 1e6),
                                chunks as f64 / fired.max(1) as f64,
                            );
                        }
                    });
                }
            }
        }
    }

    // ---- task 72: FusedLinear (同じ入力を取る投影の連結) ---------------------

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct FusedGuard;
    impl FusedGuard {
        fn new() -> Self {
            set_fused_gemv_for_test(None);
            Self
        }
    }
    impl Drop for FusedGuard {
        fn drop(&mut self) {
            set_fused_gemv_for_test(None);
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 行数だけ違う層を何本か作る (ヒープ実体)。
    fn parts_of(ty: GgmlType, k: usize, ns: &[usize], seed: u32) -> Vec<GgufLinear> {
        ns.iter()
            .enumerate()
            .map(|(i, &n)| {
                let w = make_matrix(ty, k, n, seed.wrapping_add(i as u32 * 37));
                GgufLinear::from_slice(ty, k, n, &w).expect("GgufLinear")
            })
            .collect()
    }

    /// `DENNO_FUSED_GEMV` の真理値表。既定 on、明示的な偽値だけ off。
    #[test]
    fn fused_gemv_gate_defaults() {
        for v in [None, Some("1"), Some("on"), Some("ON"), Some("true"), Some(" on "), Some("")] {
            assert!(fused_gemv_from_value(v), "value={v:?} は on のはず");
        }
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some("OFF"), Some("False")] {
            assert!(!fused_gemv_from_value(v), "value={v:?} は off のはず");
        }
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(false));
        assert!(!fused_gemv_enabled(), "off の上書きが効いていない");
        set_fused_gemv_for_test(Some(true));
        assert!(fused_gemv_enabled(), "on の上書きが効いていない");
    }

    /// 連結版と個別版の出力が `to_bits()` で一致する (タスクの指定テスト)。
    ///
    /// 行は独立・1 行の内積は `i` 昇順の逐次加算のままなので、連結しても
    /// 丸め順序は変わらない。全量子型 + F32/F16/Bf16 で固定する。
    #[test]
    fn fused_split_matches_separate_bits() {
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(true));
        let k = 256usize;
        let ns = [5usize, 3, 2];
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_1,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
        ] {
            let x = make_x(k, 61);
            let parts = parts_of(ty, k, &ns, 7);
            // 個別版 (従来の呼び出し) の期待値。
            let mut want: Vec<Vec<f32>> = Vec::new();
            for (p, &n) in parts.iter().zip(ns.iter()) {
                let mut o = vec![0f32; n];
                p.gemv(&x, &mut o).expect("gemv (個別)");
                want.push(o);
            }
            // 連結版。
            let fused = FusedLinear::concat(parts);
            assert!(fused.is_fused(), "{ty:?}: 連結されていない");
            assert_eq!(fused.n(), ns.iter().sum::<usize>(), "{ty:?}: 合計行数");
            let mut buf = vec![0f32; fused.n()];
            let mut outs: Vec<Vec<f32>> = ns.iter().map(|&n| vec![0f32; n]).collect();
            {
                let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|v| &mut v[..]).collect();
                fused.gemv_split("t", &x, &mut buf, &mut refs).expect("gemv_split");
            }
            for i in 0..ns.len() {
                assert_eq!(bits(&outs[i]), bits(&want[i]), "{ty:?}: 層 {i} がビット不一致");
            }
            // 連結バッファの側も同じ (切り分け位置の検算)。
            for i in 0..ns.len() {
                assert_eq!(
                    bits(&buf[fused.range(i)]),
                    bits(&want[i]),
                    "{ty:?}: 連結バッファの層 {i}"
                );
            }
        }
    }

    /// 連結できない (型・k が違う) ときは個別のまま落ち、値は従来どおり。
    #[test]
    fn fused_falls_back_to_separate_on_type_mismatch() {
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(true));
        let k = 256usize;
        let a = GgufLinear::from_slice(GgmlType::Q8_0, k, 3, &make_matrix(GgmlType::Q8_0, k, 3, 3))
            .expect("Q8_0");
        let b = GgufLinear::from_slice(GgmlType::F32, k, 2, &make_matrix(GgmlType::F32, k, 2, 5))
            .expect("F32");
        let fused = FusedLinear::concat(vec![a, b]);
        assert!(!fused.is_fused(), "型が違うのに連結した");
        assert_eq!(fused.splits(), &[3, 5]);
        assert_eq!(fused.range(1), 3..5);

        let x = make_x(k, 9);
        let mut outs = [vec![0f32; 3], vec![0f32; 2]];
        {
            let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|v| &mut v[..]).collect();
            let mut buf = vec![0f32; fused.n()];
            fused.gemv_split("t", &x, &mut buf, &mut refs).expect("gemv_split");
        }
        let mut buf = vec![0f32; fused.n()];
        fused.gemv_concat_into("t", &x, &mut buf).expect("gemv_concat_into");
        // 個別版と同じ (Q8_0 の 3 行 + F32 の 2 行)。
        let mut want0 = vec![0f32; 3];
        let lin0 = GgufLinear::from_slice(GgmlType::Q8_0, k, 3, &make_matrix(GgmlType::Q8_0, k, 3, 3))
            .expect("Q8_0");
        lin0.gemv(&x, &mut want0).expect("gemv");
        let mut want1 = vec![0f32; 2];
        let lin1 = GgufLinear::from_slice(GgmlType::F32, k, 2, &make_matrix(GgmlType::F32, k, 2, 5))
            .expect("F32");
        lin1.gemv(&x, &mut want1).expect("gemv");
        assert_eq!(bits(&outs[0]), bits(&want0));
        assert_eq!(bits(&outs[1]), bits(&want1));
        assert_eq!(bits(&buf[0..3]), bits(&want0));
        assert_eq!(bits(&buf[3..5]), bits(&want1));
    }

    /// mmap 上で隣接していれば**写さずに**連結する (並びが逆でも行範囲で吸収)。
    #[test]
    fn fused_zero_copy_when_adjacent_in_mmap() {
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(true));
        let ty = GgmlType::Q8_0;
        let k = 64usize;
        let a_bytes = make_matrix(ty, k, 3, 71); // ファイル先頭
        let b_bytes = make_matrix(ty, k, 2, 73); // その直後
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&a_bytes);
        file_bytes.extend_from_slice(&b_bytes);

        let path = std::env::temp_dir().join(format!(
            "denno_fused_zero_copy_{}_{}.bin",
            std::process::id(),
            ty.type_size()
        ));
        std::fs::write(&path, &file_bytes).expect("write");
        let f = std::fs::File::open(&path).expect("open");
        // SAFETY: 読み取り専用の共有写像。テスト中にこのファイルを書き換えない。
        let map = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).expect("mmap") });
        drop(f);

        // 呼び出し順は [後ろ, 前] (ファイルの並びと逆)。
        let second = GgufLinear::new(
            ty,
            k,
            2,
            WeightBlocks::mapped(map.clone(), a_bytes.len(), b_bytes.len()),
        )
        .expect("second");
        let first = GgufLinear::new(ty, k, 3, WeightBlocks::mapped(map.clone(), 0, a_bytes.len()))
            .expect("first");
        let fused = FusedLinear::concat(vec![second, first]);
        assert!(fused.is_fused(), "隣接しているのに連結されていない");
        assert!(fused.is_zero_copy(), "隣接しているのに写している");
        // バッファはファイル順 [前 3 行][後ろ 2 行]。呼び出し順の対応表で引く。
        assert_eq!(fused.range(0), 3..5, "後ろの層の行範囲");
        assert_eq!(fused.range(1), 0..3, "前の層の行範囲");

        let x = make_x(k, 77);
        let mut buf = vec![0f32; fused.n()];
        fused.gemv_concat_into("t", &x, &mut buf).expect("gemv");
        assert_eq!(bits(&buf[3..5]), bits(&naive_gemv(ty, &b_bytes, k, 2, &x)));
        assert_eq!(bits(&buf[0..3]), bits(&naive_gemv(ty, &a_bytes, k, 3, &x)));
        let _ = std::fs::remove_file(&path);
    }

    /// ヒープ実体 (隣接情報なし) は写して連結する。値は同じ。
    #[test]
    fn fused_copies_when_not_mapped() {
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(true));
        let ty = GgmlType::Q5_1;
        let k = 256usize;
        let parts = parts_of(ty, k, &[4, 3], 11);
        let fused = FusedLinear::concat(parts);
        assert!(fused.is_fused());
        assert!(!fused.is_zero_copy(), "ヒープ実体なのに写しなしになっている");
        assert_eq!(fused.weight_bytes(), 7 * (k / 32) * ty.type_size());
    }

    /// バッチ経路 (`gemv_batch_split`) も個別の `gemv_batch` とビット一致する。
    #[test]
    fn fused_batch_split_matches_separate_bits() {
        let _g = par_test_lock();
        let _p = FusedGuard::new();
        set_fused_gemv_for_test(Some(true));
        let ty = GgmlType::Q8_0;
        let k = 256usize;
        let ns = [5usize, 3];
        let nt = 3usize;
        let xs: Vec<f32> = (0..nt * k)
            .map(|i| ((pat(i, 91) as f32) / 128.0) - 1.0)
            .collect();

        let parts = parts_of(ty, k, &ns, 13);
        let mut want: Vec<Vec<f32>> = Vec::new();
        for (p, &n) in parts.iter().zip(ns.iter()) {
            let mut o = vec![0f32; nt * n];
            crate::model::qwen4exp::gemv_batch(p, &xs, nt, &mut o).expect("gemv_batch (個別)");
            want.push(o);
        }
        let fused = FusedLinear::concat(parts);
        assert!(fused.is_fused());
        let mut outs: Vec<Vec<f32>> = ns.iter().map(|&n| vec![0f32; nt * n]).collect();
        {
            let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|v| &mut v[..]).collect();
            fused.gemv_batch_split(&xs, nt, &mut refs).expect("gemv_batch_split");
        }
        for i in 0..ns.len() {
            assert_eq!(bits(&outs[i]), bits(&want[i]), "層 {i}");
        }
    }

    /// P86: `two_row` の呼び出しは GPU GEMV に奪われず CPU の 2 行カーネルに届く
    /// (cuda ビルドのみ。GPU 実体は無くても intercept フックで再現する)。
    ///
    /// intercept は `maybe_gpu_gguf_gemv_into` を「成功」させ、かつ out を
    /// 書かない。旧経路 (GPU 素通り) なら two_row の呼び出しも out が埋まらない
    /// まま返る。修正後は down / up の 2 行入口が CPU で計算する。
    #[cfg(feature = "cuda")]
    #[test]
    fn two_row_reaches_cpu_while_plain_entry_is_intercepted() {
        use crate::cuda::GemvInterceptGuard;
        use crate::model::linear_gguf::TWO_ROW_CPU_CALLS;

        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &make_matrix(GgmlType::Q8_0, K, N, 3))
            .expect("from_slice")
            .with_fast_q8(true);
        let x = make_x(K, 1);
        let delta = || TWO_ROW_CPU_CALLS.load(std::sync::atomic::Ordering::Relaxed);

        // 対照: plain 入口 (two_row = false) は intercept に奪われ、out は書かれない。
        let before = delta();
        let mut out = vec![f32::NAN; N];
        {
            let _g = GemvInterceptGuard::new();
            lin.gemv_label("plain", &x, &mut out).expect("gemv_label");
        }
        assert!(
            out.iter().all(|v| v.is_nan()),
            "GPU に奪われたはずの呼び出しが out を書いた (intercept が効いていない)"
        );
        assert_eq!(delta(), before, "plain 入口で two_row カウンタが動いた");

        // two_row 入口は CPU に落ちる (intercept 中でも out が正しく埋まる)。
        // 参照値は先に (intercept 無しで) 取る。参照呼び出し自体も two_row なので
        // カウンタを進める。
        let mut want = vec![0f32; N];
        lin.gemv_label_par_2row("hc_2row_ref", &x, &mut want, 64)
            .expect("gemv_label_par_2row (参照)");
        let before2 = delta();

        let mut out2 = vec![f32::NAN; N];
        {
            let _g = GemvInterceptGuard::new();
            lin.gemv_label_par_2row("hc_2row", &x, &mut out2, 64)
                .expect("gemv_label_par_2row");
        }
        assert_eq!(bits(&out2), bits(&want), "intercept 下の 2 行結果が CPU 参照と違う");
        assert_eq!(delta(), before2 + 1, "two_row の呼び出しが CPU に届いていない");
    }

    /// task 87: `gemv_impl_par` は intercept を**試みない**呼び出し (two_row
    /// バイパス / F32 の CPU 参照) でも実行先フラグを毎回 false に戻す。
    /// 戻さないと直前の呼び出しの true が残り、attn_kprof の `note_gate`
    /// (gate の直前は qkv の GPU GEMV) が gate を「GPU を通った」ように誤って
    /// 数える。cuda ビルドのみ (フラグ自体が cfg(feature = "cuda"))。GPU 実体不要。
    #[cfg(feature = "cuda")]
    #[test]
    fn gemv_impl_par_resets_gpu_flag_without_intercept() {
        // two_row 入口 (hc mix decode / P86 バイパス): intercept を試みない。
        let lin =
            GgufLinear::from_slice(GgmlType::Q8_0, K, N, &make_matrix(GgmlType::Q8_0, K, N, 3))
                .expect("from_slice")
                .with_fast_q8(true);
        let x = make_x(K, 1);
        let mut out = vec![0f32; N];
        crate::cuda::set_last_linear_ran_on_gpu(true);
        lin.gemv_label_par_2row("hc_2row", &x, &mut out, 64)
            .expect("gemv_label_par_2row");
        assert!(
            !crate::cuda::last_linear_ran_on_gpu(),
            "two_row (CPU) の呼び出し後にフラグが true のまま"
        );

        // F32 (ssm の beta / alpha): GPU 経路が存在しない (router_gpu は未設定)。
        let (lin, x) = f32_lin(K, 64, 43);
        let mut out = vec![0f32; 64];
        crate::cuda::set_last_linear_ran_on_gpu(true);
        lin.gemv(&x, &mut out).expect("f32 gemv");
        assert!(
            !crate::cuda::last_linear_ran_on_gpu(),
            "F32 (CPU) の呼び出し後にフラグが true のまま"
        );
    }
}
