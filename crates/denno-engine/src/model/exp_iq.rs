//! task 144: IQ 系の量子型 expert を GPU で回すための入口 (GPU 非依存の部分)。
//!
//! # なぜ要るか
//!
//! P39/P43 の VRAM 常駐経路は `cuda::gguf_row_bytes` が返す **Q8_0 / Q5_1 / Q4_K /
//! Q5_K の 4 型**しか GPU で読めない。本番 (swift15 IQ3_S) の expert は
//! gate/up = IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS、down = IQ4_NL / Q2_0 なので、
//! `expert_vram::acquire` の `moe_gemv_dev_ready` が **型の判定で false** になり、
//! 1 本も VRAM に載らなかった (実測: VRAM 使用 0 MiB / ヒット 0 / 全 480 expert が
//! CPU で 99〜108 ms/tok)。`f_moe` (PTX のモジュール) は正常で、原因は型。
//!
//! ここでは新しいカーネル (`kernels_exp_iq.cu`) が読める型の判定・起動形・
//! ゲートの解釈を持つ。cuda.rs は `feature = "cuda"` のときしかコンパイルされない
//! ので、**GPU 無しで検証できる部分をこちらに置く**。
//!
//! # カーネルの契約 (cuda 側と共有)
//!
//! - 1 行 = 1 warp。32 レーンが 8 要素ずつ (IQ4_NL / Q2_0 は 1 レーン = 1 ブロック)
//!   受け持ち、`__shfl_down_sync` の固定順で足す。
//! - ブロック = 4 warp (4 行)、共有メモリ無し。
//! - 累積の順序は CPU と違う (task 144 はビット一致を要求しない)。**決定性**は
//!   起動形が同じなら保たれる。
//!
//! [`dequant_mirror`] / [`dot_mirror`] はカーネルと同じ索引・同じ累積順を Rust で
//! 書き下したもの。GPU の無いコンテナでも「要素 → バイト」の対応が正しいことを
//! `kquant::dequant_row` とのビット一致で確かめられる。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::kquant::GgmlType;

/// `DENNO_EXPERT_IQ_GPU=1`: IQ 系 expert を VRAM 常駐 + GPU GEMV で回す。既定 off。
pub const EXPERT_IQ_GPU_ENV: &str = "DENNO_EXPERT_IQ_GPU";

/// `DENNO_EXPERT_IQ_UPLOAD`: 1 トークンあたりに VRAM へ上げ直す expert の本数の上限。
///
/// ミスした expert は CPU で計算し、**トークンを止めない範囲**で後から VRAM へ
/// 上げる。0 で「上げ直さない」(= 最初に載ったものだけがヒットする)。
pub const EXPERT_IQ_UPLOAD_ENV: &str = "DENNO_EXPERT_IQ_UPLOAD";

/// [`EXPERT_IQ_UPLOAD_ENV`] の既定値 (1 トークンあたり)。
///
/// 1 本 = 約 2.05 MB。8 本で 16.4 MB / トークン = H2D で約 2 ms。
/// ヒット率が 0.5 を超えれば目標 (65 ms/tok) に届くので、詰め込みすぎない。
pub const DEFAULT_UPLOAD_PER_TOKEN: usize = 8;

/// カーネルのスロット数 (型の数)。
pub const SLOTS: usize = 6;

/// `DENNO_EXPERT_IQ_GPU` の解釈 (純関数)。`"1"` / `"true"` / `"on"` だけが真。
pub fn iq_gpu_from_value(v: Option<&str>) -> bool {
    crate::model::expert_vram::flag_on(v)
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn iq_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| iq_gpu_from_value(std::env::var(EXPERT_IQ_GPU_ENV).ok().as_deref()))
}

/// [`EXPERT_IQ_UPLOAD_ENV`] の解釈 (純関数)。未設定・数値でない値は既定値。
pub fn upload_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_UPLOAD_PER_TOKEN),
        _ => DEFAULT_UPLOAD_PER_TOKEN,
    }
}

/// 1 トークンあたりの上げ直し上限 (プロセスで 1 回だけ env を読む)。
pub fn upload_per_token() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| upload_from_value(std::env::var(EXPERT_IQ_UPLOAD_ENV).ok().as_deref()))
}

/// 型 → カーネルのスロット。未対応は `None` (呼び出し側は CPU へ = fail closed)。
pub fn slot(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::IQ2S => Some(0),
        GgmlType::IQ3Xxs => Some(1),
        GgmlType::IQ3S => Some(2),
        GgmlType::IQ4Xs => Some(3),
        GgmlType::IQ4Nl => Some(4),
        GgmlType::Q2_0 => Some(5),
        _ => None,
    }
}

/// この型を新しいカーネルで読めるか。
pub fn supported(ty: GgmlType) -> bool {
    slot(ty).is_some()
}

/// カーネル名 (PTX の `.visible .entry`)。`cuModuleGetFunction` に渡す形。
pub fn kernel_name(slot: usize) -> Option<&'static str> {
    Some(match slot {
        0 => "dno_exp_gemv_iq2s",
        1 => "dno_exp_gemv_iq3xxs",
        2 => "dno_exp_gemv_iq3s",
        3 => "dno_exp_gemv_iq4xs",
        4 => "dno_exp_gemv_iq4nl",
        5 => "dno_exp_gemv_q2_0",
        _ => return None,
    })
}

/// silu カーネルの名前。
pub const SILU_KERNEL: &str = "dno_exp_silu_mul";

/// 1 行あたりのバイト数。ブロックサイズで割り切れない k は `None`。
///
/// `moe_expert::row_bytes` と同じ式だが、こちらは型の許可制を持たない純関数
/// (`(k / blck) * type_size`)。
pub fn row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if bs == 0 || ts == 0 || k == 0 || k % bs != 0 {
        return None;
    }
    Some((k / bs) * ts)
}

/// カーネルの起動形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

/// 層一括 GEMV の起動形 (純関数)。`None` = この形では起動しない。
///
/// 行は warp 単位なので `n_out` は 4 で切り上げる (端数 warp は `row >= n_out` で
/// 何も書かない)。`nseg` は grid.y、上限は 65535。
pub fn launch_plan(nseg: usize, n_out: usize) -> Option<IqLaunch> {
    const ROWS: usize = 4; // kernels_exp_iq.cu の EI_ROWS
    if nseg == 0 || n_out == 0 {
        return None;
    }
    let tiles = n_out.div_ceil(ROWS);
    if nseg > 65535 || tiles > 65535 {
        return None;
    }
    Some(IqLaunch {
        grid: (tiles as u32, nseg as u32, 1),
        block: (32 * ROWS as u32, 1, 1),
    })
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 1 リクエストぶんの観測 (task 144 の REPORT 用)
// ---------------------------------------------------------------------------
//
// ヒット率は「層で参照した expert のうち VRAM に載っていた割合」。GPU に倒した
// トークンだけでなく CPU のトークンも数える (載っていないと永久に上がらないため、
// 上げ直しの効果を同じ物差しで見たい)。台帳 (`DENNO_Q4_PROFILE`) とは独立に、
// リクエスト終端で 1 行だけ出す。

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static GPU_LAYERS: AtomicU64 = AtomicU64::new(0);
static GPU_EXPERTS: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんの参照を積む (`lookups` = 選択 expert 数、`hits` = 載っていた数)。
pub fn note_lookup(lookups: u32, hits: u32) {
    LOOKUPS.fetch_add(lookups as u64, Ordering::Relaxed);
    HITS.fetch_add(hits as u64, Ordering::Relaxed);
}

/// 層一括 GEMV を 1 層ぶん回したことを積む (`n` = GPU で回した expert 数)。
pub fn note_gpu_layer(n: u32) {
    GPU_LAYERS.fetch_add(1, Ordering::Relaxed);
    GPU_EXPERTS.fetch_add(n as u64, Ordering::Relaxed);
}

/// 上げ直し (H2D) を 1 本ぶん積む。
pub fn note_upload() {
    UPLOADS.fetch_add(1, Ordering::Relaxed);
}

/// 上げ直しに使ったホスト時間 (ns)。acquire = ページフォルト + H2D。
static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

/// 上げ直しのホスト時間を積む。
pub fn note_upload_ns(ns: u64) {
    UPLOAD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行。ゲートが off / 参照 0 件なら `None`。
pub fn stats_line() -> Option<String> {
    if !iq_gpu_enabled() {
        return None;
    }
    let l = LOOKUPS.load(Ordering::Relaxed);
    if l == 0 {
        return None;
    }
    let h = HITS.load(Ordering::Relaxed);
    let layers = GPU_LAYERS.load(Ordering::Relaxed);
    let exps = GPU_EXPERTS.load(Ordering::Relaxed);
    let up = UPLOADS.load(Ordering::Relaxed);
    let vram = match crate::model::expert_vram::budget_usage() {
        Some((used, pinned, cap)) => format!(
            "vram={}MiB/{}MiB pin={}MiB",
            used >> 20,
            cap >> 20,
            pinned >> 20
        ),
        None => "vram=-".to_string(),
    };
    let up_ms = UPLOAD_NS.load(Ordering::Relaxed) as f64 / 1e6;
    Some(format!(
        "[exp-iq] lookups={l} hits={h} hit_rate={:.3} gpu_layers={layers} gpu_experts={exps} uploads={up} up_ms={up_ms:.1} {vram}",
        h as f64 / l as f64
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [&LOOKUPS, &HITS, &GPU_LAYERS, &GPU_EXPERTS, &UPLOADS, &UPLOAD_NS] {
        c.store(0, Ordering::Relaxed);
    }
}

// カーネルのミラー (GPU 無しで索引と累積順を確かめるため)
// ---------------------------------------------------------------------------

/// カーネルが 1 レーンぶん読む要素数 (256 要素ブロックの型)。
const LANE_ELEMS: usize = 8;

/// カーネルと同じ手順で 1 ブロック (`blk`) の 256 要素 (IQ4_NL は 32、Q2_0 は 64) を
/// `out_block` に書く。`lane` は 0..32。
///
/// 返り値は書いた要素数 (`None` ならこの型はミラー未対応)。
#[allow(clippy::too_many_arguments)]
fn mirror_block(ty: GgmlType, blk: &[u8], lane: usize, out_block: &mut [f32]) -> Option<usize> {
    use crate::kquant_tables as t;
    match ty {
        GgmlType::IQ2S => {
            // d(2) + qs[32] + signs[32] + qh[8] + scales[8]
            let d = f16(blk, 0);
            let ib32 = lane >> 2;
            let l4 = lane & 3;
            let qh = blk[66 + ib32] as u32;
            let idx = blk[2 + lane] as usize | (((qh << (8 - 2 * l4)) & 0x300) as usize);
            let sg = blk[34 + lane];
            let sc = blk[74 + ib32];
            let nib = if l4 & 2 != 0 { sc >> 4 } else { sc & 0x0f };
            let db = d * (0.5 + nib as f32) * 0.25;
            let grid = t::IQ2S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                let gv = grid[j] as f32;
                let sgn = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                out_block[lane * LANE_ELEMS + j] = db * gv * sgn;
            }
            Some(256)
        }
        GgmlType::IQ3Xxs => {
            let d = f16(blk, 0);
            let ib32 = lane >> 2;
            let l4 = lane & 3;
            let aux = u32::from_le_bytes([
                blk[66 + ib32 * 4],
                blk[67 + ib32 * 4],
                blk[68 + ib32 * 4],
                blk[69 + ib32 * 4],
            ]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            let signs = t::KSIGNS_IQ2XS[((aux >> (7 * l4)) & 127) as usize];
            let g1 = t::IQ3XXS_GRID[blk[2 + ib32 * 8 + 2 * l4] as usize].to_le_bytes();
            let g2 = t::IQ3XXS_GRID[blk[2 + ib32 * 8 + 2 * l4 + 1] as usize].to_le_bytes();
            for j in 0..4 {
                let s0 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & (1 << (j + 4)) != 0 {
                    -1.0
                } else {
                    1.0
                };
                out_block[lane * LANE_ELEMS + j] = db * g1[j] as f32 * s0;
                out_block[lane * LANE_ELEMS + 4 + j] = db * g2[j] as f32 * s1;
            }
            Some(256)
        }
        GgmlType::IQ3S => {
            let d = f16(blk, 0);
            let g = lane >> 2;
            let l4 = lane & 3;
            let sc = blk[106 + (g >> 1)];
            let nib = if g & 1 != 0 { sc >> 4 } else { sc & 0x0f };
            let db = d * (1.0 + 2.0 * nib as f32);
            let qh = blk[66 + g] as u32;
            let idx1 = blk[2 + g * 8 + 2 * l4] as usize | (((qh << (8 - 2 * l4)) & 256) as usize);
            let idx2 =
                blk[2 + g * 8 + 2 * l4 + 1] as usize | (((qh << (7 - 2 * l4)) & 256) as usize);
            let signs = blk[74 + g * 4 + l4];
            let g1 = t::IQ3S_GRID[idx1].to_le_bytes();
            let g2 = t::IQ3S_GRID[idx2].to_le_bytes();
            for j in 0..4 {
                let s0 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & (1 << (j + 4)) != 0 {
                    -1.0
                } else {
                    1.0
                };
                out_block[lane * LANE_ELEMS + j] = db * g1[j] as f32 * s0;
                out_block[lane * LANE_ELEMS + 4 + j] = db * g2[j] as f32 * s1;
            }
            Some(256)
        }
        GgmlType::IQ4Xs => {
            let d = f16(blk, 0);
            let g = lane >> 2;
            let h4 = lane & 3;
            let half = h4 >> 1;
            let jbase = (h4 & 1) * 8;
            let sh = blk[2] as u32 | ((blk[3] as u32) << 8);
            let sl = blk[4 + (g >> 1)] as u32;
            let ls = ((sl >> (4 * (g & 1))) & 0x0f) | (((sh >> (2 * g)) & 3) << 4);
            let dl = d * (ls as i32 - 32) as f32;
            for j in 0..8 {
                let byte = blk[8 + g * 16 + jbase + j];
                let q = if half != 0 { byte >> 4 } else { byte & 0x0f } as usize;
                let gv = t::KVALUES_IQ4NL[q] as f32;
                out_block[lane * LANE_ELEMS + j] = dl * gv;
            }
            Some(256)
        }
        GgmlType::IQ4Nl => {
            // レーン 1 本 = 1 ブロック (32 要素)。`out_block` は 32 要素。
            let d = f16(blk, 0);
            for j in 0..16 {
                let lo = (blk[2 + j] & 0x0f) as usize;
                let hi = (blk[2 + j] >> 4) as usize;
                out_block[j] = d * t::KVALUES_IQ4NL[lo] as f32;
                out_block[16 + j] = d * t::KVALUES_IQ4NL[hi] as f32;
            }
            Some(32)
        }
        GgmlType::Q2_0 => {
            // レーン 1 本 = 1 ブロック (64 要素)。
            let d = f16(blk, 0);
            for j in 0..64 {
                let q = ((blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 3) as i32;
                out_block[j] = (q - 1) as f32 * d;
            }
            Some(64)
        }
        _ => None,
    }
}

/// 行のバイト列を 1 行ぶんデコードする (カーネルと同じ索引)。
///
/// `out.len()` は `k`。要素の並びは [`crate::kquant::dequant_row`] と一致する
/// (テストがビット一致で確かめる)。
pub fn dequant_mirror(ty: GgmlType, row: &[u8], k: usize, out: &mut [f32]) -> bool {
    let Some(bs) = mirror_block_size(ty) else {
        return false;
    };
    if out.len() != k || k % bs != 0 {
        return false;
    }
    let nb = k / bs;
    let mut block = vec![0.0f32; bs];
    for b in 0..nb {
        let bytes = &row[b * ty.type_size()..(b + 1) * ty.type_size()];
        if bs == 256 {
            // 32 レーンが 8 要素ずつ (lane * 8 + j の位置へ書く)。
            for lane in 0..32 {
                mirror_block(ty, bytes, lane, &mut block).expect("mirror");
            }
        } else {
            // 1 レーン = 1 ブロック。ブロック内の並びをそのまま使う。
            mirror_block(ty, bytes, 0, &mut block).expect("mirror");
        }
        out[b * bs..(b + 1) * bs].copy_from_slice(&block);
    }
    true
}

/// 1 ブロックの要素数 (カーネルのレーン割り当ての単位)。
pub fn mirror_block_size(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::IQ2S | GgmlType::IQ3Xxs | GgmlType::IQ3S | GgmlType::IQ4Xs => Some(256),
        GgmlType::IQ4Nl => Some(32),
        GgmlType::Q2_0 => Some(64),
        _ => None,
    }
}

/// カーネルの累積順そのままで 1 行の内積を取る (テスト用)。
///
/// レーンごとに自分の要素を昇順に足し、最後に `__shfl_down_sync` と同じ木で束ねる。
pub fn dot_mirror(ty: GgmlType, row: &[u8], k: usize, x: &[f32]) -> Option<f32> {
    let bs = mirror_block_size(ty)?;
    if x.len() != k || k % bs != 0 {
        return None;
    }
    let nb = k / bs;
    let mut lane_acc = [0.0f32; 32];
    let mut block = vec![0.0f32; bs];
    for b in 0..nb {
        let bytes = &row[b * ty.type_size()..(b + 1) * ty.type_size()];
        let xb = &x[b * bs..(b + 1) * bs];
        if bs == 256 {
            for lane in 0..32 {
                mirror_block(ty, bytes, lane, &mut block).expect("mirror");
                for j in 0..8 {
                    lane_acc[lane] += block[lane * 8 + j] * xb[lane * 8 + j];
                }
            }
        } else {
            mirror_block(ty, bytes, 0, &mut block).expect("mirror");
            // 1 レーン = 1 ブロック: レーン b % 32 がこのブロックを持つ。
            let lane = b % 32;
            for j in 0..bs {
                lane_acc[lane] += block[j] * xb[j];
            }
        }
    }
    // __shfl_down_sync と同じ木 (同時更新)。
    let mut a = lane_acc;
    let mut off = 16;
    while off > 0 {
        for l in 0..off {
            a[l] += a[l + off];
        }
        off >>= 1;
    }
    Some(a[0])
}

/// blk の先頭 2 バイトを f16 として f32 にする (`kquant` の `f16_to_f32` と同じ式)。
fn f16(blk: &[u8], off: usize) -> f32 {
    let h = u16::from_le_bytes([blk[off], blk[off + 1]]);
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquant;

    /// 決定的な擬似乱数 (外部クレートを使わない)。
    fn fill(seed: u64, buf: &mut [u8]) {
        let mut s = seed | 1;
        for b in buf.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 24) as u8;
        }
    }

    fn types() -> Vec<GgmlType> {
        vec![
            GgmlType::IQ2S,
            GgmlType::IQ3Xxs,
            GgmlType::IQ3S,
            GgmlType::IQ4Xs,
            GgmlType::IQ4Nl,
            GgmlType::Q2_0,
        ]
    }

    /// ミラーの索引が `kquant::dequant_row` とビット一致する (全 6 型 x 実形の k)。
    #[test]
    fn mirror_matches_dequant_row() {
        for ty in types() {
            let ts = ty.type_size();
            for k in [256usize, 640, 2560, 5120] {
                if k % ty.blck_size() != 0 {
                    continue;
                }
                let nblk = k / ty.blck_size();
                let mut row = vec![0u8; nblk * ts];
                fill(
                    0x9e37_79b9_7f4a_7c15u64.wrapping_mul(k as u64 + 1),
                    &mut row,
                );
                // f16 のスケールが inf/NaN になると比較が意味を失うので指数部を抑える。
                for b in 0..nblk {
                    row[b * ts + 1] &= 0x3b; // 指数の上位ビットを落とす (|exp| を小さく)
                }
                let want = kquant::dequant_row(ty, &row, k);
                let mut got = vec![0.0f32; k];
                assert!(
                    dequant_mirror(ty, &row, k, &mut got),
                    "{ty:?} k={k}: mirror 未対応"
                );
                for i in 0..k {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "{ty:?} k={k} 要素 {i}: mirror と dequant_row がずれた"
                    );
                }
                assert!(
                    got.iter().any(|v| *v != 0.0),
                    "{ty:?}: 全部 0 では比較にならない"
                );
            }
        }
    }

    /// ミラーの内積が逐次 f32 内積と相対 1e-5 以内 (累積順が違うだけ)。
    #[test]
    fn dot_mirror_is_close_to_sequential_dot() {
        for ty in types() {
            let k = 2560usize;
            let ts = ty.type_size();
            let nblk = k / ty.blck_size();
            let mut row = vec![0u8; nblk * ts];
            fill(0x1234_5678_9abc_def1, &mut row);
            for b in 0..nblk {
                row[b * ts + 1] &= 0x3b;
            }
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
                .collect();
            let w = kquant::dequant_row(ty, &row, k);
            let mut seq = 0.0f32;
            for i in 0..k {
                seq += w[i] * x[i];
            }
            let got = dot_mirror(ty, &row, k, &x).expect("dot");
            let denom = seq.abs().max(1e-6);
            assert!(
                (got - seq).abs() / denom < 1e-5,
                "{ty:?}: mirror {got} と逐次 {seq} が離れすぎ"
            );
        }
    }

    /// スロット・行長・起動形の基本。
    #[test]
    fn slots_and_launch_plan() {
        assert_eq!(slot(GgmlType::IQ2S), Some(0));
        assert_eq!(slot(GgmlType::Q2_0), Some(5));
        assert_eq!(slot(GgmlType::Q4K), None, "既存 4 型は IQ 経路の担当外");
        assert!(kernel_name(0).unwrap().starts_with("dno_exp_gemv_"));
        assert!(kernel_name(SLOTS).is_none());
        // gate (k=2560 IQ2_S): 820 B/行、down (k=640 IQ4_NL): 360 B/行
        assert_eq!(row_bytes(GgmlType::IQ2S, 2560), Some(820));
        assert_eq!(row_bytes(GgmlType::IQ4Nl, 640), Some(360));
        assert_eq!(row_bytes(GgmlType::Q2_0, 640), Some(180));
        assert_eq!(row_bytes(GgmlType::IQ2S, 640), None, "256 で割り切れない");
        // gate/up: 640 行 -> 160 タイル、down: 2560 行 -> 640 タイル
        let p = launch_plan(10, 640).expect("plan");
        assert_eq!(p.grid, (160, 10, 1));
        assert_eq!(p.block, (128, 1, 1));
        let q = launch_plan(10, 2560).expect("plan");
        assert_eq!(q.grid, (640, 10, 1));
        assert!(launch_plan(0, 640).is_none());
        assert!(launch_plan(10, 0).is_none());
    }

    /// ホスト側が読む名前が .cu のカーネル名と一致する (名前の食い違いは
    /// 実機でしか分からないので、ソースを読んで突き合わせる)。
    #[test]
    fn kernel_names_exist_in_cu_source() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let cu = std::fs::read_to_string(dir.join("kernels_exp_iq.cu")).expect(".cu が読めない");
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert!(
                cu.contains(&format!("__global__ void {name}(")),
                "{name} が kernels_exp_iq.cu に無い"
            );
        }
        assert!(
            cu.contains(&format!("__global__ void {SILU_KERNEL}(")),
            "{SILU_KERNEL} が kernels_exp_iq.cu に無い"
        );
        // PTX の生成先 (build_ptx.sh の exp_iq) と cuda.rs が読む名前が同じであること。
        let sh = std::fs::read_to_string(dir.join("build_ptx.sh")).expect("build_ptx.sh");
        assert!(
            sh.contains("exp_iq)       echo \"kernels_exp_iq.cu"),
            "build_ptx.sh に exp_iq の規則が無い"
        );
    }

    /// コミット済みの PTX が .cu と同じ引数の数を持つ (ABI の取り違えは実機で
    /// しか分からないので、PTX の `.param` 行を数えて固定する)。
    #[test]
    fn committed_ptx_has_expected_params() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq.ptx")) else {
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
            assert_eq!(count(name), Some(8), "{name} の引数の数が .cu と違う");
        }
        assert_eq!(count(SILU_KERNEL), Some(4), "{SILU_KERNEL} の引数の数");
    }

    /// ゲートは既定 off、`DENNO_EXPERT_IQ_UPLOAD` の既定は 8。
    #[test]
    fn env_parsing() {
        assert!(!iq_gpu_from_value(None));
        assert!(iq_gpu_from_value(Some("1")));
        assert!(iq_gpu_from_value(Some(" on ")));
        assert!(!iq_gpu_from_value(Some("0")));
        assert_eq!(upload_from_value(None), DEFAULT_UPLOAD_PER_TOKEN);
        assert_eq!(upload_from_value(Some("")), DEFAULT_UPLOAD_PER_TOKEN);
        assert_eq!(upload_from_value(Some("0")), 0);
        assert_eq!(upload_from_value(Some("24")), 24);
        assert_eq!(upload_from_value(Some("x")), DEFAULT_UPLOAD_PER_TOKEN);
    }
}
