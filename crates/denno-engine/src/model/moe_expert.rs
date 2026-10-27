//! MoE expert のスライス切り出しと GEMV (P-MOE-B1)。
//!
//! 出典: expert テンソルの配置 (`ne = [n_in, n_out, n_expert]`) と量子型の使い分けは
//! llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! 行のデコードは本リポジトリの `kquant.rs` (同じく MIT 由来) を再利用している。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! Flash-Next 系の expert テンソルは GGUF 上で 3 次元 (`ne = [n_in, n_out, n_expert]`、
//! ne[0] が最内 = 行長) で、1 ファイルに全 expert ぶんが連結されている。
//! 1 トークンで使うのは各層 top-k 個だけなので、expert 単位でバイト範囲を切り出して
//! GEMV する経路が要る。ここに置くのはその **CPU 参照実装**だけで、
//! GPU カーネルのビット一致比較の基準になる。推論経路には接続しない。
//!
//! 契約:
//!
//! - 対象量子型は実ファイルで使われている 4 種のみ
//!   (**Q4_K / Q5_K / Q5_1 / Q8_0**)。それ以外は `Err`。黙って別経路に落とさない。
//! - 行のデ量子化は [`crate::kquant::dequant_row_into`] そのものを使う
//!   ([`crate::kquant::dequant_row`] は同じ本体に `Vec` を渡すだけなので値は同一)。
//! - **スカラー経路** ([`expert_gemv_scalar`]) の内積は `0..n_in` の昇順に逐次加算。
//!   SIMD 化・分割アキュムレータ・FMA 化をしない。
//!   Rust は既定で浮動小数の再結合を許さないので、素直な `acc += w[i] * x[i]` は
//!   丸め順序が固定される (これがビット一致の根拠)。GPU カーネルとの比較相手はこちら。
//! - **Q4_K だけ**は AVX2+FMA の融合カーネル ([`gemv_q4k_avx2`]) を持つ。行を f32 に
//!   展開せず、144 バイトの量子化ブロックから直接 FMA で内積を取る。累積順が変わるので
//!   スカラー経路とはビット一致しない (相対 1e-5 / 絶対 1e-4 以内)。
//!   実行時に `is_x86_feature_detected!` で判定し、AVX2/FMA が無ければスカラーへ落ちる。
//! - 行バッファは 1 本だけ確保して行ごとに使い回す (行ごとに `Vec` を作らない)。

use anyhow::{bail, Result};

use crate::kquant::{dequant_row_into, GgmlType};

/// 1 expert ぶんのバイト範囲 (テンソル先頭からの相対)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlice {
    /// テンソルのデータ先頭からのオフセット (バイト)
    pub off: usize,
    /// 1 expert ぶんのバイト数
    pub len: usize,
}

impl ExpertSlice {
    /// `ne = [n_in, n_out, n_expert]` の expert テンソルから expert `e` の範囲を返す。
    ///
    /// `off = e * n_out * row_bytes(n_in, ty)`、`len = n_out * row_bytes(n_in, ty)`。
    /// メモリ上は expert が最外なので、expert のバイト列は連続する。
    pub fn new(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize, e: usize) -> Result<Self> {
        let rb = row_bytes(ty, n_in)?;
        if n_out == 0 {
            bail!("n_out が 0 (行が無い)");
        }
        if n_expert == 0 {
            bail!("n_expert が 0");
        }
        if e >= n_expert {
            bail!("expert index {} が範囲外 (n_expert={})", e, n_expert);
        }
        let len = n_out
            .checked_mul(rb)
            .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", n_out))?;
        let off = e
            .checked_mul(len)
            .ok_or_else(|| anyhow::anyhow!("expert offset が usize を溢れる (e={})", e))?;
        Ok(ExpertSlice { off, len })
    }

    /// 範囲の終端 (排他)。テンソル総バイト数以下であることを呼び出し側で確認する。
    pub fn end(&self) -> usize {
        self.off + self.len
    }
}

/// 1 行 (`n_in` 要素) あたりのバイト数。`n_in` は量子型のブロックサイズで割り切れる必要がある。
///
/// 対象外の量子型は `Err` (Q4_K / Q5_K / Q5_1 / Q8_0 以外)。
pub fn row_bytes(ty: GgmlType, n_in: usize) -> Result<usize> {
    match ty {
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q5_1 | GgmlType::Q8_0 => {}
        other => bail!("expert GEMV の対象外の量子型: {:?}", other),
    }
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if n_in == 0 {
        bail!("n_in が 0");
    }
    if bs == 0 || n_in % bs != 0 {
        bail!("n_in={} がブロックサイズ {} で割り切れない (端数は扱わない)", n_in, bs);
    }
    Ok((n_in / bs) * ts)
}

/// 内積 (f32 逐次加算・`0..n` 昇順)。
///
/// ビット一致の基準なので分割アキュムレータや FMA 化をしない
/// (`#[inline(never)]` は最適化で演算列が畳まれないようにするための目印)。
#[inline(never)]
fn dot_seq(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += w[i] * x[i];
    }
    acc
}

/// `kquant::get_scale_min_k4` と同じ式 (あちらは非公開なので写しを持つ)。
///
/// `q` は Q4_K ブロックの `scales[12]`。戻り値は `(scale, min)`。
/// 値が一致することは `q4k_fused_matches_scalar_within_tolerance` が
/// スカラー経路との突き合わせで間接的に固定している。
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// 32 要素ごとの `x` の和。Q4_K の最小値項 `Σ m*x` は `x` だけで決まるので、
/// 行ループの外で 1 回だけ作って使い回す (行ごとに作り直さない)。
fn group_sums_32(x: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(x.len() / 32);
    for c in x.chunks_exact(32) {
        let mut s = 0.0f32;
        for &e in c {
            s += e;
        }
        v.push(s);
    }
    v
}

/// Q4_K 融合カーネルが実行時に使えるか (x86_64 + AVX2 + FMA)。
pub fn q4k_fused_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// テスト用: 融合カーネルが実際に発火した回数。
/// 「テストが緑」= 「融合経路が通った」ではないので、ゲートの通過を数える。
#[cfg(test)]
static FUSED_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Q4_K の融合 GEMV (AVX2 + FMA)。`out[r] = Σ_i w_ri * x_i` を、
/// 行を f32 に展開せず 144 バイトブロックから直接 FMA で取る。
///
/// Q4_K ブロックの値は `y = d*sc*q - dmin*m` (q は 4bit、sc/m は 32 要素ごと) なので
/// `Σ y*x = Σ (d*sc)*(q*x) - Σ (dmin*m)*(Σ_32 x)`。
/// 前半を 8 レーン FMA で積み、後半は [`group_sums_32`] の 32 要素和で引く。
/// 累積順がスカラー経路と違うためビット一致しない (差は相対 1e-5 / 絶対 1e-4 以内)。
///
/// # Safety
///
/// - `n_in % 256 == 0`、`blocks.len() == n_out * (n_in/256)*144`、`x.len() == n_in`、
///   `out.len() == n_out` を呼び出し側 ([`expert_gemv`]) が検証済みであること。
/// - 実行 CPU が AVX2 と FMA を持つこと ([`q4k_fused_supported`] で確認する)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_q4k_avx2(blocks: &[u8], n_in: usize, n_out: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    const BLK: usize = 144; // Q4_K 1 ブロック = 256 要素

    let nb = n_in / 256; // 行あたりのブロック数
    let rb = nb * BLK; // 行バイト数
    let gsum = group_sums_32(x);
    let xp = x.as_ptr();

    /// 32 バイト (= 64 要素の nibble 面) を f32 へ開き、`x` と FMA して 8 レーン累積を返す。
    /// `dv` は 32 要素ブロックの実効スケール `d*sc`。
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fma32(acc: __m256, nib: __m256i, xp: *const f32, dv: __m256) -> __m256 {
        let l = _mm256_castsi256_si128(nib);
        let h = _mm256_extracti128_si256::<1>(nib);
        let mut a = acc;
        a = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(l)), dv),
            _mm256_loadu_ps(xp),
            a,
        );
        a = _mm256_fmadd_ps(
            _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(l))),
                dv,
            ),
            _mm256_loadu_ps(xp.add(8)),
            a,
        );
        a = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(h)), dv),
            _mm256_loadu_ps(xp.add(16)),
            a,
        );
        a = _mm256_fmadd_ps(
            _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(h))),
                dv,
            ),
            _mm256_loadu_ps(xp.add(24)),
            a,
        );
        a
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    for r in 0..n_out {
        let row = blocks.as_ptr().add(r * rb);
        // 累積はブロック (256 要素) ごとに畳んでから f32 で足す。
        // 行全体を 1 本のベクタ累積で回すと、1 レーンあたり 320 項・部分和が
        // 数十まで育って丸め誤差が 1e-4 台になる (スカラー経路との差が許容を超える)。
        // ブロック単位なら 1 レーン 32 項で済み、差は 1e-5 台に収まる。
        let mut total = 0.0f32;
        for b in 0..nb {
            let blk = std::slice::from_raw_parts(row.add(b * BLK), BLK);
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let sc = &blk[4..16];
            let qrow = row.add(b * BLK + 16);
            // 低 nibble 面と高 nibble 面で別のアキュムレータを持つ (依存チェーンを短くする)。
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut minsum = 0.0f32;
            for j in 0..4usize {
                let (s0, m0) = scale_min_k4(2 * j, sc);
                let (s1, m1) = scale_min_k4(2 * j + 1, sc);
                let dv0 = _mm256_set1_ps(d * s0 as f32);
                let dv1 = _mm256_set1_ps(d * s1 as f32);
                let qv = _mm256_loadu_si256(qrow.add(j * 32) as *const __m256i);
                let lo = _mm256_and_si256(qv, _mm256_set1_epi8(0x0f));
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), _mm256_set1_epi8(0x0f));
                let g = b * 8 + j * 2; // 32 要素グループの添字 (x 側)
                let xb = xp.add(b * 256 + j * 64);
                acc0 = fma32(acc0, lo, xb, dv0);
                acc1 = fma32(acc1, hi, xb.add(32), dv1);
                minsum += dmin * m0 as f32 * *gsum.get_unchecked(g);
                minsum += dmin * m1 as f32 * *gsum.get_unchecked(g + 1);
            }
            total += hsum8(_mm256_add_ps(acc0, acc1)) - minsum;
        }
        *out.get_unchecked_mut(r) = total;
    }
}

/// 引数検証 (スカラー経路と融合経路で共通)。行バイト数を返す。
fn check_gemv_args(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &[f32],
) -> Result<usize> {
    let rb = row_bytes(ty, n_in)?;
    if n_out == 0 {
        bail!("n_out が 0 (行が無い)");
    }
    if x.len() != n_in {
        bail!("x の長さ {} が n_in={} と違う", x.len(), n_in);
    }
    if out.len() != n_out {
        bail!("out の長さ {} が n_out={} と違う", out.len(), n_out);
    }
    let want = n_out
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", n_out))?;
    if blocks.len() != want {
        bail!("blocks の長さ {} が n_out * row_bytes = {} と違う", blocks.len(), want);
    }
    Ok(rb)
}

/// スカラー経路の本体 (検証済みの引数を受ける)。
///
/// 行バッファは 1 本だけ。行ごとに Vec を確保しない。
fn gemv_scalar_body(
    ty: GgmlType,
    rb: usize,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let mut row = vec![0.0f32; n_in];
    for r in 0..n_out {
        let src = &blocks[r * rb..(r + 1) * rb];
        dequant_row_into(ty, src, &mut row);
        out[r] = dot_seq(&row, x);
    }
}

/// ページ単位に切り下げた範囲へ `MADV_WILLNEED` を投げる (非同期の先読み依頼)。
///
/// expert の重みは mmap 越しなので、初回のタッチはページフォルトになる。
/// 1 トークンで触る expert の量は 487 MB (48 層 × 11 行列 × 921 KB) あり、
/// フォルト待ちが forward の 9 割を占めていた (層 4594 ms のうち計算は 1.2 ms/層)。
/// GEMV を回す前にまとめて投げると、I/O と計算が重なる。
///
/// Linux 以外・非 x86_64 では何もしない (戻り値も見ない。失敗しても遅くなるだけ)。
pub fn prefetch_range(bytes: &[u8]) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // libc クレートを足さずに済むよう、必要な 1 本だけ宣言する。
        extern "C" {
            fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        const MADV_WILLNEED: i32 = 3; // linux/mman-common.h
        const PAGE: usize = 4096;

        if bytes.is_empty() {
            return;
        }
        let start = bytes.as_ptr() as usize;
        let aligned = start & !(PAGE - 1);
        let len = (start - aligned) + bytes.len();
        // SAFETY: `aligned` はページ境界で、`len` はその境界から元のスライス末尾まで。
        // どちらも同じ mmap の読み取り専用写像の内側なので、範囲外を指さない。
        unsafe {
            madvise(aligned as *mut core::ffi::c_void, len, MADV_WILLNEED);
        }
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = bytes;
    }
}

/// 行方向の並列化しきい値。これ未満の行列はスレッドを立てる損のほうが大きい。
const PAR_MIN_ROWS: usize = 512;

/// `n_out` 行を rayon で分けて `f(chunk_blocks, chunk_out)` を回す。
///
/// **行は互いに独立**なので、どの行をどのスレッドが担当しても各要素の計算順序は
/// 変わらない (ビット一致は保たれる)。`f` は行チャンクぶんの `blocks` と `out` を受ける。
///
/// `par == false` のときは呼び出しスレッドだけで全行を回す。外側で expert を
/// rayon 並列化しているとき (P-MOE-G5) に内側で二重にスレッドを立てないための入口で、
/// 分割の有無は各要素の計算順序を変えないので結果はビット一致する。
#[cfg(feature = "par")]
fn for_each_row_chunk<F>(blocks: &[u8], n_out: usize, rb: usize, out: &mut [f32], par: bool, f: F)
where
    F: Fn(&[u8], &mut [f32]) + Sync + Send,
{
    use rayon::prelude::*;
    if !par || n_out < PAR_MIN_ROWS {
        f(blocks, out);
        return;
    }
    // 1 スレッドあたり最低 64 行は残す (チャンクが細かすぎるとオーバーヘッドが勝つ)。
    let threads = rayon::current_num_threads().max(1);
    let want = n_out.div_ceil(threads).max(64);
    let rows = want.min(n_out);
    let bytes = rows * rb;
    blocks
        .par_chunks(bytes)
        .zip(out.par_chunks_mut(rows))
        .for_each(|(b, o)| f(b, o));
}

#[cfg(not(feature = "par"))]
fn for_each_row_chunk<F>(blocks: &[u8], _n_out: usize, _rb: usize, out: &mut [f32], _par: bool, f: F)
where
    F: Fn(&[u8], &mut [f32]),
{
    f(blocks, out);
}

/// 1 expert ぶんの生バイト (`blocks`) に対する GEMV。`out[r] = dot(row_r, x)`。
///
/// - `blocks` は [`ExpertSlice`] で切り出した 1 expert ぶん (`len` バイトちょうど)。
/// - 行は `0..n_out` の昇順。
/// - **Q4_K かつ AVX2+FMA が使える環境では融合カーネル** ([`gemv_q4k_avx2`]) を通る。
///   それ以外 (他 3 型・AVX2/FMA 無し) は [`expert_gemv_scalar`] と同じスカラー経路。
///   融合経路は累積順が変わるので、スカラー経路とはビット一致しない。
pub fn expert_gemv(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    expert_gemv_inner(ty, blocks, n_in, n_out, x, out, true)
}

/// [`expert_gemv`] の本体。`par == false` なら行方向の rayon 並列を使わない
/// (外側で expert を並列化しているとき用。P-MOE-G5)。
fn expert_gemv_inner(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
    par: bool,
) -> Result<()> {
    let rb = check_gemv_args(ty, blocks, n_in, n_out, x, out)?;

    if ty == GgmlType::Q4K && q4k_fused_supported() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: check_gemv_args が n_in%256==0・blocks/x/out の長さを検証済みで、
            // q4k_fused_supported() が AVX2 と FMA の存在を実行時に確認している。
            // 行は独立なので、チャンク分割しても各要素の計算順序は変わらない。
            for_each_row_chunk(blocks, n_out, rb, out, par, |b, o| unsafe {
                gemv_q4k_avx2(b, n_in, o.len(), x, o)
            });
            #[cfg(test)]
            FUSED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
    }

    gemv_scalar_body(ty, rb, blocks, n_in, n_out, x, out);
    Ok(())
}

/// Q8_0 (34 バイトブロック = 32 要素) の AVX2+FMA 融合カーネル。
///
/// ブロックは `d: f16` (2 バイト) + `qs: [i8; 32]`。行を f32 に開かず、i8 を 8 本ずつ
/// f32 へ変換して FMA する。累積順が変わるのでスカラー経路とはビット一致しない
/// (相対 1e-6 程度)。AVX2/FMA の判定は呼び出し側で行う。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_q8_0_avx2(blocks: &[u8], n_in: usize, n_out: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    const BLK: usize = 34; // d(2) + qs(32)
    let nb = n_in / 32; // 行あたりのブロック数
    let rb = nb * BLK; // 行バイト数
    let xp = x.as_ptr();
    let bp = blocks.as_ptr();

    for r in 0..n_out {
        let row = bp.add(r * rb);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let dv = _mm256_set1_ps(d);
            let q = p.add(2);
            let lo = _mm_loadu_si128(q as *const __m128i);
            let hi = _mm_loadu_si128(q.add(16) as *const __m128i);
            let x0 = xp.add(b * 32);
            let v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            let v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo)));
            let v2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            let v3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi)));
            a0 = _mm256_fmadd_ps(_mm256_mul_ps(v0, dv), _mm256_loadu_ps(x0), a0);
            a1 = _mm256_fmadd_ps(_mm256_mul_ps(v1, dv), _mm256_loadu_ps(x0.add(8)), a1);
            a0 = _mm256_fmadd_ps(_mm256_mul_ps(v2, dv), _mm256_loadu_ps(x0.add(16)), a0);
            a1 = _mm256_fmadd_ps(_mm256_mul_ps(v3, dv), _mm256_loadu_ps(x0.add(24)), a1);
        }
        let s = _mm256_add_ps(a0, a1);
        let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        out[r] = _mm_cvtss_f32(h);
    }
}

/// Q5_1 (24 バイトブロック = 32 要素) の AVX2+FMA 融合カーネル。
///
/// ブロックは `d: f16` + `m: f16` + `qh: u32` (5 ビット目) + `qs: [u8; 16]`
/// (下位 4 ビット、要素 j と j+16 を 1 バイトに詰める)。`w = d*q + m` なので
/// `dot(w, x) = d*dot(q, x) + m*sum(x)` に分け、`sum(x)` はブロック位置ごとに
/// 1 度だけ作る。5 ビットの展開はスカラーで行う (32 要素で 100 命令程度)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_q5_1_avx2(blocks: &[u8], n_in: usize, n_out: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    const BLK: usize = 24;
    let nb = n_in / 32;
    let rb = nb * BLK;
    let xp = x.as_ptr();
    let bp = blocks.as_ptr();

    // ブロック位置ごとの `sum(x[32])` (行に依らないので先に 1 回だけ)。
    let mut gsum = vec![0f32; nb];
    for (b, g) in gsum.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..32 {
            acc += *xp.add(b * 32 + j);
        }
        *g = acc;
    }

    let mut q = [0f32; 32];
    for r in 0..n_out {
        let row = bp.add(r * rb);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut macc = 0f32;
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let m = half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16)).to_f32();
            let qh = std::ptr::read_unaligned(p.add(4) as *const u32);
            let qs = p.add(8);
            for j in 0..16 {
                let byte = *qs.add(j);
                let xh0 = (((qh >> j) << 4) & 0x10) as u8 as i32;
                let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                q[j] = (((byte & 0x0f) as i32) | xh0) as f32;
                q[j + 16] = (((byte >> 4) as i32) | xh1) as f32;
            }
            let dv = _mm256_set1_ps(d);
            let xb = xp.add(b * 32);
            let qp = q.as_ptr();
            a0 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(qp), dv), _mm256_loadu_ps(xb), a0);
            a1 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(qp.add(8)), dv),
                _mm256_loadu_ps(xb.add(8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(qp.add(16)), dv),
                _mm256_loadu_ps(xb.add(16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(qp.add(24)), dv),
                _mm256_loadu_ps(xb.add(24)),
                a3,
            );
            macc += m * gsum[b];
        }
        let s01 = _mm256_add_ps(a0, a1);
        let s23 = _mm256_add_ps(a2, a3);
        let s = _mm256_add_ps(s01, s23);
        let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        out[r] = _mm_cvtss_f32(h) + macc;
    }
}

/// Q5_1 専用の入口。融合カーネルが使えなければ [`expert_gemv`] へ落ちる。
pub fn expert_gemv_q5_1_fast(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    expert_gemv_q5_1_inner(blocks, n_in, n_out, x, out, true)
}

fn expert_gemv_q5_1_inner(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
    par: bool,
) -> Result<()> {
    check_gemv_args(GgmlType::Q5_1, blocks, n_in, n_out, x, out)?;
    if n_in % 32 == 0 && q8_0_fused_supported() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: 引数は check_gemv_args が検証済み (n_in%32==0 は直前で確認) で、
            // q8_0_fused_supported() が AVX2 と FMA の存在を実行時に確認している。
            let rb = n_in / 32 * 24;
            for_each_row_chunk(blocks, n_out, rb, out, par, |b, o| unsafe {
                gemv_q5_1_avx2(b, n_in, o.len(), x, o)
            });
            return Ok(());
        }
    }
    expert_gemv_inner(GgmlType::Q5_1, blocks, n_in, n_out, x, out, par)
}

/// Q8_0 の融合カーネルが使えるか (AVX2 + FMA の実行時判定)。
pub fn q8_0_fused_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Q8_0 専用の入口。融合カーネルが使えなければ [`expert_gemv`] へ落ちる。
///
/// [`expert_gemv`] から自動では呼ばない (27B の 1 本の Q8_0 テンソルが
/// ビット一致契約の対象なので、使うモデル側が明示的に選ぶ)。
pub fn expert_gemv_q8_fast(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    expert_gemv_q8_inner(blocks, n_in, n_out, x, out, true)
}

fn expert_gemv_q8_inner(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
    par: bool,
) -> Result<()> {
    check_gemv_args(GgmlType::Q8_0, blocks, n_in, n_out, x, out)?;
    if n_in % 32 == 0 && q8_0_fused_supported() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: 引数は check_gemv_args が検証済み (n_in%32==0 は直前で確認) で、
            // q8_0_fused_supported() が AVX2 と FMA の存在を実行時に確認している。
            let rb = n_in / 32 * 34;
            for_each_row_chunk(blocks, n_out, rb, out, par, |b, o| unsafe {
                gemv_q8_0_avx2(b, n_in, o.len(), x, o)
            });
            return Ok(());
        }
    }
    expert_gemv_inner(GgmlType::Q8_0, blocks, n_in, n_out, x, out, par)
}

/// expert テンソルの量子型で GEMV を振り分ける (P-MOE-G1 で追加)。
///
/// mmap 経路 ([`crate::model::qwen4exp`] の `ExpertW::gemv`) とキャッシュ経路
/// (`ExpertCacheSet::get` が返したバイト列) の**両方がこれを通る**ので、
/// 同じバイト列に対しては必ず同じカーネル・同じ丸め順になる (ビット一致の根拠)。
///
/// - Q5_1 / Q8_0 は AVX2+FMA の融合カーネル、それ以外 (Q4_K / Q5_K) は
///   [`expert_gemv`] (Q4_K は AVX2+FMA があれば融合)。
pub fn expert_gemv_any(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    expert_gemv_any_inner(ty, blocks, n_in, n_out, x, out, true)
}

/// テスト用: 行並列なしの入口 ([`expert_gemv_any_seq`]) を通った回数。
/// 「外側 expert 並列の内側が逐次版を通っている」ことをテストで数えるためのもの。
#[cfg(test)]
pub(crate) static SEQ_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// [`expert_gemv_any`] の**行並列を使わない**版 (P-MOE-G5)。
///
/// 外側で expert ループを rayon 並列化しているときに呼ぶ。内側でも並列にすると
/// スレッドが二重に立ってオーバーサブスクライブするので、ここでは行を分割しない。
/// 行は互いに独立で、分割の有無は各要素の計算順序を変えないため、
/// [`expert_gemv_any`] と**ビット一致**する (同じカーネルを通る)。
pub fn expert_gemv_any_seq(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    #[cfg(test)]
    SEQ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    expert_gemv_any_inner(ty, blocks, n_in, n_out, x, out, false)
}

fn expert_gemv_any_inner(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
    par: bool,
) -> Result<()> {
    match ty {
        GgmlType::Q5_1 => expert_gemv_q5_1_inner(blocks, n_in, n_out, x, out, par),
        GgmlType::Q8_0 => expert_gemv_q8_inner(blocks, n_in, n_out, x, out, par),
        _ => expert_gemv_inner(ty, blocks, n_in, n_out, x, out, par),
    }
}

/// スカラー経路そのもの (`dequant_row_into` + [`dot_seq`])。**ビット一致の基準**。
///
/// [`expert_gemv`] が融合カーネルへ振る Q4_K でも、こちらは常に逐次加算のままなので、
/// GPU カーネル・参照実装との比較にはこの関数を使う。
pub fn expert_gemv_scalar(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let rb = check_gemv_args(ty, blocks, n_in, n_out, x, out)?;
    gemv_scalar_body(ty, rb, blocks, n_in, n_out, x, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! 実ファイルはこのコンテナから見えないので、ブロックはテスト内で組み立てる。
    //! 期待値は「dequant_row + 素直な逐次内積」を独立に書き下した参照ループと
    //! `to_bits()` で比較する (Q8_0 の 1 本だけは手計算値も突き合わせる)。

    use super::*;
    use crate::kquant::dequant_row;

    const N_IN: usize = 256;
    const N_OUT: usize = 4;
    const N_EXPERT: usize = 3;

    /// テスト用の決定的バイト列 (0..=255)。
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

    /// 1 ブロックを組み立てる。スケール類は有限値に固定する
    /// (生バイトをそのまま d に入れると f16 が NaN/Inf になり、テストが無意味になる)。
    fn make_block(ty: GgmlType, seed: u32) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b = vec![0u8; ts];
        match ty {
            // d(2) + dmin(2) + scales[12] + qs[128]
            GgmlType::Q4K => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..16 {
                    // 下位 6bit = scale (0..63)、上位 2bit = min の上位ビット
                    b[i] = (pat(i, seed) & 0x3f) | ((pat(i + 7, seed) & 0x3) << 6);
                }
                for i in 16..144 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
            GgmlType::Q5K => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..16 {
                    b[i] = (pat(i, seed) & 0x3f) | ((pat(i + 7, seed) & 0x3) << 6);
                }
                for i in 16..176 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + m(2) + qh(4) + qs[16]
            GgmlType::Q5_1 => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..24 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + qs[32]
            GgmlType::Q8_0 => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                for i in 2..34 {
                    b[i] = pat(i, seed);
                }
            }
            other => panic!("テスト対象外の型: {:?}", other),
        }
        b
    }

    /// 1 expert ぶんのブロック列を、`(expert, row, block)` ごとに別 seed で組む。
    fn make_expert(ty: GgmlType, n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let bs = ty.blck_size();
        let nblk = n_in / bs;
        let mut v = Vec::new();
        for r in 0..n_out {
            for blk in 0..nblk {
                let seed = (e * 10_000 + r * 100 + blk) as u32;
                v.extend_from_slice(&make_block(ty, seed));
            }
        }
        v
    }

    /// expert テンソル全体 (expert が最外で連結)。
    fn make_tensor(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for e in 0..n_expert {
            v.extend_from_slice(&make_expert(ty, n_in, n_out, e));
        }
        v
    }

    fn make_x(n_in: usize) -> Vec<f32> {
        (0..n_in)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect()
    }

    /// 参照: `dequant_row` (Vec 確保版) + 素直な逐次内積。
    fn ref_gemv(ty: GgmlType, blocks: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
        let rb = row_bytes(ty, n_in).unwrap();
        let mut out = vec![0.0f32; n_out];
        for r in 0..n_out {
            let w = dequant_row(ty, &blocks[r * rb..(r + 1) * rb], n_in);
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += w[i] * x[i];
            }
            out[r] = acc;
        }
        out
    }

    fn expect_bits(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "長さが違う");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "row {} がビット一致しない: got={} ({:#010x}) want={} ({:#010x})",
                i,
                g,
                g.to_bits(),
                w,
                w.to_bits()
            );
        }
    }

    const ALL_TYPES: [GgmlType; 4] =
        [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q5_1, GgmlType::Q8_0];

    // ---- 1. スライス切り出し ------------------------------------------------

    #[test]
    fn slice_matches_manual_expert_split() {
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            let total = tensor.len();
            let mut covered = 0usize;
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                let want = make_expert(ty, N_IN, N_OUT, e);
                assert_eq!(sl.len, want.len(), "{:?}: expert {} の len が違う", ty, e);
                assert_eq!(
                    &tensor[sl.off..sl.end()],
                    &want[..],
                    "{:?}: expert {} の範囲が手で分割した範囲と違う",
                    ty,
                    e
                );
                covered += sl.len;
            }
            assert_eq!(covered, total, "{:?}: スライスが全体を過不足なく覆っていない", ty);
            // 最後の expert の終端がテンソル総バイト数に一致する
            let last = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT - 1).unwrap();
            assert_eq!(last.end(), total, "{:?}: 最終 expert の終端が総バイト数と違う", ty);
        }
    }

    #[test]
    fn slice_offset_is_contiguous_and_ordered() {
        let ty = GgmlType::Q4K;
        let rb = row_bytes(ty, N_IN).unwrap();
        let per = N_OUT * rb;
        for e in 0..N_EXPERT {
            let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
            assert_eq!(sl.off, e * per);
            assert_eq!(sl.len, per);
        }
    }

    // ---- 2. GEMV が dequant_row + 逐次内積とビット一致 -----------------------

    #[test]
    fn gemv_bits_equal_dequant_row_and_sequential_dot() {
        // Q4_K の `expert_gemv` は AVX2+FMA の融合カーネルへ振れるようになったので、
        // ここは**スカラー経路 (`expert_gemv_scalar`) の値を固定する**。
        // 融合経路の一致は `q4k_fused_matches_scalar_within_tolerance` が見る。
        let x = make_x(N_IN);
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                let blocks = &tensor[sl.off..sl.end()];
                let mut got = vec![0.0f32; N_OUT];
                expert_gemv_scalar(ty, blocks, N_IN, N_OUT, &x, &mut got).unwrap();
                let want = ref_gemv(ty, blocks, N_IN, N_OUT, &x);
                expect_bits(&got, &want);
                // 縮退していないこと (全行 0 なら一致しても意味が無い)
                assert!(
                    got.iter().any(|v| *v != 0.0),
                    "{:?}: expert {} の出力が全 0",
                    ty,
                    e
                );
                assert!(got.iter().all(|v| v.is_finite()), "{:?}: 非有限値", ty);
            }
        }
    }

    #[test]
    fn gemv_on_extracted_slice_equals_gemv_on_standalone_bytes() {
        // ExpertSlice で切り出したバイト列と、expert 単体で組んだバイト列は同一のはず。
        let x = make_x(N_IN);
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            let standalone = make_expert(ty, N_IN, N_OUT, 1);
            let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, 1).unwrap();
            let mut a = vec![0.0f32; N_OUT];
            let mut b = vec![0.0f32; N_OUT];
            expert_gemv(ty, &tensor[sl.off..sl.end()], N_IN, N_OUT, &x, &mut a).unwrap();
            expert_gemv(ty, &standalone, N_IN, N_OUT, &x, &mut b).unwrap();
            expect_bits(&a, &b);
        }
    }

    #[test]
    fn gemv_q8_0_matches_hand_computed_value() {
        // d = 0.5、qs = 1..=32、x = 全 1.0 → out = 0.5 * (1+...+32) = 0.5 * 528 = 264。
        // 0.5 * q は f32 で厳密、和も 264 まで厳密なので丸めの余地が無い。
        let n_in = 32usize;
        let n_out = 1usize;
        let mut b = vec![0u8; GgmlType::Q8_0.type_size()];
        b[0..2].copy_from_slice(&f16_bytes(0.5));
        for j in 0..32 {
            b[2 + j] = (j as u8) + 1;
        }
        let x = vec![1.0f32; n_in];
        let mut out = vec![0.0f32; n_out];
        expert_gemv(GgmlType::Q8_0, &b, n_in, n_out, &x, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 264.0f32.to_bits(), "got {}", out[0]);
    }

    // ---- 3. n_in がブロックサイズで割り切れない -----------------------------

    #[test]
    fn row_bytes_rejects_ragged_n_in() {
        for (ty, bad) in [
            (GgmlType::Q4K, 257usize),
            (GgmlType::Q5K, 255),
            (GgmlType::Q5_1, 33),
            (GgmlType::Q8_0, 31),
        ] {
            assert!(row_bytes(ty, bad).is_err(), "{:?}: n_in={} が通ってしまった", ty, bad);
            assert!(
                ExpertSlice::new(ty, bad, N_OUT, N_EXPERT, 0).is_err(),
                "{:?}: ExpertSlice が端数を許した",
                ty
            );
            // gemv も同じ検査で落ちる (デ量子化の debug_assert より前に弾く)
            let blocks = vec![0u8; 4096];
            let x = vec![0.0f32; bad];
            let mut out = vec![0.0f32; N_OUT];
            assert!(expert_gemv(ty, &blocks, bad, N_OUT, &x, &mut out).is_err());
        }
        // 0 も弾く
        assert!(row_bytes(GgmlType::Q4K, 0).is_err());
    }

    // ---- 4. e >= n_expert ---------------------------------------------------

    #[test]
    fn slice_rejects_out_of_range_expert() {
        for ty in ALL_TYPES {
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT).is_err());
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT + 5).is_err());
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT - 1).is_ok());
        }
        // n_expert = 0 は誰も選べない
        assert!(ExpertSlice::new(GgmlType::Q4K, N_IN, N_OUT, 0, 0).is_err());
    }

    // ---- 5. 範囲外読みが起きないこと ---------------------------------------

    #[test]
    fn last_expert_slice_fits_in_tensor() {
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                assert!(
                    sl.end() <= tensor.len(),
                    "{:?}: expert {} の終端 {} がテンソル長 {} を超える",
                    ty,
                    e,
                    sl.end(),
                    tensor.len()
                );
                // 実際にスライスして GEMV しても落ちない (境界の実読み)
                let x = make_x(N_IN);
                let mut out = vec![0.0f32; N_OUT];
                expert_gemv(ty, &tensor[sl.off..sl.end()], N_IN, N_OUT, &x, &mut out).unwrap();
            }
        }
    }

    #[test]
    fn gemv_rejects_wrong_buffer_lengths() {
        let ty = GgmlType::Q4K;
        let blocks = make_expert(ty, N_IN, N_OUT, 0);
        let x = make_x(N_IN);
        let mut out = vec![0.0f32; N_OUT];
        // blocks が 1 バイト短い / 長い
        assert!(expert_gemv(ty, &blocks[..blocks.len() - 1], N_IN, N_OUT, &x, &mut out).is_err());
        let mut longer = blocks.clone();
        longer.push(0);
        assert!(expert_gemv(ty, &longer, N_IN, N_OUT, &x, &mut out).is_err());
        // x / out の長さ違い
        assert!(expert_gemv(ty, &blocks, N_IN, N_OUT, &x[..N_IN - 1], &mut out).is_err());
        let mut short_out = vec![0.0f32; N_OUT - 1];
        assert!(expert_gemv(ty, &blocks, N_IN, N_OUT, &x, &mut short_out).is_err());
        // n_out = 0
        assert!(expert_gemv(ty, &[], N_IN, 0, &x, &mut []).is_err());
    }

    // ---- 型の門番 -----------------------------------------------------------

    #[test]
    fn unsupported_types_are_rejected() {
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::Q6K,
            GgmlType::Q8_0, // 下で「対象」であることを確認する対照
        ] {
            let want_ok = ty == GgmlType::Q8_0;
            assert_eq!(row_bytes(ty, N_IN).is_ok(), want_ok, "{:?}", ty);
            assert_eq!(
                ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, 0).is_ok(),
                want_ok,
                "{:?}",
                ty
            );
        }
        let x = make_x(N_IN);
        let mut out = vec![0.0f32; N_OUT];
        assert!(expert_gemv(GgmlType::Q6K, &[0u8; 4096], N_IN, N_OUT, &x, &mut out).is_err());
        assert!(expert_gemv(GgmlType::F32, &[0u8; 4096], N_IN, N_OUT, &x, &mut out).is_err());
    }

    // ---- 実ファイルの形 (前提の実測値) との突き合わせ -----------------------

    #[test]
    fn row_bytes_match_flash_next_measurements() {
        // 前提表: ffn_{gate,up}_exps は [2560, 640, 512] で Q4_K の行が 1440 B、
        // ffn_down_exps は [640, 2560, 512] で Q5_1 の行が 480 B。
        assert_eq!(row_bytes(GgmlType::Q4K, 2560).unwrap(), 1440);
        assert_eq!(row_bytes(GgmlType::Q5K, 2560).unwrap(), 1760);
        assert_eq!(row_bytes(GgmlType::Q5_1, 640).unwrap(), 480);
        assert_eq!(row_bytes(GgmlType::Q8_0, 640).unwrap(), 680);
        // 1 expert ぶんのバイト数 (前提表の 921.7 KB / 1.23 MB に対応)
        assert_eq!(ExpertSlice::new(GgmlType::Q4K, 2560, 640, 512, 0).unwrap().len, 921_600);
        assert_eq!(ExpertSlice::new(GgmlType::Q5_1, 640, 2560, 512, 0).unwrap().len, 1_228_800);
        // 512 expert 全体のバイト数
        let sl = ExpertSlice::new(GgmlType::Q4K, 2560, 640, 512, 511).unwrap();
        assert_eq!(sl.off, 511 * 921_600);
        assert_eq!(sl.end(), 512 * 921_600);
    }

    // ---- 6. Q4_K 融合カーネル (AVX2 + FMA) ---------------------------------

    /// 実形状 (640 行 × 2560 列 = 10 ブロック/行)。
    const PERF_N_IN: usize = 2560;
    const PERF_N_OUT: usize = 640;

    /// 融合経路の x。`make_x` と同じく決定的で、行ごとに符号が変わる。
    fn make_x_wide(n_in: usize) -> Vec<f32> {
        (0..n_in)
            .map(|i| ((i * 61 % 127) as f32 - 63.0) / 96.0)
            .collect()
    }

    /// AVX2/FMA が無い環境では黙って緑にせず、理由を出して skip する。
    fn fused_ready(what: &str) -> bool {
        if q4k_fused_supported() {
            return true;
        }
        eprintln!(
            "skip {}: この CPU には AVX2/FMA が無い (融合経路は選ばれずスカラーへ落ちる)",
            what
        );
        false
    }

    #[test]
    fn q4k_fused_matches_scalar_within_tolerance() {
        if !fused_ready("q4k_fused_matches_scalar_within_tolerance") {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;

        // 256 = 1 ブロック/行 (境界の最小)、2560 = 実形状の 10 ブロック/行
        for (n_in, n_out) in [(256usize, 4usize), (PERF_N_IN, PERF_N_OUT)] {
            let x = make_x_wide(n_in);
            let blocks = make_expert(GgmlType::Q4K, n_in, n_out, 0);
            let mut fused = vec![0.0f32; n_out];
            let mut scalar = vec![0.0f32; n_out];

            let before = FUSED_CALLS.load(Relaxed);
            expert_gemv(GgmlType::Q4K, &blocks, n_in, n_out, &x, &mut fused).unwrap();
            let fired = FUSED_CALLS.load(Relaxed) - before;
            expert_gemv_scalar(GgmlType::Q4K, &blocks, n_in, n_out, &x, &mut scalar).unwrap();

            // ゲートが実際に開いたこと (緑 = 経路が発動した証拠、ではないので数を数える)
            assert!(fired >= 1, "n_in={}: 融合カーネルが発火していない", n_in);

            // 縮退していないこと
            assert!(
                scalar.iter().any(|v| *v != 0.0),
                "n_in={}: スカラー出力が全 0 (比較が無意味)",
                n_in
            );
            assert!(
                scalar.iter().all(|v| v.is_finite()),
                "n_in={}: 非有限値が出た",
                n_in
            );

            let truth = ref_gemv_f64(&blocks, n_in, n_out, &x);
            let (mut d_fs, mut d_ft, mut d_st, mut r_fs) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for (r, ((g, w), t)) in fused.iter().zip(scalar.iter()).zip(truth.iter()).enumerate() {
                let diff = (g - w).abs();
                // 許容は「絶対 1e-4 + 相対 1e-5」。実測では差の大半は**スカラー経路側**の
                // 逐次 f32 累積の誤差で、f64 真値に対しては融合のほうが約 10 倍正確
                // (640x2560 で 融合 1.7e-5 / スカラー 1.7e-4)。下の assert で毎回確認する。
                let tol = 1e-4 + 1e-5 * w.abs();
                assert!(
                    diff <= tol,
                    "n_in={} row {}: 融合 {} とスカラー {} の差 {} が許容 {} を超える",
                    n_in,
                    r,
                    g,
                    w,
                    diff,
                    tol
                );
                // 真値 (f64) からの誤差も見る。融合が系統的に悪化していないことの根拠。
                d_ft = d_ft.max((g - *t as f32).abs());
                d_st = d_st.max((w - *t as f32).abs());
                d_fs = d_fs.max(diff);
                if *t != 0.0 {
                    r_fs = r_fs.max(diff / t.abs() as f32);
                }
            }
            eprintln!(
                "Q4_K {} 行 x {} 列: 融合vsスカラー 最大絶対差 {:.3e} (最大相対差 {:.3e}) / \
                 f64真値との最大差 融合 {:.3e} スカラー {:.3e}",
                n_out, n_in, d_fs, r_fs, d_ft, d_st
            );
            // 融合経路はスカラー経路より悪化しない (丸め順が違うだけであることの確認)。
            // 係数 2 は「同じ f32 精度で、レーン分割により誤差が同程度以下」の意味。
            assert!(
                d_ft <= 2.0 * d_st + 1e-6,
                "n_in={}: 融合の真値誤差 {:.3e} がスカラーの {:.3e} より大きすぎる",
                n_in,
                d_ft,
                d_st
            );
        }
    }

    /// Q8_0 の融合カーネル (attention の投影が Q8_0 なので、こちらのほうが効く)。
    #[test]
    fn q8_0_fused_matches_scalar_within_tolerance() {
        if !fused_ready("q8_0_fused_matches_scalar_within_tolerance") {
            return;
        }
        // 256 = 8 ブロック/行、2560 = 実形状 (n_embd) の 80 ブロック/行
        for (n_in, n_out) in [(256usize, 4usize), (PERF_N_IN, PERF_N_OUT)] {
            let x = make_x_wide(n_in);
            let blocks = make_expert(GgmlType::Q8_0, n_in, n_out, 0);
            let mut fused = vec![0.0f32; n_out];
            let mut scalar = vec![0.0f32; n_out];

            expert_gemv_q8_fast(&blocks, n_in, n_out, &x, &mut fused).unwrap();
            expert_gemv_scalar(GgmlType::Q8_0, &blocks, n_in, n_out, &x, &mut scalar).unwrap();

            assert!(
                scalar.iter().any(|v| *v != 0.0),
                "n_in={}: スカラー出力が全 0 (比較が無意味)",
                n_in
            );

            let rb = row_bytes(GgmlType::Q8_0, n_in).unwrap();
            let (mut d_fs, mut d_ft, mut d_st) = (0.0f32, 0.0f32, 0.0f32);
            for r in 0..n_out {
                let w = dequant_row(GgmlType::Q8_0, &blocks[r * rb..(r + 1) * rb], n_in);
                let mut acc = 0.0f64;
                for i in 0..n_in {
                    acc += w[i] as f64 * x[i] as f64;
                }
                let t = acc as f32;
                let diff = (fused[r] - scalar[r]).abs();
                d_fs = d_fs.max(diff);
                d_ft = d_ft.max((fused[r] - t).abs());
                d_st = d_st.max((scalar[r] - t).abs());
                let tol = 1e-4 + 1e-5 * scalar[r].abs();
                assert!(
                    diff <= tol,
                    "n_in={} row {}: 融合 {} とスカラー {} の差 {} が許容 {} を超える",
                    n_in,
                    r,
                    fused[r],
                    scalar[r],
                    diff,
                    tol
                );
            }
            eprintln!(
                "Q8_0 {} 行 x {} 列: 融合vsスカラー 最大絶対差 {:.3e} / f64真値との最大差 融合 {:.3e} スカラー {:.3e}",
                n_out, n_in, d_fs, d_ft, d_st
            );
            assert!(
                d_ft <= 2.0 * d_st + 1e-6,
                "n_in={}: 融合の真値誤差 {:.3e} がスカラーの {:.3e} より大きすぎる",
                n_in,
                d_ft,
                d_st
            );
        }
    }

    /// Q5_1 の融合カーネル (Flash-Next の `ffn_down_exps` がこの型)。
    #[test]
    fn q5_1_fused_matches_scalar_within_tolerance() {
        if !fused_ready("q5_1_fused_matches_scalar_within_tolerance") {
            return;
        }
        // 640 = 実形状 (expert の列数) の 20 ブロック/行、2560 = 行数側の実形状
        for (n_in, n_out) in [(256usize, 4usize), (640, 256)] {
            let x = make_x_wide(n_in);
            let blocks = make_expert(GgmlType::Q5_1, n_in, n_out, 0);
            let mut fused = vec![0.0f32; n_out];
            let mut scalar = vec![0.0f32; n_out];

            expert_gemv_q5_1_fast(&blocks, n_in, n_out, &x, &mut fused).unwrap();
            expert_gemv_scalar(GgmlType::Q5_1, &blocks, n_in, n_out, &x, &mut scalar).unwrap();

            assert!(
                scalar.iter().any(|v| *v != 0.0),
                "n_in={}: スカラー出力が全 0 (比較が無意味)",
                n_in
            );

            let rb = row_bytes(GgmlType::Q5_1, n_in).unwrap();
            let (mut d_fs, mut d_ft, mut d_st) = (0.0f32, 0.0f32, 0.0f32);
            for r in 0..n_out {
                let w = dequant_row(GgmlType::Q5_1, &blocks[r * rb..(r + 1) * rb], n_in);
                let mut acc = 0.0f64;
                for i in 0..n_in {
                    acc += w[i] as f64 * x[i] as f64;
                }
                let t = acc as f32;
                let diff = (fused[r] - scalar[r]).abs();
                d_fs = d_fs.max(diff);
                d_ft = d_ft.max((fused[r] - t).abs());
                d_st = d_st.max((scalar[r] - t).abs());
                let tol = 1e-4 + 1e-5 * scalar[r].abs();
                assert!(
                    diff <= tol,
                    "n_in={} row {}: 融合 {} とスカラー {} の差 {} が許容 {} を超える",
                    n_in,
                    r,
                    fused[r],
                    scalar[r],
                    diff,
                    tol
                );
            }
            eprintln!(
                "Q5_1 {} 行 x {} 列: 融合vsスカラー 最大絶対差 {:.3e} / f64真値との最大差 融合 {:.3e} スカラー {:.3e}",
                n_out, n_in, d_fs, d_ft, d_st
            );
            assert!(
                d_ft <= 2.0 * d_st + 1e-6,
                "n_in={}: 融合の真値誤差 {:.3e} がスカラーの {:.3e} より大きすぎる",
                n_in,
                d_ft,
                d_st
            );
        }
    }

    /// P-MOE-G5: 行並列なしの入口 [`expert_gemv_any_seq`] は行並列版と**ビット一致**する。
    ///
    /// `n_out` を [`PAR_MIN_ROWS`] (512) 以上にして、行並列版が実際にチャンクへ
    /// 分割する条件で比べる (分割の有無が各要素の計算順序を変えないことの確認)。
    #[test]
    fn seq_entry_matches_row_parallel_entry_on_large_matrix() {
        let n_in = N_IN;
        let n_out = 640;
        assert!(n_out >= PAR_MIN_ROWS, "行並列が分割する大きさで比べる");
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * 0.01)
            .collect();
        for ty in [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0] {
            // 1 行 = n_in 要素ぶんのブロック (Q4_K は 256 要素/blk、Q5_1・Q8_0 は 32)。
            let per_row = row_bytes(ty, n_in).expect("row_bytes") / ty.type_size();
            let mut blocks = Vec::new();
            for r in 0..n_out {
                for b in 0..per_row {
                    blocks.extend_from_slice(&make_block(ty, (r * per_row + b) as u32 + 1));
                }
            }
            let mut par = vec![0.0f32; n_out];
            let mut seq = vec![0.0f32; n_out];
            expert_gemv_any(ty, &blocks, n_in, n_out, &x, &mut par).expect("行並列");
            expert_gemv_any_seq(ty, &blocks, n_in, n_out, &x, &mut seq).expect("行並列なし");
            assert!(
                par.iter().any(|v| *v != 0.0),
                "{ty:?}: 出力が全 0 (テストが無意味)"
            );
            for r in 0..n_out {
                assert_eq!(
                    par[r].to_bits(),
                    seq[r].to_bits(),
                    "{ty:?}: 行 {r} がビット一致しない"
                );
            }
        }
    }

    /// f64 の真値。`kquant::dequant_row` (参照実装由来) の値を f64 で逐次内積する。
    /// 融合カーネルとは独立な経路なので、nibble の取り違え・スケールの取り違えを検出する。
    fn ref_gemv_f64(blocks: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f64> {
        let rb = row_bytes(GgmlType::Q4K, n_in).unwrap();
        let mut out = vec![0.0f64; n_out];
        for r in 0..n_out {
            let w = dequant_row(GgmlType::Q4K, &blocks[r * rb..(r + 1) * rb], n_in);
            let mut acc = 0.0f64;
            for i in 0..n_in {
                acc += w[i] as f64 * x[i] as f64;
            }
            out[r] = acc;
        }
        out
    }

    /// 実形状 (640x2560) の合成データでスカラー経路と融合経路の ns/行 を出す。
    /// 実モデルは読まない。速度は環境依存なので合否判定には使わない (数値だけ出す)。
    #[test]
    fn perf_q4k_fused_vs_scalar_ns_per_row() {
        if !fused_ready("perf_q4k_fused_vs_scalar_ns_per_row") {
            return;
        }
        use std::hint::black_box;
        use std::time::Instant;

        let x = make_x_wide(PERF_N_IN);
        let blocks = make_expert(GgmlType::Q4K, PERF_N_IN, PERF_N_OUT, 0);
        let mut out = vec![0.0f32; PERF_N_OUT];
        let (xs, bs) = (x.as_slice(), blocks.as_slice());

        // ウォームアップ (初回のページフォルトを測り込みから外す)
        expert_gemv_scalar(GgmlType::Q4K, black_box(bs), PERF_N_IN, PERF_N_OUT, black_box(xs), &mut out)
            .unwrap();
        expert_gemv(GgmlType::Q4K, black_box(bs), PERF_N_IN, PERF_N_OUT, black_box(xs), &mut out)
            .unwrap();

        let reps = 10usize;
        let t = Instant::now();
        for _ in 0..reps {
            expert_gemv_scalar(
                GgmlType::Q4K,
                black_box(bs),
                PERF_N_IN,
                PERF_N_OUT,
                black_box(xs),
                &mut out,
            )
            .unwrap();
            black_box(&out);
        }
        let scalar = t.elapsed();

        let t = Instant::now();
        for _ in 0..reps {
            expert_gemv(
                GgmlType::Q4K,
                black_box(bs),
                PERF_N_IN,
                PERF_N_OUT,
                black_box(xs),
                &mut out,
            )
            .unwrap();
            black_box(&out);
        }
        let fused = t.elapsed();

        let rows = (reps * PERF_N_OUT) as f64;
        let sc_ns = scalar.as_nanos() as f64 / rows;
        let fu_ns = fused.as_nanos() as f64 / rows;
        eprintln!(
            "Q4_K {} 行 x {} 列: スカラー {:.1} ns/行, 融合 {:.1} ns/行, {:.2} 倍 (reps={})",
            PERF_N_OUT,
            PERF_N_IN,
            sc_ns,
            fu_ns,
            sc_ns / fu_ns,
            reps
        );
        assert!(sc_ns > 0.0 && fu_ns > 0.0, "計測値が 0 (測れていない)");
    }
}
