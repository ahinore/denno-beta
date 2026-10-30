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
//! - **decode (M=1) 専用**に、活性を 32 要素ブロックで int8 にして `vpdpbusd` で
//!   積む経路を持つ ([`expert_gemv_any_vnni`])。Q4_K / Q5_1 / Q8_0 が対象で、
//!   `DENNO_GEMV_VNNI=1` のときだけ [`expert_gemv_any`] の代わりに使われる
//!   (既定 off。活性を量子化するので f32 経路とはビット一致しない)。
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
    #[cfg(test)]
    ANY_CALLS.with(|c| c.set(c.get() + 1));
    expert_gemv_any_inner(ty, blocks, n_in, n_out, x, out, true)
}

/// テスト用: [`expert_gemv_any`] を通った回数 (スレッドごと)。
///
/// 「GPU 経路を切ったときに CPU の入口が実際に使われた」ことを数えるためのもの。
/// スレッドローカルにしてあるので、並列に走る他のテストの呼び出しと混ざらない
/// (行並列の内側は別スレッドだが、数えるのは top-level の呼び出しだけ)。
#[cfg(test)]
thread_local! {
    pub(crate) static ANY_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// テスト用: このスレッドの [`ANY_CALLS`]。
#[cfg(test)]
pub(crate) fn any_calls() -> usize {
    ANY_CALLS.with(|c| c.get())
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

// ---- H3: 量子型のバッチ GEMV (GEMM) --------------------------------------
//
// prefill はプロンプト全体 (m トークン) を 1 回で流す。1 行版 ([`expert_gemv_any`]) を
// m 回呼ぶと同じ重み行を m 回読むことになり、mmap 越しの 1 expert 921 KB が
// そのまま RAM 帯域に当たる。ここでは**重み行のデコードを 1 回だけ行い m トークンに
// 使い回す**。
//
// 数値契約: `outs[t * n_out + r]` は [`expert_gemv_any`] を m 回呼んだ結果とビット一致。
// 根拠は「1 行版と同じカーネル・同じ順序」:
//
// - Q8_0 / Q5_1 / Q4_K かつ AVX2+FMA がある環境では、1 行版と同じ融合カーネルと
//   同じレーンの累積順を持つバッチ版を通す。ブロックの int→f32 変換
//   (`w = q * d` の丸め) は行に依らないので、ブロックごとに 1 回へ括り出しても
//   積和の順序は変わらない (各行の累積チェーンは独立で、行間で共有しない)。
// - それ以外 (Q5_K、AVX2/FMA 無し) は 1 行版がスカラー経路に落ちるので、
//   [`dequant_row_into`] の結果 1 本を m 行で共有して [`dot_seq`] する。
// - Q4_K / Q5_1 のバッチ融合カーネルは k 方向をチャンクに切って**外側**で回す
//   ([`gemm_k_blocks`])。行ループの内側で x のチャンク (`m × KB × ブロック`) を
//   使い回すので、行ごとに x 全体を L2 から引き直さない。チャンク内のブロックは
//   昇順のままなので、(行, トークン) ごとの加算順は 1 行版と変わらない。

/// [`expert_gemm_any`] の引数検証。行バイト数を返す。
///
/// 対象型の門番は [`row_bytes`] と同じ (Q4_K / Q5_K / Q5_1 / Q8_0)。
fn check_gemm_args(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
    outs: &[f32],
) -> Result<usize> {
    let rb = check_gemm_matrix(ty, blocks, n_in, n_out, xs, m)?;
    let want_o = m
        .checked_mul(n_out)
        .ok_or_else(|| anyhow::anyhow!("outs の長さが usize を溢れる (m={}, n_out={})", m, n_out))?;
    if outs.len() != want_o {
        bail!("outs の長さ {} が m * n_out = {} と違う", outs.len(), want_o);
    }
    Ok(rb)
}

/// [`check_gemm_args`] のうち **出力スライスを見ない** 部分 (行バイト数を返す)。
///
/// 層全体のタスク表を 1 本の rayon で回す経路 ([`crate::model::qwen4exp`] の
/// flat スケジューラ) は、出力を `*mut f32` で受けて行チャンクごとに書く。
/// 出力の長さ検証は呼び出し側が 1 度だけ行うので、ここでは行列と `xs` だけを見る。
/// 検証内容・エラー文は [`check_gemm_args`] と同一。
pub(crate) fn check_gemm_matrix(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
) -> Result<usize> {
    let rb = row_bytes(ty, n_in)?;
    if n_out == 0 {
        bail!("n_out が 0 (行が無い)");
    }
    if m == 0 {
        bail!("m が 0 (トークンが無い)");
    }
    let want_x = m
        .checked_mul(n_in)
        .ok_or_else(|| anyhow::anyhow!("xs の長さが usize を溢れる (m={}, n_in={})", m, n_in))?;
    if xs.len() != want_x {
        bail!("xs の長さ {} が m * n_in = {} と違う", xs.len(), want_x);
    }
    let want_b = n_out
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", n_out))?;
    if blocks.len() != want_b {
        bail!("blocks の長さ {} が n_out * row_bytes = {} と違う", blocks.len(), want_b);
    }
    Ok(rb)
}

/// テスト用: バッチ融合カーネルが実際に発火した回数 (スレッドローカル)。
/// 「テストが緑」= 「融合経路が通った」ではないので、ゲートの通過を数える。
/// テストは 1 本 1 スレッドで走るので、スレッドローカルなら他のテストと混ざらない
/// (行数が少ない行列は rayon を使わないので、数えるのは常に呼び出しスレッド)。
#[cfg(test)]
thread_local! {
    static GEMM_FUSED_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

#[inline]
fn gemm_fused_fired() {
    #[cfg(test)]
    GEMM_FUSED_CALLS.with(|c| c.set(c.get() + 1));
}

/// テスト用: このスレッドでバッチ融合カーネルが発火した累計回数。
#[cfg(test)]
fn gemm_fused_count() -> usize {
    GEMM_FUSED_CALLS.with(|c| c.get())
}

/// 行範囲 `r0..r1` を計算し、`out.add(t * stride + (r - r0))` に書く。
///
/// 行範囲で切るのは、行が互いに独立で、切っても各要素の計算順序が変わらないため
/// (どのスレッドが担当しても結果はビット一致)。
///
/// # 出力ポインタの安全性 (呼び出し側が守る条件)
///
/// - `out` は `m * stride` 要素以上の有効な領域を指すこと。呼び出し元は
///   `stride = n_out` を渡し、`outs.len() == m * n_out` は [`check_gemm_args`] が
///   検証済み。
/// - `[r0, r1)` は [`gemm_row_ranges`] が返した `[0, n_out)` の分割の 1 つで
///   `r1 <= n_out`。書き込みは `t * n_out + [r0, r1)` に限られるので、チャンク
///   同士 (rayon のタスク同士) で重ならない。
/// - どのカーネルも `(m-1)*n_out + n_out - 1` より先へは書かない。
///
/// 上の条件は [`expert_gemm_inner`] (行列 1 本を丸ごと回す) と、層全体の
/// タスク表を 1 本の rayon で回す flat スケジューラ
/// (`crate::model::qwen4exp::moe_experts_flat`) の両方が満たす。flat 側は
/// 行列・`xs`・出力長を [`check_gemm_matrix`] で 1 度だけ検証し、あとは
/// `[0, n_out)` を重ならない行範囲に切ってここへ渡す (チャンクごとの再検証はしない)。
pub(crate) fn gemm_rows(
    ty: GgmlType,
    rb: usize,
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
    fused: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if fused && ty == GgmlType::Q8_0 && q8_0_fused_supported() {
            gemm_fused_fired();
            // SAFETY: 引数は check_gemm_args が検証済み (n_in%32==0 は row_bytes が保証) で、
            // q8_0_fused_supported() が AVX2 と FMA の存在を実行時に確認している。
            // `out` は m*stride 要素ちょうど (debug_assert 済み)。
            unsafe { gemm_q8_0_avx2(blocks, n_in, r0, r1, xs, m, out, stride) };
            return;
        }
        if fused && ty == GgmlType::Q5_1 && q8_0_fused_supported() {
            gemm_fused_fired();
            // SAFETY: 同上 (Q5_1 のブロックは 32 要素 = 24 バイト)。
            unsafe { gemm_q5_1_avx2(blocks, n_in, r0, r1, xs, m, out, stride) };
            return;
        }
        if fused && ty == GgmlType::Q4K && q4k_fused_supported() {
            gemm_fused_fired();
            // SAFETY: 同上 (Q4_K は n_in%256==0 が row_bytes で保証される)。
            unsafe { gemm_q4k_avx2(blocks, n_in, r0, r1, xs, m, out, stride) };
            return;
        }
    }
    let _ = fused;
    gemm_rows_scalar(ty, rb, blocks, n_in, r0, r1, xs, m, out, stride);
}

/// スカラー経路のバッチ版 ([`expert_gemv_scalar`] を m 回呼んだのと同じ順序)。
///
/// 行を f32 に開かない。**1 ブロックずつデコードし、その場で m トークンぶんの
/// 積和へ配る** (作業バッファは 1 ブロック = 32 or 256 要素だけ)。重みのバイト列を
/// 読むのは 1 行あたり 1 回で済む。
///
/// 数値: トークン t の累積は「ブロックを先頭から、ブロック内の要素 i を昇順」に
/// `acc += w[i] * x[i]` する逐次加算そのもの ([`dot_seq`] と同じ順序) なので、
/// `dequant_row_into` + [`dot_seq`] と**ビット一致**する。トークン方向のループを
/// 内側に置くか外側に置くかは、各トークンの累積チェーンが独立なので値に影響しない。
fn gemm_rows_scalar(
    ty: GgmlType,
    rb: usize,
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    debug_assert_eq!((n_in / bs) * ts, rb, "gemm_rows_scalar: 行長");
    let nb = n_in / bs;
    let mut w = vec![0f32; bs];
    let mut acc = vec![0f32; m];
    for r in r0..r1 {
        for a in acc.iter_mut() {
            *a = 0.0f32;
        }
        let row = &blocks[r * rb..(r + 1) * rb];
        for b in 0..nb {
            crate::kquant::dequant_block_into(ty, &row[b * ts..(b + 1) * ts], &mut w);
            let base = b * bs;
            for (t, a) in acc.iter_mut().enumerate() {
                let xb = &xs[t * n_in + base..t * n_in + base + bs];
                let mut s = *a;
                for i in 0..bs {
                    s += w[i] * xb[i];
                }
                *a = s;
            }
        }
        for (t, a) in acc.iter().enumerate() {
            // SAFETY: out は m * stride 要素以上 (gemm_rows の条件) で、
            // t < m / r0 <= r < r1 <= n_out = stride なので範囲内。
            unsafe { *out.add(t * stride + (r - r0)) = *a };
        }
    }
}

/// Q8_0 のバッチ融合カーネル。[`gemv_q8_0_avx2`] と**同じレーン累積順**。
///
/// 1 行版はブロックごとに `w = q*d` を変換し直すが、`w` は行に依らないので
/// ここではブロックごとに 1 回だけ変換し、m トークンの積和で共有する。
/// 累積は 8 レーン 2 本 (`a0`/`a1`) で、行ごとに同じ順序で回るので値は 1 行版と同一。
///
/// # Safety
///
/// `blocks.len() == (r1 - r0 を含む全行) * (n_in/32)*34`、`xs.len() == m*n_in`、
/// `out` が `m*stride` 要素以上の有効な領域を指す (`r0..r1` は `[0, n_out)` の
/// 分割の 1 つ = [`gemm_rows`] の条件)、`r1 <= n_out`、実行 CPU が AVX2 と FMA を持つこと。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q8_0_avx2(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 34; // d(2) + qs(32)
    let nb = n_in / 32;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = xs.as_ptr();

    // トークンごとの 8 レーン累積 2 本。レジスタには収まらないのでメモリに置く
    // (行あたり nb 回のロード/ストアだけ。重みの読みは 1 回で済む)。
    let mut a0 = vec![_mm256_setzero_ps(); m];
    let mut a1 = vec![_mm256_setzero_ps(); m];

    for r in r0..r1 {
        for v in a0.iter_mut() {
            *v = _mm256_setzero_ps();
        }
        for v in a1.iter_mut() {
            *v = _mm256_setzero_ps();
        }
        let row = bp.add(r * rb);
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let dv = _mm256_set1_ps(d);
            let q = p.add(2);
            let lo = _mm_loadu_si128(q as *const __m128i);
            let hi = _mm_loadu_si128(q.add(16) as *const __m128i);
            // ブロックのデコード (行に依らない) は 1 回だけ。
            let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), dv);
            let w1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo))),
                dv,
            );
            let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), dv);
            let w3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi))),
                dv,
            );
            for t in 0..m {
                let xb = xp.add(t * n_in + b * 32);
                let mut c0 = a0[t];
                let mut c1 = a1[t];
                c0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), c0);
                c1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb.add(8)), c1);
                c0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb.add(16)), c0);
                c1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb.add(24)), c1);
                a0[t] = c0;
                a1[t] = c1;
            }
        }
        for t in 0..m {
            let s = _mm256_add_ps(a0[t], a1[t]);
            let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            let h = _mm_add_ss(h, _mm_shuffle_ps::<0x55>(h, h));
            *out.add(t * stride + (r - r0)) = _mm_cvtss_f32(h);
        }
    }
}

/// Q5_1 のバッチ融合カーネル (旧実装 / 参照用)。[`gemv_q5_1_avx2`] と**同じレーン累積順**。
///
/// k ブロック版 ([`gemm_q5_1_avx2`]) のビット一致テストとマイクロベンチの
/// 「変更前」としてのみ使う (`#[cfg(test)]`)。
///
/// 5 ビットの展開 (`qh` + `qs` → 32 要素) は行に依らないのでブロックごとに 1 回だけ。
/// 最小値項 `Σ m*x` は `sum(x[32])` をトークンごとに先に作っておく (1 行版と同じ)。
///
/// # Safety
///
/// [`gemm_q8_0_avx2`] と同じ (Q5_1 のブロックは 32 要素 = 24 バイト)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[cfg(test)] // ビット一致の参照実装 (旧カーネル)。テストからのみ呼ぶ。
unsafe fn gemm_q5_1_avx2_ref(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 24; // d(2) + m(2) + qh(4) + qs(16)
    let nb = n_in / 32;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = xs.as_ptr();

    // ブロック位置ごとの `sum(x[32])` (1 行版と同じ昇順の和)。トークンごとに違う。
    let mut gsums = vec![0f32; m * nb];
    for t in 0..m {
        for b in 0..nb {
            let mut acc = 0f32;
            for j in 0..32 {
                acc += *xp.add(t * n_in + b * 32 + j);
            }
            gsums[t * nb + b] = acc;
        }
    }

    let mut a0 = vec![_mm256_setzero_ps(); m];
    let mut a1 = vec![_mm256_setzero_ps(); m];
    let mut a2 = vec![_mm256_setzero_ps(); m];
    let mut a3 = vec![_mm256_setzero_ps(); m];
    let mut macc = vec![0f32; m];
    let mut q = [0f32; 32];

    for r in r0..r1 {
        for t in 0..m {
            a0[t] = _mm256_setzero_ps();
            a1[t] = _mm256_setzero_ps();
            a2[t] = _mm256_setzero_ps();
            a3[t] = _mm256_setzero_ps();
            macc[t] = 0f32;
        }
        let row = bp.add(r * rb);
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let mv = half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16)).to_f32();
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
            let qp = q.as_ptr();
            // ブロックのデコード (行に依らない) は 1 回だけ。
            let w0 = _mm256_mul_ps(_mm256_loadu_ps(qp), dv);
            let w1 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(8)), dv);
            let w2 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(16)), dv);
            let w3 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(24)), dv);
            for t in 0..m {
                let xb = xp.add(t * n_in + b * 32);
                let mut c0 = a0[t];
                let mut c1 = a1[t];
                let mut c2 = a2[t];
                let mut c3 = a3[t];
                c0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), c0);
                c1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb.add(8)), c1);
                c2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb.add(16)), c2);
                c3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb.add(24)), c3);
                a0[t] = c0;
                a1[t] = c1;
                a2[t] = c2;
                a3[t] = c3;
                macc[t] += mv * gsums[t * nb + b];
            }
        }
        for t in 0..m {
            let s01 = _mm256_add_ps(a0[t], a1[t]);
            let s23 = _mm256_add_ps(a2[t], a3[t]);
            let s = _mm256_add_ps(s01, s23);
            let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            let h = _mm_add_ss(h, _mm_shuffle_ps::<0x55>(h, h));
            *out.add(t * stride + (r - r0)) = _mm_cvtss_f32(h) + macc[t];
        }
    }
}

/// Q4_K のバッチ融合カーネル (旧実装 / 参照用)。[`gemv_q4k_avx2`] と**同じレーン累積順**。
///
/// k ブロック版 ([`gemm_q4k_avx2`]) のビット一致テストとマイクロベンチの
/// 「変更前」としてのみ使う (`#[cfg(test)]`)。
///
/// 1 行版は 256 要素ブロックごとに nibble を f32 へ開き直すが、その値
/// (`w = nib * d*sc`) は行に依らないので、ブロックごとに 1 回だけ作って m トークンで
/// 共有する。累積は 8 レーン 2 本をブロック内でリセットし、ブロックごとに
/// `hsum8(acc0+acc1) - minsum` を `total` へ足す (1 行版と同じ順序)。
///
/// # Safety
///
/// [`gemm_q8_0_avx2`] と同じ。加えて Q4_K は `n_in % 256 == 0` が [`row_bytes`] で
/// 保証される。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[cfg(test)] // ビット一致の参照実装 (旧カーネル)。テストからのみ呼ぶ。
unsafe fn gemm_q4k_avx2_ref(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 144; // Q4_K 1 ブロック = 256 要素

    let nb = n_in / 256;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = xs.as_ptr();

    /// 8 レーンの水平和 ([`gemv_q4k_avx2`] の `hsum8` と同じ順序)。
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    // 32 要素グループごとの `sum(x)` (最小値項 Σ m*x 用)。1 行版と同じ昇順の和。
    let ng32 = n_in / 32;
    let mut gsums = vec![0f32; m * ng32];
    for t in 0..m {
        for g in 0..ng32 {
            let mut acc = 0f32;
            for j in 0..32 {
                acc += *xp.add(t * n_in + g * 32 + j);
            }
            gsums[t * ng32 + g] = acc;
        }
    }

    let mut a0 = vec![_mm256_setzero_ps(); m];
    let mut a1 = vec![_mm256_setzero_ps(); m];
    let mut minsum = vec![0f32; m];
    let mut total = vec![0f32; m];

    for r in r0..r1 {
        for t in 0..m {
            total[t] = 0f32;
        }
        let row = bp.add(r * rb);
        for b in 0..nb {
            let blk = std::slice::from_raw_parts(row.add(b * BLK), BLK);
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let sc = &blk[4..16];
            let qrow = row.add(b * BLK + 16);
            for t in 0..m {
                a0[t] = _mm256_setzero_ps();
                a1[t] = _mm256_setzero_ps();
                minsum[t] = 0f32;
            }
            for j in 0..4usize {
                let (s0, m0) = scale_min_k4(2 * j, sc);
                let (s1, m1) = scale_min_k4(2 * j + 1, sc);
                let dv0 = _mm256_set1_ps(d * s0 as f32);
                let dv1 = _mm256_set1_ps(d * s1 as f32);
                let qv = _mm256_loadu_si256(qrow.add(j * 32) as *const __m256i);
                let lo = _mm256_and_si256(qv, _mm256_set1_epi8(0x0f));
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), _mm256_set1_epi8(0x0f));
                // ブロックのデコード (行に依らない) は 1 回だけ。
                let lol = _mm256_castsi256_si128(lo);
                let loh = _mm256_extracti128_si256::<1>(lo);
                let hil = _mm256_castsi256_si128(hi);
                let hih = _mm256_extracti128_si256::<1>(hi);
                let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lol)), dv0);
                let w1 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(lol))),
                    dv0,
                );
                let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(loh)), dv0);
                let w3 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(loh))),
                    dv0,
                );
                let w4 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hil)), dv1);
                let w5 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(hil))),
                    dv1,
                );
                let w6 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hih)), dv1);
                let w7 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(hih))),
                    dv1,
                );
                let g = b * 8 + j * 2; // 32 要素グループの添字 (x 側)
                for t in 0..m {
                    let xb = xp.add(t * n_in + b * 256 + j * 64);
                    let mut c0 = a0[t];
                    let mut c1 = a1[t];
                    c0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), c0);
                    c0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb.add(8)), c0);
                    c0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb.add(16)), c0);
                    c0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb.add(24)), c0);
                    c1 = _mm256_fmadd_ps(w4, _mm256_loadu_ps(xb.add(32)), c1);
                    c1 = _mm256_fmadd_ps(w5, _mm256_loadu_ps(xb.add(40)), c1);
                    c1 = _mm256_fmadd_ps(w6, _mm256_loadu_ps(xb.add(48)), c1);
                    c1 = _mm256_fmadd_ps(w7, _mm256_loadu_ps(xb.add(56)), c1);
                    a0[t] = c0;
                    a1[t] = c1;
                    minsum[t] += dmin * m0 as f32 * *gsums.get_unchecked(t * ng32 + g);
                    minsum[t] += dmin * m1 as f32 * *gsums.get_unchecked(t * ng32 + g + 1);
                }
            }
            for t in 0..m {
                total[t] += hsum8(_mm256_add_ps(a0[t], a1[t])) - minsum[t];
            }
        }
        for t in 0..m {
            *out.add(t * stride + (r - r0)) = total[t];
        }
    }
}

/// x の k チャンクを L1 に保つための目標バイト数。
///
/// L1D は 32 KB (Zen4 / Ice Lake)。重みのストリーム (行 × KB ブロック) と
/// 累積バッファに半分を残し、x には 16 KB を割り当てる。
const GEMM_X_L1_BUDGET: usize = 16 * 1024;

/// 1 つの k チャンクに含めるブロック数を、x が L1 に収まる範囲で選ぶ。
///
/// x の 1 ブロックぶんは `blk_elems * 4 * m` バイト (m トークンぶん)。
/// これが [`GEMM_X_L1_BUDGET`] に収まる本数を `1..=cap` に丸める。
/// - Q4_K (256 要素) / m = 15: 15,360 B/ブロック → 1 ブロック (x は 15 KB)
/// - Q5_1 (32 要素) / m = 15: 1,920 B/ブロック → 8 ブロック (x は 15 KB)
/// - m = 1 なら cap まで広げる (x は小さいのでチャンクを跨ぐ回数だけが増える)。
///
/// m > 15 では 1 ブロックでも予算を超えるので 1 に張り付く。その場合 x は L2 に
/// 落ちるが、それでも「行ごとに x 全体を引き直す」現行より L2 の再読みが
/// 行数ぶん減る。
#[inline]
fn gemm_k_blocks(blk_elems: usize, m: usize, cap: usize) -> usize {
    let per_block = blk_elems * 4 * m.max(1);
    (GEMM_X_L1_BUDGET / per_block).clamp(1, cap)
}

/// Q4_K のバッチ融合カーネル (k ブロック版)。[`gemm_q4k_avx2_ref`] と**ビット一致**。
///
/// ループ順は `k チャンク → 行 → ブロック → トークン`。行ループの内側で
/// x のチャンク部分 (m × KB × 256 × 4 バイト) を使い回すので、行ごとに x 全体
/// (m × n_in × 4 バイト = m=15 で 150 KB) を L2 から引き直さずに済む。
/// KB は [`gemm_k_blocks`] が L1 32 KB に収まる値として選ぶ。
///
/// 数値: (行, トークン) ごとの累計 `total` を k チャンクを跨いで持ち、
/// ブロック昇順に `hsum8(a0+a1) - minsum` を足す。チャンク内のブロック順も
/// 昇順なので、加算順は現行と同一 (ブロックを跨いだ部分和は作らない)。
/// `w0..w7` のレーン割り当て・`minsum` の加算順・`hsum8` の畳み方も変えていない。
///
/// # Safety
///
/// [`gemm_q8_0_avx2`] と同じ。加えて Q4_K は `n_in % 256 == 0` が [`row_bytes`] で
/// 保証される。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q4k_avx2(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 144; // Q4_K 1 ブロック = 256 要素
    const KB_MAX: usize = 4;

    let nb = n_in / 256;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = xs.as_ptr();
    let rows = r1.saturating_sub(r0);
    if rows == 0 || m == 0 {
        return;
    }

    /// 8 レーンの水平和 ([`gemm_q4k_avx2_ref`] の `hsum8` と同じ順序)。
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    // 32 要素グループごとの `sum(x)` (最小値項 Σ m*x 用)。1 行版と同じ昇順の和。
    let ng32 = n_in / 32;
    let mut gsums = vec![0f32; m * ng32];
    for t in 0..m {
        for g in 0..ng32 {
            let mut acc = 0f32;
            for j in 0..32 {
                acc += *xp.add(t * n_in + g * 32 + j);
            }
            gsums[t * ng32 + g] = acc;
        }
    }

    // (行, トークン) ごとの累計。k チャンクを跨いでブロック昇順の和を保つ。
    let mut total = vec![0f32; rows * m];
    let mut a0 = vec![_mm256_setzero_ps(); m];
    let mut a1 = vec![_mm256_setzero_ps(); m];
    let mut minsum = vec![0f32; m];

    let kb = gemm_k_blocks(256, m, KB_MAX);
    let mut b0 = 0usize;
    while b0 < nb {
        let b1 = (b0 + kb).min(nb);
        for r in r0..r1 {
            let row = bp.add(r * rb);
            let tot = total.as_mut_ptr().add((r - r0) * m);
            for b in b0..b1 {
                let blk = std::slice::from_raw_parts(row.add(b * BLK), BLK);
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                let sc = &blk[4..16];
                let qrow = row.add(b * BLK + 16);
                for t in 0..m {
                    a0[t] = _mm256_setzero_ps();
                    a1[t] = _mm256_setzero_ps();
                    minsum[t] = 0f32;
                }
                for j in 0..4usize {
                    let (s0, m0) = scale_min_k4(2 * j, sc);
                    let (s1, m1) = scale_min_k4(2 * j + 1, sc);
                    let dv0 = _mm256_set1_ps(d * s0 as f32);
                    let dv1 = _mm256_set1_ps(d * s1 as f32);
                    let qv = _mm256_loadu_si256(qrow.add(j * 32) as *const __m256i);
                    let lo = _mm256_and_si256(qv, _mm256_set1_epi8(0x0f));
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), _mm256_set1_epi8(0x0f));
                    // ブロックのデコード (行に依らない) は 1 回だけ。
                    let lol = _mm256_castsi256_si128(lo);
                    let loh = _mm256_extracti128_si256::<1>(lo);
                    let hil = _mm256_castsi256_si128(hi);
                    let hih = _mm256_extracti128_si256::<1>(hi);
                    let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lol)), dv0);
                    let w1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(lol))),
                        dv0,
                    );
                    let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(loh)), dv0);
                    let w3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(loh))),
                        dv0,
                    );
                    let w4 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hil)), dv1);
                    let w5 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(hil))),
                        dv1,
                    );
                    let w6 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hih)), dv1);
                    let w7 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(hih))),
                        dv1,
                    );
                    let g = b * 8 + j * 2; // 32 要素グループの添字 (x 側)
                    for t in 0..m {
                        let xb = xp.add(t * n_in + b * 256 + j * 64);
                        let mut c0 = a0[t];
                        let mut c1 = a1[t];
                        c0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), c0);
                        c0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb.add(8)), c0);
                        c0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb.add(16)), c0);
                        c0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb.add(24)), c0);
                        c1 = _mm256_fmadd_ps(w4, _mm256_loadu_ps(xb.add(32)), c1);
                        c1 = _mm256_fmadd_ps(w5, _mm256_loadu_ps(xb.add(40)), c1);
                        c1 = _mm256_fmadd_ps(w6, _mm256_loadu_ps(xb.add(48)), c1);
                        c1 = _mm256_fmadd_ps(w7, _mm256_loadu_ps(xb.add(56)), c1);
                        a0[t] = c0;
                        a1[t] = c1;
                        minsum[t] += dmin * m0 as f32 * *gsums.get_unchecked(t * ng32 + g);
                        minsum[t] += dmin * m1 as f32 * *gsums.get_unchecked(t * ng32 + g + 1);
                    }
                }
                for t in 0..m {
                    *tot.add(t) += hsum8(_mm256_add_ps(a0[t], a1[t])) - minsum[t];
                }
            }
        }
        b0 = b1;
    }
    for t in 0..m {
        let base = t * stride;
        for ri in 0..rows {
            *out.add(base + ri) = *total.get_unchecked(ri * m + t);
        }
    }
}

/// Q5_1 のバッチ融合カーネル (k ブロック版)。[`gemm_q5_1_avx2_ref`] と**ビット一致**。
///
/// 1 行版は 4 本のベクトル累積 (`a0..a3`) を**行全体**で回し、最後に 1 回だけ
/// hsum する。そのため k ブロック化では、この 4 本 + スカラー (`macc`) を
/// (行, トークン) ごとに保持したまま k チャンクを回す必要がある。
/// 保持する量は行タイル `rt` で L1 に収まるように区切る
/// (`rt × m × 4 × 32 バイト ≦ 16 KB` → m=15 なら 8 行)。
///
/// ループ順は `行タイル → k チャンク → 行 → ブロック → トークン`。
/// 行ループの内側で x のチャンク部分 (m × KB × 32 × 4 バイト) を使い回す。
///
/// 数値: (行, トークン) ごとにブロック昇順で 4 本の FMA と `macc` の加算を進め、
/// 最後に `hsum(a0+a1+a2+a3) + macc` を取る。チャンクを跨いでもブロック順は
/// 昇順のままなので、加算順は現行と同一。
///
/// # Safety
///
/// [`gemm_q8_0_avx2`] と同じ (Q5_1 のブロックは 32 要素 = 24 バイト)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q5_1_avx2(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    xs: &[f32],
    m: usize,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 24; // d(2) + m(2) + qh(4) + qs(16)
    const KB_MAX: usize = 8;
    let nb = n_in / 32;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = xs.as_ptr();
    let rows = r1.saturating_sub(r0);
    if rows == 0 || m == 0 {
        return;
    }

    // ブロック位置ごとの `sum(x[32])` (1 行版と同じ昇順の和)。トークンごとに違う。
    let mut gsums = vec![0f32; m * nb];
    for t in 0..m {
        for b in 0..nb {
            let mut acc = 0f32;
            for j in 0..32 {
                acc += *xp.add(t * n_in + b * 32 + j);
            }
            gsums[t * nb + b] = acc;
        }
    }

    // 1 チャンク = 8 ブロック (256 要素) 以下。m=15 なら 8 ブロック = x 15 KB。
    //
    // ただし 1 行ぶんの x (m * n_in * 4 バイト) が予算に収まるなら、行ループの
    // 内側で x は元から L1 に載っている。分けても x の再読みは減らず、チャンクを
    // 跨ぐぶんループだけが増えるので 1 チャンクにする (m = 1 の 10 KB がこれ)。
    let kb = if m * n_in * 4 <= GEMM_X_L1_BUDGET {
        nb
    } else {
        gemm_k_blocks(32, m, KB_MAX)
    };
    // 4 本のベクトル + スカラー 1 本を (行, トークン) ごとに持つので、
    // 行タイルはそれが L1 (16 KB) に収まる行数にする。
    let rt = (GEMM_X_L1_BUDGET / (m * 4 * 32)).max(1);
    let tile = rt.min(rows);
    let mut acc = vec![_mm256_setzero_ps(); tile * m * 4];
    let mut macc = vec![0f32; tile * m];
    let mut q = [0f32; 32];

    let mut ra = r0;
    while ra < r1 {
        let re = (ra + tile).min(r1);
        let rn = re - ra;
        for v in acc[..rn * m * 4].iter_mut() {
            *v = _mm256_setzero_ps();
        }
        for v in macc[..rn * m].iter_mut() {
            *v = 0f32;
        }

        let mut b0 = 0usize;
        while b0 < nb {
            let b1 = (b0 + kb).min(nb);
            for r in ra..re {
                let row = bp.add(r * rb);
                let ai = (r - ra) * m * 4;
                let mi = (r - ra) * m;
                for b in b0..b1 {
                    let p = row.add(b * BLK);
                    let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
                    let mv =
                        half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16)).to_f32();
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
                    let qp = q.as_ptr();
                    // ブロックのデコード (行に依らない) は 1 回だけ。
                    let w0 = _mm256_mul_ps(_mm256_loadu_ps(qp), dv);
                    let w1 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(8)), dv);
                    let w2 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(16)), dv);
                    let w3 = _mm256_mul_ps(_mm256_loadu_ps(qp.add(24)), dv);
                    for t in 0..m {
                        let xb = xp.add(t * n_in + b * 32);
                        let o = ai + t * 4;
                        let mut c0 = *acc.get_unchecked(o);
                        let mut c1 = *acc.get_unchecked(o + 1);
                        let mut c2 = *acc.get_unchecked(o + 2);
                        let mut c3 = *acc.get_unchecked(o + 3);
                        c0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), c0);
                        c1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb.add(8)), c1);
                        c2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb.add(16)), c2);
                        c3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb.add(24)), c3);
                        *acc.get_unchecked_mut(o) = c0;
                        *acc.get_unchecked_mut(o + 1) = c1;
                        *acc.get_unchecked_mut(o + 2) = c2;
                        *acc.get_unchecked_mut(o + 3) = c3;
                        *macc.get_unchecked_mut(mi + t) +=
                            mv * *gsums.get_unchecked(t * nb + b);
                    }
                }
            }
            b0 = b1;
        }

        for t in 0..m {
            for r in ra..re {
                let o = ((r - ra) * m + t) * 4;
                let a0 = *acc.get_unchecked(o);
                let a1 = *acc.get_unchecked(o + 1);
                let a2 = *acc.get_unchecked(o + 2);
                let a3 = *acc.get_unchecked(o + 3);
                let s01 = _mm256_add_ps(a0, a1);
                let s23 = _mm256_add_ps(a2, a3);
                let s = _mm256_add_ps(s01, s23);
                let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
                let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
                let h = _mm_add_ss(h, _mm_shuffle_ps::<0x55>(h, h));
                *out.add(t * stride + (r - r0)) =
                    _mm_cvtss_f32(h) + *macc.get_unchecked((r - ra) * m + t);
            }
        }
        ra = re;
    }
}

/// 行チャンクの切り方。`par == false`・`n_out` が小さい・`par` feature 無しでは
/// 1 チャンク (= 呼び出しスレッドだけで全行)。
fn gemm_row_ranges(n_out: usize, par: bool) -> Vec<(usize, usize)> {
    #[cfg(feature = "par")]
    {
        if par && n_out >= PAR_MIN_ROWS {
            let threads = rayon::current_num_threads().max(1);
            // 64 行固定だと n_out=640 / 16 スレッドで 10 チャンクしか立たず 6 本が遊ぶ。
            // スレッド数で割り切れる幅にして 16 本すべてを埋める (下限 32 行は据え置き)。
            let want = n_out.div_ceil(threads).max(32).min(n_out);
            return (0..n_out)
                .step_by(want)
                .map(|a| (a, (a + want).min(n_out)))
                .collect();
        }
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = par;
    }
    vec![(0, n_out)]
}

/// 量子型のバッチ GEMV (GEMM)。`outs[t * n_out + r] = dot(row_r, xs[t])`。
///
/// - `xs` は `m * n_in` (行優先、1 行 = 1 トークン)、`outs` は `m * n_out`。
/// - 重み (`blocks`) は 1 expert ぶん ([`ExpertSlice`] で切り出したもの)。
/// - 結果は [`expert_gemv_any`] を m 回呼んだ結果と**ビット一致**する
///   (根拠は上のモジュール内コメント)。
/// - 行 (`n_out`) 方向に rayon で分割する。行は独立なので分割しても値は変わらない。
pub fn expert_gemm_any(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
    outs: &mut [f32],
) -> Result<()> {
    expert_gemm_inner(ty, blocks, n_in, n_out, xs, m, outs, true, true)
}

/// [`expert_gemm_any`] の**融合カーネルを使わない**版。
///
/// 1 行版がスカラー経路に落ちる入口 (`GgufLinear::gemv` の Q5_1 / `fast_q8` 無しの
/// Q8_0) をバッチ化するときに使う。融合カーネルは同じ型でも 1 行版と累積順が
/// 違う (1 ulp ずれる) ので、ここを取り違えるとビット一致が壊れる。
///
/// 結果は [`expert_gemv_scalar`] を m 回呼んだ結果とビット一致する。
pub fn expert_gemm_scalar(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
    outs: &mut [f32],
) -> Result<()> {
    expert_gemm_inner(ty, blocks, n_in, n_out, xs, m, outs, true, false)
}

/// [`expert_gemm_any`] の**行並列を使わない**版。
///
/// 外側で expert ループを rayon 並列化しているときに呼ぶ (内側で二重にスレッドを
/// 立てない)。行の分割は各要素の計算順序を変えないので結果はビット一致する。
pub fn expert_gemm_any_seq(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
    outs: &mut [f32],
) -> Result<()> {
    expert_gemm_inner(ty, blocks, n_in, n_out, xs, m, outs, false, true)
}

/// rayon のタスクへ `*mut f32` を渡すための薄いラッパ。
///
/// # 安全性 (この型を `Send`/`Sync` にしてよい理由)
///
/// - `ranges` ([`gemm_row_ranges`] の戻り値) は `[0, n_out)` の分割なので、
///   タスクが書く `t*n_out + [a, b)` はタスク間で重ならない。
/// - `outs.len() == m * n_out` は [`check_gemm_args`] が検証済み。
/// - どのカーネルも `(m-1)*n_out + n_out - 1` より先へは書かない。
///
/// したがって異なるタスクが同じ要素へ書くことはなく、ポインタを共有しても
/// data race にならない。
#[derive(Clone, Copy)]
pub(crate) struct OutPtr(*mut f32);

impl OutPtr {
    /// 生ポインタから作る。指す先の条件 (この型のドキュメント) は呼び出し側が守る。
    #[inline]
    pub(crate) fn new(p: *mut f32) -> Self {
        Self(p)
    }

    /// ラッパごと (構造体の値として) 取り出す。フィールドを直接触ると
    /// Rust 2021 の閉包キャプチャが `*mut f32` 単体を掴み、`Send`/`Sync` を
    /// 満たさなくなるため、メソッド経由にして `OutPtr` を捕捉させる。
    #[inline]
    pub(crate) fn get(self) -> *mut f32 {
        self.0
    }

    /// `n` 要素先を指す新しいラッパ (元のラッパの範囲は呼び出し側が管理する)。
    ///
    /// # Safety
    /// 元のポインタの指す領域が `n` 要素以上あること (この型のドキュメント)。
    #[inline]
    pub(crate) fn add(self, n: usize) -> Self {
        // SAFETY: 呼び出し側が領域の大きさを保証している (上の Safety)。
        Self(unsafe { self.0.add(n) })
    }
}

// SAFETY: 上の条件により書き込み領域はタスク間で重ならない (ポインタ値の
// コピーは読み取りだけで、指す先への書き込みは各タスクの範囲に閉じる)。
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

fn expert_gemm_inner(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    xs: &[f32],
    m: usize,
    outs: &mut [f32],
    par: bool,
    fused: bool,
) -> Result<()> {
    let rb = check_gemm_args(ty, blocks, n_in, n_out, xs, m, outs)?;
    let ranges = gemm_row_ranges(n_out, par);
    // `outs` はトークン優先 (`outs[t * n_out + r]`) なので行チャンクは連続しないが、
    // `ranges` は [0, n_out) の分割なので、タスクが書く `t * n_out + [a, b)` は
    // 互いに重ならない。よって各タスクは `outs.as_mut_ptr().add(a)` を基点に
    // stride = n_out, r0 = a で直接書ける (中間バッファの確保・ゼロ埋め・コピー無し)。
    if ranges.len() > 1 {
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            let base = OutPtr(outs.as_mut_ptr());
            ranges.par_iter().for_each(move |&(a, b)| {
                // SAFETY: `[a, b)` は [0, n_out) の分割の 1 つで、outs.len() == m * n_out
                // は check_gemm_args が検証済み。書き込みは base.get().add(a) から
                // t * n_out + (r - a) (r < b) に限られるので他タスクと重ならない。
                gemm_rows(
                    ty,
                    rb,
                    blocks,
                    n_in,
                    a,
                    b,
                    xs,
                    m,
                    unsafe { base.get().add(a) },
                    n_out,
                    fused,
                );
            });
            return Ok(());
        }
    }
    let (a, b) = ranges[0];
    gemm_rows(ty, rb, blocks, n_in, a, b, xs, m, outs.as_mut_ptr(), n_out, fused);
    Ok(())
}

// ===========================================================================
// AVX-512 VNNI (`vpdpbusd`) 経路
// ===========================================================================
//
// 2026-10-28 のオペレーター指示により **prefill はビット一致契約に縛られない**。
// 活性を int8 に量子化する誤差はこの経路の中だけに閉じ、f32 経路
// ([`gemm_rows`] / [`expert_gemm_any`] / `expert_gemv*`) の値は 1 ビットも変えない。
//
// - 実行時判定: `is_x86_feature_detected!` + `#[target_feature]`。
//   `-C target-cpu=native` は使わない (CI は x86-64-v3)。
// - `DENNO_VNNI=0` で f32 経路に戻る ([`vnni_enabled`])。
// - 量子化の粒度は型のブロックに合わせる: Q4_K は 256 要素、Q5_1 は 32 要素。
//   32 要素ごとの `Σ x` (f32 の厳密和) は Q4_K の最小値項と Q5_1 の加算項に使う。
// - Q4_K は `y_i = d*sc*q_i - dmin*m_i` を
//   `Σ y x = (d*sc) * Σ q x - dmin * Σ m x` に分ける。前者を `vpdpbusd`
//   (q を u8 のまま 0..15、x を s8)、後者を 32 要素グループごとの `Σ x` で取る。
// - Q5_1 は `y_i = d*q_i + m` を `Σ y x = d * Σ q x + m * Σ x` に分ける
//   (q は 0..31 なので u8 に収まる)。

/// `vpdpbusd` (u8 × s8 → i32) が実行時に使えるか。
///
/// 512 ビット版は 1 命令 64 MAC、256 ビット版は 32 MAC。
pub fn vnni_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// このプロセスで VNNI 経路を使ってよいか。`DENNO_VNNI=0` で無効 (f32 経路へ)。
///
/// 判定はプロセスで 1 回だけ ([`std::sync::OnceLock`])。
pub fn vnni_enabled() -> bool {
    #[cfg(test)]
    {
        // テストが「VNNI を切ったときのビット一致」を明示的に確かめられるようにする
        // (`moe_test_guard` を取ったテストだけが触る)。
        match VNNI_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => return false,
            1 => return vnni_supported(),
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let off = matches!(std::env::var("DENNO_VNNI").ok().as_deref(), Some("0"));
        vnni_supported() && !off
    })
}

/// テスト用: VNNI の可否を上書きする (`None` で既定に戻す)。
///
/// `moe_test_guard` (MoE のテストを直列化する mutex) を取ったテストだけが呼ぶこと。
#[cfg(test)]
pub(crate) fn set_vnni_for_test(on: Option<bool>) {
    let v = match on {
        None => -1,
        Some(false) => 0,
        Some(true) => 1,
    };
    VNNI_TEST_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
static VNNI_TEST_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// 量子型から活性の量子化ブロック長を返す (VNNI 経路を持たない型は `None`)。
pub(crate) fn vnni_act_block(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::Q4K => Some(256),
        GgmlType::Q5_1 => Some(32),
        _ => None,
    }
}

/// 量子化済みの活性 (`m × n_in`)。[`quant_act_i8`] が埋める。
#[derive(Clone, Copy)]
pub(crate) struct VnniAct<'a> {
    /// `m * n_in` の int8 (トークン優先、1 トークン = `n_in` 要素)
    pub q: &'a [i8],
    /// `m * (n_in / blk)` のスケール (`max|x| / 127`、0 のとき 1.0)
    pub sc: &'a [f32],
    /// `m * (n_in / 32)` の `Σ x` (量子化前の f32 の逐次和)
    pub gs: &'a [f32],
    pub m: usize,
    pub n_in: usize,
    /// 活性の量子化ブロック長 (Q4_K は 256、Q5_1 は 32)
    pub blk: usize,
}

/// 活性 `xs` (`m × n_in`、トークン優先) を int8 に量子化する。
///
/// - `blk` 要素ごとに 1 スケール: `scale = max|x| / 127` (max が 0 なら 1.0)。
/// - `q[i] = round(x[i] * (1/scale))` を `[-127, 127]` に丸める (round-to-nearest)。
/// - `gs` は **32 要素ごと** の `Σ x`。量子化前の f32 を昇順に足す
///   (Q4_K の `Σ m x` と Q5_1 の `m Σ x` に使う)。
///
/// 引数の長さは `q = m*n_in`、`sc = m*(n_in/blk)`、`gs = m*(n_in/32)`。
pub(crate) fn quant_act_i8(
    xs: &[f32],
    m: usize,
    n_in: usize,
    blk: usize,
    q: &mut [i8],
    sc: &mut [f32],
    gs: &mut [f32],
) -> Result<()> {
    if m == 0 || n_in == 0 {
        bail!("quant_act_i8: m={} n_in={} が 0", m, n_in);
    }
    if blk < 32 || blk % 32 != 0 || n_in % blk != 0 {
        bail!("quant_act_i8: blk={} が 32 の倍数でないか n_in={} を割り切れない", blk, n_in);
    }
    let nb = n_in / blk;
    let ng32 = n_in / 32;
    if xs.len() != m * n_in || q.len() != m * n_in || sc.len() != m * nb || gs.len() != m * ng32 {
        bail!(
            "quant_act_i8: 長さが違う (xs={} q={} sc={} gs={}, 期待 {} {} {} {})",
            xs.len(),
            q.len(),
            sc.len(),
            gs.len(),
            m * n_in,
            m * n_in,
            m * nb,
            m * ng32
        );
    }
    for t in 0..m {
        for b in 0..nb {
            let xb = &xs[t * n_in + b * blk..t * n_in + (b + 1) * blk];
            let mut mx = 0.0f32;
            for &v in xb {
                mx = mx.max(v.abs());
            }
            let scale = if mx == 0.0 { 1.0 } else { mx / 127.0 };
            let inv = 1.0 / scale;
            let qb = &mut q[t * n_in + b * blk..t * n_in + (b + 1) * blk];
            for (qi, &v) in qb.iter_mut().zip(xb.iter()) {
                let r = (v * inv).round().clamp(-127.0, 127.0);
                *qi = r as i8;
            }
            sc[t * nb + b] = scale;
            let base = b * (blk / 32);
            for g in 0..blk / 32 {
                let mut s = 0.0f32;
                for &v in &xb[g * 32..(g + 1) * 32] {
                    s += v;
                }
                gs[t * ng32 + base + g] = s;
            }
        }
    }
    Ok(())
}

/// テスト用: VNNI カーネルが発火した回数 (プロセス全体)。
/// 「テストが緑」= 「VNNI 経路が通った」ではないので通過を数える。
/// flat スケジューラは rayon のワーカースレッドから呼ぶので、スレッドローカルではなく
/// プロセス全体で数える。
#[cfg(test)]
static VNNI_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn vnni_fired() {
    #[cfg(test)]
    VNNI_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// テスト用: VNNI カーネルが発火した累計回数。
#[cfg(test)]
pub(crate) fn vnni_calls() -> usize {
    VNNI_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Q4_K の VNNI バッチ融合カーネル。行範囲 `r0..r1` を計算して
/// `out.add(t * stride + (r - r0))` に書く。
///
/// ループ順は `ブロック → 行 → トークン → チャンク (64 要素)`。ブロック (256 要素) を
/// 外側に置くので、そのブロックぶんの活性 `m × 256 バイト` (m=40 で 10 KB) が
/// 行ループの間ずっと L1 に載る。`total` は (行, トークン) ごとの f32 累計で、
/// ブロックごとに `xd * (Σ q x) - dmin * (Σ m x)` を足す。
///
/// # Safety
///
/// - `blocks.len() == n_out * row_bytes(Q4K, n_in)` (`n_in % 256 == 0`)。
/// - `act` は `quant_act_i8` を通した活性で、`blk == 256`・長さは型どおり。
/// - `out` は `m * stride` 要素以上の有効な領域を指し、`r0..r1` は `[0, n_out)` の分割。
/// - 実行 CPU が AVX-512 (F/BW/VL/DQ/VNNI) を持つこと。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn gemm_q4k_vnni(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    act: &VnniAct<'_>,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 144; // Q4_K 1 ブロック = 256 要素
    /// 同じタスクの中で何行先まで先読みするか (重みは毎層 cold で来る)。
    const PF_ROWS: usize = 3;

    let nb = n_in / 256;
    let rb = nb * BLK;
    let ng32 = n_in / 32;
    let rows = r1.saturating_sub(r0);
    let m = act.m;
    if rows == 0 || m == 0 {
        return;
    }
    debug_assert_eq!(act.n_in, n_in, "gemm_q4k_vnni: 活性の n_in");
    debug_assert_eq!(act.blk, 256, "gemm_q4k_vnni: 活性のブロック長");
    let bp = blocks.as_ptr();
    let xp = act.q.as_ptr();
    let scp = act.sc.as_ptr();
    let gp = act.gs.as_ptr();

    /// 8 レーンの水平和 (既存 AVX2 カーネルと同じ順序)。
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// 16 レーンの水平和 (hsum8 の 2 本を足す)。
    #[inline]
    #[target_feature(enable = "avx2,fma,avx512f")]
    unsafe fn hsum16(v: __m512) -> f32 {
        let lo = _mm512_castps512_ps256(v);
        // 128 ビット組を [2,3,2,3] に並べ替えると下位 256 ビットが上半分になる。
        let hi = _mm512_castps512_ps256(_mm512_shuffle_f32x4::<0b11_10_11_10>(v, v));
        hsum8(_mm256_add_ps(lo, hi))
    }

    let mut total = vec![0f32; rows * m];
    // ブロック (行に依らない) のデコード結果をトークンループの外へ出す。
    let mut qv = [_mm512_setzero_si512(); 4];
    let mut av = [_mm512_setzero_ps(); 4];

    for b in 0..nb {
        for r in r0..r1 {
            let row = bp.add(r * rb + b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(row as *const u16)).to_f32();
            let dmin =
                half::f16::from_bits(std::ptr::read_unaligned(row.add(2) as *const u16)).to_f32();
            let scq = std::slice::from_raw_parts(row.add(4), 12);
            let mut mnf = [0f32; 8];
            for j in 0..4usize {
                let qs = _mm256_loadu_si256(row.add(16 + j * 32) as *const __m256i);
                let lo = _mm256_and_si256(qs, _mm256_set1_epi8(0x0f));
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qs), _mm256_set1_epi8(0x0f));
                // 要素順: 下位 nibble が先頭 32 要素、上位 nibble が次の 32 要素。
                qv[j] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                let (s0, m0) = scale_min_k4(2 * j, scq);
                let (s1, m1) = scale_min_k4(2 * j + 1, scq);
                mnf[2 * j] = m0 as f32;
                mnf[2 * j + 1] = m1 as f32;
                let v0 = _mm256_set1_ps(d * s0 as f32);
                let v1 = _mm256_set1_ps(d * s1 as f32);
                av[j] = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v1, 1);
            }
            let mnv = _mm256_loadu_ps(mnf.as_ptr());
            let tot = total.as_mut_ptr().add((r - r0) * m);
            for t in 0..m {
                let xb = xp.add(t * n_in + b * 256);
                let mut acc = _mm512_setzero_ps();
                for j in 0..4usize {
                    let xv = _mm512_loadu_si512(xb.add(j * 64) as *const __m512i);
                    // q は u8 (0..15)、x は s8。64 要素 = 2 グループを 1 命令で取る。
                    let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv[j], xv);
                    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), av[j], acc);
                }
                // 最小値項: dmin * Σ_g m_g * (Σ x)。
                let gsv = _mm256_loadu_ps(gp.add(t * ng32 + b * 8));
                let msum = hsum8(_mm256_mul_ps(mnv, gsv));
                let xd = *scp.add(t * nb + b);
                *tot.add(t) += xd * hsum16(acc) - dmin * msum;
            }
            // 重みは毎層 cold: 数行先の行頭を L1 へ先読みする (残りは HW に任せる)。
            if r + PF_ROWS < r1 {
                let nx = row.add(PF_ROWS * rb);
                let mut o = 0usize;
                while o < rb {
                    _mm_prefetch(nx.add(o) as *const i8, _MM_HINT_T0);
                    o += 128;
                }
            }
        }
    }

    for t in 0..m {
        let base = t * stride;
        for k in 0..rows {
            *out.add(base + k) = *total.get_unchecked(k * m + t);
        }
    }
}

/// Q5_1 の VNNI バッチ融合カーネル。[`gemm_q5_1_vnni`] の説明は下。
///
/// 1 ブロック = 32 要素 = 24 バイト。`y_i = d*q_i + m` なので
/// `Σ y x = d * Σ q x + m * Σ x`。`vpdpbusd` (256 ビット = 32 MAC) で `Σ q x` を取り、
/// 8 レーンの累積に `d * xd` を掛けながら足す (ブロックごとに `d` が違うため、
/// 32 要素ずつスケールを掛けてから累積する)。`m * Σ x` はスカラーで足す。
///
/// ループ順は `行タイル (16 行) → ブロック → 行 → トークン`。8 レーンの累積は
/// (行, トークン) ごとに持つので、行タイルはそれが L1 に収まる 16 行にする
/// (m=40 で 16 × 40 × 32 バイト = 20 KB)。
///
/// # Safety
///
/// [`gemm_q4k_vnni`] と同じ (Q5_1 は `n_in % 32 == 0` が [`row_bytes`] で保証される)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn gemm_q5_1_vnni(
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    act: &VnniAct<'_>,
    out: *mut f32,
    stride: usize,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 24; // d(2) + m(2) + qh(4) + qs(16)
    const RT: usize = 16; // 行タイル
    const PF_ROWS: usize = 3;

    let nb = n_in / 32;
    let rb = nb * BLK;
    let ng32 = n_in / 32;
    let rows = r1.saturating_sub(r0);
    let m = act.m;
    if rows == 0 || m == 0 {
        return;
    }
    debug_assert_eq!(act.n_in, n_in, "gemm_q5_1_vnni: 活性の n_in");
    debug_assert_eq!(act.blk, 32, "gemm_q5_1_vnni: 活性のブロック長");
    let bp = blocks.as_ptr();
    let xp = act.q.as_ptr();
    let scp = act.sc.as_ptr();
    let gp = act.gs.as_ptr();

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// 16 ビットの各ビットを 1 バイトずつ (0x10 / 0x00) に展開する。
    /// Q5_1 の 5 ビット目 (`qh`) を 32 バイトの u8 に載せるために使う。
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn spread_hi_bits(v: u16) -> __m128i {
        let mask = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
        let lo = _mm_set1_epi16(v as i16);
        let hi = _mm_set1_epi16((v >> 8) as i16);
        let a = _mm_and_si128(
            _mm_cmpeq_epi16(_mm_and_si128(lo, mask), mask),
            _mm_set1_epi16(0x10),
        );
        let b = _mm_and_si128(
            _mm_cmpeq_epi16(_mm_and_si128(hi, mask), mask),
            _mm_set1_epi16(0x10),
        );
        _mm_packus_epi16(a, b)
    }

    let mut ra = r0;
    while ra < r1 {
        let re = (ra + RT).min(r1);
        let rn = re - ra;
        let mut acc = vec![_mm256_setzero_ps(); rn * m];
        let mut ms = vec![0f32; rn * m];

        for b in 0..nb {
            for r in ra..re {
                let p = bp.add(r * rb + b * BLK);
                let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
                let mc = half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16))
                    .to_f32();
                let qh = std::ptr::read_unaligned(p.add(4) as *const u32);
                let qs = _mm_loadu_si128(p.add(8) as *const __m128i);
                let nib_lo = _mm_and_si128(qs, _mm_set1_epi8(0x0f));
                let nib_hi = _mm_and_si128(_mm_srli_epi16::<4>(qs), _mm_set1_epi8(0x0f));
                // 要素 0..31 = [下位 nibble 16 個, 上位 nibble 16 個] | 5 ビット目
                let qv = _mm256_set_m128i(
                    _mm_or_si128(nib_hi, spread_hi_bits((qh >> 16) as u16)),
                    _mm_or_si128(nib_lo, spread_hi_bits(qh as u16)),
                );
                let ai = (r - ra) * m;
                for t in 0..m {
                    let xv = _mm256_loadu_si256(xp.add(t * n_in + b * 32) as *const __m256i);
                    let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qv, xv);
                    let xd = *scp.add(t * nb + b);
                    let sv = _mm256_set1_ps(d * xd);
                    let o = ai + t;
                    let a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), sv, *acc.get_unchecked(o));
                    *acc.get_unchecked_mut(o) = a;
                    // gs は量子化前の Σ x なので xd を掛けない (掛けると最小値項が xd 倍に潰れる)。
                    *ms.get_unchecked_mut(o) += mc * *gp.add(t * ng32 + b);
                }
                // 重みは毎層 cold: 数行先の同じブロック位置を L1 へ先読みする。
                if r + PF_ROWS < re {
                    let nx = bp.add((r + PF_ROWS) * rb + b * BLK);
                    _mm_prefetch(nx as *const i8, _MM_HINT_T0);
                    _mm_prefetch(nx.add(64) as *const i8, _MM_HINT_T0);
                }
            }
        }

        for t in 0..m {
            for r in ra..re {
                let o = (r - ra) * m + t;
                *out.add(t * stride + (r - r0)) =
                    hsum8(*acc.get_unchecked(o)) + *ms.get_unchecked(o);
            }
        }
        ra = re;
    }
}

/// VNNI 経路で行範囲 `r0..r1` を計算する。使えないときは `false` を返す
/// (呼び出し側は [`gemm_rows`] の f32 経路へ落とす)。
///
/// 使えない条件: 対象外の量子型、活性の形 (`m` / `n_in` / `blk`) が合わない、
/// 実行 CPU が AVX-512 VNNI を持たない。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_rows_vnni(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    r0: usize,
    r1: usize,
    act: &VnniAct<'_>,
    out: *mut f32,
    stride: usize,
) -> bool {
    let Some(blk) = vnni_act_block(ty) else {
        return false;
    };
    if act.m == 0 || act.n_in != n_in || act.blk != blk {
        return false;
    }
    let nb = n_in / blk;
    let ng32 = n_in / 32;
    if act.q.len() < act.m * n_in || act.sc.len() < act.m * nb || act.gs.len() < act.m * ng32 {
        return false;
    }
    let rb = match row_bytes(ty, n_in) {
        Ok(rb) => rb,
        Err(_) => return false,
    };
    if blocks.len() < r1 * rb || r0 > r1 || r1 > blocks.len() / rb {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !vnni_supported() {
            return false;
        }
        vnni_fired();
        // SAFETY: 上の検証で `blocks` の行数・`act` の形・`r0..r1` の範囲は確認済み。
        // `out` の指す先 (m * stride 要素以上) は呼び出し側の契約
        // ([`gemm_rows`] の Safety) と同じ。実行 CPU の機能は `vnni_supported` が見た。
        unsafe {
            match ty {
                GgmlType::Q4K => {
                    gemm_q4k_vnni(blocks, n_in, r0, r1, act, out, stride);
                }
                GgmlType::Q5_1 => {
                    gemm_q5_1_vnni(blocks, n_in, r0, r1, act, out, stride);
                }
                _ => return false,
            }
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (blocks, r0, r1, out, stride);
        false
    }
}

// ===========================================================================
// ===========================================================================
// M=1 (decode) の VNNI GEMV: 活性を 32 要素ブロックで int8 にして `vpdpbusd`
// ===========================================================================
//
// prefill のバッチ経路 ([`gemm_rows_vnni`]) は活性の量子化ブロック長が型ごとに
// 違う (Q4_K は 256、Q5_1 は 32) が、decode は 1 トークン = 1 行なので
// **型によらず 32 要素固定**で 1 回だけ量子化し、行ループの外に置く。
//
// 数値: f32 経路 ([`expert_gemv_any`]) とはビット一致しない。差の出所は
// (a) 累積順、(b) 活性の int8 量子化。重み側は整数のまま `vpdpbusd` に載るので、
// 重みの復元式 (Q4_K `d*sc*q - dmin*m`、Q5_1 `d*q + m`、Q8_0 `d*q`) は f32 経路と同一。

/// M=1 用に量子化した活性 (32 要素ブロック。`n_in % 32 == 0` が前提)。
///
/// 1 回の GEMV 呼び出しの中で 1 度だけ埋め、行ループでは読むだけにする
/// (行ごとに量子化し直さない)。
#[derive(Default)]
pub(crate) struct VnniActM1 {
    /// `n_in` 要素の int8 (`round(x / sc[g])` を `[-127, 127]` に clamp)
    pub q: Vec<i8>,
    /// `n_in / 32` のスケール (`max|x| / 127`。max が 0 のブロックは 1.0)
    pub sc: Vec<f32>,
    /// `n_in / 32` の `Σ x` (量子化前の f32。Q4_K の `Σ m x` と Q5_1 の `m Σ x` 用)
    pub gs: Vec<f32>,
    /// `n_in / 32` の `Σ q` (int)。Q8_0 は重みを +128 して u8 にするので、
    /// その分を i32 で引き戻すのに使う。
    pub sq: Vec<i32>,
    /// Q8_0 の引き戻しベクトル (8 レーン × グループ数)。レーン 0 に `128 * Σ q`、
    /// 残りは 0。ブロックごとに `_mm256_setr_epi32` を組まないための表。
    pub q8_fix: Vec<i32>,
}

impl VnniActM1 {
    /// `x` を 32 要素ブロックで int8 に量子化する (スケールはブロック単位で f32)。
    pub fn quantize(&mut self, x: &[f32]) -> Result<()> {
        let n_in = x.len();
        if n_in == 0 || n_in % 32 != 0 {
            bail!("VNNI GEMV の活性長 {} が 32 の倍数でない", n_in);
        }
        let ng = n_in / 32;
        self.q.clear();
        self.q.resize(n_in, 0);
        self.sc.clear();
        self.sc.resize(ng, 0.0);
        self.gs.clear();
        self.gs.resize(ng, 0.0);
        // ブロック長 32 固定 (`max|x| / 127`)。gs も 32 要素ごとに作られる。
        quant_act_i8(x, 1, n_in, 32, &mut self.q, &mut self.sc, &mut self.gs)?;
        self.sq.clear();
        self.q8_fix.clear();
        self.q8_fix.resize(ng * 8, 0);
        for (g, c) in self.q.chunks_exact(32).enumerate() {
            let mut s = 0i32;
            for &v in c {
                s += v as i32;
            }
            self.sq.push(s);
            self.q8_fix[g * 8] = 128 * s;
        }
        Ok(())
    }
}

/// 8 レーンの水平和 (既存 AVX2 カーネルと同じ順序)。
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum8_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
    _mm_cvtss_f32(s)
}

/// 16 レーンの水平和 (`hsum8_avx2` の 2 本を足す)。
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma,avx512f")]
unsafe fn hsum16_avx512(v: std::arch::x86_64::__m512) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm512_castps512_ps256(v);
    // 128 ビット組を [2,3,2,3] に並べ替えると下位 256 ビットが上半分になる。
    let hi = _mm512_castps512_ps256(_mm512_shuffle_f32x4::<0b11_10_11_10>(v, v));
    hsum8_avx2(_mm256_add_ps(lo, hi))
}

/// 16 ビットの各ビットを 1 バイトずつ (0x10 / 0x00) に展開する。
///
/// Q5_1 の 5 ビット目 (`qh`) を 32 バイトの u8 に載せるために使う。
/// `gemm_q5_1_vnni` の中にも同じものがある (prefill 側はそのままにしてある)。
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn spread_hi_bits(v: u16) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let mask = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
    let lo = _mm_set1_epi16(v as i16);
    let hi = _mm_set1_epi16((v >> 8) as i16);
    let a = _mm_and_si128(
        _mm_cmpeq_epi16(_mm_and_si128(lo, mask), mask),
        _mm_set1_epi16(0x10),
    );
    let b = _mm_and_si128(
        _mm_cmpeq_epi16(_mm_and_si128(hi, mask), mask),
        _mm_set1_epi16(0x10),
    );
    _mm_packus_epi16(a, b)
}

/// 数行先の行頭を L1 へ先読みする (重みは decode では毎層 cold で来る)。
///
/// 範囲は `r + PF_ROWS < rows` のときに限るので、行チャンクの外へは触らない。
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn prefetch_rows_ahead(row: *const u8, rb: usize, r: usize, rows: usize, ahead: usize) {
    use std::arch::x86_64::*;
    if ahead == 0 {
        return; // A/B 用の逃げ道 (定数なので最適化で消える)
    }
    if r + ahead < rows {
        let nx = row.add(ahead * rb);
        let mut o = 0usize;
        while o < rb {
            _mm_prefetch(nx.add(o) as *const i8, _MM_HINT_T0);
            o += 128;
        }
    }
}

/// Q4_K の M=1 VNNI カーネル。`out[r] = dot(row_r, x)` (r は渡したチャンク内)。
///
/// 1 ブロック (256 要素) = 4 個の 64 要素チャンク。各チャンクは
/// `[下位 nibble 32 B][上位 nibble 32 B]` を 1 本の 512 ビットにして `vpdpbusd`
/// 1 回で 64 MAC を取る (`gemm_q4k_vnni` と同じ nibble 順)。
/// 復元式は `y = (d*sc)*q - dmin*m` なので
/// `Σ y x = (d*sc)*sd*Σ q xq - dmin*m*Σ x` (sd は 32 要素グループの活性スケール)。
/// 最小値項の `Σ x` は量子化前の f32 ([`VnniActM1::gs`]) を使う
/// (活性スケールを掛けると最小値項だけが sd 倍に潰れる)。
///
/// # Safety
///
/// - `n_in % 256 == 0`、`blocks.len() == n_out * row_bytes(Q4K, n_in)`、
///   `act` は `n_in` 要素を量子化済み、`out.len() == n_out`。
/// - 実行 CPU が AVX-512 (F/BW/VL/DQ/VNNI) を持つこと ([`vnni_supported`])。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn gemv_q4k_vnni(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    act: &VnniActM1,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    const BLK: usize = 144; // Q4_K 1 ブロック = 256 要素
    const PF_ROWS: usize = 3;

    let nb = n_in / 256; // 行あたりのブロック数
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = act.q.as_ptr();
    let scp = act.sc.as_ptr();
    let gsp = act.gs.as_ptr();
    let nib = _mm256_set1_epi8(0x0f);

    for r in 0..n_out {
        let row = bp.add(r * rb);
        // 行全体で 1 本のベクタ累積。1 レーンあたりの項数は 4 (チャンク内の
        // 4 要素) × 4 (チャンク) × ブロック数 = 160 で、誤差は量子化より十分小さい。
        let mut acc = _mm512_setzero_ps();
        let mut minsum = 0.0f32;
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let dmin =
                half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16)).to_f32();
            let scq = std::slice::from_raw_parts(p.add(4), 12);
            let g0 = b * 8; // 32 要素グループの先頭添字
            let xb = xp.add(b * 256);
            for j in 0..4usize {
                let (s0, m0) = scale_min_k4(2 * j, scq);
                let (s1, m1) = scale_min_k4(2 * j + 1, scq);
                // 64 要素 = 2 グループ。スケールは [g(2j) ×8, g(2j+1) ×8]。
                let v0 = _mm256_set1_ps(d * s0 as f32 * *scp.add(g0 + 2 * j));
                let v1 = _mm256_set1_ps(d * s1 as f32 * *scp.add(g0 + 2 * j + 1));
                let av = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v1, 1);
                let qs = _mm256_loadu_si256(p.add(16 + j * 32) as *const __m256i);
                let lo = _mm256_and_si256(qs, nib);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qs), nib);
                // 要素順: 下位 nibble が先頭 32 要素、上位 nibble が次の 32 要素。
                let qv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                let xv = _mm512_loadu_si512(xb.add(j * 64) as *const __m512i);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv, xv);
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), av, acc);
                minsum += dmin * m0 as f32 * *gsp.add(g0 + 2 * j);
                minsum += dmin * m1 as f32 * *gsp.add(g0 + 2 * j + 1);
            }
        }
        prefetch_rows_ahead(row, rb, r, n_out, PF_ROWS);
        *out.get_unchecked_mut(r) = hsum16_avx512(acc) - minsum;
    }
}

/// Q5_1 の M=1 VNNI カーネル (1 ブロック = 32 要素 = 24 バイト)。
///
/// `y = d*q + m` なので `Σ y x = d*sd*Σ q xq + m*Σ x`。`vpdpbusd` (256 ビット =
/// 32 MAC) 1 回で `Σ q xq` を取り、ブロックごとに `d*sd` を掛けて累積する。
/// `m*Σ x` は量子化前の `Σ x` をスカラーで足す。
///
/// # Safety
///
/// [`gemv_q4k_vnni`] と同じ (Q5_1 は `n_in % 32 == 0` が [`row_bytes`] で保証される)。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn gemv_q5_1_vnni(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    act: &VnniActM1,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    const BLK: usize = 24; // d(2) + m(2) + qh(4) + qs(16)
    const PF_ROWS: usize = 3;

    let nb = n_in / 32;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = act.q.as_ptr();
    let scp = act.sc.as_ptr();
    let gsp = act.gs.as_ptr();
    let nib = _mm_set1_epi8(0x0f);

    for r in 0..n_out {
        let row = bp.add(r * rb);
        let mut acc = _mm256_setzero_ps();
        let mut msum = 0.0f32;
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let mc =
                half::f16::from_bits(std::ptr::read_unaligned(p.add(2) as *const u16)).to_f32();
            let qh = std::ptr::read_unaligned(p.add(4) as *const u32);
            let qs = _mm_loadu_si128(p.add(8) as *const __m128i);
            let nib_lo = _mm_and_si128(qs, nib);
            let nib_hi = _mm_and_si128(_mm_srli_epi16::<4>(qs), nib);
            // 要素 0..31 = [下位 nibble 16 個, 上位 nibble 16 個] | 5 ビット目
            let qv = _mm256_set_m128i(
                _mm_or_si128(nib_hi, spread_hi_bits((qh >> 16) as u16)),
                _mm_or_si128(nib_lo, spread_hi_bits(qh as u16)),
            );
            let xv = _mm256_loadu_si256(xp.add(b * 32) as *const __m256i);
            let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qv, xv);
            let sd = *scp.add(b);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(d * sd), acc);
            // gs は量子化前の Σ x なので sd を掛けない (掛けると最小値項が潰れる)。
            msum += mc * *gsp.add(b);
        }
        prefetch_rows_ahead(row, rb, r, n_out, PF_ROWS);
        *out.get_unchecked_mut(r) = hsum8_avx2(acc) + msum;
    }
}

/// Q8_0 の M=1 VNNI カーネル (1 ブロック = 32 要素 = 34 バイト)。
///
/// `y = d*q` (q は i8) だが `vpdpbusd` の片側は u8 でなければならない。
/// そこで重みを `q ^ 0x80 = q + 128` で u8 にし、`128 * Σ q` を i32 のうちに
/// 引き戻す (`dpbusd` の後に `_mm256_sub_epi32`)。f32 で引くと大きな値の差し引きに
/// なって桁落ちする。`Σ q` は 32 要素グループごとに 1 回だけ作る
/// ([`VnniActM1::q8_fix`])。
///
/// # Safety
///
/// [`gemv_q4k_vnni`] と同じ。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
unsafe fn gemv_q8_0_vnni(
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    act: &VnniActM1,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    const BLK: usize = 34; // d(2) + qs(32)
    // Q8_0 は行が 2720 B と短く、先読みの命令数が本体に比べて重い (実測で 1T が
    // 33.7 -> 22.6 GMAC/s に落ちた)。ここだけ先読みしない。
    const PF_ROWS: usize = 0;

    let nb = n_in / 32;
    let rb = nb * BLK;
    let bp = blocks.as_ptr();
    let xp = act.q.as_ptr();
    let scp = act.sc.as_ptr();
    let fixp = act.q8_fix.as_ptr();
    let bias = _mm256_set1_epi8(-128);

    for r in 0..n_out {
        let row = bp.add(r * rb);
        let mut acc = _mm256_setzero_ps();
        for b in 0..nb {
            let p = row.add(b * BLK);
            let d = half::f16::from_bits(std::ptr::read_unaligned(p as *const u16)).to_f32();
            let q = _mm256_loadu_si256(p.add(2) as *const __m256i);
            let qu = _mm256_xor_si256(q, bias);
            let xv = _mm256_loadu_si256(xp.add(b * 32) as *const __m256i);
            let dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qu, xv);
            let dp = _mm256_sub_epi32(dp, _mm256_loadu_si256(fixp.add(b * 8) as *const __m256i));
            let sd = *scp.add(b);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(d * sd), acc);
        }
        prefetch_rows_ahead(row, rb, r, n_out, PF_ROWS);
        *out.get_unchecked_mut(r) = hsum8_avx2(acc);
    }
}

/// このプロセスで M=1 (decode) の VNNI GEMV を使ってよいか。
///
/// 既定は無効。`DENNO_GEMV_VNNI=1` のときだけ有効 (判定はプロセスで 1 回)。
/// prefill のバッチ経路 (`DENNO_VNNI`) とは別のスイッチにしてある
/// (decode と prefill を独立に A/B できるようにするため)。
pub fn gemv_vnni_enabled() -> bool {
    #[cfg(test)]
    {
        match GEMV_VNNI_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => return false,
            1 => return vnni_supported(),
            _ => {}
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        vnni_supported()
            && matches!(
                std::env::var("DENNO_GEMV_VNNI").ok().as_deref(),
                Some("1")
            )
    })
}

/// テスト用: M=1 VNNI のゲートを上書きする (`None` で既定に戻す)。
///
/// プロセス全体の状態なので、[`gemv_vnni_guard`] (テスト内) を取ったテストだけが呼ぶこと。
#[cfg(test)]
pub(crate) fn set_gemv_vnni_for_test(on: Option<bool>) {
    let v = match on {
        None => -1,
        Some(false) => 0,
        Some(true) => 1,
    };
    GEMV_VNNI_TEST_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
static GEMV_VNNI_TEST_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// テスト用: M=1 VNNI のゲートを開け、Drop で必ず既定へ戻す (panic でも戻る)。
#[cfg(test)]
pub(crate) struct VnniGateGuard;

#[cfg(test)]
impl VnniGateGuard {
    pub(crate) fn on() -> Self {
        set_gemv_vnni_for_test(Some(true));
        VnniGateGuard
    }
}

#[cfg(test)]
impl Drop for VnniGateGuard {
    fn drop(&mut self) {
        set_gemv_vnni_for_test(None);
    }
}

/// テスト用: MoE の forward と VNNI のゲートを直列化する共有ロック。
///
/// ゲートはプロセス全体の状態なので、**ゲートを開くテストは forward を回すテストと
/// 同じミューテックスを取る**必要がある (取らないと、par/seq のビット一致を
/// 前後 2 回の呼び出しで見ているテストの間にゲートが開いて落ちる)。
/// `qwen4exp` 側の `moe_test_guard` もここを指す。
#[cfg(test)]
pub(crate) fn moe_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match L.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

/// テスト用: ゲートを触るテストの直列化 ([`moe_test_lock`] と同じミューテックス)。
#[cfg(test)]
pub(crate) fn gemv_vnni_test_guard() -> std::sync::MutexGuard<'static, ()> {
    moe_test_lock()
}

/// テスト用: M=1 VNNI カーネルが発火した回数 (プロセス全体)。
#[cfg(test)]
static GEMV_VNNI_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn gemv_vnni_fired() {
    #[cfg(test)]
    GEMV_VNNI_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// テスト用: M=1 VNNI カーネルが発火した累計回数。
#[cfg(test)]
pub(crate) fn gemv_vnni_calls() -> usize {
    GEMV_VNNI_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

// 活性の量子化を 1 回だけ行うためのスクラッチ (スレッドごとに使い回す)。
// GEMV は 1 トークンで 1440 回呼ばれるので、呼び出しごとに `Vec` を 5 本
// 確保しない (行並列のワーカーも同じスレッドのスクラッチを再利用する)。
thread_local! {
    static GEMV_VNNI_ACT: std::cell::RefCell<VnniActM1> =
        std::cell::RefCell::new(VnniActM1::default());
}

/// Flash-Next の decode (M=1) 用 VNNI GEMV。使えたときだけ `true` を返す。
///
/// - 対象は Q4_K / Q5_1 / Q8_0。`n_in` は Q4_K が 256 の倍数、Q5_1 / Q8_0 が 32 の倍数。
/// - `DENNO_GEMV_VNNI=1` が無い / AVX-512 VNNI が無い / 対象外の型・形 → `Ok(false)`
///   (呼び出し側は [`expert_gemv_any`] へ落とす)。
/// - [`expert_gemv_any`] の既定動作は変えない (27B のビット一致契約に触らない)。
pub fn expert_gemv_any_vnni(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    expert_gemv_any_vnni_inner(ty, blocks, n_in, n_out, x, out, true)
}

/// [`expert_gemv_any_vnni`] の**行並列を使わない**版。
///
/// [`expert_gemv_any_seq`] と同じ用途 (外側で expert を並列化しているとき)。
/// 行は独立なので、行の分割の有無で各要素の計算順序は変わらない
/// (どちらの入口も同じカーネル・同じ活性量子化を通るので結果は一致する)。
pub fn expert_gemv_any_vnni_seq(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<bool> {
    expert_gemv_any_vnni_inner(ty, blocks, n_in, n_out, x, out, false)
}

fn expert_gemv_any_vnni_inner(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
    par: bool,
) -> Result<bool> {
    // ゲートを最初に見る (既定 off のときは引数検証もせずに f32 経路へ返す)。
    if !gemv_vnni_enabled() {
        return Ok(false);
    }
    match ty {
        GgmlType::Q4K if n_in % 256 == 0 && n_in > 0 => {}
        GgmlType::Q5_1 | GgmlType::Q8_0 if n_in % 32 == 0 && n_in > 0 => {}
        _ => return Ok(false),
    }
    let rb = check_gemv_args(ty, blocks, n_in, n_out, x, out)?;

    #[cfg(target_arch = "x86_64")]
    {
        GEMV_VNNI_ACT.with(|cell| {
            let mut act = cell.borrow_mut();
            act.quantize(x)?;
            gemv_vnni_fired();
            // クロージャには `&VnniActM1` だけを渡す (RefMut を持たせると !Sync)。
            let a: &VnniActM1 = &act;
            // SAFETY: check_gemv_args が blocks/x/out の長さと n_in の粒度を検証済みで、
            // 直前の n_in % 256 / % 32 の確認と vnni_supported() (gemv_vnni_enabled が
            // 見ている) がカーネルの前提を満たす。行は独立なので分割しても値は変わらない。
            for_each_row_chunk(blocks, n_out, rb, out, par, |b, o| unsafe {
                match ty {
                    GgmlType::Q4K => gemv_q4k_vnni(b, n_in, o.len(), a, o),
                    GgmlType::Q5_1 => gemv_q5_1_vnni(b, n_in, o.len(), a, o),
                    GgmlType::Q8_0 => gemv_q8_0_vnni(b, n_in, o.len(), a, o),
                    _ => unreachable!("対象外の型は上で弾いている"),
                }
            });
            Ok(true)
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (rb, par);
        Ok(false)
    }
}

// ===========================================================================
// 可変 m の grouped GEMM: タスク表 (Segment) と CPU 参照実装
// ===========================================================================
//
// 1 層ぶんの expert 呼び出しを「expert ごとの (m, 入力の範囲, 出力の範囲)」の列に
// ならしたもの。GPU 側 (重みを都度転送して可変 m の GEMM を回すカーネル) に渡す
// 入力形式と、その正解データを出す CPU 参照実装をここに置く。推論経路には
// 接続しない (既存の flat / legacy 経路の数値は 1 ビットも変えない)。
//
// - `Segment` は GPU の引数にそのまま載る固定幅 (`u16` / `u32`)。
//   オフセットは **f32 要素単位** (バイトではない)。
// - 並びは **m の降順** (同 m は expert 昇順)。GPU 側のブロック割当で重い仕事が
//   先に始まるようにするため。segment は互いに素な範囲を書くだけなので、
//   並び順は値に影響しない。
// - `xs` は expert ごとにトークンを詰め直した入力 (m_total × n_in)、
//   `outs` は同じ行順の出力 (m_total × n_out)。行 → 元トークンの対応は
//   呼び出し側の gather 索引 (`by_expert[expert][i].0`) と同じ。

/// 1 層ぶんの expert 呼び出し 1 本 (可変 m の grouped GEMM の 1 単位)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// 重みの何番目の expert か。
    pub expert: u16,
    /// この expert に割り当たったトークン数。
    pub m: u32,
    /// `xs` の先頭 (f32 要素単位)。範囲は `m * n_in`。
    pub x_off: u32,
    /// `outs` の先頭 (f32 要素単位)。範囲は `m * n_out`。
    pub out_off: u32,
}

/// [`build_grouped_batch`] の結果 (タスク表 + 詰め直した入力)。
#[derive(Debug, Clone)]
pub struct GroupedBatch {
    /// m の降順 (同 m は expert 昇順) のタスク表。
    pub segs: Vec<Segment>,
    /// expert ごとに詰め直した入力 (`m_total × n_in`、行優先)。
    pub xs: Vec<f32>,
    /// `xs` / `outs` の行数 (= Σ m)。
    pub m_total: usize,
}

impl GroupedBatch {
    /// `outs` に必要な要素数 (`m_total * n_out`)。
    pub fn outs_len(&self, n_out: usize) -> usize {
        self.m_total * n_out
    }
}

/// gather 索引から segment 表と詰め直した入力を作る。
///
/// `by_expert[expert]` は既存の flat 経路と同じ形 (`(トークン, ルータのスロット)`
/// の列、expert 内はトークン昇順)。`xs` はトークン優先の入力 (`nt × n_in`)。
///
/// 行の詰め方は既存の gather と同一 (expert 内はトークン昇順) だが、segment を
/// **m の降順** に並べるので `xs` の行順もその並びになる。`outs` は呼び出し側が
/// `outs_len(n_out)` 要素で確保する (中身は [`grouped_expert_gemm_cpu`] が全部書く)。
pub fn build_grouped_batch(
    by_expert: &[Vec<(usize, usize)>],
    xs: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<GroupedBatch> {
    if n_in == 0 {
        bail!("build_grouped_batch: n_in が 0");
    }
    if n_out == 0 {
        bail!("build_grouped_batch: n_out が 0");
    }
    if xs.len() % n_in != 0 {
        bail!(
            "build_grouped_batch: xs の長さ {} が n_in={} で割り切れない",
            xs.len(),
            n_in
        );
    }
    let nt = xs.len() / n_in;

    // 1. 空でない expert と件数を集める。
    let mut active: Vec<(usize, usize)> = Vec::new(); // (expert, m)
    let mut m_total = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        for &(t, _) in picks {
            if t >= nt {
                bail!(
                    "build_grouped_batch: expert {ei} のトークン {t} が範囲外 (nt={nt})"
                );
            }
        }
        active.push((ei, picks.len()));
        m_total += picks.len();
    }

    // 2. m の降順・同 m は expert 昇順 (決定的な並び)。
    active.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // 3. オフセットを詰めながら gather する (行は segment の並びに従う)。
    let mut segs: Vec<Segment> = Vec::with_capacity(active.len());
    let mut gx: Vec<f32> = vec![0.0; m_total * n_in];
    let (mut xo, mut oo) = (0usize, 0usize);
    for (ei, m) in active {
        let x_off = u32::try_from(xo)
            .map_err(|_| anyhow::anyhow!("build_grouped_batch: x_off {xo} が u32 を超える"))?;
        let out_off = u32::try_from(oo)
            .map_err(|_| anyhow::anyhow!("build_grouped_batch: out_off {oo} が u32 を超える"))?;
        let expert = u16::try_from(ei)
            .map_err(|_| anyhow::anyhow!("build_grouped_batch: expert {ei} が u16 を超える"))?;
        let m32 = u32::try_from(m)
            .map_err(|_| anyhow::anyhow!("build_grouped_batch: m {m} が u32 を超える"))?;
        for (i, &(t, _)) in by_expert[ei].iter().enumerate() {
            let dst = xo + i * n_in;
            gx[dst..dst + n_in].copy_from_slice(&xs[t * n_in..(t + 1) * n_in]);
        }
        segs.push(Segment {
            expert,
            m: m32,
            x_off,
            out_off,
        });
        xo += m * n_in;
        oo += m * n_out;
    }
    // 4. 合計の検算 (Σ m == m_total、x/out の範囲が総量に一致)。
    if xo != m_total * n_in || oo != m_total * n_out {
        bail!(
            "build_grouped_batch: 合計が合わない (Σm={m_total}, x={xo}/{}, out={oo}/{})",
            m_total * n_in,
            m_total * n_out
        );
    }
    Ok(GroupedBatch {
        segs,
        xs: gx,
        m_total,
    })
}

/// segment 表の整合性を検証し、`m_total` を返す。
///
/// 契約: `x_off` / `out_off` は 0 から**隙間なく連続**し (重複・飛びなし)、
/// 最後の segment の終端が `xs_len` / `outs_len` に一致する。
/// `m == 0` の segment は認めない (GPU 側で空タスクを回さない)。
pub fn check_segments(
    segs: &[Segment],
    n_in: usize,
    n_out: usize,
    xs_len: usize,
    outs_len: usize,
) -> Result<usize> {
    let (mut xo, mut oo, mut m_total) = (0usize, 0usize, 0usize);
    for (i, s) in segs.iter().enumerate() {
        if s.m == 0 {
            bail!("segment {i}: m が 0");
        }
        if s.x_off as usize != xo {
            bail!(
                "segment {i}: x_off {} が連続でない (期待 {xo})",
                s.x_off
            );
        }
        if s.out_off as usize != oo {
            bail!(
                "segment {i}: out_off {} が連続でない (期待 {oo})",
                s.out_off
            );
        }
        xo += s.m as usize * n_in;
        oo += s.m as usize * n_out;
        m_total += s.m as usize;
    }
    if xo != xs_len {
        bail!("segment の x 範囲の合計 {xo} が xs の長さ {xs_len} と違う");
    }
    if oo != outs_len {
        bail!("segment の out 範囲の合計 {oo} が outs の長さ {outs_len} と違う");
    }
    Ok(m_total)
}

/// CPU 参照実装 (GPU カーネルの正解データ)。
///
/// Segment ごとに「expert の重み (量子化ブロック) を 1 行ずつ f32 に開いて、
/// `xs` の m 行と内積を取る」。`weights[expert]` は [`ExpertSlice`] で切り出した
/// 1 expert ぶんのバイト列 (`n_out * row_bytes(ty, n_in)`)。
///
/// # ビット一致の根拠
///
/// 行のデコードは [`crate::kquant::dequant_row_into`] (中身は [`crate::kquant::dequant_block`]
/// を 1 ブロックずつ呼ぶだけ)、内積は [`dot_seq`] (`0..n_in` の昇順に
/// `acc += w[i] * x[i]`)。これは [`expert_gemm_scalar`] の
/// `gemm_rows_scalar` と同じ演算列 (ブロック昇順・ブロック内要素昇順の逐次加算)
/// なので、**丸めの順序が変わらずビット一致**する。
///
/// 行 (`n_out`) とトークンのループは segment 内で独立、segment 同士も互いに素な
/// 範囲を書くので、どの順に回しても値は変わらない。
///
/// 注意: 本体の推論で使う融合カーネル ([`gemm_rows`] の AVX2 版) は
/// レーン累積のため**この参照とは丸め順が違う** (相対 1e-6 程度)。GPU との
/// 比較相手は、モジュール冒頭の契約どおりスカラー経路 (= この関数) にする。
#[allow(clippy::too_many_arguments)]
pub fn grouped_expert_gemm_cpu(
    ty: GgmlType,
    weights: &[&[u8]],
    n_in: usize,
    n_out: usize,
    segs: &[Segment],
    xs: &[f32],
    outs: &mut [f32],
) -> Result<()> {
    let rb = row_bytes(ty, n_in)?;
    check_segments(segs, n_in, n_out, xs.len(), outs.len())?;
    grouped_range_cpu(ty, rb, n_in, n_out, weights, segs, xs, 0, outs)
}

/// [`grouped_expert_gemm_cpu`] の本体 (segment の部分列を回す)。
///
/// `outs` は全体の `out_base` 要素目から始まる部分スライス。並列化するときは
/// segment の列を切り、その先頭の `out_off` を `out_base` にして渡す
/// (segment の範囲は連続なので、切り口は必ず重ならない部分スライスになる)。
#[allow(clippy::too_many_arguments)]
fn grouped_range_cpu(
    ty: GgmlType,
    rb: usize,
    n_in: usize,
    n_out: usize,
    weights: &[&[u8]],
    segs: &[Segment],
    xs: &[f32],
    out_base: usize,
    outs: &mut [f32],
) -> Result<()> {
    let want_b = n_out
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={n_out})"))?;
    let mut w = vec![0.0f32; n_in];
    for (i, seg) in segs.iter().enumerate() {
        let ei = seg.expert as usize;
        let Some(blocks) = weights.get(ei) else {
            bail!("segment {i}: expert {ei} の重みが無い (weights={})", weights.len());
        };
        if blocks.len() != want_b {
            bail!(
                "segment {i}: expert {ei} の重みの長さ {} が n_out * row_bytes = {want_b} と違う",
                blocks.len()
            );
        }
        let m = seg.m as usize;
        let x0 = seg.x_off as usize;
        let o0 = seg.out_off as usize - out_base;
        for r in 0..n_out {
            dequant_row_into(ty, &blocks[r * rb..(r + 1) * rb], &mut w);
            for t in 0..m {
                let xb = &xs[x0 + t * n_in..x0 + (t + 1) * n_in];
                outs[o0 + t * n_out + r] = dot_seq(&w, xb);
            }
        }
    }
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

    // ---- 7. バッチ GEMM (H3) -----------------------------------------------

    /// m トークンぶんの入力 (`xs[t*n_in + i]`)。トークンごとに値の列を変える
    /// (トークンと行の取り違え、行の使い回しを検出するため)。
    fn make_xs(m: usize, n_in: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(m * n_in);
        for t in 0..m {
            for i in 0..n_in {
                let k = (i * 37 + t * 911) % 101;
                v.push((k as f32 - 50.0) / 64.0);
            }
        }
        v
    }

    /// 参照: `expert_gemv_any` を m 回呼んだ結果を、GEMM と同じ並び (`t*n_out + r`) に詰める。
    fn gemv_any_m_times(
        ty: GgmlType,
        blocks: &[u8],
        n_in: usize,
        n_out: usize,
        xs: &[f32],
        m: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n_out];
        let mut one = vec![0.0f32; n_out];
        for t in 0..m {
            one.iter_mut().for_each(|v| *v = 0.0);
            expert_gemv_any(ty, blocks, n_in, n_out, &xs[t * n_in..(t + 1) * n_in], &mut one)
                .expect("gemv_any");
            out[t * n_out..(t + 1) * n_out].copy_from_slice(&one);
        }
        out
    }

    #[test]
    fn gemm_matches_m_times_gemv_any_bits() {
        // 本体の契約: `outs[t*n_out + r]` が 1 行版を m 回呼んだ結果とビット一致。
        // m = 1 (1 行版と同値) / 4 / 17 (タイル外れ) で見る。
        for ty in ALL_TYPES {
            let blocks = make_expert(ty, N_IN, N_OUT, 0);
            for m in [1usize, 4, 17] {
                let xs = make_xs(m, N_IN);
                let mut got = vec![0.0f32; m * N_OUT];
                expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs, m, &mut got)
                    .unwrap_or_else(|e| panic!("{ty:?} m={m}: {e}"));
                let want = gemv_any_m_times(ty, &blocks, N_IN, N_OUT, &xs, m);
                expect_bits(&got, &want);
                // 縮退していないこと (全 0 なら一致しても意味が無い)
                assert!(
                    got.iter().any(|v| *v != 0.0),
                    "{ty:?} m={m}: 出力が全 0 (テストが無意味)"
                );
                assert!(got.iter().all(|v| v.is_finite()), "{ty:?} m={m}: 非有限値");
                // トークンごとに値が違うこと (行の取り違え・共有の検出)
                if m > 1 {
                    let t0 = &got[0..N_OUT];
                    assert!(
                        (1..m).any(|t| got[t * N_OUT..(t + 1) * N_OUT] != *t0),
                        "{ty:?} m={m}: 全トークンが同じ値 (入力を共有している疑い)"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_matches_m_times_gemv_scalar_bits() {
        // スカラー経路 (`dequant_row_into` + 逐次内積) のバッチ本体 `gemm_rows_scalar` を
        // 直接呼び、1 行版のスカラー経路を m 回呼んだ結果とビット一致することを見る。
        // 「デコードを 1 回に括り出す」変形が丸め順を変えていないことの直接確認で、
        // AVX2/FMA が無い環境で実際に使われる経路でもある。
        // (融合カーネルへ振れる型の `expert_gemm_any` は 1 行版の融合経路と一致するので、
        //  そちらは `gemm_matches_m_times_gemv_any_bits` が見る。)
        for ty in ALL_TYPES {
            let rb = row_bytes(ty, N_IN).unwrap();
            let blocks = make_expert(ty, N_IN, N_OUT, 0);
            for m in [1usize, 4, 17] {
                let xs = make_xs(m, N_IN);
                let mut got = vec![0.0f32; m * N_OUT];
                gemm_rows_scalar(ty, rb, &blocks, N_IN, 0, N_OUT, &xs, m, got.as_mut_ptr(), N_OUT);
                let mut want = vec![0.0f32; m * N_OUT];
                for t in 0..m {
                    let mut one = vec![0.0f32; N_OUT];
                    expert_gemv_scalar(
                        ty,
                        &blocks,
                        N_IN,
                        N_OUT,
                        &xs[t * N_IN..(t + 1) * N_IN],
                        &mut one,
                    )
                    .unwrap();
                    want[t * N_OUT..(t + 1) * N_OUT].copy_from_slice(&one);
                }
                expect_bits(&got, &want);
                assert!(got.iter().any(|v| *v != 0.0), "{ty:?} m={m}: 出力が全 0");
            }
        }
    }

    #[test]
    fn gemm_matches_gemv_any_on_real_shape() {
        // 実形状 (640 行 x 2560 列 = 10 ブロック/行) かつ n_out >= PAR_MIN_ROWS なので、
        // 行チャンクの並列経路とチャンク詰めもここで通る。
        for ty in ALL_TYPES {
            let blocks = make_expert(ty, PERF_N_IN, PERF_N_OUT, 0);
            for m in [1usize, 4] {
                let xs = make_xs(m, PERF_N_IN);
                let mut got = vec![0.0f32; m * PERF_N_OUT];
                expert_gemm_any(ty, &blocks, PERF_N_IN, PERF_N_OUT, &xs, m, &mut got).unwrap();
                let want = gemv_any_m_times(ty, &blocks, PERF_N_IN, PERF_N_OUT, &xs, m);
                expect_bits(&got, &want);
                assert!(got.iter().any(|v| *v != 0.0), "{ty:?}: 出力が全 0");
            }
        }
    }

    #[test]
    fn gemm_seq_entry_matches_parallel_entry() {
        // 行の分割の有無は各要素の計算順序を変えない (ビット一致)。
        let n_out = 640;
        assert!(n_out >= PAR_MIN_ROWS, "行並列が分割する大きさで比べる");
        for ty in ALL_TYPES {
            let blocks = make_expert(ty, N_IN, n_out, 0);
            let m = 3usize;
            let xs = make_xs(m, N_IN);
            let mut par = vec![0.0f32; m * n_out];
            let mut seq = vec![0.0f32; m * n_out];
            expert_gemm_any(ty, &blocks, N_IN, n_out, &xs, m, &mut par).unwrap();
            expert_gemm_any_seq(ty, &blocks, N_IN, n_out, &xs, m, &mut seq).unwrap();
            assert!(par.iter().any(|v| *v != 0.0), "{ty:?}: 出力が全 0");
            expect_bits(&par, &seq);
        }
    }

    #[test]
    fn gemm_fused_gate_fires() {
        // 「緑 = 経路が発動した証拠」ではないので、バッチ融合カーネルの発火を数える。
        // 行数が PAR_MIN_ROWS 未満なので rayon は使わず、このスレッドで数えられる。
        if !q8_0_fused_supported() {
            eprintln!("skip gemm_fused_gate_fires: この CPU には AVX2/FMA が無い");
            return;
        }
        assert!(N_OUT < PAR_MIN_ROWS, "ゲートの計数は逐次経路で行う");
        let xs = make_xs(4, N_IN);
        for ty in [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0] {
            let blocks = make_expert(ty, N_IN, N_OUT, 0);
            let mut out = vec![0.0f32; 4 * N_OUT];
            let before = gemm_fused_count();
            expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs, 4, &mut out).unwrap();
            let fired = gemm_fused_count() - before;
            assert!(fired >= 1, "{ty:?}: バッチ融合カーネルが発火していない");
        }
        // 対照: Q5_K は 1 行版もスカラー経路なので、バッチ融合は選ばれない。
        let blocks = make_expert(GgmlType::Q5K, N_IN, N_OUT, 0);
        let mut out = vec![0.0f32; 4 * N_OUT];
        let before = gemm_fused_count();
        expert_gemm_any(GgmlType::Q5K, &blocks, N_IN, N_OUT, &xs, 4, &mut out).unwrap();
        assert_eq!(
            gemm_fused_count(),
            before,
            "Q5_K でバッチ融合カーネルが発火した (スカラー経路のはず)"
        );
        assert!(out.iter().any(|v| *v != 0.0), "Q5_K: 出力が全 0");
    }

    #[test]
    fn gemm_rejects_wrong_lengths() {
        let ty = GgmlType::Q8_0;
        let blocks = make_expert(ty, N_IN, N_OUT, 0);
        let m = 3usize;
        let xs = make_xs(m, N_IN);
        let mut out = vec![0.0f32; m * N_OUT];

        // m = 0 / xs の長さ違い / outs の長さ違い
        assert!(expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs, 0, &mut []).is_err());
        assert!(expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs[..m * N_IN - 1], m, &mut out).is_err());
        let mut short = vec![0.0f32; m * N_OUT - 1];
        assert!(expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs, m, &mut short).is_err());
        // blocks の長さ違い
        assert!(expert_gemm_any(ty, &blocks[..blocks.len() - 1], N_IN, N_OUT, &xs, m, &mut out).is_err());
        // n_out = 0
        assert!(expert_gemm_any(ty, &[], N_IN, 0, &[], 0, &mut []).is_err());
        // 対象外の型 (1 行版と同じ門番)
        assert!(expert_gemm_any(GgmlType::F32, &blocks, N_IN, N_OUT, &xs, m, &mut out).is_err());
        assert!(expert_gemm_any(GgmlType::Q6K, &blocks, N_IN, N_OUT, &xs, m, &mut out).is_err());
        // 正常系が通ること (上の否定だけでは「常に Err」でも緑になる)
        assert!(expert_gemm_any(ty, &blocks, N_IN, N_OUT, &xs, m, &mut out).is_ok());
    }

    // ---- 8. k ブロック版 GEMM (moe_xblk) -------------------------------------

    #[test]
    fn gemm_k_block_choice_matches_l1_budget() {
        // KB の選び方そのものを見る (k チャンクが 1 回で終わると、
        // 「k ブロックを外側に出す」変更が発動した証拠にならない)。
        // Q4_K (256 要素): m=1 は cap の 4、m=15 は 1 ブロック (15,360 B)。
        assert_eq!(gemm_k_blocks(256, 1, 4), 4);
        assert_eq!(gemm_k_blocks(256, 8, 4), 2);
        assert_eq!(gemm_k_blocks(256, 15, 4), 1);
        assert_eq!(gemm_k_blocks(256, 32, 4), 1);
        // Q5_1 (32 要素): m=15 は 8 ブロック (256 要素 = 15,360 B)。
        assert_eq!(gemm_k_blocks(32, 1, 8), 8);
        assert_eq!(gemm_k_blocks(32, 15, 8), 8);
        assert_eq!(gemm_k_blocks(32, 32, 8), 4);
        // 実形状では Q4_K / Q5_1 とも k チャンクが複数回に分かれる。
        assert!(PERF_N_IN / 256 > gemm_k_blocks(256, 15, 4));
        assert!(PERF_N_IN / 32 > gemm_k_blocks(32, 15, 8));
    }

    /// k ブロック版と旧カーネル (参照実装) のビット一致を見る共通部。
    fn expect_kblock_bits(
        ty: GgmlType,
        new_k: unsafe fn(&[u8], usize, usize, usize, &[f32], usize, *mut f32, usize),
        ref_k: unsafe fn(&[u8], usize, usize, usize, &[f32], usize, *mut f32, usize),
        ms: &[usize],
    ) -> usize {
        let n_in = PERF_N_IN;
        let n_out = PERF_N_OUT;
        let blocks = make_expert(ty, n_in, n_out, 0);
        let mut compared = 0usize;
        for &m in ms {
            let xs = make_xs(m, n_in);
            let mut got = vec![0.0f32; m * n_out];
            let mut want = vec![0.0f32; m * n_out];
            // SAFETY: 形状は check_gemm_args と同じ条件を満たし、AVX2/FMA は
            // 呼び出し側が確認している。
            unsafe { new_k(&blocks, n_in, 0, n_out, &xs, m, got.as_mut_ptr(), n_out) };
            unsafe { ref_k(&blocks, n_in, 0, n_out, &xs, m, want.as_mut_ptr(), n_out) };
            expect_bits(&got, &want);
            assert!(
                got.iter().any(|v| *v != 0.0),
                "{ty:?} m={m}: 出力が全 0 (テストが無意味)"
            );
            assert!(got.iter().all(|v| v.is_finite()), "{ty:?} m={m}: 非有限値");
            compared += 1;
        }
        // 行範囲を途中から取る経路 (rayon の行チャンクと同じ形) も見る。
        let m = 15usize;
        let (r0, r1) = (137usize, 381usize);
        let xs = make_xs(m, n_in);
        let mut got = vec![0.0f32; m * (r1 - r0)];
        let mut want = vec![0.0f32; m * (r1 - r0)];
        unsafe { new_k(&blocks, n_in, r0, r1, &xs, m, got.as_mut_ptr(), r1 - r0) };
        unsafe { ref_k(&blocks, n_in, r0, r1, &xs, m, want.as_mut_ptr(), r1 - r0) };
        expect_bits(&got, &want);
        compared += 1;
        compared
    }

    #[test]
    fn gemm_kblock_q4k_matches_ref_bits() {
        if !q8_0_fused_supported() {
            eprintln!("skip gemm_kblock_q4k_matches_ref_bits: AVX2/FMA が無い");
            return;
        }
        // 実形状 (640 行 x 2560 列 = 10 ブロック/行) で m = 1 / 3 / 15 / 16 / 33。
        // m=15 は KB=1 (k チャンク 10 回)、m=1 は KB=4 (3 回) を通る。
        let n = expect_kblock_bits(
            GgmlType::Q4K,
            gemm_q4k_avx2,
            gemm_q4k_avx2_ref,
            &[1, 3, 15, 16, 33],
        );
        eprintln!("gemm_kblock_q4k_matches_ref_bits: {n} 形状がビット一致");
        assert!(n >= 6);
    }

    #[test]
    fn gemm_kblock_q5_1_matches_ref_bits() {
        if !q8_0_fused_supported() {
            eprintln!("skip gemm_kblock_q5_1_matches_ref_bits: AVX2/FMA が無い");
            return;
        }
        let n = expect_kblock_bits(
            GgmlType::Q5_1,
            gemm_q5_1_avx2,
            gemm_q5_1_avx2_ref,
            &[1, 3, 15, 16, 33],
        );
        eprintln!("gemm_kblock_q5_1_matches_ref_bits: {n} 形状がビット一致");
        assert!(n >= 6);
    }

    #[test]
    fn gemm_kblock_through_expert_gemm_any_bits() {
        // 公開入口から呼んでも (行並列あり / なし) 旧カーネルとビット一致すること。
        // `expert_gemm_any` は n_out >= PAR_MIN_ROWS で rayon 分割するので、
        // 行チャンク版の経路がここでも通る。
        if !q8_0_fused_supported() {
            eprintln!("skip gemm_kblock_through_expert_gemm_any_bits: AVX2/FMA が無い");
            return;
        }
        assert!(PERF_N_OUT >= PAR_MIN_ROWS);
        for ty in [GgmlType::Q4K, GgmlType::Q5_1] {
            let n_in = PERF_N_IN;
            let n_out = PERF_N_OUT;
            let blocks = make_expert(ty, n_in, n_out, 0);
            for m in [1usize, 15] {
                let xs = make_xs(m, n_in);
                let mut got = vec![0.0f32; m * n_out];
                let mut want = vec![0.0f32; m * n_out];
                expert_gemm_any(ty, &blocks, n_in, n_out, &xs, m, &mut got).unwrap();
                unsafe {
                    match ty {
                        GgmlType::Q4K => {
                            gemm_q4k_avx2_ref(
                                &blocks,
                                n_in,
                                0,
                                n_out,
                                &xs,
                                m,
                                want.as_mut_ptr(),
                                n_out,
                            )
                        }
                        GgmlType::Q5_1 => {
                            gemm_q5_1_avx2_ref(
                                &blocks,
                                n_in,
                                0,
                                n_out,
                                &xs,
                                m,
                                want.as_mut_ptr(),
                                n_out,
                            )
                        }
                        _ => unreachable!(),
                    }
                }
                expect_bits(&got, &want);
                assert!(got.iter().any(|v| *v != 0.0), "{ty:?} m={m}: 出力が全 0");
            }
        }
    }

    /// マイクロベンチ (CPU だけで完結)。旧カーネルと k ブロック版を同じ
    /// ハーネスで測る。実行:
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_gemm_kblock`
    #[test]
    #[ignore = "マイクロベンチ (500 回 x 4 形状 x 2 カーネル x 2 型)"]
    fn bench_gemm_kblock_vs_ref_gmacs() {
        use std::hint::black_box;
        use std::time::Instant;
        if !q8_0_fused_supported() {
            eprintln!("skip bench_gemm_kblock_vs_ref_gmacs: AVX2/FMA が無い");
            return;
        }
        const ITERS: usize = 500;
        const WARM: usize = 20;
        let n_in = PERF_N_IN;
        let n_out = PERF_N_OUT;

        macro_rules! run {
            ($label:expr, $k:expr, $m:expr, $blocks:expr, $xs:expr, $out:expr) => {{
                let k: unsafe fn(&[u8], usize, usize, usize, &[f32], usize, *mut f32, usize) = $k;
                for _ in 0..WARM {
                    unsafe {
                        k(
                            black_box(&$blocks),
                            n_in,
                            0,
                            n_out,
                            black_box(&$xs),
                            $m,
                            $out.as_mut_ptr(),
                            n_out,
                        )
                    };
                }
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    unsafe {
                        k(
                            black_box(&$blocks),
                            n_in,
                            0,
                            n_out,
                            black_box(&$xs),
                            $m,
                            $out.as_mut_ptr(),
                            n_out,
                        )
                    };
                }
                let dt = t0.elapsed().as_secs_f64();
                let macs = ($m * n_in * n_out * ITERS) as f64;
                eprintln!(
                    "{:<20} m={:<3} {:8.3} ms/回  {:7.1} GMAC/s (1 スレッド)",
                    $label,
                    $m,
                    dt * 1e3 / ITERS as f64,
                    macs / dt / 1e9
                );
                black_box(&$out);
            }};
        }

        for m in [1usize, 8, 15, 32] {
            let xs = make_xs(m, n_in);
            let mut out = vec![0.0f32; m * n_out];
            let q4 = make_expert(GgmlType::Q4K, n_in, n_out, 0);
            let q51 = make_expert(GgmlType::Q5_1, n_in, n_out, 0);
            run!("Q4_K 旧 (ref)", gemm_q4k_avx2_ref, m, q4, xs, out);
            run!("Q4_K 新 (k block)", gemm_q4k_avx2, m, q4, xs, out);
            run!("Q5_1 旧 (ref)", gemm_q5_1_avx2_ref, m, q51, xs, out);
            run!("Q5_1 新 (k block)", gemm_q5_1_avx2, m, q51, xs, out);
        }
    }

    /// 行チャンクごとの確保・ゼロ埋め・コピーを消した効果を、実形状
    /// (n_in=2560, n_out=640, m=60, Q4_K) の 1 呼び出しあたり µs で見る。
    ///
    /// 実行:
    /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_gemm_chunk_alloc`
    #[test]
    #[ignore = "マイクロベンチ (実形状 640x2560, Q4_K, m=60)"]
    fn bench_gemm_chunk_alloc_q4k_m60() {
        use std::hint::black_box;
        use std::time::Instant;
        if !q8_0_fused_supported() {
            eprintln!("skip bench_gemm_chunk_alloc_q4k_m60: AVX2/FMA が無い");
            return;
        }
        const ITERS: usize = 100;
        const WARM: usize = 10;
        const REPS: usize = 5;
        let n_in = PERF_N_IN;
        let n_out = PERF_N_OUT;
        let m = 60usize;
        assert!(n_out >= PAR_MIN_ROWS, "行並列が分割する大きさで測る");
        let blocks = make_expert(GgmlType::Q4K, n_in, n_out, 0);
        let xs = make_xs(m, n_in);
        let mut out = vec![0.0f32; m * n_out];
        let ranges = gemm_row_ranges(n_out, true);
        eprintln!(
            "threads={} ranges={} rows/range={}",
            rayon::current_num_threads(),
            ranges.len(),
            ranges[0].1 - ranges[0].0
        );
        for _ in 0..WARM {
            expert_gemm_any(GgmlType::Q4K, &blocks, n_in, n_out, &xs, m, &mut out).unwrap();
        }
        let mut best = f64::INFINITY;
        let mut worst = 0f64;
        for _ in 0..REPS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                expert_gemm_any(
                    GgmlType::Q4K,
                    black_box(&blocks),
                    n_in,
                    n_out,
                    black_box(&xs),
                    m,
                    black_box(&mut out),
                )
                .unwrap();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            best = best.min(us);
            worst = worst.max(us);
        }
        let macs = (m * n_in * n_out) as f64;
        eprintln!(
            "bench_gemm_chunk_alloc_q4k_m60: {best:.1} us/回 (min, max={worst:.1})  {:.1} GMAC/s",
            macs / (best * 1e-6) / 1e9
        );
        black_box(&out);
    }

    // ---- VNNI (AVX-512) 経路 ------------------------------------------------

    /// 活性を量子化して (`q`, `sc`, `gs`) を確保する (テスト用)。
    fn make_act(xs: &[f32], m: usize, n_in: usize, blk: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0i8; m * n_in];
        let mut sc = vec![0f32; m * (n_in / blk)];
        let mut gs = vec![0f32; m * (n_in / 32)];
        quant_act_i8(xs, m, n_in, blk, &mut q, &mut sc, &mut gs).expect("quant");
        (q, sc, gs)
    }

    /// カーネルが計算する式の f64 参照実装。
    ///
    /// `Σ_i (w_i の量子化式) * (q_i * xd_b)` を f64 で足す。ただし最小値項 (Q4_K の
    /// `-dmin*m`、Q5_1 の `+m`) は **量子化前の `Σ x`** を使う (カーネルと同じ)。
    /// ここが一致すれば、nibble / 5 ビット目の並び・スケールの割り当て・グループの
    /// 対応は正しい (差は f32 の累積順だけになる)。
    fn ref_kernel_formula(
        ty: GgmlType,
        blocks: &[u8],
        n_in: usize,
        n_out: usize,
        q: &[i8],
        sc: &[f32],
        gs: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let rb = row_bytes(ty, n_in).unwrap();
        let ts = ty.type_size();
        let blk = vnni_act_block(ty).unwrap();
        let nb = n_in / blk;
        let ng32 = n_in / 32;
        let mut out = vec![0f32; n_out];
        let xq = |i: usize| q[t * n_in + i] as f64;
        for (r, o) in out.iter_mut().enumerate() {
            let row = &blocks[r * rb..(r + 1) * rb];
            let mut acc = 0.0f64;
            for b in 0..nb {
                let xd = sc[t * nb + b] as f64;
                let bl = &row[b * ts..(b + 1) * ts];
                let d = half::f16::from_le_bytes([bl[0], bl[1]]).to_f64();
                match ty {
                    GgmlType::Q4K => {
                        let dmin = half::f16::from_le_bytes([bl[2], bl[3]]).to_f64();
                        let scq = &bl[4..16];
                        for j in 0..4usize {
                            for half_i in 0..2usize {
                                let g = 2 * j + half_i;
                                let (sg, mg) = scale_min_k4(g, scq);
                                let mut dot = 0.0f64;
                                for i in 0..32usize {
                                    let byte = bl[16 + j * 32 + i];
                                    let qn =
                                        if half_i == 0 { byte & 0x0f } else { byte >> 4 } as f64;
                                    dot += qn * xq(b * 256 + j * 64 + half_i * 32 + i);
                                }
                                acc += xd * d * (sg as f64) * dot
                                    - dmin * (mg as f64) * gs[t * ng32 + b * 8 + g] as f64;
                            }
                        }
                    }
                    GgmlType::Q5_1 => {
                        let mc = half::f16::from_le_bytes([bl[2], bl[3]]).to_f64();
                        let qh = u32::from_le_bytes([bl[4], bl[5], bl[6], bl[7]]);
                        let mut dot = 0.0f64;
                        for j in 0..16usize {
                            let byte = bl[8 + j];
                            let lo = ((byte & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as f64;
                            let hi = (((byte >> 4) as u32) | (((qh >> (16 + j)) & 1) << 4)) as f64;
                            dot += lo * xq(b * 32 + j) + hi * xq(b * 32 + 16 + j);
                        }
                        acc += xd * d * dot + mc * gs[t * ng32 + b] as f64;
                    }
                    _ => unreachable!(),
                }
            }
            *o = acc as f32;
        }
        out
    }

    /// `quant_act_i8` のスケール・丸め・グループ和を手計算と突き合わせる。
    #[test]
    fn quant_act_i8_scales_rounding_and_groups() {
        // 第 1 ブロック (32 要素) は両端だけ ±1.0 → scale = 1/127、両端は ±127 に張り付く。
        // 第 2 ブロックは全部 0 → scale 1.0・q は全部 0。
        let mut xs = vec![0.0f32; 64];
        xs[0] = -1.0;
        xs[31] = 1.0;
        let (q, sc, gs) = make_act(&xs, 1, 64, 32);
        assert!((sc[0] - 1.0 / 127.0).abs() < 1e-9, "scale {}", sc[0]);
        assert_eq!(q[0], -127, "最小値は -127 に張り付く");
        assert_eq!(q[31], 127, "最大値は 127 に張り付く");
        assert!(q[1..31].iter().all(|&v| v == 0), "0 は 0 のまま");
        assert_eq!(sc[1], 1.0, "全部 0 のブロックのスケールは 1.0");
        assert!(q[32..].iter().all(|&v| v == 0));
        // gs は量子化前の f32 の和 (第 1 ブロックは -1 + 1 = 0)
        assert!(gs[0].abs() < 1e-6, "gs[0]={}", gs[0]);
        assert_eq!(gs[1], 0.0);

        // round-to-nearest: max=1.0 なので q = round(x * 127)
        let mut x2 = vec![0.0f32; 32];
        x2[1] = 1.0;
        x2[2] = -1.0;
        x2[3] = 0.6;
        x2[4] = 0.4;
        x2[5] = 0.0039; // 127 * 0.0039 = 0.4953 → 0
        let (q2, _, gs2) = make_act(&x2, 1, 32, 32);
        assert_eq!(
            &q2[..6],
            &[0, 127, -127, 76, 51, 0],
            "丸めが round-to-nearest でない"
        );
        assert!((gs2[0] - x2.iter().sum::<f32>()).abs() < 1e-6);
    }

    /// VNNI 経路が実際に発火し、f32 融合カーネルと許容誤差で一致する。
    ///
    /// 誤差の出所は活性の int8 量子化だけ (重みは量子化しない) なので、
    /// 相対誤差は 1 ブロックの量子化刻み (`max|x|/127`) の数倍に収まる。
    #[test]
    fn vnni_gemm_matches_fused_kernels_within_tolerance() {
        if !vnni_supported() {
            eprintln!("skip vnni_gemm_matches_fused_kernels_within_tolerance: avx512vnni が無い");
            return;
        }
        for (ty, n_in, n_out, blk) in [
            (GgmlType::Q4K, 512usize, 16usize, 256usize),
            (GgmlType::Q5_1, 64usize, 512usize, 32usize),
        ] {
            let m = 5usize;
            let blocks = make_expert(ty, n_in, n_out, 0);
            let rb = row_bytes(ty, n_in).unwrap();
            let xs: Vec<f32> = (0..m * n_in)
                .map(|i| ((i * 53 % 211) as f32 - 105.0) / 128.0)
                .collect();
            let (q, sc, gs) = make_act(&xs, m, n_in, blk);
            let act = VnniAct {
                q: &q,
                sc: &sc,
                gs: &gs,
                m,
                n_in,
                blk,
            };
            let mut got = vec![0f32; m * n_out];
            let before = vnni_calls();
            let used =
                gemm_rows_vnni(ty, &blocks, n_in, 0, n_out, &act, got.as_mut_ptr(), n_out);
            assert!(used, "{ty:?}: VNNI 経路が使われていない");
            assert!(
                vnni_calls() > before,
                "{ty:?}: VNNI カーネルが発火していない (ゲートの通過を数える)"
            );

            let mut want = vec![0f32; m * n_out];
            gemm_rows(
                ty,
                rb,
                &blocks,
                n_in,
                0,
                n_out,
                &xs,
                m,
                want.as_mut_ptr(),
                n_out,
                true,
            );

            assert!(
                got.iter().any(|v| *v != 0.0),
                "{ty:?}: 出力が全 0 (経路の生死が分からない)"
            );

            // (a) 量子化した式そのものとの一致 (カーネルの数式・並びの検証)。
            for t in 0..m {
                let want_q = ref_kernel_formula(ty, &blocks, n_in, n_out, &q, &sc, &gs, t);
                let g = &got[t * n_out..(t + 1) * n_out];
                let s_q = want_q.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
                let rel_q = g
                    .iter()
                    .zip(&want_q)
                    .map(|(a, b)| (a - b).abs() / s_q)
                    .fold(0f32, f32::max);
                eprintln!("{ty:?} t={t}: カーネルの式との差 (f64 基準) {rel_q:.3e}");
                // f32 の累積順の違いだけで 1e-4 台まで出る (キャンセルが大きい形)。
                // 式や並びが違えば O(1) になるので、1e-3 で十分に切り分けられる。
                assert!(
                    rel_q < 1e-3,
                    "{ty:?} t={t}: カーネルの式と一致しない (rel {rel_q:.3e}) — 式か並びが違う"
                );
            }
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let max_rel = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs() / scale)
                .fold(0f32, f32::max);
            let mean_rel = got
                .iter()
                .zip(&want)
                .map(|(g, w)| ((g - w).abs() / scale) as f64)
                .sum::<f64>()
                / got.len() as f64;
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{ty:?}: max rel {max_rel:.3e} / mean rel {mean_rel:.3e} / max abs {max_abs:.3e} \
                 (基準の最大 |out| = {scale:.3e})"
            );
            // ここは活性の int8 量子化そのものの誤差 (合成データはキャンセルが大きく
            // 最悪側に出る)。実形状での値は `bench_vnni_vs_f32_gmacs` が数字で出す。
            assert!(max_rel < 5e-2, "{ty:?}: 相対誤差 {max_rel} が大きすぎる");
        }
    }

    /// 対象外の型・粒度の違う活性では VNNI 経路を名乗らない (黙って別経路に落とさない)。
    #[test]
    fn vnni_refuses_unsupported_types_and_block_sizes() {
        let (n_in, n_out) = (256usize, 4usize);
        let xs = make_x(n_in);
        let (q, sc, gs) = make_act(&xs, 1, n_in, 256);
        let act = VnniAct {
            q: &q,
            sc: &sc,
            gs: &gs,
            m: 1,
            n_in,
            blk: 256,
        };
        let mut out = vec![0f32; n_out];
        for ty in [GgmlType::Q8_0, GgmlType::Q5K] {
            let blocks = make_expert(ty, n_in, n_out, 0);
            assert!(
                !gemm_rows_vnni(ty, &blocks, n_in, 0, n_out, &act, out.as_mut_ptr(), n_out),
                "{ty:?}: 対象外の型なのに VNNI 経路を使おうとした"
            );
        }
        // Q5_1 の活性ブロックは 32 要素なので、256 で作った活性は使えない
        let blocks51 = make_expert(GgmlType::Q5_1, n_in, n_out, 0);
        assert!(
            !gemm_rows_vnni(
                GgmlType::Q5_1,
                &blocks51,
                n_in,
                0,
                n_out,
                &act,
                out.as_mut_ptr(),
                n_out
            ),
            "Q5_1 に 256 要素ブロックの活性を渡せてしまう"
        );
    }

    /// `vnni_enabled` は機能検出と矛盾しない (機能が無ければ必ず false)。
    #[test]
    fn vnni_enabled_needs_feature_detection() {
        if !vnni_supported() {
            assert!(!vnni_enabled(), "機能が無いのに有効になっている");
        }
    }

    /// m の実測分布を模した列 (130 expert / 合計 5120 / 最小 5 / 最大 492)。
    fn bench_m_distribution() -> Vec<usize> {
        let n = 130usize;
        let total = 5120usize;
        let mut ms: Vec<usize> = (0..n).map(|i| 20 + (i * 7) % 41).collect();
        ms[0] = 492;
        ms[1] = 300;
        let mut sum: usize = ms.iter().sum();
        let mut i = 2usize;
        while sum > total {
            let room = ms[i % n].saturating_sub(5);
            let take = (sum - total).min(room).min(20);
            ms[i % n] -= take;
            sum -= take;
            i += 1;
            assert!(i < 100_000, "m 分布を作れない");
        }
        while sum < total {
            ms[i % n] += 1;
            sum += 1;
            i += 1;
        }
        assert_eq!(ms.iter().sum::<usize>(), total);
        ms
    }

    /// 実形状マイクロベンチ (実モデル不要)。
    ///
    /// - 形状: (Q4_K) n_in=2560 / n_out=640、(Q5_1) n_in=640 / n_out=2560。
    /// - m: 130 expert で合計 5120 (5〜492 の裾) の分布と、固定 m=15。
    /// - 重みは 130 expert ぶんを確保し 1 周で全部触るので、重みは L3 (16 MB) に載らない
    ///   (cold を模す)。固定 m=15 は 1 expert だけを繰り返し触る (hot)。
    /// - 1 スレッドと 8 スレッド (この機は 8 コア / 16 SMT)。
    ///
    /// 実行:
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_vnni_vs_f32`
    #[test]
    #[ignore = "マイクロベンチ (130 expert x 実形状、約 300 MB を確保)"]
    fn bench_vnni_vs_f32_gmacs() {
        use std::time::Instant;
        if !vnni_supported() {
            eprintln!("skip bench_vnni_vs_f32_gmacs: avx512vnni が無い");
            return;
        }
        const ROWS: usize = 64; // 本番の flat スケジューラ (MOE_FLAT_ROWS) と同じ行チャンク
        const REPS: usize = 3;
        const HOT_ITERS: usize = 200;
        let n_expert = 130usize;
        let ms = bench_m_distribution();
        let m_max = *ms.iter().max().unwrap();

        for (tag, ty, n_in, n_out) in [
            ("Q4_K (gate/up)", GgmlType::Q4K, 2560usize, 640usize),
            ("Q5_1 (down)", GgmlType::Q5_1, 640usize, 2560usize),
        ] {
            let blk = vnni_act_block(ty).expect("VNNI 対象型");
            let rb = row_bytes(ty, n_in).unwrap();
            eprintln!(
                "\n== {tag}: n_in={n_in} n_out={n_out} expert={n_expert} 合計 m={} (max {m_max}) \
                 重み {} MB ==",
                ms.iter().sum::<usize>(),
                n_expert * n_out * rb / (1024 * 1024)
            );

            // 重み (130 expert) と活性 (m_max トークン) を確保する。
            let ws: Vec<Vec<u8>> = (0..n_expert)
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let xs: Vec<f32> = (0..m_max * n_in)
                .map(|i| ((i * 61 % 127) as f32 - 63.0) / 96.0)
                .collect();
            // VNNI 用の量子化済み活性 (本番では層ごとに expert あたり 1 回だけ作る)。
            let mut acts: Vec<(Vec<i8>, Vec<f32>, Vec<f32>)> = ms
                .iter()
                .map(|&m| make_act(&xs[..m * n_in], m, n_in, blk))
                .collect();

            // 1 スイープ = 全 expert × 行チャンク 64 行。`vnni` で経路を選ぶ。
            struct Ctx<'a> {
                ty: GgmlType,
                n_in: usize,
                n_out: usize,
                blk: usize,
                rb: usize,
                rows: usize,
                xs: &'a [f32],
            }
            fn sweep(
                c: &Ctx<'_>,
                ws: &[Vec<u8>],
                acts: &[(Vec<i8>, Vec<f32>, Vec<f32>)],
                ms: &[usize],
                threads: usize,
                vnni: bool,
                scratches: &mut [Vec<f32>],
            ) -> f64 {
                use std::time::Instant;
                /// 1 スレッドぶんの本体 (expert `lo..hi`)。出力は使い捨て。
                fn run(
                    c: &Ctx<'_>,
                    ws: &[Vec<u8>],
                    acts: &[(Vec<i8>, Vec<f32>, Vec<f32>)],
                    ms: &[usize],
                    vnni: bool,
                    p: *mut f32,
                    lo: usize,
                    hi: usize,
                ) {
                    use std::hint::black_box;
                    for e in lo..hi {
                        let m = ms[e];
                        let x = &c.xs[..m * c.n_in];
                        let mut a = 0usize;
                        while a < c.n_out {
                            let b = (a + c.rows).min(c.n_out);
                            // SAFETY: p は m_max * n_out 要素。a < n_out なので範囲内。
                            let pa = unsafe { p.add(a) };
                            if vnni {
                                let (q, sc, gs) = &acts[e];
                                let act = VnniAct {
                                    q,
                                    sc,
                                    gs,
                                    m,
                                    n_in: c.n_in,
                                    blk: c.blk,
                                };
                                let used = gemm_rows_vnni(
                                    c.ty,
                                    black_box(&ws[e]),
                                    c.n_in,
                                    a,
                                    b,
                                    &act,
                                    pa,
                                    c.n_out,
                                );
                                assert!(used, "VNNI 経路が使われていない");
                            } else {
                                gemm_rows(
                                    c.ty,
                                    c.rb,
                                    black_box(&ws[e]),
                                    c.n_in,
                                    a,
                                    b,
                                    black_box(x),
                                    m,
                                    pa,
                                    c.n_out,
                                    true,
                                );
                            }
                            a = b;
                        }
                    }
                    black_box(p);
                }
                let per = ws.len().div_ceil(threads.max(1));
                let t0 = Instant::now();
                if threads <= 1 {
                    let p = scratches[0].as_mut_ptr();
                    // SAFETY: scratches[0] は m_max * n_out 要素。書き込みは n_out 列まで。
                    run(c, ws, acts, ms, vnni, p, 0, ws.len());
                } else {
                    std::thread::scope(|s| {
                        let mut hs = Vec::with_capacity(threads);
                        for (ti, sc) in scratches.iter_mut().enumerate().take(threads) {
                            let lo = (ti * per).min(ws.len());
                            let hi = ((ti + 1) * per).min(ws.len());
                            hs.push(s.spawn(move || {
                                let p = sc.as_mut_ptr();
                                run(c, ws, acts, ms, vnni, p, lo, hi);
                            }));
                        }
                        for h in hs {
                            h.join().unwrap();
                        }
                    });
                }
                t0.elapsed().as_secs_f64()
            }

            /// 量子化だけのスイープ (本番では expert ごとに 1 回)。
            fn quant_sweep(
                c: &Ctx<'_>,
                acts: &mut [(Vec<i8>, Vec<f32>, Vec<f32>)],
                ms: &[usize],
                threads: usize,
            ) -> f64 {
                use std::time::Instant;
                let per = acts.len().div_ceil(threads);
                let t0 = Instant::now();
                let parts: Vec<&mut [(Vec<i8>, Vec<f32>, Vec<f32>)]> =
                    acts.chunks_mut(per).collect();
                std::thread::scope(|s| {
                    let mut hs = Vec::with_capacity(parts.len());
                    for (ti, part) in parts.into_iter().enumerate() {
                        let msl = &ms[ti * per..];
                        hs.push(s.spawn(move || {
                            for (k, (q, sc, gs)) in part.iter_mut().enumerate() {
                                let m = msl[k];
                                quant_act_i8(&c.xs[..m * c.n_in], m, c.n_in, c.blk, q, sc, gs)
                                    .unwrap();
                            }
                        }));
                    }
                    for h in hs {
                        h.join().unwrap();
                    }
                });
                t0.elapsed().as_secs_f64()
            }
            let ctx = Ctx {
                ty,
                n_in,
                n_out,
                blk,
                rb,
                rows: ROWS,
                xs: &xs,
            };

            let macs = |ms: &[usize]| -> f64 {
                ms.iter().map(|&m| (m * n_in * n_out) as f64).sum::<f64>()
            };
            let best = |f: &mut dyn FnMut() -> f64| -> f64 {
                let mut b = f64::INFINITY;
                for _ in 0..REPS {
                    b = b.min(f());
                }
                b
            };

            // (a) 固定 m=15 / 1 expert / hot (従来の表と同じ条件)
            {
                let m = 15usize;
                let one = vec![make_expert(ty, n_in, n_out, 0)];
                let one_ms = vec![m];
                let (q, sc, gs) = make_act(&xs[..m * n_in], m, n_in, blk);
                let one_act = vec![(q, sc, gs)];
                let mac = macs(&one_ms) * HOT_ITERS as f64;
                let mut hot_scratch = vec![vec![0f32; m_max * n_out]];
                let t_f = {
                    let mut f = || {
                        let t0 = Instant::now();
                        for _ in 0..HOT_ITERS {
                            sweep(&ctx, &one, &one_act, &one_ms, 1, false, &mut hot_scratch);
                        }
                        t0.elapsed().as_secs_f64()
                    };
                    best(&mut f)
                };
                let t_v = {
                    let mut v = || {
                        let t0 = Instant::now();
                        for _ in 0..HOT_ITERS {
                            sweep(&ctx, &one, &one_act, &one_ms, 1, true, &mut hot_scratch);
                        }
                        t0.elapsed().as_secs_f64()
                    };
                    best(&mut v)
                };
                eprintln!(
                    "  m=15 hot 1T : f32 {:8.1} GMAC/s | VNNI {:8.1} GMAC/s | 比 {:.2}x",
                    mac / t_f / 1e9,
                    mac / t_v / 1e9,
                    t_f / t_v
                );
            }

            // (a2) f32 経路に対する相対誤差 (全 expert・全行。誤差の出所は活性の int8 量子化だけ)
            {
                let mut out_f = vec![0f32; m_max * n_out];
                let mut out_v = vec![0f32; m_max * n_out];
                let (mut max_abs, mut max_ref) = (0f32, 0f32);
                let (mut sum_abs, mut sum_ref) = (0f64, 0f64);
                for e in 0..n_expert {
                    let m = ms[e];
                    let (q, sc, gs) = &acts[e];
                    let act = VnniAct {
                        q,
                        sc,
                        gs,
                        m,
                        n_in,
                        blk,
                    };
                    let used = gemm_rows_vnni(
                        ty,
                        &ws[e],
                        n_in,
                        0,
                        n_out,
                        &act,
                        out_v.as_mut_ptr(),
                        n_out,
                    );
                    assert!(used, "VNNI 経路が使われていない");
                    gemm_rows(
                        ty,
                        rb,
                        &ws[e],
                        n_in,
                        0,
                        n_out,
                        &xs[..m * n_in],
                        m,
                        out_f.as_mut_ptr(),
                        n_out,
                        true,
                    );
                    for k in 0..m * n_out {
                        let (a, b) = (out_v[k], out_f[k]);
                        max_abs = max_abs.max((a - b).abs());
                        max_ref = max_ref.max(b.abs());
                        sum_abs += (a - b).abs() as f64;
                        sum_ref += b.abs() as f64;
                    }
                }
                eprintln!(
                    "  誤差 (f32 経路比): max |Δ|/max |ref| = {:.3e} / mean |Δ|/mean |ref| = {:.3e}",
                    max_abs / max_ref.max(1e-30),
                    sum_abs / sum_ref.max(1e-30)
                );
            }

            // (b) 分布 / 1 スレッド / cold、(c) 分布 / 8 スレッド / cold
            for threads in [1usize, 8] {
                let mac = macs(&ms);
                let mut scratches: Vec<Vec<f32>> =
                    (0..threads).map(|_| vec![0f32; m_max * n_out]).collect();
                let t_f = {
                    let mut f = || sweep(&ctx, &ws, &acts, &ms, threads, false, &mut scratches);
                    best(&mut f)
                };
                let t_v = {
                    let mut v = || sweep(&ctx, &ws, &acts, &ms, threads, true, &mut scratches);
                    best(&mut v)
                };
                let t_q = {
                    let mut q = || quant_sweep(&ctx, &mut acts, &ms, threads);
                    best(&mut q)
                };
                eprintln!(
                    "  dist cold {threads}T: f32 {:8.1} GMAC/s ({:7.1} ms) | VNNI {:8.1} GMAC/s \
                     ({:7.1} ms) | 量子化 {:6.1} ms | 比 {:.2}x",
                    mac / t_f / 1e9,
                    t_f * 1e3,
                    mac / t_v / 1e9,
                    t_v * 1e3,
                    t_q * 1e3,
                    t_f / (t_v + t_q)
                );
            }
        }
    }

    // ---- M=1 (decode) の VNNI GEMV -----------------------------------------

    /// M=1 VNNI カーネルが計算する式の f64 参照実装。
    ///
    /// **量子化済みの活性** (`q`, `sc`, `gs`) を使うので、差は f32 の累積順だけになる。
    /// ここが一致すれば nibble / 5 ビット目の並び・32 要素グループと重みスケールの
    /// 対応・最小値項の扱いが正しい。
    fn ref_gemv_quantized_f64(
        ty: GgmlType,
        blocks: &[u8],
        n_in: usize,
        n_out: usize,
        q: &[i8],
        sc: &[f32],
        gs: &[f32],
    ) -> Vec<f64> {
        let rb = row_bytes(ty, n_in).unwrap();
        let ts = ty.type_size();
        let xq = |i: usize| q[i] as f64;
        let mut out = vec![0f64; n_out];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &blocks[r * rb..(r + 1) * rb];
            let mut acc = 0.0f64;
            match ty {
                GgmlType::Q4K => {
                    let nb = n_in / 256;
                    for b in 0..nb {
                        let bl = &row[b * ts..(b + 1) * ts];
                        let d = half::f16::from_le_bytes([bl[0], bl[1]]).to_f64();
                        let dmin = half::f16::from_le_bytes([bl[2], bl[3]]).to_f64();
                        let scq = &bl[4..16];
                        for j in 0..4usize {
                            for half_i in 0..2usize {
                                let g8 = 2 * j + half_i;
                                let (sg, mg) = scale_min_k4(g8, scq);
                                let grp = b * 8 + g8; // 32 要素グループ
                                let mut dot = 0.0f64;
                                for i in 0..32usize {
                                    let byte = bl[16 + j * 32 + i];
                                    let qn = if half_i == 0 {
                                        byte & 0x0f
                                    } else {
                                        byte >> 4
                                    } as f64;
                                    dot += qn * xq(grp * 32 + i);
                                }
                                acc += sc[grp] as f64 * d * sg as f64 * dot
                                    - dmin * mg as f64 * gs[grp] as f64;
                            }
                        }
                    }
                }
                GgmlType::Q5_1 => {
                    let nb = n_in / 32;
                    for b in 0..nb {
                        let bl = &row[b * ts..(b + 1) * ts];
                        let d = half::f16::from_le_bytes([bl[0], bl[1]]).to_f64();
                        let mc = half::f16::from_le_bytes([bl[2], bl[3]]).to_f64();
                        let qh = u32::from_le_bytes([bl[4], bl[5], bl[6], bl[7]]);
                        let mut dot = 0.0f64;
                        for j in 0..16usize {
                            let byte = bl[8 + j];
                            let lo = ((byte & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as f64;
                            let hi =
                                (((byte >> 4) as u32) | (((qh >> (16 + j)) & 1) << 4)) as f64;
                            dot += lo * xq(b * 32 + j) + hi * xq(b * 32 + 16 + j);
                        }
                        acc += sc[b] as f64 * d * dot + mc * gs[b] as f64;
                    }
                }
                GgmlType::Q8_0 => {
                    let nb = n_in / 32;
                    for b in 0..nb {
                        let bl = &row[b * ts..(b + 1) * ts];
                        let d = half::f16::from_le_bytes([bl[0], bl[1]]).to_f64();
                        let mut dot = 0.0f64;
                        for i in 0..32usize {
                            dot += (bl[2 + i] as i8) as f64 * xq(b * 32 + i);
                        }
                        acc += sc[b] as f64 * d * dot;
                    }
                }
                other => panic!("テスト対象外の型: {:?}", other),
            }
            *o = acc;
        }
        out
    }

    /// 量子化のブロック長・スケール・`Σ x`・`Σ q` を手計算と突き合わせる。
    #[test]
    fn vnni_m1_quantize_blocks_scales_and_sums() {
        let mut x = vec![0f32; 64];
        x[0] = 2.0;
        x[1] = -1.0;
        // 第 2 ブロック (32..64) は全部 0 → スケール 1.0・q は全部 0。
        let mut a = VnniActM1::default();
        a.quantize(&x).unwrap();
        assert_eq!(a.q.len(), 64);
        assert_eq!(a.sc.len(), 2);
        assert!((a.sc[0] - 2.0 / 127.0).abs() < 1e-12, "sc[0]={}", a.sc[0]);
        assert_eq!(a.sc[1], 1.0, "max=0 のブロックのスケールは 1.0");
        assert_eq!((a.q[0], a.q[1]), (127, -64), "丸めが round-to-nearest でない");
        assert!(a.q[32..].iter().all(|&v| v == 0));
        assert!((a.gs[0] - 1.0).abs() < 1e-6, "gs[0]={}", a.gs[0]);
        assert_eq!(a.gs[1], 0.0);
        assert_eq!(a.sq[0], 127 - 64);
        assert_eq!(a.sq[1], 0);
        assert_eq!(a.q8_fix[0], 128 * (127 - 64));
        assert_eq!(a.q8_fix[8], 0);
        assert_eq!(a.q8_fix.len(), 16, "8 レーン x グループ数");
        // 32 の倍数でない活性は弾く (黙って端数を落とさない)。
        assert!(a.quantize(&vec![0f32; 48]).is_err());
    }

    /// 実形 3 種で f32 経路と比べる (誤差の出所は活性の int8 量子化だけ)。
    ///
    /// `|x|` が大きいケース (i8 に落とすときの飽和側) も同じループで見る。
    /// 量子化はブロックごとに `max|x|/127` で正規化するので、値の絶対値では
    /// 相対誤差は変わらない (変わるのは i8 の飽和ではなく f32 の丸め)。
    #[test]
    fn gemv_vnni_matches_f32_path_within_tolerance() {
        let _g = gemv_vnni_test_guard();
        if !vnni_supported() {
            eprintln!("skip gemv_vnni_matches_f32_path_within_tolerance: avx512vnni が無い");
            return;
        }
        let _gate = VnniGateGuard::on();
        for (tag, ty, n_in, n_out) in [
            ("Q4_K 2560x640", GgmlType::Q4K, 2560usize, 640usize),
            ("Q5_1 640x2560", GgmlType::Q5_1, 640usize, 2560usize),
            ("Q8_0 2560x640", GgmlType::Q8_0, 2560usize, 640usize),
        ] {
            let blocks = make_expert(ty, n_in, n_out, 0);
            for (case, k) in [("通常", 1.0f32), ("|x| 大 (1e4 倍)", 1.0e4f32)] {
                let x: Vec<f32> = make_x_wide(n_in).iter().map(|v| v * k).collect();
                let mut want = vec![0f32; n_out];
                expert_gemv_any(ty, &blocks, n_in, n_out, &x, &mut want).unwrap();
                let mut got = vec![0f32; n_out];
                let before = gemv_vnni_calls();
                let used = expert_gemv_any_vnni(ty, &blocks, n_in, n_out, &x, &mut got).unwrap();
                assert!(used, "{tag}: VNNI 経路が使われていない");
                assert!(
                    gemv_vnni_calls() > before,
                    "{tag}: ゲートは開いたがカーネルが発火していない"
                );
                assert!(
                    want.iter().any(|v| *v != 0.0),
                    "{tag}: f32 経路の出力が全 0 (比較が無意味)"
                );
                assert!(got.iter().all(|v| v.is_finite()), "{tag}: 非有限値が出た");

                let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30);
                let max_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0f32, f32::max);
                let mean_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| ((g - w).abs() / scale) as f64)
                    .sum::<f64>()
                    / got.len() as f64;
                // 行ごとの相対誤差 (|want| が基準の 1% 未満の行は除く)。
                let mut row_rel = 0f32;
                for (g, w) in got.iter().zip(&want) {
                    if w.abs() >= 0.01 * scale {
                        row_rel = row_rel.max((g - w).abs() / w.abs());
                    }
                }
                eprintln!(
                    "{tag} [{case}]: max|Δ|/max|ref| = {:.3e} / mean = {:.3e} / 行相対の最大 = {:.3e}",
                    max_abs / scale,
                    mean_abs,
                    row_rel
                );
                // 量子化の刻みは `max|x|/127` の半分 = 入力の 0.39%。積和で
                // 平均化されるので、出力の相対誤差はその 1 桁上に出る。
                assert!(
                    max_abs / scale < 1e-2,
                    "{tag} [{case}]: 相対誤差 {:.3e} が大きすぎる",
                    max_abs / scale
                );
            }
        }
    }

    /// カーネルが「量子化済みの式」をそのまま計算していること (f64 参照との比較)。
    ///
    /// ここは活性の量子化誤差が入らないので、nibble 順・スケール割り当て・最小値項の
    /// 対応が違えば O(1) の差になり、正しければ f32 の累積順だけで 1e-4 台に収まる。
    #[test]
    fn gemv_vnni_matches_quantized_f64_reference() {
        let _g = gemv_vnni_test_guard();
        if !vnni_supported() {
            eprintln!("skip gemv_vnni_matches_quantized_f64_reference: avx512vnni が無い");
            return;
        }
        let _gate = VnniGateGuard::on();
        for (tag, ty, n_in, n_out) in [
            ("Q4_K 2560x640", GgmlType::Q4K, 2560usize, 640usize),
            ("Q5_1 640x2560", GgmlType::Q5_1, 640usize, 2560usize),
            ("Q8_0 2560x640", GgmlType::Q8_0, 2560usize, 640usize),
        ] {
            let blocks = make_expert(ty, n_in, n_out, 0);
            let x = make_x_wide(n_in);
            let mut act = VnniActM1::default();
            act.quantize(&x).unwrap();
            let mut got = vec![0f32; n_out];
            assert!(
                expert_gemv_any_vnni(ty, &blocks, n_in, n_out, &x, &mut got).unwrap(),
                "{tag}: VNNI 経路が使われていない"
            );
            let want = ref_gemv_quantized_f64(ty, &blocks, n_in, n_out, &act.q, &act.sc, &act.gs);
            let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-30);
            let max_rel = got
                .iter()
                .zip(&want)
                .map(|(g, w)| ((*g as f64) - *w).abs() / scale)
                .fold(0f64, f64::max);
            eprintln!("{tag}: 量子化した式との差 (f64 基準) {max_rel:.3e}");
            assert!(
                max_rel < 1e-3,
                "{tag}: カーネルの式と一致しない (rel {max_rel:.3e}) — nibble 順かスケールが違う"
            );
        }
    }

    /// 既定 (env 未設定) では VNNI 経路を名乗らない = `expert_gemv_any` の動作を変えない。
    #[test]
    fn gemv_vnni_is_off_by_default() {
        let _g = gemv_vnni_test_guard();
        set_gemv_vnni_for_test(Some(false));
        let (n_in, n_out) = (2560usize, 640usize);
        let blocks = make_expert(GgmlType::Q4K, n_in, n_out, 0);
        let x = make_x_wide(n_in);
        let mut out = vec![0f32; n_out];
        let before = gemv_vnni_calls();
        assert!(
            !expert_gemv_any_vnni(GgmlType::Q4K, &blocks, n_in, n_out, &x, &mut out).unwrap(),
            "既定で VNNI 経路を名乗った"
        );
        assert_eq!(gemv_vnni_calls(), before, "既定でカーネルが発火した");
        set_gemv_vnni_for_test(None);
    }

    /// M=1 VNNI のマイクロベンチ (1 スレッド / 8 スレッド、f32 経路との比)。
    ///
    /// - 形は実形: Q4_K 2560x640 / Q5_1 640x2560 / Q8_0 2560x640 (n_in x n_out)。
    /// - decode は 1 トークンで 48 層 x 10 expert の別々の重み (合計 487 MB) を触るので、
    ///   ここでは 24 枚を巡回して L3 (16 MB) に載りきらない条件にする
    ///   (実機の mmap cold よりは温かい = 比は上振れしうる)。
    /// - 実行:
    ///   `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_gemv_vnni`
    #[test]
    #[ignore = "マイクロベンチ (24 枚 x 3 形の重みを確保、約 90 MB)"]
    fn bench_gemv_vnni_m1_gmacs() {
        use std::time::Instant;
        let _g = gemv_vnni_test_guard();
        if !vnni_supported() {
            eprintln!("skip bench_gemv_vnni_m1_gmacs: avx512vnni が無い");
            return;
        }
        let _gate = VnniGateGuard::on();
        const N_MAT: usize = 24;
        const REPS: usize = 7;

        /// f32 と VNNI を交互に測り、それぞれの最小を返す。
        ///
        /// このコンテナはホスト共有で load average が動くので、別々に測ると
        /// 比が ±30% ぶれる。同じ rep の中で続けて測って交互に取る。
        fn paired(reps: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
            let (mut ba, mut bb) = (f64::INFINITY, f64::INFINITY);
            for i in 0..reps {
                let t0 = Instant::now();
                a();
                let ta = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                b();
                let tb = t0.elapsed().as_secs_f64();
                if i > 0 {
                    // 1 回目はウォームアップとして捨てる。
                    ba = ba.min(ta);
                    bb = bb.min(tb);
                }
            }
            (ba, bb)
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("8 スレッドの rayon プール");

        for (tag, ty, n_in, n_out) in [
            ("Q4_K 2560x640", GgmlType::Q4K, 2560usize, 640usize),
            ("Q5_1 640x2560", GgmlType::Q5_1, 640usize, 2560usize),
            ("Q8_0 2560x640", GgmlType::Q8_0, 2560usize, 640usize),
        ] {
            let ws: Vec<Vec<u8>> = (0..N_MAT)
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let x = make_x_wide(n_in);
            let mut of = vec![0f32; n_out]; // f32 経路の出力
            let mut ov = vec![0f32; n_out]; // VNNI 経路の出力
            let bytes = N_MAT * n_out * row_bytes(ty, n_in).unwrap();
            let macs = (N_MAT * n_in * n_out) as f64;
            let before = gemv_vnni_calls();

            // (a) 24 枚を巡回 (L3 16 MB に載りきらない = 帯域も見る)。
            let (t_f1, t_v1) = paired(
                REPS,
                || {
                    for w in &ws {
                        expert_gemv_any_seq(ty, w, n_in, n_out, &x, &mut of).unwrap();
                    }
                    std::hint::black_box(&of);
                },
                || {
                    for w in &ws {
                        assert!(expert_gemv_any_vnni_seq(ty, w, n_in, n_out, &x, &mut ov).unwrap());
                    }
                    std::hint::black_box(&ov);
                },
            );
            let (t_f8, t_v8) = paired(
                REPS,
                || {
                    pool.install(|| {
                        for w in &ws {
                            expert_gemv_any(ty, w, n_in, n_out, &x, &mut of).unwrap();
                        }
                    });
                },
                || {
                    pool.install(|| {
                        for w in &ws {
                            assert!(expert_gemv_any_vnni(ty, w, n_in, n_out, &x, &mut ov).unwrap());
                        }
                    });
                },
            );
            // (b) 1 枚を 24 回 (毎回同じ重み = L2 に載る)。帯域ではなく
            //     カーネルとスレッド起動のコストを見る。
            let one = &ws[0];
            let (h_f1, h_v1) = paired(
                REPS,
                || {
                    for _ in 0..N_MAT {
                        expert_gemv_any_seq(ty, one, n_in, n_out, &x, &mut of).unwrap();
                    }
                    std::hint::black_box(&of);
                },
                || {
                    for _ in 0..N_MAT {
                        assert!(expert_gemv_any_vnni_seq(ty, one, n_in, n_out, &x, &mut ov).unwrap());
                    }
                    std::hint::black_box(&ov);
                },
            );
            let (h_f8, h_v8) = paired(
                REPS,
                || {
                    pool.install(|| {
                        for _ in 0..N_MAT {
                            expert_gemv_any(ty, one, n_in, n_out, &x, &mut of).unwrap();
                        }
                    });
                },
                || {
                    pool.install(|| {
                        for _ in 0..N_MAT {
                            assert!(
                                expert_gemv_any_vnni(ty, one, n_in, n_out, &x, &mut ov).unwrap()
                            );
                        }
                    });
                },
            );
            assert!(
                gemv_vnni_calls() > before,
                "{tag}: VNNI カーネルが発火していない (ゲートの通過を数える)"
            );
            eprintln!(
                "{tag}: 重み {} MB / {} MAC/枚\n  \
                 24 枚 (L3 超)  1T f32 {:6.1} -> VNNI {:6.1} GMAC/s ({:.2}x) \
                 | 8T f32 {:6.1} -> VNNI {:6.1} GMAC/s ({:.2}x)\n  \
                 1 枚 x24 (L2)  1T f32 {:6.1} -> VNNI {:6.1} GMAC/s ({:.2}x) \
                 | 8T f32 {:6.1} -> VNNI {:6.1} GMAC/s ({:.2}x)",
                bytes / (1024 * 1024),
                n_in * n_out,
                macs / t_f1 / 1e9,
                macs / t_v1 / 1e9,
                t_f1 / t_v1,
                macs / t_f8 / 1e9,
                macs / t_v8 / 1e9,
                t_f8 / t_v8,
                macs / h_f1 / 1e9,
                macs / h_v1 / 1e9,
                h_f1 / h_v1,
                macs / h_f8 / 1e9,
                macs / h_v8 / 1e9,
                h_f8 / h_v8,
            );
        }
    }

    // ---- 9. grouped GEMM (可変 m) ------------------------------------------

    /// m の一覧から gather 索引 (`by_expert`) を組み立てる。
    ///
    /// expert 内のトークンは `(e * 37) % (ntok - m + 1)` から連続 m 個 (昇順・重複なし)。
    /// ルータのスロットは `i % k`。
    fn by_expert_from_ms(ms: &[usize], ntok: usize, k: usize) -> Vec<Vec<(usize, usize)>> {
        ms.iter()
            .enumerate()
            .map(|(e, &m)| {
                assert!(m <= ntok, "m {m} がトークン数 {ntok} を超える");
                let base = (e * 37) % (ntok - m + 1);
                (0..m).map(|i| (base + i, i % k)).collect()
            })
            .collect()
    }

    #[test]
    fn grouped_segments_are_descending_and_contiguous() {
        let ms = [7usize, 3, 7, 1, 6, 3];
        let (n_in, n_out) = (8usize, 4usize);
        let ntok = 9usize;
        let x_all = make_xs(ntok, n_in);
        let by = by_expert_from_ms(&ms, ntok, 3);
        let gb = build_grouped_batch(&by, &x_all, n_in, n_out).unwrap();

        assert_eq!(gb.m_total, ms.iter().sum::<usize>());
        assert_eq!(gb.segs.len(), ms.len());
        // m の降順 (同 m は expert 昇順)
        for w in gb.segs.windows(2) {
            assert!(w[0].m >= w[1].m, "m が降順でない: {:?} -> {:?}", w[0], w[1]);
            if w[0].m == w[1].m {
                assert!(
                    w[0].expert < w[1].expert,
                    "同 m で expert 昇順でない: {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
        }
        // オフセットは 0 から隙間なく連続し、終端が xs / outs の長さに一致する
        let (mut xo, mut oo) = (0usize, 0usize);
        for s in &gb.segs {
            assert_eq!(s.x_off as usize, xo, "x_off が連続でない: {s:?}");
            assert_eq!(s.out_off as usize, oo, "out_off が連続でない: {s:?}");
            xo += s.m as usize * n_in;
            oo += s.m as usize * n_out;
        }
        assert_eq!(xo, gb.xs.len());
        assert_eq!(xo, gb.m_total * n_in);
        assert_eq!(oo, gb.outs_len(n_out));
        assert_eq!(
            check_segments(&gb.segs, n_in, n_out, gb.xs.len(), gb.outs_len(n_out)).unwrap(),
            gb.m_total
        );
        // xs は gather した行 (expert 内はトークン昇順)
        for s in &gb.segs {
            let picks = &by[s.expert as usize];
            assert_eq!(picks.len(), s.m as usize);
            for (i, &(t, _)) in picks.iter().enumerate() {
                let a = s.x_off as usize + i * n_in;
                assert_eq!(
                    &gb.xs[a..a + n_in],
                    &x_all[t * n_in..(t + 1) * n_in],
                    "segment expert={} 行 {i} が gather と違う",
                    s.expert
                );
            }
        }
        // 空の expert は segment にならない (m = 0 の空タスクを作らない)
        let mut by2 = by.clone();
        by2[2] = Vec::new();
        let gb2 = build_grouped_batch(&by2, &x_all, n_in, n_out).unwrap();
        assert_eq!(gb2.segs.len(), ms.len() - 1);
        assert!(gb2.segs.iter().all(|s| s.expert != 2));
    }

    #[test]
    fn grouped_cpu_matches_scalar_entry_all_types() {
        // 小さな形で全 4 型を「既存の per-expert 経路 (スカラー)」とビット比較する。
        let (n_in, n_out) = (N_IN, N_OUT);
        let ms = [3usize, 1, 4, 2];
        let ntok = 5usize;
        let x_all = make_xs(ntok, n_in);
        let by = by_expert_from_ms(&ms, ntok, 2);
        for ty in ALL_TYPES {
            let ws: Vec<Vec<u8>> = (0..ms.len())
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let wrefs: Vec<&[u8]> = ws.iter().map(|v| v.as_slice()).collect();
            let gb = build_grouped_batch(&by, &x_all, n_in, n_out).unwrap();
            let mut got = vec![0.0f32; gb.outs_len(n_out)];
            grouped_expert_gemm_cpu(ty, &wrefs, n_in, n_out, &gb.segs, &gb.xs, &mut got)
                .unwrap_or_else(|e| panic!("{ty:?}: {e}"));
            for s in &gb.segs {
                let m = s.m as usize;
                let x0 = s.x_off as usize;
                let o = s.out_off as usize;
                let mut want = vec![0.0f32; m * n_out];
                expert_gemm_scalar(
                    ty,
                    wrefs[s.expert as usize],
                    n_in,
                    n_out,
                    &gb.xs[x0..x0 + m * n_in],
                    m,
                    &mut want,
                )
                .unwrap();
                expect_bits(&got[o..o + m * n_out], &want);
                assert!(
                    want.iter().any(|v| *v != 0.0),
                    "{ty:?} expert {}: 出力が全 0 (テストが無意味)",
                    s.expert
                );
            }
            assert!(got.iter().all(|v| v.is_finite()), "{ty:?}: 非有限値");
        }
    }

    #[test]
    fn grouped_cpu_matches_per_expert_bits_real_shape() {
        // 実形状 (n_in=2560 / n_out=640、130 expert、合計 5120、m は 5〜492)。
        // Q4_K は gate/up 形、Q5_1 は down 形 (n_in=640 → n_out=2560)。
        let ms = bench_m_distribution();
        let ntok = 512usize; // 512 トークン × k=10 = 5120
        assert_eq!(ms.len(), 130);
        assert_eq!(ms.iter().sum::<usize>(), 5120);
        assert_eq!(ms.iter().max(), Some(&492));
        assert_eq!(ms.iter().min(), Some(&5));
        let by = by_expert_from_ms(&ms, ntok, 10);
        for (ty, n_in, n_out) in [
            (GgmlType::Q4K, PERF_N_IN, PERF_N_OUT),
            (GgmlType::Q5_1, PERF_N_OUT, PERF_N_IN),
        ] {
            let ws: Vec<Vec<u8>> = (0..ms.len())
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let wrefs: Vec<&[u8]> = ws.iter().map(|v| v.as_slice()).collect();
            let x_all = make_xs(ntok, n_in);
            let gb = build_grouped_batch(&by, &x_all, n_in, n_out).unwrap();
            assert_eq!(gb.segs.len(), 130);
            let mut got = vec![0.0f32; gb.outs_len(n_out)];
            grouped_expert_gemm_cpu(ty, &wrefs, n_in, n_out, &gb.segs, &gb.xs, &mut got)
                .unwrap_or_else(|e| panic!("{ty:?}: {e}"));
            // (a) 既存の per-expert 経路 (スカラー) とビット一致
            for s in &gb.segs {
                let m = s.m as usize;
                let x0 = s.x_off as usize;
                let o = s.out_off as usize;
                let mut want = vec![0.0f32; m * n_out];
                expert_gemm_scalar(
                    ty,
                    wrefs[s.expert as usize],
                    n_in,
                    n_out,
                    &gb.xs[x0..x0 + m * n_in],
                    m,
                    &mut want,
                )
                .unwrap();
                expect_bits(&got[o..o + m * n_out], &want);
                assert!(
                    want.iter().any(|v| *v != 0.0),
                    "{ty:?} expert {}: 出力が全 0",
                    s.expert
                );
            }
            assert!(got.iter().all(|v| v.is_finite()), "{ty:?}: 非有限値");
            // トークン (行) の取り違えが無いこと: 最大 m の expert の先頭 2 行は別の値
            let s0 = gb.segs[0];
            assert!(s0.m > 1, "先頭 segment の m が 1");
            let o0 = s0.out_off as usize;
            assert_ne!(
                got[o0..o0 + n_out],
                got[o0 + n_out..o0 + 2 * n_out],
                "{ty:?}: 同じ expert の 2 トークンが同値 (行を共有している疑い)"
            );
        }
    }

    #[test]
    fn grouped_cpu_does_not_depend_on_segment_order() {
        // m 降順の並びを逆 (m 昇順) に組み直しても結果が変わらないこと。
        let ms = [5usize, 3, 9, 1, 6, 4, 7];
        let ntok = 12usize;
        // n_in は Q4_K のブロック長 256 で割り切れる必要がある (row_bytes の門番)。
        let (n_in, n_out) = (256usize, 8usize);
        let x_all = make_xs(ntok, n_in);
        let by = by_expert_from_ms(&ms, ntok, 4);
        for ty in [GgmlType::Q4K, GgmlType::Q5_1] {
            let ws: Vec<Vec<u8>> = (0..ms.len())
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let wrefs: Vec<&[u8]> = ws.iter().map(|v| v.as_slice()).collect();
            let gb = build_grouped_batch(&by, &x_all, n_in, n_out).unwrap();
            let mut want = vec![0.0f32; gb.outs_len(n_out)];
            grouped_expert_gemm_cpu(ty, &wrefs, n_in, n_out, &gb.segs, &gb.xs, &mut want).unwrap();

            let mut perm: Vec<usize> = (0..gb.segs.len()).collect();
            perm.reverse();
            let mut segs2: Vec<Segment> = Vec::with_capacity(gb.segs.len());
            let mut xs2: Vec<f32> = Vec::with_capacity(gb.xs.len());
            let (mut xo, mut oo) = (0usize, 0usize);
            for &i in &perm {
                let s = gb.segs[i];
                let m = s.m as usize;
                let x0 = s.x_off as usize;
                segs2.push(Segment {
                    expert: s.expert,
                    m: s.m,
                    x_off: xo as u32,
                    out_off: oo as u32,
                });
                xs2.extend_from_slice(&gb.xs[x0..x0 + m * n_in]);
                xo += m * n_in;
                oo += m * n_out;
            }
            let mut got = vec![0.0f32; gb.outs_len(n_out)];
            grouped_expert_gemm_cpu(ty, &wrefs, n_in, n_out, &segs2, &xs2, &mut got).unwrap();
            // 元の並びへ戻して突き合わせる。
            let mut back = vec![0.0f32; gb.outs_len(n_out)];
            let mut o = 0usize;
            for &i in &perm {
                let d = gb.segs[i].out_off as usize;
                let n = gb.segs[i].m as usize * n_out;
                back[d..d + n].copy_from_slice(&got[o..o + n]);
                o += n;
            }
            expect_bits(&back, &want);
            assert!(want.iter().any(|v| *v != 0.0), "{ty:?}: 出力が全 0");
            assert_ne!(
                want[..n_out],
                want[gb.segs[0].m as usize * n_out..],
                "{ty:?}: segment の範囲がずれている疑い"
            );
        }
    }

    #[test]
    fn grouped_cpu_rejects_bad_tables() {
        let (n_in, n_out) = (N_IN, N_OUT);
        for ty in [GgmlType::Q4K, GgmlType::Q5_1] {
            let w0 = make_expert(ty, n_in, n_out, 0);
            let weights: [&[u8]; 1] = [&w0];
            let xs = make_xs(3, n_in);
            let mut outs = vec![0.0f32; 3 * n_out];
            let ok = [Segment {
                expert: 0,
                m: 3,
                x_off: 0,
                out_off: 0,
            }];
            grouped_expert_gemm_cpu(ty, &weights, n_in, n_out, &ok, &xs, &mut outs).unwrap();
            assert!(outs.iter().any(|v| *v != 0.0), "{ty:?}: 出力が全 0");

            // xs が短い (m の合計と合わない)
            let e = grouped_expert_gemm_cpu(ty, &weights, n_in, n_out, &ok, &xs[..xs.len() - 1], &mut outs)
                .unwrap_err();
            assert!(format!("{e}").contains("xs"), "エラー文が違う: {e}");
            // outs が短い
            let n = outs.len() - 1;
            let e = grouped_expert_gemm_cpu(ty, &weights, n_in, n_out, &ok, &xs, &mut outs[..n])
                .unwrap_err();
            assert!(format!("{e}").contains("outs"), "エラー文が違う: {e}");
            // オフセットが連続でない (重複)
            let dup = [
                Segment {
                    expert: 0,
                    m: 2,
                    x_off: 0,
                    out_off: 0,
                },
                Segment {
                    expert: 0,
                    m: 1,
                    x_off: 0,
                    out_off: (2 * n_out) as u32,
                },
            ];
            assert!(
                grouped_expert_gemm_cpu(ty, &weights, n_in, n_out, &dup, &xs, &mut outs).is_err(),
                "{ty:?}: 重複した x_off を通した"
            );
            // m == 0 の segment
            let zero = [Segment {
                expert: 0,
                m: 0,
                x_off: 0,
                out_off: 0,
            }];
            assert!(check_segments(&zero, n_in, n_out, 0, 0).is_err());
            // 重みが無い expert
            let missing = [Segment {
                expert: 1,
                m: 3,
                x_off: 0,
                out_off: 0,
            }];
            let e = grouped_expert_gemm_cpu(ty, &weights, n_in, n_out, &missing, &xs, &mut outs)
                .unwrap_err();
            assert!(format!("{e}").contains("重みが無い"), "エラー文が違う: {e}");
            // 重みの長さが違う
            let short: [&[u8]; 1] = [&w0[..w0.len() - 1]];
            assert!(
                grouped_expert_gemm_cpu(ty, &short, n_in, n_out, &ok, &xs, &mut outs).is_err(),
                "{ty:?}: 短い重みを通した"
            );
        }
    }

    /// grouped CPU 参照実装のマイクロベンチ (GPU 側の比較対象)。
    ///
    /// - 形状: Q4_K は gate/up (n_in=2560 / n_out=640)、Q5_1 は down (640 / 2560)。
    /// - m: 130 expert で合計 5120 (5〜492) の実測分布。
    /// - 1 スレッドと 8 スレッド (segment の列を割る。segment は互いに素な範囲を
    ///   書くだけなので、スレッド数を変えても値は変わらない)。
    /// - 重み 130 expert を 1 周で全部触るので cold (Q4_K 120 MB / Q5_1 160 MB)。
    ///
    /// 実行:
    /// `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_grouped_cpu_gmacs`
    #[test]
    #[ignore = "マイクロベンチ (130 expert x 実形状、約 300 MB を確保)"]
    fn bench_grouped_cpu_gmacs() {
        use std::time::Instant;
        const REPS: usize = 2;
        let ms = bench_m_distribution();
        let n_expert = ms.len();
        let ntok = 512usize;
        let by = by_expert_from_ms(&ms, ntok, 10);

        /// segment の列を `threads` 本に割り、重ならない outs の部分スライスで回す。
        #[allow(clippy::too_many_arguments)]
        fn run_split(
            ty: GgmlType,
            n_in: usize,
            n_out: usize,
            weights: &[&[u8]],
            segs: &[Segment],
            xs: &[f32],
            outs: &mut [f32],
            threads: usize,
        ) -> f64 {
            let rb = row_bytes(ty, n_in).unwrap();
            let per = segs.len().div_ceil(threads.max(1));
            let mut ranges: Vec<(usize, usize, usize, usize)> = Vec::new(); // lo, hi, o_lo, o_hi
            let mut i = 0usize;
            while i < segs.len() {
                let hi = (i + per).min(segs.len());
                let o_lo = segs[i].out_off as usize;
                let last = &segs[hi - 1];
                let o_hi = last.out_off as usize + last.m as usize * n_out;
                ranges.push((i, hi, o_lo, o_hi));
                i = hi;
            }
            // outs を ranges の順に重ならない部分スライスへ切る (範囲は連続している)。
            let mut parts: Vec<&mut [f32]> = Vec::with_capacity(ranges.len());
            let mut rest: &mut [f32] = outs;
            let mut cur = 0usize;
            for &(_, _, o_lo, o_hi) in &ranges {
                let (gap, tail) = rest.split_at_mut(o_lo - cur);
                debug_assert!(gap.is_empty(), "outs の範囲に隙間がある");
                let (mid, tail) = tail.split_at_mut(o_hi - o_lo);
                parts.push(mid);
                rest = tail;
                cur = o_hi;
            }
            let t0 = Instant::now();
            std::thread::scope(|s| {
                let mut hs = Vec::with_capacity(parts.len());
                for ((lo, hi, _, _), part) in ranges.into_iter().zip(parts) {
                    let part_segs = &segs[lo..hi];
                    let base = part_segs[0].out_off as usize;
                    hs.push(s.spawn(move || {
                        grouped_range_cpu(
                            ty, rb, n_in, n_out, weights, part_segs, xs, base, part,
                        )
                        .unwrap();
                    }));
                }
                for h in hs {
                    h.join().unwrap();
                }
            });
            t0.elapsed().as_secs_f64()
        }

        for (tag, ty, n_in, n_out) in [
            ("Q4_K (gate/up)", GgmlType::Q4K, PERF_N_IN, PERF_N_OUT),
            ("Q5_1 (down)", GgmlType::Q5_1, PERF_N_OUT, PERF_N_IN),
        ] {
            let rb = row_bytes(ty, n_in).unwrap();
            let ws: Vec<Vec<u8>> = (0..n_expert)
                .map(|e| make_expert(ty, n_in, n_out, e))
                .collect();
            let wrefs: Vec<&[u8]> = ws.iter().map(|v| v.as_slice()).collect();
            let x_all = make_xs(ntok, n_in);
            let gb = build_grouped_batch(&by, &x_all, n_in, n_out).unwrap();
            let mut outs = vec![0.0f32; gb.outs_len(n_out)];
            let macs = gb.m_total as f64 * n_in as f64 * n_out as f64;
            eprintln!(
                "\n== {tag}: n_in={n_in} n_out={n_out} expert={n_expert} m_total={} (m {}..{}) \
                 重み {} MB ({rb} B/行) ==",
                gb.m_total,
                ms.iter().min().unwrap(),
                ms.iter().max().unwrap(),
                n_expert * n_out * rb / (1024 * 1024)
            );
            let mut base: Option<Vec<f32>> = None;
            for threads in [1usize, 8] {
                let mut best = f64::INFINITY;
                for _ in 0..REPS {
                    best = best.min(run_split(
                        ty, n_in, n_out, &wrefs, &gb.segs, &gb.xs, &mut outs, threads,
                    ));
                }
                assert!(outs.iter().any(|v| *v != 0.0), "{tag}: 出力が全 0 (計測が無意味)");
                assert!(best > 0.0, "{tag}: 計測値が 0");
                match &base {
                    None => base = Some(outs.clone()),
                    Some(b) => {
                        // スレッド数で値が変わらないこと (segment は互いに素)
                        for (a, w) in outs.iter().zip(b.iter()) {
                            assert_eq!(a.to_bits(), w.to_bits(), "{tag}: {threads}T で値が違う");
                        }
                    }
                }
                eprintln!(
                    "  {threads}T: {:6.3} s = {:6.2} GMAC/s ({:.2} G MAC)",
                    best,
                    macs / best / 1e9,
                    macs / 1e9
                );
            }
        }
    }

}
