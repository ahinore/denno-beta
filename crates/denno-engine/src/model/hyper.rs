//! 出典: llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! P-MOE-D1: hyper-connection の CPU 参照実装 (Qwen3.8-Flash-Next / `qwen4exp`)。
//!
//! 層の入口/出口に置かれる「hc 本の並列 residual ストリーム」の混合 (mix) と
//! 書き戻し (combine) だけを扱う。推論経路には接続しない (参照実装)。
//! **f32 の逐次演算だけ**で計算し、f64 を使わず、並列化もしない。
//! GPU カーネル側とビット一致させる基準になるので、加算順序を固定する。
//!
//! 演算順序 (この順序自体が数値契約。入れ替えない):
//!
//! `hc_mix`:
//! 1. ストリームごとに RMSNorm (各ストリームの `n_embd` 要素で二乗平均、eps 加算、
//!    rsqrt)。参照実装の `ggml_rms_norm` はストリーム単位 (`ne0 = n_embd` の行ごと)。
//! 2. gamma 乗算: `xn[s][d] = norm[s][d] * w_norm[s*n_embd + d]`
//! 3. ストリームを連結して `[hc_dim]` とみなす。
//! 4. `lo = w_down @ xn` を f32 逐次 dot で計算し、その後 `1/hc` を掛ける
//!    (掛けるのは matmul の後)。
//! 5. `lo = silu(lo)` (`silu(x) = x * sigmoid(x)`、`sigmoid` は `1/(1+exp(-x))`)
//! 6. `gate = w_up @ lo` (f32 逐次 dot)
//! 7. `gated[s][d] = xn[s][d] * sigmoid(gate[s][d])`
//! 8. `mixed[d] = (1/hc) * sum_s gated[s][d]` (s の昇順に加算)
//!
//! inject:
//! - `w_down_inject` があるとき: `lo` と `inject` は 1 回の matmul の行を分けた
//!   もの。行 `0..low_rank` が `lo`、行 `low_rank..low_rank+hc` が `inject`。
//!   `1/hc` は分割前に掛ける。`inject` には silu を掛けない。
//! - 無いとき: `inject` は 0 で返す (呼び出し側が別経路で作る)。
//!
//! `hc_combine`:
//! 1. `w[s] = 2 * sigmoid(inject[s])` (0 のとき 1 になり、素の residual 加算に一致)
//! 2. `out[s][d] = residual[s][d] + block_out[d] * w[s]`
//!
//! `hc_head` は `hc_mix` と同じ手順で `mixed` を作って返す (combine はしない)。
//! 参照実装では `output_norm` が無く、head の mixer がそれを担う。
//!
//! 重みの読み出し・量子型の展開は呼び出し側の責務。ここは f32 スライスだけを見る。
//! レイアウトはすべて行優先で、ストリームは `[stream][dim]` (`streams[s*n_embd + d]`)。

use crate::tensor::rmsnorm;

/// hyper-connection の形状パラメータ。
///
/// 実ファイル (GGUF メタデータ) の値: `hc = 4`、`low_rank = 320`、`n_embd = 2560`
/// (`hc_dim = hc * n_embd = 10240`)。`eps` は `hparams.f_norm_rms_eps` をそのまま受ける。
pub struct HyperParams {
    /// 並列 residual ストリーム本数 (`hyper_connection.count`)。
    pub hc: usize,
    /// 埋め込み次元 (`n_embd`)。`hc_dim = hc * n_embd`。
    pub n_embd: usize,
    /// mix の低ランク次元 (`hyper_connection.low_rank`)。
    pub low_rank: usize,
    /// RMSNorm の eps (`f_norm_rms_eps`、実ファイルは 1e-6 相当)。
    pub eps: f32,
}

/// hc 本のストリーム。レイアウトは `[stream][dim]` を 1 トークン分 (row-major)。
/// `streams[s*n_embd + d]`。
pub struct HyperStreams {
    pub data: Vec<f32>,
    pub hc: usize,
    pub n_embd: usize,
}

impl HyperStreams {
    /// 同じ内容を hc 回複製する (参照実装の `res_hc = ggml_repeat_4d(inpL, n_embd, hc, ...)`)。
    pub fn from_embed(embd: &[f32], p: &HyperParams) -> Self {
        assert_eq!(embd.len(), p.n_embd, "embd.len() は n_embd と一致する必要がある");
        assert!(p.hc >= 1, "hc が 0");
        let mut data = Vec::with_capacity(p.hc * p.n_embd);
        for _ in 0..p.hc {
            data.extend_from_slice(embd);
        }
        Self { data, hc: p.hc, n_embd: p.n_embd }
    }

    /// ストリーム `s` の `n_embd` 要素。
    pub fn stream(&self, s: usize) -> &[f32] {
        assert!(s < self.hc, "stream index {} は hc={} の範囲外", s, self.hc);
        let e = self.n_embd;
        &self.data[s * e..(s + 1) * e]
    }

    /// 全ストリームの平均。`s` の昇順に逐次加算してから `1/hc` を掛ける
    /// (参照実装 `hc_stream_mean` と同じ順序)。
    pub fn mean(&self) -> Vec<f32> {
        assert!(self.hc >= 1, "hc が 0");
        let e = self.n_embd;
        assert_eq!(self.data.len(), self.hc * e, "data.len() は hc*n_embd と一致する必要がある");
        let mut acc = self.stream(0).to_vec();
        for s in 1..self.hc {
            let src = &self.data[s * e..(s + 1) * e];
            for d in 0..e {
                acc[d] += src[d];
            }
        }
        let inv_hc = 1.0f32 / self.hc as f32;
        for v in acc.iter_mut() {
            *v *= inv_hc;
        }
        acc
    }
}

/// `sigmoid(x) = 1/(1+exp(-x))`。
///
/// 参照実装の `ggml_sigmoid` と同じ式。GPU カーネルはこの式 (除算 1 回) に合わせること。
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(x) = x * sigmoid(x)`。
///
/// 参照実装の `ggml_silu` は `x / (1+exp(-x))` (除算) だが、本ファイルは
/// タスク仕様の `x * (1/(1+exp(-x)))` (逆数の乗算) を採る。最終ビットが
/// 除算版とずれ得るので、GPU 側はこの式に合わせること。
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// 行ベクトル同士の内積 (f32 逐次加算)。
///
/// ビット一致の基準なので 4 分割アキュムレータや FMA 化をしない
/// (`#[inline(never)]` は最適化で演算列が変わらないようにするための目印)。
#[inline(never)]
fn dot_row(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += w[i] * x[i];
    }
    acc
}

/// ストリームごとの RMSNorm + gamma 乗算の結果を `[hc][n_embd]` で返す。
///
/// `crate::tensor::rmsnorm` は `x * inv * w` の順 (二乗和は f32 逐次) なので、
/// 参照実装の「`ggml_rms_norm` → `ggml_mul(w_norm)`」と同じ丸め順になる。
fn normed_streams(x: &HyperStreams, w_norm: &[f32], p: &HyperParams) -> Vec<f32> {
    let e = p.n_embd;
    let hc = p.hc;
    assert_eq!(x.hc, hc, "ストリーム本数が hc と一致しない");
    assert_eq!(x.n_embd, e, "ストリーム次元が n_embd と一致しない");
    assert_eq!(x.data.len(), hc * e, "data.len() は hc*n_embd と一致する必要がある");
    assert_eq!(w_norm.len(), hc * e, "w_norm は [hc][n_embd] (変換器が 1+w に畳んだ値)");
    let mut xn = Vec::with_capacity(hc * e);
    for s in 0..hc {
        let gamma = &w_norm[s * e..(s + 1) * e];
        xn.extend_from_slice(&rmsnorm(x.stream(s), gamma, e, p.eps));
    }
    xn
}

/// `hc_mix` / `hc_head` の共通本体。`mixed` と `inject` を返す。
fn mix_internal(
    x: &HyperStreams,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_down_inject: Option<&[f32]>,
    p: &HyperParams,
) -> (Vec<f32>, Vec<f32>) {
    let e = p.n_embd;
    let hc = p.hc;
    let lr = p.low_rank;
    let hc_dim = hc * e;
    assert!(hc >= 1, "hc が 0");
    assert!(e >= 1, "n_embd が 0");
    assert!(lr >= 1, "low_rank が 0");
    assert_eq!(w_down.len(), lr * hc_dim, "w_down は [low_rank][hc_dim]");
    assert_eq!(w_up.len(), hc_dim * lr, "w_up は [hc_dim][low_rank]");
    if let Some(di) = w_down_inject {
        assert_eq!(di.len(), (lr + hc) * hc_dim, "w_down_inject は [low_rank+hc][hc_dim]");
    }

    // 1-3. ストリームごとの RMSNorm + gamma 乗算 → [hc_dim] に連結
    let xn = normed_streams(x, w_norm, p);

    // 4-5. lo = silu( (w_down @ xn) * 1/hc )
    //      w_down_inject があるときは行 0..low_rank が lo の行 (分割前の 1 回の matmul)。
    let down_src: &[f32] = match w_down_inject {
        Some(di) => di,
        None => w_down,
    };
    let mut lo = Vec::with_capacity(lr);
    for r in 0..lr {
        let row = &down_src[r * hc_dim..(r + 1) * hc_dim];
        lo.push(dot_row(row, &xn));
    }
    hc_lo_apply(&mut lo, p);

    // inject: 行 low_rank..low_rank+hc を 1/hc 倍 (silu は掛けない)。
    // w_down_inject が無ければ 0 (呼び出し側が別経路で作る)。
    let mut inject = vec![0.0f32; hc];
    if let Some(di) = w_down_inject {
        for r in 0..hc {
            let row = &di[(lr + r) * hc_dim..(lr + r + 1) * hc_dim];
            inject[r] = dot_row(row, &xn);
        }
        hc_inject_apply(&mut inject, p);
    }

    // 6. gate = w_up @ lo
    let mut gate = Vec::with_capacity(hc_dim);
    for r in 0..hc_dim {
        gate.push(dot_row(&w_up[r * lr..(r + 1) * lr], &lo));
    }

    // 7. gated[s][d] = xn[s][d] * sigmoid(gate[s][d])
    // 8. mixed[d] = (1/hc) * sum_s gated[s][d]  (s の昇順に加算してから掛ける)
    let mixed = hc_mixed(&xn, &gate, p);

    (mixed, inject)
}

/// ストリームごとの RMSNorm + gamma 乗算 (`hc_mix` の手順 1-3)。
///
/// 量子化重みの経路 (`GgufLinear`) からも**同じ丸め順序**で使えるように公開して
/// いる。戻り値は `[hc][n_embd]` の連結 (`[hc_dim]`)。
pub fn grouped_rms_norm(x: &HyperStreams, w_norm: &[f32], p: &HyperParams) -> Vec<f32> {
    normed_streams(x, w_norm, p)
}

/// `hc_mix` の手順 4-5 を in-place で当てる: `lo[r] = silu(lo[r] * (1/hc))`。
///
/// 入力は `w_down @ xn` の**生の内積** (量子化重みの GEMV の出力をそのまま渡せる)。
/// 1/hc は matmul の後に掛ける (掛ける位置が変わると最終ビットが変わる)。
pub fn hc_lo_apply(lo: &mut [f32], p: &HyperParams) {
    assert_eq!(lo.len(), p.low_rank, "lo は [low_rank]");
    let inv_hc = 1.0f32 / p.hc as f32;
    for v in lo.iter_mut() {
        *v = silu(*v * inv_hc);
    }
}

/// `hc_mix` の手順 7-8: `mixed[d] = (1/hc) * Σ_s xn[s][d] * sigmoid(gate[s][d])`。
///
/// 加算は `s` の昇順、`1/hc` は和の後 (順序を変えると最終ビットが変わる)。
pub fn hc_mixed(xn: &[f32], gate: &[f32], p: &HyperParams) -> Vec<f32> {
    let e = p.n_embd;
    let hc = p.hc;
    assert_eq!(xn.len(), hc * e, "xn は [hc][n_embd]");
    assert_eq!(gate.len(), hc * e, "gate は [hc][n_embd]");
    let inv_hc = 1.0f32 / hc as f32;
    let mut mixed = vec![0.0f32; e];
    for d in 0..e {
        let mut acc = xn[d] * sigmoid(gate[d]);
        for s in 1..hc {
            let i = s * e + d;
            acc += xn[i] * sigmoid(gate[i]);
        }
        mixed[d] = acc * inv_hc;
    }
    mixed
}

/// `hc_mix` の inject の後処理: `inject[r] *= 1/hc` (silu は掛けない)。
pub fn hc_inject_apply(inject: &mut [f32], p: &HyperParams) {
    assert_eq!(inject.len(), p.hc, "inject は [hc]");
    let inv_hc = 1.0f32 / p.hc as f32;
    for v in inject.iter_mut() {
        *v *= inv_hc;
    }
}

/// attn / ffn の入口。`mixed` は `[n_embd]`、`inject` は `[hc]`。
///
/// 形状:
/// - `w_norm`: `[hc][n_embd]` gamma (変換器が `1+w` に畳んだ値)
/// - `w_down`: `[low_rank][hc_dim]` row-major
/// - `w_up`: `[hc_dim][low_rank]` row-major
/// - `w_down_inject`: `[low_rank + hc][hc_dim]` (あれば down と inject を 1 回で計算)
pub fn hc_mix(
    x: &HyperStreams,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_down_inject: Option<&[f32]>,
    p: &HyperParams,
) -> (Vec<f32>, Vec<f32>) {
    mix_internal(x, w_norm, w_down, w_up, w_down_inject, p)
}

/// 出口。`residual` に `block_out` をストリーム別の重みで足す。
///
/// `w[s] = 2 * sigmoid(inject[s])` なので、`inject` が全て 0 なら `w = 1` になり
/// 素の residual 加算 (`residual[s][d] + block_out[d]`) に一致する。
pub fn hc_combine(
    residual: &HyperStreams,
    block_out: &[f32],
    inject: &[f32],
    p: &HyperParams,
) -> HyperStreams {
    let e = p.n_embd;
    let hc = p.hc;
    assert_eq!(residual.hc, hc, "residual.hc が hc と一致しない");
    assert_eq!(residual.n_embd, e, "residual.n_embd が n_embd と一致しない");
    assert_eq!(residual.data.len(), hc * e, "residual.data.len() は hc*n_embd と一致する必要がある");
    assert_eq!(block_out.len(), e, "block_out は [n_embd]");
    assert_eq!(inject.len(), hc, "inject は [hc]");

    let mut w = Vec::with_capacity(hc);
    for s in 0..hc {
        w.push(2.0f32 * sigmoid(inject[s]));
    }

    let mut data = Vec::with_capacity(hc * e);
    for s in 0..hc {
        let ws = w[s];
        let src = residual.stream(s);
        for d in 0..e {
            data.push(src[d] + block_out[d] * ws);
        }
    }
    HyperStreams { data, hc, n_embd: e }
}

/// 最終段。`output_norm` は無く、head の mixer がそれを担う。
/// `hc_mix` と同じ手順で `mixed` を作って返す (combine はしない)。
pub fn hc_head(
    x: &HyperStreams,
    w_norm: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    p: &HyperParams,
) -> Vec<f32> {
    mix_internal(x, w_norm, w_down, w_up, None, p).0
}

// ---------------------------------------------------------------------------
// バッチ版 (トークン方向にまとめる)
//
// 層優先 prefill は nt トークンをまとめて持っているのに、要素演算をトークンごとに
// 呼ぶと、関数呼び出しと Vec 確保がトークン数ぶん乗る (1 トークン = hc_dim 要素
// = 実形 40 KB)。ここでは `[nt][...]` の連続バッファを 1 回で埋める。
//
// 数値: どの関数も要素ごとに独立で、トークン内の演算順序は逐次版と同一
// (加算する順序・1/hc を掛ける位置を変えない)。したがって `to_bits()` で一致する。
// トークン方向に rayon で割っても、トークン間で共有する状態は無いので同じ。
// ---------------------------------------------------------------------------

/// バッチ版の要素演算を rayon で割る最小トークン数。
///
/// 1 トークンあたり hc_dim (実形 10240) 要素を触るので、64 トークンで 640 K 要素
/// = 2.5 MB になり、分割の損益分岐を超える。`low_rank` 行 (実形 320) や
/// `hc` 行 (実形 4) の小さい行はトークン並列にしない (1 行が L1 に収まり、
/// rayon の起動コストだけが乗る)。
pub const HC_BATCH_PAR_MIN_TOKENS: usize = 64;

/// テスト用: バッチ版が rayon 経路に入った回数。
/// 「緑 = 分岐が発動した証拠」ではないので、nt >= [`HC_BATCH_PAR_MIN_TOKENS`] の
/// テストでここを数える。
#[cfg(test)]
pub(crate) static HC_BATCH_PAR_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn bump_par_calls() {
    #[cfg(test)]
    HC_BATCH_PAR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`grouped_rms_norm`] のバッチ版。戻り値は `[nt][hc][n_embd]` の連結。
///
/// 1 トークンぶんは [`grouped_rms_norm`] と同じ計算 (ストリームごとに逐次二乗和 →
/// rsqrt → `x * inv * gamma`) なので、トークンをまとめても値はビット一致する。
pub fn grouped_rms_norm_batch(xs: &[HyperStreams], w_norm: &[f32], p: &HyperParams) -> Vec<f32> {
    let e = p.n_embd;
    let hc = p.hc;
    let hc_dim = hc * e;
    assert!(hc >= 1, "hc が 0");
    assert!(e >= 1, "n_embd が 0");
    assert_eq!(w_norm.len(), hc_dim, "w_norm は [hc][n_embd] (変換器が 1+w に畳んだ値)");
    for (t, x) in xs.iter().enumerate() {
        assert_eq!(x.hc, hc, "トークン {t}: ストリーム本数が hc と一致しない");
        assert_eq!(x.n_embd, e, "トークン {t}: ストリーム次元が n_embd と一致しない");
        assert_eq!(
            x.data.len(),
            hc_dim,
            "トークン {t}: data.len() は hc*n_embd と一致する必要がある"
        );
    }

    let mut out = vec![0.0f32; xs.len() * hc_dim];
    let fill = |row: &mut [f32], x: &HyperStreams| {
        for s in 0..hc {
            let src = &x.data[s * e..(s + 1) * e];
            let gamma = &w_norm[s * e..(s + 1) * e];
            // `crate::tensor::rmsnorm` と同じ丸め順 (逐次二乗和 → rsqrt → x*inv*w)
            let ss = src.iter().map(|v| v * v).sum::<f32>() / e as f32;
            let inv = 1.0 / (ss + p.eps).sqrt();
            let dst = &mut row[s * e..(s + 1) * e];
            for i in 0..e {
                dst[i] = src[i] * inv * gamma[i];
            }
        }
    };

    #[cfg(feature = "par")]
    if xs.len() >= HC_BATCH_PAR_MIN_TOKENS {
        use rayon::prelude::*;
        bump_par_calls();
        out.par_chunks_mut(hc_dim)
            .zip(xs.par_iter())
            .for_each(|(row, x)| fill(row, x));
        return out;
    }

    for (row, x) in out.chunks_mut(hc_dim).zip(xs.iter()) {
        fill(row, x);
    }
    out
}

/// [`hc_lo_apply`] のバッチ版。`lo` は `[nt][low_rank]`。
///
/// 行が小さい (実形 320 要素 = 1.25 KB) のでトークン並列はしない。
pub fn hc_lo_apply_batch(lo: &mut [f32], nt: usize, p: &HyperParams) {
    assert!(p.low_rank >= 1, "low_rank が 0");
    assert_eq!(lo.len(), nt * p.low_rank, "lo は [nt][low_rank]");
    let inv_hc = 1.0f32 / p.hc as f32;
    for row in lo.chunks_exact_mut(p.low_rank) {
        for v in row.iter_mut() {
            *v = silu(*v * inv_hc);
        }
    }
}

/// [`hc_mixed`] のバッチ版。`xn` / `gate` は `[nt][hc][n_embd]` の連結、
/// 戻り値は `[nt][n_embd]` の連結。
///
/// 加算は `s` の昇順、`1/hc` は和の後 (逐次版と同じ順序)。内側は `n_embd` 方向に
/// 連続アクセスする (`d` ごとに `s` を回すと stride が n_embd になる)。
pub fn hc_mixed_batch(xn: &[f32], gate: &[f32], nt: usize, p: &HyperParams) -> Vec<f32> {
    let e = p.n_embd;
    let hc = p.hc;
    let hc_dim = hc * e;
    assert!(hc >= 1, "hc が 0");
    assert!(e >= 1, "n_embd が 0");
    assert_eq!(xn.len(), nt * hc_dim, "xn は [nt][hc][n_embd]");
    assert_eq!(gate.len(), xn.len(), "gate は xn と同じ長さ ([nt][hc][n_embd])");

    let inv_hc = 1.0f32 / hc as f32;
    let mut mixed = vec![0.0f32; nt * e];
    let fill = |m: &mut [f32], x: &[f32], g: &[f32]| {
        for d in 0..e {
            m[d] = x[d] * sigmoid(g[d]);
        }
        for s in 1..hc {
            let xs = &x[s * e..(s + 1) * e];
            let gs = &g[s * e..(s + 1) * e];
            for d in 0..e {
                m[d] += xs[d] * sigmoid(gs[d]);
            }
        }
        for v in m.iter_mut() {
            *v *= inv_hc;
        }
    };

    #[cfg(feature = "par")]
    if nt >= HC_BATCH_PAR_MIN_TOKENS {
        use rayon::prelude::*;
        bump_par_calls();
        mixed
            .par_chunks_mut(e)
            .zip(xn.par_chunks(hc_dim))
            .zip(gate.par_chunks(hc_dim))
            .for_each(|((m, x), g)| fill(m, x, g));
        return mixed;
    }

    for (t, m) in mixed.chunks_mut(e).enumerate() {
        fill(
            m,
            &xn[t * hc_dim..(t + 1) * hc_dim],
            &gate[t * hc_dim..(t + 1) * hc_dim],
        );
    }
    mixed
}

/// [`hc_inject_apply`] のバッチ版。`v` は `[nt][hc]`。
pub fn hc_inject_apply_batch(v: &mut [f32], nt: usize, p: &HyperParams) {
    assert!(p.hc >= 1, "hc が 0");
    assert_eq!(v.len(), nt * p.hc, "inject は [nt][hc]");
    let inv_hc = 1.0f32 / p.hc as f32;
    for row in v.chunks_exact_mut(p.hc) {
        for x in row.iter_mut() {
            *x *= inv_hc;
        }
    }
}

/// [`hc_combine`] のバッチ版 (in-place)。`res` は `[nt][hc][n_embd]`、
/// `block_out` は `[nt][n_embd]`、`inject` は `[nt][hc]`。
///
/// 逐次版はトークンごとに `HyperStreams` を作り直す (実形 40 KB の Vec を
/// 層あたり 2 回 x トークン数)。ここは `res` をそのまま書き換える。
pub fn hc_combine_batch(
    res: &mut [HyperStreams],
    block_out: &[f32],
    inject: &[f32],
    p: &HyperParams,
) {
    let e = p.n_embd;
    let hc = p.hc;
    let hc_dim = hc * e;
    let nt = res.len();
    assert!(hc >= 1, "hc が 0");
    assert!(e >= 1, "n_embd が 0");
    assert_eq!(block_out.len(), nt * e, "block_out は [nt][n_embd]");
    assert_eq!(inject.len(), nt * hc, "inject は [nt][hc]");
    for (t, r) in res.iter().enumerate() {
        assert_eq!(r.hc, hc, "トークン {t}: residual.hc が hc と一致しない");
        assert_eq!(r.n_embd, e, "トークン {t}: residual.n_embd が n_embd と一致しない");
        assert_eq!(
            r.data.len(),
            hc_dim,
            "トークン {t}: residual.data.len() は hc*n_embd と一致する必要がある"
        );
    }

    let fill = |r: &mut HyperStreams, bo: &[f32], inj: &[f32]| {
        for s in 0..hc {
            // 逐次版と同じ式・同じ順序: out = residual + block_out * (2*sigmoid(inject))
            let ws = 2.0f32 * sigmoid(inj[s]);
            let row = &mut r.data[s * e..(s + 1) * e];
            for d in 0..e {
                row[d] = row[d] + bo[d] * ws;
            }
        }
    };

    #[cfg(feature = "par")]
    if nt >= HC_BATCH_PAR_MIN_TOKENS {
        use rayon::prelude::*;
        bump_par_calls();
        res.par_iter_mut()
            .zip(block_out.par_chunks(e))
            .zip(inject.par_chunks(hc))
            .for_each(|((r, bo), inj)| fill(r, bo, inj));
        return;
    }

    for (t, r) in res.iter_mut().enumerate() {
        fill(r, &block_out[t * e..(t + 1) * e], &inject[t * hc..(t + 1) * hc]);
    }
}
