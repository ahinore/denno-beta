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

    let inv_hc = 1.0f32 / hc as f32;

    // 4-5. lo = silu( (w_down @ xn) * 1/hc )
    //      w_down_inject があるときは行 0..low_rank が lo の行 (分割前の 1 回の matmul)。
    let down_src: &[f32] = match w_down_inject {
        Some(di) => di,
        None => w_down,
    };
    let mut lo = Vec::with_capacity(lr);
    for r in 0..lr {
        let row = &down_src[r * hc_dim..(r + 1) * hc_dim];
        lo.push(silu(dot_row(row, &xn) * inv_hc));
    }

    // inject: 行 low_rank..low_rank+hc を 1/hc 倍 (silu は掛けない)。
    // w_down_inject が無ければ 0 (呼び出し側が別経路で作る)。
    let mut inject = vec![0.0f32; hc];
    if let Some(di) = w_down_inject {
        for r in 0..hc {
            let row = &di[(lr + r) * hc_dim..(lr + r + 1) * hc_dim];
            inject[r] = dot_row(row, &xn) * inv_hc;
        }
    }

    // 6. gate = w_up @ lo
    let mut gate = Vec::with_capacity(hc_dim);
    for r in 0..hc_dim {
        gate.push(dot_row(&w_up[r * lr..(r + 1) * lr], &lo));
    }

    // 7. gated[s][d] = xn[s][d] * sigmoid(gate[s][d])
    // 8. mixed[d] = (1/hc) * sum_s gated[s][d]  (s の昇順に加算してから掛ける)
    let mut mixed = vec![0.0f32; e];
    for d in 0..e {
        let mut acc = xn[d] * sigmoid(gate[d]);
        for s in 1..hc {
            let i = s * e + d;
            acc += xn[i] * sigmoid(gate[i]);
        }
        mixed[d] = acc * inv_hc;
    }

    (mixed, inject)
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
