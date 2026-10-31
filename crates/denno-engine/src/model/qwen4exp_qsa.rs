//! P-MOE-F5b: QSA (indexer による key 選択) の材料づくり。
//!
//! 出典: llama.cpp `src/models/qwen4exp.cpp` (MIT License,
//! Copyright (c) 2023-2026 The ggml authors) の `build_qsa_sel` と
//! `src/llama-memory-hybrid-idx.cpp` の仕様。本ファイルは Rust による再実装であり、
//! 逐語コピーは含まない。ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! # 担当範囲 (F5b)
//!
//! full attention 層が「どの cell (= KV 位置) を見るか」を決める indexer を作る。
//! [`QsaState::push_token`] は 1 トークンぶんの生キーを積み、その時点で見えるべき
//! cell の添字列 (`0..=pos` の部分集合、昇順・重複なし) を返す。
//! 選択結果を attention に配線するのは G2 (呼び出し側
//! [`crate::model::qwen4exp_attn::FullAttn`] が `DENNO_QSA=1` のとき
//! [`crate::model::qwen4exp_attn::AttnLayer::attend_visible`] に渡す)。
//!
//! # 数値の契約
//!
//! 1. 生の index key は `k_raw = k_proj @ cur` の `idx_dim` 次元。**正規化も rope もしない**。
//!    トークンごとに 1 本ずつ、層ごとに別の配列へ積む。
//! 2. pool は `kpool` 個そろって初めて成立する。端数は pool にせず tail として残す。
//!    `pooled = mean(k_raw[first..first+kpool])` の平均は
//!    「先頭セルをコピー → セル昇順に加算 → `1/kpool` の乗算」の順
//!    (参照実装の `ggml_add` の鎖 + `ggml_scale` と同じ。除算ではない)。
//! 3. そのあと `rmsnorm(k_norm)` → `rope(pos = 先頭メンバーの位置)`。
//!    **この順序が本質**で、平均 → norm → rope を入れ替えると値が変わる (テストで固定)。
//! 4. クエリは `q_proj @ cur` を head ごとに `rmsnorm(q_norm)` → `rope(pos = トークン位置)`。
//! 5. `score[pool] = Σ_h max(0, dot(q_h, pooled_pool)) * (1/sqrt(idx_dim))`。
//!    ReLU は head ごとの dot の後、head 昇順に加算する。
//! 6. 選ぶのは `min(n_pool, top_k / kpool)` 個 (同点は添字の小さい方)。
//!    残すのは「選ばれた pool の全 cell」+「tail」+「現在のトークン」。
//!    文脈が短く `n_pool <= top_k / kpool` なら全 pool が入るので dense と一致する。
//!
//! pool / score / select の実体は [`crate::model::indexer`] (P-MOE-D2 の CPU 参照実装) を
//! 共有する。D2 側は端数の pool も平均してスコアに載せる (ceil) が、本モジュールは
//! **完成 pool だけ**を渡す (floor) ので、端数が top-k の枠を食うことはない。
//!
//! # 仮定 (実機アンカーで確認する)
//!
//! indexer の rope はモデル共通の `rope.dimension_count` / `rope.freq_base` を
//! `idx_dim` 次元ベクトルの先頭 `n_rot` 次元に当てる (参照実装が attention と同じ
//! rope 引数で呼ばれる前提)。`n_rot` が奇数・`idx_dim` 超・0 なら読込時に落とす。
//!
//! # 重みの持ち方
//!
//! `indexer.q_proj` / `indexer.k_proj` は量子化ブロック (実ファイルは BF16) のまま
//! [`GgufLinear`] で持つ。f32 へ展開しない。`q_norm` / `k_norm` (各 `[idx_dim]`) だけ
//! f32 常駐。

use anyhow::{bail, Context, Result};

use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::indexer::{pool_keys, score_pools, select_cells, IndexerParams};
use crate::model::linear_gguf::GgufLinear;
use crate::model::qwen4exp::{LayerKind, Qwen4ExpParams};
use crate::model::qwen4exp_attn::rope_neox_partial;
use crate::quant::WeightBlocks;
use crate::tensor::qknorm_rows_inplace;

// ---------------------------------------------------------------------------
// 1 層ぶんの indexer
// ---------------------------------------------------------------------------

/// full attention 層 1 つぶんの indexer の重みと形状。
pub struct QsaLayer {
    /// 入力長 (`embedding_length`)
    pub n_embd: usize,
    /// indexer の head 数 (`attention.indexer.head_count` = 4)
    pub n_idx_h: usize,
    /// head あたりのキー長 (`attention.indexer.key_length` = 128)
    pub idx_dim: usize,
    /// pool の cell 数 (`compress_ratios[il]`。QSA 層は 4)
    pub kpool: usize,
    /// 残す生 cell 数の上限 (`attention.indexer.top_k` = 2048)
    pub top_k: usize,
    /// rope を当てる次元数 (`rope.dimension_count`)
    pub n_rot: usize,
    /// rope の底 (`rope.freq_base`)
    pub rope_theta: f32,
    /// RMSNorm の eps (`attention.layer_norm_rms_epsilon`)
    pub rms_eps: f32,
    /// `indexer.q_norm.weight` — `[idx_dim]` (head 間で共有)
    pub q_norm: Vec<f32>,
    /// `indexer.k_norm.weight` — `[idx_dim]`
    pub k_norm: Vec<f32>,
    /// `indexer.q_proj.weight` — in `n_embd` → out `n_idx_h * idx_dim`
    q_proj: GgufLinear,
    /// `indexer.k_proj.weight` — in `n_embd` → out `idx_dim`
    k_proj: GgufLinear,
}

impl QsaLayer {
    /// `blk.{il}.indexer.*` から読む。形状・型の取り違えはここで落とす。
    pub fn from_gguf(
        f: &GgufFile,
        p: &Qwen4ExpParams,
        il: usize,
        kpool: usize,
    ) -> Result<Self> {
        if kpool == 0 {
            bail!("層 {il}: kpool = 0 では pool を作れない");
        }
        let d = p.indexer_head_size;
        let nh = p.indexer_n_head;
        if d == 0 || nh == 0 {
            bail!("層 {il}: indexer の key_length / head_count が 0 (idx_dim={d}, n_head={nh})");
        }
        if p.n_embd == 0 {
            bail!("層 {il}: embedding_length が 0 では indexer の投影が作れない");
        }
        if p.indexer_top_k == 0 || p.indexer_top_k < kpool {
            bail!(
                "層 {il}: indexer.top_k {} が kpool {kpool} 未満 (pool を 1 つも選べない)",
                p.indexer_top_k
            );
        }
        if p.n_rot == 0 || p.n_rot % 2 != 0 || p.n_rot > d {
            bail!(
                "層 {il}: rope.dimension_count {} が indexer.key_length {d} に対して不正 (偶数かつ 1..={d})",
                p.n_rot
            );
        }

        let name = |s: &str| format!("blk.{il}.indexer.{s}");
        expect_2d(f, &name("q_proj.weight"), p.n_embd, nh * d)?;
        expect_2d(f, &name("k_proj.weight"), p.n_embd, d)?;
        expect_1d(f, &name("q_norm.weight"), d)?;
        expect_1d(f, &name("k_norm.weight"), d)?;

        let q_norm = dequant_exact(f, &name("q_norm.weight"), d)?;
        let k_norm = dequant_exact(f, &name("k_norm.weight"), d)?;

        Ok(Self {
            n_embd: p.n_embd,
            n_idx_h: nh,
            idx_dim: d,
            kpool,
            top_k: p.indexer_top_k,
            n_rot: p.n_rot,
            rope_theta: p.rope_freq_base,
            rms_eps: p.rms_eps,
            q_norm,
            k_norm,
            q_proj: linear(f, &name("q_proj.weight"), p.n_embd, nh * d)?,
            k_proj: linear(f, &name("k_proj.weight"), p.n_embd, d)?,
        })
    }

    /// 共有実装 ([`crate::model::indexer`]) に渡すパラメータ。
    pub fn indexer_params(&self) -> IndexerParams {
        IndexerParams {
            n_head: self.n_idx_h,
            idx_dim: self.idx_dim,
            top_k: self.top_k,
            kpool: self.kpool,
            eps: self.rms_eps,
        }
    }

    /// 生の index key (`k_proj @ cur`、`idx_dim` 次元)。norm も rope もしない。
    pub fn project_k(&self, cur: &[f32]) -> Result<Vec<f32>> {
        if cur.len() != self.n_embd {
            bail!(
                "QsaLayer::project_k: 入力長 {} が n_embd = {} と違う",
                cur.len(),
                self.n_embd
            );
        }
        let mut out = vec![0f32; self.idx_dim];
        self.k_proj.gemv_label("qsa", cur, &mut out)?;
        Ok(out)
    }

    /// indexer のクエリ (`q_proj @ cur` → head ごとの rmsnorm → rope)。
    pub fn query(&self, cur: &[f32], pos: usize) -> Result<Vec<f32>> {
        if cur.len() != self.n_embd {
            bail!(
                "QsaLayer::query: 入力長 {} が n_embd = {} と違う",
                cur.len(),
                self.n_embd
            );
        }
        let mut q = vec![0f32; self.n_idx_h * self.idx_dim];
        self.q_proj.gemv_label("qsa", cur, &mut q)?;
        qknorm_rows_inplace(
            &mut q,
            1,
            self.n_idx_h,
            self.idx_dim,
            &self.q_norm,
            self.rms_eps,
        );
        rope_neox_partial(
            &mut q,
            self.n_idx_h,
            self.idx_dim,
            self.n_rot,
            pos,
            self.rope_theta,
        );
        Ok(q)
    }

    /// 完成した pool 1 つぶんのキー: 平均 → rmsnorm(k_norm) → rope(先頭メンバーの位置)。
    ///
    /// `raw` は `kpool` 本ぶん (`kpool * idx_dim` 要素) ちょうど。`start` は pool の
    /// 先頭 cell の通し番号で、rope の位置に使う。
    pub fn pool_key(&self, raw: &[f32], start: usize) -> Result<Vec<f32>> {
        if raw.len() != self.kpool * self.idx_dim {
            bail!(
                "QsaLayer::pool_key: 入力長 {} が kpool * idx_dim = {} と違う",
                raw.len(),
                self.kpool * self.idx_dim
            );
        }
        let ip = self.indexer_params();
        let mut row = pool_keys(raw, self.kpool, &ip);
        qknorm_rows_inplace(&mut row, 1, 1, self.idx_dim, &self.k_norm, self.rms_eps);
        rope_neox_partial(
            &mut row,
            1,
            self.idx_dim,
            self.n_rot,
            start,
            self.rope_theta,
        );
        Ok(row)
    }

    /// 1 層あたりの重みバイト数 (量子化ブロック + norm)。
    pub fn weight_bytes(&self) -> usize {
        self.q_proj.weight_bytes()
            + self.k_proj.weight_bytes()
            + self.q_norm.len() * 4
            + self.k_norm.len() * 4
    }
}

// ---------------------------------------------------------------------------
// 残す cell の決定
// ---------------------------------------------------------------------------

/// 選ばれた pool の全 cell + tail (未完成 pool の生 cell) + 現在のトークン。
///
/// `scores` は**完成 pool だけ**のスコア (`n_cells / kpool` 本)。
/// 戻り値は `0..n_cells` の部分集合で、昇順・重複なし。
///
/// `cur` は現在のトークンの cell 番号。`(cur + 1) % kpool == 0` のとき cur は
/// 完成 pool の末尾メンバーで、その pool が選ばれないと attention が自分自身を
/// 見られなくなる。参照実装と同じく cur は選択に関わらず常に残す。
pub fn select_visible(
    scores: &[f32],
    n_cells: usize,
    p: &IndexerParams,
    cur: usize,
) -> Result<Vec<usize>> {
    if p.kpool == 0 {
        bail!("select_visible: kpool = 0");
    }
    let n_complete = n_cells / p.kpool;
    if scores.len() != n_complete {
        bail!(
            "select_visible: scores の長さ {} が完成 pool 数 {n_complete} と違う (n_cells={n_cells}, kpool={})",
            scores.len(),
            p.kpool
        );
    }
    if cur >= n_cells {
        bail!("select_visible: 現在の cell {cur} が n_cells = {n_cells} の範囲外");
    }
    // 完成 pool だけを対象に top-k (端数は渡さないので top-k の枠を食わない)。
    let mut keep: Vec<usize> = select_cells(scores, n_complete * p.kpool, p)
        .iter()
        .map(|&c| c as usize)
        .collect();
    // tail: 未完成 pool の生 cell は常に残す。
    keep.extend(n_complete * p.kpool..n_cells);
    // 現在のトークンは常に残す。
    keep.push(cur);
    keep.sort_unstable();
    keep.dedup();
    Ok(keep)
}

// ---------------------------------------------------------------------------
// 全層ぶんの状態
// ---------------------------------------------------------------------------

/// QSA を持つ層の indexer と、トークンごとに積む生キー / pool キー。
///
/// 層ごとに独立した配列を持つ (`raw[layer]` / `pooled[layer]`)。
/// 容量は `max_ctx` で固定し、超過は**エラー**にする (黙って捨てない)。
pub struct QsaState {
    /// QSA を持つ層だけ `Some`。持たない層 (dense / ssm) は `None`。
    layers: Vec<Option<QsaLayer>>,
    /// 層ごとの生キー列 `[n_cells][idx_dim]` (平坦)
    raw: Vec<Vec<f32>>,
    /// 層ごとの pool キー列 `[n_pool][idx_dim]` (norm と rope 済み)
    pooled: Vec<Vec<f32>>,
    /// 層ごとの tail の先頭 cell 番号 (= 完成 pool の本数 * kpool)
    tail_start: Vec<usize>,
    /// 直近のトークンのクエリ (norm と rope 済み) `[n_idx_h][idx_dim]`
    q: Vec<Vec<f32>>,
    /// 直近のスコア `[n_pool]` (完成 pool のみ)
    scores: Vec<Vec<f32>>,
    max_ctx: usize,
}

impl QsaState {
    /// `compress_ratios[il] != 0` の層に indexer を組む。QSA は full attention 層だけ。
    ///
    /// 容量は `max_ctx`。`max_ctx == 0` は不正。
    pub fn from_gguf(f: &GgufFile, p: &Qwen4ExpParams, max_ctx: usize) -> Result<Self> {
        if max_ctx == 0 {
            bail!("QsaState: max_ctx = 0 は不正 (容量が無い)");
        }
        if p.compress_ratios.len() != p.n_layer {
            bail!(
                "QsaState: compress_ratios の要素数 {} が block_count {} と違う",
                p.compress_ratios.len(),
                p.n_layer
            );
        }
        let mut layers = Vec::with_capacity(p.n_layer);
        let mut n_qsa = 0usize;
        for il in 0..p.n_layer {
            let kpool = p.compress_ratios[il] as usize;
            if kpool == 0 {
                layers.push(None);
                continue;
            }
            match p.layer_kind(il) {
                Ok(LayerKind::FullAttn) => {}
                Ok(LayerKind::Ssm) => bail!(
                    "層 {il}: compress_ratio = {kpool} だが ssm 層 (QSA は full attention 層だけ)"
                ),
                Err(e) => bail!("層 {il}: {e}"),
            }
            layers.push(Some(
                QsaLayer::from_gguf(f, p, il, kpool).with_context(|| format!("層 {il} (QSA)"))?,
            ));
            n_qsa += 1;
        }
        if n_qsa == 0 {
            bail!("QsaState: QSA を持つ層が 1 本も無い (compress_ratios が全部 0)");
        }
        let n_layer = layers.len();
        Ok(Self {
            layers,
            raw: vec![Vec::new(); n_layer],
            pooled: vec![Vec::new(); n_layer],
            tail_start: vec![0; n_layer],
            q: vec![Vec::new(); n_layer],
            scores: vec![Vec::new(); n_layer],
            max_ctx,
        })
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// 層 `il` が QSA を持つか (範囲外は `false`)。
    pub fn is_qsa(&self, il: usize) -> bool {
        matches!(self.layers.get(il), Some(Some(_)))
    }
    /// 層 `il` の indexer (QSA を持たない層・範囲外は `None`)。
    pub fn layer(&self, il: usize) -> Option<&QsaLayer> {
        self.layers.get(il).and_then(|l| l.as_ref())
    }

    fn layer_ref(&self, layer: usize) -> Result<&QsaLayer> {
        match self.layers.get(layer) {
            Some(Some(l)) => Ok(l),
            Some(None) => bail!("QsaState: 層 {layer} は QSA を持たない"),
            None => bail!(
                "QsaState: 層 {layer} は範囲外 (n_layer = {})",
                self.layers.len()
            ),
        }
    }

    /// 層 `layer` に積まれている cell 数
    pub fn cells(&self, layer: usize) -> Result<usize> {
        let l = self.layer_ref(layer)?;
        Ok(self.raw[layer].len() / l.idx_dim)
    }
    /// 層 `layer` の完成 pool 数
    pub fn n_pools(&self, layer: usize) -> Result<usize> {
        let l = self.layer_ref(layer)?;
        Ok(self.pooled[layer].len() / l.idx_dim)
    }
    /// 層 `layer` の生キー `cell` (`idx_dim` 要素、norm も rope もしていない)
    pub fn raw_key(&self, layer: usize, cell: usize) -> Result<&[f32]> {
        let l = self.layer_ref(layer)?;
        let n = self.cells(layer)?;
        if cell >= n {
            bail!("QsaState::raw_key: cell {cell} が積んだ本数 {n} の範囲外");
        }
        Ok(&self.raw[layer][cell * l.idx_dim..(cell + 1) * l.idx_dim])
    }
    /// 層 `layer` の pool キー列 (norm と rope 済み、`n_pool * idx_dim` 要素)
    pub fn pooled_keys(&self, layer: usize) -> Result<&[f32]> {
        self.layer_ref(layer)?;
        Ok(&self.pooled[layer])
    }
    /// 層 `layer` の tail の先頭 cell 番号
    pub fn tail_start(&self, layer: usize) -> Result<usize> {
        self.layer_ref(layer)?;
        Ok(self.tail_start[layer])
    }
    /// 直近のトークンのクエリ (norm と rope 済み、`n_idx_h * idx_dim` 要素)
    pub fn query(&self, layer: usize) -> Result<&[f32]> {
        self.layer_ref(layer)?;
        Ok(&self.q[layer])
    }
    /// 直近のスコア (完成 pool のみ)
    pub fn scores(&self, layer: usize) -> Result<&[f32]> {
        self.layer_ref(layer)?;
        Ok(&self.scores[layer])
    }

    /// QSA を持つ全層の重みバイト数
    pub fn weight_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.weight_bytes())
            .sum::<usize>()
    }

    /// 全層の生キー / pool / クエリ / スコアを空にする (重みは残す)。
    pub fn reset(&mut self) {
        for v in self.raw.iter_mut() {
            v.clear();
        }
        for v in self.pooled.iter_mut() {
            v.clear();
        }
        for v in self.q.iter_mut() {
            v.clear();
        }
        for v in self.scores.iter_mut() {
            v.clear();
        }
        self.tail_start.iter_mut().for_each(|t| *t = 0);
    }

    /// トークンを 1 つ積み、その時点で見えるべき cell の添字列を返す。
    ///
    /// `pos` は層 `layer` に積む位置で、これまでに積んだ本数と一致すること
    /// (順序の取り違えを黙って通さない)。戻り値は昇順・重複なしの `0..=pos`。
    pub fn push_token(&mut self, layer: usize, cur: &[f32], pos: usize) -> Result<Vec<usize>> {
        // 層の借用と可変な配列の借用を分けるため、ここでは `self.layers` を直接見る。
        let l = match self.layers.get(layer) {
            Some(Some(l)) => l,
            Some(None) => bail!("QsaState::push_token: 層 {layer} は QSA を持たない"),
            None => bail!(
                "QsaState::push_token: 層 {layer} は範囲外 (n_layer = {})",
                self.layers.len()
            ),
        };
        if cur.len() != l.n_embd {
            bail!(
                "QsaState::push_token: 入力長 {} が n_embd = {} と違う",
                cur.len(),
                l.n_embd
            );
        }
        let pushed = self.raw[layer].len() / l.idx_dim;
        if pos != pushed {
            bail!("QsaState::push_token: 層 {layer} の pos {pos} が積んだ本数 {pushed} と違う");
        }
        if pos >= self.max_ctx {
            bail!(
                "QsaState::push_token: 層 {layer} が容量 max_ctx = {} に達した (pos {pos})",
                self.max_ctx
            );
        }

        // 生キー (norm も rope もしない) を積む。
        let k_raw = l.project_k(cur)?;
        self.raw[layer].extend_from_slice(&k_raw);

        // クエリ (head ごとの norm → rope)。
        let q_idx = l.query(cur, pos)?;

        let n_cells = pos + 1;
        let n_complete = n_cells / l.kpool;
        // pool は kpool 個そろったときだけ作る (端数は tail として残す)。
        if n_cells % l.kpool == 0 {
            let start = n_cells - l.kpool;
            let d = l.idx_dim;
            let row = l.pool_key(&self.raw[layer][start * d..n_cells * d], start)?;
            self.pooled[layer].extend_from_slice(&row);
        }
        self.tail_start[layer] = n_complete * l.kpool;

        let ip = l.indexer_params();
        if n_complete == 0 {
            // 完成 pool が無い = 全 cell が tail。dense と一致する。
            self.q[layer] = q_idx;
            self.scores[layer].clear();
            return Ok((0..n_cells).collect());
        }
        let scores = score_pools(&q_idx, &self.pooled[layer], n_complete, &ip);
        let visible = select_visible(&scores, n_cells, &ip, pos)?;
        self.q[layer] = q_idx;
        self.scores[layer] = scores;
        Ok(visible)
    }
}

// ---------------------------------------------------------------------------
// GGUF からの読み出し
// ---------------------------------------------------------------------------

/// `(ne0, ne1)` が期待どおりで、余次元が 1 であることを確かめる。
fn expect_2d(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != k || t.dims[1] as usize != n {
        bail!("{name}: ne={:?} が期待 [{k}, {n}] と違う", &t.dims[..2]);
    }
    if t.dims[2] != 1 || t.dims[3] != 1 {
        bail!("{name}: 2 次元を超えるテンソル (ne={:?})", t.dims);
    }
    Ok(())
}

/// `ne0 == n` の 1 次元テンソルであることを確かめる。
fn expect_1d(f: &GgufFile, name: &str, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != n || t.dims[1..].iter().any(|&d| d != 1) {
        bail!("{name}: ne={:?} が期待 [{n}] と違う", t.dims);
    }
    Ok(())
}

/// [`GgufLinear`] の `gemv` が扱える型か。
fn gemv_supported(ty: GgmlType) -> bool {
    // 融合カーネルの有無ではなく「デコードできるか」で通す (2026-10-31)。
    crate::kquant::dequant_supported(ty)
}

/// 量子化ブロックを保持したまま [`GgufLinear`] を作る (mmap のスライスを共有)。
fn linear(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<GgufLinear> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if !gemv_supported(t.ty) {
        bail!("{name}: 未対応の量子型 {:?}", t.ty);
    }
    let ty = t.ty;
    if let Some((file, off, len)) = f.tensor_mapping_parts(name) {
        if let Some(map) = f.mmap_handle_at(file) {
            return GgufLinear::new(ty, k, n, WeightBlocks::mapped(map, off, len))
                .map(|l| l.with_fast_q8(true))
                .with_context(|| format!("{name}: mmap ブロックから GgufLinear を作れない"));
        }
    }
    let bytes = f
        .tensor_bytes(name)
        .with_context(|| format!("tensor {name}"))?;
    GgufLinear::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
        .with_context(|| format!("{name}: GgufLinear を作れない"))
}

/// 小さい f32 テンソルを厳密な要素数で読み出す。
fn dequant_exact(f: &GgufFile, name: &str, want: usize) -> Result<Vec<f32>> {
    let v = f
        .dequant_tensor(name)
        .with_context(|| format!("tensor {name}"))?;
    if v.len() != want {
        bail!("{name}: 要素数 {} が期待 {want} と違う", v.len());
    }
    Ok(v)
}
