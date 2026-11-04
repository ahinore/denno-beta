//! P-MOE-F2: Flash-Next (`qwen4exp`) の ssm 層を既存の GDN 実装で回す薄いアダプタ。
//!
//! Flash-Next の 48 層のうち 36 層 (`il % 4 != 3`) は GDN (ゲート付きデルタ則 +
//! 深さ方向 conv) と同じ演算で、`gdn::forward_token_core` がそのまま使える。
//! このモジュールがやるのは「GGUF のテンソル名から投影を読み、投影済みの値を
//! `forward_token_core` に渡す」ことだけ。新しい数式は書かない。
//!
//! 出典: llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`) の
//! `build_layer_attn_linear` / `build_qkvz` / `build_conv_state_at`。
//!
//! 諸元 (GGUF メタデータで確定):
//! - `ssm.state_size = 128` → head_k_dim = head_v_dim = 128
//! - `ssm.group_count = 16` → n_k_heads、`ssm.time_step_rank = 48` → n_v_heads
//! - `ssm.inner_size = 6144`、`ssm.conv_kernel = 4`
//! - `attn_qkv` の出力順は **[q 2048][k 2048][v 6144]** (q が 6144 ではない)。
//!   参照 `build_layer_attn_linear` の `q_conv` / `k_conv` / `v_conv` の view
//!   オフセット (0 / head_k_dim*num_k_heads / 2*head_k_dim*num_k_heads) が根拠。
//!
//! 重みの持ち方:
//! - 大きい 3 本 (`attn_qkv` / `attn_gate` / `ssm_out`) は量子化ブロックのまま
//!   [`GgufLinear`] で持つ (f32 へ展開しない。1 層 61 MB・36 層 2.2 GB)。
//!   mmap が使えるならブロックを写さずに共有写像を参照する。
//! - `ssm_beta` / `ssm_alpha` も [`GgufLinear`] の F32 経路で持つ (491 KB/層)。
//! - 小さい 4 本 (`ssm_conv1d` / `ssm_a` / `ssm_dt` / `ssm_norm`) は f32 常駐。
//!
//! **既知の差分 (F4 で判定する)**: 参照実装の出力ゲートは `sigmoid(z)` で、
//! `gdn.rs` の `silu(z)` とは違う (`qwen4exp.cpp` の `build_norm_gated`:
//! "the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not
//! silu")。本タスクの指示により `gdn.rs` の数式は変えていないので、この
//! アダプタは `silu(z)` で計算する。詳細は REPORT_P_MOE_F2.md。

use anyhow::{bail, Context, Result};

use crate::gdn::{self, GdnCore, GdnParams, GdnState};
use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::linear_gguf::{FusedLinear, GgufLinear};
use crate::model::qwen4exp::{
    one, IdentityAttn, LayerAttn, LayerKind, Qwen4ExpError, Qwen4ExpParams, SpecGdnSnap,
};
use crate::quant::WeightBlocks;
use std::sync::atomic::{AtomicU64, Ordering};

/// task 128: SsmLayer インスタンスごとの通し番号 (デバイス状態ストアの鍵)。
///
/// アドレス再利用 (ABA) で別モデルのデバイス状態を流用しないための identity。
/// 構築時 (`from_gguf_with` とテストの構造体リテラル) に 1 回発号する。
static DEVICE_NONCE: AtomicU64 = AtomicU64::new(1);

/// task 128: SsmLayer インスタンスごとの nonce を発号する (構築時に 1 回だけ)。
fn next_device_nonce() -> u64 {
    DEVICE_NONCE.fetch_add(1, Ordering::Relaxed)
}

// ---- task 128 の診断 (`DENNO_GDN_GPU_CHECK=1`。既定 off・診断専用) ----------
//
// decode の最初の数トークンだけ、GPU の層出力と scan 後の状態 (conv / ssm) を
// CPU 参照と突き合わせて 1 行ずつ出す。数値経路からは読まない (env 未指定では
// `gdn_gpu_check_on()` の static read 1 回だけ)。

/// 突き合わせる decode トークン数 (1 始まりの窓番号で判定)。
#[cfg(feature = "cuda")]
const GDN_CHECK_TOKENS: u64 = 3;

/// 出す行の上限 (層 × トークンで増えるので頭打ちにする)。
///
/// task 137: 8 では 1 トークン目の 8 層 (li=0..7) しか見えず、9 層目以降で
/// 起きた失敗を隠した。36 層を 1 トークンぶん覆える 40 にする (診断専用・
/// CPU 参照の計算は上限に関係なく走るので、増えるのはログ行だけ)。
#[cfg(feature = "cuda")]
const GDN_CHECK_MAX_LINES: u64 = 40;

#[cfg(feature = "cuda")]
static GDN_CHECK_LINES: AtomicU64 = AtomicU64::new(0);

/// 2 つの f32 列の (最大絶対差, 最初に許容を超えた (添字, gpu, cpu))。
///
/// 許容は絶対 1e-4 + 相対 1e-5 (AGENTS.md の標準)。
#[cfg(feature = "cuda")]
fn gdn_check_diff(gpu: &[f32], cpu: &[f32]) -> (f64, Option<(usize, f32, f32)>) {
    let n = gpu.len().min(cpu.len());
    let mut max_abs = 0f64;
    let mut first = None;
    for i in 0..n {
        let d = (gpu[i] as f64 - cpu[i] as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        let tol = 1e-4 + 1e-5 * (cpu[i].abs() as f64);
        if d > tol && first.is_none() {
            first = Some((i, gpu[i], cpu[i]));
        }
    }
    (max_abs, first)
}

/// [`gdn_check_diff`] の第 2 要素をログ用の短い文字列にする。
#[cfg(feature = "cuda")]
fn fmt_first(v: &Option<(usize, f32, f32)>) -> String {
    match v {
        None => "none".to_string(),
        Some((i, g, c)) => format!("[{i}] gpu={g:e} cpu={c:e}"),
    }
}

/// ssm 層 1 本ぶんの重みと状態。
pub struct SsmLayer {
    /// 層番号 (`blk.{il}.*`)。診断ダンプのゲートに使う。
    pub il: usize,
    /// task 128: デバイス状態ストア (`cuda::gdn_decode_submit`) の鍵。
    /// 構築時に 1 回発号する (アドレス再利用で別インスタンスのデバイス状態を
    /// 流用しない。cuda.rs の `GdnDecodeKey` 参照)。
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    nonce: u64,
    /// GDN のハイパーパラメータ (`GdnParams::qwen4exp()`)。
    pub p: GdnParams,
    /// [2560 -> 10240] q/k/v の融合投影 (出力順は [q 2048][k 2048][v 6144]) と
    /// [2560 -> 6144] z (出力ゲート)。
    ///
    /// task 72: どちらも同じ入力 `cur` を取るので、**行方向に連結して 1 本の
    /// GEMV** にする (量子型が同じとき)。行は独立なので値はビット一致する。
    qkv_gate: FusedLinear,
    /// [2560 -> 48] beta と alpha の生値 (どちらも F32・同じ入力なので連結)。
    beta_alpha: FusedLinear,
    /// [6144 -> 2560] 出力投影
    w_out: GgufLinear,
    /// [conv_dim][d_conv] — チャネルが外・tap が内
    conv1d: Vec<f32>,
    /// [n_v_heads] — 負値のまま (-exp(A_log))。符号を反転しない。
    ssm_a: Vec<f32>,
    /// [n_v_heads] — dt bias
    ssm_dt: Vec<f32>,
    /// [head_v_dim] — head ごとの gated RMS norm の重み
    ssm_norm: Vec<f32>,
    /// conv 履歴 + SSM 行列
    pub state: GdnState,
}

impl SsmLayer {
    /// `blk.{il}.*` から読む。ハイパーパラメータは [`GdnParams::qwen4exp`] (実モデル) 固定。
    pub fn from_gguf(f: &GgufFile, il: usize) -> Result<Self> {
        Self::from_gguf_with(f, il, GdnParams::qwen4exp())
    }

    /// ハイパーパラメータを外から与えて `blk.{il}.*` から読む。
    ///
    /// 形状・型・本数の検査を全部ここで行い、欠けたら Err。
    /// 実モデルでは [`gdn_params`] の導出値が [`GdnParams::qwen4exp`] と一致する。
    pub fn from_gguf_with(f: &GgufFile, il: usize, p: GdnParams) -> Result<Self> {
        let name = |suffix: &str| format!("blk.{il}.{suffix}");

        let n_qkv = name("attn_qkv.weight");
        let n_gate = name("attn_gate.weight");
        let n_beta = name("ssm_beta.weight");
        let n_alpha = name("ssm_alpha.weight");
        let n_conv = name("ssm_conv1d.weight");
        let n_a = name("ssm_a");
        let n_dt = name("ssm_dt.bias");
        let n_norm = name("ssm_norm.weight");
        let n_out = name("ssm_out.weight");

        // ---- 形状 (GGUF の ne0 が入力・ne1 が出力) ----------------------
        expect_2d(f, &n_qkv, p.n_embd, p.qkv_dim())?;
        expect_2d(f, &n_gate, p.n_embd, p.value_dim())?;
        expect_2d(f, &n_beta, p.n_embd, p.n_v_heads)?;
        expect_2d(f, &n_alpha, p.n_embd, p.n_v_heads)?;
        expect_2d(f, &n_out, p.value_dim(), p.n_embd)?;
        // conv1d は ne0 = tap、ne1 = チャネル。素直に f32 へ落とすと
        // [channel][tap] になる (gdn.rs が欲しい並びそのもの)。
        expect_2d(f, &n_conv, p.d_conv, p.conv_dim())?;
        expect_1d(f, &n_a, p.n_v_heads)?;
        expect_1d(f, &n_dt, p.n_v_heads)?;
        expect_1d(f, &n_norm, p.head_v_dim)?;

        // ---- 大きい 3 本 + beta/alpha はブロックのまま -------------------
        let w_qkv = linear(f, &n_qkv, p.n_embd, p.qkv_dim())?;
        let w_gate = linear(f, &n_gate, p.n_embd, p.value_dim())?;
        let w_beta = linear(f, &n_beta, p.n_embd, p.n_v_heads)?;
        let w_alpha = linear(f, &n_alpha, p.n_embd, p.n_v_heads)?;
        let w_out = linear(f, &n_out, p.value_dim(), p.n_embd)?;
        // task 72: 同じ入力を取る投影を 1 本の GEMV にまとめる。qkv+gate は
        // 実モデルではどちらも Q8_0、beta/alpha はどちらも F32 なので 2 組に
        // 分かれて連結される (型が揃わなければ自動で個別のまま)。
        let qkv_gate = FusedLinear::concat(vec![w_qkv, w_gate]);
        let beta_alpha = FusedLinear::concat(vec![w_beta, w_alpha]);

        // ---- 小さい 4 本は f32 常駐 -------------------------------------
        let conv1d = dequant_exact(f, &n_conv, p.conv_dim() * p.d_conv)?;
        let ssm_a = dequant_exact(f, &n_a, p.n_v_heads)?;
        let ssm_dt = dequant_exact(f, &n_dt, p.n_v_heads)?;
        let ssm_norm = dequant_exact(f, &n_norm, p.head_v_dim)?;

        let state = GdnState::new(&p);
        Ok(Self {
            il,
            nonce: next_device_nonce(),
            p,
            qkv_gate,
            beta_alpha,
            w_out,
            conv1d,
            ssm_a,
            ssm_dt,
            ssm_norm,
            state,
        })
    }

    /// `cur: [n_embd]` → 出力 `[n_embd]`。state を 1 トークンぶん進める。
    pub fn forward_token(&mut self, cur: &[f32]) -> Result<Vec<f32>> {
        if cur.len() != self.p.n_embd {
            bail!(
                "SsmLayer::forward_token: 入力長 {} が n_embd={} と違う",
                cur.len(),
                self.p.n_embd
            );
        }
        let prof = crate::model::qwen4exp::q4prof::enabled();

        // task 128: decode 窓 (`gdn_gpu_win`。`forward_token_at` が nt == 1 で開ける)
        // の中では、qkv_gate GEMV → GDN scan → attn_out GEMV を 1 つの GPU
        // submission (`crate::cuda::gdn_decode_submit`) にまとめて出す
        // (beta/alpha だけ CPU f32 で先に計算する)。`Some(out)` で GPU 完走
        // (この層の CPU scan は省略)、`None` で従来の CPU 経路に落ちる
        // (fail-closed)。off / 窓の外 / cuda feature 無しでは何も呼ばれない。
        #[cfg(feature = "cuda")]
        if crate::model::qwen4exp::gdn_gpu_win::active() && crate::cuda::gdn_gpu_scan_on() {
            if let Some(out) = self.forward_token_gpu(cur, prof)? {
                return Ok(out);
            }
        }

        // task 128: CPU 経路が state を読む前に、デバイスが正ならホストへ回復する
        // (直前の decode トークンを GPU が処理して device_authoritative になった
        // あとに、GPU が decline した / 窓が閉じた (バッチ prefill・spec verify)
        // とき)。HostCurrent = エントリ無し or ホスト正で no-op。
        #[cfg(feature = "cuda")]
        if crate::cuda::gdn_gpu_scan_on() {
            match crate::cuda::gdn_decode_state_take(self.nonce, self.il) {
                crate::cuda::GdnDecodeFlush::HostCurrent => {}
                crate::cuda::GdnDecodeFlush::Recovered(conv, ssm) => {
                    self.state.conv = conv;
                    self.state.ssm = ssm;
                }
                crate::cuda::GdnDecodeFlush::Lost => bail!(
                    "SsmLayer::forward_token: gdn decode 状態の回復 (D2H) に失敗 (li={}) → 要求を落とす",
                    self.il
                ),
            }
        }

        let t = std::time::Instant::now();
        // task 72: qkv と z (出力ゲート) は同じ入力 cur を取るので 1 本の GEMV に
        // まとめる。行は独立・1 行の中の加算順は連結しても変わらないので、
        // 個別に呼んだ場合とビット一致する。ゲートの時間もここ (A_QKV) に入る。
        let mut qkvg = vec![0f32; self.qkv_gate.n()];
        self.qkv_gate.gemv_concat_into("qkv", cur, &mut qkvg)?;
        let qkv = &qkvg[self.qkv_gate.range(0)];
        let z = &qkvg[self.qkv_gate.range(1)];
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_QKV, t);
        }
        // task 87: 経路内訳 (GPU intercept か CPU か)。off なら分岐 1 回。
        crate::model::qwen4exp::attn_kprof::note_qkv(t);
        let t = std::time::Instant::now();

        // task 72: beta と alpha も同じ入力なので 1 本にまとめる (どちらも F32)。
        let mut ba = vec![0f32; self.beta_alpha.n()];
        self.beta_alpha.gemv_concat_into("ssm_beta", cur, &mut ba)?;
        let beta_raw = &ba[self.beta_alpha.range(0)];
        let alpha = &ba[self.beta_alpha.range(1)];

        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_GATE, t);
        }
        crate::model::qwen4exp::attn_kprof::note_gate(t);
        let t = std::time::Instant::now();

        // 診断ダンプ (DENNO_DEBUG_GDN_DUMP_LAYER) のゲートに層番号を渡す。
        gdn::gdn_dbg_set_layer(self.il as i64);
        let cw = GdnCore {
            conv1d: &self.conv1d,
            ssm_a: &self.ssm_a,
            ssm_dt: &self.ssm_dt,
            ssm_norm: &self.ssm_norm,
        };
        // 数式・演算順序は gdn.rs のものをそのまま使う (この層では書き直さない)。
        // qwen4exp の出力ゲートは **sigmoid** (qwen35 の silu ではない)。
        let y = gdn::forward_token_core_gate(
            &self.p,
            &cw,
            qkv,
            z,
            beta_raw,
            alpha,
            &mut self.state,
            gdn::GateKind::Sigmoid,
        );

        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_GDN, t);
        }
        let t = std::time::Instant::now();

        let mut out = vec![0f32; self.p.n_embd];
        self.w_out.gemv_label("attn_out", &y, &mut out)?;
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_OUT, t);
        }
        crate::model::qwen4exp::attn_kprof::note_out(t);
        Ok(out)
    }


    /// task 128: decode 窓内の GPU submission 経路 ([`Self::forward_token`] だけが
    /// 呼ぶ。窓 + `DENNO_GDN_GPU_SCAN` の確認は呼び出し側)。
    ///
    /// - `Ok(Some(out))`: GPU で qkv_gate GEMV → scan → attn_out GEMV まで完走。
    ///   state はデバイス正本 (ホストは 1 トークン古い)。計測はここで積む。
    /// - `Ok(None)`: GPU が decline した (発行前の失敗)。呼び出し側は CPU 経路へ
    ///   落ちる。**計測は 1 つも積まない** (CPU 経路が同じ区間を積むので二重に
    ///   数えない。beta/alpha だけ二度計算になるが F32 の 2560x48 で微細)。
    /// - scan 発行後の失敗で状態回復できたときは、ホスト state へ写して `Ok(None)`
    ///   を返す (scan 出力 y が失われるので CPU で計算し直す)。
    /// - `Err`: 状態回復にも失敗した (poisoned)。要求を落とす。
    #[cfg(feature = "cuda")]
    fn forward_token_gpu(&mut self, cur: &[f32], prof: bool) -> Result<Option<Vec<f32>>> {
        // task 128 の診断 (`DENNO_GDN_GPU_CHECK=1`): 最初の数トークンだけ
        // CPU 参照と突き合わせる。参照を作る前にホスト state を最新化する
        // (直前のトークンを GPU が処理していればホストは 1 トークン古い)。
        let check = crate::cuda::gdn_gpu_check_on()
            && crate::model::qwen4exp::gdn_gpu_win::token_index() <= GDN_CHECK_TOKENS;
        let mut ref_state = if check {
            if let crate::cuda::GdnDecodeFlush::Recovered(conv, ssm) =
                crate::cuda::gdn_decode_state_take(self.nonce, self.il)
            {
                self.state.conv = conv;
                self.state.ssm = ssm;
            }
            Some(self.state.clone())
        } else {
            None
        };

        // beta/alpha は CPU F32 のまま (GPU 経路でも値を変えない。Req の契約)。
        let t_ba = std::time::Instant::now();
        let mut ba = vec![0f32; self.beta_alpha.n()];
        self.beta_alpha.gemv_concat_into("ssm_beta", cur, &mut ba)?;
        let beta = &ba[self.beta_alpha.range(0)];
        let alpha = &ba[self.beta_alpha.range(1)];

        let qkv_units = self.gdn_qkv_units();
        let t_sub = std::time::Instant::now();
        let res = {
            let req = self.gdn_decode_req(cur, beta, alpha, &qkv_units);
            crate::cuda::gdn_decode_submit(&req)
        };
        match res {
            crate::cuda::GdnDecodeRes { poisoned: true, .. } => bail!(
                "SsmLayer::forward_token: gdn decode submission の状態回復に失敗 (li={}) → 要求を落とす",
                self.il
            ),
            crate::cuda::GdnDecodeRes {
                out: Some(out),
                state_dev,
                ..
            } => {
                // 診断: CPU 参照 (qkv GEMV → scan → attn_out) と突き合わせる。
                if let (Some(mut st), Some((conv_gpu, ssm_gpu))) = (ref_state.take(), state_dev) {
                    self.gdn_check_report(cur, &mut st, &out, &conv_gpu, &ssm_gpu, beta, alpha);
                }
                if prof {
                    crate::model::qwen4exp::q4prof::add(
                        &crate::model::qwen4exp::q4prof::A_GATE,
                        t_ba,
                    );
                    crate::model::qwen4exp::q4prof::add(
                        &crate::model::qwen4exp::q4prof::A_GDN,
                        t_sub,
                    );
                }
                crate::model::qwen4exp::attn_kprof::note_gate(t_ba);
                // GPU 側の qkv / attn_out が GPU で走ったことを attn_kprof に見せる
                // (時間はこの submission の壁時計からは切り分けられないので ~0 を
                // 積む。合計 ms は scan バケット側 (q4prof A_GDN) が持つ)。
                crate::cuda::set_last_linear_ran_on_gpu(true);
                let t_mark = std::time::Instant::now();
                crate::model::qwen4exp::attn_kprof::note_qkv(t_mark);
                crate::model::qwen4exp::attn_kprof::note_out(t_mark);
                Ok(Some(out))
            }
            crate::cuda::GdnDecodeRes {
                out: None,
                state_back: Some((conv, ssm)),
                ..
            } => {
                // scan 発行後の失敗: 新状態 (デバイス正本だったぶんを含む) を
                // ホストへ戻して CPU 経路へ落ちる。このトークンは CPU で計算し直す。
                self.state.conv = conv;
                self.state.ssm = ssm;
                Ok(None)
            }
            crate::cuda::GdnDecodeRes { out: None, .. } => Ok(None),
        }
    }

    /// task 128: qkv_gate の「1 発 GEMV の単位」を呼び出し順で列挙する
    /// ([`FusedLinear::gemv_units`] + 行オフセット = [`FusedLinear::range`] の
    /// 起点。連結バッファ内の置き場所は `gemv_concat_into` と同一規約)。
    #[cfg(feature = "cuda")]
    fn gdn_qkv_units(&self) -> Vec<(crate::cuda::GdnGgufUnit<'_>, usize)> {
        self.qkv_gate
            .gemv_units()
            .iter()
            .enumerate()
            .map(|(i, u)| {
                (
                    crate::cuda::GdnGgufUnit {
                        uid: u.uid(),
                        ty: u.ty(),
                        blocks: u.blocks().as_slice(),
                        k: u.k(),
                        n: u.n(),
                    },
                    self.qkv_gate.range(i).start,
                )
            })
            .collect()
    }

    /// task 128: [`crate::cuda::gdn_decode_submit`] への要求を組む
    /// (field の意味は cuda.rs の `GdnDecodeReq` doc)。
    #[cfg(feature = "cuda")]
    fn gdn_decode_req<'a>(
        &'a self,
        cur: &'a [f32],
        beta: &'a [f32],
        alpha: &'a [f32],
        qkv_units: &'a [(crate::cuda::GdnGgufUnit<'a>, usize)],
    ) -> crate::cuda::GdnDecodeReq<'a> {
        crate::cuda::GdnDecodeReq {
            nonce: self.nonce,
            li: self.il,
            tok: crate::model::qwen4exp::gdn_gpu_win::token_index(),
            n_embd: self.p.n_embd,
            qkv_total: self.qkv_gate.n(),
            qkv_row_start: self.qkv_gate.range(0).start,
            z_row_start: self.qkv_gate.range(1).start,
            qkv_units,
            out_unit: crate::cuda::GdnGgufUnit {
                uid: self.w_out.uid(),
                ty: self.w_out.ty(),
                blocks: self.w_out.blocks().as_slice(),
                k: self.w_out.k(),
                n: self.w_out.n(),
            },
            cur,
            beta,
            alpha,
            conv_host: &self.state.conv,
            ssm_host: &self.state.ssm,
            wconv: &self.conv1d,
            wa: &self.ssm_a,
            wdt: &self.ssm_dt,
            wnorm: &self.ssm_norm,
            nk: self.p.n_k_heads,
            nv: self.p.n_v_heads,
            dk: self.p.head_k_dim,
            dv: self.p.head_v_dim,
            conv_dim: self.p.conv_dim(),
            d_conv: self.p.d_conv,
            eps: self.p.eps,
        }
    }

    /// task 128 の診断 (`DENNO_GDN_GPU_CHECK=1`): このトークンの GPU 結果
    /// (層出力 / scan 後の conv・ssm) を CPU 参照と突き合わせ、最大差と
    /// **最初に違った要素**を 1 行で出す。
    ///
    /// 比較の許容は絶対 1e-4 + 相対 1e-5 (AGENTS.md の標準)。行数は
    /// `GDN_CHECK_MAX_LINES` で頭打ちにする (診断でログを埋めない)。
    #[cfg(feature = "cuda")]
    fn gdn_check_report(
        &self,
        cur: &[f32],
        st: &mut GdnState,
        out_gpu: &[f32],
        conv_gpu: &[f32],
        ssm_gpu: &[f32],
        beta: &[f32],
        alpha: &[f32],
    ) {
        let n = GDN_CHECK_LINES.fetch_add(1, Ordering::Relaxed);
        if n >= GDN_CHECK_MAX_LINES {
            return;
        }
        // CPU 参照 (通常の CPU 経路と同じ呼び出し: qkv GEMV は intercept 込み)。
        let mut qkvg = vec![0f32; self.qkv_gate.n()];
        if let Err(e) = self.qkv_gate.gemv_concat_into("qkv", cur, &mut qkvg) {
            tracing::info!("[gdn-gpu-check] li={} CPU 参照の GEMV に失敗: {e:#}", self.il);
            return;
        }
        let qkv = &qkvg[self.qkv_gate.range(0)];
        let z = &qkvg[self.qkv_gate.range(1)];
        let cw = GdnCore {
            conv1d: &self.conv1d,
            ssm_a: &self.ssm_a,
            ssm_dt: &self.ssm_dt,
            ssm_norm: &self.ssm_norm,
        };
        let y = gdn::forward_token_core_gate(
            &self.p,
            &cw,
            qkv,
            z,
            beta,
            alpha,
            st,
            gdn::GateKind::Sigmoid,
        );
        let mut out_cpu = vec![0f32; self.p.n_embd];
        if let Err(e) = self.w_out.gemv_label("attn_out", &y, &mut out_cpu) {
            tracing::info!("[gdn-gpu-check] li={} CPU 参照の attn_out に失敗: {e:#}", self.il);
            return;
        }
        let (max_out, first_out) = gdn_check_diff(out_gpu, &out_cpu);
        let (max_conv, first_conv) = gdn_check_diff(conv_gpu, &st.conv);
        let (max_ssm, first_ssm) = gdn_check_diff(ssm_gpu, &st.ssm);
        tracing::info!(
            "[gdn-gpu-check] tok={} li={} maxabs out={max_out:.3e} conv={max_conv:.3e} ssm={max_ssm:.3e} first out={} conv={} ssm={}",
            crate::model::qwen4exp::gdn_gpu_win::token_index(),
            self.il,
            fmt_first(&first_out),
            fmt_first(&first_conv),
            fmt_first(&first_ssm),
        );
    }

    /// task 128: 系列の切れ目で GDN 状態を 0 に戻す ([`SsmAttn::reset`] が呼ぶ)。
    ///
    /// ホスト state を 0 にすると同時に、**デバイス側のコピーを無効化する**
    /// (`gdn_decode_state_invalidate`)。無効化を落とすと、次の CPU 経路
    /// (prefill の `forward_batch`) の `take` が前の系列の状態をホストへ戻し、
    /// モデル使い回しで系列を跨いで状態が化ける。
    pub fn reset_state(&mut self) {
        #[cfg(feature = "cuda")]
        if crate::cuda::gdn_gpu_scan_on() {
            crate::cuda::gdn_decode_state_invalidate_counted(self.nonce, self.il);
        }
        self.state.reset();
    }

    /// チャンクぶんの入力をまとめて回す (バッチ prefill 用)。
    ///
    /// 変えるのは行列積と GDN 本体の呼び方だけ:
    /// - 5 本の投影をチャンクで 1 回ずつ [`gemv_batch`] に流す (GPU 常駐時は
    ///   1 トークンあたり 5 往復していたのがチャンクあたり 5 往復になる)。
    /// - GDN 本体は [`gdn::forward_batch_core_gate`] にチャンク一括で渡す
    ///   (conv1d をトークン方向にまとめ、状態スキャンをヘッド範囲で並列化)。
    ///   ヘッドは互いに独立なので、トークン順を保てば要素ごとの丸め順は不変。
    ///
    /// 数値: 行列積は逐次と同じカーネルをトークンごとに当てるだけ、GDN 本体は
    /// 要素ごとに独立な演算を同じ順序で並べ替えただけなので、`forward_token` と
    /// ビット一致する (テスト `gdn_batch_matches_token_by_token_bitexact`、
    /// `ssm_batch_matches_token_by_token_bitexact`)。
    ///
    /// 診断ダンプ (`DENNO_DEBUG_GDN_DUMP_LAYER`) がこの層を指しているときだけ
    /// 従来のトークン逐次に落ちる (サブステージの追記順を保つため)。
    pub fn forward_batch(&mut self, curs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let nt = curs.len();
        if nt == 0 {
            return Ok(Vec::new());
        }
        if nt == 1 {
            return Ok(vec![self.forward_token(&curs[0])?]);
        }
        let e = self.p.n_embd;
        let mut flat = Vec::with_capacity(nt * e);
        for cur in curs {
            if cur.len() != e {
                bail!(
                    "SsmLayer::forward_batch: 入力長 {} が n_embd={e} と違う",
                    cur.len()
                );
            }
            flat.extend_from_slice(cur);
        }

        let qd = self.p.qkv_dim();
        let vd = self.p.value_dim();
        let nh = self.p.n_v_heads;

        let t = crate::model::qwen4exp::q4prof::stamp();
        // task 128: バッチ経路 (prefill / spec verify) が state を上書きする前に、
        // デバイスが正ならホストへ回復する (GPU decode のあとにバッチ経路が来ると
        // ここで戻る。HostCurrent は no-op。ゲート off では呼ばれない)。
        #[cfg(feature = "cuda")]
        if crate::cuda::gdn_gpu_scan_on() {
            match crate::cuda::gdn_decode_state_take(self.nonce, self.il) {
                crate::cuda::GdnDecodeFlush::HostCurrent => {}
                crate::cuda::GdnDecodeFlush::Recovered(conv, ssm) => {
                    self.state.conv = conv;
                    self.state.ssm = ssm;
                }
                crate::cuda::GdnDecodeFlush::Lost => bail!(
                    "SsmLayer::forward_batch: gdn decode 状態の回復 (D2H) に失敗 (li={}) → 要求を落とす",
                    self.il
                ),
            }
        }
        // task 72: qkv と z、beta と alpha をそれぞれ 1 回のバッチ GEMV にまとめる。
        // トークンごとの値は個別に呼んだ場合と同じ (行の独立)。
        let mut qkv = vec![0f32; nt * qd];
        let mut z = vec![0f32; nt * vd];
        self.qkv_gate
            .gemv_batch_split(&flat, nt, &mut [&mut qkv[..], &mut z[..]])?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_QKV,
            self.il,
            crate::model::qwen4exp::q4prof::ST_QKV,
        );
        crate::model::qwen4exp::q4prof::bump_nt(self.il, nt);

        let t = crate::model::qwen4exp::q4prof::stamp();
        let mut beta_raw = vec![0f32; nt * nh];
        let mut alpha = vec![0f32; nt * nh];
        self.beta_alpha
            .gemv_batch_split(&flat, nt, &mut [&mut beta_raw[..], &mut alpha[..]])?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_GATE,
            self.il,
            crate::model::qwen4exp::q4prof::ST_GATE,
        );

        let t = crate::model::qwen4exp::q4prof::stamp();
        gdn::gdn_dbg_set_layer(self.il as i64);
        let cw = GdnCore {
            conv1d: &self.conv1d,
            ssm_a: &self.ssm_a,
            ssm_dt: &self.ssm_dt,
            ssm_norm: &self.ssm_norm,
        };
        // サブステージのダンプがこの層を指しているときだけトークン逐次に落ちる
        // (追記順 = llama.cpp の lchk と同一に保つ)。通常はチャンク一括。
        let ys = if gdn::gdn_dbg_enabled_for(self.il as i64) {
            let mut ys = Vec::with_capacity(nt * vd);
            for i in 0..nt {
                let y = gdn::forward_token_core_gate(
                    &self.p,
                    &cw,
                    &qkv[i * qd..(i + 1) * qd],
                    &z[i * vd..(i + 1) * vd],
                    &beta_raw[i * nh..(i + 1) * nh],
                    &alpha[i * nh..(i + 1) * nh],
                    &mut self.state,
                    gdn::GateKind::Sigmoid,
                );
                if y.len() != vd {
                    bail!(
                        "SsmLayer::forward_batch: GDN の出力長 {} が value_dim={vd} と違う",
                        y.len()
                    );
                }
                ys.extend_from_slice(&y);
            }
            ys
        } else {
            gdn::forward_batch_core_gate(
                &self.p,
                &cw,
                &qkv,
                &z,
                &beta_raw,
                &alpha,
                &mut self.state,
                gdn::GateKind::Sigmoid,
            )
        };
        if ys.len() != nt * vd {
            bail!(
                "SsmLayer::forward_batch: GDN の出力長 {} が nt={nt} * value_dim={vd} と違う",
                ys.len()
            );
        }
        crate::model::qwen4exp::q4prof::mark_l(
            t,
            self.il,
            crate::model::qwen4exp::q4prof::ST_GDN,
        );

        let t = crate::model::qwen4exp::q4prof::stamp();
        let mut outs = vec![0f32; nt * e];
        crate::model::qwen4exp::gemv_batch(&self.w_out, &ys, nt, &mut outs)?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_OUT,
            self.il,
            crate::model::qwen4exp::q4prof::ST_OUT,
        );
        Ok(outs.chunks_exact(e).map(|c| c.to_vec()).collect())
    }

    /// `attn_qkv` の投影だけを行う (state は進めない)。
    ///
    /// 戻り値の並びは **[q key_dim][k key_dim][v value_dim]**。F4 の切り分けと
    /// レイアウト検証で使う。
    ///
    /// task 72: qkv は gate と連結されているので、この入口でも連結ぶん
    /// (gate) を計算する。診断・テスト用の入口で、decode の本線は
    /// [`Self::forward_token`] (1 回の GEMV で qkv と z を同時に得る)。
    pub fn project_qkv(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut buf = vec![0f32; self.qkv_gate.n()];
        self.qkv_gate.gemv_concat_into("qkv", cur, &mut buf)?;
        Ok(buf[self.qkv_gate.range(0)].to_vec())
    }

    /// task 72: qkv+gate が 1 本の GEMV に連結されているか (発動確認用)。
    pub fn qkv_gate_is_fused(&self) -> bool {
        self.qkv_gate.is_fused()
    }

    /// task 72: beta+alpha が 1 本の GEMV に連結されているか (発動確認用)。
    pub fn beta_alpha_is_fused(&self) -> bool {
        self.beta_alpha.is_fused()
    }

    /// qkv 出力の切り分け `(q_end, k_end, v_end)`。qwen4exp では `(2048, 4096, 10240)`。
    pub fn qkv_split(&self) -> (usize, usize, usize) {
        let kd = self.p.key_dim();
        let vd = self.p.value_dim();
        (kd, 2 * kd, 2 * kd + vd)
    }

    /// 1 トークンあたりの重みバイト数 (大きい 5 本の合計)。
    pub fn weight_bytes(&self) -> usize {
        self.qkv_gate.weight_bytes()
            + self.beta_alpha.weight_bytes()
            + self.w_out.weight_bytes()
            + self.conv1d.len() * 4
            + self.ssm_a.len() * 4
            + self.ssm_dt.len() * 4
            + self.ssm_norm.len() * 4
    }
}

/// `(ne0, ne1)` が期待どおりで、余次元が 1 であることを確かめる。
fn expect_2d(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != k || t.dims[1] as usize != n {
        bail!(
            "{name}: ne={:?} が期待 [{k}, {n}] と違う",
            &t.dims[..2]
        );
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

/// 量子化ブロックを保持したまま [`GgufLinear`] を作る。
///
/// mmap のスライスが取れるときはバイト列を写さない (`WeightBlocks::mapped`)。
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
            let l = GgufLinear::new(ty, k, n, WeightBlocks::mapped(map, off, len))
                .with_context(|| format!("{name}: mmap ブロックから GgufLinear を作れない"))?;
            return crate::model::linear_gguf::maybe_requant(l)
                .map(|l| l.with_fast_q8(true))
                .with_context(|| format!("{name}: Q8_0 へ作り直せない"));
        }
    }
    let bytes = f
        .tensor_bytes(name)
        .with_context(|| format!("tensor {name}"))?;
    let l = GgufLinear::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
        .with_context(|| format!("{name}: GgufLinear を作れない"))?;
    crate::model::linear_gguf::maybe_requant(l)
        .map(|l| l.with_fast_q8(true))
        .with_context(|| format!("{name}: Q8_0 へ作り直せない"))
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

// ---------------------------------------------------------------------------
// P-MOE-F3: ssm 層を `LayerAttn` として層ループに接続する
// ---------------------------------------------------------------------------

/// [`Qwen4ExpParams`] (GGUF メタデータ) から GDN のハイパーパラメータを導出する。
///
/// 対応 (参照実装 `qwen4exp.cpp` の `load_arch_hparams` と同じ読み替え):
/// - `embedding_length` → `n_embd`
/// - `ssm.group_count` → `n_k_heads`
/// - `ssm.time_step_rank` → `n_v_heads`
/// - `ssm.state_size` → `head_k_dim`
/// - `ssm.inner_size / ssm.time_step_rank` → `head_v_dim`
/// - `ssm.conv_kernel` → `d_conv`
///
/// 実モデル (2560 / 16 / 48 / 128 / 6144 / 4 / 1e-6) では
/// [`GdnParams::qwen4exp`] と同じ値になる (`tests/qwen4exp_layers.rs` で固定)。
pub fn gdn_params(p: &Qwen4ExpParams) -> Result<GdnParams> {
    if p.ssm_dt_rank == 0 {
        bail!("qwen4exp.ssm.time_step_rank が 0 (head_v_dim を決められない)");
    }
    if p.ssm_d_inner % p.ssm_dt_rank != 0 {
        bail!(
            "qwen4exp.ssm.inner_size {} が time_step_rank {} で割り切れない",
            p.ssm_d_inner,
            p.ssm_dt_rank
        );
    }
    let g = GdnParams {
        n_embd: p.n_embd,
        n_k_heads: p.ssm_n_group,
        n_v_heads: p.ssm_dt_rank,
        head_k_dim: p.ssm_d_state,
        head_v_dim: p.ssm_d_inner / p.ssm_dt_rank,
        d_conv: p.ssm_d_conv,
        eps: p.rms_eps,
    };
    // 派生値どうしの突き合わせ (メタデータの 3 つの数が同じ次元を指していること)。
    if g.key_dim() != p.ssm_key_dim() {
        bail!(
            "ssm key dim {} が state_size * group_count = {} と違う",
            g.key_dim(),
            p.ssm_key_dim()
        );
    }
    if g.value_dim() != p.ssm_value_dim() {
        bail!(
            "ssm value dim {} が inner_size = {} と違う",
            g.value_dim(),
            p.ssm_value_dim()
        );
    }
    if g.conv_dim() != p.ssm_conv_dim() {
        bail!(
            "conv dim {} が 2*key + value = {} と違う",
            g.conv_dim(),
            p.ssm_conv_dim()
        );
    }
    Ok(g)
}

/// ssm 層を [`LayerAttn`] として回す。full attention 層は `full` に委譲する
/// (F3 の既定は [`IdentityAttn`]。F4 が実装を差し替える)。
///
/// `layers[il]` が `Some` なら ssm 層 (`il % interval != interval-1`)、
/// `None` なら full attention 層。状態 (`conv` / `ssm`) は層ごとに独立で、
/// `forward` を呼ぶたびに 1 トークンぶん進む。
pub struct SsmAttn {
    layers: Vec<Option<SsmLayer>>,
    full: Box<dyn LayerAttn>,
}

impl SsmAttn {
    /// `p.layer_kind(il)` を見て ssm 層だけ [`SsmLayer`] を組む。
    /// full attention 層は [`IdentityAttn`] に委譲する。
    pub fn from_gguf(f: &GgufFile, p: &Qwen4ExpParams) -> Result<Self> {
        Self::with_full(f, p, Box::new(IdentityAttn))
    }

    /// full attention 層の実装を差し替えて組む (F4 用)。
    pub fn with_full(
        f: &GgufFile,
        p: &Qwen4ExpParams,
        full: Box<dyn LayerAttn>,
    ) -> Result<Self> {
        let g = gdn_params(p)?;
        let mut layers = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            let kind = p
                .layer_kind(il)
                .map_err(|e| anyhow::anyhow!("層 {il}: {e}"))?;
            layers.push(match kind {
                LayerKind::Ssm => Some(
                    SsmLayer::from_gguf_with(f, il, g.clone())
                        .with_context(|| format!("層 {il}: ssm 層を組めない"))?,
                ),
                LayerKind::FullAttn => None,
            });
        }
        Ok(Self { layers, full })
    }

    /// 層ごとの種別 (組み上げた結果。テスト・診断用)。
    pub fn kinds(&self) -> Vec<LayerKind> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_some() {
                    LayerKind::Ssm
                } else {
                    LayerKind::FullAttn
                }
            })
            .collect()
    }

    /// ssm 層の本数 (= [`Qwen4ExpParams::n_ssm_layers`])
    pub fn n_ssm(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }

    /// full attention 層の本数 (= [`Qwen4ExpParams::n_full_attn_layers`])
    pub fn n_full(&self) -> usize {
        self.layers.iter().filter(|l| l.is_none()).count()
    }

    /// ssm 層の状態 (テスト・診断用)。full attention 層は `None`。
    pub fn ssm_layer(&self, il: usize) -> Option<&SsmLayer> {
        self.layers.get(il).and_then(|l| l.as_ref())
    }
}

impl LayerAttn for SsmAttn {
    fn forward(
        &mut self,
        layer: usize,
        cur: &[f32],
    ) -> std::result::Result<Vec<f32>, Qwen4ExpError> {
        let n = self.layers.len();
        match self.layers.get_mut(layer) {
            None => Err(one(format!(
                "SsmAttn::forward: 層 {layer} は範囲外 (層数 {n})"
            ))),
            Some(Some(s)) => s
                .forward_token(cur)
                // anyhow の文脈を 1 行に畳む (`{:#}` は "外側: 内側" を連結する)
                .map_err(|e| Qwen4ExpError(vec![format!("層 {layer}: {e:#}")])),
            Some(None) => self.full.forward(layer, cur),
        }
    }

    /// ssm 層は投影をチャンクでまとめる。full 層は `self.full` の
    /// [`LayerAttn::forward_batch`] に委譲する (バッチ prefill で `FullAttn` が
    /// 入っているときは KV を 1 回走査するバッチ版が走る。既定の実装 =
    /// 1 トークンずつの `forward` と同じ値)。
    fn forward_batch(
        &mut self,
        layer: usize,
        curs: &[Vec<f32>],
    ) -> std::result::Result<Vec<Vec<f32>>, Qwen4ExpError> {
        let n = self.layers.len();
        match self.layers.get_mut(layer) {
            None => Err(one(format!(
                "SsmAttn::forward_batch: 層 {layer} は範囲外 (層数 {n})"
            ))),
            Some(Some(s)) => s
                .forward_batch(curs)
                .map_err(|e| Qwen4ExpError(vec![format!("層 {layer}: {e:#}")])),
            Some(None) => self.full.forward_batch(layer, curs),
        }
    }

    /// 層優先の順序でも安全。`forward` は「1 層の状態を 1 トークン進める」だけで、
    /// 層どうしは独立している。層優先にしても各層の中でトークン順は保たれるので、
    /// GDN の conv / ssm 状態の進み方は逐次版と同じになる。
    fn layer_major_safe(&self) -> bool {
        true
    }

    /// P118: ssm 層の GDN conv/ssm を 0 に戻し、full attention 側へも委譲する。
    ///
    /// 積んだ位置だけを消す (`GdnState::reset` は領域を使い回す)。重みと
    /// uid 登録は触らない (GPU 常駐の重みを再 upload させないのが目的)。
    fn reset(&mut self) {
        for l in self.layers.iter_mut().flatten() {
            // task 128: reset はホスト state を 0 に戻す。デバイス側のコピーが
            // 正本のまま残ると、次の prefill の `take` が **古い状態** をホストへ
            // 戻してしまい、系列を跨いで状態が化ける (br114 で全トークンが同じ
            // 壊れ方をした `!!!!` の原因)。必ずデバイスを無効化する。
            l.reset_state();
        }
        self.full.reset();
    }

    /// P109: full attention 側が巻き戻しに対応しているときだけ対応
    /// (QSA 有効の [`FullAttn`] では `false` になり、spec の batch 経路は逐次に落ちる)。
    fn spec_restorable(&self) -> bool {
        self.full.spec_restorable()
    }

    /// P109: GDN 状態 (conv / ssm) を層ごと clone し、full attention 側は
    /// 委譲先の [`crate::model::qwen4exp::SpecFullSnap`] に任せる。
    fn spec_snapshot(&self) -> Box<dyn std::any::Any + Send> {
        Box::new(SpecGdnSnap {
            gdn: self
                .layers
                .iter()
                .map(|l| {
                    l.as_ref().map(|s| {
                        // task 128: デバイスが正なら (直前の decode トークンを GPU が
                        // 処理した直後なら) スナップショットはデバイスの値を取る。
                        // 読み取り専用 (正本フラグを動かさない) なので、snapshot の
                        // あとも decode を続けるときデバイスは再シード不要。
                        // Lost = D2H 失敗 (GPU 不調)。エラーを記録してホストの値で
                        // 代替する (GPU が壊れたままなら次の submit が要求を落とす)。
                        #[cfg(feature = "cuda")]
                        if crate::cuda::gdn_gpu_scan_on() {
                            match crate::cuda::gdn_decode_state_read(s.nonce, s.il) {
                                crate::cuda::GdnDecodeRead::Device(conv, ssm) => {
                                    return (conv, ssm);
                                }
                                crate::cuda::GdnDecodeRead::Lost => {
                                    tracing::error!(
                                        "[gdn-gpu-scan] snapshot の状態読み取り (D2H) に失敗 (li={}) → ホスト state で代替",
                                        s.il
                                    );
                                }
                                crate::cuda::GdnDecodeRead::HostCurrent => {}
                            }
                        }
                        (s.state.conv.clone(), s.state.ssm.clone())
                    })
                })
                .collect(),
            full: self.full.spec_snapshot(),
        })
    }

    /// P109: GDN 状態をスナップショットへ戻し、full attention 側へも委譲する。
    fn spec_restore(&mut self, snap: &dyn std::any::Any) -> std::result::Result<(), Qwen4ExpError> {
        let s = snap
            .downcast_ref::<SpecGdnSnap>()
            .expect("SsmAttn::spec_restore: スナップショットの型が違う");
        if s.gdn.len() != self.layers.len() {
            return Err(one(format!(
                "SsmAttn::spec_restore: スナップショットの層数 {} が現在の層数 {} と違う",
                s.gdn.len(),
                self.layers.len()
            )));
        }
        for (l, st) in self.layers.iter_mut().zip(s.gdn.iter()) {
            if let (Some(layer), Some((conv, ssm))) = (l.as_mut(), st) {
                layer.state.conv = conv.clone();
                layer.state.ssm = ssm.clone();
                // task 128: ホスト state をスナップショットで上書きしたので、デバイス
                // 側のコピー (あれば) はもう正でない。次の decode 1 トークン目で
                // ホストから再シードする。
                #[cfg(feature = "cuda")]
                if crate::cuda::gdn_gpu_scan_on() {
                    crate::cuda::gdn_decode_state_invalidate(layer.nonce, layer.il);
                }
            }
        }
        self.full.spec_restore(&*s.full)
    }
}

// ---------------------------------------------------------------------------
// P-PF-GDN: ssm 層のバッチ経路 (`forward_batch` → `gdn::forward_batch_core_gate`)
// が 1 トークンずつの `forward_token` とビット一致することの固定。
//
// 実モデル (111 GB) は使わない。合成した F32 の投影行列で「構造は実モデルと同じ・
// 寸法だけ小さい」層を組み、同じ入力列を 2 つの層に「逐次」と「チャンク」で
// 食わせて比べる。出力だけでなく conv / ssm 状態も比べる (状態がずれると次の
// チャンクで静かに壊れる)。
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn lcg_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut rng = seed;
        move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (u32::MAX as f32) - 0.5
        }
    }

    /// F32 の重みで [`GgufLinear`] を作る (`WeightBlocks` は行優先 [n][k])。
    fn f32_linear(k: usize, n: usize, rnd: &mut impl FnMut() -> f32) -> GgufLinear {
        let data: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        GgufLinear::new(GgmlType::F32, k, n, WeightBlocks::from_vec(bytes))
            .expect("GgufLinear (F32)")
    }

    /// 構造は実モデルと同じ・寸法だけ小さい ssm 層 (nk=2 / nv=4 で GQA の
    /// タイル対応 h % nk も通る)。
    fn small_layer() -> (SsmLayer, GdnParams) {
        let p = GdnParams {
            n_embd: 16,
            n_k_heads: 2,
            n_v_heads: 4,
            head_k_dim: 8,
            head_v_dim: 8,
            d_conv: 4,
            eps: 1e-6,
        };
        let mut rnd = lcg_rng(0x0f0f_1234_5678_9abc);
        let vd = p.value_dim();
        let layer = SsmLayer {
            il: 0,
            nonce: next_device_nonce(),
            p: p.clone(),
            qkv_gate: FusedLinear::concat(vec![
                f32_linear(p.n_embd, p.qkv_dim(), &mut rnd),
                f32_linear(p.n_embd, vd, &mut rnd),
            ]),
            beta_alpha: FusedLinear::concat(vec![
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
            ]),
            w_out: f32_linear(vd, p.n_embd, &mut rnd),
            conv1d: (0..p.conv_dim() * p.d_conv).map(|_| 0.4 * rnd()).collect(),
            ssm_a: (0..p.n_v_heads).map(|_| -0.05 - 0.05 * rnd().abs()).collect(),
            ssm_dt: (0..p.n_v_heads).map(|_| 0.1 * rnd()).collect(),
            ssm_norm: (0..p.head_v_dim).map(|_| 0.5 + rnd().abs()).collect(),
            state: GdnState::new(&p),
        };
        (layer, p)
    }

    fn inputs(p: &GdnParams, nt: usize) -> Vec<Vec<f32>> {
        (0..nt)
            .map(|t| {
                (0..p.n_embd)
                    .map(|j| 0.3 * (((t * 7 + j * 3) % 5) as f32) - 0.6)
                    .collect()
            })
            .collect()
    }

    /// 2 トークン以上で `forward_batch` のチャンク経路が発動し、`forward_token` を
    /// nt 回呼ぶのと出力・状態ともビット一致する。
    #[test]
    fn ssm_batch_matches_token_by_token_bitexact() {
        for nt in [2usize, 3, 5, 9, 17] {
            let (mut seq, p) = small_layer();
            let (mut bat, _) = small_layer();
            let curs = inputs(&p, nt);

            let want: Vec<Vec<f32>> = curs
                .iter()
                .map(|c| seq.forward_token(c).expect("forward_token"))
                .collect();
            let got = bat.forward_batch(&curs).expect("forward_batch");

            assert_eq!(got.len(), nt, "nt={nt}: 出力トークン数");
            for t in 0..nt {
                assert_eq!(
                    bits(&got[t]),
                    bits(&want[t]),
                    "nt={nt}: トークン {t} の出力がビット不一致"
                );
            }
            assert_eq!(bits(&seq.state.ssm), bits(&bat.state.ssm), "nt={nt}: ssm 状態");
            assert_eq!(bits(&seq.state.conv), bits(&bat.state.conv), "nt={nt}: conv 状態");
        }
    }

    /// チャンクの切り方を変えても同じ (状態の持ち越しが正しい)。
    #[test]
    fn ssm_batch_chunk_split_matches_single_chunk() {
        let (mut one, p) = small_layer();
        let (mut split, _) = small_layer();
        let curs = inputs(&p, 11);

        let whole = one.forward_batch(&curs).expect("forward_batch (1 回)");
        let mut pieces: Vec<Vec<f32>> = Vec::new();
        for (a, b) in [(0usize, 3usize), (3, 4), (4, 11)] {
            pieces.extend(split.forward_batch(&curs[a..b]).expect("forward_batch (分割)"));
        }
        assert_eq!(pieces.len(), whole.len());
        for t in 0..whole.len() {
            assert_eq!(bits(&pieces[t]), bits(&whole[t]), "トークン {t}");
        }
        assert_eq!(bits(&one.state.ssm), bits(&split.state.ssm), "ssm 状態");
        assert_eq!(bits(&one.state.conv), bits(&split.state.conv), "conv 状態");
    }

    /// nt=1 は `forward_token` へ委譲する (バッチ経路に入らない)。
    #[test]
    fn ssm_batch_single_token_delegates() {
        let (mut a, p) = small_layer();
        let (mut b, _) = small_layer();
        let curs = inputs(&p, 1);
        let want = a.forward_token(&curs[0]).expect("forward_token");
        let got = b.forward_batch(&curs).expect("forward_batch");
        assert_eq!(got.len(), 1);
        assert_eq!(bits(&got[0]), bits(&want));
    }

    // ---- task 72: qkv+gate / beta+alpha の融合 ------------------------------

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct FusedGuard;
    impl FusedGuard {
        fn new() -> Self {
            crate::model::linear_gguf::set_fused_gemv_for_test(None);
            Self
        }
    }
    impl Drop for FusedGuard {
        fn drop(&mut self) {
            crate::model::linear_gguf::set_fused_gemv_for_test(None);
        }
    }

    /// 融合した層と分けた層が、出力も状態も**ビット一致**する。
    ///
    /// 数値不変の根拠: 行は独立で、1 行の内積は `i` 昇順の逐次加算のまま
    /// (連結しても 1 行の中の順序は変わらない)。
    #[test]
    fn ssm_fused_matches_separate_bitexact() {
        let _g = crate::model::linear_gguf::par_test_lock();
        let _p = FusedGuard::new();

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(true));
        let (mut fused, p) = small_layer();
        assert!(fused.qkv_gate_is_fused(), "qkv+gate が連結されていない");
        assert!(fused.beta_alpha_is_fused(), "beta+alpha が連結されていない");

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(false));
        let (mut sep, _) = small_layer();
        assert!(!sep.qkv_gate_is_fused(), "off なのに連結された");
        assert!(!sep.beta_alpha_is_fused(), "off なのに連結された");

        let curs = inputs(&p, 7);
        for (t, c) in curs.iter().enumerate() {
            let a = fused.forward_token(c).expect("fused forward_token");
            let b = sep.forward_token(c).expect("separate forward_token");
            assert_eq!(bits(&a), bits(&b), "トークン {t}");
        }
        assert_eq!(bits(&fused.state.ssm), bits(&sep.state.ssm), "ssm 状態");
        assert_eq!(bits(&fused.state.conv), bits(&sep.state.conv), "conv 状態");
    }

    /// バッチ経路 (`gemv_batch_split`) も融合の有無でビット一致する。
    #[test]
    fn ssm_fused_batch_matches_separate_bitexact() {
        let _g = crate::model::linear_gguf::par_test_lock();
        let _p = FusedGuard::new();

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(true));
        let (mut fused, p) = small_layer();
        assert!(fused.qkv_gate_is_fused());

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(false));
        let (mut sep, _) = small_layer();

        let curs = inputs(&p, 5);
        let a = fused.forward_batch(&curs).expect("fused forward_batch");
        let b = sep.forward_batch(&curs).expect("separate forward_batch");
        assert_eq!(a.len(), b.len());
        for t in 0..a.len() {
            assert_eq!(bits(&a[t]), bits(&b[t]), "トークン {t}");
        }
        assert_eq!(bits(&fused.state.ssm), bits(&sep.state.ssm), "ssm 状態");
        assert_eq!(bits(&fused.state.conv), bits(&sep.state.conv), "conv 状態");
    }

    /// task 87: decode (`forward_token`) が attn_kprof の 3 区間を実際に積むことの
    /// 証拠。calls がトークン数ぶん増え、line() に 3 区間とも現れる。
    /// 合成層は F32 の投影なので GPU intercept の対象外。gpu 発数は cuda ビルドで
    /// 直前の線形呼び出しの実行先を数えるだけ (ここでは GPU を回さない) ので
    /// assert しない。終わりに必ず FORCE を戻す。
    #[test]
    fn ssm_decode_fills_attn_kprof_slots() {
        crate::model::qwen4exp::attn_kprof::set_for_test(true);
        crate::model::qwen4exp::attn_kprof::reset();
        let (mut layer, _p) = small_layer();
        for _ in 0..3 {
            let cur: Vec<f32> = (0..layer.p.n_embd)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                .collect();
            let _ = layer.forward_token(&cur).expect("forward_token");
        }
        let line = crate::model::qwen4exp::attn_kprof::line().expect("強制 on なのに line() が None");
        assert!(line.contains("qkv calls=3"), "qkv が積まれていない: {line}");
        assert!(line.contains("gate calls=3"), "gate が積まれていない: {line}");
        assert!(line.contains("out calls=3"), "out が積まれていない: {line}");
        crate::model::qwen4exp::attn_kprof::reset();
        crate::model::qwen4exp::attn_kprof::set_for_test(false);
    }

    // ---- task 128: GPU scan の差し込み (CPU で確認できる分) ----------------

    /// task 128 のテストで立てた強制フラグを panic でも必ず戻す。
    #[cfg(feature = "cuda")]
    struct GdnGpuScanGuard;
    #[cfg(feature = "cuda")]
    impl GdnGpuScanGuard {
        fn new() -> Self {
            crate::cuda::gdn_gpu_scan_set_for_test(false);
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(false);
            crate::cuda::gdn_invalidate_calls_reset_for_test();
            Self
        }
    }
    #[cfg(feature = "cuda")]
    impl Drop for GdnGpuScanGuard {
        fn drop(&mut self) {
            crate::cuda::gdn_gpu_scan_set_for_test(false);
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(false);
        }
    }

    /// task 128: ゲート off (既定) では decode 窓を開けても経路が 1 バイトも
    /// 変わらない。GPU 分岐にも state 回復にも入らない (ストアは確保されない)。
    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_gpu_scan_off_window_open_bit_identical() {
        let _g = GdnGpuScanGuard::new();
        let (mut win, p) = small_layer();
        let (mut base, _) = small_layer();
        let curs = inputs(&p, 5);
        for (t, c) in curs.iter().enumerate() {
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(true);
            let a = win.forward_token(c).expect("窓あり");
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(false);
            let b = base.forward_token(c).expect("窓なし");
            assert_eq!(bits(&a), bits(&b), "トークン {t}: ゲート off で窓の有無が違う");
        }
        assert_eq!(bits(&win.state.ssm), bits(&base.state.ssm), "ssm 状態");
        assert_eq!(bits(&win.state.conv), bits(&base.state.conv), "conv 状態");
        assert_eq!(
            crate::cuda::gdn_decode_state_entry_count(),
            0,
            "ゲート off なのにデバイス状態ストアが確保された"
        );
        assert_eq!(
            crate::cuda::gdn_invalidate_calls_for_test(),
            0,
            "ゲート off なのに reset の無効化が走った"
        );
    }

    /// task 128: ゲート強制 on + CUDA 無しのコンテナでは、decode 窓の中でも CPU
    /// 経路に落ちる (fail-closed)。出力は窓なしの CPU 経路とビット一致し、
    /// デバイス状態ストアも確保されない (submit が cuda_enabled で戻る)。
    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_gpu_scan_on_no_cuda_falls_back() {
        let _g = GdnGpuScanGuard::new();
        crate::cuda::gdn_gpu_scan_set_for_test(true);
        let (mut layer, p) = small_layer();
        let (mut base, _) = small_layer();
        let cur = inputs(&p, 1).remove(0);

        crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(true);
        let got = layer.forward_token(&cur).expect("窓あり + ゲート on");
        crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(false);

        let want = base.forward_token(&cur).expect("窓なし");
        assert_eq!(bits(&got), bits(&want), "CPU フォールバックが基準と違う");
        assert_eq!(
            crate::cuda::gdn_decode_state_entry_count(),
            0,
            "CUDA 無しでデバイス状態ストアが確保された"
        );
        assert_eq!(bits(&layer.state.ssm), bits(&base.state.ssm), "ssm 状態");
        assert_eq!(bits(&layer.state.conv), bits(&base.state.conv), "conv 状態");
    }

    /// task 128: `reset_state` が (ゲート on のとき) デバイス側のコピーを無効化し、
    /// ホスト state は 0 に戻ること。ゲート off では無効化を呼ばない。
    ///
    /// 無効化を落とすと、reset 後の CPU 経路 (prefill の take) が前の系列の
    /// 状態をホストへ戻してしまう — これが br114 の `!!!!` の原因。
    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_reset_state_invalidates_device_copy() {
        let _g = GdnGpuScanGuard::new();
        let (mut layer, p) = small_layer();
        let cur = inputs(&p, 2);
        for c in &cur {
            let _ = layer.forward_token(c).expect("forward_token");
        }
        assert!(layer.state.ssm.iter().any(|&v| v != 0.0), "状態が進んでいない");

        // ゲート off: 無効化は走らない (ストアが存在しないので no-op のまま)。
        layer.reset_state();
        assert_eq!(crate::cuda::gdn_invalidate_calls_for_test(), 0);
        assert!(layer.state.conv.iter().all(|&v| v == 0.0), "conv が 0 でない");
        assert!(layer.state.ssm.iter().all(|&v| v == 0.0), "ssm が 0 でない");

        // ゲート on: reset のたびに無効化が 1 回走る。
        crate::cuda::gdn_gpu_scan_set_for_test(true);
        for c in &cur {
            let _ = layer.forward_token(c).expect("forward_token (ゲート on)");
        }
        layer.reset_state();
        assert_eq!(
            crate::cuda::gdn_invalidate_calls_for_test(),
            1,
            "ゲート on の reset で無効化が走っていない"
        );
        assert!(layer.state.ssm.iter().all(|&v| v == 0.0), "ssm が 0 でない");
    }

    /// task 128: Req 組み立てが実モデルの寸法と一致すること (qwen4exp 実形:
    /// n_embd 2560 / qkv 10240 + gate 6144 / out [6144, 2560] / nk 16 / nv 48 /
    /// dk = dv 128 / conv 4)。重みは実モデルと同じ Q8_0 (ゼロブロック) で、
    /// qkv+gate が 1 本に融合した実形を再現する。
    #[cfg(feature = "cuda")]
    #[test]
    fn gdn_decode_req_matches_real_model_dims() {
        let _g = GdnGpuScanGuard::new();
        let p = GdnParams::qwen4exp();
        let q8_zero = |k: usize, n: usize| {
            let blocks = k * n / 32;
            GgufLinear::new(
                GgmlType::Q8_0,
                k,
                n,
                WeightBlocks::from_vec(vec![0u8; blocks * 34]),
            )
            .expect("Q8_0 linear")
        };
        let mut rnd = lcg_rng(0x5150_4154_4153_4b45);
        let layer = SsmLayer {
            il: 7,
            nonce: next_device_nonce(),
            p: p.clone(),
            qkv_gate: FusedLinear::concat(vec![
                q8_zero(p.n_embd, p.qkv_dim()),
                q8_zero(p.n_embd, p.value_dim()),
            ]),
            beta_alpha: FusedLinear::concat(vec![
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
            ]),
            w_out: q8_zero(p.value_dim(), p.n_embd),
            conv1d: vec![0f32; p.conv_dim() * p.d_conv],
            ssm_a: vec![-0.05; p.n_v_heads],
            ssm_dt: vec![0.0; p.n_v_heads],
            ssm_norm: vec![1.0; p.head_v_dim],
            state: GdnState::new(&p),
        };
        assert!(
            layer.qkv_gate.is_fused(),
            "実モデルと同じ Q8_0+Q8_0 が融合していない"
        );

        let cur = vec![0.5f32; p.n_embd];
        let beta = vec![0.25f32; p.n_v_heads];
        let alpha = vec![0.75f32; p.n_v_heads];
        let qkv_units = layer.gdn_qkv_units();
        let req = layer.gdn_decode_req(&cur, &beta, &alpha, &qkv_units);

        assert!(req.nonce != 0, "nonce が未発号");
        assert_eq!(req.li, 7);
        assert_eq!(req.n_embd, 2560);
        assert_eq!(req.qkv_total, 16384, "qkv 10240 + gate 6144");
        assert_eq!(req.qkv_row_start, 0);
        assert_eq!(req.z_row_start, 10240, "z は qkv 行数のあと");
        assert_eq!(req.qkv_units.len(), 1, "実モデルは qkv+gate 連結 1 発");
        let (u, row) = &req.qkv_units[0];
        assert_eq!(*row, 0);
        assert_eq!((u.k, u.n), (2560, 16384));
        assert_eq!(u.ty, GgmlType::Q8_0);
        assert_eq!((req.out_unit.k, req.out_unit.n), (6144, 2560));
        assert_eq!(req.cur.len(), 2560);
        assert_eq!(req.beta.len(), 48);
        assert_eq!(req.alpha.len(), 48);
        assert_eq!(req.conv_host.len(), 10240 * 3, "conv [conv_dim][d_conv-1]");
        assert_eq!(req.ssm_host.len(), 48 * 128 * 128, "ssm [nv][dk][dv]");
        assert_eq!(req.wconv.len(), 10240 * 4);
        assert_eq!(req.wa.len(), 48);
        assert_eq!(req.wdt.len(), 48);
        assert_eq!(req.wnorm.len(), 128);
        assert_eq!((req.nk, req.nv), (16, 48));
        assert_eq!((req.dk, req.dv), (128, 128));
        assert_eq!((req.conv_dim, req.d_conv), (10240, 4));
        assert_eq!(req.eps, 1e-6f32);
    }
}
