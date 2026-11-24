//! task 796: Qwen3.8-Flash-Next (`qwen4exp`) の MTP (nextn) head の
//! safetensors ローダと 1 段 draft forward。
//!
//! 参照実装は llama.cpp (`/work/llama.cpp` の `src/models/qwen4exp.cpp` の
//! `graph_mtp` と `conversion/qwen4exp.py`、MIT License)。ここは Rust による
//! 独立実装で、逐語コピーは含まない。ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! ## 何をするか
//!
//! 実ファイル (`mtp.safetensors`、31 テンソル) を mmap で開き、ヘッダの dtype /
//! shape を検証して f32 へ展開し、1 トークン分の draft forward を回す:
//!
//! ```text
//! h_norm[s]  = RMSNorm(h[s], pre_fc_norm_hidden[s*e..])     s = 0..hc
//! e_norm     = RMSNorm(e, pre_fc_norm_embedding)
//! res_hc[s]  = fc_embedding @ e_norm + fc_hidden @ h_norm[s]    ([hc][n_embd])
//! (cur, inj) = hc_mix(res_hc, attn_hyper_connection.*)
//! cur        = self_attn(cur)                                  (full attention + output gate)
//! res_hc     = hc_combine(res_hc, cur, inj, ...)
//! (cur, inj) = hc_mix(res_hc, mlp_hyper_connection.*)
//! cur        = moe(cur)                                        (routed + shared expert)
//! res_hc     = hc_combine(res_hc, cur, inj, ...)
//! mixed      = hc_mix(res_hc, hyper_connection_mixer.*)        (inject なし)
//! logits     = lm_head @ mixed
//! ```
//!
//! `res_hc` (hc*n_embd) はそのまま次の draft step の `h` になる。
//!
//! ## 数値の扱い
//!
//! - 重みは BF16 (実ファイル) または F32 (テスト) を**そのまま** f32 に展開する。
//!   f64 は使わない。内積は行方向の逐次和 (加算順序は添字昇順)。
//! - hyper-connection の演算は [`crate::model::hyper`] をそのまま使う
//!   (参照実装とビット一致する基準実装)。inject だけは `hc_mix` の
//!   `w_down_inject` 連結を使わず、`block_inject_weight @ xn * (1/hc)` を
//!   別に計算する。同じ行の同じ内積なので値は一致する (テストで固定)。
//! - RMSNorm の gamma は **`+1` を畳んだ値**にする (`*_norm.weight` /
//!   `*_layernorm.weight` のみ)。GGUF 変換器 (`conversion/qwen.py` の
//!   `Qwen3NextModel.modify_tensors` と `conversion/qwen4exp.py` の
//!   indexer 例外) が同じ規則で `1 + w` を焼き込むため。`pre_fc_norm_*` は
//!   名前が `norm.weight` で終わらないので**畳まない** (参照実装と同じ)。
//! - attention は dense (全位置) で回す。QSA の indexer 選択は
//!   `indexer_top_k` 個のセルを選ぶので、`n_pos <= indexer_top_k` では
//!   全ブロックが選ばれて dense と一致する。それを超える文脈では
//!   [`MtpModel::draft`] がエラーを返す (indexer 選択は未実装)。
//!
//! ## GPU
//!
//! このタスクでは `.cu` を触れないため CUDA 経路は無い。重みはホスト f32
//! (実形で約 250 MB。expert は選択分だけ都度展開するので常駐しない)。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use crate::model::hyper::{self, HyperParams, HyperStreams};
use crate::tensor::rmsnorm_into;

// ---------------------------------------------------------------------------
// 設定
// ---------------------------------------------------------------------------

/// MTP head の形状・数値パラメータ。HF の `config.json` の `text_config` から
/// 引く (`mtp` サブオブジェクトを含む)。
#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    /// `hidden_size` (実形 2560)
    pub n_embd: usize,
    /// `hc_count` (実形 4)
    pub hc: usize,
    /// `hc_lowrank` (実形 320)
    pub low_rank: usize,
    /// `mtp.num_hidden_layers` (実形 1。実装は 1 段だけ)
    pub n_layer: usize,
    /// `num_attention_heads` (実形 24)
    pub n_head: usize,
    /// `num_key_value_heads` (実形 2)
    pub n_head_kv: usize,
    /// `head_dim` (実形 256)
    pub head_dim: usize,
    /// rope を掛ける次元数 (`head_dim * partial_rotary_factor`、実形 64)
    pub n_rot: usize,
    /// `mtp.rope_theta` (実形 1e7)
    pub rope_theta: f32,
    /// `rms_norm_eps` (実形 1e-6)
    pub rms_eps: f32,
    /// `num_experts` (実形 512)
    pub n_expert: usize,
    /// `num_experts_per_tok` (実形 10)
    pub n_expert_used: usize,
    /// `moe_intermediate_size` (実形 640)
    pub n_ff_exp: usize,
    /// `shared_expert_intermediate_size` (実形 640)
    pub n_ff_shexp: usize,
    /// `indexer_n_heads` (実形 4)
    pub indexer_n_head: usize,
    /// `indexer_head_dim` (実形 128)
    pub indexer_head_dim: usize,
    /// `indexer_budget` (実形 2048)。dense attention が一致する上限。
    pub indexer_top_k: usize,
}

impl MtpConfig {
    /// 実ファイル (`mtp.safetensors`) の形状。テストの期待値にも使う。
    pub fn qwen38_flash_next() -> Self {
        MtpConfig {
            n_embd: 2560,
            hc: 4,
            low_rank: 320,
            n_layer: 1,
            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            n_rot: 64,
            rope_theta: 1e7,
            rms_eps: 1e-6,
            n_expert: 512,
            n_expert_used: 10,
            n_ff_exp: 640,
            n_ff_shexp: 640,
            indexer_n_head: 4,
            indexer_head_dim: 128,
            indexer_top_k: 2048,
        }
    }

    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }

    fn validate(&self) -> Result<()> {
        let bad = |msg: String| -> Result<()> { bail!("MtpConfig が不正: {msg}") };
        if self.n_embd == 0 || self.hc == 0 || self.low_rank == 0 {
            return bad("n_embd / hc / low_rank は 1 以上".into());
        }
        if self.n_layer != 1 {
            return bad(format!(
                "n_layer={} は未対応 (MTP は 1 ブロックだけ。qwen4exp の graph_mtp も同じ前提)",
                self.n_layer
            ));
        }
        if self.head_dim == 0 || self.n_head == 0 || self.n_head_kv == 0 {
            return bad("head_dim / n_head / n_head_kv は 1 以上".into());
        }
        if self.n_head % self.n_head_kv != 0 {
            return bad(format!(
                "n_head={} が n_head_kv={} で割り切れない",
                self.n_head, self.n_head_kv
            ));
        }
        if self.n_rot > self.head_dim || self.n_rot % 2 != 0 {
            return bad(format!(
                "n_rot={} は head_dim={} 以下かつ偶数",
                self.n_rot, self.head_dim
            ));
        }
        if self.n_expert_used == 0 || self.n_expert_used > self.n_expert {
            return bad(format!(
                "n_expert_used={} は 1..=n_expert({})",
                self.n_expert_used, self.n_expert
            ));
        }
        if self.n_ff_exp == 0 || self.n_ff_shexp == 0 {
            return bad("n_ff_exp / n_ff_shexp は 1 以上".into());
        }
        if self.indexer_n_head == 0 || self.indexer_head_dim == 0 {
            return bad("indexer_n_head / indexer_head_dim は 1 以上".into());
        }
        Ok(())
    }

    /// HF の `config.json` (テキスト) から読む。`text_config` があればそこを見る。
    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).context("config.json が JSON として読めない")?;
        let root = v.get("text_config").filter(|x| x.is_object()).unwrap_or(&v);
        let num = |key: &str| -> Result<u64> {
            root.get(key)
                .and_then(|x| x.as_u64())
                .with_context(|| format!("config.json に {key} (u64) が無い"))
        };
        let num_f = |key: &str| -> Result<f64> {
            root.get(key)
                .and_then(|x| x.as_f64())
                .with_context(|| format!("config.json に {key} (number) が無い"))
        };
        let mtp = root.get("mtp");
        let mtp_u = |key: &str, default: Option<u64>| -> Result<u64> {
            match mtp.and_then(|m| m.get(key)).and_then(|x| x.as_u64()) {
                Some(v) => Ok(v),
                None => default
                    .with_context(|| format!("config.json の text_config.mtp に {key} が無い")),
            }
        };

        let head_dim = num("head_dim")?;
        let n_layer = match root.get("mtp_num_hidden_layers").and_then(|x| x.as_u64()) {
            Some(v) => v,
            None => mtp_u("num_hidden_layers", Some(1))?,
        };
        // mtp.layer_types は full_attention だけでなければならない (linear は未対応)。
        if let Some(types) = mtp
            .and_then(|m| m.get("layer_types"))
            .and_then(|x| x.as_array())
        {
            for (i, t) in types.iter().enumerate() {
                let name = t.as_str().unwrap_or("<non-string>");
                if name != "full_attention" {
                    bail!("MTP 層 {i} の layer_types={name:?} は未対応 (full_attention のみ)");
                }
            }
            if types.len() as u64 != n_layer {
                bail!(
                    "mtp.layer_types の要素数 {} が mtp_num_hidden_layers={n_layer} と違う",
                    types.len()
                );
            }
        }
        let n_rot = match root
            .get("rope_parameters")
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|x| x.as_f64())
        {
            Some(p) => (head_dim as f64 * p) as u64,
            None => head_dim,
        };
        let rope_theta = match mtp
            .and_then(|m| m.get("rope_theta"))
            .and_then(|x| x.as_f64())
        {
            Some(x) => x as f32,
            None => num_f("rope_theta")? as f32,
        };

        let cfg = MtpConfig {
            n_embd: num("hidden_size")? as usize,
            hc: num("hc_count")? as usize,
            low_rank: num("hc_lowrank")? as usize,
            n_layer: n_layer as usize,
            n_head: num("num_attention_heads")? as usize,
            n_head_kv: num("num_key_value_heads")? as usize,
            head_dim: head_dim as usize,
            n_rot: n_rot as usize,
            rope_theta,
            rms_eps: num_f("rms_norm_eps")? as f32,
            n_expert: num("num_experts")? as usize,
            n_expert_used: num("num_experts_per_tok")? as usize,
            n_ff_exp: num("moe_intermediate_size")? as usize,
            n_ff_shexp: num("shared_expert_intermediate_size")? as usize,
            indexer_n_head: num("indexer_n_heads")? as usize,
            indexer_head_dim: num("indexer_head_dim")? as usize,
            indexer_top_k: num("indexer_budget")? as usize,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("{} を読めない", path.display()))?;
        Self::from_json_str(&s).with_context(|| format!("{} の解釈に失敗", path.display()))
    }
}

// ---------------------------------------------------------------------------
// 期待するテンソル一覧
// ---------------------------------------------------------------------------

/// MTP ファイルに入っているべきテンソル名と shape (torch 順)。
///
/// 実ファイル (31 テンソル) と 1 件も違わないことを [`MtpModel::open`] が
/// 検証する。過不足はそのままエラーにする。
pub fn expected_tensor_shapes(cfg: &MtpConfig) -> Vec<(String, Vec<usize>)> {
    let e = cfg.n_embd;
    let hc = cfg.hc;
    let hcd = hc * e;
    let lr = cfg.low_rank;
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| out.push((name, shape));
    push("mtp.fc_embedding.weight".into(), vec![e, e]);
    push("mtp.fc_hidden.weight".into(), vec![e, e]);
    push("mtp.pre_fc_norm_embedding.weight".into(), vec![e]);
    push("mtp.pre_fc_norm_hidden.weight".into(), vec![hcd]);
    push(
        "mtp.hyper_connection_mixer.hc_norm.weight".into(),
        vec![hcd],
    );
    push(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight".into(),
        vec![lr, hcd],
    );
    push(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight".into(),
        vec![hcd, lr],
    );
    for il in 0..cfg.n_layer {
        let p = |s: &str| format!("mtp.layers.{il}.{s}");
        for m in ["attn_hyper_connection", "mlp_hyper_connection"] {
            push(p(&format!("{m}.block_inject_weight.weight")), vec![hc, hcd]);
            push(p(&format!("{m}.hc_norm.weight")), vec![hcd]);
            push(
                p(&format!("{m}.input_mix_weight_down.weight")),
                vec![lr, hcd],
            );
            push(p(&format!("{m}.input_mix_weight_up.weight")), vec![hcd, lr]);
        }
        let gu = 2 * cfg.n_ff_exp;
        push(p("mlp.experts.gate_up_proj"), vec![cfg.n_expert, gu, e]);
        push(
            p("mlp.experts.down_proj"),
            vec![cfg.n_expert, e, cfg.n_ff_exp],
        );
        push(p("mlp.gate.weight"), vec![cfg.n_expert, e]);
        push(
            p("mlp.shared_expert.gate_proj.weight"),
            vec![cfg.n_ff_shexp, e],
        );
        push(
            p("mlp.shared_expert.up_proj.weight"),
            vec![cfg.n_ff_shexp, e],
        );
        push(
            p("mlp.shared_expert.down_proj.weight"),
            vec![e, cfg.n_ff_shexp],
        );
        push(p("mlp.shared_expert_gate.weight"), vec![1, e]);
        push(
            p("self_attn.q_proj.weight"),
            vec![cfg.n_head * 2 * cfg.head_dim, e],
        );
        push(
            p("self_attn.k_proj.weight"),
            vec![cfg.n_head_kv * cfg.head_dim, e],
        );
        push(
            p("self_attn.v_proj.weight"),
            vec![cfg.n_head_kv * cfg.head_dim, e],
        );
        push(
            p("self_attn.o_proj.weight"),
            vec![e, cfg.n_head * cfg.head_dim],
        );
        push(p("self_attn.q_norm.weight"), vec![cfg.head_dim]);
        push(p("self_attn.k_norm.weight"), vec![cfg.head_dim]);
        // indexer: q は n_head 本、k は 1 本 (indexer_kv_heads=1) を 1 本の行列に詰めた形
        push(
            p("self_attn.indexer.index_qk_proj.weight"),
            vec![(cfg.indexer_n_head + 1) * cfg.indexer_head_dim, e],
        );
        push(
            p("self_attn.indexer.q_layernorm.weight"),
            vec![cfg.indexer_head_dim],
        );
        push(
            p("self_attn.indexer.k_layernorm.weight"),
            vec![cfg.indexer_head_dim],
        );
    }
    out
}

// ---------------------------------------------------------------------------
// safetensors の読み出し
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            other => bail!("未対応の dtype {other:?} (F32 / F16 / BF16 のみ)"),
        }
    }

    fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone)]
struct Meta {
    dtype: DType,
    shape: Vec<usize>,
    /// データ領域の先頭からの相対バイト位置
    off: usize,
    len: usize,
}

impl Meta {
    fn n_elem(&self) -> usize {
        self.shape.iter().product()
    }
}

/// 生バイト (`n` 要素) を f32 に展開する。`add` は全要素に足す値。
fn decode_bytes(bytes: &[u8], n: usize, dtype: DType, add: f32) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), n * dtype.size());
    let mut out = Vec::with_capacity(n);
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let b = &bytes[i * 4..i * 4 + 4];
                out.push(f32::from_le_bytes(b.try_into().unwrap()) + add);
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let b = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out.push(f32::from_bits((b as u32) << 16) + add);
            }
        }
        DType::F16 => {
            for i in 0..n {
                let b = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out.push(half::f16::from_bits(b).to_f32() + add);
            }
        }
    }
    out
}

/// mmap した safetensors 1 ファイル。ヘッダのオフセットは昇順・隙間なし・
/// 範囲内であることをロード時に検証する。
pub struct MtpFile {
    mmap: Mmap,
    /// データ領域の先頭 (8 + ヘッダ長)
    base: usize,
    tensors: BTreeMap<String, Meta>,
}

impl MtpFile {
    fn open(path: &Path) -> Result<Self> {
        let f =
            std::fs::File::open(path).with_context(|| format!("{} を開けない", path.display()))?;
        let meta = f
            .metadata()
            .with_context(|| format!("{} の stat に失敗", path.display()))?;
        if meta.len() < 8 {
            bail!(
                "{} は safetensors として短すぎる ({} バイト)",
                path.display(),
                meta.len()
            );
        }
        // SAFETY: 読み取り専用の mmap。ファイルは開いたまま保持する。
        let mmap = unsafe { Mmap::map(&f) }
            .with_context(|| format!("{} を mmap できない", path.display()))?;
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let base = 8usize
            .checked_add(hlen)
            .with_context(|| "safetensors のヘッダ長が桁あふれ")?;
        if hlen == 0 || base > mmap.len() {
            bail!(
                "{}: ヘッダ長 {hlen} がファイルサイズ {} と矛盾",
                path.display(),
                mmap.len()
            );
        }
        let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..base])
            .with_context(|| format!("{} のヘッダが JSON として読めない", path.display()))?;
        let obj = hdr
            .as_object()
            .with_context(|| format!("{} のヘッダが object でない", path.display()))?;

        let data_len = mmap.len() - base;
        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = v
                .get("dtype")
                .and_then(|x| x.as_str())
                .with_context(|| format!("{name}: dtype が無い"))?;
            let dtype = DType::parse(dtype_s).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|x| x.as_array())
                .with_context(|| format!("{name}: shape が無い"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|u| u as usize)
                        .with_context(|| format!("{name}: shape の要素が u64 でない"))
                })
                .collect::<Result<_>>()?;
            let offs = v
                .get("data_offsets")
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 2)
                .with_context(|| format!("{name}: data_offsets が [start, end] でない"))?;
            let start = offs[0]
                .as_u64()
                .with_context(|| format!("{name}: data_offsets[0] が u64 でない"))?
                as usize;
            let end = offs[1]
                .as_u64()
                .with_context(|| format!("{name}: data_offsets[1] が u64 でない"))?
                as usize;
            let n: usize = shape.iter().product();
            let want = n
                .checked_mul(dtype.size())
                .with_context(|| format!("{name}: 要素数 {n} x {:?} が桁あふれ", dtype))?;
            if end < start || end > data_len {
                bail!("{name}: data_offsets [{start}, {end}] がデータ領域 (0..{data_len}) の外");
            }
            if end - start != want {
                bail!(
                    "{name}: dtype {} shape {:?} の期待バイト数 {want} が data_offsets の幅 {} と違う",
                    dtype.name(),
                    shape,
                    end - start
                );
            }
            tensors.insert(
                name.clone(),
                Meta {
                    dtype,
                    shape,
                    off: start,
                    len: end - start,
                },
            );
        }
        // データ領域が重なっていないこと (隙間は alignment 用に許す)。
        // serde_json の Map は名前順なので、オフセット順に並べ直してから見る。
        let mut spans: Vec<(&str, usize, usize)> = tensors
            .iter()
            .map(|(n, m)| (n.as_str(), m.off, m.off + m.len))
            .collect();
        spans.sort_by_key(|&(_, s, _)| s);
        for w in spans.windows(2) {
            let (a, _, a_end) = w[0];
            let (b, b_start, _) = w[1];
            if b_start < a_end {
                bail!("{a} と {b} のデータ領域が重なっている ({a_end} > {b_start})");
            }
        }
        Ok(MtpFile {
            mmap,
            base,
            tensors,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|m| m.shape.as_slice())
    }

    fn meta(&self, name: &str) -> Result<&Meta> {
        self.tensors
            .get(name)
            .with_context(|| format!("テンソル {name} がファイルに無い"))
    }

    /// `[off, off+len)` を f32 に展開する。`add` は全要素に足す値
    /// (`+1` を畳んだ gamma 用)。
    fn decode_f32(&self, m: &Meta, add: f32) -> Vec<f32> {
        let bytes = &self.mmap[self.base + m.off..self.base + m.off + m.len];
        decode_bytes(bytes, m.n_elem(), m.dtype, add)
    }

    fn f32_tensor(&self, name: &str, add: f32) -> Result<Vec<f32>> {
        let m = self.meta(name)?;
        Ok(self.decode_f32(m, add))
    }

    /// テンソルの途中の `n` 要素 (先頭から `elem_off` 要素目から) だけ f32 に
    /// 展開する。expert の 1 本だけを読むために使う。
    fn f32_slice(&self, name: &str, elem_off: usize, n: usize, add: f32) -> Result<Vec<f32>> {
        let m = self.meta(name)?;
        let total = m.n_elem();
        if elem_off + n > total {
            bail!(
                "{name}: 要素 [{elem_off}, {}) が全 {total} 要素の外",
                elem_off + n
            );
        }
        let sz = m.dtype.size();
        let start = self.base + m.off + elem_off * sz;
        let bytes = &self.mmap[start..start + n * sz];
        Ok(decode_bytes(bytes, n, m.dtype, add))
    }
}

/// GGUF 変換器と同じ規則: 名前が `norm.weight` で終わる gamma だけ `1 + w` に
/// 畳む (`pre_fc_norm_embedding.weight` / `pre_fc_norm_hidden.weight` は
/// 末尾が違うので畳まれない)。
fn gamma_add(name: &str) -> f32 {
    if name.ends_with("norm.weight") {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// 重み
// ---------------------------------------------------------------------------

/// hyper-connection の mixer 1 本分。
pub struct MtpMixer {
    /// `[hc][n_embd]` gamma (`1+w` 済み)
    pub norm: Vec<f32>,
    /// `[low_rank][hc_dim]`
    pub down: Vec<f32>,
    /// `[hc_dim][low_rank]`
    pub up: Vec<f32>,
    /// `[hc][hc_dim]`。head の mixer は `None`。
    pub inject: Option<Vec<f32>>,
}

/// full attention 1 層分 (indexer は形状検証だけして保持する)。
pub struct MtpAttn {
    /// `[n_head * 2 * head_dim][n_embd]` (head ごとに q, gate の順)
    pub q_proj: Vec<f32>,
    /// `[n_head_kv * head_dim][n_embd]`
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// `[n_embd][n_head * head_dim]`
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub index_qk_proj: Vec<f32>,
    pub index_q_norm: Vec<f32>,
    pub index_k_norm: Vec<f32>,
}

/// MoE 1 層分。expert の重みは mmap のまま保持し、選択された expert だけ
/// 都度 f32 に展開する (実形で 5.2 GB を常駐させないため)。
pub struct MtpFfn {
    /// `[n_expert][n_embd]`
    pub router: Vec<f32>,
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
    pub shared_expert_gate: Vec<f32>,
    /// `[n_expert][2*n_ff][n_embd]` のテンソル (mmap)
    gate_up_name: String,
    /// `[n_expert][n_embd][n_ff]` のテンソル (mmap)
    down_name: String,
}

pub struct MtpLayer {
    pub attn_mixer: MtpMixer,
    pub ffn_mixer: MtpMixer,
    pub attn: MtpAttn,
    pub ffn: MtpFfn,
}

/// ロード済みの MTP head。
pub struct MtpModel {
    pub cfg: MtpConfig,
    pub layers: Vec<MtpLayer>,
    fc_embedding: Vec<f32>,
    fc_hidden: Vec<f32>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    head_mixer: MtpMixer,
    file: MtpFile,
    bytes: u64,
    path: PathBuf,
}

impl std::fmt::Debug for MtpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpModel")
            .field("path", &self.path)
            .field("cfg", &self.cfg)
            .field("tensors", &self.file.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// 既定の MTP 重みのパスを返す環境変数。
pub const MTP_PATH_ENV: &str = "DENNO_MTP_PATH";

/// `DENNO_MTP_PATH` が設定されていればそのパス。
pub fn path_from_env() -> Option<PathBuf> {
    match std::env::var_os(MTP_PATH_ENV) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

impl MtpModel {
    /// `DENNO_MTP_PATH` から開く。未設定なら `Ok(None)` (既定起動に影響しない)。
    pub fn open_from_env() -> Result<Option<Self>> {
        match path_from_env() {
            Some(p) => Ok(Some(Self::open(&p)?)),
            None => Ok(None),
        }
    }

    /// safetensors と同じディレクトリの `config.json` を使って開く。
    pub fn open(path: &Path) -> Result<Self> {
        let cfg_path = path
            .parent()
            .map(|d| d.join("config.json"))
            .unwrap_or_else(|| PathBuf::from("config.json"));
        if !cfg_path.exists() {
            bail!(
                "{} が無い。MTP の config.json は safetensors と同じディレクトリに置くか、\
                 MtpModel::open_with_config で明示する",
                cfg_path.display()
            );
        }
        let cfg = MtpConfig::from_json_file(&cfg_path)?;
        Self::open_with_config(path, cfg)
    }

    pub fn open_with_config(path: &Path, cfg: MtpConfig) -> Result<Self> {
        cfg.validate()?;
        let file = MtpFile::open(path)?;
        let bytes = std::fs::metadata(path)
            .with_context(|| format!("{} の stat に失敗", path.display()))?
            .len();

        // 1. 期待する名前・shape と過不足なく一致するか。
        let expect = expected_tensor_shapes(&cfg);
        let mut problems: Vec<String> = Vec::new();
        for (name, shape) in &expect {
            match file.shape(name) {
                Some(got) if got == shape.as_slice() => {}
                Some(got) => {
                    problems.push(format!("{name}: shape {got:?} が期待 {shape:?} と違う"))
                }
                None => problems.push(format!("{name}: ファイルに無い")),
            }
        }
        let expect_names: BTreeMap<&str, ()> =
            expect.iter().map(|(n, _)| (n.as_str(), ())).collect();
        for name in file.names() {
            if !expect_names.contains_key(name) {
                problems.push(format!("{name}: 想定外のテンソル"));
            }
        }
        if !problems.is_empty() {
            bail!(
                "{}: MTP のテンソル検証に {} 件失敗\n  - {}",
                path.display(),
                problems.len(),
                problems.join("\n  - ")
            );
        }

        // 2. 非 expert の重みを f32 に展開する。
        let t = |name: &str| -> Result<Vec<f32>> {
            file.f32_tensor(name, gamma_add(name))
                .with_context(|| format!("{name} の展開"))
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("mtp.layers.{il}.{s}");
            let mixer = |prefix: &str| -> Result<MtpMixer> {
                Ok(MtpMixer {
                    norm: t(&p(&format!("{prefix}.hc_norm.weight")))?,
                    down: t(&p(&format!("{prefix}.input_mix_weight_down.weight")))?,
                    up: t(&p(&format!("{prefix}.input_mix_weight_up.weight")))?,
                    inject: Some(t(&p(&format!("{prefix}.block_inject_weight.weight")))?),
                })
            };
            let gate_up_name = p("mlp.experts.gate_up_proj");
            let down_name = p("mlp.experts.down_proj");
            layers.push(MtpLayer {
                attn_mixer: mixer("attn_hyper_connection")?,
                ffn_mixer: mixer("mlp_hyper_connection")?,
                attn: MtpAttn {
                    q_proj: t(&p("self_attn.q_proj.weight"))?,
                    k_proj: t(&p("self_attn.k_proj.weight"))?,
                    v_proj: t(&p("self_attn.v_proj.weight"))?,
                    o_proj: t(&p("self_attn.o_proj.weight"))?,
                    q_norm: t(&p("self_attn.q_norm.weight"))?,
                    k_norm: t(&p("self_attn.k_norm.weight"))?,
                    index_qk_proj: t(&p("self_attn.indexer.index_qk_proj.weight"))?,
                    index_q_norm: t(&p("self_attn.indexer.q_layernorm.weight"))?,
                    index_k_norm: t(&p("self_attn.indexer.k_layernorm.weight"))?,
                },
                ffn: MtpFfn {
                    router: t(&p("mlp.gate.weight"))?,
                    shared_gate: t(&p("mlp.shared_expert.gate_proj.weight"))?,
                    shared_up: t(&p("mlp.shared_expert.up_proj.weight"))?,
                    shared_down: t(&p("mlp.shared_expert.down_proj.weight"))?,
                    shared_expert_gate: t(&p("mlp.shared_expert_gate.weight"))?,
                    gate_up_name,
                    down_name,
                },
            });
        }
        let head_mixer = MtpMixer {
            norm: t("mtp.hyper_connection_mixer.hc_norm.weight")?,
            down: t("mtp.hyper_connection_mixer.input_mix_weight_down.weight")?,
            up: t("mtp.hyper_connection_mixer.input_mix_weight_up.weight")?,
            inject: None,
        };
        let model = MtpModel {
            fc_embedding: t("mtp.fc_embedding.weight")?,
            fc_hidden: t("mtp.fc_hidden.weight")?,
            enorm: t("mtp.pre_fc_norm_embedding.weight")?,
            hnorm: t("mtp.pre_fc_norm_hidden.weight")?,
            head_mixer,
            cfg,
            layers,
            file,
            bytes,
            path: path.to_path_buf(),
        };
        Ok(model)
    }

    /// 開いたファイルのパス。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// safetensors のバイト数。
    pub fn file_bytes(&self) -> u64 {
        self.bytes
    }

    /// 検証済みのテンソル数 (実ファイルは 31)。
    pub fn tensor_count(&self) -> usize {
        self.file.len()
    }

    /// ホストに常駐している f32 (expert を除く) の概算バイト数。
    pub fn host_f32_bytes(&self) -> u64 {
        use std::cell::Cell;
        let n = Cell::new(0usize);
        let add = |v: usize| n.set(n.get() + v);
        add(self.fc_embedding.len() + self.fc_hidden.len() + self.enorm.len() + self.hnorm.len());
        let add_mixer = |m: &MtpMixer| {
            add(m.norm.len() + m.down.len() + m.up.len());
            if let Some(i) = &m.inject {
                add(i.len());
            }
        };
        add_mixer(&self.head_mixer);
        for l in &self.layers {
            add_mixer(&l.attn_mixer);
            add_mixer(&l.ffn_mixer);
            add(l.attn.q_proj.len() + l.attn.k_proj.len() + l.attn.v_proj.len());
            add(l.attn.o_proj.len() + l.attn.q_norm.len() + l.attn.k_norm.len());
            add(l.attn.index_qk_proj.len() + l.attn.index_q_norm.len() + l.attn.index_k_norm.len());
            add(l.ffn.router.len() + l.ffn.shared_gate.len() + l.ffn.shared_up.len());
            add(l.ffn.shared_down.len() + l.ffn.shared_expert_gate.len());
        }
        n.get() as u64 * 4
    }

    fn hp(&self) -> HyperParams {
        HyperParams {
            hc: self.cfg.hc,
            n_embd: self.cfg.n_embd,
            low_rank: self.cfg.low_rank,
            eps: self.cfg.rms_eps,
        }
    }

    /// 入力段 (`pre_fc_norm_*` + `fc_*`) だけを回して `res_hc` を作る。
    /// テストと中間値の確認用に公開する。
    pub fn input_projection(&self, h: &[f32], e_embd: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let hc = cfg.hc;
        if h.len() != hc * e {
            bail!("h の長さ {} が hc*n_embd = {} と違う", h.len(), hc * e);
        }
        if e_embd.len() != e {
            bail!("埋め込みの長さ {} が n_embd = {} と違う", e_embd.len(), e);
        }
        let mut en = vec![0.0f32; e];
        rmsnorm_into(e_embd, &self.enorm, e, cfg.rms_eps, &mut en);
        let mut hn = vec![0.0f32; hc * e];
        for s in 0..hc {
            rmsnorm_into(
                &h[s * e..(s + 1) * e],
                &self.hnorm[s * e..(s + 1) * e],
                e,
                cfg.rms_eps,
                &mut hn[s * e..(s + 1) * e],
            );
        }
        let mut out = vec![0.0f32; hc * e];
        for s in 0..hc {
            for r in 0..e {
                let row_e = &self.fc_embedding[r * e..(r + 1) * e];
                let row_h = &self.fc_hidden[r * e..(r + 1) * e];
                let a = dot(row_e, &en);
                let b = dot(row_h, &hn[s * e..(s + 1) * e]);
                out[s * e + r] = a + b;
            }
        }
        Ok(out)
    }

    /// 1 段の draft forward。
    ///
    /// - `h`: trunk の hc-wide 残差 `[hc*n_embd]`
    /// - `e_embd`: 次トークンの埋め込み `[n_embd]`
    /// - `kv`: この MTP 層の KV (前の draft step まで)。呼ぶと 1 位置伸びる。
    /// - `lm_head`: `[n_embd] -> logits`。trunk の lm_head をそのまま渡す。
    pub fn draft(
        &self,
        h: &[f32],
        e_embd: &[f32],
        kv: &mut MtpKvCache,
        lm_head: &mut dyn FnMut(&[f32]) -> Result<Vec<f32>>,
    ) -> Result<MtpDraft> {
        let cfg = &self.cfg;
        let hp = self.hp();
        let layer = self
            .layers
            .first()
            .with_context(|| "MTP 層がロードされていない")?;
        let pos = kv.len();
        if pos >= cfg.indexer_top_k {
            bail!(
                "位置 {pos} が indexer_budget {} 以上。QSA の indexer 選択は未実装なので \
                 dense attention では一致しない",
                cfg.indexer_top_k
            );
        }

        let mut res = self.input_projection(h, e_embd)?;
        let mut streams = HyperStreams {
            data: res.clone(),
            hc: cfg.hc,
            n_embd: cfg.n_embd,
        };

        // --- attention ブロック ---
        let (mixed, inject) = mix_with_inject(&streams, &layer.attn_mixer, &hp);
        let attn_out = self.attention(&layer.attn, &mixed, kv)?;
        hyper::hc_combine_inplace(&mut streams, &attn_out, &inject, &hp);

        // --- MoE ブロック ---
        let (mixed, inject) = mix_with_inject(&streams, &layer.ffn_mixer, &hp);
        let ffn_out = self.moe(&layer.ffn, &mixed)?;
        hyper::hc_combine_inplace(&mut streams, &ffn_out, &inject, &hp);

        // --- head ---
        res.copy_from_slice(&streams.data);
        let mixed_head = hyper::hc_mix(
            &streams,
            &self.head_mixer.norm,
            &self.head_mixer.down,
            &self.head_mixer.up,
            None,
            &hp,
        )
        .0;
        let logits = lm_head(&mixed_head)?;
        Ok(MtpDraft {
            logits,
            hidden_next: res,
            hidden_head: mixed_head,
            pos,
        })
    }

    fn attention(&self, a: &MtpAttn, x: &[f32], kv: &mut MtpKvCache) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let dh = cfg.head_dim;
        let nh = cfg.n_head;
        let nkv = cfg.n_head_kv;
        let pos = kv.len();
        if kv.n_head_kv != nkv || kv.head_dim != dh {
            bail!(
                "KV の形状 ({}, {}) が設定 ({nkv}, {dh}) と違う",
                kv.n_head_kv,
                kv.head_dim
            );
        }

        let qg = matvec(&a.q_proj, x, nh * 2 * dh, e);
        let mut q = vec![0.0f32; nh * dh];
        let mut gate = vec![0.0f32; nh * dh];
        for h in 0..nh {
            let src = &qg[h * 2 * dh..(h + 1) * 2 * dh];
            q[h * dh..(h + 1) * dh].copy_from_slice(&src[..dh]);
            gate[h * dh..(h + 1) * dh].copy_from_slice(&src[dh..]);
        }
        let mut k = matvec(&a.k_proj, x, nkv * dh, e);
        let v = matvec(&a.v_proj, x, nkv * dh, e);

        // q/k は head ごとに RMSNorm してから rope (gate は正規化しない)
        for h in 0..nh {
            let mut row = vec![0.0f32; dh];
            rmsnorm_into(
                &q[h * dh..(h + 1) * dh],
                &a.q_norm,
                dh,
                cfg.rms_eps,
                &mut row,
            );
            q[h * dh..(h + 1) * dh].copy_from_slice(&row);
        }
        for h in 0..nkv {
            let mut row = vec![0.0f32; dh];
            rmsnorm_into(
                &k[h * dh..(h + 1) * dh],
                &a.k_norm,
                dh,
                cfg.rms_eps,
                &mut row,
            );
            k[h * dh..(h + 1) * dh].copy_from_slice(&row);
        }
        crate::model::qwen4exp_attn::rope_neox_partial(
            &mut q,
            nh,
            dh,
            cfg.n_rot,
            pos,
            cfg.rope_theta,
        );
        crate::model::qwen4exp_attn::rope_neox_partial(
            &mut k,
            nkv,
            dh,
            cfg.n_rot,
            pos,
            cfg.rope_theta,
        );
        kv.push(&k, &v);

        // dense GQA (q head h は kv head h / (nh/nkv) を見る)
        let group = nh / nkv;
        let scale = 1.0f32 / (dh as f32).sqrt();
        let n_pos = kv.len();
        let mut ctx = vec![0.0f32; nh * dh];
        let mut scores = vec![0.0f32; n_pos];
        for h in 0..nh {
            let kvh = h / group;
            let qh = &q[h * dh..(h + 1) * dh];
            for j in 0..n_pos {
                let kj = &kv.k[j * nkv * dh + kvh * dh..j * nkv * dh + (kvh + 1) * dh];
                scores[j] = dot(qh, kj) * scale;
            }
            softmax_inplace(&mut scores);
            let out = &mut ctx[h * dh..(h + 1) * dh];
            out.fill(0.0);
            for j in 0..n_pos {
                let p = scores[j];
                if p == 0.0 {
                    continue;
                }
                let vj = &kv.v[j * nkv * dh + kvh * dh..j * nkv * dh + (kvh + 1) * dh];
                for d in 0..dh {
                    out[d] += p * vj[d];
                }
            }
            // output gate
            for d in 0..dh {
                out[d] *= hyper::sigmoid(gate[h * dh + d]);
            }
        }
        Ok(matvec(&a.o_proj, &ctx, e, nh * dh))
    }

    fn moe(&self, f: &MtpFfn, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;
        let ne = cfg.n_expert;

        // ルータ: softmax (全 expert) → top-k → 選択分の和で再正規化
        let mut probs = matvec(&f.router, x, ne, e);
        softmax_inplace(&mut probs);
        let sel = top_k(&probs, cfg.n_expert_used);
        let sum: f32 = sel.iter().map(|&i| probs[i]).sum();
        let sum = sum.max(6.103515625e-5);

        let gu_stride = 2 * n_ff * e;
        let dn_stride = e * n_ff;

        let mut out = vec![0.0f32; e];
        let mut act = vec![0.0f32; n_ff];
        for &ex in &sel {
            let w = probs[ex] / sum;
            if w == 0.0 {
                continue;
            }
            // 選択された expert の 2 枚だけを mmap から展開する (実形で 1 本 20 MB)
            let gate_up = self
                .file
                .f32_slice(&f.gate_up_name, ex * gu_stride, gu_stride, 0.0)?;
            let down = self
                .file
                .f32_slice(&f.down_name, ex * dn_stride, dn_stride, 0.0)?;
            for r in 0..n_ff {
                let g = dot(&gate_up[r * e..(r + 1) * e], x);
                let u = dot(&gate_up[(n_ff + r) * e..(n_ff + r + 1) * e], x);
                act[r] = hyper::silu(g) * u;
            }
            for r in 0..e {
                let row = &down[r * n_ff..(r + 1) * n_ff];
                out[r] += w * dot(row, &act);
            }
        }

        // shared expert (sigmoid gate 付き) を足す
        let sg = matvec(&f.shared_gate, x, n_ff, e);
        let su = matvec(&f.shared_up, x, n_ff, e);
        let mut sact = vec![0.0f32; n_ff];
        for r in 0..n_ff {
            sact[r] = hyper::silu(sg[r]) * su[r];
        }
        let sdown = matvec(&f.shared_down, &sact, e, n_ff);
        let g = hyper::sigmoid(dot(&f.shared_expert_gate, x));
        for r in 0..e {
            out[r] += g * sdown[r];
        }
        Ok(out)
    }

    /// テンソル全体を f32 で返す (テストの参照実装用。本番経路は expert を
    /// 選択分だけ [`MtpFile::f32_slice`] で読む)。
    #[cfg(test)]
    fn tensor_f32(&self, name: &str, add: f32) -> Result<Vec<f32>> {
        self.file
            .f32_tensor(name, add)
            .with_context(|| format!("{name} の展開"))
    }
}

/// 1 段 draft の結果。
#[derive(Debug)]
pub struct MtpDraft {
    /// trunk の lm_head が出した logits (語彙長は lm_head 次第)
    pub logits: Vec<f32>,
    /// 次の draft step に渡す `h` (`[hc*n_embd]`)
    pub hidden_next: Vec<f32>,
    /// head の mixer 出力 (`[n_embd]`)。lm_head の入力。
    pub hidden_head: Vec<f32>,
    /// この step が書いた位置
    pub pos: usize,
}

impl MtpDraft {
    /// 最大 logit の添字 (同値は小さい添字)。
    pub fn argmax(&self) -> u32 {
        let mut best = 0usize;
        for (i, &v) in self.logits.iter().enumerate() {
            if v > self.logits[best] {
                best = i;
            }
        }
        best as u32
    }
}

// ---------------------------------------------------------------------------
// KV キャッシュ
// ---------------------------------------------------------------------------

/// MTP 層 1 本ぶんの KV。`k` / `v` は `[pos][n_kv_heads * head_dim]`。
pub struct MtpKvCache {
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl MtpKvCache {
    pub fn new(n_head_kv: usize, head_dim: usize) -> Self {
        MtpKvCache {
            n_head_kv,
            head_dim,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.k.len() / (self.n_head_kv * self.head_dim)
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.n_head_kv * self.head_dim);
        debug_assert_eq!(v.len(), k.len());
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }
}

// ---------------------------------------------------------------------------
// 小さなヘルパ
// ---------------------------------------------------------------------------

/// 行優先 `[rows][cols]` の行列と `[cols]` の GEMV。加算は添字昇順。
pub fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        out.push(dot(&w[r * cols..(r + 1) * cols], x));
    }
    out
}

/// 添字昇順の逐次内積 (f32)。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn softmax_inplace(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// 確率の降順 top-k (同値は添字の小さい方)。昇順の添字で返す。
pub fn top_k(probs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    idx.truncate(k.min(probs.len()));
    idx.sort_unstable();
    idx
}

/// `hc_mix` と同じ mixed と、inject (`block_inject_weight @ xn * (1/hc)`) を返す。
///
/// `hyper::hc_mix` は inject を `w_down` の後ろの行として扱うが、実ファイルは
/// 別テンソル (`[hc][hc_dim]`) なので、同じ xn に対する同じ内積を別に回す。
/// 値は連結版とビット一致する (テスト `inject_matches_hyper_concat` で固定)。
fn mix_with_inject(x: &HyperStreams, m: &MtpMixer, hp: &HyperParams) -> (Vec<f32>, Vec<f32>) {
    let (mixed, _) = hyper::hc_mix(x, &m.norm, &m.down, &m.up, None, hp);
    let inject = match &m.inject {
        Some(w) => {
            let xn = hyper::grouped_rms_norm(x, &m.norm, hp);
            let hcd = hp.hc * hp.n_embd;
            let mut inj = Vec::with_capacity(hp.hc);
            for r in 0..hp.hc {
                inj.push(dot(&w[r * hcd..(r + 1) * hcd], &xn));
            }
            hyper::hc_inject_apply(&mut inj, hp);
            inj
        }
        None => vec![0.0f32; hp.hc],
    };
    (mixed, inject)
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
