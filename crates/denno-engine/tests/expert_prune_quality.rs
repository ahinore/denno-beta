//! task 120: expert 剪定ゲート (`DENNO_EXPERT_TOPK` / `DENNO_EXPERT_MIN_W`) の
//! 品質 harness — CPU で実 GGUF を読んで測る (**測定のみ**。ゲート自体は既定 off)。
//!
//! 1 プロセスで:
//!   a. 各プロンプトを「ゲート全部 off (= top-10 フル)」で貪欲生成し、基準トークン列
//!      と各位置の基準 logits を取る。
//!   b. そのトークン列を設定 1 つぶん teacher forcing する (prompt を prefill して
//!      基準トークンを順に `step()` で食う)。各位置で
//!      - top-1 一致率 (設定の argmax が基準トークンと同じか)
//!      - KL(基準 ‖ 設定) の平均 (次トークン分布)
//!      - 層呼び出しあたりの平均実行 expert 数 (剪定後、W_STATS 経路で数える)
//!      - 最初の不一致位置 (1 始まり。無ければ `-`)
//!      を出す。
//!   c. 最後に貼り付け可能な 1 つの表を出す。
//!
//! full (= 両ゲート off) の設定も必ず測り、**自分自身との一致率 1.000 / KL 0** を
//! 主張する (harness の簿記検証。ずれたら harness 側の欠陥)。
//!
//! モード (環境変数 `DENNO_P120_GGUF` の有無で切替):
//! - FULL: `DENNO_P120_GGUF` で指定した実 GGUF (単一ファイルか分割ディレクトリ) を
//!   CPU で読んで全設定を測る。prompt はトークナイザの chat 展開
//!   (`encode_messages`、serve の既定と同じ形) を通す。
//! - SELFTEST (env 無し): 組み込みの小さな合成 qwen4exp GGUF
//!   (`tests/qwen4exp_decode.rs` と同じ組み方) で harness 経路だけを確認する。
//!   full との一致率 1.000 / KL 0 に加え、TOPK=1 で実行 expert が減ることも見る。
//!   実モデルの品質数字はここからは出ない。
//!
//! 実行 (FULL、GPU ホストでメインセッションが実行する。CPU で動くので CUDA は不要):
//!   DENNO_P120_GGUF=/models/qwen38-swift15-abl/IQ3_S \
//!   cargo test -p denno-engine --release --test expert_prune_quality -- --ignored --nocapture
//!
//! 実行 (SELFTEST、実 GGUF が無い環境):
//!   cargo test -p denno-engine --release --test expert_prune_quality -- --ignored --nocapture
//!
//! 環境変数:
//!   DENNO_P120_GGUF       対象 GGUF (無ければ SELFTEST)
//!   DENNO_P120_N_TOKENS   基準生成トークン数 (既定 32)
//!   DENNO_P120_OUT        結果表の追加出力先 (既定は標準出力のみ)
//!
//! 注意: 剪定は decode (m=1) 経路だけに効く。teacher forcing の prefill は
//! バッチ経路 (ゲート未対応) を通るので、基準と設定の両方でフル top-k になり、
//! 差は decode の expert 選択だけに現れる。これは「decode の剪定が品質にどう
//! 効くか」を単独で測る形になる。

use std::path::{Path, PathBuf};

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::{
    set_expert_prune_measurement, take_expert_w_stats, ExpertPruneMeasurement,
    ExpertWStatsSnapshot, Qwen4Exp, Qwen4ExpParams,
};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{argmax, DecoderConfig, Qwen4ExpDecoder};
use denno_engine::model::qwen4exp_ssm::SsmAttn;
use denno_engine::ple::PLE_TABLE_NAME;
use denno_engine::tok_gguf::GgufTokenizer;

/// プロンプト (FULL モード。実トラフィックに近い 5 種)。
const PROMPTS: [&str; 5] = [
    // 日本語フィクション (SF 小説の書き出し依頼)
    "小説書いてみてください。ジャンルはSFでお願いします",
    // Rust コード
    "RustのArc<Mutex<T>>とArc<RwLock<T>>はどんなときに使い分けるべきですか?",
    // Python コード
    "PythonでCSVを読み込んで列ごとの平均を出すスクリプトを書いてください",
    // チャット (会話を書いて系)
    "喫茶店での二人の会話をセリフだけで書いてください。テーマは引っ越しです",
    // 事実 QA
    "日本で一番標高の高い山は何ですか?",
];

/// SELFTEST モードの生トークン列 (合成モデルは語彙 8・PLE n-gram 長 3)。
const SELFTEST_PROMPTS: [&[u32]; 2] = [&[1, 2, 3], &[0, 4, 1]];

/// 剪定なしの基準設定。
fn full_config() -> (String, ExpertPruneMeasurement) {
    (
        "full".to_string(),
        ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: true,
        },
    )
}

/// 測定する設定のリスト (full を含む)。
///
/// FULL モード: TOPK = 9,8,7,6,5 と MIN_W = 0.01,0.02,0.05,0.1。
/// SELFTEST モード (合成モデルは top-2): TOPK=1 だけ (9..5 は k 超過で no-op)。
fn config_list(full_mode: bool) -> Vec<(String, ExpertPruneMeasurement)> {
    let mk = |topk: Option<usize>, min_w: Option<f32>| ExpertPruneMeasurement {
        topk,
        min_w,
        w_stats: true,
    };
    let mut v = vec![full_config()];
    if full_mode {
        for n in [9usize, 8, 7, 6, 5] {
            v.push((format!("TOPK={n}"), mk(Some(n), None)));
        }
    } else {
        v.push(("TOPK=1".to_string(), mk(Some(1), None)));
    }
    for x in [0.01f32, 0.02, 0.05, 0.1] {
        v.push((format!("MIN_W={x}"), mk(None, Some(x))));
    }
    v
}

/// 1 設定 × 1 プロンプトぶんの測定結果。
#[derive(Clone, Copy)]
struct RunResult {
    agree: f64,
    mean_kl: f64,
    experts_per_call: f64,
    calls: u64,
    first_diff: Option<usize>,
}

// ---------------------------------------------------------------------------
// モデルの組み方 (FULL: 実 GGUF / SELFTEST: 組み込み合成 GGUF)
// ---------------------------------------------------------------------------

fn ggufs_in(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    v.sort();
    anyhow::ensure!(!v.is_empty(), "{} に .gguf が無い", dir.display());
    Ok(v)
}

struct Target {
    f: GgufFile,
    params: Qwen4ExpParams,
    tok: Option<GgufTokenizer>,
    /// 合成モデルの一時ディレクトリ (SELFTEST)。mmap を生かすために Target と
    /// 同じ寿命で持つ。
    _dir: Option<TempDir>,
}

fn load_target() -> anyhow::Result<Target> {
    match std::env::var("DENNO_P120_GGUF") {
        Ok(s) if !s.trim().is_empty() => {
            let p = PathBuf::from(s.trim());
            let parts = if p.is_dir() { ggufs_in(&p)? } else { vec![p] };
            let f = if parts.len() == 1 {
                GgufFile::open(&parts[0])?
            } else {
                GgufFile::open_split(&parts)?
            };
            let tok = GgufTokenizer::from_gguf(&f)?;
            let params = Qwen4ExpParams::from_meta(&f)?;
            Ok(Target {
                f,
                params,
                tok: Some(tok),
                _dir: None,
            })
        }
        _ => {
            let dir = TempDir::new("p120_synth");
            let path = dir.path("model.gguf");
            write_gguf(&path, &synth_kvs(DEFAULT_LAYERS), &synth_tensors(DEFAULT_LAYERS));
            let f = GgufFile::open(&path)?;
            let params = validate_synth(&f)?;
            Ok(Target {
                f,
                params,
                tok: None,
                _dir: Some(dir),
            })
        }
    }
}

fn validate_synth(f: &GgufFile) -> anyhow::Result<Qwen4ExpParams> {
    denno_engine::model::qwen4exp::validate(f)
        .map_err(|e| anyhow::anyhow!("合成 GGUF の validate が落ちた: {e}"))
}

fn build_decoder(t: &Target, max_ctx: usize) -> anyhow::Result<Qwen4ExpDecoder> {
    let full = FullAttn::from_gguf(&t.f, &t.params, max_ctx)?;
    let attn = SsmAttn::with_full(&t.f, &t.params, Box::new(full))?;
    let m = Qwen4Exp::with_attn(&t.f, Box::new(attn))?;
    Ok(Qwen4ExpDecoder::new(m, DecoderConfig::new(max_ctx)))
}

// ---------------------------------------------------------------------------
// 測定本体
// ---------------------------------------------------------------------------

/// 基準生成: ゲート全部 off で貪欲生成し、トークン列と各位置の基準 logits を返す。
/// `ref_logits[i]` は `ref_toks[i]` を予測した logits。
fn generate_reference(
    t: &Target,
    prompt: &[u32],
    n: usize,
    eos: Option<u32>,
) -> anyhow::Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let max_ctx = prompt.len() + n + 64;
    let mut dec = build_decoder(t, max_ctx)?;
    let mut logits = dec.prefill_batched(prompt, true)?;
    let mut toks = Vec::with_capacity(n);
    let mut ref_logits = Vec::with_capacity(n);
    while toks.len() < n {
        let id = argmax(&logits)? as u32;
        if Some(id) == eos {
            break;
        }
        toks.push(id);
        ref_logits.push(std::mem::take(&mut logits));
        logits = dec.step(id)?;
    }
    anyhow::ensure!(!toks.is_empty(), "基準生成が空 (最初の位置で eos)");
    Ok((toks, ref_logits))
}

/// log-sum-exp の 2 乗和を避ける普通の実装: KL(P‖Q) = Σ exp(lp_i) (lp_i - lq_i)。
/// p と q が完全一致するときは各項が厳密に 0 になる (ln(sp) が共通で打ち消す)。
fn kl_ref_cfg(p: &[f32], q: &[f32]) -> f64 {
    let pm = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let qm = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut sp, mut sq) = (0.0f64, 0.0f64);
    for i in 0..p.len() {
        sp += ((p[i] - pm) as f64).exp();
        sq += ((q[i] - qm) as f64).exp();
    }
    let ln_sp = sp.ln();
    let ln_sq = sq.ln();
    let mut kl = 0.0f64;
    for i in 0..p.len() {
        let lp = (p[i] - pm) as f64 - ln_sp;
        let w = ((p[i] - pm) as f64).exp();
        let lq = (q[i] - qm) as f64 - ln_sq;
        kl += w * (lp - lq);
    }
    kl
}

/// 1 設定ぶんの teacher forcing 測定。
fn run_config(
    dec: &mut Qwen4ExpDecoder,
    prompt: &[u32],
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    cfg: ExpertPruneMeasurement,
) -> anyhow::Result<RunResult> {
    set_expert_prune_measurement(Some(cfg));
    let _ = take_expert_w_stats(); // 蓄積を消す
    // 状態を消して基準生成と同じ出発点に戻す (P118)。reset 無しだと前の設定の
    // KV / GDN / PLE 状態が次の設定に漏れて比較にならない。
    dec.reset();
    let mut logits = dec
        .prefill_batched(prompt, true)
        .map_err(|e| anyhow::anyhow!("prefill: {e}"))?;
    let mut agree_n = 0usize;
    let mut kl_sum = 0.0f64;
    let mut first_diff: Option<usize> = None;
    for (i, &t) in ref_toks.iter().enumerate() {
        let a = argmax(&logits)? as u32;
        if a == t {
            agree_n += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i + 1);
        }
        kl_sum += kl_ref_cfg(&ref_logits[i], &logits);
        logits = dec.step(t).map_err(|e| anyhow::anyhow!("step[{i}]: {e}"))?;
    }
    let snap: ExpertWStatsSnapshot =
        take_expert_w_stats().ok_or_else(|| anyhow::anyhow!("W_STATS が空 (収集が死んでいる)"))?;
    set_expert_prune_measurement(None);
    let n = ref_toks.len().max(1) as f64;
    Ok(RunResult {
        agree: agree_n as f64 / n,
        mean_kl: kl_sum / n,
        experts_per_call: snap.mean_experts_per_call(),
        calls: snap.calls(),
        first_diff,
    })
}

fn first_diff_str(d: Option<usize>) -> String {
    match d {
        Some(p) => p.to_string(),
        None => "-".to_string(),
    }
}

// ---------------------------------------------------------------------------
// 組み込み合成 GGUF (tests/qwen4exp_decode.rs と同じ組み方。SELFTEST 専用)
// ---------------------------------------------------------------------------

const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_F32: u32 = 6;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;
const GT_U64: u32 = 10;
const TY_F32: u32 = 0;
const TY_Q5_1: u32 = 7;
const TY_Q8_0: u32 = 8;
const TY_Q4_K: u32 = 12;
const TY_IQ4_NL: u32 = 20;
const TY_BF16: u32 = 30;

enum Kv {
    U32(u32),
    F32(f32),
    Str(&'static str),
    ArrU32(Vec<u32>),
    ArrU64(Vec<u64>),
    ArrStr(Vec<String>),
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::F32(v) => {
                b.extend_from_slice(&GT_F32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::Str(s) => {
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                push_str(b, s);
            }
            Kv::ArrU32(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Kv::ArrU64(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_U64.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
            Kv::ArrStr(v) => {
                b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for s in v {
                    push_str(b, s);
                }
            }
        }
    }
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    ty: u32,
}

fn ts(name: impl Into<String>, dims: Vec<u64>, ty: u32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dims,
        ty,
    }
}

fn blck_size(ty: u32) -> u64 {
    match ty {
        TY_F32 | TY_BF16 => 1,
        TY_Q5_1 | TY_Q8_0 | TY_IQ4_NL => 32,
        TY_Q4_K => 256,
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
}

fn type_size(ty: u32) -> u64 {
    match ty {
        TY_F32 => 4,
        TY_BF16 => 2,
        TY_Q5_1 => 24,
        TY_Q8_0 => 34,
        TY_IQ4_NL => 18,
        TY_Q4_K => 144,
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
}

fn nbytes(dims: &[u64], ty: u32) -> u64 {
    let ne: u64 = dims.iter().product();
    let bs = blck_size(ty);
    ne / bs * type_size(ty)
        + if ne % bs == 0 { 0 } else { type_size(ty) }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

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

/// テンソルの中身。**有限値だけ**を作る (スケール欄を乱数にすると NaN になる)。
fn tensor_bytes(name: &str, dims: &[u64], ty: u32, seed: u32) -> Vec<u8> {
    let nb = nbytes(dims, ty) as usize;
    let mut b: Vec<u8> = (0..nb).map(|i| pat(i, 7 + seed)).collect();
    match ty {
        TY_F32 => {
            for i in 0..nb / 4 {
                let v = if name.contains("norm") {
                    1.0f32
                } else if name.ends_with(".ssm_a") {
                    -0.5f32
                } else if name.ends_with("ssm_dt.bias") {
                    0.0f32
                } else if name.ends_with("ssm_conv1d.weight") {
                    0.05 * (((i % SSM_CONV as usize) + 1) as f32)
                } else if name.ends_with("ple_conv1d.weight") {
                    0.1 * (((i % PLE_CONV as usize) + 1) as f32)
                } else {
                    (((i as u32).wrapping_add(seed * 5) % 17) as f32 - 8.0) * 0.01
                };
                b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        TY_BF16 => {
            for i in 0..nb / 2 {
                let v = (((i as u32).wrapping_add(seed * 3) % 13) as f32 - 6.0) * 0.01;
                let bits = (v.to_bits() >> 16) as u16;
                b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        TY_Q8_0 => {
            for blk in 0..nb / 34 {
                let o = blk * 34;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                for j in 0..32u32 {
                    let q = ((j as i32 + blk as i32 * 3 + seed as i32 * 7) % 11) - 5;
                    b[o + 2 + j as usize] = q as i8 as u8;
                }
            }
        }
        TY_Q5_1 => {
            for blk in 0..nb / 24 {
                let o = blk * 24;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
            }
        }
        TY_Q4_K => {
            for blk in 0..nb / 144 {
                let o = blk * 144;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.0));
                for j in 0..8 {
                    b[o + 4 + j] = 0x01;
                }
                for j in 8..12 {
                    b[o + 4 + j] = 0x11;
                }
            }
        }
        TY_IQ4_NL => {
            for blk in 0..nb / 18 {
                let o = blk * 18;
                b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
            }
        }
        other => panic!("テストヘルパ未対応の型 {other}"),
    }
    b
}

fn write_gguf(path: &Path, kvs: &[(String, Kv)], tensors: &[TensorSpec]) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        push_str(&mut head, k);
        v.write(&mut head);
    }

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for t in tensors {
        offsets.push(cur);
        cur = align_up((cur + nbytes(&t.dims, t.ty)) as usize, ALIGN) as u64;
    }

    for (t, off) in tensors.iter().zip(&offsets) {
        push_str(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_le_bytes());
        head.extend_from_slice(&off.to_le_bytes());
    }

    let mut out = head;
    out.resize(align_up(out.len(), ALIGN), 0);
    for (i, t) in tensors.iter().enumerate() {
        out.extend_from_slice(&tensor_bytes(&t.name, &t.dims, t.ty, i as u32));
        out.resize(align_up(out.len(), ALIGN), 0);
    }
    std::fs::write(path, out).expect("テスト GGUF の書き出し");
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_p120_quality_{tag}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp dir");
        TempDir(p)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const N_EMBD: u64 = 256;
const N_VOCAB: u64 = 8;
const N_HEAD: u64 = 2;
const HEAD: u64 = 16;
const N_HEAD_KV: u64 = 1;
const N_EXPERT: u64 = 4;
const N_EXPERT_USED: u32 = 2;
const N_FF: u64 = 32;
const HC: u64 = 2;
const HC_LR: u64 = 32;
const SSM_CONV: u64 = 4;
const SSM_STATE: u64 = 8;
const SSM_GROUP: u64 = 2;
const SSM_DT: u64 = 4;
const SSM_INNER: u64 = 32;
const INTERVAL: u64 = 4;
const IDX_HEAD: u64 = 2;
const IDX_DIM: u64 = 8;
const IDX_TOP_K: u64 = 8;
const RATIO: u32 = 4;
const PLE_NGRAM: u64 = 3;
const PLE_HEADS_PER_NGRAM: u64 = 2;
const N_EMBD_PER_LAYER: u64 = N_EMBD / ((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
const PLE_LAYER: usize = 1;
const PLE_CONV: u64 = 4;
const PLE_MIN_ROWS: u64 = 40;
const DEFAULT_LAYERS: usize = 4;

const HC_DIM: u64 = N_EMBD * HC;
const KEY_DIM: u64 = SSM_STATE * SSM_GROUP;
const VALUE_DIM: u64 = SSM_INNER;
const CONV_DIM: u64 = 2 * KEY_DIM + VALUE_DIM;

fn synth_kvs(layers: usize) -> Vec<(String, Kv)> {
    let mut ratios = vec![0u32; layers];
    for (il, r) in ratios.iter_mut().enumerate() {
        if (il + 1) % INTERVAL as usize == 0 {
            *r = RATIO;
        }
    }
    vec![
        ("general.architecture".to_string(), Kv::Str("qwen4exp")),
        ("general.alignment".to_string(), Kv::U32(ALIGN as u32)),
        ("qwen4exp.block_count".to_string(), Kv::U32(layers as u32)),
        ("qwen4exp.context_length".to_string(), Kv::U32(1024)),
        ("qwen4exp.embedding_length".to_string(), Kv::U32(N_EMBD as u32)),
        (
            "qwen4exp.attention.head_count".to_string(),
            Kv::U32(N_HEAD as u32),
        ),
        (
            "qwen4exp.attention.head_count_kv".to_string(),
            Kv::U32(N_HEAD_KV as u32),
        ),
        (
            "qwen4exp.attention.key_length".to_string(),
            Kv::U32(HEAD as u32),
        ),
        (
            "qwen4exp.attention.value_length".to_string(),
            Kv::U32(HEAD as u32),
        ),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon".to_string(),
            Kv::F32(1e-6),
        ),
        ("qwen4exp.rope.dimension_count".to_string(), Kv::U32(4)),
        (
            "qwen4exp.rope.dimension_sections".to_string(),
            Kv::ArrU32(vec![2, 2, 2, 0]),
        ),
        ("qwen4exp.rope.freq_base".to_string(), Kv::F32(1e7)),
        ("qwen4exp.expert_count".to_string(), Kv::U32(N_EXPERT as u32)),
        (
            "qwen4exp.expert_used_count".to_string(),
            Kv::U32(N_EXPERT_USED),
        ),
        (
            "qwen4exp.expert_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length".to_string(),
            Kv::U32(N_FF as u32),
        ),
        (
            "qwen4exp.ssm.conv_kernel".to_string(),
            Kv::U32(SSM_CONV as u32),
        ),
        ("qwen4exp.ssm.state_size".to_string(), Kv::U32(SSM_STATE as u32)),
        (
            "qwen4exp.ssm.group_count".to_string(),
            Kv::U32(SSM_GROUP as u32),
        ),
        (
            "qwen4exp.ssm.time_step_rank".to_string(),
            Kv::U32(SSM_DT as u32),
        ),
        (
            "qwen4exp.ssm.inner_size".to_string(),
            Kv::U32(SSM_INNER as u32),
        ),
        (
            "qwen4exp.full_attention_interval".to_string(),
            Kv::U32(INTERVAL as u32),
        ),
        (
            "qwen4exp.hyper_connection.count".to_string(),
            Kv::U32(HC as u32),
        ),
        (
            "qwen4exp.hyper_connection.low_rank".to_string(),
            Kv::U32(HC_LR as u32),
        ),
        (
            "qwen4exp.attention.indexer.head_count".to_string(),
            Kv::U32(IDX_HEAD as u32),
        ),
        (
            "qwen4exp.attention.indexer.key_length".to_string(),
            Kv::U32(IDX_DIM as u32),
        ),
        (
            "qwen4exp.attention.indexer.top_k".to_string(),
            Kv::U32(IDX_TOP_K as u32),
        ),
        (
            "qwen4exp.attention.compress_ratios".to_string(),
            Kv::ArrU32(ratios),
        ),
        (
            "qwen4exp.embedding_length_per_layer_input".to_string(),
            Kv::U32(N_EMBD_PER_LAYER as u32),
        ),
        (
            "qwen4exp.ple.layers".to_string(),
            Kv::ArrU32(vec![PLE_LAYER as u32]),
        ),
        (
            "qwen4exp.ple.ngram_size".to_string(),
            Kv::U32(PLE_NGRAM as u32),
        ),
        (
            "qwen4exp.ple.heads_per_ngram".to_string(),
            Kv::U32(PLE_HEADS_PER_NGRAM as u32),
        ),
        (
            "qwen4exp.ple.conv_kernel".to_string(),
            Kv::U32(PLE_CONV as u32),
        ),
        ("qwen4exp.ple.eos_token_id".to_string(), Kv::U32(7)),
        ("qwen4exp.ple.image_token_id".to_string(), Kv::U32(9)),
        (
            "qwen4exp.ple.layer_multipliers".to_string(),
            Kv::ArrU64(vec![1, 2, 3]),
        ),
        (
            "qwen4exp.ple.head_offsets".to_string(),
            Kv::ArrU64(vec![0, 10, 20, 30]),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes".to_string(),
            Kv::ArrU64(vec![10, 10, 10, 10]),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Kv::ArrStr(vec![String::new(); N_VOCAB as usize]),
        ),
    ]
}

fn common_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], TY_F32),
        ts(p("ffn_gate_inp_shexp.weight"), vec![N_EMBD], TY_F32),
        ts(
            p("ffn_gate_exps.weight"),
            vec![N_EMBD, N_FF, N_EXPERT],
            TY_Q4_K,
        ),
        ts(
            p("ffn_up_exps.weight"),
            vec![N_EMBD, N_FF, N_EXPERT],
            TY_Q4_K,
        ),
        ts(
            p("ffn_down_exps.weight"),
            vec![N_FF, N_EMBD, N_EXPERT],
            TY_Q5_1,
        ),
        ts(p("ffn_gate_shexp.weight"), vec![N_EMBD, N_FF], TY_Q8_0),
        ts(p("ffn_up_shexp.weight"), vec![N_EMBD, N_FF], TY_Q8_0),
        ts(p("ffn_down_shexp.weight"), vec![N_FF, N_EMBD], TY_Q8_0),
        ts(p("hc_attn_norm.weight"), vec![HC_DIM], TY_F32),
        ts(p("hc_attn_up.weight"), vec![HC_LR, HC_DIM], TY_Q8_0),
        ts(p("hc_attn_down.weight"), vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(p("hc_attn_inject.weight"), vec![HC_DIM, HC], TY_F32),
        ts(p("hc_ffn_norm.weight"), vec![HC_DIM], TY_F32),
        ts(p("hc_ffn_up.weight"), vec![HC_LR, HC_DIM], TY_Q8_0),
        ts(p("hc_ffn_down.weight"), vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(p("hc_ffn_inject.weight"), vec![HC_DIM, HC], TY_F32),
    ]
}

fn ssm_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ssm_a"), vec![SSM_DT], TY_F32),
        ts(p("ssm_dt.bias"), vec![SSM_DT], TY_F32),
        ts(p("ssm_alpha.weight"), vec![N_EMBD, SSM_DT], TY_F32),
        ts(p("ssm_beta.weight"), vec![N_EMBD, SSM_DT], TY_F32),
        ts(p("ssm_conv1d.weight"), vec![SSM_CONV, CONV_DIM], TY_F32),
        ts(p("ssm_norm.weight"), vec![SSM_STATE], TY_F32),
        ts(p("ssm_out.weight"), vec![VALUE_DIM, N_EMBD], TY_Q8_0),
        ts(p("attn_qkv.weight"), vec![N_EMBD, CONV_DIM], TY_Q8_0),
        ts(p("attn_gate.weight"), vec![N_EMBD, VALUE_DIM], TY_Q8_0),
    ]
}

fn full_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(
            p("attn_q.weight"),
            vec![N_EMBD, 2 * N_HEAD * HEAD],
            TY_Q8_0,
        ),
        ts(p("attn_k.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_v.weight"), vec![N_EMBD, N_HEAD_KV * HEAD], TY_Q8_0),
        ts(p("attn_q_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_k_norm.weight"), vec![HEAD], TY_F32),
        ts(p("attn_output.weight"), vec![N_HEAD * HEAD, N_EMBD], TY_Q8_0),
        ts(
            p("indexer.q_proj.weight"),
            vec![N_EMBD, IDX_HEAD * IDX_DIM],
            TY_BF16,
        ),
        ts(p("indexer.k_proj.weight"), vec![N_EMBD, IDX_DIM], TY_BF16),
        ts(p("indexer.q_norm.weight"), vec![IDX_DIM], TY_F32),
        ts(p("indexer.k_norm.weight"), vec![IDX_DIM], TY_F32),
    ]
}

fn ple_tensors(il: usize) -> Vec<TensorSpec> {
    let p = |s: &str| format!("blk.{il}.{s}");
    vec![
        ts(p("ple_key.weight"), vec![N_EMBD, HC_DIM], TY_Q8_0),
        ts(p("ple_value.weight"), vec![N_EMBD, N_EMBD], TY_Q8_0),
        ts(p("ple_norm_key.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_norm_query.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_norm_conv.weight"), vec![HC_DIM], TY_F32),
        ts(p("ple_conv1d.weight"), vec![PLE_CONV, HC_DIM], TY_F32),
    ]
}

fn synth_tensors(layers: usize) -> Vec<TensorSpec> {
    let mut out = vec![
        ts("token_embd.weight", vec![N_EMBD, N_VOCAB], TY_Q8_0),
        ts("output.weight", vec![N_EMBD, N_VOCAB], TY_Q8_0),
        ts("output_hc_norm.weight", vec![HC_DIM], TY_F32),
        ts("output_hc_up.weight", vec![HC_LR, HC_DIM], TY_Q8_0),
        ts("output_hc_down.weight", vec![HC_DIM, HC_LR], TY_Q8_0),
        ts(
            PLE_TABLE_NAME,
            vec![N_EMBD_PER_LAYER, PLE_MIN_ROWS],
            TY_IQ4_NL,
        ),
    ];
    for il in 0..layers {
        out.extend(common_tensors(il));
        if (il + 1) % INTERVAL as usize == 0 {
            out.extend(full_tensors(il));
        } else {
            out.extend(ssm_tensors(il));
        }
        if il == PLE_LAYER {
            out.extend(ple_tensors(il));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// テスト本体
// ---------------------------------------------------------------------------

#[test]
#[ignore = "実 GGUF が必要 (env DENNO_P120_GGUF)。無い環境では組み込み合成モデルで SELFTEST"]
fn expert_prune_quality() {
    let full_mode = std::env::var("DENNO_P120_GGUF")
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let n_tokens: usize = std::env::var("DENNO_P120_N_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if full_mode { 32 } else { 6 });
    let out_path = std::env::var("DENNO_P120_OUT").ok().filter(|s| !s.is_empty());

    let mode_label = if full_mode { "FULL" } else { "SELFTEST" };
    println!("[p120] mode={mode_label} n_tokens={n_tokens}");
    let target = match load_target() {
        Ok(t) => t,
        Err(e) => {
            if full_mode {
                panic!("DENNO_P120_GGUF を開けない: {e}");
            }
            println!("[p120] 合成モデルを作れないので SELFTEST を skip: {e}");
            return;
        }
    };
    let eos = target.tok.as_ref().and_then(|t| t.eos());
    println!(
        "[p120] layers={} experts={} used={} vocab={}",
        target.params.n_layer,
        target.params.n_expert,
        target.params.n_expert_used,
        target.params.n_vocab
    );

    // プロンプトの用意 (FULL: chat 展開 / SELFTEST: 生トークン列)。
    let mut cases: Vec<(String, Vec<u32>)> = Vec::new();
    if full_mode {
        let tok = target.tok.as_ref().expect("FULL にはトークナイザが要る");
        for (i, text) in PROMPTS.iter().enumerate() {
            let msgs = vec![("user".to_string(), text.to_string())];
            let ids = tok
                .encode_messages(None, &msgs, None, false)
                .unwrap_or_else(|e| panic!("prompt[{i}] を展開できない: {e}"));
            assert!(!ids.is_empty(), "prompt[{i}] が空");
            cases.push((format!("p{i}"), ids));
        }
    } else {
        for (i, ids) in SELFTEST_PROMPTS.iter().enumerate() {
            cases.push((format!("p{i}"), ids.to_vec()));
        }
    }

    // 基準生成 (ゲート全部 off)。
    let mut refs: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::with_capacity(cases.len());
    for (i, (_, prompt)) in cases.iter().enumerate() {
        let (toks, logits) =
            generate_reference(&target, prompt, n_tokens, eos).expect("基準生成");
        println!(
            "[p120] reference p{i}: {} toks (prompt {} + n {n_tokens})",
            toks.len(),
            prompt.len()
        );
        refs.push((toks, logits));
    }

    // 設定ごとに teacher forcing。
    let configs = config_list(full_mode);
    let mut rows: Vec<(String, String, RunResult)> = Vec::new();
    for (i, (label, prompt)) in cases.iter().enumerate() {
        let max_ctx = prompt.len() + n_tokens + 64;
        let mut dec = build_decoder(&target, max_ctx).expect("decoder");
        let (ref_toks, ref_logits) = &refs[i];
        for (cfg_label, cfg) in &configs {
            let r = run_config(&mut dec, prompt, ref_toks, ref_logits, *cfg)
                .unwrap_or_else(|e| panic!("{label}/{cfg_label}: {e}"));
            println!(
                "  {label:>4} {cfg_label:<12} agree={:.3} kl={:.6} experts/call={:.2} calls={} first_diff={}",
                r.agree,
                r.mean_kl,
                r.experts_per_call,
                r.calls,
                first_diff_str(r.first_diff)
            );
            rows.push((label.clone(), cfg_label.clone(), r));
        }
        // full 設定は自分自身の再現なので必ず一致する (harness の簿記検証)。
        let full_row = rows
            .iter()
            .find(|(l, c, _)| l == label && c == "full")
            .expect("full row")
            .2
            .clone();
        assert_eq!(
            full_row.agree, 1.0,
            "{label}: full 設定の一致率が 1.000 でない (harness の欠陥)"
        );
        assert_eq!(
            full_row.mean_kl, 0.0,
            "{label}: full 設定の KL が 0 でない (harness の欠陥)"
        );
    }

    // 最終表 (貼り付け用)。
    let mut table = String::new();
    table.push_str("config,prompt,agree,mean_kl,experts_per_call,calls,first_diff\n");
    for (label, cfg_label, r) in &rows {
        table.push_str(&format!(
            "{cfg_label},{label},{:.3},{:.6},{:.2},{},{}\n",
            r.agree,
            r.mean_kl,
            r.experts_per_call,
            r.calls,
            first_diff_str(r.first_diff)
        ));
    }
    println!("\n[p120] result table ({mode_label})\n{table}");
    if let Some(p) = out_path {
        std::fs::write(&p, &table).expect("DENNO_P120_OUT の書き出し");
        println!("[p120] 表を {p} に書いた");
    }

    // full 設定の全行は一致率 1.000 / KL 0 (全体の最終検査)。
    for (label, cfg_label, r) in &rows {
        if cfg_label == "full" {
            assert_eq!(r.agree, 1.0, "{label}: full の一致率");
            assert_eq!(r.mean_kl, 0.0, "{label}: full の KL");
        }
    }
    println!("[p120] ok: full config vs itself は全プロンプトで agree=1.000 / kl=0");
}
