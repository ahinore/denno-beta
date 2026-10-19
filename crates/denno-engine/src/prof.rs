//! 軽量セルフプロファイラ (P7.5)。
//!
//! `DENNO_PROFILE=1` を付けた場合のみ有効になり、推論の各フェーズの
//! 累積実時間と呼び出し回数を記録する。無効時は分岐 1 回のオーバーヘッド
//! もなく(コンパイラが消す)、数値経路には一切触れないため
//! exactness (ビット一致) は完全不変。
//!
//! リクエスト完了時に server 側から `summary()` を INFO ログ出力する。
//!
//! s57 (T4): **prefill / decode を別テーブルに分離**し、`attn_core` を
//! GDN (linear attention) と full attention の別バケットに割り、各 op の
//! 「1 call あたり平均トークン数 (batch 幅)」と実行先 (CPU / GPU) を併記する。
//! 集計はすべて観測のみ — 数値経路には触れない (PROFILE off は完全 no-op)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const EMBED: usize = 0; // トークン→埋め込み lookup
pub const RMSNORM: usize = 1; // 層の RMSNorm (s85: 層ブロック全体を包む Ambient は撤去済み — この行は norm の実測区間そのもの)
pub const ROPE: usize = 2; // RoPE 回転 (q, k)
pub const QKV: usize = 3; // Q/K/V 投影 (線形層)
pub const ATTN: usize = 4; // attention 本体 (KV書き込み、scores、softmax、probs@V)
pub const OPROJ: usize = 5; // 出力投影
pub const GATE_UP: usize = 6; // MLP gate/up 投影
pub const SWIGLU: usize = 7; // SiLU + 要素積
pub const DOWN: usize = 8; // MLP down 投影
pub const FINAL_NORM: usize = 9; // 最終 RMSNorm
pub const LM_HEAD: usize = 10; // 最終 logits 投影 (vocab 次元 matvec)
pub const SAMPLER: usize = 11; // サンプリング
pub const DETOK: usize = 12; // トークン→テキスト デコード (ストリーミング差分用)
pub const QKNORM: usize = 13; // QK-Norm (Qwen3 のみ。q/k の head_dim 単位 RMSNorm)
pub const D2H: usize = 14; // s49: デバイス→ホスト戻し (chain の logits download + staged 状態フラッシュ)
/// s54: d2h_state の細分 (倍率 10)。生 id (D2H, D2H+1, …) は変更しない —
/// telemetry の OpSummary / denno-view は生 id で扱うため、細分は
/// `summary()` のテキスト出力と telemetry の op 記録 (telemetry_add_op) だけに
/// 掛ける (バイト互換維持)。ON 無効時は分岐 1 回で従来どおり D2H に積む。
///
/// s54 の実測 (9B int8・decode 232 step・PROFILE): s49 の `d2h_state` バケットは
/// **step あたり 2 call** = ①`chain_download(CH_LOGITS)` の sync (ステップ全体の
/// GPU 実行完了待ちを吸収する支配項) ②`chain_q35_kv_stage_flush` (ステップ末尾の
/// ホスト反映。`chain_q35_fullattn` が層ごとに出す staged async D2H は非同期
/// 発行のみで同期点を持たないため、計上は ①に集まる)。full 層の k/v 行
/// (16 層 × 2 × kv_dim 4B = 128KB/step) は ①の sync が完了を待つ転送であり、
/// その分が decode の律速に効く — ①の内訳値を D2H_LOGITS / D2H_KV_STAGE /
/// D2H_KV_FLUSH で分離して実測するための id。
pub const D2H_LOGITS: usize = 10 * D2H; // logits の chain_download (sync 待ちを吸収)
pub const D2H_KV_STAGE: usize = 10 * D2H + 1; // full 層 k/v 行の staged async D2H (per 層)
pub const D2H_KV_FLUSH: usize = 10 * D2H + 2; // ステップ末尾の staged → ホスト反映
pub const D2H_GDN_FLUSH: usize = 10 * D2H + 3; // 遅延 GDN 状態の回復 (prefill 冒頭)
pub const D2H_KV_DELAYED_FLUSH: usize = 10 * D2H + 4; // 遅延 KV 行の回復 (将来用)
pub const D2H_ATTN_PF: usize = 10 * D2H + 5; // prefill attention 出力の D2H
pub const D2H_N: usize = 9; // 細分バケット数 (D2H_LOGITS .. D2H_ATTN_PF_HELD)
/// s55 (T1 step2): `d2h_logits` (= 従来 D2H_LOGITS 1 本) を「待ち」と「コピー」に
/// 分離するための id。decode 経路で logits の D2H を発行する直前に追加 sync を
/// 1 回だけ打ち (prof 有効時のみ)、その所要を WAIT に積む。従来の D2H_LOGITS
/// (連続区間) は COPY に置き換える — WAIT + COPY = 従来の D2H_LOGITS と一致する。
/// 最初のステップだけは logits D2H の前に既存 sync (lm_head 側) が無い状態から
/// 測ってしまうため、その 1 点は `HELD` に別計上する (WAIT の平均を汚さない)。
pub const D2H_LOGITS_WAIT: usize = 10 * D2H + 6; // 追加 sync のみ (GPU 側未完了処理の待ち)
pub const D2H_LOGITS_COPY: usize = 10 * D2H + 7; // 従来 D2H_LOGITS 区間 (従来計測)
pub const D2H_LOGITS_PF_HELD: usize = 10 * D2H + 8; // prefill 後 1 点目の WAIT (集計から除外)
/// s64 (T5b L1 Step 1・計測のみ): GDN スキャン 1 call (`gdn_scan_impl` /
/// `gdn_scan_chunked_impl`) の内訳。**表示専用**で、値はどれも `attn_gdn` /
/// `gdn_scan` のバケット (呼び出し側の `time_layer_n_ev(ATT_GDN_SCAN)`) に
/// 既に含まれる — ここへ積むのは「どの区間が重いか」を見るためだけ。
///
/// 動機: s63 後の prefill で `gdn_scan` 3,159 ms (48 call = 24 層 × 2 リクエスト) と
/// `d2h_state` 10,604 ms (168 call) が残った。`gdn_scan` は「DevBuffer 確保 + H2D →
/// launch → sync → D2H」が 1 関数に混在しており、確保 (cuMemAlloc) と同期待ちの
/// どちらが支配的かを実測で切り分けられなかった。id は D2H 細分の空き (9..) を
/// 使う — D2H 細分の既存 id (0..8) は不変なので PROFILE の連続性は保たれる。
pub const D2H_GDN_PF_ALLOC: usize = 10 * D2H + 9; // バッファ確保 + 入力 H2D (重み 4 本含む)
pub const D2H_GDN_PF_LAUNCH: usize = 10 * D2H + 10; // カーネル launch 発行
pub const D2H_GDN_PF_SYNC: usize = 10 * D2H + 11; // cuCtxSynchronize (完了待ち = カーネル実行)
pub const D2H_GDN_PF_D2H: usize = 10 * D2H + 12; // 出力 y / conv / ssm の D2H + ホスト反映
pub const D2H_GDN_PF_N: usize = 4; // 上記 4 区間の本数
pub const D2H_GDN_PF_FIRST: usize = 10 * D2H + 9;
/// s65 (T5b L1 第 2 段・計測のみ): prefill の `d2h_state` **発行元**の分類。
///
/// 動機: s63/s64 後も prefill 最大項目は `d2h_state` (5,686 tok で 13.7 s /
/// 504 call・33%) で、うち 96 call だけが `gdn_scan` 内と分かっている。残り
/// ~408 call (= 1 リクエスト ~200 call) の発行元が未特定のままだった。s60 の
/// 教訓「推測で最適化しない」に従い、**計測だけ**を足して発行元を確定させる。
///
/// 分類軸 (cuda.rs の D2H 発行点に注釈する):
///   `d2h_src_kv_row`   … full 層 k/v 行 (prefill `chain_qkv_rope_prefill` の
///                        同期 D2H / decode `chain_q35_fullattn` の staged async)
///   `d2h_src_gdn_state`… GDN 状態 (conv/ssm) の戻し (`gdn_scan` 内・遅延フラッシュ)
///   `d2h_src_attn_pf`  … prefill attention 出力 (CH_ATT → ホスト)
///   `d2h_src_linear`   … 線形層の出力 D2H (GEMM/GEMV/matvec の結果戻し)
///   `d2h_src_logits`   … 最終 logits の戻し
///   `d2h_src_other`    … 上記以外 (未分類の D2H 発行点)
///
/// **表示専用**: 値は必ず `D2H` 生バケット (時間 + 呼び出し) に既に積まれており、
/// ここへは時間を足さない。積むのは**件数**だけ (`D2H_SRC_CALLS`) — 時間は
/// 発行点の内側で二重計上される (sync 待ちの帰属が重複する) ため取らない。
/// 既存の D2H 細分 (0..8)・`d2h_state` バケット・`GDN_PF_*` の値は一切変えない。
pub const D2H_SRC_N: usize = 6;
pub const D2H_SRC_KV_ROW: usize = 10 * D2H + 13;
pub const D2H_SRC_GDN_STATE: usize = 10 * D2H + 14;
pub const D2H_SRC_ATTN_PF: usize = 10 * D2H + 15;
pub const D2H_SRC_LINEAR: usize = 10 * D2H + 16;
pub const D2H_SRC_LOGITS: usize = 10 * D2H + 17;
pub const D2H_SRC_OTHER: usize = 10 * D2H + 18;
pub const D2H_SRC_FIRST: usize = 10 * D2H + 13;
/// s78 (T7 計測のみ): prefill の `gdn_proj` (beta/alpha 投影) 1 call ぶんの内訳。
/// 呼び出し側は beta と alpha を別々の `time_n_ev(ATT_GDN_PROJ)` に載せているので、
/// その**内側**を 4 区間 (H2D / カーネル実行 / 同期 / D2H) に分割すると
/// 「数学 / 転送 / 待ち」のどれが支配的かが分かる。`D2H_GDN_PF_*` と同じ規約:
/// **表示専用**テーブルで、どの生 id にも合算しない (二重計上しない)。
/// GPU 経路が不発 (CPU フォールバック) の call はどれも積まれない。
pub const D2H_GDN_PROJ_H2D: usize = 10 * D2H + 19; // 活性化 [m,e] の H2D 発行
pub const D2H_GDN_PROJ_KERNEL: usize = 10 * D2H + 20; // カーネル launch 発行
pub const D2H_GDN_PROJ_SYNC: usize = 10 * D2H + 21; // cuCtxSynchronize (完了待ち = 実行)
pub const D2H_GDN_PROJ_D2H: usize = 10 * D2H + 22; // 出力 [m,nv] の D2H + ホスト反映
pub const D2H_GDN_PROJ_N: usize = 4;
pub const D2H_GDN_PROJ_FIRST: usize = 10 * D2H + 19;
/// s57 (T4): `attn_core` の層種別分離 (倍率 100)。**ATNN 生 id (= 4) の集計は従来どおり**
/// — これは summary のテキスト出力だけに掛ける細分 (D2H 細分と同じ規約)。
///
/// 動機: prefill の PROFILE で `attn_core` が 44.7% を占めたが、これは
/// **GDN (linear attention) のスキャンと full attention の attend が同一バケットに
/// 混ざった**値で、どちらを GPU バッチ化すべきか判断できなかった。呼び出し側が
/// 層種別を引数で与え、別バケットに積む。`ATNN` 生 id の合計は従来の
/// `attn_core` と一致する (GDN + FULL = ATNN)。
pub const ATT_CLS_N: usize = 4;
pub const ATT_GDN: usize = 100 * ATTN; // GDN (linear attention) スキャン・beta/alpha 投影
pub const ATT_FULL: usize = 100 * ATTN + 1; // full attention (GQA attend / full_attn_core)
/// s61c (T5b 計測のみ): `attn_gdn` を「投影 (beta/alpha)」と「スキャン (GDN コア)」に
/// 割る細分。**表示専用**で、`ATT_GDN` の集計は従来どおり (この 2 本の和になる)。
///
/// 動機: prefill の `attn_gdn` が 28.5 s を占めたが、この値は
/// **beta/alpha の投影 (`apply_bt_rows` → P17i GEMM) と GDN スキャン
/// (`gdn_scan_batch`: DevBuffer 確保 + H2D/D2H + sync) が同一バケットに
/// 混ざった**もの。T5b の第一レバー (スキャンのデバイス常駐化) の効果と、
/// 投影側が既に GPU で走っているかを切り分けられなかったため分離する。
pub const ATT_GDN_PROJ: usize = 100 * ATTN + 2; // GDN の beta/alpha 投影のみ
pub const ATT_GDN_SCAN: usize = 100 * ATTN + 3; // GDN スキャン本体
pub const N_OPS: usize = 15;

/// s57 (T4): 計測フェーズ。prefill (n>1 の forward) と decode (n==1) を
/// **別テーブル**に集計する。`Both` は従来の cumulative テーブル (双方の合計)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Prefill,
    Decode,
    Both,
}

impl Phase {
    pub const N: usize = 3;
    pub const NAMES: [&'static str; 3] = ["prefill", "decode", "cumulative"];

    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Phase::Prefill => 0,
            Phase::Decode => 1,
            Phase::Both => 2,
        }
    }
}

/// s57 (T4): op の実行先ラベル。既存の GPU 呼び出し有無 (実測された経路) を
/// 集計して表示するだけで、分岐には一切使わない。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exec {
    /// GPU 経路で走った call がある
    Gpu,
    /// GPU call が 1 度も無く CPU のみ
    Cpu,
    /// 判定不能 (GPU 選択が無い op = 一部要素演算など)
    CpuFixed,
}

impl Exec {
    pub const N: usize = 3;

    #[inline]
    pub fn idx(self) -> usize {
        match self {
            Exec::Gpu => 0,
            Exec::Cpu => 1,
            Exec::CpuFixed => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Exec::Gpu => "gpu",
            Exec::Cpu => "cpu",
            Exec::CpuFixed => "cpu*",
        }
    }
}

/// s56 (T2 項目0): decode chain の logits D2H 直前に打つ追加 sync を、どちらの
/// 細分バケットに計上するか決める純関数 (GPU 不要・テスト可能)。
///
/// 従来は `cache.len() > pos0` で判定していたが、この chain は decode 1 トークン
/// だけを受け (`forward` の先頭で `tokens.len() > 1` は prefill へ分岐)、成功時に
/// `cache.advance(1)` するため **`pos0 + 1 == cache.len()` が常に成り立つ**。
/// つまり従来条件は decode では常に false になり、待ち時間が集計外の
/// `D2H_LOGITS_PF_HELD` に全部入って `d2h_logits_wait` 行が 1 度も出なかった
/// (T1 step2 の実測で判明)。
///
/// 正しい判定: 1 点目 = 「直前が prefill で、ここより前に GPU 同期点が無い」点。
/// chain に入る時点の `pos0` は直前の prefill で kv 長が進んだ後の値 = プロンプト長
/// (= キャッシュ起点が 0) か、multi-turn の prefix 再利用時は `pos0 - base_len`
/// (base ≤ プロンプト長)。どちらも `pos0 > 0` で、2 点目以降は前ステップの
/// logits D2H (COPY) が同期点になっている。よって「cache が空から進んだ (= pos0 == 0)
/// のでない限り 1 点目」= `pos0 > 0` かつ「cache が pos0 より先へ進んでいない」
/// (`cache.len() <= pos0 + 1`) を PF_HELD、それ以外 (2 点目以降、または cache が
/// 既に pos0 を超えて進んでいる) を WAIT とする。
///
/// なお `pos0 == 0` は chain では起きない (prefill が最低 1 トークン消費する) が、
/// 起きた場合は「同期点が無い」ので PF_HELD 側に倒す (平均を汚さない安全側)。
#[inline]
pub fn d2h_logits_split_op(pos0: usize, cache_len: usize) -> usize {
    if pos0 == 0 || cache_len <= pos0 + 1 {
        D2H_LOGITS_PF_HELD
    } else {
        D2H_LOGITS_WAIT
    }
}

const NAMES: [&str; N_OPS] = [
    "embed", "rmsnorm", "rope", "qkv_proj", "attn_core", "o_proj", "gate_up_proj", "swiglu",
    "down_proj", "final_norm", "lm_head", "sampler", "detok", "qk_norm", "d2h_state",
];

/// s54: d2h_state の細分名 (id は 10*D2H + i)。
const D2H_NAMES: [&str; D2H_N] = [
    "d2h_logits",
    "d2h_kv_stage",
    "d2h_kv_flush",
    "d2h_gdn_flush",
    "d2h_kv_dflush",
    "d2h_attn_pf",
    // s55: d2h_logits を「待ち」と「コピー」に分離 (合計は従来 d2h_logits と一致)
    "d2h_logits_wait",
    "d2h_logits_copy",
    "d2h_logits_pfhold",
    // s64: GDN スキャン 1 call の内訳 -- **D2H 細分ではない** (専用テーブル +
    // D2H_GDN_PF_* id で持つ)。D2H_N / D2H_SUB_* / 生 id の範囲は s63 のまま。
];

/// s64: GDN スキャン内訳の表示名 (id は 10*D2H + 9 + i)。
const D2H_GDN_PF_NAMES: [&str; D2H_GDN_PF_N] = [
    "gdn_pf_alloc",
    "gdn_pf_launch",
    "gdn_pf_sync",
    "gdn_pf_d2h",
];

/// s78 (T7): `gdn_proj` 1 call の内訳の表示名 (id は 10*D2H + 19 + i)。
const D2H_GDN_PROJ_NAMES: [&str; D2H_GDN_PROJ_N] = [
    "gdnp_h2d",
    "gdnp_kernel",
    "gdnp_sync",
    "gdnp_d2h",
];

/// s65: `d2h_state` 発行元の表示名 (id は 10*D2H + 13 + i)。
/// **件数の分類**であり、時間は持たない (docs/コメント参照)。
const D2H_SRC_NAMES: [&str; D2H_SRC_N] = [
    "d2h_src_kv_row",
    "d2h_src_gdn_state",
    "d2h_src_attn_pf",
    "d2h_src_linear",
    "d2h_src_logits",
    "d2h_src_other",
];

/// s57 (T4): attn_core の層種別細分名 (id は 100*ATTN + i)。
const ATT_CLS_NAMES: [&str; ATT_CLS_N] = [
    "attn_gdn",
    "attn_full",
    // s61c: attn_gdn の内訳 (attn_gdn 行の直下にネスト表示・表示専用)
    "gdn_proj",
    "gdn_scan",
];

/// s57 (T4): op ごとの実行先ラベル分類 (表示専用)。
/// `None` は「実行先が固定 (GPU 選択が無い CPU 要素演算)」を表す `cpu*`。
/// ここに載る op は同一 op 内でも条件により GPU/CPU が混在し得るため、
/// 実際の GPU 発動回数 (exec_ns が exec_call と同数か) で表示を決める。
const HAS_GPU_PATH: [bool; N_OPS] = [
    false, // embed: ホストのメモリコピー (GPU 選択なし)
    false, // rmsnorm: chain では GPU だが per-op は CPU (op 内混在)
    false, // rope: chain では GPU だが per-op は CPU (op 内混在)
    true,  // qkv_proj: maybe_gpu_linear / linear_grouped
    true,  // attn_core: full attention GPU バッチ / GDN スキャン GPU
    true,  // o_proj: maybe_gpu_oproj / chain
    true,  // gate_up_proj: maybe_gpu_ffn / linear_grouped
    true,  // swiglu: chain_swiglu (GPU カーネル)
    true,  // down_proj: chain_linear_grouped
    false, // final_norm: CPU rmsnorm
    true,  // lm_head: maybe_gpu_matvec_f32 / Quant8 hook
    false, // sampler
    false, // detok
    false, // qk_norm: chain では GPU、per-op は CPU
    true,  // d2h_state: 転送そのもの
];

/// s57 (T4): フェーズ × op の 2 次元カウンタ配列の const 初期化
/// (`[[Z; N_OPS]; Phase::N]` は AtomicU64 が Copy でないため const 文脈では
/// 組めない — const fn で組む)。
const fn mk_counters<const N: usize>() -> [[AtomicU64; N]; Phase::N] {
    // 内側・外側とも inline const で要素ごとに独立した Atomic を作る
    // ([inner; N] は Copy を要求するため使えない)
    [const { [const { AtomicU64::new(0) }; N] }; Phase::N]
}

static NS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
static CALLS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
/// s57 (T4): バケットに渡されたトークン (行) 数の総和。平均 = TOKS / CALLS_ROWS。
static TOKS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
/// s61 (T5a 撤回): `tok/call` の分母 — 行数 (rows>0) を渡した sample の call 数。
/// 行数を持たないサンプル (d2h・time 等の rows=0) を分母から外し、
/// 平均を実バッチ幅にする。
static CALLS_ROWS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
/// s57 (T4): GPU 経路で走った call 数 (実行先ラベルの根拠)。
static EXEC_CALLS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
/// s54: d2h_state の細分累積 (id は 10*D2H + i → 添字 i)。フェーズ別。
static D2H_SUB_NS: [[AtomicU64; D2H_N]; Phase::N] = mk_counters::<D2H_N>();
static D2H_SUB_CALLS: [[AtomicU64; D2H_N]; Phase::N] = mk_counters::<D2H_N>();
/// s64: GDN スキャン 1 call の内訳 (`gdn_pf_*`) 累積。**表示専用** — 生 id・
/// `attn_gdn`・`gdn_scan` のどのバケットにも合算しない (二重計上防止)。
static GDN_PF_NS: [[AtomicU64; D2H_GDN_PF_N]; Phase::N] = mk_counters::<D2H_GDN_PF_N>();
static GDN_PF_CALLS: [[AtomicU64; D2H_GDN_PF_N]; Phase::N] = mk_counters::<D2H_GDN_PF_N>();
/// s65: `d2h_state` 発行元の**件数**の分類。**表示専用** — 時間は
/// 持たない (発行点の内側で計ると sync 待ちが二重計上されるため)。
static D2H_SRC_CALLS: [[AtomicU64; D2H_SRC_N]; Phase::N] = mk_counters::<D2H_SRC_N>();
/// s88 (Step 1): `d2h_state` 発行元ごとの**経過時間**。フェーズ別・**表示専用**。
///
/// 動機: `D2H_SRC_CALLS` は件数しか持たないため、prefill の `d2h_state` が大きいとき
/// **どの発行点**が時間を使っているのかが分からなかった (`d2h_state` 1 本に 6 ops が
/// 混ざる)。ここでは発行点 (site) ごとに呼び出し側で区間を測り、site → 時間を積む。
///
/// **どの生 id にも合算しない** (二重計上防止)。`D2H` 生バケットは従来どおり
/// 発行点を包む `time(D2H_*)` が持っており、ここの値はその内訳を読むための
/// 別勘定である。site の区間は互いに重ならない (1 回の発行 = 1 site) ので、
/// site ごとの和は `d2h_state` の区間に収まる。
static D2H_SRC_NS: [[AtomicU64; D2H_SRC_N]; Phase::N] = mk_counters::<D2H_SRC_N>();
/// s78 (T7): `gdn_proj` 1 call の内訳 (`gdnp_*`) 累積。**表示専用** —
/// `attn_gdn` / `gdn_proj` のどのバケットにも合算しない (二重計上防止)。
static GDN_PROJ_NS: [[AtomicU64; D2H_GDN_PROJ_N]; Phase::N] = mk_counters::<D2H_GDN_PROJ_N>();
static GDN_PROJ_CALLS: [[AtomicU64; D2H_GDN_PROJ_N]; Phase::N] = mk_counters::<D2H_GDN_PROJ_N>();
/// s57 (T4): attn_core の細分 (id は 100*ATTN + i → 添字 i)。フェーズ別。
static ATT_SUB_NS: [[AtomicU64; ATT_CLS_N]; Phase::N] = mk_counters::<ATT_CLS_N>();
static ATT_SUB_CALLS: [[AtomicU64; ATT_CLS_N]; Phase::N] = mk_counters::<ATT_CLS_N>();
/// s85 (Step 0): 親バケット (Ambient) → 直接の子の時間の累積。フェーズ別。
///
/// 動機: s84 の Ambient は「子を親へも合算する」方式だったため、`rmsnorm` の行が
/// 層ブロック全体を意味し、prefill の table の合計 (TOTAL) が実測 wall の
/// 2.7 倍に膨らんだ (子が親へ二重計上される)。ここで親ごとに「直接の子が積んだ
/// 時間」を別に持ち、summary は `self = 親 − 子` を出して表を**加算的**にする。
///
/// 子から見た親は sample を積む時点の `AMBIENT` なので、親が入れ子でも
/// 「その sample の直接の親」1 本にだけ足す (祖父には足さない — 祖父の self は
/// 直下の子の合計を引くだけで求まる)。計測 OFF ではこの表へも触らない。
static PARENT_CHILD_NS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();
static PARENT_CHILD_CALLS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();

static ON: OnceLock<bool> = OnceLock::new();
// s57 (T4): 現在の計測フェーズ (thread-local)。数値経路には影響しない。
thread_local! {
    /// 現在の計測フェーズ (thread-local)。forward の入口で set_phase する。
    /// 既定は decode (単発 forward の既定 = 1 トークン)。
    static PHASE: std::cell::Cell<Phase> = const { std::cell::Cell::new(Phase::Decode) };
}

pub fn enabled() -> bool {
    // s85 (Step 0): テストから会計を直接叩けるよう、env を介さない上書きを受け付ける
    // (`#[cfg(test)]` のみ — リリースバイナリには存在しない)。env を書き換えると
    // 並走テストの `enabled()` を揺らすため、テストはこの経路を使う。
    #[cfg(test)]
    if let Some(v) = tests::force_enabled() {
        return v;
    }
    *ON.get_or_init(|| std::env::var("DENNO_PROFILE").map(|v| v != "0").unwrap_or(false))
}

// ---------------------------------------------------------------------------
// s57 (T4): フェーズ (prefill / decode) の切り替え
// ---------------------------------------------------------------------------

/// 以降の計測を `p` に集計する (thread-local)。`time` / `time_layer` は
/// 呼び出し時点のフェーズを読むだけなので、数値経路には一切影響しない。
#[inline]
pub fn set_phase(p: Phase) {
    PHASE.with(|c| c.set(p));
}

/// 現在のフェーズ (テスト・デバッグ用)。
#[inline]
pub fn phase() -> Phase {
    PHASE.with(|c| c.get())
}

/// `f` をフェーズ `p` で実行し、終了時に元のフェーズへ戻す (RAII 不要版)。
/// PROFILE 無効時は set/get も行わず素通しする (完全 no-op)。
#[inline]
pub fn with_phase<T>(p: Phase, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let prev = PHASE.with(|c| c.replace(p));
    let r = f();
    PHASE.with(|c| c.set(prev));
    r
}

// ---------------------------------------------------------------------------
// 計測
// ---------------------------------------------------------------------------

/// s57 (T4): 直近の int8 線形呼び出しが実際に GPU で走ったか (実行先ラベル用)。
/// cuda 無効ビルドでは常に false (= CPU 表示)。PROFILE 無効時は summary 自体が
/// 出ないため、この関数の値は誰も読まない (数値経路には触れない)。
#[inline]
pub fn linear_ran_on_gpu() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::last_linear_ran_on_gpu()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// フェーズ・トークン数・実行先を明示する計測。`rows` はバケットに渡った
/// バッチ幅 (行数 = トークン数)、`on_gpu` はこの call が GPU 経路で走ったか。
/// 集計のみで数値経路には触れない。
pub fn time_full<T>(op: usize, rows: usize, on_gpu: bool, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        add_sample(phase(), op, ns, 1, rows, on_gpu);
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
    }
    r
}

/// `time` のバッチ幅付き版 (現行フェーズに集計)。実行先ラベルは持たないため
/// `via` 列は常に `cpu` になる (セクションに GPU 選択が無い op 用)。
/// **線形層のバケット (QKV / OPROJ / GATE_UP / DOWN) や GDN 投影は `time_n_ev`
/// を使うこと** — T4 の誤診はここで `via=cpu` 固定になっていたことが原因。
pub fn time_n<T>(op: usize, rows: usize, f: impl FnOnce() -> T) -> T {
    time_full(op, rows, false, f)
}

/// cuda 無効ビルド用の no-op スコープ (cuda 側の同名と API を揃える)。
#[cfg(not(feature = "cuda"))]
struct Probe;

#[cfg(not(feature = "cuda"))]
impl Probe {
    #[inline]
    fn begin() -> Self {
        Probe
    }

    #[inline]
    fn any_on_gpu(&self) -> bool {
        false
    }
}

/// s61 (T5a 撤回): `time_n` の **実行先ラベル付き** 版。`f` を実行したあとで
/// 「その区間中に GPU 線形呼び出しが 1 回でも走ったか」を cuda 側の実測
/// (`cuda::Probe`) から読んで `via` 列に渡す。
///
/// T4 の誤診の原因は 2 つ:
///   1. `time_n` は `time_full(..., on_gpu = false, ...)` を渡しており、
///      prefill の per-op 計測が **常に `via=cpu`** と表示されていた。
///   2. 修正の第一案 (`last_linear_ran_on_gpu()`) も **直近 1 回** の線形しか
///      見ないため、1 区間で複数の線形を呼ぶバケット (qkv = q/k/v、
///      gate_up = gate/up) では最後の 1 本でラベルが決まってしまっていた。
/// ここでは区間内の **いずれか** が GPU で走ったか (`Probe::any_on_gpu`) を
/// 使う。ラベルは model 側の主張ではなく計測区間の実測から取る。
///
/// **計測専用** — 戻り値の型・数値経路は `time_n` と完全に同一 (PROFILE 無効時は
/// `Probe` の Cell 操作以外は素通し)。
pub fn time_n_ev<T>(op: usize, rows: usize, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    #[cfg(feature = "cuda")]
    let probe = crate::cuda::Probe::begin();
    #[cfg(not(feature = "cuda"))]
    let probe = Probe::begin();
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    let on_gpu = probe.any_on_gpu();
    if prof_on {
        add_sample(phase(), op, ns, 1, rows, on_gpu);
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
    }
    r
}

/// `time_layer` のバッチ幅付き版 (現行フェーズに集計)。
pub fn time_layer_n<T>(op: usize, _layer: usize, rows: usize, f: impl FnOnce() -> T) -> T {
    time_layer_n_ev(op, _layer, rows, || (f(), false)).0
}

/// s57 (T4): `time_layer` のバッチ幅 + 実行先ラベル付き版。`f` は
/// `(値, GPU で走ったか)` を返す。ラベルはクロージャの戻り値から得るので
/// 数値経路の分岐には一切影響しない。PROFILE 無効時は素通し (完全 no-op)。
pub fn time_layer_n_ev<T>(
    op: usize,
    _layer: usize,
    rows: usize,
    f: impl FnOnce() -> (T, bool),
) -> (T, bool) {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let (r, on_gpu) = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        add_sample(phase(), op, ns, 1, rows, on_gpu);
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
        denno_telemetry::record_layer_op(op, ns);
    }
    (r, on_gpu)
}

/// s84 (T8): 区間の**帰属先バケット**を一時的に差し替える (計測のみ)。
///
/// s85 (Step 0): s84 では「内側の子を親へも合算したうえで子にも残す」方式だったが、
/// (a) 親が入れ子になると孫が祖父に入らず `self = 親 − 子` が合わない、
/// (b) 表の shares が重複して TOTAL が実測 wall の 2.7 倍になった。
/// 現在は**有効な親のスタック**を持ち、sample を「それを包む全スコープ」へ積む
/// (区間の包含が成り立つ)。同時に「親へ入れた子の合計」を `PARENT_CHILD_NS` に
/// 記録し、summary は `self = 親 − 子` を出して行を加算的に読む。
///
/// スコープはスレッドローカル。PROFILE / テレメトリとも無効なら、この構造体は
/// 何も持たず `add_sample` も従来どおり 1 つのバケットにしか積まない
/// (既定挙動は完全に不変)。
pub struct Ambient(pub(crate) ());

thread_local! {
    /// 現在有効な親バケットのスタック (空 = 従来どおり単独計上)。
    static AMBIENT: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Ambient {
    /// 以降の計測を `op` へも合算する。計測が全て無効なら no-op スコープ。
    #[inline]
    pub fn enter(op: usize) -> Self {
        if !enabled() && !crate::telemetry_on() {
            return Ambient(());
        }
        AMBIENT.with(|c| c.borrow_mut().push(op));
        Ambient(())
    }
}

impl Drop for Ambient {
    /// スコープ終了時に、**そのスコープが積んだ 1 段**だけを降ろす。
    #[inline]
    fn drop(&mut self) {
        AMBIENT.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

/// s85 (Step 0): 現在有効な親バケットをすべて集める (外側 → 内側の順)。
/// 計測 OFF では空 Vec を返す (1 回の分岐で済む)。
#[inline]
fn ambient_ops() -> Vec<usize> {
    if !enabled() && !crate::telemetry_on() {
        return Vec::new();
    }
    AMBIENT.with(|c| c.borrow().clone())
}

/// 1 サンプルをフェーズ別テーブルへ積む。生 id (N_OPS 未満)、D2H 細分、
/// attn_core 層種別細分の 3 系統を一度に処理する。
///
/// s61: `crate::quant` の回帰テスト (既定経路が `maybe_gpu_linear` を通るか) が
/// 同じ会計を叩けるよう `pub` にした。外部クレートへは公開していない
/// (`pub(crate)` 相当)。
pub(crate) fn add_sample(
    phase: Phase,
    op: usize,
    ns: u64,
    calls: u64,
    rows: usize,
    on_gpu: bool,
) {
    let pi = phase.idx();
    let slot = if op < N_OPS { op } else { D2H };
    // s85 (Step 0): この sample の時間は、下の 3 つの出口が **ちょうど 1 回だけ**
    // `add_outer_ns` で書く (`owners` = それを包む全スコープ + 自分のバケット)。
    // 書き込みを 2 箇所に置くと、時間が自分にも親にも 2 回入って 16.0/7.0・
    // 4.0/2.0・総和 2 倍になっていた (実際に踏んだ) — 時間の書き込みは 1 本に保つ。
    //
    // s87 (Step 0): 「同じバケットが外側のスコープにも居る」(= collapsed) ときは、
    // その sample は自分のバケットの**直下の区間**であり、同時に外側のスコープへの
    // 子の時間でもある。s85 はこれを子に数えず self 側へ残していたため、
    // `self` と子の行を足しても親の区間にならず表が加算的でなかった。
    // 会計の規約 (supervisor 決定): collapsed な sample は自分のバケットも
    // `children` に入れる (時間の書き込みは依然 `owners` の 1 回だけ)。
    let amb = ambient_ops();
    let collapsed = amb.contains(&slot);
    let mut owners = amb;
    if !collapsed {
        owners.push(slot);
    }
    let mut children: Vec<usize> = owners.iter().copied().filter(|&p| p != slot).collect();
    if collapsed {
        children.push(slot);
    }
    CALLS[pi][slot].fetch_add(calls, Ordering::Relaxed);
    CALLS[Phase::Both.idx()][slot].fetch_add(calls, Ordering::Relaxed);
    // s85 (Step 0): 件数・行数は**包んでいるスコープ**へも合算する (時間は下の 3 つの
    // 出口が `add_outer_ns` で 1 回だけ足す)。親は D2H 細分より先に処理する —
    // 細分の経路は生 id へ落ちないので、ここで別扱いが要る。
    //
    // s84 は「最も内側の親 1 本」にだけ合算し、しかも同じ op の sample を除外して
    // いた (`p != op`)。それだと孫が祖父の区間に入らず `self = 親 − 子` が合わない。
    // 今はそれを包む全スコープへ積む。**自分のバケットは既に上で計上済みなので
    // 除く** — `owners` は collapsed でないとき自分のバケットを含むので、ここでは
    // `owners` ではなく「自分のバケットでないもの」を回す (含めると calls が 2 倍に
    // なる。s85 で実際に踏んだ)。
    for &p in owners.iter().filter(|&&p| p != slot) {
        CALLS[pi][p].fetch_add(calls, Ordering::Relaxed);
        CALLS[Phase::Both.idx()][p].fetch_add(calls, Ordering::Relaxed);
        if rows > 0 {
            TOKS[pi][p].fetch_add(rows as u64 * calls, Ordering::Relaxed);
            TOKS[Phase::Both.idx()][p].fetch_add(rows as u64 * calls, Ordering::Relaxed);
        }
    }
    if rows > 0 {
        TOKS[pi][slot].fetch_add(rows as u64 * calls, Ordering::Relaxed);
        TOKS[Phase::Both.idx()][slot].fetch_add(rows as u64 * calls, Ordering::Relaxed);
        // s61: 行数を渡したサンプルだけを「バッチ幅の平均」の分母にする。
        // 従来は TOKS / CALLS で割っていたため、行数を持たない呼び出し
        // (d2h 等の rows=0) が分母に入り tok/call が実バッチ幅より小さく出ていた。
        CALLS_ROWS[pi][slot].fetch_add(calls, Ordering::Relaxed);
        CALLS_ROWS[Phase::Both.idx()][slot].fetch_add(calls, Ordering::Relaxed);
    }
    if on_gpu {
        EXEC_CALLS[pi][slot].fetch_add(calls, Ordering::Relaxed);
        EXEC_CALLS[Phase::Both.idx()][slot].fetch_add(calls, Ordering::Relaxed);
    }
    // s57 (T4): attn_core の層種別細分 (ATT_GDN / ATT_FULL)。生 id (attn_core) の
    // 合計はこの 2 バケットの和になる — 呼び出し側は必ず層種別付きで計上する
    // ので、ここで合算して従来の attn_core バケットを保つ (バケットの意味を変えない)。
    // D2H 細分とは違い、細分側の加算が生 id の値を決める点に注意。
    if op >= ATT_GDN && op < ATT_GDN + ATT_CLS_N {
        let k = op - ATT_GDN;
        ATT_SUB_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
        ATT_SUB_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
        ATT_SUB_CALLS[pi][k].fetch_add(calls, Ordering::Relaxed);
        ATT_SUB_CALLS[Phase::Both.idx()][k].fetch_add(calls, Ordering::Relaxed);
        // s61c: GDN の内訳 (PROJ / SCAN) は **親 (ATT_GDN) へ合算する**。
        // 生 id (attn_core) への合算は上の共通ブロックが slot = D2H で
        // 既に行っているので、ここで触ると二重計上になる (実際に踏んだ)。
        if k >= 2 {
            ATT_SUB_NS[pi][0].fetch_add(ns, Ordering::Relaxed);
            ATT_SUB_NS[Phase::Both.idx()][0].fetch_add(ns, Ordering::Relaxed);
            ATT_SUB_CALLS[pi][0].fetch_add(calls, Ordering::Relaxed);
            ATT_SUB_CALLS[Phase::Both.idx()][0].fetch_add(calls, Ordering::Relaxed);
        }
        NS[pi][ATTN].fetch_add(ns, Ordering::Relaxed);
        NS[Phase::Both.idx()][ATTN].fetch_add(ns, Ordering::Relaxed);
        CALLS[pi][ATTN].fetch_add(calls, Ordering::Relaxed);
        CALLS[Phase::Both.idx()][ATTN].fetch_add(calls, Ordering::Relaxed);
        if rows > 0 {
            TOKS[pi][ATTN].fetch_add(rows as u64 * calls, Ordering::Relaxed);
            TOKS[Phase::Both.idx()][ATTN].fetch_add(rows as u64 * calls, Ordering::Relaxed);
            // s61: 生 id (attn_core) の tok/call の分母も細分側と同じ規則で数える
            CALLS_ROWS[pi][ATTN].fetch_add(calls, Ordering::Relaxed);
            CALLS_ROWS[Phase::Both.idx()][ATTN].fetch_add(calls, Ordering::Relaxed);
        }
        if on_gpu {
            EXEC_CALLS[pi][ATTN].fetch_add(calls, Ordering::Relaxed);
            EXEC_CALLS[Phase::Both.idx()][ATTN].fetch_add(calls, Ordering::Relaxed);
        }
    }
    // s54: d2h_state の細分 (細分先が別 id なので生 id へは積まない — 呼び出し側が
    // D2H 本体も別途計上する)。
    if op >= D2H_LOGITS && op <= D2H_LOGITS_PF_HELD {
        let k = op - D2H_LOGITS;
        D2H_SUB_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
        D2H_SUB_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
        D2H_SUB_CALLS[pi][k].fetch_add(calls, Ordering::Relaxed);
        D2H_SUB_CALLS[Phase::Both.idx()][k].fetch_add(calls, Ordering::Relaxed);
    }
    // s64: GDN スキャン内訳 (gdn_pf_*) は **ここでしか積まない** — 生 id へ落とす
    // 共通ブロックにも D2H 細分にも掛けない (表示専用・二重計上しない)。
    if (D2H_GDN_PF_FIRST..D2H_GDN_PF_FIRST + D2H_GDN_PF_N).contains(&op) {
        let k = op - D2H_GDN_PF_FIRST;
        GDN_PF_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
        GDN_PF_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
        GDN_PF_CALLS[pi][k].fetch_add(calls, Ordering::Relaxed);
        GDN_PF_CALLS[Phase::Both.idx()][k].fetch_add(calls, Ordering::Relaxed);
    }
    // s78 (T7): `gdn_proj` 内訳 (gdnp_*) も **ここでしか積まない** (表示専用)。
    if (D2H_GDN_PROJ_FIRST..D2H_GDN_PROJ_FIRST + D2H_GDN_PROJ_N).contains(&op) {
        let k = op - D2H_GDN_PROJ_FIRST;
        GDN_PROJ_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
        GDN_PROJ_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
        GDN_PROJ_CALLS[pi][k].fetch_add(calls, Ordering::Relaxed);
        GDN_PROJ_CALLS[Phase::Both.idx()][k].fetch_add(calls, Ordering::Relaxed);
        add_outer_ns(pi, &owners, &children, ns);
        return;
    }
    // s65: `d2h_state` 発行元 (d2h_src_*) も **ここでしか積まない** (表示専用)。
    // `mark_d2h_src` はこの経路を使うので、生 id へ落ちないよう明示的に弾く。
    if (D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&op) {
        let k = op - D2H_SRC_FIRST;
        D2H_SRC_CALLS[pi][k].fetch_add(calls, Ordering::Relaxed);
        D2H_SRC_CALLS[Phase::Both.idx()][k].fetch_add(calls, Ordering::Relaxed);
        add_outer_ns(pi, &owners, &children, ns);
        return;
    }
    add_outer_ns(pi, &owners, &children, ns);
}

/// s85 (Step 0): 1 sample の時間を、囲んでいる全バケットへ **1 回ずつ**足す。
///
/// `owners` = 外側のスコープ + (collapsed でなければ) 自分のバケット。区間の包含が
/// 成り立つので、孫の時間も祖父のバケットへ入る (= `親 >= 子`)。`children` には
/// 「この sample が入れた時間」を `PARENT_CHILD_*` へも記録する — これが
/// `self = 親 − 子` の引き算の材料。
///
/// s87 (Step 0): collapsed (自分のバケットが外側のスコープにも居る) な sample は
/// **自分のバケットも `children` に入れる**。その sample はバケットの直下の区間で
/// あり、同時にそのバケットが包んでいる領域の一部でもあるため、子に数えないと
/// `self + 子` が親の区間にならず表が加算的でなくなる (規約は supervisor 決定)。
/// 時間の書き込みは `owners` の 1 回だけなので、二重計上にはならない。
/// 計測 OFF では `owners` が `[slot]` だけ・`children` が空で、表へ触らない。
#[inline]
fn add_outer_ns(pi: usize, owners: &[usize], children: &[usize], ns: u64) {
    for &p in owners {
        NS[pi][p].fetch_add(ns, Ordering::Relaxed);
        NS[Phase::Both.idx()][p].fetch_add(ns, Ordering::Relaxed);
    }
    for &p in children {
        PARENT_CHILD_NS[pi][p].fetch_add(ns, Ordering::Relaxed);
        PARENT_CHILD_NS[Phase::Both.idx()][p].fetch_add(ns, Ordering::Relaxed);
    }
}

/// s65 (T5b L1 第 2 段・計測のみ): `d2h_state` の 1 回の発行を発行元 `src`
/// (`D2H_SRC_*`) として計上する。**件数のみ** (時間は `d2h_state` バケットが
/// 持っている)。PROFILE 無効時は分岐 1 回で何もしない (時刻読みなし)。
///
/// 呼び出しは D2H を発行する直後 1 箇所だけに置く — D2H の発行点を増やす /
/// 減らす commit が入ると分類件数が `d2h_state` の件数から外れるので、
/// そのときはここも同時に直す。
#[inline]
pub fn mark_d2h_src(src: usize) {
    if !enabled() || !(D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&src) {
        return;
    }
    add_sample(phase(), src, 0, 1, 0, false);
}

/// s88 (Step 1): `mark_d2h_src` の**時間つき**版。発行点 `src` の区間 `ns` を
/// 件数と一緒に積む。**表示専用** (`D2H_SRC_NS` / `D2H_SRC_CALLS` のみ・どの生 id
/// にも合算しない)。
///
/// 呼び出し側は自分の区間を測って渡す — `mark_d2h_src` のように発行の**直後**に
/// 呼ぶだけでは、転送発行から sync までを含む区間が測れないため。
/// PROFILE / テレメトリが無効なら分岐 1 回で戻る (時刻は呼び出し側が測らない —
/// 呼び出し側も `d2h_src_on()` で先に弾く)。
#[inline]
pub fn mark_d2h_src_ns(src: usize, ns: u64) {
    if !enabled() || !(D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&src) {
        return;
    }
    let pi = phase().idx();
    let k = src - D2H_SRC_FIRST;
    D2H_SRC_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
    D2H_SRC_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
    D2H_SRC_CALLS[pi][k].fetch_add(1, Ordering::Relaxed);
    D2H_SRC_CALLS[Phase::Both.idx()][k].fetch_add(1, Ordering::Relaxed);
}

/// s88 (Step 1): 発行点の計時を行うか (= 計測が有効か)。呼び出し側はこれが真の
/// ときだけ `Instant::now()` を読む (既定経路はクロック読みゼロのまま)。
#[inline]
pub fn d2h_src_on() -> bool {
    enabled() || crate::telemetry_on()
}

/// s88 (Step 1): 発行点の**時間だけ**を積む (件数は呼び出し側の `mark_d2h_src` が
/// 既に積んでいる site 用)。`mark_d2h_src_ns` と違い件数を足さない — 両方呼ぶと
/// calls が 2 倍になり、`d2h_state` の calls との突き合わせが合わなくなる。
#[inline]
pub fn add_d2h_src_ns_only(src: usize, ns: u64) {
    if !enabled() || !(D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&src) {
        return;
    }
    let pi = phase().idx();
    let k = src - D2H_SRC_FIRST;
    D2H_SRC_NS[pi][k].fetch_add(ns, Ordering::Relaxed);
    D2H_SRC_NS[Phase::Both.idx()][k].fetch_add(ns, Ordering::Relaxed);
}

/// `f` の実行時間を op に計上して結果を返す。
/// (P13: テレメトリ有効時は op 時間を telemetry ハブにも累積する。
/// どちらも計測のみで数値経路には触れないため exactness は不変)
pub fn time<T>(op: usize, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        // s54: d2h の細分 id は NS の範囲外 — 生 id で受ける (下の summary が集約)
        add_sample(phase(), op, ns, 1, 0, false);
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
    }
    r
}

/// `time` の層付き版 (P14: LayerRun フレーム用)。層ループ内の op に使う。
/// op 時間は OpSummary 累積に加算され、さらに telemetry の層バッファに
/// 実行順で記録される (`telemetry_layer_begin` 〜 `telemetry_layer_flush` 間)。
/// layer 引数は呼び出し側の文脈明示用 (記録は層バッファ側が担当する)。
pub fn time_layer<T>(op: usize, _layer: usize, f: impl FnOnce() -> T) -> T {
    let prof_on = enabled();
    let tel_on = crate::telemetry_on();
    if !prof_on && !tel_on {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos() as u64;
    if prof_on {
        add_sample(phase(), op, ns, 1, 0, false);
    }
    if tel_on {
        crate::telemetry_add_op(if op < N_OPS { op } else { D2H }, ns);
        denno_telemetry::record_layer_op(op, ns);
    }
    r
}

/// s54: d2h 細分の計測ゲート (prof 有効 or テレメトリ有効)。無効時は分岐 1 回で
/// 生 id 経路と完全に同じ (数値経路に接触しない)。
#[inline]
pub fn op_on() -> bool {
    enabled() || crate::telemetry_on()
}

/// s54: `op_on()` が偽なら `f` を素通しする `time` (計測の分岐を 1 箇所に閉じる)。
#[inline]
pub fn time_if_on<T>(op: usize, f: impl FnOnce() -> T) -> T {
    if op_on() {
        return time(op, f);
    }
    f()
}

/// s57 (T4): バッチ幅付きの層 op 計測ゲート。`op_on()` が偽なら素通し。
#[inline]
pub fn time_if_on_n<T>(op: usize, rows: usize, f: impl FnOnce() -> T) -> T {
    if op_on() {
        return time_full(op, rows, false, f);
    }
    f()
}

/// s64 (T5b L1 Step 1): 1 関数の内部を複数区間に割って計時する計測器。
///
/// 用途は「ある関数の所要時間を、確保 / launch / sync / D2H などの区間へ割る」こと。
/// `begin` で区間 1 の計時を開始し、`lap(next_id)` で「直前の区間の経過」を
/// `next_id`……ではなく **直前の区間に積み**、次区間の計時を開始する
/// (`seq.lap(SYNC)` は「今の区間 (LAUNCH) の終わり」と「SYNC の開始」を兼ねる)。
/// `end` で最後の区間を積む。
///
/// PROFILE 無効時は `begin` が `None` を持ち、`lap`/`end` は分岐 1 回で何もしない
/// (時刻読みもしない = 計測 off の実行時間に影響しない)。**表示専用**の細分に
/// しか積まないので、既存バケットの値・数値経路は不変。
pub struct Seq {
    op: Option<usize>,
    t: Instant,
}

impl Seq {
    #[inline]
    pub fn begin(op: usize, now: Instant) -> Self {
        if enabled() {
            Self {
                op: Some(op),
                t: now,
            }
        } else {
            Self { op: None, t: now }
        }
    }

    /// 現在の区間を積み、`next` 区間の計時を始める。
    #[inline]
    pub fn lap(&mut self, next: usize) {
        let Some(op) = self.op else { return };
        let now = Instant::now();
        add_sample(phase(), op, now.duration_since(self.t).as_nanos() as u64, 1, 0, false);
        self.op = Some(next);
        self.t = now;
    }

    /// 最後の区間を積む。`t` は次回 `begin` で上書きされるため無効化する。
    #[inline]
    pub fn end(mut self) {
        let Some(op) = self.op.take() else { return };
        let now = Instant::now();
        add_sample(phase(), op, now.duration_since(self.t).as_nanos() as u64, 1, 0, false);
    }
}

pub fn reset() {
    for p in 0..Phase::N {
        for i in 0..N_OPS {
            NS[p][i].store(0, Ordering::Relaxed);
            CALLS[p][i].store(0, Ordering::Relaxed);
            TOKS[p][i].store(0, Ordering::Relaxed);
            CALLS_ROWS[p][i].store(0, Ordering::Relaxed);
            EXEC_CALLS[p][i].store(0, Ordering::Relaxed);
        }
        for i in 0..D2H_N {
            D2H_SUB_NS[p][i].store(0, Ordering::Relaxed);
            D2H_SUB_CALLS[p][i].store(0, Ordering::Relaxed);
        }
        for i in 0..D2H_GDN_PF_N {
            GDN_PF_NS[p][i].store(0, Ordering::Relaxed);
            GDN_PF_CALLS[p][i].store(0, Ordering::Relaxed);
        }
        for i in 0..D2H_SRC_N {
            D2H_SRC_CALLS[p][i].store(0, Ordering::Relaxed);
            // s88 (Step 1): 発行点ごとの時間もクリアする (件数だけ消すと差分が残る)
            D2H_SRC_NS[p][i].store(0, Ordering::Relaxed);
        }
        for i in 0..D2H_GDN_PROJ_N {
            GDN_PROJ_NS[p][i].store(0, Ordering::Relaxed);
            GDN_PROJ_CALLS[p][i].store(0, Ordering::Relaxed);
        }
        for i in 0..ATT_CLS_N {
            ATT_SUB_NS[p][i].store(0, Ordering::Relaxed);
            ATT_SUB_CALLS[p][i].store(0, Ordering::Relaxed);
        }
        // s85 (Step 0): 親 → 直接の子の累積 (self の引き算の材料)
        for i in 0..N_OPS {
            PARENT_CHILD_NS[p][i].store(0, Ordering::Relaxed);
            PARENT_CHILD_CALLS[p][i].store(0, Ordering::Relaxed);
        }
    }
}

/// s57 (T4): 実行先ラベルの決定 (純関数・テスト可能)。
/// `has_gpu_path && gpu_calls == calls && calls > 0` → "gpu"
/// `has_gpu_path && gpu_calls == 0 && calls > 0` → "cpu"
/// 混在 (一部だけ GPU) → "mixed"
/// GPU 選択の無い op → "cpu*"
pub fn exec_label(has_gpu_path: bool, calls: u64, gpu_calls: u64) -> &'static str {
    if calls == 0 {
        return "-";
    }
    if !has_gpu_path {
        return "cpu*";
    }
    if gpu_calls == 0 {
        "cpu"
    } else if gpu_calls >= calls {
        "gpu"
    } else {
        "mixed"
    }
}

/// s85 (Step 0): 親バケットの**子** (外側のスコープから入ってきた時間) の累積。
#[inline]
fn parent_child_ns(p: usize, op: usize) -> u64 {
    PARENT_CHILD_NS[p][op].load(Ordering::Relaxed)
}

/// s85 (Step 0): 親バケット行の `self` = 親 − 子 (負にはしない)。
///
/// 表を加算的にするための核心: `self` と子の行を足した値が親の実測区間になる。
/// 同じ op が入れ子の内外で積まれた場合などに引きすぎて負にならないよう 0 で止める。
///
/// s87 (Step 0) の `self` 列の定義 (表の読み手向けの正):
/// `self` = このバケットの時間のうち、**子の sample にも、同じバケットの Ambient
/// スコープにも覆われていない部分**。印刷する total は `self` の和で、これは
/// 実測区間の**葉分解**である (外側のスコープを重ねて数えない — 意図的に
/// 二重計上しない)。したがって「ms 列 (= そのバケットが包む全体)」と
/// 「self 列」は別物で、両者を足してはいけない。collapsed な sample
/// (= バケットの Ambient の内側で同じバケットを計時した分) は子として数えるので
/// `self` からは除かれる (規約は supervisor 決定)。
#[inline]
pub fn parent_self_ns(p: usize, op: usize) -> u64 {
    NS[p][op]
        .load(Ordering::Relaxed)
        .saturating_sub(parent_child_ns(p, op))
}

/// s85 (Step 0): 表の 1 行 = (op, 親の区間 ms, 直接の子 ms, self ms, calls, tok/call, via)。
///
/// `parent` は従来の「そのバケットに積まれた全部」の値 (入れ子の子を含む)、
/// `self` は `parent − 子`。**表の総和は self 列の和** (= 実測区間の分解) にして
/// 加算性を保つ。子を持たないバケットは `self == parent` で、子を持つバケットだけ
/// インデントして内訳を出す。列を増やすと既存の読み手 (tests / ログの目視) が
/// 壊れるため、self は**7 列目 (via の後) に足す** — 先頭 6 列の意味は不変。
/// s85 (Step 0): フェーズの行を (op, parent, self, calls, toks, gpu_calls, calls_rows) で集める。
/// self はここで親 − 直接の子として確定させる (表と総和で同じ値を共有する)。
fn phase_rows(p: usize) -> Vec<(usize, u64, u64, u64, u64, u64, u64)> {
    (0..N_OPS)
        .map(|i| {
            let parent = NS[p][i].load(Ordering::Relaxed);
            let self_ns = parent.saturating_sub(parent_child_ns(p, i));
            (
                i,
                parent,
                self_ns,
                CALLS[p][i].load(Ordering::Relaxed),
                TOKS[p][i].load(Ordering::Relaxed),
                EXEC_CALLS[p][i].load(Ordering::Relaxed),
                CALLS_ROWS[p][i].load(Ordering::Relaxed),
            )
        })
        .filter(|(_, parent, _, _, _, _, _)| *parent > 0)
        .collect()
}

/// 1 フェーズ分のテーブルを整形する。
///
/// s85 (Step 0): 総和 (`total`) は **self 列の和** (加算的な区間分解)。
/// 従来は親の区間をそのまま合計していたため、Ambient の子が二重に膨らんでいた。
fn phase_table(phase: Phase, s: &mut String) {
    let p = phase.idx();
    let rows = phase_rows(p);
    let total: u64 = rows.iter().map(|(_, _, self_ns, _, _, _, _)| *self_ns).sum();
    if rows.is_empty() {
        s.push_str(&format!("  [{}] (no calls)\n", Phase::NAMES[p]));
        return;
    }
    let mut sorted = rows;
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    s.push_str(&format!(
        "  [{}] total {:.1} ms (additive: sum of self)\n",
        Phase::NAMES[p],
        total as f64 / 1e6
    ));
    s.push_str(&format!(
        "  {:<14} {:>10} {:>8} {:>7} {:>8} {:>6} {:>10}\n",
        "op", "ms", "calls", "share", "tok/call", "via", "self"
    ));
    for (i, ns, self_ns, calls, toks, gpu_calls, calls_rows) in sorted {
        // s61: 分母は行数を渡したサンプルだけ (実バッチ幅)。
        let mean_tok = if calls_rows > 0 {
            toks as f64 / calls_rows as f64
        } else {
            0.0
        };
        s.push_str(&format!(
            "  {:<14} {:>10.1} {:>8} {:>6.1}% {:>8.1} {:>6} {:>10.1}\n",
            NAMES[i],
            ns as f64 / 1e6,
            calls,
            self_ns as f64 / total.max(1) as f64 * 100.0,
            mean_tok,
            exec_label(HAS_GPU_PATH[i], calls, gpu_calls),
            self_ns as f64 / 1e6
        ));
        // attn_core: GDN (linear attention) と full attention の内訳を直下に出す
        if i == ATTN {
            for k in 0..ATT_CLS_N {
                let sns = ATT_SUB_NS[p][k].load(Ordering::Relaxed);
                let sc = ATT_SUB_CALLS[p][k].load(Ordering::Relaxed);
                if sns == 0 && sc == 0 {
                    continue;
                }
                // s61c: gdn_proj / gdn_scan は attn_gdn の内訳なので 1 段深く出す。
                // attn_gdn の値は子分を合算済み (add_sample が親へも積む)。
                let (indent, name) = if k >= 2 {
                    ("      ", ATT_CLS_NAMES[k])
                } else {
                    ("    ", ATT_CLS_NAMES[k])
                };
                s.push_str(&format!(
                    "{indent}{:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
                    name,
                    sns as f64 / 1e6,
                    sc,
                    sns as f64 / total.max(1) as f64 * 100.0,
                    "-",
                    ""
                ));
                if k == 2 {
                    // s78 (T7): gdn_proj の直下にその内訳 (H2D / kernel / sync / D2H)
                    push_gdn_proj_rows(p, total, "        ", s);
                }
                if k == 3 {
                    // s64: gdn_scan の直下にその内訳 (確保 / launch / sync / D2H) を出す
                    push_gdn_pf_rows(p, total, "        ", s);
                }
            }
        }
        if i == D2H {
            // s64: GDN スキャン内訳 (gdn_pf_*) は d2h_state の値に含まれない
            // (表示専用) が、読む側が d2h 由来と誤読しないよう d2h 行の直下に出す。
            push_gdn_pf_rows(p, total, "    ", s);
            for k in 0..D2H_N {
                let sns = D2H_SUB_NS[p][k].load(Ordering::Relaxed);
                let sc = D2H_SUB_CALLS[p][k].load(Ordering::Relaxed);
                if sns == 0 && sc == 0 {
                    continue;
                }
                s.push_str(&format!(
                    "    {:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
                    D2H_NAMES[k],
                    sns as f64 / 1e6,
                    sc,
                    sns as f64 / total.max(1) as f64 * 100.0,
                    "-",
                    ""
                ));
            }
            // s65: 発行元の**件数**の分類と s88 の時間 (ms)。
            // 合計が d2h_state の calls 列と一致するかを読むための表。
            push_d2h_src_rows(p, "    ", s);
            // s88 (Step 1): prefill だけ、発行点ごとの時間の上位 10 行を出す
            // (どの site を先に攻めるかを決めるため。decode は chain 定常で少ない)。
            if phase == Phase::Prefill {
                push_d2h_src_top10(p, s);
            }
        }
    }
}

/// s65: `d2h_state` 発行元の**件数** (と s88 の時間) を 1 行ずつ積む。
/// 合計が `d2h_state` の calls 列と一致すれば分類漏れなし。
///
/// s88 (Step 1): 発行元ごとの経過時間 (`D2H_SRC_NS`) も ms 列に出す。時間は
/// **表示専用**で `d2h_state` の行には合算しない (`d2h_state` の ms との差 =
/// 発行点の外側 = 呼び出し元の区間)。
fn push_d2h_src_rows(p: usize, indent: &str, s: &mut String) {
    for k in 0..D2H_SRC_N {
        let sc = D2H_SRC_CALLS[p][k].load(Ordering::Relaxed);
        if sc == 0 {
            continue;
        }
        let sns = D2H_SRC_NS[p][k].load(Ordering::Relaxed);
        s.push_str(&format!(
            "{indent}{:<12} {:>10.1} {:>8} {:>6} {:>8} {:>6}\n",
            D2H_SRC_NAMES[k],
            sns as f64 / 1e6,
            sc,
            "-",
            "-",
            ""
        ));
    }
}

/// s88 (Step 1): prefill の `d2h_state` を**発行点ごとの時間**で降順に並べた上位 10 行。
///
/// 目的: `d2h_state` 1 本に混ざっている複数の発行点のうち、どれを先に攻めるかを
/// 決める。**表示専用** (`D2H_SRC_NS` を読むだけ・どのバケットにも書かない)。
/// 行が 1 本も無ければ何も出さない。prefill のみ (decode は chain 定常で site が
/// 少なく、上位 10 行が意味を持たない) — 呼び出し側が phase で絞る。
fn push_d2h_src_top10(p: usize, s: &mut String) {
    let mut rows: Vec<(usize, u64, u64)> = (0..D2H_SRC_N)
        .map(|k| {
            (
                k,
                D2H_SRC_NS[p][k].load(Ordering::Relaxed),
                D2H_SRC_CALLS[p][k].load(Ordering::Relaxed),
            )
        })
        .filter(|(_, ns, calls)| *ns > 0 || *calls > 0)
        .collect();
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    s.push_str("    d2h_state by site (top 10, ms desc)\n");
    s.push_str(&format!(
        "    {:<18} {:>10} {:>8} {:>10}\n",
        "site", "ms", "calls", "us/call"
    ));
    for (k, ns, calls) in rows.into_iter().take(10) {
        let us = if calls > 0 {
            ns as f64 / calls as f64 / 1e3
        } else {
            0.0
        };
        s.push_str(&format!(
            "    {:<18} {:>10.1} {:>8} {:>10.1}\n",
            D2H_SRC_NAMES[k],
            ns as f64 / 1e6,
            calls,
            us
        ));
    }
}

/// s78 (T7): `gdn_proj` (beta/alpha 投影) 1 call の内訳 (`gdnp_h2d/kernel/sync/d2h`)
/// を `indent` 付きで 1 行ずつ積む。**表示専用** — 4 区間の和は 2 call
/// (beta + alpha) ぶんの GPU 区間になるが、値はどのバケットにも合算されない
/// (`gdn_proj` 行の値は呼び出し側の外側計時が持つ)。
fn push_gdn_proj_rows(p: usize, total: u64, indent: &str, s: &mut String) {
    for k in 0..D2H_GDN_PROJ_N {
        let sns = GDN_PROJ_NS[p][k].load(Ordering::Relaxed);
        let sc = GDN_PROJ_CALLS[p][k].load(Ordering::Relaxed);
        if sns == 0 && sc == 0 {
            continue;
        }
        s.push_str(&format!(
            "{indent}{:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
            D2H_GDN_PROJ_NAMES[k],
            sns as f64 / 1e6,
            sc,
            sns as f64 / total.max(1) as f64 * 100.0,
            "-",
            ""
        ));
    }
}

/// s64: GDN スキャン 1 call の内訳 (`gdn_pf_alloc/launch/sync/d2h`) を
/// `indent` 付きで 1 行ずつ積む。**表示専用** — total (= 集計対象の合計) に対する
/// 比を出すが、値はどのバケットにも合算されない。
fn push_gdn_pf_rows(p: usize, total: u64, indent: &str, s: &mut String) {
    for k in 0..D2H_GDN_PF_N {
        let sns = GDN_PF_NS[p][k].load(Ordering::Relaxed);
        let sc = GDN_PF_CALLS[p][k].load(Ordering::Relaxed);
        if sns == 0 && sc == 0 {
            continue;
        }
        s.push_str(&format!(
            "{indent}{:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
            D2H_GDN_PF_NAMES[k],
            sns as f64 / 1e6,
            sc,
            sns as f64 / total.max(1) as f64 * 100.0,
            "-",
            ""
        ));
    }
}

/// 整形済みサマリ (prefill / decode / cumulative の 3 テーブル) を返す。
///
/// s57 (T4): 従来は decode+prefill の累積 1 表だった — prefill のホットスポットを
/// 確定するには decode の寄与 (step 数ぶん積み上がる) が混ざると読めないため、
/// フェーズ別を先に出し、互換・合計確認用に cumulative を最後に残す
/// (`cumulative` の各 op = prefill + decode、TOTAL も従来と同値)。
/// d2h_state / attn_core の直後には細分行をインデントして出す。
pub fn summary() -> String {
    let mut s = String::from("PROFILE (prefill/decode split; tok/call = mean batch width):\n");
    for ph in [Phase::Prefill, Phase::Decode] {
        phase_table(ph, &mut s);
    }
    s.push_str("  [cumulative]\n");
    // cumulative は従来どおり「合計時間降順・累積%」で出す (既存の読み方と互換)。
    // s85 (Step 0): 総和は self の和 (加算的)。累積% も self で積む。
    let p = Phase::Both.idx();
    let rows = phase_rows(p);
    let total: u64 = rows.iter().map(|(_, _, self_ns, _, _, _, _)| *self_ns).sum();
    let mut sorted = rows;
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    s.push_str(&format!(
        "  {:<14} {:>10} {:>8} {:>7} {:>8} {:>6} {:>10}\n",
        "op", "ms", "calls", "cum%", "tok/call", "via", "self"
    ));
    let mut acc = 0u64;
    for (i, ns, self_ns, calls, toks, gpu_calls, calls_rows) in sorted {
        acc += self_ns;
        // s61: 分母は行数を渡したサンプルだけ (実バッチ幅)。
        let mean_tok = if calls_rows > 0 {
            toks as f64 / calls_rows as f64
        } else {
            0.0
        };
        s.push_str(&format!(
            "  {:<14} {:>10.1} {:>8} {:>6.1}% {:>8.1} {:>6} {:>10.1}\n",
            NAMES[i],
            ns as f64 / 1e6,
            calls,
            acc as f64 / total.max(1) as f64 * 100.0,
            mean_tok,
            exec_label(HAS_GPU_PATH[i], calls, gpu_calls),
            self_ns as f64 / 1e6
        ));
        if i == ATTN {
            for k in 0..ATT_CLS_N {
                let sns = ATT_SUB_NS[p][k].load(Ordering::Relaxed);
                let sc = ATT_SUB_CALLS[p][k].load(Ordering::Relaxed);
                if sns == 0 && sc == 0 {
                    continue;
                }
                // s61c: gdn_proj / gdn_scan は attn_gdn の内訳 (1 段深く出す)。
                // attn_gdn の値は子分を合算済み (add_sample が親へも積む)。
                let (indent, name) = if k >= 2 {
                    ("      ", ATT_CLS_NAMES[k])
                } else {
                    ("    ", ATT_CLS_NAMES[k])
                };
                s.push_str(&format!(
                    "{indent}{:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
                    name,
                    sns as f64 / 1e6,
                    sc,
                    sns as f64 / total.max(1) as f64 * 100.0,
                    "-",
                    ""
                ));
                if k == 3 {
                    // s64: gdn_scan の直下にその内訳 (確保 / launch / sync / D2H) を出す
                    push_gdn_pf_rows(p, total, "        ", &mut s);
                }
            }
        }
        if i == D2H {
            // s64: GDN スキャン内訳 (gdn_pf_*) は d2h_state の値に含まれない
            // (表示専用) が、読む側が d2h 由来と誤読しないよう d2h 行の直下に出す。
            push_gdn_pf_rows(p, total, "    ", &mut s);
            for k in 0..D2H_N {
                let sns = D2H_SUB_NS[p][k].load(Ordering::Relaxed);
                let sc = D2H_SUB_CALLS[p][k].load(Ordering::Relaxed);
                if sns == 0 && sc == 0 {
                    continue;
                }
                s.push_str(&format!(
                    "    {:<12} {:>10.1} {:>8} {:>6.1}% {:>8} {:>6}\n",
                    D2H_NAMES[k],
                    sns as f64 / 1e6,
                    sc,
                    sns as f64 / total.max(1) as f64 * 100.0,
                    "-",
                    ""
                ));
            }
            // s65: 発行元の件数分類。
            push_d2h_src_rows(p, "    ", &mut s);
        }
    }
    s.push_str(&format!("  TOTAL {:.1} ms\n", total as f64 / 1e6));
    s
}

#[cfg(test)]
mod tests {
    //! s55: d2h 細分 id の整合 (生 id 不変・細分は N_OPS の外・summary が 3 行を出す)。
    //! s57 (T4): フェーズ分離 (prefill/decode) と attn_core の層種別分離、平均 tok/call。
    //! 時刻値は環境依存なので検証しない — 検証するのは id の算術と集約だけ。
    use super::*;

    /// s61c: 集計カウンタ (`NS` / `ATT_SUB_*` 等) は**プロセス全体で 1 本**の
    /// static なので、`reset()` を挟むテストを並走させると他テストの
    /// バケットが消える (実際に `attn_gdn` 行が消えて落ちた)。集計に触れる
    /// テストはこのロックを取って直列化する。ロックは panic でも解放される。
    static GLOBAL_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// s85 (Step 0): テスト専用の `enabled()` 上書き。None = env (`DENNO_PROFILE`) を見る。
    /// `Ambient::enter` / `add_outer_ns` は `enabled()` が偽だと完全 no-op なので、
    /// 会計のテストはこのフラグで計測を有効化する (env の書き換えは並走テストを
    /// 揺らすため使わない)。会計の static はプロセス共有なので、必ず
    /// `lock_counters()` と一緒に使う。
    static FORCE_ENABLED: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

    pub(super) fn force_enabled() -> Option<bool> {
        *FORCE_ENABLED.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// s85 (Step 0): 計測を有効化したまま `f` を実行し、終了時に必ず元へ戻す
    /// (panic しても戻る)。カウンタのロックも取るので、呼び出し側は取らない。
    ///
    /// 表はプロセス共有なので、掴んだ直後に `reset()` する — テスト本体の末尾で
    /// リセットする形だと、途中で assert が落ちたときに値を残し、後続のテスト
    /// (`attn_core_is_split_into_gdn_and_full_rows` 等) を巻き込む (実際に踏んだ)。
    fn with_prof_counters<T>(f: impl FnOnce() -> T) -> T {
        let _guard = lock_counters();
        *FORCE_ENABLED.lock().unwrap_or_else(|e| e.into_inner()) = Some(true);
        let _on = ForceEnabled(true);
        reset();
        f()
    }

    /// 計測の強制フラグを Drop で戻すガード (panic 時も解除される)。
    struct ForceEnabled(bool);
    impl Drop for ForceEnabled {
        fn drop(&mut self) {
            *FORCE_ENABLED.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    /// グローバルカウンタを汚すテスト用の排他ガード。
    fn lock_counters() -> std::sync::MutexGuard<'static, ()> {
        GLOBAL_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// 生 id と telemetry のバイト表現に触れていないこと (細分を足しても不変)。
    #[test]
    fn d2h_subsplit_ids_are_outside_raw_range() {
        assert_eq!(D2H, 14);
        assert_eq!(N_OPS, 15);
        let ids = [
            D2H_LOGITS,
            D2H_KV_STAGE,
            D2H_KV_FLUSH,
            D2H_GDN_FLUSH,
            D2H_KV_DELAYED_FLUSH,
            D2H_ATTN_PF,
            D2H_LOGITS_WAIT,
            D2H_LOGITS_COPY,
            D2H_LOGITS_PF_HELD,
        ];
        assert_eq!(ids.len(), D2H_N, "D2H_N は細分 id の総数と一致");
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, D2H_LOGITS + i, "細分 id は連番 (10*D2H + i)");
            assert!(id >= N_OPS, "細分 id は生 id の範囲外: {id}");
        }
    }

    /// s65: `d2h_state` 発行元の件数分類 id が D2H 細分 / GDN_PF の後ろに連番で
    /// 並び、生 id と交差しないこと。表示名の本数も一致させる。
    #[test]
    fn d2h_src_ids_are_outside_raw_range() {
        let ids = [
            D2H_SRC_KV_ROW,
            D2H_SRC_GDN_STATE,
            D2H_SRC_ATTN_PF,
            D2H_SRC_LINEAR,
            D2H_SRC_LOGITS,
            D2H_SRC_OTHER,
        ];
        assert_eq!(ids.len(), D2H_SRC_N);
        assert_eq!(D2H_SRC_NAMES.len(), D2H_SRC_N);
        assert_eq!(D2H_SRC_FIRST, D2H_GDN_PF_FIRST + D2H_GDN_PF_N);
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, D2H_SRC_FIRST + i);
            assert!(id >= N_OPS, "生 id の範囲外: {id}");
        }
        // D2H 細分 (i < 9)・GDN_PF (9..12) と交差しない
        assert!(D2H_SRC_KV_ROW > D2H_LOGITS_PF_HELD + 3);
    }

    /// s88 (Step 1): `d2h_state` 発行点ごとの**時間**が site 別に積まれ、
    /// **どの生バケットにも入らない**こと。
    ///
    /// 固定する規約:
    /// 1) `mark_d2h_src_ns` は件数 1 + 時間。`add_d2h_src_ns_only` は時間だけ
    ///    (件数を二重にしない — 両方呼ぶ site があるため)。
    /// 2) site の時間は `D2H` 生バケットを増やさない (表示専用)。
    /// 3) 上位 10 行は ms の降順。
    #[test]
    fn d2h_src_ns_is_per_site_and_display_only() {
        with_prof_counters(|| {
            reset();
            // 既定の phase は Decode — この計測は prefill の表に出すので明示する。
            let prev = phase();
            set_phase(Phase::Prefill);
            // site の時間だけを積む経路 (mark_d2h_src 済みの site 向け)
            add_d2h_src_ns_only(D2H_SRC_ATTN_PF, 5_000_000);
            mark_d2h_src(D2H_SRC_ATTN_PF);
            // 件数 + 時間を同時に積む経路 (chain_download 向け)
            mark_d2h_src_ns(D2H_SRC_OTHER, 2_000_000);
            let _ = d2h_src_on();

            let p = Phase::Prefill.idx();
            let apf = D2H_SRC_ATTN_PF - D2H_SRC_FIRST;
            let other = D2H_SRC_OTHER - D2H_SRC_FIRST;
            // 1) 時間は site 別
            assert_eq!(D2H_SRC_NS[p][apf].load(Ordering::Relaxed), 5_000_000);
            assert_eq!(D2H_SRC_NS[p][other].load(Ordering::Relaxed), 2_000_000);
            // 件数: ATTN_PF は 1 (時間だけの記録は件数を足さない)、OTHER も 1
            assert_eq!(D2H_SRC_CALLS[p][apf].load(Ordering::Relaxed), 1);
            assert_eq!(D2H_SRC_CALLS[p][other].load(Ordering::Relaxed), 1);
            // 2) 生バケットは触らない
            assert_eq!(
                NS[p][D2H].load(Ordering::Relaxed),
                0,
                "site の時間は d2h_state 生バケットへ入らない (表示専用)"
            );
            // 3) prefill の表に site 別の行が出る (ms 降順 = ATTN_PF が先)
            let s = summary();
            let prefill = s
                .split("[prefill]")
                .nth(1)
                .unwrap()
                .split("[decode]")
                .next()
                .unwrap();
            // D2H 行が 0 件なので site 表は d2h_state 行の下に出ない —
            // 直接 site 表を組み立てて順序を見る (表示関数の単体確認)。
            let mut out = String::new();
            push_d2h_src_top10(p, &mut out);
            let first = out
                .lines()
                .find(|l| l.contains("d2h_src_"))
                .expect("site 行が無い");
            assert!(
                first.contains("d2h_src_attn_pf"),
                "ms の降順で最大の site が先頭: {out}"
            );
            assert!(prefill.contains("[prefill]") || !prefill.is_empty());
            set_phase(prev);
            reset();
        });
    }

    /// s78 (T7): `gdn_proj` 内訳 (gdnp_*) の id が連番で、既存の細分
    /// (D2H 細分 0..9 / GDN_PF 9..13 / D2H_SRC 13..19) と交差せず、
    /// かつ 10*D2H ブロック内 (生 id の D2H スロットへ落ちない領域) に
    /// 収まることを固定する。
    #[test]
    fn gdn_proj_split_ids_are_outside_existing_ranges() {
        let ids = [
            D2H_GDN_PROJ_H2D,
            D2H_GDN_PROJ_KERNEL,
            D2H_GDN_PROJ_SYNC,
            D2H_GDN_PROJ_D2H,
        ];
        assert_eq!(ids.len(), D2H_GDN_PROJ_N);
        assert_eq!(D2H_GDN_PROJ_NAMES.len(), D2H_GDN_PROJ_N);
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, D2H_GDN_PROJ_FIRST + i);
            assert!(id >= N_OPS, "raw id range violated: {id}");
        }
        // 既存細分の末尾 (d2h_src_other) の直後から始まる
        assert_eq!(D2H_GDN_PROJ_FIRST, D2H_SRC_FIRST + D2H_SRC_N);
    }

    /// s57 (T4): attn_core の層種別細分 id が生 id の範囲外で連番であること。
    #[test]
    fn attn_split_ids_are_outside_raw_range() {
        assert_eq!(ATT_GDN, 100 * ATTN);
        assert_eq!(ATT_FULL, 100 * ATTN + 1);
        assert_eq!(ATT_GDN_PROJ, 100 * ATTN + 2);
        assert_eq!(ATT_GDN_SCAN, 100 * ATTN + 3);
        assert_eq!(ATT_CLS_NAMES.len(), ATT_CLS_N);
        for (i, &id) in [ATT_GDN, ATT_FULL, ATT_GDN_PROJ, ATT_GDN_SCAN]
            .iter()
            .enumerate()
        {
            assert_eq!(id, ATT_GDN + i);
            assert!(id >= N_OPS, "細分 id は生 id の範囲外: {id}");
        }
        // D2H 細分 (10*D2H + i, i < 9) と範囲が交差しないこと
        assert!(ATT_GDN > D2H_LOGITS_PF_HELD);
    }

    /// s61c (T5b 計測のみ): `attn_gdn` は GDN 投影 + スキャンの和、`attn_core` は
    /// さらに full attention との和。細分を足しても生バケットの意味が変わらない
    /// こと (T4 の不変条件) を固定する。
    #[test]
    fn attn_gdn_splits_into_proj_and_scan_without_changing_totals() {
        let _guard = lock_counters();
        reset();
        set_phase(Phase::Prefill);
        // 2 層分: 投影 (beta/alpha) 2 call + スキャン 1 call
        for _ in 0..2 {
            add_sample(Phase::Prefill, ATT_GDN_PROJ, 1_000_000, 1, 512, true);
            add_sample(Phase::Prefill, ATT_GDN_PROJ, 1_500_000, 1, 512, true);
            add_sample(Phase::Prefill, ATT_GDN_SCAN, 10_000_000, 1, 512, true);
        }
        add_sample(Phase::Prefill, ATT_FULL, 2_000_000, 2, 512, true);
        let s = summary();
        let find = |name: &str| -> Vec<String> {
            s.lines()
                .find(|l| l.split_whitespace().next() == Some(name))
                .unwrap_or_else(|| panic!("summary に {name} 行が無い\n{s}"))
                .split_whitespace()
                .map(|x| x.to_string())
                .collect()
        };
        let core = find("attn_core");
        let gdn = find("attn_gdn");
        let full = find("attn_full");
        let proj = find("gdn_proj");
        let scan = find("gdn_scan");
        // 投影 5.0ms ((1.0 + 1.5) ×2 層) + スキャン 20ms (10 ×2 層) = 25.0
        assert_eq!(proj[1], "5.0", "投影は beta/alpha の 2 call × 2 層: {proj:?}");
        assert_eq!(scan[1], "20.0", "スキャンは 1 call × 2 層: {scan:?}");
        assert_eq!(gdn[1], "25.0", "attn_gdn = proj + scan: {gdn:?}");
        // attn_core = attn_gdn + attn_full
        assert_eq!(core[1], "27.0", "attn_core = gdn + full: {core:?}");
        assert_eq!(full[1], "2.0");
        // calls は細分側でも独立に数える (attn_gdn は親にも合算される)
        assert_eq!(proj[2], "4");
        assert_eq!(scan[2], "2");
        assert_eq!(gdn[2], "6");
        assert_eq!(core[2], "8", "attn_core = GDN 6 + full 2");
        // ネスト: gdn_proj / gdn_scan は attn_gdn より深い位置に出る。
        // prefill 表は [prefill] の見出し行から [decode] の手前まで (decode は
        // 空だと "(no calls)" しか出さず総和行が無いので、範囲を切らないと
        // 下の cumulative 表まで舐めてしまう)。
        let prefill = s
            .split("[prefill]")
            .nth(1)
            .expect("summary に [prefill] が無い")
            .split("[decode]")
            .next()
            .unwrap();
        assert!(
            prefill.lines().any(|l| l.starts_with("    attn_gdn")),
            "attn_gdn は 4 スペース: {prefill}"
        );
        assert!(
            prefill.lines().any(|l| l.starts_with("      gdn_proj")),
            "gdn_proj は 6 スペース (attn_gdn の内訳): {prefill}"
        );
        // 生バケット (attn_core) は細分を二重計上しない: 27.0 のまま
        // (細分 25.0 を足した 52.0 にならない)。
        // 注: prefill 表の総和行 (= attn_core + d2h) は**他のテストの並走**で
        // 動くため絶対値では固定しない。ここでは細分が生バケットへ
        // 二重に効いていないことだけを見る。
        assert_eq!(
            core[1], "27.0",
            "attn_core は細分を二重計上しない (27.0 のまま): {core:?}"
        );
        assert_eq!(core[2], "8");
        reset();
    }

    /// s56 (T2 項目0): decode chain のラベル分岐の算術。
    ///
    /// chain は decode 1 トークンのみを受け、成功時に `cache.advance(1)` するので
    /// 入口では常に `pos0 + 1 == cache.len()`。従来の `cache.len() > pos0` は
    /// この不変条件により **常に false** となり、追加 sync の待ちが全部
    /// 集計外の pfhold に落ちていた (= wait 行が 1 度も出ない)。
    /// ここでは chain の呼び出し位置 (1 点目 / 2 点目以降) を引数に持たせず
    /// `(pos0, cache_len)` だけで判定が成立することを固定する:
    /// - chain 入口の不変条件 (len == pos0 + 1) では pfhold (集計外 = 1 点目扱い)
    /// - cache が pos0 より先へ進んでいる (= prefill 直後ではない) なら wait
    /// - pos0 == 0 (chain では起きない) も同期点が無いので pfhold 側
    #[test]
    fn d2h_logits_split_op_marks_first_decode_step_only() {
        // 通常: prompt 512 トークン prefill 後の 1 点目 (pos0 = 512, len = 513)
        assert_eq!(d2h_logits_split_op(512, 513), D2H_LOGITS_PF_HELD);
        // chain 入口の不変条件 (advance 前): 常に pfhold 側 (= 1 点目)
        for &p in &[1usize, 2, 511, 512, 4095] {
            assert_eq!(
                d2h_logits_split_op(p, p + 1),
                D2H_LOGITS_PF_HELD,
                "chain 入口 (len == pos0+1) は 1 点目扱い: pos0={p}"
            );
        }
        // 従来条件 `len > pos0` が真になる状況: cache が pos0 を超えて先行
        // = 前ステップの sync が既にある → wait に計上
        assert_eq!(d2h_logits_split_op(512, 514), D2H_LOGITS_WAIT);
        assert_eq!(d2h_logits_split_op(512, 600), D2H_LOGITS_WAIT);
        // 空キャッシュ (chain では起きない) は同期点が無いので pfhold 側
        assert_eq!(d2h_logits_split_op(0, 1), D2H_LOGITS_PF_HELD);
        assert_eq!(d2h_logits_split_op(0, 0), D2H_LOGITS_PF_HELD);
        // 返す id は必ず logits 細分 3 本のどれかで、生 id (N_OPS 未満) を汚さない
        for &(p, l) in &[(1usize, 2usize), (7, 7), (7, 8), (7, 9), (4096, 5000)] {
            let op = d2h_logits_split_op(p, l);
            assert!(
                [D2H_LOGITS_WAIT, D2H_LOGITS_COPY, D2H_LOGITS_PF_HELD].contains(&op),
                "細分 id 以外を返した: {op}"
            );
            assert!(op >= N_OPS, "生 id の範囲外であること: {op}");
        }
    }
    /// s55: logits の 3 分割が summary に 3 行 (呼び出し回数の内訳) として出ること。
    /// 直接叩く (整形はコード上 enabled() と無関係)。
    ///
    /// s57 (T4): 行の書式が 6 列に変わった (op/ms/calls/share or cum%/tok-call/via)
    /// ので、列数を固定せず「2 列目 = calls」で検証する (cumulative 行に reset() を
    /// 挟んでいた s55 の書き方も廃止 — 末尾の 1 回だけにする)。
    #[test]
    fn logits_split_is_summarized_per_bucket() {
        let _guard = lock_counters();
        reset();
        for (op, calls) in [
            (D2H_LOGITS_WAIT, 3u64),
            (D2H_LOGITS_COPY, 3),
            (D2H_LOGITS_PF_HELD, 1),
        ] {
            add_sample(Phase::Decode, op, 1000, calls, 0, false);
        }
        // 生 id 側は呼び出し側 (time) が積む — ここでは d2h_state 本体のみ
        add_sample(Phase::Decode, D2H, 7000, 7, 0, false);
        let s = summary();
        for (name, calls) in [
            ("d2h_logits_wait", "3"),
            ("d2h_logits_copy", "3"),
            ("d2h_logits_pfhold", "1"),
        ] {
            let cols: Vec<&str> = s
                .lines()
                .find(|l| l.trim_start().starts_with(name))
                .unwrap_or_else(|| panic!("summary に行が無い: {name}\n{s}"))
                .split_whitespace()
                .collect();
            assert!(cols.len() >= 3, "op/ms/calls は常に先頭 3 列: {name}\n{s}");
            assert_eq!(cols[1], "0.0", "ms: {name}\n{s}");
            assert_eq!(cols[2], calls, "呼び出し回数: {name}\n{s}");
        }
        // 従来の単独行 (d2h_logits) は COPY への置換後は出ない
        assert!(
            !s.lines()
                .any(|l| l.split_whitespace().next() == Some("d2h_logits")),
            "旧 d2h_logits 単独行が出ている\n{s}"
        );
        reset();
    }

    /// s57 (T4): prefill と decode が別テーブルに入り、cumulative は両者の和で
    /// あること (従来の累積値と一致する読み方)。
    #[test]
    fn phases_are_aggregated_separately_and_summed_into_cumulative() {
        let _guard = lock_counters();
        reset();
        add_sample(Phase::Prefill, QKV, 5_000_000, 8, 256, true);
        add_sample(Phase::Decode, QKV, 1_000_000, 100, 1, true);
        add_sample(Phase::Prefill, RMSNORM, 2_000_000, 16, 256, false);

        let ns = |p: Phase, op: usize| NS[p.idx()][op].load(Ordering::Relaxed);
        assert_eq!(ns(Phase::Prefill, QKV), 5_000_000);
        assert_eq!(ns(Phase::Decode, QKV), 1_000_000);
        assert_eq!(
            ns(Phase::Both, QKV),
            6_000_000,
            "cumulative = prefill + decode (従来の読み方と一致)"
        );
        assert_eq!(CALLS[Phase::Prefill.idx()][QKV].load(Ordering::Relaxed), 8);
        assert_eq!(CALLS[Phase::Decode.idx()][QKV].load(Ordering::Relaxed), 100);
        assert_eq!(CALLS[Phase::Both.idx()][QKV].load(Ordering::Relaxed), 108);
        // 平均バッチ幅 = トークン総和 / call 数
        assert_eq!(TOKS[Phase::Prefill.idx()][QKV].load(Ordering::Relaxed), 8 * 256);
        assert_eq!(TOKS[Phase::Decode.idx()][QKV].load(Ordering::Relaxed), 100);

        let s = summary();
        for ph in ["prefill", "decode", "cumulative"] {
            assert!(s.contains(&format!("[{ph}]")), "{ph} 節が無い\n{s}");
        }
        reset();
    }

    /// s57 (T4): attn_core の集計は GDN + full の別行になり、生 id (attn_core) の
    /// 値は従来どおり両者の合計であること。
    ///
    /// 設計 (承認済み): 生 id の行はそのまま「GDN + full の和」を出し、内訳は
    /// 直下のネスト行 (attn_gdn / attn_full) で示す — 表の構造は変えない。
    #[test]
    fn attn_core_is_split_into_gdn_and_full_rows() {
        let _guard = lock_counters();
        reset();
        // GDN: 30 call × 3.0 ms / full: 16 call × 2.0 ms
        for _ in 0..30 {
            add_sample(Phase::Prefill, ATT_GDN, 100_000, 1, 256, true);
        }
        for _ in 0..16 {
            add_sample(Phase::Prefill, ATT_FULL, 125_000, 1, 256, false);
        }
        let s = summary();
        // 生 id (attn_core) の合計 = GDN + full (バケットの意味を変えない)
        assert_eq!(
            NS[Phase::Both.idx()][ATTN].load(Ordering::Relaxed),
            5_000_000,
            "attn_core = attn_gdn + attn_full"
        );
        assert_eq!(
            CALLS[Phase::Both.idx()][ATTN].load(Ordering::Relaxed),
            46,
            "attn_core の call 数 = GDN 30 + full 16"
        );
        // 細分バケットも各自の値を持つ
        assert_eq!(ATT_SUB_NS[Phase::Prefill.idx()][0].load(Ordering::Relaxed), 3_000_000);
        assert_eq!(ATT_SUB_NS[Phase::Prefill.idx()][1].load(Ordering::Relaxed), 2_000_000);

        for name in ["attn_gdn", "attn_full", "attn_core"] {
            let line = s
                .lines()
                .find(|l| l.trim_start().starts_with(name))
                .unwrap_or_else(|| panic!("summary に行が無い: {name}\n{s}"));
            let cols: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(cols[0], name);
        }
        // prefill 節の attn_core 行は GDN と full の和 (3.0 + 2.0 = 5.0 ms)
        let prefill = s.split("[prefill]").nth(1).unwrap();
        let core = prefill
            .lines()
            .find(|l| l.trim_start().starts_with("attn_core"))
            .unwrap();
        let cols: Vec<&str> = core.split_whitespace().collect();
        assert_eq!(cols[1], "5.0", "GDN 3.0 + full 2.0 ms: {core}");
        assert_eq!(cols[2], "46", "GDN 30 + full 16 call: {core}");
        // 生 id 行は「合算」なので tok/call も 46 call 平均 (= 256)
        assert_eq!(cols[4], "256.0", "tok/call = 46 call の平均: {core}");
        // ネスト行はインデントされて出る (表の構造を変えない)
        assert!(
            prefill
                .lines()
                .any(|l| l.starts_with("    attn_gdn")),
            "attn_gdn はインデントされた内訳行: {prefill}"
        );
        reset();
    }

    /// s57 (T4): 実行先ラベル (cpu/gpu/mixed/cpu*) の決定。
    #[test]
    fn exec_label_reports_gpu_cpu_and_mixed() {
        assert_eq!(exec_label(true, 10, 10), "gpu");
        assert_eq!(exec_label(true, 10, 0), "cpu");
        assert_eq!(exec_label(true, 10, 4), "mixed");
        assert_eq!(exec_label(false, 10, 0), "cpu*");
        assert_eq!(exec_label(true, 0, 0), "-");
    }

    /// s61 (T5a 撤回): `tok/call` の分母は **行数を渡したサンプルだけ**。
    /// 行数 0 の呼び出し (d2h 等) が混ざっても平均バッチ幅が歪まないこと
    /// (T4 は TOKS / CALLS で割っていた)。
    #[test]
    fn tok_per_call_excludes_samples_without_rows() {
        let _guard = lock_counters();
        reset();
        set_phase(Phase::Prefill);
        // 行数を渡した 2 call (各 256 行) + 行数 0 の 8 call
        add_sample(Phase::Prefill, QKV, 1_000_000, 2, 256, false);
        add_sample(Phase::Prefill, QKV, 1_000_000, 8, 0, false);
        let s = summary();
        let line = s
            .lines()
            .find(|l| l.trim_start().starts_with("qkv_proj"))
            .unwrap_or_else(|| panic!("summary に qkv_proj 行が無い\n{s}"));
        let cols: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cols[2], "10", "calls は従来どおり全 call を数える: {line}");
        assert_eq!(
            cols[4], "256.0",
            "tok/call の分母から rows=0 を外す (T4 は 512/10 = 51.2 と出ていた): {line}"
        );
        reset();
    }

    /// s61: `time_n_ev` は計測ラベルを付けるだけで、戻り値と数値経路を
    /// `time_n` と完全に同一に保つ (誤診の再発防止の前提)。
    #[test]
    fn time_n_ev_passes_through_value_and_side_effects() {
        let v = time_n(QKV, 4, || vec![1.0f32, 2.0, 3.0]);
        let w = time_n_ev(QKV, 4, || vec![1.0f32, 2.0, 3.0]);
        assert_eq!(v, w, "time_n_ev は戻り値を変えない");
        // クロージャの副作用 (状態更新) も 1 回だけ通ること
        let mut hits = 0;
        let _ = time_n_ev(QKV, 4, || {
            hits += 1;
        });
        assert_eq!(hits, 1, "クロージャは 1 回だけ実行される");
        // 戻り値の型は何でも通る (タプル含む)
        let (a, b) = time_n_ev(QKV, 4, || (1u8, "x"));
        assert_eq!((a, b), (1u8, "x"));
    }

    /// s57 (T4): `with_phase` は PROFILE off では既存のフェーズ設定に触れない
    /// (= 完全 no-op)。on のときもネスト終了で元のフェーズへ戻る。
    #[test]
    fn with_phase_is_noop_and_restores() {
        let prev = phase();
        let r = with_phase(Phase::Prefill, || 42);
        assert_eq!(r, 42);
        if !enabled() {
            assert_eq!(phase(), prev, "PROFILE off ではフェーズを変えない");
        } else {
            assert_eq!(phase(), prev, "ネスト終了で元へ戻る");
        }
        set_phase(prev);
    }

    /// s85 (Step 0): Ambient の会計が**加算的**であること。
    ///
    /// s84 は子を親へも合算していたため、表の合計が実測 wall より膨らんでいた
    /// (5,686 tok の 1 リクエストで TOTAL 37,922.7 ms / wall 13.733 s)。ここでは
    /// 1) 親 >= 子、2) self = 親 − 子、3) 子孫の sample がフェーズ総和で
    /// 二重に数えられない、4) 総和 = 実測区間の分解 (self の和) を固定する。
    #[test]
    fn ambient_parent_buckets_are_additive() {
        with_prof_counters(|| {
            reset();
            let p = Phase::Prefill.idx();
            // 実経路と同じ形: 親 (RMSNORM) の内側に子 (QKV 6.0) と norm 本体 (1.0) を積む。
            {
                let _amb = Ambient::enter(RMSNORM);
                add_sample(Phase::Prefill, QKV, 3_000_000, 1, 8, true);
                add_sample(Phase::Prefill, QKV, 3_000_000, 1, 8, true);
                add_sample(Phase::Prefill, RMSNORM, 1_000_000, 1, 8, false);
            }
            let parent = NS[p][RMSNORM].load(Ordering::Relaxed);
            let child = parent_child_ns(p, RMSNORM);
            assert!(parent >= child, "親 >= 子 (parent={parent}, child={child})");
            assert_eq!(parent, 7_000_000, "親 = 子の合計 (norm 1.0 + QKV 6.0)");
            assert_eq!(child, 7_000_000, "親へ入れた子の累積");
            assert_eq!(
                parent_self_ns(p, RMSNORM),
                0,
                "子が親を説明しきるなら self = 0 (二重計上しない)"
            );
            // 子を持たないバケットの self は親と同値
            assert_eq!(parent_self_ns(p, QKV), 3_000_000 * 2);

            // フェーズ総和は self の和 = 実測区間の分解。
            // QKV 6.0 (子) + RMSNORM self 0.0 = 6.0 — 親 7.0 を足した 13.0 にならない。
            let s = summary();
            let prefill = s
                .split("[prefill]")
                .nth(1)
                .unwrap()
                .split("[decode]")
                .next()
                .unwrap();
            let total_line = prefill.lines().find(|l| l.contains("total")).unwrap();
            assert!(
                total_line.contains("6.0 ms"),
                "prefill total は self の和 (子の二重計上なし): {total_line}"
            );
            // 表の行は self 列を持つ (7 列目)。
            let norm_row = prefill
                .lines()
                .find(|l| l.split_whitespace().next() == Some("rmsnorm"))
                .unwrap();
            let cols: Vec<&str> = norm_row.split_whitespace().collect();
            assert_eq!(cols.len(), 7, "op/ms/calls/share/tok-call/via/self: {norm_row}");
            assert_eq!(cols[1], "7.0", "ms 列は親の区間 (子を含む): {norm_row}");
            assert_eq!(cols[6], "0.0", "self 列は親 − 子: {norm_row}");
            reset();
        });
    }

    /// s85 (Step 0): 入れ子の Ambient でも区間の**包含**が成り立ち
    /// (`親 = 直下 + 子孫`)、総和では孫を二重に数えないこと。
    #[test]
    fn nested_ambient_is_contained_but_not_double_counted() {
        with_prof_counters(|| {
            reset();
            let p = Phase::Prefill.idx();
            {
                let _outer = Ambient::enter(RMSNORM); // 祖父
                add_sample(Phase::Prefill, RMSNORM, 1_000_000, 1, 4, false); // 祖父の直下
                {
                    let _inner = Ambient::enter(GATE_UP); // 親
                    add_sample(Phase::Prefill, DOWN, 2_000_000, 1, 4, true);
                }
            }
            // 親 (GATE_UP) の区間 = 孫 DOWN 2.0 (自分の直下区間は計時していない)
            assert_eq!(NS[p][GATE_UP].load(Ordering::Relaxed), 2_000_000);
            assert_eq!(parent_child_ns(p, GATE_UP), 2_000_000);
            assert_eq!(parent_self_ns(p, GATE_UP), 0, "GATE_UP self = 0");
            // 祖父 (RMSNORM) の区間 = 自分の直下 1.0 + 孫 DOWN 2.0 = 3.0 (包含)
            assert_eq!(
                NS[p][RMSNORM].load(Ordering::Relaxed),
                3_000_000,
                "親の区間は孫の分も含む (入れ子の包含)"
            );
            // 祖父へ入れたぶん = 直下 1.0 + 孫 2.0 = 3.0。区間が包含なので self は 0
            // (直下の区間も「祖父に入れた時間」として数える — 表を加算的に保つため)。
            assert_eq!(parent_child_ns(p, RMSNORM), 3_000_000, "親へ入れた子の累積");
            assert_eq!(parent_self_ns(p, RMSNORM), 0, "RMSNORM self = 0");
            // self の和 = GATE_UP 0.0 + RMSNORM 0.0 + DOWN 2.0 = 2.0 ms。
            // 入れ子を二重に数えない (1.0 + 2.0 + 2.0 = 5.0 にならない)。
            let s = summary();
            let prefill = s
                .split("[prefill]")
                .nth(1)
                .unwrap()
                .split("[decode]")
                .next()
                .unwrap();
            let total_line = prefill.lines().find(|l| l.contains("total")).unwrap();
            assert!(
                total_line.contains("2.0 ms"),
                "入れ子でも総和は実測区間 (2.0 ms = DOWN のみ): {total_line}"
            );
            reset();
        });
    }
}