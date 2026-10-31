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

// ---------------------------------------------------------------------------
// s138 (Task B6・計測のみ): 「毎ミリ秒がどれか 1 つの行に入る」ための 2 系統の
// 分割表。**表示専用** — どの生 id (N_OPS 未満) にも D2H 細分にも合算しない
// (既存バケットの値はこの commit で 1 bit も動かさない)。
//
// 動機 (supervisor 実測: 27B q4 serve, DENNO_PROFILE=1, 187 token, wall 19.342 s):
//   (1) decode のプロファイル合計は 8.7 ms/token だが wall は約 58 ms/token。
//       残り約 49 ms は「どのバケットにも入っていない」。decode の q4 chain 経路
//       (chain launch / sync / logits D2H / q4 GEMV launch) には行が無く、
//       `d2h_src_linear` は 83,514 call で 0.0 ms (= 実際にホストが止まる場所を
//       指していない) だった。
//   (2) prefill の gate_up_proj は 26.7 ms/call (tok/call 18.5) で、bench の
//       m=8 (0.514) / m=64 (2.323) と 5-10 倍食い違う。線形ラッパ 1 call の
//       中身 (確保 / memset / H2D / 量子化 / launch / D2H / 待ち) が未分化。
// ---------------------------------------------------------------------------

/// 表 A (decode chain): 1 ステップの区間を **互いに排他** に割る。実装は
/// cuda.rs 側の網掛け (`chain_step_begin` / `chain_seg`) が `CHAIN_SEG` と
/// `CHAIN_ACC` で持ち、ステップ終端で `split_flush_step` に流す。
/// chain 側は 1 ステップで使う GPU 資源が 1 本の stream に直列なので、
/// 区間ごとに「発行 (host が API から戻るまで)」と「待ち (sync の中)」を
/// 別の行に分ける — 合計がステップの host 時間になる。
pub const SPLIT_CHAIN_N: usize = 8;
/// 開始位置は **`10*D2H` の 23 ブロック (0..22) の直後**にして、以後の追加でも
/// 番号を詰め直さない (過去のプロファイルログとの比較を壊さないため)。
pub const SPLIT_CHAIN_FIRST: usize = 10 * D2H + 23;
/// chain の launch API 発行 (rmsnorm / GEMV / swiglu / attention / q4 GEMM の
/// cuLaunchKernel など。「発行した」時間で、GPU の実行時間ではない)
pub const CHAIN_LAUNCH: usize = SPLIT_CHAIN_FIRST;
/// decode 1 ステップ 1 回の logits D2H 発行 (`chain_download(CH_LOGITS)`)
pub const CHAIN_D2H_LOGITS: usize = SPLIT_CHAIN_FIRST + 1;
/// chain のその他の D2H 発行 (full 層 k/v 行の staged async など)
pub const CHAIN_D2H_ARRAY: usize = SPLIT_CHAIN_FIRST + 2;
/// chain の sync のうち、**直前の launch に帰属できる** 待ち (ホストに入る cuda 呼び出し)
pub const CHAIN_SYNC_WAIT: usize = SPLIT_CHAIN_FIRST + 3;
/// CPU 側の計算・側付け (Vec 確保・ホストコピー・rope 表生成など)
pub const CHAIN_CPU: usize = SPLIT_CHAIN_FIRST + 4;
/// ホストで起きる GPU API 呼び出し (cuMemcpy 同期版・Context 切り替え等。
/// 多くは GPU 側の完了待ちになる)
pub const CHAIN_HOST_API: usize = SPLIT_CHAIN_FIRST + 5;
/// (デコード) sync の直前までの区間のうち、どこにも当たらなかった残余。
/// 実際は「待ち」であって「発行」ではないものが混ざる (sync で締める)
pub const CHAIN_SYNC_RESIDUAL: usize = SPLIT_CHAIN_FIRST + 6;
/// (デコード) ステップの前後 (embedding 用意・sampler など chain の外)
pub const CHAIN_OUTER: usize = SPLIT_CHAIN_FIRST + 7;

/// 表 A の行数は 8 だが、上の 7 段に加えて「直前の launch に帰属できる待ち」
/// = `chain_kernel` を **待ちの内訳** として 1 本持つため、表 A は 9 行。
/// (`chain_kernel` は `chain_sync_wait` の内訳 = 待ち行から同じ秒数を引いて
/// 移すので、待ちの合計も排他の和も動かない。)
pub const SPLIT_CHAIN_ROWS: usize = 9;
pub const CHAIN_KERNEL_WAIT: usize = SPLIT_CHAIN_FIRST + 8;

/// 表 B (prefill の線形ラッパ 1 call): `maybe_gpu_linear_*` の内側を排他に割る。
pub const SPLIT_Q4_N: usize = 8;
/// 表 B の開始 = 表 A の全行 (8 段 + `chain_kernel` の 1 行) の直後。
pub const SPLIT_Q4_FIRST: usize = SPLIT_CHAIN_FIRST + SPLIT_CHAIN_ROWS;
/// `vec![0.0; m*n]` の確保 + ゼロ埋め (GPU 経路でも CPU フォールバックでも通る)
pub const Q4P_OUT_ALLOC: usize = SPLIT_Q4_FIRST;
/// 非同期転送用のピン留めホスト側バッファの確保/伸長 (cuMemHostAlloc 相当)
pub const Q4P_PIN_ALLOC: usize = SPLIT_Q4_FIRST + 1;
/// ホスト f32 → ピン留め領域への H2D ステージング copy (SIMD memcpy)
pub const Q4P_H2D_STAGE: usize = SPLIT_Q4_FIRST + 2;
/// H2D 発行
pub const Q4P_H2D_ISSUE: usize = SPLIT_Q4_FIRST + 3;
/// 量子化カーネル (Q4 経路は量子化しないので 0)、GEMV/GEMM カーネル発行
pub const Q4P_KERNEL_LAUNCH: usize = SPLIT_Q4_FIRST + 4;
/// 出力 D2H 発行
pub const Q4P_D2H_ISSUE: usize = SPLIT_Q4_FIRST + 5;
/// cuCtxSynchronize (= カーネル実行 + 転送の完了待ち)
pub const Q4P_SYNC_WAIT: usize = SPLIT_Q4_FIRST + 6;
/// ピン留め領域 → 呼び出し側 `out` のコピー (sync の後)
pub const Q4P_HOST_COPY: usize = SPLIT_Q4_FIRST + 7;

// ---------------------------------------------------------------------------
// s138: 汎用の分割表 (表示専用)。上 2 表はこの 1 本の実装を共有する。
// ---------------------------------------------------------------------------
/// 分割表の総数 (0 = decode chain, 1 = prefill の線形ラッパ)。
/// 表を増やすときは `SPLIT_TABLES` の添字と `push_split_rows` の呼び出しを対で足す。
pub const SPLIT_TABLES: usize = 3;
/// 0 = decode chain (表 A) / 1 = prefill の線形ラッパ (表 B) / 2 = serve (q4) 経路 (表 C)
pub const SPLIT_CHAIN_TABLE: usize = 0;
pub const SPLIT_Q4_TABLE: usize = 1;
/// s148: serve (q4) 経路の表 C。行数は `SPLIT_SERVE_N` (s154 で 16 → 32 行)。
pub const SPLIT_SERVE_TABLE: usize = 2;
/// 表 1 本の行数の最大値 (表 C = 32 行 / 表 A = 9 行 / 表 B = 8 行)。積算器はこの本数で持つ。
pub const SPLIT_ROWS: usize = SPLIT_SERVE_N;
/// 積算器 (`split_flush_step` の `acc`) の本数 = 一番多い表の行数。
/// s138 後半: 表 A が `chain_kernel` の 1 行を得て 9 行になったため、
/// 「表ごとに行数が同じ」前提の `SPLIT_N` は廃止した (行数は `split_rows()`)。
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

/// s138: 2 次元版 (`[Phase::N]` × 表 × 行)。`[t; N]` は Copy を要求するため
/// 内側も inline const で要素ごとに独立した Atomic を作る。
const fn mk_counters2<const R: usize, const T: usize>() -> [[[AtomicU64; R]; T]; Phase::N] {
    [const { [const { [const { AtomicU64::new(0) }; R] }; T] }; Phase::N]
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
/// s138 (Task B6): 分割表 (decode chain / prefill の線形ラッパ) の累積。フェーズ別。
/// **表示専用** — どの生 id にも D2H 細分にも合算しない (二重計上防止)。
/// 行は表ごとに違う (表 A = `SPLIT_CHAIN_ROWS` / 表 B = `SPLIT_Q4_N`) が、
/// 積算器は `SPLIT_ROWS` 本で共通。`split_slot` が表と行を 1 つの id に畳む。
static SPLIT_NS: [[[AtomicU64; SPLIT_ROWS]; SPLIT_TABLES]; Phase::N] =
    mk_counters2::<SPLIT_ROWS, SPLIT_TABLES>();
static SPLIT_CALLS: [[[AtomicU64; SPLIT_ROWS]; SPLIT_TABLES]; Phase::N] =
    mk_counters2::<SPLIT_ROWS, SPLIT_TABLES>();
/// s138: 分割表の段のうち「待ち」に当たる行 (呼び出し側の site が無く、
/// sync 区間の残余をここへ入れる)。表 A = CHAIN_SYNC_WAIT / 表 B = Q4P_SYNC_WAIT。
const SPLIT_WAIT_ROW: [usize; SPLIT_TABLES] = [
    CHAIN_SYNC_WAIT - SPLIT_CHAIN_FIRST,
    Q4P_SYNC_WAIT - SPLIT_Q4_FIRST,
    // s148: 表 C は「待ち」を `sl_lin_sync` に明示的に積む (flush 経由の残余は無い)。
    SL_LIN_SYNC - SPLIT_SERVE_FIRST,
];
/// s138: 分割表 A (decode chain) の行名 (`SPLIT_CHAIN_FIRST + i`)。
/// s140: `chain_cpu` / `chain_sync_resid` は 1 ステップ内で 1 か所からしか
/// 積まれない (CPU 区間は per-op 経路の 1 ループ・残余はステップ末尾) ので、
/// 積算器でまとめて足すと tick を打った瞬間の経過が落ちて排他の和が縮む。
/// この 2 行は `split_sample` (積んだ時点の時刻で直接加算) で積む。
const SPLIT_CHAIN_NAMES: [&str; SPLIT_CHAIN_ROWS] = [
    "chain_launch",
    "chain_d2h_logits",
    "chain_d2h_array",
    "chain_sync_wait",
    "chain_cpu",
    "chain_host_api",
    "chain_sync_resid",
    "chain_outer",
    // 待ちの内訳 (sync の直前に発行したカーネルの実行待ち = 待ち行から移す)
    "chain_kernel",
];
/// s138: 分割表 B (prefill の線形ラッパ 1 call) の行名 (`SPLIT_Q4_FIRST + i`)。
const SPLIT_Q4_NAMES: [&str; SPLIT_Q4_N] = [
    "lin_out_alloc",
    "lin_pin_alloc",
    "lin_h2d_stage",
    "lin_h2d",
    "lin_kernel",
    "lin_d2h",
    "lin_sync_wait",
    "lin_host_copy",
];

/// s148 (Task B10・計測のみ): **serve (q4) 経路**の 1 ステップ / 1 線形呼び出しを
/// 排他に割る表 (表 C)。s138 の表 A (`chain_*`) / 表 B (`lin_*`) は「q4 serve が
/// 実際に通る関数」の内側に無く、27B q4 の実測では
///   表 A … decode 1,602 step に対し 0 サンプル (chain は q4 で発火しない)
///   表 B … prefill 608 call / 7,846 ms に対し 802 サンプル / 445.8 ms
/// だった (どちらも wrapper の外側を数えている)。ここは **表 A / 表 B を置き換えず
/// に並置する** 追加表で、本番経路の最も内側 (1 step に 1 回 / 1 call に 1 回) から
/// 積む。既存バケット・既存表の値は 1 bit も動かさない (表示専用)。
///
/// 段はすべて「1 step (または 1 call) の中で互いに排他」で、合計がその step
/// (または call) の壁時計になる。`sl_other` は「上記のどれにも当たらなかった
/// 残余」で、必ず実測残余 (全体 − 既知段の和) を入れる — 何も測っていない穴が
/// 0 ms として静かに通らないようにするため。
/// s156 (Task B19): 40 → 45 (`rsv_a` / `rsv_b` の 2 行と、後ろの空きを gap_fwd の
/// 内訳 7 行へ流用)。
pub const SPLIT_SERVE_N: usize = 45;
/// 表 C の開始は表 B の末尾 (詰め直しをしない — 過去ログとの位置関係を保つ)。
pub const SPLIT_SERVE_FIRST: usize = SPLIT_Q4_FIRST + SPLIT_Q4_N;
/// decode 1 ステップの全体 (engine の step 入口 → forward 戻り)。サーバは
/// `step_chat_batch` を呼ぶので 1 step = 1 サンプル。
pub const SL_STEP: usize = SPLIT_SERVE_FIRST;
/// forward 本体 (prompt 準備 = KV 容量確保を含む) のうち embed → 層ループ
/// (投影 / attention / FFN) だけを除いた残余。
pub const SL_STEP_PREP: usize = SPLIT_SERVE_FIRST + 1;
/// 層ループの投影 (線形層) 呼び出し: q4 の GEMM/GEMV ディスパッチ入口から戻るまで。
/// 内訳は下の `sl_lin_*` (排他)。
pub const SL_LIN: usize = SPLIT_SERVE_FIRST + 2;
/// 線形呼び出しのうち行列算カーネルの発行 (実行列算の時間は下の sync に寄る)。
pub const SL_LIN_LAUNCH: usize = SPLIT_SERVE_FIRST + 3;
/// 線形呼び出しのうちホスト同期待ち (cuCtxSynchronize = カーネル実行 + 転送完了)。
pub const SL_LIN_SYNC: usize = SPLIT_SERVE_FIRST + 4;
/// 線形呼び出しのうち活性化の H2D (ピン留め staging copy + 発行)。
pub const SL_LIN_H2D: usize = SPLIT_SERVE_FIRST + 5;
/// 線形呼び出しのうち出力 D2H (発行 + ピン留め領域 → 呼び出し側バッファのコピー)。
pub const SL_LIN_D2H: usize = SPLIT_SERVE_FIRST + 6;
/// 線形呼び出しのうち出力領域の確保 (`vec![0.0; m*n]` / scratch の grow)。
pub const SL_LIN_OUTALLOC: usize = SPLIT_SERVE_FIRST + 7;
/// 線形呼び出しのうち量子化・準備 (Q4 は量子化しない = ほぼ 0)。
pub const SL_LIN_QUANT: usize = SPLIT_SERVE_FIRST + 8;
/// 線形呼び出しのうち残余 (上記のどれにも当たらない時間)。
pub const SL_LIN_OTHER: usize = SPLIT_SERVE_FIRST + 9;
/// 層ループの投影以外 (rmsnorm / attention コア / swiglu / GDN スキャン / 残差)。
pub const SL_STEP_LAYERS_OTHER: usize = SPLIT_SERVE_FIRST + 10;
/// 最終 norm + lm_head (logits をホストへ戻す前まで)。
pub const SL_STEP_FINAL: usize = SPLIT_SERVE_FIRST + 11;
/// ステップ末尾の logits D2H (sync 待ちで GPU の全処理完了を吸収する)。
pub const SL_STEP_LOGITS_D2H: usize = SPLIT_SERVE_FIRST + 12;
/// サンプラ (penalty / 拘束マスクを含む sample の呼び出し)。
pub const SL_STEP_SAMPLER: usize = SPLIT_SERVE_FIRST + 13;
/// detok (差分テキストの生成。qwen35 は KV/状態の D2H を持たない)。
pub const SL_STEP_DETOK: usize = SPLIT_SERVE_FIRST + 14;
/// ステップの残余 (上記のどれにも当たらない時間 = 未帰属)。
pub const SL_STEP_OTHER: usize = SPLIT_SERVE_FIRST + 15;
/// s154 (Task B15・計測のみ): `sl_step_lay_other` (層ループの投影以外) の内訳。
/// 層あたり 1 回しか起きない段 (`rmsnorm` ×2 / `gdn_proj` / `gdn_scan` /
/// `full_attn` ×4) は `calls` が steps × 層数 になるため、**表示は転記のみ**で
/// COVERAGE の断言 (samples == steps) は per-step の段にだけ置く。
///
///   sl_lay_embed     ステップ入口の embed 行の複製 (1 step 1 回)
///   sl_lay_norm_in   層頭の入力 rmsnorm (per layer)
///   sl_lay_gdn_proj  GDN の 4 射影 (qkv / gate / beta / alpha を 1 区間)
///   sl_lay_gdn_scan  GDN スキャン (conv + SSM・ここは**トークン逐次**の関数)
///   sl_lay_full_qk   full 層の q/k/v 射影 (q4 GEMV)
///   sl_lay_full_core full attention コア (qknorm + rope + KV 書き込み + attend)
///   sl_lay_full_oproj full 層の o_proj
///   sl_lay_ffn_gate  FFN の gate/up 射影
///   sl_lay_swiglu    swiglu (ホスト f32)
///   sl_lay_ffn_down  FFN の down 射影
///   sl_lay_norm_post 層尾の post rmsnorm (= 次層の入力 norm)
///   sl_lay_residual  残差加算 (attn 後 / FFN 後・ホスト f32)
///   sl_lay_other     上記のどれにも入らない層ループの時間 (≈ 0 を期待する穴)
///
/// そして `sl_step_other` (59.0%) の名前:
///
///   sl_so_lin_sum     この step の間に `sl_lin` が積んだ分 (転記)
///   sl_so_lin_d2h     ↑のうち D2H 段 (カーネルの完了待ちを吸収している)
///   sl_so_sample_pen  サンプラ前の penalty + tool マスク (1 step 1 回)
///   sl_so_stream      detok 後の delta / flush / stop 判定 (1 step 1 回)
///   sl_so_logits     logits のホスト反映 (`lm_head` 終端 → 戻り値・1 step 1 回)
///   sl_so_gap_fwd    forward 呼び出しの残り (1 step 1 回)
///   sl_so_end        最後の forward の戻り → `step_end` (1 step 1 回)
///   sl_so_resid      残余 = `sl_step_other` 全体 − 上の child の和 (1 step 1 回)
pub const SL_LAY_EMBED: usize = SPLIT_SERVE_FIRST + 16;
pub const SL_LAY_NORM_IN: usize = SPLIT_SERVE_FIRST + 17;
pub const SL_LAY_GDN_PROJ: usize = SPLIT_SERVE_FIRST + 18;
pub const SL_LAY_GDN_SCAN: usize = SPLIT_SERVE_FIRST + 19;
pub const SL_LAY_FULL_QK: usize = SPLIT_SERVE_FIRST + 20;
pub const SL_LAY_FULL_CORE: usize = SPLIT_SERVE_FIRST + 21;
pub const SL_LAY_FULL_OPROJ: usize = SPLIT_SERVE_FIRST + 22;
pub const SL_LAY_FFN_GATE: usize = SPLIT_SERVE_FIRST + 23;
pub const SL_LAY_SWIGLU: usize = SPLIT_SERVE_FIRST + 24;
pub const SL_LAY_FFN_DOWN: usize = SPLIT_SERVE_FIRST + 25;
pub const SL_LAY_NORM_POST: usize = SPLIT_SERVE_FIRST + 26;
pub const SL_LAY_RESIDUAL: usize = SPLIT_SERVE_FIRST + 27;
pub const SL_LAY_OTHER: usize = SPLIT_SERVE_FIRST + 28;
pub const SL_SO_LIN_SUM: usize = SPLIT_SERVE_FIRST + 29;
pub const SL_SO_LIN_D2H: usize = SPLIT_SERVE_FIRST + 30;
pub const SL_SO_SAMPLE_PEN: usize = SPLIT_SERVE_FIRST + 31;
pub const SL_SO_STREAM: usize = SPLIT_SERVE_FIRST + 32;
pub const SL_SO_LOGITS: usize = SPLIT_SERVE_FIRST + 33;
pub const SL_SO_GAP_FWD: usize = SPLIT_SERVE_FIRST + 34;
pub const SL_SO_END: usize = SPLIT_SERVE_FIRST + 35;
pub const SL_SO_RESID: usize = SPLIT_SERVE_FIRST + 36;
/// s154 (B15): 最終 norm + lm_head の**転記** (`sl_step_final` と同じ区間を
/// 違う行に積む — `sl_step_final` の値は 1 bit も動かさない)。
pub const SL_LAY_FINAL: usize = SPLIT_SERVE_FIRST + 37;
/// s156 (Task B19): `sl_so_gap_fwd` (B18 実測 54.158 ms/step = instrumented step の
/// 41%) の内訳。forward 入口 → 層ループ末尾を**排他な 7 段**に割る。行名はすべて
/// `sl_gap_*` で揃える:
///
///   sl_gap_embed    embed 行の複製 (1 step 1 回。`sl_lay_embed` と同じ区間)
///   sl_gap_rmsnorm  層あたり 2 回の rmsnorm (attn 前 / FFN 前)
///   sl_gap_gdn      GDN 層の attn 区間 (4 射影 + スキャン + out_proj)
///   sl_gap_attn     full 層の attn 区間。**rope を除いた正味**で出す
///                   (生の区間は rope を含むので、表示と残余計算で rope を引く)
///   sl_gap_rope     partial interleaved RoPE (full 層のみ・attn の内数)
///   sl_gap_quant    活性の量子化。q4 経路は活性を量子化しない (重みは事前量子化・
///                   活性は f32) ので**構造的に 0**。int8 の活性量子化は
///                   `sl_lin_quant` (`sl_lin` の子) に乗る — 穴ではなく経路の性質。
///   sl_gap_ffn      FFN 区間 (post norm の後・gate/up + swiglu + down)
///
/// `sl_gap_resid` は上の和を `sl_so_gap_fwd` から引いた**残余** (転記行・ns は
/// 積まない: 表示のたびに計算する)。B16 の恒等式 (`sl_step_other` = step − 名付きの
/// 兄弟) は動かさない — これらは `sl_so_gap_fwd` の**内側**の子であって、step の
/// 兄弟ではない。
pub const SL_GAP_EMBED: usize = SPLIT_SERVE_FIRST + 38;
pub const SL_GAP_RMSNORM: usize = SPLIT_SERVE_FIRST + 39;
pub const SL_GAP_GDN: usize = SPLIT_SERVE_FIRST + 40;
pub const SL_GAP_ATTN: usize = SPLIT_SERVE_FIRST + 41;
pub const SL_GAP_ROPE: usize = SPLIT_SERVE_FIRST + 42;
pub const SL_GAP_QUANT: usize = SPLIT_SERVE_FIRST + 43;
pub const SL_GAP_FFN: usize = SPLIT_SERVE_FIRST + 44;

/// 表 C の行名 (`SPLIT_SERVE_FIRST + i`)。並びは id の並びと一致させる。
const SPLIT_SERVE_NAMES: [&str; SPLIT_SERVE_N] = [
    "sl_step",
    "sl_step_prep",
    "sl_lin",
    "sl_lin_launch",
    "sl_lin_sync",
    "sl_lin_h2d",
    "sl_lin_d2h",
    "sl_lin_outalloc",
    "sl_lin_quant",
    "sl_lin_other",
    "sl_step_lay_other",
    "sl_step_final",
    "sl_step_logits_d2h",
    "sl_step_sampler",
    "sl_step_detok",
    "sl_step_other",
    // s154 (B15): 層ループの内訳 (calls = steps × 層の数)
    "sl_lay_embed",
    "sl_lay_norm_in",
    "sl_lay_gdn_proj",
    "sl_lay_gdn_scan",
    "sl_lay_full_qk",
    "sl_lay_full_core",
    "sl_lay_full_oproj",
    "sl_lay_ffn_gate",
    "sl_lay_swiglu",
    "sl_lay_ffn_down",
    "sl_lay_norm_post",
    "sl_lay_residual",
    "sl_lay_other",
    // s154 (B15): sl_step_other の内訳 (per step)
    "sl_so_lin_sum",
    "sl_so_lin_d2h",
    "sl_so_sample_pen",
    "sl_so_stream",
    "sl_so_logits",
    "sl_so_gap_fwd",
    "sl_so_end",
    "sl_so_resid",
    "sl_lay_final",
    // s156 (Task B19): sl_so_gap_fwd の内訳 (forward 入口 → 層ループ末尾)
    "sl_gap_embed",
    "sl_gap_rmsnorm",
    "sl_gap_gdn",
    "sl_gap_attn",
    "sl_gap_rope",
    "sl_gap_quant",
    "sl_gap_ffn",
];

/// s148: 表 C の「全体」行と「内訳」行 (排他な内訳の和が全体になる組)。
/// `(全体, [内訳…])` — COVERAGE が両者の差を必ず出す。
const SL_GROUPS: [(usize, &[usize]); 2] = [
    (
        SL_STEP,
        &[
            SL_STEP_PREP,
            SL_LIN,
            SL_STEP_LAYERS_OTHER,
            SL_STEP_FINAL,
            SL_STEP_LOGITS_D2H,
            SL_STEP_SAMPLER,
            SL_STEP_DETOK,
            SL_STEP_OTHER,
        ],
    ),
    (
        SL_LIN,
        &[
            SL_LIN_LAUNCH,
            SL_LIN_SYNC,
            SL_LIN_H2D,
            SL_LIN_D2H,
            SL_LIN_OUTALLOC,
            SL_LIN_QUANT,
            SL_LIN_OTHER,
        ],
    ),
];

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
    if tel_on && !step_skipped() {
        // s156 (Task B19): 間引いた step はテレメトリにも積まない (表 C と同じ間引き)。
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
    if tel_on && !step_skipped() {
        // s156 (Task B19): 間引いた step はテレメトリにも積まない。
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
    if tel_on && !step_skipped() {
        // s156 (Task B19): 間引いた step はテレメトリにも積まない。
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

/// s156 (Task B19): `sl_so_gap_fwd` の内訳 (ns)。`rope` は `attn` の**内数**
/// なので、`attn` には rope を除いた正味を入れる — これで 7 段が排他になり、
/// 和が `sl_so_gap_fwd` に一致する (一致しない分は `gap_fwd_residual_ns`)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GapFwdParts {
    pub embed: u64,
    pub rmsnorm: u64,
    pub gdn: u64,
    pub attn: u64,
    pub rope: u64,
    pub quant: u64,
    pub ffn: u64,
}

impl GapFwdParts {
    /// 表示順の行名。**サンプリングの有無・経路の別で変わらない** (CPU フィクスチャが
    /// 「unsampled でも同じ名前が出る」ことを固定する)。
    pub const NAMES: [&'static str; 7] = [
        "sl_gap_embed",
        "sl_gap_rmsnorm",
        "sl_gap_gdn",
        "sl_gap_attn",
        "sl_gap_rope",
        "sl_gap_quant",
        "sl_gap_ffn",
    ];

    /// 排他な 7 段の和 (= gap_fwd から引く量)。
    #[inline]
    pub fn sum(&self) -> u64 {
        self.embed + self.rmsnorm + self.gdn + self.attn + self.rope + self.quant + self.ffn
    }

    /// 表示用の (行名, ns) 列。順序は `NAMES` と一致。
    #[inline]
    pub fn as_pairs(&self) -> [(&'static str, u64); 7] {
        [
            (Self::NAMES[0], self.embed),
            (Self::NAMES[1], self.rmsnorm),
            (Self::NAMES[2], self.gdn),
            (Self::NAMES[3], self.attn),
            (Self::NAMES[4], self.rope),
            (Self::NAMES[5], self.quant),
            (Self::NAMES[6], self.ffn),
        ]
    }
}

/// s156 (Task B19): gap_fwd の 7 段 — (行名, 実測行, 転記元の行)。
///
/// 27B q4 の per-op 経路は `sl_gap_*` を直接測る (この経路の層ループは
/// `sl_lay_*` のスパンを切っていない)。9B int8 の chain 経路は `sl_lay_*` を
/// 測っているので、`sl_gap_*` の号数が 0 のときだけ `sl_lay_*` の和へ落とす。
/// **両方を足さない** (同じ区間を二重に数えない) ための「どちらか一方」規則。
const GAP_ROWS: [(&str, usize, &[usize]); 7] = [
    ("sl_gap_embed", SL_GAP_EMBED, &[SL_LAY_EMBED]),
    (
        "sl_gap_rmsnorm",
        SL_GAP_RMSNORM,
        &[SL_LAY_NORM_IN, SL_LAY_NORM_POST],
    ),
    ("sl_gap_gdn", SL_GAP_GDN, &[SL_LAY_GDN_PROJ, SL_LAY_GDN_SCAN]),
    (
        "sl_gap_attn",
        SL_GAP_ATTN,
        &[SL_LAY_FULL_QK, SL_LAY_FULL_CORE, SL_LAY_FULL_OPROJ],
    ),
    ("sl_gap_rope", SL_GAP_ROPE, &[]),
    // q4 経路は活性を量子化しない (構造的に 0)。転記元も無い。
    ("sl_gap_quant", SL_GAP_QUANT, &[]),
    (
        "sl_gap_ffn",
        SL_GAP_FFN,
        &[SL_LAY_FFN_GATE, SL_LAY_SWIGLU, SL_LAY_FFN_DOWN],
    ),
];

/// s156 (Task B19): 1 段分の (ns, 号数) を「実測行 → 転記元の行」の順で読む。
#[inline]
fn gap_row(p: usize, primary: usize, fallback: &[usize]) -> (u64, u64) {
    if serve_row_ticks(p, primary) > 0 {
        (serve_row_ns(p, primary), serve_row_ticks(p, primary))
    } else {
        let ns = fallback.iter().map(|&op| serve_row_ns(p, op)).sum();
        let ticks = fallback.iter().map(|&op| serve_row_ticks(p, op)).sum();
        (ns, ticks)
    }
}

/// s156 (Task B19): `sl_so_gap_fwd` の内訳を表 C から読む (転記のみ・ns を積まない)。
#[inline]
pub fn gap_fwd_parts(p: usize) -> GapFwdParts {
    let row = |i: usize| gap_row(p, GAP_ROWS[i].1, GAP_ROWS[i].2).0;
    // rope は attn の内数 — 排他にするため引く (負なら 0 に張り付ける)。
    let rope = row(4);
    GapFwdParts {
        embed: row(0),
        rmsnorm: row(1),
        gdn: row(2),
        attn: row(3).saturating_sub(rope),
        rope,
        quant: row(5),
        ffn: row(6),
    }
}

/// s156 (Task B19): 内訳 7 行の号数 (表示用)。`gap_fwd_parts` と同じ行・同じ
/// 「実測行 → 転記元」規則で読む (chain 経路でも ms と号数が食い違わない)。
#[inline]
pub fn gap_fwd_part_ticks(p: usize) -> [u64; 7] {
    let mut t = [0u64; 7];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = gap_row(p, GAP_ROWS[i].1, GAP_ROWS[i].2).1;
    }
    t
}

/// s156 (Task B19): `sl_so_gap_fwd` の**残余** = gap_fwd − 内訳 7 段の和 (ns)。
///
/// 転記行は ns を積まない (`push` に 0 を渡す) ので、この残余は「表示のたびに
/// 引き算で出す」— 内訳が gap_fwd を超えた (区間の切り方が壊れた) ときは負に
/// なるので 0 に張り付ける (`overflow_drops` と同じ規約)。実経路では 0 を
/// 期待する穴で、**0 でも行を出す** (消すと穴が見えなくなる)。
#[inline]
pub fn gap_fwd_residual_ns(p: usize) -> u64 {
    serve_row_ns(p, SL_SO_GAP_FWD).saturating_sub(gap_fwd_parts(p).sum())
}

#[cfg(test)]
pub(crate) fn d2h_src_calls_for_test(phase: Phase, src: usize) -> u64 {
    if !(D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&src) {
        return 0;
    }
    D2H_SRC_CALLS[phase.idx()][src - D2H_SRC_FIRST].load(Ordering::Relaxed)
}

/// s108: `chain_download` / `chain_download_at` が発行元の件数を積むスロットを
/// 決める純関数 (GPU 不要・テスト可能)。
///
/// 件数を呼び出し側 (slot) から決めるのは、この 2 関数が「スロットをどの用途で
/// 使っているか」を知っている唯一の場所だから。従来は無条件で
/// `D2H_SRC_OTHER` に落としていたため、decode 1 ステップ 1 回の最大の D2H
/// (logits) が未分類側に埋もれ、「どの発行点が重いか」の並びから消えていた。
///
/// - CH_LOGITS … 最終 logits (decode chain のステップ末尾・prefill の lm_head 入力)
/// - それ以外 … `chain_download_at` (活性化の一部行を戻す) を含め `D2H_SRC_LINEAR`
///   — 線形層の結果、または線形層に渡す活性化の部分コピー
pub fn chain_download_site(slot: usize) -> usize {
    // CH_LOGITS は cuda モジュール (feature = "cuda") の定数。非 cuda ビルドでも
    // この純関数だけをテストできるよう、分岐の相手を `cfg` で分ける。
    #[cfg(feature = "cuda")]
    let is_logits = slot == crate::cuda::CH_LOGITS;
    #[cfg(not(feature = "cuda"))]
    let is_logits = false;
    if is_logits {
        D2H_SRC_LOGITS
    } else {
        D2H_SRC_LINEAR
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
    // s156 (Task B19): 間引いた step では読まない (表 C と同じサンプリング)。
    row_on()
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
    if tel_on && !step_skipped() {
        // s156 (Task B19): 間引いた step はテレメトリにも積まない。
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
    if tel_on && !step_skipped() {
        // s156 (Task B19): 間引いた step はテレメトリにも積まない。
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

/// s138 (Task B6): 秒 (f64) を ns へ。分割表の加算だけは**秒**で行う —
/// 行ごとに ns へ丸めて足すと 1 step で 100 回 × 16ns の丸め誤差が積もり、
/// 排他の和が host 時間からずれる (表 A の各行は 1 ステップで 100 回近く積む)。
#[inline]
pub fn secs_to_ns(s: f64) -> u64 {
    if !(s > 0.0) {
        return 0;
    }
    (s * 1e9).round() as u64
}

/// s138 (Task B6): 分割表の行 id を (表, 行) へ畳む純関数 (GPU 不要・テスト可能)。
/// 範囲外は `None` (= 積まない)。表の行 id は `SPLIT_*_FIRST` から表の行数ぶん
/// 連番 (`SPLIT_CHAIN_ROWS` / `SPLIT_Q4_N`) なので、表どうしの範囲が交差してはならない。
#[inline]
pub fn split_slot(op: usize) -> Option<(usize, usize)> {
    if (SPLIT_CHAIN_FIRST..SPLIT_CHAIN_FIRST + SPLIT_CHAIN_ROWS).contains(&op) {
        Some((SPLIT_CHAIN_TABLE, op - SPLIT_CHAIN_FIRST))
    } else if (SPLIT_Q4_FIRST..SPLIT_Q4_FIRST + SPLIT_Q4_N).contains(&op) {
        Some((SPLIT_Q4_TABLE, op - SPLIT_Q4_FIRST))
    } else if (SPLIT_SERVE_FIRST..SPLIT_SERVE_FIRST + SPLIT_SERVE_N).contains(&op) {
        Some((SPLIT_SERVE_TABLE, op - SPLIT_SERVE_FIRST))
    } else {
        None
    }
}

/// s138 (Task B6): 分割表に 1 区間を積む。**表示専用** — `SPLIT_NS` /
/// `SPLIT_CALLS` にしか書かない (生 id・D2H 細分・attn 細分には触らない)。
///
/// `calls` を渡せるのは「その区間が無条件に 1 call 発生した」段用
/// (確保 / launch / D2H / sync など)。区間が起きなかった呼び出し (フラグで
/// 飛ばした段) を分母から外したいときは呼び出し側が 0 を渡す。
/// PROFILE / テレメトリ無効時は分岐 1 回で戻る (時刻は呼び出し側が測らない)。
#[inline]
pub fn split_sample(op: usize, ns: u64, calls: u64) {
    if !enabled() && !crate::telemetry_on() {
        return;
    }
    let Some((t, i)) = split_slot(op) else { return };
    let pi = phase().idx();
    // s156 (Task B19): 間引いた step では表 C / 表 A / 表 B の ns も積まない
    // (号数だけは積む — 表ごとの「何 sample 見たか」の会計を落とさない)。
    if !step_skipped() {
        SPLIT_NS[pi][t][i].fetch_add(ns, Ordering::Relaxed);
        SPLIT_NS[Phase::Both.idx()][t][i].fetch_add(ns, Ordering::Relaxed);
    }
    if calls > 0 {
        SPLIT_CALLS[pi][t][i].fetch_add(calls, Ordering::Relaxed);
        SPLIT_CALLS[Phase::Both.idx()][t][i].fetch_add(calls, Ordering::Relaxed);
    }
}

/// s138 (Task B6): 分割表の計時を行うか (呼び出し側はこれが真のときだけ
/// `Instant::now()` を読む — 既定経路はクロック読みゼロのまま)。
#[inline]
pub fn split_on() -> bool {
    // s156 (Task B19): 表 C の間引き (`DENNO_PROFILE_SAMPLE`) に合わせる —
    // 間引いた step ではクロックを読まない (ns は `split_sample` が捨てる)。
    row_on()
}

// ---------------------------------------------------------------------------
// s148 (Task B10・計測のみ): serve (q4) 経路の表 C の会計
//
// 使い方は 3 段。どれも `on` が偽なら時刻を読まず・確保もせず・表にも書かない
// (`span_begin` は `None` を返し、`span_end` / `push` は 1 分岐で戻る):
//
//   1. ステップ全体: `let st = step_begin(); ... ; step_end(st);`
//      → `sl_step` に 1 サンプル (1 step = 1 サンプル)。
//   2. ステップ内の区間: `let s = span_begin(SL_...); ...; span_end(s);`
//      → 閉じた区間を 1 サンプルとして積む (区間は呼び出し側が排他に保つ)。
//   3. ステップ末尾: `push(SL_STEP_OTHER, 残余 ns)` + `close_table(SPLIT_SERVE_TABLE)`
//      → 残余を明示し、「内訳の和 ≠ 全体」を `step_gap` で次回以降に見せる。
//
// **どの生 id・D2H 細分・attn 細分・表 A / 表 B にも合算しない** (表示専用)。
// ---------------------------------------------------------------------------

/// s148: 表 C の計時を行うか (表 A / 表 B と同じゲート)。
#[inline]
pub fn serve_on() -> bool {
    enabled() || crate::telemetry_on()
}

// ---------------------------------------------------------------------------
// s156 (Task B19): サンプリング (`DENNO_PROFILE_SAMPLE=N`)
//
// B18 の実測 (表 C・92 step の serve パス) は 1 step あたり 392 本の線形 D2H を
// 個別に計時していたため、**計測そのものが測っている対象を変えていた**
// (instrumented 131.8 ms/step に対し実 KPI は 55.1 ms/token)。
// ここでは step 単位で間引く: N step に 1 step だけ区間を取る。
//
// 約束は 3 つ。
//   1. `DENNO_PROFILE` 未設定なら今までどおり素通し (環境変数は 1 回だけ読む)。
//   2. `N == 1` (既定) は今までの挙動と 1 bit も変わらない。
//   3. 間引いた step では **時刻を読まない**。表 C の行には「その step を通した
//      こと」(行の号数) だけを積み、ns は 1 ns も積まない。ns と号数は別の
//      カウンタなので、平均は必ず **実際に測った step / 呼び出し**で割る
//      (`serve_samples` / `serve_lin_samples`)。間引き率を掛けて全 step 相当に
//      直すことはしない — かけ算は分布の仮定で、号数は事実。
//
// 端数の食い違い (sampled な step で子が親を超える等) で `sl_step_other` が
// 負に振れないよう、負の残余は 0 に張り付けて**件数として**数える
// (`overflow_drops`)。0.0 が静かに「穴なし」として通るのを防ぐ。
// ---------------------------------------------------------------------------

/// s156 (Task B19): サンプリング周期の既定 (1 = 全 step を測る = 従来挙動)。
pub const SAMPLE_EVERY_DEFAULT: u64 = 1;

/// s156 (Task B19): 上限 (step 番号の飽和よけ。「測らない」状態は作らない)。
pub const SAMPLE_EVERY_MAX: u64 = 1_000_000;

/// s156 (Task B19): `DENNO_PROFILE_SAMPLE` の文字列を周期に読む (純関数)。
///
/// - `None` / 空 / 空白 / `0` / 数値でない → 既定 1 (従来挙動)
///   「計測しない」は `DENNO_PROFILE` の役目なので、ここへ無言で落とさない。
/// - `N > SAMPLE_EVERY_MAX` → `SAMPLE_EVERY_MAX`
pub fn parse_sample_every(raw: Option<&str>) -> u64 {
    let Some(s) = raw else {
        return SAMPLE_EVERY_DEFAULT;
    };
    match s.trim().parse::<u64>() {
        Ok(n) if n >= 1 => n.min(SAMPLE_EVERY_MAX),
        _ => SAMPLE_EVERY_DEFAULT,
    }
}

static SAMPLE_EVERY: OnceLock<u64> = OnceLock::new();

/// s156 (Task B19): サンプリング周期 N (1 = 全 step)。環境変数は 1 回だけ読む。
///
/// 呼び出し側は必ず `serve_on()` / `row_on()` で先に弾かれるので、`DENNO_PROFILE`
/// 未設定の実行ではこの関数も環境変数の読みも起きない (既定経路は完全 no-op)。
#[inline]
pub fn sample_every() -> u64 {
    #[cfg(test)]
    if let Some(n) = tests::force_sample_every() {
        return n;
    }
    *SAMPLE_EVERY
        .get_or_init(|| parse_sample_every(std::env::var("DENNO_PROFILE_SAMPLE").ok().as_deref()))
}

/// s156 (Task B19): この step 番号 (0 起点) を測るか。N=1 なら常に真。
#[inline]
pub fn sample_hit(step_index: u64) -> bool {
    step_index % sample_every().max(1) == 0
}

thread_local! {
    /// s156 (Task B19): いま開いている step の番号 (0 起点)。
    /// `None` = step の外 (prefill forward など `step_begin` を通らない区間) で、
    /// そこでは従来どおり計時する。
    static STEP_IDX: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// s156 (Task B19): **この区間・この行を計時してよいか**。既存の `serve_on()`
/// ゲートの後ろに 1 本置くだけで「sampled な step だけを測る」に切り替わる
/// (N=1 では常に真 = 従来挙動)。
#[inline]
pub fn row_on() -> bool {
    if !serve_on() {
        return false;
    }
    match STEP_IDX.with(|c| c.get()) {
        Some(i) => sample_hit(i),
        None => true,
    }
}

/// s156 (Task B19): いまの step 番号 (テスト・表示用)。step の外は `None`。
#[inline]
pub fn step_index() -> Option<u64> {
    STEP_IDX.with(|c| c.get())
}

/// s156 (Task B19): 計測は有効だが、この step は間引かれた (= ns を積まない)。
#[inline]
fn step_skipped() -> bool {
    match STEP_IDX.with(|c| c.get()) {
        Some(i) => !sample_hit(i),
        None => false,
    }
}

/// s156 (Task B19): 表 C で実際に **ns を測った step 数** (フェーズ `p`)。
/// 表示の分母はこれ (号数ではない) — 「測った step の平均」を出す。
#[inline]
pub fn serve_samples(p: usize) -> u64 {
    SL_SAMPLED[p].load(Ordering::Relaxed)
}

/// s156 (Task B19): 表 C で実際に **ns を測った線形呼び出し数** (フェーズ `p`)。
#[inline]
pub fn serve_lin_samples(p: usize) -> u64 {
    LIN_SAMPLED[p].load(Ordering::Relaxed)
}

/// s156 (Task B19): 区間を開けずに閉じた回数 (= 間引いた step で `span_begin` が
/// `None` を返した回数)。sampled step の値に待ち時間が寄っている兆候なので表示する。
#[inline]
pub fn skipped_spans() -> u64 {
    SKIPPED_SPANS.load(Ordering::Relaxed)
}

/// s156 (Task B19): 負の残余を 0 に張り付けた回数 (sampled step の内訳が全体を
/// 超えた回数)。実経路では恒等的に 0。
#[inline]
pub fn overflow_drops() -> u64 {
    OVERFLOW_DROPS.load(Ordering::Relaxed)
}

/// s156 (Task B19): 1 行だけのカウンタ表を const で作る。
const fn mk_counters1<const N: usize>() -> [AtomicU64; N] {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; N]
}

static SL_SAMPLED: [AtomicU64; Phase::N] = mk_counters1::<{ Phase::N }>();
static LIN_SAMPLED: [AtomicU64; Phase::N] = mk_counters1::<{ Phase::N }>();
static STEPS_SEEN: [AtomicU64; Phase::N] = mk_counters1::<{ Phase::N }>();
static SKIPPED_SPANS: AtomicU64 = AtomicU64::new(0);
static OVERFLOW_DROPS: AtomicU64 = AtomicU64::new(0);

/// s148: 表 C に 1 区間を積む。`calls` は「この区間が起きた回数」(ステップ由来の
/// 段は 1、線形呼び出し由来の段は 1)。表 C の行でない id は無視する。
///
/// s151 (Task B12): 段に属するサンプルの**実測号数**も数える (`SL_TICKS`)。COVERAGE の
/// 「samples」は `SPLIT_CALLS` ではなくこの実測号数を出す — `SPLIT_CALLS` の行は
/// 内訳 (`SL_STEP_SAMPLER` / `SL_STEP_DETOK` / `SL_LIN_OTHER`) なら号数と一致するが、
/// 全体行 (`SL_STEP` / `SL_LIN`) は「自分のサンプル + 内側の子を包んだ回数」でも
/// 増えるため、同じ数値で「何 sample 測ったか」を語れない (B11 実測で
/// `sl_lin_other` の calls が線形呼び出しの 3 倍 = 内訳 3 本 × 1 call になっていた)。
#[inline]
pub fn push(op: usize, ns: u64, calls: u64) {
    // s156 (Task B19): 間引いた step では **ns を積まず号数だけ** 積む。号数を
    // 隠すと「何 step のうち何 sample 見たか」が表から消え、0.0 が「その段が
    // 0 ms」と読めてしまう。
    if !serve_on() {
        return;
    }
    let skipped = step_skipped();
    let Some((SPLIT_SERVE_TABLE, r)) = split_slot(op) else {
        return;
    };
    let pi = phase().idx();
    if !skipped {
        SPLIT_NS[pi][SPLIT_SERVE_TABLE][r].fetch_add(ns, Ordering::Relaxed);
        SPLIT_NS[Phase::Both.idx()][SPLIT_SERVE_TABLE][r].fetch_add(ns, Ordering::Relaxed);
    }
    if calls > 0 {
        SPLIT_CALLS[pi][SPLIT_SERVE_TABLE][r].fetch_add(calls, Ordering::Relaxed);
        SPLIT_CALLS[Phase::Both.idx()][SPLIT_SERVE_TABLE][r].fetch_add(calls, Ordering::Relaxed);
        SL_TICKS[pi][r].fetch_add(calls, Ordering::Relaxed);
    }
}

/// s151 (Task B12): 表 C の段ごとの**実測サンプル号数** (`push` が積んだ号数)。
/// COVERAGE の samples はこちらを読む (SPLIT_CALLS は全体行だと子の回数が混ざる)。
static SL_TICKS: [[AtomicU64; SPLIT_SERVE_N]; Phase::N] = mk_counters::<SPLIT_SERVE_N>();

/// s154 (Task B15): 表 C の段ごとの**閉じた号数** (`span_end` が `Some` を受けて
/// 閉じた回数)。`sl_lay_embed` のように ns が 0 になりうる段でも「アダプタが
/// 閉じたか」を判定できるように、ns の会計とは別に持つ (ns には足さない)。
static SPLIT_END: [[[AtomicU64; SPLIT_ROWS]; SPLIT_TABLES]; Phase::N] = mk_counters2();

#[cfg(test)]
thread_local! {
    /// s151 (B12): 表 C のステップ入口 (`step_begin`) を開いたが `step_end` で
    /// まだ閉じていない数 (このスレッド)。1 ステップ = 1 サンプルを、サーバの
    /// 実呼び出し形 (scheduler の decode ループ / `step_chat` の prefill) で
    /// 検査するための**計測内部**の状態で、数値経路には現れない。
    static SL_OPEN: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// s148: 区間の開始 tick (計測 OFF は `None`)。呼び出し側は戻り値をそのまま
/// `span_end` / `fn_end` に渡す。
///
/// s156 (Task B19): 間引いた step では **時刻を読まず** `None` を返す
/// (`span_end` が `skipped_spans` を数え、ns は積まない)。
#[inline]
pub fn span_begin() -> Option<Instant> {
    if row_on() {
        Some(Instant::now())
    } else {
        None
    }
}

/// s148: 区間の終了を `op` に積む (1 サンプル)。`None` は素通し。
///
/// s156 (Task B19): 間引いた step では **ns を積まず、号数だけ** 積む (`push` が
/// ns を捨てる)。`None` (区間を開けなかった) は待ち時間が sampled step に寄って
/// いる兆候なので 1 つ数える — 静かに落とさない。数えるのは **step の中だけ**
/// (`step_index()` が `Some`): step の外の `None` は「計測が OFF だった」だけなので、
/// 数えると `skipped_spans` が「間引きの兆候」でなくなる。
#[inline]
pub fn span_end(t: Option<Instant>, op: usize) {
    if t.is_none() && step_index().is_some() {
        SKIPPED_SPANS.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(t0) = t {
        let ns = t0.elapsed().as_nanos() as u64;
        push(op, ns, 1);
        // s154 (Task B15): 区間を**閉じた**ことの号数。ns とは別に数える — 実機の
        // 1 区間は数十 ns より長いが、合成時刻を使うテストでは 0 ns になりうる。
        // COVERAGE が「アダプタが閉じたか」を ns で判定すると 0 ns の step で
        // MISS になるので、閉じた号数をこちらで持つ (ns の会計には足さない)。
        if serve_on() {
            let Some((SPLIT_SERVE_TABLE, r)) = split_slot(op) else {
                return;
            };
            SPLIT_END[phase().idx()][SPLIT_SERVE_TABLE][r].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// s156 (Task B19): **step の中だけで**区間を開く (`gap_fwd` の内訳用)。
///
/// `gap_fwd` は decode 1 歩の区間なので、その内訳も step の外では意味を持たない
/// (単体の forward 呼び出し・prefill・モデルのテストは step を開かない)。
/// ここで `None` を返せば、そういう経路では**時刻も読まず表 C の行も積まない** —
/// `sl_gap_*` の号数が「serve の step 数 × 層数」からずれない。
#[inline]
pub fn span_begin_in_step() -> Option<Instant> {
    if step_index().is_some() {
        span_begin()
    } else {
        None
    }
}

/// s153 (Task B14): 既に読み取り済みの ns を `op` に積む (1 サンプル)。
/// 段のサイトは `lin_stage_taken_ns` が読んだ ns をそのまま表 C へ渡す —
/// 子 (`sl_lin_*`) と残余の計算が**同じ 1 回のクロック読み**になるので、
/// 子の和が親 (`sl_lin`) を超えない (超えると残余が 0 に張り付き、表 C の
/// 「残余 = 全体 − 段」が嘘になる)。
#[inline]
pub fn span_end_ns(ns: u64, op: usize) {
    if !serve_on() {
        return;
    }
    push(op, ns, 1);
}

/// s169 (P8): 既に読めている ns を `calls` 回ぶんまとめて積む (1 区間 = 1 サンプル)。
///
/// `span_end` と同じく「閉じた号数」(`SPLIT_END`) も `calls` だけ進めるので、
/// COVERAGE の samples 列 (閉じた号数) と ms が食い違わない。prefill の位相
/// 計測は 1 リクエスト分をまとめて読む (層あたり 1 サンプル = `calls` = 層数)。
#[inline]
pub fn span_end_ns_calls(ns: u64, op: usize, calls: u64) {
    if calls == 0 || !serve_on() {
        return;
    }
    push(op, ns, calls);
    let Some((SPLIT_SERVE_TABLE, r)) = split_slot(op) else {
        return;
    };
    SPLIT_END[phase().idx()][SPLIT_SERVE_TABLE][r].fetch_add(calls, Ordering::Relaxed);
}

/// s154 (Task B15): `SL_LIN` の累積 ns (フェーズ `p`)。`sl_so_lin_sum` の素材。
#[inline]
fn lin_sum_ns(p: usize) -> u64 {
    serve_row_ns(p, SL_LIN)
}

/// s154 (Task B15): `sl_so_lin_sum` = **この step の間に sl_lin が積んだ分**。
/// `step_begin` の控え (`SO_SNAP`) の `sl_lin` 入口値を引く。サーバは複数セッションを
/// 交互に 1 歩ずつ進めるので、`sl_lin` の前 step までの累積をそのまま積むと
/// 「この step の other」より転記行が大きくなりうる (恒等式が壊れて見える)。
/// 控えが無い (step の外) ときは 0 を返す — 帰属先の step が無いので転記しない。
#[inline]
pub fn so_lin_sum_ns() -> u64 {
    if !serve_on() {
        return 0;
    }
    let p = phase().idx();
    match SO_SNAP.with(|c| c.get()) {
        Some(s) => lin_sum_ns(p).saturating_sub(s.lin0),
        None => 0,
    }
}

/// s154 (B15): `step_begin` の控え。`kids0` は実測の子 (`SL_SO_MEASURED`) 5 本の
/// 累積 ns、`lin0` は `sl_lin` の累積 ns — どちらも「この step の間に増えた分」を
/// 出すために引く。
#[derive(Clone, Copy)]
struct SoSnap {
    kids0: u64,
    lin0: u64,
}

thread_local! {
    /// s154 (B15): この step の入口で控えた値 (子 5 本の累積と `sl_lin` の累積)。
    static SO_SNAP: std::cell::Cell<Option<SoSnap>> = const { std::cell::Cell::new(None) };
}

/// s154 (Task B15): child 5 本の累積 ns (フェーズ `p`)。
#[inline]
fn so_measured_ns(p: usize) -> u64 {
    SL_SO_MEASURED
        .iter()
        .map(|&op| serve_row_ns(p, op))
        .sum()
}

/// s154 (Task B15): 1 ステップ分の開始 (表 C の `sl_step` の tick)。戻り値の 2 番目は
/// 「その時点で積まれていた表 C の各行の和」— サーバは複数セッションを交互に 1 歩
/// ずつ進めるため、あるステップの壁時計には別セッションの区間が重なる。終端で
/// 「自分が積んだ分」だけを引くための基準として使う (s154 で `sl_step_other` の
/// child も同じ 1 個の値に畳んだ)。計測 OFF は `None`。
///
/// s154 (Task B15): 併せて入口で child 5 本の累積を控える (`SO_SNAP`) — 実測の子は
/// `step_end` の後にも 1 本 (`sl_so_end`) 来るので、入口の全体累積だけを基準にすると
/// その 1 本が次の step の差分に紛れ込む。子ごとに 1 本なので控えとの差が正しい。
#[inline]
pub fn step_begin() -> Option<(Instant, SoCarry)> {
    if !serve_on() {
        return None;
    }
    #[cfg(test)]
    SL_OPEN.with(|c| c.set(c.get() + 1));
    split_step_begin();
    // s156 (Task B19): この step の番号を採り、測るかどうかを決める。間引いた
    // step では以降の `span_begin` が時刻を読まない (`row_on()` が偽)。
    let p = phase().idx();
    let idx = STEPS_SEEN[p].fetch_add(1, Ordering::Relaxed);
    let sampled = sample_hit(idx);
    if sampled {
        SL_SAMPLED[p].fetch_add(1, Ordering::Relaxed);
    }
    STEP_IDX.with(|c| c.set(Some(idx)));
    let now = Instant::now();
    // s155 (Task B16): 表 C の全 40 行の**行ごとの入口値**を控える。
    // s154 はここで「全行の和」1 個だけを控えていたため、入口の後に積まれる段が
    // 差に現れず `own` が恒等的に 0 になっていた (`SoCarry` の説明を参照)。
    let carry = SoCarry::new(p);
    // s154 (B15): この step の間の実測の子 (`SL_SO_MEASURED`) と `sl_lin` の増分を
    // 切り分けるための控え。
    SO_SNAP.with(|c| {
        c.set(Some(SoSnap {
            kids0: so_measured_ns(p),
            lin0: lin_sum_ns(p),
        }))
    });
    Some((now, carry))
}

/// s151 (B12): 表 C のステップ入口を開いたが閉じていない数 (テスト専用)。
/// 0 でない状態で次の `step_begin` が来たら、それは「1 ステップ 1 サンプル」の
/// 契約違反 (入れ子)。通常ビルドでは常に 0 (関数自体が無い)。
#[cfg(test)]
fn step_open_count() -> i64 {
    SL_OPEN.with(|c| c.get())
}

/// s155 (Task B16): `sl_step_other` から引く**実測の子**。
///
/// `step_end_tail` は `other = step − 名付きの兄弟` を出し、残余 `sl_so_resid` を
/// 「other − ここに並べた実測の子」で出す — 並べた行が `other` の内訳として
/// 排他でないと残余が負に張り付くので、以下は**この step の間に 1 度だけ**
/// 積まれる実測行に限る。
///
/// `sl_so_lin_sum` / `sl_so_lin_d2h` は転記 (線形呼び出しの内側) なので入れない —
/// 入れても `other` から引く量が二重になり、転記行の号数も step 数と一致しない。
/// `sl_so_gap_fwd` も入れない: per-op 経路では forward 入口 → 層ループ末尾の区間で、
/// **層ループ (`sl_step_lay_other`) の内側**にある (兄弟ではなく子のさらに内側)。
/// ここに入れると other から二重に引くことになる。
pub const SL_SO_MEASURED: [usize; 4] = [
    SL_SO_SAMPLE_PEN,
    SL_SO_STREAM,
    SL_SO_LOGITS,
    SL_SO_END,
];

/// s155 (Task B16): `step_begin` が控える**行ごとの入口値**。
///
/// s154 は「入口の全行の和」を 1 つの数で控えていたため、入口の後に積まれる段
/// (実経路では sl_lin / sl_step_lay_other / 層内訳が全部そう) が 1 つ残らず
/// `own` の差に現れず、`sl_step_other` が `sl_step` と同値になっていた。
/// 行ごとに控えれば `step_end` は「同じ行の末尾 − 入口」で自分の増分を出せる。
///
/// 表は固定長 (40 行) の配列で、`phase().idx()` の 1 面だけを写す — メモリ確保
/// なし・アロケータを踏まない (1 step 1 回)。
#[derive(Clone, Copy)]
pub struct SoCarry {
    p: usize,
    ns: [u64; SPLIT_SERVE_N],
    calls: [u64; SPLIT_SERVE_N],
}

impl SoCarry {
    /// いまの表 C の 1 面の写し (フェーズ `p`)。
    #[inline]
    fn new(p: usize) -> Self {
        let mut ns = [0u64; SPLIT_SERVE_N];
        let mut calls = [0u64; SPLIT_SERVE_N];
        for r in 0..SPLIT_SERVE_N {
            ns[r] = SPLIT_NS[p][SPLIT_SERVE_TABLE][r].load(Ordering::Relaxed);
            calls[r] = SL_TICKS[p][r].load(Ordering::Relaxed);
        }
        Self { p, ns, calls }
    }

    /// この行の入口値 (ns)。表 C の行でなければ 0。
    #[inline]
    fn get(&self, op: usize) -> u64 {
        match split_slot(op) {
            Some((SPLIT_SERVE_TABLE, r)) => self.ns[r],
            _ => 0,
        }
    }

    /// この行の入口号数 (ns の会計には足さない)。
    #[cfg(test)]
    #[inline]
    fn ticks(&self, op: usize) -> u64 {
        match split_slot(op) {
            Some((SPLIT_SERVE_TABLE, r)) => self.calls[r],
            _ => 0,
        }
    }

    #[cfg(test)]
    #[inline]
    fn phase_idx(&self) -> usize {
        self.p
    }
}

/// s155 (Task B16): `step_end` の残余から引く「層ループの内訳」(`sl_lay_*`)。
///
/// 親 (`sl_step_lay_other`) は `SL_GROUPS[0]` の名付き段として既に引いている。
/// その子は親の区間の**内側**で積まれるので、親の増分は「子の和 + 区間の外側」
/// に一致する (兄弟ではない)。ここで子の増分も引くのは、**親を閉じた後に積まれた
/// 子** (経路の取り違え) を `sl_step_other` から外に出さないため。
#[allow(dead_code)]
pub const SL_LAY_CHILDREN: [usize; 12] = [
    SL_LAY_NORM_IN,
    SL_LAY_GDN_PROJ,
    SL_LAY_GDN_SCAN,
    SL_LAY_FULL_QK,
    SL_LAY_FULL_CORE,
    SL_LAY_FULL_OPROJ,
    SL_LAY_FFN_GATE,
    SL_LAY_SWIGLU,
    SL_LAY_FFN_DOWN,
    SL_LAY_NORM_POST,
    SL_LAY_RESIDUAL,
    SL_LAY_EMBED,
];

/// s154 (B15): `sl_step_other` の child 行 (残余 `sl_so_resid` を除く)。
/// この step の間に child が積んだ ns = 末尾の和 − 入口の和 (`step_begin` の
/// payload が入口の和で、`step_end` がその差を引く)。
/// 線形段 (`sl_lin_launch` / `h2d` / `d2h` /
/// `outalloc`) は `step_begin` の payload (表 C 全 40 行の和) に**既に含まれて
/// いる**ので、末尾との差分は恒等的に 0 になる (入れても二重計上にはならないが、
/// 会計に何も足さない)。`sl_so_lin_sum` は「この step で sl_lin が積んだ分」を
/// そのまま積む転記行で、残余は「other 全体 − 下の 5 本」で出す。
pub const SL_SO_CHILD_ROWS: [usize; 5] = [
    SL_SO_SAMPLE_PEN,
    SL_SO_STREAM,
    SL_SO_LOGITS,
    SL_SO_GAP_FWD,
    SL_SO_END,
];

/// s155 (Task B16): `sl_step` から **名付きの兄弟**を引いて残余 (`other`) を出す。
///
/// s154 (B15) の `own` は `payload1 - payload0` (=「末尾の全行の和 − 入口の全行の
/// 和」) だった。入口の和 (`payload0`) は `push` が **同じ step の中で** 先に積んだ
/// 行 (実経路では sl_lin / sl_step_lay_other / 層内訳が全部そう) を既に含むので、
/// 差は毎 step ちょうど 0 になり、`other = ns = sl_step` になっていた (B15 の GPU
/// 実測: sl_step 131.737 / sl_step_other 131.737 ms/step)。CPU の合成 fixture は
/// 線形呼び出しを `step_begin` の前に積むため入口の和が 0 になり、この壊れ方を
/// 再現できていなかった。
///
/// 直し方は「同じ行の末尾 − 同じ行の入口 = この step が自分で積んだ分」:
///
///   own = Σ_{op ∈ SL_GROUPS[0].1 \\ {sl_step_other}} (row(op) − carry(op))
///
/// `SL_GROUPS[0].1` は「step の直下の排他な内訳」= prep / 投影 (sl_lin) / 層ループ
/// (`sl_step_lay_other`) / final / logits_d2h / sampler / detok。投影 (`sl_lin`) は
/// 層ループの内側で走るが、`sl_step_lay_other` の区間は**投影を除いた**層ループ
/// (engine が投影の前後で区間を切る) なので、両者は兄弟であって包含ではない。
/// 層内訳 (`sl_lay_*`) は `sl_step_lay_other` の内側 (包含) なのでここでは引かない
/// — 引くと二重減算で other が 0 に張り付く。
#[inline]
fn step_own_ns(p: usize, carry: &SoCarry) -> u64 {
    let mut own = 0u64;
    for &op in SL_GROUPS[0].1 {
        if op != SL_STEP_OTHER {
            own += serve_row_ns(p, op).saturating_sub(carry.get(op));
        }
    }
    own
}

/// s148: 1 ステップ分の終了。`sl_step` に 1 サンプル積み、内訳で説明できなかった
/// 分を `sl_step_other` へ**必ず**入れる (未帰属が 0 ms として静かに通らない)。
/// `None` は素通し (計測 OFF は時刻も読まない)。
///
/// s155 (Task B16): `other` = **step − 名付きの兄弟** (1 本ずつ、同じ行の差)。
/// s154 (B15) は `own = payload1 - payload0` としていた。`payload1` は
/// `SL_GROUPS[0].1` (8 行の内訳) の和、`payload0` は `step_begin` が控えた
/// **表 C 全 40 行の和** — 比べる集合が違うので差は恒等的に 0 になり、
/// `other = ns = sl_step` になっていた (B15 の GPU 実測: sl_step 131.737 /
/// sl_step_other 131.737 ms/step)。ここでは `SoCarry` (行ごとの入口値) を使って
/// 同じ行の「末尾 − 入口」= **この step が自分で積んだ分**を引く。
///
/// `sl_lin` も名付きの兄弟として引く: 投影の区間は `sl_step_lay_other` (層ループの
/// 区間) の**外側**で切られている (`span_end(st)` → 投影 → `span_begin` の順) ので
/// prep / lin / lay_other / final / logits_d2h / sampler / detok は排他な兄弟になる。
#[inline]
pub fn step_end(st: Option<(Instant, SoCarry)>) {
    let Some((t0, carry)) = st else { return };
    #[cfg(test)]
    SL_OPEN.with(|c| c.set(c.get() - 1));
    let ns = t0.elapsed().as_nanos() as u64;
    push(SL_STEP, ns, 1);
    let p = phase().idx();
    let own = step_own_ns(p, &carry);
    // s156 (Task B19): 間引きで子が親を超えたら 0 に張り付けて件数で数える
    // (0.0 を「穴なし」と読ませない)。
    let other = ns.checked_sub(own).unwrap_or_else(|| {
        OVERFLOW_DROPS.fetch_add(1, Ordering::Relaxed);
        0
    });
    let snap = SO_SNAP.with(|c| c.replace(None));
    let kids_own = match snap {
        Some(s) => so_measured_ns(p).saturating_sub(s.kids0),
        None => 0,
    };
    // s155 (Task B16): `sl_step_other` の push は **この step の締め** まで待つ。
    // 理由は 2 つ。① 兄弟 (`sl_step_lay_other` 等) の一部は `step_end` の後に
    // 閉じる (per-op 経路は 投影 → サンプラ → step_end → 次の投影 の順) ので、
    // ここで確定すると自分の step の段が 1 本も差に現れない (s154 の 0 化と
    // 同じ罠を別の形で踏む)。② `sl_so_resid` は `other` の後で `push` する行なので、
    // 同じ関数から読むと自分の分がまだ入っていない。
    TAIL_CARRY.with(|c| {
        c.set(Some(TailCarry {
            p,
            other,
            own,
            kids_own,
            sib0: carry,
        }))
    });
}

/// s155 (Task B16): `step_end` が控える「この step の締め」の材料。
/// `other` は `step_end` の時点の暫定値 (step の真ん中で閉じた親がまだ入っていない)、
/// `own` は入口から `step_end` までに積まれた名付き段の和、`kids_own` は実測の子の和、
/// `sib0` は兄弟の**入口の累積** (`step_begin` の `SoCarry`)。
/// `step_end_tail` が「兄弟の今の累積 − 入口」を引き直して確定する。
#[derive(Clone, Copy)]
struct TailCarry {
    p: usize,
    other: u64,
    own: u64,
    kids_own: u64,
    sib0: SoCarry,
}

thread_local! {
    /// s155 (Task B16): 直近の `step_end` の控え (`step_end_tail` が 1 回だけ食う)。
    static TAIL_CARRY: std::cell::Cell<Option<TailCarry>> = const { std::cell::Cell::new(None) };
}

/// s155 (Task B16): `sl_step` から **名付きの兄弟**を引いて残余 (`other`) を出す。
///
/// `other` = `sl_step` − (名付きの兄弟の和)。兄弟は `SL_GROUPS[0].1` のうち
/// `sl_step_other` 自身を除いた 7 行 (`sl_step_prep` / `sl_lin` /
/// `sl_step_lay_other` / `sl_step_final` / `sl_step_logits_d2h` /
/// `sl_step_sampler` / `sl_step_detok`)。`: `sl_so_*` は**その other の内側**の
/// 子なので兄弟には含めない。
///
/// `sl_lin` も兄弟として引く。B15 のコメントは「投影は層ループの内側」としていたが、
/// 実際の per-op 経路 (`model/qwen35.rs` の decode ループ) は
/// `span_end(st)` → 投影 → `span_begin()` の順で区間を切るので、投影は層ループの
/// 区間の**外側** = step の直下の兄弟になる。引かないと「other = step − 5 行」に
/// 投影の分が残り、other が step の 8 割に膨らむ。
///
/// 締め (`step_end_tail`) で**入口の控えを使わずに累積から引き直す**のは、
/// per-op 経路の 1 step が「投影 (前 step の logits D2H) → `sl_so_*` → サンプラ →
/// `step_end` → 次の投影 (この step の forward)」の順に走り、**兄弟の一部が
/// `step_end` の後に閉じる**ため。ここで「今の累積 − 入口」を引くと、その step の
/// 兄弟を 1 本残らず引ける (cumulative 差は括弧の外側で相殺する)。
#[inline]
pub fn step_end_tail() {
    let Some(c) = TAIL_CARRY.with(|x| x.replace(None)) else {
        return;
    };
    let p = c.p;
    let mut own_now = 0u64;
    for &op in SL_GROUPS[0].1 {
        if op != SL_STEP_OTHER {
            own_now += serve_row_ns(p, op).saturating_sub(c.sib0.get(op));
        }
    }
    // `step_end` の後に兄弟が 1 本も来なければ `own_now == c.own` で other は不変。
    // 来た経路では `own_now > c.own` になり、その分だけ other から引く。
    let other = c.other.saturating_sub(own_now.saturating_sub(c.own));
    // s156 (Task B19): 残余の push は **`STEP_IDX` を落とす前**に行う — `push` は
    // 「間引いた step か」を `STEP_IDX` で判定するので、先に落とすと間引いた step の
    // 残余だけが計時されて内訳が食い違う。
    push(SL_STEP_OTHER, other, 1);
    push(SL_SO_RESID, other.saturating_sub(c.kids_own), 1);
    // この step は閉じた (外側 = サンプラ / 次の投影 の地は step の外)。
    STEP_IDX.with(|x| x.set(None));
}

/// s148: 表 C の `whole` 行と、その内訳の和との差 (ns)。負 (内訳が全体を超える =
/// 区間の切り方の誤り) は 0 にして「穴」だけを見せる。
#[inline]
pub fn step_gap(whole: usize) -> u64 {
    let p = phase().idx();
    let total = serve_row_ns(p, whole);
    let Some((_, parts)) = SL_GROUPS.iter().find(|(w, _)| *w == whole) else {
        return 0;
    };
    let sum: u64 = parts.iter().map(|&op| serve_row_ns(p, op)).sum();
    total.saturating_sub(sum)
}

/// s148: 表 C の 1 行の累積 ns (フェーズ別)。
#[inline]
pub fn serve_row_ns(p: usize, op: usize) -> u64 {
    match split_slot(op) {
        Some((SPLIT_SERVE_TABLE, r)) => SPLIT_NS[p][SPLIT_SERVE_TABLE][r].load(Ordering::Relaxed),
        _ => 0,
    }
}

/// s148: 表 C の 1 行の累積サンプル数 (フェーズ別)。
///
/// s151 (Task B12): **全体行 (`sl_step` / `sl_lin`) には内側の子の回数が混ざる**
/// (`add_sample` が子を包む全体へも calls を足す、s148 の設計)。「実際に何 sample
/// 測ったか」を語る値は `serve_row_ticks` を使うこと。内訳行 (`sl_lin_h2d` 等) は
/// 1 sample = 1 号なので両者は一致する。
#[inline]
pub fn serve_row_calls(p: usize, op: usize) -> u64 {
    match split_slot(op) {
        Some((SPLIT_SERVE_TABLE, r)) => {
            SPLIT_CALLS[p][SPLIT_SERVE_TABLE][r].load(Ordering::Relaxed)
        }
        _ => 0,
    }
}

/// s151 (Task B12): 表 C の 1 行の**実測サンプル号数** (フェーズ別)。`push` が
/// 積んだ回数だけで、内側の子の回数は混ざらない。COVERAGE の samples と
/// 「samples == steps / lin calls」の判定はこちらを見る。
#[inline]
pub fn serve_row_ticks(p: usize, op: usize) -> u64 {
    match split_slot(op) {
        Some((SPLIT_SERVE_TABLE, r)) => SL_TICKS[p][r].load(Ordering::Relaxed),
        _ => 0,
    }
}

/// s154 (Task B15): 表 C の段が `span_end` で**閉じられた**号数 (フェーズ `p`)。
/// ns とは独立なので、合成時刻テスト (0 ns) でも「アダプタが置いたか」が判る。
#[inline]
pub fn serve_row_ends(p: usize, op: usize) -> u64 {
    match split_slot(op) {
        Some((SPLIT_SERVE_TABLE, r)) => SPLIT_END[p][SPLIT_SERVE_TABLE][r].load(Ordering::Relaxed),
        _ => 0,
    }
}

/// s154 (Task B15): 表 C の「全体行」の実測号数 (steps / lin calls の分母)。
/// `serve_expected` と `step_end` の残余計算が同じ量を読むために切り出した
/// (s151 から `serve_expected` の内側にあった式そのもの)。
#[inline]
pub fn serve_ticks_for(p: usize) -> u64 {
    serve_row_ticks(p, SL_STEP)
}

/// s148: 1 ステップ分の区間分けを開いた回数 (COVERAGE の「何 step の内訳を
/// 実際に分割できたか」)。ステップ全体 (`step_begin`) とは独立に数える —
/// 分割の付け漏れはここが全体より小さくなるので見える。
static SL_SPLIT_STEPS: AtomicU64 = AtomicU64::new(0);

/// s148: 分割を開いたステップ数を 1 つ数える (`step_begin` が呼ぶ)。
#[inline]
pub fn split_step_begin() {
    if serve_on() {
        SL_SPLIT_STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// s148: 分割を開いたステップ数。
pub fn split_step_count() -> u64 {
    SL_SPLIT_STEPS.load(Ordering::Relaxed)
}

/// s148: 1 本の HTTP リクエストの壁時計 (ns)。サーバのハンドラが要求の入口から
/// 出口までを 1 回だけ積む。COVERAGE が「プロファイルの和 / リクエストの壁時計」
/// を出すための分母で、テレメトリでは数えない (リクエスト単位の観測のみ)。
///
/// s151 (Task B12): そのリクエストの壁時計を **prefill 分 / decode 分** に割って
/// 持つ。q4 serve の 1 リクエストは「prefill 1 回 + decode N 回」なので、全壁時計を
/// 両フェーズの分母に使うと decode の shares が prefill の秒数で薄まる
/// (B11 実測: decode の 6,000 ms に対しプリフィル込みの壁は 12.1 s)。フェーズ別の
/// 壁 = `prefill 分 + そのフェーズのステップ内側時間` として per-request で積む。
static REQ_NS: AtomicU64 = AtomicU64::new(0);
static REQ_N: AtomicU64 = AtomicU64::new(0);
/// s151: フェーズ別の内訳 (添字は `Phase::Prefill` / `Phase::Decode`)。
/// 0 = そのフェーズのステップ内側時間の累積 (表 C の `sl_step` と同じ区間)
/// 1 = リクエストの壁時計のうちそのフェーズへ割り当てる分 (prefill 分 / decode 分)
static REQ_PHASE_NS: [[AtomicU64; 2]; Phase::N] = mk_counters::<2>();
static REQ_PHASE_N: [[AtomicU64; 2]; Phase::N] = mk_counters::<2>();

/// s148: 1 リクエスト分の壁時計を積む。
#[inline]
pub fn add_request_wall(ns: u64) {
    if enabled() {
        REQ_NS.fetch_add(ns, Ordering::Relaxed);
        REQ_N.fetch_add(1, Ordering::Relaxed);
    }
}

/// s151 (Task B12): サーバのハンドラが要求の入口 / 出口で呼ぶ。戻り値は
/// 「prefill ステップで DOM 区間が動いた集合」で、出口で 1 本の壁時計を
/// prefill 分と decode 分に割るための勘定。**計測 ON のときだけ**時刻を読み、
/// 返す票は `(req_t, 入口の表 C スナップショット, 出口の表 C スナップショット)`。
/// s151 (Task B12): サーバのハンドラ (scheduler の Run) が要求の入口 / 出口で使う
/// 壁時計の票。**中身は非公開** (壊れた票を作れないように) で、`request_begin` →
/// `request_end` の 1 往復にだけ使う。計測 OFF の `request_begin` は `None` を返す。
pub type RequestMarker = RequestMark;

#[derive(Clone, Copy)]
pub struct RequestMark {
    t0: Instant,
    /// 入口時点の (prefill の `sl_step` ns, prefill の非 DOM 内訳 ns, decode の `sl_step` ns)
    before: (u64, u64, u64),
}

/// 表 C の「ステップ内側」のうち、DOM 区間 (`sl_step_detok`) を除いた和。
/// q4 serve の 1 ステップは `sl_step_prep → sl_lin/lay_other → final → sampler →
/// detok` で、prefill 側は `sl_step_prep` (forward 全体) を積む。入口／出口の差分が
/// 「このリクエストが回した non-DOM 時間」になる。
#[inline]
fn serve_step_non_dom_ns(p: usize) -> u64 {
    let sum = |ops: &[usize]| -> u64 { ops.iter().map(|&op| serve_row_ns(p, op)).sum() };
    let has = |ops: &[usize]| -> bool { ops.iter().any(|&op| serve_row_ticks(p, op) > 0) };
    let mut ns = 0;
    if has(SL_GROUPS[0].1) {
        let all = sum(SL_GROUPS[0].1);
        let dom = sum(SL_GROUPS[1].1);
        ns += all.saturating_sub(dom);
    }
    // prefill 側は `sl_step_prep` 1 本 (= forward 全体)。入れ子の投影は数えない。
    if serve_row_ticks(p, SL_STEP_PREP) > 0 && serve_row_ticks(p, SL_STEP) == 0 {
        ns += serve_row_ns(p, SL_STEP_PREP);
    }
    ns
}

/// s151 (B12): リクエストの入口。計測 OFF は `None` (クロック読みなし・表読みなし)。
#[inline]
pub fn request_begin() -> Option<RequestMark> {
    if !enabled() {
        return None;
    }
    Some(RequestMark {
        t0: Instant::now(),
        before: (
            serve_row_ns(Phase::Prefill.idx(), SL_STEP),
            serve_step_non_dom_ns(Phase::Prefill.idx()),
            serve_row_ns(Phase::Decode.idx(), SL_STEP),
        ),
    })
}

/// s151 (B12): リクエストの出口。壁時計を 1 本積み、そのうち **prefill 区間**
/// (`sl_step_prep` = forward 全体がこのリクエストで増えた分) を prefill へ、
/// 残りを decode へ割り当てる。1 リクエスト = 1 本だけ数える。
/// `wall(prefill) + wall(decode) == wall(全体)` が常に成り立つ。
#[inline]
pub fn request_end(mark: Option<RequestMark>) {
    let Some(m) = mark else { return };
    let ns = m.t0.elapsed().as_nanos() as u64;
    REQ_NS.fetch_add(ns, Ordering::Relaxed);
    REQ_N.fetch_add(1, Ordering::Relaxed);
    let after_pf = serve_step_non_dom_ns(Phase::Prefill.idx());
    let pf = after_pf.saturating_sub(m.before.1).min(ns);
    let (pi, di) = (Phase::Prefill.idx(), Phase::Decode.idx());
    let after_dec = serve_row_ns(Phase::Decode.idx(), SL_STEP);
    REQ_PHASE_NS[pi][0].fetch_add(pf, Ordering::Relaxed);
    REQ_PHASE_NS[di][0].fetch_add(after_dec.saturating_sub(m.before.2), Ordering::Relaxed);
    REQ_PHASE_NS[pi][1].fetch_add(pf, Ordering::Relaxed);
    REQ_PHASE_NS[di][1].fetch_add(ns - pf, Ordering::Relaxed);
    REQ_PHASE_N[pi][0].fetch_add(1, Ordering::Relaxed);
    REQ_PHASE_N[di][0].fetch_add(1, Ordering::Relaxed);
}

/// s148: (リクエスト本数, 壁時計の和 ns)。
pub fn request_wall() -> (u64, u64) {
    (
        REQ_N.load(Ordering::Relaxed),
        REQ_NS.load(Ordering::Relaxed),
    )
}

/// s151 (B12): フェーズ別の壁時計 (`(リクエスト本数, 壁 ns)`)。
/// prefill / decode の shares の分母はここから取る (両者の和が全壁時計になる)。
pub fn request_wall_for(p: usize) -> (u64, u64) {
    let k = if p == Phase::Decode.idx() { Phase::Decode.idx() } else { Phase::Prefill.idx() };
    (
        REQ_PHASE_N[k][0].load(Ordering::Relaxed),
        REQ_PHASE_NS[k][1].load(Ordering::Relaxed),
    )
}

/// s148: decode のステップ数 (バケットの最大 call 数 = ヘッダと同じ定義)。
///
/// s151 (Task B12): **表 C の `sl_step` の実測サンプル号数**に同じ値 (`serve_ticks`)
/// を返す — COVERAGE の「expected steps」と「samples」が同じ量であることを型で
/// 保証するため。全バケットの最大 call 数は「どの op も毎ステップ通る」前提に
/// 依存するので、COVERAGE の分母には使わない (B11 実測では decode 92 step に対して
/// プロセス全体の最大 call 数 57,780 が分母に入り、期待とサンプルが別の量だった)。
pub fn decode_steps(p: usize) -> u64 {
    serve_ticks(p)
}

/// s151 (Task B12): 表 C の `sl_step` を積んだ回数 (このフェーズで実際に
/// 分割できたステップ数)。COVERAGE の「steps」と期待値はこの 1 本から取る。
pub fn serve_ticks(p: usize) -> u64 {
    serve_row_ticks(p, SL_STEP)
}

/// s148: 表の添字 → フェーズ (Both は Prefill として扱う = 期待値は prefill 側)。
fn phase_for(p: usize) -> Phase {
    if p == Phase::Decode.idx() {
        Phase::Decode
    } else {
        Phase::Prefill
    }
}

/// s148: **実 q4 ディスパッチの call 数** (表 C の線形行の分母)。`lin_stage` の
/// ように wrapper で数えるのではなく、`maybe_gpu_linear_q4_into` の入口
/// (= 本番サーバが実際に通る唯一の q4 線形入口) で数える。
///
/// s148 追記: ここは 1 本のカウンタで、`phase()` が `Both` でも **1 回だけ** 加算する
/// (`Both` 行が事前に埋まっている状態で `Both` にもう一度足すと、この行だけ 2 倍に
/// なる — `denno-server` のビルドでは `#[cfg(test)]` の上書きが無く、`Both` は
/// 実際に `Prefill` / `Decode` から作られる行なので、二重加算は起きない)。
static LIN_DISPATCH_CALLS: [[AtomicU64; 2]; Phase::N] = mk_counters::<2>();

/// s148: 実 q4 ディスパッチ入口を 1 回通ったことを数える (計測 ON のみ)。
#[inline]
pub fn count_lin_dispatch() {
    if serve_on() {
        LIN_DISPATCH_CALLS[phase().idx()][0].fetch_add(1, Ordering::Relaxed);
    }
}

/// s148: 実 q4 ディスパッチの call 数 (フェーズ別)。`Both` (累積) は
/// `Prefill` / `Decode` の和として引く (行を二重に持たない)。
pub fn lin_dispatch_calls(phase: Phase) -> u64 {
    match phase {
        Phase::Both => lin_dispatch_calls(Phase::Prefill) + lin_dispatch_calls(Phase::Decode),
        _ => LIN_DISPATCH_CALLS[phase.idx()][0].load(Ordering::Relaxed),
    }
}

/// s148: 表 C の「1 step あたり / 1 call あたりの期待値」(COVERAGE の分母)。
/// 戻り値は `(表, 行, 期待サンプル数, 期待の名前)`。期待値は **表 C が自分で
/// 駆動した件数** (ステップは `sl_step` の実測号数、線形は実 q4 ディスパッチの
/// call 数) で、行が 0 サンプルなのに期待が 1 以上なら COVERAGE MISS として出す。
///
/// s151 (B12) の変更点:
///   (1) `steps` の分母は「プロセス全体のステップカウンタ」ではなく **表 C の
///       `sl_step` の実測号数** (`serve_ticks`) にした — 期待とサンプルが同じ量。
///   (2) 行 (`SL_STEP_SAMPLER` / `SL_STEP_FINAL`) は列挙の対象のまま、`SL_STEP_PREP`
///       は **prefill だけ**を期待する (decode では `sl_lin` / `sl_step_lay_other`
///       が前段を覆うので `sl_step_prep` は積まれない — 期待すると偽の MISS になる)。
///   (3) `sl_lin` は **`sl_lin_other` と同じ期待** (= 線形呼び出し数) しか持たないため
///       per-row の MISS は出さない (両者は同じ push を共有する)。
pub fn serve_expected(p: usize) -> Vec<(usize, usize, u64, &'static str)> {
    let steps = serve_ticks_for(p);
    let lin = lin_dispatch_calls(phase_for(p));
    let is_decode = p == Phase::Decode.idx();
    let prep = if is_decode { 0 } else { serve_row_ticks(p, SL_STEP_PREP) };
    vec![
        (SPLIT_SERVE_TABLE, SL_STEP, steps, "steps"),
        // prefill は forward 1 回 = `sl_step_prep` 1 号 (`sl_step` は積まれない)。
        // decode の「前段」は `sl_step_prep` ではなく投影 + 層ループの残余。
        (SPLIT_SERVE_TABLE, SL_STEP_PREP, prep, "prefill"),
        (SPLIT_SERVE_TABLE, SL_LIN, lin, "lin calls"),
        (SPLIT_SERVE_TABLE, SL_STEP_FINAL, steps, "steps"),
        // `SL_STEP_LOGITS_D2H` はこの一覧に入れない: q4 serve の logits D2H は
        // `logits()` の内側 (final 段) で起き、独立した区間としては存在しない
        // (= 行は予約のまま 0 サンプル。予約を消すと過去ログの id がずれる)。
        (SPLIT_SERVE_TABLE, SL_STEP_SAMPLER, steps, "steps"),
    ]
}

/// s151 (B12): `sl_lin_other` のサンプル数 = **線形呼び出し 1 回につき 1 号**。
///
/// s148 は内訳の号数 (`SPLIT_CALLS`) を出していたため、1 回の呼び出しのうち
/// 3 か所 (入口の地 / 重み登録 / ピン留め) がそれぞれ other を閉じた回は
/// 3 号に見えていた (B11 実測: 線形 36,090 call に対し 108,270 号 = ちょうど 3 倍)。
/// ここでは「other を 1 回以上閉じた**呼び出し**の数」を数える — 段の号数ではなく
/// 呼び出しの数なので、`lin calls` と直接突き合わせられる。
pub fn lin_other_samples(p: usize) -> u64 {
    LIN_OTHER_CALLS[p][0].load(Ordering::Relaxed)
}

/// s151 (B12): `sl_lin_other` を閉じた**呼び出し**の数 (フェーズ別)。
static LIN_OTHER_CALLS: [[AtomicU64; 1]; Phase::N] = mk_counters::<1>();

thread_local! {
    /// s151 (B12): いま走っている線形呼び出しの内側の積算。
    ///   `(起動したか, sl_lin_other の ns, sl_lin_other の号数)`
    /// **1 呼び出しにつき other は最大 1 号** — 内側の何か所で閉じても、ns を
    /// 足し合わせたうえで `lin_call_end` が 1 号だけ積む (s148 は入口の地 /
    /// 重み登録 / ピン留めの 3 か所がそれぞれ 1 号積み、36,090 call に対し
    /// 108,270 号 = 3 倍になっていた)。
    static LIN_STATE: std::cell::RefCell<LinAcc> =
        const {
            std::cell::RefCell::new(LinAcc {
                open: false,
                other_ns: 0,
                other_n: 0,
                taken_ns: 0,
                taken_n: 0,
            })
        };
}

/// s151 (B12): 線形呼び出し 1 回の内側の積算 (thread-local)。
struct LinAcc {
    /// `lin_call_begin` から `lin_call_end` の間か
    open: bool,
    /// この呼び出しで閉じた `sl_lin_other` **内側区間**の ns の和
    /// (s153 以降は会計の主役ではない — 残余は `call_ns - taken_ns` で出し、
    /// これは明示的な other 区間として残すだけ。)
    other_ns: u64,
    /// この呼び出しで other を閉じた回数 (2 回以上なら会計の誤り = テストで見る)
    other_n: u64,
    /// s153 (Task B14): この呼び出しで**実際に測った段** (`SL_LIN_LAUNCH` /
    /// `SL_LIN_H2D` / `SL_LIN_D2H` / `SL_LIN_OUTALLOC`) の ns の和。
    /// 残余 `sl_lin_other` は `call_ns - taken_ns` で出る (実質の加算なので
    /// 1/16 ns の切り捨て誤差も出ない)。
    taken_ns: u64,
    /// s153 (Task B14): `taken_ns` を積んだ段の本数 (0 なら「何も測っていない
    /// 呼び出し」= 全体が残余。テストと COVERAGE の突き合わせに使う)。
    taken_n: u64,
}

/// s151 (B12): 線形呼び出しの入口。`sl_lin` の全体 tick と対で 1 回だけ呼ぶ。
/// 計測 OFF は `None` (クロック読みなし)。
#[inline]
pub fn lin_call_begin() -> Option<Instant> {
    if !serve_on() {
        return None;
    }
    LIN_STATE.with(|c| {
        let mut a = c.borrow_mut();
        a.open = true;
        a.other_ns = 0;
        a.other_n = 0;
        a.taken_ns = 0;
        a.taken_n = 0;
    });
    Some(Instant::now())
}

/// s151 (B12): 線形呼び出しの内側で「別の段を測った」ことを記録する
/// (号数だけ。s153 以降は `lin_stage_taken_ns` が本来の口)。
#[inline]
pub fn lin_stage_taken() {
    #[cfg(test)]
    LIN_STAGE_TAKEN.with(|c| c.set(c.get() + 1));
}

/// s153 (Task B14): 線形呼び出しの内側で**段を 1 本測った**ことを記録する
/// (`span_end` に渡したのと同じ開始 tick をそのまま渡す)。
///
/// ここで積んだ ns の和が `taken_ns` で、`lin_call_end` は
/// `sl_lin_other = call_ns - taken_ns` を出す — つまり
/// `sl_lin == sl_lin_launch + sl_lin_h2d + sl_lin_d2h + sl_lin_outalloc + sl_lin_other`
/// が 1 call ごとに成り立つ。s151 までは other が「明示的に other として
/// 閉じた区間」だけだったため、段を測っている呼び出しでも other が全体と
/// 等しかった (B14 実測 17:21Z: sl_lin 2225.3 ms = sl_lin_other 2225.3 ms)。
///
/// **クロック読みは 1 回** — 段の ns をここで読み、表 C へは
/// `span_end_ns(ns, op)` で同じ値を渡す (2 回読むと後から読む方が必ず長くなり、
/// 子の和が親を超える = 残余が 0 に張り付く)。
/// 計測 OFF は `None` を受けて何もしない (時刻も読まない)。
#[inline]
pub fn lin_stage_taken_ns(t: Option<Instant>) -> u64 {
    #[cfg(test)]
    LIN_STAGE_TAKEN.with(|c| c.set(c.get() + 1));
    if !serve_on() {
        return 0;
    }
    let Some(t0) = t else { return 0 };
    let ns = t0.elapsed().as_nanos() as u64;
    lin_stage_add(ns);
    ns
}

/// s153 (Task B14): 測った段の ns を明示的に積む (`lin_stage_taken_ns` の下請け)。
/// 呼び出しの外 (begin 前 / end 後) に来た分は捨てる — 別の呼び出しの
/// 残余を汚さない。
#[inline]
pub fn lin_stage_add(ns: u64) {
    if !serve_on() {
        return;
    }
    LIN_STATE.with(|c| {
        let mut a = c.borrow_mut();
        if !a.open {
            return;
        }
        a.taken_ns += ns;
        a.taken_n += 1;
    });
}

/// s153 (Task B14): この呼び出しで測った段の ns と本数 (テスト用)。
#[cfg(test)]
fn lin_taken() -> (u64, u64) {
    LIN_STATE.with(|c| (c.borrow().taken_ns, c.borrow().taken_n))
}

#[cfg(test)]
thread_local! {
    /// s151 (B12): `lin_stage_taken` の回数 (テスト専用・会計には使わない)。
    static LIN_STAGE_TAKEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// s151 (B12): `sl_lin_other` の内側区間を閉じる。**積むのは出口
/// (`lin_call_end`) だけ** — ここでは「この呼び出しの other」へ ns と回数を
/// 足すだけなので、内側で何回呼ばれても表 C の `sl_lin_other` は 1 号になる。
///
/// s153 (Task B14): ここで足した ns は**もう残余の値にはならない** (残余は
/// `call_ns - taken_ns`)。この口は「other として明示したい区間」の記録として
/// 残し、号数 (`lin_other_samples`) の会計を壊さないための fallback に使う。
#[inline]
pub fn lin_other_push(ns: u64) {
    if !serve_on() {
        return;
    }
    LIN_STATE.with(|c| {
        let mut a = c.borrow_mut();
        a.other_ns += ns;
        a.other_n += 1;
    });
}

/// s151 (B12): 線形呼び出しの内側の `sl_lin_other` 区間を閉じる (`span_begin` の
/// 戻り値をそのまま渡す)。
#[inline]
pub fn lin_other_close(t: Option<Instant>) {
    if let Some(t0) = t {
        lin_other_push(t0.elapsed().as_nanos() as u64);
    }
}

/// s151 (B12): 線形呼び出しの出口。`sl_lin` の全体を 1 号積み、`sl_lin_other`
/// は**この呼び出しで 1 回以上閉じたときだけ 1 号**積む。子 (`sl_lin_*`) の
/// 号数は最大でも call 数 + 1 (other) で、`sl_lin` の号数は常に call 数。
#[inline]
pub fn lin_call_end(t0: Option<Instant>) {
    let Some(t0) = t0 else { return };
    lin_call_end_at(t0.elapsed().as_nanos() as u64);
}

/// s153 (Task B14): 線形呼び出しの出口の会計 (経過 ns を渡す形)。`lin_call_end` と
/// テストが**同じ 1 本の算術**を通るように切り出した — 残余は
/// `ns - taken_ns`、`sl_lin` / `sl_lin_other` はどちらも 1 号。
#[inline]
pub fn lin_call_end_at(ns: u64) {
    let taken_ns = LIN_STATE.with(|c| {
        let mut a = c.borrow_mut();
        a.open = false;
        a.taken_ns
    });
    push(SL_LIN, ns, 1);
    // s153 (Task B14): 残余 = 全体 − **実際に測った段** (`SL_LIN_LAUNCH` /
    // `SL_LIN_H2D` / `SL_LIN_D2H` / `SL_LIN_OUTALLOC` の ns の和)。
    // 段を 1 本も測らなかった呼び出し (CPU フォールバック等) は全体が残余 =
    // 今日までの挙動と同じ。明示的に other として閉じた区間
    // (`lin_other_close` の `other_ns`) は残余の中身の一部なので、残余の値には
    // 足さない (s151 まではこれが主値だったため、段を測っている呼び出しでも
    // other == 全体になっていた)。`min(ns)` は「段の和 > 全体」= 会計の穴が
    // 出たときの番人で、主値ではない。
    let residual = ns.saturating_sub(taken_ns).min(ns);
    push(SL_LIN_OTHER, residual, 1);
    let pi = phase().idx();
    LIN_OTHER_CALLS[pi][0].fetch_add(1, Ordering::Relaxed);
    LIN_OTHER_CALLS[Phase::Both.idx()][0].fetch_add(1, Ordering::Relaxed);
}

/// s148: 表 C の 1 行の表示名 (`SL_` の id から引く)。
pub fn serve_row_name(op: usize) -> Option<&'static str> {
    split_row_name(SPLIT_SERVE_TABLE, op - SPLIT_SERVE_FIRST).filter(|_| {
        (SPLIT_SERVE_FIRST..SPLIT_SERVE_FIRST + SPLIT_SERVE_N).contains(&op)
    })
}

/// s138 (Task B6): 1 ステップ分の分割を表へ流し込む。`acc` はそのステップの
/// 各段の秒数 (呼び出し側の積算器 = cuda.rs の `CHAIN_ACC`)。`steps` は「この分割が何ステップ分か」
/// (decode は 1)。**表示専用**。
///
/// 段の呼び出し回数は「その段が起きたステップ数」= `acc[i] > 0` の個数で数える
/// (launch を 1 回もしなかった段は 0 のまま。「0.0 ms / 0 call」は起きていない、
/// と読める)。`wait_total` (省略可) は sync 区間の残余 — sync に催促された
/// 待ちなので、来たら**待ち行**へ入れて sup の残りから引く。
pub fn split_flush_step(
    table: usize,
    acc: &[f64; SPLIT_ROWS],
    steps: u64,
    wait_total: Option<(usize, f64)>,
) {
    if !enabled() && !crate::telemetry_on() {
        return;
    }
    if table >= SPLIT_TABLES || steps == 0 {
        return;
    }
    let pi = phase().idx();
    let rows = split_rows(table);
    if rows == 0 {
        return;
    }
    for (i, &secs) in acc.iter().enumerate().take(rows) {
        if secs <= 0.0 {
            continue;
        }
        let ns = secs_to_ns(secs);
        // 段が起きたステップ数を calls として積む (0 なら件数は増えない)
        SPLIT_NS[pi][table][i].fetch_add(ns, Ordering::Relaxed);
        SPLIT_NS[Phase::Both.idx()][table][i].fetch_add(ns, Ordering::Relaxed);
        SPLIT_CALLS[pi][table][i].fetch_add(steps, Ordering::Relaxed);
        SPLIT_CALLS[Phase::Both.idx()][table][i].fetch_add(steps, Ordering::Relaxed);
    }
    if let Some((supp_row, secs)) = wait_total {
        if secs > 0.0 && supp_row < SPLIT_ROWS {
            let ns = secs_to_ns(secs);
            let w = SPLIT_WAIT_ROW[table];
            SPLIT_NS[pi][table][w].fetch_add(ns, Ordering::Relaxed);
            SPLIT_NS[Phase::Both.idx()][table][w].fetch_add(ns, Ordering::Relaxed);
            if supp_row != w {
                // 残余は sync を待たせた区間の側から引く (排他の和を保つ)
                let sub = secs_to_ns(secs.max(0.0));
                SPLIT_NS[pi][table][supp_row].fetch_sub(sub, Ordering::Relaxed);
                SPLIT_NS[Phase::Both.idx()][table][supp_row]
                    .fetch_sub(sub, Ordering::Relaxed);
            }
        }
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
        // s138: 分割表 (decode chain / 線形ラッパ) もクリアする
        for t in 0..SPLIT_TABLES {
            for i in 0..SPLIT_ROWS {
                SPLIT_NS[p][t][i].store(0, Ordering::Relaxed);
                SPLIT_CALLS[p][t][i].store(0, Ordering::Relaxed);
            }
        }
        // s140 (Task B6b): カバレッジの分母・tick の件数もクリアする
        for op in 0..N_OPS {
            LIN_STAGE_CALLS[p][op].store(0, Ordering::Relaxed);
            LIN_STAGE_NS[p][op].store(0, Ordering::Relaxed);
        }
        for t in 0..TICK_CALLS.len() {
            TICK_CALLS[t].store(0, Ordering::Relaxed);
        }
        // s148: 表 C の分母 (実ディスパッチ件数)・分割ステップ数・リクエスト壁時計
        LIN_DISPATCH_CALLS[p][0].store(0, Ordering::Relaxed);
        // s151 (B12): 表 C の実測号数と線形の other 号数もフェーズごとにクリアする
        for r in 0..SPLIT_SERVE_N {
            SL_TICKS[p][r].store(0, Ordering::Relaxed);
            // s154 (B15): `span_end` で閉じた号数 (COVERAGE の samples) も同じく。
            SPLIT_END[p][SPLIT_SERVE_TABLE][r].store(0, Ordering::Relaxed);
        }
        LIN_OTHER_CALLS[p][0].store(0, Ordering::Relaxed);
        if p == Phase::Prefill.idx() {
            SL_SPLIT_STEPS.store(0, Ordering::Relaxed);
            REQ_NS.store(0, Ordering::Relaxed);
            REQ_N.store(0, Ordering::Relaxed);
            for k in 0..2 {
                for j in 0..2 {
                    REQ_PHASE_NS[k][j].store(0, Ordering::Relaxed);
                    REQ_PHASE_N[k][j].store(0, Ordering::Relaxed);
                }
            }
            #[cfg(test)]
            SL_OPEN.with(|c| c.set(0));
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
        // s156 (Task B19): サンプリングの会計 (step 番号・測った step 数・測った線形
        // 呼び出し数) もフェーズごとにクリアする — クリアしないとテストの期待値が
        // 前のテストの実行回数に依存する。
        STEPS_SEEN[p].store(0, Ordering::Relaxed);
        SL_SAMPLED[p].store(0, Ordering::Relaxed);
        LIN_SAMPLED[p].store(0, Ordering::Relaxed);
    }
    // s156 (Task B19): プロセス全体で 1 本のカウンタ (間引き・飽和の兆候)。
    SKIPPED_SPANS.store(0, Ordering::Relaxed);
    OVERFLOW_DROPS.store(0, Ordering::Relaxed);
    // s156 (Task B19): 開いている step の番号も落とす (テストは同じスレッドを
    // 使い回すので、落とさないと次のテストが「step の途中」から始まる)。
    STEP_IDX.with(|c| c.set(None));
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

/// s138 (Task B6): 分割表 (`table`) の 1 行ずつを `indent` 付きで積む。
/// 各行 = (名前, 合計 ms, その段が起きた回数, 合計に対する比, 1 回あたり ms)。
/// **表示専用** — `SPLIT_NS` / `SPLIT_CALLS` を読むだけで、どのバケットにも
/// 書かないし、`total` にも足さない (分割の和は外側の区間と別勘定)。
fn push_split_rows(p: usize, table: usize, total: u64, indent: &str, s: &mut String) {
    if table >= SPLIT_TABLES {
        return;
    }
    let n_rows = split_rows(table);
    let mut sum = 0u64;
    for i in 0..n_rows {
        // 行名は表ごとに行数が違うので、行数で switch せず `names()` を引く。
        let name = split_row_name(table, i).unwrap_or("?");
        let sns = SPLIT_NS[p][table][i].load(Ordering::Relaxed);
        if sns == 0 {
            continue;
        }
        let sc = SPLIT_CALLS[p][table][i].load(Ordering::Relaxed);
        sum += sns;
        let per = if sc > 0 {
            sns as f64 / sc as f64 / 1e6
        } else {
            0.0
        };
        s.push_str(&format!(
            "{indent}{:<14} {:>10.1} {:>8} {:>6.1}% {:>9.3}\n",
            name,
            sns as f64 / 1e6,
            sc,
            sns as f64 / total.max(1) as f64 * 100.0,
            per
        ));
    }
    if sum > 0 {
        s.push_str(&format!(
            "{indent}{:<14} {:>10.1}\n",
            "  = split sum",
            sum as f64 / 1e6
        ));
    }
}

/// s138 (Task B6): 分割表の 1 行の表示名 (未定義の行は `None`)。
/// 表ごとに行数が違う (`SPLIT_CHAIN_ROWS` = 9 / `SPLIT_Q4_N` = 8) ので、
/// 「行数で配列を選ぶ」のではなく行の添字で名前を引く。
pub fn split_row_name(table: usize, row: usize) -> Option<&'static str> {
    if table == SPLIT_CHAIN_TABLE && row < SPLIT_CHAIN_ROWS {
        Some(SPLIT_CHAIN_NAMES[row])
    } else if table == SPLIT_Q4_TABLE && row < SPLIT_Q4_N {
        Some(SPLIT_Q4_NAMES[row])
    } else if table == SPLIT_SERVE_TABLE && row < SPLIT_SERVE_N {
        Some(SPLIT_SERVE_NAMES[row])
    } else {
        None
    }
}

/// s138 (Task B6): 表 `table` の行数 (表 A = 9 / 表 B = 8 / 表 C = 16)。表示側と
/// テストが「その表に何行あるか」を 1 か所から引くためのもの (行数で分岐を散らかさない)。
#[inline]
pub fn split_rows(table: usize) -> usize {
    if table == SPLIT_CHAIN_TABLE {
        SPLIT_CHAIN_ROWS
    } else if table == SPLIT_Q4_TABLE {
        SPLIT_Q4_N
    } else if table == SPLIT_SERVE_TABLE {
        SPLIT_SERVE_N
    } else {
        0
    }
}

/// s140 (Task B6b): 線形ステージのバケット呼び出し数 (分岐できなかった線形呼び出し
/// も件数だけは数える — 表 B のカバレッジ率の分母)。**表示専用**。
static LIN_STAGE_CALLS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();

/// s140 (Task B6b): 線形ステージのバケット ns (表 B のカバレッジ率の分母)。
/// バケット値そのものは書き換えない (`add_sample` が本体を積む)。
static LIN_STAGE_NS: [[AtomicU64; N_OPS]; Phase::N] = mk_counters::<N_OPS>();

/// s140 (Task B6b): 表 B の分母を更新する。**表示専用** — 生 id・分割表には
/// 書かない。`on=false` (計測 OFF) は分岐 1 回で戻る。
#[inline]
pub fn count_linear_stage(op: usize, ns: u64, calls: u64, on: bool) {
    if !on || op >= N_OPS {
        return;
    }
    let pi = phase().idx();
    LIN_STAGE_CALLS[pi][op].fetch_add(calls, Ordering::Relaxed);
    LIN_STAGE_NS[pi][op].fetch_add(ns, Ordering::Relaxed);
}

/// s140 (Task B6b): (線形呼び出し数, 線形 ms) — 表 B のカバレッジ率の分母。
/// `phase` = Prefill / Decode のどちらか (Both は数えていない)。
pub fn linear_stage_total(phase: Phase) -> (u64, u64) {
    let p = phase.idx();
    let mut calls = 0u64;
    let mut ns = 0u64;
    for op in 0..N_OPS {
        calls += LIN_STAGE_CALLS[p][op].load(Ordering::Relaxed);
        ns += LIN_STAGE_NS[p][op].load(Ordering::Relaxed);
    }
    (calls, ns)
}

/// s140 (Task B6b): 分割表のサンプル数 (全段の calls の和) と駆動件数 (split_on のまま
/// 走っている線形呼び出し数 = `TICK_CALLS`)。「何 call のうち何 call を分割できたか」。
static TICK_CALLS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// s140 (Task B6b): 分割 tick を打った線形呼び出し数を 1 つ数える (**表示専用**)。
/// 1 呼び出しが 2 タイル以上に割れたときは fragment ぶん数える (tick の回数と
/// 一致させる = 「分割できた割合」が 100% を超えない)。
#[inline]
pub fn split_tick_call(table: usize) {
    if table < 2 {
        TICK_CALLS[table].fetch_add(1, Ordering::Relaxed);
    }
}

/// s140 (Task B6b): 分割 tick を打った回数 (分母)。
pub fn split_tick_count(table: usize) -> u64 {
    if table < 2 {
        TICK_CALLS[table].load(Ordering::Relaxed)
    } else {
        0
    }
}

/// s140 (Task B6b): 分割表のサンプル数 = 全行の calls の和。
pub fn split_sample_count(phase: Phase, table: usize) -> u64 {
    if table >= SPLIT_TABLES {
        return 0;
    }
    let p = phase.idx();
    (0..split_rows(table))
        .map(|i| SPLIT_CALLS[p][table][i].load(Ordering::Relaxed))
        .sum()
}

/// s138 (Task B6): 分割表の**待ち合計** (1 行)。summary の先頭ヘッダで
/// 「プロファイルが捕まえた時間」と「捕まえていない時間」を並べるために使う。
/// 行が無ければ 0。
fn split_wait_ns(p: usize, table: usize) -> u64 {
    if table >= SPLIT_TABLES {
        return 0;
    }
    SPLIT_NS[p][table][SPLIT_WAIT_ROW[table]].load(Ordering::Relaxed)
}

/// s138 (Task B6): サンプル数のヘッダ (1 行)。**187 token の平均を 1 step の
/// 値と取り違えないため**、decode の行が何回のサンプルに基づくかを先に出す。
///
/// 出す値 (decode は「1 token = 1 サンプル」なので calls = steps):
///   decode steps        … decode の行が積まれたステップ数 (= 応答トークン数 − 1 程度)
///   chain split / sum   … ステップ分割 (表 A) を閉じた回数とその和 (待ち込み)
///   chain wait          … そのうち sync で待った時間
///   lin calls / split   … 線形ステージの呼び出し数 / 分割 tick を打った回数
///                         (= 表 B の排他の和が覆う call 数。s140)
///   lin sum / sync wait … その和と、そのうち sync で待った時間
///
/// 注: `chain split` < `decode steps` は「decode の行は積まれたが分割されていない
/// ステップ」がある = 分割の付け漏れ。数字を突き合わせて気づけるようにする。
fn push_sample_header(p: usize, s: &mut String) {
    // decode のステップ数 = 行を積んだ回数の最大値 (どのステップでも通る op を
    // 1 つ選ぶのは脆いので最大を採る — sampler / detok が無い構成でも動く)
    let mut decode_steps = 0u64;
    for i in 0..N_OPS {
        decode_steps = decode_steps.max(CALLS[p][i].load(Ordering::Relaxed));
    }
    // s140 (Task B6b): chain のサンプル数は「ステップ分割を開いた回数」= tick の数。
    // s138 は launch 段 / sync_wait 段の calls を代用していたが、両方とも
    // 「その段が起きたステップ数」であって「分割できたステップ数」ではない
    // (段が 1 つも積まれないステップを見落とす)。
    let chain_splits = split_tick_count(SPLIT_CHAIN_TABLE);
    let mut chain_sum = 0u64;
    for i in 0..SPLIT_CHAIN_ROWS {
        chain_sum += SPLIT_NS[p][SPLIT_CHAIN_TABLE][i].load(Ordering::Relaxed);
    }
    // s140 (Task B6b): 線形の分母 = 線形ステージのバケット件数 (同じ呼び出しが
    // `lin_stage` で数えたもの)、分割できた数 = tick を打った呼び出し数。
    // s138 は「出力確保の段の calls」を call 数の代用にしていたため、tile 分割で
    // 1 呼び出しが複数サンプルになっても気づけなかった (608 call に 802 サンプル)。
    let (lin_bucket_calls0, _) = linear_stage_total(Phase::Prefill);
    let lin_calls = lin_bucket_calls0.max(split_tick_count(SPLIT_Q4_TABLE));
    let lin_splits = split_tick_count(SPLIT_Q4_TABLE);
    let mut q4_sum = 0u64;
    for i in 0..SPLIT_Q4_N {
        q4_sum += SPLIT_NS[p][SPLIT_Q4_TABLE][i].load(Ordering::Relaxed);
    }
    let chain_wait = split_wait_ns(p, SPLIT_CHAIN_TABLE);
    let q4_wait = split_wait_ns(p, SPLIT_Q4_TABLE);
    // s140 (Task B6b): カバレッジ。分母は同じ呼び出しが積んだバケット側の件数・ns で、
    // 分割表のサンプル数と突き合わせる (split calls < 分母 = 付け漏れ)。
    let (lin_bucket_calls, lin_bucket_ns) = linear_stage_total(Phase::Prefill);
    let lin_pct = if lin_bucket_ns > 0 {
        q4_sum as f64 / lin_bucket_ns as f64 * 100.0
    } else {
        0.0
    };
    let (_chain_bucket_calls, chain_bucket_ns) = linear_stage_total(Phase::Decode);
    let chain_pct = if chain_bucket_ns > 0 {
        chain_sum as f64 / chain_bucket_ns as f64 * 100.0
    } else {
        0.0
    };
    s.push_str(&format!(
        "  samples: decode steps {} / chain split {} (sum {:.1} ms, wait {:.1} ms) | \
         lin calls {} / split {} (sum {:.1} ms, sync wait {:.1} ms)\n",
        decode_steps,
        chain_splits,
        chain_sum as f64 / 1e6,
        chain_wait as f64 / 1e6,
        lin_calls,
        lin_splits,
        q4_sum as f64 / 1e6,
        q4_wait as f64 / 1e6,
    ));
    // s140 (Task B6b): カバレッジ行 — 分割表の和がバケットの何割を掴んだか。
    //   lin  … prefill の線形ステージ (QKV/GATE_UP/DOWN/OPROJ) の件数・ms に対する比
    //   chain … decode の線形ステージの ms に対する比 (件数は 1 step 1 回の分割済み
    //           呼び出しが駆動件数を超えるため出さない)
    //   samples … 分割表のサンプル数の和 / tick を打った回数
    s.push_str(&format!(
        "  coverage: lin {} calls split {} / bucket {} calls {:.1} ms ({:.1} pct of linear ms) | \
         chain {:.1} pct of decode linear ms | samples chain {} / {} ticks, \
         lin {} / {} ticks\n",
        lin_calls,
        lin_splits,
        lin_bucket_calls,
        lin_bucket_ns as f64 / 1e6,
        lin_pct,
        chain_pct,
        split_sample_count(Phase::Decode, SPLIT_CHAIN_TABLE),
        split_tick_count(SPLIT_CHAIN_TABLE),
        split_sample_count(Phase::Prefill, SPLIT_Q4_TABLE),
        split_tick_count(SPLIT_Q4_TABLE),
    ));
    // s148: COVERAGE は summary の**末尾**に出す (ヘッダに足すと、`[prefill]` /
    // `[decode]` で節を切って読む既存の読み手・テストが節の切れ目を取り違える)。
    // ここは分割表の後ろなので、節の見出し行は増えない。
}

/// s148 (Task B10): 1 行目 (steps / lin calls / リクエスト壁時計) を前置する
/// COVERAGE 節。`p` のフェーズの表 C の行を見る。
fn push_coverage(phase: Phase, s: &mut String) {
    push_coverage_for(phase.idx(), s);
}

/// s148: **COVERAGE 行**。表 C (serve/q4 経路) の各段について
/// 「サンプル数 / 期待サンプル数」と「帰属 ms / リクエストの壁時計」を出し、
/// 期待があるのにサンプルが 0 の段は `COVERAGE MISS` と明示する。
///
/// 動機 (27B q4 の実測): decode の表は 120 token で 960.9 ms (= 8.0 ms/token) なのに
/// リクエストの壁時計は約 108 ms/token で、85% がどの行にも入っていなかった。
/// しかも `decode steps 1602 / chain split 0` と「0 サンプル」がヘッダに並んで
/// いたのに、それが計測の穴だとは読めなかった。ここでは
///   (1) 必ず一致すべき 2 組 (steps = サンプル / 線形 call = サンプル) を 1 行で並べ、
///   (2) 各段の「サンプル vs 期待」と「ms vs リクエストの壁時計」を出し、
///   (3) サンプル 0 で期待 > 0 の段を `COVERAGE MISS` として名指しする。
fn push_coverage_for(p: usize, s: &mut String) {
    let phase_name = Phase::NAMES[p];
    // s151 (B12): 期待 = 表 C の実測号数と同じ量 (`serve_ticks` / `lin_dispatch_calls`)。
    // samples も `serve_row_ticks` (push の号数) を出す。両者が同じ量なので、
    // この 1 行が [ok] なら「1 step = 1 サンプル」「1 線形 call = 1 サンプル」が
    // そのまま成り立っている。`SPLIT_CALLS` (全体行は子の回数も含む) は使わない。
    let steps = serve_ticks(p);
    let lin = lin_dispatch_calls(phase_for(p));
    let step_samples = serve_row_ticks(p, SL_STEP);
    let lin_samples = serve_row_ticks(p, SL_LIN);
    let (req_n, req_ns) = request_wall();
    let (phase_n, phase_ns) = request_wall_for(p);
    // 必ず一致すべき 2 組 (この 1 行だけで突き合わせられるようにする)
    let steps_ok = steps == step_samples;
    let lin_ok = lin == lin_samples;
    s.push_str(&format!(
        "  COVERAGE [{phase_name}]: steps {} / samples {} [{}] | lin calls {} / samples {} [{}] | \
         split steps {} | requests {} wall {:.1} ms (phase {} wall {:.1} ms)\n",
        steps,
        step_samples,
        if steps_ok { "ok" } else { "COVERAGE MISS" },
        lin,
        lin_samples,
        if lin_ok { "ok" } else { "COVERAGE MISS" },
        split_step_count(),
        req_n,
        req_ns as f64 / 1e6,
        phase_n,
        phase_ns as f64 / 1e6,
    ));
    // 段ごとのサンプルと帰属 ms。分母は「期待サンプル数」と「そのフェーズの壁時計」
    // (prefill 分と decode 分を別々に積んだもの — 全壁時計を両方に使うと decode が
    // 薄まる)。フェーズ分がまだ無いときは全壁時計へ落とす (0 除算を避ける)。
    let denom = if phase_ns > 0 { phase_ns } else { req_ns };
    let pct_of = |ns: u64| -> f64 {
        if denom > 0 {
            ns as f64 / denom as f64 * 100.0
        } else {
            0.0
        }
    };
    s.push_str(&format!(
        "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10} {:>9}\n",
        "bucket", "samples", "expected", "of", "ms", "pct wall"
    ));
    let mut misses: Vec<&'static str> = Vec::new();
    for (_, row, exp, exp_name) in serve_expected(p) {
        let name = split_row_name(SPLIT_SERVE_TABLE, row - SPLIT_SERVE_FIRST).unwrap_or("?");
        let n = serve_row_ticks(p, row);
        let ms = serve_row_ns(p, row) as f64 / 1e6;
        if n == 0 && exp > 0 {
            misses.push(name);
        }
        s.push_str(&format!(
            "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
            name,
            n,
            exp,
            exp_name,
            ms,
            pct_of(serve_row_ns(p, row))
        ));
    }
    // s154 (B15): `sl_step_lay_other` (層ループの投影以外) の内訳。ここが
    // 40.3% の無名部分だった。層ごとの段は `expected` を `steps × layers` で出す
    // (層数を出さないので "-" ではなく倍率だけを表示する)。
    let lay_rows: [(usize, &'static str); 13] = [
        (SL_LAY_EMBED, "step"),
        (SL_LAY_NORM_IN, "layer"),
        (SL_LAY_GDN_PROJ, "layer"),
        (SL_LAY_GDN_SCAN, "layer"),
        (SL_LAY_FULL_QK, "layer"),
        (SL_LAY_FULL_CORE, "layer"),
        (SL_LAY_FULL_OPROJ, "layer"),
        (SL_LAY_FFN_GATE, "layer"),
        (SL_LAY_SWIGLU, "layer"),
        (SL_LAY_FFN_DOWN, "layer"),
        (SL_LAY_NORM_POST, "layer"),
        (SL_LAY_RESIDUAL, "layer"),
        (SL_LAY_OTHER, "step"),
    ];
    let lay_sum: u64 = lay_rows
        .iter()
        .map(|&(op, _)| serve_row_ns(p, op))
        .sum::<u64>()
        + serve_row_ns(p, SL_LAY_FINAL);
    let lay_other_ns = serve_row_ns(p, SL_STEP_LAYERS_OTHER);
    // s154 (B15): 2 本はアダプタでは埋まらない — `sl_lay_embed` (実経路は
    // forward ごとに 1 回で、サーバの step の外 = prefill / 補助 forward もある) と
    // `sl_lay_other` (層ループの child の外 = 実測上 0 を期待する穴)。この 2 本は
    // 「アダプタが閉じたか」だけを見る (号数の縛りを置かない)。
    let adapter_only = [SL_LAY_EMBED, SL_LAY_OTHER];
    for (op, unit) in lay_rows {
        let name = split_row_name(SPLIT_SERVE_TABLE, op - SPLIT_SERVE_FIRST).unwrap_or("?");
        // 層あたりの段は表示のみ (層数・層種別は呼び出し側が知らない。full 層が
        // 0 本の step もあるので `steps × 層数` を期待値にはしない)。per-step の段
        // (`sl_lay_final` のような 1 step 1 回) は `steps` と突き合わせる。
        let exp = if unit == "step" { steps } else { 0 };
        // s169 (P8): prefill は step を開かない (`sl_step` が 0) ので、層あたりの
        // 段の期待値は「prefill の回数 × layers」で出す。以前は "0xlayers" と出て
        // いて、サンプル 0 と倍率 0 の区別が付かなかった。step があれば従来どおり。
        let layer_mult = if steps > 0 {
            steps
        } else {
            serve_row_ends(p, SL_STEP_PREP)
        };
        if exp > 0 && !adapter_only.contains(&op) {
            // s154 (B15): 号数ではなく「`span_end` で閉じられたか」で判定する
            // (合成時刻テストでは 1 区間が 0 ns になり、ns は 0 のまま)。
            let ends = serve_row_ends(p, op);
            if ends < exp {
                misses.push(name);
            }
        }
        s.push_str(&format!(
            "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
            name,
            // s154 (B15): 閉じた号数 (ns が 0 でも区間は置かれている)。ns を積んだ
            // 号数 (`serve_row_ticks`) と食い違うときは `ticks` として添える。
            serve_row_ends(p, op),
            if unit == "step" {
                if adapter_only.contains(&op) {
                    "-".to_string()
                } else {
                    exp.to_string()
                }
            } else {
                format!("{layer_mult}x{}", "layers")
            },
            if unit == "step" { "steps" } else { "layer" },
            serve_row_ns(p, op) as f64 / 1e6,
            pct_of(serve_row_ns(p, op))
        ));
    }
    s.push_str(&format!(
        "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
        "sl_lay_final",
        serve_row_ticks(p, SL_LAY_FINAL),
        steps,
        "steps",
        serve_row_ns(p, SL_LAY_FINAL) as f64 / 1e6,
        pct_of(serve_row_ns(p, SL_LAY_FINAL))
    ));
    // 層内の内訳の和 vs 親 (`sl_step_lay_other`)。s154 の計測点は層ループの
    // 内側を全部覆うので、差は「層ループのうち child の外側」(≈ 0) だけになる。
    s.push_str(&format!(
        "  COVERAGE layer residual: children {} / lay_other {} ns over {} steps ({} ms/step, {:.1}%)\n",
        lay_sum,
        lay_other_ns,
        steps,
        if steps > 0 {
            lay_other_ns as f64 / steps as f64 / 1e6
        } else {
            0.0
        },
        if lay_other_ns > 0 {
            lay_sum as f64 / lay_other_ns as f64 * 100.0
        } else {
            0.0
        }
    ));
    // s155 (Task B16): MISS の対象は「**1 step 1 回**で、実経路で必ず発火するべき」
    // 行だけ。判定は `serve_row_ends` (span を閉じた号数) と `steps` の比較で、
    // 行を外すのは「step 数と 1 対 1 にならない」理由があるときだけ:
    //   sl_so_sample_pen 1 step 1 回 (engine の decode ループで条件の外側に置いた)
    //   sl_so_stream     1 step 1 回 (detok / delta / stop の 3 出口すべてで閉じる)
    //   sl_so_logits     1 step 1 回 (cuda feature 非依存の per-op forward が閉じる —
    //                    B15 は cuda の内側にだけ置いていたため 0 call だった)
    //   sl_so_gap_fwd    **step 数と一致しない**ので外す: 区間は step の内側
    //                    (`sl_so_logits` の中) なのでモデル経路の step 数を数えず、
    //                    しかも cuda の chain と per-op の**両方**が同じ step で
    //                    閉じうる。値だけ出し、号数は `serve_row_ends` を裸で表示する
    //                    (外しても「発火したか」は表に見える — 期待値は置かない)。
    //   sl_so_end        1 step 1 回 (step_end の直後に閉じる)
    // 転記の 2 本 (`sl_so_lin_sum` / `sl_so_lin_d2h`) は線形呼び出しに比例するので
    // 期待値を置かない (元からこの一覧に無い)。
    let step_other = serve_row_ns(p, SL_STEP_OTHER);
    for (name, op, expect_steps) in [
        ("sl_so_sample_pen", SL_SO_SAMPLE_PEN, true),
        ("sl_so_stream", SL_SO_STREAM, true),
        ("sl_so_logits", SL_SO_LOGITS, true),
        ("sl_so_gap_fwd", SL_SO_GAP_FWD, false),
        ("sl_so_end", SL_SO_END, true),
    ] {
        let ends = serve_row_ends(p, op);
        if expect_steps && ends < steps {
            misses.push(name);
        }
        s.push_str(&format!(
            "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
            name,
            ends,
            steps,
            "steps",
            serve_row_ns(p, op) as f64 / 1e6,
            pct_of(serve_row_ns(p, op))
        ));
    }
    // s154 (B15): `sl_so_resid` は `step_end` の中で `push` する残余行なので、
    // 「閉じた号数」ではなく **積んだ号数** (`serve_row_calls`) を `steps` と比べる。
    {
        let calls = serve_row_calls(p, SL_SO_RESID);
        if calls < steps {
            misses.push("sl_so_resid");
        }
        s.push_str(&format!(
            "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
            "sl_so_resid",
            calls,
            steps,
            "steps",
            serve_row_ns(p, SL_SO_RESID) as f64 / 1e6,
            pct_of(serve_row_ns(p, SL_SO_RESID))
        ));
    }
    // 残余行 (`sl_so_resid`) と `sl_step_other` の恒等式: 残余 + 実測の子
    // + 転記行 (lin_sum) が other に一致する。**ここは 0 許容で出す** — 実測の子は
    // `step_end` の後に 1 本 (`sl_so_end`) 来るので、その分だけ other 側に先に
    // 入っている step で差が出る (差の合計は steps のぶんだけ残る = 穴ではない)。
    let resid_ns = serve_row_ns(p, SL_SO_RESID);
    let kids_ns: u64 = SL_SO_MEASURED
        .iter()
        .map(|&op| serve_row_ns(p, op))
        .sum();
    let lin_sum = serve_row_ns(p, SL_SO_LIN_SUM);
    s.push_str(&format!(
        "  COVERAGE other residual: resid {} + kids {} + lin_sum {} = {} / other {} ns \
         over {} steps ({} ms/step, diff {})\n",
        resid_ns,
        kids_ns,
        lin_sum,
        resid_ns + kids_ns + lin_sum,
        step_other,
        steps,
        if steps > 0 {
            resid_ns as f64 / steps as f64 / 1e6
        } else {
            0.0
        },
        (resid_ns + kids_ns + lin_sum) as i64 - step_other as i64
    ));
    // 残余 (どの段にも入らなかった ms) と、全体 − 内訳の差
    let step_ns = serve_row_ns(p, SL_STEP);
    let step_other = serve_row_ns(p, SL_STEP_OTHER);
    let lin_ns = serve_row_ns(p, SL_LIN);
    let lin_other = serve_row_ns(p, SL_LIN_OTHER);
    let gap_step = step_gap(SL_STEP);
    let gap_lin = step_gap(SL_LIN);
    s.push_str(&format!(
        "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
        "sl_step_other",
        serve_row_ticks(p, SL_STEP_OTHER),
        "-",
        "-",
        step_other as f64 / 1e6,
        pct_of(step_other)
    ));
    // s151 (B12): `sl_lin_other` の samples は **線形呼び出し 1 回につき 1** を出す。
    // s148 は `SPLIT_CALLS` を出していたため、内訳 3 本 (入口の地 / 登録 / ピン留め) が
    // 1 call に 3 号入り、線形呼び出しの 3 倍 (36,090 → 108,270) に見えていた。
    s.push_str(&format!(
        "  COVERAGE {:<18} {:>9} {:>9} {:>7} {:>10.1} {:>8.1}%\n",
        "sl_lin_other",
        lin_other_samples(p),
        lin,
        "lin calls",
        lin_other as f64 / 1e6,
        pct_of(lin_other)
    ));
    // s153 (Task B14): `sl_lin` の内訳は **排他の和 = 全体** でなければならない
    // (`sl_lin_other` は残余 = 全体 − 測った段)。ここでは 1) 段の和が全体を
    // 超えない、2) 残余の号数 = 線形呼び出し数、3) 未帰属 (`gap_lin`) が 1 呼び出し
    // あたり時計の粒度未満、の 3 点を 1 行で出す。号数は `sl_lin_other` の
    // 実測号数 (`serve_row_ticks`) と線形呼び出し数の対比なので、s148 の
    // 「other が 3 倍」もここで [MISS] になる。
    let lin_children: u64 = SL_GROUPS[1].1.iter().map(|&op| serve_row_ns(p, op)).sum();
    let lin_other_ticks = serve_row_ticks(p, SL_LIN_OTHER);
    // s153 (B14): 判定に使うのは**号数**だけにする。`sl_lin` の 1 号に対して
    // `sl_lin_other` が 1 号 (s148 の 3 倍は 3 号で捕まる)。ns の恒等式
    // (親 = 段の和 + 残余) は上の行に数値として出すが、合成時刻を使うテストでは
    // 段の ns が実時計なので判定には使わない (実経路では恒等的に成り立つ)。
    let lin_ticks = serve_row_ticks(p, SL_LIN);
    let lin_ok_rel = lin_other_ticks == lin_ticks;
    s.push_str(&format!(
        "  COVERAGE lin residual: {} + children {} = {} / {} ns over {} calls \
         (other samples {} / lin samples {}) [{}]\n",
        lin_other, lin_children, lin_other + lin_children, lin_ns, lin,
        lin_other_ticks, lin_ticks,
        if lin_ok_rel { "ok" } else { "COVERAGE MISS" },
    ));
    if !lin_ok_rel {
        misses.push("sl_lin_other");
    }
    // s155 (Task B16): `sl_step_other` が `sl_step` と同値 (= 何も名付けられていない)
    // なら MISS にする。B15 の GPU 実測は sl_step 131.737 に対し sl_step_other
    // 131.737 ms/step で、CPU の試験は全部の段を `step_end` の前に積んでいたため
    // これを見逃した。ここでは「other < step」かつ「兄弟 > 0」を 1 行で出す。
    {
        let sib: u64 = SL_GROUPS[0]
            .1
            .iter()
            .filter(|&&op| op != SL_STEP_OTHER)
            .map(|&op| serve_row_ns(p, op))
            .sum();
        let other_ticks = serve_row_ticks(p, SL_STEP_OTHER);
        let ok = steps > 0 && step_other < step_ns && sib > 0;
        if steps > 0 && !ok {
            misses.push("sl_step_other");
        }
        s.push_str(&format!(
            "  COVERAGE step residual: other {} + siblings {} = {} / {} ns over {} steps \
             (other samples {}) [{}]\n",
            step_other,
            sib,
            step_other + sib,
            step_ns,
            steps,
            other_ticks,
            if ok { "ok" } else { "COVERAGE MISS" },
        ));
    }
    s.push_str(&format!(
        "  COVERAGE accounted {:.1} ms of wall {:.1} ms ({:.1} pct) | \
         gaps: step {:.1} ms, lin {:.1} ms\n",
        (step_ns + lin_ns) as f64 / 1e6,
        denom as f64 / 1e6,
        pct_of(step_ns + lin_ns),
        gap_step as f64 / 1e6,
        gap_lin as f64 / 1e6,
    ));
    if misses.is_empty() {
        s.push_str("  COVERAGE status: all buckets have samples\n");
    } else {
        s.push_str(&format!(
            "  COVERAGE status: COVERAGE MISS in {} bucket(s): {}\n",
            misses.len(),
            misses.join(", ")
        ));
    }
}

/// s138 (Task B6): フェーズ節の最後に置く分割表。prefill は線形ラッパ (表 B)、
/// decode は chain (表 A)。どちらも**そのフェーズでしか起きない**ので、表を
/// 出し分けても読み手の混乱は無い (decode に lin_* は出ない)。
/// 行が 1 本も無ければ何も出さない (PROFILE off の計測で空の節を出さない)。
fn push_split_section(phase: Phase, s: &mut String) {
    let p = phase.idx();
    let table = match phase {
        Phase::Prefill => SPLIT_Q4_TABLE,
        Phase::Decode => SPLIT_CHAIN_TABLE,
        Phase::Both => return,
    };
    let rows = phase_rows(p);
    let total: u64 = rows.iter().map(|(_, _, self_ns, _, _, _, _)| *self_ns).sum();
    let n_rows = if table == SPLIT_CHAIN_TABLE {
        SPLIT_CHAIN_ROWS
    } else {
        SPLIT_Q4_N
    };
    let any = (0..n_rows).any(|i| SPLIT_NS[p][table][i].load(Ordering::Relaxed) > 0);
    if !any {
        return;
    }
    // decode は 1 token = 1 サンプル (calls 列 = その段が起きたステップ数)、
    // prefill は 1 回の線形ラッパ呼び出しが 1 サンプル。
    let steps = SPLIT_CALLS[p][table][SPLIT_WAIT_ROW[table]].load(Ordering::Relaxed);
    // 見出しに `[prefill]` / `[decode]` を入れない (節の切れ目と紛れないため)。
    // 表の種類は名前 (`chain_*` / `lin_*`) で分かる。
    s.push_str(&format!(
        "  SPLIT {} (exclusive, display only; {})\n",
        Phase::NAMES[p].to_uppercase(),
        if steps > 0 {
            "calls = samples (decode: steps)"
        } else {
            "calls = per-call sites"
        }
    ));
    s.push_str(&format!(
        "  {:<14} {:>10} {:>8} {:>7} {:>9}\n",
        "segment", "ms", "calls", "share", "ms/call"
    ));
    push_split_rows(p, table, total, "  ", s);
}

/// 整形済みサマリ (prefill / decode / cumulative の 3 テーブル) を返す。
///
/// s57 (T4): 従来は decode+prefill の累積 1 表だった — prefill のホットスポットを
/// 確定するには decode の寄与 (step 数ぶん積み上がる) が混ざると読めないため、
/// フェーズ別を先に出し、互換・合計確認用に cumulative を最後に残す
/// (`cumulative` の各 op = prefill + decode、TOTAL も従来と同値)。
/// d2h_state / attn_core の直後には細分行をインデントして出す。
///
/// s138 (Task B6): 先頭にサンプル数ヘッダ、`prefill` 節の最後に線形ラッパの分割
/// (lin_*)、`decode` 節の最後に chain の分割 (chain_*) を出す。**どちらも表示専用**
/// で、TOTAL (= self の和) には入らない。
pub fn summary() -> String {
    let mut s = String::from(
        "PROFILE (prefill/decode split; tok/call = mean batch width; \
         ms/calls = per-sample when calls = steps):\n",
    );
    push_sample_header(Phase::Prefill.idx(), &mut s);
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
    // s138 (Task B6): 分割表は **最後** (cumulative の後) に置く。節の途中に
    // 見出し行 (`[prefill]` / `[decode]`) を増やすと、既存の読み手 (ログの
    // 切り出し・テスト) が節の切れ目を取り違えるため。ここは 1 本の表として
    // 読み、「どの段が何 ms か」だけを見る。
    push_split_section(Phase::Prefill, &mut s);
    push_split_section(Phase::Decode, &mut s);
    // s148 (Task B10): serve (q4) 経路の表 C。段は 1 step / 1 call で排他なので
    // 合計がその step の壁時計になる。prefill の節に 1 本だけ出す (decode の
    // step は同じ表の別フェーズに積まれるが、段の意味は共通)。
    push_serve_section(Phase::Prefill, &mut s);
    push_serve_section(Phase::Decode, &mut s);
    // s148: COVERAGE は **両フェーズ** に出す (summary の末尾 = 既存の節の切れ目を
    // 動かさない位置)。q4 serve の decode step は decode 表に積まれるので、prefill 側
    // だけ出すと decode の穴がそのまま隠れる (「steps 0 / samples 0 [ok]」だけが
    // 並んで MISS が見えない状態を実際に踏んだ)。
    push_coverage(Phase::Prefill, &mut s);
    push_coverage(Phase::Decode, &mut s);
    // s240 (P40): GDN chunked 3 カーネルの実時間 (`DENNO_GDN_KPROF=1` のときだけ)。
    // 計測専用の 1 行で、上の表の値には一切混ぜない (`DENNO_GDN_KPROF` が未設定
    // なら `gdn_kprof_line()` が None = このブロックは 1 バイトも足さない)。
    #[cfg(feature = "cuda")]
    if let Some(line) = crate::cuda::gdn_kprof_line() {
        s.push_str(&line);
        s.push('\n');
    }
    // decode の GDN コア 4 区間 (`DENNO_GDN_KPROF=1` のときだけ)。
    // prefill の上の行 (`[GDN_KPROF] pre/state/scan`) とは別系統で、こちらは
    // `forward_token_core_gate` の conv / l2 / scan / rms を積んだもの。
    // 未設定なら `line()` が None = 1 バイトも足さない。
    if let Some(line) = crate::gdn::kprof::line() {
        s.push_str(&line);
        s.push('\n');
    }
    s
}

/// s148 (Task B10): 表 C (serve/q4 経路) の節。`sl_step` (1 step の全体) の
/// サンプル数と、その内訳 (`sl_lin` = 投影の全体、`sl_step_other` = 残余) を
/// 「ms / calls / ms per step」で出す。内訳の和が全体に足りない分は
/// `unattributed` 行として**必ず**出す (0 でも行を消さない — 消すと穴が見えなくなる)。
fn push_serve_section(phase: Phase, s: &mut String) {
    let p = phase.idx();
    let total_calls = serve_row_calls(p, SL_STEP);
    if total_calls == 0 && serve_row_calls(p, SL_LIN) == 0 {
        return;
    }
    let total_ns = serve_row_ns(p, SL_STEP);
    // s156 (Task B19): 平均の分母は **実際に測った step 数** (`DENNO_PROFILE_SAMPLE`
    // で間引いても嘘にならない)。0 のときは号数へ落とす (合成テストの互換)。
    let measured = {
        let n = serve_samples(p);
        if n > 0 {
            n
        } else {
            total_calls
        }
    };
    let per = |ns: u64| {
        if measured > 0 {
            ns as f64 / measured as f64 / 1e6
        } else {
            0.0
        }
    };
    s.push_str(&format!(
        "  SPLIT SERVE ({}, q4 serve path; calls = steps for sl_step / calls for sl_lin)\n",
        Phase::NAMES[p]
    ));
    // s156 (Task B19): 間引きの状態を必ず 1 行で出す — 何 step のうち何 step を
    // 測ったか (N=1 でも「N=1 / 全部測った」として出る)。推測値は書かない。
    s.push_str(&format!(
        "  SPLIT SERVE sampling: N={} (DENNO_PROFILE_SAMPLE) | steps seen {} | steps measured {} \
         | lin calls measured {} | spans skipped {}\n",
        sample_every(),
        measured,
        serve_samples(p),
        serve_lin_samples(p),
        skipped_spans()
    ));
    s.push_str(&format!(
        "  {:<18} {:>10} {:>8} {:>7} {:>12}\n",
        "segment", "ms", "calls", "share", "ms/step"
    ));
    for row in 0..SPLIT_SERVE_N {
        let op = SPLIT_SERVE_FIRST + row;
        let ns = serve_row_ns(p, op);
        let calls = serve_row_calls(p, op);
        if ns == 0 && calls == 0 {
            continue;
        }
        let name = split_row_name(SPLIT_SERVE_TABLE, row).unwrap_or("?");
        s.push_str(&format!(
            "  {:<18} {:>10.1} {:>8} {:>6.1}% {:>12.3}\n",
            name,
            ns as f64 / 1e6,
            calls,
            if total_ns > 0 {
                ns as f64 / total_ns as f64 * 100.0
            } else {
                0.0
            },
            per(ns)
        ));
    }
    // 内訳の和が全体に届かない分は「未帰属」として明示する (0 でも出す)。
    let gap_step = step_gap(SL_STEP);
    let gap_lin = step_gap(SL_LIN);
    s.push_str(&format!(
        "  {:<18} {:>10.1} {:>8} {:>7} {:>12.3}\n",
        "unattributed step",
        gap_step as f64 / 1e6,
        total_calls,
        "-",
        per(gap_step)
    ));
    s.push_str(&format!(
        "  {:<18} {:>10.1} {:>8} {:>7} {:>12.3}\n",
        "unattributed lin",
        gap_lin as f64 / 1e6,
        serve_row_calls(p, SL_LIN),
        "-",
        if serve_row_calls(p, SL_LIN) > 0 {
            gap_lin as f64 / serve_row_calls(p, SL_LIN) as f64 / 1e6
        } else {
            0.0
        }
    ));
    // s154 (B15): `sl_step_other` (無名の 59%) の内訳を 4 スペース深く出す。
    // 転記行 (lin_sum / lin_d2h) と実測行 (sample_pen / stream / logits / gap_fwd /
    // end) と残余 (`sl_so_resid`) の和が `sl_step_other` に一致する。
    let kid_rows: [(&str, usize); 8] = [
        ("sl_so_lin_sum", SL_SO_LIN_SUM),
        ("sl_so_lin_d2h", SL_SO_LIN_D2H),
        ("sl_so_sample_pen", SL_SO_SAMPLE_PEN),
        ("sl_so_stream", SL_SO_STREAM),
        ("sl_so_logits", SL_SO_LOGITS),
        ("sl_so_gap_fwd", SL_SO_GAP_FWD),
        ("sl_so_end", SL_SO_END),
        ("sl_so_resid", SL_SO_RESID),
    ];
    let step_other_ns = serve_row_ns(p, SL_STEP_OTHER);
    for (name, op) in kid_rows {
        let ns = serve_row_ns(p, op);
        s.push_str(&format!(
            "    {:<14} {:>10.1} {:>8} {:>7} {:>12.3}\n",
            name,
            ns as f64 / 1e6,
            serve_row_ticks(p, op),
            if step_other_ns > 0 {
                format!("{:.1}%", ns as f64 / step_other_ns as f64 * 100.0)
            } else {
                "-".to_string()
            },
            per(ns)
        ));
    }
    // s156 (Task B19): `sl_so_gap_fwd` (B18 実測 54.158 ms/step = instrumented
    // step の 41%) の内訳 (7 段・すべて `sl_gap_*`)。6 スペース深く出す。
    // 和 + 残余 = gap_fwd が**表示のたびに**確かめられる形で並べる (残余は 0 でも行を出す)。
    // `sl_gap_attn` は rope を除いた正味 — 表示値の和が gap_fwd に一致する。
    let gap_fwd_ns = serve_row_ns(p, SL_SO_GAP_FWD);
    let gap_parts = gap_fwd_parts(p);
    let gap_ticks = gap_fwd_part_ticks(p);
    if gap_fwd_ns > 0 || gap_parts.sum() > 0 {
        s.push_str(&format!(
            "      {:<20} {:>10} {:>8} {:>7} {:>12}\n",
            "sl_gap_* (gap_fwd)", "ms", "calls", "share", "ms/step"
        ));
        let mut sum = 0u64;
        for (i, (name, ns)) in gap_parts.as_pairs().iter().enumerate() {
            sum += *ns;
            s.push_str(&format!(
                "      {:<20} {:>10.1} {:>8} {:>7} {:>12.3}\n",
                name,
                *ns as f64 / 1e6,
                gap_ticks[i],
                if gap_fwd_ns > 0 {
                    format!("{:.1}%", *ns as f64 / gap_fwd_ns as f64 * 100.0)
                } else {
                    "-".to_string()
                },
                per(*ns)
            ));
        }
        let resid = gap_fwd_residual_ns(p);
        s.push_str(&format!(
            "      {:<20} {:>10.1} {:>8} {:>7} {:>12.3}\n",
            "sl_gap_resid",
            resid as f64 / 1e6,
            "-",
            if gap_fwd_ns > 0 {
                format!("{:.1}%", resid as f64 / gap_fwd_ns as f64 * 100.0)
            } else {
                "-".to_string()
            },
            per(resid)
        ));
        s.push_str(&format!(
            "      {:<20} {:>10.1} {:>8} {:>7} {:>12.3}  (parts {} + resid {} = {} / gap_fwd {})\n",
            "= gap_fwd",
            gap_fwd_ns as f64 / 1e6,
            "",
            "",
            per(gap_fwd_ns),
            sum,
            resid,
            sum + resid,
            gap_fwd_ns
        ));
    }
}

#[cfg(test)]
static GLOBAL_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// s163 (Task K4): グローバルカウンタ (`NS` / `CALLS` 等) を汚し得るテストの
/// 排他ガード (crate 内テスト専用)。ロックの実体は `GLOBAL_COUNTER_LOCK` 1 本で、
/// `prof::tests::lock_counters()` もこれを通る (別 Mutex は作らない)。
///
/// 集計の static はプロセス全体で 1 本なので、`FORCE_ENABLED = Some(true)` の
/// 窓の間に別スレッドの forward が sample を積むと、厳密比較 (`CALLS[ATTN] == 46`
/// 等) が崩れる。prof 側のテストは `lock_counters()` で、長い forward を回す
/// テストはこの関数で同じロックを取る — 双方向に排他にして窓を消す。
/// ロックは panic でも解放される。
#[cfg(test)]
pub(crate) fn counters_guard() -> std::sync::MutexGuard<'static, ()> {
    GLOBAL_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    //! s55: d2h 細分 id の整合 (生 id 不変・細分は N_OPS の外・summary が 3 行を出す)。
    //! s57 (T4): フェーズ分離 (prefill/decode) と attn_core の層種別分離、平均 tok/call。
    //! 時刻値は環境依存なので検証しない — 検証するのは id の算術と集約だけ。
    use super::*;

    /// s85 (Step 0): テスト専用の `enabled()` 上書き。None = env (`DENNO_PROFILE`) を見る。
    /// `Ambient::enter` / `add_outer_ns` は `enabled()` が偽だと完全 no-op なので、
    /// 会計のテストはこのフラグで計測を有効化する (env の書き換えは並走テストを
    /// 揺らすため使わない)。会計の static はプロセス共有なので、必ず
    /// `lock_counters()` と一緒に使う。
    static FORCE_ENABLED: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

    pub(super) fn force_enabled() -> Option<bool> {
        *FORCE_ENABLED.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// s156 (Task B19): テスト専用のサンプリング周期の上書き (None = env を見る)。
    /// `DENNO_PROFILE_SAMPLE` を書き換えると並走テストを揺らすので、テストはこの
    /// 経路を使う (`force_enabled` と同じ理由)。
    static FORCE_SAMPLE_EVERY: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);

    pub(super) fn force_sample_every() -> Option<u64> {
        *FORCE_SAMPLE_EVERY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// s156 (Task B19): サンプリング周期を上書きして `f` を実行し、必ず戻す。
    fn with_sample_every<T>(n: u64, f: impl FnOnce() -> T) -> T {
        struct G;
        impl Drop for G {
            fn drop(&mut self) {
                *FORCE_SAMPLE_EVERY.lock().unwrap_or_else(|e| e.into_inner()) = None;
            }
        }
        *FORCE_SAMPLE_EVERY
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(n);
        let _g = G;
        f()
    }

    /// s85 (Step 0): 計測を有効化したまま `f` を実行し、終了時に必ず元へ戻す
    /// (panic しても戻る)。カウンタのロックも取るので、呼び出し側は取らない。
    ///
    /// 表はプロセス共有なので、掴んだ直後に `reset()` する — テスト本体の末尾で
    /// リセットする形だと、途中で assert が落ちたときに値を残し、後続のテスト
    /// (`attn_core_is_split_into_gdn_and_full_rows` 等) を巻き込む (実際に踏んだ)。
    fn with_prof_counters<T>(f: impl FnOnce() -> T) -> T {
        // P74: `serve_on()` は `enabled() || crate::telemetry_on()` なので、
        // 並走するテストがテレメトリを ON にしていると表 C にサンプルが入る。
        // テレメトリを切り替えるテスト (qwen4exp) は `moe_test_lock` を握るので、
        // ここでも同じロックを取って「計測の窓」を排他する (ロック順は
        // moe -> counters で固定。逆順を取るテストは無い)。
        let _moe = crate::model::moe_expert::moe_test_lock();
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
    ///
    /// s163 (Task K4): ロックの実体は `crate::prof::counters_guard()` に移した
    /// (prof 以外のテスト — 長文脈 forward 等 — も同じ 1 本を取れるようにする)。
    /// ここは名前を保つだけの薄いラッパで、**別 Mutex は作らない**。
    fn lock_counters() -> std::sync::MutexGuard<'static, ()> {
        counters_guard()
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

    /// s108: `chain_download_site` が「decode 1 ステップ 1 回の最大 D2H」を
    /// 未分類側に落とさないことを固定する (GPU 不要の純関数)。
    ///
    /// 固定する規約:
    /// 1) CH_LOGITS は `D2H_SRC_LOGITS` (従来は OTHER に混ざっていた)
    /// 2) それ以外のスロットは `D2H_SRC_LINEAR`
    /// 3) どのスロットでも `D2H_SRC_OTHER` は返さない (未分類を増やさない)
    /// 4) 返す id は `mark_d2h_src_ns` が受理する範囲内 (範囲外だと件数が静かに落ちる)
    ///
    /// CH_LOGITS は cuda モジュールの定数なので、このテストは cuda feature のみ。
    #[test]
    #[cfg(feature = "cuda")]
    fn chain_download_site_classifies_logits_apart_from_other_slots() {
        assert_eq!(chain_download_site(crate::cuda::CH_LOGITS), D2H_SRC_LOGITS);
        for slot in [
            crate::cuda::CH_X,
            crate::cuda::CH_H,
            crate::cuda::CH_QKV,
            crate::cuda::CH_ATT,
            crate::cuda::CH_O,
            crate::cuda::CH_GA,
            crate::cuda::CH_ACT,
            crate::cuda::CH_DOWN,
        ] {
            assert_eq!(
                chain_download_site(slot),
                D2H_SRC_LINEAR,
                "slot {slot} は LINEAR 側"
            );
        }
        for slot in 0..16usize {
            let site = chain_download_site(slot);
            assert!(
                (D2H_SRC_FIRST..D2H_SRC_FIRST + D2H_SRC_N).contains(&site),
                "slot {slot}: site {site} が d2h_src の範囲外"
            );
            assert_ne!(site, D2H_SRC_OTHER, "slot {slot} が未分類へ落ちている");
        }
    }

    /// s108: `chain_download` が積む件数の**スロット**を固定する。実 GPU 呼び出し
    /// はせず、件数の積み先 (LOGITS へ +1 / LINEAR・OTHER は不変) だけを確認する。
    /// GPU 非対応の環境では発行自体が起きない (`chain_ctx` が None) ので、その
    /// 場合だけ判定をスキップする — GPU ホストと cuda feature のビルドでは走る。
    #[test]
    #[cfg(feature = "cuda")]
    fn chain_download_counts_logits_slot_as_logits() {
        with_prof_counters(|| {
            set_phase(Phase::Decode);
            let before_l = d2h_src_calls_for_test(Phase::Decode, D2H_SRC_LOGITS);
            let before_i = d2h_src_calls_for_test(Phase::Decode, D2H_SRC_LINEAR);
            let before_o = d2h_src_calls_for_test(Phase::Decode, D2H_SRC_OTHER);
            let r = crate::cuda::chain_download(crate::cuda::CH_LOGITS, 4);
            if r.is_none() {
                return; // GPU 無し: 発行も件数も起きない (判定不能)
            }
            assert_eq!(
                d2h_src_calls_for_test(Phase::Decode, D2H_SRC_LOGITS),
                before_l + 1,
                "logits は D2H_SRC_LOGITS に積む"
            );
            assert_eq!(d2h_src_calls_for_test(Phase::Decode, D2H_SRC_LINEAR), before_i);
            assert_eq!(
                d2h_src_calls_for_test(Phase::Decode, D2H_SRC_OTHER),
                before_o,
                "未分類 (OTHER) には積まない"
            );
            // site の件数は生バケット (d2h_state) へ入らない — 表示専用の規約。
            assert_eq!(
                NS[Phase::Decode.idx()][D2H].load(Ordering::Relaxed),
                0,
                "site の件数は d2h_state 生バケットへ入らない"
            );
        });
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
    ///
    /// s96 (P3-1): GPU 経路 (`cuda::gdn_proj_f32_into`) は削除したが、**id の予約は
    /// 残す** — 番号を詰め直すと過去のプロファイルログとの比較が壊れる。誰も
    /// 積まなくなったので表は 0 のまま出る (表示専用・数値経路には無関係)。
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

    /// s138 (Task B6): 分割表 (decode chain / 線形ラッパ) の id が連番で、
    /// 既存の細分 (D2H 細分 0..9 / GDN_PF 9..13 / D2H_SRC 13..19 / GDN_PROJ 19..23)
    /// と交差せず、生 id (N_OPS = 15) の外側に収まること。
    /// 番号を詰め直すと過去のプロファイルログとの比較が壊れるので、開始位置も固定する。
    ///
    /// s138 後半: 表ごとに行数が違う (表 A = `SPLIT_CHAIN_ROWS` 9 / 表 B =
    /// `SPLIT_Q4_N` 8) ので、「表 A の行数ぶん進んだ所が表 B」という関係だけを固定し、
    /// 「両表の行数は同じ」という前提は捨てた (`SPLIT_N` 廃止)。
    #[test]
    fn split_ids_are_outside_existing_ranges() {
        assert_eq!(SPLIT_CHAIN_FIRST, 10 * D2H + 23);
        assert_eq!(SPLIT_CHAIN_ROWS, SPLIT_CHAIN_NAMES.len());
        assert_eq!(SPLIT_Q4_N, SPLIT_Q4_NAMES.len());
        assert_eq!(SPLIT_Q4_FIRST, SPLIT_CHAIN_FIRST + SPLIT_CHAIN_ROWS);
        // s148 (Task B10): 表 C (serve/q4 経路) は表 B の末尾に並置する (詰め直しを
        // しない — 過去ログの id 位置を保つ)。行数は `SPLIT_SERVE_N` 16。
        assert_eq!(SPLIT_SERVE_FIRST, SPLIT_Q4_FIRST + SPLIT_Q4_N);
        assert_eq!(SPLIT_SERVE_N, SPLIT_SERVE_NAMES.len());
        // 積算器は**一番多い表**の本数で確保する (s148: 表 C = 16 行が最大)。
        // 「表 A の本数」ではなく「3 表の最大」であることが本当の不変条件。
        assert_eq!(
            SPLIT_ROWS,
            SPLIT_CHAIN_ROWS.max(SPLIT_Q4_N).max(SPLIT_SERVE_N),
            "積算器の本数は 3 表の最大でなければならない"
        );
        assert!(SPLIT_ROWS >= SPLIT_CHAIN_ROWS && SPLIT_ROWS >= SPLIT_Q4_N);
        assert!(SPLIT_ROWS >= SPLIT_SERVE_N);
        // どの表の行数も積算器の本数を超えない (acc の添字が必ず入る)
        for t in 0..SPLIT_TABLES {
            assert!(
                split_rows(t) <= SPLIT_ROWS,
                "表 {t} の行数 {} が積算器の本数 {SPLIT_ROWS} を超える",
                split_rows(t)
            );
        }
        for i in 0..SPLIT_CHAIN_ROWS {
            let a = SPLIT_CHAIN_FIRST + i;
            assert!(a >= N_OPS, "生 id の範囲外: {a}");
            assert!(a < SPLIT_Q4_FIRST, "表 A と表 B の範囲が交差: {a}");
            // 既存の細分 (GDN_PROJ の末尾) と重ならない
            assert!(a >= D2H_GDN_PROJ_FIRST + D2H_GDN_PROJ_N);
            // split_slot が (表, 行) へ正しく畳む
            assert_eq!(split_slot(a), Some((SPLIT_CHAIN_TABLE, i)));
        }
        for i in 0..SPLIT_Q4_N {
            let b = SPLIT_Q4_FIRST + i;
            assert!(b >= N_OPS, "生 id の範囲外: {b}");
            assert!(b < SPLIT_SERVE_FIRST, "表 B と表 C の範囲が交差: {b}");
            assert_eq!(split_slot(b), Some((SPLIT_Q4_TABLE, i)));
        }
        // s148: 表 C (serve/q4 経路) の 16 行も (表 C, i) へ畳まれる
        for i in 0..SPLIT_SERVE_N {
            let c = SPLIT_SERVE_FIRST + i;
            assert!(c >= N_OPS, "生 id の範囲外: {c}");
            assert_eq!(split_slot(c), Some((SPLIT_SERVE_TABLE, i)), "表 C の行 {i} が折れない");
            assert_eq!(split_row_name(SPLIT_SERVE_TABLE, i), Some(SPLIT_SERVE_NAMES[i]));
            assert_eq!(split_rows(SPLIT_SERVE_TABLE), SPLIT_SERVE_N);
        }
        assert_eq!(split_row_name(SPLIT_SERVE_TABLE, SPLIT_SERVE_N), None);
        // 表 A / B / C の隣接: 表の末尾の次が次の表の先頭 (間が空かない) /
        // 一番後ろの表 (C) の次だけが範囲外
        assert_eq!(split_slot(SPLIT_Q4_FIRST - 1), Some((SPLIT_CHAIN_TABLE, SPLIT_CHAIN_ROWS - 1)));
        assert_eq!(split_slot(SPLIT_SERVE_FIRST - 1), Some((SPLIT_Q4_TABLE, SPLIT_Q4_N - 1)));
        assert_eq!(split_slot(SPLIT_SERVE_FIRST + SPLIT_SERVE_N), None);
        // 3 表の id 範囲は互いに交差しない (連番で隙間も重なりも無い)
        assert!(SPLIT_CHAIN_FIRST + SPLIT_CHAIN_ROWS == SPLIT_Q4_FIRST);
        assert!(SPLIT_Q4_FIRST + SPLIT_Q4_N == SPLIT_SERVE_FIRST);
        assert!(SPLIT_SERVE_FIRST >= D2H_GDN_PROJ_FIRST + D2H_GDN_PROJ_N);
        // 生 id・D2H 細分・既存細分の id は split_slot の範囲外 (誤って積まない)
        for op in (0..N_OPS).chain([
            D2H_LOGITS,
            D2H_LOGITS_PF_HELD,
            D2H_GDN_PF_FIRST,
            D2H_SRC_FIRST,
            D2H_GDN_PROJ_FIRST,
            ATT_GDN,
            ATT_GDN_SCAN,
        ]) {
            assert_eq!(split_slot(op), None, "既存 id {op} が分割表に入る");
        }
        // 待ち行はどちらの表でも**その表の行数**の範囲内
        assert!(SPLIT_WAIT_ROW[SPLIT_CHAIN_TABLE] < SPLIT_CHAIN_ROWS);
        assert!(SPLIT_WAIT_ROW[SPLIT_Q4_TABLE] < SPLIT_Q4_N);
        assert_eq!(SPLIT_WAIT_ROW[SPLIT_CHAIN_TABLE], CHAIN_SYNC_WAIT - SPLIT_CHAIN_FIRST);
        assert_eq!(SPLIT_WAIT_ROW[SPLIT_Q4_TABLE], Q4P_SYNC_WAIT - SPLIT_Q4_FIRST);
        // 表 B の `lin_*` の id は表 A の行数に依らず `SPLIT_Q4_FIRST + i` で連番
        assert_eq!(Q4P_OUT_ALLOC, SPLIT_Q4_FIRST);
        assert_eq!(Q4P_HOST_COPY, SPLIT_Q4_FIRST + SPLIT_Q4_N - 1);
        // 表 A の `chain_kernel` は 9 行目 (8 段 + 待ちの内訳 1 本)
        assert_eq!(CHAIN_KERNEL_WAIT, SPLIT_CHAIN_FIRST + SPLIT_CHAIN_ROWS - 1);
        // 行名は表ごとに引ける (行数を超える添字は None)
        for i in 0..SPLIT_CHAIN_ROWS {
            assert_eq!(split_row_name(SPLIT_CHAIN_TABLE, i), Some(SPLIT_CHAIN_NAMES[i]));
            assert_eq!(split_rows(SPLIT_CHAIN_TABLE), SPLIT_CHAIN_ROWS);
        }
        for i in 0..SPLIT_Q4_N {
            assert_eq!(split_row_name(SPLIT_Q4_TABLE, i), Some(SPLIT_Q4_NAMES[i]));
            assert_eq!(split_rows(SPLIT_Q4_TABLE), SPLIT_Q4_N);
        }
        assert_eq!(split_row_name(SPLIT_CHAIN_TABLE, SPLIT_CHAIN_ROWS), None);
        assert_eq!(split_row_name(SPLIT_Q4_TABLE, SPLIT_Q4_N), None);
        // 表 A の 9 行目 (chain_kernel) は表 B には存在しない
        assert_eq!(split_row_name(SPLIT_Q4_TABLE, SPLIT_CHAIN_ROWS - 1), None);
        assert_eq!(split_rows(SPLIT_TABLES), 0);
    }

    /// s138 (Task B6): `split_flush_step` は段を**排他**に割る — 積んだ段の和が
    /// 1 ステップの host 時間に一致し、段が起きなかった行の calls は増えない。
    /// `split_sample` (区間を直接 1 本積む版) と同じ会計に入る。PROFILE 無効時は
    /// 何も積まない。
    ///
    /// 表ごとに行数が違うので、表 A は `chain_kernel` (9 行目) まで、
    /// 表 B は 8 行目 (`lin_host_copy`) までを 1 回ずつ確かめる。
    #[test]
    fn split_flush_is_exclusive_and_display_only() {
        with_prof_counters(|| {
            reset();
            set_phase(Phase::Decode);
            // 有効時: launch → sync → cpu の 3 段に割る (経過時間は測り直さない —
            // sleep の粒度に依存しないよう、段の値は自分で決める)
            let mut acc = [0.0; SPLIT_ROWS];
            acc[CHAIN_LAUNCH - SPLIT_CHAIN_FIRST] = 0.002;
            acc[CHAIN_SYNC_WAIT - SPLIT_CHAIN_FIRST] = 0.010;
            acc[CHAIN_CPU - SPLIT_CHAIN_FIRST] = 0.001;
            let sum: f64 = acc.iter().sum();
            assert!((sum - 0.013).abs() < 1e-12, "段の和が host 時間に一致しない: {sum}");
            // 触っていない段は 0 (排他)
            assert_eq!(acc[CHAIN_D2H_LOGITS - SPLIT_CHAIN_FIRST], 0.0);
            assert_eq!(acc[CHAIN_OUTER - SPLIT_CHAIN_FIRST], 0.0);
            assert_eq!(acc[CHAIN_KERNEL_WAIT - SPLIT_CHAIN_FIRST], 0.0);
            // 表へ流し込む: 行ごとに時間と「起きた回数」が入る
            split_flush_step(SPLIT_CHAIN_TABLE, &acc, 1, None);
            let p = Phase::Decode.idx();
            for row in [CHAIN_LAUNCH, CHAIN_SYNC_WAIT, CHAIN_CPU] {
                let i = row - SPLIT_CHAIN_FIRST;
                assert!(SPLIT_NS[p][SPLIT_CHAIN_TABLE][i].load(Ordering::Relaxed) > 0);
                assert_eq!(SPLIT_CALLS[p][SPLIT_CHAIN_TABLE][i].load(Ordering::Relaxed), 1);
            }
            for row in [CHAIN_D2H_LOGITS, CHAIN_OUTER, CHAIN_KERNEL_WAIT] {
                let i = row - SPLIT_CHAIN_FIRST;
                assert_eq!(
                    SPLIT_CALLS[p][SPLIT_CHAIN_TABLE][i].load(Ordering::Relaxed),
                    0,
                    "起きていない段の calls は増えない (行 {i})"
                );
            }
            // 表 B (8 行) 側も同じ規約: 最終行 (`lin_host_copy`) まで積める
            set_phase(Phase::Prefill);
            let mut accq = [0.0; SPLIT_ROWS];
            accq[Q4P_OUT_ALLOC - SPLIT_Q4_FIRST] = 0.004;
            accq[Q4P_HOST_COPY - SPLIT_Q4_FIRST] = 0.003;
            split_flush_step(SPLIT_Q4_TABLE, &accq, 1, None);
            let pp = Phase::Prefill.idx();
            for row in [Q4P_OUT_ALLOC, Q4P_HOST_COPY] {
                let i = row - SPLIT_Q4_FIRST;
                assert!(SPLIT_NS[pp][SPLIT_Q4_TABLE][i].load(Ordering::Relaxed) > 0);
                assert_eq!(SPLIT_CALLS[pp][SPLIT_Q4_TABLE][i].load(Ordering::Relaxed), 1);
            }
            // 表 A の 9 行目は表 B の範囲外 — 表 B の flush では 8 行しか見ない
            // (acc に 9 行目の値が入っていても表 B には積まれない)
            set_phase(Phase::Decode);
            // 表示専用: 生 id (N_OPS 未満) には 1 サンプルも入らない。会計の static は
            // プロセス共有で、`with_prof_counters` のロックを取らない他テストが
            // 並走して値を積むため、絶対値 0 ではなく「分割の前後で不変」で見る。
            let snap: Vec<(u64, u64)> = (0..N_OPS)
                .map(|i| {
                    (
                        NS[p][i].load(Ordering::Relaxed),
                        CALLS[p][i].load(Ordering::Relaxed),
                    )
                })
                .collect();
            let mut acc2 = [0.0; SPLIT_ROWS];
            acc2[CHAIN_OUTER - SPLIT_CHAIN_FIRST] = 0.003;
            split_flush_step(SPLIT_CHAIN_TABLE, &acc2, 1, None);
            for i in 0..N_OPS {
                assert_eq!(
                    (NS[p][i].load(Ordering::Relaxed), CALLS[p][i].load(Ordering::Relaxed)),
                    snap[i],
                    "生 id {i} を分割表が汚した"
                );
            }
            reset();
        });
    }

    /// s138: 範囲外の id は黙って捨てる (誤って生 id へ落ちない)。
    /// `split_sample` は id を `split_slot` で畳むので、表の外 (生 id・隣の表の
    /// 終端より先) は会計に届かない。
    #[test]
    fn split_sample_ignores_unknown_ids() {
        with_prof_counters(|| {
            reset();
            set_phase(Phase::Decode);
            // 生 id は分割表に入らない
            split_sample(QKV, 1_000, 1);
            split_sample(GATE_UP, 1_000, 1);
            // 表 A の終端より先 (表 A の終端 + 1 = 表 B の先頭だが、次の表の外側の id として追加済み)
            // s148: 表 C ができたので、表 B の終端 (= 表 C の先頭) は「範囲外」では
            // なくなった。本当に全表の外にある id で確かめる (表 C の終端 + 1)。
            let outside = SPLIT_SERVE_FIRST + SPLIT_SERVE_N + 1;
            assert_eq!(outside, SPLIT_SERVE_FIRST + SPLIT_SERVE_N + 1);
            assert_eq!(split_slot(outside), None, "probe id がどの表にも入ってはいけない");
            // 表 C の先頭と終端も表に入る (probe が「表の外」であることの対照)
            assert_eq!(split_slot(SPLIT_SERVE_FIRST), Some((SPLIT_SERVE_TABLE, 0)));
            assert_eq!(
                split_slot(SPLIT_SERVE_FIRST + SPLIT_SERVE_N - 1),
                Some((SPLIT_SERVE_TABLE, SPLIT_SERVE_N - 1))
            );
            split_sample(outside, 1_000, 1);
            for t in 0..SPLIT_TABLES {
                for i in 0..SPLIT_ROWS {
                    assert_eq!(
                        SPLIT_NS[Phase::Decode.idx()][t][i].load(Ordering::Relaxed),
                        0,
                        "範囲外の id が表 {t} の行 {i} に入った"
                    );
                    assert_eq!(
                        SPLIT_CALLS[Phase::Decode.idx()][t][i].load(Ordering::Relaxed),
                        0
                    );
                }
            }
            // 表の中の id は入る (上の「入らない」が範囲の取り違えでないことの対照)
            split_sample(CHAIN_KERNEL_WAIT, 1_000, 1);
            split_sample(Q4P_HOST_COPY, 2_000, 1);
            let p = Phase::Decode.idx();
            assert_eq!(
                SPLIT_NS[p][SPLIT_CHAIN_TABLE][CHAIN_KERNEL_WAIT - SPLIT_CHAIN_FIRST]
                    .load(Ordering::Relaxed),
                1_000
            );
            assert_eq!(
                SPLIT_NS[p][SPLIT_Q4_TABLE][Q4P_HOST_COPY - SPLIT_Q4_FIRST].load(Ordering::Relaxed),
                2_000
            );
            // `calls = 0` は時間だけ積む (起きなかった段を分母から外す用)
            split_sample(CHAIN_CPU, 500, 0);
            assert_eq!(
                SPLIT_NS[p][SPLIT_CHAIN_TABLE][CHAIN_CPU - SPLIT_CHAIN_FIRST].load(Ordering::Relaxed),
                500
            );
            assert_eq!(
                SPLIT_CALLS[p][SPLIT_CHAIN_TABLE][CHAIN_CPU - SPLIT_CHAIN_FIRST]
                    .load(Ordering::Relaxed),
                0
            );
            reset();
        });
    }

    /// s140 (Task B6b): 分割 tick の件数・サンプル数・カバレッジ分母が
    /// 1 call = 1 tick で対応し、ヘッダにカバレッジ行が出る。
    /// 「サンプル数 = tick 数」「分割の和 = バケット ms」を数で確かめる
    /// (s138 は tick の分母を持たず、608 call のバケットに 802 サンプルを
    /// 積んでいた = 件数が食い違ったまま表を見ていた)。
    #[test]
    fn lin_split_covers_every_call_once_and_header_prints_coverage() {
        with_prof_counters(|| {
            reset();
            set_phase(Phase::Prefill);
            // 3 call の線形ステージ: バケット 10 ms / 分割の和 10 ms
            for _ in 0..3 {
                count_linear_stage(GATE_UP, 10_000_000, 1, true);
                split_tick_call(SPLIT_Q4_TABLE);
                split_sample(GATE_UP, 10_000_000, 0);
                split_sample(Q4P_SYNC_WAIT, 10_000_000, 1);
            }
            // 分割できなかった 1 call (バケットだけ 4 ms) — 分母には入る
            count_linear_stage(QKV, 4_000_000, 1, true);
            let (calls, ns) = linear_stage_total(Phase::Prefill);
            assert_eq!(calls, 4, "分母は線形呼び出し数");
            assert_eq!(ns, 34_000_000, "分母は線形バケットの ns");
            assert_eq!(split_tick_count(SPLIT_Q4_TABLE), 3, "tick は分割できた call 数");
            // サンプル数 = 表 B の calls の和 (1 call で sync 段 1 + 総和段 1 は
            // 「段の回数」なので、ここでは tick と一致するかを検査する)
            // ここは「段の回数」の和 (calls 列 = バケット状の段の発生数)
            assert_eq!(split_sample_count(Phase::Prefill, SPLIT_Q4_TABLE), 3);
            let s = summary();
            let cov = s
                .lines()
                .find(|l| l.contains("coverage: lin "))
                .unwrap_or_else(|| panic!("カバレッジ行が無い\n{s}"));
            assert!(cov.contains("pct of linear ms"), "{cov}");
            assert!(cov.contains("ticks"), "{cov}");
            // 表 B の和 / 分母 = 30 / 34 ms = 88.2%
            assert!(cov.contains("88.2 pct"), "{cov}");
            assert!(cov.contains("lin 4 calls split 3"), "{cov}");
            // ヘッダの従来の形も壊れていない
            let head = s
                .lines()
                .find(|l| l.contains("samples: decode steps"))
                .expect("samples ヘッダ");
            assert!(head.contains("lin calls 4 / split 3"), "{head}");
            // 分母 4 call のうち tick したのは 3 call (4 本目は分割されていない)
            assert!(cov.contains("lin 4 calls split 3"), "{cov}");
            reset();
        });
    }

    /// s140 (Task B6b): chain の様相 — stage を x ms、sync 待ちを y ms に積む。
    fn chain_fixture_steps(steps: usize, stage: f64, wait: f64) {
        set_phase(Phase::Decode);
        for _ in 0..steps {
            // 1 ステップ = 1 サンプル (cuda::chain_step_entered と同じ数え方)
            split_tick_call(SPLIT_CHAIN_TABLE);
            let mut acc = [0.0; SPLIT_ROWS];
            acc[CHAIN_LAUNCH - SPLIT_CHAIN_FIRST] = stage;
            acc[CHAIN_SYNC_WAIT - SPLIT_CHAIN_FIRST] = wait;
            split_flush_step(SPLIT_CHAIN_TABLE, &acc, 1, None);
        }
    }

    /// s140 (Task B6b): chain の排他の和 = ステップ数 x (段 + 待ち) で、
    /// サンプル数 (tick) がステップ数と一致する。tick を打たないステップ
    /// (表に積まない) があれば分母とサンプル数がずれる = 付け漏れが見える。
    #[test]
    fn chain_exclusive_sum_matches_steps_and_sample_count() {
        with_prof_counters(|| {
            reset();
            chain_fixture_steps(4, 0.002, 0.030);
            let p = Phase::Decode.idx();
            let sum: u64 = (0..SPLIT_CHAIN_ROWS)
                .map(|i| SPLIT_NS[p][SPLIT_CHAIN_TABLE][i].load(Ordering::Relaxed))
                .sum();
            // 1 ステップ = 32 ms (段 2 + 待ち 30) x 4 = 128 ms
            assert_eq!(sum, 128_000_000, "排他の和がサンプルの壁時計の和にならない");
            assert_eq!(split_tick_count(SPLIT_CHAIN_TABLE), 4);
            // 表 A のサンプル数も 4 ステップ分 (launch / wait の 2 段)
            assert_eq!(split_sample_count(Phase::Decode, SPLIT_CHAIN_TABLE), 8);
            let s = summary();
            let head = s
                .lines()
                .find(|l| l.contains("samples: decode steps"))
                .expect("samples ヘッダ");
            // chain split は launch 段の calls (= 4) — 集計行の意味は s138 のまま
            assert!(head.contains("chain split 4"), "{head}");
            let cov = s
                .lines()
                .find(|l| l.contains("coverage: lin "))
                .expect("カバレッジ行");
            assert!(cov.contains("samples chain 8 / 4 ticks"), "{cov}");
            reset();
        });
    }

    /// s138: サンプル数のヘッダが decode の steps と chain の分割回数を並べ、
    /// 分割表の行 (chain_* / lin_*) が summary に出る。TOTAL は従来どおり
    /// self の和で、分割表を足し込まない (表示専用)。
    #[test]
    fn summary_prints_sample_counts_and_split_rows() {
        with_prof_counters(|| {
            reset();
            // decode 2 ステップ (op は 1 行ずつ): 分割は 2 回とも閉じる
            set_phase(Phase::Decode);
            for _ in 0..2 {
                add_sample(Phase::Decode, GATE_UP, 3_000_000, 1, 1, true);
                let mut acc = [0.0; SPLIT_ROWS];
                acc[CHAIN_LAUNCH - SPLIT_CHAIN_FIRST] = 0.002;
                acc[CHAIN_SYNC_WAIT - SPLIT_CHAIN_FIRST] = 0.010;
                acc[CHAIN_D2H_LOGITS - SPLIT_CHAIN_FIRST] = 0.001;
                split_flush_step(SPLIT_CHAIN_TABLE, &acc, 1, None);
            }
            // prefill: 線形ラッパ 1 call 分 (launch / sync / 確保)
            set_phase(Phase::Prefill);
            add_sample(Phase::Prefill, GATE_UP, 20_000_000, 1, 18, true);
            let mut acc = [0.0; SPLIT_ROWS];
            acc[Q4P_OUT_ALLOC - SPLIT_Q4_FIRST] = 0.004;
            acc[Q4P_H2D_STAGE - SPLIT_Q4_FIRST] = 0.001;
            acc[Q4P_KERNEL_LAUNCH - SPLIT_Q4_FIRST] = 0.002;
            acc[Q4P_SYNC_WAIT - SPLIT_Q4_FIRST] = 0.013;
            split_flush_step(SPLIT_Q4_TABLE, &acc, 1, None);
            set_phase(Phase::Decode);

            // 表はプロセス共有 (並走テストが値を消し得る) なので、このテストは
            // ロック内で積み終えてから 1 回だけ summary を作る。値の絶対性ではなく
            // 「ヘッダの形」と「表の出し分け」を見る。
            let s = summary();
            // 1) サンプル数のヘッダが 1 行出る (187 token の平均を 1 step と読まないため)
            let head = s
                .lines()
                .find(|l| l.contains("samples: decode steps"))
                .unwrap_or_else(|| panic!("samples ヘッダが無い\n{s}"));
            assert!(head.contains("decode steps "), "decode の steps: {head}");
            assert!(head.contains("chain split "), "chain の分割回数: {head}");
            assert!(head.contains("lin calls "), "線形ラッパの呼び出し数: {head}");
            // 2) 分割表 (cumulative の後) に chain_* / lin_* 行が出る。
            //    表は節の外 (summary の末尾) なので、SPLIT の見出しで切る。
            let dec = s
                .split("SPLIT DECODE")
                .nth(1)
                .unwrap_or_else(|| panic!("SPLIT DECODE が無い\n{s}"))
                .to_string();
            let pre = s
                .split("SPLIT PREFILL")
                .nth(1)
                .unwrap_or_else(|| panic!("SPLIT PREFILL が無い\n{s}"))
                .split("SPLIT DECODE")
                .next()
                .unwrap()
                .to_string();
            for name in ["chain_launch", "chain_sync_wait", "chain_d2h_logits"] {
                let line = dec
                    .lines()
                    .find(|l| l.split_whitespace().next() == Some(name))
                    .unwrap_or_else(|| panic!("decode の分割に {name} 行が無い\n{s}"));
                let cols: Vec<&str> = line.split_whitespace().collect();
                let calls: u64 = cols[2].parse().expect("calls は数値");
                assert!(calls >= 2, "calls はステップ数 (>=2): {line}");
            }
            for name in ["lin_out_alloc", "lin_kernel", "lin_sync_wait"] {
                let line = pre
                    .lines()
                    .find(|l| l.split_whitespace().next() == Some(name))
                    .unwrap_or_else(|| panic!("prefill の分割に {name} 行が無い\n{s}"));
                let cols: Vec<&str> = line.split_whitespace().collect();
                let calls: u64 = cols[2].parse().expect("calls は数値");
                assert!(calls >= 1, "calls は線形ラッパの呼び出し数 (>=1): {line}");
            }
            // 表の出し分け: decode 側に lin_* / prefill 側に chain_* は出ない
            assert!(!dec.contains("lin_kernel"));
            assert!(!pre.contains("chain_launch"));
            // 3) 表示専用: decode の TOTAL は self の和のまま (分割表を足し込まない)。
            //    並走テストの値が混ざるので、TOTAL = 行の self の和 を確かめる。
            // s148: 末尾に SPLIT SERVE (decode) の節も `[decode]` として現れるので、
            // 本体の decode 節は「最後の [decode]」ではなく先頭側から取る。
            let decode_section = s.split("[decode]").nth(1).expect("decode 節が無い");
            let decode_section = decode_section
                .split("SPLIT ")
                .next()
                .unwrap_or(decode_section);
            // 節の見出しは `[decode]` の直後 (節の切れ目で落とした) なので、
            // TOTAL 行は節の先頭そのもの。行の先頭は `" total ..."` になる。
            let dec_total = decode_section
                .lines()
                .find(|l| l.trim_start().starts_with("total "))
                .expect("decode TOTAL 行が無い");
            // `  [decode] total 6.0 ms ...` → "total" の次が ms
            let total_ms: f64 = dec_total
                .split_whitespace()
                .skip_while(|t| *t != "total")
                .nth(1)
                .and_then(|v| v.parse().ok())
                .expect("decode TOTAL の ms");
            let mut self_sum = 0.0f64;
            // TOTAL 行より後ろは次の節 (`[cumulative]`) — 表の行だけを足す
            for line in decode_section
                .lines()
                .skip(2)
                .take_while(|l| !l.contains("[cumulative]"))
            {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() == 7 {
                    if let Ok(v) = cols[6].parse::<f64>() {
                        self_sum += v;
                    }
                }
            }
            assert!(
                (total_ms - self_sum).abs() < 0.2,
                "decode TOTAL は self の和 (分割表は足さない): total={total_ms} self_sum={self_sum}"
            );
            reset();
        });
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

    // -----------------------------------------------------------------------
    // s148 (Task B10): 表 C (serve/q4 経路) の会計。GPU 無しで検証できる範囲
    // -----------------------------------------------------------------------

    /// s148: 表 C の最小の段を手で積む合成ステップ (GPU 無し)。
    /// `step_begin/step_end` は本物を通すため、残余 (`sl_step_other`) の
    /// 計算まで含めた会計がそのまま検査できる。`steps` 回だけ 1 step ずつ進める。
    ///
    /// s151 (B12): サーバの decode ループと同じ形にする — ステップごとに
    /// (1) サンプラ 1 号、(2) `sl_lin` 1 call (内訳 3 段 + 残余 2 か所の 5 号を
    /// 1 call に畳む)、(3) `sl_step_final` / `sl_step_lay_other` 1 号ずつ、
    /// (4) `step_begin` → `step_end`。生 id (`SAMPLER`) も 1 回ずつ積む
    /// (ヘッダの decode steps は生 id の最大 call 数)。
    ///
    /// s155 (B16): 実順 (per-op 経路) に合わせた — `step_begin` → forward (投影 /
    /// lay_other / final) → `step_end` → サンプラ → 次の投影 + 層ループ →
    /// `step_end_tail`。残余 (`sl_step_other`) は **step_end_tail** が積む
    /// (s155 で締めの位置が `step_end` の後ろに移った)。
    ///
    /// フェーズは **decode** (COVERAGE のヘッダは prefill を見るため、この fixture は
    /// `push_serve_section` の decode 節 = 表 C の decode 行で検査する)。
    fn serve_fixture_steps(steps: usize) {
        set_phase(Phase::Decode);
        for _ in 0..steps {
            // バケットの駆動件数 (ヘッダの decode steps の分母は CALLS の最大値)
            crate::prof::time(SAMPLER, || {});
            let st = step_begin();
            // 線形呼び出し 1 回: 入口 → 内訳 (q4 タイルの 5 段) → 出口。
            // `sl_lin` の全体は `lin_call_end` が積む (`q4_dispatch` は
            // `maybe_gpu_linear_q4_into` の内側なので、ディスパッチ件数も 1 回)。
            serve_fixture_linear_call();
            count_lin_dispatch();
            let l = span_begin();
            span_end(l, SL_STEP_LAYERS_OTHER);
            let f = span_begin();
            span_end(f, SL_STEP_FINAL);
            // s154 (B15): `sl_step_other` の child (実経路は 1 step 1 回)。
            for op in [
                SL_SO_SAMPLE_PEN,
                SL_SO_STREAM,
                SL_SO_LOGITS,
                SL_SO_GAP_FWD,
                SL_SO_END,
            ] {
                let k = span_begin();
                span_end(k, op);
            }
            step_end(st);
            // step_end の後 = サンプラ → 次の投影 / 層ループ (実経路の後半)。
            let s = span_begin();
            span_end(s, SL_STEP_SAMPLER);
            let d = span_begin();
            span_end(d, SL_STEP_DETOK);
            serve_fixture_linear_call();
            count_lin_dispatch();
            let l2 = span_begin();
            span_end(l2, SL_STEP_LAYERS_OTHER);
            crate::prof::step_end_tail();
        }
    }

    /// s151 (B12): `maybe_gpu_linear_q4_into` の口 (`lin_call_begin`) だけを開く。
    fn serve_fixture_linear_begin() -> Option<std::time::Instant> {
        super::lin_call_begin()
    }

    /// s151 (B12): `q4_dispatch_tile` の closure 1 枚の中身 (4 段 + ピン留め)。
    ///
    /// s153 (Task B14): 段は**連続した 1 本の区間**で、`lin_stage_taken_ns` が読んだ
    /// ns を `span_end_ns` で表 C へ渡す (cuda.rs の実サイトと同じ形 = クロック読み
    /// 1 回)。ピン留めの確保は残余の中身 (other) だが、表 C には積まない —
    /// 実経路ではピン留めの区間もタイルの段の合間 (親の内側) で独立した段では
    /// ないため、積むと子の和が親を超える。ここでは
    /// `lin_other_close` だけ通して号数 (`lin_other_samples`) の会計を壊さない。
    fn serve_fixture_tile_stages() {
        let a = span_begin();
        let ns = super::lin_stage_taken_ns(a);
        span_end_ns(ns, SL_LIN_OUTALLOC);
        let b = span_begin();
        super::lin_other_close(b); // ピン留め確保 (残余の中身・表 C へは積まない)
        let c = span_begin();
        let ns = super::lin_stage_taken_ns(c);
        span_end_ns(ns, SL_LIN_H2D);
        let d = span_begin();
        let ns = super::lin_stage_taken_ns(d);
        span_end_ns(ns, SL_LIN_LAUNCH);
        let e = span_begin();
        let ns = super::lin_stage_taken_ns(e);
        span_end_ns(ns, SL_LIN_D2H);
    }

    /// s151 (B12): 線形呼び出し 1 回を、`maybe_gpu_linear_q4_into` と
    /// `q4_dispatch_tile` と同じ順・同じ号数で積む:
    ///   口 `lin_call_begin` → 入口の地 (other) → タイルの 5 段 → 重み登録 (other)
    ///   → 口 `lin_call_end` (ここで `sl_lin` 1 号 + `sl_lin_other` 1 号)
    /// **s148 は other が 3 か所でそれぞれ 1 号 (= 3 号) になっていた**。
    fn serve_fixture_linear_call() {
        let call_t = serve_fixture_linear_begin();
        // 入口の地 (shared_ctx / make_current)
        {
            let a = span_begin();
            super::lin_other_close(a);
        }
        // タイル 1 枚の内側
        serve_fixture_tile_stages();
        // 重み登録 (dispatch の地)
        {
            let a = span_begin();
            super::lin_other_close(a);
        }
        super::lin_call_end(call_t);
    }

    /// s148: 表 C の行を 1 本取り出す。COVERAGE はヘッダ (先頭の `[prefill]` より前)、
    /// `SPLIT SERVE` の節は末尾にあるので、行頭一致で探す (同名のヘッダ行と
    /// 取り違えないよう `starts_with` を使う)。
    fn serve_decode_line(s: &str, needle: &str) -> String {
        s.lines()
            .find(|l| l.trim_start().starts_with(needle))
            .unwrap_or_else(|| panic!("summary に {needle} で始まる行が無い\n{s}"))
            .to_string()
    }

    /// s148 (Task B10): **表 C のサンプル数 = ステップ数**。小さな合成ステップ数
    /// (7) を回し、`sl_step` のサンプル数がステップ数と一致し、内訳の和が全体を
    /// 超えず、残余 (`sl_step_other`) が「全体 − 内訳」に一致することを固定する
    /// (GPU 不要・DENNO_PROFILE はテスト内の上書きで有効化)。
    #[test]
    fn serve_table_samples_match_steps_without_gpu() {
        with_prof_counters(|| {
            reset();
            let steps = 7u64;
            serve_fixture_steps(steps as usize);
            let p = Phase::Decode.idx();
            // s151 (B12): 線形の期待もサンプルも `sl_lin` (タイル入口) の 1 号。
            // s155 (B16): per-op の 1 step は「前 step の logits D2H」+「この step の
            // forward」で**投影が 2 回**走る (`sl_step` の内訳としては step の内側に
            // 1 本だけ入る)。ここでは fixture が実順に 2 本積むので 2 × steps。
            assert_eq!(
                serve_row_ticks(p, SL_LIN),
                steps * 2,
                "sl_lin の号数 ≠ steps × 2 (per-op は 1 step に投影 2 本)"
            );
            assert_eq!(
                lin_dispatch_calls(Phase::Decode),
                steps * 2,
                "ディスパッチ件数 ≠ steps × 2"
            );
            // 1 step = 1 サンプル (この 1 点が崩れると coverage の分母が嘘になる)
            assert_eq!(serve_row_calls(p, SL_STEP), steps, "sl_step のサンプル数 ≠ ステップ数");
            assert_eq!(
                serve_row_calls(p, SL_STEP_SAMPLER),
                steps,
                "sampler 段が毎 step 積まれていない"
            );
            assert_eq!(serve_row_calls(p, SL_STEP_DETOK), steps, "detok 段が毎 step 積まれていない");
            // 内訳の和は全体を超えない / 残余は「全体 − 内訳」と一致する
            let whole = serve_row_ns(p, SL_STEP);
            let other = serve_row_ns(p, SL_STEP_OTHER);
            assert_eq!(serve_row_calls(p, SL_STEP_OTHER), steps, "残余は毎 step 1 サンプル");
            assert_eq!(step_gap(SL_STEP), 0, "全体 − 内訳 = 残余 が 0 でない");
            assert!(other <= whole, "残余 {other} ns が全体 {whole} ns を超える");
            // 表 C の節に「1 step = 1 サンプル」がそのまま出る
            let s = summary();
            let line = serve_decode_line(&s, "SPLIT SERVE (decode");
            assert!(line.contains("q4 serve path"), "{line}");
            let row = serve_decode_line(&s, "sl_step  ");
            assert!(row.contains(&steps.to_string()), "sl_step 行にステップ数が無い: {row}");
            // COVERAGE の decode 節は「steps = サンプル」で MISS を出さない
            let head = serve_decode_line(&s, "COVERAGE [decode]: steps");
            assert!(
                head.contains(&format!("steps {steps} / samples {steps} [ok]")),
                "{head}"
            );
            assert!(!head.contains("COVERAGE MISS"), "steps が一致しているのに MISS: {head}");
            reset();
        });
    }

    /// s148 (Task B10): COVERAGE の「steps」の期待値は**ヘッダと同じ定義** (生 id の
    /// 最大 call 数) で、表 C の `sl_step` の実測サンプル数と突き合わせられること。
    /// 期待だけあってサンプル 0 のときは per-bucket でも MISS として名指しされる。
    #[test]
    fn serve_coverage_marks_miss_and_reports_wall_share() {
        with_prof_counters(|| {
            reset();
            let p = Phase::Decode.idx();
            // s151 (B12): 期待 steps の分母は表 C の実測号数 = 「まだ積んでいない」= 0。
            // 生 id の call 数 (プロセス全体のカウンタ) は分母にしない — ヘッダの
            // `samples:` 行だけが従来どおり生 id の最大 call 数を見る。
            for _ in 0..2 {
                crate::prof::time(SAMPLER, || {});
            }
            let exp = serve_expected(p);
            let step_exp = exp
                .iter()
                .find(|(_, row, ..)| *row == SL_STEP)
                .expect("sl_step の期待");
            assert_eq!(step_exp.2, 0, "期待がプロセス全体のカウンタ由来になっている");
            assert_eq!(serve_row_calls(p, SL_STEP), 0);
            // 壁時計を 1 本積む (計測 ON のときだけ数える)
            add_request_wall(500_000_000); // 500 ms
            let s = summary();
            let head = serve_decode_line(&s, "COVERAGE [decode]: steps");
            assert!(head.contains("steps 0 / samples 0 [ok]"), "{head}");
            // MISS を名指しする経路は「期待があるのにサンプル 0」— 線形だけは
            // ディスパッチ件数で期待するので、件数だけ積んでサンプル 0 にすると走る。
            count_lin_dispatch();
            let s = summary();
            let status = s
                .lines()
                .rev()
                .find(|l| l.trim_start().starts_with("COVERAGE status:"))
                .unwrap_or_else(|| panic!("COVERAGE status 行が無い\n{s}"));
            // s155 (B16): 名指しの**集合**を見る (行が増えても壊れない形)。
            // ここが sl_lin を落とすと「線形の付け漏れ」が静かに通る。
            assert!(
                status.contains("sl_lin"),
                "線形の付け漏れが MISS として名指しされない: {status}"
            );
            // リクエストの壁時計が同じ行に出る (帰属 ms の分母)
            let head = serve_decode_line(&s, "COVERAGE [decode]: steps");
            assert!(head.contains("requests 1 wall 500.0 ms"), "壁時計が出ない: {head}");
            // 実測のステップを積むと期待と一致する (期待 ≠ 捏造)
            reset();
            serve_fixture_steps(2);
            assert_eq!(serve_row_calls(p, SL_STEP), 2);
            assert_eq!(serve_expected(p)[0].2, serve_row_calls(p, SL_STEP), "期待 ≠ 実測サンプル");
            reset();
        });
    }

    /// s148 (Task B10): 計測 OFF (DENNO_PROFILE 未設定) は表 C に 1 サンプルも積まず、
    /// 時刻も読まない (`step_begin` / `span_begin` が `None`)。表 C の関数を
    /// 直に呼んでも会計は 0 のまま — 既定経路が厳密な no-op であることの固定。
    #[test]
    fn serve_table_is_strict_noop_when_profiling_off() {
        with_prof_counters_off(|| {
            let st = step_begin();
            assert!(st.is_none(), "計測 OFF なのに step_begin が時刻を読んだ");
            assert!(span_begin().is_none(), "計測 OFF なのに span_begin が時刻を読んだ");
            step_end(st);
            span_end(None, SL_STEP_SAMPLER);
            // s154 (B15): 追加した段も OFF では行に 1 ns も入らない。
            for op in [SL_LAY_GDN_SCAN, SL_LAY_FINAL, SL_SO_LIN_SUM, SL_SO_END] {
                span_end(span_begin(), op);
            }
            push(SL_STEP, 1_000, 1);
            count_lin_dispatch();
            add_request_wall(1_000);
            let p = Phase::Decode.idx();
            for row in 0..SPLIT_SERVE_N {
                let op = SPLIT_SERVE_FIRST + row;
                assert_eq!(serve_row_ns(p, op), 0, "計測 OFF で表 C の行 {row} に ns が入った");
                assert_eq!(
                    serve_row_calls(p, op),
                    0,
                    "計測 OFF で表 C の row {row} に sample が入った"
                );
            }
            assert_eq!(lin_dispatch_calls(Phase::Prefill), 0);
            assert_eq!(lin_dispatch_calls(Phase::Decode), 0);
            assert_eq!(request_wall(), (0, 0));
        });
    }

    /// s148: 計測を**明示的に無効にして** `f` を実行するガード (OFF 側の no-op を
    /// 検査する用)。`with_prof_counters` の対。
    fn with_prof_counters_off<T>(f: impl FnOnce() -> T) -> T {
        // P74: `with_prof_counters` と同じ理由で moe 側のロックも取る
        // (`serve_on()` の第 2 項 `crate::telemetry_on()` を並走テストが
        //  true にすると、計測 OFF でも `step_begin` が時刻を読む)。
        let _moe = crate::model::moe_expert::moe_test_lock();
        let _guard = lock_counters();
        *FORCE_ENABLED.lock().unwrap_or_else(|e| e.into_inner()) = Some(false);
        let _on = ForceEnabled(false);
        reset();
        f()
    }

    // -----------------------------------------------------------------------
    // s151 (Task B12): serve 経路の会計を「サーバの実呼び出し形」で固定する
    // -----------------------------------------------------------------------

    /// s151 (B12): 1 回の fake serve ループ (prefill 1 forward + decode N step) を
    /// サーバと同じ粒度で回す。prefill は `sl_step_prep` を 1 号 + 線形 1 call、
    /// decode は `step_begin` → forward (embed / 投影 / 層ループ / final / logits) →
    /// `sl_so_*` → `step_end` → サンプラ → 次の投影。**時刻の値は問わない**
    /// (会計の形だけを見る)。
    ///
    /// s155 (B16): 順序を **per-op 経路の実順**にした。per-op は 1 step の
    /// 呼び出し ("投影 → サンプラ → step_end") の中で「前 step の logits D2H」と
    /// 「この step の forward」を回すので、兄弟 (`sl_lin` / `sl_step_lay_other`) は
    /// 入口側と `step_end` の後の両方に来る。ここが s154 の CPU 試験の盲点
    /// (fixture は全部の段を `step_end` の前に積んでいた) で、`other == step` の
    /// 壊れ方を再現できていなかった。
    fn serve_fake_loop_prefill_then_decode(steps: usize) {
        // prefill (エンジンは prefill forward を `sl_step_prep` で積む)
        set_phase(Phase::Prefill);
        let pf = span_begin();
        serve_fixture_linear_call();
        span_end(pf, SL_STEP_PREP);
        // decode
        set_phase(Phase::Decode);
        for _ in 0..steps {
            // s154 (B15): 実経路の形を保つ — step 入口で embed と (前 step の lin を)
            // 控え、内側に `sl_lin` / 層内の実測の子 / final を積み、`step_end` の
            // **後**に `sl_so_end` を置く (エンジンは `step_end` で締めてから戻る)。
            let st = step_begin();
            let emb = span_begin();
            span_end(emb, SL_LAY_EMBED);
            // 前 step の logits D2H = この step の先頭の投影 (sl_step_lay_other の外側)
            serve_fixture_linear_call();
            let v = span_begin();
            span_end(v, SL_SO_LIN_SUM);
            let l = span_begin();
            span_end(l, SL_STEP_LAYERS_OTHER);
            let ln = span_begin();
            span_end(ln, SL_LAY_NORM_IN);
            let f = span_begin();
            span_end(f, SL_STEP_FINAL);
            let ff = span_begin();
            span_end(ff, SL_LAY_FINAL);
            // s154 (B15): 実経路の detok / delta / stop 判定 (`sl_so_stream`) と
            // ホスト側の logits 反映 (`sl_so_logits`) も 1 step 1 本。
            let ss = span_begin();
            span_end(ss, SL_SO_STREAM);
            let sl = span_begin();
            span_end(sl, SL_SO_LOGITS);
            let sg = span_begin();
            span_end(sg, SL_SO_GAP_FWD);
            step_end(st);
            let se = span_begin();
            span_end(se, SL_SO_END);
            // サンプラと「この step の forward の本体」は step_end の後 (= 次の呼び出し)。
            let s = span_begin();
            span_end(s, SL_STEP_SAMPLER);
            let sp = span_begin();
            span_end(sp, SL_SO_SAMPLE_PEN);
            let d = span_begin();
            span_end(d, SL_STEP_DETOK);
            serve_fixture_linear_call();
            let l2 = span_begin();
            span_end(l2, SL_STEP_LAYERS_OTHER);
            crate::prof::step_end_tail();
        }
    }

    /// s151 (B12): **1 decode step = 1 サンプル**。fake serve ループ 1 ステップを
    /// 回して、`steps == samples == 1`、`step_begin/step_end` が入れ子にならない、
    /// COVERAGE の期待値が `sl_step` の実測号数と同じであることを固定する。
    #[test]
    fn serve_step_ticks_once_per_decode_step_on_serve_loop() {
        with_prof_counters(|| {
            reset();
            assert_eq!(step_open_count(), 0, "reset 直後に開いたステップが残っている");
            serve_fake_loop_prefill_then_decode(1);
            let d = Phase::Decode.idx();
            assert_eq!(step_open_count(), 0, "step_begin と step_end が対になっていない");
            assert_eq!(serve_row_ticks(d, SL_STEP), 1, "sl_step の号数 ≠ ステップ数");
            assert_eq!(serve_ticks(d), 1);
            assert_eq!(serve_row_ticks(d, SL_STEP_FINAL), 1, "sl_step_final が毎 step 積まれない");
            assert_eq!(serve_row_ticks(d, SL_STEP_SAMPLER), 1, "sl_step_sampler が毎 step 積まれない");
            // s155 (B16): 層ループは step の前半 (forward) と後半 (次の呼び出し) の 2 本。
            assert_eq!(
                serve_row_ticks(d, SL_STEP_LAYERS_OTHER),
                2,
                "sl_step_lay_other の号数 ≠ 2 (per-op は 1 step に前後 2 本)"
            );
            // COVERAGE の期待 steps はプロセス全体のカウンタではなく表 C の実測
            let exp = serve_expected(d);
            for (_, row, e, _) in &exp {
                let samples = serve_row_ticks(d, *row);
                if *e > 0 {
                    assert!(samples > 0, "期待 {e} なのに行 {row} のサンプルが 0");
                }
            }
            // 3 ステップに増やすと号数も 3 (1 step 1 号の比例)
            reset();
            serve_fake_loop_prefill_then_decode(3);
            assert_eq!(serve_row_ticks(d, SL_STEP), 3);
            assert_eq!(step_open_count(), 0);
            reset();
        });
    }

    /// s155 (Task B16): **`sl_step_other` は「step − 名付きの兄弟」の実残余**。
    ///
    /// B15 の GPU 実測では `sl_step` と `sl_step_other` が同じ値 (131.737 ms/step)
    /// だった = 何も名付けられていない。CPU の試験がこれを見逃したのは、fixture が
    /// lin / lay_other を**全部 `step_end` の前**に積んでいたため (`own` の比較集合が
    /// 同じになり差が 0 でなくても、other が step と同値になる経路を通らなかった)。
    ///
    /// ここでは per-op 経路の実順 (step_begin → 投影 → lay_other → final →
    /// step_end → サンプラ → 次の投影 → step_end_tail) で 1 step を積み、
    ///   ① named siblings (sl_lin + sl_step_lay_other) > 0
    ///   ② other < step
    ///   ③ other + siblings <= step (兄弟の外側 = 前後の地だけが残余)
    /// を固定する。s154 (4b83cbc) の `own = payload1 − payload0` は集合違いで
    /// 恒等的に 0 になるので ② が落ちる。
    #[test]
    fn serve_step_other_is_step_minus_named_siblings() {
        with_prof_counters(|| {
            reset();
            set_phase(Phase::Decode);
            let d = Phase::Decode.idx();
            let mark = request_begin();
            // 1 step: 入口 → forward (embed / 投影 / 層ループ / final / logits) →
            // step_end → サンプラ → 次の投影 + 層ループ → 締め。
            let st = step_begin();
            let emb = span_begin();
            span_end(emb, SL_LAY_EMBED);
            serve_fixture_linear_call();
            // s163 (K4): 実経路の 1 step は投影を数百本ふくむ (tail は 1 本) ので、
            // 合成 fixture でも step の内側の投影を tail より十分に厚くする。ns スケールで
            // 「step 内の兄弟」と「step_end の後に閉じる兄弟」が同程度だと、
            // `other + siblings <= step` が実測ノイズで反転する (1 本のときは 20 回中
            // 7 回落ちた)。比較式は緩めず、本数だけを実経路の比に寄せる。
            for _ in 0..15 {
                serve_fixture_linear_call();
            }
            let lo = span_begin();
            span_end(lo, SL_STEP_LAYERS_OTHER);
            let fi = span_begin();
            span_end(fi, SL_STEP_FINAL);
            let lg = span_begin();
            span_end(lg, SL_SO_LOGITS);
            step_end(st);
            let sm = span_begin();
            span_end(sm, SL_STEP_SAMPLER);
            serve_fixture_linear_call();
            let lo2 = span_begin();
            span_end(lo2, SL_STEP_LAYERS_OTHER);
            step_end_tail();
            request_end(mark);

            let step = serve_row_ns(d, SL_STEP);
            let other = serve_row_ns(d, SL_STEP_OTHER);
            let sib_lin = serve_row_ns(d, SL_LIN);
            let sib_lay = serve_row_ns(d, SL_STEP_LAYERS_OTHER);
            let siblings = sib_lin + sib_lay;
            assert_eq!(serve_row_ticks(d, SL_STEP), 1, "1 step 1 サンプルでない");
            assert_eq!(serve_row_ticks(d, SL_STEP_OTHER), 1, "残余は毎 step 1 サンプル");
            assert!(step > 0, "sl_step が 0 ns");
            assert!(siblings > 0, "名付きの兄弟 (sl_lin + lay_other) が 0 ns");
            assert!(
                other < step,
                "other {other} ns == step {step} ns (残余が全体を飲んでいる = B15 の再発)"
            );
            assert!(
                other + siblings <= step,
                "other {other} + siblings {siblings} が step {step} を超える"
            );
            // 兄弟を積まない step では other == step (残余の箱が空でないことも固定)
            reset();
            let st = step_begin();
            step_end(st);
            step_end_tail();
            assert_eq!(
                serve_row_ns(d, SL_STEP_OTHER),
                serve_row_ns(d, SL_STEP),
                "兄弟が 1 本も無い step で other != step"
            );
            // summary の COVERAGE も「other < step」を見る (B15 の再発で MISS)
            let s = summary();
            let head = serve_decode_line(&s, "COVERAGE [decode]:");
            assert!(!head.contains("COVERAGE MISS"), "{head}");
            reset();
        });
    }

    /// s151 (B12): **子は親に 1 回だけ**。s148 は `sl_lin_other` を 1 call につき
    /// 3 か所 (入口の地 / ピン留め / 重み登録) で閉じていたため、線形 call 数の
    /// 3 倍の号数が写っていた (B11 実測: 36,090 call → 108,270)。
    ///
    /// いまは 1 call = `sl_lin` 1 号 + `sl_lin_other` 1 号で、**親 = 内訳の和**。
    /// 時刻値は環境依存なので、内訳 = 1 ns / 全体 = 5 ns を与えて
    /// `1 + 1 + 1 + 1 + 1 = 5` が厳密に成り立つことを固定する
    /// (残余 = 全体 − 子 が 0 = 未帰属なし)。
    #[test]
    fn serve_linear_children_sum_to_parent_exactly_once() {
        with_prof_counters(|| {
            reset();
            let d = Phase::Decode.idx();
            set_phase(Phase::Decode);
            // 線形呼び出し 1 回。子の 5 段は **0 ns** で積み、全体 (実際の経過) を
            // `lin_call_end` が積む — 経過は数 ns しかないので、未帰属は 0 になる。
            let call_t = serve_fixture_linear_begin();
            let t = |ns: u64| std::time::Instant::now() - std::time::Duration::from_nanos(ns);
            // 入口の地 (other)
            super::lin_other_close(Some(t(0)));
            // タイルの 5 段 (out_alloc / other / h2d / launch / d2h)。
            // s153 (B14): 段は `span_end` + `lin_stage_taken_ns` の対で積む
            // (cuda.rs の実サイトと同じ) — 残余 = 全体 − 段の和。
            for op in [SL_LIN_OUTALLOC, SL_LIN_H2D, SL_LIN_LAUNCH, SL_LIN_D2H] {
                let ns = super::lin_stage_taken_ns(Some(t(0)));
                span_end_ns(ns, op);
            }
            // ピン留め確保 / 重み登録 (残余の中身・表 C へは積まない)
            super::lin_other_close(Some(t(0)));
            // 重み登録 (other) — 2 回目の other だが号数は増えない
            super::lin_other_close(Some(t(0)));
            super::lin_call_end(call_t);
            // 親 (sl_lin) = 1 号、子 (other) = 1 号 (3 倍にならない)
            assert_eq!(serve_row_ticks(d, SL_LIN), 1, "sl_lin の号数 ≠ call 数");
            assert_eq!(
                serve_row_ticks(d, SL_LIN_OTHER),
                1,
                "sl_lin_other が call ごとに 1 号になっていない (3 倍の再発)"
            );
            assert_eq!(lin_other_samples(d), 1);
            for (op, name) in [
                (SL_LIN_OUTALLOC, "sl_lin_outalloc"),
                (SL_LIN_H2D, "sl_lin_h2d"),
                (SL_LIN_LAUNCH, "sl_lin_launch"),
                (SL_LIN_D2H, "sl_lin_d2h"),
            ] {
                assert_eq!(serve_row_ticks(d, op), 1, "{name} の号数 ≠ call 数");
            }
            // 親 = 子の和 + 残余。時刻は環境依存なので等式は範囲で固定する:
            // (1) 子 (5 段) は親より大きくなれない (二重計上なら 3 倍になる)
            // (2) 残余 (親 − 子) は 2 割以下 = 未帰属の箱が空のまま働いている
            let parent = serve_row_ns(d, SL_LIN);
            let children: u64 = SL_GROUPS[1].1.iter().map(|&op| serve_row_ns(d, op)).sum();
            assert!(parent > 0, "sl_lin の全体が 0 ns");
            assert!(children <= parent, "子の和 {children} ns が親 {parent} ns を超える");
            // 残余 = 親 − 子 (空でない箱)。テスト側の staging (Instant 読み・
            // RefCell 借用) も経過に入るため比率は緩く見る —
            // 「子が親の 3 倍」(s148 の 108,270) はここで必ず落ちる。
            let gap = step_gap(SL_LIN);
            assert_eq!(gap, parent - children);
            assert!(
                children * 2 >= parent,
                "子の和が親の半分未満 — 段の閉じ漏れ: 親 {parent} / 子 {children}"
            );
            // 3 call に増やしても号数は 3 倍のまま (各 call 1 号)
            for _ in 0..2 {
                serve_fixture_linear_call();
            }
            assert_eq!(serve_row_ticks(d, SL_LIN), 3);
            assert_eq!(
                serve_row_ticks(d, SL_LIN_OTHER),
                3,
                "sl_lin_other が 9 になった (3 倍の再発)"
            );
            assert_eq!(lin_other_samples(d), 3);
            reset();
        });
    }

    /// s153 (Task B14): **`sl_lin_other` は残余 = 全体 − 測った段**。
    /// 合成呼び出しを厳密に 720 ns にし、段を 100 + 50 + 40 + 10 = 200 ns 測る。
    /// 残余は 520 ns、号数は 1 号、親 = 段の和 + 残余。
    ///
    /// 呼び出しの全体 720 ns は `lin_call_begin` が返す tick を「いま − 720 ns」に
    /// 差し替えて与える (経過は実時間ではなく厳密な値になる)。段の ns も
    /// `lin_stage_add` で同じ値を入れる — `span_end` は実時間で積むので、
    /// この検査では「会計の算術」だけを見る (cuda.rs の段サイトは
    /// `lin_stage_taken_ns` で同じ開始 tick を両方へ渡している)。
    #[test]
    fn serve_linear_other_is_residual_parent_minus_stages() {
        with_prof_counters(|| {
            reset();
            let d = Phase::Decode.idx();
            set_phase(Phase::Decode);
            // **全体 = 720 ns ちょうど**: `Instant::now()` は「いま」なので、
            // 実経過は 720 ns を必ず超える (段の仕込みだけで 1 us 前後)。
            // `Instant - Duration` はプロセス起動直後だと 0 に張り付くため、
            // 「過去の tick」を作って経過を縮める手は使えない。ここでは
            // LinAcc を**直接**仕込み、`lin_call_end` を呼ばずに残余の式
            // (`ns.saturating_sub(taken_ns)`) と同じ算術を出口の 1 か所で回す。
            super::LIN_STATE.with(|c| {
                let mut a = c.borrow_mut();
                a.open = true;
                a.other_ns = 0;
                a.other_n = 0;
                a.taken_ns = 0;
                a.taken_n = 0;
            });
            // 4 段を測る (100 + 50 + 40 + 10 = 200 ns)
            for ns in [100u64, 50, 40, 10] {
                super::lin_stage_add(ns);
            }
            let (taken_ns, taken_n) = super::lin_taken();
            assert_eq!(taken_ns, 200, "測った段の ns が 200 でない");
            assert_eq!(taken_n, 4, "測った段の本数が 4 でない");
            super::lin_call_end_at(720);
            assert_eq!(serve_row_ns(d, SL_LIN), 720, "sl_lin の全体が 720 ns でない");
            assert_eq!(
                serve_row_ns(d, SL_LIN_OTHER),
                520,
                "sl_lin_other が 520 ns (720 - 200) でない"
            );
            assert_eq!(serve_row_ticks(d, SL_LIN_OTHER), 1, "残余の号数が 1 でない");
            assert_eq!(lin_other_samples(d), 1, "残余の呼び出し数が 1 でない");
            assert_eq!(serve_row_ticks(d, SL_LIN), 1, "sl_lin の号数が 1 でない");
            // 親 = 段の和 + 残余 (号数は揃う)
            let parent = serve_row_ns(d, SL_LIN);
            let other = serve_row_ns(d, SL_LIN_OTHER);
            assert_eq!(taken_ns + other, parent, "親 ≠ 段の和 + 残余");
            assert_eq!(
                serve_row_ticks(d, SL_LIN_OTHER),
                serve_row_ticks(d, SL_LIN),
                "残余の号数 ≠ 親の号数"
            );
            reset();
        });
    }

    /// s153 (Task B14): **段を 1 つも測らない呼び出しは今日までどおり全体が残余**。
    /// (CPU フォールバック / 計測が入らない呼び出し。号数は 1 のまま。)
    #[test]
    fn serve_linear_other_is_whole_call_when_no_stage_measured() {
        with_prof_counters(|| {
            reset();
            let d = Phase::Decode.idx();
            set_phase(Phase::Decode);
            super::LIN_STATE.with(|c| {
                let mut a = c.borrow_mut();
                a.open = true;
                a.other_ns = 0;
                a.other_n = 0;
                a.taken_ns = 0;
                a.taken_n = 0;
            });
            let (taken_ns, taken_n) = super::lin_taken();
            assert_eq!((taken_ns, taken_n), (0, 0), "段を測っていない会計が汚れている");
            // 段を 1 本も測らない呼び出し (全体 = 残余)
            super::lin_call_end_at(640);
            assert_eq!(serve_row_ns(d, SL_LIN), 640, "sl_lin の全体が 640 ns でない");
            assert_eq!(
                serve_row_ns(d, SL_LIN_OTHER),
                640,
                "段を測らない呼び出しの残余が全体でない"
            );
            assert_eq!(serve_row_ticks(d, SL_LIN_OTHER), 1, "残余の号数が 1 でない");
            assert_eq!(lin_other_samples(d), 1);
            reset();
        });
    }

    /// s151 (B12): **COVERAGE は OK**。fake serve ループ (prefill 1 + decode 4) を
    /// 回すと、両フェーズのヘッダが [ok]、status は MISS を出さず、
    /// `requests` と `wall` が正になり、phase 別の壁が両フェーズで正になる。
    #[test]
    fn serve_coverage_is_ok_on_fake_serve_loop() {
        with_prof_counters(|| {
            reset();
            let mark = request_begin();
            serve_fake_loop_prefill_then_decode(4);
            // 線形の期待 = 実 q4 ディスパッチ数 (cuda 無しでも 0 でないよう直接数える)
            // s155 (B16): prefill 1 本 + per-op の decode は 1 step に投影 2 本 (steps × 2)。
            let p = Phase::Prefill.idx();
            let d = Phase::Decode.idx();
            for i in 0..(1 + 4 * 2) {
                set_phase(if i == 0 { Phase::Prefill } else { Phase::Decode });
                count_lin_dispatch();
            }
            request_end(mark);
            let (n, wall) = request_wall();
            assert_eq!(n, 1, "requests が 1 でない");
            assert!(wall > 0, "リクエストの壁時計が 0");
            let (pn, pw) = request_wall_for(p);
            let (dn, dw) = request_wall_for(d);
            assert_eq!((pn, dn), (1, 1), "prefill / decode の壁が別々に数えられていない");
            assert_eq!(pw + dw, wall, "phase 別の壁の和 ≠ 全体の壁");
            let s = summary();
            for phase_name in ["prefill", "decode"] {
                let head = serve_decode_line(&s, &format!("COVERAGE [{phase_name}]: steps"));
                assert!(!head.contains("COVERAGE MISS"), "{head}");
                assert!(head.contains("requests 1 wall"), "壁時計がヘッダに出ない: {head}");
            }
            let st = s
                .lines()
                .rev()
                .find(|l| l.trim_start().starts_with("COVERAGE status:"))
                .unwrap_or_else(|| panic!("COVERAGE status 行が無い\n{s}"));
            let rel = s
                .lines()
                .find(|l| l.contains("layer residual"))
                .unwrap_or("NO LAYER RESIDUAL LINE");
            assert!(
                st.contains("all buckets have samples"),
                "COVERAGE status が MISS のまま: {st}\nREL {rel}\n{s}"
            );
            reset();
        });
    }

    /// s151 (B12): 計測 OFF は表 C とリクエスト壁時計の**どれにも触らない**。
    /// `request_begin` が `None` を返し (クロックを読まない)、fake ループを
    /// 回しても全行 0 のまま = 既定経路に何も残さない。
    #[test]
    fn serve_request_wall_is_noop_when_profiling_off() {
        with_prof_counters_off(|| {
            assert!(request_begin().is_none(), "計測 OFF なのに request_begin が票を返した");
            assert!(lin_call_begin().is_none(), "計測 OFF なのに lin_call_begin が時刻を読んだ");
            request_end(None);
            lin_call_end(None);
            lin_other_push(1_000);
            lin_stage_taken();
            lin_stage_taken_ns(None);
            lin_stage_add(1_000);
            serve_fake_loop_prefill_then_decode(2);
            for p in [Phase::Prefill.idx(), Phase::Decode.idx()] {
                for row in 0..SPLIT_SERVE_N {
                    let op = SPLIT_SERVE_FIRST + row;
                    assert_eq!(serve_row_ns(p, op), 0, "計測 OFF で行 {row} に ns が入った");
                    assert_eq!(serve_row_calls(p, op), 0, "計測 OFF で行 {row} に calls が入った");
                    assert_eq!(serve_row_ticks(p, op), 0, "計測 OFF で行 {row} に号数が入った");
                }
                assert_eq!(lin_other_samples(p), 0);
                let (pn, pwall) = request_wall_for(p);
                assert_eq!((pn, pwall), (0, 0), "計測 OFF でフェーズ別の壁が動いた");
            }
            assert_eq!(request_wall(), (0, 0));
        });
    }

    // -----------------------------------------------------------------------
    // s156 (Task B19): サンプリング (`DENNO_PROFILE_SAMPLE`) と gap_fwd の内訳
    // -----------------------------------------------------------------------

    /// 境界: 未設定 / 空 / 空白 / "0" / 数値でない → 既定 1 (従来挙動)。
    /// 「測らない」は `DENNO_PROFILE` の役目なので、ここへ無言で落とさない。
    #[test]
    fn parse_sample_every_falls_back_to_one_on_garbage() {
        for raw in [
            None,
            Some(""),
            Some("   "),
            Some("0"),
            Some("00"),
            Some("abc"),
            Some("-3"),
            Some("1.5"),
            Some("16 steps"),
        ] {
            assert_eq!(
                parse_sample_every(raw),
                SAMPLE_EVERY_DEFAULT,
                "raw={raw:?} は既定 1 に落ちる"
            );
        }
        assert_eq!(SAMPLE_EVERY_DEFAULT, 1);
    }

    /// 境界: "1" / "16" はそのまま、上限超えは丸める、u64 桁あふれは既定に落ちる。
    #[test]
    fn parse_sample_every_accepts_positive_integers_and_saturates() {
        assert_eq!(parse_sample_every(Some("1")), 1);
        assert_eq!(parse_sample_every(Some("16")), 16);
        assert_eq!(parse_sample_every(Some(" 16 ")), 16);
        assert_eq!(parse_sample_every(Some("1000000")), SAMPLE_EVERY_MAX);
        assert_eq!(parse_sample_every(Some("999999999")), SAMPLE_EVERY_MAX);
        // u64 を超える値は parse が失敗する → 既定 (飽和させない: 桁あふれは入力ミス)
        assert_eq!(
            parse_sample_every(Some("18446744073709551616")),
            SAMPLE_EVERY_DEFAULT
        );
        assert!(SAMPLE_EVERY_MAX > 0);
    }

    /// 周期性: N step に 1 回だけ真。N=1 は全 step、N=16 は 0/16/32、上限は 0 だけ。
    #[test]
    fn sample_hit_is_periodic_and_one_hits_every_step() {
        let _guard = lock_counters();
        let hits = |n: u64| -> Vec<u64> {
            with_sample_every(n, || (0..40u64).filter(|&i| sample_hit(i)).collect())
        };
        assert_eq!(hits(1), (0..40u64).collect::<Vec<_>>());
        assert_eq!(hits(16), vec![0, 16, 32]);
        assert_eq!(hits(3), vec![0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]);
        assert_eq!(hits(SAMPLE_EVERY_MAX), vec![0]);
    }

    /// 間引いた step では **ns が 1 ns も増えず号数だけ**が増える
    /// (`SPLIT_NS` が間引きで膨らまないことの直接の検査)。
    #[test]
    fn skipped_steps_accumulate_ticks_but_no_ns() {
        with_prof_counters(|| {
            set_phase(Phase::Decode);
            with_sample_every(4, || {
                // `skipped_spans` はプロセス全体で 1 本のカウンタ (`NS` と同じ) なので、
                // このテストが積んだ分だけを差で見る。
                let skip0 = skipped_spans();
                for i in 0..8u64 {
                    let st = step_begin();
                    assert_eq!(step_index(), Some(i), "step 番号が 0 起点で進んでいない");
                    // 区間は sampled な step だけ開く (間引いた step は時刻を読まない)。
                    let t = span_begin();
                    assert_eq!(t.is_some(), sample_hit(i), "step {i} の span_begin が食い違う");
                    span_end(t, SL_SO_GAP_FWD);
                    // 直接 push しても、間引いた step の ns は捨てられる (号数は残る)。
                    push(SL_GAP_GDN, 2_000_000, 1);
                    step_end(st);
                    step_end_tail();
                }
                assert_eq!(step_index(), None, "step を閉じた後に番号が残っている");
                let p = Phase::Decode.idx();
                // 8 step のうち測ったのは 0 と 4 の 2 本。
                assert_eq!(serve_samples(p), 2);
                // ns は「測った 2 step × 2 ms」だけ — 8 step 分 (16 ms) は入らない。
                assert_eq!(serve_row_ns(p, SL_GAP_GDN), 2 * 2_000_000);
                // 号数は 8 step 全部 (間引いても「何 step 見たか」は落とさない)。
                assert_eq!(serve_row_ticks(p, SL_GAP_GDN), 8);
                // 区間を開けずに閉じた回数 = 間引いた 6 step。
                assert_eq!(skipped_spans() - skip0, 6);
                // 残余は張り付いていない (子が親を超えていない)。
                assert_eq!(overflow_drops(), 0);
            });
        });
    }

    /// 52 ms/step の gap と同じ**比率**の合成イベント。値は比率だけを写した合成値で、
    /// 実測値ではない (`sl_gap_rope` は attn の**内数**なので gap に足さない)。
    /// 戻り値は (gap_fwd の合成値, 内訳)。
    fn gap_fixture_events() -> (u64, GapFwdParts) {
        const LAYERS: u64 = 8;
        let mut gap = 0u64;
        let mut ev = |op: usize, ns: u64| {
            push(op, ns, 1);
            gap += ns;
        };
        ev(SL_GAP_EMBED, 40_000);
        for _ in 0..LAYERS {
            ev(SL_GAP_RMSNORM, 60_000);
            ev(SL_GAP_GDN, 1_800_000);
            ev(SL_GAP_RMSNORM, 60_000);
            ev(SL_GAP_ATTN, 2_400_000); // rope を含む総計
            push(SL_GAP_ROPE, 250_000, 1); // attn の内数 (gap には足さない)
            ev(SL_GAP_FFN, 3_000_000);
        }
        // 名付けた 7 段に入らない「層間の地」(残差加算など) — 残余として出る。
        ev(SL_LAY_RESIDUAL, 300_000);
        push(SL_SO_GAP_FWD, gap, 1);
        (gap, gap_fwd_parts(Phase::Decode.idx()))
    }

    /// CPU フィクスチャ: 合成イベント → 内訳表。内訳の和 + 残余 = gap_fwd、残余は
    /// 10% 未満。**サンプリングの有無で行名が変わらない**ことも同じテストで固定する。
    #[test]
    fn gap_fwd_buckets_sum_to_the_gap_and_names_are_sampling_independent() {
        let run = |n: u64| -> (u64, GapFwdParts, u64, [u64; 7]) {
            with_prof_counters(|| {
                set_phase(Phase::Decode);
                with_sample_every(n, || {
                    let mut last = (0u64, GapFwdParts::default());
                    // 2 周期ぶん回す (N=16 なら 32 step 中 2 step だけ測る)。
                    for _ in 0..(2 * n.max(1)) {
                        let st = step_begin();
                        last = gap_fixture_events();
                        step_end(st);
                        step_end_tail();
                    }
                    let p = Phase::Decode.idx();
                    (
                        // 2 周期ぶん回すので、測った step は必ず 2 本 = 累積は 2 倍。
                        2 * last.0,
                        last.1,
                        gap_fwd_residual_ns(p),
                        gap_fwd_part_ticks(p),
                    )
                })
            })
        };
        let (gap_all, parts_all, resid_all, ticks_all) = run(1);
        let (gap_s16, parts_s16, resid_s16, ticks_s16) = run(16);
        // 和 + 残余 = gap_fwd (どちらのサンプリングでも同じ恒等式)。
        assert_eq!(parts_all.sum() + resid_all, gap_all, "N=1 で内訳が gap_fwd に一致しない");
        assert_eq!(parts_s16.sum() + resid_s16, gap_s16, "N=16 で内訳が gap_fwd に一致しない");
        // 残余は 10% 未満 (= 名付けが足りている)。合成値なので比で見る。
        assert!(
            resid_all * 10 < gap_all,
            "残余が大きすぎる: resid={resid_all} gap={gap_all}"
        );
        assert!(
            resid_s16 * 10 < gap_s16,
            "残余が大きすぎる (N=16): resid={resid_s16} gap={gap_s16}"
        );
        // サンプリングしても **同じ 7 行が同じ順で出る** (名前は周期に依存しない)。
        assert_eq!(GapFwdParts::NAMES, GAP_ROWS.map(|r| r.0));
        // 号数: 実測した 6 行は >0、`sl_gap_quant` だけは構造的に 0
        // (q4 は活性を量子化しない — 行は出るが号数は付かない)。
        for (i, &t) in ticks_all.iter().enumerate() {
            if i == 5 {
                assert_eq!(t, 0, "sl_gap_quant は q4 経路で 0 のはず");
            } else {
                assert!(t > 0, "行 {} の号数が 0", GapFwdParts::NAMES[i]);
            }
        }
        assert_eq!(ticks_all, ticks_s16.map(|t| t / 16), "号数が step 数に比例していない");
        // ns は**測った step だけ**を積むので、周期を変えても同じ値になる
        // (N=1 で 2 step 測った分と、N=16 で 2 step 測った分が一致する)。
        assert_eq!(parts_all, parts_s16, "サンプリングで測った値が変わった");
        // 内訳の形: rope は attn の内数として独立行に出て、attn からは引かれている。
        // (測った step は 2 本なので、合成値は 1 step 分の 2 倍。)
        const K: u64 = 2;
        assert_eq!(parts_all.rope, K * 8 * 250_000);
        assert_eq!(parts_all.attn, K * 8 * (2_400_000 - 250_000));
        assert_eq!(parts_all.rmsnorm, K * 8 * 120_000);
        assert_eq!(parts_all.gdn, K * 8 * 1_800_000);
        assert_eq!(parts_all.ffn, K * 8 * 3_000_000);
        assert_eq!(parts_all.embed, K * 40_000);
        // q4 経路は活性を量子化しない = 構造的に 0 (穴ではなく経路の性質)。
        assert_eq!(parts_all.quant, 0);
        // gdn + attn + ffn が gap の主成分 (52 ms の形)。
        assert!(parts_all.gdn + parts_all.attn + parts_all.ffn > gap_all / 2);
    }

    /// 表示: 内訳 7 行・残余行・「= gap_fwd」の合計行・サンプリング行が実際に出る
    /// (GPU 不要・整形の退行を拾う)。
    #[test]
    fn summary_prints_gap_fwd_breakdown_and_sampling_line() {
        with_prof_counters(|| {
            set_phase(Phase::Decode);
            // 表 C の節は `sl_step` / `sl_lin` の号数が 0 だと出ないので、1 step 回す。
            let st = step_begin();
            let _ = gap_fixture_events();
            step_end(st);
            step_end_tail();
            let s = summary();
            for name in GapFwdParts::NAMES {
                assert!(s.contains(name), "summary に {name} が無い");
            }
            assert!(s.contains("sl_gap_resid"), "残余行が出ていない");
            assert!(s.contains("= gap_fwd"), "合計行が出ていない");
            assert!(
                s.contains("SPLIT SERVE sampling: N="),
                "サンプリング行が出ていない"
            );
        });
    }

    /// 表 C の末尾 7 行が `sl_gap_*` と 1 対 1 で並んでいる (id と名前の取り違え防止)。
    #[test]
    fn gap_fwd_row_ids_and_names_line_up() {
        let ids = [
            SL_GAP_EMBED,
            SL_GAP_RMSNORM,
            SL_GAP_GDN,
            SL_GAP_ATTN,
            SL_GAP_ROPE,
            SL_GAP_QUANT,
            SL_GAP_FFN,
        ];
        assert_eq!(SPLIT_SERVE_N, 45);
        for (i, &op) in ids.iter().enumerate() {
            assert_eq!(GAP_ROWS[i].0, GapFwdParts::NAMES[i], "行 {i} の名前が食い違う");
            assert_eq!(GAP_ROWS[i].1, op, "行 {i} の id が食い違う");
            assert_eq!(
                split_row_name(SPLIT_SERVE_TABLE, op - SPLIT_SERVE_FIRST),
                Some(GapFwdParts::NAMES[i])
            );
            // すべて表 C の行であること (push が黙って捨てない)。
            assert!(split_slot(op).is_some(), "行 {i} が表 C の外にある");
        }
        // gap_fwd の内訳は step の兄弟ではない (B16 の恒等式を動かさない)。
        for &op in ids.iter() {
            assert!(!SL_GROUPS[0].1.contains(&op), "sl_gap_* を step の兄弟に入れない");
        }
    }
}
