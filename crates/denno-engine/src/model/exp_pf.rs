//! task 148: prefill の選択 expert を GPU で回すための入口 (GPU 非依存の部分)。
//!
//! # なぜ要るか
//!
//! prefill の選択 expert は丸ごと CPU で、cold prefill (3,826 tok) の 53% を占める
//! (lab t147a の台帳 `[q4ledger-pf]`: `experts` 47.4 s / `wall` 89.3 s。launch 0 /
//! sync 0 / H2D 0 = GPU を 1 度も通っていない)。
//!
//! prefill は 512 トークンのチャンクで回るので、1 層 1 チャンクで見ると
//! 「256 expert × 平均 20 行」の GEMM になる。decode の m=1 カーネル
//! (`kernels_exp_iq.cu`) を **行方向へ広げた** カーネルなら、重みの読みを
//! 1 回に畳んだまま 20 行ぶんの積和を回せる。
//!
//! # カーネルの契約 (`kernels_exp_iq_pf.cu` と共有)
//!
//! - 1 warp = 2 出力行 (PF_OUTS)。32 レーンが 8 要素ずつ (IQ4_NL / Q2_0 は
//!   1 レーン = 1 ブロック) 持ち、**行方向は 16 行ずつ** (PF_MR) まとめて
//!   レジスタに積む。重みは 1 回デコードして 16 行に配る。
//! - block = 4 warp = 8 出力行、grid = (ceil(n_out / 8), segment 数, 1)。
//! - 1 行ぶんの累積順は decode の m=1 カーネルと同じ (b 昇順 → j 昇順 →
//!   `__shfl_down_sync` の木)。**同じ行を m=1 と m 行で回すとビット一致する**
//!   (FMA の縮約は nvcc の既定でどちらも同じ)。
//! - segment 表 (1 segment = 4 語):
//!
//!   seg[s*4 + 0] = m        (この expert の行数)
//!   seg[s*4 + 1] = row_off  (`rows` の先頭添字。出力も同じ添字 × n_out)
//!   seg[s*4 + 2] = 0        (予約)
//!   seg[s*4 + 3] = 0        (予約)
//!
//! - `rows[i]` は行 i の**トークン番号**。入力は `xbase + rows[i] * n_in`。
//!   down 行列は入力が `act` (行が詰まっている) なので `rows[i] = i` を渡す。
//!
//! # 実測 (lab t151c、SHA 9bb62f7d6、swift15-abl IQ3_S、14T)
//!
//! 条件は `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=20000
//! DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。off が PF=0 の基準。
//!
//! | 項目 | off | on (既定) |
//! | --- | --- | --- |
//! | c2000 cold prefill | 89.6 s | 63.6 s |
//! | c2000 turn2 prefill | 0.95 s | 0.69 s |
//! | c8000 cold prefill | 329.3 s | 218.3 s |
//! | c8000 turn2 prefill | 1.03 s | 0.75 s |
//! | decode 28 / 3,954 / 14,139 tok | 56 / 51 / 52 | 56 / 52 / 52 |
//! | md5 (a, b, c2000_cold, c2000_turn2) | 一致 | **一致** |
//!
//! c8000 の md5 だけ違う (丸めで greedy の 1 トークンが入れ替わる)。text は
//! ベースラインと同じ日本語の要約。
//!
//! **上げ直しを既定で切った理由**: `DENNO_EXPERT_IQ_PF_UPLOAD_MB=512` にすると
//! 14k の decode が 52 → 59 ms/tok (+13.5%、lab t151c の `onup`)。prefill が
//! 空きに積んだ expert が decode の作業集合を奪う。prefill は VRAM を**読むだけ**
//! にして、decode の在庫を変えないのを既定にする。
//!
//! # ゲート
//!
//! `DENNO_EXPERT_IQ_GPU_PF` (既定 on) かつ `DENNO_EXPERT_IQ_GPU=1` のときだけ。
//! 片方でも欠けたら [`pf_enabled`] が false を返し、呼び出し側は従来の CPU 経路
//! (`moe_experts_flat`) をそのまま回す (fail closed)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::kquant::GgmlType;

/// `DENNO_EXPERT_IQ_GPU_PF=1`: prefill の選択 expert を GPU (m 行) で回す。
///
/// **`DENNO_EXPERT_IQ_GPU=1` が前提** (VRAM 常駐の LRU を使うため)。
pub const EXPERT_IQ_PF_ENV: &str = "DENNO_EXPERT_IQ_GPU_PF";

/// `DENNO_EXPERT_IQ_PF_MIN_HITS`: 1 層を GPU で回すのに要るヒット expert 数 (既定 4)。
///
/// 1 層 1 同期は prefill の実測で 30 us 前後 (t147a の `sync_floor`)。ヒットが
/// 数本しか無い層は CPU のままの方が安い。
pub const EXPERT_IQ_PF_MIN_HITS_ENV: &str = "DENNO_EXPERT_IQ_PF_MIN_HITS";

/// `DENNO_EXPERT_IQ_PF_UPLOAD_MB`: 1 チャンクで VRAM へ上げ直す上限 (MiB、**既定 0**)。
///
/// **LRU に空きがある間だけ**使う。埋まっているときの上げ直しは「1 本上げるために
/// 1 本追い出す」= 次のチャンクで同じだけミスが増えるだけで、H2D の時間が純損に
/// なる (ミス 1 本の CPU 計算 0.5 ms に対し H2D 0.27 ms)。
///
/// task 151: **既定 0 にした**。空きに積んだ expert は decode が使うはずだった
/// スロットを奪い、次の decode のヒット率が下がる (lab t151b: 14k の decode が
/// +17.3%)。prefill は VRAM を **読むだけ** (peek) にして、decode の作業集合を
/// 変えないのを既定にする。
pub const EXPERT_IQ_PF_UPLOAD_MB_ENV: &str = "DENNO_EXPERT_IQ_PF_UPLOAD_MB";

/// `DENNO_EXPERT_IQ_PF_UPLOAD_ROWS`: 上げ直すミスの最小行数 (既定 8)。
///
/// 19 トークンの prefill では 1 expert あたり平均 0.74 行なので、この床で
/// **短い prefill では 1 本も上げない** (0.97 s の next turn に 512 MB を
/// 載せるような真似をしない)。
pub const EXPERT_IQ_PF_UPLOAD_ROWS_ENV: &str = "DENNO_EXPERT_IQ_PF_UPLOAD_ROWS";

/// `DENNO_PF_STREAM=1`: prefill のミス expert を GPU のステージング領域へ
/// 上げて m 行カーネルで回す (task 162、task 156 の移植)。
///
/// decode の VRAM 常駐 ([`crate::model::expert_vram`]) は **読むだけ** (peek)
/// で、1 本も入れ替えない。ステージングは decode の在庫とは別の領域から取る。
pub const STREAM_ENV: &str = "DENNO_PF_STREAM";

/// `DENNO_PF_STREAM_MB`: ステージング領域の合計 (MiB、既定 512)。
///
/// 2 枚 (ダブルバッファ) に等分する。空き VRAM から取り、確保できなければ
/// 小さいサイズへ落とし、それでも駄目ならこのリクエストではストリーミング
/// しない (fail closed)。
pub const STREAM_MB_ENV: &str = "DENNO_PF_STREAM_MB";

/// `DENNO_PF_STREAM_MIN_TOK`: ストリーミングを始める最小トークン数 (既定 512)。
///
/// 19 トークンの next turn prefill と decode を変えないための床。
pub const STREAM_MIN_TOK_ENV: &str = "DENNO_PF_STREAM_MIN_TOK";

/// `DENNO_PF_STREAM_GROUP`: 1 回に上げる expert 数 (既定 32)。
///
/// 上げ (H2D) と m 行カーネルの計算を重ねる単位。小さいほど重なりは細かく
/// なるが、発行とイベントの固定費が増える。
pub const STREAM_GROUP_ENV: &str = "DENNO_PF_STREAM_GROUP";

/// `DENNO_PF_STREAM_PIN`: 上げの経路 (既定 1)。
///
/// - `0` = pageable のまま同期コピー (task 156 と同じ。新リンク単体の効果を見る)
/// - `1` = pinned のバウンスバッファ経由 + 別ストリームの非同期コピー (既定)
///
/// どちらもデバイスへ渡るバイト列は同じ (数値には触らない)。
pub const STREAM_PIN_ENV: &str = "DENNO_PF_STREAM_PIN";

/// `DENNO_PF_STREAM_UT`: 上げを専用スレッドへ出すか (task 165)。
///
/// - `1` = ホストの写しと H2D の積み込みを上げ専用スレッドが行い、計算スレッドは
///   グループ g のカーネルを積みながら g+1 の上げを進める
/// - `0` = task 162 と同じ (写しと積み込みを計算スレッドが直列に行う。**既定**)
///
/// lab t165a の実測 (4 条件): ut=1 は 14k cold 102.10 s、ut=0 は 99.97 s。
/// 写しと DMA が同じメモリ帯域を食うのでスレッドを分けても速くならず、DMA を
/// 1 本しか積めないため `enq` が前の DMA の完了を待つ (enq_s 0.04 → 9.53)。
/// 数値には触らない (デバイスへ渡るバイト列は同じ)。
pub const STREAM_UT_ENV: &str = "DENNO_PF_STREAM_UT";

/// `DENNO_PF_STREAM_WC`: pinned のバウンスを write-combined にするか (task 165)。
///
/// バウンスは CPU が書いて DMA が読むだけなので、write-combined にすると
/// 書き込みの read-for-ownership が消え、DMA もキャッシュ coherence を
/// 取らないぶん速くなる。既定 **on** (lab t165b/t165c で採用)。
/// 確保に失敗したら通常の pinned へ落ちる (数値は同じ)。
pub const STREAM_WC_ENV: &str = "DENNO_PF_STREAM_WC";

/// ストリーミング時の層ループのチャンク既定 (task 162 の lab t162c の最良)。
///
/// `DENNO_PF_STREAM=1` のとき `DENNO_PREFILL_BATCH_CHUNK` 未設定ならこれを使う。
pub const DEFAULT_PREFILL_CHUNK_STREAM: usize = 1024;

/// `DENNO_PF_CHUNK`: prefill の層ループのチャンク (トークン数) を上書きする (task 197)。
///
/// **既定 = [`DEFAULT_PF_CHUNK_STREAM`] (1536、ストリーミング経路のとき)**。
/// `0` / `off` / 空 / 解析不能で旧経路 (`DENNO_PREFILL_BATCH_CHUNK` と既存の既定) に戻る。
///
/// なぜ効くか: ストリーミング経路は「1 層 1 チャンク」でミス expert を VRAM へ
/// 上げる。1 チャンク 512 トークンでも全 expert が選ばれるので、**チャンクを
/// 2 倍にすると同じ重みを上げる回数が半分になる** (アップロード量がほぼ半減)。
/// 台帳 (lab t199a の base) では 3,826 トークンで 105.5 GB / copy 7.11 s、
/// 14,018 トークンで 357.8 GB / copy 17.04 s を上げており、これが cold prefill の
/// 最大項目。`DEFAULT_PREFILL_CHUNK_STREAM` が 1024 なのに本番 env が 512 を
/// 明示しているため、1024 の利得が今は出ていない。
///
/// lab t197a (5 COND、depth8k、比較は base と base2 の両方):
///
/// | COND | 3,826 cold | 14,018 cold | 上げた量 (3.8k / 14k) | copy_s | md5 |
/// | --- | --- | --- | --- | --- | --- |
/// | base (chunk 512) | 13.21 s | 50.01 s | 105.5 / 357.8 GB | 4.13 / 14.78 | — |
/// | chunk 1024 | 10.36 s (-21.6%) | 36.87 s (-26.3%) | 71.0 / 223.6 GB | 2.78 / 8.70 | DIFF |
/// | chunk 1536 | 9.89 s (-25.1%) | 35.44 s (-29.1%) | 56.4 / 158.0 GB | 2.21 / 6.17 | DIFF |
/// | chunk 2048 | 13.22 s (+0.1%) | 69.81 s (+39.6%) | 41.6 / 92.3 GB | 1.68 / — | DIFF |
///
/// 2048 が落ちるのは容量ではなく `gdn_pf_chain_shape_ok` の
/// `nt >= GDN_CHUNK_MIN_TOKENS (2048)` で ssm のデバイス連鎖が外れるため
/// (ssm が 3.96 → 24.7 s、d2h 39.6 → 124.7 GB)。1536 はこの境界の 3/4 で、
/// 構造的に余裕がある。
///
/// **task 270: この 2048 の壁は `DENNO_PF_CHAIN_BIG=1` で外せる** (既定 off)。
/// 連鎖の判定は「単発 sigmoid スキャン (`dno_gdn_scan_sig`) が nt < 2048」で
/// 切られていたが、task 186/225 で v2/v3 (どちらもトークン長の上限を持たない)
/// が入り、既定では v3 が選ばれる。よって形 (dk = dv = 128 / d_conv = 4) が
/// 揃えば nt >= 2048 でも連鎖をそのまま使える。`gdn_pf_chain` 側も、投影を
/// 撃つ前にスキャンカーネルの有無を確かめてから進む (無ければ状態を触らずに
/// 従来経路へ)。`DENNO_PF_CHUNK` を 3072 以上にすると、14k のチャンク枚数が
/// 10 → 5 (3072) / 4 (4096) に減り、同じ expert 重みを上げる回数がその分減る。
///
/// lab t207a (merge 済み int-auto = br198 + br200 の上で測り直した 5 COND、
/// base は自分の run の `DENNO_PF_CHUNK=0`):
///
/// | COND | 3,826 cold | 14,018 cold | 3,826 turn2 | 14,018 turn2 |
/// | --- | --- | --- | --- | --- |
/// | old2 | 11.69 / 11.71 s | 40.41 s | 0.40 s | 0.47 s |
/// | old3 | 11.75 / 11.76 s | 39.67 s | 0.40 s | 0.47 s |
/// | def (既定) | 9.18 / 8.48 s | 31.48 s | 0.40 s | 0.47 s |
/// | def2 (既定) | 8.50 / 8.50 s | 30.47 s | 0.40 s | 0.47 s |
///
/// def/def2 の平均 vs old2/old3 の平均: **3,826 が -26.1%、14,018 が -22.6%**。
/// decode の層時間は 19.20/19.31 vs 19.24/19.34 ms/tok (3,826) で差なし。
/// md5 は 3,826 cold が全 COND 一致、14,018 cold と両 turn2 が DIFF (丸め順のみ、
/// 下の「なぜ md5 が変わるか」)。
///
/// なぜ md5 が変わるか: チャンクはトークンのまとめ方だけで、式も expert の選択も
/// 変えない。変わるのは **チャンク末尾が `DENNO_PF_STREAM_MIN_TOK (512)` を
/// 跨ぐかどうか**である。3,826 を 512 で切ると末尾が 242 トークン (< 512) になり
/// そのチャンクだけ従来経路 (常駐ヒット + ミスは CPU の flat) へ落ち、1536 で切ると
/// 末尾が 754 (>= 512) でストリーミング経路 (全行 GPU の m 行カーネル) になる。
/// GPU の m 行カーネルと CPU の flat は縮約順が違う (task 162 で許容済み) ので
/// ビットが変わる。14,018 は 512 でも 1536 でも末尾が 194 で同じ経路だが、
/// 常駐 (LRU) の履歴が違うため末尾チャンクの「ヒット / ミス」の分かれ方が変わる。
/// どちらも丸め・縮約順の差で、演算も expert 選択も同じ。
pub const PF_CHUNK_ENV: &str = "DENNO_PF_CHUNK";

/// [`PF_CHUNK_ENV`] の既定 (ストリーミング経路のとき)。lab t197a の最良。
pub const DEFAULT_PF_CHUNK_STREAM: usize = 1536;

/// `DENNO_PF_CHUNK_BALANCE`: チャンクをプロンプト長の**均等割り**で選ぶ (task 239)。
///
/// ミス expert は**チャンクごとに上げ直す**ので、転送量はほぼ
/// 「チャンク数 × 1 チャンクあたりの distinct expert 数」に比例する。
/// 既定は固定 1536 なので、3,222 トークンのプロンプトは 3 チャンクになり、
/// 3 回とも同じ expert を上げる (lab t239b の c8869: 3 チャンクで 17,510 本 /
/// 35.8 GB、c2000: 3 チャンクで 23,732 本 / 48.6 GB)。
///
/// on のとき「上限 [`CHUNK_BALANCE_CAP`] の下でチャンク数を最小にする」長さ
/// (= プロンプト長 / チャンク数 の切り上げ) を使う。3,222 なら 1,611 × 2 枚。
/// 1 枚の長さが上限を超えないので、GDN のデバイス連鎖の境界 (2048) は踏まない。
pub const CHUNK_BALANCE_ENV: &str = "DENNO_PF_CHUNK_BALANCE";

/// [`CHUNK_BALANCE_ENV`] の既定 (off = 従来の固定 1536)。
///
/// lab t239d (tools、c8869=3,222 tok / c2000=3,826 tok、base/bal/base2/bal2/bal3):
///
/// | cond | c8869 cold | c2000 cold | c8869 decode |
/// | --- | --- | --- | --- |
/// | base2 (基準) | 6.07 s | 7.13 s | 18.2 ms/tok |
/// | bal/bal2/bal3 | **5.53 s** | 6.91 s | 18.5-18.6 ms/tok |
///
/// prefill は -8.9% (c8869) / -3.1% (c2000) だが、c8869 の decode が +2%、
/// その機序は expert のヒット率 0.929 → 0.912 (最後のチャンクが LRU に残す
/// 集合が変わる)。目標の「decode は 1% 以内」を超えるので**既定 off**。
///
/// task 246 で 2 回測り直した (br239 を int-auto 1d36583f6 に載せ、ゲートは既定 off)。
/// t246a (5 COND: x0/base/bal/reseed/base2) と t246c (3 COND: x0/reseed/base2):
///
/// | cond | c8869 cold prefill | c2000 cold | c8869 cold decode | c8869 turn2 decode |
/// | --- | --- | --- | --- | --- |
/// | base / base2 | 6.35 / 6.01 s | 7.06 / 7.08 s | 18.8 / 18.7 | 19.6 / 19.5 |
/// | bal (balance のみ) | **5.46 s** | 6.87 s | 20.0 | 19.0 |
/// | reseed (balance + DEC_RESEED) | **5.47 / 5.48 s** | 6.84 / 6.82 s | 19.8 / 19.8 | 19.2 / 19.1 |
///
/// prefill は -9.0% (c8869) / -3.3% (c2000) で目標を満たすが、**c8869 cold の直後の
/// decode だけ +5.8%** が 2 run で再現する (turn2 の decode は -3% と逆)。
/// 機序は t239d のヒット率ではない: 同じ run の `[exp-iq]` は base も bal も
/// `lookups=46560 hits=42349 hit_rate=0.910 uploads=876` で**完全に一致**する。
/// 候補は生成列が変わることによる page cache (PLE 表で 1 回 216 ms の跳ねを観測)。
/// 目標 (decode +1% 以内) を満たさないので既定 off のまま。
pub const DEFAULT_CHUNK_BALANCE: bool = false;

/// 均等割りの 1 チャンク上限 (GDN 連鎖の境界 `GDN_CHUNK_MIN_TOKENS` = 2048 の手前)。
pub const CHUNK_BALANCE_CAP: usize = 2047;

/// [`CHUNK_BALANCE_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn chunk_balance_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
        None => DEFAULT_CHUNK_BALANCE,
    }
}

/// 均等割りのチャンクを選ぶ (純関数)。
///
/// - `n <= base` なら 1 チャンク (`n`)。
/// - それ以外は「`cap` で割ったときのチャンク数」を保ったまま均等に割る
///   (端数の小さい最後の 1 枚を作らない)。
pub fn balanced_chunk(n: usize, base: usize, cap: usize) -> usize {
    let base = base.max(1);
    let cap = cap.max(1);
    if n == 0 || n <= base {
        return n.max(1).min(base);
    }
    let nch = n.div_ceil(cap);
    n.div_ceil(nch).clamp(1, cap)
}

/// この prefill のチャンク長 (純関数)。
///
/// 均等割りは「ストリーミング経路」「`DENNO_PF_CHUNK_BALANCE` が on」
/// 「`DENNO_PF_CHUNK` の**明示**指定が無い」の 3 つが揃ったときだけ効く。
/// 長いプロンプト (2 枚を超える) まで広げるのは `long_on`
/// ([`CHUNK_BALANCE_LONG_ENV`]) が on のときだけ (task 246 で追加)。
///
/// **`chunk_override()` で判定してはいけない**: あれは未設定のとき
/// `Some(DEFAULT_PF_CHUNK_STREAM)` を返すので、常に「明示指定あり」に見えて
/// ゲートが一度も発火しない (lab t239c で実際に踏んだ)。明示判定は
/// [`chunk_override_from_value`] に生の env を渡して行う。
pub fn chunk_for_prefill(
    n: usize,
    base: usize,
    cap: usize,
    balance_on: bool,
    long_on: bool,
    stream_on: bool,
    explicit_override: bool,
) -> usize {
    let cap = cap.max(1);
    if !balance_on || !stream_on || explicit_override {
        return base.max(1);
    }
    // 2 枚以内は task 239 の範囲。それを超える長さは long ゲートが要る。
    if n > cap.saturating_mul(2) && !long_on {
        return base.max(1);
    }
    balanced_chunk(n, base, cap)
}


/// 均等割りを使うか (プロセスで 1 回だけ env を読む)。
pub fn chunk_balance() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| chunk_balance_from_value(std::env::var(CHUNK_BALANCE_ENV).ok().as_deref()))
}

/// `DENNO_PF_CHUNK_BALANCE_LONG`: 均等割りを 2 枚を超える長さ (14k など) まで
/// 広げる (task 246)。
///
/// 3,222 トークンは 1,611 × 2 枚で済むが、14,018 は 1,536 固定だと 10 枚
/// (9 × 1,536 + 194) になり、同じ expert を 10 回上げる。上限
/// [`CHUNK_BALANCE_CAP`] (2,047) の下で枚数を最小にすると 2,003 × 7 枚になり、
/// 転送量が約 3 割減る見込み (推定)。
///
/// 既定は off (第 1 ラウンドの lab で測ってから決める)。
pub const CHUNK_BALANCE_LONG_ENV: &str = "DENNO_PF_CHUNK_BALANCE_LONG";

/// [`CHUNK_BALANCE_LONG_ENV`] の既定 (off = 従来の固定 1,536)。
///
/// lab t246b (depth8k、int-auto 1d36583f6 + br239、x0/base/lbal/lbal2/base2):
///
/// | cond | c8000 cold (14,018 tok) | 転送 up_MB | copy_s | c8000 cold decode | c2000 cold |
/// | --- | --- | --- | --- | --- | --- |
/// | base / base2 | 25.11 / 25.11 s | 151,898 | 5.89 | 18.4 / 18.4 | 7.07 / 7.09 s |
/// | lbal | 24.41 s | 122,068 | 4.80 | 18.9 | 6.80 s |
/// | lbal2 | 24.33 s | 122,068 | 4.74 | 18.3 | 6.80 s |
///
/// チャンクは 10 枚 (9 枚ストリーム + 194 トークンの従来経路) から 2,003 × 7 枚へ。
/// 転送は -20%、copy は -1.1 s だが**壁時計は -2.9%** にとどまる (14k の prefill は
/// 転送よりホスト側の計算が支配的で、7 枚が上限 = GDN 連鎖の境界 2,048 の手前)。
/// 目標の -5% に届かないので**既定 off**。
pub const DEFAULT_CHUNK_BALANCE_LONG: bool = false;

/// [`CHUNK_BALANCE_LONG_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn chunk_balance_long_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
        None => DEFAULT_CHUNK_BALANCE_LONG,
    }
}

/// 長いプロンプトまで均等割りするか (プロセスで 1 回だけ env を読む)。
pub fn chunk_balance_long() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        chunk_balance_long_from_value(std::env::var(CHUNK_BALANCE_LONG_ENV).ok().as_deref())
    })
}

// ---------------------------------------------------------------------------
// task 246: prefill 終端で decode の作業集合を LRU の先頭へ寄せる
// ---------------------------------------------------------------------------
//
// 機序 (REPORT_P239): 固定 1,536 で切ると 3,222 トークンの末尾が 150 トークン
// (< `stream_min_tok` = 512) になり、その 1 枚だけ**従来経路**へ落ちる。従来経路は
// 1 トークンずつ `moe_ffn_quant` を通るので、ヒットした expert を `acquire` で
// 触る (= LRU の先頭へ寄せる)。均等割りで 2 枚にすると全チャンクがストリーミング
// 経路 (peek のみ = **LRU を触らない**) になり、末尾の温めが消える。これが
// c8869 の decode ヒット率 0.929 → 0.912 / decode +2% の正体。
//
// 対策は H2D を伴わない「触り直し」: 最後のチャンクの末尾 `TOK` トークンが選んだ
// expert を、**トークン順 → 層順 → j 順**に `touch_many` する。これは従来経路が
// やっていたことと同じ順序で、載っていない鍵は黙って飛ばす (アップロードしない)。
// `TOK` は従来経路に落ちていた末尾チャンクの長さ (c8869 で 150、14k で 194) を
// 覆う 256 を既定にする。

/// `DENNO_PF_DEC_RESEED`: prefill の終端で末尾トークンの expert を LRU の先頭へ
/// 寄せ直す (task 246、H2D なし)。
pub const DEC_RESEED_ENV: &str = "DENNO_PF_DEC_RESEED";

/// [`DEC_RESEED_ENV`] の既定 (off)。
///
/// lab t246a / t246c (tools) の実測: 発火はする (`[pf-reseed] tok=256 keys=122880
/// resident=89844 (73.1%) in 1.84ms`、`[pf-reseed] tok=19 ... in 0.25ms` = 短い
/// prefill でも末尾 19 トークンぶん)。decode のヒット率は 0.910 → 0.913 (+0.3 pp)
/// で、COND 全体の decode 中央値は 20.3 → 20.3 (変化なし)。
/// **c8869 cold の直後の decode +5.8% は消えない** (bal 20.0 / reseed 19.8 /
/// base 18.8) ので、目標 (decode +1% 以内) は満たさない。既定 off。
pub const DEFAULT_DEC_RESEED: bool = false;

/// `DENNO_PF_DEC_RESEED_TOK`: 何トークンぶんを寄せ直すか (既定 256)。
pub const DEC_RESEED_TOK_ENV: &str = "DENNO_PF_DEC_RESEED_TOK";

/// [`DEC_RESEED_TOK_ENV`] の既定。
pub const DEFAULT_DEC_RESEED_TOK: usize = 256;

/// [`DEC_RESEED_TOK_ENV`] の上限 (これを超える指定は丸める)。
pub const DEC_RESEED_TOK_MAX: usize = 4096;

/// [`DEC_RESEED_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn dec_reseed_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
        None => DEFAULT_DEC_RESEED,
    }
}

/// 寄せ直しを使うか (プロセスで 1 回だけ env を読む)。
pub fn dec_reseed_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dec_reseed_from_value(std::env::var(DEC_RESEED_ENV).ok().as_deref()))
}

/// [`DEC_RESEED_TOK_ENV`] の解釈 (純関数)。0 と解析不能は既定。
pub fn dec_reseed_tok_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(n) if n > 0 => n.min(DEC_RESEED_TOK_MAX),
        _ => DEFAULT_DEC_RESEED_TOK,
    }
}

/// 寄せ直すトークン数 (プロセスで 1 回だけ env を読む)。
pub fn dec_reseed_tok() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| dec_reseed_tok_from_value(std::env::var(DEC_RESEED_TOK_ENV).ok().as_deref()))
}

/// 末尾トークンの選択を控える器。
///
/// `per_layer[layer]` は「層 `layer` の末尾 `tok` トークン × k 本」を
/// **トークン昇順 → j 昇順**で平らに持つ (先頭が `nt - tok` 番目のトークン)。
#[derive(Default)]
struct DecHot {
    /// 最後に観測したチャンクのトークン数 (層 0 で更新)。
    nt: usize,
    /// 1 トークンあたりの選択数 (層で同じ)。
    k: usize,
    /// 控えた末尾トークン数。
    tok: usize,
    /// 層 → 末尾トークンの選択 (平ら)。
    per_layer: Vec<Vec<u16>>,
}

thread_local! {
    static DEC_HOT: std::cell::RefCell<DecHot> = std::cell::RefCell::new(DecHot::default());
}

/// 1 層ぶんの選択を控える本体 (純粋・テスト可能)。
///
/// 層 0 がチャンクの先頭 (このエンジンは層を 0 から順に回す)。チャンクが
/// 変わると前のチャンクの内容は捨てるので、prefill が終わった時点では
/// **最後のチャンク**の末尾トークンだけが残る。
fn note_tail_into(h: &mut DecHot, layer: usize, nt: usize, k: usize, sels: &[usize], tok: usize) {
    if nt == 0 || k == 0 || tok == 0 || sels.len() != nt * k {
        return;
    }
    if layer == 0 || h.nt != nt || h.k != k {
        if layer != 0 {
            // 層によってチャンク長が違うのは想定外 (控えない方が安全)。
            return;
        }
        h.nt = nt;
        h.k = k;
        h.tok = tok.min(nt);
        h.per_layer.clear();
    }
    if h.tok != tok.min(nt) || h.k != k {
        return;
    }
    if h.per_layer.len() <= layer {
        h.per_layer.resize_with(layer + 1, Vec::new);
    }
    let lo = nt - h.tok;
    let row = &mut h.per_layer[layer];
    row.clear();
    row.extend(sels[lo * k..].iter().map(|&e| e as u16));
}

/// 1 層ぶんの選択を控える (`sels` は `nt * k` の平らな配列、行はトークン昇順)。
pub fn note_tail_selection(layer: usize, nt: usize, k: usize, sels: &[usize]) {
    if !dec_reseed_on() {
        return;
    }
    let tok = dec_reseed_tok();
    DEC_HOT.with(|c| note_tail_into(&mut c.borrow_mut(), layer, nt, k, sels, tok));
}

/// 控えた選択を「触る順」で取り出す本体 (純粋・テスト可能)。取り出したら消す。
///
/// 順序は従来経路 (1 トークンずつ回す経路) と同じ
/// 「トークン昇順 → 層昇順 → j 昇順」。`touch_many` は渡した順に新しくするので、
/// 最後のトークンの選択が一番「最近使った」になる。
fn take_dec_hot_from(h: &mut DecHot, tok: usize) -> (Vec<(u16, u16)>, usize) {
    let (ntok, k) = (h.tok.min(tok), h.k);
    let mut out: Vec<(u16, u16)> = Vec::new();
    if ntok > 0 && k > 0 {
        out.reserve(ntok * k * h.per_layer.len());
        for t in 0..ntok {
            for (layer, row) in h.per_layer.iter().enumerate() {
                if row.len() != ntok * k {
                    continue;
                }
                for j in 0..k {
                    out.push((layer as u16, row[t * k + j]));
                }
            }
        }
    }
    *h = DecHot::default();
    (out, ntok)
}

/// 控えた選択を「触る順」で取り出す (**取り出したら消す**)。ゲート off なら空。
///
/// 戻り値は `(keys, 末尾トークン数)`。
pub fn take_dec_hot_keys() -> (Vec<(u16, u16)>, usize) {
    if !dec_reseed_on() {
        return (Vec::new(), 0);
    }
    let tok = dec_reseed_tok();
    DEC_HOT.with(|c| take_dec_hot_from(&mut c.borrow_mut(), tok))
}

/// [`PF_CHUNK_ENV`] の解釈 (純関数)。0・空・偽値・解析不能は `None` (= 上書きしない)。
pub fn chunk_override_from_value(v: Option<&str>) -> Option<usize> {
    let s = v?.trim();
    if s.is_empty() {
        return None;
    }
    if matches!(
        s.to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "no"
    ) {
        return None;
    }
    let n: usize = s.parse().ok()?;
    if n == 0 {
        None
    } else {
        Some(n.clamp(1, 65536))
    }
}

/// 実際に使う上書き値。
///
/// 優先順: `DENNO_PF_CHUNK` (明示) > ストリーミング経路の既定
/// ([`DEFAULT_PF_CHUNK_STREAM`]) > 上書きなし (`DENNO_PREFILL_BATCH_CHUNK` の経路)。
pub fn chunk_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var(PF_CHUNK_ENV).ok().as_deref() {
        Some(s) => chunk_override_from_value(Some(s)),
        // 未設定: ストリーミング経路 (1 層 1 チャンクで上げる) のときだけ既定を当てる。
        // stream off では従来の `DENNO_PREFILL_BATCH_CHUNK` をそのまま使う。
        None => stream_enabled().then_some(DEFAULT_PF_CHUNK_STREAM),
    })
}

/// [`EXPERT_IQ_PF_MIN_HITS_ENV`] の既定値。
pub const DEFAULT_MIN_HITS: usize = 4;

/// [`EXPERT_IQ_PF_UPLOAD_MB_ENV`] の既定値 (0 = 上げ直さない)。
pub const DEFAULT_UPLOAD_MB: usize = 0;

/// [`EXPERT_IQ_PF_UPLOAD_ROWS_ENV`] の既定値。
pub const DEFAULT_UPLOAD_ROWS: usize = 8;

/// [`STREAM_ENV`] の既定値。**task 165 で採用して on**。
///
/// task 162 は lab t162c で不採用だった (14,018 が -12.4% で目標 -15% に届かず)。
/// task 165 の lab で残りを詰めた (depth8k、比較は各 run の off に対して):
///
/// | 条件 | 3,826 cold | 14,018 cold | decode |
/// | --- | --- | --- | --- |
/// | t165a ut=1 (スレッド) | 29.30 s | 102.10 s | 28 ms/tok |
/// | t165a ut=0 (直列) | 28.68 s | 99.97 s | 28 ms/tok |
/// | t165b g64+WC | 27.30 s (-22.5%) | 95.06 s (-15.3%) | 28 ms/tok |
/// | t165c g64+WC | 27.32 s (-22.7%) | 95.31 s (-15.3%) | 28 ms/tok |
///
/// 効いたのは **write-combined のバウンス** (probe 26.8 → 53.5 GB/s、copy も
/// 2 割減) と **グループ 64** (per-group の固定費が半分)。上げ専用スレッドは
/// 効かなかった (`STREAM_UT_ENV` を見よ)。
pub const DEFAULT_STREAM_ON: bool = true;

/// [`STREAM_MB_ENV`] の既定値 (2 枚で 512 MiB)。
pub const DEFAULT_STREAM_MB: usize = 512;

/// [`STREAM_MIN_TOK_ENV`] の既定値 (これ未満は従来経路)。
pub const DEFAULT_STREAM_MIN_TOK: usize = 512;

/// [`STREAM_GROUP_ENV`] の既定値 (1 回に上げる expert 数)。
///
/// task 165: 32 → 64 (固定費の削減。lab t165c: 14k cold は g32 96.81 s に対し
/// g64 95.31 s)。
///
/// task 324: **64 → 32**。カーネルの転送の裏への重なり (`DENNO_PF_UP_GPUWAIT`)
/// が既定になると、残る露出は**層の入口の段差** (最初のグループの転送) になる。
/// 段差はグループのバイト数に比例するので、g64 → g32 で半減する。lab t324b
/// (3,826 cold、2 ラウンド): g64 2,605 / 2,607 → g32 2,532 / 2,532 ms (-2.8%)、
/// 14k cold 6,641 → 6,517 (-1.9%)。`DENNO_PF_STREAM_GROUP=64` で切り戻せる。
pub const DEFAULT_STREAM_GROUP: usize = 32;

/// [`STREAM_PIN_ENV`] の既定値 (1 = pinned バウンス + 非同期)。
pub const DEFAULT_STREAM_PIN: usize = 1;

/// [`STREAM_UT_ENV`] の既定値 (task 320: **on**。`0` / `off` / `false` で直列経路)。
///
/// task 165: 当時は既定 off だった (lab t165a、層優先になる前のチャンク優先で
/// ut=1 は 14k cold 102.10 s、ut=0 は 99.97 s)。当時は写しと DMA が同じ
/// 帯域を食ううえ、依頼が 1 グループ先だけだったため隠れなかった。
///
/// task 320 で層優先 + グループ 64 + 3 枚のステージングの経路で測り直した
/// (lab t320a / t320b、depth8k、比較は同じ run の base/base2):
///
/// | 条件 | 3,826 cold | 14k cold | decode |
/// | --- | --- | --- | --- |
/// | t320a ut=1 (load 4.7) | 2.891 s (対 base 3.112 / base2 2.818) | 7.30 s (対 base 8.56 / base2 8.72) | 17.1 (base 17.1) |
/// | t320b ut=1 (load 1.9) | **2.642 s (対 base 2.776 / base2 2.791)** | 6.77 s (対 base 6.80 / base2 6.84) | 17.3 (base 17.1) |
/// | t320b ut=1 + 先読み 2 | 2.647 s | 6.75 s | 17.4 |
///
/// 3,826 は 2 ラウンドの中央値で **-4.9%** (t320b、負荷の高い t320a でも -5%)。
/// 14k は負荷の高い run で -14.7%、空いている run では -0.4% (床は DDR 帯域)。
/// decode は 17.1-17.4 で 0.1 ms の刻み幅の範囲 (差は経路の外)。
/// `exposed_host` は 1,402 → 1,270 ms、`stage_copy` はワーカーへ移り
/// (`up_wait` 405 → 1,112 ms)、`gpu_busy` は 1,364 → 1,376 ms で変わらない。
pub const DEFAULT_STREAM_UT: bool = true;

/// `DENNO_PF_STREAM_LOOKAHEAD`: 上げ専用スレッドへ **何グループ先まで** 依頼するか
/// (task 320、既定 1 = task 165 のまま)。
///
/// 依頼を 1 つ先に限ると、計算スレッドが待つ `ev_up` は「1 反復ぶん前に
/// 依頼したグループ」のもので、そのグループはまだ写し (1 グループ 約 3 ms) と
/// DMA を終えていない。つまり写しがカーネルの裏に隠れず、直列のときと
/// ほとんど変わらない (lab t320a: ut=1 で wall -5.0%、`up_wait` 405 → 1,112 ms)。
/// 2 つ先まで依頼すると、待つグループは 2 反復前に依頼済みになり、写しは
/// カーネルの裏に隠れる。
///
/// 上限はステージングの枚数 - 1 ([`crate::cuda`] の 3 枚 = 2)。
/// 依頼が 2 つ先まで進むと 3 枚すべてが「上げ中」になり、スロットの再利用は
/// 前のグループのカーネル完了 (`ev_use`) を待つ (正しさは変わらない)。
pub const STREAM_LOOKAHEAD_ENV: &str = "DENNO_PF_STREAM_LOOKAHEAD";

/// [`STREAM_LOOKAHEAD_ENV`] の既定値 (1 = 1 つ先だけ)。
pub const DEFAULT_STREAM_LOOKAHEAD: usize = 1;

/// 依頼を何グループ先まで出すか (純関数)。1..=2 に丸める。
pub fn stream_lookahead_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_STREAM_LOOKAHEAD;
    };
    match s.parse::<usize>() {
        Ok(n) => n.clamp(1, 2),
        Err(_) => DEFAULT_STREAM_LOOKAHEAD,
    }
}

/// 依頼を何グループ先まで出すか (プロセスで 1 回だけ env を読む)。
pub fn stream_lookahead() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        stream_lookahead_from_value(std::env::var(STREAM_LOOKAHEAD_ENV).ok().as_deref())
    })
}

/// `DENNO_PF_UP_DECOUPLE=1`: 上げ専用スレッドのホスト待ちを「pinned を前回
/// 読んだ DMA の完了」だけにし、VRAM ステージングの再利用待ち (カーネルの完了) は
/// 上げストリームの `cuStreamWaitEvent` として GPU に守らせる (task 320、既定 off)。
///
/// 既定 (off) ではワーカーがカーネルの完了をホストで待つので、写しは常に
/// 「前のグループのカーネルが終わってから」始まる。1 グループの内訳は
/// 写し 3.1 ms / DMA 2.5 ms (競合時 3.2) / カーネル 4.0 ms で、直列に近い
/// 並びだと壁時計は 3 つの和になる (lab t320b: 3,826 cold で
/// wall 2,644 = gpu_busy 1,376 + exposed 1,268、ut=1 のまま)。
/// 分離すると写しがカーネルの裏に入り、壁時計は「カーネル」と「転送」の
/// 大きいほうに近づく (推定)。
pub const UP_DECOUPLE_ENV: &str = "DENNO_PF_UP_DECOUPLE";

/// [`UP_DECOUPLE_ENV`] の解釈 (明示的な真値だけ on)。
pub fn up_decouple_from_value(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// ホスト待ちを DMA の完了だけにするか (プロセスで 1 回だけ env を読む)。
pub fn up_decouple_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| up_decouple_from_value(std::env::var(UP_DECOUPLE_ENV).ok().as_deref()))
}

/// `DENNO_PF_UP_GPUWAIT=1`: 計算スレッドがグループの H2D 完了を**ホストで待たず**、
/// 計算ストリームに `cuStreamWaitEvent` として入れる (task 324、既定 off)。
///
/// # なぜ要るか (lab t320c の台帳、3,826 cold、既定 ut=1)
///
/// 転送は写しが床 (22.65 GB / 1.08 s = 21 GB/s、DDR 律速) で、MoE のカーネルは
/// 同じ区間に 598 ms ある。ところが計算スレッドはグループごとに
/// `cuEventSynchronize(ev_up)` で H2D の完了を待ってからカーネルを積むので、
/// 待っているあいだ GPU は遊び、カーネルは転送の**後ろ**に直列に付く
/// (wall 2,604 = gpu_busy 1,375 + exposed 1,228、うち `up_wait` 1,111)。
///
/// 待ちを GPU の後ろへ送ると、GPU は前のグループの転送完了で自分から走り出し、
/// 計算スレッドは次の依頼を積み続けられる。壁時計は「転送」と「カーネル」の
/// 大きいほう (推定 1.1 s) に近づく。
///
/// 依頼 (`PfUpJob`) の H2D は**積まれた後**でないと `cuStreamWaitEvent` が
/// 古い記録を見るので、`ready` (ワーカーが ev_up を記録した) まではホストで待つ。
pub const UP_GPUWAIT_ENV: &str = "DENNO_PF_UP_GPUWAIT";

/// [`UP_GPUWAIT_ENV`] の既定値 (lab t324b で採ったので on)。
///
/// lab t324b (3,826 cold、2 ラウンド、t324b の `[q4ledger-pf]`): gpuwait + nosync +
/// group 32 で 2,532 / 2,532 ms、旧既定 (off) は 2,607 / 2,609 ms = **-2.9%**。
/// 14k cold は 6,517 対 6,655 = -2.1% (round 1 の 1 サンプル)。
pub const DEFAULT_UP_GPUWAIT: bool = true;

/// [`UP_GPUWAIT_ENV`] の解釈。未設定は [`DEFAULT_UP_GPUWAIT`]、
/// `0` / `off` などは偽。
pub fn up_gpuwait_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_UP_GPUWAIT,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// グループの H2D 完了を計算ストリームの待ちにするか (プロセスで 1 回だけ env を読む)。
pub fn up_gpuwait_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| up_gpuwait_from_value(std::env::var(UP_GPUWAIT_ENV).ok().as_deref()))
}

/// `DENNO_PF_STREAM_NOSYNC=1`: 層優先 MoE の出力がデバイスのままのとき、
/// 層の終端で `cuCtxSynchronize` しない (task 324、既定 off)。
///
/// # なぜ要るか
///
/// 層の終端の同期は「最後のグループのカーネル + 累積」が終わるまでホストを
/// 止める。その間 GPU は遊び、次の層の attention はその後でしか積まれない。
/// 待たなくても、次の層のホスト側の書き込みは同期コピー (`cuMemcpyHtoD` /
/// `cuMemcpyDtoH`) なので、既定ストリームの前の仕事を待つ = 順序は保たれる。
/// 出力をホストへ戻す経路 (`PfStreamIo::Host`) は従来どおり待つ。
pub const STREAM_NOSYNC_ENV: &str = "DENNO_PF_STREAM_NOSYNC";

/// [`STREAM_NOSYNC_ENV`] の既定値 (gpuwait と一緒に lab t324b で採ったので on)。
///
/// 単独では効果なし (lab t324b の gwn 2,606 対 gw 2,611 ms) だが、採った形
/// (gpuwait + nosync + group 32) の一部なので同じ既定にする。
pub const DEFAULT_STREAM_NOSYNC: bool = true;

/// [`STREAM_NOSYNC_ENV`] の解釈。未設定は [`DEFAULT_STREAM_NOSYNC`]。
pub fn stream_nosync_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_NOSYNC,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// 層の終端で同期しないか (プロセスで 1 回だけ env を読む)。
pub fn stream_nosync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stream_nosync_from_value(std::env::var(STREAM_NOSYNC_ENV).ok().as_deref()))
}

/// `DENNO_PF_STAGE_NOSYNC`: prefill の mix / combine の終端でホストが待たない。
///
/// # なぜ要るか (lab t329b の台帳、UD-Q4_K_XL の 14k cold、1 サンプル)
///
/// 14k は 48 層 × 7 チャンク = 336 チャンク層で、1 層あたり mix 2 回 + combine 2 回。
/// `other` 5,169 ms のうち **combine の待ちが 1,668 ms** (672 回 × 2.5 ms)、
/// `mix` も 870 ms (672 回 × 1.3 ms) を待ちに使っていた。`cuCtxSynchronize` は
/// **文脈の全ストリーム**を待つので、上げ専用スレッドの非同期 H2D (non-blocking
/// ストリーム) まで毎回吸い付ける。GPU は 21.5 s 動いていて壁は 27.7 s、差の
/// 6.1 s がこの露出。
///
/// # なぜ外してよいか
///
/// 段の終端でホストが待たなくても、次の段は
/// - 同じ (レガシー既定) ストリームへの非同期起動、または
/// - 同じストリームの**同期**コピー (`cuMemcpyHtoD` / `cuMemcpyDtoH`)
///
/// のどちらかで、どちらもストリーム順で前のカーネルの完了を待つ。ホストが
/// デバイスの結果を読む経路は必ず同期コピーを通る (`pf_take_mixed` /
/// `pf_download_res` / KV の `to_slice`) ので、古いバイト列は読まない。
/// 数値には触らない (バイト列も丸め順も同じ)。
///
/// # 実測 (lab t334a / t334b、UD-Q4_K_XL、depth8k、PROFILE=1)
///
/// 14k cold: t334a の b1 28.74 → ng **27.23 s (-5.3%)**、t334b の b1 29.11 →
/// ngw **27.33 s (-6.1%)**。3,826 cold (各 2 ラウンドの中央値) は t334a
/// 10.41 → 10.18 (-2.2%)、t334b 10.22 → 9.86 (-3.5%)。md5 は 8 run すべて一致。
/// 採用は lab の 2 ラウンド中央値 -2.0% (14k) / -2.2 と -3.5% (3,826) による。
pub const STAGE_NOSYNC_ENV: &str = "DENNO_PF_STAGE_NOSYNC";

/// [`STAGE_NOSYNC_ENV`] の既定 (lab t334 で採ったので **on**)。
pub const DEFAULT_STAGE_NOSYNC: bool = true;

/// [`STAGE_NOSYNC_ENV`] の解釈。未設定 / 空は [`DEFAULT_STAGE_NOSYNC`]。
pub fn stage_nosync_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STAGE_NOSYNC,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// mix / combine の終端で待たないか (プロセスで 1 回だけ env を読む)。
pub fn stage_nosync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stage_nosync_from_value(std::env::var(STAGE_NOSYNC_ENV).ok().as_deref()))
}

/// `DENNO_PF_ATTN_NOSYNC`: prefill の attention 連鎖 (full 層 / ssm 層 / shared) の
/// **段の終端**でホストが待たない。
///
/// # なぜ要るか (lab t329b の台帳、UD-Q4_K_XL の 14k cold、1 サンプル)
///
/// 336 チャンク層の待ちは `ssm` 1,546 ms (756 回 = 36 層 × 7 チャンク × 3 段) /
/// `full` 631 ms (168 回) / `other` 1,656 ms / `mix` 826 ms で、合計 4.66 s。
/// 壁 27.7 s のうち GPU が 21.5 s しか動いていない差がこれ。
///
/// ssm の連鎖は 1 チャンク層につき 3 回 (`投影` / `スキャン` / `out_proj`) 待つ。
/// もともと「段ごとの壁時計を意味のある値にするため」に置かれたもので、
/// 待った直後にホストが読むのは**同期コピーで落とした結果**か、`out_proj` の
/// 出力をそのまま combine に渡すデバイス経路だけ。full 層の 3 本も
/// 「待ってから `to_slice` で D2H」の形で、`to_slice` 自体がストリーム順。
///
/// # 数値
///
/// 待つ位置を消すだけで、起動するカーネルも引数もバイト列も変えない
/// (lab t334 の 8 run で md5 一致)。段ごとの壁時計 (`phase_ns`) は
/// 「起動だけ」の値になる。
pub const ATTN_NOSYNC_ENV: &str = "DENNO_PF_ATTN_NOSYNC";

/// [`ATTN_NOSYNC_ENV`] の既定 (lab t334 で採ったので **on**)。
pub const DEFAULT_ATTN_NOSYNC: bool = true;

/// [`ATTN_NOSYNC_ENV`] の解釈。未設定 / 空は [`DEFAULT_ATTN_NOSYNC`]。
pub fn attn_nosync_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_ATTN_NOSYNC,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// attention 連鎖の段の終端で待たないか (プロセスで 1 回だけ env を読む)。
pub fn attn_nosync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| attn_nosync_from_value(std::env::var(ATTN_NOSYNC_ENV).ok().as_deref()))
}

/// `DENNO_PF_MIX_GAMMA_CACHE`: mix の norm (`gamma`) をデバイスに常駐させる。
///
/// 14k の層優先 prefill は 48 層 × 7 チャンク × 2 mix = 672 回、mix のたびに
/// 40 KB を `cuMemcpyHtoD` で上げ直していた。同期コピーなので、そのストリームに
/// 積まれた前のカーネルを待つ = 段の待ちを外した木では新しい直列点になる
/// (lab t334a の `g` で h2d 4,750.7 → 4,723.2 MB = -27.5 MB がこの 672 回ぶん)。
/// norm は層の定数なので、1 回上げて使い回せば同じバイト列になる。
pub const MIX_GAMMA_CACHE_ENV: &str = "DENNO_PF_MIX_GAMMA_CACHE";

/// [`MIX_GAMMA_CACHE_ENV`] の既定 (lab t334 で採ったので **on**)。
pub const DEFAULT_MIX_GAMMA_CACHE: bool = true;

/// [`MIX_GAMMA_CACHE_ENV`] の解釈。未設定 / 空は [`DEFAULT_MIX_GAMMA_CACHE`]。
pub fn mix_gamma_cache_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_MIX_GAMMA_CACHE,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// mix の norm を常駐させるか (プロセスで 1 回だけ env を読む)。
pub fn mix_gamma_cache_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        mix_gamma_cache_from_value(std::env::var(MIX_GAMMA_CACHE_ENV).ok().as_deref())
    })
}

/// `DENNO_PF_GDN_WB_DIRECT`: GDN の状態のホスト書き戻しを、呼び出し側の配列へ
/// 直接 D2H する (**既定 off**)。
///
/// 従来は `to_vec` で 1.6 MB の Vec を確保して D2H し、`copy_from_slice` で
/// もう 1 回写していた (36 層 × 7 チャンク × 2 本 = 504 回、台帳の `ssm` の
/// d2h 823.7 MB)。`to_slice` は同じ同期 D2H を 1 回だけ行う。
/// デバイスへ渡るバイト列もホストの配列も同じ。
///
/// lab t334b の `ngw` (これ + 上の 3 本) は 14k cold 27.33 s (同じ run の b1
/// 29.11 に対して -6.1%) だったが、**3,826 cold が 10.45 対 10.22 (+2.3%)** で
/// 1 サンプルしか無いので既定 off のまま残す (`ssm` の cpu は 68 → 23 ms に
/// なっており、効かないのは待ちであって写しではない)。次のタスクで 2 ラウンド
/// 測り直すこと。
pub const GDN_WB_DIRECT_ENV: &str = "DENNO_PF_GDN_WB_DIRECT";

/// [`GDN_WB_DIRECT_ENV`] の既定 (測り直すまで off)。
pub const DEFAULT_GDN_WB_DIRECT: bool = false;

/// [`GDN_WB_DIRECT_ENV`] の解釈 (明示的な真値だけ on)。
pub fn gdn_wb_direct_from_value(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// 状態の書き戻しを直接 D2H にするか (プロセスで 1 回だけ env を読む)。
pub fn gdn_wb_direct_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| gdn_wb_direct_from_value(std::env::var(GDN_WB_DIRECT_ENV).ok().as_deref()))
}

/// `DENNO_PF_GDN_WB_LAST`: GDN 状態のホスト書き戻しを**層の最終チャンクだけ**にする
/// (task 379、lab t379a / t379b で採ったので既定 on。`=0` で切り戻す)。
///
/// # なぜ要るか (lab t377c の台帳、既定モデル、14k cold)
///
/// 層優先 prefill は 36 ssm 層 × 7 チャンク = 252 チャンク層で、各チャンク層の終端に
/// `gdn_host_writeback` (conv + ssm の **同期** D2H 2 本、計 1.63 MB) が入る。
/// 14k の `[q4ledger-pf]` は `ssm` の d2h **823.69 MB / 504 回**、`ssm` の
/// exposed 1,052.5 ms (その段の壁 2,610.4 の 40%) で、これが ssm 段でいちばん
/// 大きいホスト側の項目だった。同期 D2H はそのストリームの前のカーネルを待つので、
/// チャンク層ごとにホストが GPU の終わりを待つ直列点にもなっている。
///
/// # なぜ最終チャンクだけでよいか
///
/// prefill のスキャン状態はデバイスが正本 (`gdn_state_begin` は `valid[li]` なら
/// ホストから seed しない) なので、途中チャンクのホスト控えは次チャンクが読まない。
/// ホスト控えを読むのは prefill が終わったあとの decode (`GdnDecodeReq::conv_host`)
/// と CPU 経路だけで、どちらも層の最終チャンクのあと。層の中のチャンクは必ず
/// 昇順に回るので、最終チャンクの書き戻しで prefill 終端のホスト控えは従来と
/// 同じバイト列になる (値は 1 bit も変わらない)。
///
/// 途中チャンクで連鎖に乗れずホスト経路へ落ちる場合 (`DENNO_DEBUG_GDN_DUMP_LAYER`
/// の診断時) は、ホスト控えが古いので**飛ばさない** ([`crate::cuda`] の
/// `pf_gdn_last_chunk_skippable`)。
pub const GDN_WB_LAST_ENV: &str = "DENNO_PF_GDN_WB_LAST";

/// [`GDN_WB_LAST_ENV`] の既定 (lab t379a / t379b で採ったので **on**)。
///
/// lab 実測 (既定モデル、depth8k、PROFILE=1、`[q4ledger-pf]` の ssm 段):
///
/// | 段 | 3,826 cold | 14k cold |
/// | --- | --- | --- |
/// | ssm の壁 (ゲート off -> on) | 616.9/622.6/620.0/622.7 -> 456.1/449.2/454.6 | 2,609.2/2,610.5 -> 2,458.2/2,451.7 |
/// | ssm の d2h | 235.34 MB/144 -> 117.67 MB/72 | 823.69 MB/504 -> 117.67 MB/72 |
/// | ssm の exposed | 183.6/189.7 -> 23.1 | 1,052.9/1,053.0 -> 901.1/894.2 |
/// | wall | 2,451.5/2,462.6 -> 2,446.5/2,439.0 | 6,165.4 -> 6,001.4 |
///
/// 3,826 は ssm 段 **-26.5%** (2 run の 6 サンプル中央値) だが、その分の
/// 待ちは `mix` 段の壁 (137.3 -> 285.0 ms) へ移り、wall は -0.6% (同じ)。3,826 は
/// expert stream の写しが壁なので、ホストの待ちを消しても壁は動かない。
/// 14k は ssm 段 -5.8%、wall は 6,165.4 -> 6,001.4 = **-2.66%** (同じ run の対)。
/// md5 は 3 run の全 COND・全リクエストで SAME、decode 16.6 (0.1 ms 刻みで不変)。
pub const DEFAULT_GDN_WB_LAST: bool = true;

/// [`GDN_WB_LAST_ENV`] の解釈。未設定 / 空は [`DEFAULT_GDN_WB_LAST`]、
/// `0` / `off` などの明示的な偽値で切り戻せる。
pub fn gdn_wb_last_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_GDN_WB_LAST,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// 最終チャンクだけ書き戻すか (プロセスで 1 回だけ env を読む)。
pub fn gdn_wb_last_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| gdn_wb_last_from_value(std::env::var(GDN_WB_LAST_ENV).ok().as_deref()))
}

/// `DENNO_PF_GDN_WB_ASYNC`: 状態の書き戻し (conv 窓 + ssm 状態) を
/// `cuMemcpyDtoHAsync` (pinned) + **null ストリームだけの待ち**にする (task 397)。
///
/// # なぜ要るか
///
/// `DevBuffer::to_slice` は `cuMemcpyDtoH` で、この API は**コンテキスト全体**
/// の完了を待つ (task 369 が full 層の k/v で測ったのと同じ性質)。14k の
/// prefill は層の最終チャンクで 1 回 = 36 回この待ちが入り、`[q4ledger-pf]` の
/// `ssm` 段の exposed 900 ms がここに乗っていた。pinned へ非同期で落としてから
/// null ストリームのイベントだけを待てば、**デバイスから出るバイト列は同じ**で
/// 待ちの範囲だけが狭くなる (mix の別ストリーム / expert の上げストリームを
/// 待たない)。
///
/// 既定 **off** (lab で採否を決める。既定 on は採ったあと)。
pub const GDN_WB_ASYNC_ENV: &str = "DENNO_PF_GDN_WB_ASYNC";

/// [`GDN_WB_ASYNC_ENV`] の既定 (測るまで off)。
pub const DEFAULT_GDN_WB_ASYNC: bool = false;

/// [`GDN_WB_ASYNC_ENV`] の解釈 (明示的な真値だけ on)。
pub fn gdn_wb_async_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
        None | Some("") => DEFAULT_GDN_WB_ASYNC,
    }
}

/// 状態の書き戻しを非同期 D2H で行うか (プロセスで 1 回だけ env を読む)。
pub fn gdn_wb_async_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| gdn_wb_async_from_value(std::env::var(GDN_WB_ASYNC_ENV).ok().as_deref()))
}

/// `DENNO_PF_STREAM_FIRST_G=<n>`: 層の**先頭グループ**の expert 数を n 本に絞る
/// (task 324、既定 0 = 絞らない。`DENNO_PF_UP_GPUWAIT=1` のときだけ効く)。
///
/// 先頭グループの転送は、層の入口で GPU が遊ぶ時間そのもの (lab t324b:
/// `experts` の exposed 849 ms のうち約 290 ms が 48 層ぶんの段差)。
pub const STREAM_FIRST_G_ENV: &str = "DENNO_PF_STREAM_FIRST_G";

/// [`STREAM_FIRST_G_ENV`] の解釈 (純関数)。読めない値・0 は「絞らない」。
pub fn stream_first_group_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.parse::<usize>().map(|n| n.min(256)).unwrap_or(0),
        _ => 0,
    }
}

/// 先頭グループの expert 数 (0 = 絞らない)。
pub fn stream_first_group() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        stream_first_group_from_value(std::env::var(STREAM_FIRST_G_ENV).ok().as_deref())
    })
}

/// [`STREAM_WC_ENV`] の既定値 (true = write-combined のバウンス)。
///
/// lab t165b: グループ 64 で 14k cold が 100.01 s (wc=0) → 95.06 s (wc=1)、
/// probe 27.3 → 53.5 GB/s。t165c でも 95.31 s で再現。
pub const DEFAULT_STREAM_WC: bool = true;

/// `DENNO_PF_PIN_GROW`: pinned バウンスを「必要量ちょうど」ではなく余裕を足して
/// 確保する (task 296、既定 off)。
///
/// 従来は 1 グループに必要なバイト数 `need` ちょうどで `cuMemHostAlloc` する。
/// グループの必要量は expert の本数で数 MB 刻みに変わるので、前より 1 バイトでも
/// 大きいグループが来ると **そのスロットの pinned を確保し直す**
/// (`cuMemHostFree` + `cuMemHostAlloc`)。3,826 cold prefill の MoE 段には
/// 写しでも DMA でも説明できない ~0.7 s が残っており、その候補がこれ
/// (`DENNO_PF_LM_DBG=1` の `pin=` と `x回数` で読む)。
///
/// `=1` で [`PF_PIN_GROW_ENV`] の余裕つき確保、未設定 / `0` で従来どおり。
pub const PIN_GROW_ENV: &str = "DENNO_PF_PIN_GROW";

/// 余裕の付け方 (純関数)。`need` に +25% (最低 16 MiB) を足し、8 MiB へ切り上げる。
/// 上限はステージング 1 枚 (`slot_cap`) で、それを超えるなら `need` のまま。
pub fn pin_grow_target(need: usize, slot_cap: usize) -> usize {
    if need == 0 {
        return 0;
    }
    const Q: usize = 8 << 20;
    let head = (need / 4).max(16 << 20);
    let want = need.saturating_add(head);
    let want = want.div_ceil(Q).saturating_mul(Q);
    if slot_cap > need {
        want.min(slot_cap)
    } else {
        need
    }
}

/// [`PIN_GROW_ENV`] の解釈 (明示的な真値だけ on)。
///
/// lab t296a / t296c: 余裕なし (従来) は 3,826 cold で 3.37 / 3.41 s、余裕ありは
/// 3.18 / 3.24 s (-5.6% / -5.1%)。pinned の確保し直しが 31 回・255 ms あった
/// (`[pf-lm-dbg] stream` の `pin=`)。task 296 の目標 (-8%) には届かなかったので
/// いったん既定 off にした (採否は REPORT_P296.md)。task 300 で既定 on へ。
///
/// [`PIN_GROW_ENV`] の解釈。未設定は [`DEFAULT_PIN_GROW`]。
pub fn pin_grow_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => DEFAULT_PIN_GROW,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// [`PIN_GROW_ENV`] の既定値。
///
/// task 300: **true**。task 296 の lab t296c は同じ木で 3,826 cold prefill を
/// 3.415 → 3.235 s (-5.3%、14k と decode は無傷、md5 全 SAME) にしたが、
/// task 296 の目標 -8% に届かないので既定 off のまま残されていた。task 300 の
/// 目標は -6% なので、そのまま採る (t300 の lab で測り直す)。
pub const DEFAULT_PIN_GROW: bool = true;

/// pinned バウンスを余裕つきで確保するか (既定 off)。
pub fn pin_grow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pin_grow_from_value(std::env::var(PIN_GROW_ENV).ok().as_deref()))
}

/// `DENNO_PF_STREAM_SPLIT`: ステージングの写しと H2D を何分割して重ねるか (task 239)。
///
/// 従来 (1) は「グループ全体を pinned へ写す → 非同期 H2D を 1 回積む → 完了を待つ」
/// なので、**写しと DMA が直列**になる。lab t239a の c8869 cold では
/// up_MB=35,799 に対して copy_s=1.39、probe_GBs=54.6 (DMA 0.66 s) で、
/// この 2.05 s が prefill 6.02 s の 34% を占めていた。
///
/// n 分割すると DMA(i) が写し(i+1) と重なる (DMA は非ブロッキングストリーム上)。
/// 端数は最後の 1 枚へ寄せるので、デバイスへ渡るバイト列は 1 分割と同一 (数値不変)。
pub const STREAM_SPLIT_ENV: &str = "DENNO_PF_STREAM_SPLIT";

/// [`STREAM_SPLIT_ENV`] の既定値 (1 = 従来の 1 回写し + 1 回 DMA)。
pub const DEFAULT_STREAM_SPLIT: usize = 1;

/// [`STREAM_SPLIT_ENV`] の解釈 (純関数)。0・解析不能は既定へ、上限 16。
pub fn stream_split_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_STREAM_SPLIT;
    };
    match s.parse::<usize>() {
        Ok(n) if n > 0 => n.clamp(1, 16),
        _ => DEFAULT_STREAM_SPLIT,
    }
}

/// 写し / H2D の分割数 (プロセスで 1 回だけ env を読む)。
pub fn stream_split() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| stream_split_from_value(std::env::var(STREAM_SPLIT_ENV).ok().as_deref()))
}

/// 1 回の起動で受ける行数 (`kernels_exp_iq_pf.cu` の `PF_MR`)。
pub const ROWS_PER_LAUNCH: usize = 16;

/// 1 warp が受ける出力行数 (`PF_OUTS`)。256 要素ブロックの型だけ 2 行。
pub const OUT_PER_WARP: usize = 2;

/// 1 block の warp 数 (`PF_WARPS`)。
pub const WARPS: usize = 4;

/// このスロットの 1 block あたりの出力行数。
///
/// 256 要素ブロックの型 (IQ のスロット 0..=3、K の Q4_K / Q5_K / Q6_K) は
/// 1 warp = 2 行で 8 行 / block、32 要素ブロックの型 (IQ4_NL / Q2_0 と
/// K の Q5_1 / Q8_0) は 1 行で 4 行 / block (重み 32/64 要素をレジスタに
/// 持つので 2 行ぶんは入らない)。
pub fn tile_of(slot: usize) -> usize {
    if slot <= 3 || (K_SLOT0..K_SLOT0 + 3).contains(&slot) {
        WARPS * OUT_PER_WARP
    } else {
        WARPS
    }
}

/// IQ 系 (`kernels_exp_iq_pf.cu`) のスロット数。
pub const IQ_SLOTS: usize = 6;

/// K 系 (`kernels_exp_k_pf.cu`) のスロット数
/// (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0、task 309)。
pub const K_SLOTS: usize = 5;

/// K 系の最初のスロット。これ以上は `kernels_exp_k_pf.ptx` のカーネル。
pub const K_SLOT0: usize = IQ_SLOTS;

/// カーネルのスロット数 (IQ 6 + K 5)。
pub const SLOTS: usize = IQ_SLOTS + K_SLOTS;

/// K 系のスロットか (PTX のモジュールが違う)。
pub fn is_k_slot(slot: usize) -> bool {
    slot >= K_SLOT0
}

/// 既定で有効か (タスクの採否で決める)。
///
/// **true** (lab t151c で採用): `DENNO_EXPERT_IQ_GPU_PF` 未設定でも
/// `DENNO_EXPERT_IQ_GPU=1` なら prefill の選択 expert を GPU で回す。
/// `DENNO_EXPERT_IQ_GPU_PF=0` で従来の CPU 経路へ戻せる。
pub const DEFAULT_ON: bool = true;

/// `DENNO_EXPERT_IQ_GPU_PF` の解釈 (純関数)。`0` / `off` / `false` だけが偽で、
/// 未設定は [`DEFAULT_ON`]。**`iq_on` が false なら常に false**。
pub fn pf_from_value(iq_on: bool, v: Option<&str>) -> bool {
    if !iq_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_ON,
        Some(s) => crate::model::expert_vram::flag_on(Some(s)) || !matches!(s, "0" | "off" | "false"),
    }
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn pf_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_from_value(
            crate::model::exp_iq::iq_gpu_enabled(),
            std::env::var(EXPERT_IQ_PF_ENV).ok().as_deref(),
        )
    })
}

/// `DENNO_PF_EXP_K=1`: prefill の選択 expert を **K 系の型**
/// (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0) でも GPU の m 行カーネル
/// (`kernels_exp_k_pf.cu`) で回す (task 309)。
///
/// **`DENNO_EXPERT_IQ_GPU=1` が前提** (VRAM 常駐の LRU とストリーミングを使うため。
/// [`pf_enabled`] が同じ条件で立つ)。既定は lab の採否で決める。
pub const K_ENV: &str = "DENNO_PF_EXP_K";

/// 既定値 (**on**。lab t309b で採用)。
///
/// t309b (UD-Q4_K_XL、depth8k、PROFILE=1) の採否の数字は REPORT_P309.md。
/// `DENNO_PF_EXP_K=0` で従来の CPU 経路へ戻せる。
pub const DEFAULT_K_ON: bool = true;

/// [`K_ENV`] の解釈 (純関数)。`iq_on` が false なら常に false。
/// 明示的な偽値 (`0` / `off` / `false`) だけが偽で、未設定は [`DEFAULT_K_ON`]。
pub fn k_from_value(iq_on: bool, v: Option<&str>) -> bool {
    if !iq_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_K_ON,
        Some(s) => crate::model::expert_vram::flag_on(Some(s)) || !matches!(s, "0" | "off" | "false"),
    }
}

/// K 系のゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn k_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        k_from_value(
            crate::model::exp_iq::iq_gpu_enabled(),
            std::env::var(K_ENV).ok().as_deref(),
        )
    })
}

/// `DENNO_PF_EXP_K_MIN_ROWS`: 1 expert を GPU に回すのに要る最小の行数 (既定 8)。
///
/// これ未満の expert は **CPU のまま**回す (重み 1 本 3 MB の転送が計算より
/// 高い。1 チャンクの expert 平均は実形で 20 行前後)。`0` で床を外す。
pub const K_MIN_ROWS_ENV: &str = "DENNO_PF_EXP_K_MIN_ROWS";

/// [`K_MIN_ROWS_ENV`] の既定値。
pub const DEFAULT_K_MIN_ROWS: usize = 8;

/// [`K_MIN_ROWS_ENV`] の解釈 (純関数)。
pub fn k_min_rows_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_K_MIN_ROWS),
        _ => DEFAULT_K_MIN_ROWS,
    }
}

/// 1 expert を GPU に回す最小行数 (プロセスで 1 回だけ env を読む)。
pub fn k_min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| k_min_rows_from_value(std::env::var(K_MIN_ROWS_ENV).ok().as_deref()))
}

/// task 419: `DENNO_PF_EXP_IQ_MIN_ROWS` — **IQ 系**の expert にも行数の床を置く。
///
/// [`K_MIN_ROWS_ENV`] は K 系 (`UD-Q4_K_XL`) にしか効かない
/// (`!iq_expert_mode(lw)` の条件)。既定モデルの expert は
/// IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS / type 42 の混在 = **IQ 系**なので、
/// 行が 1-2 本しかない expert でも重み 2 MB を丸ごと GPU へ上げている
/// (lab t419a の `[exp-pf-stream]`: 14k で 13,616 本 27.9 GB、
/// 3,826 で 10,938 本 22.4 GB。1 本あたり 2.05 MB)。
/// 上げた重みで計算する行数が床未満なら、CPU の flat 経路
/// (`moe_experts_flat`) のほうが安い。
///
/// 既定 **0** (床なし = 従来どおり全部上げる)。
///
/// # 結果: **採らなかった** (lab t419c、`ADOPT: no`)
///
/// 既定モデル (UD-IQ3_XXS、depth8k、PROFILE=1、全 COND `DENNO_START_DROP_CACHE=1`):
/// 床 16 で 3,826 が 8.32 -> 12.43 s / 14k が 7.05 -> 13.38 s と **大幅に悪化**した
/// (CPU の flat 経路に回した expert の数だけ損をする。1 本 2 MB の読みより
/// そちらのほうが高い)。床 32 / 64 は 8.32 / 7.02 s で baseline と同じ
/// (分割が立っても差が出ない = CPU に回す量が変わるだけで得が無い)。
/// 掃引は 3 ラウンド目を使い切ったので、この枝は閉じる。
pub const IQ_MIN_ROWS_ENV: &str = "DENNO_PF_EXP_IQ_MIN_ROWS";

/// [`IQ_MIN_ROWS_ENV`] の既定値 (床なし)。
pub const DEFAULT_IQ_MIN_ROWS: usize = 0;

/// [`IQ_MIN_ROWS_ENV`] の解釈 (純関数)。
pub fn iq_min_rows_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_IQ_MIN_ROWS),
        _ => DEFAULT_IQ_MIN_ROWS,
    }
}

/// IQ 系の行数の床 (プロセスで 1 回だけ env を読む)。
pub fn iq_min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| iq_min_rows_from_value(std::env::var(IQ_MIN_ROWS_ENV).ok().as_deref()))
}

/// この expert の行数を GPU に回すか (純関数。`min_rows = 0` は全部 GPU)。
pub fn k_expert_pays(rows: usize, min_rows: usize) -> bool {
    rows > 0 && rows >= min_rows
}

fn env_usize(key: &str, dflt: usize) -> usize {
    match std::env::var(key).ok() {
        Some(s) if !s.trim().is_empty() => s.trim().parse().unwrap_or(dflt),
        _ => dflt,
    }
}

/// 1 層を GPU で回すのに要るヒット本数。
pub fn min_hits_per_layer() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_MIN_HITS_ENV, DEFAULT_MIN_HITS))
}

/// 1 チャンクで上げ直す上限 (MiB)。
pub fn upload_mb() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_UPLOAD_MB_ENV, DEFAULT_UPLOAD_MB))
}

/// 上げ直すミスの最小行数。
pub fn upload_min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(EXPERT_IQ_PF_UPLOAD_ROWS_ENV, DEFAULT_UPLOAD_ROWS))
}

/// `DENNO_PF_STREAM` の解釈 (純関数)。**`0` / `off` / `false` だけが偽**。
///
/// `pf_enabled()` (IQ ゲート + prefill ゲート) が偽なら常に偽。
pub fn stream_from_value(pf_on: bool, v: Option<&str>) -> bool {
    if !pf_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_ON,
        Some(s) => !matches!(s, "0" | "off" | "false"),
    }
}

/// ミス expert のストリーミングが有効か (プロセスで 1 回だけ env を読む)。
pub fn stream_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stream_from_value(pf_enabled(), std::env::var(STREAM_ENV).ok().as_deref()))
}

/// ステージング領域の合計 (バイト)。
pub fn stream_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(STREAM_MB_ENV, DEFAULT_STREAM_MB).saturating_mul(1 << 20))
}

/// ストリーミングを始める最小トークン数。
pub fn stream_min_tok() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(STREAM_MIN_TOK_ENV, DEFAULT_STREAM_MIN_TOK))
}

/// 1 回に上げる expert 数。
pub fn stream_group() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(STREAM_GROUP_ENV, DEFAULT_STREAM_GROUP).clamp(1, 256))
}

/// 上げの経路 (0 = pageable 同期 / 1 = pinned + 非同期)。
pub fn stream_pin() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(STREAM_PIN_ENV, DEFAULT_STREAM_PIN).min(1))
}

/// `DENNO_PF_STREAM_UT` の解釈 (純関数)。**`0` / `off` / `false` だけが偽**。
///
/// ストリーミング自体が無効なら常に偽。
pub fn stream_ut_from_value(pf_on: bool, v: Option<&str>) -> bool {
    if !pf_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_UT,
        Some(s) => !matches!(s, "0" | "off" | "false"),
    }
}

/// 上げ専用スレッドを使うか (プロセスで 1 回だけ env を読む)。
pub fn stream_ut() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        stream_ut_from_value(
            stream_enabled(),
            std::env::var(STREAM_UT_ENV).ok().as_deref(),
        )
    })
}

/// task 333: ストリーミングが門で落ちたとき、写しを計算スレッドへ落として
/// **1 回だけ**やり直すか。
///
/// task 328 は 3 ラウンドとも層優先 dev MoE が `理由: stream` で落ちたが、
/// どの門かは分からなかった。上げスレッド / 表の不整合で落ちたときに限り、
/// 同じカーネル・同じ順序のまま転送の担い手だけを替えて再試行する。
/// 既定 on (落ちた経路でしか走らないので、正常時の数値は変わらない)。
pub const STREAM_RETRY_ENV: &str = "DENNO_PF_STREAM_RETRY";

/// [`STREAM_RETRY_ENV`] の既定値。
pub const DEFAULT_STREAM_RETRY: bool = true;

/// [`STREAM_RETRY_ENV`] の解釈 (純関数)。`0` / `off` / `false` / `no` だけが偽。
pub fn stream_retry_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_RETRY,
        Some(s) => !matches!(s, "0" | "off" | "false" | "no"),
    }
}

/// 落ちた門の理由が、写しを計算スレッドへ落とす再試行で直りうるか (純関数)。
///
/// `no_upload` = 上げスレッドが居ない、`ptr_table` = 依頼したグループの
/// デバイスポインタ表が合わない (直列経路はポインタ表を自分で組む)。
/// `stage` (ステージングの確保) や `no_miss` (全 expert が常駐) は
/// 転送の担い手を替えても直らないので再試行しない。
pub fn stream_retry_pays(why: &str) -> bool {
    matches!(why, "no_upload" | "ptr_table")
}

/// 再試行するか (プロセスで 1 回だけ env を読む)。
pub fn stream_retry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        stream_retry_from_value(std::env::var(STREAM_RETRY_ENV).ok().as_deref())
    })
}

/// `DENNO_PF_STREAM_WC` の解釈 (純関数)。**明示的な真値だけ on**。
///
/// ストリーミング自体が無効なら常に偽。
pub fn stream_wc_from_value(pf_on: bool, v: Option<&str>) -> bool {
    if !pf_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_WC,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// バウンスを write-combined にするか (既定 false)。
pub fn stream_wc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        stream_wc_from_value(
            stream_enabled(),
            std::env::var(STREAM_WC_ENV).ok().as_deref(),
        )
    })
}

/// task 309: 層の expert 型が K 系のときの既定も込めた解釈 (純関数)。
///
/// 明示値 (`DENNO_PF_STREAM_WC`) はそのまま。**未設定のときだけ**型で変える:
/// IQ 系は従来どおり write-combined (lab t165b: probe 53.5 GB/s)、
/// K 系は通常の pinned (lab t309b: UD-Q4_K_XL の 3,826 cold が 17.0 s 対
/// write-combined の 29.6 s。WC だとホスト側の写しが 4.3 GB/s で頭打ち)。
pub fn stream_wc_for_from(v: Option<&str>, pf_on: bool, is_k: bool) -> bool {
    if !pf_on {
        return false;
    }
    match v.map(str::trim) {
        None | Some("") => !is_k,
        Some(s) => matches!(s, "1" | "on" | "true" | "yes"),
    }
}

/// このモードの write-combined 既定 (プロセスで 1 回だけ env を読む)。
pub fn stream_wc_for(is_k: bool) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static ON_K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let init = || {
        stream_wc_for_from(
            std::env::var(STREAM_WC_ENV).ok().as_deref(),
            stream_enabled(),
            is_k,
        )
    };
    if is_k {
        *ON_K.get_or_init(init)
    } else {
        *ON.get_or_init(init)
    }
}

// ---------------------------------------------------------------------------
// task 227: 転送 (写し + DMA) の詰まりを取るための 3 本
// ---------------------------------------------------------------------------
//
// 台帳 (REPORT_P226 §2) では 14k の転送 14.76 s のうち写しが 5.90 s、
// H2D のブロックが 8.85 s。どちらも同じホストメモリ帯域を食い、しかも
// **1 グループ = 「全部写す → 1 回 DMA → 待つ」の直列**なので、
// バイトを減らしても時間が動かなかった。以下は独立に切り替えられる。

/// `DENNO_PF_STREAM_CHUNK_KB`: バウンスと DMA を何 KB ずつ交互に流すか
/// (0 = 従来どおりグループ全体を 1 回で写して 1 回で DMA)。
///
/// 小さく刻むと、DMA が読むバイトが L3 (16 MB) / L2 に残っている間に
/// 流せる可能性がある (推定。lab で測る)。値は 64 KiB..65536 KiB に丸める。
pub const STREAM_CHUNK_KB_ENV: &str = "DENNO_PF_STREAM_CHUNK_KB";

/// [`STREAM_CHUNK_KB_ENV`] の既定値 (0 = 従来の直列)。
pub const DEFAULT_STREAM_CHUNK_KB: usize = 0;

/// 1 回に流すバイト数 (0 = 従来経路)。
pub fn stream_chunk_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let kb = env_usize(STREAM_CHUNK_KB_ENV, DEFAULT_STREAM_CHUNK_KB);
        if kb == 0 {
            return 0;
        }
        kb.clamp(64, 65536) << 10
    })
}

/// `DENNO_PF_STREAM_SRC`: DMA の転送元。
///
/// - `pin` (既定): mmap → WC pinned のバウンスを挟む (task 162 の経路)
/// - `page`: mmap のページをそのまま pageable として DMA する (写しを消す。
///   driver が内部でバウンスする。ホスト側は呼び出しの間ブロックする)
/// - `reg`: mmap のページを `cuMemHostRegister` で pin して直接 DMA する
///   (写しを消し、DMA も pinned 速度。driver が file backed を拒否したら
///   スライス単位で `page` へ落ちる)
pub const STREAM_SRC_ENV: &str = "DENNO_PF_STREAM_SRC";

/// [`STREAM_SRC_ENV`] の既定値 (0 = pin バウンス)。
pub const DEFAULT_STREAM_SRC: u8 = 0;

/// 転送元の解釈 (純関数)。未知の値は既定 (pin) に落とす。
pub fn stream_src_from_value(v: Option<&str>) -> u8 {
    match v.map(str::trim) {
        Some("page") => 1,
        Some("reg") => 2,
        _ => DEFAULT_STREAM_SRC,
    }
}

/// 転送元 (`DENNO_PF_STREAM_SRC`、プロセスで 1 回だけ読む)。
pub fn stream_src() -> u8 {
    static N: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *N.get_or_init(|| stream_src_from_value(std::env::var(STREAM_SRC_ENV).ok().as_deref()))
}

/// `DENNO_PF_REG_MB`: `DENNO_PF_STREAM_SRC=reg` で pin する上限 (MiB、既定 16384)。
///
/// pin したページはカーネルが回収できないので、モデル全体 (約 50 GB) を
/// pin すると RAM が足りなくなる。上限を超えたぶんは pageable で積む。
pub const REG_MB_ENV: &str = "DENNO_PF_REG_MB";

/// [`REG_MB_ENV`] の既定値 (MiB)。
pub const DEFAULT_REG_MB: usize = 16384;

/// pin してよいバイト数。
pub fn reg_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(REG_MB_ENV, DEFAULT_REG_MB).saturating_mul(1 << 20))
}

/// `DENNO_PF_TC_EXP=1`: prefill の expert グループを tensor core の grouped GEMM
/// (`kernels_exp_iq_tc.cu`) で回す (task 250、既定 off)。
///
/// 現行の m 行 GEMV は 1 block = 8 出力行なので、1 expert あたりの活性読みが
/// `n_out/8 × m × n_in × 4 B` になる。1 expert の重みは約 2 MB なので、
/// 行数が多い (m = 300 前後) layer major では活性の再読みが重みを上回る
/// (REPORT_P242)。新カーネルは 64 トークン × 64 行のタイルで受けるので、
/// 活性の読みが 8 分の 1 になり、積は bf16 の mma (FP32 アキュムレート) で行う。
///
/// 数値は変わる (重みと活性を bf16 に丸める。丸めだけの差)。
///
/// # 実測 (task 250、lab t250a / t250c / t250d、`DENNO_PF_LM_LAYER=1` と併用)
///
/// 層優先の MoE をこのカーネルに替えると prefill は大きく縮む。
/// 14k cold prefill は base の 25.1-26.4 s に対し **9.9 s (t250c)** /
/// **15.0 s (t250d、`DENNO_PF_LM_TILE` でトークン分割)**、3,826 は 7.03 → 3.25 s。
/// ところが 14k 後の decode が 18.4 → 32-52 ms/tok に落ちる。機序は作業域の
/// 確保で、VRAM はすでに expert 22 GB + 重みで埋まっているため、1.3-1.5 GB の
/// 連続確保が「空き 123 MiB」で失敗し、追い出し階段が expert を 33,497 MiB
/// **全捨て** する (decode のヒット率 0.95 → 0.86、moe experts 573 → 1,679 ms)。
/// `DENNO_PF_LM_RESERVE` / `DENNO_PF_LM_RELOAD` / トークン分割のどれでも
/// 確保そのものは通らず (予約 2,186 MiB でも実空き 1,243 MiB)、**採らない**
/// (`ADOPT: no` t250d)。ゲートは既定 off のまま残す。確保器が作業域のぶんを
/// 実際に返せるようになるまで (task 248 の領域)、この経路は既定にしない。
pub const TC_EXP_ENV: &str = "DENNO_PF_TC_EXP";

/// [`TC_EXP_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn tc_exp_from_value(v: Option<&str>) -> bool {
    // task 265 (2026-11-13, operator approved): default on; only an explicit
    // false value (0 / off / false / no) turns it off.
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// グループ GEMM を tensor core で回すか (プロセスで 1 回だけ env を読む)。
pub fn tc_exp_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| tc_exp_from_value(std::env::var(TC_EXP_ENV).ok().as_deref()))
}

/// `DENNO_PF_TC_EXP_K=1`: **K 系** (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0) の prefill
/// expert グループも tensor core の grouped GEMM (`kernels_exp_k_tc.cu`) で回す
/// (task 416、**既定 on** = lab t416b で採用)。
///
/// `DENNO_PF_TC_EXP` (IQ 系) とは別のゲートにしてある。IQ 系は 2026-11-13 から
/// 既定 on だが、K 系は新しいカーネルなので lab の A/B で採否を決めた。
/// 形が合わない (k がブロックで割り切れない / gate と up の型が違う) ときは
/// 従来の m 行 GEMV (`kernels_exp_k_pf.cu`) へ落ちる。
///
/// # 実測 (task 416、lab t416b、MODEL=flash-next-UD-Q4_K_XL、depth8k、PROFILE=1)
///
/// 14k cold prefill **17.98 s** 対 baseline 25.18 / 25.22 / 24.96 s = **-28.6%**。
/// experts 段の gpu_busy は 18.34 -> 0.43 s (3,826 は 4.70 -> 0.43 s)。
/// 3,826 cold は read (44.4 GB の pread) が律速になり、copy_s が 7.2 s の
/// ラウンドでは 8.76 s 対 baseline 9.13/9.17 s (-4%)、12.6 s のラウンドでは
/// 14.68 s (ディスクの振れ)。decode は 16.3 対 16.4 ms/tok で不変。
/// 既定モデル (IQ 系) は `exp_pf::slot` が IQ のスロットを返すので
/// `ktc_shapes_ok` が偽 = 構造的に不発 (lab t416c で md5 全一致)。
pub const TC_EXP_K_ENV: &str = "DENNO_PF_TC_EXP_K";

/// 既定値 (**on**。lab t416b で採用)。
pub const DEFAULT_TC_EXP_K_ON: bool = true;

/// [`TC_EXP_K_ENV`] の解釈 (純関数)。明示的な偽値 (`0` / `off` / `false` /
/// `no`) だけが偽で、未設定は [`DEFAULT_TC_EXP_K_ON`]。
pub fn tc_exp_k_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_TC_EXP_K_ON,
        Some(s) => crate::model::expert_vram::flag_on(Some(s))
            || !matches!(s.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
    }
}

/// K 系のグループ GEMM ゲート (プロセスで 1 回だけ env を読む)。
pub fn tc_exp_k_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| tc_exp_k_from_value(std::env::var(TC_EXP_K_ENV).ok().as_deref()))
}

/// `DENNO_PF_POOL_K_PLAN=1`: prefill のプール計画 (タイル / 余白) を **K 系の
/// dev MoE + tensor core 経路**に合わせる (task 420)。
///
/// # 機序
///
/// task 318 (`DENNO_PF_LM_K_DEV`、既定 on) で層優先のデバイス MoE は K 系
/// (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0) でも回るようになり、task 416 で K 系の
/// TC grouped GEMM (`DENNO_PF_TC_EXP_K`、既定 on) が入った。ところがプール計画の
/// 2 か所は IQ 版の述語のままだった:
///
/// 1. `pf_pool_need_for` の `tc` が IQ の `tc_shapes_ok` だけを見るので、
///    UD-Q4_K_XL では **使わない g / u の作業域**を予約する (14k で 686 MiB)。
/// 2. `pf_pool_reserve_for` の `dev_ok` が `iq_expert_mode` のままで、UD は
///    「dev MoE が使えない」= CPU 経路と判定され、余白が 3,072 MiB になる
///    (dev 経路なら 1,024 MiB)。
///
/// その結果 14k の 1 パス (タイル = 14,018) が上限に収まらず、7,009 の 2 パスに
/// なっていた。2 パスは expert 重みの pread が 98.5 GB (1 パスの 2 倍) で、
/// 14k cold prefill の 74% を占める。
///
/// # 実測 (lab t420a、sha ff1a1ec、MODEL=flash-next-UD-Q4_K_XL、depth8k)
///
/// ゲート無しで `DENNO_PF_POOL_BIG_MB=8192` を渡すと 1 パスに届く
/// (`形を上げた (タイル 5120 → 14018 = 3 → 1 パス / 作業域 4779 MiB)`)。
/// 14k cold 17.47 / 17.50 s -> **15.03 s (-14.1%)**、`up_MB` 98,460 -> 53,323、
/// copy_s 12.69 -> 9.75。ただし作業域 4,779 MiB + 余白 3,072 MiB のぶん
/// expert を 1,500 本追い出して戻すため、14k 後の decode が 23-24 -> 26 ms/tok に
/// 落ちた。
///
/// # 実測 (lab t420b、sha b187ed1、同じモデル / シナリオ、本ゲート on)
///
/// 余白 1,024 MiB / 作業域 **4,094 MiB** (g / u を取らない) / 追い出し **543 本**
/// (base は 957 本) で 1 パス。14k cold **13.56 s** 対 base 17.53 / 17.76 s =
/// **-23%**、`up_MB` 98,173 -> 52,516、copy_s 21.64 (外れ回) / 8.64。
/// 3,826 cold 9.56 対 10.13 / 10.70 / 9.36 (差はノイズ)、14k 後 decode
/// **22.0 対 21.8 ms/tok** (悪化なし)、短い decode 16.4 対 16.2、md5 は全文脈 SAME。
/// 14k 後の decode が落ちた t420a の形 (作業域 4,779 + 余白 3,072) と違い、
/// 追い出しが base より少ないので悪化しない。
pub const POOL_K_PLAN_ENV: &str = "DENNO_PF_POOL_K_PLAN";

/// [`POOL_K_PLAN_ENV`] の既定 (**on**。lab t420b で採用)。
pub const DEFAULT_POOL_K_PLAN: bool = true;

/// [`POOL_K_PLAN_ENV`] の解釈 (純関数)。既定 on、**明示的な偽値だけ off**。
pub fn pool_k_plan_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_POOL_K_PLAN,
        Some(s) => !matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
    }
}

/// K 系の dev MoE + TC 経路に合わせたプール計画を使うか (プロセスで 1 回だけ env を読む)。
pub fn pool_k_plan_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pool_k_plan_from_value(std::env::var(POOL_K_PLAN_ENV).ok().as_deref()))
}

/// `DENNO_PF_LM_LAYER=1`: prefill の層ループを層優先にする (task 227、既定 off)。
///
/// いまはチャンク優先 (チャンク → 全 48 層) なので、1 つの expert が
/// **チャンクの数だけ** H2D で運ばれる (14k で 9 パス = 158.5 GB)。層優先にすると
/// 1 プロンプトで 1 回 (48 層 × 512 本 = 50 GB が上限) になり、VRAM の
/// expert キャッシュも要らない (同じ expert を二度使わない)。
/// 代償は、層のあいだ全チャンクの残差を保持すること (n_tokens × n_embd × hc × 4)。
///
/// # 実測 (lab t227b、sha 9de5b3d2b、depth8k)
///
/// バイトは狙いどおり減った: 14k の `[exp-pf-stream] up_MB` は 158.5 GB →
/// **46.7 GB (-70%)**、copy_s 5.90 → 2.40。ところが壁時計は 3.8k +2.3% /
/// 14k **+21.6%**、14k 後の decode は 18.4 → 36.0 ms/tok で、**採らない**
/// (`ADOPT: no`)。理由は層でまとめたぶん、全トークンの活性 (14k で 143 MB) を
/// router / shared / expert 段で 3 回上げ直し (48 層で 20.6 GB)、shared と
/// expert の出力 13.8 GB を戻すため。加えて MoE の作業域 2.7 GB と事前の
/// 追い出し 6 GB が decode のヒット率を落とす。
/// 次に採るなら **活性をデバイスに残したまま** 層優先を回すこと (REPORT_P227 §7)。
///
/// # 実測 (task 233 / 242 で活性をデバイスに残した版)
///
/// 往復 (14k で 34 GB) は消え、3,826 tok は base 比 -10〜12% まで来た回もある。
/// しかし 14,018 tok は **+8〜17%** で目標 (-10%) に届かず、14k 後の decode も
/// base の 1.6 倍 (lab t242c: ヒット率 0.829 対 0.973、decode 23.0 対 18.1 ms/tok)。
/// 原因は 3 つ:
/// (1) グループのポインタ表の同期 H2D (層あたり 243-410 ms) は非同期化しても
///     待ちが「ステージングの再利用待ち」に移っただけで、14k では消えない
///     (`use_s` 11.2-11.9 s)。層優先は 1 expert あたりの行数が多いので、GEMV が
///     重みではなく活性の再読みで律速になる。
/// (2) 14k では `h` (1,505 MiB) の連続確保が失敗し、追い出し階段が expert の
///     LRU を全部捨てる (slab がチャンク単位でしか返さない)。
/// (3) 作業域を小さくする `DENNO_PF_LM_GL` と、追い出した expert を戻す
///     `DENNO_PF_LM_RELOAD` はどちらも壁時計を改善しなかった (下記の各 doc)。
/// **採らない** (`ADOPT: no` t242c)。ゲートは 3 本とも既定 off のまま残す。
pub const LM_LAYER_ENV: &str = "DENNO_PF_LM_LAYER";

/// `DENNO_PF_LM_TILE`: 層優先 prefill の MoE を **何トークンずつ**回すか
/// (task 250、既定 0 = 分割しない)。
///
/// 層優先は 1 層ぶんの全トークンの expert 出力 `h` を連続領域で取る。
/// 14k では 1.5 GB になり、lab t250a / t250c では `cuMemAlloc` が
/// 「要求 1,505 MiB / 空き 1,243 MiB」で失敗し、追い出し階段が LRU を
/// **全部**捨てた (22 GB → 0、decode のヒット率 0.25)。トークンで割ると
/// `h` が 1/分割数になり、連続確保が通る。
///
/// 代償は expert の H2D が分割数だけ増えること (14k で 1 パス 46 GB、写し 2.5 s)。
/// 1024 未満は 1024 に丸める。
///
/// # 実測 (lab t250d、sha 35273d56e、depth8k)
///
/// `h` は 14k で 1.5 GB → 471 MiB (5,120) / 660 MiB (7,168) になり、
/// **それでも確保は通らなかった**。VRAM は expert 22 GB + 重みで埋まっていて、
/// 「要求 1,368 MiB / 空き 123 MiB」→ 追い出し階段が 33,497 MiB 全捨て、が再現する。
/// 14k cold prefill は 15.0 s (base 25.6 s の -41%)、3,826 は 3.25 s (-54%) だが、
/// 14k 後の decode は 18.4 → 31.9 ms/tok (+69%) で目標 (+1% 以内) に届かない。
/// 分割しない t250c の 52.0 ms/tok よりはましだが、採らない (`ADOPT: no` t250d)。
pub const LM_TILE_ENV: &str = "DENNO_PF_LM_TILE";

/// task 265: the token tile used when `DENNO_PF_LM_TILE` is unset (lab t265c).
pub const DEFAULT_LM_TILE: usize = 5120;

/// [`LM_TILE_ENV`] の解釈 (純関数)。0 / 解析不能は分割しない。
pub fn lm_tile_from_value(v: Option<&str>) -> usize {
    // task 265 (2026-11-13): unset or empty = 5120 (lab t265c); explicit 0 = no split.
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_LM_TILE;
    };
    match s.parse::<usize>() {
        Ok(n) if n >= 1024 => n,
        _ => 0,
    }
}

/// 1 回の MoE で受けるトークン数 (nt 以下に丸める。0 相当なら分割しない)。
pub fn lm_tile(nt: usize) -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let t = *N.get_or_init(|| lm_tile_from_value(std::env::var(LM_TILE_ENV).ok().as_deref()));
    if t == 0 || t >= nt {
        nt
    } else {
        t
    }
}

/// `DENNO_PF_LM_RESERVE=1`: 層優先 prefill の作業域を **外部予約**として
/// expert の予算へ計上する (task 250、既定 off)。
///
/// いまは作業域 (14k で約 2 GB) を `DevBuffer::alloc_tagged` で取るので、
/// 確保のたびに LRU の追い出し階段が走り、確保したまま解放しない作業域のぶん
/// LRU の上限が縮んだままになる (lab t250a: 22 GB → 8.2 GB、decode の
/// ヒット率 0.247)。予約にすると上限が作業域のぶんだけ下がり、prefill の
/// 終端で返るので、次の decode で上限が戻る。
///
/// # 実測 (lab t250c、sha a57edfdec)
///
/// 予約自体は効いた (`作業域 2186 MiB を予約した (会計=true)`) が、
/// **確保は通らなかった**: 予約後の実空きは 1,243 MiB で、要求 1,505 MiB に
/// 足りず、追い出し階段が 39,171 MiB 全捨て。expert の slab はチャンクが
/// 空になるまで実メモリを返さないので、予約で LRU を 2 GB 削っても連続領域は
/// 空かない。14k 後の decode は 22.4 ms/tok (+24.8%) で目標に届かない。
pub const LM_RESERVE_ENV: &str = "DENNO_PF_LM_RESERVE";

/// [`LM_RESERVE_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn lm_reserve_from_value(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// 作業域を外部予約するか (プロセスで 1 回だけ env を読む)。
pub fn lm_reserve_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| lm_reserve_from_value(std::env::var(LM_RESERVE_ENV).ok().as_deref()))
}

/// `DENNO_PF_LM_LAYER` の解釈 (純関数)。**明示的な真値だけ on**。
pub fn lm_layer_from_value(v: Option<&str>) -> bool {
    // task 265 (2026-11-13, operator approved): default on; only an explicit
    // false value (0 / off / false / no) turns it off.
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// 層優先 prefill を使うか (プロセスで 1 回だけ env を読む)。
pub fn lm_layer_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| lm_layer_from_value(std::env::var(LM_LAYER_ENV).ok().as_deref()))
}

/// `DENNO_PF_DMA_PROBE=1`: prefill の先頭で 1 回だけ H2D の素の速度を測る
/// (診断専用・既定 off)。測るのは `[pf-probe]` の 1 行群で、数値経路は変えない。
pub const DMA_PROBE_ENV: &str = "DENNO_PF_DMA_PROBE";

// ---------------------------------------------------------------------------
// task 325: 1 回目の読み (mmap → pinned) を速くする 2 本
// ---------------------------------------------------------------------------
//
// REPORT_P321 の実測: UD-Q4_K_XL の 1 パスは 52.4 GB / 20.2 s = 2.6 GB/s、
// 2 パスは 98.5 GB / 10.9 s = 9.0 GB/s (2 枚目は page cache)。つまり速さを
// 決めているのは **冷えたページの 1 回目の読み**で、これは写しスレッドが
// mmap をフォールトしながら memcpy しているため。候補は 2 つ:
//
// - `DENNO_PF_RA_GROUPS=n`: 写しの外 (専用スレッド) から、n グループ先の
//   mmap 範囲へ `MADV_WILLNEED` を投げる。フォールトは 128 KB ずつなので、
//   先読みを別スレッドから重ねると同時に飛ぶ I/O が増える (仮説)。
// - `DENNO_PF_PREAD=1`: フォールトをやめ、`read_at` で GGUF のパートから
//   pinned へ直接読む。1 回の長さは `DENNO_PF_PREAD_KB` (既定 4096 KiB)。
//   フォールトの 1.1M 回のトラップと 128 KB の readahead を消せる (仮説)。
//
// どちらも**バイト列は変えない** (同じ範囲を同じ順に置くだけ)。既定 off。

/// [`RA_GROUPS_ENV`] の既定値 (0 = 先読みしない)。
pub const DEFAULT_RA_GROUPS: usize = 0;

/// 先読みを何グループ先まで投げるか。
pub const RA_GROUPS_ENV: &str = "DENNO_PF_RA_GROUPS";

/// [`RA_GROUPS_ENV`] の解釈 (純関数)。0..=64 に丸める。未設定・解析不能は既定。
pub fn ra_groups_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_RA_GROUPS;
    };
    match s.parse::<usize>() {
        Ok(n) => n.min(64),
        Err(_) => DEFAULT_RA_GROUPS,
    }
}

/// 先読みのグループ数 (プロセスで 1 回だけ env を読む)。
pub fn ra_groups() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| ra_groups_from_value(std::env::var(RA_GROUPS_ENV).ok().as_deref()))
}

/// `DENNO_PF_PREAD=1`: ミス expert の写しを `read_at` にする。
pub const PREAD_ENV: &str = "DENNO_PF_PREAD";

/// [`PREAD_ENV`] の既定値。**task 325 の lab t325a で採用して on**。
///
/// `mmap` をフォールトしながらの memcpy は 1 回の I/O が 128 KB (readahead) に
/// なる。GGUF のパートから `read_at` で直接 pinned へ読むと 1 回が 1-4 MB に
/// なり、同じホストで 1 回目の読みが速い (lab t325a、UD-Q4_K_XL、depth8k、
/// すべて同じ run の baseline との比較):
///
/// | 条件 | 3,826 cold (2 ラウンド) | `copy_s` (44.4 GB) | 実速度 | 14k cold |
/// | --- | --- | --- | --- | --- |
/// | b0t (baseline) | 14.74 / 12.78 s | 10.47 s | 4.2 GB/s | 29.03 s |
/// | **pread (14 本)** | **10.97 / 11.37 s** | **7.70 s** | **5.8 GB/s** | 27.12 s |
/// | pread + 32 本 | 10.28 / 11.33 s | 7.20 s | 6.2 GB/s | 26.79 s |
/// | b1 (baseline の repeat) | 14.88 / 14.72 s | 11.15 s | 4.0 GB/s | 32.98 s |
///
/// 3,826 は中央値で -19% (b0t) / -25% (b1)。14k は copy が page cache 律速
/// (9 GB/s) なので差が出ず、壁時計は -6.6% (b0t) / -18% (b1) で悪化しない。
/// md5 は c2000_cold / c8000_cold が全 COND で同一 (バイト列は変えない)。

/// [`PREAD_ENV`] の既定値。
pub const DEFAULT_PREAD: bool = true;

/// [`PREAD_ENV`] の解釈 (純関数)。未設定は既定 (on)、`0` 系だけが偽。
pub fn pread_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_PREAD,
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" => true,
            "0" | "off" | "false" | "no" => false,
            _ => DEFAULT_PREAD,
        },
    }
}

/// `read_at` 経路を使うか (プロセスで 1 回だけ env を読む)。
pub fn pread_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pread_from_value(std::env::var(PREAD_ENV).ok().as_deref()))
}

/// 1 回の `read_at` の長さ (KiB)。
pub const PREAD_KB_ENV: &str = "DENNO_PF_PREAD_KB";

/// [`PREAD_KB_ENV`] の既定値 (4 MiB)。
pub const DEFAULT_PREAD_KB: usize = 4096;

/// [`PREAD_KB_ENV`] の解釈 (純関数)。64..=65536 KiB に丸める。
pub fn pread_kb_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_PREAD_KB;
    };
    match s.parse::<usize>() {
        Ok(n) if n > 0 => n.clamp(64, 65536),
        _ => DEFAULT_PREAD_KB,
    }
}

/// 1 回の `read_at` の長さ (バイト)。
pub fn pread_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| pread_kb_from_value(std::env::var(PREAD_KB_ENV).ok().as_deref()) << 10)
}

/// `len` バイトを `blk` ずつに切った (転送元の中のオフセット, 長さ) の列。
///
/// 端数は最後の 1 個に寄せる (合計は必ず `len`)。切る位置が違うだけで、
/// 書くバイト列は 1 回読みと同一。
pub fn pread_blocks(len: usize, blk: usize) -> Vec<(usize, usize)> {
    let blk = blk.max(1);
    let mut out = Vec::with_capacity(len.div_ceil(blk));
    let mut at = 0usize;
    while at < len {
        let n = blk.min(len - at);
        out.push((at, n));
        at += n;
    }
    out
}

/// `DENNO_PF_UP_SPLIT_KB`: 上げスレッドの写しと DMA を何 KiB ずつ重ねるか
/// (**既定 8192 = 8 MiB**。`0` で従来の「全部写してから 1 回 DMA」)。
///
/// # なぜ要るか (task 481 の台帳)
///
/// 3,826 cold の `[exp-pf-stream]` は `up_MB=22455.5` に対して `copy_s=1.09`
/// (20.6 GB/s) で、これが壁 2,119 ms の 51%。中身は 3 往復:
///
/// 1. page cache を読む (22.4 GB)
/// 2. write-combined のバウンスへ書く (22.4 GB)
/// 3. DMA がバウンスを読み直す (22.4 GB。読み出しは 1 と帯域を食い合う)
///
/// 小さなバウンス 2 枚を交互に使い、写し(i+1) を DMA(i) と重ねると、DMA が
/// 読むのは**ついさっき書いた行**になる。キャッシュコヒーレントな PCIe では
/// その行がまだ L3 (9800X3D は 96 MB) にあれば DRAM を読み直さない。
///
/// # 実測 (lab t481b、`model=flash-next-UD-IQ3_XXS`、depth8k)
///
/// | | base2 (従来) | sp8 (8 MiB) | sp16 (16 MiB) |
/// | --- | --- | --- | --- |
/// | 3,826 cold | 2.12 s (copy 1.09) | **1.96 s** (copy 0.80) | 1.96 s |
/// | 14k cold | 5.18 s (copy 1.32) | **5.03 s** (copy 0.94) | 5.09 s |
/// | decode | 11.1 ms/tok | 11.2 | 11.2 |
///
/// copy_s が -27% / -29%、壁が -7.5% / -2.9%。md5 は全コンテキスト SAME。
///
/// 値は 1 バイトも変えない (デバイスへ渡るバイト列は 1 回写しと同一。
/// `pf_stream_tests` の分割一致テストが同じ契約を固定している)。
pub const UP_SPLIT_KB_ENV: &str = "DENNO_PF_UP_SPLIT_KB";

/// [`UP_SPLIT_KB_ENV`] の既定値 (8 MiB)。
///
/// lab t481b (3,826 / 14k cold、`model=flash-next-UD-IQ3_XXS`): 8 MiB と 16 MiB は
/// 同じだけ効いた (3,826 2.12 -> 1.96 / 1.96、14k 5.18 -> 5.03 / 5.09) ので、
/// L3 (96 MB) に余裕のある 8 MiB を既定にした。`DENNO_PF_UP_SPLIT_KB=0` で
/// 従来の「全部写してから 1 回 DMA」へ戻せる。
pub const DEFAULT_UP_SPLIT_KB: usize = 8192;

/// [`UP_SPLIT_KB_ENV`] の解釈 (純関数)。未設定 / 空 / 解析不能は既定 (8 MiB)、
/// 明示的な `0` / `off` は分割しない、正の値は 64..=65536 KiB に丸める。
pub fn up_split_kb_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_UP_SPLIT_KB;
    };
    match s.parse::<usize>() {
        Ok(0) => 0,
        Ok(n) => n.clamp(64, 65536),
        Err(_) => match s.to_ascii_lowercase().as_str() {
            "off" | "false" | "no" => 0,
            _ => DEFAULT_UP_SPLIT_KB,
        },
    }
}

/// 分割の大きさ (バイト、0 = 分割しない)。プロセスで 1 回だけ env を読む。
pub fn up_split_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        up_split_kb_from_value(std::env::var(UP_SPLIT_KB_ENV).ok().as_deref()) << 10
    })
}

/// 分割 DMA を積んだ回数 (診断。0 ならゲートが発火していない)。
static UP_SPLIT_CHUNKS: AtomicU64 = AtomicU64::new(0);

/// 分割 DMA を 1 回積んだ。
pub fn note_up_split() {
    UP_SPLIT_CHUNKS.fetch_add(1, Ordering::Relaxed);
}

/// 分割 DMA を積んだ回数 (リクエスト境界で 0 に戻す)。
pub fn up_split_chunks() -> u64 {
    UP_SPLIT_CHUNKS.load(Ordering::Relaxed)
}

/// `DENNO_PF_PREAD_THREADS`: pread の写し専用のスレッド数 (既定 **32**、
/// `0` で従来の共有プール = 14 本)。`read_at` は同期なので、デバイスが
/// レイテンシ律速なら本数を増やすと同時に飛ぶ I/O が増える。
pub const PREAD_THREADS_ENV: &str = "DENNO_PF_PREAD_THREADS";

/// [`PREAD_THREADS_ENV`] の既定値 (32)。task 358 の lab t358b で採った。
///
/// task 325 は 14 (共有プール) と 32 を比べて「壁時計は動かない」として
/// 0 のまま残した (当時の 3,826 は 10.3-11.4 s、copy 7.2-7.7 s)。その後の
/// task 353/357 で PLE と層優先の前処理が速くなり、**prefill の壁時計に
/// 占める expert の pread (44.4 GB / 98.2 GB) の比率が上がった**ので、
/// 同じ 32 本をもう一度測った:
///
/// lab t358b (UD-Q4_K_XL、depth8k、`[exp-pf-stream]` の copy_s と壁時計):
///
/// | cond | 3,826 copy | 3,826 wall | 14k copy | 14k wall |
/// | --- | --- | --- | --- | --- |
/// | x0 (14 本) | 10.37 / 9.26 s | 12.81 / 11.54 s | 14.30 s | 30.25 s |
/// | **32 本** | **8.61 / 7.74 s** | **10.83 / 9.87 s** | **11.84 s** | **25.71 s** |
/// | 48 本 | 7.41 / 8.58 s | 9.84 / 11.04 s | 14.11 s | 34.65 s |
/// | b1 (14 本) | 9.96 / 14.12 s | 12.12 / 18.28 s | 14.02 s | 31.18 s |
///
/// 3,826 は中央値 12.12 -> 10.35 s (-14.6%)、14k は 30.25 -> 25.71 s (-15.0%)。
/// 48 本は 3,826 で最速だが 14k で負け (ホストの外乱) なので 32 を採る。
/// decode は変わらない (a1/a2 の生秒 2.3-2.6 s)。
pub const DEFAULT_PREAD_THREADS: usize = 32;

/// [`PREAD_THREADS_ENV`] の解釈 (純関数)。0..=64 に丸める。`0` = 共有プール。
pub fn pread_threads_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_PREAD_THREADS;
    };
    match s.parse::<usize>() {
        Ok(n) => n.min(64),
        Err(_) => DEFAULT_PREAD_THREADS,
    }
}

/// pread 専用プールのスレッド数 (プロセスで 1 回だけ env を読む)。
pub fn pread_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pread_threads_from_value(std::env::var(PREAD_THREADS_ENV).ok().as_deref())
    })
}

/// `DENNO_PF_STREAM_PREAD`: task 349。IQ 系のストリームでも `read_at` を使う。
///
/// task 325 の pread は「pinned バウンスが write-combined でない」スロットだけ
/// で発火する (`pf_stage_plan` の `!pin_wc`)。IQ 系は
/// [`stream_wc_for`] の既定が WC なので、**IQ3_S では pread が一度も
/// 発火していなかった** (lab t343c の `[exp-pf-stream]` に `pread_` の欄が無い)。
///
/// このゲートを on にすると、ストリームのスロットだけ pin を**非 WC** に
/// 落として pread を発火させる (`DENNO_PF_STREAM_WC` より優先する)。
/// WC へ `read(2)` すると極端に遅いため (task 325)、WC のままでは使えない。
/// 既定 off: 非 WC のバウンスはホスト側の memcpy が遅いので、cold で勝つかを
/// lab で測ってから決める。
pub const STREAM_PREAD_ENV: &str = "DENNO_PF_STREAM_PREAD";

/// [`STREAM_PREAD_ENV`] の既定値 (off)。
pub const DEFAULT_STREAM_PREAD: bool = false;

/// [`STREAM_PREAD_ENV`] の解釈 (純関数)。未設定は既定、`1` 系だけが真。
pub fn stream_pread_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_STREAM_PREAD,
        Some(s) => matches!(
            s.to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
    }
}

/// ストリームの pread を使うか (プロセスで 1 回だけ env を読む)。
pub fn stream_pread() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stream_pread_from_value(std::env::var(STREAM_PREAD_ENV).ok().as_deref()))
}

/// `DENNO_PF_PLE_PIPE`: task 349。プロンプト全体の PLE 行 gather を 1 本の
/// 生産スレッドで先に流す (パイプライン)。
///
/// 既存の `DENNO_PF_PLE_PRE` は **次の 1 チャンク**しか先読みしない。冷えた
/// page cache では 14k の gather が 2.7 s かかる (`ple.rs` の実測) のに、1 チャンクの
/// 層ループは 0.46 s しかないので、隠れずに `other` へ出る (lab t349a の
/// 14k cold: `other` 2,608 ms / 露出 2,334 ms、base は 830 ms)。
///
/// ここでは gather を 1 本のスレッドで **チャンク順に全部**作り、`sync_channel` で
/// 1 チャンクずつ渡す。層ループ (GPU) の裏でディスクのランダム読みが進むので、
/// 2.7 s のうち層ループに隠れる分が増える。並列度 (`DENNO_PLE_PAR`) も
/// 表の置き場所 (RAM 常駐) も変えない。
///
/// task 353: 層優先 (`forward_prefill_batch_resident_lm`) では gather が
/// **層ループの前に全チャンク直列**なので、パイプの生産スレッドは使わない。
/// 代わりに gather を 32 本の専用プール (`ple_par_parts` の上限 32) で引く。
/// 冷えた page cache の実測は [`DEFAULT_PLE_PIPE`] の doc を見ること。
pub const PLE_PIPE_ENV: &str = "DENNO_PF_PLE_PIPE";

/// [`PLE_PIPE_ENV`] の既定値。**task 353 の lab t353a で採ったので on**。
///
/// lab t353a (depth8k、`DENNO_START_DROP_CACHE=1` で全 COND を冷えた状態に揃え、
/// 同じ run の baseline b1/b2 = 8.85/8.82 s と 9.29/9.21 s):
///
/// | COND | 3,826 cold | 14k cold | `[pre-kprof]` ple (3.8k / 14k) | decode |
/// | --- | --- | --- | --- | --- |
/// | b1 (baseline) | 8.85 s | 9.29 s | 771 / 1,851 ms | 16.6 |
/// | pl (`DENNO_PF_PLE_PIPE=1`) | **8.41 s (-5.0%)** | **8.06 s (-13.2%)** | 328 / 681 ms | 16.7 |
/// | b2 (baseline) | 8.82 s | 9.24 s | — | 16.6 |
///
/// md5 は pl の 8 本すべて SAME (`a91e17da` / `5a8f5488` / `e2292eb6` / `561c6da8`)、
/// errors 0、text sane。効いているのは**層優先の先頭 gather の並列度**
/// (32 本、`ple_par_parts`) で、パイプ側 (`forward_prefill_batch_resident`) は
/// このモデルでは層優先経路が使われるため主因ではない (lab の ple 時間は
/// 1,851 → 681 ms に落ちた)。decode は 0.1 ms/tok の刻み内。
pub const DEFAULT_PLE_PIPE: bool = true;

/// [`PLE_PIPE_ENV`] の解釈 (純関数)。未設定は既定 (on)、`0` 系だけが偽。
pub fn ple_pipe_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_PLE_PIPE,
        Some(s) => matches!(s.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"),
    }
}

/// PLE のパイプライン先読みを使うか (プロセスで 1 回だけ env を読む)。
pub fn ple_pipe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ple_pipe_from_value(std::env::var(PLE_PIPE_ENV).ok().as_deref()))
}

/// `DENNO_PF_PLE_THREADS`: task 357。層優先 prefill の PLE gather を引く専用
/// プールの本数 (= 同時に飛ぶ 90 B のランダム pread の数)。
///
/// task 353 は 32 本で固定していた (`qwen4exp.rs` の `PLE_PIPE_THREADS`)。1 行
/// 90 B の pread は **レイテンシ律速**で、14k cold の gather は 32 本でも
/// 681 ms かかる (lab t353a の pl、`[pre-kprof]` ple=681.3、166,728 reads =
/// 245k IOPS、1 本あたり 79 us)。ホストは 8C/16T だが pread はカーネル内で
/// 待つので、本数を増やすとデバイスのキューに同時に乗る読みが増える
/// (CPU は遊んだまま)。task 357 の lab t357a/t357b で 64/96/128 を測り、
/// task 419 の lab t419b で 128 本も測り、そちらが 8% 速かったので
/// **既定 128** にした ([`DEFAULT_PLE_THREADS`])。
pub const PLE_THREADS_ENV: &str = "DENNO_PF_PLE_THREADS";

/// [`PLE_THREADS_ENV`] の既定値 (**128 本**)。
///
/// task 357 が 96 本を既定にした (14k cold の gather は 32 本で 680 ms、
/// 64 本で 488 ms、96 本で 425 ms)。task 419 の lab t419b で 128 本を
/// 2 COND x 2 ラウンド測り、**ple 段が 14k 415.3 / 414.2 -> 381.4 / 382.9 ms
/// (-8.1%)、3,826 215.4 / 213.9 -> 197.4 / 196.9 ms (-8.2%)** と両方で
/// 同じだけ下がったので既定を 128 にした。壁への寄与は 14k で -0.4%
/// (7.02 -> 6.99 s)、3,826 は 0 (壁は expert の読み)、md5 は全 COND SAME。
/// pread はカーネル内で待つので、128 本でも CPU は遊んでいる。
pub const DEFAULT_PLE_THREADS: usize = 128;

/// [`PLE_THREADS_ENV`] の解釈 (純関数)。1..=128 に丸める。未設定・解析不能は既定。
///
/// task 423: 上限を 512 に上げて 256 本を lab t423a で測ったが、`ple_par_parts`
/// (ple.rs) 側にも 128 の丸めがあり、**ゲートを上げても 128 本のまま**だった
/// (ple 段 239.8 -> 241.2 ms で差なし)。ここは 128 に戻す。
pub fn ple_threads_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_PLE_THREADS;
    };
    match s.parse::<usize>() {
        Ok(n) if n > 0 => n.min(128),
        _ => DEFAULT_PLE_THREADS,
    }
}

/// PLE gather の専用プールの本数 (プロセスで 1 回だけ env を読む)。
pub fn ple_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| ple_threads_from_value(std::env::var(PLE_THREADS_ENV).ok().as_deref()))
}

/// `DENNO_PF_EMBED_PAR`: 層優先 prefill の**埋め込み** (`token_embd` の行デコード)
/// を引く本数 (task 415)。
///
/// 層優先の prefill は層ループの前に全トークンぶんの埋め込みを引く。ここは
/// **1 トークン 1 回の直列ループ**で、1 行は 2,100 B (Q6_K) のランダム読み。
/// 表は 317 MB あるのでページキャッシュから落ちていることがあり、そのときは
/// 1 本 100 us のディスク待ちがトークン数だけ並ぶ (14k で 1.4 s 相当)。
/// 行は互いに独立なので N 本で引ける (値はビット一致)。
///
/// 既定 **16** (lab t415b で採った。ホストは 8C/16T)。
///
/// # 実測 (task 415、lab t415b、depth8k、PROFILE=1、既定モデル)
///
/// 14k cold の `[pre-kprof] embed` (同じ run): b1 (直列) **165.9 ms** ->
/// 16 本 **24.4** / 32 本 **27.3** = **-85%**。3,826 cold の壁は
/// b1 2,338.2 -> 16 本 2,282.5 ms (b2 2,306.2、直列)。decode 13.3-13.6、
/// md5 は全 COND・全文脈 SAME、errors 0。
pub const EMBED_PAR_ENV: &str = "DENNO_PF_EMBED_PAR";

/// [`EMBED_PAR_ENV`] の既定値 (16 本)。
pub const DEFAULT_EMBED_PAR: usize = 16;

/// [`EMBED_PAR_ENV`] の解釈 (純関数)。未設定・空・解析不能は既定、1..=64 に丸める。
/// `0` と `1` は従来の直列。
pub fn embed_par_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_EMBED_PAR;
    };
    match s.parse::<usize>() {
        Ok(0) => 0,
        Ok(n) => n.min(64),
        Err(_) => DEFAULT_EMBED_PAR,
    }
}

/// 埋め込みを引く本数 (プロセスで 1 回だけ env を読む)。
pub fn embed_par() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| embed_par_from_value(std::env::var(EMBED_PAR_ENV).ok().as_deref()))
}

/// `DENNO_PF_EMBED_OVL`: PLE の gather を背景スレッドへ出し、その裏で埋め込みを
/// 引く (task 415)。
///
/// どちらもホストの**ファイル読み**で、待ちはカーネル内 (pread) かページフォルト。
/// 埋め込み (直列) と gather (96 本) は同じディスクを別々に叩くので、重ねると
/// 短いほうが丸ごと隠れる。読む行・綴じ方は変わらない (値はビット一致)。
///
/// 既定 **on** (lab t415b で採った: ovl は 14k で 5.63 s 対 b1 6.53 s、b2 5.70 s)。
/// `DENNO_PF_EMBED_PAR` が 1 以下のときは効かない (重ねる相手が無い)。
pub const EMBED_OVL_ENV: &str = "DENNO_PF_EMBED_OVL";

/// [`EMBED_OVL_ENV`] の既定値 (on)。
pub const DEFAULT_EMBED_OVL: bool = true;

/// [`EMBED_OVL_ENV`] の解釈 (純関数)。未設定・空は既定 (on)、
/// `0` 系の明示値だけ off。
pub fn embed_ovl_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_EMBED_OVL,
        Some(s) => !matches!(s.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
    }
}

/// PLE gather と埋め込みを重ねるか (プロセスで 1 回だけ env を読む)。
pub fn embed_ovl() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| embed_ovl_from_value(std::env::var(EMBED_OVL_ENV).ok().as_deref()))
}

/// `DENNO_PF_PLE_LJOIN`: 背景に出す PLE の gather を **層ループの PLE 層まで
/// 待たない** (task 423)。
///
/// 層優先 prefill は埋め込みの裏で gather を走らせる (`DENNO_PF_EMBED_OVL`) が、
/// 層ループへ入る**前に join** していた。gather は 90 B の pread が IOPS 律速で
/// 14k で約 380 ms / 3,826 で約 200 ms かかり、その待ちが丸ごと GPU の遊びになる。
/// PLE 層は層 1 で、行を使うのはその層の入口だけなので、join をそこへ移すと
/// 層 0 の 1 層ぶん (14k で約 140 ms / 3,826 で約 170 ms) が gather と重なる。
/// 受け渡しは**チャンクごと** (背景スレッドがチャンク順に送り、PLE 層が 1 チャンク
/// ずつ受ける) なので、層 1 の後半のチャンクは更に待たない。
/// 読む行・綴じ方は変えない (値はビット一致)。
///
/// # 実測 (lab t423a / t423b / t423c、既定モデル、depth8k、全 COND `DROP_CACHE=1`)
///
/// `[pre-kprof]` の **ple 段 (gather の待ち)**: 14k 378.8 / 377.8 -> 239.8 / 241.2 ms
/// (層の入口でまとめて待つ版、t423a/t423b)、**チャンクごとに受ける版 (t423c) で
/// 380.4 / 375.1 -> 37.1 / 21.8 ms**、3,826 200.4 / 200.9 -> **0.0 / 0.0 ms**。
/// 壁は 14k が t423c の交互の対で **7.20 / 6.97 -> 6.94 / 6.68 s (-3.6 / -4.2%)**、
/// 3,826 は 8.37 / 8.37 -> 8.30 / 8.36 (-0.8 / -0.2%。expert の読みで詰まっており、
/// gather を層ループへ移すとそのぶん層が伸びる)、decode 12.0-12.2 ms/tok 不変、
/// md5 は全 COND・全 6 文脈で SAME、errors 0、text sane。**既定 on**。
pub const PLE_LJOIN_ENV: &str = "DENNO_PF_PLE_LJOIN";

/// [`PLE_LJOIN_ENV`] の既定値 (on)。
pub const DEFAULT_PLE_LJOIN: bool = true;

/// [`PLE_LJOIN_ENV`] の解釈 (純関数)。未設定・空・解析不能は既定 (on)、
/// `0` 系の明示値だけ off。
pub fn ple_ljoin_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_PLE_LJOIN,
        Some(s) => !matches!(s.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
    }
}

/// PLE の gather を層ループの PLE 層まで待たないか (プロセスで 1 回だけ env を読む)。
pub fn ple_ljoin() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ple_ljoin_from_value(std::env::var(PLE_LJOIN_ENV).ok().as_deref()))
}

/// パート番号ごとの読み口 (pread 経路だけが使う)。
///
/// モデルの構築時に 1 回だけ開く。開けなかったパートは `None` で、その
/// パートを含む区間は従来どおり memcpy へ落ちる (fail open)。
static PARTS: std::sync::OnceLock<Vec<Option<std::sync::Arc<std::fs::File>>>> =
    std::sync::OnceLock::new();

/// GGUF のパートを開いて覚える (2 回目以降は捨てる)。
///
/// **添字はパート番号そのもの**。開けなかったパートは `None` を残す
/// (詰めると別のパートを読む事故になる)。
pub fn install_parts(paths: Vec<Option<std::path::PathBuf>>) {
    let files: Vec<Option<std::sync::Arc<std::fs::File>>> = paths
        .iter()
        .map(|p| {
            p.as_ref()
                .and_then(|p| std::fs::File::open(p).ok())
                .map(std::sync::Arc::new)
        })
        .collect();
    let opened = files.iter().filter(|f| f.is_some()).count();
    if PARTS.set(files).is_ok() && pread_on() {
        eprintln!("[pf-pread] 読み口: {opened} パートを開いた (DENNO_PF_PREAD=1)");
    }
}

/// パートが 1 本でも開けているか (診断用)。
pub fn parts_open() -> usize {
    PARTS
        .get()
        .map(|v| v.iter().filter(|f| f.is_some()).count())
        .unwrap_or(0)
}

/// `part` の `off` から `dst` (長さ `len`) を読み切る。成功したら true。
///
/// 失敗 (開けていない / 短読み / エラー) は false を返し、呼び出し側は
/// memcpy へ落ちる。**部分的な書き込みは残る**が、そのあと memcpy が同じ
/// 範囲を上書きするので中身は正しい。
///
/// # Safety
/// `dst` は `len` バイト書ける領域を指すこと。区間は呼び出し側が重複させない。
pub unsafe fn read_part_ptr(part: u32, off: u64, dst: *mut u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    let Some(Some(f)) = PARTS.get().and_then(|v| v.get(part as usize)) else {
        PREAD_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    };
    let buf = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut done = 0usize;
        while done < len {
            match f.read_at(&mut buf[done..], off + done as u64) {
                Ok(0) => {
                    PREAD_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
                Ok(n) => done += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    PREAD_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (buf, off);
        PREAD_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    PREAD_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    PREAD_BYTES.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
    true
}

static PREAD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PREAD_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PREAD_FAIL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `[exp-pf-stream]` の末尾へ足す pread の台帳 (発火していなければ空)。
pub fn pread_stats() -> String {
    let calls = PREAD_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let fail = PREAD_FAIL.load(std::sync::atomic::Ordering::Relaxed);
    if calls == 0 && fail == 0 {
        return String::new();
    }
    let mb = PREAD_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    format!(" pread_MB={mb:.1} pread_calls={calls} pread_fail={fail}")
}

/// pread の台帳を 0 に戻す。
pub fn pread_stats_reset() {
    for c in [&PREAD_BYTES, &PREAD_CALLS, &PREAD_FAIL] {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// 先読みの台帳 (グループ数 / madvise したバイト / その時間)。
static RA_JOBS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RA_GROUP_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RA_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 先読み 1 グループぶんを積む (診断用)。
pub fn note_ra_job(bytes: usize, ns: u64) {
    RA_JOBS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    RA_GROUP_BYTES.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
    RA_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
}

/// `[exp-pf-stream]` の末尾へ足す先読みの台帳 (発火していなければ空)。
///
/// task 325 の `DENNO_PF_RA_GROUPS` (ページキャッシュへ投げるだけ) 用。
/// task 474 の RAM の置き場は [`rc_stats`] が別に出す。
pub fn ra_head_stats() -> String {
    let jobs = RA_JOBS.load(std::sync::atomic::Ordering::Relaxed);
    if jobs == 0 {
        return String::new();
    }
    let mb = RA_GROUP_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let s = RA_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    format!(" ra_groups={jobs} ra_MB={mb:.1} ra_s={s:.2}")
}

/// 先読みの台帳を 0 に戻す。
pub fn ra_head_stats_reset() {
    for c in [&RA_JOBS, &RA_GROUP_BYTES, &RA_NS] {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}


/// `DENNO_PF_DMA_PROBE` の解釈 (純関数)。**明示的な真値だけ on**。
pub fn dma_probe_from_value(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// 素の H2D を測るか (プロセスで 1 回だけ env を読む)。
pub fn dma_probe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dma_probe_from_value(std::env::var(DMA_PROBE_ENV).ok().as_deref()))
}

/// 型 → カーネルのスロット (decode の `kernels_exp_iq` と同じ順)。
pub fn slot(ty: GgmlType) -> Option<usize> {
    if let Some(s) = crate::model::exp_iq::slot(ty) {
        return Some(s);
    }
    // task 309: K 系 (prefill のバッチ用。`DENNO_PF_EXP_K` で有効になる)。
    Some(match ty {
        GgmlType::Q4K => K_SLOT0,
        GgmlType::Q5K => K_SLOT0 + 1,
        GgmlType::Q6K => K_SLOT0 + 2,
        GgmlType::Q5_1 => K_SLOT0 + 3,
        GgmlType::Q8_0 => K_SLOT0 + 4,
        _ => return None,
    })
}

/// この型を m 行カーネルで読めるか。
pub fn supported(ty: GgmlType) -> bool {
    slot(ty).is_some()
}

/// 1 行あたりのバイト数。ブロックサイズで割り切れない k は `None`。
///
/// decode の [`crate::model::exp_iq::row_bytes`] と同じ式 (型の許可制は持たない)。
pub fn row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if bs == 0 || ts == 0 || k == 0 || !k.is_multiple_of(bs) {
        return None;
    }
    Some((k / bs) * ts)
}

/// カーネル名 (`kernels_exp_iq_pf.cu` の `.visible .entry`)。
pub fn kernel_name(slot: usize) -> Option<&'static str> {
    Some(match slot {
        0 => "dno_exp_pf_iq2s",
        1 => "dno_exp_pf_iq3xxs",
        2 => "dno_exp_pf_iq3s",
        3 => "dno_exp_pf_iq4xs",
        4 => "dno_exp_pf_iq4nl",
        5 => "dno_exp_pf_q2_0",
        // task 309: K 系 (`kernels_exp_k_pf.cu`)。PTX は別モジュール。
        6 => "dno_exp_k_pf_q4k",
        7 => "dno_exp_k_pf_q5k",
        8 => "dno_exp_k_pf_q6k",
        9 => "dno_exp_k_pf_q5_1",
        10 => "dno_exp_k_pf_q8_0",
        _ => return None,
    })
}

/// silu カーネルの名前 (このモジュール専用。decode の silu とは別モジュール)。
pub const SILU_KERNEL: &str = "dno_exp_pf_silu_mul";

/// ヒット行の加重和カーネルの名前 (`od[t*e+i] = sum_j w[t*k+j] * h[row_j*e+i]`)。
///
/// これを GPU で回すことで、1 層あたりホストへ戻す量が「全行 × e」から
/// 「nt × e」になる (42 MB → 5.2 MB)。
pub const ACCUM_KERNEL: &str = "dno_exp_pf_accum";

/// 起動形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PfLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

/// 層一括 m 行 GEMV の起動形 (純関数)。`None` = この形では起動しない。
///
/// block = 128 (4 warp)。1 block が受ける出力行数は型で違う ([`tile_of`])。
/// `n_out` がタイルで割り切れない形は起動しない (端数の行は CPU で計算する。
/// 実形は 640 と 2,560 でどちらも割り切れる)。grid.y = segment 数 (上限 65535)。
pub fn launch_plan(slot: usize, nseg: usize, n_out: usize) -> Option<PfLaunch> {
    if nseg == 0 || n_out == 0 || nseg > 65535 {
        return None;
    }
    let tile = tile_of(slot);
    if tile == 0 || !n_out.is_multiple_of(tile) {
        return None;
    }
    let tiles = n_out / tile;
    if tiles > 65535 {
        return None;
    }
    Some(PfLaunch {
        grid: (tiles as u32, nseg as u32, 1),
        block: ((32 * WARPS) as u32, 1, 1),
    })
}

// ---------------------------------------------------------------------------
// 層 1 回ぶんの segment 表 (GPU 無しで検証できる部分)
// ---------------------------------------------------------------------------

/// 1 層 1 チャンクぶんの segment 表。
///
/// `seg` は 4 語 × ヒット expert 数、`rows` は行数ぶんのトークン番号。
/// 行は expert index 昇順、expert の中はトークン昇順 (`by_expert` の順)。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PfPlan {
    /// segment 表 (4 語 × nseg)
    pub seg: Vec<u32>,
    /// 行 → トークン番号
    pub rows: Vec<u32>,
    /// ヒットした expert の番号 (昇順)
    pub experts: Vec<usize>,
    /// 行の総数
    pub total_rows: usize,
}

impl PfPlan {
    /// segment 数。
    pub fn nseg(&self) -> usize {
        self.experts.len()
    }

    /// 1 行ぶんのトークン番号。
    pub fn token_of(&self, row: usize) -> u32 {
        self.rows[row]
    }
}

/// ヒットした expert だけの segment 表を作る (`pick[ei]` が true の expert)。
///
/// 行の並びは `by_expert` のまま (トークン昇順)。出力も同じ順に詰めるので、
/// `rows` の添字がそのまま出力行になる。
pub fn build_plan(by_expert: &[Vec<(usize, usize)>], pick: &[bool]) -> PfPlan {
    let mut plan = PfPlan::default();
    let mut off = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() || !pick.get(ei).copied().unwrap_or(false) {
            continue;
        }
        plan.seg.push(picks.len() as u32);
        plan.seg.push(off as u32);
        plan.seg.push(0);
        plan.seg.push(0);
        plan.experts.push(ei);
        for &(t, _) in picks {
            plan.rows.push(t as u32);
        }
        off += picks.len();
    }
    plan.total_rows = off;
    plan
}

/// down 行列用の行表 (`act` は行が詰まっているので 0,1,2,...)。
pub fn identity_rows(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// task 162: 行の加重和カーネルへ渡す行表 `prow[t * k + j]` = `h` の行番号。
///
/// `plan` は [`build_plan`] の結果。`pick` が false の expert は CPU に残るので、
/// その (t, j) は `u32::MAX` のまま返す。呼び出し側は重みを 0 にして
/// (`row_table_zero_slots`) 足し込みから外す。範囲外・重複があれば `None`。
pub fn row_table_partial(
    by_expert: &[Vec<(usize, usize)>],
    plan: &PfPlan,
    k: usize,
    nt: usize,
) -> Option<Vec<u32>> {
    if k == 0 || nt == 0 || plan.seg.len() != plan.experts.len() * 4 {
        return None;
    }
    let mut prow = vec![u32::MAX; nt * k];
    let mut seg = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        // plan は expert 昇順。i 番目の非空 expert が plan に無ければ CPU 側。
        if plan.experts.get(seg).copied() != Some(ei) {
            continue;
        }
        let off = plan.seg[seg * 4 + 1] as usize;
        for (idx, &(t, j)) in picks.iter().enumerate() {
            if t >= nt || j >= k {
                return None;
            }
            let wi = t * k + j;
            if prow[wi] != u32::MAX {
                return None;
            }
            prow[wi] = (off + idx) as u32;
        }
        seg += 1;
    }
    if seg != plan.experts.len() {
        return None;
    }
    Some(prow)
}

/// [`row_table_partial`] の**全行が載っている**版 (穴が 1 つでもあれば `None`)。
pub fn row_table_full(
    by_expert: &[Vec<(usize, usize)>],
    plan: &PfPlan,
    k: usize,
    nt: usize,
) -> Option<Vec<u32>> {
    let prow = row_table_partial(by_expert, plan, k, nt)?;
    if prow.contains(&u32::MAX) {
        return None;
    }
    Some(prow)
}

/// task 233: 行の加重和の表に「共有 expert の行」を 1 列足す (純関数)。
///
/// `prow` / `wrow` は `ntok * k` (行は expert 昇順、加重和の項の順そのもの)。
/// 戻り値は `ntok * (k + 1)` で、各トークンの**最後の 1 項**が共有 expert の
/// 行 (`shared_row0 + t`) と係数 `sh_w[t]` になる。共有 expert の出力を MoE の
/// 作業域の末尾の行へ書いておけば、加重和のカーネル 1 回で routed と共有が
/// 同じ出力に載る (ホストへ戻さない)。
pub fn row_table_with_shared(
    prow: &[u32],
    wrow: &[f32],
    ntok: usize,
    k: usize,
    shared_row0: usize,
    sh_w: &[f32],
) -> Option<(Vec<u32>, Vec<f32>)> {
    if k == 0 || ntok == 0 || prow.len() != ntok * k || wrow.len() < ntok * k {
        return None;
    }
    if sh_w.len() < ntok || shared_row0 > u32::MAX as usize {
        return None;
    }
    let kk = k + 1;
    let mut pr = vec![0u32; ntok * kk];
    let mut wr = vec![0f32; ntok * kk];
    for t in 0..ntok {
        pr[t * kk..t * kk + k].copy_from_slice(&prow[t * k..(t + 1) * k]);
        wr[t * kk..t * kk + k].copy_from_slice(&wrow[t * k..(t + 1) * k]);
        pr[t * kk + k] = (shared_row0 + t) as u32;
        wr[t * kk + k] = sh_w[t];
    }
    Some((pr, wr))
}

/// task 162: 載っていない (t, j) を「重み 0・行 0」にして和から外す。
///
/// `prow` の穴を 0 で埋め、`weights` の同じ位置を 0.0 にする。0 × 有限値 = 0
/// なので和は変わらない (CPU 側の寄与は呼び出し側が足す)。戻り値は穴の数。
pub fn row_table_zero_slots(prow: &mut [u32], weights: &mut [f32]) -> usize {
    let mut n = 0usize;
    for (wi, p) in prow.iter_mut().enumerate() {
        if *p == u32::MAX {
            *p = 0;
            weights[wi] = 0.0;
            n += 1;
        }
    }
    n
}

/// task 162: ステージングへ 1 回に上げる範囲 (plan 添字の `[start, end)`) を切る。
///
/// `needs[i]` は plan 順の expert の必要バイト数 (常駐は 0)。1 範囲の合計が
/// `cap` を超えず、本数が `max_group` を超えないように前から詰める。
/// 1 本だけで `cap` を超える expert があれば `None` (この層はストリーミング
/// しない = 呼び出し側が従来経路へ落ちる)。
pub fn group_bounds(needs: &[usize], cap: usize, max_group: usize) -> Option<Vec<(usize, usize)>> {
    if needs.is_empty() || cap == 0 {
        return None;
    }
    let max_group = max_group.max(1);
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    let mut sum = 0usize;
    for (i, &need) in needs.iter().enumerate() {
        if need > cap {
            return None;
        }
        if i > start && (sum + need > cap || i - start >= max_group) {
            out.push((start, i));
            start = i;
            sum = 0;
        }
        sum += need;
    }
    if start < needs.len() {
        out.push((start, needs.len()));
    }
    Some(out)
}

/// task 324: 層の**先頭グループだけ** `first` 本に絞って切る。
///
/// 層の入口では、前の層の MoE が終わってからでないと次の層の計画 (router) が
/// 立たないので、**最初のグループの転送がまるごと露出する** (lab t324b の
/// `[q4ledger-pf] experts` の exposed 849 ms のうち、g64 で 1 層 6 ms 級 x 48 層
/// = 約 290 ms が段差)。先頭だけ小さく切ると段差がその比で縮む。
///
/// `first == 0` か 1 グループしか作れないときは [`group_bounds`] と同じ結果になる
/// (既定 off で従来と 1 バイトも変わらない)。担当は重複なし・取りこぼしなし。
pub fn group_bounds_first(
    needs: &[usize],
    cap: usize,
    max_group: usize,
    first: usize,
) -> Option<Vec<(usize, usize)>> {
    let max_group = max_group.max(1);
    let first = first.clamp(0, max_group);
    if first == 0 || needs.len() <= 1 {
        return group_bounds(needs, cap, max_group);
    }
    if needs.is_empty() || cap == 0 || needs.iter().any(|&n| n > cap) {
        // 既存の判定 (None / 空) に任せる。
        return group_bounds(needs, cap, max_group);
    }
    let n = needs.len().min(first);
    let mut sum = 0usize;
    let mut end = 0usize;
    for (i, &need) in needs.iter().enumerate().take(n) {
        if i > 0 && sum + need > cap {
            break;
        }
        sum += need;
        end = i + 1;
    }
    if end == 0 {
        return group_bounds(needs, cap, max_group);
    }
    let mut out: Vec<(usize, usize)> = vec![(0usize, end)];
    if end < needs.len() {
        let rest = group_bounds(&needs[end..], cap, max_group)?;
        out.extend(rest.into_iter().map(|(a, b)| (a + end, b + end)));
    }
    Some(out)
}

/// task 242: グループ局所の g/u/act に要る行数 (グループの最大行数)。
///
/// `seg` は 4 語 × segment (`[行数, 行の先頭, 0, 0]`)、`bounds` は
/// [`group_bounds`] の結果 (segment の index 範囲)。層優先 prefill は
/// カーネルの出力ポインタを `先頭 - グループ先頭行 * n_out * 4` で渡すので、
/// g/u/act は**この行数**だけ持てばよい (14k で 0.86 GB → 0.11 GB)。
///
/// 範囲外・空なら `None` (呼び出し側は従来の全体サイズへ落ちる)。
pub fn max_group_rows(seg: &[u32], bounds: &[(usize, usize)]) -> Option<usize> {
    if seg.len() % 4 != 0 || bounds.is_empty() {
        return None;
    }
    let nseg = seg.len() / 4;
    let mut mx = 0usize;
    for &(a, b) in bounds {
        if a >= b || b > nseg {
            return None;
        }
        let last = b - 1;
        let start = seg[a * 4 + 1] as usize;
        let end = (seg[last * 4 + 1] as usize).checked_add(seg[last * 4] as usize)?;
        if end < start {
            return None;
        }
        mx = mx.max(end - start);
    }
    if mx == 0 {
        return None;
    }
    Some(mx)
}

/// task 242: グループ局所用に、各 segment の行の先頭を**グループの先頭から**の
/// 相対番号へ直した `seg` を返す (`[行数, 行の先頭, 0, 0]` × segment)。
///
/// カーネルは出力を `obase + (s.ro + r)*n_out` に書き、入力を
/// `rows[s.ro + r]` から読む。グループ局所では `obase` をグループのバッファの
/// 先頭にしたいので `s.ro` を 0 起点にし、行表 (`rows_gu` / `rows_id`) の
/// ポインタをグループ先頭行だけ足して渡す (呼び出し側)。
///
/// 行の先頭がグループの先頭より小さければ `None` (plan の並びが壊れている)。
pub fn seg_group_local(seg: &[u32], bounds: &[(usize, usize)]) -> Option<Vec<u32>> {
    if seg.len() % 4 != 0 || bounds.is_empty() {
        return None;
    }
    let nseg = seg.len() / 4;
    let mut out = seg.to_vec();
    for &(a, b) in bounds {
        if a >= b || b > nseg {
            return None;
        }
        let base = seg[a * 4 + 1];
        for i in a..b {
            let off = out[i * 4 + 1];
            out[i * 4 + 1] = off.checked_sub(base)?;
        }
    }
    Some(out)
}

/// task 242: グループ局所の g/u/act を使うか (`DENNO_PF_LM_GL`)。
/// **既定 off**。`1` / `on` / `true` で有効。
///
/// 効くのは層優先 prefill の**デバイス出力**のときだけ (チャンク経路の
/// ホスト出力は従来どおり全体サイズ)。
///
/// lab t242b の実測では狙いどおり作業域が 2,870 → 2,100 MiB、追い出しが
/// 2,371 → 1,603 MiB、14k のストリーミングが 46.1 → 35.0 GB になった。
/// しかし lab t242c (GL のみ) の壁時計は 3,826 tok -1% / 14,018 tok **+17%** で、
/// GL を切った t242b の lmno (+7.6%) より悪い (別 run なので断定はできない)。
/// 採否 no なので既定 off。
pub fn lm_group_local() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("DENNO_PF_LM_GL").ok() {
        Some(s) => matches!(s.trim(), "1" | "on" | "true"),
        None => false,
    })
}

/// task 242: 層優先 prefill が追い出した expert を終端で戻すか
/// (`DENNO_PF_LM_RELOAD`)。**既定 off**。`1` / `on` / `true` で有効。
///
/// lab t242b の実測: 戻すのに 809 本 / 1,603 MiB で +0.8-1.5 秒かかり、
/// しかも 14k 後の decode のヒット率は 0.837 のまま (base は 0.973) だった。
/// 理由は prefill の途中で 1,505 MiB の `h` を確保できず、`DevBuffer` の
/// 追い出し階段が expert の LRU を全部捨てるため (戻した分も一緒に消える)。
/// つまり戻す相手が残っていない。ゲートは残すが既定 off。
pub fn lm_reload_evicted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("DENNO_PF_LM_RELOAD").ok() {
        Some(s) => matches!(s.trim(), "1" | "on" | "true"),
        None => false,
    })
}

/// task 394: 層優先 prefill の終端で戻す expert の転送を**複数スレッド**で行う
/// (`DENNO_PF_RESTORE_MT`)。**既定 on** (lab t394a / t394c で採った)。
///
/// # なぜ要るか (lab t390b / REPORT_P390 §1 の実測)
///
/// UD-Q4_K_XL の 14k cold では、prefill の終端で 1,057 本 (3,310 MiB) を LRU へ
/// 戻すのに `other` 段の未分類ぶんを使っていた。戻しの 1 本は「gate/up/down を
/// 3 回の同期 H2D」で、待ちが直列に並ぶ。ページは page cache に載っているので
/// (P390 §1 の先読みは 3.3 GB を 2 ms で触った)、律速は I/O ではなく
/// **同期転送 3 回 + 確保** の側にある。
///
/// ここでは [`crate::model::expert_vram::acquire_deferred`] (転送を LRU のロックの
/// 外で行う版。task 214 が背景の上げ直しで使っている) を N 本のスレッドで回し、
/// 同期の待ちを重ねる。VRAM へ渡るバイト列・LRU の内容は 1 本ずつの経路と同一で、
/// 変わるのは**順序と重なり**だけ。
///
/// # 実測 (lab t394a / t394c、`[pf-restore]` の ms)
///
/// | 条件 | 戻す本数 | mt=0 | mt=1 (8 本) | mt=1 (16 本) |
/// | --- | ---: | ---: | ---: | ---: |
/// | UD 3,826 (2 ラウンド) | 289 | 70 / 72 | **55 / 53** | 59 / 57 |
/// | UD 14k | 1,057 | 549 | **213** | 177 |
/// | 既定モデル 14k | 827 | 121 | **71** | — |
///
/// 壁時計への寄与は小さい (戻しは 14k の壁の 2%) が、段は 2 ラウンドとも
/// 2 割以上速く、decode・md5・既定モデルの数字はどれも変わらなかった
/// (t394c の既定モデルは 4 COND 全リクエストで md5 SAME)。
pub const PF_RESTORE_MT_ENV: &str = "DENNO_PF_RESTORE_MT";

/// [`PF_RESTORE_MT_ENV`] の既定 (on)。
pub const DEFAULT_PF_RESTORE_MT: bool = true;

/// [`PF_RESTORE_MT_ENV`] の解釈 (純関数)。未設定は既定 (on)、
/// **明示的な偽値だけ off** (`pf_pool_big_from_value` と同じ規約)。
pub fn pf_restore_mt_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_PF_RESTORE_MT,
        Some(s) => !matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
    }
}

/// 戻しを複数スレッドで行うか (プロセスで 1 回だけ env を読む)。
pub fn pf_restore_mt() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_restore_mt_from_value(std::env::var(PF_RESTORE_MT_ENV).ok().as_deref()))
}

/// `DENNO_PF_RESTORE_THREADS`: 戻しを回すスレッド数 (既定 8)。
///
/// 1 本あたり 3.1 MB のホスト写し (DDR) と DMA (PCIe) なので、本数を増やしても
/// 帯域で頭打ちになる。8 は DDR の 21 GB/s と PCIe の 52 GB/s に対して
/// 「1 本 0.2 ms」近くまで詰められる見込みの値 (推定)。
pub const PF_RESTORE_THREADS_ENV: &str = "DENNO_PF_RESTORE_THREADS";

/// [`PF_RESTORE_THREADS_ENV`] の既定値。
pub const DEFAULT_PF_RESTORE_THREADS: usize = 8;

/// [`PF_RESTORE_THREADS_ENV`] の解釈 (純関数)。1..=32 に丸める。0 も 1 にする。
pub fn pf_restore_threads_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_PF_RESTORE_THREADS;
    };
    match s.parse::<usize>() {
        Ok(n) => n.clamp(1, 32),
        Err(_) => DEFAULT_PF_RESTORE_THREADS,
    }
}

/// 戻しのスレッド数 (プロセスで 1 回だけ env を読む)。
pub fn pf_restore_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_restore_threads_from_value(std::env::var(PF_RESTORE_THREADS_ENV).ok().as_deref())
    })
}

/// task 394: 層優先 prefill で、**次の層の expert 重み全体**をページキャッシュへ
/// 先読みする (`DENNO_PF_LAYER_PRE`)。**既定 off** (測ってから決める)。
///
/// # なぜ要るか (lab t394a の実測)
///
/// 3,826 cold の壁 9.43-9.66 s に対し、`[exp-pf-stream] copy_s` は 7.11-8.22 s
/// (warm な回は 7.11 s = 6.24 GB/s で、P325 が測ったこのデバイスの上限 6.2 GB/s
/// と一致する)。つまり **壁はディスクで決まっていて、それ以上は読めない**。
/// ところが層ループは「attention (phase 1) → MoE (phase 2)」の順で、expert の
/// 読みは phase 2 の routing が決まってからしか始まらない。phase 1 のあいだ
/// ディスクは遊んでいる (3,826 で 1.2 s / 14k で 4.5 s、14k の `copy_s` は
/// 12.2-12.6 s で壁 25.3 s の 49%)。
///
/// ここでは routing ではなく「**その層の expert は全部要る**」という前提で
/// 1 層先を読む。読むのはページキャッシュへで、Linux の `readahead(2)` を使う
/// (ユーザー側の写しを伴わない)。
///
/// # 結果: **効かなかったので既定 off** (lab t394b、`ADOPT: no`)
///
/// 前提が外れていた。3,826 の `[exp-pf-stream] experts=14192 / 48 層 = 296 本`
/// に対し、層の全数は **512 本** (`lpre_MB` 77,018 MB / 48 層 / 3.13 MB = 512)。
/// つまり routing は 58% しか使わない。全数を読むと 44.4 GB の 1.73 倍 =
/// 77 GB を投げることになり、ディスクの空き (3,826 の `copy_s` は壁の 75-78%)
/// を無駄読みが食う。t394b の round 2 は b1 copy_s 7.36 / lp 7.25 / b2 7.20 で
/// **区別できない** (round 1 の -8% は b2 の位置効果)。
///
/// 予測で絞る道は task 337 が否定済み。この枝は閉じる。
pub const LAYER_PRE_ENV: &str = "DENNO_PF_LAYER_PRE";

/// [`LAYER_PRE_ENV`] の既定 (off)。
pub const DEFAULT_LAYER_PRE: bool = false;

/// [`LAYER_PRE_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn layer_pre_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => DEFAULT_LAYER_PRE,
        Some(s) => matches!(s.to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes"),
    }
}

/// 次の層を先読みするか (プロセスで 1 回だけ env を読む)。
pub fn layer_pre() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| layer_pre_from_value(std::env::var(LAYER_PRE_ENV).ok().as_deref()))
}

/// 先読みの台帳 (層数 / バイト / 呼び出し回数)。
static LAYER_PRE_LAYERS: AtomicU64 = AtomicU64::new(0);
static LAYER_PRE_BYTES: AtomicU64 = AtomicU64::new(0);
static LAYER_PRE_CALLS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんの先読みを積んだ (診断用)。`bytes` は投げたバイト数。
pub fn note_layer_pre(bytes: usize, calls: usize) {
    LAYER_PRE_LAYERS.fetch_add(1, Ordering::Relaxed);
    LAYER_PRE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    LAYER_PRE_CALLS.fetch_add(calls as u64, Ordering::Relaxed);
}

/// `[exp-pf-stream]` の末尾へ足す先読みの台帳 (発火していなければ空)。
pub fn layer_pre_stats() -> String {
    let n = LAYER_PRE_LAYERS.load(Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    let mb = LAYER_PRE_BYTES.load(Ordering::Relaxed) as f64 / 1e6;
    format!(
        " lpre_layers={n} lpre_MB={mb:.1} lpre_calls={}",
        LAYER_PRE_CALLS.load(Ordering::Relaxed)
    )
}

/// 先読みの台帳を 0 に戻す。
pub fn layer_pre_stats_reset() {
    for c in [&LAYER_PRE_LAYERS, &LAYER_PRE_BYTES, &LAYER_PRE_CALLS] {
        c.store(0, Ordering::Relaxed);
    }
}

// ===========================================================================
// task 470: 1 チャンク目の routing を解いてから先読みする (`DENNO_PF_EARLY_ROUTE`)
// ===========================================================================
//
// task 394 の `DENNO_PF_LAYER_PRE` は「その層の expert は全部要る」という前提で
// 1 層先を `readahead(2)` したが、実測は層に **512 本** あるのに routing は
// **296 本 (58%)** しか使わず、77 GB を投げて 44.4 GB を読む 1.73 倍の無駄読みに
// なって負けた (t394b)。無駄読みはディスクの遊びをそのまま食うので、58% では
// 差が出ない。
//
// ここでは予測ではなく**事実**を使う。層優先 prefill は 1 チャンク目の
// attention が終わった時点で、そのチャンクのトークンの MoE 入力がデバイスに
// 確定しているので、**そのチャンクの routing を実際に解いて**選ばれた expert
// だけを先に読む。チャンク 0 の expert 集合は層全体の集合の**部分集合**なので
// (トークンが増えて減ることはない)、無駄読みは 0。残りのチャンクの attention が
// 走っているあいだディスクが遊ぶぶん (3,826 で 1 層 9.5 ms / 14k で 46 ms) を
// そのまま取り返す。
//
// 値は 1 バイトも変えない (router の結果は先読みにしか使わず、MoE は従来どおり
// 全トークンで routing を解き直す)。ゲート off では分岐 1 回だけ。

/// `DENNO_PF_EARLY_ROUTE`: 1 チャンク目の routing で選ばれた expert を先読みし、
/// **RAM の置き場**へ置く (MoE の写しがそこから memcpy する)。**既定 on**
/// (task 474 の lab t474b で採用)。
///
/// - `0` / `off` — 従来 (先読みしない)
/// - `1` / `on` — 置き場へ読む (残りの窓が 4 チャンク以上ある層だけ)
/// - `2` / `pread` — `1` と同じ (task 470 の pread 版の名前。互換のため残す)
///
/// task 470 の `readahead(2)` / 「pread して捨てる」はページキャッシュに残らず
/// `copy_s` が投げたぶんの 11-28% しか縮まなかったので、task 474 で置き場を
/// プロセスが持つ形に変えた (lab t474b: 14k cold で -25%、3,826 はゲート on でも
/// 1 命令も変わらない = 残り 1 チャンクの窓には収まらないので先読みしない)。
pub const EARLY_ROUTE_ENV: &str = "DENNO_PF_EARLY_ROUTE";

/// [`EARLY_ROUTE_ENV`] の既定 (task 474 で on)。
pub const DEFAULT_EARLY_ROUTE: bool = true;

/// ゲートの解釈 (純関数)。未設定は既定 (on)、`0` 系だけが偽。
pub fn early_route_mode_from_value(v: Option<&str>) -> u8 {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("2") | Some("pread") => 2,
        Some("1") | Some("on") | Some("true") | Some("yes") => 1,
        Some("0") | Some("off") | Some("false") | Some("no") => 0,
        // 未設定 / 空 / 解析不能は既定。
        _ => u8::from(DEFAULT_EARLY_ROUTE),
    }
}

/// ゲートの解釈 (真偽だけ)。明示的な真値で on。
pub fn early_route_from_value(v: Option<&str>) -> bool {
    early_route_mode_from_value(v) != 0
}

/// 先読みのやり方 (プロセスで 1 回だけ env を読む)。0 = off。
pub fn early_route_mode() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| early_route_mode_from_value(std::env::var(EARLY_ROUTE_ENV).ok().as_deref()))
}

/// 1 チャンク目の routing で先読みするか (プロセスで 1 回だけ env を読む)。
pub fn early_route_on() -> bool {
    early_route_mode() != 0
}

/// `DENNO_PF_EARLY_ROUTE_MB`: 1 層で先読みする上限 (MiB)。**0 = 自動**
/// ([`early_route_budget`])。
pub const EARLY_ROUTE_MB_ENV: &str = "DENNO_PF_EARLY_ROUTE_MB";

/// `DENNO_PF_EARLY_ROUTE_KB`: 自動のときの「残りのトークン 1 個あたり何 KiB 投げるか」。
///
/// 実測の比 (UD-Q4_K_XL、t470a の b1): attention は 3,826 で 0.240 ms/tok、
/// 14,018 で 0.254 ms/tok でほぼ同じ、ディスクは 5.9 GB/s。1 チャンク目を除いた
/// 残りトークンが遊ぶディスクは 0.25 ms × 5.9 GB/s = 1.45 MB/tok ではなく、
/// **残りチャンクの GPU 時間**ぶんなので、14k (残り 12,000 tok、1 層 46 ms) で
/// 271 MB = 23 KiB/tok、3,826 (残り 1,913 tok、1 層 9.5 ms) で 45 MB = 24 KiB/tok。
/// 既定はその中央の 24 KiB。
pub const EARLY_ROUTE_KB_ENV: &str = "DENNO_PF_EARLY_ROUTE_KB";

/// [`EARLY_ROUTE_KB_ENV`] の既定 (KiB/token)。
pub const DEFAULT_EARLY_ROUTE_KB: usize = 24;

/// 自動の上限の下限 / 上限 (バイト)。1 本の expert は約 3 MB なので、
/// 下限でも 1 本は投げられる。
pub const EARLY_ROUTE_MIN_BYTES: usize = 8 << 20;
/// 自動の上限の上限 (バイト)。ページキャッシュを 1 層ぶん以上汚さない。
pub const EARLY_ROUTE_MAX_BYTES: usize = 768 << 20;

/// [`EARLY_ROUTE_MB_ENV`] の解釈 (純関数)。数値だけを受け付け、0 は自動。
pub fn early_route_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s.parse::<usize>().unwrap_or(0),
        None => 0,
    }
}

/// 1 層で先読みする上限 (MiB、0 = 自動)。
pub fn early_route_mb() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| early_route_mb_from_value(std::env::var(EARLY_ROUTE_MB_ENV).ok().as_deref()))
}

/// [`EARLY_ROUTE_KB_ENV`] の解釈 (純関数)。数値だけ、0 も許す。
pub fn early_route_kb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s.parse::<usize>().unwrap_or(DEFAULT_EARLY_ROUTE_KB),
        None => DEFAULT_EARLY_ROUTE_KB,
    }
}

/// 自動のときの KiB/token。
pub fn early_route_kb() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        early_route_kb_from_value(std::env::var(EARLY_ROUTE_KB_ENV).ok().as_deref())
    })
}

/// 1 層ぶんの先読み上限 (バイト) を決める (純関数)。
///
/// - `mb > 0` ならその MiB (env の明示指定が最優先)。
/// - `mb == 0` なら `残りトークン数 × kb KiB` を
///   [`EARLY_ROUTE_MIN_BYTES`]..=[`EARLY_ROUTE_MAX_BYTES`] に丸める。
pub fn early_route_budget(mb: usize, kb: usize, rest_tok: usize) -> usize {
    if mb > 0 {
        return mb.saturating_mul(1 << 20);
    }
    let want = rest_tok.saturating_mul(kb).saturating_mul(1 << 10);
    want.clamp(EARLY_ROUTE_MIN_BYTES, EARLY_ROUTE_MAX_BYTES)
}

/// 先読みの窓として要る「残りチャンク数」の下限。
///
/// lab t474a: 3,826 (残り 1 チャンク) では 1 層 42 MB の先読みが窓 (9.5 ms) に
/// 収まらず (11.3 ms)、壁は ±0 だった。14k (残り 6 チャンク) は収まって -13.6%。
pub const EARLY_ROUTE_MIN_REST_CHUNKS: usize = 4;

/// 先読みの窓が十分か (純関数)。
///
/// 「残りのチャンクが [`EARLY_ROUTE_MIN_REST_CHUNKS`] 本以上」= 窓が
/// `4 × nt0` トークン以上あること。1 チャンクで終わる層 (窓なし) も false。
///
/// task 482: この判定は [`crate::model::qwen4exp`] の
/// `pf_early_route_prefetch` の**先頭**でも使う ([`er_early_exit_on`])。
/// 以前は routing (GPU の router GEMV + D2H) と VRAM 参照の後ろにあり、
/// 窓が足りない 3,826 でも 48 層ぶんそれを払っていた (lab t482a の
/// `sub other pf_er=72.6/90.1 ms`)。
pub fn early_route_window_ok(nt0: usize, nt_all: usize) -> bool {
    if nt0 == 0 || nt_all <= nt0 {
        return false;
    }
    nt_all - nt0 >= nt0.saturating_mul(EARLY_ROUTE_MIN_REST_CHUNKS)
}

/// task 482: 窓の判定を `pf_early_route_prefetch` の**先頭**へ移すか
/// (`DENNO_PF_ER_EARLY_EXIT`)。**既定 on**、明示的な偽値だけ切る。
///
/// on では、窓が足りない層 (3,826 の 2 チャンクなど) で router GEMV + D2H +
/// VRAM 参照 + 置き場の計画を**まるごと飛ばす**。窓が足りない層はこの先の
/// 経路が `None` を返して何もしないので、**値も読みも 1 バイトも変わらない**
/// (捨てる計算を先に捨てるだけ)。off は task 474 と同じ順序。
pub const ER_EARLY_EXIT_ENV: &str = "DENNO_PF_ER_EARLY_EXIT";

/// [`ER_EARLY_EXIT_ENV`] の解釈 (純関数)。既定 on、明示的な偽値だけ off。
pub fn er_early_exit_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no") | Some("")
    )
}

/// [`ER_EARLY_EXIT_ENV`] をプロセスで 1 回だけ読む。
pub fn er_early_exit_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        er_early_exit_from_value(std::env::var(ER_EARLY_EXIT_ENV).ok().as_deref())
    })
}

/// 先読みの台帳 (層数 / expert 本数 / バイト / 呼び出し回数)。
///
/// task 474: 実行役の join と読み終えるまでの時間は置き場の側
/// ([`rc_join`] / [`note_rc_read_ns`]) が持つ。ここは「何層ぶん投げたか」だけ。
static ER_LAYERS: AtomicU64 = AtomicU64::new(0);
static ER_EXPERTS: AtomicU64 = AtomicU64::new(0);
static ER_BYTES: AtomicU64 = AtomicU64::new(0);
static ER_CALLS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶんの先読みを積んだ (診断用)。
pub fn note_early_route(experts: usize, bytes: usize, calls: usize) {
    ER_LAYERS.fetch_add(1, Ordering::Relaxed);
    ER_EXPERTS.fetch_add(experts as u64, Ordering::Relaxed);
    ER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    ER_CALLS.fetch_add(calls as u64, Ordering::Relaxed);
}

/// `[exp-pf-stream]` の末尾へ足す台帳 (発火していなければ空)。
pub fn early_route_stats() -> String {
    let n = ER_LAYERS.load(Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    let mb = ER_BYTES.load(Ordering::Relaxed) as f64 / 1e6;
    format!(
        " er_layers={n} er_experts={} er_MB={mb:.1} er_calls={}",
        ER_EXPERTS.load(Ordering::Relaxed),
        ER_CALLS.load(Ordering::Relaxed)
    )
}

/// 先読みの台帳を 0 に戻す。
pub fn early_route_stats_reset() {
    for c in [&ER_LAYERS, &ER_EXPERTS, &ER_BYTES, &ER_CALLS] {
        c.store(0, Ordering::Relaxed);
    }
}


/// `part` の `off..off+len` をページキャッシュへ先に読む (Linux の `readahead(2)`)。
///
/// - ユーザー側のバッファを持たない (カーネルがページキャッシュを埋めるだけ)。
/// - 戻り値は**投げたバイト数** (0 = パートが開けていない / Linux 以外)。
/// - 呼び出しは待たない (カーネルが非同期に読む)。読み終わっていないページを
///   触ると、あとからの `pread` はそのページの完了を待つだけ (二重読みはしない)。
pub fn readahead_part(part: u32, off: u64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let Some(Some(f)) = PARTS.get().and_then(|v| v.get(part as usize)) else {
        return 0;
    };
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn readahead(fd: i32, offset: i64, count: usize) -> isize;
        }
        const PER_CALL: usize = 1 << 30; // 1 GiB ずつ投げる (巨大範囲を分割)
        let fd = f.as_raw_fd();
        let mut done = 0usize;
        while done < len {
            let n = (len - done).min(PER_CALL);
            let r = unsafe { readahead(fd, (off + done as u64) as i64, n) };
            if r < 0 {
                break;
            }
            done += n;
        }
        done
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (f, off);
        0
    }
}

// ---------------------------------------------------------------------------
// task 474: 先読みを「捨てる」のをやめて RAM に置き、MoE の写しがそこから取る
// ---------------------------------------------------------------------------
//
// task 470 の実測: 1 チャンク目の routing を解いて **残りチャンクの attention の
// あいだ** に expert を読む窓は実在する (3,826 で 1 層 9.5 ms、14k で 62 ms) が、
// (a) `readahead(2)` はほぼ無効、(b) 背景 pread で捨ててもページキャッシュに
// 残らず、`copy_s` が縮んだのは投げたバイトの 11-28% だけだった。
//
// そこで **読んだバイトをプロセスの RAM に置く**。MoE の写し
// (`pf_copy_into_pin`) は、同じ (パート, 絶対オフセット, 長さ) が置き場にあれば
// ディスクを読まずに memcpy する。値はバイト単位で同じなので**数値は 1 bit も
// 変わらない** (読む場所が変わるだけ)。
//
// 置き場は 1 層ぶんを使い回す。1 本の expert は約 3 MB、窓の中で読める量は
// `DENNO_PF_EARLY_ROUTE` の予算 (既定: 残りトークン × 24 KiB) で決まる。

/// `DENNO_PF_RA_CACHE_MB`: 先読みの置き場 (MiB)。**既定 512**。
///
/// 1 層ぶんしか使わない。14k の予算は 287 MB (24 KiB × 11,970 tok) なので
/// 512 MiB あれば足りる。0 にすると置き場を作らない (先読みもしない)。
pub const RC_CACHE_MB_ENV: &str = "DENNO_PF_RA_CACHE_MB";

/// [`RC_CACHE_MB_ENV`] の既定 (MiB)。
pub const DEFAULT_RC_CACHE_MB: usize = 512;

/// `DENNO_PF_RA_THREADS`: 先読みの並列度。**既定 8**。
///
/// 窓のあいだディスクを回す本数。コピー側は 32 本 (`DENNO_PF_PREAD_THREADS`) だが、
/// 窓は短いので上げすぎても立ち上げで損をする (t470d の実測: 14 本で 3.2-5.5 GB/s)。
pub const RC_THREADS_ENV: &str = "DENNO_PF_RA_THREADS";

/// [`RC_THREADS_ENV`] の既定。
pub const DEFAULT_RC_THREADS: usize = 8;

/// [`RC_CACHE_MB_ENV`] の解釈 (純関数)。0 は「置き場なし」、それ以外は
/// 32..=2048 MiB に丸める。
pub fn rc_cache_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some("") | None => DEFAULT_RC_CACHE_MB,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => 0,
            Ok(n) => n.clamp(32, 2048),
            Err(_) => DEFAULT_RC_CACHE_MB,
        },
    }
}

/// 置き場のバイト数 (プロセスで 1 回だけ env を読む)。
pub fn rc_cache_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        rc_cache_mb_from_value(std::env::var(RC_CACHE_MB_ENV).ok().as_deref()) << 20
    })
}

/// [`RC_THREADS_ENV`] の解釈 (純関数)。1..=32。
pub fn rc_threads_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some("") | None => DEFAULT_RC_THREADS,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => DEFAULT_RC_THREADS,
            Ok(n) => n.clamp(1, 32),
            Err(_) => DEFAULT_RC_THREADS,
        },
    }
}

/// 先読みの並列度。
pub fn rc_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| rc_threads_from_value(std::env::var(RC_THREADS_ENV).ok().as_deref()))
}

/// 置き場の 1 エントリ。`base` は [`RcBuf`] の先頭からのオフセット。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RcEnt {
    pub file: u32,
    pub off: u64,
    pub len: u32,
    pub base: u32,
}

/// `(file, off, len)` = ディスク上の範囲を、置き場のどの位置へ読むか。
///
/// [`rc_layout`] の戻り値の 1 本。`base` は置き場の先頭からのバイト。
pub type RcJob = (u32, u32, u64, u32);

/// 先読みの範囲を置き場のオフセットへ割り当てる (純関数)。
///
/// `runs` は **(パート, 絶対オフセット, 長さ)** を MoE が読む順に並べたもの。
/// 先頭から `cap` バイトに収まる手前までを割り当てる (expert 昇順の先頭 =
/// MoE が最初に読むぶん)。戻り値は (割り当て, 合計バイト)。1 本も入らなければ
/// `None`。
pub fn rc_layout(runs: &[(u32, u64, usize)], cap: usize) -> Option<(Vec<RcJob>, usize)> {
    let mut jobs: Vec<RcJob> = Vec::with_capacity(runs.len());
    let mut at = 0usize;
    for &(file, off, len) in runs {
        if len == 0 {
            continue;
        }
        if at.saturating_add(len) > cap {
            break;
        }
        jobs.push((at as u32, file, off, len as u32));
        at += len;
    }
    if jobs.is_empty() {
        return None;
    }
    Some((jobs, at))
}

/// 置き場そのもの (プロセスで 1 回だけ確保し、層ごとに使い回す)。
///
/// 中身は先読みスレッドが**互いに重ならない範囲**へ書き、書き終えた範囲だけを
/// 索引へ入れる (読みかけのバイトを MoE が掴まない)。
pub struct RcBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: 確保した領域は解放しない (プロセスの終わりまで有効)。書き込みは
// `rc_layout` が割り当てた重ならない範囲だけで、索引への登録は `rc_commit_one`
// (Mutex) を通す。読み出しは登録済みの範囲だけ。
unsafe impl Send for RcBuf {}
unsafe impl Sync for RcBuf {}

impl RcBuf {
    /// 置き場の先頭。
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    /// 置き場のバイト数。
    pub fn len(&self) -> usize {
        self.len
    }
    /// 空か。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

static RC_BUF: std::sync::OnceLock<Option<std::sync::Arc<RcBuf>>> = std::sync::OnceLock::new();

/// 置き場を用意する (初回だけ確保する)。0 バイト設定なら `None`。
fn rc_buf() -> Option<std::sync::Arc<RcBuf>> {
    RC_BUF
        .get_or_init(|| {
            let n = rc_cache_bytes();
            if n == 0 {
                return None;
            }
            let mut v = vec![0u8; n].into_boxed_slice();
            let ptr = v.as_mut_ptr();
            std::mem::forget(v);
            Some(std::sync::Arc::new(RcBuf { ptr, len: n }))
        })
        .clone()
}

struct RcIdx {
    ents: Vec<RcEnt>,
}

static RC_IDX: std::sync::Mutex<RcIdx> = std::sync::Mutex::new(RcIdx { ents: Vec::new() });

/// 先読みを止めるか (MoE が始まったら窓は終わり)。
static RC_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 先読みの実行役 (高々 1 本)。
static RC_WORKER: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> =
    std::sync::Mutex::new(None);

/// 置き場に載ったバイト / MoE が置き場から取ったバイト / その本数 / 外れた本数。
static RC_BYTES: AtomicU64 = AtomicU64::new(0);
static RC_HIT_BYTES: AtomicU64 = AtomicU64::new(0);
static RC_HIT_CALLS: AtomicU64 = AtomicU64::new(0);
static RC_MISS_CALLS: AtomicU64 = AtomicU64::new(0);
/// 先読みが読み終えるまでの時間 (層ごとの最大を積む)。
static RC_READ_NS: AtomicU64 = AtomicU64::new(0);

/// 前の層の実行役を回収する (まだ走っていれば止めてから待つ)。
pub fn rc_join() {
    RC_STOP.store(true, Ordering::Relaxed);
    let Ok(mut g) = RC_WORKER.lock() else {
        return;
    };
    if let Some(h) = g.take() {
        let _ = h.join();
    }
}

/// 窓の終わりを知らせる (実行役は次の範囲の手前で止まる)。
pub fn rc_stop() {
    RC_STOP.store(true, Ordering::Relaxed);
}

/// 先読みが止められたか (実行役が範囲ごとに見る)。
pub fn rc_stopped() -> bool {
    RC_STOP.load(Ordering::Relaxed)
}

/// 索引の本数 (ロックを取らずに「載っていない」を判定するため)。
static RC_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 索引を空にする (置き場のバイトは消さない)。
pub fn rc_clear() {
    if let Ok(mut g) = RC_IDX.lock() {
        g.ents.clear();
    }
    RC_N.store(0, Ordering::Relaxed);
}

/// 1 層ぶんの先読みを始める。
///
/// 前の層の実行役を回収し、索引を空にし、`runs` を置き場へ割り当てて返す
/// (読み込みは呼び出し側が別スレッドで行い、[`rc_commit_one`] で索引へ入れる)。
pub fn rc_plan(runs: &[(u32, u64, usize)]) -> Option<(std::sync::Arc<RcBuf>, Vec<RcJob>)> {
    rc_join();
    rc_clear();
    let buf = rc_buf()?;
    let (jobs, _total) = rc_layout(runs, buf.len())?;
    RC_STOP.store(false, Ordering::Relaxed);
    Some((buf, jobs))
}

/// 実行役を登録する (次の層の [`rc_plan`] が回収する)。
pub fn rc_worker_set(h: std::thread::JoinHandle<()>) {
    if let Ok(mut g) = RC_WORKER.lock() {
        if let Some(old) = g.replace(h) {
            let _ = old.join();
        }
    }
}

/// 1 範囲を読み終えたので索引へ入れる (読みかけを見せない)。
pub fn rc_commit_one(file: u32, off: u64, len: usize, base: u32) {
    if len == 0 || len > u32::MAX as usize {
        return;
    }
    if let Ok(mut g) = RC_IDX.lock() {
        g.ents.push(RcEnt {
            file,
            off,
            len: len as u32,
            base,
        });
        RC_N.store(g.ents.len(), Ordering::Relaxed);
        RC_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// 置き場に `(file, off, len)` がちょうど載っているか。
pub fn rc_has(file: u32, off: u64, len: usize) -> bool {
    if RC_N.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let Ok(g) = RC_IDX.lock() else {
        return false;
    };
    g.ents
        .iter()
        .any(|e| e.file == file && e.off == off && e.len as usize == len)
}

/// 置き場から `dst` へ写す。載っていなければ false (呼び出し側がディスクへ落ちる)。
///
/// # Safety
///
/// `dst` は `len` バイト以上書ける領域を指すこと。
pub unsafe fn rc_copy_into(file: u32, off: u64, len: usize, dst: *mut u8) -> bool {
    if len == 0 || dst.is_null() {
        return false;
    }
    if RC_N.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let Ok(g) = RC_IDX.lock() else {
        return false;
    };
    let Some(e) = g
        .ents
        .iter()
        .find(|e| e.file == file && e.off == off && e.len as usize == len)
    else {
        RC_MISS_CALLS.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let src = unsafe { rc_buf_ptr().add(e.base as usize) };
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
    RC_HIT_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    RC_HIT_CALLS.fetch_add(1, Ordering::Relaxed);
    true
}

/// 置き場の先頭 (索引が空でないときだけ呼ばれる)。
fn rc_buf_ptr() -> *const u8 {
    match RC_BUF.get().and_then(|b| b.as_ref()) {
        Some(b) => b.ptr as *const u8,
        None => std::ptr::null(),
    }
}

/// 先読み専用の rayon プール (プロセスで 1 個)。
///
/// 層ループの rayon を奪わないよう、置き場を埋めるスレッドは専用にする。
pub fn rc_pool(nt: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .thread_name(|i| format!("pf-ra-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// 先読みが 1 層ぶん読み終えるまでの時間を積む (診断)。
pub fn note_rc_read_ns(ns: u64) {
    RC_READ_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 置き場の台帳 (`[exp-pf-stream]` の末尾へ足す。載っていなければ空)。
pub fn rc_stats() -> String {
    let bytes = RC_BYTES.load(Ordering::Relaxed);
    let hit = RC_HIT_BYTES.load(Ordering::Relaxed);
    let miss = RC_MISS_CALLS.load(Ordering::Relaxed);
    if bytes == 0 && hit == 0 && miss == 0 {
        return String::new();
    }
    let ms = RC_READ_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let hit_mb = hit as f64 / 1e6;
    let put_mb = bytes as f64 / 1e6;
    let pct = if bytes > 0 {
        hit as f64 * 100.0 / bytes as f64
    } else {
        0.0
    };
    format!(
        " ra_MB={put_mb:.1} ra_hit_MB={hit_mb:.1} ra_hit={} ra_miss={miss} ra_ms={ms:.1} ra_pct={pct:.0}",
        RC_HIT_CALLS.load(Ordering::Relaxed)
    )
}

/// 置き場の台帳を 0 に戻す。
pub fn rc_stats_reset() {
    for c in [&RC_BYTES, &RC_HIT_BYTES, &RC_HIT_CALLS, &RC_MISS_CALLS] {
        c.store(0, Ordering::Relaxed);
    }
    RC_READ_NS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// 1 リクエストぶんの観測 (REPORT 用)
// ---------------------------------------------------------------------------

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static LAYERS: AtomicU64 = AtomicU64::new(0);
static EXPERTS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static D2H_NS: AtomicU64 = AtomicU64::new(0);
static ADD_NS: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static SKIP_LAYERS: AtomicU64 = AtomicU64::new(0);
/// task 162: ストリーミングで GPU に載せた層の数。
static STREAM_LAYERS: AtomicU64 = AtomicU64::new(0);
/// task 333: 上げスレッドを使わない写しでやり直した回数 (`DENNO_PF_STREAM_RETRY`)。
static STREAM_RETRY: AtomicU64 = AtomicU64::new(0);
/// task 379: `DENNO_PF_GDN_WB_LAST` で飛ばした GDN 状態の書き戻し回数
/// (1 回 = conv + ssm の 2 本)。ゲートが効いた証拠として `[exp-pf-stream]` に出す。
static GDN_WB_SKIPS: AtomicU64 = AtomicU64::new(0);
/// task 261: 旧 4 型 (Q8_0 / Q5_1 / Q4_K / Q5_K) の prefill GPU 経路。
/// `QUANT_PF_LAYERS` = GPU で回し切った層、`QUANT_PF_SKIPS` = 途中で諦めた層。
static QUANT_PF_LAYERS: AtomicU64 = AtomicU64::new(0);
static QUANT_PF_ROWS: AtomicU64 = AtomicU64::new(0);
static QUANT_PF_HITS: AtomicU64 = AtomicU64::new(0);
static QUANT_PF_SKIPS: AtomicU64 = AtomicU64::new(0);
static QUANT_PF_NS: AtomicU64 = AtomicU64::new(0);
/// task 162: ステージングへ上げた expert 本数 (層またぎの再送を含む)。
static STREAM_EXPERTS: AtomicU64 = AtomicU64::new(0);
/// task 162: ステージングへ上げたバイト数。
static STREAM_BYTES: AtomicU64 = AtomicU64::new(0);
/// task 162: H2D の完了待ちで止まった時間 (重なりの指標)。
static STREAM_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// task 162: ホストのバウンスバッファへの写しの所要。
static STREAM_COPY_NS: AtomicU64 = AtomicU64::new(0);
/// task 165: H2D の積み込み (cuMemcpyHtoDAsync + イベント記録) の所要。
static STREAM_ENQ_NS: AtomicU64 = AtomicU64::new(0);
/// task 165: 上げ専用スレッドの順番待ち (計算スレッドが ready を待った時間)。
static STREAM_READY_NS: AtomicU64 = AtomicU64::new(0);
/// task 165: ステージングの再利用待ち (前のグループの計算完了待ち)。
static STREAM_SLOT_NS: AtomicU64 = AtomicU64::new(0);
/// task 233: スロット再利用前の「計算完了イベント」待ち (診断。今まで台帳に出ていなかった)。
static STREAM_USE_NS: AtomicU64 = AtomicU64::new(0);
/// task 324: H2D の完了を**計算ストリーム**に待たせたグループ数 (ゲートの発火の証拠)。
static STREAM_GPUWAIT: AtomicU64 = AtomicU64::new(0);
/// task 324: 層の終端で同期を省いた回数 (ゲートの発火の証拠)。
static STREAM_NOSYNC: AtomicU64 = AtomicU64::new(0);
/// task 162: ステージングへ載せた expert の行数 (1 本あたりの行数を出すため)。
static STREAM_ROWS: AtomicU64 = AtomicU64::new(0);
/// task 162: 上げの回数 (グループ数)。
static STREAM_GROUPS: AtomicU64 = AtomicU64::new(0);
/// task 162: ステージングに載らず CPU に残した expert 本数。
static STREAM_SPILL: AtomicU64 = AtomicU64::new(0);
/// task 162: CPU に残したミスの計算に要した時間。
static STREAM_CPU_MISS_NS: AtomicU64 = AtomicU64::new(0);
/// task 162: 最初のグループで測った実 H2D 速度 (GB/s × 1000、0 = 未測)。
static STREAM_PROBE_MBPS: AtomicU64 = AtomicU64::new(0);
/// task 227: `cuMemHostRegister` を試したバイト数 / 通ったバイト数 / 回数 / 時間。
static REG_BYTES: AtomicU64 = AtomicU64::new(0);
static REG_OK_BYTES: AtomicU64 = AtomicU64::new(0);
static REG_CALLS: AtomicU64 = AtomicU64::new(0);
static REG_NS: AtomicU64 = AtomicU64::new(0);
/// task 227: 通ったフラグ (u32::MAX = まだ通っていない)。
static REG_FLAGS: AtomicU64 = AtomicU64::new(u32::MAX as u64);

/// task 227: `cuMemHostRegister` の 1 回ぶんを積む。
pub fn note_reg(bytes: usize, ok: bool, ns: u64, flags: u32) {
    REG_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    REG_CALLS.fetch_add(1, Ordering::Relaxed);
    REG_NS.fetch_add(ns, Ordering::Relaxed);
    if ok {
        REG_OK_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        REG_FLAGS.store(flags as u64, Ordering::Relaxed);
    }
}

/// 1 層ぶんの参照を積む (`lookups` = 選択 expert 数、`hits` = 載っていた数)。
pub fn note_lookup(lookups: u32, hits: u32) {
    LOOKUPS.fetch_add(lookups as u64, Ordering::Relaxed);
    HITS.fetch_add(hits as u64, Ordering::Relaxed);
}

/// 1 層ぶんの GPU 実行を積む (`rows` = GPU に載った行数、`experts` = 本数)。
pub fn note_layer(rows: u32, experts: u32, gpu_ns: u64) {
    LAYERS.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    EXPERTS.fetch_add(experts as u64, Ordering::Relaxed);
    GPU_NS.fetch_add(gpu_ns, Ordering::Relaxed);
}

/// 1 層ぶんの発行〜同期 (カーネル本体) の時間を積む。
pub fn note_kernel_ns(ns: u64) {
    KERNEL_NS.fetch_add(ns, Ordering::Relaxed);
}

/// 1 層ぶんの D2H (加重和の nt×e) の時間を積む。
pub fn note_d2h_ns(ns: u64) {
    D2H_NS.fetch_add(ns, Ordering::Relaxed);
}

/// ヒット寄与の足し込み (ホスト側) の時間を積む。
pub fn note_add_ns(ns: u64) {
    ADD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// ヒットが少なくて GPU を呼ばなかった層を積む。
pub fn note_skip_layer() {
    SKIP_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// 上げ直し 1 本と、そのホスト時間。
pub fn note_upload_ns(ns: u64) {
    UPLOADS.fetch_add(1, Ordering::Relaxed);
    UPLOAD_NS.fetch_add(ns, Ordering::Relaxed);
}

/// task 162: ステージングへ 1 グループ上げた。
///
/// `bytes` / `rows` / `experts` はこのグループの合計、`copy_ns` はホスト側の
/// 写しの時間、`wait_ns` は H2D の完了待ちで止まった時間 (0 に近いほど重なっている)。
///
/// task 165: 上げ専用スレッドの内訳も同じ行に積む。
/// `enq_ns` は積み込み (非同期コピー + イベント記録)、`ready_ns` は計算スレッドが
/// ワーカーの順番待ちで止まった時間、`slot_ns` はワーカーがステージングの
/// 再利用 (前のグループの計算完了) を待った時間。
#[allow(clippy::too_many_arguments)]
pub fn note_stream_group(
    bytes: usize,
    rows: usize,
    experts: usize,
    copy_ns: u64,
    enq_ns: u64,
    ready_ns: u64,
    wait_ns: u64,
    slot_ns: u64,
) {
    STREAM_GROUPS.fetch_add(1, Ordering::Relaxed);
    STREAM_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    STREAM_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    STREAM_EXPERTS.fetch_add(experts as u64, Ordering::Relaxed);
    STREAM_COPY_NS.fetch_add(copy_ns, Ordering::Relaxed);
    STREAM_ENQ_NS.fetch_add(enq_ns, Ordering::Relaxed);
    STREAM_READY_NS.fetch_add(ready_ns, Ordering::Relaxed);
    STREAM_WAIT_NS.fetch_add(wait_ns, Ordering::Relaxed);
    STREAM_SLOT_NS.fetch_add(slot_ns, Ordering::Relaxed);
}

/// task 233: スロット再利用前の計算完了待ちの時間を積む (診断)。
pub fn note_stream_use_ns(ns: u64) {
    STREAM_USE_NS.fetch_add(ns, Ordering::Relaxed);
}

/// task 324: H2D の完了を計算ストリーム (`cuStreamWaitEvent`) に待たせた。
pub fn note_stream_gpuwait() {
    STREAM_GPUWAIT.fetch_add(1, Ordering::Relaxed);
}

/// task 324: 層の終端で同期を省いた (デバイス出力のまま次へ渡した)。
pub fn note_stream_nosync() {
    STREAM_NOSYNC.fetch_add(1, Ordering::Relaxed);
}

/// task 162: ストリーミングで 1 層を GPU に載せた。
pub fn note_stream_layer() {
    STREAM_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// task 333: 上げスレッドを使わない写しで 1 回やり直した。
pub fn note_stream_retry() {
    STREAM_RETRY.fetch_add(1, Ordering::Relaxed);
}

/// task 379: GDN 状態の書き戻しを 1 回飛ばした (conv + ssm の 2 本)。
pub fn note_gdn_wb_skip() {
    GDN_WB_SKIPS.fetch_add(1, Ordering::Relaxed);
}

/// task 379: 飛ばした書き戻しの回数 (累積)。
pub fn gdn_wb_skips() -> u64 {
    GDN_WB_SKIPS.load(Ordering::Relaxed)
}

/// task 162: CPU に残したミスの計算時間。
pub fn note_stream_cpu_miss_ns(ns: u64) {
    STREAM_CPU_MISS_NS.fetch_add(ns, Ordering::Relaxed);
}

/// task 162: 最初のグループで測った実 H2D 速度 (bytes / ns)。
pub fn note_stream_probe(bytes: usize, ns: u64) {
    if bytes == 0 || ns == 0 {
        return;
    }
    // MB/s を 1000 倍して整数で持つ (AtomicF64 を使わない)。
    let mbps = (bytes as f64 / 1e6) / (ns as f64 / 1e9);
    let _ = STREAM_PROBE_MBPS.compare_exchange(
        0,
        (mbps * 1000.0) as u64,
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
}

/// task 162: ストリーミングの累計 (バイト, 本数, 行数, 待ち秒)。REPORT 用。
pub fn stream_totals() -> (u64, u64, u64, f64) {
    (
        STREAM_BYTES.load(Ordering::Relaxed),
        STREAM_EXPERTS.load(Ordering::Relaxed),
        STREAM_ROWS.load(Ordering::Relaxed),
        STREAM_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e9,
    )
}

/// 1 リクエストぶんの 1 行。ゲート off / GPU 層 0 なら `None`。
/// task 261: 旧 4 型の prefill 経路で 1 層を GPU で回し切った。
pub fn note_quant_pf_layer(rows: u32, hits: u32) {
    QUANT_PF_LAYERS.fetch_add(1, Ordering::Relaxed);
    QUANT_PF_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    QUANT_PF_HITS.fetch_add(hits as u64, Ordering::Relaxed);
}

/// task 261: 旧 4 型の prefill 経路が GPU を諦めた (CPU 経路へ)。
pub fn note_quant_pf_skip() {
    QUANT_PF_SKIPS.fetch_add(1, Ordering::Relaxed);
}

/// task 261: 旧 4 型 prefill の GPU 時間 (ns)。
pub fn note_quant_pf_ns(ns: u64) {
    QUANT_PF_NS.fetch_add(ns, Ordering::Relaxed);
}

/// task 261: 旧 4 型 prefill の 1 行 (1 層も回っていなければ `None`)。
pub fn quant_pf_line() -> Option<String> {
    let layers = QUANT_PF_LAYERS.load(Ordering::Relaxed);
    let skips = QUANT_PF_SKIPS.load(Ordering::Relaxed);
    if layers == 0 && skips == 0 {
        return None;
    }
    let rows = QUANT_PF_ROWS.load(Ordering::Relaxed);
    let hits = QUANT_PF_HITS.load(Ordering::Relaxed);
    let ns = QUANT_PF_NS.load(Ordering::Relaxed);
    Some(format!(
        "[exp-quant-pf] layers={layers} skips={skips} rows={rows} hits={hits} gpu_ms={:.1} rows/layer={:.1}",
        ns as f64 / 1e6,
        if layers > 0 {
            rows as f64 / layers as f64
        } else {
            0.0
        }
    ))
}

/// task 261: 旧 4 型 prefill の積算を 0 に戻す。
pub fn quant_pf_reset() {
    for c in [
        &QUANT_PF_LAYERS,
        &QUANT_PF_ROWS,
        &QUANT_PF_HITS,
        &QUANT_PF_SKIPS,
        &QUANT_PF_NS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

pub fn stats_line() -> Option<String> {
    if !pf_enabled() {
        return None;
    }
    let layers = LAYERS.load(Ordering::Relaxed);
    if layers == 0
        && SKIP_LAYERS.load(Ordering::Relaxed) == 0
        && STREAM_LAYERS.load(Ordering::Relaxed) == 0
    {
        return None;
    }
    let vram = match crate::model::expert_vram::budget_usage() {
        Some((used, pinned, cap)) => format!(
            "vram={}MiB/{}MiB pin={}MiB",
            used >> 20,
            cap >> 20,
            pinned >> 20
        ),
        None => "vram=-".to_string(),
    };
    let lk = LOOKUPS.load(Ordering::Relaxed);
    let hr = if lk > 0 {
        HITS.load(Ordering::Relaxed) as f64 / lk as f64
    } else {
        0.0
    };
    // 診断 (`DENNO_EXPERT_GPU_CHECK=1`) を回したときだけ、末尾に併記する。
    let chk = match take_check() {
        Some(c) => format!(
            " check_rows={} check_argmax={} check_max_abs={:.3e} check_max_rel={:.3e}",
            c.rows, c.argmax, c.max_abs, c.max_rel
        ),
        None => String::new(),
    };
    let mut out = format!(
        "[exp-pf] lookups={lk} hit_rate={hr:.3} gpu_layers={layers} skip_layers={} experts={} \
         rows={} gpu_ms={:.1} kern_ms={:.1} d2h_ms={:.1} add_ms={:.1} uploads={} up_ms={:.1} {vram}{chk}",
        SKIP_LAYERS.load(Ordering::Relaxed),
        EXPERTS.load(Ordering::Relaxed),
        ROWS.load(Ordering::Relaxed),
        GPU_NS.load(Ordering::Relaxed) as f64 / 1e6,
        KERNEL_NS.load(Ordering::Relaxed) as f64 / 1e6,
        D2H_NS.load(Ordering::Relaxed) as f64 / 1e6,
        ADD_NS.load(Ordering::Relaxed) as f64 / 1e6,
        UPLOADS.load(Ordering::Relaxed),
        UPLOAD_NS.load(Ordering::Relaxed) as f64 / 1e6,
    );
    // task 162: ストリーミングの内訳 (task 7 の台帳)。
    // 「1 本あたり何行を載せたか」「上げたバイトと実 H2D 速度」「CPU に残した
    // ミスの ms」を 1 行で読めるようにする。
    let sl = STREAM_LAYERS.load(Ordering::Relaxed);
    if sl > 0 || STREAM_GROUPS.load(Ordering::Relaxed) > 0 {
        let bytes = STREAM_BYTES.load(Ordering::Relaxed);
        let wait = STREAM_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let copy = STREAM_COPY_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let enq = STREAM_ENQ_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let ready = STREAM_READY_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let slotw = STREAM_SLOT_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let use_s = STREAM_USE_NS.load(Ordering::Relaxed) as f64 / 1e9;
        let exps = STREAM_EXPERTS.load(Ordering::Relaxed);
        let rows = STREAM_ROWS.load(Ordering::Relaxed);
        // 1000 倍して持っている MB/s を GB/s へ。
        let probe = STREAM_PROBE_MBPS.load(Ordering::Relaxed) as f64 / 1e6;
        out.push_str(&format!(
            "\n[exp-pf-stream] layers={sl} groups={} experts={exps} rows={rows} \
             rows_per_expert={:.1} up_MB={:.1} copy_s={copy:.2} enq_s={enq:.2} \
             ready_s={ready:.2} wait_s={wait:.2} slotwait_s={slotw:.2} use_s={use_s:.2} \
             probe_GBs={probe:.1} spill={} cpu_miss_ms={:.1} pin={} wc={} spread={} ut={} dec={} gpuw={} nsync={} group={} min_tok={} \
             chunk_kb={} src={} reg_MB={:.1} reg_ok_MB={:.1} reg_calls={} reg_s={:.2} reg_flags={:#x} retry={} wbl={} wbskip={} usplit={} usplit_kb={}",
            STREAM_GROUPS.load(Ordering::Relaxed),
            if exps > 0 {
                rows as f64 / exps as f64
            } else {
                0.0
            },
            bytes as f64 / 1e6,
            STREAM_SPILL.load(Ordering::Relaxed),
            STREAM_CPU_MISS_NS.load(Ordering::Relaxed) as f64 / 1e6,
            stream_pin(),
            u8::from(stream_wc()),
            u8::from(stream_pread()),
            u8::from(stream_ut()),
            u8::from(up_decouple_on()),
            STREAM_GPUWAIT.load(Ordering::Relaxed),
            STREAM_NOSYNC.load(Ordering::Relaxed),
            stream_group(),
            stream_min_tok(),
            stream_chunk_bytes() >> 10,
            stream_src(),
            REG_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
            REG_OK_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
            REG_CALLS.load(Ordering::Relaxed),
            REG_NS.load(Ordering::Relaxed) as f64 / 1e9,
            REG_FLAGS.load(Ordering::Relaxed) as u32,
            STREAM_RETRY.load(Ordering::Relaxed),
            u8::from(gdn_wb_last_on()),
            GDN_WB_SKIPS.load(Ordering::Relaxed),
            UP_SPLIT_CHUNKS.load(Ordering::Relaxed),
            up_split_bytes() >> 10,
        ));
        // task 254: 表の非同期 H2D が発火したか (gate on の証拠)。
        #[cfg(feature = "cuda")]
        {
            let (calls, oks) = crate::cuda::pf_tab_counts();
            if calls > 0 {
                out.push_str(&format!(" tab_async_calls={calls} tab_async_ok={oks}"));
            }
        }
        // task 325: 1 回目の読みの経路 (pread / 先読み) が発火したかの台帳。
        out.push_str(&pread_stats());
        out.push_str(&rc_stats());
        // task 394: 次の層の先読みが発火したかの台帳。
        out.push_str(&layer_pre_stats());
        // task 470: 1 チャンク目の routing による先読みの台帳。
        out.push_str(&early_route_stats());
    }
    Some(out)
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [
        &LOOKUPS,
        &HITS,
        &LAYERS,
        &EXPERTS,
        &ROWS,
        &GPU_NS,
        &KERNEL_NS,
        &D2H_NS,
        &ADD_NS,
        &UPLOADS,
        &UPLOAD_NS,
        &SKIP_LAYERS,
        &STREAM_LAYERS,
        &STREAM_RETRY,
        &GDN_WB_SKIPS,
        &STREAM_EXPERTS,
        &STREAM_BYTES,
        &STREAM_WAIT_NS,
        &STREAM_COPY_NS,
        &STREAM_ENQ_NS,
        &STREAM_READY_NS,
        &STREAM_SLOT_NS,
        &STREAM_GPUWAIT,
        &STREAM_NOSYNC,
        &STREAM_ROWS,
        &STREAM_GROUPS,
        &STREAM_SPILL,
        &STREAM_CPU_MISS_NS,
        &STREAM_PROBE_MBPS,
        &REG_BYTES,
        &REG_OK_BYTES,
        &REG_CALLS,
        &REG_NS,
        &UP_SPLIT_CHUNKS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    // task 470: 1 チャンク目の routing による先読みの台帳も戻す。
    early_route_stats_reset();
    // task 474: 先読みの置き場 (RAM) の台帳も戻す。
    rc_stats_reset();
    CHECK_LAYERS.store(0, Ordering::Relaxed);
    if let Ok(mut g) = CHECK.lock() {
        *g = None;
    }
    // task 254: 表の非同期 H2D の累計もリクエスト単位にする。
    #[cfg(feature = "cuda")]
    crate::cuda::pf_tab_counts_reset();
    // task 325: 1 回目の読みの台帳もリクエスト単位にする。
    pread_stats_reset();
    rc_stats_reset();
    // task 394: 次の層の先読みの台帳もリクエスト単位にする。
    layer_pre_stats_reset();
}

// ---------------------------------------------------------------------------
// 診断: prefill の GPU 行と CPU 行の突き合わせ (`DENNO_EXPERT_GPU_CHECK=1`)
// ---------------------------------------------------------------------------

/// `DENNO_EXPERT_PF_CHECK_LAYERS`: 1 リクエストで突き合わせる層数 (既定 3)。
pub const PF_CHECK_LAYERS_ENV: &str = "DENNO_EXPERT_PF_CHECK_LAYERS";

/// `DENNO_EXPERT_PF_CHECK_ROWS`: 1 層で突き合わせる行数 (先頭から、既定 8)。
pub const PF_CHECK_ROWS_ENV: &str = "DENNO_EXPERT_PF_CHECK_ROWS";

/// [`PF_CHECK_LAYERS_ENV`] の既定値。
pub const DEFAULT_CHECK_LAYERS: usize = 3;

/// [`PF_CHECK_ROWS_ENV`] の既定値。
pub const DEFAULT_CHECK_ROWS: usize = 8;

/// 突き合わせの結果 (1 リクエストぶん)。`max_abs` は全要素の最大絶対差、
/// `max_rel` は `|gpu-cpu| / max(|cpu|, 1e-6)` の最大、`argmax` は行ごとの
/// argmax が一致した数。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PfCheck {
    pub layers: u64,
    pub rows: u64,
    pub argmax: u64,
    pub max_abs: f32,
    pub max_rel: f32,
}

static CHECK: Mutex<Option<PfCheck>> = Mutex::new(None);
static CHECK_LAYERS: AtomicU64 = AtomicU64::new(0);

/// 突き合わせる層数 (`DENNO_EXPERT_PF_CHECK_LAYERS`、既定 3)。
pub fn check_layers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(PF_CHECK_LAYERS_ENV, DEFAULT_CHECK_LAYERS))
}

/// 1 層で突き合わせる行数 (segment あたり、既定 4)。
pub fn check_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| env_usize(PF_CHECK_ROWS_ENV, DEFAULT_CHECK_ROWS))
}

/// この層を突き合わせるか (1 リクエストで先頭 [`check_layers`] 層だけ)。
pub fn claim_check_layer() -> bool {
    let n = check_layers();
    CHECK_LAYERS.fetch_add(1, Ordering::Relaxed) < n as u64
}

/// 1 層ぶんの結果を併合する。
pub fn note_check(c: PfCheck) {
    let mut g = match CHECK.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let a = g.get_or_insert_with(PfCheck::default);
    a.layers += c.layers;
    a.rows += c.rows;
    a.argmax += c.argmax;
    if c.max_abs > a.max_abs {
        a.max_abs = c.max_abs;
    }
    if c.max_rel > a.max_rel {
        a.max_rel = c.max_rel;
    }
}

/// 溜めた結果を取り出す (リクエスト終端の 1 行で使う)。
pub fn take_check() -> Option<PfCheck> {
    CHECK.lock().ok()?.take()
}

/// 1 チャンクぶんの上げ直し予算 (本数)。層 0 で戻す。
///
/// `DENNO_EXPERT_IQ_PF_UPLOAD_MB` を 1 本 2.05 MB で割った本数を上限にする。
pub fn upload_budget_experts() -> usize {
    let mb = upload_mb();
    if mb == 0 {
        return 0;
    }
    (mb << 20) / (2_048_000usize.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// task 394: 戻しの並列化のゲートは既定 on (lab t394a / t394c で採った)。
    /// 明示的な偽値だけ off。スレッド数は 1..=32 に丸める。
    #[test]
    fn restore_mt_gate_truth_table() {
        assert!(DEFAULT_PF_RESTORE_MT, "lab で採ったのに既定 off に戻っている");
        assert!(pf_restore_mt_from_value(None), "未設定は既定 (on)");
        assert!(pf_restore_mt_from_value(Some("")), "空も既定 (on)");
        assert!(!pf_restore_mt_from_value(Some("0")));
        assert!(!pf_restore_mt_from_value(Some("off")));
        assert!(!pf_restore_mt_from_value(Some("false")));
        assert!(!pf_restore_mt_from_value(Some("no")));
        assert!(!pf_restore_mt_from_value(Some(" OFF ")));
        assert!(pf_restore_mt_from_value(Some("1")));
        assert!(pf_restore_mt_from_value(Some("on")));
        assert!(pf_restore_mt_from_value(Some(" true ")));
        assert!(pf_restore_mt_from_value(Some("YES")));
        // スレッド数: 未設定は既定 8、0 も 1 (0 本のプールは作れない)、上限 32。
        assert_eq!(DEFAULT_PF_RESTORE_THREADS, 8);
        assert_eq!(pf_restore_threads_from_value(None), 8);
        assert_eq!(pf_restore_threads_from_value(Some("")), 8);
        assert_eq!(pf_restore_threads_from_value(Some("0")), 1);
        assert_eq!(pf_restore_threads_from_value(Some("1")), 1);
        assert_eq!(pf_restore_threads_from_value(Some(" 12 ")), 12);
        assert_eq!(pf_restore_threads_from_value(Some("999")), 32);
        assert_eq!(pf_restore_threads_from_value(Some("x")), 8);
    }

    /// task 394: 次の層の先読みのゲートは既定 off、明示の真値だけ on。
    #[test]
    fn layer_pre_gate_truth_table() {
        assert!(!DEFAULT_LAYER_PRE, "既定が on になっている");
        assert!(!layer_pre_from_value(None), "未設定は既定 (off)");
        assert!(!layer_pre_from_value(Some("")), "空も既定 (off)");
        assert!(!layer_pre_from_value(Some("0")));
        assert!(!layer_pre_from_value(Some("off")));
        assert!(!layer_pre_from_value(Some("false")));
        assert!(!layer_pre_from_value(Some("no")));
        assert!(layer_pre_from_value(Some("1")));
        assert!(layer_pre_from_value(Some("on")));
        assert!(layer_pre_from_value(Some(" TRUE ")));
        assert!(layer_pre_from_value(Some("yes")));
        // 台帳は積んだぶんだけ増え、0 に戻せる。
        layer_pre_stats_reset();
        assert_eq!(layer_pre_stats(), "", "発火していないのに台帳が出る");
        note_layer_pre(1000, 3);
        note_layer_pre(2000, 3);
        let s = layer_pre_stats();
        assert!(s.contains("lpre_layers=2"), "{s}");
        assert!(s.contains("lpre_calls=6"), "{s}");
        assert!(s.contains("lpre_MB=0.0"), "{s}");
        layer_pre_stats_reset();
        assert_eq!(layer_pre_stats(), "");
    }

    /// task 470 / 474: 1 チャンク目の routing による先読みのゲートは**既定 on**、
    /// `0` 系だけが偽。上限は「明示 MiB > 自動 (残りトークン × KiB)」で、
    /// 自動は下限 / 上限に丸める。
    #[test]
    fn early_route_gate_and_budget() {
        assert!(DEFAULT_EARLY_ROUTE, "task 474 で既定 on にした");
        assert!(early_route_from_value(None), "未設定は既定 (on)");
        assert!(early_route_from_value(Some("")), "空も既定 (on)");
        assert!(!early_route_from_value(Some("0")));
        assert!(!early_route_from_value(Some("off")));
        assert!(!early_route_from_value(Some("no")));
        assert!(!early_route_from_value(Some("false")));
        assert!(early_route_from_value(Some("1")));
        assert!(early_route_from_value(Some("on")));
        assert!(early_route_from_value(Some(" TRUE ")));
        assert!(early_route_from_value(Some("yes")));
        // `2` は task 470 の pread 版の名前 (置き場を使うので 1 と同じ)。
        assert_eq!(early_route_mode_from_value(None), 1);
        assert_eq!(early_route_mode_from_value(Some("")), 1);
        assert_eq!(early_route_mode_from_value(Some("0")), 0);
        assert_eq!(early_route_mode_from_value(Some("1")), 1);
        assert_eq!(early_route_mode_from_value(Some("on")), 1);
        assert_eq!(early_route_mode_from_value(Some("2")), 2);
        assert_eq!(early_route_mode_from_value(Some(" PREAD ")), 2);
        assert_eq!(early_route_mode_from_value(Some("x")), 1, "解析不能は既定");
        assert!(early_route_from_value(Some("2")), "pread も on");
        // 明示 MiB はそのまま (0 は自動)。
        assert_eq!(early_route_mb_from_value(None), 0);
        assert_eq!(early_route_mb_from_value(Some("0")), 0);
        assert_eq!(early_route_mb_from_value(Some("128")), 128);
        assert_eq!(early_route_mb_from_value(Some("x")), 0);
        // KiB/token は数値だけ (解析不能は既定)。
        assert_eq!(early_route_kb_from_value(None), DEFAULT_EARLY_ROUTE_KB);
        assert_eq!(early_route_kb_from_value(Some("0")), 0);
        assert_eq!(early_route_kb_from_value(Some(" 48 ")), 48);
        assert_eq!(early_route_kb_from_value(Some("x")), DEFAULT_EARLY_ROUTE_KB);
        // 自動の上限: 3,826 の 1 チャンク (残り 1,913 tok) と 14k の 6 チャンク。
        assert_eq!(
            early_route_budget(0, 24, 1_913),
            1_913 * 24 * 1024,
            "3,826 の残り 1 チャンク"
        );
        assert_eq!(
            early_route_budget(0, 24, 12_000),
            294_912_000,
            "14k の残り 6 チャンク (288 MiB)"
        );
        assert_eq!(
            early_route_budget(0, 24, 64_000),
            EARLY_ROUTE_MAX_BYTES,
            "長すぎるときは上限 (768 MiB) に当たる"
        );
        assert_eq!(
            early_route_budget(0, 24, 1),
            EARLY_ROUTE_MIN_BYTES,
            "小さすぎるときは下限"
        );
        // 明示 MiB は自動より優先。
        assert_eq!(early_route_budget(200, 24, 1_913), 200 << 20);
        assert_eq!(early_route_budget(0, 0, 12_000), EARLY_ROUTE_MIN_BYTES);
        // 台帳は積んだぶんだけ出て、0 に戻せる。
        early_route_stats_reset();
        assert_eq!(early_route_stats(), "", "発火していないのに台帳が出る");
        note_early_route(40, 128 << 20, 120);
        note_early_route(40, 128 << 20, 120);
        let s = early_route_stats();
        assert!(s.contains("er_layers=2"), "{s}");
        assert!(s.contains("er_experts=80"), "{s}");
        assert!(s.contains("er_calls=240"), "{s}");
        assert!(s.contains("er_MB=268.4"), "{s}");
        early_route_stats_reset();
        assert_eq!(early_route_stats(), "");
    }

    /// task 474: 先読みの置き場 (RAM) の割り当てと索引。
    ///
    /// `rc_layout` は「MoE が読む順の先頭から、置き場に収まる手前まで」を
    /// 割り当てる (expert 昇順の先頭 = 一番早く要るぶん)。索引は
    /// **(パート, 絶対オフセット, 長さ) の完全一致**だけで当たる
    /// (1 バイトでも違えばディスクへ落ちる = 値は絶対に変わらない)。
    #[test]
    fn ra_cache_layout_and_index() {
        // 1 本 100 バイト × 3 本。200 バイトの置き場には 2 本しか入らない。
        let runs = [(0u32, 1_000u64, 100usize), (0, 2_000, 100), (1, 3_000, 100)];
        let (jobs, total) = rc_layout(&runs, 200).expect("2 本は入る");
        assert_eq!(total, 200);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0], (0u32, 0u32, 1_000u64, 100u32), "先頭は置き場の 0");
        assert_eq!(jobs[1], (100, 0, 2_000, 100), "次は前の直後");
        // ぴったり 300 バイトなら 3 本。
        assert_eq!(rc_layout(&runs, 300).map(|(j, t)| (j.len(), t)), Some((3, 300)));
        // 1 本も入らない / 空。
        assert!(rc_layout(&runs, 99).is_none());
        assert!(rc_layout(&runs, 0).is_none());
        assert!(rc_layout(&[], 1 << 20).is_none());
        // 長さ 0 の範囲は飛ばす (置き場を無駄にしない)。
        let z = [(0u32, 10u64, 0usize), (0, 20, 8)];
        let (j, t) = rc_layout(&z, 64).expect("1 本");
        assert_eq!((j.len(), t), (1, 8));
        assert_eq!(j[0].0, 0, "長さ 0 は場所を食わない");

        // 索引: 完全一致だけが当たる。
        rc_clear();
        assert!(!rc_has(0, 1_000, 100), "空なら当たらない");
        rc_commit_one(0, 1_000, 100, 0);
        rc_commit_one(1, 3_000, 100, 100);
        assert!(rc_has(0, 1_000, 100));
        assert!(rc_has(1, 3_000, 100));
        assert!(!rc_has(0, 1_000, 99), "長さが違えば当たらない");
        assert!(!rc_has(0, 1_001, 100), "位置が違えば当たらない");
        assert!(!rc_has(2, 1_000, 100), "パートが違えば当たらない");
        // 台帳 (載ったバイト) が積まれる。
        let s = rc_stats();
        assert!(s.contains("ra_MB=0.0"), "{s}");
        assert!(s.contains("ra_hit=0"), "{s}");
        rc_stats_reset();
        rc_clear();
        assert_eq!(rc_stats(), "", "空に戻したら台帳も空");
        assert!(!rc_has(0, 1_000, 100), "clear で消える");
    }

    /// task 474: 置き場の大きさと並列度の env 解釈。
    #[test]
    fn ra_cache_env_truth_table() {
        assert_eq!(rc_cache_mb_from_value(None), DEFAULT_RC_CACHE_MB);
        assert_eq!(rc_cache_mb_from_value(Some("")), DEFAULT_RC_CACHE_MB);
        assert_eq!(rc_cache_mb_from_value(Some("0")), 0, "0 は置き場なし");
        assert_eq!(rc_cache_mb_from_value(Some("256")), 256);
        assert_eq!(rc_cache_mb_from_value(Some(" 64 ")), 64);
        assert_eq!(rc_cache_mb_from_value(Some("1")), 32, "下限 32 MiB");
        assert_eq!(rc_cache_mb_from_value(Some("99999")), 2048, "上限 2048 MiB");
        assert_eq!(rc_cache_mb_from_value(Some("x")), DEFAULT_RC_CACHE_MB);
        assert_eq!(rc_threads_from_value(None), DEFAULT_RC_THREADS);
        assert_eq!(rc_threads_from_value(Some("0")), DEFAULT_RC_THREADS);
        assert_eq!(rc_threads_from_value(Some("4")), 4);
        assert_eq!(rc_threads_from_value(Some("999")), 32, "上限 32");
        assert_eq!(rc_threads_from_value(Some("x")), DEFAULT_RC_THREADS);
    }

    /// task 474: 窓の下限 (残りチャンク 4 本)。
    ///
    /// 3,826 (1,913 トークンのチャンク 2 本 = 残り 1 本) は窓が短すぎて
    /// 先読みがはみ出す (t474a で ±0)。14k (2,003 × 7) は残り 6 本で収まる。
    #[test]
    fn early_route_window_gate() {
        assert!(!early_route_window_ok(1_913, 3_826), "3,826 は残り 1 チャンク");
        assert!(early_route_window_ok(2_003, 14_018), "14k は残り 6 チャンク");
        assert!(!early_route_window_ok(0, 14_018), "チャンク 0 は窓なし");
        assert!(!early_route_window_ok(2_048, 2_048), "1 チャンクは窓なし");
        assert!(early_route_window_ok(2_048, 10_240), "残り 4 本ちょうどは可");
        assert!(!early_route_window_ok(2_048, 10_239), "残り 4 本未満は不可");
    }

    /// task 325: `DENNO_PF_RA_GROUPS` の真理値表 (未設定は off)。
    #[test]
    fn ra_groups_gate_truth_table() {
        assert_eq!(ra_groups_from_value(None), 0);
        assert_eq!(ra_groups_from_value(Some("0")), 0);
        assert_eq!(ra_groups_from_value(Some("1")), 1);
        assert_eq!(ra_groups_from_value(Some(" 3 ")), 3);
        assert_eq!(ra_groups_from_value(Some("999")), 64, "上限 64");
        assert_eq!(ra_groups_from_value(Some("x")), 0, "解析不能は既定");
    }

    /// task 333: `DENNO_PF_STREAM_RETRY` は既定 on、`0` 系だけが偽。
    #[test]
    fn stream_retry_gate_truth_table() {
        assert!(stream_retry_from_value(None), "未設定は既定 (on)");
        assert!(stream_retry_from_value(Some("")), "空も既定 (on)");
        assert!(!stream_retry_from_value(Some("0")));
        assert!(!stream_retry_from_value(Some("off")));
        assert!(!stream_retry_from_value(Some("false")));
        assert!(!stream_retry_from_value(Some("no")));
        assert!(stream_retry_from_value(Some("1")));
        assert!(stream_retry_from_value(Some(" on ")));
        assert!(stream_retry_from_value(Some("x")), "解析不能は既定 (on)");
    }

    /// task 333: 再試行して意味がある門だけ true (転送の担い手を替えて直るもの)。
    #[test]
    fn retry_pays_only_for_the_transfer_guards() {
        assert!(stream_retry_pays("no_upload"), "上げスレッドが居ない");
        assert!(stream_retry_pays("ptr_table"), "ポインタ表の不整合");
        // 1 グループでも計算した後は `out` を触っているので再試行しない
        // (呼び出し側の従来経路と二重加算になる)。
        assert!(!stream_retry_pays("no_upload_late"));
        assert!(!stream_retry_pays("ptr_table_late"));
        for why in [
            "stage",
            "no_miss",
            "dead",
            "type",
            "slot",
            "ctx",
            "kernels",
            "accum",
            "current",
            "row_bytes",
            "shape",
            "bounds",
            "group_rows",
            "seg_local",
            "args",
            "entries",
            "plan",
            "row_table",
            "shared_table",
            "host_missing",
            "no_bytes",
            "",
        ] {
            assert!(!stream_retry_pays(why), "{why} は再試行しない");
        }
    }

    /// task 420: `DENNO_PF_POOL_K_PLAN` は既定 on、**明示的な偽値だけ** off
    /// (lab t420b で採用。task 265 の 3 本と同じ規約)。
    #[test]
    fn pool_k_plan_gate_truth_table() {
        assert!(pool_k_plan_from_value(None), "未設定は既定 (on)");
        assert!(pool_k_plan_from_value(Some("")), "空も既定 (on)");
        assert!(!pool_k_plan_from_value(Some("0")));
        assert!(!pool_k_plan_from_value(Some("off")));
        assert!(!pool_k_plan_from_value(Some("FALSE")));
        assert!(!pool_k_plan_from_value(Some("no")));
        assert!(pool_k_plan_from_value(Some("x")), "解析不能は既定 (on)");
        assert!(pool_k_plan_from_value(Some("1")));
        assert!(pool_k_plan_from_value(Some("on")));
        assert!(pool_k_plan_from_value(Some("YES")));
        assert!(pool_k_plan_from_value(Some(" 1 ")), "前後の空白は無視する");
        assert!(DEFAULT_POOL_K_PLAN, "既定は on (lab t420b)");
    }

    /// task 325: `DENNO_PF_PREAD` は既定 on、`0` 系だけが偽。
    #[test]
    fn pread_gate_truth_table() {
        assert!(pread_from_value(None), "未設定は既定 (on)");
        assert!(pread_from_value(Some("")), "空も既定 (on)");
        assert!(!pread_from_value(Some("0")));
        assert!(!pread_from_value(Some("off")));
        assert!(!pread_from_value(Some("false")));
        assert!(!pread_from_value(Some("no")));
        assert!(pread_from_value(Some("1")));
        assert!(pread_from_value(Some("on")));
        assert!(pread_from_value(Some("ON")));
        assert!(pread_from_value(Some("x")), "解析不能は既定 (on)");
    }

    /// task 325: `DENNO_PF_PREAD_KB` の丸めと既定。
    #[test]
    fn pread_kb_rounds_and_defaults() {
        assert_eq!(pread_kb_from_value(None), DEFAULT_PREAD_KB);
        assert_eq!(pread_kb_from_value(Some("")), DEFAULT_PREAD_KB);
        assert_eq!(pread_kb_from_value(Some("0")), DEFAULT_PREAD_KB);
        assert_eq!(pread_kb_from_value(Some("1")), 64, "下限 64 KiB");
        assert_eq!(pread_kb_from_value(Some("1024")), 1024);
        assert_eq!(pread_kb_from_value(Some("999999")), 65536, "上限 64 MiB");
        assert_eq!(pread_kb_from_value(Some("x")), DEFAULT_PREAD_KB);
    }

    /// task 349: `DENNO_PF_STREAM_PREAD` は既定 off、`1` 系だけが真。
    #[test]
    fn stream_pread_gate_truth_table() {
        assert!(!stream_pread_from_value(None), "未設定は既定 (off)");
        assert!(!stream_pread_from_value(Some("")), "空も既定 (off)");
        assert!(!stream_pread_from_value(Some("0")));
        assert!(!stream_pread_from_value(Some("off")));
        assert!(!stream_pread_from_value(Some("x")), "解析不能は既定 (off)");
        assert!(stream_pread_from_value(Some("1")));
        assert!(stream_pread_from_value(Some("on")));
        assert!(stream_pread_from_value(Some("TRUE")));
        assert!(stream_pread_from_value(Some(" yes ")));
    }

    /// task 415: `DENNO_PF_EMBED_PAR` の丸めと既定 (16 本。lab t415b で採用)。
    #[test]
    fn task415_embed_par_rounds_and_defaults() {
        assert_eq!(embed_par_from_value(None), 16, "未設定は既定 16");
        assert_eq!(embed_par_from_value(Some("")), 16);
        assert_eq!(embed_par_from_value(Some("x")), 16, "解析不能は既定");
        assert_eq!(embed_par_from_value(Some("0")), 0, "0 は直列 (切り戻し)");
        assert_eq!(embed_par_from_value(Some("1")), 1, "1 も直列 (プールを作らない)");
        assert_eq!(embed_par_from_value(Some(" 32 ")), 32);
        assert_eq!(embed_par_from_value(Some("999")), 64, "上限 64");
        assert_eq!(EMBED_PAR_ENV, "DENNO_PF_EMBED_PAR");
        assert_eq!(DEFAULT_EMBED_PAR, 16);
    }

    /// task 415: `DENNO_PF_EMBED_OVL` は既定 on (lab t415b で採用)、0 系だけが偽。
    #[test]
    fn task415_embed_ovl_gate_truth_table() {
        assert!(embed_ovl_from_value(None), "未設定は既定 (on)");
        assert!(embed_ovl_from_value(Some("")), "空も既定 (on)");
        assert!(embed_ovl_from_value(Some("x")), "解析不能は既定 (on)");
        assert!(!embed_ovl_from_value(Some("0")));
        assert!(!embed_ovl_from_value(Some("off")));
        assert!(embed_ovl_from_value(Some("1")));
        assert!(embed_ovl_from_value(Some(" TRUE ")));
        assert_eq!(EMBED_OVL_ENV, "DENNO_PF_EMBED_OVL");
        assert!(DEFAULT_EMBED_OVL);
    }

    /// task 423: `DENNO_PF_PLE_LJOIN` は既定 on (t423a/t423b で採用)、`0` 系だけが偽。
    #[test]
    fn task423_ple_ljoin_gate_truth_table() {
        assert!(ple_ljoin_from_value(None), "未設定は既定 (on)");
        assert!(ple_ljoin_from_value(Some("")), "空も既定 (on)");
        assert!(ple_ljoin_from_value(Some("x")), "解析不能は既定 (on)");
        assert!(!ple_ljoin_from_value(Some("0")));
        assert!(!ple_ljoin_from_value(Some("off")));
        assert!(!ple_ljoin_from_value(Some(" FALSE ")));
        assert!(!ple_ljoin_from_value(Some("no")));
        assert!(ple_ljoin_from_value(Some("1")));
        assert!(ple_ljoin_from_value(Some("on")));
        assert_eq!(PLE_LJOIN_ENV, "DENNO_PF_PLE_LJOIN");
        assert!(DEFAULT_PLE_LJOIN, "lab t423a/t423b で採ったので既定 on");
    }

    /// task 349 / 353: `DENNO_PF_PLE_PIPE` は既定 on (t353a で採用)、`0` 系だけが偽。
    #[test]
    fn ple_pipe_gate_truth_table() {
        assert!(ple_pipe_from_value(None), "未設定は既定 (on)");
        assert!(ple_pipe_from_value(Some("")), "空も既定 (on)");
        assert!(!ple_pipe_from_value(Some("0")));
        assert!(!ple_pipe_from_value(Some("off")));
        assert!(!ple_pipe_from_value(Some("x")), "解析不能は偽 (明示の真値だけ on)");
        assert!(ple_pipe_from_value(Some("1")));
        assert!(ple_pipe_from_value(Some("on")));
        assert!(ple_pipe_from_value(Some("TRUE")));
        assert!(ple_pipe_from_value(Some(" yes ")));
        assert_eq!(PLE_PIPE_ENV, "DENNO_PF_PLE_PIPE");
        assert!(DEFAULT_PLE_PIPE);
    }

    /// task 357 / 419: `DENNO_PF_PLE_THREADS` の丸めと既定 (128 本)。
    #[test]
    fn ple_threads_rounds_and_defaults() {
        assert_eq!(ple_threads_from_value(None), 128, "未設定は既定 128");
        assert_eq!(ple_threads_from_value(Some("")), 128);
        assert_eq!(ple_threads_from_value(Some("x")), 128, "解析不能は既定");
        assert_eq!(ple_threads_from_value(Some("0")), 128, "0 は既定へ");
        assert_eq!(ple_threads_from_value(Some("64")), 64);
        assert_eq!(ple_threads_from_value(Some(" 32 ")), 32, "切り戻しは 32");
        assert_eq!(ple_threads_from_value(Some("999")), 128, "上限 128");
        assert_eq!(PLE_THREADS_ENV, "DENNO_PF_PLE_THREADS");
        assert_eq!(DEFAULT_PLE_THREADS, 128, "task 419 で 128 にした");
    }

    /// task 325 / 358: `DENNO_PF_PREAD_THREADS` の丸めと既定 (32 本)。
    ///
    /// task 358 で既定を 0 (共有プール 14 本) から 32 本へ変えた (lab t358b:
    /// 3,826 -14.6%、14k -15.0%)。**`0` は明示的な切り戻し**として 0 を返す
    /// (未設定・空・解析不能だけが新しい既定へ行く)。
    #[test]
    fn pread_threads_rounds_and_defaults() {
        assert_eq!(pread_threads_from_value(None), 32, "未設定は新しい既定 32");
        assert_eq!(pread_threads_from_value(Some("")), 32);
        assert_eq!(pread_threads_from_value(Some("0")), 0, "0 = 共有プールへ戻す");
        assert_eq!(pread_threads_from_value(Some("32")), 32);
        assert_eq!(pread_threads_from_value(Some(" 16 ")), 16);
        assert_eq!(pread_threads_from_value(Some("999")), 64, "上限 64");
        assert_eq!(pread_threads_from_value(Some("x")), 32);
        assert_eq!(PREAD_THREADS_ENV, "DENNO_PF_PREAD_THREADS");
        assert_eq!(DEFAULT_PREAD_THREADS, 32);
    }

    /// task 325: 塊に切っても合計は元の長さで、隙間も重なりも無い。
    #[test]
    fn pread_blocks_cover_exactly() {
        for (len, blk) in [(0usize, 64usize), (1, 64), (63, 64), (64, 64), (65, 64), (1000, 300)] {
            let v = pread_blocks(len, blk);
            let total: usize = v.iter().map(|&(_, n)| n).sum();
            assert_eq!(total, len, "len={len} blk={blk}");
            let mut at = 0usize;
            for (off, n) in v {
                assert_eq!(off, at, "隙間か重なりがある (len={len} blk={blk})");
                assert!(n > 0 && n <= blk.max(1));
                at += n;
            }
            assert_eq!(at, len);
        }
        // 0 バイトは 0 個 (空の区間を作らない)。
        assert!(pread_blocks(0, 4096).is_empty());
    }

    fn by_expert_fixture() -> Vec<Vec<(usize, usize)>> {
        // expert 0: 3 行、1: 0 行、2: 5 行、3: 1 行、4: 2 行
        let mut v: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 5];
        v[0] = vec![(0, 0), (2, 1), (9, 2)];
        v[2] = vec![(0, 1), (1, 0), (3, 3), (4, 1), (9, 0)];
        v[3] = vec![(7, 0)];
        v[4] = vec![(5, 2), (6, 1)];
        v
    }

    /// 選んだ expert の行だけが 1 回ずつ入り、`seg` の row_off が詰まっていること。
    #[test]
    fn plan_keeps_expert_order_and_offsets() {
        let be = by_expert_fixture();
        let pick = [true, true, false, true, true];
        let plan = build_plan(&be, &pick);
        assert_eq!(plan.nseg(), 3, "空の expert 1 と未選択の expert 2 は入らない");
        assert_eq!(plan.experts, vec![0, 3, 4]);
        assert_eq!(plan.total_rows, 3 + 1 + 2);
        assert_eq!(plan.seg.len(), 3 * 4);
        assert_eq!(&plan.seg[0..4], &[3, 0, 0, 0]);
        assert_eq!(&plan.seg[4..8], &[1, 3, 0, 0]);
        assert_eq!(&plan.seg[8..12], &[2, 4, 0, 0]);
        assert_eq!(plan.rows, vec![0, 2, 9, 7, 5, 6]);
        // 各行はちょうど 1 回、(expert, token) の対応が保たれている
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for (s, &ei) in plan.experts.iter().enumerate() {
            let m = plan.seg[s * 4] as usize;
            let off = plan.seg[s * 4 + 1] as usize;
            for r in 0..m {
                let t = plan.rows[off + r] as usize;
                seen.push((ei, t));
            }
        }
        let mut want: Vec<(usize, usize)> = be
            .iter()
            .enumerate()
            .filter(|(i, _)| pick[*i])
            .flat_map(|(i, v)| v.iter().map(move |&(t, _)| (i, t)))
            .collect();
        seen.sort_unstable();
        want.sort_unstable();
        assert_eq!(seen, want, "行の取りこぼし・重複がある");
    }

    /// task 242: グループ局所の g/u/act に要る行数は「グループの最大行数」。
    ///
    /// カーネルは行の添字が絶対なので、呼び出し側は
    /// `ポインタ - グループ先頭行 * n_out * 4` で渡す。この行数だけ持てば、
    /// どのグループも自分の範囲しか触らない。
    #[test]
    fn max_group_rows_is_the_largest_group_span() {
        // 6 expert / 行数 3,1,2,4,1,2 (合計 13)。
        let seg: Vec<u32> = vec![
            3, 0, 0, 0, // rows 0..3
            1, 3, 0, 0, // rows 3..4
            2, 4, 0, 0, // rows 4..6
            4, 6, 0, 0, // rows 6..10
            1, 10, 0, 0, // rows 10..11
            2, 11, 0, 0, // rows 11..13
        ];
        // 2 グループ: 0..6 は 6 行、6..13 は 7 行。
        assert_eq!(max_group_rows(&seg, &[(0, 3), (3, 6)]), Some(7));
        // 1 グループなら全体 (13 行)。
        assert_eq!(max_group_rows(&seg, &[(0, 6)]), Some(13));
        // 3 グループ: 6 / 4 / 2 行。
        assert_eq!(max_group_rows(&seg, &[(0, 3), (3, 4), (4, 6)]), Some(6));
        // 壊れた入力は None (呼び出し側は従来の全体サイズへ落ちる)。
        assert_eq!(max_group_rows(&seg, &[]), None);
        assert_eq!(max_group_rows(&seg[..11], &[(0, 3)]), None);
        assert_eq!(max_group_rows(&seg, &[(0, 9)]), None);
        assert_eq!(max_group_rows(&seg, &[(2, 2)]), None);
        // group_bounds の結果と組み合わせても一致する。
        let needs = vec![100usize, 100, 100, 100, 100, 100];
        let bounds = group_bounds(&needs, 350, 64).expect("3 本ずつ 2 グループ");
        assert_eq!(bounds, vec![(0, 3), (3, 6)]);
        assert_eq!(max_group_rows(&seg, &bounds), Some(7));
    }

    /// task 242: グループ局所の `seg` は各行の先頭がグループ先頭から 0 起点に
    /// なり、行表をグループ先頭行だけ足して渡せば元の行に対応する。
    #[test]
    fn seg_group_local_rebases_each_group_to_zero() {
        let seg: Vec<u32> = vec![
            3, 0, 0, 0, // rows 0..3
            1, 3, 0, 0, // rows 3..4
            2, 4, 0, 0, // rows 4..6
            4, 6, 0, 0, // rows 6..10
            1, 10, 0, 0, // rows 10..11
            2, 11, 0, 0, // rows 11..13
        ];
        let bounds = vec![(0usize, 3usize), (3, 6)];
        let gl = seg_group_local(&seg, &bounds).expect("直せる");
        assert_eq!(gl.len(), seg.len());
        // 行数は変えない。
        for i in 0..6 {
            assert_eq!(gl[i * 4], seg[i * 4], "segment {i} の行数が変わった");
        }
        // グループ 0 は元のまま (先頭が 0)、グループ 1 は 6 を引く。
        assert_eq!(&gl[0..12], &seg[0..12]);
        assert_eq!(&gl[12..24], &[4, 0, 0, 0, 1, 4, 0, 0, 2, 5, 0, 0]);
        // 相対化しても「グループの最大行数」は同じ。
        assert_eq!(max_group_rows(&gl, &bounds), max_group_rows(&seg, &bounds));
        // 1 グループなら全体が 0 起点 (先頭は元から 0 なので不変)。
        assert_eq!(seg_group_local(&seg, &[(0, 6)]).as_deref(), Some(&seg[..]));
        // 壊れた入力は None。
        assert_eq!(seg_group_local(&seg, &[]), None);
        assert_eq!(seg_group_local(&seg[..11], &[(0, 3)]), None);
        assert_eq!(seg_group_local(&seg, &[(0, 9)]), None);
        // 先頭より小さい行番号が来たら None (plan の並びが壊れている)。
        // segment 4 はグループ 1 (先頭 6) の中なので、0 は先頭より小さい。
        let mut bad = seg.clone();
        bad[4 * 4 + 1] = 0;
        assert_eq!(seg_group_local(&bad, &[(0, 3), (3, 6)]), None);
    }

    /// task 242: グループ局所の添字をカーネルと同じ式で辿り、どの領域も
    /// はみ出さないこと (lab t242a は down の行表を足して error 700 になった)。
    ///
    /// - gate/up: 出力 = `(rel + r) * ff + row0` → `rows_cap * ff` 未満
    /// - gate/up: 入力 = `x[rows_gu[abs_start + rel + r]]` → `ntok` 未満
    /// - down: 入力 = `act[rows_id[rel + r]]` = `act[rel + r]` → グループ局所
    ///   (`abs_start` を足したら act の範囲外になる)
    #[test]
    fn group_local_indices_stay_inside_the_group_scratch() {
        // 5 expert / 行数 40,10,80,25,5 (= 160 行) / トークンは 7 個を巡回。
        let rows_per = [40usize, 10, 80, 25, 5];
        let ntok = 7usize;
        let (ff, e) = (4usize, 3usize);
        let mut by_expert: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut t = 0usize;
        for &m in rows_per.iter() {
            let mut v = Vec::new();
            for _ in 0..m {
                v.push((t % ntok, 0));
                t += 1;
            }
            by_expert.push(v);
        }
        let pick = vec![true; 5];
        let plan = build_plan(&by_expert, &pick);
        let total_rows = plan.total_rows;
        assert_eq!(total_rows, 160);
        let needs = vec![100usize; 5];
        let bounds = group_bounds(&needs, 250, 64).expect("2 本ずつ + 端数");
        let gl = seg_group_local(&plan.seg, &bounds).expect("直せる");
        let cap = max_group_rows(&plan.seg, &bounds).expect("行数");
        let mut seen = 0usize;
        for &(a, b) in &bounds {
            let abs_start = plan.seg[a * 4 + 1] as usize;
            for i in a..b {
                let m = gl[i * 4] as usize;
                let rel0 = gl[i * 4 + 1] as usize;
                for r in 0..m {
                    let rel = rel0 + r;
                    // gate/up の出力はグループの作業域に収まる。
                    assert!(rel < cap, "g/u の行 {rel} が cap {cap} を超えた");
                    assert!(rel * ff < cap * ff);
                    // gate/up の入力は絶対の行表 (abs_start を足したもの) で引く。
                    let abs_row = abs_start + rel;
                    assert_eq!(
                        plan.rows[abs_row], plan.rows[abs_start + rel0 + r],
                        "行表の対応がずれた"
                    );
                    assert!((plan.rows[abs_row] as usize) < ntok);
                    // down の入力はグループ局所の act (identity のまま)。
                    assert_eq!(rel, abs_row - abs_start);
                    assert!(rel * e < cap * e, "act の行 {rel} が範囲外");
                    // down の出力は全体の h (絶対)。
                    assert!((abs_row) * e < (total_rows + ntok) * e);
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, total_rows, "全ての行を 1 回ずつ辿る");
    }

    /// task 233: 行の加重和の表に共有 expert を 1 列足す。行は
    /// 「routed (expert 昇順) のあと、各トークンの共有行」で、係数は sh_w。
    #[test]
    fn row_table_with_shared_appends_one_term_per_token() {
        let (ntok, k) = (3usize, 2usize);
        let prow: Vec<u32> = vec![0, 5, 1, 6, 2, 7];
        let wrow: Vec<f32> = vec![0.5, 0.5, 0.25, 0.75, 1.0, 0.0];
        let sh: Vec<f32> = vec![0.9, 0.8, 0.7];
        let (pr, wr) = row_table_with_shared(&prow, &wrow, ntok, k, 6, &sh).expect("足せる");
        assert_eq!(pr.len(), ntok * (k + 1));
        assert_eq!(wr.len(), ntok * (k + 1));
        for t in 0..ntok {
            assert_eq!(
                &pr[t * 3..t * 3 + 2],
                &prow[t * 2..(t + 1) * 2],
                "routed の行が動いた"
            );
            assert_eq!(wr[t * 3], wrow[t * 2], "routed の重みが動いた");
            assert_eq!(wr[t * 3 + 1], wrow[t * 2 + 1]);
            assert_eq!(pr[t * 3 + 2], (6 + t) as u32, "共有の行は h の末尾");
            assert_eq!(wr[t * 3 + 2], sh[t], "共有の係数");
        }
        // 形が合わないときは None (黙って詰めない)。
        assert!(row_table_with_shared(&prow, &wrow, ntok, k, 6, &sh[..2]).is_none());
        assert!(row_table_with_shared(&prow[..5], &wrow, ntok, k, 6, &sh).is_none());
        assert!(row_table_with_shared(&prow, &wrow, ntok, 0, 6, &sh).is_none());
    }

    /// 全部ヒットなら行数は選択数 (nt x k) と一致する。
    #[test]
    fn plan_all_hits_covers_every_pair() {
        let mut be: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 4];
        for t in 0..32usize {
            for j in 0..3usize {
                be[(t * 3 + j) % 4].push((t, j));
            }
        }
        let pick = [true; 4];
        let plan = build_plan(&be, &pick);
        assert_eq!(plan.total_rows, 96);
        assert_eq!(plan.nseg(), 4);
        assert_eq!(identity_rows(4), vec![0, 1, 2, 3]);
    }

    /// 起動形: 256 ブロック型は 8 行 / block、レーンブロック型は 4 行 / block。
    #[test]
    fn launch_plan_shapes() {
        let p = launch_plan(2, 200, 640).expect("plan");
        assert_eq!(p.grid, (80, 200, 1));
        assert_eq!(p.block, (128, 1, 1));
        let q = launch_plan(2, 200, 2560).expect("plan");
        assert_eq!(q.grid, (320, 200, 1));
        // down は IQ4_NL / Q2_0 (スロット 4/5) = 4 行 / block
        let r = launch_plan(4, 200, 2560).expect("plan");
        assert_eq!(r.grid, (640, 200, 1));
        assert_eq!(tile_of(0), 8);
        assert_eq!(tile_of(5), 4);
        assert!(launch_plan(0, 0, 640).is_none());
        assert!(launch_plan(0, 1, 0).is_none());
        assert!(launch_plan(0, 65536, 640).is_none());
        // タイルで割り切れない形は起動しない (端数は CPU で計算する)
        assert!(launch_plan(0, 1, 641).is_none());
        assert!(launch_plan(4, 1, 642).is_none());
    }

    /// task 309: K 系のスロット・カーネル名・起動形 (UD-Q4_K_XL の実形)。
    #[test]
    fn k_slots_names_and_plans() {
        use crate::kquant::GgmlType as T;
        assert_eq!(slot(T::Q4K), Some(K_SLOT0));
        assert_eq!(slot(T::Q5K), Some(K_SLOT0 + 1));
        assert_eq!(slot(T::Q6K), Some(K_SLOT0 + 2));
        assert_eq!(slot(T::Q5_1), Some(K_SLOT0 + 3));
        assert_eq!(slot(T::Q8_0), Some(K_SLOT0 + 4));
        // IQ 系のスロットは動かない (既存のカーネル名と一致)。
        assert_eq!(slot(T::IQ3S), Some(2));
        assert_eq!(kernel_name(2), Some("dno_exp_pf_iq3s"));
        assert_eq!(kernel_name(5), Some("dno_exp_pf_q2_0"));
        assert_eq!(kernel_name(6), Some("dno_exp_k_pf_q4k"));
        assert_eq!(kernel_name(7), Some("dno_exp_k_pf_q5k"));
        assert_eq!(kernel_name(8), Some("dno_exp_k_pf_q6k"));
        assert_eq!(kernel_name(9), Some("dno_exp_k_pf_q5_1"));
        assert_eq!(kernel_name(10), Some("dno_exp_k_pf_q8_0"));
        assert_eq!(kernel_name(SLOTS), None, "スロットの外は名前を持たない");
        assert!(is_k_slot(K_SLOT0) && is_k_slot(SLOTS - 1) && !is_k_slot(IQ_SLOTS - 1));
        // 256 要素ブロックの型は 8 行 / block、32 要素ブロックの型は 4 行 / block。
        assert_eq!(tile_of(K_SLOT0), 8, "Q4_K");
        assert_eq!(tile_of(K_SLOT0 + 2), 8, "Q6_K");
        assert_eq!(tile_of(K_SLOT0 + 3), 4, "Q5_1");
        assert_eq!(tile_of(K_SLOT0 + 4), 4, "Q8_0");
        // UD-Q4_K_XL の形: gate/up は k=2560 → n=640、down は k=640 → n=2560。
        let g = launch_plan(slot(T::Q4K).unwrap(), 200, 640).expect("gate の形");
        assert_eq!(g.grid, (80, 200, 1));
        assert_eq!(g.block, (128, 1, 1));
        let d = launch_plan(slot(T::Q5_1).unwrap(), 200, 2560).expect("down の形");
        assert_eq!(d.grid, (640, 200, 1));
        // k の割り切り: Q4_K は 256 の倍数だけ、Q5_1 は 32 の倍数だけ。
        assert!(row_bytes(T::Q4K, 2560).is_some());
        assert!(row_bytes(T::Q4K, 640).is_none(), "down を Q4_K で読んではいけない");
        assert_eq!(row_bytes(T::Q5_1, 640), Some(480));
        assert_eq!(row_bytes(T::Q8_0, 2560), Some(2720));
        assert_eq!(row_bytes(T::Q6K, 2560), Some(2100));
    }

    /// task 419: IQ 系の行数の床は既定 0 (床なし)。値はそのまま読む。
    #[test]
    fn iq_min_rows_truth_table() {
        assert_eq!(iq_min_rows_from_value(None), 0, "未設定は床なし");
        assert_eq!(iq_min_rows_from_value(Some("")), 0);
        assert_eq!(iq_min_rows_from_value(Some("x")), 0, "解析不能は床なし");
        assert_eq!(iq_min_rows_from_value(Some("0")), 0, "0 は床なし");
        assert_eq!(iq_min_rows_from_value(Some("16")), 16);
        assert_eq!(iq_min_rows_from_value(Some(" 32 ")), 32, "空白は落とす");
        assert_eq!(DEFAULT_IQ_MIN_ROWS, 0, "既定が動いている");
        assert_eq!(IQ_MIN_ROWS_ENV, "DENNO_PF_EXP_IQ_MIN_ROWS");
        // 床の判定は K 系と同じ関数 (行 0 は回さない)。
        assert!(!k_expert_pays(15, 16));
        assert!(k_expert_pays(16, 16));
        assert!(k_expert_pays(1, 0), "床 0 は全部 GPU");
    }

    /// task 309: K 系のゲートは IQ ゲートの内側・既定 off・明示の真値だけ on。
    #[test]
    fn k_gate_truth_table() {
        assert!(!k_from_value(false, Some("1")), "IQ ゲート off なら動かない");
        assert!(!k_from_value(false, None));
        assert_eq!(k_from_value(true, None), DEFAULT_K_ON);
        assert_eq!(k_from_value(true, Some("")), DEFAULT_K_ON);
        assert!(k_from_value(true, Some("1")));
        assert!(k_from_value(true, Some("on")));
        assert!(!k_from_value(true, Some("0")));
        assert!(!k_from_value(true, Some("off")));
        assert!(!k_from_value(true, Some("false")));
        // task 309: lab t309b で採用して既定 on (IQ ゲートが前提なので、
        // env 未設定でも `DENNO_EXPERT_IQ_GPU=1` が無ければ k_enabled() は false)。
        assert!(DEFAULT_K_ON, "既定 off に戻っている");
        assert!(k_from_value(true, None), "未設定は既定 (on) でない");
        // write-combined の既定は型で変わる (K 系は通常 pinned)。
        assert!(!stream_wc_for_from(None, true, true), "K 系の既定が WC");
        assert!(stream_wc_for_from(None, true, false), "IQ 系の既定が WC でない");
        assert!(stream_wc_for_from(Some("1"), true, true), "明示値は型より優先");
        assert!(!stream_wc_for_from(Some("0"), true, false), "明示値は型より優先");
        assert!(!stream_wc_for_from(None, false, false), "ストリーミング off");
        // 行数の床 (純関数)。
        assert_eq!(k_min_rows_from_value(None), DEFAULT_K_MIN_ROWS);
        assert_eq!(k_min_rows_from_value(Some("")), DEFAULT_K_MIN_ROWS);
        assert_eq!(k_min_rows_from_value(Some("16")), 16);
        assert_eq!(k_min_rows_from_value(Some("0")), 0);
        assert_eq!(k_min_rows_from_value(Some("x")), DEFAULT_K_MIN_ROWS);
        assert_eq!(DEFAULT_K_MIN_ROWS, 8);
        assert!(!k_expert_pays(7, 8), "床未満は CPU");
        assert!(k_expert_pays(8, 8));
        assert!(!k_expert_pays(0, 0), "行の無い expert は回さない");
        assert!(k_expert_pays(3, 0), "床 0 は全部 GPU");
    }

    /// task 334: 段の終端の待ちを外す 3 本のゲート (lab t334 で採って**既定 on**)と、
    /// 測り直し待ちの 1 本 (既定 off)。
    ///
    /// どれも待つ位置を消すだけで、起動するカーネルも引数もバイト列も変えない
    /// (lab t334 の 8 run で md5 一致)。0 / off / false / no で切り戻せる。
    #[test]
    fn stage_nosync_gate_truth_table() {
        assert!(DEFAULT_STAGE_NOSYNC, "lab t334 で採った既定 on が戻っている");
        assert!(stage_nosync_from_value(None), "未設定は既定 (on)");
        assert!(stage_nosync_from_value(Some("")), "空も既定 (on)");
        assert!(!stage_nosync_from_value(Some("0")));
        assert!(!stage_nosync_from_value(Some("off")));
        assert!(!stage_nosync_from_value(Some("false")));
        assert!(!stage_nosync_from_value(Some("no")));
        assert!(stage_nosync_from_value(Some("1")));
        assert!(stage_nosync_from_value(Some("on")));
        assert!(stage_nosync_from_value(Some(" on ")), "前後の空白は許す");
        assert!(stage_nosync_from_value(Some("true")));
        assert!(stage_nosync_from_value(Some("yes")));
        assert_eq!(STAGE_NOSYNC_ENV, "DENNO_PF_STAGE_NOSYNC");
        // attention 連鎖のほうも同じ規約 (別のゲート名)。
        assert!(DEFAULT_ATTN_NOSYNC);
        assert!(attn_nosync_from_value(None), "未設定は既定 (on)");
        assert!(attn_nosync_from_value(Some("")));
        assert!(!attn_nosync_from_value(Some("0")));
        assert!(!attn_nosync_from_value(Some("off")));
        assert!(attn_nosync_from_value(Some("1")));
        assert!(attn_nosync_from_value(Some("on")));
        assert_eq!(ATTN_NOSYNC_ENV, "DENNO_PF_ATTN_NOSYNC");
        assert_ne!(STAGE_NOSYNC_ENV, ATTN_NOSYNC_ENV, "2 本は独立に切れること");
        // norm (gamma) の常駐。
        assert!(DEFAULT_MIX_GAMMA_CACHE);
        assert!(mix_gamma_cache_from_value(None), "未設定は既定 (on)");
        assert!(mix_gamma_cache_from_value(Some("")));
        assert!(!mix_gamma_cache_from_value(Some("0")));
        assert!(!mix_gamma_cache_from_value(Some("off")));
        assert!(mix_gamma_cache_from_value(Some("1")));
        assert!(mix_gamma_cache_from_value(Some("on")));
        assert_eq!(MIX_GAMMA_CACHE_ENV, "DENNO_PF_MIX_GAMMA_CACHE");
        // GDN の状態の書き戻し (直接 D2H) は 1 サンプルだけなので既定 off。
        assert!(!DEFAULT_GDN_WB_DIRECT);
        assert!(!gdn_wb_direct_from_value(None), "未設定は既定 off");
        assert!(!gdn_wb_direct_from_value(Some("0")));
        assert!(gdn_wb_direct_from_value(Some("1")));
        assert_eq!(GDN_WB_DIRECT_ENV, "DENNO_PF_GDN_WB_DIRECT");
    }

    /// ゲートは `DENNO_EXPERT_IQ_GPU=1` の内側だけ。既定は [`DEFAULT_ON`]。
    #[test]
    fn gate_needs_iq_and_parses_values() {
        assert!(!pf_from_value(false, Some("1")), "IQ ゲート off なら動かない");
        assert!(!pf_from_value(false, None));
        assert_eq!(pf_from_value(true, None), DEFAULT_ON);
        assert_eq!(pf_from_value(true, Some("")), DEFAULT_ON);
        assert!(pf_from_value(true, Some("1")));
        assert!(pf_from_value(true, Some("on")));
        assert!(!pf_from_value(true, Some("0")));
        assert!(!pf_from_value(true, Some("off")));
        assert!(!pf_from_value(true, Some("false")));
        // task 151: lab t151c で採用。未設定でも `DENNO_EXPERT_IQ_GPU=1` なら on。
        assert!(DEFAULT_ON, "既定 off に戻っている");
        assert!(pf_from_value(true, None), "未設定は既定 (on) でない");
        assert_eq!(DEFAULT_UPLOAD_MB, 0, "既定で prefill が LRU に積んでいる");
        assert_eq!(upload_mb(), 0, "既定の上げ直しが 0 でない");
    }

    /// 上げ直しの予算は MB から本数へ (1 本 2.05 MB)。
    #[test]
    fn upload_budget_is_bytes_over_expert_size() {
        assert_eq!(upload_budget_experts(), (upload_mb() << 20) / 2_048_000);
    }

    /// task 239: 均等割りの**発火**条件。
    ///
    /// lab t239c では `chunk_override().is_some()` で判定していたため、
    /// 未設定でも `Some(既定)` が返ってゲートが一度も発火せず、up_MB が
    /// バイト一致のままだった。ここで 4 条件の全組み合わせを固定する。
    #[test]
    fn chunk_for_prefill_fires_only_on_all_three_conditions() {
        let (n, base, cap) = (3222usize, DEFAULT_PF_CHUNK_STREAM, CHUNK_BALANCE_CAP);
        // 3 条件が揃ったときだけ均等割り (3,222 → 1,611 × 2 枚)。
        assert_eq!(chunk_for_prefill(n, base, cap, true, false, true, false), 1611);
        // ゲート off。
        assert_eq!(chunk_for_prefill(n, base, cap, false, false, true, false), base);
        // ストリーミング off (従来経路)。
        assert_eq!(chunk_for_prefill(n, base, cap, true, false, false, false), base);
        // `DENNO_PF_CHUNK` の明示指定がある。
        assert_eq!(chunk_for_prefill(n, base, cap, true, false, true, true), base);
        // 1 枚で収まる長さはゲート on でもそのまま。
        assert_eq!(chunk_for_prefill(1000, base, cap, true, false, true, false), 1000);
        // 2 枚を超える長さ (14k) は long ゲート off なら触らない (task 197 の実測領域)。
        assert_eq!(
            chunk_for_prefill(14018, base, cap, true, false, true, false),
            base,
            "長いプロンプトのチャンクを変えてしまった"
        );
        assert_eq!(chunk_for_prefill(4095, base, cap, true, false, true, false), base);
        // 2 枚ちょうどまでは均等割り。
        assert_eq!(chunk_for_prefill(4094, base, cap, true, false, true, false), 2047);
    }

    /// task 246: 長いプロンプトの均等割りは専用ゲート (`BALANCE_LONG`) の内側だけ。
    ///
    /// 14,018 は固定 1,536 で 10 枚 (9 × 1,536 + 194)、均等割りで 2,003 × 7 枚。
    /// 枚数が 3 割減るので、ミス expert の上げ直しもそのぶん減る (推定)。
    #[test]
    fn chunk_balance_long_extends_to_long_prompts() {
        let (base, cap) = (DEFAULT_PF_CHUNK_STREAM, CHUNK_BALANCE_CAP);
        assert!(!chunk_balance_long_from_value(None), "既定は off");
        assert!(!chunk_balance_long_from_value(Some("0")));
        assert!(chunk_balance_long_from_value(Some("1")));
        assert!(chunk_balance_long_from_value(Some(" on ")));
        // long off: 14k は従来の 1,536 のまま (10 枚)。
        assert_eq!(
            chunk_for_prefill(14018, base, cap, true, false, true, false),
            base
        );
        // long on: 2,003 × 7 枚。
        let c = chunk_for_prefill(14018, base, cap, true, true, true, false);
        assert_eq!(c, 2003);
        assert_eq!(14018usize.div_ceil(c), 7);
        // 上限は GDN 連鎖の境界 (2,048) の手前を守る。
        assert!(c <= cap && c < 2048);
        // 短いプロンプトの答えは long の有無で変わらない。
        for n in [1000usize, 3222, 3826, 4094] {
            assert_eq!(
                chunk_for_prefill(n, base, cap, true, true, true, false),
                chunk_for_prefill(n, base, cap, true, false, true, false),
                "n={n} で long が短い側の答えを変えた"
            );
        }
    }

    /// task 246: 寄せ直しのゲート解釈とトークン数の丸め。
    #[test]
    fn dec_reseed_parses_values() {
        assert!(!DEFAULT_DEC_RESEED, "既定 off でない");
        assert!(!dec_reseed_from_value(None));
        assert!(!dec_reseed_from_value(Some("0")));
        assert!(!dec_reseed_from_value(Some("off")));
        assert!(dec_reseed_from_value(Some("1")));
        assert!(dec_reseed_from_value(Some("on")));
        assert!(dec_reseed_from_value(Some(" true ")));
        // 既定は 256 (従来経路に落ちていた末尾チャンクの 150 / 194 を覆う)。
        assert_eq!(DEFAULT_DEC_RESEED_TOK, 256);
        assert_eq!(dec_reseed_tok_from_value(None), DEFAULT_DEC_RESEED_TOK);
        assert_eq!(dec_reseed_tok_from_value(Some("0")), DEFAULT_DEC_RESEED_TOK);
        assert_eq!(dec_reseed_tok_from_value(Some("x")), DEFAULT_DEC_RESEED_TOK);
        assert_eq!(dec_reseed_tok_from_value(Some("64")), 64);
        assert_eq!(
            dec_reseed_tok_from_value(Some("999999")),
            DEC_RESEED_TOK_MAX,
            "上限で丸めていない"
        );
    }

    /// task 246: 控えるのは**最後のチャンクの末尾トークン**だけで、
    /// 触る順は「トークン昇順 → 層昇順 → j 昇順」(従来経路と同じ)。
    #[test]
    fn dec_hot_keeps_only_the_tail_in_token_major_order() {
        let (nt, k, tok) = (4usize, 2usize, 2usize);
        let mut h = DecHot::default();
        // 1 枚目のチャンク (層 0..1)。トークン t の選択は (10*t + layer) 系。
        for layer in 0..2usize {
            let sels: Vec<usize> = (0..nt * k)
                .map(|i| 100 + layer * 10 + (i / k) * 2 + (i % k))
                .collect();
            note_tail_into(&mut h, layer, nt, k, &sels, tok);
        }
        // 2 枚目のチャンク (同じ長さ) は前のチャンクを捨てる。
        for layer in 0..2usize {
            let sels: Vec<usize> = (0..nt * k)
                .map(|i| layer * 10 + (i / k) * 2 + (i % k))
                .collect();
            note_tail_into(&mut h, layer, nt, k, &sels, tok);
        }
        let (keys, ntok) = take_dec_hot_from(&mut h, tok);
        assert_eq!(ntok, 2, "末尾 2 トークンだけのはず");
        // t=2: 層 0 の j=0,1 → 4,5 / 層 1 の j=0,1 → 14,15
        // t=3: 層 0 → 6,7 / 層 1 → 16,17
        assert_eq!(
            keys,
            vec![(0, 4), (0, 5), (1, 14), (1, 15), (0, 6), (0, 7), (1, 16), (1, 17)],
            "トークン昇順 → 層昇順 → j 昇順になっていない"
        );
        // 取り出したら空 (次の prefill のぶんと混ざらない)。
        let (again, _) = take_dec_hot_from(&mut h, tok);
        assert!(again.is_empty(), "取り出し後に残っている");
    }

    /// task 246: `tok` がチャンク長以上なら全トークン、0 相当なら空。
    #[test]
    fn dec_hot_tail_window_is_clamped() {
        let (nt, k) = (3usize, 1usize);
        let mut h = DecHot::default();
        for layer in 0..2usize {
            let sels: Vec<usize> = (0..nt).map(|t| layer * 10 + t).collect();
            note_tail_into(&mut h, layer, nt, k, &sels, nt * 4);
        }
        let (keys, ntok) = take_dec_hot_from(&mut h, nt * 4);
        assert_eq!(ntok, nt);
        assert_eq!(keys.len(), 2 * nt);
        // 先頭のトークンから順に出る。
        assert_eq!(keys[0], (0, 0));
        assert_eq!(keys[1], (1, 10));
    }

    /// task 246: 形が合わない入力は控えない (壊れた配列で層を跨がない)。
    #[test]
    fn dec_hot_ignores_malformed_selection() {
        let mut h = DecHot::default();
        note_tail_into(&mut h, 0, 4, 2, &[1, 2, 3], 2); // 長さ不足
        assert!(take_dec_hot_from(&mut h, 2).0.is_empty());
        note_tail_into(&mut h, 0, 0, 2, &[], 2);
        assert!(take_dec_hot_from(&mut h, 2).0.is_empty());
    }

    /// task 239: 均等割りは「上限の下でチャンク数を最小にする」長さを返す。
    ///
    /// 実際のプロンプト長 (lab t239a/t239b の c2000=3,826 / c8869=3,222) で、
    /// 固定 1536 の 3 枚が 2 枚になること (転送量が 1/3 減る) を固定する。
    #[test]
    fn balanced_chunk_minimizes_chunk_count_under_cap() {
        let (base, cap) = (DEFAULT_PF_CHUNK_STREAM, CHUNK_BALANCE_CAP);
        // 1 枚で収まる長さはそのまま。
        assert_eq!(balanced_chunk(150, base, cap), 150);
        assert_eq!(balanced_chunk(1536, base, cap), 1536);
        // 3,222 → 1,611 × 2 枚 (固定 1536 なら 1,536 + 1,536 + 150 の 3 枚)。
        let c = balanced_chunk(3222, base, cap);
        assert_eq!(c, 1611, "c8869 が 2 枚にならない");
        assert_eq!(3222usize.div_ceil(c), 2);
        // 3,826 → 1,913 × 2 枚。
        let c = balanced_chunk(3826, base, cap);
        assert_eq!(3826usize.div_ceil(c), 2, "c2000 が 2 枚にならない");
        // 14,018 → 7 枚 (固定 1536 なら 10 枚)。
        let c = balanced_chunk(14018, base, cap);
        assert_eq!(14018usize.div_ceil(c), 7);
        // どの長さでも 1 枚は上限を超えない (GDN 連鎖の境界を踏まない)。
        for n in [1usize, 512, 1536, 2000, 3222, 3826, 14018, 65536] {
            let c = balanced_chunk(n, base, cap);
            assert!(c > 0 && c <= cap.max(base), "n={n} のチャンク {c} が上限外");
            assert!(n.div_ceil(c) >= 1);
        }
    }

    /// task 162: 行表は「トークン × j」→ GPU の行番号。載っていない expert の
    /// 位置は穴 (`u32::MAX`) のまま残り、`row_table_zero_slots` で 0 重みになる。
    #[test]
    fn stream_row_table_maps_and_zeroes_holes() {
        let be = by_expert_fixture();
        // expert 0,2,4 は GPU、1,3 は CPU (1 は空なので穴にならない)。
        let pick = [true, true, true, false, true];
        let plan = build_plan(&be, &pick);
        assert_eq!(plan.experts, vec![0, 2, 4]);
        assert_eq!(plan.seg[0..4], [3, 0, 0, 0]);
        assert_eq!(plan.seg[4..8], [5, 3, 0, 0]);
        assert_eq!(plan.seg[8..12], [2, 8, 0, 0]);
        // k = 4、トークンは 0..=9 (fixture のトークン番号) → nt = 10。
        let k = 4;
        let nt = 10;
        let mut prow = row_table_partial(&be, &plan, k, nt).expect("行表");
        // expert 0 (先頭 0): トークン 2 の j=1 → 行 1。
        assert_eq!(prow[2 * k + 1], 1);
        // expert 2 (先頭 3): トークン 3 の j=3 は 3 番目の行 → 3 + 2 = 5。
        assert_eq!(prow[3 * k + 3], 5);
        // expert 4 (先頭 8): トークン 6 の j=1 は 2 番目の行 → 9。
        assert_eq!(prow[6 * k + 1], 9);
        // expert 3 (CPU) の (7,0) は穴。
        assert_eq!(prow[7 * k], u32::MAX);
        let mut w = vec![1.0f32; nt * k];
        let holes = row_table_zero_slots(&mut prow, &mut w);
        assert!(holes >= 1);
        assert_eq!(prow[7 * k], 0);
        assert_eq!(w[7 * k], 0.0);
        assert!(
            row_table_full(&be, &plan, k, nt).is_none(),
            "穴があるのに full"
        );
        // 全部 GPU で全 (t, j) が埋まる表なら strict も通る。
        let mut v: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 2];
        for t in 0..nt {
            for j in 0..k {
                v[(t + j) % 2].push((t, j));
            }
        }
        let plan = build_plan(&v, &[true, true]);
        assert!(row_table_full(&v, &plan, k, nt).is_some(), "全行が載っている");
        assert!(row_table_partial(&be, &plan, 0, nt).is_none(), "k = 0 は表を作らない");
    }

    /// task 162: 上げのグループ分けは「容量」と「本数」の両方で切る。
    /// 1 本だけで容量を超えるなら、その層はストリーミングしない (`None`)。
    #[test]
    fn stream_groups_respect_capacity_and_count() {
        // 常駐 0 (10) と ミス 100 が交互。
        let needs = [0usize, 100, 0, 100, 0, 100, 0, 100, 0, 100];
        let g = group_bounds(&needs, 250, 3).expect("分けられる");
        // 本数の上限 3 が先に効き、どの範囲もミス 2 本 (200 バイト) に収まる。
        assert_eq!(g, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
        for &(a, b) in &g {
            let sum: usize = needs[a..b].iter().sum();
            assert!(sum <= 250, "容量超過 {sum}");
            assert!(b - a <= 3, "本数超過");
        }
        // 本数の上限だけで切る形。
        let needs = [10usize; 7];
        assert_eq!(
            group_bounds(&needs, 1000, 3).expect("分けられる"),
            vec![(0, 3), (3, 6), (6, 7)]
        );
        // 1 本で容量を超える → この層は上げない。
        assert!(group_bounds(&[500, 10], 400, 8).is_none());
        assert!(group_bounds(&[], 400, 8).is_none());
        assert!(group_bounds(&[10], 0, 8).is_none());
    }

    /// task 324: 先頭だけ小さいグループに切る。担当は重複なし・取りこぼしなしで、
    /// `first == 0` なら従来とまったく同じ。
    #[test]
    fn first_group_is_short_and_covers_everything() {
        let needs = [100usize; 10];
        // 絞らないときは従来と一致する。
        assert_eq!(
            group_bounds_first(&needs, 1000, 4, 0),
            group_bounds(&needs, 1000, 4)
        );
        // 先頭は 1 本、残りは従来の切り方。
        let g = group_bounds_first(&needs, 1000, 4, 1).expect("分けられる");
        assert_eq!(g, vec![(0, 1), (1, 5), (5, 9), (9, 10)]);
        // 先頭 2 本。
        let g = group_bounds_first(&needs, 1000, 4, 2).expect("分けられる");
        assert_eq!(g[0], (0, 2));
        // 容量でも先頭は切られる。
        let needs = [90usize, 90, 90, 90];
        let g = group_bounds_first(&needs, 100, 4, 4).expect("1 本ずつ");
        assert_eq!(g, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        // 先頭の指定が上限 (max_group) を超えても本数上限は守る。
        let g = group_bounds_first(&needs, 1000, 2, 9).expect("分けられる");
        assert_eq!(Some(g), group_bounds(&needs, 1000, 2));
        // 全範囲が被覆され、重複が無い。
        let needs = [3usize, 1, 4, 1, 5, 9, 2, 6];
        let g = group_bounds_first(&needs, 1000, 3, 2).expect("分けられる");
        let mut seen = vec![0usize; needs.len()];
        for &(a, b) in &g {
            assert!(a < b && b <= needs.len());
            for s in seen.iter_mut().take(b).skip(a) {
                *s += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "重複か穴がある: {g:?}");
        // 1 本で容量を超えるものは従来どおり `None`。
        assert!(group_bounds_first(&[500, 10], 400, 8, 1).is_none());
        assert!(group_bounds_first(&[], 400, 8, 1).is_none());
        assert!(group_bounds_first(&[10], 0, 8, 1).is_none());
    }

    /// task 162: ストリーミングのゲートは prefill ゲートの内側だけ。
    /// 床 (min_tok) は 19 トークンの next turn prefill を巻き込まない。
    #[test]
    fn stream_gate_parses_values() {
        assert!(!stream_from_value(false, Some("1")), "pf off なら動かない");
        assert!(!stream_from_value(false, None));
        assert_eq!(stream_from_value(true, None), DEFAULT_STREAM_ON);
        assert_eq!(stream_from_value(true, Some("")), DEFAULT_STREAM_ON);
        assert!(stream_from_value(true, Some("1")));
        assert!(stream_from_value(true, Some("on")));
        assert!(!stream_from_value(true, Some("0")));
        assert!(!stream_from_value(true, Some("off")));
        assert!(!stream_from_value(true, Some("false")));
        assert!(stream_min_tok() > 19, "next turn prefill を巻き込む");
        assert!(stream_bytes() >= 64 << 20, "ステージングが小さすぎる");
        assert!(stream_group() >= 1);
        assert!(stream_pin() <= 1);
        // task 165: 上げ専用スレッドのゲート (ストリーミングの内側)。
        assert!(!stream_ut_from_value(false, Some("1")), "stream off なら動かない");
        assert_eq!(stream_ut_from_value(true, None), DEFAULT_STREAM_UT);
        assert_eq!(stream_ut_from_value(true, Some("")), DEFAULT_STREAM_UT);
        assert!(stream_ut_from_value(true, Some("1")));
        assert!(!stream_ut_from_value(true, Some("0")));
        assert!(!stream_ut_from_value(true, Some("off")));
        // task 320: 依頼の先読み数 (1..=2 に丸める。既定 1 = 従来)。
        assert_eq!(stream_lookahead_from_value(None), DEFAULT_STREAM_LOOKAHEAD);
        assert_eq!(stream_lookahead_from_value(Some("")), DEFAULT_STREAM_LOOKAHEAD);
        assert_eq!(stream_lookahead_from_value(Some("1")), 1);
        assert_eq!(stream_lookahead_from_value(Some("2")), 2);
        assert_eq!(stream_lookahead_from_value(Some("0")), 1, "0 は下限へ");
        assert_eq!(stream_lookahead_from_value(Some("9")), 2, "上限は 2 (3 枚)");
        assert_eq!(stream_lookahead_from_value(Some("x")), DEFAULT_STREAM_LOOKAHEAD);
        assert_eq!(stream_lookahead_from_value(Some(" 2 ")), 2);
        // task 320: 分離モードは明示的な真値だけ on。
        assert!(!up_decouple_from_value(None));
        assert!(!up_decouple_from_value(Some("0")));
        assert!(!up_decouple_from_value(Some("")));
        assert!(up_decouple_from_value(Some("1")));
        assert!(up_decouple_from_value(Some("yes")));
        // task 379: GDN 状態の書き戻しを最終チャンクだけにするゲート
        // (lab t379a / t379b で採ったので既定 on)。
        assert!(gdn_wb_last_from_value(None), "既定 on");
        assert!(gdn_wb_last_from_value(Some("")), "未設定と同じ");
        assert!(!gdn_wb_last_from_value(Some("0")), "明示的な 0 で切り戻せる");
        assert!(!gdn_wb_last_from_value(Some("off")));
        assert!(!gdn_wb_last_from_value(Some("no")));
        assert!(gdn_wb_last_from_value(Some("1")));
        assert!(gdn_wb_last_from_value(Some("on")));
        assert!(gdn_wb_last_from_value(Some("true")));
        assert!(gdn_wb_last_from_value(Some("yes")));
        assert!(gdn_wb_last_from_value(Some(" 1 ")), "前後の空白は無視する");
        assert!(DEFAULT_GDN_WB_LAST, "lab で採ったので既定 on");
        // task 324: H2D の完了を計算ストリームに待たせるゲート (lab t324b で採った)。
        assert_eq!(up_gpuwait_from_value(None), DEFAULT_UP_GPUWAIT);
        assert_eq!(up_gpuwait_from_value(Some("")), DEFAULT_UP_GPUWAIT);
        assert!(DEFAULT_UP_GPUWAIT, "採ったので既定 on");
        assert!(!up_gpuwait_from_value(Some("0")), "明示的な 0 で切り戻せる");
        assert!(!up_gpuwait_from_value(Some("off")));
        assert!(up_gpuwait_from_value(Some("1")));
        assert!(up_gpuwait_from_value(Some(" on ")), "前後の空白は無視する");
        assert!(up_gpuwait_from_value(Some("yes")));
        // task 324: 層の終端の同期を省くゲート (同じく既定 on、0 で切り戻す)。
        assert_eq!(stream_nosync_from_value(None), DEFAULT_STREAM_NOSYNC);
        assert_eq!(stream_nosync_from_value(Some("")), DEFAULT_STREAM_NOSYNC);
        assert!(DEFAULT_STREAM_NOSYNC);
        assert!(!stream_nosync_from_value(Some("0")));
        assert!(!stream_nosync_from_value(Some("false")));
        assert!(stream_nosync_from_value(Some("1")));
        assert!(stream_nosync_from_value(Some(" true ")));
        // グループ 1 個の本数は 32 (task 324 で 64 から)。切り戻しは env で。
        assert_eq!(DEFAULT_STREAM_GROUP, 32);
        // task 324: 先頭グループの本数 (0 = 絞らない。読めない値は 0)。
        assert_eq!(stream_first_group_from_value(None), 0);
        assert_eq!(stream_first_group_from_value(Some("")), 0);
        assert_eq!(stream_first_group_from_value(Some("0")), 0);
        assert_eq!(stream_first_group_from_value(Some("16")), 16);
        assert_eq!(stream_first_group_from_value(Some(" 8 ")), 8);
        assert_eq!(stream_first_group_from_value(Some("x")), 0);
        assert_eq!(stream_first_group_from_value(Some("999")), 256, "上限 256");
        // task 165: write-combined のバウンスは明示的な真値だけ on。
        assert!(!stream_wc_from_value(false, Some("1")), "stream off なら動かない");
        assert_eq!(stream_wc_from_value(true, None), DEFAULT_STREAM_WC);
        assert_eq!(stream_wc_from_value(true, Some("")), DEFAULT_STREAM_WC);
        assert!(stream_wc_from_value(true, Some("1")));
        assert!(!stream_wc_from_value(true, Some("0")));
        assert_eq!(
            stream_wc_from_value(true, Some(" ")),
            DEFAULT_STREAM_WC,
            "空白だけは既定"
        );
        assert!(stream_wc_from_value(true, Some(" on ")), "前後の空白は無視する");
        assert!(!stream_wc_from_value(true, Some("no")));
        // task 227: 転送元の選択 (既定は pin = バウンス) と、素の H2D の探針。
        assert_eq!(stream_src_from_value(None), DEFAULT_STREAM_SRC);
        assert_eq!(stream_src_from_value(Some("")), DEFAULT_STREAM_SRC);
        assert_eq!(stream_src_from_value(Some("pin")), 0);
        assert_eq!(stream_src_from_value(Some("page")), 1);
        assert_eq!(stream_src_from_value(Some("reg")), 2);
        assert_eq!(stream_src_from_value(Some(" reg ")), 2, "前後の空白は無視する");
        assert_eq!(stream_src_from_value(Some("bogus")), DEFAULT_STREAM_SRC);
        assert_eq!(DEFAULT_STREAM_SRC, 0, "既定は従来のバウンス");
        assert_eq!(DEFAULT_STREAM_CHUNK_KB, 0, "既定は従来の直列");
        // task 227: 探針は明示的な真値だけ on。
        assert!(!dma_probe_from_value(None));
        assert!(!dma_probe_from_value(Some("")));
        assert!(!dma_probe_from_value(Some("0")));
        assert!(!dma_probe_from_value(Some("off")));
        assert!(dma_probe_from_value(Some("1")));
        assert!(dma_probe_from_value(Some(" on ")));
        // task 265: 層優先 prefill は既定 on、明示的な偽値だけ off。
        assert!(lm_layer_from_value(None));
        assert!(lm_layer_from_value(Some("")));
        assert!(!lm_layer_from_value(Some("0")));
        assert!(!lm_layer_from_value(Some("off")));
        assert!(lm_layer_from_value(Some("1")));
        assert!(lm_layer_from_value(Some(" yes ")));
        assert_eq!(DEFAULT_PREFILL_CHUNK_STREAM, 1024, "lab t162c の最良");
        // task 197: チャンク上書きは「明示した正の数だけ」効く。
        assert_eq!(chunk_override_from_value(None), None);
        assert_eq!(chunk_override_from_value(Some("")), None, "空は上書きしない");
        assert_eq!(chunk_override_from_value(Some("   ")), None);
        assert_eq!(chunk_override_from_value(Some("0")), None, "0 は無効");
        assert_eq!(chunk_override_from_value(Some("off")), None);
        assert_eq!(chunk_override_from_value(Some("false")), None);
        assert_eq!(chunk_override_from_value(Some("abc")), None);
        assert_eq!(chunk_override_from_value(Some("1024")), Some(1024));
        assert_eq!(chunk_override_from_value(Some(" 1536 ")), Some(1536));
        assert_eq!(chunk_override_from_value(Some("999999")), Some(65536), "上限で切る");
        assert_eq!(chunk_override_from_value(Some("1")), Some(1));
        // task 197: 既定 1536 は ssm のデバイス連鎖の形の境界 (< 2048) の内側にある。
        assert_eq!(DEFAULT_PF_CHUNK_STREAM, 1536, "lab t197a の最良");
        #[cfg(feature = "cuda")]
        assert!(
            DEFAULT_PF_CHUNK_STREAM < crate::cuda::GDN_CHUNK_MIN_TOKENS,
            "既定チャンクが GDN 連鎖の境界 (2048) に触ると ssm がホスト経路へ落ちる"
        );
        // task 165 で決めた既定 (lab t165b/t165c の最良)。
        assert!(DEFAULT_STREAM_ON, "lab t165c で採用して既定 on");
        assert!(DEFAULT_STREAM_WC, "write-combined のバウンスが効いた");
        // task 324: 32 (gpuwait の既定化で、残る露出は層の入口の段差 = グループの
        // バイト数に比例するようになった。lab t324b で g64 2,605 → g32 2,532)。
        assert_eq!(stream_group(), 32, "グループ 32 が最良");
        // task 239: 分割は既定 1 (従来の 1 回写し + 1 回 DMA)。
        assert_eq!(
            stream_split_from_value(None),
            DEFAULT_STREAM_SPLIT,
            "未設定は既定 (1 = 従来)"
        );
        assert_eq!(stream_split_from_value(Some("")), 1);
        assert_eq!(stream_split_from_value(Some("0")), 1, "0 は既定へ");
        assert_eq!(stream_split_from_value(Some("x")), 1, "解析不能は既定へ");
        assert_eq!(stream_split_from_value(Some("4")), 4);
        assert_eq!(stream_split_from_value(Some("64")), 16, "上限 16");
        // task 481: 上げスレッドの分割 DMA は既定 8 MiB (lab t481b で採用)。
        assert_eq!(up_split_kb_from_value(None), 8192, "未設定は既定 (on)");
        assert_eq!(up_split_kb_from_value(Some("")), 8192);
        assert_eq!(up_split_kb_from_value(Some("0")), 0, "明示的な 0 は従来");
        assert_eq!(up_split_kb_from_value(Some("off")), 0);
        assert_eq!(up_split_kb_from_value(Some("false")), 0);
        assert_eq!(up_split_kb_from_value(Some("x")), 8192, "解析不能は既定へ");
        assert_eq!(up_split_kb_from_value(Some("16")), 64, "下限 64 KiB");
        assert_eq!(up_split_kb_from_value(Some("4096")), 4096);
        assert_eq!(up_split_kb_from_value(Some(" 8192 ")), 8192, "空白は無視");
        assert_eq!(up_split_kb_from_value(Some("200000")), 65536, "上限 64 MiB");
        // task 239: 均等割りのゲートと上限。
        assert!(
            !chunk_balance_from_value(None),
            "既定 off (従来の固定チャンク)"
        );
        assert!(!chunk_balance_from_value(Some("0")));
        assert!(chunk_balance_from_value(Some("1")));
        assert!(chunk_balance_from_value(Some(" on ")));
        #[cfg(feature = "cuda")]
        assert_eq!(
            CHUNK_BALANCE_CAP + 1,
            crate::cuda::GDN_CHUNK_MIN_TOKENS,
            "上限は GDN 連鎖の境界の手前 (2048 を踏むと ssm がホスト経路へ落ちる)"
        );
        // 分け合った 1 枚 (cuda 側は 3 枚) に、既定のグループ (64 本、
        // 平均 2.05 MB/本) が載ること。
        assert!(
            (stream_bytes() / 3) / stream_group() >= 2_500_000,
            "1 枚に既定のグループが載らない"
        );
        // task 300: pinned バウンスの余裕つき確保は**既定 on** (task 296 の
        // lab t296c で 3,826 cold -5.3%、14k と decode は無傷)。
        assert!(pin_grow(), "既定は余裕つき確保 (task 300)");
        assert!(pin_grow_from_value(None));
        assert!(pin_grow_from_value(Some("")));
        assert!(!pin_grow_from_value(Some("0")));
        assert!(!pin_grow_from_value(Some("off")));
        assert!(!pin_grow_from_value(Some("False")));
        assert!(pin_grow_from_value(Some("1")));
        assert!(pin_grow_from_value(Some(" on ")));
        assert!(pin_grow_from_value(Some("yes")));
        // 余裕は +25% (最低 16 MiB) を 8 MiB へ切り上げ、ステージング 1 枚で頭打ち。
        assert_eq!(pin_grow_target(0, 1 << 30), 0);
        assert_eq!(pin_grow_target(64 << 20, 1 << 30), 80 << 20);
        assert_eq!(pin_grow_target(8 << 20, 1 << 30), 24 << 20, "最低 16 MiB の余裕");
        assert_eq!(pin_grow_target(160 << 20, 170 << 20), 170 << 20, "上限で切る");
        assert_eq!(pin_grow_target(200 << 20, 170 << 20), 200 << 20, "上限が足りない");
        // 余裕の目的: 64 MiB を頼んだあと 72 MiB を頼んでも確保し直さない。
        let t = pin_grow_target(64 << 20, 1 << 30);
        assert!(t >= 72 << 20, "±数 MB の揺れを吸収しない: {t}");
    }

    /// ホストが読む名前が .cu のカーネル名と一致する (実機でしか分からない
    /// 名前の食い違いをソースで突き合わせる)。
    #[test]
    fn kernel_names_exist_in_cu_source() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let cu = std::fs::read_to_string(dir.join("kernels_exp_iq_pf.cu")).expect(".cu が読めない");
        // task 309: K 系は別ファイル (別モジュール)。
        let kcu = std::fs::read_to_string(dir.join("kernels_exp_k_pf.cu"))
            .expect("kernels_exp_k_pf.cu が読めない");
        for i in 0..SLOTS {
            let name = kernel_name(i).expect("slot");
            let src = if is_k_slot(i) { &kcu } else { &cu };
            let file = if is_k_slot(i) {
                "kernels_exp_k_pf.cu"
            } else {
                "kernels_exp_iq_pf.cu"
            };
            // 256 要素ブロックの型はマクロで生成するので、どちらの書き方でも受ける。
            assert!(
                src.contains(&format!("__global__ void {name}("))
                    || src.contains(&format!("PF_KERNEL_256({name},"))
                    || src.contains(&format!("KPF_KERNEL_256({name},")),
                "{name} が {file} に無い"
            );
        }
        assert!(
            cu.contains(&format!("__global__ void {SILU_KERNEL}(")),
            "{SILU_KERNEL} が kernels_exp_iq_pf.cu に無い"
        );
        assert!(
            cu.contains(&format!("__global__ void {ACCUM_KERNEL}(")),
            "{ACCUM_KERNEL} が kernels_exp_iq_pf.cu に無い"
        );
        // 起動形の定数が .cu と揃っていること (片方だけ変えると行が落ちる)。
        assert!(cu.contains("#define PF_MR 16"), "PF_MR が 16 でない");
        assert!(cu.contains("#define PF_OUTS 2"), "PF_OUTS が 2 でない");
        assert!(cu.contains("#define PF_WARPS 4"), "PF_WARPS が 4 でない");
        assert_eq!(ROWS_PER_LAUNCH, 16);
        assert!(kcu.contains("#define PF_MR 16"), "K 系の PF_MR が 16 でない");
        assert!(kcu.contains("#define PF_OUTS 2"), "K 系の PF_OUTS が 2 でない");
        assert!(kcu.contains("#define PF_WARPS 4"), "K 系の PF_WARPS が 4 でない");
        // 生成規則 (build_ptx.sh の exp_iq_pf) と cuda.rs が読む名前が同じこと。
        let sh = std::fs::read_to_string(dir.join("build_ptx.sh")).expect("build_ptx.sh");
        assert!(
            sh.contains("exp_iq_pf)") && sh.contains("kernels_exp_iq_pf.cu|fmad"),
            "build_ptx.sh に exp_iq_pf の規則が無い"
        );
        assert!(
            sh.contains("targets_all=(") && sh.contains(" exp_iq_pf"),
            "build_ptx.sh の targets_all に exp_iq_pf が無い"
        );
        // task 309: K 系も同じ規則で生成する。
        assert!(
            sh.contains("exp_k_pf)") && sh.contains("kernels_exp_k_pf.cu|fmad"),
            "build_ptx.sh に exp_k_pf の規則が無い"
        );
        assert!(
            sh.contains(" exp_k_pf)"),
            "build_ptx.sh の targets_all に exp_k_pf が無い"
        );
    }

    /// コミット済みの PTX が .cu と同じ引数の数を持つ (ABI の取り違えは実機で
    /// しか分からないので、PTX の `.param` 行を数えて固定する)。
    #[test]
    fn committed_ptx_has_expected_params() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let count = |ptx: &str, name: &str| -> Option<usize> {
            let head = format!(".visible .entry {name}(");
            let i = ptx.find(&head)?;
            let rest = &ptx[i + head.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            Some(rest[..end].matches(".param").count())
        };
        let Ok(ptx) = std::fs::read_to_string(dir.join("kernels_exp_iq_pf.ptx")) else {
            // PTX 未生成のツリーでは何も言えない (カーネル無し = CPU へ落ちるだけ)。
            return;
        };
        for i in 0..IQ_SLOTS {
            let name = kernel_name(i).expect("slot");
            assert_eq!(count(&ptx, name), Some(9), "{name} の引数の数が .cu と違う");
        }
        assert_eq!(count(&ptx, SILU_KERNEL), Some(4), "{SILU_KERNEL} の引数の数");
        // accum は task 151 でホスト側の足し込みに戻したので**現状未使用**。
        // PTX には残っていて、ABI (引数の数) だけを固定する。
        assert_eq!(count(&ptx, ACCUM_KERNEL), Some(7), "{ACCUM_KERNEL} の引数の数");
        // task 309: K 系の PTX (未生成なら省略)。
        let Ok(kptx) = std::fs::read_to_string(dir.join("kernels_exp_k_pf.ptx")) else {
            return;
        };
        for i in K_SLOT0..SLOTS {
            let name = kernel_name(i).expect("slot");
            assert_eq!(count(&kptx, name), Some(9), "{name} の引数の数が .cu と違う");
        }
    }

    /// task 151: 診断の併合と取り出し (`DENNO_EXPERT_GPU_CHECK=1` の prefill 版)。
    /// env を触らない部分だけ固定する。
    #[test]
    fn check_accumulates_and_takes() {
        let _ = take_check();
        note_check(PfCheck {
            layers: 1,
            rows: 8,
            argmax: 8,
            max_abs: 1.0e-6,
            max_rel: 2.0e-6,
        });
        note_check(PfCheck {
            layers: 1,
            rows: 4,
            argmax: 3,
            max_abs: 5.0e-6,
            max_rel: 1.0e-6,
        });
        let c = take_check().expect("溜まっている");
        assert_eq!(c.layers, 2);
        assert_eq!(c.rows, 12);
        assert_eq!(c.argmax, 11);
        assert_eq!(c.max_abs, 5.0e-6);
        assert_eq!(c.max_rel, 2.0e-6);
        assert!(take_check().is_none(), "取り出したら空になる");
    }
}
