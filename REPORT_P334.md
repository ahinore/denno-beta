ADOPT: yes t334a t334b t334c

理由: UD-Q4_K_XL の 14k cold は、同じ run の末尾 baseline に対して t334a **-5.3%**、
t334b **-6.1%** (2 ラウンド中央値では -2.0%)。3,826 cold は 2 ラウンドとも勝ち
(-2.2% / -3.5%)、既定モデル (swift15-abl-IQ3_S) は 3,826 が 2.50-2.54 s・decode
16.9-17.0 ms/tok で**変化なし**、md5 は UD 8 run + 既定 3 run の**全部一致**。
**タスクの目標 -8% には届いていない**。採否はオペレーターの規則 (2 ラウンドの
中央値で 2% 以上、他が 1% を超えて悪化しない、text 正常) に拠る。3 本を既定 on
にし、`DENNO_PF_GDN_WB_DIRECT` だけは 1 サンプルなので off のまま残した。

# REPORT P334 — UD-Q4_K_XL の 14k `other`: 段の待ちを外したら、待ちは同期コピーへ移った

branch: br334 = int-auto `716e9a8` から。lab **3 リクエスト** (上限 3):
**t334a** (`c5cfd76`、ゲート 3 本 + 部分集合の切り分け 5 COND) / **t334b** (`d486535`、
`GDN_WB_DIRECT` を足して 4 COND) / **t334c** (`d486535`、既定モデル、3 COND)。
UD は `MODEL=flash-next-UD-Q4_K_XL`、baseline COND = int-auto 既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に baseline の使い捨て・末尾に repeat、`PROFILE=1`。

## 0. 前提の訂正 (Do 1) — 14k の「672 チャンク層」は 336 チャンク層

REPORT_P329 §7 は「96 層 × 7 チャンク = 672 チャンク層」と書いたが、**層数は 48**。
`[q4prof] prefill tok=14018 layer_exec=672864` = 48 × 14,018 で確定する
(`[exp-pf-stream] layers=96` は 48 層 × タイル 2 パスのこと)。
よって層優先ループの内側は **48 層 × 7 チャンク = 336 チャンク層**で、1 チャンク層に
つき **mix 2 回 + combine 2 回** (attn 入口と ffn 入口) が回る。台帳の回数と一致する:
`mix` の sync 672 = 336 × 2、`ssm` の sync 756 = 36 層 × 7 × 3 段、`full` の sync
168 = 12 層 × 7 × 2、`other` の sync 680 ≈ 336 × 2。

## 1. 変えたこと (4 本、すべて env ゲート)

待ちの**位置**を消すだけで、起動するカーネル・引数・バイト列・丸め順は 1 つも
変えていない。だから md5 は全 run で一致する (それが狙い)。

| ゲート | 何を消すか | 台帳の対象 (t329b の 14k) | 既定 |
| --- | --- | ---: | --- |
| `DENNO_PF_STAGE_NOSYNC` | `pf_mix` / `pf_combine_core` の終端の `cuCtxSynchronize` | mix 826 ms (672 回) + combine 1,656 ms (680 回) | **on** |
| `DENNO_PF_ATTN_NOSYNC` | `gdn_pf_chain` の 3 段 (投影 / スキャン / out_proj) と `pf_full_qkv` / `fused` / `full_out` / `pf_shared_dev` の待ち | ssm 1,546 ms (756 回) + full 631 ms (168 回) | **on** |
| `DENNO_PF_MIX_GAMMA_CACHE` | mix の norm (層の定数 40 KB) の H2D 672 回 → デバイス常駐 | h2d 4,750.7 MB のうち 27.5 MB | **on** |
| `DENNO_PF_GDN_WB_DIRECT` | 状態の書き戻しの `to_vec` + `copy_from_slice` → `to_slice` | ssm の d2h 823.7 MB / 504 回 | off (1 サンプル) |

**なぜ外してよいか**: `cuCtxSynchronize` は**文脈の全ストリーム**を待つので、上げ
専用スレッドの非同期 H2D (non-blocking ストリーム) まで毎回吸い付けていた。待ちを
外しても、次の段は同じ (レガシー既定) ストリームへの非同期起動か、同じストリームの
**同期コピー** (`cuMemcpyHtoD` / `DtoH`) で、どちらもストリーム順で前のカーネルを
待つ。ホストがデバイスの結果を読む経路 (`pf_take_mixed` / `pf_download_res` / KV の
`to_slice` / GDN 状態の書き戻し) は必ず同期コピーを通るので、古いバイト列は読まない。
`pf_steer_*` (ホストが直後に読む) は**直呼びのまま**残した。

## 2. lab の数字 (実測、すべて同じ run の中の比較)

### 14k cold (c8000_cold、depth8k の round 1 のみ)

| request | cond | 14k cold (s) | 同じ run の末尾 baseline | 差 | copy_s (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| t334a | b0t (先頭・バイアス) | 31.25 | — | — | 10.10 |
| t334a | g (gamma のみ) | 28.63 | 28.74 | -0.4% | 8.06 |
| t334a | n (待ち 2 本) | 27.86 | 28.74 | -3.1% | 7.33 |
| t334a | ng (待ち 2 本 + gamma) | **27.23** | 28.74 | **-5.3%** | 7.84 |
| t334a | b1 (末尾 baseline) | 28.74 | — | — | 8.22 |
| t334b | b0t (先頭) | 27.94 | — | — | 11.75 |
| t334b | ng | 29.48 | 29.11 | +1.3% | 13.91 |
| t334b | ngw (4 本) | **27.33** | 29.11 | **-6.1%** | 11.81 |
| t334b | b1 (末尾 baseline) | 29.11 | — | — | 11.76 |

- `copy_s` は MoE の上げ (98.2 GB の pread) で、デバイスとページキャッシュで 2 割
  動く。t334b の `ng` だけ 13.91 s (`enq_s` 0.64 対 他 0.12-0.14) で、この 1 本だけ
  ディスク競合の外れ値。**層の中の時間**で見ると: b1 25,113 ms に対し ng 24,267
  (t334a) / ngw 24,321 (t334b) で、2 ラウンドとも **-3.2%** で一致する。
- 2 ラウンドの中央値: baseline (28.74, 29.11) = 28.93、ゲート (27.23, 29.48) = 28.36
  = **-2.0%**。ゲート on を 4 本にすると (27.33, 4 本は 1 サンプル)。

### 3,826 cold (各 COND 2 ラウンドの中央値) と turn2 / decode

| request | cond | 3,826 cold (s) | 14k turn2 (s) | warm short decode (ms/tok) |
| --- | --- | ---: | ---: | ---: |
| t334a | b0t | 17.39 (先頭バイアス) | 0.95 | 20.4 |
| t334a | g | 10.41 | 0.61 | 19.5 |
| t334a | n | **9.99** | 0.56 | 19.9 |
| t334a | ng | 10.18 | 0.60 | 19.5 |
| t334a | b1 | 10.41 | 0.72 | 19.8 |
| t334b | b0t | 11.58 | 0.60 | 21.2 |
| t334b | ng | **9.86** | 0.73 | 19.8 |
| t334b | ngw | 10.45 | 0.59 | 20.2 |
| t334b | b1 | 10.22 | 0.68 | 19.8 |

3,826 は t334a -2.2% / t334b -3.5% (ng)。`ngw` の +2.3% は 1 サンプルで、
`to_slice` は `to_vec` + `copy_from_slice` より**仕事が少ない**ので機序として説明
できない (ssm の cpu は 68 → 23 ms に落ちている)。既定モデルも -1.6% で、これも
「悪化」とは読めない範囲。decode は 19.5-20.4 で 0.1 ms/tok 刻みの範囲。

### 既定モデル (t334c、swift15-abl-IQ3_S、depth、3 COND)

| cond | 3,826 cold (s) | 3,826 turn2 (s) | decode (ms/tok) | md5 |
| --- | ---: | ---: | ---: | --- |
| b0t | 2.54 | 0.39 | 16.9 (cold 16.1) | 65ededc3 / f18c75fc / a91e17da / 5a8f5488 |
| ngw | 2.50 | 0.39 | 17.0 (cold 16.2) | 同上 (完全一致) |
| b1 | 2.53 | 0.39 | 16.9 (cold 16.1) | 同上 (完全一致) |

`VERDICT ngw vs b0t OK decode +0.7% prefill +0.0% md5 SAME errors 0`。
16.9 と 17.0 は 0.1 ms/tok の刻み 1 つで、t329d の範囲 (16.9-17.1)。

## 3. 台帳 — 待ちはどこへ移ったか (t334b の 14k、実測)

| 行 | b1 (待ちあり) | ngw (待ちなし) | 差 |
| --- | ---: | ---: | ---: |
| syncs (全体) | 2,565 | **201** | -2,364 回 |
| `mix` wall / sync | 1,022 / 826 | 23 / 0 | -999 |
| `full` wall / sync | 1,140 / 631 | 1,297 / 0 | +157 |
| `ssm` wall / sync | 1,792 / 1,546 | **3,540** / 0 | **+1,748** |
| `experts` wall / sync | 19,094 / 1,220 | 19,001 / 1,295 | -93 |
| `other` wall / sync | 6,051 / 1,656 | 3,464 / 9 | -2,587 |
| `other` の sub combine | 1,658 | 3 | -1,655 |
| wall | 29.10 s | 27.33 s | -1.77 s |

- **消した待ち 4.66 s のうち約 2.9 s は、同期コピーの待ちとして戻ってきた** (`ssm`
  が +1.75 s、`full` が +0.16 s)。`ssm` の正体は `gdn_host_writeback` (36 層 × 7
  チャンク × 2 本、d2h 823.7 MB) で、これは**連鎖のカーネルの後ろに順序付けられた
  同期 D2H** なので、待ちを消すと新しい直列点になる。`ssm` の sub (qkv/scan/out) は
  7 ms しかない = 壁のほぼ全部が待ちと写し。
- `mix` は gamma を常駐させたので 1,022 → 23 ms。`g` (gamma だけ) の回で h2d が
  4,750.7 → 4,723.2 MB = **-27.5 MB** = 672 回 × 40 KB で、常駐が効いている証拠。
- `experts` の sync は 1,220 → 1,295 で変わらない (MoE 側は触っていない)。

## 4. 数値の扱い

- md5 は t334a の 10 run・t334b の 8 run で**全 COND 同一**: a `737a73af`、
  b `87d8a97c`、c2000_cold `f9557457`、c2000_turn2 `9cbe03ca`、c8000_cold
  `27c0c9dc`、c8000_turn2 `d88fd518`。既定モデル (t334c) も 3 COND で完全一致:
  a `65ededc3`、b `f18c75fc`、c2000_cold `a91e17da`、c2000_turn2 `5a8f5488`。
  errors は全 run 0。
- 4 本とも**バイト列を変えない**ので md5 が一致するのは設計どおり。丸め順の変更も
  近似も無い (「丸めだけの差」ですらない)。
- 決定性・行チャンク不変・par/seq 一致・アンカーは触っていない (prefill の待ちの
  位置だけ)。

## 5. 見積り (推定。測っていない)

1. **GDN 状態の書き戻しを最後のチャンクだけにする**と -1.5〜-2.0 s。デバイス状態が
   正本で、ホスト側の配列を読むのは (a) 層の終わり、 (b) 連鎖が降りたときの
   フォールバック、だけ。1〜6 チャンク目を飛ばせば `ssm` の 2.0 s の大半が消える。
   連鎖がスキャン後に降りない設計 (降りたら `bail!`) なので、降りた門では
   デバイス状態から読み直せば安全。
2. **full 層の KV の D2H** (689 MB / 168 回) を同じ形にすると -0.5 s。
3. `other` の残り 3.46 s のうち **2.49-2.65 s は PLE の gather** (UD は 104 GB で
   59 GiB の RAM に収まらず、166,952 回の pread がディスクに落ちる)。共通ルールで
   既に不採用 (表を RAM に載せる / 並列 pread) なので触っていない。
4. `experts` の露出 618-833 ms。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1228 passed / 0 failed / 102 ignored** (最終木、int-auto `4e4923d` を merge 済み)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **102 passed / 0 failed**
- 追加: `exp_pf::tests::stage_nosync_gate_truth_table` (4 本の真理値表と既定。
  `0` / `off` / `false` / `no` / 未設定・空の扱いを固定)。
- 更新: `kernel_invariants::pf_mix_stage_sync_uses_an_event` — mix / combine が
  `pf_stage_sync_maybe` を通り、steer の 2 本は直呼びのままであることを数える。

## 7. GPU で確認してほしいこと

1. 本番 (8098) の UD-Q4_K_XL で 3,826 / 14k cold を 2 回ずつ。**14k の 2 サンプル
   中央値**を取ってほしい (この木の 14k は 1 ラウンド 1 サンプルで、t334b の `ng`
   1 本がディスク競合で +1.3% に振れている)。`[q4ledger-pf]` の `syncs` が
   2,500 台 → 200 前後になっていればゲートは発火している。
2. `DENNO_PF_GDN_WB_DIRECT=1` を 2 ラウンド測る (t334b の 1 サンプルは 14k -6.1% /
   3,826 +2.3%)。効くなら -0.3 s 程度、効かないなら `ssm` の待ちが本体。
3. 既定モデルは t334c のとおり (3,826 2.50-2.54 s、decode 16.9-17.0 ms/tok、md5 一致)。
4. 次の一手は §5.1 の「状態の書き戻しを最後のチャンクだけに」(推定 -1.5〜-2.0 s)。

## 8. lab の VERDICT 行 (そのまま)

**注意**: この lab の `VERDICT` は**先頭の COND (b0t) を基準**にするので、先頭
バイアス (t334a では何も変えていない b1 が b0t に対して prefill -8.3%) が全部の
差に乗る。採否は**同じ run の末尾 baseline (b1)** との比較 (§2) で判断した。

```
# t334a (c5cfd76)
VERDICT g vs b0t OK decode -4.0% prefill -6.9% md5 SAME errors 0
VERDICT n vs b0t OK decode -2.3% prefill -7.6% md5 SAME errors 0
VERDICT ng vs b0t OK decode -4.0% prefill -6.2% md5 SAME errors 0
VERDICT b1 vs b0t OK decode -2.9% prefill -8.3% md5 SAME errors 0
VERDICT_DEPTH ng c8000_cold vs b0t decode -39.6% prefill -12.9%
VERDICT_DEPTH ng c2000_cold vs b0t decode -71.2% prefill -41.5%
VERDICT baseline b0t errors 0 text ok

# t334b (d486535)
VERDICT ng vs b0t OK decode -6.4% prefill -7.5% md5 SAME errors 0
VERDICT ngw vs b0t OK decode -4.8% prefill -4.8% md5 SAME errors 0
VERDICT b1 vs b0t OK decode -6.8% prefill -8.2% md5 SAME errors 0
VERDICT_DEPTH ngw c8000_cold vs b0t decode -3.5% prefill -2.2%
VERDICT_DEPTH ng c8000_cold vs b0t decode +1.9% prefill +5.5%
VERDICT baseline b0t errors 0 text ok

# t334c (d486535、既定モデル swift15-abl-IQ3_S、depth)
VERDICT ngw vs b0t OK decode +0.7% prefill +0.0% md5 SAME errors 0
VERDICT b1 vs b0t OK decode +0.2% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH ngw c2000_cold vs b0t decode +0.4% prefill -1.6%
VERDICT_DEPTH ngw c2000_turn2 vs b0t decode +0.3% prefill -1.3%
VERDICT baseline b0t errors 0 text ok
```

## 9. text (そのまま引用)

- TXT (全 COND): `# B-Trees: A Comprehensive Explanation ... ## 1. What is a
  B-Tree? ... A **B-tree (Balanced Tree) is a self-balancing tree data structure
  that maintains sorted data and allows searches, sequential access, insertions,
  and deletions in logarithmic time.`
- TXTB: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
  最も重要であり、かつ最も難しい概念です。」
- TXTD (t334b の c8000): 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）
  における **Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録した
  ものです。」
- 既定モデル (t334c): TXT `A **B-tree** (specifically, the **B+ tree) is the
  fundamental data structure used by most modern relational databases ...`、TXTB
  「Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、Rust の最も重要な
  概念であり、同時に初心者が最もつまずきやすい部分です。」
- errors は全 run 0。

