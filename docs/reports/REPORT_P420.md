ADOPT: yes t420b t420c

# task 420 (dev3) — prefill のプール計画を K 系の dev MoE + TC 経路に合わせる
# (UD-Q4_K_XL の 14k cold 17.5 -> 13.6 s = -23%、既定モデルは md5 全一致)

branch: **br420** = 開始時の int-auto `ff1a1ec` (merge: br416) から。
br416 は既に int-auto に入っていたので merge は不要。終端でも int-auto は
`ff1a1ec` のまま (動いていない) なので merge は無い。

数値はすべて GPU ラボの実測 (`PROFILE=1`)。lab 名と測った SHA:
**t420a = `ff1a1ec`** (UD、depth8k、4 COND x 2 ラウンド、env だけの切り分け)、
**t420b = `b187ed1`** (UD、depth8k、4 COND x 2 ラウンド、本ゲート on)、
**t420c = `eb2c129`** (既定モデル `flash-next-UD-IQ3_XXS`、depth8k、4 COND x 2 ラウンド)。
3 ラウンド (上限 3)。

## 0. 結論 (5 行)

1. **採る (`ADOPT: yes t420b t420c`)。** UD-Q4_K_XL の **14k cold prefill は
   13.56 s** 対 base 17.53 / 17.76 s = **-23%**。目標 (-5%) の 4.6 倍。
2. **3,826 cold は変わらない** (9.56 / 8.37 対 base 10.13 / 9.54 / 10.70 / 8.94。
   ラウンド間のディスクの振れの内側)。3,826 は既に 1 パスで 44 GB の pread が壁。
3. **やったこと**: prefill のプール計画の 2 か所が IQ 版の述語のままだったのを
   K 系 (Q4_K/Q5_K/Q6_K/Q5_1/Q8_0) の dev MoE + TC 経路に合わせた
   (`DENNO_PF_POOL_K_PLAN`、lab t420b で**既定 on**)。14k が 2 パス (タイル
   7,009) から **1 パス (タイル 14,018)** になり、expert 重みの pread が
   98.5 GB -> 52.5 GB になった。
4. **既定モデルは無変化。** 4 COND x 6 文脈の md5 が**全部 SAME**
   (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)、
   decode 13.2-13.3、3,826 2.30-2.31 s、14k 5.63-6.88 s、errors 0。
5. **decode は悪化しない。** UD の短い decode 16.4 対 16.2、14k 後の decode
   22.0 対 21.8 / 22.0 ms/tok。t420a で decode が +10% 落ちた形 (作業域 4,779 +
   余白 3,072 MiB) と違い、追い出しが base より少ない (543 本 対 957 本)。

## 1. 前提の測り直しと、なぜこの塊を選んだか (Do 1 / Do 2)

### 1.1 現行 int-auto の UD-Q4_K_XL 14k の段 (t416b ktc = いまの既定と同じ)

`[q4ledger-pf] tok=14018 wall=17,980.8ms` (REPORT_P416 §1.2 と同じ台帳):

| 段 | wall | 割合 | busy / exposed |
| --- | ---: | ---: | --- |
| experts | 13,349.7 ms | 74.2% | 1,321.0 / 12,028.8 |
| ssm | 2,167.2 | 12.1% | 1,314.3 / 852.9 |
| other | 1,578.9 | 8.8% | 171.4 / 1,407.4 |
| full | 862.6 | 4.8% | 638.8 / 223.8 |
| mix | 22.3 | 0.1% | 882.8 / 0.0 |

`[exp-pf-stream] layers=96 groups=1421 experts=31343 rows_per_expert=214.7
up_MB=98460.4 copy_s=12.69 pread_calls=94029` で、experts 段の 12.0 s は
**pread 待ち**。task 416 で GPU 側 (gpu_busy 18.34 -> 0.43 s) が消えたので、
prefill の最大の塊は**重みの読み**に移っていた。ここを触るのが最大の一手。

### 1.2 なぜ 98.5 GB 読んでいたか (この task の発見)

`[pf-pool-big] 形を上げた (チャンク 1536 → 2047 / タイル 5120 → 7009 = 3 → 2 パス
/ 作業域 3203 MiB)` = **14k を 2 パスで回していた**。1 パスにすると同じ層で
expert を読み直さないので 98.5 -> 約 49 GB になる (実測 52.5 GB)。

1 パスが選ばれない理由は、プール計画の上限 (4,864 MiB) に 1 パスのレイアウトが
収まらないこと。収まらない理由は**計画側が K 系の経路を知らない**ことだった:

1. `pf_pool_need_for` の `tc` が IQ 版の `tc_shapes_ok` だけを見る。UD の
   expert は Q4_K / Q5_1 で IQ のスロットに無いので「TC に乗らない」= GEMV と
   判定され、**実際には使わない g / u の作業域**を予約していた
   (14k で 686 MiB)。
2. `pf_pool_reserve_for` の `dev_ok` が `iq_expert_mode` のままで、UD は
   「dev MoE が使えない」= CPU 経路と判定され、余白が 3,072 MiB
   (dev 経路は 1,024 MiB)。doc にも「IQ カーネル非対応 (UD-Q4_K_XL など)」と
   書いてあるが、これは task 309/318 で dev MoE が K 系を受けるようになる前の話。

task 279 が IQ3_S で 1 パス (-39/-50%) を採ったときの作業域は 4,094-4,341 MiB で
上限に収まっていた。UD は上記 2 つで 5,000 MiB を超え、**同じ 1 パスの仕組みが
黙って効かない**状態だった。ここが「まだ誰も触っていない」最大の塊。

## 2. 変えたこと

1. **`DENNO_PF_POOL_K_PLAN`** (exp_pf.rs、lab t420b で既定 on)。純関数
   `pool_k_plan_from_value` は既定 on・**明示的な偽値 (`0`/`off`/`false`/`no`)
   だけ off** (task 265 の 3 本と同じ規約)。切り戻しはこの 1 本。
2. **`pf_pool_need_for` の `tc`** (qwen4exp.rs)。ゲート on では実行時と**同じ
   2 段** (`IQ → K`) を**全 MoE 層**で評価する。1 層でも乗らない層があれば
   `false` = g / u を取る (安全側。取り過ぎても遅いだけで壊れない)。
   - 純関数 `pf_pool_tc_all(per_layer, iq, k)`: 「全層 IQ」または「全層 K かつ
     どの層も IQ の形に合わない」のときだけ真。IQ の形に合って PTX が無い層は
     K へ落ちない (実行時もそう) ので、そこも区別する。
   - `pf_pool_tc_layers`: 層ごとに `(IQ の形, K の形)` を `tc_shapes_ok` /
     `ktc_shapes_ok` で作る。
3. **`pf_pool_reserve_for` の `dev_ok`** (qwen4exp.rs)。ゲート on では
   `iq_expert_mode` ではなく `pf_expert_mode` (IQ or K) を使う。これは dev MoE の
   実行時の判定 (task 318) と同じ述語で、余白が CPU 経路の 3,072 から dev 経路の
   1,024 MiB になる。
4. **発火の証拠**: `[pf-pool-big] 形を上げた (... タイル 5120 → 14018 = 3 → 1 パス
   / 作業域 4094 MiB)` が kk の 14k のログ (両ラウンド) に出る。
   `[exp-pf-stream] layers=48` (base は 96 = 2 パス) と
   `up_MB=52516` (base 98,173) が台帳側の証拠。

## 3. 数値の扱い

- **値は変えない。** 変わるのは作業域の大きさ・タイル・余白だけで、重みも活性も
  丸めは変えない (TC 経路は task 416 で既に既定 on)。予想どおり UD の
  `c8000_cold` `20779eab`、`c8000_turn2` `e6398d57`、`a` `737a73af`、`b` `87d8a97c`
  は **4 COND とも SAME**。
- `c2000_turn2` だけ `95095dd7` と `c519179d` の 2 値が出るが、**ゲート off の
  base でも両方出る** (b2_1 95095dd7 / b2_2 c519179d) 既存の揺れ (t416b の b1 も
  `67e08087 / c519179d` の 2 本)。この task の変更とは無関係。
- 決定性: パス数が変わると「どの層でどの expert をまとめて上げるか」は変わるが、
  1 トークンの和は k 昇順のままなので値は変わらない (md5 が SAME であることが
  その証拠)。行チャンク不変・par/seq 一致はテストで維持。

## 4. lab t420a: env だけで 1 パスの効果を切り分ける (sha `ff1a1ec`)

`DENNO_PF_POOL_BIG_MB=8192` (ゲート無しで上限だけ上げる) を `p8` として 4 COND。

| cond | env | decode ms/tok | 3,826 cold (中央値 r1 / r2) | 14k cold | 14k 後 decode |
| --- | --- | ---: | --- | ---: | ---: |
| x0 (捨て) | 基準 | 18.1 | 10.28 (12.06 / 8.50) | 17.41 | 24 |
| b1 | 基準 | 16.3 | 14.85 (20.69 / 9.00) | 17.47 | 23 |
| p8 | `POOL_BIG_MB=8192` | 22.6 (再送回) | 9.88 (10.84 / 8.91) | **15.03** | 26 |
| b2 | 基準 | 16.2 | 10.00 (11.16 / 8.85) | 17.50 | 24 |

- 14k は **-14.1%** (15.03 対 17.41-17.50)。3 本の base が ±0.5% に収まっている。
- `[exp-pf-stream] layers=48 up_MB=53322.9 copy_s=9.75` (base 98,460 / 12.69)。
  作業域 **4,779 MiB**、追い出し **1,500 本**、終端で 4,602 MiB を 269 ms で戻す。
- **14k 後の decode が 26 対 23-24 ms/tok に落ちた** (-10% の悪化)。
  1,500 本の追い出しが原因と見て、次のラウンドで「作業域と余白を小さくする」
  方向 (本ゲート) に振った。→ これが当たった (§5)。

### VERDICT (t420a)

```
VERDICT b1 vs x0 OK decode -10.1% prefill -18.0% md5 DIFF errors 0
VERDICT p8 vs x0 OK decode +24.6% prefill -20.7% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -10.8% prefill -20.7% md5 DIFF errors 0
VERDICT_DEPTH p8 c2000_cold vs x0 decode -6.5% prefill -3.9%
VERDICT_DEPTH p8 c8000_cold vs x0 decode +17.4% prefill -13.7%
VERDICT_DEPTH p8 c8000_turn2 vs x0 decode +9.4% prefill +8.2%
VERDICT_DEPTH b2 c8000_cold vs x0 decode +4.3% prefill +0.5%
VERDICT baseline x0 errors 0 text ok
```
(14k は depth8k の仕様で**ラウンド 1 にしか流れない**ので COND あたり 1 本。
p8 の decode +24.6% は再送の回 (a の md5 が 2 本) の外れ値。判定は上の表で読む。)

## 5. lab t420b: 本ゲート (sha `b187ed1`、gate on = `kk`)

| cond | env | decode ms/tok | 3,826 cold (中央値 r1 / r2) | 14k cold | 14k 後 decode |
| --- | --- | ---: | --- | ---: | ---: |
| x0 (捨て) | 基準 | 16.9 | 10.70 (12.46 / 8.94) | 17.76 | 25.2 |
| b1 | 基準 | 16.7 | 9.36 (10.21 / 8.50) | 37.46 (外れ) | 25.9 |
| kk | `POOL_K_PLAN=1` | **16.4** | 9.56 (10.76 / 8.37) | **13.56** | 22.0 |
| b2 | 基準 | 16.2 | 10.13 (10.72 / 9.54) | 17.53 | 21.8 |

- **14k は 13.56 s 対 17.53 / 17.76 s = -23%**。
  `[pf-pool-big] 形を上げた (... タイル 5120 → 14018 = 3 → 1 パス / 作業域
  4094 MiB)`、`[exp-pf-stream] layers=48 up_MB=52515.9 copy_s=8.64`、
  `up_wait=8549.5` (base 12,528)。余白は **1,024 MiB**、追い出しは **543 本**
  (base 957 本) で、t420a の 1,500 本より少ない。
- b1 の 37.46 s は**ディスクの外れ回** (`copy_s=21.64`、同じ COND の 2 ラウンド目は
  9.36 s)。判定は b2 / x0 と比べる。
- **3,826 は差なし** (9.56 対 9.36 / 10.13。ラウンド間の振れ 8.37-10.76 の内側)。
- **14k 後の decode 22.0 対 21.8 / 25.2 ms/tok**、14k cold の decode 27.5 対
  32.3 / 36.3、短い decode 16.4 対 16.2。どこも悪化しない。
- md5 は §3 のとおり (**c8000 の 2 文脈と a / b は 4 COND とも SAME**)。
- errors 0、text 健全 (§7)。

### VERDICT (t420b)

```
VERDICT b1 vs x0 OK decode -1.4% prefill -13.7% md5 DIFF errors 0
VERDICT kk vs x0 OK decode -2.8% prefill -14.4% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -4.4% prefill -14.4% md5 DIFF errors 0
VERDICT_DEPTH kk c2000_cold vs x0 decode -22.6% prefill -10.6%
VERDICT_DEPTH kk c8000_cold vs x0 decode -24.3% prefill -23.6%
VERDICT_DEPTH kk c8000_turn2 vs x0 decode -13.0% prefill -10.6%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -11.2% prefill -1.3%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode -13.6% prefill -12.1%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT は x0 比。x0 は捨て COND なので判定は本文の b2 との比較で読む。
`kk c8000_cold -23.6%` が x0 17.76 に対する数字。)

### 段の台帳 (kk、14k、`[q4ledger-pf]` は base の値と比べる)

kk の 14k は `up_wait=8549.5 ms` で experts 段の露出が 12.0 -> 8.5 s に縮み、
壁は 13.56 s。残りは ssm / other / full (base で 2.17 / 1.58 / 0.86 s) と
GPU 側 (base で busy 4.33 s) で、**pread 8.6 s がまだ壁の 64%**。

## 6. lab t420c: 既定モデル (sha `eb2c129`、`kk` = ゲート未設定 = 既定 on)

| cond | env | decode ms/tok | 3,826 cold | 14k cold | 14k 後 decode |
| --- | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | `POOL_K_PLAN=0` | 13.2 | 2.30 | 6.88 | 15.2 |
| b1 | `POOL_K_PLAN=0` | 13.2 | 2.31 | 6.03 | 14.8 |
| kk | (未設定 = on) | 13.3 | 2.31 | 5.63 | 14.8 |
| b2 | `POOL_K_PLAN=0` | 13.2 | 2.30 | 5.65 | 14.8 |

- **md5 は 4 COND x 6 文脈の全部が SAME** (`edaa84ad / 1938059f / b42d79d2 /
  12fc5c36 / 672fec07 / 98e879d2`)。IQ 系は `pf_pool_tc_all` が「全層 IQ」で真、
  `pf_expert_mode` は元から真なので、**計画が 1 バイトも変わらない**。
- 14k は kk 5.63 対 b2 5.65 (ゲート off) で同じ。b1 6.03 / x0 6.88 の差は
  ラウンド順のノイズ (同じゲート off どうしが 15% 離れている)。
- errors 0、text 健全。

### VERDICT (t420c)

```
VERDICT b1 vs x0 OK decode -0.3% prefill +1.7% md5 SAME errors 0
VERDICT kk vs x0 OK decode +0.6% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.3% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH kk c8000_cold vs x0 decode -3.0% prefill -18.2%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -3.0% prefill -17.9%
VERDICT baseline x0 errors 0 text ok
```

## 7. text (抜粋)

- UD kk (c8000_turn2、md5 e6398d57、base と SAME):
  `"ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と技術的深掘り**を記録したものです。\n\n以下に、文書の構造、主要な発見、および技術的な詳細を体�"`
- UD kk (b、md5 87d8a97c):
  `"Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリ管理をコンパイラが自動的に行"`
- 既定モデル: 4 COND とも `"# B-Trees: A Comprehensive Explanation..."` /
  Rust の文ではじまり、md5 も SAME。

## 8. 見積り (推定。実測ではない)

- 14k の壁は「1 パスの pread 8.6 s + GPU 4.3 s」で、重なりが効いて 13.56 s。
  pread は 52.5 GB / 8.64 s = **6.1 GB/s** で、これはこのホストの
  1 MB ランダム pread の帯域 (t416b の 98.5 GB / 12.7 s = 7.8 GB/s と同帯域)。
  これ以上速くするには **読む量を減らす**しかない。
- 3,826 は既に 1 パス (44 GB)。床は pread で、8.4-9.6 s のあいだで振れる。
- 次の一手の候補: (a) ssm 段 (14k で 2.17 s、うち exposed 0.85)、
  (b) other 段 (1.58 s、うち exposed 1.41。ple 214 / embed 297 / res_up 152 ms)、
  (c) decode の expert gpu_ms (14k 後で 155-215 ms)。experts はもう 0.43 s。

## 9. テスト (最終状態)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1,297 passed / 0 failed / 103 ignored**
- `... --test kernel_invariants`: **120 passed / 0 failed**
- cuda 無しの CI 相当: `cargo build --workspace --release` 通過、
  `cargo test --workspace --release -- --test-threads=1` **failed 0**
- 追加 4 本:
  - `pool_k_plan_gate_truth_table` (exp_pf): 既定 on と明示的な偽値だけ off。
  - `pf_pool_tc_all_truth_table` (qwen4exp): 全層 IQ / 全層 K / 混在 / PTX 無し /
    ゲート off の真理値表。
  - `ud_q4_k_xl_shape_takes_the_k_path_only` (cuda): UD の層 0 の形
    (e=2560 / ff=640 / Q4_K + Q5_1) が K 版だけに乗ることを固定。
  - `ud_k_model_one_pass_needs_the_k_tc_plan` (pf_pool): UD の 14k が
    旧計画 (g/u + 余白 3,072) ではタイル 7,009、新計画 (g/u 無し + 余白 1,024)
    では 14,018 (1 パス) になることを**純関数だけで**固定。lab の実測
    (7,009 / 14,018、作業域 4,094 MiB) と一致する。

## 10. GPU で確認してほしいこと / 残し

1. 既定 on にした。切り戻しは `DENNO_PF_POOL_K_PLAN=0` (純関数テストで固定)。
2. 既定モデルは構造的に不発 (全層 IQ なので計画が同じ)。t420c で md5 全一致。
3. 3,826 は 1 パスのまま変わらない。次に 3,826 を判定に使うときは `copy_s` を
   併記する (ラウンド間で 8.4-21.6 s 振れた)。
4. 次の最大の塊は prefill の pread (14k で 8.6 s = 壁の 64%)。読む量を減らす
   以外に手が無いので、ssm / other の露出を削るのが現実的。
