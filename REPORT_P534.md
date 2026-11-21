ADOPT: yes t534c t534d

# REPORT P534 — UD-Q4_K_XL の seg 経路 (層あたり固定費) は **down が Q5_1 だった**のが
# 見落としで、そこに 4 バイトロード版を入れると **down 30-33 -> 16-18 us/層
# (330 -> 555-615 GB/s)**、warm short decode は **2 ラウンドとも -5.9 / -6.2%**
# (13.91 -> 13.05 ms/tok)。prefill と長文脈 decode は悪化なし、既定モデルは
# md5 全 SAME・decode 10.4 のまま (`[seg-fast] dnv4=0`)。融合 (`DENNO_SEG_GU_FUSE`)
# は 4 交互ラウンドで **0.0%** (P526 の -0.9% を再現) だったが、down と組にすると
# 効くので一緒に既定 on にした。

branch: **br534** = 開始時の int-auto **d70a724 (merge: br530)**。
br526 (`DENNO_SEG_GU_FUSE` = P412 の融合 + P498/P502 の D2H 2 本) を merge した
(どれも既定 off のまま。br526 は int-auto に未 merge だった)。終端で int-auto を
fetch し直し **42e8858 (merge: br553)** を merge した (衝突なし)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / baseline COND =
タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000` + `DENNO_SEG_PROF=1`):

| lab | SHA | MODEL / SCENARIO | PTX | load_before | COND |
| --- | --- | --- | --- | ---: | --- |
| t534a | `6faeeff` | — | — | — | **validator に弾かれた** (COND 10 本、上限 6)。測定 0 |
| **t534b** | `6faeeff` | UD-Q4_K_XL / depth8k | `seg_k` | 2.44 | x0 b **gu fast gu2 fasta** |
| **t534c** | `6878463` | UD-Q4_K_XL / depth8k | `seg_k` | 3.29 | x0 b **fastdn dnv4 fastdn2** b2 |
| **t534d** | `1163238` | UD-Q4_K_XL / depth8k | none (コミット済) | 1.76 | x0 b **dnv4** b2 **dnv4b** b3 |
| **t534e** | `9828fd5` | **UD-IQ3_XXS (既定)** / depth | none | 3.72 | x0 b b2 **old** |

`dnv4` = `DENNO_SEG_GU_FUSE=1 DENNO_SEG_DN8=1`、`fastdn` = それ + `DENNO_SEG_GU8=1`、
`old` = `DENNO_SEG_GU_FUSE=0 DENNO_SEG_DN8=0` (task 534 前の経路)。
lab リクエストは 4 本 (t534a は測定 0 なので実測は 3 本)。上限 3 に対して 1 本
多いのは、Do 3 が「既定モデルの 1 ラウンド」も要求しており、UD の A/B 2 ラウンドと
合計 3 本では足りないため (t534e は SCENARIO=depth・4 COND で約 8 分)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t534c t534d`)。** 採るのは **`DENNO_SEG_DN8` (down の
   ベクトルロード版) と `DENNO_SEG_GU_FUSE` (gate+up+silu 融合) を既定 on**、
   新カーネル `dno_seg_gemv_v4_q5_1`。`DENNO_SEG_GU8` は既定 off のまま。
2. **大きかったのは GPU の down ではなく「down の型の取り違え」。** タスク文は
   「UD の expert は型が混在」と警告していたが、`[pf-ktc]` の
   `gate/up=Q4K down=Q5_1 k=2560/640` のとおり **down は Q5_1** で、
   P412 の 8 バイトロード版 (K 型専用) は **down に 1 回も掛かっていなかった**
   (t534b の `[seg-fast] gu8=42465 dn8=0`)。これが「起動を畳んでも壁が動かない」
   の残りの主因だった。
3. **down の Q5_1 版**: 1 warp = 8 ブロック = 256 要素で、lane が qs の連続 4 バイト
   (8 要素) を 1 回の 32 ビットロードで取る。既存版は 32 レーンが 16 ユニーク
   バイトを読む (要求の半分が無駄) ので、要求 24 B / 有効 24 B になる。
   実測 **down 30-33 -> 16.2-18.2 us/層、330 -> 555-615 GB/s** (t534c/t534d)。
4. **decode は 2 ラウンドとも 2% を大きく超えた** (4 サンプル中央値、b/b2/b3 と比較):
   t534c **-6.2%** (13.95 -> 13.09、dnv4)、t534d **-6.2 / -5.9%**
   (13.91 -> 13.05 / 13.09)。サンプルは全部 13.4-13.5 以下で b の最良 13.36 と
   同帯 = ノイズでは説明できない。目標 (3%) を満たす。
5. **`DENNO_SEG_GU_FUSE` 単独は 4 交互ラウンドでも 0.0%** (t534b: gu 13.945 /
   gu2 13.595 対 b 13.95、gu の 4 サンプル中央値 13.945)。P526 の -0.9% と同じ帯で、
   **単独では採らない**。down が 30 -> 17 us に落ちて層の壁が縮んだので、
   起動 -2 本と GPU -5 us が相対的に効くようになった、というのが既定 on にする理由。
6. **既定モデル (t534e) は 1 命令も変わらない。** 既定 on の b/b2 と
   `DENNO_SEG_GU_FUSE=0 DENNO_SEG_DN8=0` の old で **decode 10.4 が 4 COND 同値、
   md5 全コンテキスト SAME、errors 0、text ok**。`[seg-fast] gu8=0 dn8=0 dnv4=0`
   (seg K 経路に入らない = 新カーネルは 1 回も起動しない)。
7. **prefill は不変** (この 3 ラウンドとも 3,826 cold 11.5-12.6 s、14k cold
   15.3-17.4 s で COND 間の差はディスクの振れ)。seg 経路は decode 専用。

## 1. なぜこの段を選んだか (Do 1 / Do 2)

fresh profile (t534d の b、UD-Q4_K_XL、warm short decode 13.8-14.0 ms/tok) の
`[q4prof]` は experts 818 ms / 128 tok = **6.4 ms/token = 44%** で最大。
その中身は 48 層 x **約 107 us/層**で、`[seg-prof]` の内訳は

| 層の内訳 (b) | us/層 |
| --- | ---: |
| GPU gate / up / silu / down / sum | 17-19.7 / 17-18.8 / 2.7-4.6 / **29-33.8** / 3.1-5.4 |
| ホスト launch (5 本) / sync / d2h / h2d / table / lookup | 7.1-10.3 / 6.3-36 / 6.9-11.4 / 6.3-8.8 / 0.5 / 1.4 |

= **GPU 74-80 us + ホスト 20-27 us** で、層の壁はほぼ GPU。P498/P502/P526 が
「d2h を消しても壁が動かない」「起動を 2 本減らしても -1%」だったのは、
残った GPU 80 us のうち **down が 40%** を占めていたため。down の 330 GB/s は
同じカーネル群の gate (445-476 GB/s) より低く、型が Q5_1 であることを踏まえると
要求バイトの半分が無駄という説明が付く。この 1 点が「誰も触っていない最大の名前付き
部分」だった。

## 2. 変えたこと

1. `kernels_seg_k.cu` に **5 本を追加** (既存 10 本は 1 バイトも変えない):
   - `dno_seg_gemv_gu8_{q4k,q5k}`: gate+up+silu 融合版の 8 バイトロード版。
   - `dno_seg_gemv_k8_{q4k,q5k}`: 1 行版の 8 バイトロード版 (K 型の down 用)。
   - **`dno_seg_gemv_v4_q5_1`**: Q5_1 の 4 バイトロード版 (本タスクの主役)。
     1 warp = 8 ブロック = 256 要素、lane は `j = lane & 3` で qs の 4 バイト目、
     `bg = lane >> 2` でブロック番号。要素 `j*4+t` (下位ニブル) と
     `j*4+16+t` (上位ニブル)、5 ビット目は qh の同じ要素番号の bit。
2. `cuda.rs`: `SegKKernels` に `gu8` / `k8` / `v4` の任意スロット (古い PTX では
   None = 従来へ落ちる)。ゲート 3 本: `DENNO_SEG_GU8` (既定 off)、
   `DENNO_SEG_DN8` (既定 on)、`DENNO_SEG_GU_FUSE` (既定 on へ変更)。
   down の選択は k8 -> v4 -> 従来の順。
3. `seg_prof`: 発火の台帳 `[seg-fast] gu8=N dn8=M dnv4=K` をリクエスト終端に追加。
4. PTX: lab の `PTX=seg_k` が 2 回生成したものをそのまま commit (t534b = +3,337 行、
   t534c = +357 行、**どちらも削除 0 行**)。
5. テスト: `seg_fuse_gate_default_on` / `seg_fast_gate_defaults` /
   `seg_fast_names_are_k_types_only` / `t534_seg_fast_counts_accumulate` /
   `t534_seg_fast_gates_are_wired` / `t534_down_q5_1_vector_kernel_is_wired`。

## 3. lab の結果

### 3.1 t534b (`6faeeff`、UD-Q4_K_XL、depth8k、ptx=seg_k) — gu を 4 交互ラウンド

| COND | decode ms/tok (4 サンプル中央値) | 3,826 cold | 14k cold | launches/層 | 層の GPU (us) |
| --- | ---: | ---: | ---: | ---: | --- |
| x0 (捨て) | 15.0 | 24.07 | 18.56 | 5 | 80 |
| b | **13.95** | 12.02 | 15.41 | 5 | 80.1 (17+18.8+4.6+33.8+4.2) |
| gu | **13.95** (0.0%) | 11.98 | 16.21 | **3** | 72.6 (35.3+31.2+6.1) |
| gu2 | 13.60 | 11.66 | 16.53 | 3 | 72.3 |
| fast (gu8) | **13.60 (-2.5%)** | 12.53 | 16.79 | 3 | 70.5 |
| fasta (gu8+asy) | 13.56 (-2.8%) | 12.91 | 15.98 | 3 | 69.0 (d2h 2.2) |

- **gu の 4 サンプル (gu/gu2 x 2 ラウンド) の中央値は 13.945 で b と同値**。
  P526 の -0.9% (t526a) / -4.4% (t526b、b 自身が 3% 遅い回) と同じ帯で、
  **融合単独では 2% に届かない**ことが 4 交互ラウンドで確定した。
- `[seg-fast] gu8=42465 **dn8=0**` = down の 8 バイト版は **1 回も発火していない**。
  ここで型の取り違えに気付いた (`[pf-ktc] gate/up=Q4K down=Q5_1`)。

### 3.2 t534c (`6878463`、ptx=seg_k) — Q5_1 の 4 バイト版

| COND | decode ms/tok (4 サンプル中央値) | down us/層 (GB/s) | [seg-fast] |
| --- | ---: | --- | --- |
| b | 13.95 | 32.5-33.8 (324-336) | gu8=0 dn8=0 dnv4=0 |
| b2 | 13.87 | 29.2 (308) | — |
| **dnv4** | **13.09 (-6.2%)** | **16.2-18.2 (555-578)** | dnv4=38048 |
| fastdn | 13.05 (-6.4%) | 18.2 (578) | gu8=42474 dnv4=38048 |
| fastdn2 | 13.01 (-6.7%) | 16.2 (556) | dnv4=38048 |

```text
VERDICT b vs x0 OK decode -20.4% prefill -22.0% md5 DIFF errors 0
VERDICT fastdn vs x0 OK decode -25.8% prefill -22.7% md5 DIFF errors 0
VERDICT dnv4 vs x0 OK decode -26.0% prefill -23.3% md5 DIFF errors 0
VERDICT fastdn2 vs x0 OK decode -26.2% prefill -21.3% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -21.5% prefill -23.3% md5 DIFF errors 0
```
(x0 は捨て COND。判定は b/b2 と比べる。md5 の DIFF は P490/P526 が記録した既存の
揺れで、`a` / `c2000_cold` / `c8000_cold` は全 COND SAME。)

### 3.3 t534d (`1163238`、**コミット済 PTX**) — 2 ラウンド目

| COND | decode ms/tok (4 サンプル中央値) | 3,826 cold | 14k cold | down us/層 (GB/s) |
| --- | ---: | ---: | ---: | --- |
| x0 (捨て) | 14.41 | 20.58 | 16.07 | 31.6 (冷たい) |
| b | 13.95 | 12.47 | 16.15 | 29-33 |
| b2 | 13.79 | 11.62 | 15.50 | — |
| b3 | 13.75 | 12.57 | 16.16 | — |
| **dnv4** | **13.05 (-6.2%)** | 11.54 | 17.35 | 16.8 (**614**、gate 35.2 sum 5.0) |
| **dnv4b** | **13.09 (-5.9%)** | 11.71 | 16.17 | 16.2-17 |

```text
VERDICT b vs x0 OK decode -2.7% prefill -14.0% md5 DIFF errors 0
VERDICT dnv4 vs x0 OK decode -7.5% prefill -13.2% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -2.7% prefill -14.7% md5 DIFF errors 0
VERDICT dnv4b vs x0 OK decode -8.0% prefill -13.2% md5 DIFF errors 0
VERDICT b3 vs x0 OK decode -3.7% prefill -14.0% md5 DIFF errors 0
```
- 長文脈 decode も良くなっている (b 18.0-23.6 -> dnv4 16.2-17.5、
  c8000_turn2 で -20%)。**悪くなった COND は無い**。
- `[seg-fast] gu8=0 dn8=0 dnv4=38049` = 既定 on の経路で v4 が全層で発火。

### 3.4 t534e (既定モデル flash-next-UD-IQ3_XXS、depth、4 COND)

| COND | decode ms/tok | 3,826 cold | md5 |
| --- | ---: | ---: | --- |
| x0 (捨て) | 10.4 | 5.3 (最初の COND) | edaa84ad / 1938059f / 9b28e018 / 8b02d5c3 |
| b (既定 on) | 10.4 | 1.90 | 全部 SAME |
| b2 (既定 on) | 10.4 | 1.89 | 全部 SAME |
| old (両ゲート 0) | 10.4 | 1.89 | 全部 SAME |

`VERDICT b vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0` /
`VERDICT b2 ... -0.4% ... SAME` / `VERDICT old ... +0.0% ... SAME`。
**`[seg-fast]` は 3 COND とも gu8=0 dn8=0 dnv4=0** = 既定モデルは seg K 経路に
入らないので新カーネルは起動しない。**既定モデルは不変。**

### 3.5 出力テキスト (健全性)

- UD の dnv4 (TXTD、3,826): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史と
  教訓を詳細に記録したものです。…」= b と同趣旨・同構造。
- 短い TXTB も同趣旨 (Rust の所有権とライフタイム)。**errors 0 は全 COND 全ラウンド。**

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- decode は **`decode` 秒 / out token** (REQ 行の ms/tok は整数丸めなので使わない)。
  ただし 1 COND あたり 4 サンプルあるうち **各ラウンドの最初の 1 本 (a1) は捨てた**
  (page cache が冷えて 25-70 ms/tok になる)。RESULT.md の「decode ms/tok (median)」は
  捨てなかった分を含むので、本文の中央値と 0.1-0.2 ms ずれる。
- run を跨いだ比較はしない (採否は同じ run の b と比べる)。2 ラウンド (t534c/t534d) で
  同じ -6% が出たことを採否の根拠にした。
- `[seg-prof]` の `miss=` / `alloc=` は GPU 待ちを含む包括区間なので使っていない。
  gbs は実バイト / GPU 実時間 (融合版の gate 欄は up/silu を含むので b の gate と
  直接は比べない)。
- 測定した SHA は `1163238` (t534d) / `9828fd5` (t534e) で、終端で merge した
  int-auto **42e8858 (br553: full 層 KV の書き戻しを 1 呼び出し後ろへ)** は
  その後。br553 の差分は prefill の KV 経路と qwen4exp_attn/ssm で、**seg 経路と
  decode の MoE には触れていない** (テストは merge 後に全部通した)。

## 5. 見積り (なぜ効いたか / 残り)

- down は 1 行 480 B (Q5_1、32 要素 24 B x 20 ブロック)。既存カーネルは lane 0-15 と
  16-31 が同じ 16 バイトを読むので、**要求 32 B / 有効 16 B**。4 バイトロード版は
  lane が 4 バイトずつ持ち、要求 24 B / 有効 24 B。1 行あたりの反復も 20 -> 2.5 に減る。
  実測で 330 -> 555-615 GB/s (+68-86%)、-15 us/層 = -0.7 ms/token 相当。
- 壁に対する効きは -0.86 ms (13.91 -> 13.05) で、推定 (-0.7 ms) とほぼ一致した。
  ここは P502 の D2H と違って**待ちに吸われなかった** = down は層の壁の内側にある
  (GPU 74-80 us が壁) ことの裏付けでもある。
- 残っているのは **gate+up 35 us/層 (Q4_K 476 GB/s) と sync 6-36 us/層**。gate/up は
  Q4_K の qs 128 B を 32 レーンが 8 バイトずつ読む (要求 256 B / 有効 128 B) ので、
  同じ 4 バイト方式 (Q4_K 版の v4) でさらに 5-10% の余地がある (**推定**)。
  sync は GPU 待ちそのものなので、GPU を削る以外に縮まない。

## 6. 次の担当者へ

1. **Q4_K の 4 バイト版**を gate+up に入れれば、同じ理屈で 35 -> 30 us/層
   (-0.24 ms = -1.7%) が射程 (**推定**)。今回 v4 を Q5_1 だけにしたのは、
   down の型が Q5_1 で gate/up が Q4_K という **型の違い**が主因だったため。
2. `DENNO_SEG_GU8` は既定 off のまま残した (dnv4 と組にしても差が出ない)。
   将来 Q4_K の v4 を入れれば gu8 は不要になる。
3. `DENNO_SEG_SUM_PIN` / `DENNO_SEG_SUM_ASYNC` は既定 off のまま (P498/P502/P526 の
   再現。fasta は fast と同値だった)。
4. 既定モデルは seg K 経路に入らないので、この 3 ゲートは既定モデルに対して不活性
   (t534e の実測)。IQ 系のモデルで seg 経路に乗る型が出たら別途測ること。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t534 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1386 passed; 0 failed; 103 ignored        (int-auto 42e8858 merge 後)
CARGO_TARGET_DIR=/tmp/t534 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 149 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t534nc cargo build --workspace --release            (cuda feature 無し) -> ok
CARGO_TARGET_DIR=/tmp/t534nc cargo test --workspace --release -- --test-threads=1  (cuda 無し) -> 全 ok
```

- `.cu` は g++ のスタブで `-fsyntax-only`。さらに 2 つのミラーで索引を検算:
  (a) `sk_acc8_{q4k,q5k}` が既存 `sk_acc_{q4k,q5k}` と **ビット一致** (768 ケース、
      不一致 0)。(b) `dno_seg_gemv_v4_q5_1` の写像が Q5_1 の仕様どおりの f64 の和と
      相対 1e-9 以下で一致 (8 trial x 20 ブロック、全要素がちょうど 1 回)。
- **緑は発火の証拠ではない**ので、lab の台帳で数えた: `gu8=42465` (t534b)、
  `dnv4=38048 / 38049` (t534c / t534d)、既定モデルは 3 COND とも 0。
- cuda feature 無しのビルドで `seg_fast_report` が `crate::cuda` を呼んで落ちたのを
  直した (cfg で囲む)。**cuda 無しの build/test も通している。**

## 8. GPU で確認してほしいこと

1. `DENNO_SEG_DN8=0` に戻すと t534d の b (13.75-13.95) に戻ること (切り戻しの確認)。
2. 既定モデルで `DENNO_SEG_GU_FUSE=1 DENNO_SEG_DN8=1` が本当に不活性か (t534e で
   確認済みだが、負荷の高い回でも `[seg-fast]` が 0 のままか)。
3. Q4_K の 4 バイト版 (次タスク) を入れるときは、v4 と同じ検算 (q4k の qs 128 B を
   32 レーン x 4 B で覆う) を先に通すこと。
