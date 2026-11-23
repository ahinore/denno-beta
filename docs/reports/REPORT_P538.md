ADOPT: yes t538b t538c

# REPORT P538 — UD-Q4_K_XL の staging copy (53 の残り) は「**1 回の写し = 8 MiB の
# 塊にディスク読みが 8 本しか入らない**」のが壁で、`DENNO_PF_SPLIT_WIDE_KB`
# (ディスクから読むグループだけ塊を **32 MiB** へ) + `DENNO_PF_PREAD_KB` (組で
# 1 MiB) にすると `copy_s` が **10.4 -> 8.0 s**、3,826 cold が **11.63 -> 9.38 s
# (-19.3%)**、14k cold が **16.52 -> 14.34 s (-13.2%)** (t538c、2 サンプル)。
# 短い decode と prefill after cut は不変 (13.1 -> 13.2 ms/tok / 0.47 -> 0.48 s)、
# 既定モデルは発火せず不変 (t538a で rd_n=0 / md5 全 SAME)。塊を全グループで
# 広げると既定モデルが **1.89 -> 2.12 s (+12%)** になるので、ゲートは
# `src` を持つ (= ファイルから読む) グループだけに限った。

branch: **br538** = 開始時の int-auto **7510bc0 (merge: br534)**。br534 は既に
int-auto に merge 済みだったので取り込む物は無い。終端でも int-auto を fetch し
直したが **7510bc0 のまま** (merge する物は無し)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `SCENARIO=depth8k`):

| lab | SHA | MODEL | COND | 備考 |
| --- | --- | --- | --- | --- |
| t538a | `96332c1` | **flash-next-UD-IQ3_XXS (既定)** | x0 b **u2 t64 u2t** b2 | `MODEL=` を書き忘れて既定モデルで走った (§3.1) |
| **t538b** | `b807800` | flash-next-UD-Q4_K_XL | x0 b **k1 k1t k2** b2 | 本命の A/B 1 回目 |
| **t538c** | `b03d34a` | flash-next-UD-Q4_K_XL | x0 **b k2 w8 old** b2 | 既定を入れた後の確認 (2 回目) |

baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
COND の意味: `u2` = `DENNO_PF_UP_SPLIT_KB=32768` (全グループで塊を広げる)、
`t64` = `DENNO_PF_PREAD_THREADS=64`、`k1`/`b` = `DENNO_PF_PREAD_KB=1024
DENNO_PF_SPLIT_WIDE_KB=32768` (= 採った既定)、`k1t` = k1 + threads 64、
`k2` = 1024 + wide 65536、`w8` = `DENNO_PF_PREAD_KB=384` (wide は既定 on)、
`old` = `DENNO_PF_PREAD_KB=384 DENNO_PF_SPLIT_WIDE_KB=0` (task 534 の経路)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t538b t538c`)。** `DENNO_PF_SPLIT_WIDE_KB` を既定
   **32768 KiB** にし、`DENNO_PF_PREAD_KB` 未設定なら wide のとき **1024 KiB** を
   使う (`pread_kb_default`)。両方 env で戻せる。
2. **壁の正体 (530 の続き)**: 分割 DMA は 8 MiB の塊ごとに `pf_copy_into_pin` を
   呼び、その塊の中の**ディスク読み (0.93 MB の区間) は 8 本**しかない。塊の
   最後の 1 本まで barrier が待つので、同時に飛ぶディスク読みは 8 本 (= 実効
   1.6 GB/s) で頭打ち。**塊を広げると本数が比例して増える**。
3. **実測 (t538b / t538c、3,826 cold、各 2 サンプル)**: old 11.69/11.58
   (t538c)、base 12.05/11.56 (t538b) に対し、wide+1024 は **9.66/9.30 (t538b) /
   9.43/9.34 (t538c)** = **-19.3%** (中央値 11.63 -> 9.38 s)。
   `copy_s` は **10.4 -> 8.0 s**。14k cold は **16.52 -> 14.34 s (-13.2%)**
   (t538c、1 サンプル)。
4. **ゲートの切り方**: p 広げると バウンス 2 枚 (2 x 32 MiB) が L3 (96 MB) から
   溢れて 481 の利得 (DMA が L3 を読む) を失う。**全グループで広げる `u2` は
   既定モデルの 3,826 cold を 1.89 -> 2.12 s (+12%、t538a で 2 ラウンド同値)**
   にした。だから wide は **`src` を持つ (= ファイルから pread する) グループ
   だけ**に効かせる。既定モデルは `src` を持たないので 1 命令も変わらない。
5. **`PREAD_KB` と wide は組で効く**。1024 単独は t530b で 16.0 s (384 は 11.8 s)
   と負けた (8 MiB の塊に 3 本しか入らない)。wide と組にすると勝つ。
   384 + wide (`w8`) も 9.18/9.19 = ほぼ同じ (§3.4) なので、既定は既に 2 ラウンド
   測った 1024 を残した (384 にしたい人は env 1 本で戻せる)。
6. **他の段は悪くない** (t538c、b 対 old): 短い decode 13.2 対 13.1 ms/tok、
   prefill after cut 0.48 対 0.47 s、14k 後の decode 23.5 対 26.6 ms/tok
   (wide の方が良い)。md5 は `a` / `c2000_cold` / `c8000_cold` が**全 COND SAME**、
   errors 0、text は全 COND `looks ok` (§5)。
7. **t538a は私のミスで既定モデルのラウンドになった** (req に `MODEL=` を
   書かなかった)。ただしそのおかげで (a) 既定モデルは wide を全グループに
   当てると +12% 遅くなる、(b) threads 64 と wide 単独では既定モデルの 3,826
   cold が動かない (`t64` 1.89、`u2t` 2.12 = `u2` と同値 → 差は块だけ)、
   (c) 既定モデルは staging の pread 経路を通らない (`rd_n=0`、台帳に
   `[exp-pf-stream]` の pread/rn 欄すら出ない) ことが分かった。無駄ではなかった
   が、UD のラウンドは 1 回損した。

## 1. なぜこの段を選んだか (Do 1 / Do 2)

Do 2 は「530 が見つけた staging copy の原因を直せ、閉じているなら 530 の数字から
UD prefill の壁の次の最大の部分を取れ」だった。530 は閉じている (ADOPT: yes、
`PREAD_KB` 384 KiB で 3,826 cold 15.9 -> 11.8 s)。530 の §6 が残した次は
**「ディスク実効 1.6 GB/s そのもの」**で、その数字は `copy_s` の内訳そのもの:

t538b の b (`UD-Q4_K_XL`、PROFILE=1、3,826 cold):

- `[exp-pf-stream]` `up_MB=43515 copy_s=10.72 enq_s=0.11 pread_MB=43745
  pread_calls=138644 group=32 usplit=5541 usplit_kb=8192`
- `rd_n=136924 rd_MB=41950.9 rd_max=29` (task 538 で足した計器。pread の同時本数)
- 壁は 3,826 cold 11.5-11.8 s で **copy_s が 91%**。compute は残り 1.4 s。
- 塊は 8 MiB x 5,541。1 塊 = 25 ブロックで、うちディスク読みは **8 本**
  (43.5 GB のうち 17 GB がディスク = 384 KiB x 44,000 本、5,541 塊で割ると 8 本/塊)。
- P522 の自前プローブでは 1 MiB x 8 本 = **3.5 GB/s**、4 MiB x 8 本 = 4.6 GB/s
  出ている。**1 本 0.93 MB を 8 本同時では 3.4 GB/s** なのに、我々は 384 KiB
  を 8 本で 1.6 GB/s。**サイズを上げると本数が落ち、本数を戻すと塊を広げる
  必要がある**、というのがこの段の構造。

触っていない最大の名前付き部分であり、530 の「次はここ」と一致する。

## 2. 変えたこと

1. `exp_pf::split_chunk_for(base, wide, has_src)` (純関数): `src` を持つ区間が
   1 本でもあるグループだけ `wide` を使う。
2. `DENNO_PF_SPLIT_WIDE_KB` (既定 **32768**、0 で従来)。`pf_up_run_split` が
   `split_chunk_for(up_split_bytes(), split_wide_bytes(), parts に src があるか)`
   で塊を決める。0 のときは `pf_up_run` が分割経路に入らないので off は完全に
   従来どおり。
3. `pread_kb_default(wide_on)`: `DENNO_PF_PREAD_KB` が**未設定**で wide が on
   なら 1 MiB、それ以外は従来の 384 KiB。明示値はいつでも優先。
4. 塊がグループごとに変わりうるので、バウンスを作り直す前に前の DMA を
   `cuEventSynchronize` で待つ (安全側。従来は塊がプロセス定数だったので
   この経路は踏まれなかった)。
5. 計器: `[exp-pf-stream]` に `usplit_wide` / `usplit_wide_kb` (発火の台帳) と
   `rd_n` / `rd_MB` / `rd_max` (pread の同時本数) を足した。数値経路は不変。
6. テスト: `task538_rd_inflight_ledger` / `task538_split_wide_gate_truth_table` /
   `task538_pread_kb_default_follows_wide` (lib)、
   `t538_staging_disk_inflight_is_counted` (kernel_invariants、配線の静的検査)。

## 3. 数値

### 3.1 t538a (`96332c1`、**既定モデル** flash-next-UD-IQ3_XXS、depth8k) — 既定の確認

| COND | decode ms/tok | 3,826 cold r1 / r2 | 14k cold (r1) | md5 |
| --- | ---: | --- | ---: | --- |
| x0 | 10.4 | **40.22** (最初の COND) / 1.91 | 6.94 | edaa84ad… |
| b | 10.4 | 1.89 / 1.89 | 5.81 | 全 SAME |
| u2 | 10.4 | **2.12 / 2.12** | 5.44 | 全 SAME |
| t64 | 10.4 | 1.89 / 1.89 | 4.99 | 全 SAME |
| u2t | 10.4 | **2.12 / 2.12** | 5.20 | 全 SAME |
| b2 | 10.4 | 1.90 / 1.89 | 5.04 | 全 SAME |

- **全 COND・全リクエストで md5 SAME、errors 0、text ok。** decode 10.4 不変。
- `u2` = 全グループで塊 32 MiB。**既定モデルの 3,826 cold だけ +12%** (1.89 ->
  2.12、2 ラウンド同値、threads を増やしても同じ)。推定: バウンス 2 枚
  (64 MiB) が L3 96 MB から溢れ、DMA が DRAM を読み直す (481 が避けた経路)。
  → これが「wide を pread グループだけに限る」根拠。
- 既定モデルは `[exp-pf-stream]` に `pread_MB=` も `rd_n=` も出さない
  (= staging の写しで pread を 1 回も使わない = `src` が全部 None)。
  だから wide も `pread_kb_default` も**このモデルには効かない**。
- 既定モデルの decode は 10.4 ms/tok で P530/P534 の値と同じ。

### 3.2 t538b (`b807800`、UD-Q4_K_XL) — A/B 1 回目

| COND | decode ms/tok | 3,826 cold r1 / r2 | 14k cold | `copy_s` | pread_calls | rd_max |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 14.7 | 16.26 / 11.63 | 17.66 | — | — | — |
| b | 13.2 | **12.05 / 11.56** | 23.58 (暴れ) | 10.72 | 138,644 | 29 |
| k1 (= 採った既定) | 13.2 | **9.66 / 9.30** | 13.01 | 8.25 | 55,808 | 32 |
| k1t (+threads 64) | 13.0 | 9.33 / 9.22 | 14.58 | 7.93 | 55,825 | 43 |
| k2 (wide 65536) | 13.2 | **9.00 / 8.89** | 14.09 | 7.03 | 55,709 | 32 |
| b2 | 13.1 | 11.60 / 11.41 | 16.43 | 10.33 | 138,520 | 28 |

- 3,826 cold の中央値: b 11.80 / k1 **9.48** (-19.7%) / k1t 9.28 / k2 8.95 / b2 11.50。
- t538b の 14k は 1 サンプル/COND で、b が 23.58 と暴れた回 (b2 は 16.43) =
  この回の 14k は判定に使わない (t538c で取り直した)。
- `copy_s` 10.72 (b) -> 8.25 (k1) -> 7.03 (k2)、`usplit_wide=678` は k1/k2 で
  発火し b/b2 は 0 (発火の証拠)。

### 3.3 t538c (`b03d34a`、既定を入れた後) — A/B 2 回目

| COND | decode ms/tok | 3,826 cold r1 / r2 | 14k cold | `copy_s` | pread_calls | usplit_wide |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 13.3 | 10.81 / 9.29 | 14.80 | — | — | — |
| **b (既定)** | 13.2 | **9.43 / 9.34** | **14.34** | 8.05 | 55,736 | 677 |
| k2 (65536) | 13.0 | 9.00 / 8.97 | 12.41 | 7.05 | 55,734 | 678 |
| w8 (384 + wide) | 13.2 | 9.17 / 9.19 | 12.99 | 7.80 | 136,578 | 677 |
| old (534 の経路) | 13.1 | **11.69 / 11.58** | **16.52** | 10.42 | 138,576 | 0 |
| b2 | 13.1 | 10.44 / **26.03** (ディスク暴れ) | 12.83 | 8.13 | 55,675 | 678 |

- **3,826 cold の中央値**: old 11.63 → b **9.38 (-19.3%)** / k2 8.98 (-22.8%) /
  w8 9.18 (-21.1%)。t538b の k1 (9.48/9.28) と同じ帯 = **2 ラウンド再現**。
- **14k cold**: old 16.52 → b 14.34 (-13.2%) / k2 12.41 / w8 12.99。
- `copy_s` 10.42 → 8.05 s (-23%)、`up_MB` 43.5 GB と `rd_MB` 41.9 GB は**不変**
  (読むバイトは 1 バイトも変えていない。変わったのは本数とサイズ)。
- `b2` の 26.03 s は同じバイナリの 2 ラウンド目で、`copy_s` も 8.13 と正常
  (P494/P522/P530 と同じ「lab のディスクが回によって 2 倍動く」性質)。
  3,826 の 1 サンプル/COND は判定に使わない。14k も同じ (1 サンプル/COND)。

### 3.4 分解 (wide と pread の塊はどちらが効いたか)

t538c の same-run 比較 (3,826 cold、2 サンプル中央値):

| COND | pread の塊 | DMA の塊 | 3,826 cold | 14k | `copy_s` |
| --- | ---: | ---: | ---: | ---: | ---: |
| old | 384 KiB | 8 MiB | 11.63 | 16.52 | 10.42 |
| w8 | 384 KiB | 32 MiB | 9.18 | 12.99 | 7.80 |
| b | 1 MiB | 32 MiB | 9.38 | 14.34 | 8.05 |
| k2 | 1 MiB | 64 MiB | 8.98 | 12.41 | 7.05 |

- **主因は DMA の塊 (wide)**。`copy_s` の 10.42 -> 7.80 s は wide だけで出る。
  `pread_calls` は old 138,576 / w8 136,578 で**ほぼ同じ**なのに速い =
  1 回の pread の固定費ではなく**同時本数**が効いている。
- 1 MiB の塊 (`b`) は w8 と 2% 以内 (9.38 対 9.18) で、14k は b の方が
  1 サンプルで悪い。差は判定できない。既定を 1024 にしたのは、1024 が
  t538b と t538c の**両方**で測れているから (384+wide は 1 回だけ)。
- 64 MiB (`k2`) は 3,826 がさらに -4% だが、バウンス 2 枚 = 128 MiB になる。
  14k は 1 サンプルで 12.41 と良いが、3,826 と 14k を同じ 2 ラウンドで
  比べると 32 MiB が無難。env 1 本で選べる (`DENNO_PF_SPLIT_WIDE_KB=65536`)。

### 3.5 decode (悪化していないこと)

- 短い decode (a1/a2 の中央値): t538c で b 13.2 / old 13.1 / k2 13.0 /
  w8 13.2 ms/tok。t538b も b 13.2 / k1 13.2 / k2 13.2。**差は 0.3 ms 以内**。
- prefill after cut (turn2): 0.48 対 0.47 s (t538c)、0.47-0.48 (t538b)。
- 14k の後の decode: old 26.6 / b 23.5 / k2 22.6 ms/tok (t538c、1 サンプル)。
  **wide の方が速い** (バウンスが広いぶん pread の本数が減る)。
- staging は prefill だけの経路なので、decode が動かないのは想定どおり。

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取って
  いない。
- 採否は**同じ run の b / old と比べた**。run を跨いだ比較はしていない
  (ディスクの日差が 1.5-2 倍ある)。
- 3,826 cold / 14k cold は **COND あたり 2 サンプル (r1/r2)**、14k は
  `SCENARIO=depth8k` の仕様でラウンド 1 にしか流れないので **1 サンプル**。
  外れ値 (b2 の 26.03 s、t538b の b の 23.58 s) は `copy_s` が正常なことと
  他 COND の同値から「ディスクの暴れ」と判断し、中央値/隣のラウンドで読んだ。
- `rd_max` は **pread 全体** (page cache ヒットも含む) の同時本数で、
  ディスクだけの本数ではない (mincore を既定 off にしたまま数えるため)。
  ディスクの本数は「1 塊のディスク区間数 x 塊の数」から推定した (§1)。
- md5 の DIFF は UD の**既知のラウンド間の揺れ** (P518/P522/P530/P534 と同じ)。
  安定している `a` / `c2000_cold` / `c8000_cold` は**全 COND SAME**。
  テキストは全 COND 同趣旨 (§5)。数値経路は 1 バイトも変えていない
  (`up_MB` / `rd_MB` 不変、デバイスへ渡るバイト列は同一。切り方と本数だけ)。

## 5. 出力テキスト (健全性)

- TXTD b (3,826): 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）に
  おける **Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録した
  ものです。…」= old と**同じ書き出し・同じ構成** (old: 「…**Prefill（事前計算
  フェーズの性能最適化の歴史と教訓を詳細に記録したものです。…」)。
- TXT (a) / TXTB (短いプロンプト) も全 COND 同趣旨。**errors 0 は全 COND・
  全ラウンド。**

## 6. 見積り (効いた理由 / 残り)

- 8 MiB の塊に 0.93 MB の区間が 8 本、その 1/3 がディスク。wide 32 MiB で
  32 本 -> ディスク ~10 本、64 MiB で ~21 本。実測の `copy_s` は
  10.42 -> 8.05 -> 7.05 s で、**本数にほぼ比例**して縮んだ (§3.4)。
- ただし P522 のプローブが「1 MiB x 8 本 = 3.5 GB/s で頭打ち」と言うので、
  この方向の天井は **17 GB / 3.5 GB/s = 4.9 s** (copy_s)。今は 7-8 s なので
  まだ 1.5-2 倍の余地がある (**推定**)。1 本のサイズを 0.93 MB のまま
  同時本数だけ増やすのが次の一手 (wide をさらに広げる = k2 の方向)。
- 残っている段: `copy_s` 8.0 s のうち 17 GB がディスク (実効 2.1 GB/s)、
  26.5 GB が page cache (memcpy)。wide を 64 MiB にすると 3,826 で -4% だが
  14k の確認が 1 サンプルしか無い。次の担当者は k2 を 2 ラウンドで測るのが最短
  (**推定**: 3,826 で -4%、14k で -4% 程度)。
- `DENNO_PF_PREAD_THREADS=64` は wide と組にしても効かなかった (`k1t` =
  9.28 対 k1 9.28/9.48、`rd_max` 43 まで上がるが壁は動かない) = 32 本で
  足りている (t424 の棄却と整合)。

## 7. 次の担当者へ

1. **`DENNO_PF_SPLIT_WIDE_KB=65536`** を 2 ラウンドで測る (t538c の k2 は
   3,826 8.98/8.97 と最速、14k 12.41 は 1 サンプル)。バウンス 2 枚 = 128 MiB
   pinned になるので VRAM/RAM の余裕は確認すること。
2. **1 本のサイズを 0.93 MB に戻す**方向 (`DENNO_PF_PREAD_KB` を 0.93 MB 以上に
   すると区間 1 本 = 1 pread)。wide 64 MiB と組にすれば 34 本/塊。未測定。
3. `DENNO_PF_PREAD_THREADS` は 32 のままで良い (t538b の k1t)。
4. 既定モデルは `src` を持たないので、このゲートは**構造的に不活性**
   (t538a で `rd_n=0`、`[exp-pf-stream]` に pread 欄が出ない)。IQ 系で
   `src` を持つモデルが出たら別途測ること。
5. `DENNO_PF_PREAD_NC_FIRST` / `DENNO_PF_PREAD_ODIRECT` / `DENNO_PF_PREAD_MERGE`
   は既定 off のまま (530/522 の棄却を再確認していない)。

## 8. テスト

```text
CARGO_TARGET_DIR=/tmp/t538 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1389 passed; 0 failed; 103 ignored            (b03d34a)
CARGO_TARGET_DIR=/tmp/t538 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 150 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t538nc cargo build --workspace --release                     -> ok
CARGO_TARGET_DIR=/tmp/t538nc cargo test  --workspace --release -- --test-threads=1 -> ok
  (cuda feature 無しの側も緑)
```

- **緑は発火の証拠ではない**ので lab の台帳で数えた: `usplit_wide=677-678`
  (t538b k1/k2、t538c b/k2/w8) に対し `old`/`b`(t538b) は **0**。
  `rd_max` 29 (old) -> 32 (wide)、`pread_calls` 138,644 -> 55,808 (1 MiB の塊)。
- `.cu` / `.ptx` は触っていない (ホスト側の写しの切り方だけ)。

## 9. GPU で確認してほしいこと

1. **既定モデルで既定のまま 1 ラウンド** (t538a は wide が無い前の commit
   `96332c1` なので、`b03d34a` 以降の既定で取り直すのが本当は望ましい)。
   予想: md5 全 SAME、decode 10.4、`usplit_wide=0`、`rd_n=0`。
2. `DENNO_PF_SPLIT_WIDE_KB=65536` の 2 ラウンド (14k 込み)。
3. `DENNO_PF_PREAD_KB` を 0.93 MB 以上にしたときの A/B (wide 64 MiB と組)。
4. 切り戻し: `DENNO_PF_SPLIT_WIDE_KB=0` で t538c の old (11.63 / 16.52 s) に
   戻ること。
