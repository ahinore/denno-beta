ADOPT: yes t637c

# task 637 (dev2) — 既定モデル (flash-next-UD-IQ3_XXS) の prefill: fresh な段
# プロファイルを取ったうえで、残っていた最大の 1 本 = **expert 上げの写し
# (3,826 で 1.07 s、壁の 52%)** のバウンスの大きさを、上げスレッドが 2 -> 3 本に
# 増えた後の L3 (96 MB) に合わせ直した (`DENNO_PF_SPLIT_WIDE_KB` の既定
# 32 MiB -> **16 MiB**)。**3,826 cold が -8.0 / -8.3%** (対 同じ run の b/b2)、
# **14k cold が -2.0 / -5.1%**、`experts` 段は 1,296.4 -> 1,144.4 ms (-11.7%)、
# `[exp-pf-stream] ready_s` は 1.07 -> 0.93。decode 9.8 ms/tok 不変、errors 0、
# md5 は全 COND 全 6 文脈 SAME。**タスク目標 (2.5%) を満たす。**

branch **`br637`** = 開始時の `int-auto` **`69b5dac46`** (merge: br606)。
`br633` (私の前タスク、`ADOPT: no`) は **merge していない**: 中身は既定 off の
診断ゲート (`DENNO_PF_KV_D2H_STREAM`) と REPORT だけで、本タスクの経路
(`exp_pf.rs` の分割 DMA の既定値) と 1 行も重ならず、P633 の結論
(「`full` の `kv_wait` は転送そのもので、専用ストリームでは動かない」) は
前提としてそのまま使えるため。終端で fetch し直したが `int-auto` は
`69b5dac46` のまま (merge する物は無い)。push 無し、docker 無し、GPU 直叩き無し。
`.cu` / `.ptx` / `build_ptx.sh` は 1 バイトも触っていない (PTX 再生成不要)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = タスク指定の env
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND `x0`、末尾に baseline の繰り返し `b2`)。
lab 名と SHA: **t637a** (`69b5dac46`、段プロファイル用、`x0 b b2`、load 1.66)、
**t637b** (`69b5dac46`、`x0 b wc b2`、load 1.87)、
**t637c** (`69b5dac46`、`x0 b sw8 sw16 b2`、load 4.09)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes`)。** `DENNO_PF_SPLIT_WIDE_KB` の既定を 32 -> 16 MiB。
   3,826 cold は **2.09/2.085 -> 1.895 s (median、-9.1%)**、14k cold は
   4.87/5.03 -> 4.77 s (対 `b2` -2.0%、対 `b` -5.1%)。
2. **機序は台帳で閉じている**: 変えたのは expert 上げの写しだけで、
   `experts` 段 1,296.4 -> 1,144.4 ms (-11.7%、exposed 735.4 -> 578.7)、
   `ready_s` 1.07 -> 0.93、`up_MB` 22,427.5 は不変、他の段 (mix / ssm / full /
   other) は 1% 以内で不変、`gpu_busy` も 1,180 -> 1,180 (不変)。
3. **理由は「上げスレッドが 2 -> 3 本に増えたのに、塊は 538 のときの 32 MiB の
   まま」だったこと。** 32 MiB x 3 = 96 MiB = 9800X3D の L3 そのもので、
   task 481 の利得 (DMA が直前に書いた行を L3 から読む) が消えていた。
   16 MiB x 3 = 48 MiB で戻る。
4. **測って棄却した**: `DENNO_PF_STREAM_WC=0` (t637b)。**3,826 が 2.08 -> 2.42 s
   (+16%)、14k が 5.03 -> 5.82 s。** 既定モデルの IQ 系で WC を外すと
   `copy_s` 1.05 -> 3.16 s、`probe_GBs` 29 -> 22.5 で、写しも DMA も遅くなる。
   task 309 が K 系で WC を外して勝った理由 (WC への写しが 4.3 GB/s で頭打ち) は
   既定モデルには当てはまらない。**既定は WC のまま。**
5. fresh な段プロファイル (t637a `b2`、3,826 cold、壁 2,071.6 / `gpu_busy` 1,179.7):
   experts 1,305.6 (busy 560.4、exposed **745.2**) / ssm 349.0 (360.9) /
   mix 289.1 (127.6) / other 91.1 (51.0) / full 36.8 (79.8)。
   **14k (壁 4,819.8 / `gpu_busy` 4,055.6)**: ssm 1,969.9 (1,315.5) /
   experts 1,818.1 (1,543.5) / full 678.5 (544.4) / other 309.9 (185.5) /
   mix 43.4 (466.7)。**壁の 84% は GPU の実時間で、残るホスト露出はほぼ全部が
   expert 上げ待ち** (§2)。16 MiB はその露出を 3,826 で 157 ms、14k で 68 ms 削った。
6. decode 9.8-9.9 ms/tok 不変、errors 0、md5 は**全 COND 全 6 文脈 SAME**、
   本文は健全 (§6)。

## 1. なぜこれを選んだか (largest piece nobody has measured)

t637a の fresh な台帳で段を数え直し、**既に誰が何を試したか**を並べた
(このモデルについて task 500 以降の REPORT を全部読んだ: P550 / 553 / 554 / 558 /
561 / 562 / 565 / 566 / 570 / 572 / 576 / 581 / 592 / 594 / 602 / 606 / 609 /
617 / 633 / 640 / 664 / 668)。

| 段 / 内訳 | 3,826 (wall / busy) | 14k (wall / busy) | それまでに試されたこと |
| --- | --- | --- | --- |
| experts (上げ + IQ TC) | 1,305.6 / 560.4 | 1,818.1 / 1,543.5 | P337 (予測・先読み) 棄却 / P497 (枚数・先読み深さ) flat / P550 (読みの形) 棄却 / P554 (上げ 2 本) / P570 (3 本 + pread 先読み) / P606 (pread 16 本) / P629 (バウンス枚数・decouple・split) flat / P227 (mmap の pin = ゼロコピー) は driver が file backed を拒否 |
| ↑ のうち **写し (バウンスの大きさ)** | **1.07 s (`ready_s`)** | **1.39 s** | **task 481 (8 MiB 分割) / 538 (広い塊 32 MiB) の後、上げが 2 -> 3 本になった (570) 以降は未測定 → これを選んだ** |
| ssm | 349.0 / 360.9 | 1,969.9 / 1,315.5 | P379 (最終チャンクだけ書き戻す) / P397 (非同期書き戻し) / P592 (decode の scan 展開)。14k の busy 1,315 は `.cu` |
| full (attention + KV 戻し) | 36.8 / 79.8 | 678.5 / 544.4 | P553 (KV defer / push par) / P561 (KV reserve) / **P633 (KV D2H を専用ストリーム、kv_wait 94 ms は転送そのもの)** |
| mix (inject / lm) | 289.1 / 127.6 | 43.4 / 466.7 | P431 / P448 / P457 / **P617 (inject を BF16 カーネルへ、gpu_busy -4.3%)** |
| other (res / teardown / iter) | 91.1 / 51.0 | 309.9 / 185.5 | P565 (res pin / restore nosync) / P581 (res の並列写し) / P602・P606 (ra / pread の本数) |

選んだ理由: (a) 3,826 の壁 2,071 のうち **1.07 s がこの写し**(52%) で、
GPU の実時間 1,180 ms とほぼ同じ大きさ、(b) **壁の形が「写し + 計算が直列」**
なので、ここが縮めば壁が直接縮む (P629 が「3,826 は expert stream の写しが壁」
と確定済み)、(c) **task 481 の L3 前提が、その後に入った変更 (上げ 3 本) で
崩れている**という筋の通った差分がある、(d) env だけで測れる (コード変更ゼロで
1 ラウンドに 2 つの値を入れられる)。

## 2. fresh な段プロファイル (t637a `b2`、PROFILE=1)

### 2.1 3,826 cold (`tok=3826`、壁 2,071.6 ms、`gpu_busy` 1,179.7)

| 段 | wall ms | busy ms | exposed ms |
| --- | ---: | ---: | ---: |
| experts | **1,305.6** | 560.4 | **745.2** |
| ssm | 349.0 | 360.9 | 0.0 |
| mix | 289.1 | 127.6 | 161.5 |
| other | 91.1 | 51.0 | 40.1 |
| full | 36.8 | 79.8 | 0.0 |
| 合計 | 2,071.6 (壁) | 1,179.7 | 946.8 |

`[pf-host]` 相当は `[exp-pf-stream]`: `up_MB=22,427.5 copy_s=0.54 ready_s=1.07
probe_GBs=29.1 wc=1 usplit=1013 usplit_wide=673 usplit_wide_kb=32768`。
**`ready_s` 1.07 s が壁の 52%** で、これが「上げスレッドが計算スレッドを
待たせた合計」。P633 の t633b (壁 2,133.9 / busy 1,181) と 3% 以内で一致。

### 2.2 14k cold (`tok=14018`、壁 4,819.8 ms、`gpu_busy` 4,055.6)

| 段 | wall ms | busy ms | exposed ms | 主な中身 |
| --- | ---: | ---: | ---: | --- |
| ssm | 1,969.9 | 1,315.5 | 654.3 | scan (`wy5` + `scan_sig3`) + 上げ待ち |
| experts | 1,818.1 | 1,543.5 | 274.6 | 上げ `ready_s=1.38` + IQ TC GEMM |
| full | 678.5 | 544.4 | 134.1 | `kv_wait=95.6` / `kv_push=38.4` |
| other | 309.9 | 185.5 | 124.4 | res_up 83.7 / pf_teardown 78.8 / iter_top 129.2 |
| mix | 43.4 | 466.7 | 0.0 | 別ストリーム (隠れている) |

`close wall=4,819.8 gpu_busy=4,055.6 exposed_host=1,187.5`。**壁の 84% は GPU が
実行中**。GPU が遊ぶ 764 ms は上げ待ちで、ssm 段の窓に 654 ms が乗っている
(ssm 自身の cpu は 29 ms しかない = 上げのイベント待ちで GPU が空く)。
つまり 14k でも **ホスト側で動かせる最大の 1 本は上げの写し**。

## 3. 変えたこと (既存ゲートの既定値 1 つ、切り戻し可)

1. `DENNO_PF_SPLIT_WIDE_KB` の既定を **32 MiB -> 16 MiB**
   (`DEFAULT_SPLIT_WIDE_KB`、`exp_pf.rs`)。`0` で 8 MiB 固定、`=32768` で
   task 538 の値へ戻せる。**バイト列・順序・値を 1 ビットも変えない**
   (分割の大きさだけ。`pf_stream_tests` の分割一致契約はそのまま)。
2. doc を書き換え (538 の測定は上げ 2 本のときのもの、という但し書きと
   t637c の表)、単体テスト `task538_split_wide_gate_truth_table` の期待値を更新、
   `kernel_invariants` に `t637_wide_split_fits_three_up_workers` を追加
   (上げ 3 本 x 16 MiB = 48 MiB が L3 の 2/3 以下、32 MiB への切り戻しが可能、
   を字面で固定)。
3. 触っていない: `.cu` / `.ptx` / `build_ptx.sh` / 他のゲート / 既定の他値。

## 4. lab の結果

### 4.1 t637b — 測って棄却: `DENNO_PF_STREAM_WC=0`

| COND | 3,826 cold r1/r2 | 14k cold r1 | `copy_s` | `probe_GBs` |
| --- | --- | --- | ---: | ---: |
| b | 2.08 / 2.10 | 5.03 | 1.05 | 32.9 |
| wc (WC=0) | **2.42 / -** | **5.82** | **3.16** | 22.5 |
| b2 | 2.07 / 2.10 | 4.87 | 0.70 | 29.1 |

WC を外すと写しが 3 倍遅く、DMA も遅い。**既定モデルでは WC が正しい**
(3,826 で +16%、14k で +16%)。棄却。

### 4.2 t637c — 本命 (`x0 b sw8 sw16 b2`、2 ラウンド)

| COND | 3,826 cold r1 / r2 (median) | 14k cold r1 |
| --- | --- | ---: |
| x0 (捨て) | 2.15 / 2.07 | 5.79 |
| b (32 MiB) | 2.08 / 2.10 (**2.09**) | 5.03 |
| sw8 (8 MiB 固定) | 1.99 / 1.85 (1.92) | 4.71 |
| **sw16 (16 MiB)** | **1.90 / 1.89 (1.895)** | **4.77** |
| b2 (32 MiB) | 2.07 / 2.10 (**2.085**) | 4.87 |

3,826: sw16 1.895 対 b 2.09 / b2 2.085 = **-9.3% / -9.1%**。
14k: sw16 4.77 対 b 5.03 / b2 4.87 = **-5.2% / -2.0%**。
sw8 も両方勝つ (3,826 -8.1%、14k -6.4% 対 b) が、3,826 の 2 ラウンドが
1.99 / 1.85 と振れ、16 MiB の方が安定して速い。**16 MiB を採る**。

### 4.3 台帳 (同じ run、3,826 / 14k)

| 指標 | b (32 MiB) | sw16 (16 MiB) | b2 (32 MiB) |
| --- | ---: | ---: | ---: |
| 3,826 壁 | 2,070.9 | **1,899.0** | 2,065.0 |
| 3,826 `experts` 段 | 1,296.4 | **1,144.4** | 1,297.3 |
| 3,826 `experts` exposed | 735.4 | **578.7** | 736.1 |
| 3,826 `ready_s` | 1.07 | **0.93** | 1.07 |
| 3,826 `gpu_busy` | 1,180 | 1,180 | 1,180 |
| 14k 壁 | 5,024.5 | **4,769.7** | 4,865.0 |
| 14k `experts` 段 | 1,838.2 | **1,771.3** | 1,822.3 |
| 14k `ready_s` | 1.41 | **1.35** | 1.39 |

`up_MB` は 3 者とも 22,427.5 (3,826) / 27,809.4 (14k) で**同値**、
`gpu_busy` も同値。**動いたのは写しの待ちだけ**で、GPU の仕事は 1 ms も
変わっていない = 数値が変わる余地が無いことの裏付けでもある (md5 も SAME)。

## 5. なぜ効いたか (機序)

- task 481 は「8 MiB のバウンス 2 枚を交互に使い、写し(i+1) を DMA(i) と重ねる。
  DMA が読むのは直前に書いた行なので、まだ L3 (96 MB) にあれば DRAM を
  読み直さない」という形で、`copy_s` を 1.09 -> 0.80 s にした。
- task 538 はディスク読みの同時本数を戻すために**広い塊 32 MiB** を足し、
  そのとき (上げ 2 本) は 2 x 32 = 64 MiB で L3 に収まっていた。
- task 570 が上げを **3 本**にした。3 x 32 = 96 MiB = L3 ちょうどで、
  3 枚が同時に生きると L3 から溢れて DMA が DRAM を読み直す (3 往復に戻る)。
- 16 MiB なら 3 x 16 = 48 MiB で余裕が残る。実測でも `ready_s` が
  3,826 で -13%、14k で -4% になり、壁が同じだけ縮んだ。
- **推定**: 残る `ready_s` 0.93 s (3,826) は 22.4 GB の読み + 書きで、
  DRAM 62 GB/s なら 0.72 s。まだ 0.2 s の余地があるが、これは
  ゼロコピー (P227 で driver が file backed の pin を拒否) を除くと詰めにくい。

## 6. 本文 (健全性)

t637c `sw16` の `c8000_cold` (md5 `672fec07` = 全 COND 全 6 文脈 SAME):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
> **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録したものです。

`!!!!` / 空 / プロンプトの反復なし。RESULT の text 欄は 5 COND x 2 ラウンドとも
`looks ok`、errors 0。

## 7. 数値の扱い / 前提

- §2・§4 は GPU ラボの実測。§5 の DRAM の床 (22.4 GB x 3 往復 / 62 GB/s) は
  実測からの割り算で P629 と同じ式 (**推定**)。
- 3,826 は 2 ラウンド・5 COND で読めた (ばらつきは 1-2%)。14k は
  1 COND 1 サンプル (ラウンド 1 だけ) で、run の中の位置で 2-4% 動く
  (t637c の x0 5.79 対 b2 4.87 がその証拠)。採否は **同じ run の b と b2 の
  両方**に勝っていることで読んだ。
- `vs x0` の VERDICT 行は使わない。x0 は run の先頭で page cache が冷たい
  (14k 5.79 対 b2 4.87 = -15.9% は効果ではなく位置)。
- task 文の「3,826 cold 2.26 s / 14k 5.0 s」は今も妥当 (今回 2.07-2.09 /
  4.87-5.03)。
- UD-Q4_K_XL は本タスクで測っていない。広い塊の既定は両モデル共通なので
  `DENNO_PF_SPLIT_WIDE_KB` の既定変更は UD にも効く (**推定**: UD は 1 MiB の
  pread を 16 本並べる形になるので、32 -> 16 MiB でも同時本数は足りる。
  実測はしていない。UD で 32 MiB に戻すなら `DENNO_PF_SPLIT_WIDE_KB=32768`)。

## 8. 次にやるなら (推定を含む)

1. **残る `ready_s` 0.93 s を詰める**: 3,826 で壁の 45%。DRAM の床 0.72 s との
   差 0.2 s は、分割 DMA の塊をもう一段下げる (8 MiB x 3 = 24 MiB) か、
   上げスレッドの担当をグループではなく**バイトで均等に割る**形にすると
   取れる可能性がある。sw8 が 14k で sw16 より速かった (4.71 対 4.77) のは
   その方向の証拠。
2. **14k の `ssm` 段に乗っている上げ待ち 647-655 ms** は、層の依存
   (routing -> 上げ -> MoE) が直列なため。ここを崩すには routing 前に層の
   全 expert を上げる (予測なしの全数先読み) しかないが、P337 が近い形を
   棄却している。VRAM 予算 (22 GB) が常駐 20.2 GB で埋まっているので、
   同時に 1 層ぶん (467 MB) 余計に持つには予算の調整が要る (**推定**)。
3. `.cu` を触れるタスクなら 14k の GPU 実時間 (ssm 1,315 + experts 1,543) が
   本命。mix は P617 で 467 ms まで下がった。

## 9. テストと merge

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> **1,432 passed / 0 failed / 103 ignored**
  (`task538_split_wide_gate_truth_table` の期待値を 16 MiB に更新)。
- `... --test kernel_invariants -- --test-threads=1` -> **156 passed / 0 failed**
  (155 + 新規 `t637_wide_split_fits_three_up_workers`)。
- cuda feature 無し: `cargo build --workspace --release` rc=0、
  `cargo test --workspace --release -- --test-threads=1` 全 suite 0 failed。
- 終端で `int-auto` を fetch し直したが `69b5dac46` のままで merge 不要。
- commit: 本 REPORT と既定値・テストを 1 commit。

## 10. GPU で確認してほしいこと

1. 既定のまま (`DENNO_PF_SPLIT_WIDE_KB` 未設定) で `[exp-pf-stream]` が
   `usplit_wide_kb=16384` / `ready_s≈0.93` (3,826)・`≈1.35` (14k)、
   3,826 cold が **1.89-1.92 s**、14k が **4.7-4.8 s** になること (§4.2-4.3)。
2. `DENNO_PF_SPLIT_WIDE_KB=32768` で 2.07-2.09 s / 4.87-5.03 s に戻ること
   (切り戻しの確認)。
3. UD-Q4_K_XL の 14k cold が既定変更で悪化していないこと (未測定。§7)。

## 11. VERDICT 行 (lab の出力そのまま)

t637b (`69b5dac46`、`x0 b wc b2`):

```text
VERDICT b vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT wc vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t637c (`69b5dac46`、`x0 b sw8 sw16 b2`):

```text
VERDICT b vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT sw8 vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT sw16 vs x0 OK decode +0.0% prefill -2.6% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH b c2000_cold vs x0 decode +0.0% prefill -0.9%
VERDICT_DEPTH b c2000_turn2 vs x0 decode +0.0% prefill +1.1%
VERDICT_DEPTH b c8000_cold vs x0 decode +0.0% prefill -13.1%
VERDICT_DEPTH b c8000_turn2 vs x0 decode -0.7% prefill +0.0%
VERDICT_DEPTH sw8 c2000_cold vs x0 decode +0.0% prefill -9.0%
VERDICT_DEPTH sw8 c2000_turn2 vs x0 decode +0.0% prefill +1.1%
VERDICT_DEPTH sw8 c8000_cold vs x0 decode -3.1% prefill -18.7%
VERDICT_DEPTH sw8 c8000_turn2 vs x0 decode -1.4% prefill -1.9%
VERDICT_DEPTH sw16 c2000_cold vs x0 decode -0.4% prefill -10.2%
VERDICT_DEPTH sw16 c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH sw16 c8000_cold vs x0 decode -3.9% prefill -17.6%
VERDICT_DEPTH sw16 c8000_turn2 vs x0 decode -1.4% prefill -1.9%
VERDICT_DEPTH b2 c2000_cold vs x0 decode +0.0% prefill -1.2%
VERDICT_DEPTH b2 c2000_turn2 vs x0 decode -0.3% prefill +0.0%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -3.9% prefill -15.9%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode -2.1% prefill -1.9%
VERDICT baseline x0 errors 0 text ok
```

(`vs x0` は捨て COND との比較。x0 の 14k 5.79 s は run の先頭で page cache が
冷たいためで、`-13.1%` / `-17.6%` は効果ではなく位置の差。採否は §4.2 の
同じ run の b / b2 と medians で読んだ。)
