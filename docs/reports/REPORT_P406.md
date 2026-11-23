ADOPT: yes t406b t406c

# REPORT P406 — 既定モデル decode: expert の上げ直し予算を空きがある間 4 倍にする
# (`DENNO_EXP_UP_FILL` + `DENNO_EXP_UP_FILL_MULT`)

branch `br406` = 開始時の `int-auto` **8aaf48807** (merge: br392 = 既に merge 済み
だったので取り込み不要)。作業中に int-auto は **b23e2ebf8** (merge: br397) へ動いたので
終端で merge した (衝突なし。br397 の `kernels_gdn_pf_wy5.*` / `exp_pf.rs` /
`kernel_invariants.rs` はこのタスクと別の場所)。最終 HEAD の関門は §6。
数値はすべて GPU ラボの実測。lab 名と測った SHA は **t406a = 8aaf48807**
(プロファイル) / **t406b = ca2552de8** (2 倍と 4 倍の A/B) /
**t406c = 7f824a6ca** (既定 on の確認)。3 ラウンドで上限。`RESULT.md` 2 行目は
3 run とも `model=flash-next-UD-IQ3_XXS ptx=none`。
最終 HEAD はこの 3 本に int-auto `b23e2ebf8` (br397 = prefill の scan カーネル
`DENNO_PF_SCAN_WY5` 既定 on) を merge したもので、測った SHA との差は prefill 側
だけ (decode の expert 経路は同じ)。

## 0. 結論 (6 行)

1. **採る。** `DENNO_EXP_UP_FILL` (既定 on) + `DENNO_EXP_UP_FILL_MULT` (既定 **4**)。
   expert の VRAM キャッシュに**空きがある間だけ**、1 トークンあたりの上げ直し予算を
   4 倍にする (48 -> 192 本 = 1 -> 4 本/層)。埋まっているとき (追い出しが起きる) は
   従来の 16 本のまま。
2. **短い decode (同じ run の対、中央値)**: t406b は base **15.4** -> 2 倍 13.7 /
   **4 倍 13.2**、t406c は off 15.6・off2 15.4 -> 2 倍 14.1 / **4 倍 13.2** ms/tok。
   2 ラウンドとも 4 倍が最良で **-14.3% / -14.8%**。task の目標 (2%) を大きく超える。
3. **段 (`[q4ledger] experts`、同じ run)**: base **4.43** -> 2 倍 **3.15** (-29%) /
   4 倍 **2.94** (-34%) ms/token。仕組みはヒット率で、1 つ目の窓で
   **0.566 -> 0.752 (2 倍) / 0.835 (4 倍)**、2 つ目の窓で **0.917 -> 0.984 / 0.998**。
   VRAM は 15,311 MiB から **16,512 / 20,081 MiB** へ早く埋まる (上限 22,000)。
4. **他の段は悪化していない**: prefill は 0.66-0.68 s -> 0.56-0.60 s (4 倍)、
   14k 後 decode は t406b の c8000_turn2 で base 15.7 -> 2 倍 15.6 / 4 倍 14.5。
   **md5 は 3 run の全 COND・全コンテキストで SAME** (`a:edaa84ad b:1938059f
   c2000_cold:b42d79d2 c2000_turn2:2081cdeb c8000_cold:672fec07 c8000_turn2:bc4aaf54`)、
   errors 0、text 健全 (§4)。
5. **効くのは充填の間だけ**。3 つ目の窓では全 COND が hit_rate 0.946-0.948
   (VRAM 21,999 MiB = 満杯) に収束し、段も 3.61-3.66 ms/token で並ぶ。
   つまり**定常に入れば差は無く、悪化もしない**。効くのは起動後 ~200 トークン
   (ページキャッシュ/統計ファイルが温まっていない間) で、そこが lab の測定帯。
6. 予算 1 本/層という上限は **task 144/214 より前 = 上げ直しが同期 H2D だった頃**の
   もの。いまは背景スレッドが送るので `[dec-up-w] queue_max=1` (行列はほぼ空) で、
   メインスレッドの代価は積むだけの約 1 us しかない。そこが古いままだった。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

### 1.1 t406a = int-auto 8aaf48807 の取り直し (depth8k、PROFILE=1、load_before 2.75)

| cond | decode ms/tok | prefill s | md5 | errors |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 15.5 | 0.68 | 全 SAME | 0 |
| base | 15.5 | 0.68 | 全 SAME | 0 |
| kp (計器 6 種) | 15.5 | 0.67 | 全 SAME | 0 |
| base2 | 15.5 | 0.69 | 全 SAME | 0 |

14k: base の c8000_cold 6.46 s / c8000_turn2 14.8 ms/tok、base2 6.26 / 15.4。

段の台帳 (kp COND、`[q4ledger]`、step 15.45 ms/token の窓):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| **experts** | **4.48** | cpu 4.48 のみ (ミス CPU 2.28 + 起動) |
| mix | 4.32 | cpu 3.65 / launch 0.35/242 |
| ssm | 3.41 | GDN 36 層、GPU の完了待ちが本体 (P245/P392) |
| full | 1.41 | sync 0.98 / gpu 0.96 |
| head | 0.73 | gpu 0.70 (787 GB/s) |
| shared | 0.63 | sh_begin 0.43 |
| router | 0.18 | mix の鎖に載っている (task 364) |
| loop | 0.01 | sync 2.23/49 (= 層端で expert の GPU を待つ) |

### 1.2 なぜ mix / ssm / full / head を選ばなかったか (測って分かっている床)

- **mix 4.32**: `DENNO_GDN_DEC_KPROF=1` の平均表で **1 回 79.6 us** =
  kern 30.0 (down v5 **17.2** + upmix 9.6 + hc_mixed) + copy 7.8 + gap 14.3 +
  launch 6.7 + 名前付きホスト区間 19.0 (mix_inj 15.6 / norm 2.7)。97 回/token。
  down は 0.87 MB を 17.2 us = **50 GB/s** で、帯域ではなく**カーネル起動と
  レイテンシ**の床。ホスト仕事は gap の内側に完全に隠れている (t372c)、
  CUDA graph は driver の固定費で負け (t384a / t401)。残る手は .cu の融合だけで、
  このタスクでは触れない。
- **ssm 3.41**: GDN 1 層 **105.7 us** = kern 87.8 (qkv 35.1 + scan 13.2 + ba 5.2 +
  out 35.1) + copy 9.3 (x-mix 後) + launch 10.3。3 カーネルが直列依存で、
  重ね打ちは効かない (t392b)。.cu の仕事。
- **full 1.41**: 1 層 163 us = kern 92.5 (qkv/out GEMV が 31.6 + 31.6 = 1.1 TB/s) +
  copy 13.3 + wait 88.4。GEMV は VRAM の帯域床、待ちは埋める仕事が無い。
- **head 0.73**: v5 カーネル (task 187 で 1.05 -> 0.71) で 787 GB/s。.cu の仕事。

### 1.3 そこで experts (4.48) の中を見た

`[exp-iq]` / `[exp-iq-hit]` / `[exp-iq-kprof]` (同じ t406a の窓):

- ヒット 1 本: **5.2-5.5 us** (GPU、358-417 GB/s)。ミス 1 本: **59 us** (CPU)。
- ミス **38.3 本/token = 2.28 ms/token**、上げ直しは **19.4 本/token** しか積めない。
- ヒット率 **0.917**、expert の VRAM は **15,311 MiB** (expert 予算 22,000 MiB、
  全体表示は 34,000 MiB) = **7 GB 空いている**。
- ミス CPU の型別実効 (1 スレッド) は IQ3_XXS 2.49 GB/s / IQ2_S 1.94 GB/s だが、
  16 スレッド合計では **35.8-53.1 GB/s** で **DRAM の実効上限**に張り付く
  (P361 が IQ3_S で出した結論と同じ)。カーネルを速くしても効かない。

つまり「空きがあるのに埋めていない」ぶんがそのまま CPU の 59 us/本になっている。
上限の出どころを追うと、`quota = 予算 / 層数 = 48 / 48 = 1` 本/層で、
**1 層で 2 本以上ミスした分は永久にミスのまま**だった (h8 の層は 2 本ミスで
1 本しか上げない、h4 の層は 6 本ミスで 1 本だけ)。これが選んだ理由。

## 2. 変えたこと (ゲート 2 本)

### 2.1 `DENNO_EXP_UP_FILL` (既定 on。`0` / `off` / `false` / `no` で切り戻し)

`iq_warm_misses` の予算計算で、**LRU に空きがある間** (`used + 32 MiB < cap`) だけ
`upload_budget(base, mult)` = `base * mult` を使う。埋まっているとき
(`upload_when_full()` = 16) は従来のまま。倍率を当てるのは**背景スレッド経路だけ**
(同期 H2D に落ちる経路 = ワーカー無し/起動失敗はメインスレッドが 1 本 0.16 ms を
払うので従来の 48 のまま)。

### 2.2 `DENNO_EXP_UP_FILL_MULT` (既定 4、1..=8 に丸め)

- `2` = lab t406b の `f1` / t406c の `def` と同じ (96 本/token = 2 本/層)。
- `4` = lab t406b の `f4` / t406c の `f4b` と同じ (192 本/token = 4 本/層)。
  最終的な既定はこれ。**t406c の `f4b` (予算 192) が既定と同じコード経路**。

### 2.3 安全側: 積めなかったら同期へ落とさない

ゲート on のとき、ワーカーが `Already` (同じ expert が飛行中) / `Refused`
(行列が満杯 / ワーカー死亡) を返したら**この層の残りを諦める**。0.16 ms の
同期 H2D をメインスレッドで撃つのは、予算を増やした目的と逆行するため。
写像できない重み (`NoMap` = ヒープ実体) は従来どおり同期経路へ。
`[dec-up-w]` の行に `skip=` を足した (3 run とも **skip=0** = この分岐は
lab では発火していない。単体テストで真理値表を固定)。

## 3. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。

### 3.1 t406b (`ca2552de8`、5 COND x 2 ラウンド、load_before 1.77)

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 15.5 | 0.67 | 全 SAME |
| base | — | 15.4 | 0.66 | 全 SAME |
| **f1** | `FILL=1` (2 倍) | **13.7** | 0.60 | 全 SAME |
| **f4** | `FILL=1` + `EXPERT_IQ_UPLOAD=96` (4 倍) | **13.2** | 0.56 | 全 SAME |
| base2 | — | 15.5 | 0.66 | 全 SAME |

```text
VERDICT base  vs x0 OK decode -0.3% prefill -0.8% md5 SAME errors 0
VERDICT f1    vs x0 OK decode -11.4% prefill -9.8% md5 SAME errors 0
VERDICT f4    vs x0 OK decode -14.6% prefill -15.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH f4 c8000_turn2 vs x0 decode -17.0% prefill -2.0%
VERDICT_DEPTH f1 c8000_turn2 vs x0 decode -10.3% prefill -2.0%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -1.0% prefill -5.6%
VERDICT baseline x0 errors 0 text ok
```

仕組み (同じ run の窓ごと。`[exp-iq]` / `[q4ledger] experts`):

| cond | 1 つ目の窓 hit | 上げ | vram MiB | experts | 2 つ目の窓 hit | 上げ | vram | experts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.566 | 5,384 | 10,496 | 14.97 | 0.917 | 2,486 | 15,311 | 4.43 |
| f1 (2 倍) | 0.752 | 7,584 | 14,752 | 9.41 | 0.984 | 906 | 16,512 | 3.15 |
| **f4 (4 倍)** | **0.835** | 8,394 | 16,324 | **7.30** | **0.998** | 150 | 16,617 | **2.94** |
| base2 | 0.566 | 5,384 | 10,496 | 14.54 | 0.917 | 2,486 | 15,311 | 4.42 |

3 つ目の窓: base 0.947 / f1 0.946 / f4 0.948 (vram 21,999 = 満杯)、
experts 3.66 / 3.61 / 3.61 = **収束して差が消える**。

### 3.2 t406c (`7f824a6ca` = 既定 on のビルド、5 COND x 2、load_before 2.88)

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) `FILL=0` | 17.3 | 0.72 | 全 SAME |
| off | `FILL=0` (従来) | 15.6 | 0.68 | 全 SAME |
| **def** | 既定 (2 倍で測ったビルド) | **14.1** | 0.66 | 全 SAME |
| **f4b** | `FILL=1` + `EXPERT_IQ_UPLOAD=96` (4 倍) | **13.2** | 0.56 | 全 SAME |
| off2 | `FILL=0` | 15.4 | 0.66 | 全 SAME |

```text
VERDICT off  vs x0 OK decode -9.7%  prefill -5.6%  md5 SAME errors 0
VERDICT def  vs x0 OK decode -18.3% prefill -7.7%  md5 SAME errors 0
VERDICT f4b  vs x0 OK decode -23.7% prefill -21.0% md5 SAME errors 0
VERDICT off2 vs x0 OK decode -10.8% prefill -7.7%  md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

この run は `load_before=2.88` で最初の COND (x0) が 17.3 ms/tok と大きく振れて
いるため `vs x0` の行は読めない。読むのは **off 15.6 / off2 15.4 対 def 14.1 /
f4b 13.2** の対で、-9.0% (def) / -14.8% (f4b)。
(この t406c の時点の既定は 2 倍。`def` が 2 倍、`f4b` が 4 倍で、最終的な既定
= 倍率 4 は `f4b` と同じ予算 192 本/token・同じ分岐。§2.2)

### 3.3 2 ラウンドのまとめ (短い decode の中央値)

| 設定 | t406b | t406c | 対 baseline |
| --- | ---: | ---: | ---: |
| 従来 (倍率なし) | 15.4 / 15.5 | 15.6 / 15.4 | — |
| 2 倍 | 13.7 | 14.1 | -11.4% / -9.0% |
| **4 倍 (既定)** | **13.2** | **13.2** | **-14.6% / -14.8%** |

## 4. 出力テキスト (sane) と md5

t406b の f4 から:

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

md5 は t406b / t406c の全 COND・全コンテキストで baseline と **1 バイトも違わない**
(`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:2081cdeb
c8000_cold:672fec07 c8000_turn2:bc4aaf54`)。errors は全 COND 0。

## 5. 数値の扱い

- **値は 1 つも変えない**。変えたのは「どの expert をいつ VRAM へ上げるか」だけで、
  計算に入る重み・カーネル・丸め順は同じ。md5 が全 SAME なのはその証拠。
- 採否は**同じ run の `[q4ledger] experts` と `[exp-iq]` のヒット率**を主、
  decode の中央値を従にした。段は 4.43 -> 3.15 (2 倍) / 2.94 (4 倍) = **-29% / -34%**
  で、2 ラウンドとも同じ向き・同じ大きさ。
- **正直な限界**: この利得は**充填の間**のもの。3 つ目の窓以降 (VRAM 満杯) は
  全 COND が 0.946-0.948 に収束し、段も 3.61-3.66 で並ぶ = 定常では差が無い
  (悪化もしない)。lab の depth8k は 1 COND = 1 サーバ起動で ~500 トークン
  なので、測っている帯がまるごと充填の間に入る。**起動直後の ~200 トークンが
  速くなる**のがこの変更の意味で、長時間走り続けるサーバの定常値は変わらない。
- 4 倍を既定にした理由: 2 ラウンドとも 4 倍 > 2 倍 (13.2 対 13.7 / 14.1) で、
  悪化した段・コンテキストが 1 つも無い。充填の DRAM 追加は一過性
  (1 トークンあたり最大 192 本 x 2.18 MB = 418 MB、~70 トークンで終わる)。
- 見積りは書かない (全部 lab の実測)。

## 6. テスト

関門 (`--features cuda`、最終 HEAD = merge 後で実行):

```text
CARGO_TARGET_DIR=/tmp/t406 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1284 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 117 passed; 0 failed
```

足したテスト (4 本、`model::exp_up_worker::tests`):
- `fill_gate_defaults_on_and_only_false_values_turn_it_off` (既定 on の真理値表)
- `upload_budget_multiplies_and_keeps_one_as_identity` (48 -> 48/96/192、0 倍は恒等、飽和)
- `fill_mult_parses_and_clamps` (既定 4、1..=8、不正値は既定)
- `skip_on_refusal_is_gated_and_keeps_queued_and_nomap` (諦める条件)

## 7. GPU で確認してほしいこと / 次のタスクへの申し送り

1. 既定のビルドで `[exp-iq]` の `vram=` が**早く** 22,000 MiB に近づき、
   1-2 窓目の `hit_rate` が 0.83-0.99 になること。`DENNO_EXP_UP_FILL=0` で
   従来 (0.566 -> 0.917) に戻る。
2. **定常では差が無い**ことを前提に読んでほしい。長く走るサーバで「効かない」のは
   仕様 (充填が終われば予算は `upload_when_full()` = 16 に戻る)。
3. もっと速くしたいなら `DENNO_EXPERT_VRAM_MB` を上げるのが筋だが、これは
   棄却済み (task 337/338 系)。今回の結果は「同じ VRAM でも埋め方で
   ヒット率が 0.917 -> 0.998 まで動く」ことを示したので、**VRAM を増やす案を
   再評価するならこのゲートと一緒に**測る価値がある (充填が速いほど上限も活きる)。
4. decode の残りは `.cu` の仕事: mix の down (0.87 MB を 17.2 us = 50 GB/s、
   起動とレイテンシの床) の融合、GDN の scan+out、expert の IQ カーネル。
   ホスト側は miss CPU が DRAM 上限 (35-53 GB/s) に張り付いていて、もう削れない。
5. `[dec-up-w]` に `skip=` が増えた。既定 on のビルドで skip>0 が増えるようなら
   ワーカーの行列が詰まっている = `DENNO_EXP_UP_FILL_MULT` を 2 に落とす合図。
