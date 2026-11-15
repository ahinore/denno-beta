ADOPT: yes t342a t342b t342c

# REPORT P342 — IQ3_S decode: hc_mixed の fp64 の床 (ブロック幅 256 -> 32) と、効かなかった 2 本

branch: **br342** = int-auto `ef21be403` (merge: br334) から。**br338 は int-auto に
入っていない** (HEAD は `e0fda8842` で int-auto の子孫ではない) が、br338 の中身は
doc コメントの修正と REPORT_P338.md だけで、コードの既定は 1 つも変えていない。
このタスクは既定 off のゲートの話ではないので**merge していない** (必要が無い)。
3 ラウンドの間 int-auto は `ef21be403` のまま (最終確認でも同じ = merge 不要)。

## 0. 結論 (3 行)

1. **採ったのは 1 本だけ**: `DENNO_DEC_MIX_HCBLK` = **32** (既定を 32 にした)。
   `dno_hc_mixed_f32` は 1 出力 1 スレッドで出力ごとに `dno_expf` を hc=4 回呼び、
   `dno_expf` は glibc expf のビット一致実装 = **fp64**。RTX 5090 の fp64 は
   fp32 の 1/64 (SM あたり 2 レーン、warp 命令 1 本 ≒ 16 サイクル) なので、
   既定の grid = ceil(e/256) = **10 ブロック (e=2560) = 10 SM** では 1 SM あたり
   288 fp64 命令 = 約 4,600 サイクルが床になっていた。32 スレッドにすると
   80 ブロック = 80 SM に散る。**値はビット一致** (md5 3 ラウンド全 COND SAME)。
   実測: `dno_hc_mixed_f32` **5.8 -> 3.9 us/窓**、mix 段の gpu **2.71-2.75 ->
   2.61-2.64 ms/token (-3〜-4%)**、mix 段の wall -1.4〜-2.8%、**decode 中央値
   -0.5%** (16.7 -> 16.6)。task の目標 (decode 2%) には**届いていない** が、
   「自分の段で 2% 以上を 2 ラウンド以上」の規則 (task 文の Why) を満たす。
2. **効かなかった 2 本**: (a) expert IQ2_S/IQ3_S GEMV の v4 (`kernels_exp_iq_v4.cu`
   = メタデータを nb ブロックぶん先読み) は **v3 と同速** (gate+up 23.2 -> 23.3
   us/層) で、**撤去した**。(b) `DENNO_DEC_TINY32` (expert の silu_mul / exp_sum を
   32 スレッドのブロックで撃つ) も **完全に無効** (silu 4.9 / sum 5.3 us/層のまま)
   で、**撤去した**。`DENNO_DEC_MIX_DTHR=512` (down を 512 スレッドで撃つ) も 0%
   (mix gpu 2.73/2.71 = base と同じ、decode -0.2%) だったので既定 256 のまま。
3. **プロファイルの読み方**: decode の mix 窓 (1 層 1 回) は wall 44 us = launch 5 +
   cpu 6.5 + sync 28 で、**ホスト側 11.5 us (26%) が GPU と直列**。GPU カーネル
   そのものは 27-28 us で、うち down 10.4 / up 7.5 / hc_mixed 3.9 / router 6.9。
   つまり mix はもう「カーネルの中」ではなく **97 回の往復 (H2D 41 KB + 3.5 起動 +
   D2H 10 KB + 同期) の固定費**が主で、ここを削るには残差をデバイスに常駐させる
   しかない (P338 §9 の結論と同じ。今回それを数字で裏づけた)。

## 1. base の測り直し (Do 1)

`SCENARIO=depth8k PROFILE=1`、baseline COND = 既定 + `DENNO_EXPERT_IQ_GPU=1
DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
t342a/b の toss (= baseline + `DENNO_GDN_DEC_KPROF=1` + `DENNO_ATTN_KPROF=1`) と
base の実測。decode は **16.6-16.8 ms/tok** (P338 の 16.6 と同じ帯)。

`[q4ledger]` の a2 (28 -> 128 tok、ms/token、t342b base_1 / base_2):

| item | wall | 内訳 (r1 / r2) |
| --- | --- | --- |
| mix | **4.29 / 4.29** | gpu 2.71 / 2.72・sync 2.75 / 2.76・launch 0.50・cpu 0.63 |
| ssm | 3.62 / 3.61 | launch 0.22/161、残りは GDN 3 カーネルの完了待ち |
| experts | 4.29-5.48 | GPU 約 2.6 + **CPU ミス 2.87** (hit_rate 0.910、5338 本/128 tok) |
| full | 1.55 | gpu 1.02・sync 1.03/18 |
| head | 0.74 | gpu 0.70 |
| shared | 0.69 | launch 0.24/192・cpu 0.31 (GPU は side stream で重なっている) |
| router | 0.19 | cpu のみ |
| sample | 0.11 | — |

`[dec-kprof-avg]` の mix 1 窓 (t342a toss、kprof 込みなので gap は過大):

| 列 | 値 (us/窓) |
| --- | --- |
| wall / kern / copy | 68.0 / 27.0 / 3.2 |
| gap / wait / launch | 22.4 / 25.0 / 10.6 |

op 別 (t342a base、us、n = 呼び出し数): `dno_hc_mixed_f32` grid=10x256 **5.8** /
`dno_mix_down_bf16_inj3` grid=324x256 **10.3** / `dno_mix_up_bf16` grid=1280x256
**7.5** / `dno_matvec_f32_r` (router) grid=512x256 **6.9**。
ssm (1 層): `dno_gemv_q5k_v5` grid=2048x16 **30.7** / `dno_gemv_q8_0_v5`
grid=320x256 **28.5** / `dno_gemv_q4k_v5` grid=768x16 **21.5** /
`dno_gdn_scan_sig2` grid=48x512 **12.6** / `dno_matvec_bf16_r` (ba) 96x256 **5.7**。

**この段で分かったこと**: (a) mix の 3 カーネルはもう帯域の近くに居る
(down 6.55 MB / 10.3 us = 636 GB/s、up = 873 GB/s) ので、ここの改善は 2-3 us/窓
が上限。(b) expert の IQ GEMV は **441 GB/s** で、同じ run の Q5_K v5 は
**940 GB/s**。この差を「ロードの細さ (1 レーン 6 B) による MLP 不足」と読み、
v4 を書いて測ったが**仮説は外れた** (§3.2)。(c) mix の壁は**ホスト往復** (§0-3)。

## 2. 変えたこと

- **`DENNO_DEC_MIX_HCBLK`** (新規、既定 **32**、切り戻しは `=256`):
  chain3 (decode の mix 本線) の `dno_hc_mixed_f32` のブロック幅。
  1 行 (grid = ceil(e/blk)、block = blk) だけの変更で、カーネルは
  `d = blockIdx.x * blockDim.x + threadIdx.x` のまま = **PTX 再生成不要**。
- `DENNO_DEC_TINY32` は**撤去** (効果 0)、`kernels_exp_iq_v4.cu` とその配線
  (`DENNO_EXP_IQ_V4` / `kernels_exp_iq_v4.ptx` / `build_ptx.sh exp_iq_v4`) も
  **撤去** (v3 と同速)。`DENNO_DEC_MIX_DTHR` は既定 256 のまま (0%)。
- コードの既定を変えたのは **HCBLK の 1 つだけ**。

## 3. ラウンド (Do 3)

### 3.1 t342a (`3530a22fd`、6 COND、PTX=none、load_before 2.12)

COND = toss(base + `GDN_DEC_KPROF=1 ATTN_KPROF=1`) / base / hc32 /
tiny(`TINY32=1`) / all(hc32+tiny) / base2。

| cond | decode ms/tok | prefill s | md5 |
| --- | --- | --- | --- |
| toss | 16.7 | 0.59 | SAME |
| base | 16.7 | 0.59 | SAME |
| hc32 | **16.6** | 0.59 | SAME |
| tiny | 16.7 | 0.59 | SAME |
| all | **16.6** | 0.58 | SAME |
| base2 | 16.7 | 0.58 | SAME |

```text
VERDICT base vs toss OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT hc32 vs toss OK decode -0.5% prefill +0.0% md5 SAME errors 0
VERDICT tiny vs toss OK decode -0.2% prefill +0.8% md5 SAME errors 0
VERDICT all vs toss OK decode -0.7% prefill -1.7% md5 SAME errors 0
VERDICT base2 vs toss OK decode +0.0% prefill -1.7% md5 SAME errors 0
```

op 別の証拠: `dno_hc_mixed_f32` が **grid=10x256 gpu 5.8 us -> grid=80x32 gpu 3.9 us**
(−33%)。mix 段の gpu は base 2.74/2.73 -> hc32 2.70/2.68。hc32 の COND は kprof を
入れたので、この run の mix wall (4.66) は kprof の固定費で**使えない** (launch
0.50 -> 0.67、cpu 0.66 -> 0.86 に膨らんでいる)。

### 3.2 t342b (`f4cd09981`、6 COND、**PTX=exp_iq_v4**、load_before 1.91)

COND = toss / base / iq4 / iq4hc(iq4+hc32) / hc32 / base2。

| cond | decode ms/tok | prefill s | md5 |
| --- | --- | --- | --- |
| toss | 16.8 | 0.59 | SAME |
| base | 16.8 | 0.59 | SAME |
| iq4 | 16.7 | 0.60 | SAME |
| iq4hc | 16.6 | 0.59 | SAME |
| hc32 | 16.7 | 0.59 | SAME |
| base2 | 16.6 | 0.61 | SAME |

```text
VERDICT base vs toss OK decode -0.2% prefill -0.8% md5 SAME errors 0
VERDICT iq4 vs toss OK decode -0.7% prefill +1.7% md5 SAME errors 0
VERDICT iq4hc vs toss OK decode -0.9% prefill -0.8% md5 SAME errors 0
VERDICT hc32 vs toss OK decode -0.7% prefill -0.8% md5 SAME errors 0
VERDICT base2 vs toss OK decode -0.9% prefill +2.5% md5 SAME errors 0
```

この run は **base2 が toss に対して -0.9%** で、床 (0.7-0.9%) が大きい
(c8000 の page cache が COND 後半で冷えた: toss/base2 の c8000 decode が
42/38 ms/tok)。したがって decode 中央値では何も分離できない。**分離は stage で読む**:

| COND | mix wall | mix gpu |
| --- | --- | --- |
| base r1 / r2 | 4.29 / 4.29 | 2.71 / 2.72 |
| hc32 r1 / r2 | **4.24 / 4.28** | **2.65 / 2.68** |
| base2 r1 | 4.31 | 2.73 |

`[dec-exp-kprof]` (a2 の窓) で **v4 が効いていない**ことがはっきり出た:
base gate 23.2 us/層 / up 4.3 -> iq4 gate **23.3** / up 4.7。IQ3_S (型 1/1) の
帯域も 306 -> 301 GB/s で、**先読みを深くしても速くならない**。
(仮説が外れた理由の推定: 1 ブロック 110 B の読みはレイテンシではなく
L2 セクタ効率と LDS の表引きで決まっている。あるいは nvcc が v3 の時点で
ロードをまとめていた — task 327 が inject で見たのと同じ結論。)

### 3.3 t342c (`f4cd09981`、6 COND、PTX=none、load_before 5.53)

COND = toss / base / hc32 / dthr(`DTHR=512`) / all(hc32+DTHR512+TINY32) / base2。

| cond | decode ms/tok | prefill s | md5 |
| --- | --- | --- | --- |
| toss | 16.8 | 0.60 | SAME |
| base | **16.6** | 0.59 | SAME |
| hc32 | **16.6** | 0.58 | SAME |
| dthr | 16.8 | 0.59 | SAME |
| all | **16.6** | 0.59 | SAME |
| base2 | 16.7 | 0.59 | SAME |

```text
VERDICT base vs toss OK decode -0.9% prefill -1.7% md5 SAME errors 0
VERDICT hc32 vs toss OK decode -1.4% prefill -3.3% md5 SAME errors 0
VERDICT dthr vs toss OK decode -0.2% prefill -1.7% md5 SAME errors 0
VERDICT all vs toss OK decode -1.2% prefill -3.3% md5 SAME errors 0
VERDICT base2 vs toss OK decode -0.7% prefill -2.5% md5 SAME errors 0
```

stage (この run が採否の主根拠。**base と base2 の対**で読む):

| COND | mix wall | mix gpu | ssm wall | experts wall |
| --- | --- | --- | --- | --- |
| base r1 / r2 | 4.35 / 4.30 | 2.75 / 2.71 | 3.64 / 3.61 | — |
| **hc32 r1 / r2** | **4.23 / 4.24** | **2.64 / 2.63** | 3.61 / 3.61 | — |
| dthr r1 / r2 | 4.34 / 4.29 | 2.73 / 2.71 | — | — |
| all r1 / r2 | **4.22 / 4.19** | **2.64 / 2.61** | 3.62 / 3.61 | — |
| base2 r1 | 4.32 | 2.75 | — | — |

hc32 の mix 段: **gpu -4.0% / -3.0%**、**wall -2.8% / -1.4%** (base 対)。
3 ラウンド合計で hc32 の mix gpu は −1.5〜−4.0%、decode 中央値は −0.5〜−0.6%
(t342a/b/c の base と base2 の床 0.5-0.9% の内側だが、3 ラウンドとも同符号)。

## 4. テキストと状態

- md5 は 3 ラウンド x 全 COND で **SAME** (base と 1 バイトも変わらない)。errors 0。
- 引用 (t342c base、TXTB): 「Rust の「所有権（Ownership）とライフタイム
  （Lifetimes）は、Rust の最も重要な概念であり、同時に初心者が最もつまずきやすい
  部分です。」
  TXT: "A **B-tree** (specifically, the **B+ tree) is the fundamental data
  structure used by most modern relational databases (such as MySQL, PostgreSQL,
  Oracle, and SQL Server)."
  TXTD (c2000): 「ご提示いただいた文書は、LLM推論エンジン（c2000プロジェクト）に
  おける、**Prefill（事前計算）フェーズの性能改善プロセスを詳細に記録したものです。」
  すべて sane。

## 5. 数値の扱い

- 採否は **RESULT.md の decode 中央値だけでは決められなかった** (床 0.5-0.9%)。
  `[q4ledger]` の **段の gpu (イベント計測の合計)** と wall を base / base2 の
  対で読み、`[dec-kprof-avg]` の op 別 (grid / block / gpu / gap) で機構を確認した。
- 3 ラウンドの decode 中央値: base/base2 = 16.7/16.7, 16.8/16.6, 16.6/16.7。
  hc32 = 16.6, 16.7, 16.6。**同一 run 内の対で −0.5〜−0.6%、段では −3〜−4%**。
- 見積り (推定): mix 段の残りは ホスト往復 11.5 us/窓 x 97 = **1.1 ms/token**。
  hc_mixed の fp64 の床は 32 ブロック幅で **0.4 us/窓**まで下がる (実測 3.9 us は
  まだメモリ待ちを含む) ので、ここはもう 1 us/窓 しか残っていない。
- DTHR=512 / TINY32 / v4 は「効果 0」を **2 ラウンドずつ**で確認した (同じ run の
  同じ段が動かない)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1230 passed** / 0 failed / 102 ignored。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **102 passed** / 0 failed。
- 追加したテスト: `dec_mix_hcblk_truth_table` (32..=1024 の 32 の倍数だけ受ける・
  既定 32)、`hcblk_covers_every_output_once` (e=2560 で `d` が全単射 = どの
  ブロック幅でも 1 出力 1 スレッドのまま)。
- v4 の検証 (撤去前に実施): Python ミラーで v3/v4 の索引算術と f32 の丸め順
  (4 アキュムレータ・fold4・warp 木) を 1 対 1 に写し、nb=10/2/12 x 3 trial =
  18 ケースが**ビット一致**することを確認した (`.cu` の構文は g++ スタブでも確認)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) はこのタスクでも同じ (触っていない)。

## 7. GPU で確認してほしいこと

- `DENNO_DEC_MIX_HCBLK=256` との A/B を **UD-Q4_K_XL** でも 1 回だけ
  (`SCENARIO=decode`)。UD は e が同じ 2560 なので同じはずだが、mix の重み
  (k=10240/n=320) が BF16 で同じ形かは実測していない。
- 次に効きそうなのは mix の**ホスト往復 1.1 ms/token** (残差のデバイス常駐) と、
  **ssm の GEMV 3 本 (q4k 21.5 + q5k 30.7 + q8_0 28.5 us/層)**。後者は
  q4k だけ 411 GB/s で、q5k は 940 GB/s。同じ形 (grid=768x16, k=2560) なので
  Q4_K のデコード列 (6 bit スケールの展開) を q5k と同じ形にできれば 2 倍近い。
  私は今回 MLP 仮説 (IQ 側) を外しているので、次は **q4k の内側を q5k と
  1 命令ずつ比べる**ところから始めるのがよい (推定)。
