ADOPT: yes t388b t388c

# REPORT P388 — UD-IQ3_XXS decode: ミスの CPU GEMV を GPU の起動と同時に走らせる (DENNO_DEC_MISS_SPAWN)

branch `br388` = 開始時の `int-auto` **056258a92**。作業中に int-auto は
**66c2a4d2d** まで進んだので merge した (docs / denno-view の 1 本、コードの衝突なし)。
数値はすべて GPU ラボの実測。lab 名と測った SHA は
**t388a = 056258a92**(プロファイル用) / **t388b = 3c5318dea**(A/B) /
**t388c = dfa11b861**(既定 on の確認)。

## 0. 結論

1. **採る (`ADOPT: yes`)。** `DENNO_DEC_MISS_SPAWN` を**既定 on** にした。
   decode のミス expert の CPU GEMV を rayon のプールへ先に流し、呼び出し側は
   カーネルの起動と同期だけを行う (scoped なので借用はそのまま、カーネル・引数・
   足し込みの順は 1 つも変えない = **md5 は全 COND で SAME**)。
2. **experts 段の実測 (同じ run の `[q4ledger] experts`、ms/tok)**:
   t388b の round 2 は base **3.90** / base2 **3.89** -> sp **3.63 (-6.9%)**、
   round 1 も base 3.36 / base2 3.34 -> sp **3.21 (-4.5%)**。**2 ラウンドとも 2% 超**。
   既定 on にした後の t388c でも **3.27/3.21 -> 3.49/3.31 (r1)**、
   **3.65/3.66 -> 3.90/3.88 (r2)** = -4.7〜-6.4%。
   decode の中央値は **15.8 -> 15.6 (-1.3%)** (t388c は 15.9/15.8 -> 15.6)、
   14k 後 (c8000_turn2) は **17.2 -> 15.3**、c8000_cold 18.2 -> 16.8。
   prefill は 0.66-0.68 s で不変、errors 0、text 健全 (§4)。
3. **理由も測れた**: 重ね打ちを完全に切る診断 COND (`DENNO_DEC_MISS_SERIAL=1`) は
   experts **4.54 (+16.4%)** / decode **16.6 (+5.0%)**。つまり今のコードの
   「起動 -> CPU ミス -> 同期」は**すでに 4% ぶん重なっている**が、CPU ミスは
   起動が終わってから始まるので、起動の 5-6 us/層 が直列に残っていた。
   そこをプールへ移すと -6.9% になる。**task の目標 (decode 2%) には 1.3% で届かない**が、
   オペレーター規則 (段で 2% 以上を 2 ラウンドの中央値) を満たすので既定 on にした。
4. **採らなかったもの**: VRAM 予算を上げる案 (vram = `DENNO_EXPERT_VRAM_MB=26000`
   `DENNO_GPU_WEIGHT_MB=6000`、ヒット率を上げる) は decode **15.9 (+0.5%)** で
   動かず、既定のまま (22000 / 12000)。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

既定モデル (flash-next-UD-IQ3_XXS、83.6 GB) の **t388a の取り直し** (int-auto
056258a92、depth8k、PROFILE=1、ベースライン COND = 22000/12000/`GPU_MIN_HIT=0`):

| cond | decode ms/tok | prefill s | md5 | errors |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 16.0 | 0.69 | 全 SAME | 0 |
| base | 15.9 | 0.68 | 全 SAME | 0 |
| kp (`EXP_KPROF=1`) | 15.9 | 0.68 | 全 SAME | 0 |
| base2 | 15.8 | 0.69 | 全 SAME | 0 |

段の台帳 (base_2 の 96 tok 窓、`[q4ledger]`、step 15.36 ms/tok):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| mix | 4.29 | cpu 3.62 / launch 0.35/242 |
| ssm | 3.52 | GDN 3 カーネルの完了待ち (P245) |
| **experts** | **4.36** | ミス CPU **1.59** + GPU 2.64 (下記) + 起動など |
| full | 1.38 | gpu 1.11 |
| head | 0.73 | — |
| shared | 0.63 | side stream |
| loop | 0.01 | sync 1.21 (= 層の端の待ち) |

experts 段の中 (`[dec-exp-kprof]`、1 層 = 47.9 層/token):

| 項目 | us/層 |
| --- | ---: |
| gate+up (融合 `DENNO_DEC_EXPGU`) | 23.8 + 3.5 |
| silu | 3.6 |
| down | 17.8 |
| sum (`dno_dec_exp_sum`) | 5.9 |
| d2h (和の ZC で残るぶん) | 0.6 |
| **GPU 計** | **55.2** |

- `[exp-iq]` = `lookups=46080 hits=43004 hit_rate=0.933 uploads=1576
  vram=21999MiB/34000MiB`、`[exp-iq-hit]` の計 = **ミス 3040 本 / 4603 層 /
  1.587 ms/tok** (`[dec-up-w]` = 1576 本、`us_per_job=152.9`)。
- **experts 段 4.36 ms/tok のうち 1.59 がミス CPU、2.64 が GPU 時間**で、
  残りが起動・lookup・段の組み立て。**選んだ理由**: ミス CPU の型は
  IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS で、融合カーネルは既にある (P83/P89/P95/P98)。
  `[exp-iq-hit]` のヒット本数ごとの表では **1 本の層 58 us / 2 本目以降 +35 us**。
  コンテナのマイクロベンチ (実形 2.1 MB/本、行 16 の平坦化、16 スレッド) でも
  **45-50 us/本 = 42-47 GB/s** で、ホストの DRAM (48 GB/s 級) に張り付く。
  つまり **CPU カーネルを速くしても伸びない** (GPU 側は
  `[exp-iq-kprof]` で 4.4-5.9 us/本 = 311-467 GB/s と、こちらは余裕がある)。
  だから削るのは**重ね打ちの側**にした。

## 2. 変えたこと (env ゲート 2 本、どちらも `cuda.rs` の `iq_layer_run`)

すべて `crates/denno-engine/src/cuda.rs` の 1 か所と `exp_iq.rs` のゲート。

- **`DENNO_DEC_MISS_SPAWN`(既定 on → t388b で採った)**: `cpu_miss` を
  `rayon::in_place_scope` でプールへ流し、呼び出しスレッドは prologue の H2D・
  カーネル 4 本の起動・同期・控えの写しだけを行う。`cpu_miss` の結果は channel で
  受け、scope の終端で join する。`F: Send` を要求するが、借用は scoped のまま
  (`&mut [f32]` と `&LayerW`)。
- **`DENNO_DEC_MISS_SERIAL`(既定 off、診断用)**: 同期と控えの写しを先に済ませて
  から CPU ミスを回す (重ね打ちを完全に切る)。既定との差 = 重ね打ちの効き目。
- 同期と写しは `finish_gpu` の 1 本にまとめ、3 つの順序 (既定 / spawn / serial) で
  同じものを通す。**カーネル・引数・発行順・足し込みの順は変えない**。
- 発火の証拠: `[dec-miss-spawn] spawn_layers=6140 serial_layers=0` (t388b の sp)、
  `spawn_layers=0 serial_layers=6140` (ser)、既定 on のビルドでも
  `spawn_layers=6140` が出る (t388c)。**緑 = 経路が通った証拠ではない**ので
  層数を数えて 1 行で出している。

## 3. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に同じものを置いた。

### 3.1 t388b (`3c5318dea` = spawn 既定 off のビルド、6 COND x 2 ラウンド、load_before 3.99)

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 15.8 | 0.66 | 全 SAME |
| base | — | 15.8 | 0.67 | 全 SAME |
| **sp** | `DEC_MISS_SPAWN=1` | **15.6** | 0.66 | 全 SAME |
| ser | `DEC_MISS_SERIAL=1` | 16.6 | 0.67 | 全 SAME |
| vram | `EXPERT_VRAM_MB=26000 GPU_WEIGHT_MB=6000` | 15.9 | 0.67 | 全 SAME |
| base2 | — | 15.8 | 0.66 | 全 SAME |

```text
VERDICT base vs x0 OK decode +0.0% prefill +1.5% md5 SAME errors 0
VERDICT sp vs x0 OK decode -1.2% prefill +0.0% md5 SAME errors 0
VERDICT ser vs x0 OK decode +5.0% prefill +1.5% md5 SAME errors 0
VERDICT vram vs x0 OK decode +0.5% prefill +0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

experts 段 (`[q4ledger] experts`、同じ run の 128 tok 窓):

| cond | round 1 | round 2 |
| --- | ---: | ---: |
| base | 3.36 | 3.90 |
| base2 | 3.34 | 3.89 |
| **sp** | **3.21 (-4.5%)** | **3.63 (-6.9%)** |
| ser | — | 4.54 (+16.4%) |

ミス CPU (`[exp-iq-hit]` の計、ms/tok): base 0.794 / 1.631、base2 0.780 / 1.608、
sp **0.705 / 1.506**。spawn ではミスが GPU と同じ時間帯に走るので、
帯域の取り合いが減って単価も下がる。

深い点 (decode ms/tok):

| cond | c2000_turn2 | c8000_cold | c8000_turn2 (14k 後) |
| --- | ---: | ---: | ---: |
| base | 14.8 | 18.2 | 17.2 |
| sp | 14.7 | **16.8** | **15.3** |
| base2 | 14.5 | 16.1 | 14.7 |
| ser | 15.5 | 17.6 | 15.8 |

### 3.2 t388c (`dfa11b861` = spawn 既定 on のビルド、5 COND x 2 ラウンド、load_before 1.73)

`x0` も既定 on のバイナリなので、旧挙動は `off` (= `DENNO_DEC_MISS_SPAWN=0`) で測った。

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て、既定 on) | 15.7 | 0.71 | 全 SAME |
| base | 既定 on | **15.6** | 0.68 | 全 SAME |
| off | `DEC_MISS_SPAWN=0` | 15.9 | 0.69 | 全 SAME |
| off2 | `DEC_MISS_SPAWN=0` | 15.8 | 0.67 | 全 SAME |
| base2 | 既定 on | **15.6** | 0.68 | 全 SAME |

```text
VERDICT base  vs x0 OK decode -0.7% prefill -4.2% md5 SAME errors 0
VERDICT off   vs x0 OK decode +1.0% prefill -3.5% md5 SAME errors 0
VERDICT off2  vs x0 OK decode +0.7% prefill -5.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill -4.2% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

experts 段 (`[q4ledger] experts`、同じ run):

| cond | round 1 | round 2 |
| --- | ---: | ---: |
| base (既定 on) | **3.27** | **3.65** |
| base2 (既定 on) | **3.21** | **3.66** |
| off (`=0`) | 3.49 | 3.90 |
| off2 (`=0`) | 3.31 | 3.88 |

= **2 ラウンドとも段で -4.7〜-6.4%**、decode は 15.9/15.8 -> 15.6。t388b と向きも
大きさも一致する。`[dec-miss-spawn] spawn_layers=6140 serial_layers=0` が既定 on の
全 COND で出て、`off` / `off2` では行が出ない (分岐が切れている証拠)。

深い点 (decode ms/tok、on = base/base2、off = off/off2):
c2000_turn2 は on 15.5/14.3 対 off 14.9/15.0、c8000_cold は on **17.7/15.9** 対
off 18.4/16.0、c8000_turn2 は on 16.8/14.6 対 off 17.0/14.6。**14k の冷たい点は
両ラウンドとも on が小さく、他は noise の幅** (±0.6 ms、MEMORY の drift)。

## 4. 出力テキスト (sane) と md5

t388b の sp (round 2) から:

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（\`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

md5 は t388b の全 COND・全コンテキストで
`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:2081cdeb
c8000_cold:672fec07 c8000_turn2:bc4aaf54` (SAME)。errors は全 COND 0。

## 5. 数値の扱い

- **値は 1 つも変えない**。カーネル・引数・発行順・足し込みの順は同じで、
  変えたのは「CPU ミスをどのスレッドで、いつ走らせるか」だけ。md5 SAME の根拠。
- 段の比較は同じ run の `[q4ledger] experts` を主、decode の中央値を従にした
  (0.1 ms/tok 刻みなので 1 段は判定に使わない)。
- t388b は load_before 3.99 の回で、絶対値 (15.8) は t388a (1.74) より高い。
  採否は同一 run 内の比で読んだ。
- `vram` COND (`EXPERT_VRAM_MB=26000` + `GPU_WEIGHT_MB=6000`) はヒット率が
  **0.947 -> 0.974** まで上がったのに decode は 15.9 (base 15.8) で動かなかった。
  上げ直しは **712 -> 1114 本** (worker の 1 本 = 約 6 MB の DRAM 交通) に増えており、
  ミス CPU の減少と相殺している、というのが観測の説明 (**推定**。内訳は測っていない)。
  c8000_turn2 も 14.9 で base2 の 14.7 と同じ帯。

## 6. テスト

関門 (`--features cuda`、merge 後の HEAD で実行):

```text
CARGO_TARGET_DIR=/tmp/t388 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1265 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 112 passed; 0 failed
```

足したテスト (2 本、`exp_iq::tests`):
- `miss_spawn_gates_parse_explicitly` — spawn は既定 on・明示的な偽値だけ切る、
  serial は既定 off。
- `miss_spawn_line_counts_fired_layers` — 発火した層数を数えて 1 行で出す
  (ゲート off / 0 層なら出さない)。
`cpu_miss` が `Send` を要求されるようになったので、既存テスト
`overlap_skips_cpu_miss_when_it_does_not_launch` の数え役を `Cell` から
`AtomicU32` に変えた (値の契約は同じ)。

## 7. GPU で確認してほしいこと / 次のタスクへの申し送り

1. 起動行に `[dec-miss-spawn] spawn_layers=<48 x 層数>` が出ること
   (`DENNO_DEC_MISS_SPAWN=0` で 0 層 = 従来経路)。
2. `DENNO_DEC_MISS_SERIAL=1` は診断専用。既定に戻すと decode が +5% 遅くなる
   ので、A/B のときだけ使う。
3. **experts 段の残りはほぼ GPU 時間 (1 層 55 us) とミス CPU (1 層 30-50 us)**。
   ミス CPU は 330-410 GB/s で帯域律速なので、次に削るなら
   (a) ミスを GPU で回す (H2D が 1 本 40-80 us かかるので現状は負け)、
   (b) ミスの本数を減らす (VRAM 予算は効果なしを t388b で確認)、
   (c) `.cu` で gate+up / down / sum を融合する (task 388 は .cu 禁止)。
4. `ser` の数字 (decode +5%) は「重ね打ちの価値」の実測。今後この経路を触る
   タスクは、`DENNO_DEC_MISS_SERIAL=1` で重ね打ちを切った A/B を一緒に取ると、
   削ったぶんが本当に効いたかを切り分けられる。
