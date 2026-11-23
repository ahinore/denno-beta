ADOPT: yes t562a t562b

# REPORT P562 — UD-Q4_K_XL の decode: seg 経路 (expert GEMV) の **gate+up も
# Q4_K の型取り違えだった** — 8 バイトロード版は同じ 8 バイトを 2 つの lane が
# 読んでいた。16 バイトロード版で **gate+up+silu の段が 33.3-34.3 -> 23.8-24.2
# us/層 (-27 / -29%、gbs 241-267 -> 330-342)**、2 ラウンド 4 対 4 で再現。
# decode の壁は -1% 前後 (ノイズ内)、prefill は md5 SAME、既定モデルは 4 COND
# 全部 md5 SAME で `[seg-fast] gu16=0` (不活性)。

branch: **br562** = 開始時の int-auto **f955c3c (merge: br558)** から派生
(br558 は既に int-auto に merge 済みだったので、merge は不要だった)。
終端で int-auto を fetch し直した結果は §7。

数値はすべて GPU ラボの実測。baseline COND = タスク文の 4 本
(`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`) + 全 COND に `DENNO_SEG_PROF=1` (段の内訳を読むため。
A/B の両側に同じだけ付けるので比較は対称)。

| lab | SHA | MODEL / SCENARIO | PTX | COND |
| --- | --- | --- | --- | --- |
| **t562a** | `900b1a0` | UD-Q4_K_XL / depth | `seg_k` | x0 b **p** b2 |
| **t562b** | `e49eec9` | UD-Q4_K_XL / depth8k | none (コミット済) | x0 b **p** b2 |
| t562c | `89efc99` | **UD-IQ3_XXS (既定)** / depth | none | x0 b **old** b2 |

`p` = `DENNO_SEG_GU16=1` (t562a/t562b の時点では既定 off)、`old` =
`DENNO_SEG_GU16=0` (t562c の時点では既定 on なので、旧経路の比較用)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t562a t562b`)。`DENNO_SEG_GU16` を既定 on にした**
   (最後のコミット)。オペレーター規則 (段で 2 ラウンド 2% 以上 + 他が 1% 超
   悪化しない + 本文健全) に当てはまる。**タスクの目標 (decode 3% / prefill 5%)
   は満たしていない** — decode の壁は -1% 前後でノイズ内 (§3.3)。
2. **段の勝ちは本物**: `[seg-prof]` の gate+up+silu (融合 1 起動) が
   t562a **33.1-33.6 -> 24.1-24.4 us/層 (-27%)**、t562b **33.3-34.3 ->
   23.8-24.2 (-29%)**。gbs は 241-267 -> 330-342。4 対 4 の全サンプルで負けなし。
3. **原因は P534 と同じ「型の取り違え」の残り**: P534 は down が Q5_1 なのに
   K 型用の 8 バイト版が掛かっていないことを直した。今回は **gate/up の Q4_K の
   8 バイト版 (`DENNO_SEG_GU8`) が、同じ 8 バイトを lane 0-3 (下位ニブル) と
   lane 4-7 (上位ニブル) の 2 回読む** = 要求 256 B / 有効 128 B だった。
   UD-Q4_K_XL の gate/up は Q4_K (`[pf-ktc] gate/up=Q4K down=Q5_1 k=2560/640
   n=640/2560` を t562a で確認) なので、ここが最後の「2 倍読んでいる」型。
4. **新しいカーネル `dno_seg_gemv_gu16_q4k`**: 1 warp = 4 ブロック、1 ブロック
   = 8 lane。lane が qs の連続 16 バイト (8 バイトロード 2 回) を読んで下位 16
   要素 + 上位 16 要素を足す。要求バイト = 有効バイト。反復は `b0 += 4` で、
   n_in=2560 (10 ブロック) なら 3 反復 (8 バイト版は 10 反復)。
5. **壁が動かない理由も分かった**: 削ったのは GPU 時間だけで、seg 層の壁は
   **ホスト側** (CPU miss GEMV `miss=60-340 us/層`、`sync` 12-29 us/層、
   `alloc` 108-385 us/層) が持っている。段の `sum` が 3.2 -> 4.9 us/層に
   増えたのは、gate が早く終わって待ちがそこへ移ったため (層の壁は不変)。
   GPU を削るだけでは decode は縮まない、が本タスクの一番 useful な結論。
6. **prefill は不変**: UD の `c2000_cold` (3,826) = `73510e1f` が t562a/t562b の
   **全 COND で SAME**、`c8000_cold` (14k) = `47c833e2` が t562b の全 COND で
   SAME (prefill は値がビット一致)。decode の短い `b` / `*_turn2` の DIFF は
   baseline 同士 (b と b2) でも起きている既存の揺れ (P490/P526/P534/P546/P558)。
7. **既定モデルは不変** (t562c): 4 COND (既定 on / `DENNO_SEG_GU16=0` / 末尾) が
   decode 32.2-32.4 ms/tok、**4 文脈すべて md5 SAME**、`[seg-fast] gu16=0`、
   `[seg-prof]` の行が 1 本も出ない (seg K 経路に入らない)。ただしこの run は
   **page cache が冷えていて絶対値は P534 の 10.4 と比べられない** (§5-2)。

## 1. 前提: fresh な段プロファイルと、読んでいる型の棚卸し (Do 1 / Do 2)

### 1.1 現 int-auto の段 (t562a の b、UD-Q4_K_XL、PROFILE=1、decode 定常)

`[q4prof] decode` (128 token、total 2,149.1 ms = 16.8 ms/tok):

| 段 | ms/128tok | ms/token | share |
| --- | ---: | ---: | ---: |
| **experts** | 1,071.7 | **8.37** | **50%** |
| attn (gate) | 352.7 | 2.76 | 16% |
| gdn | 351.2 | 2.74 | 16% |
| full | 158.1 | 1.24 | 7% |
| router | 90.7 | 0.71 | 4% |
| head | 60.4 | 0.47 | 3% |
| shared | 24.5 | 0.19 | 1% |

experts の中身 (`[seg-prof]`、1 層あたり):

| seg 層 | us/層 (b) |
| --- | ---: |
| GPU gate+up+silu (融合 1 起動) | **33.3** |
| GPU down (Q5_1 v4) | 17.2 |
| GPU sum | 3.5 |
| host lookup / table / h2d / launch / d2h | 1.0 / 0.4 / 6.7 / 4.2 / 6.6 |
| host sync (GPU 待ち) | 12.5-29.4 |
| host miss (VRAM ミスの CPU GEMV) | 76.9-340 (**包括区間**) |
| host alloc | 108.9-385 (**包括区間**) |

### 1.2 UD-Q4_K_XL の decode が実際に読む型 (棚卸し)

| 行列 | 型 | decode の経路 | ロード幅 | 状態 |
| --- | --- | --- | --- | --- |
| expert gate / up | **Q4_K** | seg K (GPU, 融合 gu) | 8 B/要素群 ×2 重複 | **本タスクで 16 B 版 (重複なし)** |
| expert down | **Q5_1** | seg K (GPU, `v4`) | 4 B/lane | P534 で修正済 (330 -> 555-615 GB/s) |
| hc mix (up/down) | Q8_0 | mix 専用 (`dn7`/`dn8`) | ブロック単位 | `[q4ledger] sub mix sum=0.24 of 2.73 ms/tok` = 9%。1 要素 1 バイトが最小 |
| shared expert | Q8_0 | shared 経路 | — | 0.19 ms/token |
| token_embd | Q6_K | `gemv_q6k_avx2` (CPU) | AVX2 32 B | 既に広い |
| ssm_alpha / beta | F32 | `gemv_f32_rows_fma_avx2` | AVX2 | 既に広い |
| ple | Q8_0 | gather 128 | — | P419 で採用済 |

「GPU の GEMV で要求バイトが有効バイトの 2 倍」になっていたのは **Q4_K の
gate/up だけ**だった (Q5_K も同じ形だが UD-Q4_K_XL の expert には無い。Q8_0 /
Q5_1 は 1 要素 1 バイトが最小なので 8 バイトロードの重複は起きない。IQ 系は
CPU の AVX2/AVX-512 経路で、1 バイトずつ読む版は無い — `gemv_iq*` の
`gemv_iq_dispatch` / `gemv_iq_ib` / `gemm_iq*_avx2`)。

### 1.3 なぜこれを選んだか

- 段で一番大きいのは **experts (50%)** で、その中の**名前の付いた GPU 部分で
  最大なのが gate+up+silu の 33 us/層**だった。down は P534 が 30-33 -> 16-18 に
  した後で、gate+up は P534 §6-1 が「同じ理屈で 35 -> 30 us が射程 (推定)」と
  書いて**誰も測っていない**唯一の大きい名前付き部分。
- 「誰も測っていない」の確認: `kernels_seg_k.cu` の最終変更は P534 (task 534)
  で、それ以降このファイルを触った commit は無い (git log で確認)。int-auto の
  P538/P546/P550/P553/P554/P558/P561/P565/P581 は全部 prefill。
- `DENNO_SEG_GU8` (P534 が既定 off で残した 8 バイト版) とは別物にした: 8 バイト
  版は**重複読みを残したまま**命令数だけ減らす版で、t534b では dnv4 と組にしても
  差が出なかった。今回の版は重複そのものを消す。

## 2. 変えたこと

1. `kernels_seg_k.cu`: **`dno_seg_gemv_gu16_q4k`** と `sk_acc16_q4k` を追加
   (既存 13 本は 1 バイトも変えない)。1 warp = 4 ブロック (SK_G16_BLKS)、
   1 ブロック = 8 lane (SK_G16_LANES)。lane は `lb = lane & 7` で 16 バイト位置、
   `bi = lane >> 3` で反復内のブロック。下位ニブルは要素 `64j+16h+t`、上位は
   `64j+32+16h+t`、スケールはサブブロック `2j` / `2j+1`。8 バイトロード 2 回
   (16 バイトのアラインメントは仮定しない = 既存の 8 バイト前提のまま)。
2. `cuda.rs`: `SegKKernels.gu16`、`SEG_K_GU16_NAMES` (Q4_K の slot 2 だけ)、
   `DENNO_SEG_GU16` (最後のコミットで**既定 on**、`0` で切り戻し)、
   選択は **gu16 -> gu8 -> gu** の順 (古い PTX では None で従来へ落ちる)。
3. `seg_prof.rs`: `[seg-fast]` に `gu16=N` を追加 (発火の台帳)。
4. 値: 1 lane の和が 32 要素になるので**丸め順が変わる = ビット一致しない**
   (タスクの規則で許容。`DENNO_SEG_GU16=0` で戻せる)。起動形は gu / gu8 と同一
   (blockDim = 128、grid.x = ceil(n_out / (4 * SK_GU_ROWS)))。

## 3. lab の結果

### 3.1 段 (発火の証拠): `[seg-prof]` の gate+up+silu

| lab / round | COND | gate+up+silu us/層 | gbs | down us/層 |
| --- | --- | ---: | ---: | ---: |
| t562a r1 / r2 | b | 33.1 / 33.3 | 241 / 246 | 16.7 / 16.4 |
| t562a r1 / r2 | **p** | **24.4 / 24.1** | **342 / 331** | 17.6 / 16.4 |
| t562a r1 / r2 | b2 | 33.5 / 33.6 | 243 / 251 | 16.7 / 17.4 |
| t562b r1 / r2 | b | 33.3 / 34.3 | 267 / 242 | 17.2 / 16.8 |
| t562b r1 / r2 | **p** | **24.2 / 23.8** | **340 / 342** | 17.6 / 17.6 |
| t562b r1 / r2 | b2 | 33.8 / 34.3 | 228 / 232 | 15.9 / 16.5 |

中央値: baseline (8 サンプル) **33.6** -> p (4 サンプル) **24.15** = **-28%**。
`[seg-fast]` は p で `gu16=5906 .. 30672`、b と b2 で `gu16=0` (両 lab)。
**ゲート on でだけ新カーネルが全層で掛かっている。**

段の副作用: `sum` が baseline 3.2 us/層 (8 サンプル中央値) -> p **4.9**
(4 サンプル) に増えた。層の壁が変わらない (§3.3) ので、gate が早く終わって
待ちが sum の窓へ移っただけと読んでいる (sum 自体のカーネルは不変)。

### 3.2 decode (warm short、`decode 秒 / out token`)

| lab | COND | a1 (2 サンプル) | a2 (2 サンプル) | 4 サンプル中央値 |
| --- | --- | --- | --- | ---: |
| t562a | b | 12.50 / 12.50 | 13.36 / 13.28 | **12.89** |
| t562a | **p** | 12.58 / 12.34 | 13.36 / 13.20 | **12.89 (-0.0%)** |
| t562a | b2 | 12.73 / 12.58 | 13.44 / 13.36 | 13.05 (-1.2% vs p) |
| t562b | b | 12.34 / 12.50 | 13.28 / 13.28 | **12.89** |
| t562b | **p** | 12.34 / 12.27 | 13.12 / 13.20 | **12.73 (-1.2%)** |
| t562b | b2 | 12.66 / 12.58 | 13.75 / 13.36 | 13.01 (-2.2% vs p) |

2 ラウンド合わせて **p 12.73-12.89 対 baseline 12.89-13.05 = -1% 前後**。
RESULT.md の `decode ms/tok (median)` 列も b 13.3 / p 13.3 / b2 13.4 (t562a)、
b 13.3 / p 13.2 / b2 13.6 (t562b) で、**ノイズの外には出ていない**。

### 3.3 長文脈 decode (depth8k、COND あたり 1 サンプル)

| 文脈 | b | **p** | b2 | p vs b2 |
| --- | ---: | ---: | ---: | ---: |
| c8000_cold (14k) | 32.50 | **19.33** | 20.29 | **-4.7%** |
| c8000_turn2 | 20.23 | **15.78** | 17.11 | **-7.8%** |
| c2000_cold (3,826) | 23.16 / 20.11 | 21.79 / 20.00 | 20.42 / 20.32 | -1 to -2% |
| c2000_turn2 | 17.81 / 16.80 | 17.58 / 16.64 | 17.11 / 17.19 | -2.0% |

p は 6/6 の長文脈サンプルで b と b2 の両方以下。ただし baseline 側に外れ値が
ある (b の c8000_cold 32.50 対 x0 28.17 / b2 20.29) ので、**-2% 程度までは
言えるが -8% は言えない**。decode の壁は全体として「ノイズ内で p がわずかに
良い」が正しい読み。

### 3.4 experts 段 (`[q4prof]`、128 token 合計 ms)

| lab | b | p | b2 |
| --- | ---: | ---: | ---: |
| t562a | 2,613.9 (r1、冷) / 1,071.7 | 1,114.9 / 1,061.0 | 1,078.5 / 1,113.3 |
| t562b | 1,424.8 / 1,075.5 | 870.7 / 1,048.1 | 1,029.1 / 1,076.7 |

p の r1 が両 lab とも一番速い (-11% / -39%) が、この段は VRAM のヒット率で
動くので**段の中央値としては読まない** (判定は §3.1 の seg 段で行った)。

### 3.5 prefill (不変の確認)

| lab | 文脈 | b | p | b2 | md5 |
| --- | --- | ---: | ---: | ---: | --- |
| t562a | c2000_cold (3,826) | 17.09 / 6.91 s | 20.81 / 6.93 | 7.06 / 6.87 | `73510e1f` 全部 SAME |
| t562b | c2000_cold | 7.38 / 6.52 | 7.14 / 6.58 | 6.75 / 6.57 | `73510e1f` 全部 SAME |
| t562b | c8000_cold (14k) | 11.63 | 10.18 | 10.28 | `47c833e2` 全部 SAME |

壁はディスクで 2 倍動く (x0 と b の差) ので比べない。**md5 が全 COND SAME =
prefill の値は 1 ビットも変わっていない** (seg 経路は decode 専用)。

### 3.6 既定モデル (t562c、UD-IQ3_XXS、4 COND)

| COND | decode ms/tok | 3,826 cold prefill | md5 (4 文脈) | gu16 |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 32.2 | 29.81 s | edaa84ad / 1938059f / b42d79d2 / 2081cdeb | 0 |
| b (既定 on) | 32.3 | 29.05 | 同上 全部 SAME | 0 |
| old (`DENNO_SEG_GU16=0`) | 32.4 | 28.85 | 同上 全部 SAME | 0 |
| b2 | 32.4 | 29.07 | 同上 全部 SAME | 0 |

`[seg-fast]` は 4 COND とも `gu8=0 dn8=0 dnv4=0 gu16=0`、`[seg-prof]` の行は
1 本も出ない = **既定モデルは seg K 経路に 1 命令も入らない**。
**注意: この run の絶対値は P534 の 10.4 ms/tok / 1.9 s と比べられない** (§5-2)。

## 4. 出力テキスト (健全性)

- UD の p、3,826 cold (TXTD): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と
  教訓を詳細に記録したものです。…」= b と同趣旨・同構造。
- 短い TXTB も同趣旨 (Rust の所有権とライフタイム)。既定モデルの TXT/TXTB も
  「looks ok」で、4 COND が同一 md5。
- **errors 0 は全 lab 全 COND 全ラウンド。** md5: UD の `a` = `737a73af`、
  `c2000_cold` = `73510e1f`、`c8000_cold` = `47c833e2` が全 COND SAME。
  短い `b` (`87d8a97c` / `ffbf4402` / `c82883fe`) と `*_turn2` の DIFF は
  **baseline 同士 (b と b2) でも起きている**既存の decode の揺れで、p に固有
  ではない (t562a の p と b2 が同じ `ffbf4402` を出している)。

## 5. 測定の注意

1. **t562a の b は最初の baseline で、expert の VRAM 昇格が終わっていない**
   (experts 2,613.9 ms、decode の a1 は 12.50 だが c2000_turn2 が 34.22)。
   採否の比較は「同じラウンドの b」と「末尾の b2」の両方で行った。
2. **t562c (既定モデル) の絶対値は使えない**: 直前の UD の 2 run (111 GB を
   2 回読む) が page cache を押し出したので、既定モデル (83.6 GB) がディスク
   から読まれている。証拠は 3,826 cold prefill が **29.05 s** (P534 の同じ
   COND は 1.9 s) で、decode も 32.3 ms/tok (10.4 のはず)。**同一 run 内の
   4 COND 比較と md5 SAME だけが言える。** 次に既定モデルを測るときは、UD の
   run の前に置くこと (そうすればページキャッシュが温かい)。
3. RESULT.md の `decode ms/tok (median)` 列は整数丸めの REQ 行から作られるので、
   本文は `decode 秒 / out token` で計算し直した値を使った (差は 0.0-0.2)。
4. `[seg-prof]` の `miss=` / `alloc=` は GPU 待ちを含む包括区間なので、
   内訳の比較には使っていない (P534 §4 と同じ)。

## 6. 見積り (次はどこか)

- **seg 層の壁はもう GPU ではない。** GPU は gate+up+silu 24 + down 17 + sum 5
  = 46 us/層まで落ちたのに、experts 段の壁は 1,061-1,115 ms/128 tok のまま。
  残りはホスト側で、大きい順に **`miss` (VRAM ミスの CPU GEMV) 76-340 us/層**、
  `alloc`、`sync` 12-29 us/層、`h2d`/`d2h` 13 us/層。
  → 次の担当者は **decode の CPU miss GEMV** を見るべき。`miss` は
  expert の VRAM ヒット率 (`[exp-iq] hit_rate=0.836`) で決まるので、手は
  (a) VRAM 予算 — P435/P452 が棄却済み (decode だけ上げると 23.5 GB の
   refill で +7.6%)、(b) CPU GEMV 自体の速さ — P442 (512 bit) / P498 が
   「もう帯域律速」と記録、(c) `miss` を GPU に回す (t562a の
   `[exp-iq] gpu_layers=0 gpu_experts=0` = 今は 1 本も GPU に載っていない)。
  (c) が未測定で、`DENNO_EXPERT_IQ_GPU=1` は prefill の話なので decode 側の
  expert GPU GEMV は別口のはず。**推定**: miss の 1/2 を GPU へ移せれば
  -1.5 ms/token (推定 -10%)。
- 同じ「2 倍読み」の残りは **Q5_K の gate/up** (同じ 16 バイト版がそのまま
  使える。UD-Q4_K_XL には無いので、その型のモデルが出たら 1 行で足せる)。
- 14k cold prefill は P558 の残り (ssm 1,902.8 ms / other 1,029.5 ms) が依然
  最大。本タスクは触っていない。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t562   cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1412 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t562   cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 153 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t562nc cargo build --workspace --release                       -> ok
CARGO_TARGET_DIR=/tmp/t562nc cargo test  --workspace --release -- --test-threads=1   -> 失敗 0
```

- 足したテスト 6 本: ゲートの真理値表 (既定 on、`0`/`off`/`false`/`no` だけ偽)、
  名は Q4_K の slot 2 だけ、**索引算術の CPU 写し** (`t562_gu16_q4k_lane_mapping_
  covers_every_element_once`: 10 ブロック x 256 要素を**ちょうど 1 回**、バイト
  位置・ニブル・サブブロックのスケールが Q4_K の仕様どおり)、`#define` と
  ホスト側の前提の一致、PTX のエントリ存在、台帳の積算。
- `.cu` は g++ のスタブで `-fsyntax-only` (既存ファイルで先に通して harness の
  妥当性も確認)。**レジスタ数・smem・本当の PTX はこのコンテナでは検証できない。**
- **緑は発火の証拠ではない**ので、lab の台帳で数えた: `gu16=5906 .. 30672`
  (t562a/t562b の p)、b/b2 は 0、既定モデルは全 COND 0 (§3.1 / §3.6)。
- PTX: `PTX=seg_k` で再生成した `kernels_seg_k.ptx` をコミット (+1838 行 /
  削除 0 行)。t562b と t562c は**コミット済み PTX で測定** (RESULT.md 2 行目が
  `ptx=none`)。`.ptx` の手編集はしていない。
- 終端で int-auto を再 fetch した: **f955c3c のまま (動きなし)**。merge
  不要だったので、`build_ptx.sh` の `targets_all` も触っていない。

## 8. GPU で確認してほしいこと

1. **既定モデルの 3,826 cold と decode を温かい page cache で 1 回**。
   t562c はディスク律速で絶対値が使えなかった (§5-2)。`DENNO_SEG_GU16` は
   Q4_K の expert にしか掛からないので、既定モデルで変わるはずはないが、
   `[seg-fast] gu16=0` と md5 SAME を温かい状態でもう一度見てほしい。
2. UD-Q4_K_XL の decode を本番の連続リクエストで。壁は -1% 前後なので、
   **比だけ**を持ち帰ること (lab の絶対値は run 間で 2 倍動く)。
3. 切り戻しは `DENNO_SEG_GU16=0` (P534 の gu 融合版に戻る)。
4. 数値の最終確認をしたいときは `DENNO_EXPERT_GPU_CHECK=1` で層ごとの
   CPU との max_abs / max_rel が出る (本タスクでは lab のラウンドが 3 で
   上限だったため取っていない。索引の写しテストで代替)。
