ADOPT: yes t356b t356c

# REPORT P356 — IQ3_S decode: 和の D2H を消す (pinned のホストへ直接書く)

branch `br356` = 開始時の `int-auto` **c1532cf5b** (merge: br357) から。
**br352 は merge していない** (task 352 は `ADOPT: no` で、br352 の中身は REPORT_P352.md だけ。
コードの変更は 1 つも無いので、このタスクに要るものが無い)。
3 ラウンドの間 int-auto は `c1532cf5b` のまま (最終確認でも同じ = 追加の merge なし)。
数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`, `swift15-abl-IQ3_S`)。
lab 名は t356a / t356b / t356c。

## 0. 結論 (5 行)

1. **採ったのは 1 本**: `DENNO_DEC_EXP_SUM_ZC` (既定 on にした)。expert 段の和
   (`dno_dec_exp_sum` の e = 2,560 要素 = 10 KB) を **VRAM ではなく pinned の
   ホストへ直接書かせる** (UVA)。層ごとの D2H と、その完了待ちが丸ごと消える。
   値は同じ (同じカーネル・同じ式・同じ並び) で、**md5 は t356a/t356b/t356c の
   全 COND (11 本) SAME**。
2. **段の実測 (同じ request 同士、`[q4ledger] experts` ms/tok)**: t356b の 28 tok 窓
   base **3.41/3.48**・base2 **3.46/3.44** -> zcsum **3.23/3.20** = **-6.7%**、
   t356c (既定 on のビルド) の同じ窓は **3.20/3.20/3.21/3.20** vs `=0` の
   **3.47/3.46** = **-7.6%**。深い点 (3826 / 3929 tok) も -3.9% / -3.6%。
   **3 ラウンドとも同じ向き。** `[dec-exp-kprof]` の 1 層では d2h **10.5 -> 1.4 us**、
   代わりに sum **5.3 -> 7.0 us** (ホストへの 10 KB 書き) で正味 **-5.0〜-7.4 us/層**。
3. **decode は 16.8 -> 16.7 ms/tok (t356b)、16.5 -> 16.4 (t356c)**。きれいな窓の
   `step` は t356b 13.80 -> 13.61、t356c 13.78 -> 13.56 = **-1.4〜-1.6%** で、
   段の -7% (段はトークンの約 2 割) と整合する。**task の目標 (decode 2%) には
   届かない** が、オペレーター規則 (段で 2% 以上を 2 ラウンドの中央値) を満たすので
   既定 on にした (task 342 の hc_blk・task 361 の行幅と同じ立場)。
   prefill は 0.58-0.59 s で不変 (むしろ -4〜-5%)、errors 0、text 健全。
4. **捨てた 1 本**: `DENNO_DEC_EXP_D2H_STREAM` (D2H を専用の非ブロッキング
   ストリームへ出す) は **decode 16.7 vs 16.7 = 0%**。`[dec-exp-d2h]` の実測で
   `us/copy=6.9` = **コピーの完了をホストが待っている**ので、ストリームを変えても
   待ちは消えない (だから「コピー自体を消す」= 本命へ切り替えた)。既定 off のまま残す。
5. **P352 の前提はもう成り立っていない (新しい事実)**: 今の int-auto は
   **hit_rate 0.990** (base の a2 窓: lookups 61,440 / hits 60,845 / ミス 595 本 =
   0.097 本/層) で、**ミス CPU は 0.403 ms/tok**。P352 が追っていた 2.78 ms/tok
   (hit 0.910) は消えており、experts 段は **GPU の鎖が主** (1 層 71.7 us のうち
   62.8 us) になっている。ここを削るには D2H のような「鎖の固定費」しか無い。

## 1. Do 1: base の取り直し (t356a、`12a22b4b0`)

decode **16.7 ms/tok** (base)、prefill 0.59 s、errors 0、md5 6 本すべて SAME。
P342/P352/P361 と同じ帯 (16.6-16.8)。b0t だけ 17.6 だが、これは最初の COND の
順序バイアス (page cache) で、以後の比較には使わない。

### 1.1 段の台帳 (base_2、プロンプト 28 tok・出力 128 tok の窓、decode 14 ms/tok)

| item | wall | 内訳 |
| --- | ---: | --- |
| mix | 4.14 | gpu 2.48・launch 0.51/339・sync 2.52/97・cpu 0.69 |
| ssm | 3.63 | launch 0.22/161・cpu 0.14 (待ちは `loop` の sync に乗る) = GDN |
| experts | **3.48** | cpu 3.48 のみ (launch/sync/copy は他行) |
| full | 1.51 | gpu 1.03・sync 1.03/18 |
| head | 0.74 | gpu 0.71 (v5 カーネル、643 GB/s) |
| shared | 0.64 | launch 0.24/192・cpu 0.26 (side stream で GPU と重なる) |
| router | 0.18 | cpu のみ |
| sample | 0.11 | — |
| loop | 0.01 | sync 1.27/47・gpu 1.25 (層の端の待ち) |
| step | 13.88 | — |

experts 段の中 (同じ窓、`[dec-exp-kprof]` はイベント計測、47.8 層/token):

| 項目 | us/層 |
| --- | ---: |
| gate+up (融合 `DENNO_DEC_EXPGU`) | 25.4 |
| up (`mark(UP)` の残り = イベント差) | 3.8 |
| silu | 4.1 |
| down | 16.8 |
| sum (`dno_dec_exp_sum`) | 4.8 |
| d2h (e 要素の D2H) | 7.9 |
| **GPU 計** | **62.8** |

段の wall 3.48 ms = 1 層 72.6 us。**GPU の鎖 62.8 + ホストの起動 ~10** で、CPU ミス
(0.403 ms/tok = 8.4 us/層) はその裏に隠れている。`[exp-iq]` は
`lookups=61440 hits=60845 hit_rate=0.990 uploads=437 vram=20057MiB/34000MiB`
(34000 = `DENNO_EXPERT_VRAM_MB` 22000 + `DENNO_GPU_WEIGHT_MB` 12000)。

## 2. Do 2: 触った 2 か所 (どちらも experts 段、env ゲート付き)

1. **`DENNO_DEC_EXP_D2H_STREAM`** (t356a で測って**捨てた**、既定 off): 和の D2H を
   専用の非ブロッキングストリーム + イベント (ready/done) へ出し、待ちを CPU ミスの
   後へ回す。仮説は「null ストリームの小さい D2H は driver が呼び出しの中で待つ」。
   **結果 0%**。`[dec-exp-d2h] stream=1 copies=4119 fallback=0 wait=28.29ms us/copy=6.9`
   = 分岐は発火したが、ホストが結局 6.9 us/層 待つので層は縮まない。
2. **`DENNO_DEC_EXP_SUM_ZC`** (t356b で採った、既定 on): 和のカーネルの出力
   ポインタへ pinned のホストバッファのアドレスを渡す (UVA なのでデバイスから同じ
   アドレスで書ける)。D2H も、その完了待ちも消える。`[dec-exp-zc]
   sum_zc_layers=4119 fallback=0` = 全層で発火、fallback 0。
   値は不変 (カーネル・引数・並びは同じで、行き先だけがホスト)。

## 3. ラウンド

### 3.1 t356a (`12a22b4b0`、4 COND、load_before 1.73) — base の取り直しと D2H ストリーム

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0t (base + `Q4_TLINE=1`、捨て) | 17.6 | 0.60 | 0 | looks ok | 全 SAME |
| base | 16.7 | 0.59 | 0 | looks ok | 全 SAME |
| d2hs (`DEC_EXP_D2H_STREAM=1`) | 16.7 | 0.60 | 0 | looks ok | 全 SAME |
| base2 | 16.6 | 0.59 | 0 | looks ok | 全 SAME |

```text
VERDICT base vs b0t OK decode -5.3% prefill -0.8% md5 SAME errors 0
VERDICT d2hs vs b0t OK decode -5.3% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs b0t OK decode -5.5% prefill -0.8% md5 SAME errors 0
```

深い点 (14k) も同一: base 6.45 s / 16.2 ms、d2hs 6.49 s / 16.2 ms、
base2 6.45 s / 15.2 ms (turn2)。**d2hs の decode は base と 1 桁まで同じ**。

b0t の `Q4_TLINE=1` は**使えなかった**: `gpu_busy=0.00 gpu_idle=31.20ms` で
イベントが張れず (プール飽和)、しかも decode が 17.6 -> 31 ms/tok に悪化する
(測定を壊す)。以後 TLINE は使わない。

### 3.2 t356b (`3ca27766f`、4 COND、load_before 1.87) — ZC の A/B

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0t (base + `GDN_DEC_KPROF=1`、捨て) | 17.2 | 0.61 | 0 | looks ok | 全 SAME |
| base | 16.8 | 0.59 | 0 | looks ok | 全 SAME |
| zcsum (`DEC_EXP_SUM_ZC=1`) | **16.7** | 0.59 | 0 | looks ok | 全 SAME |
| base2 | 16.8 | 0.59 | 0 | looks ok | 全 SAME |

段の比較 (同じ request 同士。`base` と `base2` は同じ env なので、この 2 本の
ばらつきがノイズの目安):

| 窓 | base | base2 | zcsum | 差 (対 base/base2) |
| --- | ---: | ---: | ---: | ---: |
| 28 tok (128 out) | 3.41 / 3.48 | 3.46 / 3.44 | **3.23 / 3.20** | **-6.7%** |
| 3929 tok | 4.36 / 4.19 | 4.21 / 4.19 | 4.03 / 3.99 | -5.4% |
| 3826 tok | 5.39 / 4.85 | 4.86 / 4.87 | 4.63 / 4.69 | -4.2% |

`[dec-exp-kprof]` の 1 層 (4119 層ぶんの平均): base = d2h 10.0 / sum 5.2、
zcsum = d2h 1.3 / sum 7.2。**d2h の 8.7 us の代わりに sum が 2.0 us 増える**
(ホストへの 10 KB 書き = PCIe の posted write)。正味 -6〜-7 us/層。

`step` (トークン全体、同じ窓): base 13.71/13.88、base2 13.82/13.80 -> zcsum
**13.61/13.60** (-1.4%)。段の -0.25 ms がそのまま出ている。

### 3.3 t356c (`fc23284e2`、3 COND、load_before 1.89) — 既定 on の確認

既定 on のビルド (`b0` / `b1`) と `DENNO_DEC_EXP_SUM_ZC=0` (`zcoff`) の A/B。
**b0 は最初の COND で page cache が冷えており** (c2000_cold 10.92 s / decode 18.7)、
A/B は `zcoff` と `b1` (どちらも温かい) で読む。

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0 (既定 on、捨て) | 16.5 | 0.61 | 0 | looks ok | 全 SAME |
| zcoff (`SUM_ZC=0`) | 16.5 | 0.59 | 0 | looks ok | 全 SAME |
| b1 (既定 on) | **16.4** | 0.58 | 0 | looks ok | 全 SAME |

```text
VERDICT zcoff vs b0 OK decode +0.2% prefill -4.1% md5 SAME errors 0
VERDICT b1 vs b0 OK decode -0.2% prefill -4.9% md5 SAME errors 0
VERDICT_DEPTH zcoff c8000_turn2 vs b0 decode ... (b0 が冷えているので使わない)
```

experts 段 (同じ request 同士。ZC on は 4 サンプル = b0/b1 x 2 ラウンド):

| 窓 | ZC on (b0/b1) | ZC off (zcoff) | 差 |
| --- | --- | --- | ---: |
| 28 tok | 3.20 / 3.20 / 3.21 / 3.20 | 3.47 / 3.46 | **-7.6%** |
| 3826 tok | 4.57 / 4.52 / 4.53 | 4.67 / 4.76 | -3.9% |
| 3929 tok | 4.04 / 4.03 / 3.98 | 4.19 / 4.16 | -3.6% |
| step (28 tok) | 13.60 / 13.56 / 13.57 / 13.49 | 13.74 / 13.82 | **-1.6%** |

`[dec-exp-kprof]` の 1 層 (6 窓すべてで同じ向き):

| 窓 | ZC on d2h | ZC off d2h | ZC on sum | ZC off sum | 正味 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4119 層 | 1.4 | 10.5 | 7.0 | 5.3 | -7.4 us/層 |
| 6117 層 | 0.7 | 8.1 | 6.2 | 5.0 | -6.2 |
| 2293 層 | 1.2 | 9.5 | 7.6 | 5.9 | -6.6 |
| 6144 層 | 0.4 | 5.6 | 4.5 | 4.3 | -5.0 |
| 4029 層 | 0.6 | 7.2 | 5.7 | 4.8 | -5.7 |
| 6135 層 | 0.5 | 5.9 | 4.6 | 4.2 | -5.0 |

起動行は既定で `[flash] exp sum zc: on`、リクエスト行は
`[dec-exp-zc] sum_zc_layers=4119 fallback=0` (zcoff では行が出ない = 分岐が
切れている証拠)。**全 COND で md5 SAME、errors 0。**

decode の中央値は 16.4 (既定 on) vs 16.5 (`=0`) で、0.1 ms 刻みでは 1 段しか
動かない。段 (-7.6%) と step (-1.6%) の方が分解能が高い。

## 4. 数値の扱い

- ZC は**行き先を VRAM から pinned のホストへ変えるだけ**で、カーネル・引数・
  加算順は同じ。**md5 が全 COND で SAME** なのはそのため (丸めは 1 つも変えない)。
- decode の中央値は 0.1 ms/tok 刻みなので、**0.1 ms (0.6%) は判定に使わない**。
  採否は (i) 段の `[q4ledger] experts` を同じ request 同士で比べる、(ii) `step`、
  (iii) `[dec-exp-kprof]` の 1 層の内訳、の 3 つで読んだ。
- t356b の `zcsum_1` は c2000 の 2 窓で step が 17.71 / 16.42 と上振れした
  (page cache の drift。同じ COND の round 2 は 15.72 / 14.55)。だから decode の
  中央値だけを見ると -0.6% に見える。**段と step は 2 ラウンドとも同じ向き**。
- 「P352 の 2.78 ms/tok (ミス CPU)」との差は測定条件の違い (hit_rate 0.910 ->
  0.990)。**測り直したのはこのタスクの実測**で、P352 の数字を否定はしない。

## 5. テスト

```text
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1242 passed; 0 failed; 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 107 passed; 0 failed
```

`CARGO_TARGET_DIR=/tmp/t356`。足したテスト: `sum_zc_gate_defaults_on` (既定 on・
偽値だけ切る)、`sum_zc_line_counts_layers` (積算と行の出し分け)、
`d2h_stream_gate_defaults_off` / `d2h_stream_line_counts_copies` (捨てたゲート)。

## 6. GPU で確認してほしいこと

1. 本番 8098 の起動行に `[flash] exp sum zc: on` が出ること (既定 on)。
   `DENNO_DEC_EXP_SUM_ZC=0` で `off` に戻る。
2. decode のリクエスト行に `[dec-exp-zc] sum_zc_layers=<48 x 層数> fallback=0` が
   出ること。`fallback` が増えるのは pinned の確保に失敗したときだけで、その層は
   従来の D2H 経路へ落ちる (値は同じ)。
3. `[dec-exp-kprof]` の `d2h` が 1-2 us/層 に落ち、`sum` が 6-7 us/層 に上がる
   (ホストへの書き)。合計は base より 5-6 us/層 小さい。
4. 14k の decode で `[q4ledger] experts` が base より小さいこと。

## 7. 次の手 (やっていないこと)

- **experts 段の残りは本物の GEMV 仕事**: gate+up 25.4 us/層 (420-490 GB/s)、
  down 16.8 us/層。速くするには新しい GEMV カーネルが要る (ホスト側の変更では
  もう動かない。P342 の v4 と P361 の AVX-512 はどちらも効かなかった)。
- **silu 4.1 us/層 は起動待ちだけのカーネル**。gate+up の epilogue に畳めれば
  -0.2 ms/tok だが、量子型ごと (IQ2_S / IQ3_S / IQ4_XS / IQ4_NL / Q2_0) の
  カーネルを足すことになる (nvcc が無い開発コンテナでは書けないので lab の PTX=)。
- **sum 4.8-7.2 us/層** も起動待ちが主。down の epilogue に畳むには、down の
  タイルを「出力行 j ごとに全 expert を回す」形へ変える必要がある (新しいカーネル)。
- 段の内訳は **mix 4.14 (gpu 2.48) / ssm 3.63 / experts 3.48 / full 1.51 /
  head 0.74**。mix のホスト側は launch 0.51 + cpu 0.69 で、ここも起動回数
  (339/token = 7/層) を減らすには fusion が要る。
