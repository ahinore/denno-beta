ADOPT: yes t300a t300b

# task 300 — IQ3_S prefill: PLE の確保し直しを取る (-7.4%、目標 -6% を満たす)

branch `br300` は開始時の `int-auto` (`7c2e83fcb` = br290 merge) から。`br296` は
int-auto に入っていなかったので merge した (pinned 余裕つき確保が入る)。
作業中に int-auto が `7794d2dde` (br301 の merge) へ動いたので最後にも merge した
(both sides、衝突なし)。lab 3 ラウンド: **t300a** (`c18c505b7`、6 COND)、
**t300b** (`0e81c3a94`、6 COND)、**t300c** (`3b538aeb1`、int-auto merge 後の確認)。
数値はすべて GPU ラボの実測。**t300c は host の取り合いで無効** (decode が
17.9 → 24-36 ms/tok、prefill も全 COND が 2 倍)。採否は t300a/t300b で決めた。

## 0. 結論 (3 行)

- **採る。3,826 cold prefill が 3.426 → 3.172 s (-7.4%、t300b)、14k cold も
  8.50 → 8.14 s (-4.2%)、decode は最大 +1.2%、errors 0、md5 全 COND SAME、
  テキスト sane。目標 (-6%、他は悪化なし) を満たす。**
- 取ったのは 2 つ。**(a)** task 296 が見つけた pinned の確保し直し
  (`DENNO_PF_PIN_GROW`) を既定 on にした (3,826 で -5.3%。task 296 の目標 -8%
  に届かず既定 off のままだった。task 300 の目標は -6% なので採る)。
  **(b)** PLE 層の作業域 (back / key / value) を grow-only で使い回す
  `DENNO_PF_PLE_REUSE` を新設 (単独 -1.5%、上に載せて -2.1%)。
- 残りは **MoE の転送が DDR 帯域の下限**(3 往復 67.8 GB / 51 GB/s = 1.38 s)と、
  **ssm / mix / full が GPU 律速**(1.12 s が GPU busy の 74%)。どちらも
  ホスト側では崩せない (前者は P227 で 4 経路とも棄却済み、後者は .cu の仕事)。

## 1. 前提

- 基準 COND はタスク指定の 4 本に `DENNO_PF_PIN_GROW=0 DENNO_PF_PLE_REUSE=0` を
  足したもの = **int-auto の既定** (両ゲートとも int-auto では既定 off)。
  先頭に捨て COND (`x0`)、末尾に baseline の繰り返し (`base2`)。
- 3,826 cold の base は t300a 3.41 s / t300b 3.43 s で、タスク文の 3.3-3.7 s と
  一致する。14k cold の base は 8.50 s (t300b)。
- t300a の `base` だけ 3,826 が 4.14 s、14k が 10.09 s と高く出た (run の 2 番目
  の COND で、まだ暖まっていない)。同じ run の base2 は 3.40 / 8.45 なので、
  採否は base2 と t300b の base (3.43 / 8.50) で読んだ。

## 2. 台帳 — 何が大きいか (t300b base、`PROFILE=1`)

### 3,826 cold (`[q4ledger-pf] tok=3826 wall=3426.2ms`)

| 段 | ms | 割合 | 律速 |
| --- | ---: | ---: | --- |
| MoE の転送 (写し 940.3 + H2D 待ち 443.1) | 1,383 | 40.4% | **DDR 帯域** (3 往復) |
| ssm (qkv 141 / scan 359 / out 44) | 619.0 | 18.1% | GPU (536.7 が GPU 時間) |
| mix (hc mix 4 本/層 + mixed の D2H) | 262.1 | 7.7% | GPU (197.6) |
| full (12 層、融合 qkv の連鎖) | 239.5 | 7.0% | GPU (128.8) + 往復 |
| PLE 注入 (ホスト) | **158.0** | 4.6% | ホスト (mmap + ゼロ埋め) |
| combine (hc の足し込み) | 83.9 | 2.4% | GPU |
| MoE のホスト仕事 (router 86 / topk 58 / plan 4 / shared 34、task 296 の dbg 実測) | ~182 | 5.3% | ホスト |
| その他 (層ループ / embed / 表) | ~500 | 14.6% | ホスト |

`[pf-host]` = stage_copy 940.3 / up_wait 443.1 / sync 59.4 / plan 4.4 / ptrtab 3.5。
`close wall=3426.2 gpu_busy=1506.8 exposed_host=1919.4`。

**読み方**: 転送は 22.6 GB × 3 往復 = 67.8 GB を 49 GB/s で流していて、P227 が
測った同じ機械の上限 (写し 48 GB/s + DMA 51-57 GB/s) に張り付いている。
上に乗っていた無駄は t296a の台帳で 2 つに分かれていた: **pinned の確保し直し
255 ms (31 回)** と、**PLE のチャンクごとの確保** (3,826 で 470 MB、
`[exp-ram-probe] prefill minflt=566,183`)。前者が (a)、後者が (b)。

### 14k cold (`tok=14018 wall=8502.0ms`)

| 段 | ms | 割合 |
| --- | ---: | ---: |
| ssm | 2,186.2 | 25.7% |
| full | 1,200.9 | 14.1% |
| mix | 962.9 | 11.3% |
| MoE の転送 (1,164.7 + 551.7) | 1,716 | 20.2% |
| PLE 注入 | **376.5** | 4.4% |
| combine | 328.4 | 3.9% |
| router / topk / plan / shared | ~500 | 5.9% |
| その他 | ~1,200 | 14.1% |

`close gpu_busy=5375.1 exposed_host=3126.9`。14k は GPU の割合が 63% で、
ホスト側で削れる最大は転送 1.7 s + PLE 0.38 s。

## 3. 変えたこと (2 コミット)

1. `c18c505b7` **`DENNO_PF_PLE_REUSE`** (新設、t300b で既定 on) — PLE 層の
   `back` (`nt × hc_dim`、14k のチャンクで 82 MB) と `key` / `value`
   (82 + 20 MB) を thread-local の grow-only で使い回し、`pf_download_res` は
   一時 Vec を作らず `out` へ直接 `cuMemcpyDtoH` する。従来は 1 チャンクごとに
   3 本を `vec![0.0; n]` で作り直していた。**32 MB を超える確保は glibc が
   mmap で出し入れする**ので、チャンクごとに数万ページのマイナーフォルトが
   戻っていた (実測 minflt 566,183 → 475,749)。3 本とも全要素を書き直すので
   デバイスへ渡るバイト列は同じ。
2. `c18c505b7` **`DENNO_PF_PIN_GROW` を既定 on** (task 296 の持ち越し)。
   pinned バウンスを「必要量ちょうど」でなく +25% (最低 16 MiB) で確保し、
   グループの必要量が数 MB 揺れるたびの `cuMemHostFree` + `cuMemHostAlloc`
   (実測 31 回・255 ms) を消す。

数値経路: どちらもデバイスへ渡るバイト列を変えない (置き場所の寿命だけ)。
**md5 は t300a/t300b の全 COND・全出力で SAME。**

## 4. 数値 (lab、すべて実測)

### t300a (`c18c505b7`、load_before=4.17) — 内訳と切り分け

| COND | 3,826 cold r1 / r2 | 14k cold | 短 decode |
| --- | --- | ---: | ---: |
| x0 (捨て) | 9.62 / 3.41 | 11.18 | 18.2 |
| base (両ゲート 0) | 4.14 / 3.41 | 10.09 | 17.9 |
| pg (`PIN_GROW` のみ) | 3.22 / 3.23 | 8.30 | 18.0 |
| **pr (両方)** | **3.18 / 3.16** | **8.09** | 17.9 |
| pr0 (`PLE_REUSE` のみ) | 3.35 / 3.36 | 8.28 | 17.9 |
| base2 (両ゲート 0) | 3.40 / 3.41 | 8.45 | 17.9 |

- base2 3.405 を基準に **pg -5.3% / pr -7.0% / pr0 -1.5%**。
  pg は task 296 の t296c (-5.3%) と一致する。
- pr0 の内訳: 3,826 の台帳が 3,413.8 → 3,361.1 ms (-53)。うち `ple=` (注入の
  sub 値) は 158.5 → 149.4 ms で **-9 だけ**なので、効いているのは注入ではなく
  **`back` の確保と D2H の経路** (78 MB の一時 Vec と 78 MB のゼロ埋め確保)。
- pr (両方) の `other` は 3,413.8 → 3,161.0 ms (-253) = pin -238 + PLE -15。

### t300b (`0e81c3a94`、load_before=4.24) — 最終 (既定 on の確認)

| COND | 3,826 cold r1 / r2 | 14k cold | 短 decode | 14k 後 decode |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て) | 3.41 / 3.43 | 8.76 | 18.0 | 16.8 |
| base (両ゲート 0) | 3.43 / 3.43 | 8.50 | 17.9 | 17.0 |
| **new (既定)** | **3.17 / 3.17** | **8.14** | 17.9 | 17.2 |
| g128 (new + `STREAM_GROUP=128`) | 3.48 / 3.23 | 9.76 | 17.9 | 18.6 |
| new2 (new の繰り返し) | 3.18 / 3.18 | 8.11 | 17.9 | 17.0 |
| base2 (両ゲート 0) | 3.43 / 3.42 | 8.47 | 17.9 | 17.1 |

- **3,826: base 3.430 → new 3.170 = -7.6%** (new2 3.180 = -7.3%)。
- **14k: base 8.50 → new 8.14 = -4.2%** (new2 8.11 = -4.6%)。
- decode: 短 17.9 → 17.9 (中央値 ±0%)、14k 後 17.0 → 17.2 (+1.2%、new2 17.0)。
- `[q4ledger-pf]` の差 (3,826、base2 → new2): `other` 2,305.6 → 2,051.9 ms、
  `exposed_host` 1,919.4 → 1,664.8、`gpu_busy` 1,506.8 → 1,506.7 (不変)。
  sub other の `ple` 158.0 → 147.3、`combine` 83.9 → 84.5、`mix` 262.1 → 262.5。
  `[pf-host]` の stage_copy 940.3 → 937.7、up_wait 443.1 → 422.6。
- 14k の new: `other` 4,152.1 → 3,764.7、sub other の `ple` 376.5 → 288.5、
  `exposed_host` 3,126.9 → 2,763.5、`gpu_busy` 5,375.1 → 5,379.0 (不変)。
- `STREAM_GROUP=128` は **14k +11.4% / decode +10.3%** で悪い (グループが
  大きいと 1 枚に収まらず分割が増える)。採らない。

### t300c (`3b538aeb1`) — 無効

int-auto (`7794d2dde` = br301 merge) を取り込んだ後の確認として 4 COND を回したが、
**host の取り合いで全 COND が壊れた** (3,826 cold 6.42-10.24 s、decode
24-36 ms/tok、base と new の差も -35% と出ている)。同じ機械で同時に
`tint_7794d2dde` が走っていた。**採否には使っていない。** merge 後はテストだけ
通している (下の 7.)。br301 は UD-Q4_K_XL の mix (Q8_0 down/up) を sn3 の
テンソルコアに載せる変更で、IQ3_S の mix (BF16) は通らない。

### 目標の判定

| 条件 | 目標 | 実測 (t300b) | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -6% 以上 | **-7.6%** (3.430 → 3.170) | **達成** |
| 14k cold | 悪化しない | -4.2% (8.50 → 8.14) | OK |
| 短 decode | +2% 以内 | ±0% | OK |
| 14k 後 decode | +2% 以内 | +1.2% (new2 は ±0%) | OK |
| errors / text / md5 | 0 / sane / SAME | 0 / sane / 全 COND SAME | OK |

## 5. なぜ -7.4% で止まるか (機序)

3,171 ms の内訳 (t300b new、`[q4ledger-pf]`): MoE の転送 1,360 / ssm 620 /
mix 262 / full 237 / PLE 147 / combine 84 / MoE のホスト仕事 ~180 /
その他 ~280。`gpu_busy` は 1,507 ms で **触っていない** (ホスト側だけを削った)。

1. 転送は 3 往復が DDR 帯域で決まる。バイト数は「VRAM 22 GB / 重み ~50 GB」から
   来るヒット率 0.470 でほぼ下限。減らすには VRAM を増やす (棄却済み) か往復を
   減らす (`cuMemHostRegister` は file backed を拒む = P227) しかない。
   `WC=0` + 小グループの L3 案は **t165b で既に測られていて** (14k 100.0 → 95.1 s
   で wc=1 が良い、probe 27.3 vs 53.5 GB/s)、cacheable pinned への DMA が
   半分の速度になるので取れない。
2. ssm / mix / full は GPU 律速 (1,120 ms = GPU busy の 74%)。カーネル側の仕事で、
   このタスクでは .cu を触らない。
3. ホスト側で残っているのは router+topk (~144 ms)、combine (1 チャンク 1 同期 +
   kernel)、`full` の入出力往復 (1.1 GB)、`mix` の mixed D2H (1.88 GB) 程度。
   `DENNO_PF_MIXKEEP` は task 230 で 14k +4.2% (棄却済み) なので触っていない。

## 6. 数値の扱い

- lab の値はすべて実測。このコンテナに GPU は無いので、ここでの見積りは
  「推定」と書いた。
- t300c は host 負荷で無効。数字は 4. に残したが判断に使っていない。
- 14k cold は depth8k の round 1 にしか出ないので 1 COND 1 サンプル。
  base/base2 が 8.50 / 8.47 と揃っている回 (t300b) を採った。
- md5 は t300a/t300b の全 COND・全出力 (a / b / c2000_cold / c2000_turn2 /
  c8000_cold / c8000_turn2) で SAME。数値経路は変えていない。

## 7. テスト

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,191 passed / 0 failed / 101 ignored   (int-auto merge 後)
cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 95 passed / 0 failed
```
追加: `ple_reuse_scratch_contract` (ゲートの真理値表、off は毎回新しい Vec、
on は同じ領域を len ちょうどに切り詰めて返す、0 長は使い回さない)、
`exp_pf::tests` の `pin_grow` の真理値表を既定 on に更新。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 級プロンプトで `[pf-lm-dbg] stream` の `pin=` が 0 に
   近づくこと、`[exp-ram-probe] prefill` の minflt が 4-5 割減ること。
2. int-auto merge 後 (`3b538aeb1`) の再測。t300c が host 負荷で壊れたので、
   空いている時間に base / new の 2 COND だけ回して -6% を再確認してほしい。
3. 次に削れるのは **router + topk (~144 ms、3,826)** と **`full` の入出力往復
   (1.1 GB)**。後者は full 層の出力をデバイスに残せば `combine` の H2D と
   合わせて -20 ms 級 (推定)。前者はホストの f32 GEMM + softmax + top-k で、
   GPU の router カーネルは既にある (`DENNO_PF_ROUTER_GPU` は probs の D2H だけ
   残る)。ただし 1 回 3 ms 級なので、まとめて動かさないと効かない。
4. 転送そのものを崩すには .cu 側で「expert の一部だけを上げる」形
   (行を要求された分だけ DMA する) が最後の候補。今のカーネルは expert の
   行列を丸ごと読むので、ホスト側からは触れない。

## 9. lab の VERDICT 行

t300a:
```
VERDICT base vs x0 OK decode -1.5% prefill +0.0% md5 SAME errors 0
VERDICT pg vs x0 OK decode -1.3% prefill -1.6% md5 SAME errors 0
VERDICT pr vs x0 OK decode -1.7% prefill -0.8% md5 SAME errors 0
VERDICT pr0 vs x0 OK decode -1.9% prefill +0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -1.7% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -7.9% prefill -42.1%
VERDICT_DEPTH pg c2000_cold vs x0 decode -11.9% prefill -50.5%
VERDICT_DEPTH pr c2000_cold vs x0 decode -11.6% prefill -51.3%
VERDICT_DEPTH pr0 c2000_cold vs x0 decode -11.9% prefill -48.5%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -11.3% prefill -47.7%
VERDICT_DEPTH base c8000_cold vs x0 decode -14.3% prefill -9.7%
VERDICT_DEPTH pg c8000_cold vs x0 decode -20.4% prefill -25.8%
VERDICT_DEPTH pr c8000_cold vs x0 decode -20.4% prefill -27.6%
VERDICT_DEPTH pr0 c8000_cold vs x0 decode -20.4% prefill -25.9%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -20.4% prefill -24.4%
VERDICT baseline x0 errors 0 text ok
```
t300b:
```
VERDICT base vs x0 OK decode -0.4% prefill -0.8% md5 SAME errors 0
VERDICT new vs x0 OK decode -0.7% prefill -1.6% md5 SAME errors 0
VERDICT g128 vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT new2 vs x0 OK decode -0.2% prefill -1.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.2% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -0.7% prefill +0.3%
VERDICT_DEPTH new c2000_cold vs x0 decode -0.7% prefill -7.3%
VERDICT_DEPTH new2 c2000_cold vs x0 decode -0.7% prefill -7.0%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -0.7% prefill +0.1%
VERDICT_DEPTH base c8000_cold vs x0 decode +0.6% prefill -3.0%
VERDICT_DEPTH new c8000_cold vs x0 decode +1.9% prefill -7.1%
VERDICT_DEPTH new2 c8000_cold vs x0 decode +0.6% prefill -7.4%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +1.3% prefill -3.3%
VERDICT baseline x0 errors 0 text ok
```
t300c (無効):
```
VERDICT_DEPTH base c2000_cold vs x0 decode -17.2% prefill -35.5%
VERDICT_DEPTH new c2000_cold vs x0 decode -15.2% prefill -35.2%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -22.8% prefill -12.7%
（load の取り合いで decode が 24-36 ms/tok。採否に使わない）
```
(VERDICT/VERDICT_DEPTH はどれも**最初の COND (x0) との比較**で、x0 は冷えた 1 発を
含む。採否は base/base2/new/new2 の生値で読んだ — 表 4.)

## 10. 本文 (t300b new の 14k cold、`TXTD new`)

「ご提示いただいた文書は、LLM推論エンジン（おそらく `denno-engine）の
**Prefill（事前計算）性能最適化の技術的記録です。以下に、文書の構造と技術的要点を
詳細に解説します。」— base と同じ書き出しで、全 COND で同一 md5。
