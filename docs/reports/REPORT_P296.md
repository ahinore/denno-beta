ADOPT: no t296a t296b t296c

# task 296 — IQ3_S 3,826 cold prefill: pinned バウンスの確保し直しを取る (目標 -8% には届かず)

branch `br296` は開始時の `int-auto` (`7c2e83fcb` = br290 の merge 後) から。
作業中に int-auto は動かず (`7c2e83fcb` のまま) なので merge は無い。
lab 3 ラウンド: **t296a** (内訳と 1 候補、sha `512b7f9ed`)、**t296b**
(ホスト負荷 2.8-6.1 で全 COND が 40% 以上ぶれ、使えない)、**t296c** (最終、sha `18527c66d`)。
数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採らない (目標未達)。** 3,826 cold は同じ run の baseline 3.415 s (中央値) →
  3.235 s (**-5.3%**、t296c)。t296a でも 3.37 → 3.18 (-5.6%)。
  **目標 -8% に 2.7 ポイント届かない。**
  14k / decode / md5 / errors はすべて無傷 (下の表)。
- 取ったのは 1 つだけ: **pinned バウンスを「必要量ちょうど」で確保していた**ため、
  グループの必要量が数 MB 揺れるたびに `cuMemHostFree` + `cuMemHostAlloc` が走っていた
  (実測 31 回・255 ms/リクエスト)。余裕つき確保 (`DENNO_PF_PIN_GROW`) で消える。
- 残りは **転送が DDR 帯域の下限に張り付いている** (22.6 GB × 3 往復 = 67.8 GB を
  51 GB/s で流している) ことと、**ssm / mix / full が GPU 律速** (1.12 s) で、
  層の構造上この 2 つが重ならないこと。どちらもホスト側の変更では崩せない
  (前者は 1.32 s の床、後者は .cu / .ptx の仕事)。ゲートは既定 off で残した。

## 1. 前提 (タスク文と lab の実測)

- 基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。先頭に捨て COND (`x0`)、
  末尾に baseline の繰り返し (`base2`)。
- 3,826 cold の baseline は run 間で 3.37〜3.42 s (t296a base2 3.37 / t296c base 3.41・
  base2 3.42)。タスク文の 3.3-3.7 s と一致する。
- 現在の経路: チャンク 1,913 (2 枚)・タイル 3,826 (MoE は層で 1 パス)・プール 1,673 MiB
  (追い出し 0 本)・`ut=0` (転送は計算スレッドで直列)。
  `[exp-pf-stream]` は up_MB=22,635 / 349 グループ / 11,039 本 / 1,836,480 行 /
  probe 16.4-17.1 GB/s、`[exp-pf]` のヒット率は 0.470。

## 2. 台帳 — 3,826 cold の中身 (t296a の `dbg` COND、sha `512b7f9ed`)

`DENNO_PF_LM_DBG=1` の層ループ (48 層の合計、ms)。この行は task 233 で入っていたが、
`pump` の中の**スロット準備と pinned の確保**が見えていなかったので足した (下の 3.)。

| 項目 | ms | 割合 | 中身 |
| --- | ---: | ---: | --- |
| MoE (dev) 合計 | **1,842** | 54.6% | 48 層 × 38 ms |
| ├ experts (ストリーム層) | 1,654 | 49.1% | 下の内訳 |
| ├ router | 86 | 2.6% | F32 GEMM + probs の D2H (7.8 MB/層) |
| ├ topk | 58 | 1.7% | softmax + top-k (ホスト) |
| ├ plan | 10 | 0.3% | by_expert / build_plan / 行表 |
| └ shared | 34 | 1.0% | 共有 expert |
| experts の内訳 | | | |
| ├ pump 合計 | 1,153 | | = prep 255 + 写し 898 |
| │  ├ **prep (pinned の確保し直し)** | **255** | 7.6% | **31 回**、1 回 8.2 ms |
| │  └ 写し (mmap → pinned) | 898 | 26.6% | 22.6 GB = 25.2 GB/s |
| ├ H2D 待ち | 422 | 12.5% | 22.6 GB = 53.6 GB/s (PCIe の上限) |
| ├ accum (カーネル + 同期) | 60 | 1.8% | 48 回 |
| ├ 表 / 起動 | 9 | 0.3% | |
| └ 残差 | 10 | 0.3% | |

`[q4ledger-pf]` の段 (同じ run の base2): mix 264 / full 232 / ssm 618 / other 2,254。
`other` の中は PLE 151 / combine 83 / embed 4 で、残りが上の MoE。
`[pf-host]` は stage_copy 898 / up_wait 423 / sync 60 / plan 4.3 / ptrtab 3.0 / scratch 3.4。

**読み方**: 転送の 3 往復は
「ページキャッシュ読み 22.6 GB + WC pinned 書き 22.6 GB」= 898 ms (50.3 GB/s) と
「DMA 読み 22.6 GB」= 422 ms (53.6 GB/s) で、**合わせて 67.8 GB / 1.32 s = 51.3 GB/s**。
P227 が同じ機械で測った「写し + DMA の 3 往復 = 51 GB/s が実質上限」と一致する。
つまり転送そのものはもう下限で、**上に乗っていた 255 ms が唯一の無駄**だった。

## 3. 変えたこと (2 コミット、どれも既定 off)

1. `512b7f9ed` **診断**: `PF_STREAM_T` を 5 → 8 要素にし、`pf_slot_prepare` (pump の
   内側) と pinned の確保、確保し直した回数を積む。`[pf-lm-dbg] stream` 行に
   `prep= pin= x<回数> sum= resid=` を足した。off のときの追加コストはグループあたり
   `Instant::now()` 2 回 (実測 10 µs/リクエスト未満、台帳の分解能以下)。
2. `512b7f9ed` **`DENNO_PF_PIN_GROW`** — pinned バウンスを「必要量 +25% (最低 16 MiB)、
   8 MiB 切り上げ、ステージング 1 枚が上限」で確保する (`exp_pf::pin_grow_target`、純関数)。
   従来は `cuMemHostAlloc(need)` ちょうどだったので、グループの必要量 (expert 本数 ×
   2.05 MB) が前より 1 バイトでも大きいと確保し直していた。
3. `18527c66d` **`DENNO_F32_GEMM_PIN_D2H`** — `f32_gemm_batch_run` の D2H が 1 MiB 以上の
   とき pinned へ非同期で落として 1 回の同期で待つ (router 7.8 MB/層、PLE の key/value
   22 MB/チャンク)。**t296c で単独の効果が見えなかった** (`ple` は 154 → 160 ms) ので
   既定 off のまま残した (下の 5.)。

数値経路: 2 はデバイスへ渡るバイト列が同じ (置き場所の大きさだけ違う)、3 も同じ
バイト列を同じ順でコピーする。**md5 は 3 ラウンド全 COND で SAME**。

## 4. 数値 (lab、すべて実測)

### t296a (`512b7f9ed`、load_before=2.02) — 内訳と候補

| COND | 3,826 cold (r1 / r2) | 14k cold | 短 decode | md5 |
| --- | --- | ---: | ---: | --- |
| x0 (捨て) | 9.90 / 3.38 | 11.37 | 17.9 | SAME |
| base | 4.48 / 3.39 | 11.12 | 17.9 | SAME |
| dbg (`DENNO_PF_LM_DBG=1`) | 3.38 / 3.38 | 8.35 | 17.9 | SAME |
| **pgr (`PIN_GROW=1`)** | **3.18 / 3.18** | 8.19 | 17.9 | SAME |
| g128 (`STREAM_GROUP=128`) | 3.49 / 3.50 | 8.51 | 17.9 | SAME |
| base2 | 3.37 / 3.37 | 8.38 | 18.0 | SAME |

- pgr vs base2 = **-5.6%**。g128 は **+3.8%** (グループを大きくすると悪い。
  ただしこの COND は PIN_GROW 無しなので、確保し直しが大きいぶんも乗っている)。
- `[pf-lm-dbg] stream` の `pin=` は base で 31 回 255 ms (層 0-30 に集中)、
  pgr では数回・10 ms 前後。

### t296b — 使えない

load_before=5.71、各 COND の load 2.76-6.06。3,826 cold は base 9.71/5.50、
base2 6.23/5.19 と 40% 以上ぶれる。pg32wc (`GROUP=32 WC=0`) と pg128 の数字は
このノイズに埋もれて判断できない (一番良く見えた pg128 の round 2 = 3.24 も、
同じ round の base2 5.19 との差を説明できない)。**採否には使っていない。**

### t296c (`18527c66d`、load_before=5.71 だが安定) — 最終

| COND | 3,826 cold (r1 / r2) | 14k cold | 短 decode | 14k 後 decode |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て、両ゲート 0) | 5.02 / 3.43 | 10.94 | 17.9 | 18.5 |
| base (両ゲート 0) | 3.41 / 3.41 | 8.42 | 17.9 | 17.0 |
| **new (既定 = 両ゲート on)** | **3.24 / 3.23** | 8.32 | 17.9 | 17.2 |
| rs (new + `PF_ROUTER_SM=1`) | 3.24 / 3.24 | 8.36 | 17.9 | 17.0 |
| new2 (new の繰り返し) | 3.23 / 3.24 | 8.40 | 17.9 | 17.1 |
| base2 (両ゲート 0) | 3.42 / 3.42 | 8.46 | 17.9 | 17.0 |

- **3,826: base 3.415 (中央値) → new 3.235 = -5.3%** (round ごとでは -5.0% / -5.3%)。
  14k は -0.9%、短 decode ±0%、14k 後 decode は最大 +1.2% (17.0 → 17.2、許容 +2% 以内)。
- `other` (MoE) は 2,284 → 2,114 ms (-170)。写し 940 → 928、H2D 待ち 441 → 422 は
  page cache の差。**`ple` は 153.8 → 159.7 で、pinned D2H (3.) は効いていない。**
- `PF_ROUTER_SM=1` は差 0 (3.24)。台帳の `topk`/`router` も動かない。
- errors 0、text は全 COND `looks ok`、md5 は 6 出力すべて SAME。
- 本文 (new の 14k cold の答え、`TXTD new`):
  「ご提示いただいた文書は、LLM推論エンジン（c2000プロジェクト）における、**Prefill（事前計算）フェーズの性能最適化の歴史を記録した技術レポートです。」
  (a は `A **B-tree** (specifically, the **B+ tree) is the fundamental data structure used by
  most modern relational databases...`、base2 と同一の書き出し)。

### 目標の判定

| 条件 | 目標 | 実測 | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -8% 以上 | **-5.3%** (t296a は -5.6%) | **未達** |
| 14k cold | 悪化しない | -0.9% | OK |
| 短 decode | +2% 以内 | ±0% | OK |
| 14k 後 decode | +2% 以内 | +1.2% | OK |
| errors / text | 0 / sane | 0 / sane | OK |

## 5. なぜ -8% に届かないか (機序)

3,235 ms の内訳 (t296c の new、`[q4ledger-pf]`):

| 段 | ms | 割合 | 律速 |
| --- | ---: | ---: | --- |
| MoE の転送 (写し + H2D 待ち) | 1,350 | 41.7% | **DDR 帯域** (3 往復 67.8 GB / 51 GB/s) |
| MoE のホスト仕事 (router/topk/plan/shared/pin) | ~190 | 5.9% | ホスト |
| accum の同期 (48 回) | 60 | 1.9% | GPU 待ち |
| PLE 注入 (ホストのトークンループ) | 160 | 4.9% | ホスト |
| combine | 84 | 2.6% | ホスト |
| ssm (scan_pf 359 / qkv_pf 142 / out_pf 44) | 618 | 19.1% | **GPU** |
| mix (hc mix の GEMM 4 本/層) | 263 | 8.1% | **GPU** |
| full (full attention 12 層) | 235 | 7.3% | **GPU** |
| その他 (層ループのホスト仕事 / embed / PLE の表読み) | 275 | 8.5% | ホスト |

1. 転送は 3 往復 (モデル読み → pinned 書き → DMA 読み) が DDR 帯域で決まる。
   バイト数 22.6 GB は「VRAM 22 GB / 重み合計 ~50 GB」から来ていて、3,826 トークン
   1 パスでは**ほぼ下限** (ヒット率 0.470 = VRAM に入る上限)。減らすには
   VRAM を増やす (= 過去セッションの先読み。棄却済み) か、往復を減らす
   (= `cuMemHostRegister` が file backed を拒む。P227 で棄却済み) しかない。
2. ssm / mix / full は GPU 律速で、`sync` の内訳がそのまま GPU 時間 (541 / 202 / 129 ms)。
   `scan_pf` 359 ms は 3,826 トークン × 36 層で 0.2 TFLOPS 級、mix は 12 TFLOPS 級で、
   どちらもカーネル側 (.cu / .ptx) の仕事。このタスクでは触れない。
3. この 2 つは**層の中で直列** (mix → attn → ssm → MoE の順) なので、転送中は GPU が遊び、
   GPU 工作中はホストが遊ぶ。重ねられれば理論上 -1.0 s (31%) ある。次の候補はこれ。

## 6. 数値の扱い

- lab の値はすべて実測。このコンテナに GPU は無いので、ここでの推定は「推定」と書いた。
- 転送の 3 往復と 51 GB/s は P227 の実測 (同じ機械) と、こちらの
  copy_s / wait_s / up_MB からの逆算が一致している。
- t296b はホスト負荷で壊れているので、採否の判断に使っていない (数字は 4. に残した)。
- md5 は 3 ラウンド 18 COND すべて SAME。数値経路を変える変更は入れていない。

## 7. テスト

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,184 passed / 0 failed / 100 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 95 passed / 0 failed
```
追加: `exp_pf::tests::stream_gate_parses_values` に `pin_grow` の真理値表と
`pin_grow_target` の契約 (0 / +25% / 最低 16 MiB / 8 MiB 切り上げ / 上限)、
`cuda::f32_d2h_pin_tests::f32_gemm_pin_d2h_gate_truth_table` (真理値表と 1 MiB の
しきい値が 3,826 の router / PLE に入り、decode の 1 トークンは入らないこと)。

## 8. GPU で確認してほしいこと / 次にやるなら

1. **`DENNO_PF_PIN_GROW=1` を既定にするか**の判断。3,826 cold は -5%、14k と decode は
   無傷、md5 SAME。目標 -8% に届かないので規則どおり既定 off にしてある。
   実機 (8098) の 3,826 級プロンプトで `[pf-lm-dbg] stream` の `pin=` が 0 に近づくことを
   確認してから on にするのが安全。
2. **次の本命は「転送と GPU 仕事の重ね」**。層 L の attn/ssm の間に層 L の expert を
   上げるには router が先に要る (現状は mix → router → MoE の順)。案:
   (a) チャンク c の mix 出力で router を先に回し、attn/ssm の裏で expert を上げる、
   (b) MoE をチャンク単位に分けて (バイトは増える) 前半チャンクの転送を後半の attn と重ねる。
   どちらも 1.0 s 級の余地がある (推定)。
3. `PF_STREAM_GROUP` は 128 で +3.8% (t296a)、32 は測れていない (t296b が負荷で没)。
   `WC=0` + 小グループ (L3 96 MB に 1 グループを収める) も未検証 — 3 往復のうち
   DMA 読みを L3 ヒットにできれば -0.4 s 級の可能性がある (推定)。
4. `DENNO_F32_GEMM_PIN_D2H` は 3,826 では効果が見えなかった。UD-Q4_K_XL のように
   PLE/router の D2H がもっと大きいモデルで測る価値はある (未検証)。
5. **`other` のうち層ループのホスト仕事 (~250 ms) はまだ分解できていない。**
   既知なのは `pf_take_mixed` の D2H 37 ms (96 回 × 19.6 MB = 50 GB/s) だけ。
   `DENNO_PF_LM_DBG` は層の中 (`moe_ffn_quant_batch_lm_dev_at`) しか見ていないので、
   次はチャンクループの外側 (mix の D2D 退避・attn のホスト側・combine) にも
   同じ区間計測を入れてから候補を選ぶのがよい。

## 9. lab の VERDICT 行

t296a:
```
VERDICT base vs x0 OK decode +0.2% prefill +0.8% md5 SAME errors 0
VERDICT dbg vs x0 OK decode +0.4% prefill -0.8% md5 SAME errors 0
VERDICT pgr vs x0 OK decode +0.2% prefill -0.8% md5 SAME errors 0
VERDICT g128 vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.7% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -2.0% prefill -40.7%
VERDICT_DEPTH dbg c2000_cold vs x0 decode -12.5% prefill -49.1%
VERDICT_DEPTH pgr c2000_cold vs x0 decode -12.8% prefill -52.1%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -11.8% prefill -49.2%
VERDICT baseline x0 errors 0 text ok
```
t296b (負荷で無効):
```
VERDICT base vs x0 OK decode +0.0% prefill +3.3% md5 SAME errors 0
VERDICT pg vs x0 OK decode +3.0% prefill +10.6% md5 SAME errors 0
VERDICT pg32wc vs x0 OK decode -0.8% prefill -0.8% md5 SAME errors 0
VERDICT pg128 vs x0 OK decode +3.1% prefill +0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +3.0% prefill -0.8% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
t296c:
```
VERDICT base vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
VERDICT new vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT rs vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT new2 vs x0 OK decode +0.4% prefill +0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -3.6% prefill -19.3%
VERDICT_DEPTH new c2000_cold vs x0 decode -4.3% prefill -23.4%
VERDICT_DEPTH rs c2000_cold vs x0 decode -4.3% prefill -23.3%
VERDICT_DEPTH new2 c2000_cold vs x0 decode -3.6% prefill -23.4%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -4.3% prefill -19.1%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT/VERDICT_DEPTH はどれも**最初の COND (x0) との比較**で、x0 は冷えた 1 発を含む。
採否は base/base2 と new/new2 の生値の中央値で読んだ — 表 4.)

t296c の x0 round 1 は 3,826 cold が 5.02 s = 最初の COND のバイアス (round 2 は 3.43)。
