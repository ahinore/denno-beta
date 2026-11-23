ADOPT: no t502b

# REPORT P502 — UD-Q4_K_XL: experts 段の最後の「名前の付いたホスト固定費」だった
# seg 層の和 D2H (10 KB/層) を**非同期**にしてミス CPU 計算の裏へ回した。分岐は
# 100% 発火 (6048 本/リクエスト、fallbacks 0) し、`[seg-prof] d2h` は
# 12-17 -> 1.1-1.5 us/層 に落ちたが、**壁は -1.8%** (median 14.05 -> 13.8 ms/tok)
# で、採否の線 (2%) と目標 (3%) に届かなかった。既定 off のまま残す。

branch: **br502** = 開始時の int-auto **890fc66 (merge: br496)**。`br498` は int-auto に
未 merge だったので merge した (衝突なし。498 の `DENNO_SEG_SUM_PIN` は既定 off のまま
残る)。終端で int-auto を fetch し直したが **890fc66 のまま** (動いていなかった) なので
merge するものは無い。

## 0. 結論 (6 行)

1. **採らない (`ADOPT: no`)。** 目標 (UD 3,826/14k cold prefill -5%、または UD warm
   short decode -3%) に届かず、オペレーターの採否線 (2 ラウンドの median で 2%) にも
   僅かに届かない (**-1.8%**)。
2. **機構は狙いどおり動いた。** `[seg-async] calls=6048 fallbacks=0` (48 層 x 126 token、
   1 リクエストあたり)、`[seg-prof] d2h` は **12-17 -> 1.1-1.5 us/層** (48 層で
   0.55-0.75 ms/token = decode の 4-5% を**消した**)。それでも壁が動かない。
3. **理由 (推定)**: 同期 `cuMemcpyDtoH` の「この文脈の仕事を待つ」ぶんが、
   実は**背景の上げ直し (`[dec-up-w]`) の待ちを吸収していた**。待ちを消すと
   同じ待ちがミス区間 (`miss=` は包括区間で 500-675 us/層) へ移るだけで、
   層の壁は変わらない。task 498 の pinned **同期**版が 0% だったのと、
   task 356 (専用ストリーム) が 0% だったのと同じ壁の別の見え方。
4. **既定モデルは 1 命令も変わらない (t502c)**: UD-IQ3_XXS では seg K 型経路を
   通らないので `[seg-async] calls=0` (全 14 リクエスト)、decode 10.6 ms/tok が
   x0/b/sa/b2 で同値、md5 全 COND SAME、errors 0。
5. 前回 (P498) の結論「experts = 4.1 ms 固定費 + 57 us x ミス本数」のうち、
   固定費の名前付き内訳で残っていたのは **launch 9.1 / d2h 13.8 / h2d 2.9 /
   lookup+table 2.2 us/層**。d2h はこれで**機構として閉じた** (消しても壁が動かない)。
   残るのは launch (5 起動の融合 = `.cu`、task 412 が -0.6% で棄却済み) だけ。
6. **prefill の判定はしていない**。t502a の 14k cold は b 21.78 / sa 19.76 / b2 19.24 s で
   ばらつき (この変更は prefill の層経路にも当たるが、decode 用の D2H なので
   効くのは decode だけ)。3,826 cold も t502b で b 16.79 / sa 16.23 / b2 16.15 /
   sa2 16.05 s = 分解能内。

## 1. なぜこの段を選んだか (Do 1 / Do 2)

P498 の段プロファイル (warm short decode 13.33 ms/token) は
experts **5.15 (39%)** / ssm 2.99 (22%) / mix 2.24 (17%) / full 1.27 / shared 0.61 /
head 0.61 / router 0.19。最大の段は experts で、その中身は
**「ミス 1 本 57 us (= 51 GB/s = DRAM の床)」+「層あたり 85 us の固定費」**。
固定費のうち名前が付いていて Rust から削れる最後の 1 つが
**d2h 13.8 us/層 (0.66 ms/token = 5%)** だった (P498 §7-4 が
「次に狙うならこれしかない」と明記)。ssm は task 496 (dn8) が直前に触っており、
mix は qchain が 100% 発火済み、full/head は GPU 律速で `.cu` 不可、prefill は
P494 のとおりディスクの床。よって **experts の d2h** を選んだ。

## 2. 変えたこと (既定 off)

1. **`DENNO_SEG_SUM_ASYNC`** (新規、既定 off): 和カーネル (5 起動の最後) の直後に
   `cuMemcpyDtoHAsync` を**スレッド専用**の 64 KiB pinned バウンスへ出し、
   コピーの後にイベント (`seg_ev` slot 7) を打つ。ミス CPU 計算 (1 層 ~100 us) を
   挟んだ後で、そのイベント 1 本だけを待って pinned -> `out` へ写す。
   - 同期 (`cuMemcpyDtoH`) との違いは **「待ちの範囲」**: 同期コピーはこの文脈の
     仕事を待つので、待ちが層のカーネルより広く、しかもミス計算の**後ろ**に直列で
     挟まる。非同期は stream に積むだけで、待ちは「コピーの後ろのイベント」1 本。
   - 失敗 (pinned が取れない / CUDA 無し / イベントが張れない / 範囲外) は false を
     返して従来の同期経路へ丸ごと落ちる (**fail closed**、値は同じバイト列)。
   - バンスをプロセス共有にするとミス計算の間ずっとロックを握るので **スレッド局所**に
     した (`cuMemHostAlloc` はデバイスを止めるため初回だけ)。
2. ゲートとカウンタは `model::seg_prof` 側 (cuda 無しのビルドでも行が出る)。
   起動 `[flash] seg sum async: on|off`、リクエスト終端
   `[seg-async] calls=.. fallbacks=..` (ゲート on のときだけ)。
3. テスト 5 本 (`model::seg_prof::tests::t502_*` 2 本、`cuda::task502_tests` 3 本)。

**既定値・既定の経路は変えていない。** ゲート off では `async_ok` が false で
`if async_ok` が短絡し、従来の `seg_sum_pin_on()` / `to_slice` がそのまま走る
(分岐 2 回だけ増える)。UD-IQ3_XXS では**この経路自体に入らない** (`calls=0`、§3.3)。

## 3. lab の結果 (すべて `tree=denno-dev3` / sha **8fdaa1e** / `PROFILE=1` / ptx=none)

baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。

### 3.1 t502a (`MODEL=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` / load_before=1.67)

COND: x0 (捨て) b **sa** (`DENNO_SEG_SUM_ASYNC=1`) **saq** (+`DENNO_SEG_PROF=1`) b2

| COND | short decode ms/tok | 3,826 cold s | 14k cold s | 8k 後 decode |
| --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 14.8 | 18.76 | 23.58 | 41.0 |
| b (base) | **14.1** | 16.68 | 21.78 | 32.0 |
| sa (async) | **13.8** | 16.57 | 19.76 | 22.6 |
| saq (async+prof) | 14.3 | 17.00 | 36.65 (外れ) | 22.4 |
| b2 (base) | **14.1** | 16.02 | 19.24 | 18.2 |

md5: 長い 3 本 (c2000_cold / c2000_turn2 / c8000_cold) は**全 COND SAME**、
`sa` は 48 token の `b` も SAME。errors 0、text ok。
`[seg-async] calls=6050 -> 12194 -> ... -> 42474`、**fallbacks=0**(a1 = 全ミスの
28 token 要求だけ calls=0 = ヒットが無いので seg 経路に入らない、正しい)。

### 3.2 t502b (`SCENARIO=depth` / load_before=2.26) — 判定に使った回

COND: x0 (捨て) b **sa** b2 **sa2** (`DENNO_SEG_SUM_ASYNC=1`)

| COND | short decode ms/tok | 3,826 cold s (c2000_cold) |
| --- | ---: | ---: |
| x0 (捨て) | 22.4 (冷え) | 18.80 |
| b | **13.9** | 16.79 |
| sa | **13.8** | 16.23 |
| b2 | **14.0** | 16.15 |
| sa2 | **13.8** | 16.05 |

**2 run の 4 点をまとめた median: base 14.05 (14.1 / 14.1 / 13.9 / 14.0)、
async 13.8 (13.8 / 13.8 / 13.8) = -1.8%。** md5 は x0 以外で 48 token の `b` が
既存の揺れ (P490/P494 の記録どおり)、長い 3 本は SAME。errors 0。

### 3.3 t502c (既定モデル `flash-next-UD-IQ3_XXS` / `SCENARIO=depth` / load_before=1.43)

COND: x0 b **sa** b2

| COND | short decode ms/tok | 3,826 cold s | md5 |
| --- | ---: | ---: | --- |
| x0 | 10.6 | 1.98 | a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3 |
| b | 10.6 | 1.98 | 同 |
| sa | 10.6 | 1.94 | 同 |
| b2 | 10.6 | 1.92 | 同 |

**`[seg-async] calls=0 fallbacks=0` が 14 リクエスト全部**。既定モデルの expert は
IQ 系なので seg K 型経路 (`moe_seg_layer_sum`) に入らない = **ゲート on でも
コードが 1 行も走らない**。decode/prefill/md5 が全 COND 同値なのはそのため
(「既定モデルが変わっていない」はラウンドの同値と `calls=0` の両方で示した)。

### 3.4 機構の証拠 (`[seg-prof]`、t502a の saq、定常 48 層、us/層)

| lookup | table | h2d | launch(5) | sync | d2h | miss(包括) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9-1.0 | 0.5-0.9 | 2.8 | 8.7-10.0 | **0.9-1.0** | **1.1-1.5** | 500-675 |

P498 (t498b) の同じ列は d2h **13.3-14.8** / sync 1-37 だった。**d2h は 1/10 以下に
なったが、`miss=` (包括区間) が増えており、層の壁は変わらない**。同期コピーが
吸収していた待ち (背景の `[dec-up-w]` の上げ直し) がミス区間へ移った、と読む。
`lookup`/`table`/`h2d`/`launch` は P498 と同じ値で、**他の区間は一切変わっていない**。

### 3.5 VERDICT 行

```text
t502a: VERDICT b vs x0 OK decode -4.5% prefill -8.2% md5 SAME errors 0
       VERDICT sa vs x0 OK decode -6.6% prefill -11.9% md5 SAME errors 0
       VERDICT saq vs x0 OK decode -3.4% prefill -9.7% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -4.7% prefill -9.7% md5 DIFF errors 0
t502b: VERDICT b vs x0 OK decode -38.2% prefill -25.6% md5 SAME errors 0
       VERDICT sa vs x0 OK decode -38.5% prefill -26.8% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -37.6% prefill -26.8% md5 DIFF errors 0
       VERDICT sa2 vs x0 OK decode -38.7% prefill -28.0% md5 DIFF errors 0
t502c: VERDICT b vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
       VERDICT sa vs x0 OK decode -0.4% prefill +0.9% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode -0.4% prefill -0.9% md5 SAME errors 0
```

(t502b/t502c の x0 は 1 本目で冷えているので VERDICT の % は採否に使わない。
採否は §3.2 の b/b2 対 sa/sa2 の median で読む。)

## 4. md5 と出力テキスト

- UD (t502a/b): 長い 3 本 (`c2000_cold` / `c2000_turn2` / `c8000_cold`) は
  **全 run 全 COND SAME**。48 token の `b` の md5 が COND 間で揺れるのは
  P490/P494/P498 が記録した既存の揺れ (同一 env の COND 同士でも揺れる)。
- 既定モデル (t502c): **全 COND 全 md5 SAME**。
- 出力 (t502a の `sa`、3,826 の回答):

```text
prefillの律速は計算量ではなく、線形層ごとのホスト往復と戻り値コピーであった。
pinned経由の二度手間を排除し、FFNをデバイス常駐化することで、1.5Bモデルで
395.8秒から12.2秒へ高速化した。最適化の過程では、計測の誤認や静かな退行が頻発し、
実測に基づく検証と計測手法の改善が不可欠であった。
```

- 短い `TXTB` (全 COND 同趣旨): 「Rust の「所有権（Ownership）とライフタイム
  （Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。…」

## 5. 数値の扱い

- すべて GPU ラボの実測 (`SCENARIO=depth` / `depth8k`、`PROFILE=1`)。
  このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- **-1.8% は推定ではなく実測の median** (2 run x 4 点 対 3 点)。ただし
  1 ラウンド内のばらつきは ±5-10% あり、1 run だけでは 2% は読めない。
  2 run をまとめて「base 14.05 / async 13.8」で安定したので、これを結論に使う。
- 「同期コピーが背景の上げ直しを吸収していた」は**推定** (直接の証拠は
  `d2h` が 1/10 になったのに `miss=` が増えて壁が動かないこと、と
  task 356/498 が別の見え方で 0% だったこと)。
- 期待値の計算: d2h 12 us x 48 層 = 0.58 ms/token = 14.1 の 4.1%。
  実測はその半分以下 (-1.8%) で、**段の時間は減ったが壁の時間ではない**。

## 6. テスト

最終 HEAD (int-auto 890fc66 の上、br498 merge 済み) で:

```text
CARGO_TARGET_DIR=/tmp/t502  cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1358 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t502  cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 144 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t502nc cargo build --workspace --release            (cuda 無し) -> OK
CARGO_TARGET_DIR=/tmp/t502nc cargo test --workspace --release --no-fail-fast -- --test-threads=1 -> 失敗 0
```

新規テスト 5 本は緑: `model::seg_prof::tests::t502_seg_sum_async_truth_table` /
`t502_seg_async_counts_accumulate` / `cuda::task502_tests::seg_async_issue_rejects_out_of_range` /
`seg_async_collect_rejects_out_of_range` / `seg_async_issue_falls_back_without_cuda`
(最後の 1 本は **GPU の無いこのコンテナでは fallback 側**を通る = fail closed の証拠)。
非同期経路の実発火は lab の `[seg-async] calls=6048 fallbacks=0` で確認した。

## 7. 次の担当者へ

1. **seg 層の和 D2H は機構として閉じた。** 同期 (P498) も非同期 (本タスク) も
   壁を動かさない。同期コピーは背景の上げ直しの待ちを吸収していた。
   `DENNO_SEG_SUM_ASYNC` は既定 off で残した (切り戻し不要、既定の経路は不変)。
2. **UD の decode で名前が付いていて残っているのは launch 9-10 us/層 だけ**
   (5 起動 = 1.9 us/起動)。これを削るには 5 起動の融合 = `.cu` が要る。
   task 412 (`DENNO_SEG_GU_FUSE`) が -0.6% で棄却しており、本タスクは `.cu` 不可。
3. **experts の壁はミス読み (57 us/本 = 51 GB/s) + 層固定費**。P498 の 2 項モデルは
   今の int-auto でも成り立つ (hit_rate 0.83-0.91、VRAM 21,996 MiB)。
4. **prefill の 5% は相変わらずディスク次第** (t502a の 14k: b 21.78 / b2 19.24 s =
   同じバイナリで 13% 振れる)。判定は `[exp-pf-stream] copy_s` が 6 GB/s 以上出た回で。
